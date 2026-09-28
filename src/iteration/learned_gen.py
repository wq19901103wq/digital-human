"""Resumable evaluation of a completed or already frozen learned selector."""
from copy import deepcopy
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

from ..config import ROOT, ConfigError, sha256_file
from ..generator import few_shot, history_sources, learned_sources
from ..generator.few_shot import PersonaFewShotRetriever
from ..generator.history_sources import require
from ..generator.learned_selection import LearnedSelector, POLICY, recall
from ..generator.fewshot_ranker import supplemental as extraction
from . import branches, control, datasets, experiment, pack_transport, runtime, versions
from .parallel import worker_limit
from .storage import file_lock, locked, read_json, write_json, write_once_json

TERMINAL = {'rejected', 'superseded', 'promoted', 'integrated', 'conflict', 'blocked',
            'development_accepted'}


def status(output):
    output = Path(output)
    result = read_json(output / 'state.json', default={})
    result['running'] = locked(output / '.run.lock')
    result['features'] = read_json(output / 'features/feature_progress.json', default={})
    result['recall'] = read_json(output / 'features/recall_progress.json', default={})
    result['memory_lane'] = read_json(output / 'memory_lane.json', default={})
    if result.get('experiment'):
        result['experiment_state'] = experiment.state_of(experiment.load_experiment(result['experiment']))
    return result


def prepare(output, *, base, name, data, model=None, source_branch=None,
            candidate=None, stage='development'):
    """Idempotent version creation and branch submission, including crash recovery."""
    branches._name(name)
    output = Path(output)
    current = branches.basis()
    require(stage in {'development', 'fixed_test'}, 'Invalid evaluation stage')
    require(current['data'] == data, 'Evaluation data changed')
    if candidate is not None:
        require(model is None and source_branch is None,
                'Frozen candidate cannot be combined with model or source branch')
        cfg = versions.load_generator(candidate)['config']
        require(cfg.get('retriever', {}).get('learned') == POLICY,
                'Candidate is not a frozen learned selector')
        manifest = dict(schema=2, candidate=candidate, base=base, data=data,
                        name=name, policy=POLICY, stage=stage)
        state_path = versions.PRIVATE / 'branches' / name / 'state.json'
        saved_manifest = read_json(output / 'manifest.json', default={})
        if saved_manifest.get('schema') == 1:
            # Model-based runs already froze their candidate in this branch.
            # Resume that version after its source was adopted; do not deploy
            # the old model again or rewrite its original manifest.
            require(state_path.exists() and all(saved_manifest.get(key) == manifest[key]
                    for key in ('base', 'data', 'name', 'policy')) and
                    saved_manifest.get('stage', 'development') == stage,
                    'Existing model evaluation differs from frozen-candidate resume')
        else:
            write_once_json(output / 'manifest.json', manifest)
        if state_path.exists():
            state = read_json(state_path)
            proposal = read_json(state_path.parent / 'revisions' / (state['revision'] + '.json'))
        else:
            require(current['production_gen'] == base, 'Production baseline changed')
            proposal = branches.submit(name, 'gen', 'Frozen learned selector against production on current data',
                                       candidate_ref=candidate)
        same_data = proposal['basis']['data'] == data
        if not same_data and stage == 'fixed_test':
            same_data = datasets.fixed_refresh_compatible(proposal['basis']['data'], data)
        require(proposal['candidate_ref'] == candidate and proposal['development_ref'] == base and same_data,
                'Existing branch differs from requested learned-selector comparison')
        branches.set_stage_limit(name, stage)
        return candidate
    require(model is not None, 'Evaluation requires a completed model or frozen candidate')
    if source_branch is None:
        require(current['production_gen'] == base, 'Production baseline changed')
    else:
        source = read_json(versions.PRIVATE / 'branches' / source_branch / 'state.json')
        require(source.get('development', {}).get('basis') == current and
                source['development']['candidate_ref'] == base, 'Source branch baseline changed')
    manifest = dict(schema=1, model=str(Path(model).resolve()), base=base, data=data,
                    source_branch=source_branch, name=name, policy=POLICY)
    if stage != 'development':
        manifest['stage'] = stage
    write_once_json(output / 'manifest.json', manifest)
    base_info = versions.load_generator(base)
    assets = learned_sources.deploy(model, base_info['dir'], data, output / 'assets')
    cfg = deepcopy(base_info['config'])
    cfg['retriever'] = {'enabled': True, 'learned': POLICY}
    candidate = versions.create_generator_version(cfg, data, source_dir=assets)
    state_path = versions.PRIVATE / 'branches' / name / 'state.json'
    if state_path.exists():
        state = read_json(state_path)
        proposal = read_json(state_path.parent / 'revisions' / (state['revision'] + '.json'))
        require(proposal['candidate_ref'] == candidate and proposal['development_ref'] == base,
                'Existing branch differs from requested learned-selector comparison')
    else:
        proposal = branches.submit(name, 'gen', 'Learned few-shot ranking with cached independent features',
                                   candidate_ref=candidate, from_branch=source_branch)
        require(proposal['development_ref'] == base, 'Source branch baseline changed')
    branches.set_stage_limit(name, stage)
    return candidate


def release_history():
    """Drop process-local archives before network work or another memory lane."""
    with history_sources._lock:
        history_sources._load.cache_clear()
    gc.collect()


@contextmanager
def history_phase(output):
    release_history()
    with pack_transport.history_memory_lane(output):
        try:
            yield
        finally:
            release_history()


def recall_binding(spec, cases, switches, selector, data_dir):
    """Bind recall work to exact inputs, even when a data/version ID is reused."""
    paths = [data_dir / name for name in
             ('messages.jsonl', 'purposes.json', 'fewshot_pool.jsonl', 'report.json')]
    stamps = [history_sources.stamp(path) for path in paths]
    # Precomputation recalls all rows_for(spec), including during smoke. Stage
    # bookkeeping and its limited record contract do not change those inputs.
    recall_spec = {key: value for key, value in spec.items() if key not in {
        'id', 'created', 'smoke', 'smoke_limit', 'branch_stage', 'record_contract', 'fingerprint'}}
    if 'data_snapshot' in recall_spec:
        recall_spec['data_snapshot'] = {key: value for key, value in
            recall_spec['data_snapshot'].items() if key not in {'case_count', 'cases_sha256'}}
    value = dict(schema=2, spec=recall_spec, cases_sha256=history_sources.digest(cases),
        switches=switches, feature_identity=selector.proof['feature_identity'],
        sources={str(path): sha256_file(path) for path in paths},
        code={str(path.relative_to(ROOT)): sha256_file(path) for path in runtime.files(ROOT)})
    require(stamps == [history_sources.stamp(path) for path in paths],
            'Historical recall sources changed while binding inputs')
    return history_sources.digest(value), paths, stamps


def precompute(directory, candidate, output, workers):
    """Shared target-once / example-once extraction before parallel generation."""
    limit = control.policy().get('max_active_requests', 16)
    require(type(workers) is int and 1 <= workers <= limit,
            f'feature workers must be within the instance concurrency limit (1–{limit})')
    spec = experiment.spec_of(directory)
    data_dir = versions.data_version_dir(spec['data_ref'])
    switches = versions.load_generator(candidate)['config'].get('retriever', {})
    selector = LearnedSelector(versions.generator_dir(candidate) / 'ranker',
                              versions.PRIVATE / '.cache/fewshot_ranker_features')
    cases = datasets.rows_for(spec)
    binding, source_paths, source_stamps = recall_binding(spec, cases, switches, selector, data_dir)
    checkpoint = output / 'features/recall_tasks.json'
    saved = read_json(checkpoint, default={})
    tasks, completed = {}, 0
    if saved.get('binding') == binding:
        require(saved.get('payload_sha256') == history_sources.digest(
            {key: value for key, value in saved.items() if key != 'payload_sha256'}),
            'Recall checkpoint content changed')
        tasks, completed = saved['tasks'], saved['completed']
        require(type(completed) is int and 0 <= completed <= len(cases) and
                isinstance(tasks, dict) and all(key == history_sources.digest(request)
                    for key, request in tasks.items()), 'Invalid recall task checkpoint')
    started = time.time()
    def progress(phase, completed):
        value = dict(phase=phase, completed=completed, total=len(cases), tasks=len(tasks),
                     elapsed_seconds=round(time.time() - started, 1), updated_at=time.time())
        write_json(output / 'features/recall_progress.json', value)
        print(f'History {phase}: {completed}/{len(cases)}; {len(tasks)} unique feature tasks; '
              f'{value["elapsed_seconds"]}s elapsed', flush=True)
    if completed < len(cases):
        progress('waiting_for_memory', completed)
        with history_phase(output), pack_transport.history_feature_cache(few_shot):
            retriever = None
            try:
                progress('loading', completed)
                retriever = PersonaFewShotRetriever(path=data_dir / 'fewshot_pool.jsonl')
                require(retriever.is_approved(), 'Historical recall pool is not approved')
                progress('recalling', completed)
                reported = time.time()
                for index in range(completed, len(cases)):
                    control.check()
                    prepared, _ = selector.tasks(cases[index], recall(retriever, cases[index],
                        source_overlap_policy=switches.get('source_overlap_policy')))
                    tasks.update(prepared)
                    if time.time() - reported >= 15 or index + 1 == len(cases):
                        require(source_stamps == [history_sources.stamp(path) for path in source_paths],
                                'Historical recall sources changed; checkpoint not updated')
                        value = dict(binding=binding, completed=index + 1, tasks=tasks)
                        write_json(checkpoint, {**value, 'payload_sha256': history_sources.digest(value)})
                        progress('recalling', index + 1)
                        reported = time.time()
            finally:
                retriever = None
    else:
        release_history()
    progress('complete', len(cases))
    with worker_limit(limit):
        extraction.run(tasks, selector.cache, output / 'features', selector.client, workers=workers)


def run(output, *, base, name, data, model=None, source_branch=None, candidate=None,
        stage='development', workers=16, feature_workers=None, timeout=360):
    output = Path(output).resolve()
    with file_lock(output / '.run.lock', blocking=False), control.job(output):
        state = read_json(output / 'state.json', default={})
        def update(**changes):
            state.update(changes, updated_at=time.time(), pid=os.getpid())
            write_json(output / 'state.json', state)
        try:
            state.pop('error', None)
            update(status='preparing')
            with history_phase(output):
                candidate = prepare(output, model=model, base=base, source_branch=source_branch,
                                    name=name, data=data, candidate=candidate, stage=stage)
            update(candidate=candidate, baseline=base, branch=name)
            attempts = {}
            while True:
                control.check()
                with history_phase(output):
                    jobs = branches.advance(name)
                branch = read_json(versions.PRIVATE / 'branches' / name / 'state.json')
                record = branches.round_of(f"{name}/{branch['rounds'][-1]}")
                if record['phase'] in TERMINAL:
                    update(status=record['phase'], round=record)
                    return status(output)
                require(jobs, 'Branch has no runnable job; inspect its pause/stage state')
                for job in jobs:
                    require(job['kind'] == 'experiment', 'Learned generator expects generator experiments')
                    directory = experiment.load_experiment(job['id'])
                    update(status='precomputing', experiment=job['id'], round=record)
                    precompute(directory, candidate, output, workers if feature_workers is None else feature_workers)
                    update(status='evaluating')
                    code = pack_transport.launch(directory, instance=versions.PRIVATE.name,
                        job=job['id'], workers=workers, seconds=timeout, kind='experiment')
                    if code:
                        attempts[job['id']] = attempts.get(job['id'], 0) + 1
                        require(attempts[job['id']] < 3, 'Experiment retries exhausted; successes preserved')
        except Exception as exc:
            update(status='needs_attention', error=f'{type(exc).__name__}: {exc}')
            raise


def launch(output, arguments):
    output = Path(output).resolve()
    with file_lock(output / '.launch.lock', blocking=False):
        require(not locked(output / '.run.lock'), 'Learned generator experiment is already running')
        previous = read_json(output / 'process.json', default={})
        if previous.get('pid'):
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise ConfigError('Previous experiment launcher is still alive')
        command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate_learned_fewshot.py'), *arguments]
        with (output / 'run.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
        result = dict(pid=process.pid, started_at=time.time(), command=command)
        write_json(output / 'process.json', result)
        return result
