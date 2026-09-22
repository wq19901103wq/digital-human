"""Resumable evaluation of a completed or already frozen learned selector."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

from ..config import ROOT, ConfigError
from ..generator import few_shot, learned_sources
from ..generator.few_shot import PersonaFewShotRetriever
from ..generator.history_sources import require
from ..generator.learned_selection import LearnedSelector, POLICY, recall
from ..generator.fewshot_ranker import extraction
from . import branches, control, datasets, experiment, pack_transport, versions
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
        write_once_json(output / 'manifest.json', manifest)
        state_path = versions.PRIVATE / 'branches' / name / 'state.json'
        if state_path.exists():
            state = read_json(state_path)
            proposal = read_json(state_path.parent / 'revisions' / (state['revision'] + '.json'))
        else:
            require(current['production_gen'] == base, 'Production baseline changed')
            proposal = branches.submit(name, 'gen', 'Frozen learned selector against production on current data',
                                       candidate_ref=candidate)
        require(proposal['candidate_ref'] == candidate and proposal['development_ref'] == base and
                proposal['basis']['data'] == data,
                'Existing branch differs from requested learned-selector comparison')
        branches.set_stage_limit(name, stage)
        return candidate
    require(model is not None and source_branch is not None, 'Model requires an accepted source branch')
    source = read_json(versions.PRIVATE / 'branches' / source_branch / 'state.json')
    require(current['data'] == data and source.get('development', {}).get('basis') == current and
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
    retriever = PersonaFewShotRetriever(path=data_dir / 'fewshot_pool.jsonl')
    tasks = {}
    cases = datasets.rows_for(spec)
    started = time.time()
    def progress(phase, completed):
        value = dict(phase=phase, completed=completed, total=len(cases), tasks=len(tasks),
                     elapsed_seconds=round(time.time() - started, 1), updated_at=time.time())
        write_json(output / 'features/recall_progress.json', value)
        print(f'History {phase}: {completed}/{len(cases)}; {len(tasks)} unique feature tasks; '
              f'{value["elapsed_seconds"]}s elapsed', flush=True)
    with pack_transport.history_feature_cache(few_shot):
        progress('loading', 0)
        require(retriever.is_approved(), 'Historical recall pool is not approved')
        progress('recalling', 0)
        reported = time.time()
        for index, case in enumerate(cases, 1):
            control.check()
            prepared, _ = selector.tasks(case, recall(retriever, case,
                source_overlap_policy=switches.get('source_overlap_policy')))
            tasks.update(prepared)
            if time.time() - reported >= 15 or index == len(cases):
                progress('recalling', index)
                reported = time.time()
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
            candidate = prepare(output, model=model, base=base, source_branch=source_branch,
                                name=name, data=data, candidate=candidate, stage=stage)
            update(candidate=candidate, baseline=base, branch=name)
            attempts = {}
            while True:
                control.check()
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
