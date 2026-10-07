"""Recorded transport and historical-code path compatibility for frozen workers."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from functools import lru_cache, wraps
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
from threading import Lock, local
import time
import uuid


def live_disk_history():
    """Load the storage adapter outside the authenticated historical snapshot."""
    path = Path(__file__).resolve().parents[1] / 'generator' / 'disk_history.py'
    spec = importlib.util.spec_from_file_location('_digital_human_disk_history', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_transport_profile():
    path = Path(__file__).resolve().with_name('transport_profile.py')
    spec = importlib.util.spec_from_file_location('_digital_human_transport_profile', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_retrieval_cache():
    path = Path(__file__).resolve().parents[1] / 'generator' / 'recall_reuse.py'
    spec = importlib.util.spec_from_file_location('_digital_human_retrieval_cache', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_disk_source_reads():
    path = Path(__file__).resolve().parents[1] / 'generator' / 'disk_source_reads.py'
    spec = importlib.util.spec_from_file_location('_digital_human_disk_source_reads', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_material_reuse():
    path = Path(__file__).resolve().with_name('material_reuse.py')
    spec = importlib.util.spec_from_file_location('_digital_human_material_reuse', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_code_evidence(runtime, name, expected):
    """Resolve only exact historical Python bytes in authenticated snapshots."""
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)):
        raise runtime.ConfigError('invalid historical evidence digest')
    p = Path(name)
    if p.suffix != '.py':
        return None
    for manifest in sorted((runtime.versions.PRIVATE / 'runtimes').glob('*/manifest.json')):
        value = runtime.json.loads(manifest.read_text())
        if runtime.digest(value) != manifest.parent.name:
            continue
        for relative, checksum in value.get('files', {}).items():
            path = Path(relative)
            if (checksum != expected or path.is_absolute() or '..' in path.parts
                    or not str(p).endswith('/' + relative) or path.suffix != '.py'):
                continue
            candidate = manifest.parent / path
            if (candidate.resolve().is_relative_to(manifest.parent.resolve())
                    and not candidate.is_symlink() and candidate.is_file()
                    and runtime.sha256_file(candidate) == expected):
                return candidate
    return None


@contextmanager
def archived_runtime_paths(runtime):
    """Give old executors the same exact-code lookup used by the live runtime."""
    original = runtime.evidence_path
    def resolve(name, expected):
        try:
            return original(name, expected)
        except runtime.ConfigError:
            archived = snapshot_code_evidence(runtime, name, expected)
            if archived is None:
                raise
            return archived
    runtime.evidence_path = resolve
    try:
        yield
    finally:
        runtime.evidence_path = original


def history_memory_capacity():
    """Keep the serial default; allow one measured extra archive resident."""
    value = os.environ.get('DH_HISTORY_MEMORY_LANES', '1')
    if value not in {'1', '2'}:
        raise ValueError('DH_HISTORY_MEMORY_LANES must be 1 or 2')
    return int(value)


def needs_history_memory(directory, kind):
    """Saved-reply Judge evaluation does not load the generation history archive."""
    from src.iteration.storage import read_json
    if kind in {'experiment', 'promotion'}:
        return read_json(Path(directory) / 'spec.json').get('kind') != 'judge_eval'
    return True


@contextmanager
def history_memory_lane(output, *, required=True):
    """Bound archive residents across processes, including old single-lane jobs.

    锁位于 instances 根（versions.PRIVATE.parent），是兼容旧单车道作业的已知取舍：
    并发槽全局有界，但不按实例隔离；多实例部署需要更强隔离时应改配置而非散锁。
    """
    from src.iteration import control, versions
    from src.iteration.storage import LockBusy, file_lock, write_json
    output = Path(output)
    if not required:
        control.check(output)
        yield
        return
    capacity = history_memory_capacity()
    started = time.time()
    slot = None
    def state(phase):
        write_json(output / 'memory_lane.json', dict(phase=phase, pid=os.getpid(),
            capacity=capacity, slot=slot, requested_at=started, updated_at=time.time()))
    state('waiting')
    try:
        with ExitStack() as stack:
            while slot is None:
                control.check()
                for index in range(capacity):
                    # Slot zero must remain the old lock: a live frozen worker
                    # already holding it counts toward the new two-slot limit.
                    name = '.history_memory.lock' if index == 0 else '.history_memory.1.lock'
                    try:
                        stack.enter_context(file_lock(versions.PRIVATE.parent / name, blocking=False))
                    except LockBusy:
                        continue
                    slot = index
                    break
                if slot is None:
                    time.sleep(.1)
            control.check()
            state('active')
            yield
    finally:
        state('idle')


@contextmanager
def history_feature_cache(module):
    """Share immutable terms and memoize pure analysis with fresh mutable results.

    Keys include the complete text and all feature flags. Per-case history
    eligibility and row overrides are deliberately outside this cache.
    """
    originals = {}
    def wrap(original):
        results = {}
        lock = Lock()
        @lru_cache(maxsize=256)
        def cached(*args, **kwargs):
            result = original(*args, **kwargs)
            identity = (type(result), tuple(result.items()) if isinstance(result, dict)
                        else tuple(result))
            with lock:
                if identity in results:
                    return results[identity]
                if len(results) < 4096:
                    results[identity] = result
            return result
        @wraps(original)
        def call(*args, **kwargs):
            return cached(*args, **kwargs).copy()
        return call
    original_terms = module._terms
    @wraps(original_terms)
    def terms(*args, **kwargs):
        return [sys.intern(term) for term in original_terms(*args, **kwargs)]
    try:
        originals['_terms'] = original_terms
        module._terms = terms
        for name in ('_situation_tags', '_situation_profile'):
            originals[name] = getattr(module, name)
            setattr(module, name, wrap(originals[name]))
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def timeout_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 3600:
        raise ValueError('request timeout must be finite and within (0, 3600] seconds')
    return value


def time_slice_value(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('time slice must be finite and positive')
    return value


@contextmanager
def case_time_slice(seconds, *, profile=None, unattempted_results=None):
    """Start the work budget after lazy preparation, then drain all results."""
    if seconds is None:
        yield
        return
    from src.iteration import control, progress, runner
    seconds = time_slice_value(seconds)
    original = runner.completed_map
    attempted = set(runner._final_records(Path(unattempted_results))[0]) if unattempted_results is not None else None

    def completed(function, items, workers=1):
        if attempted is not None:
            items = sorted(items, key=lambda item: str(item[1]['case_id']) in attempted)
        started = time.monotonic()
        deadline = started + seconds
        drained = False
        ready = False
        warmup = 0.0
        saved = failures = 0
        lock = Lock()

        def stage_wrapper(original_stage):
            @wraps(original_stage)
            def stage(instance, phase, **fields):
                nonlocal deadline, ready, warmup
                if phase in {'generating', 'judging'}:
                    with lock:
                        if not ready:
                            warmup = time.monotonic() - started
                            deadline = started + warmup + seconds
                            ready = True
                            print(f'Time slice ready: lazy preparation={warmup:.3f}s; '
                                  f'credited={warmup:.3f}s; '
                                  f'work budget={seconds:g}s', flush=True)
                return original_stage(instance, phase, **fields)
            return stage

        def bounded():
            nonlocal drained
            submitted = 0
            for item in items:
                with lock:
                    drained = bool(submitted and time.monotonic() >= deadline)
                if drained:
                    return
                submitted += 1
                yield item

        stages = {kind: kind.stage for kind in (progress.RunProgress, progress.ParallelCaseProgress)}
        try:
            for kind, original_stage in stages.items():
                kind.stage = stage_wrapper(original_stage)
            for result in original(function, bounded(), workers):
                yield result
                if isinstance(result, dict):
                    saved += result.get('status') == 'ok'
                    failures += result.get('status') == 'failed'
        finally:
            for kind, original_stage in stages.items():
                kind.stage = original_stage
            elapsed = time.monotonic() - started
            if profile is not None:
                profile.record_slice(elapsed_seconds=elapsed, lazy_preparation_seconds=warmup,
                    budget_seconds=seconds, success=saved, failed=failures, workers=workers,
                    drain_requested=drained)
            print(f'Time slice ended: elapsed={elapsed:.3f}s; lazy preparation={warmup:.3f}s; '
                  f'persisted success={saved}; failed={failures}', flush=True)
        if drained:
            raise control.StopRequested('time_slice_complete')

    runner.completed_map = completed
    try:
        yield
    finally:
        runner.completed_map = original


def connection_env_file():
    """Keep credentials in the live project, outside the frozen executor."""
    return str(Path(os.environ.get('DH_ENV_FILE',
        Path(__file__).resolve().parents[2] / '.env')).resolve())


@contextmanager
def timeout_floor(client_type, seconds):
    """Change only the local wait limit; preserve request and cache identities."""
    seconds = timeout_value(seconds)
    original = client_type.__init__

    def initialize(self, settings, model_cfg):
        config = {**model_cfg, 'timeout_seconds': max(
            float(model_cfg.get('timeout_seconds', 60)), seconds)}
        original(self, settings, config)

    client_type.__init__ = initialize
    try:
        yield
    finally:
        client_type.__init__ = original


def worker_timeout_floor(client_type, seconds):
    return nullcontext() if seconds == 0 else timeout_floor(client_type, seconds)


@contextmanager
def feature_response_cache_validation():
    """Reject cached responses using the frozen extractor's own evidence rules."""
    from src import cache
    from src.generator.fewshot_ranker import supplemental
    from src.generator.history import HistoryError
    from src.llm import LLMError
    original = supplemental.extract_one
    original_failed = getattr(supplemental, 'failed', None)
    failures = local()

    @wraps(original)
    def extract(directory, key, request, client):
        if request['kind'] != supplemental.concern.KIND:
            return original(directory, key, request, client)
        previous = getattr(failures, 'rejected', None)
        rejected = failures.rejected = []
        try:
            with cache.validation(lambda raw: supplemental.concern.validate(json.loads(raw), request)):
                return original(directory, key, request, client)
        except HistoryError as exc:
            if rejected and rejected[-1] is exc:
                raise LLMError(str(exc), retryable=True) from exc
            raise
        finally:
            failures.rejected = previous

    supplemental.extract_one = extract
    if original_failed is not None:
        @wraps(original_failed)
        def failed(directory, key, request, raw, exc):
            original_failed(directory, key, request, raw, exc)
            rejected = getattr(failures, 'rejected', None)
            if rejected is not None:
                rejected.append(exc)
        supplemental.failed = failed
    try:
        yield
    finally:
        supplemental.extract_one = original
        if original_failed is not None:
            supplemental.failed = original_failed


@contextmanager
def archived_gbdt_paths(module, runtime):
    """Adapt the old audit's physical-path manifest to verified code archives.

    The frozen verifier still reconstructs the complete model. Only its in-memory
    evidence-key view changes; original records, hashes and executor stay intact.
    """
    original_verify = module.verify
    if getattr(original_verify, '_archived_gbdt_paths', False):
        yield
        return

    def verify(directory, record, original, study, source_spec, arm):
        descriptor = module.read(directory / 'scorer_source.json')
        sweep = module.versions.PRIVATE / 'judge_training' / module._name(descriptor['study'])
        inputs = module.read(sweep / 'spec.json')['inputs']
        inherited = module.read(original / 'learning.json')['evidence_files']
        evidence = dict(record['evidence_files'])
        for name, expected in inputs.items():
            actual = runtime.evidence_path(name, expected)
            if str(actual.resolve()) == str(Path(name).resolve()):
                continue
            module.require(evidence.get(name) == expected,
                           'archived GBDT source is missing from original evidence')
            alias = str(actual.resolve())
            module.require(alias not in evidence or evidence[alias] == expected,
                           'archived GBDT source identity conflicts')
            evidence[alias] = expected
            if name not in inherited:
                del evidence[name]
        return original_verify(directory, {**record, 'evidence_files': evidence},
                               original, study, source_spec, arm)

    verify._archived_gbdt_paths = True
    module.verify = verify
    try:
        yield
    finally:
        module.verify = original_verify


def snapshot_generation(snapshot, runtime):
    """Classify an authenticated snapshot by its manifest file list (design §2.3a).

    The verified manifest decides the generation before any import that could
    reach a retrieval module: legacy archives carry ``src/generator/few_shot.py``;
    modern archives carry the self-contained ``src/generator/fewshot/`` package.
    Anything else is rejected (fail-closed).
    """
    try:
        manifest = json.loads((Path(snapshot) / 'manifest.json').read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise runtime.ConfigError(f'cannot classify snapshot generation: {exc}') from exc
    files = manifest.get('files')
    if not isinstance(files, dict):
        raise runtime.ConfigError('cannot classify snapshot generation: manifest has no files map')
    legacy = any(name in files for name in ('src/generator/few_shot.py',
                                            'src/digital_human/generator/few_shot.py'))
    modern = any(name == 'src/generator/fewshot' or name.startswith('src/generator/fewshot/')
                 for name in files)
    if legacy and modern:
        raise runtime.ConfigError('cannot classify snapshot generation: manifest contains both '
                                  'src/generator/few_shot.py and src/generator/fewshot/')
    if not legacy and not modern:
        raise runtime.ConfigError('cannot classify snapshot generation: manifest contains neither '
                                  'src/generator/few_shot.py nor src/generator/fewshot/')
    return 'few_shot' if legacy else 'fewshot'


@contextmanager
def fewshot_profile_instrument(profile, fewshot):
    """Hang transport timing spans on the new fewshot package entry points.

    Legacy snapshots keep using transport_profile.instrument, which patches the
    old PersonaFewShotRetriever; modern archives do not contain that class.
    """
    from src import tracing
    originals = []

    def replace(owner, name, wrapper):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        setattr(owner, name, wrapper(original))

    def measured(name):
        def wrap(original):
            @wraps(original)
            def call(*args, **kwargs):
                with profile.span(name):
                    return original(*args, **kwargs)
            return call
        return wrap

    def approval(original):
        @wraps(original)
        def call(session, *args, **kwargs):
            result = original(session, *args, **kwargs)
            engine = getattr(session, '_engine', None)
            if engine is not None:
                profile.record('recall_engine_' + type(engine).__name__, 0.0)
            return result
        return call

    def model_step(original):
        @wraps(original)
        @contextmanager
        def call(kind, request):
            span = profile.span('model_' + kind) if kind in {'llm', 'codex'} else nullcontext()
            with span, original(kind, request) as value:
                yield value
        return call

    try:
        replace(fewshot.RecallSession, 'is_approved', measured('history_pool_approval'))
        replace(fewshot.RecallSession, 'is_approved', approval)
        replace(fewshot.RecallSession, 'retrieve', measured('retrieval_compute'))
        replace(fewshot.RecallSession, '_candidate_rows', measured('retrieval_candidate_selection'))
        if hasattr(fewshot.RecallSession, '_sparse_candidates'):
            replace(fewshot.RecallSession, '_sparse_candidates', measured('retrieval_sparse_selection'))
        for engine in (fewshot.InMemoryRecallEngine, fewshot.SqliteIndexedRecallEngine):
            replace(engine, 'excluded_ids', measured('retrieval_excluded_ids'))
            replace(engine, 'term_features', measured('retrieval_term_features'))
        replace(fewshot.rank, 'score_and_select', measured('retrieval_rank'))
        replace(fewshot.RecallSession, 'render_selected', measured('retrieval_render'))
        replace(tracing, 'step', model_step)
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def launch(directory, *, instance, job, workers, seconds, kind='pack', feature_output=None):
    from . import runtime, versions
    seconds = timeout_value(seconds)
    snapshot = runtime.verify(Path(directory) / 'runtime.json')
    transport = Path(__file__).resolve()
    if snapshot_generation(snapshot, runtime) == 'fewshot':
        archived_transport = snapshot / 'src' / 'iteration' / 'pack_transport.py'
        if not archived_transport.is_file():
            raise runtime.ConfigError('modern snapshot is missing src/iteration/pack_transport.py')
    command = [sys.executable, '-B', '-u', str(transport),
               '--snapshot', str(snapshot), '--instance', instance, '--job', job,
               '--workers', str(workers), '--timeout-seconds', str(seconds), '--kind', kind]
    if feature_output is not None:
        if kind != 'features':
            raise ValueError('feature output requires features operation')
        command.extend(['--feature-output', str(Path(feature_output).resolve())])
    return subprocess.call(command, cwd=snapshot, env={**os.environ,
        'DH_ENV_FILE': connection_env_file(),
        'DH_INSTANCES_ROOT': str(versions.PRIVATE.parent),
        'DH_JOB_DIR': str(directory),
        'DH_SETTINGS_FILE': str(snapshot / 'settings.snapshot.yaml')})


def complete_saved_features(directory, output, workers, *, generation=None):
    """Extract only a complete frozen recall checkpoint, without history loading.

    generation: 'few_shot' 旧代快照走归档自带 src/generator/history_sources.py；
    'fewshot' 新代快照走 src/history/sources.py；None 默认新包（活树 shim 与
    新包同源，保持旧调用方在当前树进程中的行为）。两代路径由 _run_legacy /
    _run_modern 显式注入。
    """
    from src.config import sha256_file
    if generation in (None, 'fewshot'):
        from src.history.sources import digest, require
    elif generation == 'few_shot':
        from src.generator.history_sources import digest, require
    else:
        raise ValueError(f'unknown snapshot generation: {generation}')
    from src.iteration import control, experiment, learned_gen, learning_guard, runtime, versions
    from src.iteration.parallel import worker_limit
    from src.iteration.storage import file_lock, read_json, write_json

    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(output.is_dir() and output != directory, 'Existing feature output is required')
    with file_lock(output / '.run.lock', blocking=False), \
            file_lock(directory / '.run.lock', blocking=False), control.job(output):
        control.check(directory)
        spec = experiment.spec_of(directory)
        state = read_json(output / 'state.json')
        manifest = read_json(output / 'manifest.json')
        candidate = spec['candidate_ref']
        require(spec['kind'] == 'gen_ab' and state.get('experiment') == directory.name
                and state.get('candidate') == candidate and manifest.get('data') == spec['data_ref']
                and manifest.get('base') == spec['baseline_ref']
                and manifest.get('name') == state.get('branch')
                and manifest.get('policy') == learned_gen.POLICY,
                'Feature output does not match the frozen experiment')
        materials = spec.get('learning_snapshot', {}).get('materials', {})
        require(str(versions.generator_dir(candidate).resolve()) in materials,
                'Frozen candidate materials are missing')
        checkpoint = output / 'features/recall_tasks.json'
        files = [directory / 'spec.json', output / 'manifest.json', checkpoint,
                 *runtime.require_current(directory)]
        data_dir = versions.data_version_dir(spec['data_ref'])
        seal = learning_guard.RunSeal([data_dir, *materials], files)
        for name, expected in materials.items():
            material = Path(name)
            actual = {str(path.relative_to(material)): sha256_file(path)
                      for path in material.rglob('*')
                      if path.is_file() and path.relative_to(material) != Path('meta.json')}
            require(actual == expected, 'Frozen feature materials changed')
        switches = versions.load_generator(candidate)['config'].get('retriever', {})
        selector = learned_gen.LearnedSelector(versions.generator_dir(candidate) / 'ranker',
            versions.PRIVATE / '.cache/fewshot_ranker_features')
        cases = learned_gen.datasets.rows_for(spec)
        binding, _, _ = learned_gen.recall_binding(spec, cases, switches, selector, data_dir)
        saved = read_json(checkpoint)
        require(saved.get('binding') == binding, 'Frozen recall checkpoint binding differs')
        require(saved.get('payload_sha256') == digest({key: value for key, value in saved.items()
                                                     if key != 'payload_sha256'}),
                'Recall checkpoint content changed')
        tasks = saved.get('tasks')
        require(type(saved.get('completed')) is int and saved['completed'] == len(cases)
                and isinstance(tasks, dict) and all(key == digest(request)
                    and request.get('client') == selector.client.cache_identity()
                    for key, request in tasks.items()), 'Complete frozen recall checkpoint required')
        limit = control.policy().get('max_active_requests', 16)
        require(type(workers) is int and 1 <= workers <= limit, 'Invalid feature worker count')
        original_check = control.check

        def check(directory=None):
            original_check(directory)
            original_check(output)
            original_check(experiment_directory)

        class CheckedClient:
            def cache_identity(self):
                return selector.client.cache_identity()

            def run(self, *args, **kwargs):
                seal.check()
                result = selector.client.run(*args, **kwargs)
                seal.check()
                return result

        experiment_directory = directory
        control.check = check
        try:
            seal.check()
            state.pop('error', None)
            state.update(status='precomputing', pid=os.getpid(), updated_at=time.time())
            write_json(output / 'state.json', state)
            with worker_limit(limit):
                learned_gen.extraction.run(tasks, selector.cache, output / 'features',
                                          CheckedClient(), workers=workers)
            seal.check()
            state.update(status='features_ready', updated_at=time.time())
            write_json(output / 'state.json', state)
        except Exception as exc:
            state.update(status='needs_attention', error=f'{type(exc).__name__}: {exc}',
                         updated_at=time.time())
            write_json(output / 'state.json', state)
            raise
        finally:
            control.check = original_check


def promote_saved(directory):
    """Run the frozen promotion gates, without preparing or evaluating a job."""
    from src.config import ConfigError
    from src.iteration import experiment, promote, versions
    kind = experiment.spec_of(directory)['kind']
    functions = {'gen_ab': promote.promote_gen, 'judge_eval': promote.promote_judge}
    if kind not in functions:
        raise ConfigError('unsupported experiment kind for frozen promotion')
    previous = versions.load_pointers()['production_gen']
    pointers = functions[kind](Path(directory).name)
    print('指针:', pointers, flush=True)
    if kind == 'gen_ab' and pointers['production_gen'] != previous:
        print('生产生成器已更新；Judge 保持原版本', flush=True)
    return pointers


def load_external_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def authenticate_snapshot(snapshot):
    module = load_external_module('_digital_human_runtime_auth',
                                  Path(__file__).with_name('runtime_auth.py'))
    return module.authenticate(snapshot)


def snapshot_source_imports():
    module = load_external_module('_digital_human_runtime_source_imports',
                                  Path(__file__).with_name('runtime_auth.py'))
    return module.source_only_imports()


@contextmanager
def authenticated_snapshot_imports(snapshot):
    authenticate_snapshot(snapshot)
    previous_path = sys.path.copy()
    with snapshot_source_imports():
        sys.path.insert(0, str(snapshot))
        try:
            yield
        finally:
            sys.path[:] = previous_path


def install_judge_transport(snapshot, directory, driver):
    from src.judge import corrected
    adapter = load_external_module('_digital_human_codex_transport',
        Path(__file__).resolve().parents[1] / 'judge' / 'codex_transport.py')
    authentication = load_external_module('_digital_human_runtime_auth_receipt',
                                         Path(__file__).with_name('runtime_auth.py'))
    adapter.install(corrected, expected_root=snapshot,
        receipt_directory=directory / 'codex_transport_overrides', provenance={
            'bootstrap_sha256': authentication.file_digest(__file__),
            'archive_driver_sha256': authentication.file_digest(driver.__file__),
            'snapshot_bytecode_policy': 'empty_external_cache_prefix_v1',
            'runtime_ref': snapshot.name})


def install_modern_recall_transport(snapshot, directory, driver):
    from src.config import ConfigError
    from src.generator import fewshot, shared_history
    from src.iteration.storage import write_once_json
    if any(not Path(module.__file__).resolve().is_relative_to(snapshot.resolve())
           for module in (fewshot, shared_history)):
        raise ConfigError('modern recall transport requires authenticated archive modules')
    reuse = live_retrieval_cache()
    authentication = load_external_module('_digital_human_recall_auth',
                                         Path(__file__).with_name('runtime_auth.py'))
    receipt = dict(schema=1, kind='modern_exact_recall_transport', runtime_ref=snapshot.name,
        bootstrap_sha256=authentication.file_digest(__file__),
        archive_driver_sha256=authentication.file_digest(driver.__file__),
        recall_reuse_sha256=authentication.file_digest(reuse.__file__),
        recall_reuse_mode=reuse.MODE, capacity=reuse.CAPACITY, max_bytes=reuse.MAX_BYTES,
        source_guards='pool_and_history_before_and_after_including_hits_v1',
        model_timing='recorded_transport_attempts_v1', created_at=time.time(), pid=os.getpid())
    write_once_json(directory / 'recall_transport_overrides' / (uuid.uuid4().hex + '.json'), receipt)

    @contextmanager
    def instrument(profile, package):
        with fewshot_profile_instrument(profile, package), \
                reuse.exact_retrieval_calls(shared_history, profile=profile):
            yield

    driver.fewshot_profile_instrument = instrument


@contextmanager
def generation_routes(snapshot, directory, driver):
    if not os.environ.get('DH_LLM_DIRECT_HOSTS'):
        yield
        return
    adapter = load_external_module('_digital_human_generation_transport',
                                   Path(__file__).with_name('generation_transport.py'))
    authentication = load_external_module('_digital_human_generation_auth',
                                         Path(__file__).with_name('runtime_auth.py'))
    with adapter.worker_routes(snapshot, directory, provenance={
            'bootstrap_sha256': authentication.file_digest(__file__),
            'archive_driver_sha256': authentication.file_digest(driver.__file__),
            'generation_transport_sha256': authentication.file_digest(adapter.__file__)}):
        yield


def dispatch_worker(args, snapshot, directory, generation):
    if generation == 'fewshot':
        driver = load_external_module('_digital_human_authenticated_transport',
                                     snapshot / 'src/iteration/pack_transport.py')
        if args.timeout_seconds == 0:
            driver.timeout_floor = worker_timeout_floor
        install_judge_transport(snapshot, directory, driver)
        install_modern_recall_transport(snapshot, directory, driver)
        with generation_routes(snapshot, directory, driver):
            return driver._run_modern(args, snapshot, directory)
    if os.environ.get('DH_LLM_DIRECT_HOSTS'):
        raise ValueError('generation direct routes require the authenticated modern worker')
    return _run_legacy(args, snapshot, directory)


def main():
    # A long-running parent may have imported an older launch function. Resolve
    # the default here too, before frozen load_settings() looks under its ROOT.
    os.environ.setdefault('DH_ENV_FILE', connection_env_file())
    # Bound native model-loader threads before importing a frozen executor.
    # Request concurrency is controlled separately by --workers.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import faulthandler
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--kind', choices=['pack', 'experiment', 'training', 'promotion', 'features'], default='pack')
    parser.add_argument('--feature-output', type=Path)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=timeout_value, default=0,
                        help='Optional wait floor; omitted preserves frozen client timeouts')
    parser.add_argument('--time-slice-seconds', type=time_slice_value)
    parser.add_argument('--unattempted-first', action='store_true',
                        help='Run untouched experiment cases before failed retries within a time slice')
    args = parser.parse_args()
    if args.unattempted_first and (args.kind != 'experiment' or args.time_slice_seconds is None):
        parser.error('--unattempted-first requires an experiment time slice')
    if (args.kind == 'features') != (args.feature_output is not None):
        parser.error('features operation requires --feature-output, exclusively')
    if args.time_slice_seconds is not None and args.kind != 'experiment':
        parser.error('time slices require an experiment worker')
    snapshot = Path(args.snapshot).resolve()
    with authenticated_snapshot_imports(snapshot):
        from src.config import valid_name
        from src.iteration import runtime, versions
        versions.switch_instance(valid_name(args.instance))
        from src.iteration import jobs
        directory = jobs.directory({'kind': 'experiment' if args.kind in {'promotion', 'features'} else args.kind,
                                    'id': valid_name(args.job)})
        verified = runtime.verify(directory / 'runtime.json')
        if verified.resolve() != snapshot:
            raise RuntimeError('timeout worker must use the recorded frozen runtime')
        runtime.require_current(directory)
        dispatch_worker(args, snapshot, directory, snapshot_generation(verified, runtime))


def _run_legacy(args, snapshot, directory):
    """Legacy snapshot generation: the original assembly, kept verbatim (§2.3a)."""
    from src.config import sha256_file
    from src.generator import few_shot
    from src.iteration import gbdt_evidence, runtime
    from src.iteration.storage import read_json, write_once_json
    from src.llm import ChatClient
    needs_history = args.kind != 'features' and needs_history_memory(directory, args.kind)
    disk_history = live_disk_history() if needs_history else None
    history_storage = disk_history.disk_storage(few_shot) if needs_history else nullcontext()
    profile_module = live_transport_profile()
    feature_validation = (snapshot / 'src/generator/fewshot_ranker/supplemental.py').is_file()
    retrieval_cache = live_retrieval_cache() if needs_history and args.kind not in {'features', 'promotion'} else None
    source_reader = live_disk_source_reads() if needs_history and args.kind not in {'features', 'promotion'} else None
    material_reuse = live_material_reuse() if args.kind not in {'features', 'promotion'} else None
    receipt = {'schema': 1, 'kind': 'transport_timeout_floor',
               'operation': args.kind,
               'timeout_seconds': args.timeout_seconds, 'workers': args.workers,
               'historical_code_path_compatibility': 'verified_runtime_and_gbdt_archives_v2',
               'history_feature_cache': 'exact_text_and_flags_copy_v1',
               'history_feature_cache_capacity': 256,
               'history_term_storage': 'intern_equal_terms_v1',
               'history_feature_storage': 'shared_equal_results_v1',
               'history_storage': disk_history.MODE if needs_history else 'not_required',
               'history_storage_adapter_sha256': sha256_file(Path(disk_history.__file__)) if needs_history else None,
               'history_storage_cache_capacity': disk_history.CAPACITY if needs_history else None,
               'transport_profile': profile_module.MODE if args.kind not in {'features', 'promotion'} else 'not_required',
               'transport_profile_sha256': sha256_file(Path(profile_module.__file__)),
               'feature_response_cache_validation': 'frozen_concern_evidence_v2' if feature_validation else 'not_required',
               'history_retrieval_cache': retrieval_cache.MODE if retrieval_cache else 'not_required',
               'history_retrieval_cache_sha256': sha256_file(Path(retrieval_cache.__file__)) if retrieval_cache else None,
               'history_retrieval_cache_capacity': retrieval_cache.CAPACITY if retrieval_cache else None,
               'history_retrieval_cache_max_bytes': retrieval_cache.MAX_BYTES if retrieval_cache else None,
               'history_source_reads': source_reader.MODE if source_reader else 'not_required',
               'history_source_reads_sha256': sha256_file(Path(source_reader.__file__)) if source_reader else None,
               'history_source_offset_capacity': source_reader.CAPACITY if source_reader else None,
               'history_source_record_capacity_per_worker': source_reader.RECORD_CAPACITY if source_reader else None,
               'history_source_sqlite_cache_kib_per_worker': source_reader.SQLITE_CACHE_KIB if source_reader else None,
               'material_validation_reuse': material_reuse.MODE if material_reuse else 'not_required',
               'material_validation_reuse_sha256': sha256_file(Path(material_reuse.__file__)) if material_reuse else None,
               'history_memory_lane': 'bounded_archive_slots_legacy_lock_v2',
               'history_memory_capacity': history_memory_capacity(),
               'history_memory_required': needs_history,
               'time_slice_seconds': args.time_slice_seconds,
               'pending_case_order': 'unattempted_before_retries_v1' if args.unattempted_first else 'original',
               'time_slice_clock': 'first_operation_full_warmup_v2',
               'runtime': read_json(directory / 'runtime.json'),
               'launcher_sha256': sha256_file(Path(__file__)),
               'created_at': time.time(), 'pid': os.getpid()}
    write_once_json(directory / 'transport_overrides' / (uuid.uuid4().hex + '.json'), receipt)
    if args.kind == 'features':
        complete_saved_features(directory, args.feature_output, args.workers, generation='few_shot')
        return
    if args.kind == 'promotion':
        print('Frozen promotion: reusing saved results; all original adoption gates retained', flush=True)
        with history_memory_lane(directory, required=needs_history), archived_runtime_paths(runtime), \
                archived_gbdt_paths(gbdt_evidence, runtime), history_feature_cache(few_shot), history_storage:
            promote_saved(directory)
        return
    print(f'Frozen {args.kind} resume: request timeout floor={args.timeout_seconds:g}s; '
          'original requests and saved rows retained', flush=True)
    sys.argv = [str(snapshot / 'scripts/iterate_branches.py'), '--instance', args.instance,
                'worker', '--kind', args.kind, '--job', args.job, '--workers', str(args.workers)]
    from src.iteration import control
    try:
        profile = profile_module.TransportProfile(directory / 'transport_profiles' / (uuid.uuid4().hex + '.json'),
            dict(job=args.job, kind=args.kind, workers=args.workers,
                 timeout_seconds=args.timeout_seconds, time_slice_seconds=args.time_slice_seconds))
        from src.generator import shared_history
        retrieval_reuse = retrieval_cache.exact_retrieval_calls(shared_history, profile=profile) if retrieval_cache else nullcontext()
        source_read_reuse = source_reader.source_reads(disk_history, retrievers=few_shot,
                                                      profile=profile) if source_reader else nullcontext()
        from src.iteration import learning_guard
        material_validation_reuse = material_reuse.verified_materials(learning_guard, profile=profile,
                                                                     disk_rows=disk_history)
        feature_response_reuse = feature_response_cache_validation() if feature_validation else nullcontext()
        with profile, profile.admission(history_memory_lane(directory, required=needs_history), 'history_lane_wait'), \
                worker_timeout_floor(ChatClient, args.timeout_seconds), \
                archived_runtime_paths(runtime), archived_gbdt_paths(gbdt_evidence, runtime), \
                history_feature_cache(few_shot), history_storage, source_read_reuse, retrieval_reuse, \
                feature_response_reuse, material_validation_reuse, \
                profile.instrument(history_required=needs_history, disk_history=disk_history), \
                case_time_slice(args.time_slice_seconds, profile=profile,
                                unattempted_results=directory / 'cases.jsonl' if args.unattempted_first else None):
            runpy.run_path(sys.argv[0], run_name='__main__')
    except control.StopRequested as exc:
        if exc.reason != 'time_slice_complete':
            raise
        print('Time slice complete; all in-flight cases persisted; safe to resume', flush=True)
        raise SystemExit(75)


def _run_modern(args, snapshot, directory):
    """Modern snapshot generation: the archive is self-contained (fewshot package
    and materials package ship inside it), so no live adapter layer is loaded.
    Only the hooks the new code needs are assembled, mirroring the legacy
    with-stack semantics (profile admission, history lane, timeout floor,
    archived evidence paths, case time slice). All imports resolve against the
    archived snapshot via sys.path and stay inside this branch."""
    from src.config import sha256_file
    from src.generator import fewshot
    from src.iteration import control, gbdt_evidence, runtime, transport_profile
    from src.iteration.materials import validation
    from src.iteration.storage import read_json, write_once_json
    from src.llm import ChatClient
    needs_history = args.kind != 'features' and needs_history_memory(directory, args.kind)
    feature_validation = (snapshot / 'src/generator/fewshot_ranker/supplemental.py').is_file()
    receipt = {'schema': 1, 'kind': 'transport_timeout_floor',
               'operation': args.kind,
               'timeout_seconds': args.timeout_seconds, 'workers': args.workers,
               'snapshot_generation': 'fewshot_package_v1',
               'fewshot_package_sha256': sha256_file(Path(fewshot.__file__)),
               'material_validation': validation.MODE,
               'material_validation_sha256': sha256_file(Path(validation.__file__)),
               'history_feature_cache': 'exact_text_and_flags_copy_v1',
               'history_feature_cache_target': 'fewshot.rank',
               'history_memory_lane': 'bounded_archive_slots_legacy_lock_v2',
               'history_memory_capacity': history_memory_capacity(),
               'history_memory_required': needs_history,
               'time_slice_seconds': args.time_slice_seconds,
               'pending_case_order': 'unattempted_before_retries_v1' if args.unattempted_first else 'original',
               'time_slice_clock': 'first_operation_full_warmup_v2',
               'runtime': read_json(directory / 'runtime.json'),
               'launcher_sha256': sha256_file(Path(__file__)),
               'created_at': time.time(), 'pid': os.getpid()}
    write_once_json(directory / 'transport_overrides' / (uuid.uuid4().hex + '.json'), receipt)
    if args.kind == 'features':
        complete_saved_features(directory, args.feature_output, args.workers, generation='fewshot')
        return
    if args.kind == 'promotion':
        print('Frozen promotion: reusing saved results; all original adoption gates retained', flush=True)
        with history_memory_lane(directory, required=needs_history), archived_runtime_paths(runtime), \
                archived_gbdt_paths(gbdt_evidence, runtime):
            promote_saved(directory)
        return
    print(f'Frozen {args.kind} resume: request timeout floor={args.timeout_seconds:g}s; '
          'original requests and saved rows retained', flush=True)
    sys.argv = [str(snapshot / 'scripts/iterate_branches.py'), '--instance', args.instance,
                'worker', '--kind', args.kind, '--job', args.job, '--workers', str(args.workers)]
    try:
        profile = transport_profile.TransportProfile(directory / 'transport_profiles' / (uuid.uuid4().hex + '.json'),
            dict(job=args.job, kind=args.kind, workers=args.workers,
                 timeout_seconds=args.timeout_seconds, time_slice_seconds=args.time_slice_seconds))
        material_validation_reuse = validation.verified_materials(profile=profile)
        feature_response_reuse = feature_response_cache_validation() if feature_validation else nullcontext()
        with profile, profile.admission(history_memory_lane(directory, required=needs_history), 'history_lane_wait'), \
                worker_timeout_floor(ChatClient, args.timeout_seconds), \
                archived_runtime_paths(runtime), archived_gbdt_paths(gbdt_evidence, runtime), \
                history_feature_cache(fewshot.rank), feature_response_reuse, material_validation_reuse, \
                fewshot_profile_instrument(profile, fewshot), \
                case_time_slice(args.time_slice_seconds, profile=profile,
                                unattempted_results=directory / 'cases.jsonl' if args.unattempted_first else None):
            runpy.run_path(sys.argv[0], run_name='__main__')
    except control.StopRequested as exc:
        if exc.reason != 'time_slice_complete':
            raise
        print('Time slice complete; all in-flight cases persisted; safe to resume', flush=True)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
