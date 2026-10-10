"""Snapshot generation dispatch for pack_transport (design §2.3a)."""
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.infrastructure.config import ConfigError
from src.iteration import pack_transport, runtime
from test_leakage_guards import env as env


LEGACY_MANIFEST = {'files': {'src/generator/few_shot.py': 'ab' * 32}}
MODERN_MANIFEST = {'files': {'src/generator/fewshot/__init__.py': 'cd' * 32,
                             'src/iteration/materials/validation.py': 'ef' * 32}}


@pytest.fixture(autouse=True)
def bootstrap_environment(monkeypatch):
    monkeypatch.delenv('DH_ENV_FILE', raising=False)
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)


def write_manifest(snapshot, manifest):
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'manifest.json').write_text(json.dumps(manifest))
    return snapshot


def test_snapshot_generation_classifies_legacy_and_modern(tmp_path):
    legacy = write_manifest(tmp_path / 'legacy', LEGACY_MANIFEST)
    modern = write_manifest(tmp_path / 'modern', MODERN_MANIFEST)
    assert pack_transport.snapshot_generation(legacy, runtime) == 'few_shot'
    assert pack_transport.snapshot_generation(modern, runtime) == 'fewshot'


@pytest.mark.parametrize('package', ('fewshot_ranker', 'fewshot'))
def test_snapshot_scoring_hooks_follow_authenticated_manifest(tmp_path, package):
    path = 'src/generator/' + package + '/supplemental.py'
    snapshot = write_manifest(tmp_path, {'files': {path: 'ab' * 32}})
    assert pack_transport.snapshot_feature_module(snapshot, runtime) == 'src.generator.' + package + '.supplemental'


def test_snapshot_scoring_ignores_unlisted_files_and_rejects_mixed_manifest(tmp_path):
    snapshot = write_manifest(tmp_path, {'files': {}})
    path = snapshot / 'src/generator/fewshot/supplemental.py'
    path.parent.mkdir(parents=True)
    path.write_text('not authenticated')
    assert pack_transport.snapshot_feature_module(snapshot, runtime) is None
    write_manifest(snapshot, {'files': {
        'src/generator/fewshot/supplemental.py': 'ab' * 32,
        'src/generator/fewshot_ranker/supplemental.py': 'cd' * 32}})
    with pytest.raises(ConfigError, match='mixed implementations'):
        pack_transport.snapshot_feature_module(snapshot, runtime)


@pytest.mark.parametrize('manifest', ({}, {'files': []}))
def test_snapshot_scoring_rejects_invalid_manifest(tmp_path, manifest):
    snapshot = write_manifest(tmp_path, manifest)
    with pytest.raises(ConfigError, match='cannot classify snapshot scoring'):
        pack_transport.snapshot_feature_module(snapshot, runtime)


def test_modern_dispatch_calls_authenticated_archive_driver(tmp_path, monkeypatch):
    events = []
    driver = SimpleNamespace(_run_modern=lambda *args: events.append('archived_worker'))
    monkeypatch.setattr(pack_transport, 'load_external_module', lambda name, path:
                        events.append(('archive_load', path)) or driver)
    monkeypatch.setattr(pack_transport, 'install_judge_transport', lambda *args:
                        events.append('transport_installed'))
    monkeypatch.setattr(pack_transport, 'install_modern_recall_transport', lambda *args:
                        events.append('recall_installed'))
    monkeypatch.setattr(pack_transport, '_run_modern', lambda *args: pytest.fail('not live worker'))
    pack_transport.dispatch_worker(SimpleNamespace(timeout_seconds=300), tmp_path, tmp_path / 'job', 'fewshot')
    assert events == [('archive_load', tmp_path / 'src/iteration/pack_transport.py'),
                      'transport_installed', 'recall_installed', 'archived_worker']


def test_default_modern_worker_installs_provenance_and_preserves_timeouts(tmp_path, monkeypatch):
    from src.judge import corrected
    from src.iteration.runtime_auth import file_digest
    original_init = type('Client', (), {'__init__': lambda self: None}).__init__
    client = type('FrozenClient', (), {'__init__': original_init})
    target = tmp_path / 'src/judge/corrected.py'
    target.parent.mkdir(parents=True)
    target.write_text('authenticated = True\n')
    driver_path = tmp_path / 'src/iteration/pack_transport.py'
    driver_path.parent.mkdir(parents=True)
    driver_path.write_text('authenticated = True\n')
    monkeypatch.setattr(corrected, '__file__', str(target))
    monkeypatch.setattr(corrected, 'subprocess', SimpleNamespace())
    monkeypatch.setattr(pack_transport, 'install_modern_recall_transport', lambda *args: None)

    def run(args, snapshot, directory):
        assert corrected.subprocess._provenance == {
            'bootstrap_sha256': file_digest(pack_transport.__file__),
            'archive_driver_sha256': file_digest(driver_path),
            'snapshot_bytecode_policy': 'empty_external_cache_prefix_v1',
            'runtime_ref': tmp_path.name}
        assert corrected.subprocess._receipt_directory == directory / 'codex_transport_overrides'
        with driver.timeout_floor(client, args.timeout_seconds):
            assert client.__init__ is original_init
        return 'default-worker'

    driver = SimpleNamespace(__file__=str(driver_path), _run_modern=run)
    original_load = pack_transport.load_external_module
    preflights = []

    def load(name, path):
        if path == driver_path:
            return driver
        module = original_load(name, path)
        if name == '_digital_human_codex_transport':
            monkeypatch.setattr(module, 'settings', lambda: preflights.append('checked'))
        return module

    monkeypatch.setattr(pack_transport, 'load_external_module', load)
    assert pack_transport.dispatch_worker(SimpleNamespace(timeout_seconds=0), tmp_path,
        tmp_path / 'job', 'fewshot') == 'default-worker'
    assert preflights == ['checked']


def test_transport_preflight_stops_before_archived_worker(tmp_path, monkeypatch):
    driver = SimpleNamespace(_run_modern=lambda *args: pytest.fail('worker must not start'))

    def preflight():
        raise RuntimeError('transport configuration unavailable')

    adapter = SimpleNamespace(settings=preflight,
        install=lambda *args, **kwargs: pytest.fail('transport must not install'))
    monkeypatch.setattr(pack_transport, 'load_external_module', lambda name, path:
                        adapter if name == '_digital_human_codex_transport' else driver)
    monkeypatch.setattr(pack_transport, 'install_modern_recall_transport', lambda *args:
                        pytest.fail('retrieval preparation must not start'))
    with pytest.raises(RuntimeError, match='transport configuration unavailable'):
        pack_transport.dispatch_worker(SimpleNamespace(timeout_seconds=0), tmp_path,
                                       tmp_path / 'job', 'fewshot')


def test_saved_reply_judge_resume_does_not_install_live_recall_adapter(tmp_path, monkeypatch):
    directory = tmp_path / 'job'
    directory.mkdir()
    (directory / 'spec.json').write_text(json.dumps({'kind': 'judge_eval'}))
    events = []
    driver = SimpleNamespace(_run_modern=lambda *args: events.append('archived_worker'))
    monkeypatch.setattr(pack_transport, 'load_external_module', lambda *args: driver)
    monkeypatch.setattr(pack_transport, 'install_judge_transport',
                        lambda *args: events.append('judge_transport'))
    monkeypatch.setattr(pack_transport, 'install_modern_recall_transport',
                        lambda *args: pytest.fail('saved replies do not retrieve history'))
    monkeypatch.delenv('DH_LLM_DIRECT_HOSTS', raising=False)
    pack_transport.dispatch_worker(SimpleNamespace(kind='experiment', timeout_seconds=0),
                                   tmp_path, directory, 'fewshot')
    assert events == ['judge_transport', 'archived_worker']


def test_snapshot_generation_accepts_legacy_nested_prefix(tmp_path):
    nested = write_manifest(tmp_path / 'nested',
                            {'files': {'src/digital_human/generator/few_shot.py': 'ab' * 32}})
    assert pack_transport.snapshot_generation(nested, runtime) == 'few_shot'


@pytest.mark.parametrize('manifest', [
    {'files': {'src/generator/few_shot.py': 'ab' * 32, 'src/generator/fewshot/__init__.py': 'cd' * 32}},
    {'files': {'src/generator/other.py': 'ab' * 32}},
    {},
])
def test_snapshot_generation_rejects_ambiguous_or_missing(tmp_path, manifest):
    snapshot = write_manifest(tmp_path / str(abs(hash(json.dumps(manifest, sort_keys=True)))), manifest)
    with pytest.raises(ConfigError, match='cannot classify snapshot generation'):
        pack_transport.snapshot_generation(snapshot, runtime)


def test_snapshot_generation_rejects_absent_manifest(tmp_path):
    with pytest.raises(ConfigError, match='cannot classify snapshot generation'):
        pack_transport.snapshot_generation(tmp_path, runtime)


class DummyProfile:
    def __init__(self):
        self.spans = []
        self.admitted = []
        self.records = []

    def record(self, name, seconds):
        self.records.append((name, seconds))

    @contextmanager
    def span(self, name):
        self.spans.append(name)
        yield

    def admission(self, context, name='request_admission'):
        self.admitted.append(name)
        return context

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def main_argv(snapshot, instance, job='synthetic', kind='experiment'):
    return ['pack_transport', '--snapshot', str(snapshot), '--instance', instance,
            '--job', job, '--kind', kind, '--workers', '1', '--timeout-seconds', '300']


def test_main_authenticates_before_snapshot_imports(tmp_path, monkeypatch):
    snapshot = tmp_path / 'untrusted'
    original_path = list(sys.path)
    monkeypatch.setattr(sys, 'argv', main_argv(snapshot, 'demo'))
    monkeypatch.setattr(pack_transport, 'authenticate_snapshot', lambda path:
                        (_ for _ in ()).throw(ValueError('untrusted archive')))
    monkeypatch.setattr(pack_transport, 'dispatch_worker', lambda *args:
                        pytest.fail('must reject before executing archive'))
    with pytest.raises(ValueError, match='untrusted archive'):
        pack_transport.main()
    assert sys.path == original_path


def prepare_main(monkeypatch, env, tmp_path, snapshot, argv):
    import faulthandler
    from src.iteration import jobs
    from src.infrastructure.storage import write_json
    directory = env.root / 'experiments/synthetic'
    monkeypatch.setenv('DH_INSTANCES_ROOT', str(env.root.parent))
    write_json(directory / 'spec.json', {'kind': 'gen_ab'})
    write_json(directory / 'runtime.json', {'frozen': 'unchanged'})
    monkeypatch.setattr(jobs, 'directory', lambda job: directory)
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    monkeypatch.setattr(runtime, 'require_current', lambda path: None)
    monkeypatch.setattr(pack_transport, 'authenticate_snapshot', lambda path: None)
    monkeypatch.setattr(pack_transport, 'snapshot_source_imports', nullcontext)
    monkeypatch.setattr(pack_transport, 'load_external_module', lambda name, path: pack_transport)
    monkeypatch.setattr(pack_transport, 'install_judge_transport', lambda *args: None)
    monkeypatch.setattr(pack_transport, 'install_modern_recall_transport', lambda *args: None)
    monkeypatch.setattr(faulthandler, 'enable', lambda: None)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(sys, 'argv', argv)
    return directory


def test_main_dispatches_legacy_manifest_to_legacy_runner(env, tmp_path, monkeypatch):
    snapshot = write_manifest(tmp_path / 'snap', LEGACY_MANIFEST)
    argv = main_argv(snapshot, env.root.name)
    directory = prepare_main(monkeypatch, env, tmp_path, snapshot, argv)
    calls = []
    monkeypatch.setattr(pack_transport, '_run_legacy',
                        lambda *args: calls.append(('legacy', args)))
    monkeypatch.setattr(pack_transport, '_run_modern',
                        lambda *args: pytest.fail('legacy manifest must not run the modern branch'))
    pack_transport.main()
    kind, (args, snap, directory_arg) = calls[0]
    assert kind == 'legacy' and snap == snapshot and directory_arg == directory
    assert args.job == 'synthetic' and args.kind == 'experiment'


def test_main_dispatches_modern_manifest_without_importing_few_shot(env, tmp_path, monkeypatch):
    snapshot = write_manifest(tmp_path / 'snap', MODERN_MANIFEST)
    argv = main_argv(snapshot, env.root.name)
    directory = prepare_main(monkeypatch, env, tmp_path, snapshot, argv)
    real_import = __import__

    def guarding_import(name, *args, **kwargs):
        fromlist = kwargs.get('fromlist') or (args[2] if len(args) > 2 else ())
        if name == 'src.generator.few_shot' or (name == 'src.generator'
                                                and 'few_shot' in (fromlist or ())):
            raise AssertionError('modern branch must not import few_shot')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', guarding_import)
    from src.generator import fewshot
    from src.iteration import transport_profile
    from src.iteration.materials import validation
    reuse = DummyProfile()
    profile = DummyProfile()
    entered = []
    monkeypatch.setattr(validation, 'verified_materials',
                        lambda **kwargs: entered.append(('materials', kwargs)) or contextmanager(lambda: (yield))())
    monkeypatch.setattr(transport_profile, 'TransportProfile', lambda *args, **kwargs: profile)
    monkeypatch.setattr(pack_transport, 'history_memory_lane',
                        lambda output, **kwargs: contextmanager(lambda: (yield))())
    monkeypatch.setattr(pack_transport, 'timeout_floor',
                        lambda client, seconds: contextmanager(lambda: (yield))())
    monkeypatch.setattr(pack_transport, 'archived_runtime_paths',
                        lambda module: contextmanager(lambda: (yield))())
    monkeypatch.setattr(pack_transport, 'archived_gbdt_paths',
                        lambda module, rt: contextmanager(lambda: (yield))())
    monkeypatch.setattr(pack_transport, 'case_time_slice',
                        lambda seconds, **kwargs: contextmanager(lambda: (yield))())
    original_retrieve = fewshot.RecallSession.retrieve
    executed = []

    def run_worker(path, **kwargs):
        assert path == str(snapshot / 'scripts/iterate_branches.py')
        assert fewshot.RecallSession.retrieve is not original_retrieve
        assert entered and entered[0][1]['profile'] is profile
        assert profile.admitted == ['history_lane_wait']
        executed.append(True)

    monkeypatch.setattr(pack_transport.runpy, 'run_path', run_worker)
    pack_transport.main()
    assert executed == [True]
    assert fewshot.RecallSession.retrieve is original_retrieve
    receipt, = [json.loads(path.read_text())
                for path in (directory / 'transport_overrides').glob('*.json')]
    assert receipt['snapshot_generation'] == 'fewshot_package_v1'
    assert receipt['operation'] == 'experiment'
    assert receipt['runtime'] == {'frozen': 'unchanged'}
    assert receipt['history_memory_required'] is True


@pytest.mark.parametrize('with_sparse_filter', [False, True])
def test_fewshot_profile_instrument_times_and_restores(with_sparse_filter):
    from src.infrastructure import tracing

    class FakeMemoryEngine:
        def excluded_ids(self):
            return set()

        def term_features(self):
            return None

    class FakeIndexedEngine:
        def excluded_ids(self):
            return set()

        def term_features(self):
            return {}

    class FakeSession:
        _engine = FakeMemoryEngine()

        def is_approved(self):
            return True

        def retrieve(self):
            self._engine.excluded_ids()
            self._candidate_rows()
            if with_sparse_filter:
                self._sparse_candidates()
            self._engine.term_features()
            fake.rank.score_and_select()
            return 'rows'

        def _candidate_rows(self):
            return 'candidates'

        def _sparse_candidates(self):
            return 'sparse candidates'

        def render_selected(self, rows):
            return rows

    if not with_sparse_filter:
        del FakeSession._sparse_candidates
    fake = SimpleNamespace(RecallSession=FakeSession, InMemoryRecallEngine=FakeMemoryEngine,
                           SqliteIndexedRecallEngine=FakeIndexedEngine,
                           rank=SimpleNamespace(score_and_select=lambda: 'rows'))
    owners = [(FakeSession, name) for name in ('is_approved', 'retrieve', 'render_selected', '_candidate_rows')]
    if with_sparse_filter:
        owners.append((FakeSession, '_sparse_candidates'))
    owners.extend((engine, name) for engine in (FakeMemoryEngine, FakeIndexedEngine)
                  for name in ('excluded_ids', 'term_features'))
    owners.append((fake.rank, 'score_and_select'))
    originals = [(owner, name, getattr(owner, name)) for owner, name in owners]
    original_step = tracing.step
    profile = DummyProfile()
    with pack_transport.fewshot_profile_instrument(profile, fake):
        assert FakeSession().retrieve() == 'rows'
        assert FakeSession().render_selected('rows') == 'rows'
        assert FakeSession().is_approved() is True
        with tracing.step('llm', {'body': {'model': 'same'}}) as event:
            assert event['request'] == {'body': {'model': 'same'}}
        assert event['status'] == 'ok'
        with pytest.raises(ValueError, match='same error'):
            with tracing.step('codex', {}) as failed_event:
                raise ValueError('same error')
        assert failed_event['status'] == 'failed'
        with tracing.step('other', {}):
            pass
    assert profile.spans == ['retrieval_compute', 'retrieval_excluded_ids', 'retrieval_candidate_selection'] + (
        ['retrieval_sparse_selection'] if with_sparse_filter else []
    ) + ['retrieval_term_features', 'retrieval_rank', 'retrieval_render', 'history_pool_approval',
         'model_llm', 'model_codex']
    assert profile.records == [('recall_engine_FakeMemoryEngine', 0.0)]
    assert tracing.step is original_step
    assert all(getattr(owner, name) is original for owner, name, original in originals)


@pytest.mark.parametrize('owner_name,method,metric', [
    ('RecallSession', '_candidate_rows', 'retrieval_candidate_selection'),
    ('RecallSession', '_sparse_candidates', 'retrieval_sparse_selection'),
    ('InMemoryRecallEngine', 'excluded_ids', 'retrieval_excluded_ids'),
    ('SqliteIndexedRecallEngine', 'excluded_ids', 'retrieval_excluded_ids'),
    ('InMemoryRecallEngine', 'term_features', 'retrieval_term_features'),
    ('SqliteIndexedRecallEngine', 'term_features', 'retrieval_term_features'),
    ('rank', 'score_and_select', 'retrieval_rank'),
])
def test_fewshot_profile_substage_errors_preserve_failure_and_restore(monkeypatch, owner_name, method, metric):
    from src.generator import fewshot

    owner = getattr(fewshot, owner_name)

    def fail():
        raise ValueError('original substage error')

    monkeypatch.setattr(owner, method, fail)
    profile = DummyProfile()
    with pytest.raises(ValueError, match='original substage error'):
        with pack_transport.fewshot_profile_instrument(profile, fewshot):
            getattr(owner, method)()
    assert profile.spans == [metric]
    assert getattr(owner, method) is fail


def test_modern_recall_transport_records_and_restores_authenticated_hooks(tmp_path, monkeypatch):
    from src.generator import fewshot
    from src.generator.fewshot import shared_history
    from src.iteration.runtime_auth import file_digest

    paths = [tmp_path / 'src/generator/fewshot/__init__.py',
             tmp_path / 'src/generator/fewshot/shared_history.py',
             tmp_path / 'src/iteration/pack_transport.py']
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('authenticated = True\n')
    monkeypatch.setattr(fewshot, '__file__', str(paths[0]))
    monkeypatch.setattr(shared_history, '__file__', str(paths[1]))
    driver = SimpleNamespace(__file__=str(paths[2]))
    original_retrieve = shared_history.PinnedRetriever.retrieve
    original_session = fewshot.RecallSession.retrieve
    directory = tmp_path / 'job'
    archive_bytes = {path: path.read_bytes() for path in paths}
    pack_transport.install_modern_recall_transport(tmp_path, directory, driver)
    receipt, = [json.loads(path.read_text())
                for path in (directory / 'recall_transport_overrides').glob('*.json')]
    reuse = pack_transport.live_retrieval_cache()
    assert receipt['runtime_ref'] == tmp_path.name
    assert receipt['recall_reuse_sha256'] == file_digest(reuse.__file__)
    assert receipt['archive_driver_sha256'] == file_digest(paths[2])
    assert receipt['capacity'] == reuse.CAPACITY
    assert receipt['max_bytes'] == reuse.MAX_BYTES
    assert Path(reuse.__file__).name == 'recall_cache.py'
    with pytest.raises(ValueError, match='restore'):
        with driver.fewshot_profile_instrument(DummyProfile(), fewshot):
            assert shared_history.PinnedRetriever.retrieve is not original_retrieve
            assert fewshot.RecallSession.retrieve is not original_session
            raise ValueError('restore')
    assert shared_history.PinnedRetriever.retrieve is original_retrieve
    assert fewshot.RecallSession.retrieve is original_session
    assert all(path.read_bytes() == content for path, content in archive_bytes.items())


@pytest.mark.parametrize('module_name', ['fewshot', 'shared_history'])
def test_modern_recall_transport_rejects_modules_outside_archive(tmp_path, monkeypatch, module_name):
    from src.generator import fewshot
    from src.generator.fewshot import shared_history
    for name, module in [('fewshot', fewshot), ('shared_history', shared_history)]:
        path = tmp_path / ('outside' if name == module_name else 'snapshot') / (name + '.py')
        monkeypatch.setattr(module, '__file__', str(path))
    directory = tmp_path / 'job'
    with pytest.raises(ConfigError, match='authenticated archive modules'):
        pack_transport.install_modern_recall_transport(tmp_path / 'snapshot', directory, SimpleNamespace())
    assert not directory.exists()


def test_launch_modern_snapshot_runs_authenticated_bootstrap(tmp_path, monkeypatch):
    snapshot = write_manifest(tmp_path / 'snap', MODERN_MANIFEST)
    # 真实路径结构：冻结入口位于归档 src/iteration/pack_transport.py（同 a5118947 布局）。
    (snapshot / 'src' / 'iteration').mkdir(parents=True)
    (snapshot / 'src/iteration/pack_transport.py').write_text('# frozen transport\n')
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    calls = []
    monkeypatch.setattr(pack_transport.subprocess, 'call',
                        lambda argv, **kwargs: calls.append((argv, kwargs)) or 0)
    assert pack_transport.launch(tmp_path / 'job', instance='example-agent', job='synthetic',
                                 workers=2, seconds=180) == 0
    argv, kwargs = calls[0]
    assert argv[argv.index('--snapshot') + 1] == str(snapshot)
    assert Path(argv[3]).resolve() == Path(pack_transport.__file__).resolve()
    assert kwargs['cwd'] == snapshot
    assert kwargs['env']['DH_SETTINGS_FILE'] == str(snapshot / 'settings.snapshot.yaml')


def test_launch_modern_snapshot_rejects_missing_snapshot_transport(tmp_path, monkeypatch):
    snapshot = write_manifest(tmp_path / 'snap', MODERN_MANIFEST)
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    monkeypatch.setattr(pack_transport.subprocess, 'call',
                        lambda *args, **kwargs: pytest.fail('must not launch'))
    with pytest.raises(ConfigError, match='missing src/iteration/pack_transport.py'):
        pack_transport.launch(tmp_path / 'job', instance='example-agent', job='synthetic',
                              workers=2, seconds=180)


def test_launch_real_legacy_archive_uses_live_transport(tmp_path, monkeypatch):
    """真实 a5118947 归档：判定为旧代（含 few_shot.py），入口用活树 transport。"""
    from _reference_few_shot import archive_root
    archive = archive_root()
    if not (archive / 'manifest.json').is_file():
        pytest.skip('本机认证归档不存在（干净 checkout / CI）')
    assert (archive / 'src' / 'generator' / 'few_shot.py').is_file()
    assert pack_transport.snapshot_generation(archive, runtime) == 'few_shot'
    monkeypatch.setattr(runtime, 'verify', lambda path: archive)
    calls = []
    monkeypatch.setattr(pack_transport.subprocess, 'call',
                        lambda argv, **kwargs: calls.append((argv, kwargs)) or 0)
    assert pack_transport.launch(tmp_path / 'job', instance='example-agent', job='synthetic',
                                 workers=2, seconds=180) == 0
    argv, kwargs = calls[0]
    assert Path(argv[3]).resolve() == Path(pack_transport.__file__).resolve()
    assert kwargs['cwd'] == archive


def test_launch_modern_micro_snapshot_resolves_archived_transport(tmp_path, monkeypatch):
    """手工构造含 fewshot 包 + src/iteration/pack_transport.py 的 manifest 结构，
    验证新代入口解析到归档自身路径（真实目录布局）。"""
    manifest = {'files': {'src/generator/fewshot/__init__.py': 'cd' * 32,
                          'src/iteration/pack_transport.py': 'ef' * 32}}
    snapshot = write_manifest(tmp_path / 'snap', manifest)
    (snapshot / 'src' / 'generator' / 'fewshot').mkdir(parents=True)
    (snapshot / 'src' / 'iteration').mkdir(parents=True)
    (snapshot / 'src' / 'generator' / 'fewshot' / '__init__.py').write_text('')
    (snapshot / 'src' / 'iteration' / 'pack_transport.py').write_text('# frozen transport\n')
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    calls = []
    monkeypatch.setattr(pack_transport.subprocess, 'call',
                        lambda argv, **kwargs: calls.append((argv, kwargs)) or 0)
    assert pack_transport.launch(tmp_path / 'job', instance='example-agent', job='synthetic',
                                 workers=2, seconds=180) == 0
    argv, kwargs = calls[0]
    assert Path(argv[3]).resolve() == Path(pack_transport.__file__).resolve()
    assert kwargs['cwd'] == snapshot


def test_launch_rejects_ambiguous_manifest(tmp_path, monkeypatch):
    snapshot = write_manifest(tmp_path / 'snap',
                              {'files': {'src/generator/few_shot.py': 'ab' * 32,
                                         'src/generator/fewshot/__init__.py': 'cd' * 32}})
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    monkeypatch.setattr(pack_transport.subprocess, 'call',
                        lambda *args, **kwargs: pytest.fail('must not launch'))
    with pytest.raises(ConfigError, match='cannot classify snapshot generation'):
        pack_transport.launch(tmp_path / 'job', instance='example-agent', job='synthetic',
                              workers=2, seconds=180)
