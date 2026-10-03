"""Snapshot generation dispatch for pack_transport (design §2.3a)."""
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import ConfigError
from src.iteration import pack_transport, runtime
from test_leakage_guards import env as env


LEGACY_MANIFEST = {'files': {'src/generator/few_shot.py': 'ab' * 32}}
MODERN_MANIFEST = {'files': {'src/generator/fewshot/__init__.py': 'cd' * 32,
                             'src/iteration/materials/validation.py': 'ef' * 32}}


def write_manifest(snapshot, manifest):
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'manifest.json').write_text(json.dumps(manifest))
    return snapshot


def test_snapshot_generation_classifies_legacy_and_modern(tmp_path):
    legacy = write_manifest(tmp_path / 'legacy', LEGACY_MANIFEST)
    modern = write_manifest(tmp_path / 'modern', MODERN_MANIFEST)
    assert pack_transport.snapshot_generation(legacy, runtime) == 'few_shot'
    assert pack_transport.snapshot_generation(modern, runtime) == 'fewshot'


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


def prepare_main(monkeypatch, env, tmp_path, snapshot, argv):
    import faulthandler
    from src.iteration import jobs
    from src.iteration.storage import write_json
    directory = env.root / 'experiments/synthetic'
    monkeypatch.setenv('DH_INSTANCES_ROOT', str(env.root.parent))
    write_json(directory / 'spec.json', {'kind': 'gen_ab'})
    write_json(directory / 'runtime.json', {'frozen': 'unchanged'})
    monkeypatch.setattr(jobs, 'directory', lambda job: directory)
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    monkeypatch.setattr(runtime, 'require_current', lambda path: None)
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


def test_fewshot_profile_instrument_times_and_restores():
    class FakeSession:
        def is_approved(self):
            return True

        def retrieve(self):
            return 'rows'

        def render_selected(self, rows):
            return rows

    fake = SimpleNamespace(RecallSession=FakeSession)
    originals = {name: getattr(FakeSession, name) for name in ('is_approved', 'retrieve', 'render_selected')}
    profile = DummyProfile()
    with pack_transport.fewshot_profile_instrument(profile, fake):
        assert FakeSession().retrieve() == 'rows'
        assert FakeSession().render_selected('rows') == 'rows'
        assert FakeSession().is_approved() is True
    assert profile.spans == ['retrieval_compute', 'retrieval_render', 'history_pool_approval']
    assert all(getattr(FakeSession, name) is original for name, original in originals.items())


def test_launch_modern_snapshot_runs_its_own_transport(tmp_path, monkeypatch):
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
    assert Path(argv[3]).resolve() == (snapshot / 'src/iteration/pack_transport.py').resolve()
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
    assert Path(argv[3]).resolve() == (snapshot / 'src' / 'iteration' / 'pack_transport.py').resolve()
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
