import json
import importlib
import importlib.util
import multiprocessing
import os
import py_compile
import subprocess
import sys

import pytest

from src.infrastructure import cache
from src.infrastructure.config import ConfigError
from src.iteration import acceptance, control, runtime, versions
from src.infrastructure.storage import write_json
from test_leakage_guards import env as env
from quota_worker import consume as _consume


def test_runtime_snapshot_survives_source_changes_and_rejects_tampering(env, tmp_path):
    source = tmp_path / 'framework'
    (source / 'src').mkdir(parents=True)
    code = source / 'src/module.py'
    code.write_text('value = 1\n')
    job = env.root / 'jobs/frozen'
    saved = runtime.freeze(job, root=source)
    code.write_text('value = 2\n')
    assert runtime.verify(job / 'runtime.json') == saved
    assert (saved / 'src/module.py').read_text() == 'value = 1\n'
    (saved / 'src/module.py').write_text('value = 3\n')
    with pytest.raises(ConfigError, match='source changed'):
        runtime.verify(job / 'runtime.json')


def test_preimport_authentication_rejects_unlisted_executable(env, tmp_path):
    from src.iteration.runtime_auth import authenticate
    source = tmp_path / 'framework'
    (source / 'src').mkdir(parents=True)
    (source / 'src/module.py').write_text('value = 1\n')
    saved = runtime.freeze(env.root / 'jobs/frozen', root=source)
    authenticate(saved)
    (saved / 'src/extra.py').write_text('raise RuntimeError("not authenticated")\n')
    with pytest.raises(ValueError, match='inventory changed'):
        authenticate(saved)


def test_preimport_authentication_rejects_unlisted_symlink_directory(env, tmp_path):
    from src.iteration.runtime_auth import authenticate
    source = tmp_path / 'framework'
    (source / 'src').mkdir(parents=True)
    (source / 'src/module.py').write_text('value = 1\n')
    saved = runtime.freeze(env.root / 'jobs/frozen', root=source)
    (saved / 'src/injected').symlink_to(source / 'src', target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        authenticate(saved)


@pytest.mark.parametrize('relative', ['unlisted.py', 'unlisted.pyc', 'unlisted.pyo',
                                     'src/unlisted.pyc', 'src/unlisted.pyo',
                                     'src/__pycache__/unlisted.py'])
def test_preimport_authentication_rejects_sourceless_executables(env, tmp_path, relative):
    from src.iteration.runtime_auth import authenticate
    source = tmp_path / 'framework'
    (source / 'src').mkdir(parents=True)
    (source / 'src/module.py').write_text('value = 1\n')
    saved = runtime.freeze(env.root / 'jobs/frozen', root=source)
    extra = saved / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b'unbound executable')
    with pytest.raises(ValueError, match='inventory changed'):
        authenticate(saved)


def poison_source_cache(path):
    path.write_text('value = 9\n')
    bytecode = py_compile.compile(str(path), doraise=True)
    cached_stat = path.stat()
    path.write_text('value = 1\n')
    os.utime(path, ns=(cached_stat.st_atime_ns, cached_stat.st_mtime_ns))
    return bytecode


@pytest.mark.parametrize('entrypoint', ['worker', 'baseline', 'profile', 'promotion'])
@pytest.mark.parametrize('changed', ['unchanged', 'source', 'root_source', 'root_bytecode',
                                   'root_symlink'])
def test_snapshot_entrypoints_authenticate_before_source_execution(env, tmp_path, monkeypatch,
                                                                 entrypoint, changed):
    from types import SimpleNamespace
    from scripts import promote as cli
    from src.iteration import jobs, pack_transport, retrieval_compare, retrieval_profile
    source = tmp_path / 'framework'
    (source / 'src').mkdir(parents=True)
    (source / 'src/source_only_fixture.py').write_text('value = 1\n')
    directory = env.root / 'experiments/source-only'
    snapshot = runtime.freeze(directory, root=source)
    path = snapshot / 'src/source_only_fixture.py'
    bytecode = runtime.Path(poison_source_cache(path))
    original_cache = bytecode.read_bytes()
    if changed == 'source':
        path.write_text('value = 2\n')
    elif changed != 'unchanged':
        injected = tmp_path / 'yaml.py'
        injected.write_text('raise RuntimeError("unlisted root code executed")\n')
        if changed == 'root_source':
            (snapshot / 'yaml.py').write_bytes(injected.read_bytes())
        elif changed == 'root_bytecode':
            py_compile.compile(str(injected), cfile=str(snapshot / 'yaml.pyc'), doraise=True)
        else:
            (snapshot / 'yaml.py').symlink_to(injected)
    previous = sys.path.copy(), sys.pycache_prefix, sys.dont_write_bytecode
    events = []

    def execute_source(*args):
        events.append('executed')
        assert sys.path[0] == str(snapshot)
        assert sys.pycache_prefix is not None and sys.dont_write_bytecode
        spec = importlib.util.spec_from_file_location('entrypoint_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.value == 1
        if entrypoint == 'worker':
            assert args[0].timeout_seconds == 0
        return 'source-result'

    if entrypoint == 'worker':
        monkeypatch.setattr(jobs, 'directory', lambda job: directory)
        monkeypatch.setattr(runtime, 'verify', lambda descriptor: snapshot)
        monkeypatch.setattr(runtime, 'require_current', lambda directory: None)
        monkeypatch.setattr(pack_transport, 'snapshot_generation', lambda *args: 'fewshot')
        monkeypatch.setattr(pack_transport, 'dispatch_worker', execute_source)
        monkeypatch.setattr(sys, 'argv', ['pack_transport', '--snapshot', str(snapshot),
            '--instance', env.root.name, '--job', 'source-only', '--kind', 'experiment', '--workers', '1'])
        invoke = pack_transport.main
    elif entrypoint == 'baseline':
        monkeypatch.setattr(retrieval_compare, '_authenticated_baseline', execute_source)
        invoke = lambda: retrieval_compare._worker_baseline(SimpleNamespace(archive=snapshot))
    elif entrypoint == 'profile':
        monkeypatch.setattr(retrieval_profile, '_frozen_authenticated', execute_source)
        invoke = lambda: retrieval_profile.frozen(SimpleNamespace(snapshot=snapshot))
    else:
        monkeypatch.setattr(cli, 'execute', execute_source)
        invoke = lambda: cli.main(['judge', '--exp', 'source-only',
            '--threshold-reason', 'explicit revised threshold', '--evidence-runtime', str(snapshot)])
    if changed != 'unchanged':
        reason = ('source changed' if changed == 'source' else
                  'symlinks' if changed == 'root_symlink' else 'inventory changed')
        with pytest.raises(ValueError, match=reason):
            invoke()
        assert events == []
    else:
        invoke()
        assert events == ['executed']
    assert (sys.path, sys.pycache_prefix, sys.dont_write_bytecode) == previous
    assert bytecode.read_bytes() == original_cache


@pytest.mark.parametrize('loader', ['import', 'spec'])
def test_snapshot_source_imports_never_execute_historical_bytecode(env, tmp_path, monkeypatch, loader):
    from src.iteration.runtime_auth import authenticate, source_only_imports
    source = tmp_path / 'framework'
    (source / 'src').mkdir(parents=True)
    (source / 'src/source_only_fixture.py').write_text('value = 1\n')
    saved = runtime.freeze(env.root / 'jobs/frozen', root=source)
    path = saved / 'src/source_only_fixture.py'
    bytecode = poison_source_cache(path)
    cached_bytes = runtime.Path(bytecode).read_bytes()
    authenticate(saved)
    spec = importlib.util.spec_from_file_location('poisoned_fixture', path)
    poisoned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(poisoned)
    assert poisoned.value == 9
    monkeypatch.syspath_prepend(str(path.parent))
    previous = sys.pycache_prefix, sys.dont_write_bytecode
    try:
        with source_only_imports():
            assert sys.dont_write_bytecode
            if loader == 'import':
                module = importlib.import_module('source_only_fixture')
            else:
                spec = importlib.util.spec_from_file_location('source_only_fixture', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            assert module.value == 1
        assert (sys.pycache_prefix, sys.dont_write_bytecode) == previous
        assert runtime.Path(bytecode).read_bytes() == cached_bytes
    finally:
        sys.modules.pop('source_only_fixture', None)


def test_external_runtime_driver_executes_source_not_matching_bytecode(tmp_path):
    from src.iteration.pack_transport import load_external_module
    path = tmp_path / 'driver.py'
    path.write_text('value = 1\n')
    poison_source_cache(path)
    assert load_external_module('authenticated_driver_fixture', path).value == 1


def test_archived_code_does_not_bless_changed_data_or_missing_evidence(env, tmp_path):
    source, data = tmp_path / 'old.py', tmp_path / 'samples.json'
    source.write_text('old = True\n')
    data.write_text('{"value": 1}')
    inputs = {str(p): runtime.sha256_file(p) for p in (source, data)}
    runtime.archive_inputs(inputs)
    source.write_text('old = False\n')
    assert runtime.verify_inputs(inputs)[0].read_text() == 'old = True\n'
    data.write_text('{"value": 2}')
    with pytest.raises(ConfigError):
        runtime.verify_inputs(inputs)
    with pytest.raises(ConfigError):
        runtime.archive_inputs(inputs)


@pytest.mark.parametrize('changed', ['none', 'code', 'manifest', 'logical_path', 'data'])
@pytest.mark.parametrize('executor', ['live', 'old_transport'])
def test_exact_runtime_snapshot_is_historical_code_evidence(env, tmp_path, changed, executor):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from src.iteration import pack_transport
    root = tmp_path / 'framework'
    code = root / 'scripts/train.py'
    code.parent.mkdir(parents=True)
    code.write_text('old training\n')
    data = root / 'scripts/samples.json'
    data.write_text('{"sample": 1}')
    saved = runtime.freeze(env.root / 'jobs/code-evidence', root=root)
    original = data if changed == 'data' else code
    checksum = runtime.sha256_file(original)
    original.write_text('changed current file')
    if changed == 'code':
        (saved / 'scripts/train.py').write_text('changed snapshot')
    elif changed == 'manifest':
        path = saved / 'manifest.json'
        write_json(path, {**json.loads(path.read_text()), 'schema': 999})
    elif changed == 'logical_path':
        original = root / 'different/train.py'
    module = runtime
    context = nullcontext()
    if executor == 'old_transport':
        def old_lookup(name, expected):
            raise ConfigError('frozen input changed or missing')
        module = SimpleNamespace(**{name: getattr(runtime, name) for name in
            ('ConfigError', 'versions', 'json', 'digest', 'sha256_file')}, evidence_path=old_lookup)
        context = pack_transport.archived_runtime_paths(module)
    before = module.evidence_path
    with context:
        if changed == 'none':
            assert module.evidence_path(str(original), checksum) == saved / 'scripts/train.py'
            assert not (env.root / 'evidence_blobs').exists()
        else:
            with pytest.raises(ConfigError, match='changed or missing'):
                module.evidence_path(str(original), checksum)
    assert module.evidence_path is before


def test_threshold_cli_uses_verified_frozen_executor(env, monkeypatch):
    from scripts import promote as cli
    directory = env.root / 'experiments/completed'
    directory.mkdir(parents=True)
    (directory / 'spec.json').write_text('{}')
    (directory / 'runtime.json').write_text('{}')
    saved = env.root / 'runtimes/original'
    monkeypatch.setattr(runtime, 'verify', lambda path: saved)
    calls = []
    monkeypatch.setattr(cli.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    cli.main(['judge', '--exp', 'completed', '--threshold-reason', '用户修改为5'])
    assert len(calls) == 1
    command, options = calls[0]
    assert command[-2:] == ['--evidence-runtime', str(saved)]
    assert command[command.index('--threshold-reason') + 1] == '用户修改为5'
    assert options == {'check': True}


def test_threshold_cli_rejects_wrong_executor_and_changed_snapshot(env, monkeypatch):
    from contextlib import nullcontext
    from scripts import promote as cli
    directory = env.root / 'experiments/completed'
    directory.mkdir(parents=True)
    (directory / 'spec.json').write_text('{}')
    (directory / 'runtime.json').write_text('{}')
    monkeypatch.setattr(runtime, 'verify', lambda path: env.root / 'runtimes/original')
    monkeypatch.setattr(cli, 'review_policy', lambda: pytest.fail('must not load new policy'))
    monkeypatch.setattr(cli.sys, 'path', list(cli.sys.path))
    monkeypatch.setattr(cli, 'snapshot_imports', lambda path: nullcontext())
    args = ['judge', '--exp', 'completed', '--threshold-reason', '用户修改为5']
    with pytest.raises(ConfigError, match='原冻结执行器'):
        cli.main([*args, '--evidence-runtime', str(env.root / 'wrong')])
    def changed(path):
        raise ConfigError('runtime source changed')
    monkeypatch.setattr(runtime, 'verify', changed)
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ConfigError, match='runtime source changed'):
        cli.main(args)


@pytest.mark.parametrize('kind', ['gen', 'judge'])
def test_ordinary_promotion_cli_uses_frozen_transport_and_rejects_wrong_kind(env, monkeypatch, kind):
    from scripts import promote as cli
    from src.iteration import pack_transport, promote
    directory = env.root / 'experiments/completed'
    directory.mkdir(parents=True)
    (directory / 'spec.json').write_text(json.dumps({'kind': 'gen_ab' if kind == 'gen' else 'judge_eval'}))
    (directory / 'runtime.json').write_text('{}')
    calls = []
    monkeypatch.setattr(pack_transport, 'launch', lambda path, **kw: calls.append((path, kw)) or 0)
    monkeypatch.setattr(promote, 'promote_gen', lambda *a, **k: pytest.fail('must not use live gates'))
    monkeypatch.setattr(promote, 'promote_judge', lambda *a, **k: pytest.fail('must not use live gates'))
    cli.main([kind, '--exp', 'completed'])
    assert calls == [(directory, dict(instance=env.root.name, job='completed', workers=1,
                                    seconds=60, kind='promotion'))]
    with pytest.raises(ConfigError, match='类型与冻结实验不一致'):
        cli.main(['judge' if kind == 'gen' else 'gen', '--exp', 'completed'])
    assert len(calls) == 1
    monkeypatch.setattr(pack_transport, 'launch', lambda *a, **kw: 9)
    with pytest.raises(SystemExit) as result:
        cli.main([kind, '--exp', 'completed'])
    assert result.value.code == 9


def test_request_quota_is_atomic_across_processes(env):
    write_json(env.root / 'resource_policy.json', {'max_requests': 3, 'max_active_requests': 2})
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    children = [ctx.Process(target=_consume, args=(str(env.root.parent), env.root.name, queue)) for _ in range(6)]
    for child in children:
        child.start()
    results = [queue.get(timeout=30) for _ in children]
    for child in children:
        child.join(30)
        assert child.exitcode == 0
    assert results.count('accepted') == 3
    assert json.loads((env.root / 'resource_usage.json').read_text())['requests'] == 3


def test_cancel_blocks_cache_hit_and_deadline_blocks_request(env, monkeypatch):
    directory = env.root / 'jobs/cancel-me'
    directory.mkdir(parents=True)
    monkeypatch.setenv('DH_JOB_DIR', str(directory))
    control.cancel(directory)
    with pytest.raises(control.StopRequested, match='cancelled'):
        cache.memo('any', {}, lambda: pytest.fail('must not produce'))
    control.resume(directory)
    write_json(env.root / 'resource_policy.json', {'deadline_at': 1})
    with pytest.raises(control.StopRequested, match='deadline'):
        with control.request():
            pytest.fail('must not call')


def test_cost_reservation_is_retained_on_failure(env):
    write_json(env.root / 'resource_policy.json', {'max_cost_units': 2, 'cost_units_per_request': 2})
    with pytest.raises(RuntimeError):
        with control.request():
            raise RuntimeError('provider failed')
    with pytest.raises(control.StopRequested, match='cost_budget'):
        with control.request():
            pytest.fail('quota overspent')


def test_fixed_batches_are_reusable_with_immutable_bindings(env):
    data_ref = 'd-test'
    assert acceptance.seal(data_ref, 1)['batches'] == 2
    spec = {'id': 'first', 'kind': 'gen_ab', 'data_ref': data_ref, 'baseline_ref': 'g-1',
            'candidate_ref': 'g-2', 'judge_ref': 'j-1', 'protocol': {}}
    first = acceptance.claim(spec)
    assert acceptance.claim(spec) == first
    with pytest.raises(ConfigError, match='cannot change'):
        acceptance.claim({**spec, 'candidate_ref': 'g-3'})
    second_spec = {**spec, 'id': 'second'}
    second = acceptance.claim(second_spec)
    assert first['batch_id'] == second['batch_id']
    assert acceptance.rows(first) == acceptance.rows(second)
    acceptance.transition({**spec, 'acceptance': first}, 'finished')
    with pytest.raises(ConfigError, match='cannot be reopened'):
        acceptance.transition({**spec, 'acceptance': first}, 'opened')
    assert acceptance.claim({**spec, 'id': 'third'})['batch_id'] == first['batch_id']
    status = acceptance.status(data_ref)
    assert status['remaining'] == 2 and status['consumed'] == 0 and status['reusable']
    assert len(status['jobs']) == 3


def test_public_gate_checks_history_even_after_file_is_removed(tmp_path):
    from scripts.check_public import check
    root = tmp_path / 'release'
    root.mkdir()
    write_json(root / 'public-files.json', {'files': ['public-files.json']})
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
    git('init')
    git('config', 'user.name', 'Synthetic Test')
    git('config', 'user.email', 'test@example.invalid')
    path = root / 'private-report.md'
    path.write_text('private synthetic record')
    git('add', '.')
    git('commit', '-m', 'synthetic fixture')
    path.unlink()
    git('add', '-u')
    git('commit', '-m', 'remove fixture')
    assert any('private-report.md' in issue[0] for issue in check(root, history=True))


def test_public_gate_rejects_reverse_import_without_disclosing_content():
    from scripts.check_public import inspect
    findings = inspect('src/core.py', b'from scripts import instance_job\n', {'src/core.py'})
    assert findings == [('src/core.py', 1, 'framework_imports_instance_or_cli')]
