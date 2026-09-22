import json
import multiprocessing
import os
import subprocess

import pytest

from src import cache
from src.config import ConfigError
from src.iteration import acceptance, control, runtime, versions
from src.iteration.storage import write_json
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
    from scripts import promote as cli
    directory = env.root / 'experiments/completed'
    directory.mkdir(parents=True)
    (directory / 'spec.json').write_text('{}')
    (directory / 'runtime.json').write_text('{}')
    monkeypatch.setattr(runtime, 'verify', lambda path: env.root / 'runtimes/original')
    monkeypatch.setattr(cli, 'review_policy', lambda: pytest.fail('must not load new policy'))
    monkeypatch.setattr(cli.sys, 'path', list(cli.sys.path))
    args = ['judge', '--exp', 'completed', '--threshold-reason', '用户修改为5']
    with pytest.raises(ConfigError, match='原冻结执行器'):
        cli.main([*args, '--evidence-runtime', str(env.root / 'wrong')])
    def changed(path):
        raise ConfigError('runtime source changed')
    monkeypatch.setattr(runtime, 'verify', changed)
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ConfigError, match='runtime source changed'):
        cli.main(args)


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


def test_batches_are_unique_bound_and_not_refunded(env):
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
    assert {r['case_id'] for r in acceptance.rows(first)}.isdisjoint(r['case_id'] for r in acceptance.rows(second))
    acceptance.transition({**spec, 'acceptance': first}, 'finished')
    with pytest.raises(ConfigError, match='cannot be reopened'):
        acceptance.transition({**spec, 'acceptance': first}, 'opened')
    with pytest.raises(ConfigError, match='exhausted'):
        acceptance.claim({**spec, 'id': 'third'})
    assert acceptance.status(data_ref)['remaining'] == 0


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
