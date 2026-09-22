"""Regression checks for shared entry points and persistent job contracts."""
import ast
from argparse import Namespace

import pytest

from src.config import ConfigError, parse_overrides
from src.iteration import branch_packs, experiment, jobs, learning_guard, runtime, scheduler
from src.iteration.storage import read_json, write_json, write_once_json
from test_leakage_guards import env as _env

env = _env


def test_override_types_and_repeated_siblings_are_consistent():
    assert parse_overrides(['llm.model=test-model', 'llm.temperature=0.4',
                            'retriever.enabled=false', 'items=[1,2]', 'llm.model=updated']) == {
        'llm': {'model': 'updated', 'temperature': .4}, 'retriever': {'enabled': False}, 'items': [1, 2]}


@pytest.mark.parametrize('values', [['llm'], ['llm..model=x'], ['llm=x', 'llm.model=y'],
                                  ['llm.model=x', 'llm={}']])
def test_ambiguous_overrides_fail_before_creation(values):
    with pytest.raises(ConfigError):
        parse_overrides(values)


def test_judge_cli_keeps_false_a_boolean(monkeypatch, tmp_path):
    from scripts import evaluate_judge
    captured = []
    monkeypatch.setattr(experiment, 'create_judge_eval_experiment',
                        lambda ds, pack, overrides, **kw: captured.append(overrides) or tmp_path)
    monkeypatch.setattr(experiment, 'state_of', lambda directory: {'status': 'finished'})
    evaluate_judge.cmd_compare(Namespace(pack='pack-calibration-test',
        override=['feature_llm.enabled=false'], change='explicit switch', against_production=False))
    assert captured == [{'feature_llm': {'enabled': False}}]


def test_missing_json_and_corrupt_json_are_different(tmp_path):
    path = tmp_path / 'state.json'
    assert read_json(path, default={}) == {}
    path.write_text('{"unfinished":')
    with pytest.raises(ValueError):
        read_json(path, default={})


def test_frozen_json_retry_is_idempotent_and_never_replaces_evidence(tmp_path):
    path = tmp_path / 'spec.json'
    write_once_json(path, {'value': [1, 2]})
    before = path.read_bytes()
    write_once_json(path, {'value': [1, 2]})
    with pytest.raises(ConfigError, match='frozen artifact changed'):
        write_once_json(path, {'value': [2, 1]})
    assert path.read_bytes() == before


def test_initial_pack_uses_the_same_guarded_resume_loop(env, monkeypatch):
    monkeypatch.setattr(branch_packs, 'build_clients', lambda *args: env.client)
    ref = branch_packs.prepare_initial()
    directory = env.root / 'judge_eval' / ref
    original = branch_packs.completed_map

    def interrupted(function, cases, workers):
        yield function(cases[0])
        raise RuntimeError('synthetic interruption')

    monkeypatch.setattr(branch_packs, 'completed_map', interrupted)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        branch_packs.build(ref)
    assert len(read_json(directory / 'building.json')['rows']) == 1
    assert read_json(directory / 'progress.json')['status'] == 'stopped'
    monkeypatch.setattr(branch_packs, 'completed_map', original)
    branch_packs.build(ref)
    pack = read_json(directory / 'pack.json')
    assert len(pack['rows']) == len(env.source.roles['judge_development'])
    assert len(env.calls) == len(pack['rows'])  # Successful first case was not generated twice.
    learning_guard.verify_pack(pack, 'development')
    assert jobs.snapshot({'kind': 'pack', 'id': ref})['status'] == 'finished'


def test_standalone_validation_cannot_read_sealed_answers(env, monkeypatch):
    monkeypatch.setattr(branch_packs.gates, 'require_fixed_entry', lambda *args: {})
    (env.source.directory / 'fixed_test.jsonl').unlink()
    with pytest.raises(ConfigError, match='绑定验收批次'):
        branch_packs.prepare_initial('validation')
    assert not env.calls and not (env.root / 'judge_eval').exists()


def test_bad_runtime_does_not_prevent_the_other_job_launching(env, monkeypatch):
    queue = [{'kind': 'training', 'id': name} for name in ('broken', 'healthy')]
    for job in queue:
        directory = jobs.directory(job)
        runtime.freeze(directory)
        write_json(directory / 'state.json', {'status': 'queued'})
    descriptor = jobs.directory(queue[0]) / 'runtime.json'
    write_json(descriptor, {**read_json(descriptor), 'manifest_sha256': 'wrong'})
    monkeypatch.setattr(scheduler.branches, 'advance', lambda: [])
    monkeypatch.setattr(scheduler.training, 'pending', lambda: queue)
    commands = []

    class Child:
        pid = 99999999

        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return None

    monkeypatch.setattr(scheduler.subprocess, 'Popen', Child)
    monkeypatch.setattr(scheduler.task_state, 'process_start', lambda pid: 'synthetic-start')
    assert scheduler.Scheduler().tick() == [queue[1]]
    assert len(commands) == 1
    failed = jobs.snapshot(queue[0])
    assert failed['status'] == 'needs_attention' and failed['reason'] == 'runtime_unavailable'


def test_retry_can_queue_a_standalone_pack(env, monkeypatch):
    job = {'kind': 'pack', 'id': 'standalone'}
    runtime.freeze(jobs.directory(job))
    scheduler.retry(job['kind'], job['id'])
    monkeypatch.setattr(scheduler.branches, 'advance', lambda: [])
    monkeypatch.setattr(scheduler.training, 'pending', lambda: [])
    launched = []

    class Child:
        pid = 99999999

        def __init__(self, command, **kwargs):
            launched.append(command)

        def poll(self):
            return None

    monkeypatch.setattr(scheduler.subprocess, 'Popen', Child)
    monkeypatch.setattr(scheduler.task_state, 'process_start', lambda pid: 'synthetic-start')
    assert scheduler.Scheduler(retry_delay=0).tick() == [job]
    assert len(launched) == 1


def test_documentation_links_and_presentation_dependency_boundary():
    from scripts.check_docs import ROOT, check
    assert check() == []
    assert not (ROOT / 'src/iteration/report.py').exists()
    for name in ['dashboard/components.py', 'dashboard/workflow_view.py', 'dashboard/trace_view.py']:
        tree = ast.parse((ROOT / 'src' / name).read_text())
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(value.endswith(('runner', 'report')) for value in imports), name
    tree = ast.parse((ROOT / 'src/iteration/learning_guard.py').read_text())
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or '').endswith('branch_packs')
                   for node in ast.walk(tree))


def test_bootstrap_cannot_reinitialize_an_existing_baseline(env, monkeypatch):
    from scripts import bootstrap
    from src.iteration import versions
    before = versions.POINTERS_PATH.read_bytes()
    monkeypatch.setattr(bootstrap.versions, 'switch_instance', lambda name: versions.PRIVATE)
    monkeypatch.setattr(bootstrap.sys, 'argv', ['bootstrap.py', '--instance', 'demo', '--data', 'unused.jsonl',
                                             '--model', 'generator', '--judge-model', 'judge', '--initialize'])
    with pytest.raises(ConfigError, match='empty instance'):
        bootstrap.main()
    assert versions.POINTERS_PATH.read_bytes() == before


@pytest.mark.parametrize('error', [RuntimeError('transport'), ConfigError(branch_packs._INVALID_GENERATION)])
def test_pack_nonterminal_errors_still_stop(env, monkeypatch, error):
    monkeypatch.setattr(branch_packs, 'build_clients', lambda *args: env.client)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(branch_packs.ReplyGenerator, 'generate', fail)
    ref = branch_packs.prepare_initial()
    with pytest.raises(type(error)):
        branch_packs.build(ref)
    assert branch_packs.recover_terminal_failures(ref)['recovered_failures'] == 0
    assert not (env.root / 'judge_eval' / ref / 'pack.json').exists()


def test_pack_terminal_format_failure_is_counted(env, monkeypatch):
    monkeypatch.setattr(branch_packs, 'build_clients', lambda *args: env.client)
    monkeypatch.setattr(env.client, 'chat', lambda *args, **kwargs: '{"replies":["1","2","3","4"]}')
    ref = branch_packs.prepare_initial()
    branch_packs.build(ref)
    pack = read_json(env.root / 'judge_eval' / ref / 'pack.json')
    assert pack['generation_failures'] == len(pack['rows'])
    assert all(row['generation_error']['message'] == branch_packs._INVALID_GENERATION for row in pack['rows'])
    learning_guard.verify_pack(pack, 'development')


@pytest.mark.parametrize('tamper', [False, True])
def test_recover_legacy_terminal_failure_preserves_frozen_runtime(env, monkeypatch, tamper):
    monkeypatch.setattr(branch_packs, 'build_clients', lambda *args: env.client)
    original_chat = env.client.chat
    ref = branch_packs.prepare_initial()
    directory = env.root / 'judge_eval' / ref
    frozen = (directory / 'runtime.json').read_bytes()
    original_map = branch_packs.completed_map
    def legacy_abort(function, cases, workers):
        yield function(cases[0])
        monkeypatch.setattr(env.client, 'chat', lambda *args, **kwargs: '{"replies":["1","2","3","4"]}')
        function(cases[1])  # Old executor saved the trace but aborted before checkpointing.
        raise RuntimeError('legacy abort')
    monkeypatch.setattr(branch_packs, 'completed_map', legacy_abort)
    with pytest.raises(RuntimeError, match='legacy abort'):
        branch_packs.build(ref)
    before = (directory / 'building.json').read_bytes()
    if tamper:
        for path in (directory / 'traces').glob('*.json'):
            value = read_json(path)
            if value['status'] == 'failed':
                value['versions']['baseline_ref'] = 'changed'
                write_json(path, value)
        with pytest.raises(ConfigError, match='拒绝恢复'):
            branch_packs.recover_terminal_failures(ref)
        assert (directory / 'building.json').read_bytes() == before
        return
    assert branch_packs.recover_terminal_failures(ref)['recovered_failures'] == 1
    assert branch_packs.recover_terminal_failures(ref)['recovered_failures'] == 0
    assert (directory / 'runtime.json').read_bytes() == frozen
    saved = read_json(directory / 'building.json')['rows']
    assert saved[0]['ai_replies'] and saved[1]['generation_status'] == 'failed'
    monkeypatch.setattr(env.client, 'chat', original_chat)
    monkeypatch.setattr(branch_packs, 'completed_map', original_map)
    branch_packs.build(ref)
    assert read_json(directory / 'pack.json')['generation_failures'] == 1
    assert read_json(directory / 'building.json')['rows'][:2] == saved
