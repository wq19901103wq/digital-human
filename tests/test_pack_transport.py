import copy
import json
from types import SimpleNamespace

import pytest

from src import llm
from src.config import ConfigError
from src.generator import few_shot
from src.iteration import pack_transport, runtime, versions
from test_fewshot_selection import generator
from test_leakage_guards import env as env


def test_saved_reply_judge_does_not_wait_for_generation_archive(env):
    from src.iteration import control
    from src.iteration.storage import file_lock, write_json
    directory = env.root / 'experiments' / 'saved-judge'
    write_json(directory / 'spec.json', {'kind': 'judge_eval'})
    with file_lock(env.root.parent / '.history_memory.lock'):
        for kind in ('experiment', 'promotion'):
            required = pack_transport.needs_history_memory(directory, kind)
            assert not required
            with pack_transport.history_memory_lane(directory, required=required):
                assert not (directory / 'memory_lane.json').exists()
        control.cancel(directory)
        with pytest.raises(control.StopRequested):
            with pack_transport.history_memory_lane(directory, required=False):
                pytest.fail('cancelled Judge must not run')


@pytest.mark.parametrize('kind,spec_kind', [('pack', 'judge_eval'), ('training', 'judge_eval'),
                                         ('experiment', 'gen_ab'), ('promotion', 'gen_ab'),
                                         ('experiment', 'unknown')])
def test_history_jobs_keep_archive_capacity_limit(env, kind, spec_kind):
    from src.iteration.storage import write_json
    directory = env.root / 'job'
    write_json(directory / 'spec.json', {'kind': spec_kind})
    assert pack_transport.needs_history_memory(directory, kind)


def test_history_memory_lane_releases_on_error_and_serializes_work(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from src.iteration.storage import locked, read_json
    lock = env.root.parent / '.history_memory.lock'
    first, second = env.root / 'first', env.root / 'second'
    entered, started = Event(), Event()
    def work():
        started.set()
        with pack_transport.history_memory_lane(second):
            entered.set()
            assert read_json(second / 'memory_lane.json')['phase'] == 'active'
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='synthetic'):
            with pack_transport.history_memory_lane(first):
                assert read_json(first / 'memory_lane.json')['phase'] == 'active'
                future = pool.submit(work)
                assert started.wait(5)
                assert not entered.wait(0.05)
                raise RuntimeError('synthetic')
        future.result(timeout=5)
    assert entered.is_set() and not locked(lock)
    assert read_json(first / 'memory_lane.json')['phase'] == 'idle'
    assert read_json(second / 'memory_lane.json')['phase'] == 'idle'


@pytest.mark.parametrize('value', ['0', '3', '-1', '', 'invalid'])
def test_history_memory_capacity_rejects_unbounded_values(monkeypatch, value):
    monkeypatch.setenv('DH_HISTORY_MEMORY_LANES', value)
    with pytest.raises(ValueError, match='must be 1 or 2'):
        pack_transport.history_memory_capacity()


def test_two_history_slots_count_live_legacy_job_and_release_on_error(env, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from src.iteration.storage import LockBusy, file_lock, locked, read_json
    monkeypatch.setenv('DH_HISTORY_MEMORY_LANES', '2')
    legacy_lock = env.root.parent / '.history_memory.lock'
    extra_lock = env.root.parent / '.history_memory.1.lock'
    legacy_active, release_legacy, third_started, third_active = (Event() for _ in range(4))
    second, third = env.root / 'second', env.root / 'third'
    def legacy():
        with file_lock(legacy_lock):
            legacy_active.set()
            assert release_legacy.wait(10)
    def wait_for_slot():
        third_started.set()
        with pack_transport.history_memory_lane(third):
            third_active.set()
            assert read_json(third / 'memory_lane.json')['slot'] == 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(legacy)
        assert legacy_active.wait(5)
        try:
            # An error inside the job must propagate, even if it is LockBusy.
            with pytest.raises(LockBusy, match='job failure'):
                with pack_transport.history_memory_lane(second):
                    value = read_json(second / 'memory_lane.json')
                    assert value['capacity'] == 2 and value['slot'] == 1
                    waiting = pool.submit(wait_for_slot)
                    assert third_started.wait(5)
                    assert not third_active.wait(.2)
                    raise LockBusy('job failure')
            waiting.result(timeout=5)
        finally:
            release_legacy.set()
        old.result(timeout=5)
    assert not locked(legacy_lock) and not locked(extra_lock)
    assert read_json(second / 'memory_lane.json')['phase'] == 'idle'
    assert read_json(third / 'memory_lane.json')['phase'] == 'idle'


def test_waiting_history_job_can_be_cancelled(env, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from src.iteration import control
    from src.iteration.storage import file_lock, read_json
    monkeypatch.setenv('DH_HISTORY_MEMORY_LANES', '1')
    output = env.root / 'cancelled'
    checked = Event()
    def stop():
        checked.set()
        raise control.StopRequested('cancelled')
    def waiting():
        with pack_transport.history_memory_lane(output):
            pytest.fail('cancelled job entered archive phase')
    monkeypatch.setattr(control, 'check', stop)
    with file_lock(env.root.parent / '.history_memory.lock'):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(waiting)
            assert checked.wait(5)
            with pytest.raises(control.StopRequested, match='cancelled'):
                future.result(timeout=5)
    assert read_json(output / 'memory_lane.json')['phase'] == 'idle'


def test_history_cache_preserves_values_flags_and_mutation_isolation(monkeypatch):
    originals = {name: getattr(few_shot, name) for name in
                 ('_situation_tags', '_situation_profile')}
    expected = originals['_situation_profile']('你说什么？', enable_clarification=True)
    calls = []
    original = originals['_situation_profile']
    def profile(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(few_shot, '_situation_profile', profile)
    with pack_transport.history_feature_cache(few_shot):
        result = few_shot._situation_profile('你说什么？', enable_clarification=True)
        assert result == expected
        result['test_mutation'] = 'must not leak'
        assert few_shot._situation_profile('你说什么？', enable_clarification=True) == expected
        assert len(calls) == 1
        for text, flag in [('你说什么？', False), ('吃饭了吗', True)]:
            assert few_shot._situation_profile(text, enable_clarification=flag) == original(
                text, enable_clarification=flag)
        assert len(calls) == 3
        tags = few_shot._situation_tags('吃饭了吗')
        tags.add('must not leak')
        assert few_shot._situation_tags('吃饭了吗') == originals['_situation_tags']('吃饭了吗')
    assert few_shot._situation_profile is profile
    assert few_shot._situation_tags is originals['_situation_tags']


def test_history_cache_preserves_prompts_and_per_case_time_guard(env):
    cases = env.source.roles['development']
    expected = [generator(env).build_prompt(case) for case in cases]
    with pack_transport.history_feature_cache(few_shot):
        model = generator(env)
        assert [model.build_prompt(case) for case in cases] == expected
        rendered = json.dumps(expected[0], ensure_ascii=False)
        for future in env.source.cases[3:]:
            assert all(answer not in rendered for answer in future['human_reply'])
        source = env.source.directory / 'messages.jsonl'
        source.write_text(source.read_text() + '\n')
        with pytest.raises(ConfigError):
            model.build_prompt(cases[0])


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf'), 3601])
def test_invalid_timeout_is_rejected(seconds):
    with pytest.raises(ValueError, match='timeout'):
        with pack_transport.timeout_floor(llm.ChatClient, seconds):
            pytest.fail('invalid timeout was accepted')


@pytest.mark.parametrize('protocol', ['anthropic', 'openai'])
def test_timeout_changes_transport_only(monkeypatch, protocol):
    monkeypatch.setenv('TEST_API_BASE', 'https://example.invalid/v1')
    monkeypatch.setenv('TEST_API_KEY', 'synthetic')
    settings = {'llm': {'base_url_env': 'TEST_API_BASE', 'api_key_env': 'TEST_API_KEY'}}
    config = {'model': 'synthetic', 'protocol': protocol, 'timeout_seconds': 20,
              'temperature': 0.7, 'max_tokens': 2500}
    before = copy.deepcopy(config)
    original = llm.ChatClient(settings, config)
    identity = original.cache_identity()
    bodies = []
    monkeypatch.setattr(llm.cache, 'memo', lambda kind, key, produce, **kw: produce())
    monkeypatch.setattr(llm.ChatClient, '_send', lambda self, body: bodies.append(body) or 'ok')
    original.chat([{'role': 'user', 'content': 'synthetic prompt'}], json_mode=True)
    initializer = llm.ChatClient.__init__
    with pack_transport.timeout_floor(llm.ChatClient, 180):
        updated = llm.ChatClient(settings, config)
        assert updated._timeout == 180
        if protocol == 'openai':
            assert updated._client.timeout == 180
        assert updated.cache_identity() == identity
        updated.chat([{'role': 'user', 'content': 'synthetic prompt'}], json_mode=True)
        assert llm.ChatClient(settings, {**config, 'timeout_seconds': 360})._timeout == 360
    assert bodies[0] == bodies[1]
    assert config == before
    assert llm.ChatClient.__init__ is initializer
    assert llm.ChatClient(settings, config)._timeout == 20


@pytest.mark.parametrize('kind,folder', [('pack', 'judge_eval'), ('experiment', 'experiments'),
                                       ('promotion', 'experiments')])
@pytest.mark.parametrize('explicit_env', [False, True])
def test_frozen_launch_keeps_checkpoint_and_rejects_tampering(env, tmp_path, monkeypatch, kind, folder, explicit_env):
    from pathlib import Path
    if explicit_env:
        expected_env = tmp_path / 'connections.env'
        monkeypatch.setenv('DH_ENV_FILE', str(expected_env))
    else:
        monkeypatch.delenv('DH_ENV_FILE', raising=False)
        expected_env = Path(pack_transport.__file__).resolve().parents[2] / '.env'
    source = tmp_path / 'source'
    (source / 'src').mkdir(parents=True)
    (source / 'src/module.py').write_text('value = 1\n')
    directory = env.root / folder / 'synthetic'
    snapshot = runtime.freeze(directory, root=source)
    checkpoint = directory / 'building.json'
    checkpoint.write_text(json.dumps({'rows': [{'case_id': 'saved'}]}))
    before = {p: p.read_bytes() for p in [checkpoint, directory / 'runtime.json',
                                        snapshot / 'manifest.json', snapshot / 'src/module.py']}
    calls = []
    monkeypatch.setattr(pack_transport.subprocess, 'call', lambda argv, **kw: calls.append((argv, kw)) or 0)
    assert pack_transport.launch(directory, instance=env.root.name, job=directory.name,
                                 workers=4, seconds=180, kind=kind) == 0
    argv, kw = calls[0]
    assert argv[argv.index('--kind') + 1] == kind
    assert argv[argv.index('--snapshot') + 1] == str(snapshot)
    assert kw['env']['DH_SETTINGS_FILE'] == str(snapshot / 'settings.snapshot.yaml')
    assert kw['env']['DH_ENV_FILE'] == str(expected_env)
    assert all(p.read_bytes() == value for p, value in before.items())
    (snapshot / 'src/module.py').write_text('value = 2\n')
    with pytest.raises(Exception, match='source changed'):
        pack_transport.launch(directory, instance=env.root.name, job=directory.name,
                              workers=4, seconds=180, kind=kind)
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['gen_ab', 'judge_eval'])
def test_saved_promotion_calls_only_original_gates_and_propagates_rejection(env, monkeypatch, kind):
    from src.iteration import experiment, promote
    directory = env.root / 'experiments/completed'
    monkeypatch.setattr(experiment, 'spec_of', lambda path: {'kind': kind})
    monkeypatch.setattr(versions, 'load_pointers', lambda: {'production_gen': 'g-old'})
    monkeypatch.setattr(pack_transport.runpy, 'run_path', lambda *a, **k: pytest.fail('must not run worker'))
    monkeypatch.setattr(llm.ChatClient, 'chat', lambda *a, **k: pytest.fail('must not request'))
    calls = []
    def accepted(name):
        calls.append(name)
        return {'production_gen': 'g-new'}
    selected = 'promote_gen' if kind == 'gen_ab' else 'promote_judge'
    other = 'promote_judge' if kind == 'gen_ab' else 'promote_gen'
    monkeypatch.setattr(promote, selected, accepted)
    monkeypatch.setattr(promote, other, lambda *a, **k: pytest.fail('wrong promotion kind'))
    assert pack_transport.promote_saved(directory) == {'production_gen': 'g-new'}
    assert calls == ['completed']
    def rejected(name):
        raise ConfigError('data, evidence or production basis changed')
    monkeypatch.setattr(promote, selected, rejected)
    with pytest.raises(ConfigError, match='production basis changed'):
        pack_transport.promote_saved(directory)
