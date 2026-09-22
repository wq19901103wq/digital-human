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


@pytest.mark.parametrize('kind,folder', [('pack', 'judge_eval'), ('experiment', 'experiments')])
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
