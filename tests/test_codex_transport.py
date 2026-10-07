"""Pinned HTTP transport changes connectivity, not frozen Judge inputs."""
from concurrent.futures import ThreadPoolExecutor
import copy
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import check
from src.judge import codex_transport as transport, corrected


def command(directory, schema=True):
    result = ['codex', 'exec', '--ephemeral', '--ignore-user-config', '--ignore-rules',
              '--sandbox', 'workspace-write', '--cd', str(directory),
              '--skip-git-repo-check', '--json', '--output-last-message',
              str(directory / 'last_message.json'), '--model', 'gpt-5.6-luna',
              '--config', 'model_reasoning_effort="low"']
    if schema:
        result += ['--output-schema', str(directory / 'schema.json')]
    return result + ['-']


def layer(kind, config, **extra):
    return {'name': {'type': kind}, 'config': config, **extra}


def test_provider_mirrors_pinned_builtin_except_websockets():
    assert transport.provider_config({'layers': []}) == {
        'name': 'OpenAI', 'wire_api': 'responses', 'requires_openai_auth': True,
        'supports_websockets': False, 'http_headers': {'version': '0.144.1'},
        'env_http_headers': {'OpenAI-Organization': 'OPENAI_ORGANIZATION',
                             'OpenAI-Project': 'OPENAI_PROJECT'}}


def test_managed_endpoint_precedence_and_user_config_ignored():
    response = {'layers': [layer('user', {'model_provider': 'other',
        'openai_base_url': 'https://ignored.example'}),
        layer('legacyManagedConfigTomlFromFile', {'openai_base_url': 'https://managed.example'}),
        layer('system', {'openai_base_url': 'https://lower.example'})]}
    assert transport.provider_config(response)['base_url'] == 'https://managed.example'


@pytest.mark.parametrize('response', [
    {}, {'layers': [layer('unknown', {})]},
    {'layers': [layer('system', {'model_provider': 'other'})]},
    {'layers': [layer('system', {'profiles': {'work': {}}})]},
    {'layers': [layer('system', {'model_providers': {transport.PROVIDER: {}}})]},
    {'layers': [layer('system', {'openai_base_url': ''})]},
])
def test_unreviewed_configuration_fails_closed(response):
    with pytest.raises(RuntimeError):
        transport.provider_config(response)


@pytest.mark.parametrize('kind', ['legacyManagedConfigTomlFromFile',
                                 'legacyManagedConfigTomlFromMdm'])
@pytest.mark.parametrize('provider', ['openai', 'other', transport.PROVIDER])
def test_higher_priority_managed_provider_cannot_override_http_flags(kind, provider):
    with pytest.raises(RuntimeError, match='overrides the HTTP transport'):
        transport.provider_config({'layers': [layer(kind, {'model_provider': provider})]})


@pytest.mark.parametrize('kind', ['mdm', 'system', 'enterpriseManaged'])
def test_lower_priority_managed_builtin_provider_is_compatible(kind):
    assert transport.provider_config({'layers': [layer(kind, {'model_provider': 'openai'})]
                                     })['supports_websockets'] is False


def test_cli_version_must_be_reviewed():
    with pytest.raises(RuntimeError, match='reviewed'):
        transport.provider_config({'layers': []}, '0.145.0')


def test_configuration_rpc_preserves_pending_responses_and_skips_notifications():
    process = SimpleNamespace(stdin=io.StringIO())
    pending = b'{"method":"notice"}\n{"id":2,"result":{"layers":[]}}\nremaining'
    request = {'id': 2, 'method': 'config/read', 'params': {'includeLayers': True}}
    result, remainder = transport._rpc(process, None, pending, request,
                                      transport.time.monotonic() + 10)
    assert result == {'layers': []} and remainder == b'remaining'
    assert json.loads(process.stdin.getvalue()) == request


def test_configuration_rpc_error_does_not_expose_server_details():
    process = SimpleNamespace(stdin=io.StringIO())
    with pytest.raises(RuntimeError, match='rejected config/read') as error:
        transport._rpc(process, None, b'{"id":2,"error":{"secret":"private"}}\n',
                       {'id': 2, 'method': 'config/read'}, transport.time.monotonic() + 10)
    assert 'private' not in str(error.value)


def test_configuration_rpc_timeout_identifies_stage():
    with pytest.raises(TimeoutError, match='during config/read'):
        transport._rpc(SimpleNamespace(stdin=io.StringIO()), None, b'',
                       {'id': 2, 'method': 'config/read'}, 0)


def test_configuration_rpc_exit_identifies_stage(monkeypatch):
    process = SimpleNamespace(stdin=io.StringIO(), stdout=SimpleNamespace(fileno=lambda: 1))
    selector = SimpleNamespace(select=lambda timeout: [True])
    monkeypatch.setattr(transport.os, 'read', lambda *args: b'')
    with pytest.raises(RuntimeError, match='exited during initialize'):
        transport._rpc(process, selector, b'', {'id': 1, 'method': 'initialize'},
                       transport.time.monotonic() + 10)


@pytest.mark.parametrize('schema', [False, True])
def test_proxy_preserves_every_input_and_only_inserts_provider(tmp_path, monkeypatch, schema):
    calls = []
    original = SimpleNamespace(run=lambda argv, **kwargs: calls.append((argv, kwargs)), PIPE=object())
    proxy = transport._TransportSubprocess(original, None, {})
    config = transport.provider_config({'layers': []})
    monkeypatch.setattr(transport, 'settings', lambda: config)
    argv = command(tmp_path, schema)
    before = copy.deepcopy(argv)
    options = dict(input='unchanged prompt', text=True, capture_output=True, check=False, timeout=360)
    proxy.run(argv, **options)
    actual, actual_options = calls[0]
    assert actual[:-5] == argv[:-1]
    assert actual[-5:-2] == ['--config', 'model_provider="digital_human_https"', '--config']
    assert actual[-1] == '-'
    assert actual[-2] == f'model_providers.{transport.PROVIDER}=' + transport._toml(config)
    assert '"supports_websockets"=false' in actual[-2]
    assert actual_options == options and argv == before
    assert proxy.PIPE is original.PIPE
    proxy.run(['codex', '--version'], check=True)
    assert calls[-1] == (['codex', '--version'], {'check': True})


def test_non_judge_commands_never_change(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, 'settings', lambda: pytest.fail('no configuration lookup'))
    candidates = [command(tmp_path) + ['extra'], command(tmp_path)]
    candidates[1][8] = str(tmp_path / 'different')
    for argv in candidates:
        assert not transport.judge_command(argv)
        proxy = transport._TransportSubprocess(SimpleNamespace(run=lambda value, **kwargs: value), None, {})
        assert proxy.run(argv) == argv


def test_install_is_local_idempotent_and_checks_archive_target(tmp_path):
    module = SimpleNamespace(__file__=str(tmp_path / 'corrected.py'),
                             subprocess=subprocess, CodexJudgeClient=object())
    original = subprocess.run
    transport.install(module, expected_root=tmp_path)
    installed = module.subprocess
    transport.install(module, expected_root=tmp_path)
    assert module.subprocess is installed and subprocess.run is original
    with pytest.raises(RuntimeError, match='outside'):
        transport.install(module, expected_root=tmp_path / 'archive')


def test_concurrent_requests_write_one_separate_receipt(tmp_path, monkeypatch):
    config = transport.provider_config({'layers': []})
    monkeypatch.setattr(transport, 'settings', lambda: config)
    proxy = transport._TransportSubprocess(SimpleNamespace(run=lambda argv, **kwargs: None),
                                          tmp_path / 'codex_transport_overrides', {'runtime_ref': 'abc'})
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(proxy.run, [command(tmp_path)] * 12))
    paths = list((tmp_path / 'codex_transport_overrides').glob('*.json'))
    assert len(paths) == 1
    receipt = json.loads(paths[0].read_text())
    assert receipt['runtime_ref'] == 'abc' and receipt['supports_websockets'] is False
    assert receipt['provider_sha256'] == transport._digest(config)


def test_settings_initialized_once_under_concurrency(monkeypatch):
    calls = []
    monkeypatch.setattr(transport, '_settings', None)
    monkeypatch.setattr(transport.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='codex-cli 0.144.1'))
    monkeypatch.setattr(transport, 'read_config_layers', lambda: calls.append(True) or {'layers': []})
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda value: transport.settings(), range(12)))
    assert len(calls) == 1 and all(value == values[0] for value in values)


def test_probe_bypasses_project_cache(monkeypatch):
    requests = []
    monkeypatch.setattr(transport, 'settings', lambda: transport.provider_config({'layers': []}))
    monkeypatch.setattr(transport, 'client', lambda config: SimpleNamespace(
        _run=lambda prompt, schema: requests.append((config, prompt, schema)) or '{"ok":true}',
        run=lambda *args: pytest.fail('must not use cache')))
    result = transport.probe('gpt-6-luna')
    assert result['model_request_count'] == 1 and result['project_case_count'] == 0
    assert requests[0][0]['model'] == 'gpt-6-luna' and len(requests) == 1


def test_check_requires_live_flag_and_preserves_pointers(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(transport, 'probe', lambda model, **kwargs: calls.append(model) or {'model': model})
    assert check.main(['codex-transport']) == 0
    value = json.loads(capsys.readouterr().out)
    assert not calls and value['model_requests_prohibited']
    assert check.main(['codex-transport', '--live', '--model', 'gpt-6-luna']) == 0
    value = json.loads(capsys.readouterr().out)
    assert calls == ['gpt-6-luna'] and not value['model_requests_prohibited']
