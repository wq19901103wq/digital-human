"""Pinned HTTP transport changes connectivity, not frozen Judge inputs."""
from concurrent.futures import ThreadPoolExecutor
import copy
import io
import json
from pathlib import Path
from threading import Barrier, Event
import subprocess
from types import SimpleNamespace

import pytest

from scripts import check
from src.judge import codex_transport as transport, corrected
from src.infrastructure import cache, tracing
from src.infrastructure.config import ConfigError


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


@pytest.mark.parametrize('kind', ['mdm', 'system', 'enterpriseManaged',
                                 'legacyManagedConfigTomlFromFile',
                                 'legacyManagedConfigTomlFromMdm'])
def test_managed_required_apps_cannot_be_disabled(kind):
    with pytest.raises(RuntimeError, match='requires app integrations'):
        transport.provider_config({'layers': [layer(kind, {'features': {'apps': True}})]})


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
def test_proxy_preserves_inputs_and_disables_only_unused_apps(tmp_path, monkeypatch, schema):
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
    assert actual[:-7] == argv[:-1]
    assert actual[-7:-4] == ['--config', 'model_provider="digital_human_https"', '--config']
    assert actual[-1] == '-'
    assert actual[-4] == f'model_providers.{transport.PROVIDER}=' + transport._toml(config)
    assert '"supports_websockets"=false' in actual[-4]
    assert actual[-3:-1] == ['--config', 'features.apps=false']
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
                             subprocess=subprocess, CodexJudgeClient=type('Client', (),
                                 {'run': lambda self, prompt, schema=None: prompt}))
    original = subprocess.run
    transport.install(module, expected_root=tmp_path)
    installed = module.subprocess
    transport.install(module, expected_root=tmp_path)
    assert module.subprocess is installed and subprocess.run is original
    with pytest.raises(RuntimeError, match='outside'):
        transport.install(module, expected_root=tmp_path / 'archive')


@pytest.mark.parametrize('feature_format_retry', [False, True])
def test_concurrent_requests_write_one_separate_receipt(tmp_path, monkeypatch, feature_format_retry):
    config = transport.provider_config({'layers': []})
    monkeypatch.setattr(transport, 'settings', lambda: config)
    proxy = transport._TransportSubprocess(SimpleNamespace(run=lambda argv, **kwargs: None),
        tmp_path / 'codex_transport_overrides', {'runtime_ref': 'abc'},
        feature_format_retry=feature_format_retry)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(proxy.run, [command(tmp_path)] * 12))
    paths = list((tmp_path / 'codex_transport_overrides').glob('*.json'))
    assert len(paths) == 1
    receipt = json.loads(paths[0].read_text())
    assert receipt['runtime_ref'] == 'abc' and receipt['supports_websockets'] is False
    assert receipt['provider_sha256'] == transport._digest(config)
    assert receipt['session_config'] == {'features.apps': False}
    assert receipt['feature_format_retry']['enabled'] is feature_format_retry
    assert receipt['feature_format_retry']['max_attempts'] == (2 if feature_format_retry else 1)


def test_failure_diagnostics_preserve_result_and_omit_raw_output(tmp_path, monkeypatch):
    result = subprocess.CompletedProcess([], 1,
        stdout='not-json\n' + json.dumps({'type': 'error', 'message':
            'stream disconnected before completion: private-prompt-token'}) + '\n' +
            json.dumps({'type': 'turn.failed', 'error': {'message': 'request timed out'}}),
        stderr='rmcp::transport http/request failed private-auth-token')
    events, calls = [], []
    proxy = transport._TransportSubprocess(SimpleNamespace(run=lambda *args, **kwargs:
        calls.append(args) or result), None, {},
        tracing=SimpleNamespace(note=lambda kind, data: events.append((kind, data))))
    monkeypatch.setattr(transport, 'settings', lambda: transport.provider_config({'layers': []}))
    assert proxy.run(command(tmp_path)) is result
    assert len(calls) == 1
    assert events == [('codex_transport_failure', {'exit_code': 1,
        'terminal_events': ['error', 'turn.failed'],
        'signals': ['mcp_connection_error', 'stream_disconnected', 'request_timeout'],
        'terminal_signals': ['stream_disconnected', 'request_timeout'],
        'stderr_signals': ['mcp_connection_error']})]
    assert 'private' not in json.dumps(events)


def test_failure_details_ignore_nonterminal_output():
    result = subprocess.CompletedProcess([], 1,
        stdout='[]\n' + json.dumps({'type': 'item.completed', 'message': '401 429 timeout'}),
        stderr='failed to refresh available models: unauthorized')
    assert transport.failure_details(result) == {'exit_code': 1, 'terminal_events': [],
        'signals': ['model_catalog_error', 'authentication_error'], 'terminal_signals': [],
        'stderr_signals': ['model_catalog_error', 'authentication_error']}


@pytest.mark.parametrize('message,kind', [
    ('error sending request for url', 'request_send_failure'),
    ('error decoding response body', 'response_body_failure'),
    ('stream ended before response.completed', 'stream_incomplete'),
    ('dns error: failed to lookup address', 'dns_resolution_failure'),
    ('Connection refused', 'connection_refused'),
    ('Connection reset by peer', 'connection_reset'),
    ('invalid peer certificate', 'tls_failure'),
    ('tunnel unsuccessful', 'proxy_connect_failure'),
    ('Operation not permitted', 'operation_not_permitted'),
])
def test_failure_details_preserve_only_allowlisted_connection_kinds(message, kind):
    private = ' private-prompt-token https://private.example Bearer private-auth-token'
    result = subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'turn.failed', 'error': {'message': message + private}}),
        stderr='error sending request for url' + private)
    details = transport.failure_details(result)
    assert details['connection_errors'] == {
        'terminal': [kind], 'stderr': ['request_send_failure']}
    assert 'private' not in json.dumps(details)
    assert 'public_probe_errors' not in details


def test_failure_details_ignore_nonterminal_connection_messages():
    result = subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'item.completed', 'message': 'error sending request for url'}),
        stderr='')
    assert 'connection_errors' not in transport.failure_details(result)


@pytest.mark.parametrize('message,signal', [
    ("The model 'private-model' is not supported", 'unsupported_model'),
    ('model_not_found: private-model does not exist', 'unsupported_model'),
    ('403 permission denied: private-token', 'permission_denied'),
    ('400 Bad Request: private-prompt', 'invalid_request'),
])
def test_failure_details_classify_terminal_rejection_without_raw_messages(message, signal):
    result = subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'turn.failed', 'error': {'message': message}}),
        stderr='failed to refresh available models: timeout waiting for child process to exit')
    details = transport.failure_details(result)
    assert details['terminal_signals'] == [signal]
    assert details['stderr_signals'] == ['model_catalog_error', 'request_timeout']
    assert 'private' not in json.dumps(details)


def test_public_probe_errors_are_opt_in_bounded_and_redacted():
    message = 'gateway error: Bearer private-token sk-private-key eyJprivate.token '
    message += 'https://private.example/path?key=secret ' + 'x' * 800
    result = subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'turn.failed', 'error': {'message': message}}), stderr='')
    assert 'public_probe_errors' not in transport.failure_details(result)
    errors = transport.failure_details(result, public_probe=True)['public_probe_errors']
    assert len(errors) == 1 and len(errors[0]) == 512
    assert 'private' not in errors[0] and 'secret' not in errors[0]


def retry_client(tmp_path, produce, *, feature_format_retry=True, stop_on_service_failure=False):
    class Client:
        def run(self, prompt, schema=None):
            return cache.memo('llm_request', {'prompt': prompt, 'schema': schema},
                lambda: self._run(prompt, schema), valid=cache.response_valid)

        def _run(self, prompt, schema=None):
            return produce(prompt, schema)

    module = SimpleNamespace(__file__=str(tmp_path / 'corrected.py'), subprocess=subprocess,
                             CodexJudgeClient=Client, cache=cache, tracing=tracing)
    transport.install(module, expected_root=tmp_path, feature_format_retry=feature_format_retry,
                      stop_on_service_failure=stop_on_service_failure)
    return module, Client()


def subprocess_client(tmp_path, monkeypatch, produce, *, feature_format_retry=False,
                      stop_on_service_failure=False):
    calls = []

    def request(prompt, schema):
        if prompt == 'before subprocess':
            raise RuntimeError('unrelated failure')
        result = module.subprocess.run(command(tmp_path, schema is not None),
                                       input=prompt, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'Codex Judge 执行失败（exit {result.returncode}）：{result.stderr[-500:]}')
        return result.stdout

    module, client = retry_client(tmp_path, request, feature_format_retry=feature_format_retry,
                                 stop_on_service_failure=stop_on_service_failure)
    client.cache_identity = lambda: {'model': 'unchanged'}
    module.subprocess._original = SimpleNamespace(run=lambda argv, **kwargs:
        calls.append((argv, kwargs)) or produce(kwargs['input']))
    monkeypatch.setattr(transport, 'settings', lambda: transport.provider_config({'layers': []}))
    return module, client, calls


@pytest.mark.parametrize('feature_format_retry', [False, True])
def test_terminal_diagnostics_persist_without_tracing_or_extra_wiki_requests(
        tmp_path, monkeypatch, feature_format_retry):
    from src.share.wiki import wiki_repair

    result = subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'turn.failed', 'error': {
            'message': 'rate limit 429 private-prompt-token'}}),
        stderr='failed to refresh available models: timeout waiting for child process to exit')
    module, client, calls = subprocess_client(tmp_path, monkeypatch, lambda prompt: result,
                                             feature_format_retry=feature_format_retry)
    installed = module.CodexJudgeClient._run
    transport.install(module, expected_root=tmp_path, feature_format_retry=feature_format_retry)
    assert module.CodexJudgeClient._run is installed
    prompt, schema = 'original task', {'type': 'object'}
    with pytest.raises(RuntimeError, match='Codex transport diagnostics'):
        wiki_repair._cached_call(client, tmp_path / 'cache', prompt, schema, lambda value: None)
    assert len(calls) == 1 and calls[0][1]['input'] == prompt
    assert client.cache_identity() == {'model': 'unchanged'}
    paths = list((tmp_path / 'failures').glob('*.json'))
    assert len(paths) == 1
    failure = json.loads(paths[0].read_text())
    assert failure['attempt'] == 1 and failure['policy_attempt'] == 0
    assert failure['retry_budget'] == 'format_only_v1'
    assert failure['failure_kind'] == 'transport'
    assert failure['result'] is None
    details = json.loads(failure['error'].split('Codex transport diagnostics: ', 1)[1])
    assert details['terminal_events'] == ['turn.failed']
    assert details['terminal_signals'] == ['rate_limit']
    assert details['stderr_signals'] == ['model_catalog_error', 'request_timeout']
    assert 'private-prompt-token' not in json.dumps(failure)


@pytest.mark.parametrize('output_kind', ['text', 'bytes', 'missing'])
@pytest.mark.parametrize('feature_format_retry', [False, True])
def test_client_timeout_preserves_wiki_format_budget_and_safe_diagnostics(
        tmp_path, monkeypatch, output_kind, feature_format_retry):
    from src.share.wiki import wiki_repair

    stdout = json.dumps({'type': 'item.completed', 'message': '401 private-prompt-token'})
    stderr = 'failed to refresh available models: timeout private-auth-token'
    if output_kind == 'bytes':
        stdout, stderr = stdout.encode(), stderr.encode() + b'\xff'
    elif output_kind == 'missing':
        stdout, stderr = None, None
    timeout = subprocess.TimeoutExpired(['codex', 'private-command-token'], 900,
                                        output=stdout, stderr=stderr)

    def produce(prompt):
        if prompt == 'cached success':
            return subprocess.CompletedProcess([], 0, stdout='{}', stderr='')
        raise timeout

    _, client, calls = subprocess_client(tmp_path, monkeypatch, produce,
                                        feature_format_retry=feature_format_retry,
                                        stop_on_service_failure=True)
    directory, schema = tmp_path / 'cache', {'type': 'object', 'properties': {}}
    assert wiki_repair._cached_call(client, directory, 'cached success', schema,
                                   lambda value: None) == {}
    stale = {'terminal_signals': ['authentication_error']}
    token = transport._failure_details.set(stale)
    try:
        with pytest.raises(transport.CodexTransportError) as error:
            wiki_repair._cached_call(client, directory, 'timeout', schema, lambda value: None)
        assert error.value.__cause__ is timeout
        assert transport._failure_details.get() is stale
    finally:
        transport._failure_details.reset(token)
    details = error.value.details
    assert details['exit_code'] == 124 and details['exit_code_source'] == 'client_timeout'
    assert details['failure_kind'] == 'subprocess_timeout' and details['timeout_seconds'] == 900
    assert details['terminal_events'] == [] and details['terminal_signals'] == []
    assert 'request_timeout' in details['signals']
    assert 'authentication_error' not in details['signals']
    paths = list((tmp_path / 'failures').glob('*.json'))
    assert len(paths) == 1
    failure = json.loads(paths[0].read_text())
    assert failure['attempt'] == 1 and failure['policy_attempt'] == 0
    assert failure['retry_budget'] == 'format_only_v1' and failure['failure_kind'] == 'transport'
    assert failure['result'] is None and failure['transport'] == details
    assert wiki_repair._transport_diagnostic(failure['error']) == details
    assert 'private' not in json.dumps(failure)
    assert wiki_repair._cached_call(client, directory, 'cached success', schema,
                                   lambda value: None) == {}
    assert len(calls) == 2


def test_client_timeout_keeps_consumed_format_attempts(tmp_path, monkeypatch):
    from src.share.wiki import wiki_repair

    timeout = subprocess.TimeoutExpired('codex', 900)
    responses = iter([subprocess.CompletedProcess([], 0, stdout='{}', stderr=''), timeout])

    def produce(prompt):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    def parse(value):
        raise ValueError('format mismatch')

    _, client, calls = subprocess_client(tmp_path, monkeypatch, produce)
    with pytest.raises(transport.CodexTransportError):
        wiki_repair._cached_call(client, tmp_path / 'cache', 'original task',
                                 {'type': 'object', 'properties': {}}, parse)
    failures = sorted((json.loads(path.read_text()) for path in
                       (tmp_path / 'failures').glob('*.json')), key=lambda value: value['attempt'])
    assert len(calls) == 2 and len(failures) == 2
    assert failures[0]['policy_attempt'] == failures[1]['policy_attempt'] == 1
    assert failures[0]['result'] == {} and failures[1]['result'] is None
    assert failures[1]['failure_kind'] == 'transport'


def test_client_timeout_retains_terminal_service_block_and_tracing(tmp_path, monkeypatch):
    timeout = subprocess.TimeoutExpired('codex', 900, output=json.dumps(
        {'type': 'turn.failed', 'error': {'message': 'rate limit 429 private-token'}}).encode())

    def produce(prompt):
        raise timeout

    module, client, calls = subprocess_client(tmp_path, monkeypatch, produce,
                                             stop_on_service_failure=True)
    events = []
    module.subprocess._tracing = SimpleNamespace(note=lambda kind, details:
                                                events.append((kind, details)))
    with pytest.raises(transport.CodexTransportError) as error:
        client.run('timeout')
    assert error.value.details['terminal_signals'] == ['rate_limit']
    assert error.value.details['signals'] == ['rate_limit', 'request_timeout']
    with pytest.raises(transport.CodexServiceBlocked):
        client.run('later task')
    assert len(calls) == 1
    assert events == [('codex_transport_failure', error.value.details)]
    assert 'private' not in json.dumps(events)


def test_non_judge_timeout_keeps_original_exception_and_details():
    timeout = subprocess.TimeoutExpired(['other-command', 'private-token'], 5)

    def run(*args, **kwargs):
        raise timeout

    proxy = transport._TransportSubprocess(SimpleNamespace(run=run), None, {})
    stale = {'terminal_signals': ['rate_limit']}
    token = transport._failure_details.set(stale)
    try:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            proxy.run(['other-command'])
        assert error.value is timeout and transport._failure_details.get() is stale
    finally:
        transport._failure_details.reset(token)


@pytest.mark.parametrize('error', [
    {'message': 'exceeded retry limit, last status: 429 Too Many Requests'},
    {'message': 'HTTP 503 Service Unavailable'},
    {'message': 'unauthorized 401'},
    {'message': 'permission denied'},
    {'message': 'model not supported'},
    {'message': '', 'code': 'insufficient_quota'},
    {'message': '', 'status_code': 429},
])
def test_terminal_service_failure_stops_later_requests_without_retry(tmp_path, monkeypatch, error):
    response = subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'turn.failed', 'error': error}), stderr='')
    module, client, calls = subprocess_client(tmp_path, monkeypatch, lambda prompt: response,
                                            stop_on_service_failure=True)
    with pytest.raises(transport.CodexTransportError):
        client.run('first')
    with pytest.raises(transport.CodexServiceBlocked):
        transport.check_available(module)
    with pytest.raises(transport.CodexServiceBlocked):
        client.run('later')
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match='service policy cannot change'):
        transport.install(module, feature_format_retry=False, stop_on_service_failure=False)


@pytest.mark.parametrize('stdout,stderr', [
    ('', 'failed to refresh available models: status 429'),
    (json.dumps({'type': 'item.completed', 'message': 'HTTP 429'}), ''),
    (json.dumps({'type': 'turn.failed', 'error': {'message': 'row 429 has value 503'}}), ''),
    (json.dumps({'type': 'turn.failed', 'error': {'message': 'stream disconnected'}}), ''),
])
def test_catalog_warnings_and_unrelated_numbers_do_not_latch_service_gate(
        tmp_path, monkeypatch, stdout, stderr):
    response = subprocess.CompletedProcess([], 1, stdout=stdout, stderr=stderr)
    module, client, calls = subprocess_client(tmp_path, monkeypatch, lambda prompt: response,
                                            stop_on_service_failure=True)
    for prompt in ('first', 'later'):
        with pytest.raises(transport.CodexTransportError):
            client.run(prompt)
        transport.check_available(module)
    assert len(calls) == 2


@pytest.mark.parametrize('message,expected', [
    ('stream disconnected: error sending request', True),
    ('request timed out', True),
    ('connection reset', True),
    ('failed to lookup address', True),
    ('stream disconnected: HTTP 429', False),
    ('stream disconnected: HTTP 503', False),
    ('stream disconnected: unauthorized 401', False),
    ('stream disconnected: model not supported', False),
    ('stream disconnected: invalid request', False),
    ('stream disconnected: certificate verify failed', False),
    ('stream disconnected: proxy connect error', False),
    ('stream disconnected: operation not permitted', False),
    ('invalid JSON output', False),
])
def test_retryable_connection_failure_requires_terminal_transport_evidence(message, expected):
    response = subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'turn.failed', 'error': {'message': message}}), stderr='')
    detail = transport.failure_details(response)
    assert transport.retryable_connection_failure(transport.CodexTransportError(detail)) is expected
    assert not transport.retryable_connection_failure(transport.CodexServiceBlocked(detail))
    assert not transport.retryable_connection_failure(RuntimeError(message))


def test_connection_retry_does_not_use_catalog_warnings_or_successful_requests():
    detail = transport.failure_details(subprocess.CompletedProcess([], 1,
        stdout='', stderr='failed to refresh available models: request timeout'))
    assert not transport.retryable_connection_failure(transport.CodexTransportError(detail))
    detail.update(failure_kind='subprocess_timeout', exit_code=124)
    assert transport.retryable_connection_failure(transport.CodexTransportError(detail))
    detail['exit_code'] = 0
    assert not transport.retryable_connection_failure(transport.CodexTransportError(detail))


def test_explicit_service_evidence_retains_status_and_request_id_not_private_text():
    response = subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'turn.failed', 'error': {'message':
            'last status: 429 Too Many Requests, request id: '
            'dfb080f3-c87a-4b6a-9644-9ac9c47137ef private-token'}}), stderr='')
    detail = transport.failure_details(response)
    assert detail['service_errors'] == [{'http_statuses': [429],
        'request_id': 'dfb080f3-c87a-4b6a-9644-9ac9c47137ef'}]
    assert 'private-token' not in json.dumps(detail)


def test_service_gate_preserves_success_from_already_inflight_request(tmp_path, monkeypatch):
    overlap, rejected = Barrier(2), Event()
    module = None

    def response(prompt):
        overlap.wait(timeout=10)
        if prompt == 'success':
            assert rejected.wait(10)
            return subprocess.CompletedProcess([], 0, stdout='{}', stderr='')
        return subprocess.CompletedProcess([], 1,
            stdout=json.dumps({'type': 'turn.failed', 'error': {'message': 'HTTP 429'}}), stderr='')

    module, client, calls = subprocess_client(tmp_path, monkeypatch, response,
                                            stop_on_service_failure=True)

    def failing_request():
        try:
            with pytest.raises(transport.CodexTransportError):
                client.run('failure')
        finally:
            rejected.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        successful = executor.submit(client.run, 'success')
        executor.submit(failing_request).result(timeout=10)
        assert successful.result(timeout=10) == '{}'
    assert len(calls) == 2
    with pytest.raises(transport.CodexServiceBlocked):
        transport.check_available(module)


def test_failure_diagnostics_do_not_reuse_previous_request_details(tmp_path, monkeypatch):
    responses = {'failed': subprocess.CompletedProcess([], 1,
        stdout=json.dumps({'type': 'error', 'message': 'stream disconnected'}), stderr=''),
        'succeeded': subprocess.CompletedProcess([], 0, stdout='{}', stderr='')}
    _, client, calls = subprocess_client(tmp_path, monkeypatch, responses.__getitem__)
    stale = {'terminal_signals': ['authentication_error']}
    token = transport._failure_details.set(stale)
    try:
        with pytest.raises(RuntimeError, match='stream_disconnected'):
            client.run('failed')
        assert transport._failure_details.get() is stale
        assert client.run('succeeded') == '{}'
        assert transport._failure_details.get() is stale
        with pytest.raises(RuntimeError) as error:
            client.run('before subprocess')
        assert str(error.value) == 'unrelated failure'
        assert transport._failure_details.get() is stale
    finally:
        transport._failure_details.reset(token)
    assert len(calls) == 2


def test_terminal_diagnostics_remain_thread_local(tmp_path, monkeypatch):
    def response(prompt):
        return subprocess.CompletedProcess([], 1,
            stdout=json.dumps({'type': 'turn.failed', 'error': {'message': prompt}}), stderr='')

    _, client, calls = subprocess_client(tmp_path, monkeypatch, response)

    def request(prompt):
        with pytest.raises(RuntimeError) as error:
            client.run(prompt)
        return json.loads(str(error.value).split('Codex transport diagnostics: ', 1)[1])

    prompts = ['rate limit 429', 'request timed out', 'unauthorized 401'] * 4
    with ThreadPoolExecutor(max_workers=4) as executor:
        details = list(executor.map(request, prompts))
    assert len(calls) == len(prompts)
    assert [detail['terminal_signals'] for detail in details] == [
        ['rate_limit'], ['request_timeout'], ['authentication_error']] * 4


@pytest.mark.parametrize('feature_format_retry', [False, True])
def test_retry_policy_cannot_change_after_install(tmp_path, feature_format_retry):
    module, _ = retry_client(tmp_path, lambda *args: '{}',
                             feature_format_retry=feature_format_retry)
    installed = module.CodexJudgeClient.run
    proxy = module.subprocess
    transport.install(module, expected_root=tmp_path, feature_format_retry=feature_format_retry)
    with pytest.raises(RuntimeError, match='retry policy cannot change'):
        transport.install(module, expected_root=tmp_path,
                          feature_format_retry=not feature_format_retry)
    assert module.CodexJudgeClient.run is installed and module.subprocess is proxy


def test_wiki_repair_owns_format_retries_without_transport_multiplication(tmp_path):
    from src.share.wiki import wiki_repair

    calls = []
    module, client = retry_client(tmp_path, lambda *args: calls.append(args) or 'bad-json',
                                 feature_format_retry=False)
    client.cache_identity = lambda: {'model': 'unchanged'}
    assert not getattr(module.CodexJudgeClient.run, '_digital_human_format_retry', False)
    with pytest.raises(ValueError):
        wiki_repair._cached_call(client, tmp_path / 'cache', 'original task',
                                {'type': 'object'}, lambda value: None)
    assert len(calls) == 3
    assert calls[0][0] == 'original task'
    assert all(prompt.endswith('original task') for prompt, schema in calls)
    assert len(list((tmp_path / 'failures').glob('*.json'))) == 3


def test_format_retry_preserves_inputs_and_valid_cached_response(tmp_path, monkeypatch):
    calls, events = [], []
    responses = iter(['not-json', '{"ok":true}'])
    module, client = retry_client(tmp_path,
        lambda prompt, schema: calls.append((prompt, schema)) or next(responses))
    monkeypatch.setattr(tracing, 'note', lambda kind, data: events.append((kind, data)))
    trace = tracing.CaseTrace(tmp_path / 'experiments/current', {'case_id': 'one'},
                             {'kind': 'judge_eval', 'dataset': 'development'})
    schema = {'type': 'object'}
    installed = module.CodexJudgeClient.run
    transport.install(module, expected_root=tmp_path)
    assert module.CodexJudgeClient.run is installed
    with cache.scope(trace.exp_dir, trace.value, {'kind': 'judge', 'branch': 'candidate'}, 0), \
            cache.validation(json.loads):
        assert client.run('unchanged prompt', schema) == '{"ok":true}'
        assert client.run('unchanged prompt', schema) == '{"ok":true}'
    assert calls == [('unchanged prompt', schema)] * 2
    assert [data['valid'] for kind, data in events if kind == 'codex_response_validation'] == [False, True, True]


def test_format_retry_stops_at_two_and_does_not_stack_outer_retry(tmp_path):
    calls = []
    _, client = retry_client(tmp_path, lambda *args: calls.append(args) or 'bad-json')
    with pytest.raises(ValueError), cache.validation(json.loads):
        for attempt in range(2):
            raw = client.run('same', {})
            try:
                json.loads(raw)
            except ValueError:
                if attempt:
                    raise
    assert len(calls) == 2


@pytest.mark.parametrize('registered,schema', [(False, {}), (True, None)])
def test_no_format_retry_without_feature_validator_and_schema(tmp_path, registered, schema):
    calls = []
    _, client = retry_client(tmp_path, lambda *args: calls.append(args) or 'bad-json')
    with cache.validation(json.loads if registered else None):
        assert client.run('same', schema) == 'bad-json'
    assert len(calls) == 1


@pytest.mark.parametrize('error', [ConfigError('source changed'), RuntimeError('transport failed'),
                                 ValueError('request failed')])
def test_request_errors_are_not_format_retried(tmp_path, error):
    calls = []

    def produce(*args):
        calls.append(args)
        raise error

    _, client = retry_client(tmp_path, produce)
    with pytest.raises(type(error), match=str(error)), cache.validation(json.loads):
        client.run('same', {})
    assert len(calls) == 1


def test_source_validator_errors_are_not_retried(tmp_path):
    calls = []
    _, client = retry_client(tmp_path, lambda *args: calls.append(args) or '{}')

    def validate(raw):
        raise ConfigError('source changed')

    with pytest.raises(ConfigError, match='source changed'), cache.validation(validate):
        client.run('same', {})
    assert len(calls) == 1


def test_format_retry_validator_is_thread_local(tmp_path):
    calls = []
    _, client = retry_client(tmp_path, lambda prompt, schema: calls.append(prompt) or prompt)

    def request(number):
        prompt = json.dumps({'id': number})

        def validate(raw):
            assert json.loads(raw)['id'] == number

        with cache.validation(validate):
            return client.run(prompt, {})

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(request, range(12)))
    assert len(calls) == 12 and [json.loads(value)['id'] for value in values] == list(range(12))


def test_settings_initialized_once_under_concurrency(monkeypatch):
    calls = []
    version_calls = []

    def read_version(*args, **kwargs):
        version_calls.append((args, kwargs))
        return SimpleNamespace(stdout='codex-cli 0.144.1')

    monkeypatch.setattr(transport, '_settings', None)
    monkeypatch.setattr(transport.subprocess, 'run', read_version)
    monkeypatch.setattr(transport, 'read_config_layers', lambda: calls.append(True) or {'layers': []})
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda value: transport.settings(), range(12)))
    assert len(calls) == 1 and all(value == values[0] for value in values)
    assert len(version_calls) == 1
    assert version_calls[0][0] == (['codex', '--version'],)
    assert version_calls[0][1]['timeout'] == transport.CONFIG_TIMEOUT_SECONDS


def test_settings_does_not_cache_failed_version_preflight(monkeypatch):
    version_calls = []
    config_calls = []

    def read_version(*args, **kwargs):
        version_calls.append(kwargs['timeout'])
        if len(version_calls) == 1:
            raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
        return SimpleNamespace(stdout='codex-cli 0.144.1')

    monkeypatch.setattr(transport, '_settings', None)
    monkeypatch.setattr(transport.subprocess, 'run', read_version)
    monkeypatch.setattr(transport, 'read_config_layers', lambda:
                        config_calls.append(True) or {'layers': []})
    with pytest.raises(subprocess.TimeoutExpired):
        transport.settings()
    assert transport._settings is None and not config_calls
    value = transport.settings()
    assert value == transport.provider_config({'layers': []})
    assert transport.settings() == value
    assert version_calls == [transport.CONFIG_TIMEOUT_SECONDS] * 2
    assert len(config_calls) == 1


def test_probe_bypasses_project_cache(monkeypatch):
    requests = []
    monkeypatch.setattr(transport, 'settings', lambda: transport.provider_config({'layers': []}))
    monkeypatch.setattr(transport, 'client', lambda config: SimpleNamespace(
        _run=lambda prompt, schema: requests.append((config, prompt, schema)) or '{"ok":true}',
        run=lambda *args: pytest.fail('must not use cache')))
    result = transport.probe('gpt-6-luna')
    assert result['model_request_count'] == 1 and result['project_case_count'] == 0
    assert requests[0][0]['model'] == 'gpt-6-luna' and len(requests) == 1
    assert transport._public_probe.get() is False


def test_probe_restores_public_diagnostics_after_failure(monkeypatch):
    def fail(prompt, schema):
        assert transport._public_probe.get() is True
        raise RuntimeError('connection failed')

    monkeypatch.setattr(transport, 'settings', lambda: transport.provider_config({'layers': []}))
    monkeypatch.setattr(transport, 'client', lambda config: SimpleNamespace(_run=fail))
    with pytest.raises(RuntimeError, match='connection failed'):
        transport.probe('gpt-6-luna')
    assert transport._public_probe.get() is False


def test_check_requires_live_flag_and_preserves_pointers(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(transport, 'probe', lambda model, **kwargs: calls.append(model) or {'model': model})
    assert check.main(['codex-transport']) == 0
    value = json.loads(capsys.readouterr().out)
    assert not calls and value['model_requests_prohibited']
    assert check.main(['codex-transport', '--live', '--model', 'gpt-6-luna']) == 0
    value = json.loads(capsys.readouterr().out)
    assert calls == ['gpt-6-luna'] and not value['model_requests_prohibited']


@pytest.mark.parametrize('method', ['config/read', 'account/rateLimits/read'])
@pytest.mark.parametrize('failure', [None, 'initialize', 'read'])
def test_metadata_reader_only_reads_and_closes_process(monkeypatch, method, failure):
    requests, lifecycle, starts = [], [], []
    process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None,
        terminate=lambda: lifecycle.append('terminate'), kill=lambda: lifecycle.append('kill'))

    def wait(timeout=None):
        lifecycle.append(('wait', timeout))
        if timeout is not None:
            raise subprocess.TimeoutExpired('codex', timeout)

    process.wait = wait

    class Selector:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            lifecycle.append('selector_closed')

        def register(self, stream, events):
            assert stream is process.stdout and events == transport.selectors.EVENT_READ

    def rpc(actual, selector, pending, request, deadline):
        assert actual is process and isinstance(selector, Selector)
        assert deadline > transport.time.monotonic()
        requests.append(request)
        if failure == 'initialize' or (failure == 'read' and request['id'] == 2):
            raise RuntimeError('metadata failed')
        if request['id'] == 2:
            assert json.loads(process.stdin.getvalue()) == {'method': 'initialized'}
            assert pending == b'pending'
        return {'layers': []}, b'pending'

    monkeypatch.setattr(transport.subprocess, 'Popen',
        lambda command, **kwargs: starts.append((command, kwargs)) or process)
    monkeypatch.setattr(transport.selectors, 'DefaultSelector', Selector)
    monkeypatch.setattr(transport, '_rpc', rpc)
    if failure:
        with pytest.raises(RuntimeError, match='metadata failed'):
            transport._read_metadata(method, timeout_seconds=5)
    else:
        assert transport._read_metadata(method, timeout_seconds=5) == {'layers': []}
    assert starts[0][0] == ['codex', 'app-server']
    assert requests[0]['method'] == 'initialize'
    if failure != 'initialize':
        assert requests[1] == ({'id': 2, 'method': method, 'params': {'includeLayers': True,
            'cwd': starts[0][1]['cwd']}} if method == 'config/read' else
            {'id': 2, 'method': method})
    assert lifecycle == ['terminate', ('wait', 5), 'kill', ('wait', None), 'selector_closed']
    assert process.stdin.closed and process.stdout.closed
    assert starts[0][1]['stderr'].closed
    assert not Path(starts[0][1]['cwd']).exists()


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_metadata_reader_rejects_invalid_timeout_before_start(monkeypatch, timeout):
    monkeypatch.setattr(transport.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('no process'))
    with pytest.raises(ValueError, match='timeout'):
        transport.read_rate_limits(timeout)


def test_metadata_reader_rejects_mutating_method_before_start(monkeypatch):
    monkeypatch.setattr(transport.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('no process'))
    with pytest.raises(ValueError, match='Unsupported'):
        transport._read_metadata('account/rateLimitResetCredit/consume')


@pytest.mark.parametrize('multiple', [False, True])
def test_rate_limits_keep_only_window_metadata(monkeypatch, multiple):
    bucket = {'primary': {'usedPercent': 100, 'windowDurationMins': 300,
                         'resetsAt': 1791800000, 'secret': 'private'},
              'secondary': None, 'credits': {'balance': 'private'}, 'account': 'private'}
    response = {'rateLimits': bucket, 'rateLimitResetCredits': {'credits': ['private']}}
    if multiple:
        response['rateLimitsByLimitId'] = {'codex': {**bucket, 'rateLimitReachedType': 'usage_limit'}}
    calls = []
    monkeypatch.setattr(transport, '_read_metadata',
        lambda method, timeout: calls.append((method, timeout)) or response)
    result = transport.read_rate_limits(15)
    assert calls == [('account/rateLimits/read', 15)]
    assert result == {'rate_limits': [{'limit_id': 'codex',
        'primary': {'used_percent': 100, 'window_duration_minutes': 300, 'resets_at': 1791800000},
        'secondary': None, 'reported_limit_reached': True if multiple else None}],
        'metadata_only': True, 'model_request_count': 0, 'production_recovery_confirmed': False}
    assert 'private' not in json.dumps(result)


@pytest.mark.parametrize('response', [None, {}, {'rateLimits': {'primary': None}},
    {'rateLimits': {'primary': {'usedPercent': True}}},
    {'rateLimits': {'primary': {'usedPercent': float('nan')}}},
    {'rateLimits': {'primary': {'usedPercent': 50, 'resetsAt': 'private'}}},
    {'rateLimitsByLimitId': {'private/url': {'primary': {'usedPercent': 50}}}}])
def test_rate_limits_missing_or_invalid_fail_without_probe(monkeypatch, response):
    monkeypatch.setattr(transport, '_read_metadata', lambda *args: response)
    monkeypatch.setattr(transport, 'probe', lambda *args: pytest.fail('no model request'))
    with pytest.raises(RuntimeError, match='Codex account rate.limit'):
        transport.read_rate_limits()


def test_check_rate_limits_are_read_only_and_mutually_exclusive(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(transport, 'probe', lambda *args: pytest.fail('no model request'))
    monkeypatch.setattr(transport, 'read_rate_limits',
        lambda **kwargs: calls.append(kwargs) or {'model_request_count': 0})
    assert check.main(['codex-transport', '--rate-limits', '--timeout-seconds', '15']) == 0
    value = json.loads(capsys.readouterr().out)
    assert calls == [{'timeout_seconds': 15}]
    assert value['model_requests_prohibited'] and not value['evaluation_started']
    assert value['project_case_count'] == 0
    assert value['checks'][0]['name'] == 'codex_account_rate_limits'
    with pytest.raises(SystemExit):
        check.main(['codex-transport', '--rate-limits', '--live'])
