"""HTTP-only transport for the unchanged, version-pinned Codex Judge client."""
from __future__ import annotations

import hashlib
import json
import math
import os
from functools import wraps
from contextvars import ContextVar
from pathlib import Path
import re
import selectors
import subprocess
import tempfile
from threading import Lock
import time
import uuid


MODE = 'codex_openai_http_v1'
CLI_VERSION = '0.144.1'
PROVIDER = 'digital_human_https'
CONFIG_TIMEOUT_SECONDS = 180
FEATURE_FORMAT_ATTEMPTS = 2
SESSION_CONFIG = {'features.apps': False}
_settings_lock = Lock()
_settings = None
_failure_details = ContextVar('codex_transport_failure_details', default=None)
_public_probe = ContextVar('codex_transport_public_probe', default=False)
SERVICE_ERROR_CODES = frozenset({'rate_limit_exceeded', 'insufficient_quota',
    'usage_limit_reached', 'invalid_api_key', 'unauthorized', 'permission_denied',
    'model_not_found', 'server_error', 'service_unavailable'})


class CodexTransportError(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('Codex transport diagnostics: ' + json.dumps(details, sort_keys=True))


class CodexServiceBlocked(CodexTransportError):
    pass


def retryable_connection_failure(error):
    if not isinstance(error, CodexTransportError) or isinstance(error, CodexServiceBlocked):
        return False
    detail = error.details
    if not detail.get('exit_code') or set(detail.get('signals', [])) & {
            'rate_limit', 'authentication_error', 'permission_denied',
            'unsupported_model', 'invalid_request'}:
        return False
    if any(item.get('code') in SERVICE_ERROR_CODES or item.get('http_statuses')
           for item in detail.get('service_errors', [])):
        return False
    connections = detail.get('connection_errors', {})
    if any(set(kinds) & {'tls_failure', 'proxy_connect_failure', 'operation_not_permitted'}
           for kinds in connections.values()):
        return False
    return bool(detail.get('failure_kind') == 'subprocess_timeout'
                or set(detail.get('terminal_signals', [])) & {
                    'stream_disconnected', 'request_timeout'}
                or set(connections.get('terminal', [])) & {
                    'request_send_failure', 'response_body_failure', 'stream_incomplete',
                    'dns_resolution_failure', 'connection_refused', 'connection_reset'})


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _rpc(process, selector, pending, request, deadline):
    process.stdin.write(json.dumps(request) + '\n')
    process.stdin.flush()
    while time.monotonic() < deadline:
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            if not line.strip():
                continue
            response = json.loads(line)
            if response.get('id') == request['id']:
                if 'error' in response:
                    raise RuntimeError(f'Codex configuration preflight rejected {request["method"]}')
                return response['result'], pending
        if not selector.select(max(0, deadline - time.monotonic())):
            break
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk:
            raise RuntimeError(f'Codex configuration preflight exited during {request["method"]}')
        pending += chunk
    raise TimeoutError(f'Codex configuration preflight timed out during {request["method"]}')


def _read_metadata(method, timeout_seconds=CONFIG_TIMEOUT_SECONDS):
    """Read metadata only; no thread, prompt, tool, or model request is created."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('Codex metadata timeout must be positive and finite')
    if method not in {'config/read', 'account/rateLimits/read'}:
        raise ValueError('Unsupported Codex metadata method')
    with tempfile.TemporaryDirectory(prefix='dh-codex-config-') as directory, \
            tempfile.TemporaryFile() as errors, selectors.DefaultSelector() as selector:
        process = subprocess.Popen(['codex', 'app-server'], cwd=directory,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=errors, text=True)
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout_seconds
            _, pending = _rpc(process, selector, b'', {'id': 1, 'method': 'initialize',
                'params': {'clientInfo': {'name': 'digital_human_transport', 'version': '1'},
                           'capabilities': {'experimentalApi': True}}}, deadline)
            process.stdin.write(json.dumps({'method': 'initialized'}) + '\n')
            process.stdin.flush()
            request = {'id': 2, 'method': method}
            if method == 'config/read':
                request['params'] = {'includeLayers': True, 'cwd': directory}
            response, _ = _rpc(process, selector, pending, request,
                               time.monotonic() + timeout_seconds)
            return response
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()


def read_config_layers():
    return _read_metadata('config/read')


def read_rate_limits(timeout_seconds=CONFIG_TIMEOUT_SECONDS):
    response = _read_metadata('account/rateLimits/read', timeout_seconds)
    if not isinstance(response, dict):
        raise RuntimeError('Codex account rate limits are unavailable')
    buckets = response.get('rateLimitsByLimitId')
    if not isinstance(buckets, dict) or not buckets:
        buckets = {'codex': response.get('rateLimits')}
    limits = []
    for limit_id, bucket in sorted(buckets.items()):
        if not isinstance(bucket, dict) or not isinstance(limit_id, str) \
                or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', limit_id):
            raise RuntimeError('Codex account rate limits are unavailable')
        limit = {'limit_id': limit_id}
        for name in ('primary', 'secondary'):
            window = bucket.get(name)
            if window is None:
                limit[name] = None
                continue
            if not isinstance(window, dict):
                raise RuntimeError('Invalid Codex account rate-limit window')
            used = window.get('usedPercent')
            if isinstance(used, bool) or not isinstance(used, (int, float)) \
                    or not math.isfinite(used) or used < 0:
                raise RuntimeError('Invalid Codex account rate-limit usage')
            safe_window = {'used_percent': used}
            for source, target in (('windowDurationMins', 'window_duration_minutes'),
                                   ('resetsAt', 'resets_at')):
                value = window.get(source)
                if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                          or value < 0):
                    raise RuntimeError('Invalid Codex account rate-limit window')
                safe_window[target] = value
            limit[name] = safe_window
        if limit['primary'] is None and limit['secondary'] is None:
            raise RuntimeError('Codex account rate-limit windows are unavailable')
        limit['reported_limit_reached'] = (bucket['rateLimitReachedType'] is not None
            if 'rateLimitReachedType' in bucket else None)
        limits.append(limit)
    return {'rate_limits': limits, 'metadata_only': True, 'model_request_count': 0,
            'production_recovery_confirmed': False}


def provider_config(response, version=CLI_VERSION):
    """Mirror the pinned built-in provider, preserving managed endpoint overrides."""
    if version != CLI_VERSION:
        raise RuntimeError('HTTP transport requires the reviewed Codex CLI version')
    layers = response.get('layers')
    if not isinstance(layers, list):
        raise RuntimeError('Codex configuration layers are unavailable')
    retained = []
    allowed = {'mdm', 'system', 'enterpriseManaged', 'legacyManagedConfigTomlFromFile',
               'legacyManagedConfigTomlFromMdm'}
    for layer in layers:
        kind = layer['name']['type']
        if layer.get('disabledReason') or kind in {'user', 'project', 'sessionFlags'}:
            continue
        if kind not in allowed:
            raise RuntimeError('Unreviewed Codex configuration layer')
        config = layer['config']
        if config.get('features', {}).get('apps') is True:
            raise RuntimeError('Managed Codex configuration requires app integrations')
        if kind in {'legacyManagedConfigTomlFromFile', 'legacyManagedConfigTomlFromMdm'} and (
                'model_provider' in config):
            raise RuntimeError('Managed provider overrides the HTTP transport session flags')
        if config.get('profile') or config.get('profiles'):
            raise RuntimeError('Managed Codex profiles require a transport review')
        if PROVIDER in config.get('model_providers', {}):
            raise RuntimeError('HTTP transport provider ID already exists')
        retained.append(config)
    selected = next((config['model_provider'] for config in retained
                     if 'model_provider' in config), 'openai')
    if selected != 'openai':
        raise RuntimeError('HTTP transport cannot replace a managed custom provider')
    base_url = next((config['openai_base_url'] for config in retained
                     if 'openai_base_url' in config), None)
    if base_url is not None and (not isinstance(base_url, str) or not base_url):
        raise RuntimeError('Invalid managed OpenAI endpoint')
    config = {'name': 'OpenAI', 'wire_api': 'responses', 'requires_openai_auth': True,
              'supports_websockets': False, 'http_headers': {'version': version},
              'env_http_headers': {'OpenAI-Organization': 'OPENAI_ORGANIZATION',
                                   'OpenAI-Project': 'OPENAI_PROJECT'}}
    if base_url is not None:
        config['base_url'] = base_url
    return config


def settings():
    global _settings
    with _settings_lock:
        if _settings is None:
            version = subprocess.run(['codex', '--version'], text=True, capture_output=True,
                                     check=True, timeout=CONFIG_TIMEOUT_SECONDS).stdout.strip().split()[-1]
            _settings = provider_config(read_config_layers(), version)
        return dict(_settings)


def _toml(value):
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(key) + '=' + _toml(item)
                              for key, item in value.items()) + '}'
    return json.dumps(value, ensure_ascii=False)


def judge_command(command):
    if not isinstance(command, list) or len(command) not in {18, 20}:
        return False
    if command[:8] != ['codex', 'exec', '--ephemeral', '--ignore-user-config',
                       '--ignore-rules', '--sandbox', 'workspace-write', '--cd']:
        return False
    if command[9:12] != ['--skip-git-repo-check', '--json', '--output-last-message']:
        return False
    if command[13] != '--model' or command[15] != '--config' or command[-1] != '-':
        return False
    if not re.fullmatch(r'model_reasoning_effort="[a-z]+"', command[16]):
        return False
    directory = Path(command[8]).resolve()
    if Path(command[12]).resolve().parent != directory:
        return False
    return len(command) == 18 or (command[17] == '--output-schema'
                                  and Path(command[18]).resolve().parent == directory)


def _http_statuses(text):
    explicit = re.findall(r'(?i)\b(?:status(?:\s+code)?|http(?:/\d(?:\.\d)?)?)'
                          r'\s*[:=]?\s*([45]\d{2})\b', text)
    described = re.findall(r'(?i)\b(400|401|403|429|500|502|503|504)\s+'
                           r'(?:bad request|unauthorized|forbidden|permission denied|'
                           r'too many requests|internal server error|bad gateway|'
                           r'service unavailable|gateway timeout)\b', text)
    return sorted({int(status) for status in explicit + described})


def _failure_signals(text):
    text = text.lower()
    statuses = _http_statuses(text)
    signals = {'mcp_connection_error': 'rmcp::transport' in text and 'http/request failed' in text,
               'model_catalog_error': 'failed to refresh available models' in text,
               'stream_disconnected': 'stream disconnected' in text,
               'request_timeout': 'timed out' in text or 'timeout' in text,
               'authentication_error': 'unauthorized' in text or 'invalid_api_key' in text
                                       or 401 in statuses,
               'rate_limit': any(phrase in text for phrase in (
                   'rate limit', 'rate_limit_exceeded', 'insufficient_quota', 'usage_limit_reached'))
                             or 429 in statuses,
               'unsupported_model': 'model' in text and any(phrase in text for phrase in (
                   'not supported', 'not available', 'does not exist', 'unknown model',
                   'model_not_found')),
               'permission_denied': 'permission denied' in text or 'permission_denied' in text
                                    or 403 in statuses,
               'invalid_request': 'invalid request' in text or '400 bad request' in text}
    return [name for name, present in signals.items() if present]


def _public_error(message):
    message = re.sub(r'(?i)bearer\s+\S+', 'Bearer [redacted]', message)
    message = re.sub(r'(?i)\b(?:sk-|eyJ)[A-Za-z0-9_.-]+', '[redacted]', message)
    message = re.sub(r'(?i)https?://\S+', '[redacted-url]', message)
    return message[:512]


def _connection_errors(text):
    text = text.lower()
    patterns = {
        'request_send_failure': ('error sending request',),
        'response_body_failure': ('error decoding response body', 'error reading a body'),
        'stream_incomplete': ('stream ended before response.completed',),
        'dns_resolution_failure': ('dns error', 'failed to lookup address',
                                   'nodename nor servname provided'),
        'connection_refused': ('connection refused',),
        'connection_reset': ('connection reset', 'connection closed before message completed'),
        'tls_failure': ('tls handshake', 'certificate verify failed', 'invalid peer certificate'),
        'proxy_connect_failure': ('proxy authentication required', 'proxy connect error',
                                  'tunnel unsuccessful'),
        'operation_not_permitted': ('operation not permitted',),
    }
    return [kind for kind, phrases in patterns.items() if any(phrase in text for phrase in phrases)]


def failure_details(result, *, public_probe=False):
    terminal_events, messages, service_errors = [], [], []
    for line in (result.stdout or '').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get('type') not in {'error', 'turn.failed'}:
            continue
        terminal_events.append(event['type'])
        error = event.get('error')
        message = str(error.get('message', '')) if isinstance(error, dict) else str(event.get('message', ''))
        messages.append(message)
        fields = error if isinstance(error, dict) else event
        statuses = _http_statuses(message)
        status = fields.get('http_status', fields.get('status_code'))
        if type(status) is int and 400 <= status <= 599:
            statuses = sorted(set(statuses + [status]))
        safe = {}
        if statuses:
            safe['http_statuses'] = statuses
        code = fields.get('code')
        if isinstance(code, str) and code in SERVICE_ERROR_CODES:
            safe['code'] = code
        request_id = fields.get('request_id')
        if request_id is None:
            match = re.search(r'(?i)\brequest[_ ]id\s*[:=]\s*'
                              r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
                              r'|req_[A-Za-z0-9_-]{1,96})\b', message)
            request_id = match.group(1) if match else None
        if isinstance(request_id, str) and re.fullmatch(
                r'[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}|req_[A-Za-z0-9_-]{1,96}',
                request_id):
            safe['request_id'] = request_id
        if safe and safe not in service_errors:
            service_errors.append(safe)
    terminal_text = '\n'.join(messages)
    stderr = result.stderr or ''
    details = {'exit_code': result.returncode, 'terminal_events': terminal_events,
               'signals': _failure_signals(terminal_text + '\n' + stderr),
               'terminal_signals': _failure_signals(terminal_text),
               'stderr_signals': _failure_signals(stderr)}
    if service_errors:
        details['service_errors'] = service_errors
    connection_errors = {source: kinds for source, text in
                         (('terminal', terminal_text), ('stderr', stderr))
                         if (kinds := _connection_errors(text))}
    if connection_errors:
        details['connection_errors'] = connection_errors
    if public_probe:
        details['public_probe_errors'] = [_public_error(message) for message in messages[:3]]
    return details


class _TransportSubprocess:
    _digital_human_transport = MODE

    def __init__(self, original, receipt_directory, provenance, *, feature_format_retry=True,
                 tracing=None, stop_on_service_failure=False):
        self._original = original
        self._receipt_directory = receipt_directory
        self._provenance = provenance
        self._feature_format_retry = feature_format_retry
        self._tracing = tracing
        self._stop_on_service_failure = stop_on_service_failure
        self._service_lock = Lock()
        self._blocked = None
        self._receipt_lock = Lock()
        self._recorded = False

    def __getattr__(self, name):
        return getattr(self._original, name)

    def raise_if_blocked(self):
        with self._service_lock:
            if self._blocked is not None:
                raise CodexServiceBlocked(self._blocked)

    def run(self, command, **kwargs):
        if not judge_command(command):
            return self._original.run(command, **kwargs)
        self.raise_if_blocked()
        config = settings()
        with self._receipt_lock:
            if not self._recorded and self._receipt_directory is not None:
                self._receipt_directory.mkdir(parents=True, exist_ok=True)
                value = {'schema': 1, 'mode': MODE, 'codex_cli_version': CLI_VERSION,
                         'provider_sha256': _digest(config), 'supports_websockets': False,
                         'session_config': dict(SESSION_CONFIG),
                         'stop_on_service_failure': self._stop_on_service_failure,
                         'feature_format_retry': {'enabled': self._feature_format_retry,
                             'max_attempts': FEATURE_FORMAT_ATTEMPTS if self._feature_format_retry else 1,
                             'scope': 'registered_validator_with_schema', 'exception': 'ValueError'},
                         'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'created_at': time.time(), 'pid': os.getpid(), **self._provenance}
                with (self._receipt_directory / (uuid.uuid4().hex + '.json')).open('x') as output:
                    json.dump(value, output, ensure_ascii=False, sort_keys=True)
                self._recorded = True
        overrides = ['--config', f'model_provider="{PROVIDER}"', '--config',
                     f'model_providers.{PROVIDER}=' + _toml(config)]
        for name, value in SESSION_CONFIG.items():
            overrides += ['--config', name + '=' + _toml(value)]
        _failure_details.set(None)
        try:
            result = self._original.run(command[:-1] + overrides + ['-'], **kwargs)
        except subprocess.TimeoutExpired as error:
            captured = [value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
                        for value in (error.stdout, error.stderr)]
            detail = failure_details(subprocess.CompletedProcess([], 124,
                stdout=captured[0], stderr=captured[1]), public_probe=_public_probe.get())
            detail.update(exit_code_source='client_timeout', failure_kind='subprocess_timeout',
                          timeout_seconds=error.timeout)
            if 'request_timeout' not in detail['signals']:
                detail['signals'].append('request_timeout')
            self._record_failure(detail)
            raise
        if getattr(result, 'returncode', 0):
            self._record_failure(failure_details(result, public_probe=_public_probe.get()))
        return result

    def _record_failure(self, detail):
        _failure_details.set(detail)
        blocking = bool(set(detail['terminal_signals']) & {
            'rate_limit', 'authentication_error', 'permission_denied', 'unsupported_model'})
        blocking = blocking or any(item.get('code') in SERVICE_ERROR_CODES or
            any(status == 429 or status in (401, 403) or status >= 500
                for status in item.get('http_statuses', []))
            for item in detail.get('service_errors', []))
        if self._stop_on_service_failure and blocking:
            with self._service_lock:
                if self._blocked is None:
                    self._blocked = detail
        if self._tracing is not None:
            self._tracing.note('codex_transport_failure', detail)


def _install_failure_diagnostics(module):
    original = getattr(module.CodexJudgeClient, '_run', None)
    if not callable(original) or getattr(original, '_digital_human_failure_diagnostics', False):
        return

    @wraps(original)
    def run(client, prompt, schema=None):
        token = _failure_details.set(None)
        try:
            return original(client, prompt, schema)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            detail = _failure_details.get()
            if detail is None:
                raise
            raise CodexTransportError(detail) from error
        finally:
            _failure_details.reset(token)

    run._digital_human_failure_diagnostics = True
    module.CodexJudgeClient._run = run


def _install_feature_format_retry(module):
    original = module.CodexJudgeClient.run
    if getattr(original, '_digital_human_format_retry', False):
        return

    @wraps(original)
    def run(client, prompt, schema=None):
        validator = module.cache._validator.get()
        if schema is None or validator is None:
            return original(client, prompt, schema)
        for attempt in range(FEATURE_FORMAT_ATTEMPTS):
            raw = original(client, prompt, schema)
            try:
                if validator(raw) is False:
                    raise ValueError('Judge feature response rejected by its registered validator')
            except ValueError as error:
                module.tracing.note('codex_response_validation', {'valid': False,
                    'attempt': attempt + 1, 'max_attempts': FEATURE_FORMAT_ATTEMPTS,
                    'error_type': type(error).__name__})
                if attempt + 1 == FEATURE_FORMAT_ATTEMPTS:
                    raise
            else:
                module.tracing.note('codex_response_validation', {'valid': True,
                    'attempt': attempt + 1, 'max_attempts': FEATURE_FORMAT_ATTEMPTS})
                return raw

    run._digital_human_format_retry = True
    module.CodexJudgeClient.run = run


def install(module=None, *, expected_root=None, receipt_directory=None, provenance=None,
            feature_format_retry=True, stop_on_service_failure=False):
    """Patch only the authenticated Judge module for the entire worker lifetime."""
    if module is None:
        from . import corrected as module
    if expected_root is not None and not Path(module.__file__).resolve().is_relative_to(
            Path(expected_root).resolve()):
        raise RuntimeError('Codex transport target is outside the authenticated runtime')
    if not callable(getattr(getattr(module, 'CodexJudgeClient', None), 'run', None)):
        raise RuntimeError('Codex transport target has no pinned Judge client')
    if getattr(module.subprocess, '_digital_human_transport', None) == MODE:
        if module.subprocess._feature_format_retry != feature_format_retry:
            raise RuntimeError('Codex transport retry policy cannot change during a worker lifetime')
        if module.subprocess._stop_on_service_failure != stop_on_service_failure:
            raise RuntimeError('Codex transport service policy cannot change during a worker lifetime')
        return
    if feature_format_retry:
        _install_feature_format_retry(module)
    elif getattr(module.CodexJudgeClient.run, '_digital_human_format_retry', False):
        raise RuntimeError('Codex transport target already has a feature format retry wrapper')
    _install_failure_diagnostics(module)
    module.subprocess = _TransportSubprocess(module.subprocess,
        Path(receipt_directory) if receipt_directory is not None else None, provenance or {},
        feature_format_retry=feature_format_retry, tracing=getattr(module, 'tracing', None),
        stop_on_service_failure=stop_on_service_failure)


def check_available(module=None):
    if module is None:
        from . import corrected as module
    if getattr(module.subprocess, '_digital_human_transport', None) == MODE:
        module.subprocess.raise_if_blocked()


def client(config):
    from .corrected import CodexJudgeClient
    install()
    return CodexJudgeClient(config)


def probe(model, *, reasoning_effort='low', timeout_seconds=180):
    """One explicit public-prompt request; never uses project cases or result cache."""
    started = time.monotonic()
    config = settings()
    preflight_seconds = time.monotonic() - started
    judge = client({'provider': 'codex_cli', 'model': model,
                    'reasoning_effort': reasoning_effort, 'timeout_seconds': timeout_seconds,
                    'codex_cli_version': CLI_VERSION})
    started = time.monotonic()
    _failure_details.set(None)
    token = _public_probe.set(True)
    try:
        raw = judge._run('Return only the JSON object {"ok":true}. Do not use tools.',
                         {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                          'required': ['ok'], 'additionalProperties': False})
    except RuntimeError as error:
        if _failure_details.get() is None:
            raise
        raise RuntimeError('Codex transport probe failed: ' +
                           json.dumps(_failure_details.get(), sort_keys=True)) from error
    finally:
        _public_probe.reset(token)
    if json.loads(raw) != {'ok': True}:
        raise RuntimeError('Codex transport probe returned an unexpected response')
    return {'model': model, 'mode': MODE, 'preflight_seconds': preflight_seconds,
            'request_seconds': time.monotonic() - started, 'provider_sha256': _digest(config),
            'session_config': dict(SESSION_CONFIG),
            'model_request_count': 1, 'project_case_count': 0}
