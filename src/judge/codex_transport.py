"""HTTP-only transport for the unchanged, version-pinned Codex Judge client."""
from __future__ import annotations

import hashlib
import json
import os
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
_settings_lock = Lock()
_settings = None


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


def read_config_layers():
    """Read metadata only; no thread, prompt, tool, or model request is created."""
    with tempfile.TemporaryDirectory(prefix='dh-codex-config-') as directory, \
            tempfile.TemporaryFile() as errors, selectors.DefaultSelector() as selector:
        process = subprocess.Popen(['codex', 'app-server'], cwd=directory,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=errors, text=True)
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + CONFIG_TIMEOUT_SECONDS
            _, pending = _rpc(process, selector, b'', {'id': 1, 'method': 'initialize',
                'params': {'clientInfo': {'name': 'digital_human_transport', 'version': '1'},
                           'capabilities': {'experimentalApi': True}}}, deadline)
            process.stdin.write(json.dumps({'method': 'initialized'}) + '\n')
            process.stdin.flush()
            response, _ = _rpc(process, selector, pending, {'id': 2, 'method': 'config/read',
                'params': {'includeLayers': True, 'cwd': directory}},
                time.monotonic() + CONFIG_TIMEOUT_SECONDS)
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
                                     check=True, timeout=10).stdout.strip().split()[-1]
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


class _TransportSubprocess:
    _digital_human_transport = MODE

    def __init__(self, original, receipt_directory, provenance):
        self._original = original
        self._receipt_directory = receipt_directory
        self._provenance = provenance
        self._receipt_lock = Lock()
        self._recorded = False

    def __getattr__(self, name):
        return getattr(self._original, name)

    def run(self, command, **kwargs):
        if not judge_command(command):
            return self._original.run(command, **kwargs)
        config = settings()
        with self._receipt_lock:
            if not self._recorded and self._receipt_directory is not None:
                self._receipt_directory.mkdir(parents=True, exist_ok=True)
                value = {'schema': 1, 'mode': MODE, 'codex_cli_version': CLI_VERSION,
                         'provider_sha256': _digest(config), 'supports_websockets': False,
                         'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'created_at': time.time(), 'pid': os.getpid(), **self._provenance}
                with (self._receipt_directory / (uuid.uuid4().hex + '.json')).open('x') as output:
                    json.dump(value, output, ensure_ascii=False, sort_keys=True)
                self._recorded = True
        overrides = ['--config', f'model_provider="{PROVIDER}"', '--config',
                     f'model_providers.{PROVIDER}=' + _toml(config)]
        return self._original.run(command[:-1] + overrides + ['-'], **kwargs)


def install(module=None, *, expected_root=None, receipt_directory=None, provenance=None):
    """Patch only the authenticated Judge module for the entire worker lifetime."""
    if module is None:
        from . import corrected as module
    if expected_root is not None and not Path(module.__file__).resolve().is_relative_to(
            Path(expected_root).resolve()):
        raise RuntimeError('Codex transport target is outside the authenticated runtime')
    if not hasattr(module, 'CodexJudgeClient'):
        raise RuntimeError('Codex transport target has no pinned Judge client')
    if getattr(module.subprocess, '_digital_human_transport', None) == MODE:
        return
    module.subprocess = _TransportSubprocess(module.subprocess,
        Path(receipt_directory) if receipt_directory is not None else None, provenance or {})


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
    raw = judge._run('Return only the JSON object {"ok":true}. Do not use tools.',
                     {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                      'required': ['ok'], 'additionalProperties': False})
    if json.loads(raw) != {'ok': True}:
        raise RuntimeError('Codex transport probe returned an unexpected response')
    return {'model': model, 'mode': MODE, 'preflight_seconds': preflight_seconds,
            'request_seconds': time.monotonic() - started, 'provider_sha256': _digest(config),
            'model_request_count': 1, 'project_case_count': 0}
