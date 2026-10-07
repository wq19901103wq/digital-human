"""Bounded connection diagnostics and worker-local generation routing."""
from contextlib import contextmanager
import io
import json
import os
import ssl
from types import SimpleNamespace
import urllib.error

import pytest

from src.iteration import generation_transport as transport, pack_transport
from src.llm import ChatClient


def client(monkeypatch, model='example-model'):
    monkeypatch.setenv('TEST_LLM_URL', 'https://api.example.com/coding')
    monkeypatch.setenv('TEST_LLM_KEY', 'secret-example-key')
    return ChatClient({'llm': {'base_url_env': 'TEST_LLM_URL', 'api_key_env': 'TEST_LLM_KEY',
                              'max_retries': 2}},
                      {'model': model, 'protocol': 'anthropic', 'temperature': 0.7,
                       'max_tokens': 1234, 'timeout_seconds': 20})


@pytest.mark.parametrize('value', ['https://api.example.com', '*', 'api.example.com:443',
                                  'api..example.com', '-api.example.com', 'api/.example.com'])
def test_direct_hosts_rejects_non_host_routes(value):
    with pytest.raises(ValueError, match='explicit hostnames'):
        transport.direct_hosts(value)


def test_direct_hosts_normalizes_and_deduplicates():
    assert transport.direct_hosts('API.example.com,api.example.com, other.example.com') == (
        'api.example.com', 'other.example.com')


@pytest.mark.parametrize('lowercase', [None, '', 'lower.example.com'])
def test_direct_routes_restores_environment_even_after_error(monkeypatch, lowercase):
    monkeypatch.setenv('NO_PROXY', 'localhost')
    if lowercase is None:
        monkeypatch.delenv('no_proxy', raising=False)
    else:
        monkeypatch.setenv('no_proxy', lowercase)
    before = {name: os.environ.get(name) for name in ('NO_PROXY', 'no_proxy')}
    with pytest.raises(RuntimeError):
        with transport.direct_routes(('api.example.com',)):
            inherited = 'localhost' if lowercase is None else lowercase
            assert os.environ['NO_PROXY'] == ','.join(filter(None, (inherited, 'api.example.com')))
            assert os.environ['no_proxy'] == os.environ['NO_PROXY']
            raise RuntimeError('worker failed')
    assert {name: os.environ.get(name) for name in before} == before


def test_urllib_proxy_handler_bypasses_only_selected_host(monkeypatch):
    monkeypatch.setenv('NO_PROXY', 'localhost')
    monkeypatch.delenv('no_proxy', raising=False)
    handler = transport.urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7897'})
    with transport.direct_routes(('api.example.com',)):
        direct = transport.urllib.request.Request('https://api.example.com/v1/messages')
        proxied = transport.urllib.request.Request('https://other.example.com/v1/messages')
        handler.proxy_open(direct, 'http://127.0.0.1:7897', 'https')
        handler.proxy_open(proxied, 'http://127.0.0.1:7897', 'https')
        assert direct.host == 'api.example.com' and direct._tunnel_host is None
        assert proxied.host == '127.0.0.1:7897' and proxied._tunnel_host == 'other.example.com'
    restored = transport.urllib.request.Request('https://api.example.com/v1/messages')
    handler.proxy_open(restored, 'http://127.0.0.1:7897', 'https')
    assert restored.host == '127.0.0.1:7897'


def test_head_is_single_unauthenticated_request(monkeypatch):
    calls = []

    def send(request, *, timeout):
        calls.append((request, timeout))
        raise urllib.error.HTTPError(request.full_url, 401, 'unauthorized', {}, io.BytesIO())

    monkeypatch.setattr(transport.urllib.request, 'urlopen', send)
    result = transport.https_probe('https://api.example.com/v1/messages')
    assert result['reachable'] and result['authenticated'] is False
    assert len(calls) == 1 and calls[0][1] == 15
    assert calls[0][0].method == 'HEAD' and not calls[0][0].headers
    assert calls[0][0].data is None


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_probe_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match='positive and finite'):
        transport.https_probe('https://api.example.com', timeout_seconds=timeout)


def test_head_error_diagnostics_do_not_leak_exception_data(monkeypatch):
    model = client(monkeypatch)

    def fail(*args, **kwargs):
        raise urllib.error.URLError(ssl.SSLEOFError(8, 'secret-example-key private prompt'))

    monkeypatch.setattr(transport, 'https_probe', fail)
    result = transport.probe_clients([model], live=True)
    assert result['healthy'] is False and result['model_request_count'] == 0
    assert result['probes'][0]['connection_error'] == 'SSLEOFError'
    assert 'secret-example-key' not in json.dumps(result)
    assert 'private prompt' not in json.dumps(result)


def test_live_probe_preserves_request_settings_and_identity(monkeypatch):
    model = client(monkeypatch)
    identity = model.cache_identity()
    calls = []
    monkeypatch.setenv('NO_PROXY', 'localhost')
    monkeypatch.delenv('no_proxy', raising=False)
    monkeypatch.setattr(transport, 'https_probe', lambda *args, **kwargs: {'reachable': True})

    def send(body, event):
        calls.append((body, model._timeout, os.environ.get('no_proxy')))
        return '{"ok":true}'

    monkeypatch.setattr(model, '_chat_anthropic', send)
    result = transport.probe_clients([model, model], route='both', live=True)
    assert result['healthy'] and result['model_request_count'] == 2
    assert result['retries'] == result['project_case_count'] == 0
    assert [row[2] for row in calls] == [None, 'localhost,api.example.com']
    assert all(row[0]['temperature'] == 0.7 and row[0]['max_tokens'] == 1234 and
               row[0]['model'] == model.model_name and row[1] == 20 for row in calls)
    assert model.cache_identity() == identity
    assert os.environ['NO_PROXY'] == 'localhost' and 'no_proxy' not in os.environ


def test_live_probe_never_retries_and_redacts_key(monkeypatch):
    model = client(monkeypatch)
    calls = []
    monkeypatch.setattr(transport, 'https_probe', lambda *args, **kwargs: {'reachable': True})

    def fail(*args):
        calls.append(True)
        raise OSError('secret-example-key disconnected')

    monkeypatch.setattr(model, '_chat_anthropic', fail)
    result = transport.probe_clients([model], live=True)
    assert result['healthy'] is False and result['model_request_count'] == len(calls) == 1
    assert 'secret-example-key' not in json.dumps(result)


def test_probe_rejects_empty_clients():
    with pytest.raises(ValueError, match='at least one client'):
        transport.probe_clients([])


@pytest.mark.parametrize('url', ['http://api.example.com', 'https://user:secret@api.example.com'])
def test_probe_rejects_unsafe_endpoint(monkeypatch, url):
    model = client(monkeypatch)
    model._base_url = url
    with pytest.raises(ValueError, match='HTTPS endpoint'):
        transport.probe_clients([model])


@pytest.mark.parametrize('healthy', [True, False])
def test_worker_routes_records_preflight_and_restores(monkeypatch, tmp_path, healthy):
    from src import llm, config
    model = client(monkeypatch)
    snapshot = tmp_path / 'snapshot'
    directory = tmp_path / 'experiment'
    monkeypatch.setattr(llm, '__file__', str(snapshot / 'src/llm.py'))
    monkeypatch.setattr(config, 'load_settings', lambda: {})
    monkeypatch.setenv(transport.DIRECT_HOSTS_ENV, 'api.example.com')
    monkeypatch.setenv('NO_PROXY', 'localhost')
    monkeypatch.delenv('no_proxy', raising=False)
    spec = dict(kind='gen_ab', data_ref='d-example', baseline_ref='g-base',
                candidate_ref='g-candidate', judge_ref='j-example')
    monkeypatch.setattr(transport, 'experiment_clients', lambda *args: ([model], spec))
    calls = []

    def probe(clients):
        calls.append(os.environ['no_proxy'])
        return {'healthy': healthy, 'model_request_count': 0}

    monkeypatch.setattr(transport, 'probe_clients', probe)
    if healthy:
        with transport.worker_routes(snapshot, directory, provenance={'bootstrap_sha256': 'digest'}):
            assert os.environ['no_proxy'] == 'localhost,api.example.com'
    else:
        with pytest.raises(RuntimeError, match='worker was not started'):
            with transport.worker_routes(snapshot, directory, provenance={'bootstrap_sha256': 'digest'}):
                pytest.fail('unhealthy worker started')
    assert calls == ['localhost,api.example.com']
    assert os.environ['NO_PROXY'] == 'localhost' and 'no_proxy' not in os.environ
    receipts = list((directory / 'generation_transport_overrides').glob('*.json'))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt['comparison'] == {name: spec[name] for name in spec if name != 'kind'}
    assert receipt['direct_hosts'] == ['api.example.com']
    assert receipt['bootstrap_sha256'] == 'digest' and receipt['preflight']['healthy'] == healthy


@pytest.mark.parametrize('authenticated', [True, False])
def test_worker_rejects_unbound_route_or_client(monkeypatch, tmp_path, authenticated):
    from src import llm, config
    model = client(monkeypatch)
    monkeypatch.setattr(config, 'load_settings', lambda: {})
    monkeypatch.setattr(llm, '__file__', str((tmp_path if authenticated else tmp_path.parent) / 'src/llm.py'))
    monkeypatch.setenv(transport.DIRECT_HOSTS_ENV, 'unbound.example.com')
    monkeypatch.setattr(transport, 'experiment_clients', lambda *args: ([model], {}))
    with pytest.raises(ValueError, match='frozen generator endpoints' if authenticated else 'archive client'):
        with transport.worker_routes(tmp_path, tmp_path / 'job', provenance={}):
            pytest.fail('unbound route entered')


def test_modern_dispatch_installs_route_around_archive_run(monkeypatch, tmp_path):
    events = []

    @contextmanager
    def routes(*args):
        events.append('route-enter')
        try:
            yield
        finally:
            events.append('route-exit')

    driver = SimpleNamespace(_run_modern=lambda *args: events.append('archive-run') or 'done')
    monkeypatch.setattr(pack_transport, 'load_external_module', lambda *args: driver)
    monkeypatch.setattr(pack_transport, 'install_judge_transport', lambda *args: None)
    monkeypatch.setattr(pack_transport, 'install_modern_recall_transport', lambda *args: None)
    monkeypatch.setattr(pack_transport, 'generation_routes', routes)
    assert pack_transport.dispatch_worker(SimpleNamespace(timeout_seconds=20), tmp_path,
                                          tmp_path / 'job', 'fewshot') == 'done'
    assert events == ['route-enter', 'archive-run', 'route-exit']


def test_legacy_dispatch_rejects_explicit_direct_route(monkeypatch, tmp_path):
    monkeypatch.setenv(transport.DIRECT_HOSTS_ENV, 'api.example.com')
    with pytest.raises(ValueError, match='authenticated modern worker'):
        pack_transport.dispatch_worker(SimpleNamespace(), tmp_path, tmp_path / 'job', 'few_shot')
