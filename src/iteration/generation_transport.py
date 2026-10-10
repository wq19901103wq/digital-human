"""Bounded generation connection probes and opt-in worker-local proxy bypass."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

DIRECT_HOSTS_ENV = 'DH_LLM_DIRECT_HOSTS'
MODE = 'endpoint_host_proxy_bypass_v1'


def direct_hosts(value=None):
    value = os.environ.get(DIRECT_HOSTS_ENV, '') if value is None else value
    hosts = tuple(dict.fromkeys(host.strip().lower() for host in value.split(',') if host.strip()))
    if any(len(host) > 253 or not re.fullmatch(
            r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?', host) or
            any(not part or len(part) > 63 or part.startswith('-') or part.endswith('-')
                for part in host.split('.')) for host in hosts):
        raise ValueError('generation direct routes require explicit hostnames, without ports or wildcards')
    return hosts


@contextmanager
def direct_routes(hosts=None):
    hosts = direct_hosts() if hosts is None else direct_hosts(','.join(hosts))
    if not hosts:
        yield
        return
    previous = {name: os.environ.get(name) for name in ('NO_PROXY', 'no_proxy')}
    inherited = previous['no_proxy'] if previous['no_proxy'] is not None else previous['NO_PROXY']
    merged = ','.join(filter(None, (inherited, *hosts)))
    try:
        os.environ['NO_PROXY'] = os.environ['no_proxy'] = merged
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _url(client):
    if client._protocol != 'anthropic':
        raise ValueError('generation connection probe currently requires the Anthropic adapter')
    url = client._base_url
    if not url.endswith('/messages'):
        url += '/messages' if url.endswith('/v1') else '/v1/messages'
    parts = urlsplit(url)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
        raise ValueError('generation connection probe requires an HTTPS endpoint without URL credentials')
    return url


def https_probe(url, *, timeout_seconds=15):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('connection probe timeout must be positive and finite')
    started = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'),
                                    timeout=timeout_seconds) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
    return dict(http_status=status, reachable=200 <= status < 500 and status not in {408, 429},
                seconds=round(time.monotonic() - started, 3), authenticated=False)


def _connection_error(exc):
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    value = dict(error=f'{type(exc).__name__}: connection probe failed',
                 connection_error=type(reason).__name__)
    if isinstance(reason, ssl.SSLError):
        value.update(ssl_library=getattr(reason, 'library', None),
                     ssl_reason=getattr(reason, 'reason', None))
    return value


def probe_clients(clients, *, route='configured', live=False, timeout_seconds=15):
    if route not in {'configured', 'direct', 'both'}:
        raise ValueError('unknown generation connection route')
    routes = ('configured', 'direct') if route == 'both' else (route,)
    clients = list({json.dumps(client.cache_identity(), sort_keys=True): client
                    for client in clients}.values())
    if not clients:
        raise ValueError('generation connection probe requires at least one client')
    urls = {_url(client) for client in clients}
    hosts = tuple(sorted({urlsplit(url).hostname for url in urls}))
    rows = []
    for selected in routes:
        with direct_routes(hosts if selected == 'direct' else ()):
            for url in sorted(urls):
                item = dict(route=selected, host=urlsplit(url).hostname, kind='https')
                try:
                    item.update(https_probe(url, timeout_seconds=timeout_seconds))
                except Exception as exc:
                    item.update(reachable=False, **_connection_error(exc))
                rows.append(item)
            if live and all(item.get('reachable') for item in rows if item['route'] == selected):
                for client in clients:
                    item = dict(route=selected, model=client.model_name, kind='public_model',
                                timeout_seconds=client._timeout)
                    started = time.monotonic()
                    body = dict(model=client._model, temperature=client._temperature,
                                max_tokens=client._max_tokens,
                                system='只输出合法 JSON，不要输出任何其他文字。',
                                messages=[dict(role='user', content='Reply exactly {"ok":true}.')])
                    try:
                        result = client._chat_anthropic(body, {})
                        item.update(reachable=json.loads(result) == {'ok': True})
                    except Exception as exc:
                        item.update(reachable=False, error=str(client._error(exc)))
                    item['seconds'] = round(time.monotonic() - started, 3)
                    rows.append(item)
    return dict(mode=MODE, probes=rows, healthy=all(item['reachable'] for item in rows),
                model_request_count=sum(item['kind'] == 'public_model' for item in rows),
                retries=0, project_case_count=0)


def experiment_clients(directory, snapshot):
    import yaml
    from src.llm import build_clients
    from src.iteration import versions
    from src.infrastructure.storage import read_json

    settings = yaml.safe_load((snapshot / 'settings.snapshot.yaml').read_text())
    spec = read_json(directory / 'spec.json')
    if spec.get('kind') != 'gen_ab':
        raise ValueError('generation routes require a frozen generator comparison')
    clients = {}
    for role in ('baseline_ref', 'candidate_ref'):
        model_cfg = versions.load_generator(spec[role])['config']['llm']
        for client in build_clients(settings, model_cfg).values():
            clients[json.dumps(client.cache_identity(), sort_keys=True)] = client
    return list(clients.values()), spec


def probe_experiment(instance, experiment_id, **kwargs):
    from src.infrastructure.config import load_settings
    from src.infrastructure.config import valid_name
    from src.iteration import runtime, versions

    load_settings()
    versions.switch_instance(instance)
    directory = versions.PRIVATE / 'experiments' / valid_name(experiment_id)
    snapshot = runtime.verify(directory / 'runtime.json')
    clients, spec = experiment_clients(directory, snapshot)
    value = probe_clients(clients, **kwargs)
    value.update(experiment=experiment_id, runtime_ref=snapshot.name,
                 comparison={name: spec[name] for name in
                             ('data_ref', 'baseline_ref', 'candidate_ref', 'judge_ref')})
    return value


@contextmanager
def worker_routes(snapshot, directory, *, provenance):
    from pathlib import Path
    import uuid
    from src import llm
    from src.infrastructure.config import load_settings
    from src.infrastructure.storage import write_once_json

    hosts = direct_hosts()
    if not hosts:
        yield
        return
    if not Path(llm.__file__).resolve().is_relative_to(snapshot.resolve()):
        raise ValueError('generation routes require an authenticated archive client')
    load_settings()
    clients, spec = experiment_clients(directory, snapshot)
    endpoints = {urlsplit(_url(client)).hostname for client in clients}
    if not set(hosts).issubset(endpoints):
        raise ValueError('generation direct hosts must belong to frozen generator endpoints')
    with direct_routes(hosts):
        probe = probe_clients(clients)
        receipt = dict(schema=1, kind='generation_direct_routes', mode=MODE,
                       runtime_ref=snapshot.name, direct_hosts=list(hosts),
                       comparison={name: spec[name] for name in
                                   ('data_ref', 'baseline_ref', 'candidate_ref', 'judge_ref')},
                       preflight=probe, created_at=time.time(), pid=os.getpid(), **provenance)
        write_once_json(directory / 'generation_transport_overrides' / (uuid.uuid4().hex + '.json'),
                        receipt)
        if not probe['healthy']:
            raise RuntimeError('generation connection preflight failed; worker was not started')
        yield
