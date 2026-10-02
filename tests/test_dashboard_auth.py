"""Local direct access is password-free; forwarded access requires owner credentials."""
import functools
import http.server
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.dashboard import auth, server


@pytest.fixture
def private_server(tmp_path, monkeypatch):
    root = tmp_path / 'site'
    root.mkdir()
    monkeypatch.setattr(server, 'ROOT', root)
    monkeypatch.setenv('DH_INSTANCES_ROOT', str(root / 'instances'))
    credentials = tmp_path / 'secrets' / 'login.json'
    auth.initialize(credentials)
    for version in ('s-0001', 's-0005'):
        content = root / 'instances' / 'demo' / 'share' / version / 'content'
        content.mkdir(parents=True)
        (content / 'wiki.xml').write_text('<wiki>SYNTHETIC_PRIVATE</wiki>')
    handler = functools.partial(server._AuditGuardHandler, directory=str(root), auth_file=credentials)
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}', root, credentials
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
@pytest.mark.parametrize('path', [
    '/dashboard/index.html', '/api/live/demo', '/api/trace/demo/run/' + 'a' * 32,
    '/dashboard/demo/versions/share/s-0005/index.html?view=attributes',
    '/instances/demo/share/s-0005/content/wiki.xml',
    '/instances/demo/share/s-0001/content/wiki.xml?download=1',
    '/instances/demo/%73hare/s-0005/content/wiki.xml',
    '/instances/demo/share/s-0005/content/../content/wiki.xml',
    '/.env', '/src/dashboard/server.py', '/',
])
def test_anonymous_requests_are_rejected_before_dispatch(private_server, method, path):
    base, root, _ = private_server
    # Forwarded localhost claims cannot grant access through the public tunnel.
    request = Request(base + path, method=method, headers={
        'X-Forwarded-For': '127.0.0.1', 'X-Forwarded-Host': 'localhost', 'Range': 'bytes=0-1'})
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    response = caught.value
    assert response.code == 401
    assert response.headers['WWW-Authenticate'].startswith('Basic ')
    assert 'no-store' in response.headers['Cache-Control']
    assert response.read() == b''
    assert not (root / 'dashboard').exists()


@pytest.mark.parametrize('value', ['Basic invalid', 'Bearer secret', 'Basic b3duZXI6d3Jvbmc='])
def test_wrong_credentials_do_not_read_files(private_server, value):
    base, _, _ = private_server
    with pytest.raises(HTTPError) as caught:
        urlopen(Request(base + '/instances/demo/share/s-0005/content/wiki.xml',
                        headers={'Authorization': value, 'Forwarded': 'for=192.0.2.1'}))
    assert caught.value.code == 401


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
def test_owner_can_download_and_responses_cannot_be_cached(private_server, method):
    base, _, credentials = private_server
    with urlopen(Request(base + '/instances/demo/share/s-0005/content/wiki.xml', method=method,
                         headers={'Authorization': auth.authorization(credentials)})) as response:
        assert response.status == 200
        assert 'no-store' in response.headers['Cache-Control']
        assert response.headers['Referrer-Policy'] == 'no-referrer'
        assert response.read() == (b'' if method == 'HEAD' else b'<wiki>SYNTHETIC_PRIVATE</wiki>')


def test_owner_can_view_dynamic_share_page(private_server):
    from test_report import _run
    from test_share_view import _share
    base, root, credentials = private_server
    exp, _ = _run(root)
    _share(exp.parent.parent)
    with urlopen(Request(base + '/dashboard/demo/versions/share/s-0001/index.html',
                         headers={'Authorization': auth.authorization(credentials)})) as response:
        assert response.status == 200
        assert '人物归属' in response.read().decode()


@pytest.mark.parametrize('failure', ['missing', 'invalid', 'public_permissions', 'symlink'])
def test_credentials_fail_closed(private_server, failure):
    base, root, credentials = private_server
    if failure == 'missing':
        credentials.unlink()
    elif failure == 'invalid':
        credentials.write_text('{}')
    elif failure == 'public_permissions':
        credentials.chmod(0o644)
    else:
        target = root / 'credentials.json'
        credentials.rename(target)
        credentials.symlink_to(target)
    with pytest.raises(HTTPError) as caught:
        urlopen(Request(base + '/instances/demo/share/s-0005/content/wiki.xml',
                        headers={'Forwarded': 'for=192.0.2.1'}))
    assert caught.value.code == 503
    assert b'SYNTHETIC_PRIVATE' not in caught.value.read()


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
@pytest.mark.parametrize('headers', [{}, {'Authorization': 'Basic stale-browser-password'}])
def test_local_direct_access_needs_no_credentials(private_server, method, headers):
    base, _, credentials = private_server
    credentials.unlink()
    with urlopen(Request(base + '/instances/demo/share/s-0005/content/wiki.xml',
                         method=method, headers=headers)) as response:
        assert response.status == 200
        assert response.headers.get('WWW-Authenticate') is None


@pytest.mark.parametrize('headers', [
    {'Host': 'public.example'}, {'Host': 'localhost.evil.example'},
    {'Forwarded': 'for=127.0.0.1'}, {'X-Forwarded-Proto': 'https'},
    {'CF-Connecting-IP': '127.0.0.1'}, {'X-Real-IP': '127.0.0.1'},
    {'Via': 'proxy'}, {'Sec-Fetch-Site': 'cross-site'},
    {'Origin': 'https://public.example'},
])
def test_local_proxy_or_cross_origin_requires_auth(private_server, headers):
    base, _, _ = private_server
    with pytest.raises(HTTPError) as caught:
        urlopen(Request(base + '/instances/demo/share/s-0005/content/wiki.xml', headers=headers))
    assert caught.value.code == 401


def test_local_access_preserves_fixed_answer_guard(private_server):
    base, _, _ = private_server
    with pytest.raises(HTTPError) as caught:
        urlopen(base + '/instances/demo/data/d-0001/fixed_test.jsonl')
    assert caught.value.code == 403


def test_restart_preserves_private_credentials(tmp_path):
    path = tmp_path / 'config' / 'login.json'
    auth.initialize(path)
    original = path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(original)['username'] == 'owner'
    auth.initialize(path)
    assert path.read_bytes() == original


def test_deployment_check_covers_anonymous_and_authenticated_access(private_server):
    from src.dashboard.diagnostics import inspect_access
    from test_report import _run
    from test_share_view import _share
    base, root, credentials = private_server
    exp, _ = _run(root)
    _share(exp.parent.parent)
    result = inspect_access(base, 'demo', ['s-0001'], auth_file=credentials)
    assert len(result['requests']) == 11
    assert {row['status'] for row in result['requests']} == {200, 401}
    assert 'SYNTHETIC_PRIVATE' not in json.dumps(result)
    assert auth.read_credentials(credentials)['password'] not in json.dumps(result)


@pytest.mark.parametrize('url', ['http://example.com', 'https://user:secret@example.com',
                                'https://example.com/other', 'https://example.com?token=secret'])
def test_deployment_check_rejects_unsafe_credential_destinations(url):
    from src.dashboard.diagnostics import inspect_access
    with pytest.raises(ValueError, match='HTTPS origin'):
        inspect_access(url, 'demo', ['s-0001'])
