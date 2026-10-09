"""Dashboard overload must be bounded and recover after slow clients leave."""
import http.server
import socket
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from src.dashboard.server import DashboardServer
from test_live_dashboard import live_server
from test_report import _run


def test_overload_rejected_without_starting_more_handlers_and_recovers():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            if self.path == '/slow':
                entered.set()
                assert release.wait(5)
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()

        def log_message(self, *_):
            pass

    class Server(DashboardServer):
        max_requests = 1

        def process_request_thread(self, *args):
            try:
                super().process_request_thread(*args)
            finally:
                finished.set()

    with Server(('127.0.0.1', 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        client = socket.create_connection(server.server_address, timeout=5)
        base = f'http://127.0.0.1:{server.server_address[1]}'
        try:
            client.sendall(b'GET /slow HTTP/1.0\r\n\r\n')
            assert entered.wait(5)
            for _ in range(6):
                with pytest.raises(HTTPError) as caught:
                    urlopen(base + '/overflow', timeout=5)
                assert caught.value.code == 503
                assert caught.value.headers['Retry-After'] == '3'
                caught.value.close()
            assert calls == ['/slow']
            release.set()
            assert finished.wait(5)
            with urlopen(base + '/recovered', timeout=5) as response:
                assert response.status == 200
        finally:
            release.set()
            client.close()
            server.shutdown()
            worker.join(5)


def test_idle_connection_times_out_and_releases_capacity():
    entered, finished = threading.Event(), threading.Event()

    class Server(DashboardServer):
        max_requests = 1
        connection_timeout = 0.1

        def process_request_thread(self, *args):
            entered.set()
            try:
                super().process_request_thread(*args)
            finally:
                finished.set()

    with Server(('127.0.0.1', 0), http.server.BaseHTTPRequestHandler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            with socket.create_connection(server.server_address, timeout=5) as client:
                assert entered.wait(5)
                assert finished.wait(5)
                assert client.recv(1) == b''
            assert server._slots.acquire(blocking=False)
            server._slots.release()
        finally:
            server.shutdown()
            worker.join(5)


def test_slow_live_poll_does_not_queue_duplicate_renders_or_block_files(tmp_path, live_server, monkeypatch):
    from src.dashboard import report
    _run(tmp_path)
    (tmp_path / 'ready.txt').write_text('ready')
    entered, release = threading.Event(), threading.Event()
    responses = []

    def render(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return {'regions': {}}

    monkeypatch.setattr(report, 'live_payload', render)
    url = live_server + '/api/live/demo'

    def first():
        with urlopen(url, timeout=5) as response:
            responses.append(response.status)

    worker = threading.Thread(target=first)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(HTTPError) as caught:
            urlopen(url, timeout=5)
        assert caught.value.code == 503
        caught.value.close()
        assert urlopen(live_server + '/ready.txt', timeout=5).read() == b'ready'
    finally:
        release.set()
        worker.join(5)
    assert responses == [200]
    with urlopen(url, timeout=5) as response:
        assert response.status == 200
