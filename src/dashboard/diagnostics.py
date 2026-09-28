"""Reusable read-only profiling of the real dashboard and live render paths."""
from __future__ import annotations

import cProfile
import json
from pathlib import Path
import pstats
import time

from . import report


def inspect(instance: Path) -> dict:
    results = {}
    for name, render in (
        ('home', lambda: report.dashboard_html(instance / 'experiments')),
        ('live', lambda: report.live_payload(instance)),
    ):
        profiler = cProfile.Profile()
        started = time.perf_counter()
        content = profiler.runcall(render)
        seconds = time.perf_counter() - started
        body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        stats = pstats.Stats(profiler).stats
        slowest = sorted(stats.items(), key=lambda item: item[1][3], reverse=True)[:15]
        results[name] = {
            'seconds': round(seconds, 4), 'bytes': len(body.encode('utf-8')),
            'profile': [dict(function=f'{Path(key[0]).name}:{key[1]}:{key[2]}',
                             calls=value[1], self_seconds=round(value[2], 4),
                             cumulative_seconds=round(value[3], 4))
                        for key, value in slowest],
        }
    return results


def inspect_access(base_url: str, instance: str, shares: list[str], *, auth_file: Path | None = None,
                   download: str = 'wiki.xml') -> dict:
    """Verify the deployed boundary without printing credentials or private response bodies."""
    import re
    from urllib.error import HTTPError
    from urllib.parse import urlsplit
    from urllib.request import HTTPRedirectHandler, Request, build_opener
    from . import auth

    url = urlsplit(base_url)
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
            or url.path not in ('', '/') or url.query or url.fragment
            or (url.scheme == 'http' and url.hostname not in ('localhost', '127.0.0.1', '::1'))):
        raise ValueError('Use an HTTPS origin or a loopback HTTP origin')
    if not shares or not all(re.fullmatch(r'[\w-]+', name) for name in [instance, *shares]):
        raise ValueError('Specify an instance and share versions to verify')
    if download not in ('wiki.xml', 'knowledge.json'):
        raise ValueError('Specify a supported Share download')

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None  # Never forward owner credentials to another location.

    opener = build_opener(NoRedirect())
    authorization = auth.authorization(auth_file if auth_file is not None else auth.credentials_path())
    results = []

    def request(path, method, headers, expected):
        try:
            response = opener.open(Request(base_url.rstrip('/') + path, method=method,
                headers={'User-Agent': 'DigitalHuman-Dashboard-Check/1.0', **headers}), timeout=20)
        except HTTPError as exc:
            response = exc
        with response:
            status = response.code
            cache = ','.join(response.headers.get_all('Cache-Control', []))
            # Consume pages to let the server finish normally, without returning their content.
            while response.read(65536):
                pass
            if status != expected or 'no-store' not in cache:
                raise ValueError(f'{method} {path}: expected {expected} with no-store, received {status}')
        results.append(dict(path=path, method=method, authenticated=headers.get('Authorization') == authorization,
                            status=status))

    paths = [f'/dashboard/{instance}/index.html', f'/api/live/{instance}']
    for share in shares:
        paths.extend([f'/dashboard/{instance}/versions/share/{share}/index.html',
                      f'/instances/{instance}/share/{share}/content/{download}'])
    for path in paths:
        for method in ('GET', 'HEAD'):
            request(path, method, {}, 401)
    for share in shares:
        page = f'/dashboard/{instance}/versions/share/{share}/index.html'
        file_path = f'/instances/{instance}/share/{share}/content/{download}'
        request(file_path, 'GET', {'Authorization': 'Basic invalid', 'Range': 'bytes=0-0'}, 401)
        request(page, 'GET', {'Authorization': authorization}, 200)
        request(file_path, 'HEAD', {'Authorization': authorization}, 200)
    return {'origin': base_url, 'requests': results, 'private_content_returned': False}
