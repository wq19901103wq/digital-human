"""后台伺服（机制层）：伺服项目根目录的静态文件 + 固定集审计隔离。

用法：python scripts/serve_dashboard.py [--port 8080]
后台页面：http://localhost:8080/dashboard/index.html

SOP §8.5：fixed_test 轮次的 cases.jsonl 含固定集真人答案，线上一律 403；
开发轮（development）不受影响。
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import re
import socketserver
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]  # src/dashboard/server.py → 项目根


def _instances_root():
    return Path(os.environ.get('DH_INSTANCES_ROOT', str(ROOT / 'instances'))).resolve()


class _AuditGuardHandler(http.server.SimpleHTTPRequestHandler):
    """固定集逐题明细访问控制（SOP §8.5）。"""

    def translate_path(self, path):
        import posixpath
        norm = posixpath.normpath(unquote(urlsplit(path).path).split('?', 1)[0].split('#', 1)[0])
        parts = [p for p in norm.split('/') if p]
        if parts and parts[0] == 'instances':
            return str(_instances_root().joinpath(*parts[1:]))
        if parts and parts[0] == 'dashboard':
            return str(_instances_root().parent.joinpath(*parts))
        return str(ROOT.joinpath(*parts))

    def do_GET(self):  # noqa: N802
        from . import report
        url = urlsplit(self.path)
        parts = unquote(url.path).strip('/').split('/')
        if parts == ['dashboard', 'index.html']:
            report.write_instance_index(_instances_root(), _instances_root().parent / 'dashboard')
        instance_page = len(parts) == 3 and parts[0] == 'dashboard' and parts[-1] == 'index.html'
        if instance_page and (_instances_root() / parts[1]).is_dir():
            # 实例页随请求重建（与总索引一致），代码升级后不残留旧渲染。
            report.refresh_dashboard(_instances_root() / parts[1] / 'experiments',
                                     _instances_root().parent / 'dashboard' / parts[1])
        version_page = len(parts) >= 3 and parts[0] == 'dashboard' and parts[2] == 'versions'
        if version_page:
            self._version_page(parts, parse_qs(url.query))
            return
        api = parts[:2] == ['api', 'live']
        trace_api = parts[:2] == ['api', 'trace']
        dashboard = len(parts) == 3 and parts[0] == 'dashboard' and parts[-1] == 'index.html'
        detail = len(parts) == 5 and parts[0] == 'instances' and parts[2] == 'experiments' and parts[-1] == 'index.html'
        if not (api or trace_api or dashboard or detail):
            return super().do_GET()
        if api and len(parts) not in (3, 4):
            self.send_error(404)
            return
        if trace_api and (len(parts) != 5 or not re.fullmatch(r'[a-f0-9]{32}', parts[-1])):
            self.send_error(404)
            return
        instance = parts[2] if api or trace_api else parts[1]
        run = parts[3] if trace_api else ((parts[3] if len(parts) == 4 else '') if api else (parts[3] if detail else ''))
        if not all(re.fullmatch(r'[\w-]+', value) for value in [instance, *([run] if run else [])]):
            self.send_error(403)
            return
        directory = _instances_root() / instance
        target = directory / 'experiments' / run if run else directory
        if not target.resolve().is_relative_to(_instances_root()):
            self.send_error(403)
            return
        if not directory.is_dir() or (run and not (target / 'spec.json').is_file()):
            self.send_error(404)
            return
        try:
            if trace_api:
                from . import trace_view
                if report.experiment.spec_of(target).get('dataset') != 'development':
                    self.send_error(403, 'Fixed-test details are audit-only')
                    return
                query = parse_qs(url.query)
                operation = query.get('operation', [None])[0]
                if operation is not None and not re.fullmatch(r'[0-9]+', operation):
                    self.send_error(400, 'Invalid recorded step')
                    return
                content = json.dumps(trace_view.payload(target, parts[4], query.get('revision', [''])[0],
                    operation=int(operation) if operation is not None else None), ensure_ascii=False)
            elif api:
                query = parse_qs(url.query)
                payload = report.live_payload(directory, run,
                    cases_revision=query.get('cases_revision', [''])[0],
                    config_revision=query.get('config_revision', [''])[0])
                content = json.dumps(payload, ensure_ascii=False)
            else:
                content = report._render_run_html(report._load_run(target)) if run else report.dashboard_html(directory / 'experiments')
        except FileNotFoundError:
            self.send_error(404, 'Recorded details not found')
            return
        except (OSError, ValueError):
            # 逐题文件可能刚写到半行；保留客户端上次画面，下一次轮询再读。
            self.send_error(503, 'Snapshot temporarily unavailable')
            return
        blob = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', ('application/json' if api or trace_api else 'text/html') + '; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _version_page(self, parts: list[str], query: dict) -> None:
        from . import version_view
        if (len(parts) not in (4, 6) or parts[-1] != 'index.html'
                or not all(re.fullmatch(r'[\w-]+', part) for part in parts[1:-1])):
            self.send_error(404)
            return
        instance = _instances_root() / parts[1]
        if not instance.resolve().is_relative_to(_instances_root()):
            self.send_error(403)
            return
        if not instance.is_dir():
            self.send_error(404)
            return
        try:
            content = (version_view.index_html(instance) if len(parts) == 4
                       else version_view.detail_html(instance, parts[3], parts[4], query))
        except PermissionError:
            self.send_error(403, 'Version content is not available here')
            return
        except FileNotFoundError:
            self.send_error(404, 'Version not found')
            return
        except (ValueError, OSError):
            self.send_error(400, 'Invalid version content or query')
            return
        blob = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def send_head(self):  # noqa: N802 - 父类接口名
        import os
        import posixpath
        from urllib.parse import unquote
        # 按静态服务实际解析出的文件路径判定（SOP §5）：解码 → 归一化（. 与 ..）→ 防出根
        # 守卫路径 = 解码后剥离查询/片段再归一化——与静态服务实际解析顺序一致
        # （%3F 与 ?download=1 两种绕过在此统一失效）
        decoded = unquote(self.path)
        guard_path = decoded.split("?", 1)[0].split("#", 1)[0]
        norm = posixpath.normpath(guard_path)
        if norm.startswith("/..") or norm == "..":
            self.send_error(403, "Forbidden")
            return None
        parts = [p for p in norm.split('/') if p]
        mount = (_instances_root() if parts[:1] == ['instances'] else
                 _instances_root().parent / 'dashboard' if parts[:1] == ['dashboard'] else ROOT)
        real = os.path.realpath(self.translate_path(self.path))
        if not Path(real).is_relative_to(mount.resolve()):
            self.send_error(403, 'Forbidden')
            return None
        path = norm
        fname = parts[-1] if parts else ''
        relative = Path(real).relative_to(mount.resolve()).parts
        real_parts = (parts[0], *relative) if parts[:1] in (['instances'], ['dashboard']) else relative
        if len(real_parts) >= 4 and real_parts[0] == 'instances' and real_parts[2] == 'data':
            data_dir = _instances_root().joinpath(*real_parts[1:4])
            if (data_dir / 'purposes.json').is_file() and Path(real).name not in {
                    'manifest.json', 'purposes.json', 'report.json'}:
                self.send_error(403, 'Use the purpose view; full history contains sealed answers')
                return None
        # 实录只经按实验规格审计的接口提供，不能绕过接口直接读取或列目录。
        if 'traces' in real_parts or '.cache' in real_parts:
            self.send_error(403, 'Use the case details view')
            return None
        if len(real_parts) >= 5 and real_parts[0] == 'instances' and real_parts[2] == 'experiments' and real_parts[-1] == 'cases.jsonl':
            try:
                spec = json.loads((_instances_root().joinpath(*real_parts[1:4]) / 'spec.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                self.send_error(403)
                return None
            if spec.get('dataset') != 'development':
                self.send_error(403, 'Fixed-test details are audit-only')
                return None
        # 禁目录列表：解析结果是目录且没有 index.html 一律 403（防文件名枚举泄漏）
        import os as _os
        _real = real
        if _os.path.isdir(_real) and not (_os.path.exists(_os.path.join(_real, "index.html"))):
            self.send_error(403, "Forbidden")
            return None
        is_fixed_seg = any("fixed_test" in seg for seg in parts)  # 子串：judge-fixed_test-* / 自定义 ID 同样拦截
        # 固定集答案全量隔离（SOP §5）：实验明细、原始测试集、Judge 验证包一律 403
        if (
            ("experiments" in parts and fname == "cases.jsonl" and is_fixed_seg)
            or (fname == "fixed_test.jsonl")
            or ("judge_eval" in parts and "validation" in path and fname in {"pack.json", "building.json"})
        ):
            self.send_error(403, "Forbidden: fixed-test case details are audit-only (SOP 8.5)")
            return None
        return super().send_head()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - 父类签名
        pass  # 静态后台不刷访问日志


def serve(port: int = 8080) -> None:
    handler = functools.partial(_AuditGuardHandler, directory=str(ROOT))
    os.chdir(ROOT)

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True  # 必须在构造/bind 之前生效

    with _Server(("", port), handler) as httpd:
        print(f"后台已启动: http://localhost:{port}/dashboard/index.html")
        print("按 Ctrl+C 停止")
        httpd.serve_forever()
