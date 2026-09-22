"""Small shared HTML fragments and links, independent of experiment execution."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else '—'), quote=True)


def _version_url(instance_dir: Path, folder: str, ref: str) -> str:
    return f'/dashboard/{quote(instance_dir.name, safe="")}/versions/{folder}/{quote(str(ref), safe="")}/index.html'


def _version_link(instance_dir: Path, folder: str, ref: str) -> str:
    if not ref:
        return '<span class="help">未记录</span>'
    return f'<a class="version-link" href="{_version_url(instance_dir, folder, ref)}"><code>{_e(ref)}</code></a>'


def _pre(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f'<pre>{_e(text)}</pre>'


def _details(title: str, body: str, opened: bool = False, *, key: str = '') -> str:
    identity = f' data-detail-key="{_e(key)}"' if key else ''
    return f'<details{identity}{" open" if opened else ""}><summary>{_e(title)}</summary><div class="detail-body">{body}</div></details>'


def _badge(text: str, tone: str = 'neutral') -> str:
    return f'<span class="badge {tone}">{_e(text)}</span>'


def _title(spec: dict) -> str:
    return str(spec.get('single_change') or '未填写任务说明').removeprefix('冒烟：').removeprefix('冒烟:').strip()


def _run_url(run: dict) -> str:
    return f'/instances/{quote(run["instance_dir"].name, safe="")}/experiments/{quote(run["spec"]["id"], safe="")}/index.html'

