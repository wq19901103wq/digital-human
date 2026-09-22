"""Crash-safe case journals. Only an unterminated final write is recoverable."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..config import ConfigError
from .storage import file_lock


def _parse(blob):
    rows, offset = [], 0
    lines = blob.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        if not raw.strip():
            offset += len(raw)
            continue
        try:
            row = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            tail = index == len(lines)-1 and not raw.endswith(b'\n')
            incomplete = (isinstance(exc, UnicodeDecodeError) and exc.end == len(raw)) or (
                isinstance(exc, json.JSONDecodeError) and
                (exc.pos >= len(exc.doc.rstrip())-1 or exc.msg.startswith('Unterminated string')))
            if tail and raw.lstrip().startswith(b'{') and incomplete:
                return rows, offset
            raise ConfigError(f'检查点第 {index+1} 行损坏；禁止跳过中间记录') from exc
        if not isinstance(row, dict) or 'case_id' not in row:
            raise ConfigError(f'检查点第 {index+1} 行缺少 case_id')
        rows.append(row)
        offset += len(raw)
    return rows, None


def records(path: Path):
    """Read complete records; a concurrent unfinished final write is not a result."""
    return _parse(path.read_bytes())[0] if path.exists() else []


def recover(path: Path):
    """Caller holds the experiment run lock; preserve exact bytes before repair."""
    if not path.exists():
        return None
    with file_lock(path.with_name('.' + path.name + '.lock')):
        blob = path.read_bytes()
        _, tail = _parse(blob)
        needs_newline = bool(blob) and not blob.endswith(b'\n')
        if tail is None and not needs_newline:
            return None
        backup = path.with_name(path.name + f'.recovery-{time.time_ns()}.bak')
        with backup.open('xb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        repaired = blob[:tail] if tail is not None else blob + b'\n'
        with path.open('r+b') as stream:
            stream.seek(0)
            stream.write(repaired)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
        return backup
