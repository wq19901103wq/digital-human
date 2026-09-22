"""Shared task status: recorded phase is distinct from producer liveness."""
from __future__ import annotations

import json
import math
import os
import subprocess
from subprocess import run as inspect_process
import time
from functools import lru_cache
from datetime import datetime

from .storage import file_lock, write_json


def process_start(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return None
        result = inspect_process(['ps', '-p', str(pid), '-o', 'lstart='],
                                capture_output=True, text=True, timeout=3)
        return result.stdout.strip() or None
    except (TypeError, ValueError, OSError, subprocess.TimeoutExpired):
        return None


@lru_cache(maxsize=4)
def _own_start(pid):
    return process_start(pid)


def active(value):
    if value.get('status') != 'running':
        return False
    try:
        pid = int(value.get('pid') or 0)
        if pid <= 0:
            return False
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    # Legacy records have no process identity; new producers persist it.
    if not value.get('process_start'):
        return True
    start = _own_start(pid) if pid == os.getpid() else process_start(pid)
    return start == value['process_start']


def resolve(value, *, now=None):
    result = dict(value)
    if value.get('status') == 'running':
        result['producer_alive'] = active(value)
        if not result['producer_alive']:
            result.update(status='interrupted', last_phase=value.get('phase'), phase='interrupted',
                          interruption_reason='生产进程已退出或 PID 已被复用；需重新核验断点后恢复')
        updated_at = _timestamp(value.get('updated_at'))
        result['heartbeat_age_seconds'] = (max(0, (time.time() if now is None else now) - updated_at)
                                           if updated_at is not None else None)
    return result


def _timestamp(value):
    """Read legacy wall-clock strings without rewriting producer checkpoints."""
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        try:
            # Legacy strftime records use the producer's local timezone.
            stamp = datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
    return stamp if math.isfinite(stamp) else None


def read(directory):
    path = directory / 'state.json'
    return resolve(json.loads(path.read_text()) if path.exists() else {})


def update(directory, phase, **fields):
    with file_lock(directory / '.state.lock'):
        path = directory / 'state.json'
        value = json.loads(path.read_text()) if path.exists() else {}
        # Counters belong to a phase; preserve the previous phase before resetting.
        if value.get('phase') and value['phase'] != phase:
            counts = {key: value.pop(key) for key in ('successful', 'failed', 'completed', 'total') if key in value}
            if counts:
                value.setdefault('stage_progress', {})[value['phase']] = counts
        # A resumed producer starts a new lease; previous errors remain in logs.
        value.pop('error', None)
        value.pop('reason', None)
        value.pop('interruption_reason', None)
        value.update(status='running', phase=phase, pid=os.getpid(),
                     process_start=_own_start(os.getpid()), updated_at=time.time())
        value.update(fields)
        write_json(path, value)
        return value
