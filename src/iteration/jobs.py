"""One read-only adapter for the three on-disk job formats.

The scheduler owns launch metadata; the worker owns progress and results.
No observer rewrites a worker checkpoint to make these views agree.
"""
from __future__ import annotations

import time

from ..config import ConfigError, valid_name
from . import task_state, versions
from .storage import locked, read_json

FOLDERS = {'experiment': 'experiments', 'pack': 'judge_eval', 'training': 'judge_training'}
START_GRACE_SECONDS = 15


def directory(job: dict, instance=None):
    if job.get('kind') not in FOLDERS:
        raise ConfigError('任务类型必须是 experiment/pack/training')
    return (instance or versions.PRIVATE) / FOLDERS[job['kind']] / valid_name(job['id'])


def metadata_path(job: dict, instance=None):
    directory(job, instance)  # Validate before building any path.
    return (instance or versions.PRIVATE) / 'scheduling' / f"{job['kind']}-{job['id']}.json"


def snapshot(job: dict, metadata=None, *, instance=None, now=None):
    path = directory(job, instance)
    metadata = metadata if metadata is not None else read_json(metadata_path(job, instance), default={})
    state = read_json(path / 'state.json', default={})
    if job['kind'] == 'experiment':
        progress = state.get('progress', {})
    elif job['kind'] == 'pack':
        progress = read_json(path / 'progress.json', default={})
    else:
        progress = state
    finished = (path / 'pack.json').is_file() if job['kind'] == 'pack' else state.get('status') == 'finished'
    now = time.time() if now is None else now
    starting = 0 <= now - metadata.get('launched_at', 0) < START_GRACE_SECONDS
    # A running producer still occupies a slot even just after publishing its result.
    active = locked(path / '.run.lock') or task_state.active(progress) or starting
    status = metadata.get('status', 'queued')
    reason = metadata.get('reason')
    if finished:
        status = 'finished'
    elif state.get('status') == 'needs_attention':
        status, reason = 'needs_attention', state.get('reason', state.get('phase'))
    elif status == 'needs_attention':
        pass
    elif active:
        status = 'running'
    elif status == 'submitted':
        status = 'waiting_retry'
    return {**metadata, 'job': job, 'status': status, 'reason': reason,
            'active': active, 'finished': finished, 'progress': progress}
