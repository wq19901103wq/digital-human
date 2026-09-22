"""Process-safe instance quotas and explicit job stops, including cache hits."""
from __future__ import annotations

import math
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from ..config import ConfigError
from . import task_state, versions
from .storage import file_lock, write_json, read_json

_job_directory = ContextVar('controlled_job_directory', default=None)


class StopRequested(ConfigError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def read(path):
    return read_json(path, default={})


def job_directory():
    value = _job_directory.get() or os.environ.get('DH_JOB_DIR')
    return Path(value).resolve() if value else None


def policy():
    value = read(versions.PRIVATE / 'resource_policy.json')
    for key in ('max_requests', 'max_active_requests'):
        if key in value and (type(value[key]) is not int or value[key] < (1 if key == 'max_active_requests' else 0)):
            raise ConfigError(f'{key} must be a non-negative integer (concurrency at least one)')
    for key in ('max_cost_units', 'cost_units_per_request', 'deadline_at'):
        if key in value and (not isinstance(value[key], (int, float)) or
                             not math.isfinite(value[key]) or value[key] < 0):
            raise ConfigError(f'invalid {key}')
    if 'max_cost_units' in value and value.get('cost_units_per_request', 0) <= 0:
        raise ConfigError('cost quota requires a positive per-request reservation')
    return value


def check(directory=None):
    directory = directory or job_directory()
    if directory:
        stopped = read(Path(directory) / 'control.json')
        if stopped.get('cancelled'):
            raise StopRequested('cancelled')
    limit = policy()
    if limit.get('deadline_at') is not None and time.time() >= limit['deadline_at']:
        raise StopRequested('deadline_reached')


def cancel(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigError('job does not exist')
    with file_lock(directory / '.control.lock'):
        value = read(directory / 'control.json')
        value.update(cancelled=True, cancelled_at=time.time())
        write_json(directory / 'control.json', value)


def resume(directory):
    with file_lock(Path(directory) / '.control.lock'):
        value = read(Path(directory) / 'control.json')
        value.update(cancelled=False, resumed_at=time.time())
        write_json(Path(directory) / 'control.json', value)


@contextmanager
def request():
    """Charge every real attempt, never refund failures/unknown costs.

    Cost units are the instance's conservative reservation, not provider billing.
    Successful cached responses consume no new request allowance.
    """
    ticket = uuid.uuid4().hex
    path = versions.PRIVATE / 'resource_usage.json'
    lock = versions.PRIVATE / '.resources.lock'
    while True:
        check()
        with file_lock(lock):
            limits, value = policy(), read(path)
            active = {k: v for k, v in value.get('active', {}).items() if task_state.active(v)}
            count, spent = value.get('requests', 0), value.get('reserved_cost_units', 0)
            cost = limits.get('cost_units_per_request', 0)
            if 'max_requests' in limits and count >= limits['max_requests']:
                raise StopRequested('request_budget_exhausted')
            if 'max_cost_units' in limits and spent + cost > limits['max_cost_units']:
                raise StopRequested('cost_budget_exhausted')
            if len(active) < limits.get('max_active_requests', 16):
                active[ticket] = {'status': 'running', 'pid': os.getpid(),
                                  'process_start': task_state._own_start(os.getpid()),
                                  'job': str(job_directory() or ''), 'started_at': time.time()}
                value.update(active=active, requests=count + 1, reserved_cost_units=spent + cost)
                write_json(path, value)
                break
        time.sleep(.1)
    try:
        check()
        yield
        check()
    finally:
        with file_lock(lock):
            value = read(path)
            value.get('active', {}).pop(ticket, None)
            write_json(path, value)


@contextmanager
def job(directory):
    """Worker threads inherit the submitting context; child processes get an explicit environment."""
    token = _job_directory.set(Path(directory).resolve())
    try:
        check(directory)
        yield
    except StopRequested as exc:
        task_state.update(Path(directory), exc.reason, status='needs_attention', reason=exc.reason)
        raise
    finally:
        _job_directory.reset(token)
