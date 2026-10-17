"""Bounded object scheduling with serialized, resumable batch progress."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import logging
from pathlib import Path
from threading import BoundedSemaphore, RLock
import time
import uuid

from ...iteration.storage import read_json


class RequestBudget:
    """Share request slots across objects without changing client/cache identity."""

    def __init__(self, workers):
        if type(workers) is not int or workers < 1:
            raise ValueError('request workers must be a positive integer')
        self.slots = BoundedSemaphore(workers)

    def wrap(self, client, *, timings=None, job=None):
        return BudgetedClient(client, self.slots, timings=timings, job=job)


class BudgetedClient:
    def __init__(self, client, slots=None, *, timings=None, job=None):
        self.client = client
        self.slots = slots
        self.timings = timings
        self.job = job

    def cache_identity(self):
        return self.client.cache_identity()

    def run(self, prompt, schema=None):
        if self.timings is None:
            with self.slots if self.slots is not None else nullcontext():
                return self.client.run(prompt, schema)
        stage = self.timings.stage(self.job)
        started_at, started = time.time(), time.perf_counter()
        admitted, error_type = None, None
        try:
            with self.slots if self.slots is not None else nullcontext():
                admitted = time.perf_counter()
                return self.client.run(prompt, schema)
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            finished = time.perf_counter()
            self.timings.record(dict(schema='wiki_request_timing_v1', run_id=self.timings.run_id,
                job_id=self.job['id'], stage=stage, started_at=started_at, finished_at=time.time(),
                queue_seconds=(admitted if admitted is not None else finished) - started,
                client_seconds=finished - admitted if admitted is not None else None,
                status='error' if error_type else 'success', error_type=error_type,
                timeout_seconds=self.timings.timeout_seconds))


class RequestTimings:
    """Observe client calls, not remote inference or successful Wiki-cache reads."""

    def __init__(self, output, *, timeout_seconds=None):
        self.path = Path(output) / 'request-profile.jsonl'
        self.run_id = uuid.uuid4().hex
        self.timeout_seconds = timeout_seconds
        self.guard = RLock()

    def stage(self, job):
        try:
            return read_json(Path(job['output']) / 'progress.json', default={}).get('stage', 'unknown')
        except (OSError, ValueError):
            return 'unknown'

    def record(self, value):
        try:
            with self.guard:
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        except OSError as exc:
            logging.warning('Wiki request timing could not be saved (%s)', type(exc).__name__)


def _timing_summary(records):
    result = dict(requests=len(records), success=sum(item['status'] == 'success' for item in records),
                  errors=dict(Counter(item['error_type'] for item in records if item['error_type'])))
    result['failed'] = result['requests'] - result['success']
    for field in ('queue_seconds', 'client_seconds'):
        values = [item[field] for item in records if item[field] is not None]
        result[field] = dict(total=sum(values), mean=sum(values) / len(values) if values else 0,
                             maximum=max(values, default=0))
    result['wall_span_seconds'] = (max(item['finished_at'] for item in records) -
                                  min(item['started_at'] for item in records)) if records else 0
    return result


def _group_timings(records, field):
    grouped = defaultdict(list)
    for item in records:
        grouped[item[field]].append(item)
    return dict(sorted(grouped.items()))


def request_profile(output, *, run_id=None):
    """Summarize persisted request timings without model calls or raw content."""
    path = Path(output) / 'request-profile.jsonl'
    records, partial_lines = [], 0
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    partial_lines += 1
                    continue
                value = json.loads(line)
                if value.get('schema') != 'wiki_request_timing_v1':
                    raise ValueError('unknown Wiki request timing schema')
                if run_id is None or value['run_id'] == run_id:
                    records.append(value)
    return dict(schema='wiki_request_profile_v1', run_id=run_id,
        run_ids=sorted({item['run_id'] for item in records}), partial_lines=partial_lines,
        measurement='client calls including global cache; excludes Wiki-cache reuse and local-only work',
        summary=_timing_summary(records),
        by_stage={stage: _timing_summary(items)
                  for stage, items in _group_timings(records, 'stage').items()},
        by_job={job_id: dict(summary=_timing_summary(items),
                            by_stage={stage: _timing_summary(stage_items)
                                      for stage, stage_items in _group_timings(items, 'stage').items()})
                for job_id, items in _group_timings(records, 'job_id').items()})


def object_key(job):
    """Serialize multiple legacy pages belonging to the same known account/chat."""
    if job.get('kind') == 'person' and job.get('account'):
        return 'person', job.get('self_account', ''), job['account']
    if job.get('kind') in ('conversation', 'group') and job.get('chat_id'):
        return 'group', job.get('self_account', ''), job['chat_id']
    return 'job', job['id']


def run_jobs(jobs, execute, *, object_workers, state, save, max_jobs=None, check_available=None):
    """Execute pending jobs; the caller owns the batch lock and input checks.

    ``jobs`` may be a coordinator-side iterator that validates/reuses completed
    work. ``execute(job, attempt)`` returns fields to merge into the job state;
    ``attempt()`` records an attempt safely from the object's worker thread.
    Only a bounded set is submitted, so saved active jobs never include a
    backlog waiting behind other objects. Ordinary job errors are resumable;
    iterator errors stop admission and preserve results from admitted work.
    """
    if object_workers < 1 or (max_jobs is not None and max_jobs < 1):
        raise ValueError('object_workers and optional max_jobs must be positive')
    guard = RLock()
    active = []

    def persist():
        state['active_jobs'] = list(active)
        state['active_job'] = active[0] if active else None
        save()

    with guard:
        for item in state['jobs'].values():
            if item.get('stage') == 'running':
                item['stage'] = 'interrupted'
        persist()

    def attempt(job_id):
        with guard:
            state['jobs'][job_id]['attempts'] += 1
            persist()

    def invoke(job):
        try:
            if check_available is not None:
                check_available()
            return execute(job, lambda: attempt(job['id']))
        except Exception as exc:
            if check_available is not None:
                check_available()
            return dict(stage='incomplete', error=f'{type(exc).__name__}: {exc}'[-1600:])

    pending, waiting, active_keys = {}, [], set()
    started, exhausted = 0, False

    def finish(future):
        job_id, key = pending.pop(future)
        try:
            result = future.result()
        except BaseException:
            with guard:
                state['jobs'][job_id]['stage'] = 'interrupted'
                active.remove(job_id)
                active_keys.remove(key)
                persist()
            raise
        with guard:
            state['jobs'][job_id].update(result)
            active.remove(job_id)
            active_keys.remove(key)
            persist()

    iterator = iter(jobs)

    def next_job():
        nonlocal exhausted
        for index, job in enumerate(waiting):
            if object_key(job) not in active_keys:
                return waiting.pop(index)
        while not exhausted:
            try:
                job = next(iterator)
            except StopIteration:
                exhausted = True
                break
            if object_key(job) not in active_keys:
                return job
            waiting.append(job)
        return None

    with ThreadPoolExecutor(max_workers=object_workers, thread_name_prefix='wiki-object') as pool:
        try:
            while True:
                while len(pending) < object_workers:
                    if max_jobs is not None and started >= max_jobs:
                        break
                    if check_available is not None:
                        check_available()
                    job = next_job()
                    if job is None:
                        break
                    key = object_key(job)
                    with guard:
                        prior = state['jobs'].get(job['id'], {})
                        state['jobs'][job['id']] = dict(stage='running', account=job.get('account', ''),
                            attempts=prior.get('attempts', 0))
                        active.append(job['id'])
                        active_keys.add(key)
                        persist()
                    pending[pool.submit(invoke, job)] = job['id'], key
                    started += 1
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    finish(future)
        finally:
            # A failed coordinator input check must not discard successful
            # concurrent work or leave finished objects marked as running.
            failure = None
            for future in list(pending):
                try:
                    finish(future)
                except BaseException as exc:
                    failure = failure or exc
            if failure is not None:
                raise failure
    return started
