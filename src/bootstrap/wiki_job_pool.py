"""Bounded object scheduling with serialized, resumable batch progress."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import RLock


def object_key(job):
    """Serialize multiple legacy pages belonging to the same known account/chat."""
    if job.get('kind') == 'person' and job.get('account'):
        return 'person', job.get('self_account', ''), job['account']
    if job.get('kind') in ('conversation', 'group') and job.get('chat_id'):
        return 'group', job.get('self_account', ''), job['chat_id']
    return 'job', job['id']


def run_jobs(jobs, execute, *, object_workers, state, save, max_jobs=None):
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
            return execute(job, lambda: attempt(job['id']))
        except Exception as exc:
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
