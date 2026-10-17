"""Object exclusion, progress durability and interruption in the shared scheduler."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier, Event, Lock
from subprocess import TimeoutExpired

import pytest

from src.share.wiki import wiki_job_pool
from src.share.wiki.wiki_job_pool import RequestBudget, RequestTimings, BudgetedClient, request_profile, run_jobs
from src.judge.codex_transport import CodexServiceBlocked
from src.iteration.storage import write_json


@pytest.mark.parametrize('workers', [0, -1, 1.5, '2', None, True])
def test_request_budget_requires_positive_integer(workers):
    with pytest.raises(ValueError, match='positive integer'):
        RequestBudget(workers)


def test_request_budget_is_shared_across_clients_without_changing_requests_or_identity():
    budget = RequestBudget(2)
    ready, release, filled, guard = Barrier(9), Event(), Event(), Lock()
    active, peak, calls = 0, 0, []
    identity, schema = dict(model='unchanged', provider='test'), dict(type='object')

    class Client:
        def cache_identity(self):
            return identity

        def run(self, prompt, schema=None):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
                calls.append((prompt, schema))
                if active == 2:
                    filled.set()
            try:
                assert release.wait(10)
                return prompt
            finally:
                with guard:
                    active -= 1

    clients = [budget.wrap(Client()), budget.wrap(Client())]
    assert all(client.cache_identity() is identity for client in clients)

    def request(index):
        ready.wait(timeout=10)
        return clients[index % 2].run(f'prompt-{index}', schema)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(request, index) for index in range(8)]
        try:
            ready.wait(timeout=10)
            assert filled.wait(10)
            assert active == 2
        finally:
            release.set()
        assert [future.result(timeout=10) for future in futures] == [f'prompt-{index}' for index in range(8)]
    assert peak == 2 and active == 0
    assert {prompt for prompt, _ in calls} == {f'prompt-{index}' for index in range(8)}
    assert all(value is schema for _, value in calls)


def test_one_object_can_borrow_all_available_request_slots():
    budget, overlap = RequestBudget(3), Barrier(3)

    class Client:
        def run(self, prompt, schema=None):
            overlap.wait(timeout=10)
            return prompt

    client = budget.wrap(Client())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(client.run, f'prompt-{index}') for index in range(3)]
        assert [future.result(timeout=10) for future in futures] == ['prompt-0', 'prompt-1', 'prompt-2']


def test_failed_request_releases_shared_slot():
    budget = RequestBudget(1)

    class Client:
        def run(self, prompt, schema=None):
            raise RuntimeError('request failed')

    with pytest.raises(RuntimeError, match='request failed'):
        budget.wrap(Client()).run('prompt')
    assert budget.slots.acquire(blocking=False)
    budget.slots.release()


@pytest.mark.parametrize('failure', [False, True])
def test_profile_separates_slot_wait_and_client_time_without_raw_content(tmp_path, monkeypatch, failure):
    clock, released = [0], []
    monkeypatch.setattr(wiki_job_pool.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(wiki_job_pool.time, 'time', lambda: 1000 + clock[0])
    job = dict(id='object-1', output=str(tmp_path / 'object'))
    write_json(tmp_path / 'object/progress.json', dict(stage='structuring'))
    timings = RequestTimings(tmp_path, timeout_seconds=900)
    identity, schema = dict(model='unchanged'), dict(type='object')

    class Slots:
        def __enter__(self):
            clock[0] += 2

        def __exit__(self, *args):
            released.append(True)

    class Client:
        def cache_identity(self):
            return identity

        def run(self, prompt, value=None):
            assert prompt == 'PRIVATE_PROMPT' and value is schema
            clock[0] += 7
            if failure:
                raise TimeoutExpired('PRIVATE_EXCEPTION', 900)
            return 'PRIVATE_RESPONSE'

    client = BudgetedClient(Client(), Slots(), timings=timings, job=job)
    assert client.cache_identity() is identity
    if failure:
        with pytest.raises(TimeoutExpired):
            client.run('PRIVATE_PROMPT', schema)
    else:
        assert client.run('PRIVATE_PROMPT', schema) == 'PRIVATE_RESPONSE'
    assert released == [True]
    records = [json.loads(line) for line in timings.path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record['job_id'] == job['id'] and record['stage'] == 'structuring'
    assert record['queue_seconds'] == 2 and record['client_seconds'] == 7
    assert record['timeout_seconds'] == 900
    assert record['error_type'] == ('TimeoutExpired' if failure else None)
    assert 'PRIVATE_' not in timings.path.read_text()
    summary = request_profile(tmp_path, run_id=timings.run_id)
    assert summary['run_ids'] == [timings.run_id]
    assert summary['summary']['requests'] == 1
    assert summary['summary']['failed'] == int(failure)
    assert summary['summary']['queue_seconds']['total'] == 2
    assert summary['by_stage']['structuring']['client_seconds']['total'] == 7
    assert summary['by_job']['object-1']['summary'] == summary['summary']
    assert summary['by_job']['object-1']['by_stage'] == summary['by_stage']
    assert request_profile(tmp_path, run_id='other')['summary']['requests'] == 0


def test_profile_groups_object_stages_without_combining_parallel_call_time(tmp_path):
    timings = RequestTimings(tmp_path)
    for job_id, stage, started_at, queue_seconds, client_seconds, error_type in (
            ('object-z', 'extracting', 100, 2, 8, None),
            ('object-a', 'drafting', 101, 1, 9, 'TimeoutExpired'),
            ('object-z', 'structuring', 112, 1, 5, None)):
        timings.record(dict(schema='wiki_request_timing_v1', run_id=timings.run_id,
            job_id=job_id, stage=stage, started_at=started_at,
            finished_at=started_at + queue_seconds + client_seconds,
            queue_seconds=queue_seconds, client_seconds=client_seconds,
            status='error' if error_type else 'success', error_type=error_type,
            timeout_seconds=900))
    timings.record(dict(schema='wiki_request_timing_v1', run_id='other',
        job_id='object-a', stage='extracting', started_at=200, finished_at=205,
        queue_seconds=0, client_seconds=5, status='success', error_type=None,
        timeout_seconds=900))

    profile = request_profile(tmp_path, run_id=timings.run_id)
    assert list(profile['by_job']) == ['object-a', 'object-z']
    assert profile['summary']['requests'] == 3
    assert profile['summary']['wall_span_seconds'] == 18
    assert profile['summary']['client_seconds']['total'] == 22
    first = profile['by_job']['object-a']
    assert first['summary']['requests'] == first['summary']['failed'] == 1
    assert first['summary']['errors'] == {'TimeoutExpired': 1}
    assert set(first['by_stage']) == {'drafting'}
    second = profile['by_job']['object-z']
    assert second['summary']['requests'] == second['summary']['success'] == 2
    assert second['summary']['wall_span_seconds'] == 18
    assert second['summary']['queue_seconds']['total'] == 3
    assert second['summary']['client_seconds']['total'] == 13
    assert second['by_stage']['structuring']['client_seconds']['total'] == 5
    assert request_profile(tmp_path)['summary']['requests'] == 4
    assert request_profile(tmp_path, run_id='missing')['by_job'] == {}


def test_profiling_releases_budget_before_io_and_io_failure_preserves_client_error(tmp_path, monkeypatch):
    budget = RequestBudget(1)
    timings = RequestTimings(tmp_path)
    job = dict(id='object-1', output=str(tmp_path))

    class Client:
        def run(self, *args):
            raise TimeoutExpired('private', 900)

    original_open = Path.open

    def unavailable(path, *args, **kwargs):
        if path == timings.path:
            assert budget.slots.acquire(blocking=False)
            budget.slots.release()
            raise OSError('private filesystem error')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', unavailable)
    with pytest.raises(TimeoutExpired):
        budget.wrap(Client(), timings=timings, job=job).run('private')
    assert budget.slots.acquire(blocking=False)
    budget.slots.release()


def test_profile_handles_missing_and_partial_records_and_rejects_unknown_schema(tmp_path):
    assert request_profile(tmp_path)['summary']['requests'] == 0
    path = tmp_path / 'request-profile.jsonl'
    path.write_text('{"schema":')
    assert request_profile(tmp_path)['partial_lines'] == 1
    path.write_text('{"schema": "unknown"}\n')
    with pytest.raises(ValueError, match='unknown Wiki request timing schema'):
        request_profile(tmp_path)


def test_service_failure_stops_admission_drains_success_and_keeps_pending_untouched():
    jobs = [dict(id=name) for name in ('failure', 'success', 'pending')]
    state = dict(jobs={'pending': dict(stage='pending', attempts=0)})
    overlap, blocked = Barrier(2), Event()
    details = dict(exit_code=1, terminal_signals=['rate_limit'])

    def available():
        if blocked.is_set():
            raise CodexServiceBlocked(details)

    def execute(job, attempt):
        attempt()
        overlap.wait(timeout=10)
        if job['id'] == 'failure':
            blocked.set()
            raise RuntimeError('transport failed')
        assert blocked.wait(10)
        return dict(stage='complete', records=4)

    with pytest.raises(CodexServiceBlocked):
        run_jobs(jobs, execute, object_workers=2, state=state, save=lambda: None,
                 check_available=available)
    assert state['jobs']['failure']['stage'] == 'interrupted'
    assert state['jobs']['failure']['attempts'] == 1
    assert state['jobs']['success']['stage'] == 'complete'
    assert state['jobs']['success']['records'] == 4
    assert state['jobs']['pending'] == dict(stage='pending', attempts=0)
    assert state['active_jobs'] == [] and state['active_job'] is None


@pytest.mark.parametrize('kind,identity', [('person', 'account'), ('conversation', 'chat_id')])
def test_pages_of_one_object_wait_while_independent_objects_overlap(kind, identity):
    jobs = [dict(id='first', kind=kind, self_account='self', **{identity: 'same'}),
            dict(id='second', kind=kind, self_account='self', **{identity: 'same'}),
            dict(id='third', kind=kind, self_account='self', **{identity: 'different'})]
    state, snapshots = dict(jobs={}), []
    first_pair = Barrier(2)
    running, completed, guard = set(), [], Lock()

    def execute(job, attempt):
        key = job[identity]
        with guard:
            assert key not in running, 'two pages of one object overlapped'
            running.add(key)
        attempt()
        if job['id'] in ('first', 'third'):
            first_pair.wait(timeout=10)
        else:
            assert 'first' in completed
        with guard:
            running.remove(key)
            completed.append(job['id'])
        return dict(stage='complete')

    assert run_jobs(jobs, execute, object_workers=2, state=state,
                    save=lambda: snapshots.append(deepcopy(state))) == 3
    assert all(item['stage'] == 'complete' for item in state['jobs'].values())
    assert any(set(s['active_jobs']) == {'first', 'third'} for s in snapshots)
    assert not any({'first', 'second'} <= set(s['active_jobs']) for s in snapshots)
    assert state['active_jobs'] == []


def test_limit_counts_started_jobs_without_claiming_deferred_same_object_pages():
    jobs = [dict(id='first', kind='person', account='same'),
            dict(id='second', kind='person', account='same'),
            dict(id='third', kind='person', account='other')]
    state = dict(jobs={})
    overlap = Barrier(2)

    def execute(job, attempt):
        attempt()
        overlap.wait(timeout=10)
        return dict(stage='complete')

    assert run_jobs(jobs, execute, object_workers=3, max_jobs=2, state=state, save=lambda: None) == 2
    assert set(state['jobs']) == {'first', 'third'}
    assert state['active_jobs'] == []


def test_job_failure_keeps_other_success_and_interrupted_attempts_can_resume():
    jobs = [dict(id='bad'), dict(id='good')]
    state = dict(jobs={'bad': dict(stage='running', attempts=2)})

    def execute(job, attempt):
        attempt()
        if job['id'] == 'bad':
            raise ValueError('failed input')
        return dict(stage='complete', records=3)

    run_jobs(jobs, execute, object_workers=2, state=state, save=lambda: None)
    assert state['jobs']['bad'] == dict(stage='incomplete', account='', attempts=3,
                                      error='ValueError: failed input')
    assert state['jobs']['good']['stage'] == 'complete'
    assert state['jobs']['good']['records'] == 3
    assert state['active_jobs'] == [] and state['active_job'] is None


def test_coordinator_input_failure_drains_admitted_work_and_preserves_results():
    state = dict(jobs={})
    admitted = Event()

    def jobs():
        yield dict(id='first')
        assert admitted.wait(10)
        raise ValueError('frozen source changed')

    def execute(job, attempt):
        attempt()
        admitted.set()
        return dict(stage='complete', sha256='retained')

    with pytest.raises(ValueError, match='frozen source changed'):
        run_jobs(jobs(), execute, object_workers=2, state=state, save=lambda: None)
    assert state['jobs']['first']['sha256'] == 'retained'
    assert state['active_jobs'] == [] and state['active_job'] is None


def test_coordinator_interruption_drains_all_admitted_work_before_releasing_batch():
    state = dict(jobs={})
    admitted = Barrier(3)

    def jobs():
        yield dict(id='first')
        yield dict(id='second')
        admitted.wait(timeout=10)
        raise KeyboardInterrupt()

    def execute(job, attempt):
        attempt()
        admitted.wait(timeout=10)
        return dict(stage='complete', sha256=job['id'])

    with pytest.raises(KeyboardInterrupt):
        run_jobs(jobs(), execute, object_workers=3, state=state, save=lambda: None)
    assert {job_id: item['sha256'] for job_id, item in state['jobs'].items()} == {
        'first': 'first', 'second': 'second'}
    assert all(item['stage'] == 'complete' for item in state['jobs'].values())
    assert state['active_jobs'] == [] and state['active_job'] is None


def test_worker_interruption_is_visible_and_other_admitted_results_survive():
    state = dict(jobs={})
    overlap = Barrier(2)

    def execute(job, attempt):
        attempt()
        overlap.wait(timeout=10)
        if job['id'] == 'interrupted':
            raise KeyboardInterrupt()
        return dict(stage='complete')

    with pytest.raises(KeyboardInterrupt):
        run_jobs([dict(id='interrupted'), dict(id='done')], execute,
                 object_workers=2, state=state, save=lambda: None)
    assert state['jobs']['interrupted']['stage'] == 'interrupted'
    assert state['jobs']['done']['stage'] == 'complete'
    assert state['active_jobs'] == []


def test_concurrent_attempts_serialize_progress_writes():
    state = dict(jobs={})
    overlap, writing, second_attempt, release = Barrier(2), Event(), Event(), Event()
    writer = Lock()
    snapshots = []

    def save():
        assert writer.acquire(blocking=False), 'progress writes overlapped'
        try:
            if state['jobs'].get('first', {}).get('attempts') == 1 and not writing.is_set():
                writing.set()
                assert release.wait(10)
            snapshots.append(deepcopy(state))
        finally:
            writer.release()

    def execute(job, attempt):
        overlap.wait(timeout=10)
        if job['id'] == 'second':
            assert writing.wait(10)
            second_attempt.set()
        attempt()
        return dict(stage='complete')

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_jobs, [dict(id='first'), dict(id='second')], execute,
                             object_workers=2, state=state, save=save)
        try:
            assert second_attempt.wait(10)
        finally:
            release.set()
        assert future.result(timeout=10) == 2
    assert all(item['attempts'] == 1 for item in state['jobs'].values())
    assert snapshots[-1]['active_jobs'] == []
