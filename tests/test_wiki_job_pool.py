"""Object exclusion, progress durability and interruption in the shared scheduler."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event, Lock

import pytest

from src.bootstrap.wiki_job_pool import run_jobs


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
