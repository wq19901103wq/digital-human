from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from threading import Barrier, Event, Lock, Thread
import sqlite3

import pytest

from src import llm, tracing
from src.generator import (disk_history, few_shot, history, history_sources, learned_sources,
                           ranker_report, ranker_samples, shared_history)
from src.iteration import control, datasets, learning_guard, pack_transport, transport_profile


@pytest.fixture
def disk_reader(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text('{}\n')
    reader = disk_history.BoundedReader(source, sqlite3.connect(':memory:', check_same_thread=False))
    try:
        yield reader
    finally:
        reader.close()


def test_admission_times_entry_not_model_and_preserves_context(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(transport_profile.time, 'monotonic', lambda: clock[0])
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    exits = []

    @contextmanager
    def admitted():
        clock[0] += 3
        try:
            yield 'original'
        finally:
            exits.append(True)

    with profile.admission(admitted()) as result:
        assert result == 'original'
        clock[0] += 10
    assert exits == [True]
    assert profile.metrics['request_admission']['total_seconds'] == 3
    with pytest.raises(ValueError, match='original error'):
        with profile.admission(admitted()):
            raise ValueError('original error')
    assert exits == [True, True]
    assert profile.metrics['request_admission']['failed'] == 0


def test_admission_failure_and_suppression_are_preserved(tmp_path):
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})

    @contextmanager
    def failed():
        raise RuntimeError('entry failed')
        yield

    with pytest.raises(RuntimeError, match='entry failed'):
        with profile.admission(failed()):
            pytest.fail('entry failure was suppressed')
    assert profile.metrics['request_admission']['failed'] == 1

    @contextmanager
    def suppressed():
        try:
            yield
        except ValueError:
            pass

    with profile.admission(suppressed()):
        raise ValueError('suppressed')


def test_instrument_restores_functions_and_separates_nested_times(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(transport_profile.time, 'monotonic', lambda: clock[0])

    @contextmanager
    def request():
        clock[0] += 3
        yield

    def send(client, body):
        with llm.request(), tracing.step('llm', {'body': body}):
            clock[0] += 10
        clock[0] += 2
        return body

    monkeypatch.setattr(control, 'request', request)
    monkeypatch.setattr(llm, 'request', request)
    monkeypatch.setattr(llm.ChatClient, '_send', send)
    targets = [(control, 'request'), (llm, 'request'), (llm.ChatClient, '_send'),
        (tracing, 'step'), (few_shot.PersonaFewShotRetriever, 'retrieve'),
        (few_shot.PersonaFewShotRetriever, '_load'), (shared_history.PinnedRetriever, 'retrieve'),
        (history_sources.HistorySources, '__init__')]
    originals = [getattr(owner, name) for owner, name in targets]
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    with pytest.raises(ValueError):
        with profile.instrument():
            body = {'unchanged': True}
            assert llm.ChatClient._send(object(), body) is body
            with tracing.step('retrieval', {}):
                pass
            raise ValueError('original')
    assert [getattr(owner, name) for owner, name in targets] == originals
    assert profile.metrics['request_admission']['total_seconds'] == 3
    assert profile.metrics['model_llm']['total_seconds'] == 10
    assert profile.metrics['llm_with_retries']['total_seconds'] == 15
    assert 'model_retrieval' not in profile.metrics


@pytest.mark.parametrize('owner,method,metric,history_required', [
    (few_shot.PersonaFewShotRetriever, 'is_approved', 'history_pool_approval', True),
    (history_sources.HistorySources, 'role', 'history_role_validation', True),
    (learned_sources, 'reconstruct', 'ranker_reconstruction', True),
    (ranker_report, 'load_completed', 'ranker_completed_sources', True),
    (ranker_samples, 'teacher_overlap', 'ranker_teacher_isolation', True),
    (datasets, 'static_sources', 'material_static_sources', False),
    (learning_guard.RunSeal, '__init__', 'material_seal_capture', False),
    (learning_guard.RunSeal, 'check', 'material_seal_check', False),
])
def test_preparation_timings_preserve_calls_failures_and_restore(
        tmp_path, monkeypatch, owner, method, metric, history_required):
    clock = [0.0]
    monkeypatch.setattr(transport_profile.time, 'monotonic', lambda: clock[0])
    expected = object()
    calls = []

    def original(value, *, fail=False):
        calls.append((value, fail))
        clock[0] += 2
        if fail:
            raise ValueError('preparation failed')
        return value

    monkeypatch.setattr(owner, method, original)
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    with pytest.raises(ValueError, match='preparation failed'):
        with profile.instrument(history_required=history_required):
            assert getattr(owner, method)(expected) is expected
            getattr(owner, method)(expected, fail=True)
    assert getattr(owner, method) is original
    assert calls == [(expected, False), (expected, True)]
    assert profile.metrics[metric] == dict(count=2, failed=1, total_seconds=4,
        max_seconds=2, active=0, active_peak=1)


def test_profile_is_thread_safe_bounded_and_written_on_stop(tmp_path):
    destination = tmp_path / 'profile.json'
    profile = transport_profile.TransportProfile(destination, {'workers': 4}, interval_seconds=100)
    barrier = Barrier(4)

    def work(number):
        with profile.span('retrieval_compute'):
            barrier.wait(timeout=10)
        return number

    with pytest.raises(control.StopRequested):
        with profile:
            with ThreadPoolExecutor(max_workers=4) as pool:
                assert list(pool.map(work, range(4))) == list(range(4))
            for number in range(100):
                profile.record_slice(success=number)
            raise control.StopRequested('time_slice_complete')
    saved = json.loads(destination.read_text())
    assert saved['status'] == 'stopped'
    assert saved['slices'] == [{'success': 99}]
    assert saved['metrics']['retrieval_compute']['count'] == 4
    assert saved['metrics']['retrieval_compute']['active_peak'] == 4
    assert saved['metrics']['retrieval_compute']['active'] == 0
    assert not profile.thread.is_alive()
    assert pack_transport.live_transport_profile().MODE == transport_profile.MODE


def test_instrument_without_history_preserves_history_functions(tmp_path):
    targets = [(few_shot.PersonaFewShotRetriever, 'retrieve'),
        (few_shot.PersonaFewShotRetriever, '_load'), (shared_history.PinnedRetriever, 'retrieve'),
        (history_sources.HistorySources, '__init__')]
    originals = [getattr(owner, name) for owner, name in targets]
    original_request = control.request
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    with profile.instrument(history_required=False):
        assert [getattr(owner, name) for owner, name in targets] == originals
        assert control.request is not original_request
    assert control.request is original_request


def test_batched_counter_does_not_add_timing_or_active_calls(tmp_path):
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    profile.record('source_offset_hits', 0.0, count=1000)
    assert profile.metrics['source_offset_hits'] == dict(count=1000, failed=0, total_seconds=0.0,
        max_seconds=0.0, active=0, active_peak=0)


def test_internal_history_and_disk_load_times_are_separate_and_restored(tmp_path, monkeypatch, disk_reader):
    clock = [0.0]
    monkeypatch.setattr(transport_profile.time, 'monotonic', lambda: clock[0])

    def validate(*args, **kwargs):
        clock[0] += 2

    def filter_rows(rows, case):
        clock[0] += 3
        return rows

    monkeypatch.setattr(history_sources.HistorySources, 'validate', validate)
    monkeypatch.setattr(history, 'filter_rows', filter_rows)
    original_cached = disk_history.BoundedReader.cached
    reader = disk_reader
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})

    def load():
        clock[0] += 5
        return {'unchanged': True}

    with pytest.raises(ValueError, match='original'):
        with profile.instrument(disk_history=disk_history):
            history_sources.HistorySources.validate(None, {})
            assert history.filter_rows([1], {}) == [1]
            for cache in (reader.cache, reader.features):
                assert reader.cached(cache, 'same', load) == {'unchanged': True}
                assert reader.cached(cache, 'same', load) == {'unchanged': True}
            raise ValueError('original')
    assert history_sources.HistorySources.validate is validate
    assert history.filter_rows is filter_rows
    assert disk_history.BoundedReader.cached is original_cached
    assert profile.metrics['history_source_validation']['total_seconds'] == 2
    assert profile.metrics['history_eligibility_filter']['total_seconds'] == 3
    for name in ('history_disk_row', 'history_disk_feature'):
        assert profile.metrics[name + '_read_parse']['total_seconds'] == 5
        assert profile.metrics[name + '_hit']['count'] == 1
        assert profile.metrics[name + '_miss']['count'] == 1


def test_disk_profile_acquires_only_the_original_reader_lock(tmp_path):
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    source = tmp_path / 'source.jsonl'
    source.write_text('{}\n')
    with profile.instrument(disk_history=disk_history):
        reader = disk_history.BoundedReader(source, sqlite3.connect(':memory:'))
        original_lock = reader.lock
        try:
            def load():
                assert not reader.lock._is_owned()
                assert not profile.lock.locked()
                return 7

            assert reader.cached(reader.cache, 'same', load) == 7
            assert reader.cached(reader.cache, 'same', lambda: 8) == 7
            assert reader.lock is original_lock
            metric = profile.snapshot()['metrics']['history_disk_lock_wait']
            assert metric['count'] == 7
            assert metric['total_seconds'] >= 0
        finally:
            reader.close()
    assert profile.metrics['history_disk_row_miss']['count'] == 1
    assert profile.metrics['history_disk_row_hit']['count'] == 1


@pytest.mark.parametrize('cache_name', ['cache', 'features'])
def test_disk_profile_records_failed_load_as_miss_not_hit(tmp_path, cache_name, disk_reader):
    reader = disk_reader
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})

    def load():
        raise ValueError('original load failure')

    with profile.instrument(disk_history=disk_history):
        with pytest.raises(ValueError, match='original load failure'):
            reader.cached(getattr(reader, cache_name), 'same', load)
    name = 'history_disk_row' if cache_name == 'cache' else 'history_disk_feature'
    assert profile.metrics[name + '_miss']['count'] == 1
    assert profile.metrics[name + '_read_parse']['failed'] == 1
    assert name + '_hit' not in profile.metrics
    assert not getattr(reader, cache_name)


def test_repeated_records_and_spans_do_not_acquire_global_profile_lock(tmp_path):
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    profile.record('counter', 1.0)
    with profile.span('span'):
        pass

    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError('hot path acquired shared profile lock')

        def __exit__(self, *args):
            pass

    original = profile.lock
    profile.lock = ForbiddenLock()
    try:
        profile.record('counter', 2.0, failed=True, count=3)
        with profile.span('span'):
            pass
    finally:
        profile.lock = original
    assert profile.metrics['counter'] == dict(count=4, failed=3, total_seconds=3.0,
        max_seconds=2.0, active=0, active_peak=0)
    assert profile.metrics['span']['count'] == 2


def test_snapshots_do_not_lose_or_duplicate_thread_local_counts(tmp_path):
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    barrier = Barrier(5)
    stop = Event()
    observed = []

    def observe():
        barrier.wait(timeout=10)
        while not stop.wait(0.001):
            observed.append(profile.snapshot()['metrics'].get('counter', {}).get('count', 0))

    def work(number):
        barrier.wait(timeout=10)
        for index in range(1000):
            profile.record('counter', 0.25, failed=number == 0, count=2)

    thread = Thread(target=observe)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(work, range(4)))
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert observed == sorted(observed)
    assert all(count <= 8000 for count in observed)
    profile.flush('ok')
    metric = json.loads(profile.path.read_text())['metrics']['counter']
    assert metric == dict(count=8000, failed=2000, total_seconds=1000.0,
        max_seconds=0.25, active=0, active_peak=0)


def test_active_peak_is_real_concurrency_not_sum_of_thread_peaks(tmp_path):
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    barrier = Barrier(4)
    serial = Lock()

    def work(number):
        barrier.wait(timeout=10)
        with serial, profile.span('serial'):
            snapshot = profile.snapshot()['metrics']['serial']
            assert snapshot['active'] == 1
            assert snapshot['active_peak'] == 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(4)))
    metric = profile.metrics['serial']
    assert metric['active'] == 0
    assert metric['active_peak'] == 1
    assert metric['count'] == 4
