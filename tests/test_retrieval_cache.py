"""Exact retrieval reuse must preserve frozen output and all source guards."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
from threading import Barrier, Event, RLock
from types import SimpleNamespace

import pytest

from leakage_support import historical
from test_disk_history import FLAGS, queries, retriever
from src.config import ConfigError, sha256_file
from src.generator import disk_history, disk_source_reads, few_shot, retrieval_cache, shared_history
from src.iteration import pack_transport, transport_profile


class Guard:
    def __init__(self):
        self.changed = False
        self.checks = 0

    def check(self):
        self.checks += 1
        if self.changed:
            raise ConfigError('source changed')


class Original:
    def __init__(self, pinned):
        self._pinned = pinned
        self.history_policy = 'complete_before_input_v1'
        self.enable_clarification = False
        self._history_sources = Guard()

    def retrieve(self, **kwargs):
        self._pinned.calls += 1
        if self._pinned.fail:
            raise ValueError('original failure')
        return [{'id': 'original', 'nested': {'values': [1, 2]}, 'tuple': (3, 4)}]

    def render_selected(self, rows, max_chars=2500):
        return str(rows)[:max_chars], [row['id'] for row in rows]


class Pinned:
    def __init__(self):
        self._lock = RLock()
        self._original = Original(self)
        self.changed = False
        self.calls = 0
        self.fail = False

    def _check(self):
        if self.changed:
            raise ConfigError('pool changed')

    def is_approved(self):
        with self._lock:
            self._check()
            return True

    def retrieve(self, **kwargs):
        with self._lock:
            self._check()
            rows = self._original.retrieve(**kwargs)
            self._check()
            return rows

    def render_selected(self, rows, max_chars=2500):
        with self._lock:
            self._check()
            result = self._original.render_selected(rows, max_chars=max_chars)
            self._check()
            return result


@pytest.fixture(params=['current', 'frozen_serialized'])
def pinned_runtime(request, monkeypatch):
    if request.param == 'frozen_serialized':
        class FrozenPinned(shared_history.PinnedRetriever):
            def __init__(self, *args, **kwargs):
                self._lock = RLock()
                super().__init__(*args, **kwargs)

            def is_approved(self):
                with self._lock:
                    return super().is_approved()

            def retrieve(self, **kwargs):
                with self._lock:
                    return super().retrieve(**kwargs)

            def render_selected(self, rows, max_chars=2500):
                with self._lock:
                    return super().render_selected(rows, max_chars=max_chars)

        monkeypatch.setattr(shared_history, 'PinnedRetriever', FrozenPinned)
    return shared_history


@contextmanager
def adapted(tmp_path, **options):
    module = SimpleNamespace(PinnedRetriever=Pinned)
    originals = {name: getattr(Pinned, name) for name in ('retrieve', 'render_selected', 'is_approved')}
    profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
    instance = Pinned()
    try:
        with retrieval_cache.exact_retrieval_calls(module, profile=profile, **options) as caches:
            yield instance, caches, profile
    finally:
        assert all(getattr(Pinned, name) is method for name, method in originals.items())


def test_exact_inputs_reuse_independent_results_under_parallel_calls(tmp_path):
    with adapted(tmp_path) as (instance, caches, profile):
        request = dict(query='same', history_case={'input_cutoff': {'timestamp': 1}}, exclude_ids={'b', 'a'})
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _: instance.retrieve(**copy.deepcopy(request)), range(8)))
        assert instance.calls == 1
        assert all(result == results[0] for result in results)
        results[0][0]['nested']['values'].clear()
        results[1][0]['tuple'] = ()
        assert instance.retrieve(**request)[0]['nested']['values'] == [1, 2]
        assert instance.retrieve(**request)[0]['tuple'] == (3, 4)
        assert profile.metrics['retrieval_cache_miss']['count'] == 1
        assert profile.metrics['retrieval_cache_hit']['count'] == 9
        assert profile.metrics['retrieval_lock_wait']['count'] == 10
        assert len(caches[instance].entries) == 1
    assert caches == {}


@pytest.mark.parametrize('field,value', [
    ('query', 'different'), ('exclude_ids', {'other'}), ('limit', 2),
    ('history_case', {'input_cutoff': {'timestamp': 2}}),
    ('history_case', {'input_cutoff': {'timestamp': 1}, 'reply_message_ids': ['other']}),
    ('current_context_messages', [{'sender': 'other', 'text': 'same'}]),
    ('include_retrieval_features', True), ('use_situation', False),
])
def test_changed_query_boundary_source_exclusion_or_options_do_not_reuse(tmp_path, field, value):
    with adapted(tmp_path) as (instance, _, profile):
        request = dict(query='same', exclude_ids={'a'}, limit=3,
            history_case={'input_cutoff': {'timestamp': 1}},
            current_context_messages=[{'sender': 'original', 'text': 'same'}],
            include_retrieval_features=False, use_situation=True)
        instance.retrieve(**request)
        instance.retrieve(**dict(request, **{field: value}))
        assert instance.calls == 2
        assert profile.metrics['retrieval_cache_miss']['count'] == 2


def test_flags_environment_and_source_instance_are_separate(tmp_path, monkeypatch):
    with adapted(tmp_path) as (instance, _, _):
        instance.retrieve(query='same')
        instance._original.enable_clarification = True
        instance.retrieve(query='same')
        monkeypatch.setenv('PERSONA_FEW_SHOT_MIN_SIMILARITY', '0.7')
        instance.retrieve(query='same')
        instance._original._history_sources = Guard()
        instance.retrieve(query='same')
        assert instance.calls == 4


@pytest.mark.parametrize('changed', ['source', 'pool'])
def test_cache_hit_still_rejects_source_and_pool_changes(tmp_path, changed):
    with adapted(tmp_path) as (instance, _, _):
        instance.retrieve(query='same')
        if changed == 'source':
            instance._original._history_sources.changed = True
        else:
            instance.changed = True
        with pytest.raises(ConfigError, match='changed'):
            instance.retrieve(query='same')
        assert instance.calls == 1


def test_original_failures_unknown_inputs_and_unguarded_calls_are_not_cached(tmp_path):
    with adapted(tmp_path) as (instance, _, profile):
        instance.fail = True
        for _ in range(2):
            with pytest.raises(ValueError, match='original failure'):
                instance.retrieve(query='same')
        instance.fail = False
        unknown = object()
        for _ in range(2):
            instance.retrieve(query='same', unknown=unknown)
        instance._original._history_sources = None
        for _ in range(2):
            instance.retrieve(query='same')
        assert instance.calls == 6
        assert profile.metrics['retrieval_cache_bypass']['count'] == 4


def test_entry_byte_budget_and_lru_are_bounded(tmp_path):
    with adapted(tmp_path, capacity=2, max_bytes=4096) as (instance, caches, _):
        for query in ('first', 'second', 'first', 'third', 'first', 'second'):
            instance.retrieve(query=query)
        assert instance.calls == 4
        assert len(caches[instance].entries) == 2
        assert caches[instance].bytes <= 4096
    with adapted(tmp_path, capacity=128, max_bytes=1) as (instance, caches, _):
        for _ in range(2):
            instance.retrieve(query='same')
        assert instance.calls == 2
        assert caches[instance].entries == {}


@pytest.mark.parametrize('left,right', [(True, 1), (1, 1.0), ([1], (1,)), ({1}, frozenset({1})), (0.0, -0.0)])
def test_input_types_are_not_conflated(left, right):
    assert retrieval_cache.identity(left) != retrieval_cache.identity(right)
    assert retrieval_cache.identity({'a': 1, 'b': 2}) == retrieval_cache.identity({'b': 2, 'a': 1})


@pytest.mark.parametrize('flags', [(), *[(name,) for name in FLAGS], FLAGS])
def test_disk_retrieval_and_render_are_identical_for_every_flag(tmp_path, flags, pinned_runtime):
    source = historical(tmp_path / 'data')
    requests = list(queries(source))
    baseline = retriever(source, flags)
    expected = [baseline.retrieve(**request) for request in requests]
    pool = source.directory / 'fewshot_pool.jsonl'
    inputs = {str(path.resolve()): sha256_file(path) for path in (pool, pool.with_name('report.json'))}
    with disk_history.disk_storage(few_shot), retrieval_cache.exact_retrieval_calls(pinned_runtime):
        candidate = pinned_runtime.PinnedRetriever(few_shot.PersonaFewShotRetriever, pool, inputs)
        for name in flags:
            setattr(candidate._original, name, not getattr(candidate._original, name))
        for request, rows in zip(requests, expected):
            assert [dict(row) for row in candidate.retrieve(**request)] == rows
            assert candidate.retrieve(**request) == rows
            assert candidate.render_selected(candidate.retrieve(**request)) == baseline.render_selected(rows)
    assert pack_transport.live_retrieval_cache().MODE == retrieval_cache.MODE


@pytest.mark.parametrize('bypass', [None, 'source', 'unsupported'])
def test_different_requests_compute_concurrently(tmp_path, monkeypatch, bypass):
    barrier = Barrier(4)
    original = Original.retrieve

    def compute(instance, **kwargs):
        barrier.wait(timeout=5)
        return original(instance, **kwargs)

    monkeypatch.setattr(Original, 'retrieve', compute)
    with adapted(tmp_path) as (instance, _, profile):
        if bypass == 'source':
            instance._original._history_sources = None

        def work(number):
            request = dict(query=str(number))
            if bypass == 'unsupported':
                request['unknown'] = object()
            return instance.retrieve(**request)

        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(work, range(4)))
        assert instance.calls == 4
        assert all(result == results[0] for result in results)
        assert profile.metrics['retrieval_cache_' + ('bypass' if bypass else 'miss')]['count'] == 4


def test_hot_cache_hit_does_not_wait_for_another_requests_miss(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    original = Original.retrieve

    def compute(instance, **kwargs):
        if kwargs['query'] == 'slow':
            entered.set()
            assert release.wait(timeout=10)
        return original(instance, **kwargs)

    monkeypatch.setattr(Original, 'retrieve', compute)
    with adapted(tmp_path) as (instance, _, profile):
        expected = instance.retrieve(query='hot')
        with ThreadPoolExecutor(max_workers=2) as workers:
            slow = workers.submit(instance.retrieve, query='slow')
            try:
                assert entered.wait(timeout=5)
                hot = workers.submit(instance.retrieve, query='hot')
                assert hot.result(timeout=5) == expected
                assert not slow.done()
            finally:
                release.set()
            assert slow.result(timeout=5) == expected
        assert instance.calls == 2
        assert profile.metrics['retrieval_cache_hit']['count'] == 1


def test_frozen_render_calls_compute_concurrently(tmp_path, monkeypatch):
    barrier = Barrier(4)
    original = Original.render_selected

    def render(instance, *args, **kwargs):
        barrier.wait(timeout=5)
        return original(instance, *args, **kwargs)

    monkeypatch.setattr(Original, 'render_selected', render)
    with adapted(tmp_path) as (instance, _, _):
        rows = instance.retrieve(query='same')
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _: instance.render_selected(rows, max_chars=30), range(4)))
        assert results == [(str(rows)[:30], ['original'])] * 4


def test_hot_hit_and_approval_do_not_wait_for_frozen_render(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    original = Original.render_selected

    def render(instance, *args, **kwargs):
        entered.set()
        assert release.wait(timeout=10)
        return original(instance, *args, **kwargs)

    monkeypatch.setattr(Original, 'render_selected', render)
    with adapted(tmp_path) as (instance, _, _):
        expected = instance.retrieve(query='hot')
        with ThreadPoolExecutor(max_workers=3) as workers:
            slow = workers.submit(instance.render_selected, expected)
            try:
                assert entered.wait(timeout=5)
                hot = workers.submit(instance.retrieve, query='hot')
                approved = workers.submit(instance.is_approved)
                assert hot.result(timeout=5) == expected
                assert approved.result(timeout=5) is True
                assert not slow.done()
            finally:
                release.set()
            assert slow.result(timeout=5) == (str(expected), ['original'])


def test_frozen_methods_are_restored_after_scope_failure(tmp_path):
    with pytest.raises(ValueError, match='scope failure'):
        with adapted(tmp_path) as (instance, _, _):
            assert instance.is_approved()
            instance.retrieve(query='same')
            raise ValueError('scope failure')


@pytest.mark.parametrize('changed', ['pool', 'report'])
@pytest.mark.parametrize('method', ['retrieve', 'render_selected'])
def test_runtime_adapter_keeps_pre_and_post_file_guards(tmp_path, monkeypatch, changed, method, pinned_runtime):
    source = historical(tmp_path / 'data')
    request = next(iter(queries(source)))
    rows = retriever(source).retrieve(**request)
    pool = source.directory / 'fewshot_pool.jsonl'
    report = pool.with_name('report.json')
    inputs = {str(path.resolve()): sha256_file(path) for path in (pool, report)}
    target = pool if changed == 'pool' else report
    original = getattr(few_shot.PersonaFewShotRetriever, method)
    calls = []

    def changing(instance, *args, **kwargs):
        calls.append(method)
        result = original(instance, *args, **kwargs)
        target.write_text('changed during call')
        return result

    monkeypatch.setattr(few_shot.PersonaFewShotRetriever, method, changing)
    with retrieval_cache.exact_retrieval_calls(pinned_runtime) as caches:
        candidate = pinned_runtime.PinnedRetriever(few_shot.PersonaFewShotRetriever, pool, inputs)
        for _ in range(2):
            with pytest.raises(ConfigError, match='文件发生变化'):
                if method == 'retrieve':
                    candidate.retrieve(**request)
                else:
                    candidate.render_selected(rows)
        with pytest.raises(ConfigError, match='文件发生变化'):
            candidate.is_approved()
        assert calls == [method]
        assert not caches or not caches[candidate].entries


@pytest.mark.parametrize('changed', ['source', 'pool'])
def test_invalidated_miss_is_never_published_to_cache(tmp_path, monkeypatch, changed):
    original = Original.retrieve

    def compute(instance, **kwargs):
        result = original(instance, **kwargs)
        if changed == 'source':
            instance._history_sources.changed = True
        else:
            instance._pinned.changed = True
        return result

    monkeypatch.setattr(Original, 'retrieve', compute)
    with adapted(tmp_path) as (instance, caches, _):
        with pytest.raises(ConfigError, match='changed'):
            instance.retrieve(query='same')
        assert instance.calls == 1
        assert caches[instance].entries == {}
        assert caches[instance].bytes == 0


def test_parallel_lru_updates_stay_bounded_and_return_independent_results(tmp_path):
    with adapted(tmp_path, capacity=2, max_bytes=4096) as (instance, caches, _):
        with ThreadPoolExecutor(max_workers=8) as workers:
            results = list(workers.map(lambda number: instance.retrieve(query=str(number)), range(64)))
        assert instance.calls == 64
        cache = caches[instance]
        assert len(cache.entries) <= 2
        assert cache.bytes == sum(len(key) + len(payload) + 128 for key, payload in cache.entries.items())
        assert cache.bytes <= 4096
        results[0][0]['nested']['values'].clear()
        assert all(result[0]['nested']['values'] == [1, 2] for result in results[1:])


@pytest.mark.parametrize('flags', [(), FLAGS])
def test_parallel_disk_retrieval_cache_and_profile_preserve_frozen_output(tmp_path, monkeypatch, flags, pinned_runtime):
    source = historical(tmp_path / 'data')
    requests = list(queries(source))
    baseline = retriever(source, flags)
    expected = [baseline.retrieve(**request) for request in requests]
    rendered = [baseline.render_selected(rows) for rows in expected]
    pool = source.directory / 'fewshot_pool.jsonl'
    inputs = {str(path.resolve()): sha256_file(path) for path in (pool, pool.with_name('report.json'))}
    profile = transport_profile.TransportProfile(tmp_path / 'parallel-profile.json', {})
    barrier = Barrier(4)
    original = few_shot.PersonaFewShotRetriever.retrieve

    def compute(instance, **kwargs):
        barrier.wait(timeout=5)
        return original(instance, **kwargs)

    monkeypatch.setattr(few_shot.PersonaFewShotRetriever, 'retrieve', compute)
    with pack_transport.history_feature_cache(few_shot), disk_history.disk_storage(few_shot), \
            disk_source_reads.source_reads(disk_history, profile=profile), \
            retrieval_cache.exact_retrieval_calls(pinned_runtime, profile=profile), \
            profile.instrument(disk_history=disk_history):
        candidate = pinned_runtime.PinnedRetriever(few_shot.PersonaFewShotRetriever, pool, inputs)
        for name in flags:
            setattr(candidate._original, name, not getattr(candidate._original, name))

        def work(request):
            rows = candidate.retrieve(**request)
            return rows, candidate.render_selected(rows)

        with ThreadPoolExecutor(max_workers=4) as workers:
            actual = list(workers.map(work, requests))
            cached = list(workers.map(work, requests))
        assert actual == cached == list(zip(expected, rendered))
        assert profile.metrics['retrieval_compute']['count'] == len(requests)
        assert profile.metrics['retrieval_compute']['active_peak'] == 4
        assert profile.metrics['retrieval_cache_miss']['count'] == len(requests)
        assert profile.metrics['retrieval_cache_hit']['count'] == len(requests)
        assert profile.metrics['retrieval_total']['active_peak'] == 4
