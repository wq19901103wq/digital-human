from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from threading import Barrier, Event

import pytest

from leakage_support import historical
from test_disk_history import FLAGS, queries, retriever
from test_fewshot_selection import generator
from test_leakage_guards import env as env
from src.config import ConfigError, sha256_file
from src.generator import disk_history, disk_source_reads, few_shot, history_sources
from src.iteration import pack_transport, transport_profile


@pytest.fixture
def reader(tmp_path):
    source = tmp_path / 'rows.jsonl'
    source.write_bytes(b'{"row":1}\n{"row":2}\n')
    connection = sqlite3.connect(tmp_path / 'index.sqlite3', check_same_thread=False)
    connection.execute('CREATE TABLE examples(id TEXT, features TEXT, position INTEGER)')
    connection.executemany('INSERT INTO examples VALUES(?,?,?)',
                           [('one', '{"value":1}', 0), ('two', '{"value":2}', 1)])
    connection.commit()
    result = disk_history.BoundedReader(source, connection)
    try:
        yield result
    finally:
        result.close()


def test_worker_keeps_only_current_row_and_feature_without_shared_lru(reader, monkeypatch):
    original_loads, parsed = json.loads, []

    def loads(value):
        parsed.append(value)
        return original_loads(value)

    monkeypatch.setattr(disk_history.json, 'loads', loads)
    with disk_source_reads.source_reads(disk_history):
        first = reader.row(0, 10)
        feature = reader.feature('one')
        assert reader.row(0, 10) is first
        assert reader.feature('one') is feature
        assert len(parsed) == 2
        assert reader.row(10, 10) == {'row': 2}
        assert reader.feature('two') == {'value': 2}
        assert reader.row(0, 10) == first
        assert reader.feature('one') == feature
        assert len(parsed) == 6
        current = reader.local.source_read_state[1]
        assert current['row'] == (0, first)
        assert current['feature'] == ('one', feature)
        assert not reader.cache and not reader.features
        handle = current['connection']
        assert handle is not reader.connection
        assert handle.execute('PRAGMA query_only').fetchone() == (1,)
        assert handle.execute('PRAGMA cache_size').fetchone() == (-disk_source_reads.SQLITE_CACHE_KIB,)
        assert handle.execute('PRAGMA mmap_size').fetchone() == (0,)
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            handle.execute("INSERT INTO examples VALUES('bad','{}',2)")
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        handle.execute('SELECT 1')
    assert current['row'] is None and current['feature'] is None


def test_workers_use_independent_connections_and_overlap_queries_and_parsing(reader, monkeypatch):
    barrier = Barrier(4)
    original_connect, original_loads = sqlite3.connect, json.loads

    class Connection(sqlite3.Connection):
        def execute(self, statement, parameters=()):
            if statement == 'SELECT ?':
                assert not reader.lock._is_owned()
                barrier.wait(timeout=5)
            return super().execute(statement, parameters)

    def connect(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=Connection)

    def loads(value):
        assert not reader.lock._is_owned()
        barrier.wait(timeout=5)
        return original_loads(value)

    def read(identity):
        result = reader.query('SELECT ?', (identity,))
        feature = reader.feature(identity)
        return result, feature, reader.local.source_read_state[1]['connection']

    monkeypatch.setattr(disk_source_reads.sqlite3, 'connect', connect)
    monkeypatch.setattr(disk_history.json, 'loads', loads)
    with disk_source_reads.source_reads(disk_history):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(read, ('one', 'two', 'one', 'two')))
        assert [result for result, _, _ in results] == [[('one',)], [('two',)], [('one',)], [('two',)]]
        assert [feature for _, feature, _ in results] == [{'value': 1}, {'value': 2}] * 2
        handles = [handle for _, _, handle in results]
        assert len({id(handle) for handle in handles}) == 4
        assert not reader.cache and not reader.features
        assert reader.active_reads == 0
    for handle in handles:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            handle.execute('SELECT 1')


def test_close_drains_worker_query_and_closes_all_connections(reader, monkeypatch):
    started, released, closing = Event(), Event(), Event()
    original_connect, handles = sqlite3.connect, []

    class Connection(sqlite3.Connection):
        def execute(self, statement, parameters=()):
            if statement == 'SELECT 1':
                started.set()
                assert released.wait(timeout=5)
                assert not reader.stream.closed
            return super().execute(statement, parameters)

    def connect(*args, **kwargs):
        handle = original_connect(*args, **kwargs, factory=Connection)
        handles.append(handle)
        return handle

    def close():
        with reader.guard():
            reader.closing = True
            closing.set()
        reader.close()

    monkeypatch.setattr(disk_source_reads.sqlite3, 'connect', connect)
    with disk_source_reads.source_reads(disk_history):
        with ThreadPoolExecutor(max_workers=3) as pool:
            result = pool.submit(reader.query, 'SELECT 1')
            assert started.wait(timeout=5)
            stopped = pool.submit(close)
            assert closing.wait(timeout=5)
            try:
                with pytest.raises(ValueError, match='closed'):
                    reader.query('SELECT 2')
                assert not stopped.done()
            finally:
                released.set()
            assert result.result(timeout=5) == [(1,)]
            stopped.result(timeout=5)
            pool.submit(reader.close).result(timeout=5)
    assert reader.closed and reader.stream.closed and reader.active_reads == 0
    for handle in handles:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            handle.execute('SELECT 2')


def test_failed_load_and_scope_exit_restore_methods_and_release_resources(reader):
    originals = (disk_history.Chat.__getitem__, disk_history.BoundedReader.cached,
                 disk_history.BoundedReader.query, disk_history.BoundedReader.close)

    def load():
        assert reader.query('SELECT 1') == [(1,)]
        with pytest.raises(RuntimeError, match='active read'):
            reader.close()
        raise ValueError('load failed')

    with pytest.raises(RuntimeError, match='scope failed'):
        with disk_source_reads.source_reads(disk_history) as windows:
            with pytest.raises(ValueError, match='load failed'):
                reader.cached(reader.cache, 'bad', load)
            assert reader.active_reads == 0 and not reader.cache
            assert reader.local.source_read_state[1]['row'] is None
            handle = reader.local.source_read_state[1]['connection']
            raise RuntimeError('scope failed')
    assert not windows
    assert (disk_history.Chat.__getitem__, disk_history.BoundedReader.cached,
            disk_history.BoundedReader.query, disk_history.BoundedReader.close) == originals
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        handle.execute('SELECT 1')
    assert reader.query('SELECT 1') == [(1,)]


def test_slices_read_each_message_once_and_scalars_reuse_bounded_offsets(tmp_path, monkeypatch):
    source = historical(tmp_path / 'data')
    expected = history_sources.HistorySources(source.directory).chats['chat-0']
    original = disk_history.Chat.__getitem__
    with disk_history.disk_storage(few_shot):
        stored = history_sources.load(source.directory)
        actual = stored.chats['chat-0']
        row_calls, sql_calls = [], []
        original_row = actual.reader.row

        def row(offset, size):
            row_calls.append(offset)
            return original_row(offset, size)

        monkeypatch.setattr(actual.reader, 'row', row)
        original_connect = sqlite3.connect

        def connect(*args, **kwargs):
            connection = original_connect(*args, **kwargs)
            connection.set_trace_callback(sql_calls.append)
            return connection

        monkeypatch.setattr(disk_source_reads.sqlite3, 'connect', connect)
        profile = transport_profile.TransportProfile(tmp_path / 'profile.json', {})
        with disk_source_reads.source_reads(disk_history, capacity=8, profile=profile) as windows:
            for selection in (slice(None), slice(8, 25), slice(None, None, -1),
                              slice(30, 5, -3), slice(4, 4)):
                row_calls.clear()
                assert actual[selection] == expected[selection]
                assert len(row_calls) == len(expected[selection])
            sql_calls.clear()
            assert actual[3] == expected[3]
            assert actual[4] == expected[4]
            assert len([sql for sql in sql_calls if sql.startswith('SELECT position')]) == 1
            assert actual[-1] == expected[-1]
            for index in range(len(actual)):
                assert actual[index] == expected[index]
                assert len(windows[actual.reader][1]) <= 8
            with ThreadPoolExecutor(max_workers=4) as pool:
                assert list(pool.map(actual.__getitem__, range(len(actual)))) == expected
            with pytest.raises(IndexError):
                actual[len(actual)]
            actual.reader.close()
            assert not windows
        assert profile.metrics['history_source_offset_cache_hit']['count'] > 0
        assert profile.metrics['history_source_offset_cache_miss']['count'] > 0
    assert disk_history.Chat.__getitem__ is original


@pytest.mark.parametrize('flags', [(), FLAGS])
def test_retrieval_and_warm_index_identity_unchanged(tmp_path, flags):
    source = historical(tmp_path / 'data')
    requests = list(queries(source))
    with disk_history.disk_storage(few_shot):
        baseline = retriever(source, flags)
        expected = [baseline.retrieve(**request) for request in requests]
    root = tmp_path / '.cache/disk-history'
    checksums = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in root.glob('*.sqlite3')}
    with disk_history.disk_storage(few_shot), disk_source_reads.source_reads(disk_history, retrievers=few_shot):
        candidate = retriever(source, flags)
        assert [candidate.retrieve(**request) for request in requests] == expected
    assert {path: (sha256_file(path), path.stat().st_mtime_ns) for path in root.glob('*.sqlite3')} == checksums


def test_prompt_and_request_identity_unchanged(env):
    baseline = generator(env)
    cases = env.source.roles['development'] + env.source.roles['fixed_test']
    expected = [baseline.build_prompt(case) for case in cases]
    with disk_history.disk_storage(few_shot), disk_source_reads.source_reads(disk_history, retrievers=few_shot):
        candidate = generator(env)
        assert [candidate.build_prompt(case) for case in cases] == expected
        assert env.calls == []


def test_initialized_retrieval_protects_lifecycle_once_without_per_field_locks(tmp_path):
    source = historical(tmp_path / 'data')
    request = next(queries(source))
    original = few_shot.PersonaFewShotRetriever.retrieve
    with disk_history.disk_storage(few_shot):
        candidate = retriever(source, FLAGS)
        expected = candidate.retrieve(**request)
        readers = (candidate._disk_reader, candidate._history_sources._disk_reader)
        for reader in readers:
            reader.lock_timing = dict(count=0, total_seconds=0.0, max_seconds=0.0)
        with disk_source_reads.source_reads(disk_history, retrievers=few_shot):
            assert candidate.retrieve(**request) == expected
            assert all(reader.lock_timing['count'] <= 4 for reader in readers)
            assert all(reader.active_reads == 0 for reader in readers)
    assert few_shot.PersonaFewShotRetriever.retrieve is original


@pytest.mark.parametrize('changed', ['messages.jsonl', 'fewshot_pool.jsonl', 'purposes.json', 'report.json'])
def test_cached_offsets_cannot_bypass_source_change_guard(tmp_path, changed):
    source = historical(tmp_path / 'data')
    originals = (disk_history.Chat.__getitem__, disk_history.BoundedReader.close)
    with pytest.raises(ConfigError, match='发生变化'):
        with disk_history.disk_storage(few_shot), disk_source_reads.source_reads(disk_history) as windows:
            candidate = retriever(source)
            candidate.retrieve(**next(queries(source)))
            path = source.directory / changed
            path.write_text(path.read_text() + '\n')
            candidate.retrieve(**next(queries(source)))
    assert not windows
    assert (disk_history.Chat.__getitem__, disk_history.BoundedReader.close) == originals


def test_field_validation_still_rejects_forged_input(tmp_path):
    source = historical(tmp_path / 'data')
    with disk_history.disk_storage(few_shot), disk_source_reads.source_reads(disk_history):
        candidate = retriever(source)
        request = next(queries(source))
        candidate.retrieve(**request)
        forged = json.loads(json.dumps(request['history_case']))
        forged['context'][0]['text'] = 'forged source'
        with pytest.raises(ConfigError, match='来源'):
            candidate.retrieve(**{**request, 'history_case': forged})


def test_live_adapter_and_capacity_validation():
    live = pack_transport.live_disk_source_reads()
    assert live.MODE == disk_source_reads.MODE
    assert live.CAPACITY == 256
    with pytest.raises(ValueError, match='positive'):
        with live.source_reads(disk_history, capacity=0):
            pass
