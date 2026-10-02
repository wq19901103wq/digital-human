from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from threading import Barrier, Event

import pytest

from src.generator import disk_history


@pytest.fixture
def reader(tmp_path):
    source = tmp_path / 'rows.jsonl'
    source.write_text('{"row":1}\n{"row":2}\n')
    connection = sqlite3.connect(':memory:', check_same_thread=False)
    connection.execute('CREATE TABLE examples(id TEXT, features TEXT, position INTEGER)')
    connection.executemany('INSERT INTO examples VALUES(?,?,?)',
                           [('one', '{"value":1}', 0), ('two', '{"value":2}', 1)])
    result = disk_history.BoundedReader(source, connection)
    try:
        yield result
    finally:
        result.close()


def test_independent_and_same_key_loads_overlap_and_publish_once(reader):
    barrier = Barrier(4)

    def retrieve(key):
        def load():
            assert not reader.lock._is_owned()
            barrier.wait(timeout=5)
            return {'key': key}
        return reader.cached(reader.cache, key, load)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(retrieve, ('same', 'same', 'one', 'two')))
    assert results[0] is results[1]
    assert list(reader.cache) and len(reader.cache) == 3
    assert reader.active_reads == 0


def test_cache_stays_bounded_during_parallel_eviction(reader, monkeypatch):
    monkeypatch.setattr(disk_history, 'CAPACITY', 2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda key: reader.cached(reader.cache, key, lambda: key), range(100))) == list(range(100))
    assert len(reader.cache) == 2


def test_sql_is_serialized_but_feature_json_parse_is_not(reader, monkeypatch):
    barrier = Barrier(2)
    original = json.loads

    def parse(value):
        assert not reader.lock._is_owned()
        barrier.wait(timeout=5)
        return original(value)

    monkeypatch.setattr(disk_history.json, 'loads', parse)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(reader.feature, ('one', 'two'))) == [{'value': 1}, {'value': 2}]


def test_close_drains_active_load_and_rejects_new_reads(reader):
    started, release, closing = Event(), Event(), Event()

    def load():
        started.set()
        assert release.wait(timeout=5)
        assert not reader.stream.closed
        return 7

    def close():
        with reader.guard():
            reader.closing = True
            closing.set()
        reader.close()

    with ThreadPoolExecutor(max_workers=3) as pool:
        result = pool.submit(reader.cached, reader.cache, 'one', load)
        assert started.wait(timeout=5)
        stopped = pool.submit(close)
        assert closing.wait(timeout=5)
        try:
            with pytest.raises(ValueError, match='closed'):
                reader.query('SELECT 1')
            assert not stopped.done()
        finally:
            release.set()
        assert result.result(timeout=5) == 7
        stopped.result(timeout=5)
        pool.submit(reader.close).result(timeout=5)
    assert reader.active_reads == 0
    assert reader.closed and reader.stream.closed
    assert not reader.cache


def test_failed_load_and_nested_query_release_lifecycle(reader):
    def load():
        assert reader.query('SELECT 1') == [(1,)]
        with pytest.raises(RuntimeError, match='active read'):
            reader.close()
        raise ValueError('load failed')

    with pytest.raises(ValueError, match='load failed'):
        reader.cached(reader.cache, 'bad', load)
    assert reader.active_reads == 0
    assert not reader.cache
    reader.close()


def test_disk_row_reads_and_chat_slices_parse_once(reader, monkeypatch):
    from collections import namedtuple
    reader.connection.execute('CREATE TABLE messages(chat TEXT,position INTEGER,offset INTEGER,size INTEGER)')
    reader.connection.executemany('INSERT INTO messages VALUES(?,?,?,?)', [('chat', 0, 0, 10), ('chat', 1, 10, 10)])
    chat = disk_history.Chat(reader, 'chat', 2, namedtuple('Message', 'row'))
    original_row, calls = reader.row, []

    def row(offset, size):
        calls.append(offset)
        return original_row(offset, size)

    monkeypatch.setattr(reader, 'row', row)
    assert [message.row for message in chat[::-1]] == [2, 1]
    assert calls == [0, 10]
