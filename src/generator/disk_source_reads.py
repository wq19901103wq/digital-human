"""Worker-local disk reads without changing authenticated disk indexes."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import sqlite3

MODE = 'worker_local_sqlite_single_record_v3'
CAPACITY = 256
RECORD_CAPACITY = 1
SQLITE_CACHE_KIB = 256


@contextmanager
def source_reads(disk_history, *, retrievers=None, profile=None, capacity=CAPACITY):
    if capacity <= 0:
        raise ValueError('source offset capacity must be positive')
    original_getitem = disk_history.Chat.__getitem__
    original_close = disk_history.BoundedReader.close
    original_query = disk_history.BoundedReader.query
    original_cached = disk_history.BoundedReader.cached
    original_retrieve = retrievers.PersonaFewShotRetriever.retrieve if retrievers is not None else None
    windows = {}
    resources = {}
    paths = {}
    counts = []

    def state(reader):
        previous = getattr(reader.local, 'source_read_state', None)
        if previous is not None and previous[0] is resources:
            return previous[1]
        current = dict(connection=None, row=None, feature=None, window=(None, {}), counts=[0, 0])
        with reader.guard():
            resources.setdefault(reader, []).append(current)
            counts.append(current['counts'])
        reader.local.source_read_state = (resources, current)
        return current

    def connection(reader, current):
        if current['connection'] is None:
            with reader.guard():
                path = paths.get(reader)
                if path is None:
                    path = next(path for _, name, path in reader.connection.execute('PRAGMA database_list')
                                if name == 'main')
                    if not path:
                        raise ValueError('worker-local disk reads require a file-backed index')
                    paths[reader] = path
            handle = sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True,
                                     check_same_thread=False, cached_statements=32)
            try:
                handle.execute(f'PRAGMA cache_size=-{SQLITE_CACHE_KIB}')
                handle.execute('PRAGMA mmap_size=0')
                handle.execute('PRAGMA query_only=ON')
            except BaseException:
                handle.close()
                raise
            current['connection'] = handle
        return current['connection']

    def reader_query(reader, statement, parameters=()):
        with reader.reading():
            return connection(reader, state(reader)).execute(statement, parameters).fetchall()

    def cached(reader, cache, key, load):
        with reader.reading():
            current = state(reader)
            slot = 'row' if cache is reader.cache else 'feature'
            previous = current[slot]
            if previous is not None and previous[0] == key:
                return previous[1]
            value = load()
            current[slot] = (key, value)
            return value

    def release(reader):
        with reader.guard():
            states = resources.pop(reader, [])
            windows.pop(reader, None)
            paths.pop(reader, None)
        for current in states:
            if current['connection'] is not None:
                current['connection'].close()
            current.update(connection=None, row=None, feature=None, window=(None, {}))

    def retrieve(subject, *args, **kwargs):
        source = getattr(subject, '_history_sources', None)
        readers = (getattr(subject, '_disk_reader', None), getattr(source, '_disk_reader', None))
        with ExitStack() as stack:
            for reader in dict.fromkeys(readers):
                if reader is not None:
                    stack.enter_context(reader.reading())
            return original_retrieve(subject, *args, **kwargs)

    def query(chat, start, end, positions=None):
        timing = profile.span('history_source_offset_query') if profile is not None else nullcontext()
        with timing:
            return dict((position, (offset, size)) for position, offset, size in
                chat.reader.query(
                    'SELECT position,offset,size FROM messages WHERE chat=? AND position BETWEEN ? AND ? '
                    'ORDER BY position', (chat.name, start, end))
                if positions is None or position in positions)

    def message(chat, offset, size):
        row = chat.reader.row(offset, size)
        return chat.message(*(row.get(field) for field in chat.message._fields))

    def getitem(chat, index):
        if not isinstance(index, (int, slice)):
            return original_getitem(chat, index)
        with chat.reader.reading():
            if isinstance(index, slice):
                positions = range(*index.indices(chat.count))
                if not positions:
                    return []
                offsets = query(chat, min(positions), max(positions), positions)
                return [message(chat, *offsets[position]) for position in positions]
            if index < 0:
                index += chat.count
            if not 0 <= index < chat.count:
                raise IndexError(index)
            current = state(chat.reader)
            name, offsets = current['window']
            hit = name == chat.name and index in offsets
            current['counts'][0 if hit else 1] += 1
            if not hit:
                start = max(0, index - capacity // 2)
                end = min(chat.count - 1, start + capacity - 1)
                offsets = query(chat, start, end)
                current['window'] = (chat.name, offsets)
                windows[chat.reader] = current['window']
            return message(chat, *offsets[index])

    def close(reader):
        result = original_close(reader)
        release(reader)
        return result

    disk_history.Chat.__getitem__ = getitem
    disk_history.BoundedReader.close = close
    disk_history.BoundedReader.query = reader_query
    disk_history.BoundedReader.cached = cached
    if retrievers is not None:
        retrievers.PersonaFewShotRetriever.retrieve = retrieve
    try:
        yield windows
    finally:
        disk_history.Chat.__getitem__ = original_getitem
        disk_history.BoundedReader.close = original_close
        disk_history.BoundedReader.query = original_query
        disk_history.BoundedReader.cached = original_cached
        if retrievers is not None:
            retrievers.PersonaFewShotRetriever.retrieve = original_retrieve
        for reader in list(resources):
            release(reader)
        windows.clear()
        if profile is not None:
            for position, name in enumerate(('history_source_offset_cache_hit', 'history_source_offset_cache_miss')):
                profile.record(name, 0.0, count=sum(counter[position] for counter in counts))
        counts.clear()
