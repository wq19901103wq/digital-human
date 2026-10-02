"""Bounded disk storage for authenticated historical retrievers, including snapshots."""
from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
from threading import Condition, RLock, local
import time
import uuid


MODE = 'source_offsets_indexed_terms_eligibility_v4'
CAPACITY = 256
FEATURES = ('_latest_situations_by_id', '_latest_profiles_by_id', '_latest_turn_forms_by_id',
            '_group_participant_profiles_by_id', '_terms_by_id')
BUCKETS = ('_rows_by_group_situation', '_rows_by_group_precise_situation',
           '_rows_by_group_latest_situation')
HEADERS = ('id', 'source_span', 'context_message_ids', 'reply_message_ids', 'annotation_scope',
           'relationship', 'chat_id')


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value, *, ordered=False):
    return json.dumps(value, ensure_ascii=False, sort_keys=not ordered, separators=(',', ':'))


def database(source, binding, build):
    """Publish complete checksummed indexes under a cross-process build lock."""
    root = Path(source).parent.parent / '.cache' / 'disk-history'
    root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(encoded(dict(mode=MODE, adapter=checksum(__file__),
                                          binding=binding)).encode()).hexdigest()
    target = root / (identity + '.sqlite3')
    manifest = target.with_suffix('.json')
    with target.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists() or manifest.exists():
            if (not target.is_file() or target.is_symlink() or not manifest.is_file()
                    or manifest.is_symlink()):
                raise ValueError('disk history index publication is incomplete')
            receipt = json.loads(manifest.read_text())
            if receipt != dict(identity=identity, sha256=checksum(target)):
                raise ValueError('disk history index checksum mismatch')
        else:
            pending = target.with_suffix('.' + uuid.uuid4().hex + '.building')
            connection = sqlite3.connect(pending)
            try:
                # 构建期允许断电丢页：.building 仅在显式发布后改名，损坏即整体重建。
                connection.execute('PRAGMA synchronous=OFF')
                connection.execute('PRAGMA cache_size=-65536')
                build(connection)
                connection.commit()
                connection.close()
                receipt = dict(identity=identity, sha256=checksum(pending))
                pending_manifest = pending.with_suffix('.json')
                pending_manifest.write_text(encoded(receipt))
                os.replace(pending, target)
                os.replace(pending_manifest, manifest)
            finally:
                connection.close()
                pending.unlink(missing_ok=True)
                pending.with_suffix('.json').unlink(missing_ok=True)
        connection = sqlite3.connect(target.as_uri() + '?mode=ro', uri=True, check_same_thread=False)
        try:
            connection.execute('PRAGMA cache_size=-2048')
            connection.execute('PRAGMA mmap_size=0')
            connection.execute('PRAGMA query_only=ON')
            if connection.execute('PRAGMA quick_check').fetchone() != ('ok',):
                raise ValueError('disk history index integrity check failed')
        except BaseException:
            connection.close()
            raise
    return connection


class BoundedReader:
    def __init__(self, source, connection):
        self.source = Path(source)
        self.connection = connection
        self.stream = self.source.open('rb')
        self.lock = RLock()
        self.drained = Condition(self.lock)
        self.local = local()
        self.active_reads = 0
        self.closing = False
        self.closed = False
        self.lock_timing = None
        self.cache = OrderedDict()
        self.features = OrderedDict()

    @contextmanager
    def guard(self):
        started = time.monotonic() if self.lock_timing is not None else None
        with self.lock:
            if started is not None:
                seconds = time.monotonic() - started
                self.lock_timing['count'] += 1
                self.lock_timing['total_seconds'] += seconds
                self.lock_timing['max_seconds'] = max(self.lock_timing['max_seconds'], seconds)
            yield

    @contextmanager
    def reading(self):
        nested = getattr(self.local, 'reading', False)
        if not nested:
            with self.guard():
                if self.closing or self.closed:
                    raise ValueError('disk history reader is closed')
                self.active_reads += 1
            self.local.reading = True
        try:
            yield
        finally:
            if not nested:
                self.local.reading = False
                with self.guard():
                    self.active_reads -= 1
                    if self.closing and not self.active_reads:
                        self.drained.notify_all()

    def query(self, statement, parameters=()):
        with self.reading(), self.guard():
            return self.connection.execute(statement, parameters).fetchall()

    def cached(self, cache, key, load):
        with self.reading():
            with self.guard():
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key]
            value = load()
            with self.guard():
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key]
                cache[key] = value
                if len(cache) > CAPACITY:
                    cache.popitem(last=False)
                return value

    def row(self, offset, size):
        return self.cached(self.cache, offset,
                           lambda: json.loads(os.pread(self.stream.fileno(), size, offset)))

    def feature(self, identity):
        def load():
            records = self.query('SELECT features FROM examples WHERE id=? ORDER BY position DESC LIMIT 1',
                                 (identity,))
            if not records:
                raise KeyError(identity)
            return json.loads(records[0][0])
        return self.cached(self.features, identity, load)

    def term_overlaps(self, terms):
        """Read only matching postings, with the frozen Counter overlap formula."""
        overlaps = Counter()
        items = list(terms.items())
        for start in range(0, len(items), CAPACITY):
            part = dict(items[start:start + CAPACITY])
            placeholders = ','.join('?' for _ in part)
            for term, identity, count in self.query(
                    f'SELECT term,id,count FROM terms WHERE term IN ({placeholders})', tuple(part)):
                overlaps[identity] += min(part[term], count)
        return overlaps

    def context_length(self, identity):
        """示例正文拼接长度（冻结 len(\" \".join(context)) 公式），建索引时物化。"""
        lengths = getattr(self, '_context_lengths', None)
        if lengths is None:
            lengths = {}
            for value, length in self.query('SELECT id, context_length FROM examples ORDER BY position'):
                lengths[value] = length
            self._context_lengths = lengths
        return lengths.get(identity, 0)

    def excluded_ids(self, case, excluded=None):
        """Use indexed boundaries/message IDs; an eligible duplicate still permits its ID."""
        from src.generator.history import validate_case
        cutoff = validate_case(case)
        target = tuple(case['reply_message_ids'])
        chats = tuple(case.get('history_excluded_chat_ids', ()))
        target_sql = ','.join('?' for _ in target) or 'NULL'
        time_allowed = ('(end_timestamp < ? OR (? = 1 AND end_timestamp = ? AND order_verified = 1 '
                        'AND chat = ? AND end_index <= ?))')
        ordered = cutoff.get('order_verified') is True
        time_params = (cutoff['timestamp'], int(ordered), cutoff['timestamp'],
                       case['source_span']['chat_id'], cutoff['index'] + 1 if ordered else 0)
        chat_condition = f'chat IN ({",".join("?" for _ in chats)})' if chats else '0'
        conflict = (f'position IN (SELECT position FROM example_messages '
                    f'WHERE message_id IN ({target_sql}))') if target else '0'
        # 直接计算允许集（时间索引范围扫描 + 少量冲突/聊天排除），
        # 被禁集 = 全池 id - 允许集；避免对 blocked 集合做相关子查询逐行回表。
        allowed = (f'SELECT id FROM examples WHERE {time_allowed} AND NOT ({chat_condition}) '
                   f'AND NOT ({conflict})')
        statement = f'SELECT DISTINCT id FROM examples WHERE id NOT IN ({allowed})'
        parameters = (*time_params, *chats, *target)
        return set(excluded or ()) | {identity for identity, in self.query(statement, parameters)}

    def close(self):
        if getattr(self.local, 'reading', False):
            raise RuntimeError('cannot close disk history from an active read')
        with self.guard():
            if self.closed:
                return
            self.closing = True
            self.drained.wait_for(lambda: not self.active_reads)
            if self.closed:
                return
            self.stream.close()
            self.connection.close()
            self.cache.clear()
            self.features.clear()
            self.closed = True


class Chat(Sequence):
    def __init__(self, reader, name, count, message):
        self.reader, self.name, self.count, self.message = reader, name, count, message

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            positions = range(*index.indices(self.count))
            if not positions:
                return []
            with self.reader.reading():
                records = self.reader.query(
                    'SELECT position,offset,size FROM messages WHERE chat=? AND position BETWEEN ? AND ? '
                    'ORDER BY position', (self.name, min(positions), max(positions)))
                messages = {}
                for position, offset, size in records:
                    if position in positions:
                        row = self.reader.row(offset, size)
                        messages[position] = self.message(*(row.get(key) for key in self.message._fields))
                return [messages[position] for position in positions]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        with self.reader.reading():
            offset, size = self.reader.query(
                'SELECT offset,size FROM messages WHERE chat=? AND position=?', (self.name, index))[0]
            row = self.reader.row(offset, size)
            return self.message(*(row.get(key) for key in self.message._fields))


def load_sources(self, directory, module):
    self.directory = Path(directory)
    self.files = [self.directory / name for name in
                  ('messages.jsonl', 'purposes.json', 'fewshot_pool.jsonl', 'report.json')]
    self.stamps = [module.stamp(source) for source in self.files]
    self.hashes = {source.name: module.sha256_file(source) for source in self.files}
    self.purpose = json.loads(self.files[1].read_text())
    report = json.loads(self.files[3].read_text())
    module.require(self.purpose.get('history_policy') == report.get('history_policy') == module.POLICY,
                   '历史策略缺失或降级；禁止退回无时间过滤的召回')
    module.require(report.get('review_status') == 'approved' and
                   report.get('examples_sha256') == self.hashes['fewshot_pool.jsonl'], '历史池内容未获核验')
    self.protocol = self.purpose['protocol']
    self.segmentation = self.protocol.get('reply_segmentation')
    if self.segmentation is not None:
        module.conversations.validate_policy(self.segmentation)
    module.require(self.protocol['development_start'] < self.protocol['acceptance_start'], '历史窗口顺序非法')
    self.heldout = set(self.protocol.get('unseen_chat_ids', []))

    def build(connection):
        connection.execute('CREATE TABLE messages(chat TEXT, position INTEGER, offset INTEGER, size INTEGER, '
                           'PRIMARY KEY(chat,position)) WITHOUT ROWID')
        counts, latest = {}, {}
        with self.files[0].open('rb') as stream:
            offset = 0
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    module.require(row.get('message_id') == module.message_identity(row), '原始消息 ID 与内容、时间不符')
                    chat = row['chat_id']
                    module.require(chat not in latest or latest[chat] <= row['timestamp'], '原始聊天时间倒序')
                    position = counts.get(chat, 0)
                    connection.execute('INSERT INTO messages VALUES(?,?,?,?)', (chat, position, offset, len(line)))
                    counts[chat], latest[chat] = position + 1, row['timestamp']
                offset += len(line)
        self.check()

    connection = database(self.files[0], dict(kind='messages', hashes=self.hashes,
                          source_code=checksum(module.__file__)), build)
    reader = None
    try:
        reader = BoundedReader(self.files[0], connection)
        self._disk_reader = reader
        self.chats = {name: Chat(reader, name, count, module.Message) for name, count in
                      connection.execute('SELECT chat,count(*) FROM messages GROUP BY chat')}
        self.check()
    except BaseException:
        reader.close() if reader is not None else connection.close()
        raise


class Row(Mapping):
    __slots__ = ('reader', 'offset', 'size', 'header')

    def __init__(self, reader, offset, size, header):
        self.reader, self.offset, self.size, self.header = reader, offset, size, header

    def __getitem__(self, key):
        if key in self.header:
            return self.header[key]
        return self.reader.row(self.offset, self.size)[key]

    def __iter__(self):
        return iter(self.reader.row(self.offset, self.size))

    def __len__(self):
        return len(self.reader.row(self.offset, self.size))


class FeatureMap(Mapping):
    def __init__(self, reader, identities, name):
        self.reader, self.identities, self.name = reader, identities, name

    def __getitem__(self, identity):
        value = self.reader.feature(identity)[self.name]
        if self.name == '_terms_by_id':
            return Counter(value)
        if self.name == '_latest_situations_by_id':
            return set(value)
        if self.name == '_group_participant_profiles_by_id':
            return tuple(value)
        return value.copy() if isinstance(value, dict) else value

    def __iter__(self):
        return iter(self.identities)

    def __len__(self):
        return len(self.identities)


class ChunkPath:
    def __init__(self, path, lines):
        self.path, self.lines = path, lines

    def stat(self):
        return self.path.stat()

    def read_text(self, **kwargs):
        return b''.join(self.lines).decode('utf-8')


def load_examples(self, original, module):
    if self._history_sources is None or self.render_path:
        return original(self)
    self._history_sources.check()
    if self.path.stat().st_mtime_ns == self._mtime_ns:
        return self._rows
    flags = {key: value for key, value in vars(self).items() if type(value) is bool}
    binding = dict(kind='examples', hashes=self._history_sources.hashes, flags=flags,
                   source_code=checksum(module.__file__))

    def build(connection):
        connection.execute('CREATE TABLE examples(position INTEGER PRIMARY KEY, id TEXT, offset INTEGER, '
                           'size INTEGER, header TEXT, buckets TEXT, features TEXT, '
                           'chat TEXT, end_timestamp REAL, end_index INTEGER, order_verified INTEGER, '
                           'latest_situations TEXT, latest_profile TEXT, latest_turn_form TEXT, '
                           'group_participant_profile TEXT, context_length INTEGER)')
        connection.execute('CREATE TABLE example_messages(message_id TEXT, position INTEGER, '
                           'PRIMARY KEY(message_id,position)) WITHOUT ROWID')
        connection.execute('CREATE TABLE terms(term TEXT, id TEXT, count INTEGER, '
                           'PRIMARY KEY(term,id)) WITHOUT ROWID')
        connection.execute('CREATE INDEX term_ids ON terms(id)')
        position = 0
        def chunk(lines, offsets):
            nonlocal position
            temporary = copy.copy(self)
            temporary.path, temporary._mtime_ns = ChunkPath(self.path, lines), -1
            rows = original(temporary)
            if temporary._mtime_ns != self.path.stat().st_mtime_ns:
                raise ValueError('frozen historical loader did not complete')
            locations = iter((json.loads(line), location) for line, location in zip(lines, offsets))
            for row in rows:
                identity = str(row['id'])
                values = {name: getattr(temporary, name)[identity] for name in FEATURES}
                values['_latest_situations_by_id'] = sorted(values['_latest_situations_by_id'])
                buckets = [[key[1] for key, members in getattr(temporary, name).items()
                            if any(member is row for member in members)] for name in BUCKETS]
                for parsed, location in locations:
                    if parsed == row:
                        offset, size = location
                        break
                else:
                    raise ValueError('frozen loader changed a historical example')
                span = row['source_span']
                connection.execute('INSERT INTO examples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (position, identity, offset, size, encoded({key: row[key] for key in HEADERS if key in row}),
                     encoded(buckets), encoded(values, ordered=True), span['chat_id'],
                     span['end_timestamp'], span['end'], int(span.get('order_verified') is True),
                     json.dumps(sorted(values['_latest_situations_by_id']), ensure_ascii=False),
                     json.dumps(values['_latest_profiles_by_id'], ensure_ascii=False, sort_keys=True),
                     values['_latest_turn_forms_by_id'],
                     json.dumps(values['_group_participant_profiles_by_id'], ensure_ascii=False),
                     len(' '.join(row['context']))))
                connection.executemany('INSERT OR IGNORE INTO example_messages VALUES(?,?)',
                    ((message, position) for message in row['context_message_ids'] + row['reply_message_ids']))
                connection.execute('DELETE FROM terms WHERE id=?', (identity,))
                connection.executemany('INSERT INTO terms VALUES(?,?,?)',
                    ((term, identity, count) for term, count in values['_terms_by_id'].items()))
                position += 1
        with self.path.open('rb') as stream:
            lines, offsets, offset = [], [], 0
            for line in stream:
                json.loads(line)
                lines.append(line)
                offsets.append((offset, len(line)))
                offset += len(line)
                if len(lines) == CAPACITY:
                    chunk(lines, offsets)
                    lines, offsets = [], []
            if lines:
                chunk(lines, offsets)
        connection.execute('CREATE INDEX example_ids ON examples(id,position)')
        connection.execute('CREATE INDEX example_times ON examples(end_timestamp)')
        connection.execute('CREATE INDEX example_chats ON examples(chat)')
        self._history_sources.check()

    connection = database(self.path, binding, build)
    reader = None
    try:
        reader = BoundedReader(self.path, connection)
        self._disk_reader = reader
        self._rows = []
        self._rows_by_group = {False: [], True: []}
        for name in BUCKETS:
            setattr(self, name, {})
        for offset, size, header, buckets in connection.execute(
                'SELECT offset,size,header,buckets FROM examples ORDER BY position'):
            row = Row(reader, offset, size, json.loads(header))
            self._rows.append(row)
            group = row.get('relationship') == 'group'
            self._rows_by_group[group].append(row)
            for name, keys in zip(BUCKETS, json.loads(buckets)):
                for key in keys:
                    getattr(self, name).setdefault((group, key), []).append(row)
        identities = dict.fromkeys(str(row['id']) for row in self._rows)
        small = ('_latest_situations_by_id', '_latest_profiles_by_id', '_latest_turn_forms_by_id',
                 '_group_participant_profiles_by_id')
        for name in small:
            setattr(self, name, {})
        for (identity, situations, profile, turn_form, participants) in connection.execute(
                'SELECT id, latest_situations, latest_profile, latest_turn_form, '
                'group_participant_profile FROM examples ORDER BY position'):
            self._latest_situations_by_id[identity] = set(json.loads(situations))
            self._latest_profiles_by_id[identity] = json.loads(profile)
            self._latest_turn_forms_by_id[identity] = turn_form
            self._group_participant_profiles_by_id[identity] = tuple(json.loads(participants))
        self._terms_by_id = FeatureMap(reader, identities, '_terms_by_id')
        self._mtime_ns = self.path.stat().st_mtime_ns
        self._history_sources.check()
    except BaseException:
        reader.close() if reader is not None else connection.close()
        raise
    return self._rows


@contextmanager
def disk_storage(module):
    """Reuse certified storage and exact indexed operations, keeping frozen ranking."""
    path = Path(__file__).resolve().with_name('indexed_retrieval.py')
    spec = importlib.util.spec_from_file_location('_digital_human_indexed_retrieval', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    from src.generator import history_sources
    original_sources = history_sources.HistorySources.__init__
    original_load = module.PersonaFewShotRetriever._load
    original_retrieve = module.PersonaFewShotRetriever.retrieve
    original_approved = module.PersonaFewShotRetriever.is_approved
    indexed = adapter.indexed_retrieve(original_retrieve)
    readers = []
    def sources(self, directory):
        load_sources(self, directory, history_sources)
        readers.append(self._disk_reader)
    def examples(self):
        rows = load_examples(self, original_load, module)
        reader = getattr(self, '_disk_reader', None)
        if reader is not None and reader not in readers:
            readers.append(reader)
        return rows
    def retrieve(self, *args, **kwargs):
        self._load()
        method = indexed if getattr(self, '_disk_reader', None) is not None else original_retrieve
        return [dict(row) for row in method(self, *args, **kwargs)]
    def approved(self):
        if self._history_sources is not None:
            self._history_sources.check()
        try:
            report = json.loads(self.path.with_name('report.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        if self.render_path or not report.get('history_policy'):
            return original_approved(self)
        if report.get('review_status') != 'approved' or not report.get('examples_sha256'):
            return False
        self._history_sources = history_sources.load(self.path.parent)
        self.history_policy = self._history_sources.purpose['history_policy']
        if report['examples_sha256'] != self._history_sources.hashes['fewshot_pool.jsonl']:
            return False
        return bool(self._load())
    history_sources._load.cache_clear()
    try:
        history_sources.HistorySources.__init__ = sources
        module.PersonaFewShotRetriever._load = examples
        module.PersonaFewShotRetriever.retrieve = retrieve
        module.PersonaFewShotRetriever.is_approved = approved
        yield
    finally:
        history_sources.HistorySources.__init__ = original_sources
        module.PersonaFewShotRetriever._load = original_load
        module.PersonaFewShotRetriever.retrieve = original_retrieve
        module.PersonaFewShotRetriever.is_approved = original_approved
        history_sources._load.cache_clear()
        for reader in readers:
            reader.close()
