"""fewshot.recall_engines：逐题资格排除与词项特征的引擎协议与双实现。

- InMemoryRecallEngine：旧内存路径语义（filter_rows 资格并集 + 显式排除）；
  词项特征返回 None，由 rank 主循环走 Counter 路径（旧 749-752 行语义）。
- SqliteIndexedRecallEngine：直接委托 disk_history.BoundedReader 的
  excluded_ids / term_overlaps / context_length（不复制 SQL），
  与现有 AST 补丁路径（indexed_retrieval.py）的替换语义一致。
- prepare_engine / build_index_engine：按设计 §4 索引矩阵分三态（I9）：
  绑定有效 → 单遍水合 PreparedPool 并挂索引引擎；干净缺失 → None（由
  open_session 按 index_policy 决定内存装载或构建）；发布不完整/损坏/来源
  核验失败 → database() 抛错拒绝（fail-closed，无 try 回退）。
  disk_history.py 只读使用。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path
import sys
from threading import RLock
from typing import Mapping, Protocol


class RecallEngine(Protocol):
    """逐题引擎：资格+显式排除集与（可选的）逐题词项数值特征。"""

    def excluded_ids(self, pool, q) -> set[str]:
        """资格判定 + 显式排除后的最终 exclude_ids（旧 retrieve 遮蔽语义）。"""
        ...

    def term_features(self, pool, query_terms: Counter) -> Mapping[str, tuple[int, int]] | None:
        """id -> (overlap, context_length)；内存引擎返回 None（Counter 路径）。"""
        ...


class InMemoryRecallEngine:
    """旧内存路径：filter_rows(self._rows, history_case) 语义原样搬迁。"""

    def excluded_ids(self, pool, q) -> set[str]:
        if not pool.history_policy:
            return set(q.exclude_ids or ())
        from ..history import filter_rows
        allowed = {str(row['id']) for row in filter_rows(pool.rows, q.history_case)}
        return set(q.exclude_ids or ()) | {str(row['id']) for row in pool.rows
                                           if str(row['id']) not in allowed}

    def term_features(self, pool, query_terms: Counter) -> None:
        return None

    def close(self) -> None:
        pass


class _IndexedTermFeatures(Mapping):
    """惰性 {id: (overlap, context_length)}：与补丁的 dict.get + context_length 语义一致。

    overlaps 来自 postings 查询（每题一次）；context_length 经 reader 内部缓存，
    未知 id 返回 0（与 BoundedReader.context_length 的默认值一致）。
    """

    def __init__(self, reader, overlaps):
        self._reader = reader
        self._overlaps = overlaps

    def __getitem__(self, identity: str) -> tuple[int, int]:
        return (self._overlaps.get(identity, 0), self._reader.context_length(identity))

    def __iter__(self):
        return iter(self._overlaps)

    def __len__(self) -> int:
        return len(self._overlaps)

    def get(self, identity: str, default=None):
        return (self._overlaps.get(identity, 0), self._reader.context_length(identity))


class SqliteIndexedRecallEngine:
    """委托只读索引；同一历史边界的资格结果按容量和字节预算复用。"""

    def __init__(self, reader, *, eligibility_cache_entries=8, eligibility_cache_bytes=4 * 1024 * 1024):
        if any(not isinstance(value, int) or value < 0
               for value in (eligibility_cache_entries, eligibility_cache_bytes)):
            raise ValueError('eligibility cache limits must be nonnegative integers')
        self._reader = reader
        self._eligibility_cache = OrderedDict()
        self._eligibility_lock = RLock()
        self._eligibility_cache_entries = eligibility_cache_entries
        self._eligibility_cache_limit = eligibility_cache_bytes
        self._eligibility_cache_bytes = 0

    @staticmethod
    def _eligibility_key(case):
        from ..history import validate_case

        cutoff = validate_case(case)
        ordered = cutoff.get('order_verified') is True
        return (cutoff['timestamp'], ordered, cutoff['index'] + 1 if ordered else 0,
                case['source_span']['chat_id'],
                tuple(case.get('history_excluded_chat_ids', ())), tuple(case['reply_message_ids']))

    def _remember_eligibility(self, key, excluded, minimum_identity_length=0):
        key_size = sys.getsizeof(key)
        for item in key:
            key_size += sys.getsizeof(item)
            if isinstance(item, tuple):
                for value in item:
                    key_size += sys.getsizeof(value)
                    if key_size > self._eligibility_cache_limit:
                        return
            if key_size > self._eligibility_cache_limit:
                return
        identity_count = len(excluded)
        tuple_size = sys.getsizeof(()) + identity_count * (sys.getsizeof((None,)) - sys.getsizeof(()))
        minimum_size = key_size + tuple_size + identity_count * (sys.getsizeof('') + minimum_identity_length)
        identities, size = None, key_size
        if minimum_size <= self._eligibility_cache_limit:
            measured_size = key_size + tuple_size
            for identity in excluded:
                measured_size += sys.getsizeof(identity)
                if measured_size > self._eligibility_cache_limit:
                    break
            else:
                identities, size = tuple(excluded), measured_size
        with self._eligibility_lock:
            if key in self._eligibility_cache:
                self._eligibility_cache.move_to_end(key)
                return
            while self._eligibility_cache and (
                    len(self._eligibility_cache) >= self._eligibility_cache_entries
                    or self._eligibility_cache_bytes + size > self._eligibility_cache_limit):
                _, (_, removed_size) = self._eligibility_cache.popitem(last=False)
                self._eligibility_cache_bytes -= removed_size
            self._eligibility_cache[key] = (identities, size)
            self._eligibility_cache_bytes += size

    def excluded_ids(self, pool, q) -> set[str]:
        if not self._eligibility_cache_entries or not self._eligibility_cache_limit:
            return self._reader.excluded_ids(q.history_case, q.exclude_ids)
        with self._reader.reading():
            key = self._eligibility_key(q.history_case)
            with self._eligibility_lock:
                cached = self._eligibility_cache.get(key)
                if cached is not None:
                    self._eligibility_cache.move_to_end(key)
            if cached is None:
                excluded = self._reader.excluded_ids(q.history_case)
                self._remember_eligibility(key, excluded, getattr(pool, '_minimum_identity_length', 0))
            elif cached[0] is None:
                excluded = self._reader.excluded_ids(q.history_case)
            else:
                excluded = set(cached[0])
            excluded.update(q.exclude_ids or ())
            return excluded

    def term_features(self, pool, query_terms: Counter) -> Mapping[str, tuple[int, int]]:
        return _IndexedTermFeatures(self._reader, self._reader.term_overlaps(query_terms))

    def close(self) -> None:
        self._reader.close()
        with self._eligibility_lock:
            self._eligibility_cache.clear()
            self._eligibility_cache_bytes = 0


def _buckets_by_row(pool):
    """一次反向映射桶成员，保留行对象身份、桶顺序与重复 ID 语义。"""
    from .. import disk_history

    result = {}
    for position, name in enumerate(disk_history.BUCKETS):
        for key, members in getattr(pool, name).items():
            seen = set()
            for member in members:
                identity = id(member)
                if identity not in seen:
                    seen.add(identity)
                    buckets = result.setdefault(identity, [[] for _ in disk_history.BUCKETS])
                    buckets[position].append(key[1])
    return result


def _build_examples(connection, pool) -> None:
    """从 PreparedPool 物化状态构建 examples/terms 表（disk_history.load_examples 的
    build 等价搬迁：行已通过 PreparedPool.load 的同源校验与特征物化）。"""
    from .. import disk_history

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
    from .prepare import thaw_value

    buckets_by_row = _buckets_by_row(pool)
    locations = iter((json.loads(line), location) for line, location in _line_locations(pool))
    for row in pool.rows:
        identity = str(row['id'])
        # I10 深只读后特征值为只读映射/冻结集合视图，编码前递归解冻为普通容器。
        values = thaw_value({name: getattr(pool, name)[identity] for name in disk_history.FEATURES})
        values['_latest_situations_by_id'] = sorted(values['_latest_situations_by_id'])
        buckets = buckets_by_row.get(id(row), [[] for _ in disk_history.BUCKETS])
        for parsed, location in locations:
            if parsed == row:
                offset, size = location
                break
        else:
            raise ValueError('frozen loader changed a historical example')
        span = row['source_span']
        connection.execute('INSERT INTO examples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (position, identity, offset, size, disk_history.encoded({key: row[key] for key in disk_history.HEADERS if key in row}),
             disk_history.encoded(buckets), disk_history.encoded(values, ordered=True), span['chat_id'],
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
    connection.execute('CREATE INDEX example_ids ON examples(id,position)')
    connection.execute('CREATE INDEX example_times ON examples(end_timestamp)')
    connection.execute('CREATE INDEX example_chats ON examples(chat)')


def _line_locations(pool):
    """池文件每行的 (offset, size)（build 期一次顺序扫描）。"""
    with pool.path.open('rb') as stream:
        offset = 0
        for line in stream:
            yield line, (offset, len(line))
            offset += len(line)


def index_binding(pool) -> dict:
    """索引绑定（与旧 open_index_engine 逐字段一致：代码摘要/flags/来源哈希）。"""
    from .. import disk_history

    package_dir = Path(__file__).resolve().parent
    source_code = hashlib.sha256(
        ''.join(disk_history.checksum(path) for path in sorted(package_dir.glob('*.py'))).encode()
    ).hexdigest()
    return dict(kind='examples', hashes=dict(pool.history_sources.hashes), flags=dict(pool.flags),
                source_code=source_code)


def index_target(source: Path, binding: dict) -> tuple[Path, Path]:
    """索引 sqlite/manifest 路径（身份公式与 disk_history.database 相同）。"""
    from .. import disk_history

    root = Path(source).parent.parent / '.cache' / 'disk-history'
    identity = hashlib.sha256(disk_history.encoded(dict(
        mode=disk_history.MODE, adapter=disk_history.checksum(disk_history.__file__),
        binding=binding)).encode()).hexdigest()
    target = root / (identity + '.sqlite3')
    return target, target.with_suffix('.json')


def prepare_engine(pool, policy: str) -> SqliteIndexedRecallEngine | None:
    """探测索引绑定并（有效时）单遍水合 PreparedPool（I9）。

    - 索引存在（sqlite/manifest 任一）→ 交 database() 校验并打开：发布不完整/
      校验和不符/SQLite 损坏一律抛错拒绝，fail-closed 不回退；打开后从 sqlite
      单遍恢复物化状态（来源验证由绑定内容哈希保证，恢复后仍调 pool 的
      check() 做 stamp 轻量检查）。
    - 干净缺失 → 返回 None，由 open_session 按 index_policy 决定内存装载
      （use_if_present）或内存装载后构建（build）。
    """
    if policy not in ('build', 'use_if_present'):
        raise ValueError(f'未知索引策略: {policy!r}')
    from .. import disk_history

    source = Path(pool.path)
    binding = index_binding(pool)
    target, manifest = index_target(source, binding)
    if not target.exists() and not manifest.exists():
        return None

    def build(connection):
        # 探测与打开之间索引被并发删除时才可能到达；构建需要内存物化状态，
        # 此处池未装载，禁止产出半成品索引。
        raise ValueError('disk history index disappeared between probe and open')

    connection = disk_history.database(source, binding, build)
    reader = disk_history.BoundedReader(source, connection)
    try:
        pool.hydrate_from_index(reader)
    except BaseException:
        reader.close()
        raise
    return SqliteIndexedRecallEngine(reader)


def build_index_engine(pool) -> SqliteIndexedRecallEngine:
    """内存装载完成后构建（或复用）磁盘索引并挂索引引擎（policy=build 路径）。

    构建期来源核验失败由 database() 的 build 抛错，发布不完整/损坏拒绝语义
    与矩阵一致，不做任何 try 回退。
    """
    from .. import disk_history

    source = Path(pool.path)
    binding = index_binding(pool)

    def build(connection):
        pool.check()
        _build_examples(connection, pool)
        pool.check()

    connection = disk_history.database(source, binding, build)
    reader = disk_history.BoundedReader(source, connection)
    return SqliteIndexedRecallEngine(reader)
