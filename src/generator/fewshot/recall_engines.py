"""fewshot.recall_engines：逐题资格排除与词项特征的引擎协议与双实现。

- InMemoryRecallEngine：旧内存路径语义（filter_rows 资格并集 + 显式排除）；
  词项特征返回 None，由 rank 主循环走 Counter 路径（旧 749-752 行语义）。
- SqliteIndexedRecallEngine：直接委托 disk_history.BoundedReader 的
  excluded_ids / term_overlaps / context_length（不复制 SQL），
  与现有 AST 补丁路径（indexed_retrieval.py）的替换语义一致。
- open_index_engine：按设计 §4 索引矩阵打开/构建磁盘索引；disk_history.py
  只读使用（database() 的发布/校验/损坏拒绝语义原样沿用，不做 try 回退）。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
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
    """磁盘索引引擎：直接委托 disk_history.BoundedReader（只读使用，不复制 SQL）。"""

    def __init__(self, reader):
        self._reader = reader

    def excluded_ids(self, pool, q) -> set[str]:
        return self._reader.excluded_ids(q.history_case, q.exclude_ids)

    def term_features(self, pool, query_terms: Counter) -> Mapping[str, tuple[int, int]]:
        return _IndexedTermFeatures(self._reader, self._reader.term_overlaps(query_terms))

    def close(self) -> None:
        self._reader.close()


# -- 磁盘索引打开/构建（设计 §4 矩阵；database() 的抛错即拒绝，fail-closed）----

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
    locations = iter((json.loads(line), location) for line, location in _line_locations(pool))
    for row in pool.rows:
        identity = str(row['id'])
        values = {name: getattr(pool, name)[identity] for name in disk_history.FEATURES}
        values['_latest_situations_by_id'] = sorted(values['_latest_situations_by_id'])
        buckets = [[key[1] for key, members in getattr(pool, name).items()
                    if any(member is row for member in members)] for name in disk_history.BUCKETS]
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


def open_index_engine(pool, policy: str) -> SqliteIndexedRecallEngine | None:
    """打开（或按 policy 构建）磁盘索引引擎；干净缺失 + use_if_present 返回 None。

    矩阵（设计 §4）：发布不完整/校验和不符/SQLite 损坏/来源核验失败一律由
    disk_history.database() 抛错拒绝，这里不做任何 try 回退。
    """
    if policy not in ('build', 'use_if_present'):
        raise ValueError(f'未知索引策略: {policy!r}')
    from .. import disk_history

    source = Path(pool.path)
    package_dir = Path(__file__).resolve().parent
    source_code = hashlib.sha256(
        ''.join(disk_history.checksum(path) for path in sorted(package_dir.glob('*.py'))).encode()
    ).hexdigest()
    binding = dict(kind='examples', hashes=dict(pool.history_sources.hashes), flags=dict(pool.flags),
                   source_code=source_code)
    root = source.parent.parent / '.cache' / 'disk-history'
    identity = hashlib.sha256(disk_history.encoded(dict(
        mode=disk_history.MODE, adapter=disk_history.checksum(disk_history.__file__),
        binding=binding)).encode()).hexdigest()
    target = root / (identity + '.sqlite3')
    manifest = target.with_suffix('.json')
    if policy == 'use_if_present' and not target.exists() and not manifest.exists():
        return None

    def build(connection):
        pool.check()
        _build_examples(connection, pool)
        pool.check()

    connection = disk_history.database(source, binding, build)
    reader = disk_history.BoundedReader(source, connection)
    return SqliteIndexedRecallEngine(reader)
