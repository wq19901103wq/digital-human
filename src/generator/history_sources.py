"""Reconstruct historical inputs from raw messages, before rendering or cache lookup.

The imported message archive is the trust boundary. A timestamp declared by a
sample, or a `verified` flag on derived material, is never source evidence.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, namedtuple
from functools import lru_cache
from pathlib import Path
from threading import RLock

from ..config import sha256_file
from ..bootstrap import conversations
from ..bootstrap.ingest import message_identity
from .history import HistoryError

POLICY = 'complete_before_input_v1'
Message = namedtuple('Message', 'sender text is_self timestamp message_id chat_type chat_name source_chat_id event',
                     defaults=[None])
_lock = RLock()


def require(condition, message):
    if not condition:
        raise HistoryError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stamp(path):
    try:
        s = path.stat()
        require(not path.is_symlink(), f'历史来源不能是符号链接：{path.name}')
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    except OSError as exc:
        raise HistoryError(f'历史来源缺失：{path.name}') from exc


class HistorySources:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.files = [self.directory / n for n in
                      ('messages.jsonl', 'purposes.json', 'fewshot_pool.jsonl', 'report.json')]
        self.stamps = [stamp(p) for p in self.files]
        self.hashes = {p.name: sha256_file(p) for p in self.files}
        self.purpose = json.loads(self.files[1].read_text())
        report = json.loads(self.files[3].read_text())
        require(self.purpose.get('history_policy') == report.get('history_policy') == POLICY,
                '历史策略缺失或降级；禁止退回无时间过滤的召回')
        require(report.get('review_status') == 'approved' and
                report.get('examples_sha256') == self.hashes['fewshot_pool.jsonl'], '历史池内容未获核验')
        self.protocol = self.purpose['protocol']
        self.segmentation = self.protocol.get('reply_segmentation')
        if self.segmentation is not None:
            conversations.validate_policy(self.segmentation)
        require(self.protocol['development_start'] < self.protocol['acceptance_start'], '历史窗口顺序非法')
        self.heldout = set(self.protocol.get('unseen_chat_ids', []))
        self.chats = defaultdict(list)
        with self.files[0].open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                m = json.loads(line)
                expected = message_identity(m)
                require(m.get('message_id') == expected, '原始消息 ID 与内容、时间不符')
                rows = self.chats[m['chat_id']]
                require(not rows or rows[-1].timestamp <= m['timestamp'], '原始聊天时间倒序')
                rows.append(Message(*(m.get(k) for k in Message._fields)))
        self.check()

    def check(self):
        require(self.stamps == [stamp(p) for p in self.files], '运行中历史输入发生变化；保留断点并停止')

    def validate(self, row, *, example=False):
        """Check every field that can reach retrieval, prompts or learned vocabulary."""
        self.check()
        span = row.get('source_span', {})
        source = self.chats.get(span.get('chat_id'), [])
        start, middle, end = (span.get(k) for k in ('start', 'reply_start', 'end'))
        require(all(type(i) is int for i in (start, middle, end)) and
                0 <= start < middle < end <= len(source), '历史片段来源区间非法')
        context, reply = source[start:middle], source[middle:end]
        following_self = end < len(source) and source[end].is_self and source[end].text.strip()
        if self.segmentation is not None:
            require(not conversations.issues(context, reply, self.segmentation), '历史片段跨会话或回复时间边界不符')
            following_self = (following_self and conversations.gap(reply[-1], source[end])
                              <= self.segmentation['reply_gap_seconds'])
            if self.protocol.get('max_context'):
                expected_start = conversations.context_start(source, middle,
                    self.protocol['max_context'], self.segmentation)
                require(start == expected_start, '上文不是冻结窗口内最近一次会话的完整片段')
        require(all(m.is_self and m.text.strip() for m in reply) and
                not (source[middle-1].is_self and source[middle-1].text.strip()) and
                not following_self,
                '真人连续气泡不完整或身份不符')
        context_ids, reply_ids = [m.message_id for m in context], [m.message_id for m in reply]
        require(row.get('context_message_ids') == context_ids and row.get('reply_message_ids') == reply_ids,
                '历史片段消息 ID 与原始来源不符')
        require(span == {'chat_id': span['chat_id'], 'start': start, 'reply_start': middle, 'end': end,
                         'start_timestamp': context[0].timestamp, 'end_timestamp': reply[-1].timestamp,
                         'order_verified': False}, '历史片段时间或同秒顺序证明不符')
        require(row.get('input_cutoff') == {'timestamp': context[-1].timestamp, 'index': middle-1,
                                          'order_verified': False}, '输入截止点与原始消息不符')
        cm = [dict(sender=m.sender, text=m.text, is_self=m.is_self, timestamp=m.timestamp) for m in context]
        texts = [m.text for m in reply]
        require(row.get('context_messages' if example else 'context') == cm and
                row.get('reply' if example else 'human_reply') == texts, '历史文本、说话人或答案与来源不符')
        require(row.get('id' if example else 'case_id') == digest({'context': context_ids, 'reply': reply_ids})[:24],
                '样本编号不属于原始片段')
        require(row.get('content_sha256') == digest(dict(context_messages=cm, reply=texts,
                context_message_ids=context_ids, reply_message_ids=reply_ids)), '历史内容指纹不符')
        require(row.get('source_message_id') == f"{span['chat_id']}:{middle}" and
                row.get('source_chat_id') == reply[0].source_chat_id and
                row.get('chat_name') == reply[0].chat_name and
                row.get('relationship' if example else 'chat_type') == reply[0].chat_type,
                '聊天身份或名称来源不符')
        if example:
            require(row.get('context') == [f'{m.sender}: {m.text}' for m in context] and
                    row.get('timestamp') == reply[-1].timestamp and
                    row.get('annotation_scope') == 'example_only', '示例渲染文本或标签来源不符')
            allowed = {'id', 'context', 'context_messages', 'reply', 'reply_shape', 'relationship', 'chat_name',
                       'source_chat_id', 'source_message_id', 'source_provenance', 'context_message_ids',
                       'reply_message_ids', 'source_span', 'input_cutoff', 'timestamp', 'annotation_scope',
                       'chat_id', 'prior_interactions', 'familiarity', 'content_sha256', 'retrieval_features'}
            require(not set(row) - allowed, '历史示例含未核验的替换文本或额外标注')
            require(row.get('chat_id') == reply[0].source_chat_id and
                    row.get('reply_shape') == ('multi' if len(reply) > 1 else 'single'), '示例派生字段不符')
        else:
            require(not any(k in row for k in ('unread', 'scenario', 'context_messages', 'reply_messages')),
                    '历史输入含未经来源校验的提示词覆盖字段')
            expected = sorted(self.heldout) if span['chat_id'] in self.heldout else []
            require(row.get('history_excluded_chat_ids', []) == expected, '留出对象排除条件被修改')
        return row

    def role(self, role):
        from ..iteration.datasets import case_path
        path = case_path(self.directory, role)
        cases = [json.loads(s) for s in path.read_text().split('\n') if s.strip()]
        require(len({c['case_id'] for c in cases}) == len(cases), '角色清单包含重复题目')
        a, b = self.protocol['development_start'], self.protocol['acceptance_start']
        for c in cases:
            self.validate(c)
            span = c['source_span']
            if role in ('judge_training', 'gen_learning', 'gen_optimization'):
                require(span['end_timestamp'] < a and span['chat_id'] not in self.heldout,
                        '学习清单包含未来或留出聊天')
            elif role in ('development', 'judge_development'):
                require(a <= span['start_timestamp'] and span['end_timestamp'] < b, '开发题超出冻结窗口')
            elif role == 'fixed_test':
                require(b <= span['start_timestamp'], '固定题早于冻结窗口')
        self.check()
        return cases


@lru_cache(maxsize=2)
def _load(directory, stamps):
    return HistorySources(Path(directory))


def load(directory):
    directory = Path(directory).resolve()
    with _lock:
        signatures = tuple(stamp(directory / n) for n in
                           ('messages.jsonl', 'purposes.json', 'fewshot_pool.jsonl', 'report.json'))
        return _load(str(directory), signatures)


def prompt_case(case):
    """Explicit generator view: answers and labels are never prompt inputs."""
    return {k: case[k] for k in ('case_id', 'chat_type', 'chat_name', 'source_chat_id', 'context') if k in case}
