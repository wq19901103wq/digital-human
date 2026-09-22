"""Eligibility of complete historical examples at an input boundary."""
from __future__ import annotations

from ..config import ConfigError


class HistoryError(ConfigError):
    pass


def validate_case(case: dict) -> dict:
    cutoff = case.get('input_cutoff')
    span = case.get('source_span', {})
    if (not isinstance(cutoff, dict) or not isinstance(cutoff.get('timestamp'), (int, float))
            or not span.get('chat_id') or not case.get('reply_message_ids')
            or not case.get('context_message_ids')):
        raise HistoryError('时间回放缺少输入截止点或完整消息来源')
    return cutoff


def eligible(row: dict, case: dict) -> bool:
    cutoff = validate_case(case)
    span = row.get('source_span', {})
    if (not span.get('chat_id') or not isinstance(span.get('end_timestamp'), (int, float))
            or not row.get('reply_message_ids') or not row.get('context_message_ids')
            or row.get('annotation_scope') != 'example_only'):
        raise HistoryError('示例时间、完整来源或标签来源未核验')
    if span['chat_id'] in case.get('history_excluded_chat_ids', []):
        return False
    target_ids = set(case['reply_message_ids'])
    if target_ids.intersection(row['context_message_ids'] + row['reply_message_ids']):
        return False
    end = span['end_timestamp']
    if end < cutoff['timestamp']:
        return True
    # Same-second ordering is accepted only when its original source is certified.
    return bool(end == cutoff['timestamp'] and cutoff.get('order_verified') is True
                and span.get('order_verified') is True
                and span['chat_id'] == case['source_span']['chat_id']
                and span['end'] <= cutoff['index'] + 1)


def filter_rows(rows: list[dict], case: dict) -> list[dict]:
    validate_case(case)
    return [row for row in rows if eligible(row, case)]
