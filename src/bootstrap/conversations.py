"""Frozen, deterministic conversation boundaries shared by builders and readers."""
from __future__ import annotations

import math

from ..config import ConfigError


def policy(*, context_gap_seconds=7200, response_gap_seconds=600, reply_gap_seconds=120):
    value = dict(schema=1, context_gap_seconds=context_gap_seconds,
                 response_gap_seconds=response_gap_seconds, reply_gap_seconds=reply_gap_seconds)
    validate_policy(value)
    return value


def validate_policy(value):
    if (set(value) != {'schema', 'context_gap_seconds', 'response_gap_seconds', 'reply_gap_seconds'}
            or value['schema'] != 1
            or any(type(value[k]) is not int or value[k] <= 0 for k in value if k != 'schema')):
        raise ConfigError('会话切分策略缺失、版本未知或间隔不是正整数')
    return value


def field(message, name):
    return message[name] if isinstance(message, dict) else getattr(message, name)


def timestamp(message):
    value = field(message, 'timestamp')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ConfigError('消息时间戳缺失或无效，不能判断会话边界')
    return value


def gap(a, b):
    return timestamp(b) - timestamp(a)


def event(message):
    return (message.get('event') if isinstance(message, dict) else getattr(message, 'event', None)) or {}


def context_start(messages, middle, max_context, rule):
    start = max(0, middle - max_context)
    for offset in range(middle - 1, start - 1, -1):
        if event(messages[offset]).get('context_boundary'):
            return offset + 1
        if offset > start and gap(messages[offset - 1], messages[offset]) > rule['context_gap_seconds']:
            return offset
    return start


def issues(context, reply, rule):
    """Return aggregate-safe reason codes; never emit message text or answers."""
    found = set()
    if not context or not reply:
        return {'empty_fragment'}
    all_gaps = [gap(a, b) for a, b in zip([*context, *reply], [*context, *reply][1:])]
    if any(g < 0 for g in all_gaps):
        found.add('reversed_time')
    if field(context[-1], 'is_self'):
        found.add('reply_to_self')
    if any(not event(m).get('reply_eligible', True) for m in reply):
        found.add('non_text_reply')
    if any(event(m).get('context_boundary') for m in context):
        found.add('context_contains_boundary')
    if gap(context[-1], reply[0]) > rule['response_gap_seconds']:
        found.add('stale_response')
    if any(gap(a, b) > rule['context_gap_seconds'] for a, b in zip(context, context[1:])):
        found.add('context_crosses_session')
    if any(gap(a, b) > rule['reply_gap_seconds'] for a, b in zip(reply, reply[1:])):
        found.add('reply_crosses_burst')
    return found
