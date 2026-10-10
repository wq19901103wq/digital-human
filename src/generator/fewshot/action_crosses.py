"""Opt-in reply-action interactions from independently cached categorical views."""
import json

from ..history_sources import require
from .features import equal
from .schema import CONTEXT, REPLY

VERSION = 'reply_action_crosses_v1'
TRANSFORM = dict(semantic='local_crosses_v2', chat_id_pair=True,
                 identity_crosses='identity_crosses_v1', reply_action_crosses=VERSION)

# Explicit low-cardinality additions; existing v1/v2 crosses remain unchanged.
# Each right-hand field describes the historical example's own reply only.
REPLY_CROSSES = {
    'last_self_action': ('first_bubble_action', 'last_bubble_action', 'own_stance_continuity'),
    'last_self_tone': ('first_bubble_action', 'own_stance_continuity'),
    'self_stance': ('reply_action', 'own_stance_continuity'),
    'needs_empathy': ('first_bubble_action', 'last_bubble_action', 'secondary_action_joke'),
    'pending_intent_comfort': ('first_bubble_action', 'secondary_action_comfort', 'secondary_action_joke'),
    'has_correction_or_disagreement': ('first_bubble_action', 'last_bubble_action'),
}
CONTEXT_MATCHES = ('last_self_action', 'last_self_tone', 'self_stance', 'needs_empathy')


def category(row, prefix, field, schema):
    value = row.get(prefix + '.' + field)
    require(isinstance(value, str) and value in schema[field],
            'Missing or invalid reply-action categorical input')
    return value


def expand(row):
    """Append 19 deterministic categorical crosses; no extraction or label input."""
    result = dict(row)
    for target, replies in REPLY_CROSSES.items():
        left = category(row, 'target', target, CONTEXT)
        for reply in replies:
            right = category(row, 'example_reply', reply, REPLY)
            result[f'action_cross.{target}_x_reply_{reply}'] = json.dumps(
                [left, right], ensure_ascii=False, separators=(',', ':'))
    for field in CONTEXT_MATCHES:
        result[f'action_cross.equal_context_{field}'] = equal(
            category(row, 'target', field, CONTEXT),
            category(row, 'example_context', field, CONTEXT))
    return result
