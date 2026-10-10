"""Local identity interactions, built from independently cached categorical views."""
from collections import Counter
import json

from ..history_sources import require
from .crosses import expand as semantic_crosses
from .schema import UNKNOWN

VERSION = 'identity_crosses_v1'
IDENTITIES = ('chat_id', 'counterpart_id', 'latest_speaker_id', 'explicit_addressee_id')
REPLY_STYLE = ('reply_action', 'opening_style', 'first_bubble_action', 'reply_tone',
    'humor_style', 'has_address', 'address_type', 'formality', 'directness',
    'bubble_count_bin', 'total_length_bin', 'response_structure')
CONTEXT_STYLE = ('scene', 'primary_intent', 'topic_domain', 'environment', 'dialogue_stage')


def pair(row, left, right):
    values = [row.get(left), row.get(right)]
    require(all(isinstance(v, str) for v in values), 'Missing categorical identity input')
    return UNKNOWN if UNKNOWN in values else json.dumps(values, ensure_ascii=False, separators=(',', ':'))


def expand(row, *, style=False):
    result = semantic_crosses(row)
    for key in IDENTITIES:
        result['id_cross.' + key] = pair(row, 'target.' + key, 'example_context.' + key)
    if style:
        for identity in ('chat_id', 'latest_speaker_id'):
            for field in REPLY_STYLE:
                result[f'id_style.{identity}_x_reply_{field}'] = pair(
                    row, 'target.' + identity, 'example_reply.' + field)
            for field in CONTEXT_STYLE:
                result[f'id_style.{identity}_x_context_{field}'] = pair(
                    row, 'target.' + identity, 'example_context.' + field)
                result[f'id_style.{field}_x_example_{identity}'] = pair(
                    row, 'target.' + field, 'example_context.' + identity)
    return result


def coverage(groups, observations, rows):
    """Report cold IDs/pairs without fitting to validation or publishing identities."""
    fit = [i for g in groups if g['split'] == 'fit' for i in g['indices']]
    supervised = [i for g in groups if g['split'] == 'fit' and
        {observations[j]['z'] for j in g['indices']} == {0, 1} for i in g['indices']]
    validation = [i for g in groups if g['split'] == 'validation' for i in g['indices']]
    fields = ('target.chat_id', 'example_context.chat_id', 'id_cross.chat_id')
    result = {}
    for key in fields:
        counts = Counter(rows[i][key] for i in fit if rows[i][key] != UNKNOWN)
        learned = {rows[i][key] for i in supervised if rows[i][key] != UNKNOWN}
        result[key] = dict(fit_unique=len(counts), validation_rows=len(validation),
            validation_known_in_fit=sum(rows[i][key] in counts for i in validation),
            validation_seen_in_mixed_fit=sum(rows[i][key] in learned for i in validation),
            validation_unknown=sum(rows[i][key] == UNKNOWN for i in validation),
            fit_singleton_values=sum(count == 1 for count in counts.values()))
    by_context = [g for g in groups if g['split'] == 'validation']
    fit_chats = {rows[i]['target.chat_id'] for i in fit} - {UNKNOWN}
    result['validation_contexts'] = dict(total=len(by_context), familiar_target_chat=sum(
        rows[g['indices'][0]]['target.chat_id'] in fit_chats for g in by_context))
    return result
