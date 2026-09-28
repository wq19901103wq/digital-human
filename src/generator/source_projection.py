"""Lossless, source-backed participants and quotes for an explicit Gen policy.

Only the validated input span is projected. Participant labels are local to the
request; neither account IDs nor text outside that span become prompt inputs.
"""
from collections import Counter
from dataclasses import dataclass
import json

from ..config import ConfigError

POLICY = 'source_participants_v1'


def validate_policy(policy):
    if policy is not None and policy != POLICY:
        raise ConfigError(f'Unknown context_projection policy: {policy!r}')


@dataclass(frozen=True)
class SourceContext:
    history: str
    unread: str
    identity_rule: str
    stats: dict


def _account(event):
    value = event.get('sender_id')
    return value if isinstance(value, str) and value.strip() and value != 'unknown' else None


def _platform_id(event):
    value = event.get('id')
    return value[9:] if isinstance(value, str) and value.startswith('platform:') and value[9:] else None


def project_validated(case, sources):
    """Project after HistorySources.validate, without widening its context span.

    The caller owns source validation and its before/after mutation checks, just
    as it does for prompt_case. Never use reply rows or inferred name matches.
    """
    span = case['source_span']
    rows = sources.chats[span['chat_id']][span['start']:span['reply_start']]
    roster = {'P1': {'names': [], 'known': True, 'self': True}}
    accounts, speakers = {}, []
    stats = dict(messages=len(rows), known_account_messages=0,
                 unknown_account_messages=0, quote_messages=0, linked_quotes=0)
    for row in rows:
        event = row.event if isinstance(row.event, dict) else {}
        account = _account(event)
        stats['known_account_messages' if account else 'unknown_account_messages'] += 1
        if row.is_self is True:
            speaker = 'P1'
        else:
            # An absent/invalid account never turns matching display names into
            # a source-backed identity. Typed self evidence has precedence.
            key = (row.is_self is False, account) if account else None
            speaker = accounts.get(key) if key is not None else None
            if speaker is None:
                speaker = f'P{len(roster) + 1}'
                roster[speaker] = {'names': [], 'known': account is not None,
                                   'self': False, 'other': row.is_self is False}
                if key is not None:
                    accounts[key] = speaker
        name = str(row.sender)
        if name not in roster[speaker]['names']:
            roster[speaker]['names'].append(name)
        speakers.append(speaker)

    events = [row.event if isinstance(row.event, dict) else {} for row in rows]
    ids = [_platform_id(event) for event in events]
    counts = Counter(value for value in ids if value is not None)
    targets = {value: index for index, value in enumerate(ids)
               if value is not None and counts[value] == 1}
    lines = []
    for index, (row, event, speaker) in enumerate(zip(rows, events, speakers)):
        quote = event.get('quote')
        link = ''
        if isinstance(quote, dict):
            stats['quote_messages'] += 1
            ref = quote.get('replyToMessageId')
            ref = str(ref) if type(ref) in (str, int) else None
            target = targets.get(ref)
            if target is not None and target < index:
                link = f' | 引用 M{target + 1}（{speakers[target]}）'
                stats['linked_quotes'] += 1
        lines.append(f'[M{index + 1} | {speaker}{link}] {row.sender}: {row.text}')

    names = []
    for speaker, person in roster.items():
        if person['self']:
            role = '本人，本次回复者' + ('；当前片段无本人发言' if not person['names'] else '')
        else:
            role = '其他发言人' if person['other'] else '本人关系未标明'
            role += '；按来源账号归并' if person['known'] else '；账号未知，未与其他行合并'
        labels = json.dumps(person['names'], ensure_ascii=False)
        names.append(f'{speaker}：{role}；本片段显示名 {labels}')
    rule = ('<source_participants>\n你以 P1 身份接话，即使 P1 未在片段发言。'
            'P 编号只标识消息发出者，不直接判定引用、转述或模仿中的事实主体。'
            'M 编号定位当前片段；引用链接由来源消息 ID 确认，未定位的引用和 @ 保留原文含义。'
            '显示名不是身份键；未知账号的行可能属于同一人，但不据此合并。编号不写进回复。\n'
            + '\n'.join(names) + '\n</source_participants>\n')
    return SourceContext('\n'.join(lines), lines[-1] if lines else '', rule, stats)
