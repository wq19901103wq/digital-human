"""Answer-blind views and local, categorical feature crosses."""
from __future__ import annotations

from datetime import datetime
import re
from zoneinfo import ZoneInfo

from ..history_sources import digest, require
from .schema import NEEDS, UNKNOWN

COUNT_BINS = (0, 1, 2, 3, 5, 10)
LENGTH_BINS = (0, 4, 8, 16, 32, 64, 128)
GAP_BINS = (0, 10, 60, 300, 1800, 3600, 86400)


def bucket(value, bins=COUNT_BINS):
    if value is None:
        return UNKNOWN
    return str(sum(value > boundary for boundary in bins))


def context_view(record, *, example=False):
    # Explicit allow-list. Never use target human_reply, generation, verdict,
    # label, future summaries, or even the historical example's reply here.
    messages = record.get('context_messages' if example else 'context')
    require(isinstance(messages, list) and messages and all(isinstance(m, dict) for m in messages),
            'Ranker needs the complete structured context')
    clean = [{k: m.get(k) for k in ('sender', 'text', 'is_self', 'timestamp')} for m in messages]
    require(all(type(m['is_self']) is bool and isinstance(m['text'], str) for m in clean),
            'Ranker context lacks speaker roles or text')
    cutoff = record['input_cutoff']['timestamp']
    require(all(isinstance(m['timestamp'], (int, float)) and 0 < m['timestamp'] <= cutoff for m in clean),
            'Context contains a message after the input cutoff')
    return dict(chat_type=record.get('chat_type', record.get('relationship')),
                chat_name=record.get('chat_name', UNKNOWN), messages=clean)


def reply_view(example):
    reply = example.get('reply')
    require(isinstance(reply, list) and reply and all(isinstance(v, str) for v in reply),
            'Example must retain all reply bubbles')
    return dict(context=context_view(example, example=True), reply=list(reply))


def person(message):
    if message.get('is_self'):
        return '__self__'
    # Sender names are exporter identities, not guessed global person IDs.
    # Chat scope prevents unrelated people with identical display names matching.
    return message.get('sender') or UNKNOWN


def identity(chat, speaker):
    return speaker if speaker in (UNKNOWN, '__self__') else digest([chat, speaker])


def mention_id(text, messages, chat):
    named = {person(m) for m in messages if person(m) not in (UNKNOWN, '__self__')}
    found = [v for v in named if re.search(r'@' + re.escape(v) + r'(?=$|\s|[，。！？,:：])', text)]
    return identity(chat, found[0]) if len(found) == 1 else UNKNOWN


def surface(text):
    return {key: str(bool(re.search(pattern, text))) for key, pattern in {
        'has_question_mark': r'[?？]', 'has_exclamation': r'[!！]', 'has_mention': '@',
        'has_ellipsis': r'…|\.{3}', 'has_emoji': r'[\U0001F300-\U0001FAFF]|\[[^\]]{1,8}\]',
        'has_laughter': r'哈|嘿嘿|嘻|hhh|233|lol', 'has_link_or_media_marker': r'https?://|\[图片\]|\[语音\]|\[视频\]',
        'uses_second_person': r'你|您', 'has_number': r'\d', 'has_particle': r'[啊呀吧呢哦噢嘛啦呗哇]',
        'has_repeated_char': r'(.)\1{2}', 'has_quote': r'[“”「」]|引用',
    }.items()}


def pattern(lengths):
    return '|'.join(bucket(v, LENGTH_BINS) for v in lengths[-5:]) or UNKNOWN


def context_local(record, *, example=False):
    view = context_view(record, example=example)
    messages, kind = view['messages'], view['chat_type']
    chat = record.get('source_chat_id') or record.get('chat_id') or UNKNOWN
    speakers = [person(m) for m in messages]
    known = [s for s in speakers if s != UNKNOWN]
    tail = next((i for i, s in enumerate(reversed(speakers)) if s != speakers[-1]), len(speakers))
    stamp = datetime.fromtimestamp(messages[-1]['timestamp'], ZoneInfo('Asia/Shanghai'))
    gaps = [b['timestamp']-a['timestamp'] for a, b in zip(messages, messages[1:])]
    require(all(g >= 0 for g in gaps), 'Context message order is invalid')
    burst = 1
    for gap in reversed(gaps[-max(tail-1, 0):] if tail > 1 else []):
        if gap > 60:
            break
        burst += 1
    self_index = next((i for i, s in enumerate(reversed(speakers)) if s == '__self__'), None)
    result = dict(chat_type=kind, chat_id=chat, counterpart_id=chat if kind == 'private' else UNKNOWN,
        latest_speaker_id=identity(chat, speakers[-1]),
        explicit_addressee_id=mention_id(messages[-1]['text'], messages, chat),
        visible_participant_count_bin=bucket(len(set(known))), context_message_count_bin=bucket(len(messages)),
        speaker_switch_count_bin=bucket(sum(a != b for a, b in zip(speakers, speakers[1:]))),
        last_speaker_run_length_bin=bucket(tail), last_speaker_burst_length_bin=bucket(burst),
        has_self_history=str('__self__' in speakers), last_self_position_bin=bucket(self_index),
        hour_band=str(stamp.hour//4), weekday_or_weekend='weekend' if stamp.weekday() >= 5 else 'weekday',
        latest_gap_bin=bucket(gaps[-1] if gaps else None, GAP_BINS),
        recent_gap_pattern='|'.join(bucket(g, GAP_BINS) for g in gaps[-4:]) or UNKNOWN,
        message_length_pattern=pattern([len(m['text']) for m in messages]),
        total_length_bin=bucket(sum(len(m['text']) for m in messages), LENGTH_BINS),
        is_group_repetition_pattern=str(kind == 'group' and len(messages) >= 2 and
                                       messages[-1]['text'] == messages[-2]['text']))
    result['has_explicit_addressee'] = str(result['explicit_addressee_id'] != UNKNOWN)
    for i in range(5):
        speaker = speakers[-1-i] if i < len(speakers) else UNKNOWN
        result[f'speaker_id_{i}'] = identity(chat, speaker)
        result[f'speaker_role_{i}'] = ('self' if speaker == '__self__' else
            UNKNOWN if speaker == UNKNOWN else 'counterpart' if kind == 'private' else 'other')
        result[f'speaker_same_previous_{i}'] = str(i+1 < len(speakers) and speaker == speakers[-2-i])
    result.update(surface('\n'.join(m['text'] for m in messages)))
    return result


def reply_local(example):
    view = reply_view(example)
    bubbles, messages = view['reply'], view['context']['messages']
    lengths, text = list(map(len, bubbles)), '\n'.join(bubbles)
    chat = example.get('source_chat_id') or example.get('chat_id') or UNKNOWN
    context_text = '\n'.join(m['text'] for m in messages)
    return dict(bubble_count_bin=bucket(len(bubbles)), total_length_bin=bucket(sum(lengths), LENGTH_BINS),
        first_bubble_length_bin=bucket(lengths[0], LENGTH_BINS), last_bubble_length_bin=bucket(lengths[-1], LENGTH_BINS),
        max_bubble_length_bin=bucket(max(lengths), LENGTH_BINS), bubble_length_pattern=pattern(lengths),
        question_bubble_position=('none' if not re.search('[?？]', text) else
            'first' if re.search('[?？]', bubbles[0]) else 'last' if re.search('[?？]', bubbles[-1]) else 'middle'),
        # The snapshot has no per-reply timestamps. Never infer them from end_timestamp.
        first_reply_delay_bin=UNKNOWN, intra_reply_gap_pattern=UNKNOWN,
        reply_addressee_id=mention_id(text, messages, chat),
        reuses_own_context_phrase=str(any(len(b) >= 3 and b in context_text for b in bubbles)), **surface(text))


def equal(a, b):
    return UNKNOWN if a == UNKNOWN or b == UNKNOWN else str(a == b)


def tokens(text):
    # Fixed character bigrams + Latin words, no corpus fitting.
    return set(re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.lower())) | \
        {text[i:i+2] for i in range(len(text)-1) if all('\u4e00' <= c <= '\u9fff' for c in text[i:i+2])}


def combine(target, example, target_features, example_features, reply_features):
    """Pure local function. LLM features on either side were extracted alone."""
    t, e, r = target_features, example_features, reply_features
    cross = {f'equal_{key}': equal(t[key], e[key]) for key in
        ('chat_type', 'chat_id', 'counterpart_id', 'latest_speaker_id', 'scene', 'relationship',
         'topic_domain', 'dialogue_stage', 'primary_intent')}
    cross['same_reply_addressee'] = equal(t['explicit_addressee_id'], r['reply_addressee_id'])
    for key, other, right in (
        ('primary_intent', 'primary_intent', e), ('primary_intent', 'reply_action', r),
        ('primary_intent', 'opening_style', r), ('incoming_tone', 'reply_tone', r),
        ('emotion_intensity', 'emotion_intensity', r), ('address_register', 'address_type', r),
        ('has_explicit_addressee', 'has_address', r), ('chat_type', 'reply_addressee_scope', r),
        ('has_mention', 'reply_addressee_scope', r), ('last_speaker_burst_length_bin', 'bubble_count_bin', r),
        ('message_length_pattern', 'bubble_length_pattern', r), ('last_self_action', 'reply_action', r),
        ('last_self_tone', 'reply_tone', r), ('needs_clarification', 'clarifies_own_ambiguity', r)):
        cross[f'{key}_x_{other}'] = t[key] + '|' + right[other]
    for need in NEEDS:
        cross[f'needs_{need}_x_reply_action'] = t[f'needs_{need}'] + '|' + r['reply_action']
    a, b = ({key for key in NEEDS if side[f'needs_{key}'] == 'yes'} for side in (t, e))
    missing = any(side[f'needs_{key}'] == UNKNOWN for side in (t, e) for key in NEEDS)
    cross['needs_intersection'] = UNKNOWN if missing else bucket(len(a & b))
    cross['needs_jaccard'] = (UNKNOWN if missing else
        (bucket(len(a & b)/len(a | b), (0, .25, .5, .75)) if a | b else 'empty'))
    for key in ('answer', 'clarify', 'coordinate', 'comfort', 'acknowledge'):
        cross[f'pending_{key}_match'] = equal(t[f'pending_intent_{key}'], e[f'pending_intent_{key}'])
    for key in ('visible_participant_count_bin', 'speaker_switch_count_bin', 'last_speaker_run_length_bin',
                'formality', 'emotion_intensity'):
        levels = {'low': 0, 'medium': 1, 'high': 2}
        left, right = levels.get(t[key], t[key]), levels.get(e[key], e[key])
        cross[f'difference_{key}'] = (UNKNOWN if UNKNOWN in (left, right) else str(max(-3, min(3, int(left)-int(right)))))
    age = target['input_cutoff']['timestamp'] - example['source_span']['end_timestamp']
    require(age > 0, 'Candidate is not strictly earlier than the target cutoff')
    cross['history_age_bin'] = bucket(age / 86400, (1, 7, 30, 90, 365))
    a, b = (tokens('\n'.join(m['text'] for m in view['messages'])) for view in
            (context_view(target), context_view(example, example=True)))
    cross['lexical_jaccard'] = bucket(len(a & b)/len(a | b), (0, .05, .1, .2, .4, .6)) if a | b else 'empty'
    return {f'{prefix}.{key}': str(value) for prefix, side in
            (('target', t), ('example_context', e), ('example_reply', r), ('cross', cross)) for key, value in side.items()}
