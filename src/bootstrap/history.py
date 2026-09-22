"""Complete historical examples and globally ordered purpose manifests."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..config import ConfigError, sha256_file
from ..iteration import datasets, versions
from ..iteration.storage import write_json
from .build_fewshot_pool import extract_examples
from .build_testsets import _to_case
from . import conversations
from .ingest import message_identity


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def examples(messages: list[dict], max_context: int = 8, *, segmentation=None, statistics=None) -> list[dict]:
    rule = conversations.validate_policy(segmentation or conversations.policy())
    by_chat = defaultdict(list)
    for message in messages:
        by_chat[message['chat_id']].append(message)
    identities = {}
    for chat, source in by_chat.items():
        if any(a['timestamp'] > b['timestamp'] for a, b in zip(source, source[1:])):
            raise ConfigError('历史消息时间倒序')
        identities[chat] = [message_identity(m) for m in source]
        for message, identity in zip(source, identities[chat]):
            message['message_id'] = identity
    rows = extract_examples(messages, max_context=max_context, statistics=statistics,
                            **{k: v for k, v in rule.items() if k != 'schema'})
    prior = Counter()
    for row in rows:
        chat, _, offset = row['source_message_id'].rpartition(':')
        source, ids, i = by_chat[chat], identities[chat], int(offset)
        start, end = i - len(row['context_messages']), i + len(row['reply'])
        # Preserve full reply bursts and the actual input boundary, not answer time.
        row.update(context_message_ids=ids[start:i], reply_message_ids=ids[i:end],
                   source_span={'chat_id': chat, 'start': start, 'reply_start': i, 'end': end,
                                'start_timestamp': source[start]['timestamp'],
                                'end_timestamp': source[end - 1]['timestamp'], 'order_verified': False},
                   input_cutoff={'timestamp': source[i - 1]['timestamp'], 'index': i - 1,
                                 'order_verified': False},
                   timestamp=source[end - 1]['timestamp'], annotation_scope='example_only',
                   chat_id=row.get('source_chat_id'), prior_interactions=prior[chat],
                   familiarity='familiar' if prior[chat] else 'observed_first')
        row['id'] = digest({'context': row['context_message_ids'], 'reply': row['reply_message_ids']})[:24]
        row['content_sha256'] = digest({k: row[k] for k in
                                       ('context_messages', 'reply', 'context_message_ids', 'reply_message_ids')})
        prior[chat] += 1
    return rows


def as_case(row: dict, heldout=()) -> dict:
    case = {**_to_case(row), **{k: row[k] for k in
            ('source_span', 'input_cutoff', 'context_message_ids', 'reply_message_ids',
             'content_sha256', 'prior_interactions', 'familiarity')}}
    if row['source_span']['chat_id'] in heldout:
        case.update(familiarity='unseen_holdout', history_excluded_chat_ids=sorted(heldout))
    return case


def choose(rows: list[dict], total: int, group_ratio: float, cap: int, seed: int,
           occupied: set | None = None) -> list[dict]:
    pool = sorted(rows, key=lambda r: r['id'])
    random.Random(seed).shuffle(pool)
    quota = {'group': round(total * group_ratio), 'private': total - round(total * group_ratio)}
    counts, chats, selected = Counter(), Counter(), []
    blocked = occupied or set()
    for row in pool:
        kind, chat = row['relationship'], row['source_span']['chat_id']
        ids = set(row['context_message_ids'] + row['reply_message_ids'])
        if counts[kind] >= quota[kind] or chats[chat] >= cap or ids & blocked:
            continue
        selected.append(row)
        counts[kind] += 1
        chats[chat] += 1
    if dict(counts) != {k: v for k, v in quota.items() if v}:
        raise ConfigError(f'完整片段/每聊天上限下容量不足：需要 {quota}，取得 {dict(counts)}')
    return sorted(selected, key=lambda r: (r['input_cutoff']['timestamp'], r['id']))


def plan(rows: list[dict], *, total=1000, train_total=1200, acceptance_total=None, seed=42,
         development_start=None, acceptance_start=None, familiar_weight=.8, group_weight=.75,
         training_group_weight=.5, heldout_chat_fraction=.2, evaluation_chat_cap=25, learning_chat_cap=125,
         segmentation=None) -> tuple[dict, dict]:
    if any(not 0 <= v <= 1 for v in (familiar_weight, group_weight, training_group_weight, heldout_chat_fraction)):
        raise ConfigError('数据比例必须在 0 到 1 之间')
    acceptance_total = total if acceptance_total is None else acceptance_total
    if min(total, train_total, acceptance_total, evaluation_chat_cap, learning_chat_cap) <= 0:
        raise ConfigError('数据规模和每聊天上限必须为正数')
    times = sorted(r['input_cutoff']['timestamp'] for r in rows)
    if len(times) < 3:
        raise ConfigError('历史样本不足')
    a = development_start if development_start is not None else times[len(times) // 4]
    b = acceptance_start if acceptance_start is not None else times[65 * len(times) // 100]
    if not times[0] < a < b <= times[-1]:
        raise ConfigError('全局时间窗口无效')
    windows = {
        'learning': [r for r in rows if r['source_span']['end_timestamp'] < a],
        'development': [r for r in rows if r['source_span']['start_timestamp'] >= a
                        and r['source_span']['end_timestamp'] < b],
        'acceptance': [r for r in rows if r['source_span']['start_timestamp'] >= b],
    }
    # A separate, explicitly labelled unseen-contact condition supplements replay.
    chats = {kind: sorted({r['source_span']['chat_id'] for r in rows if r['relationship'] == kind})
             for kind in ('group', 'private')}
    counts = {window: {kind: Counter(r['source_span']['chat_id'] for r in values
                                    if r['relationship'] == kind and r['familiarity'] == 'familiar')
                       for kind in chats} for window, values in windows.items()}
    learning_counts = {kind: Counter(r['source_span']['chat_id'] for r in windows['learning']
                                     if r['relationship'] == kind) for kind in chats}
    # Capacity-only rejection sampling, frozen before any model output is observed.
    heldout = set()
    for attempt in range(1000):
        rng = random.Random(seed + attempt)
        trial = set()
        for values in chats.values():
            trial.update(rng.sample(values, min(len(values), max(1, round(len(values) * heldout_chat_fraction))) if familiar_weight < 1 else 0))
        valid = True
        # The holdout must leave enough learning data too. Checking evaluation
        # alone can choose an infeasible split even when another split fits the
        # exact same windows, quotas and per-chat caps.
        for size, ratio in ((total, group_weight), (train_total, training_group_weight)):
            quota = {'group': round(size * ratio), 'private': size - round(size * ratio)}
            for kind, need in quota.items():
                if sum(min(n, learning_chat_cap) for chat, n in learning_counts[kind].items()
                       if chat not in trial) < need:
                    valid = False
        for window in ('development', 'acceptance'):
            window_total = acceptance_total if window == 'acceptance' else total
            for kind, ratio in [('group', group_weight), ('private', 1-group_weight)]:
                capacity = counts[window][kind]
                familiar_need = round(round(window_total * familiar_weight) * ratio)
                unseen_need = round((window_total - round(window_total * familiar_weight)) * ratio)
                if (sum(min(n, evaluation_chat_cap) for chat, n in capacity.items() if chat not in trial) < familiar_need
                        or sum(min(n, evaluation_chat_cap) for chat, n in capacity.items() if chat in trial) < unseen_need):
                    valid = False
        if valid:
            heldout = trial
            break
    if not valid:
        raise ConfigError('按类型留出对象仍无法同时满足学习及评测配额，请调整全局窗口或明确修改配额')
    learning = [r for r in windows['learning'] if r['source_span']['chat_id'] not in heldout]
    roles = {'gen_learning': learning,
             'gen_optimization': choose(learning, total, group_weight, learning_chat_cap, seed),
             'judge_training': choose(learning, train_total, training_group_weight, learning_chat_cap, seed + 1)}
    occupied = set()
    for role, window in [('development', 'development'), ('fixed_test', 'acceptance')]:
        candidates = windows[window]
        familiar = [r for r in candidates if r['source_span']['chat_id'] not in heldout
                    and r['familiarity'] == 'familiar']
        unseen = [r for r in candidates if r['source_span']['chat_id'] in heldout]
        window_total = acceptance_total if window == 'acceptance' else total
        familiar_total = round(window_total * familiar_weight)
        selected = (choose(familiar, familiar_total, group_weight, evaluation_chat_cap, seed, occupied)
                    + choose(unseen, window_total - familiar_total, group_weight, evaluation_chat_cap, seed, occupied))
        roles[role] = selected
        occupied.update(mid for row in selected for mid in row['context_message_ids'] + row['reply_message_ids'])
    roles['judge_development'] = list(roles['development'])
    audit = {'reply_segmentation': conversations.validate_policy(segmentation or conversations.policy()),
             'development_start': a, 'acceptance_start': b, 'seed': seed,
             'development_total': total, 'acceptance_total': acceptance_total, 'training_total': train_total,
             'split_strategy': 'global_time_and_purpose', 'familiar_weight': familiar_weight,
             'group_weight': group_weight,
             'training_group_weight': training_group_weight, 'heldout_chat_fraction': heldout_chat_fraction,
             'evaluation_chat_cap': evaluation_chat_cap, 'learning_chat_cap': learning_chat_cap, 'unseen_condition': 'whole_chat_holdout',
             'unseen_chat_ids': sorted(heldout), 'natural_first_contact_verified': False,
             'window_available': {k: dict(Counter(r['relationship'] for r in v)) for k, v in windows.items()},
             'public_history_before': a, 'fixed_answers_for_optimization': False,
             'learning_roles_may_overlap': True, 'development_roles_share_messages': True,
             'development_sharing_reason': '共享开发选择材料；不声称 Gen/Judge 两次独立验证',
             'holdout_selection_attempt': attempt, 'holdout_selection_uses_model_results': False,
             'independent_acceptance_requires_verified_static_sources': True}
    return roles, audit


def write_rows(path: Path, rows) -> None:
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def publish(messages, rows, roles, protocol, source_files) -> str:
    """Create data only: no generator copies, no pointer writes, no model calls."""
    with versions.file_lock(versions.PRIVATE / '.iteration.lock'):
        vid = versions.create_data_version(None, {'schema': 2, 'source_messages': len(messages),
            'split_strategy': protocol['split_strategy'], 'source_files': source_files,
            'source_assessment': '沿用已导入历史导出的真人来源认定；自然首次接触覆盖未认证'})
        directory = versions.data_version_dir(vid)
        write_rows(directory / 'messages.jsonl', ({**m, 'message_id': message_identity(m)} for m in messages))
        write_rows(directory / 'fewshot_pool.jsonl', rows)
        pool_hash = sha256_file(directory / 'fewshot_pool.jsonl')
        purpose = {'schema': 2, 'data_ref': vid, 'history_policy': 'complete_before_input_v1',
                   'protocol': protocol, 'roles': {}}
        for role, selected in roles.items():
            filename = datasets.ROLES[role][1]
            cases = [as_case(r, protocol['unseen_chat_ids']) for r in selected]
            write_rows(directory / filename, cases)
            purpose['roles'][role] = {'label': datasets.ROLES[role][0], 'file': filename,
                'total': len(cases), 'sha256': sha256_file(directory / filename),
                'chat_types': dict(Counter(c['chat_type'] for c in cases)),
                'cohorts': dict(Counter(c['familiarity'] for c in cases)),
                'chats': len({c['source_span']['chat_id'] for c in cases}),
                'sealed': role == 'fixed_test'}
        write_json(directory / 'purposes.json', purpose)
        write_json(directory / 'report.json', {'review_status': 'approved',
            'examples_sha256': pool_hash, 'total': len(rows), 'history_policy': purpose['history_policy']})
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest.update(purposes_sha256=sha256_file(directory / 'purposes.json'),
            fewshot_pool={'total': len(rows), 'sha256': pool_hash},
            testsets={role: {**purpose['roles'][role],
                      'group': purpose['roles'][role]['chat_types'].get('group', 0),
                      'private': purpose['roles'][role]['chat_types'].get('private', 0),
                      'file_sha256': purpose['roles'][role]['sha256']}
                      for role in ('development', 'fixed_test')})
        write_json(directory / 'manifest.json', manifest)
        # The same source-backed audit used by check.py is mandatory for every
        # new build. On failure, stop without finalizing or returning a version.
        from .data_quality import audit, require_passed
        quality = audit(directory)
        write_json(directory / 'data_quality.json', quality)
        require_passed(quality)
        versions.finalize_data_version(vid)
    return vid
