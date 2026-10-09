"""Directed forms of address from chat evidence; review drafts, never identity merges."""
from collections import defaultdict
from datetime import datetime, timezone
import re

from .wiki_aliases import account_of, cell, excerpt, identity_of, key


USAGE = {
    'direct_address_candidate': '当面称呼候选（指向待核）',
    'scoped_name_use': '在原话中提及（不等于当面称呼）',
    'requested_name': '说话人希望被叫',
    'reported_name': '说话人转述别人叫法',
    'disliked_name': '说话人表示不喜欢的叫法',
}
DIRECTION = {'self_to_other': '本人→对方', 'other_to_self': '对方→本人',
             'other_to_other': '他人→他人', 'self_reference': '谈及自己',
             'naming_statement': '称呼声明，未观察到实际使用',
             'other_account': '其他来源账号，未映射为本人', 'unknown': '方向待核'}


def self_role(identity):
    roles = set(identity.get('self_roles', {}))
    return True if roles == {'True'} else False if roles == {'False'} else None


class AddressCollector:
    def __init__(self, identities, decisions, messages, self_account=None):
        self.identities, self.messages, self.records = identities, messages, {}
        accounts = {item['account'] for item in identities.values()}
        self.self_account = self_account or (next(iter(accounts)) if len(accounts) == 1
                                            and not next(iter(accounts)).startswith('unknown') else None)
        self.scoped_names = defaultdict(list)
        for decision in decisions.get('decisions', []):
            if decision.get('status') != 'user_confirmed':
                continue
            chat_id = decision.get('scope', {}).get('chat_id')
            if chat_id:
                self.scoped_names[chat_id].append(decision)

    def add(self, row, line, alias, target, target_name, usage, binding, previous, decision=None):
        source = identity_of(row)
        in_perspective = account_of(row) == self.self_account
        source_self = row.get('is_self') if in_perspective else None
        target_self = self_role(self.identities.get(target, {})) if in_perspective else None
        if usage in {'requested_name', 'reported_name', 'disliked_name'}:
            direction = 'naming_statement'
        elif not in_perspective:
            direction = 'other_account'
        elif source == target:
            direction = 'self_reference'
        elif source_self is True and target_self is False:
            direction = 'self_to_other'
        elif source_self is False and target_self is True:
            direction = 'other_to_self'
        elif source_self is False and target_self is False:
            direction = 'other_to_other'
        else:
            direction = 'unknown'
        rid = key([source, target, alias, row['chat_id'], usage, binding])
        record = self.records.setdefault(rid, dict(id=rid, source_identity=source,
            source_name=row.get('sender', ''), source_is_self=source_self,
            source_export_is_self=row.get('is_self'), perspective_account=self.self_account,
            target_identity=target, target_name=target_name, target_is_self=target_self,
            alias=alias, usage=usage, direction=direction, binding=binding,
            chat_id=row['chat_id'], chat_name=row.get('chat_name', ''),
            status='pending_review', runtime_usable=False, historical_input_status='not_admitted',
            count=0, first_observed_at=row.get('timestamp'), last_observed_at=row.get('timestamp'),
            evidence=[], generic_title_not_identity=bool(re.search(r'哥|姐|总|老板|老师|弟|奶奶', alias))))
        record['count'] += 1
        timestamp = row.get('timestamp')
        if timestamp is not None:
            record['first_observed_at'] = min(record['first_observed_at'] or timestamp, timestamp)
            record['last_observed_at'] = max(record['last_observed_at'] or timestamp, timestamp)
        if decision:
            record['identity_decision_id'] = decision['id']
            record['target_person_id'] = decision.get('person_id')
            record['identity_binding_note'] = '仅本轮确认的范围内对应；不证明历史时刻已知，也不确认称呼偏好'
        if len(record['evidence']) < 3:
            record['evidence'].append(dict(path=str(self.messages), **excerpt(row, line),
                                           context_before=list(previous)))

    def naming_statement(self, row, line, alias, preference, previous):
        usage = {'disliked': 'disliked_name', 'reported': 'reported_name'}.get(preference, 'requested_name')
        self.add(row, line, alias, identity_of(row), row.get('sender', ''), usage,
                 'statement_about_speaker', previous)

    def confirmed_mentions(self, row, line, text, vocative, previous):
        for decision in self.scoped_names.get(row['chat_id'], []):
            account = row['chat_id'].split(':', 2)[1]
            targets = [ident for ident in decision.get('observed_identity_records', [])
                       if ident.startswith(account + '/')]
            if len(targets) != 1:
                continue
            for alias in decision.get('aliases', []):
                if alias not in text:
                    continue
                # Alias substring use proves mention only; it does not establish an addressee.
                usage = 'direct_address_candidate' if alias == vocative else 'scoped_name_use'
                self.add(row, line, alias, targets[0], decision['canonical_name'], usage,
                         'user_confirmed_scoped_identity', previous, decision)


def write_addresses(review, output):
    records = review['context']['addresses']
    rows = ['# 谁怎么称呼谁：按人物与聊天范围整理', '',
            '自动提取候选，不因称呼合并身份。仅已确认身份在其限定群内绑定；'
            '私聊对方仍可能不是句中所指的人。一次出现不代表惯用称呼。', '',
            '实际用过、转述、希望被叫、不喜欢的叫法分别记录；提及一个人不等于正在对他讲话。'
            '本人角色仅使用指定来源账号内的 is_self；其他来源的自己发送不等于本人发送。', '',
            f'当前本人视角的来源账号：`{review["context"]["perspective_account"] or "未指定，方向待核"}`。', '',
            '数字是当前提取规则命中的消息数，非全部历史使用频次；每项保留最多三条代表原话。', '',
            '## 概况', '']
    for direction, label in DIRECTION.items():
        items = [r for r in records if r['direction'] == direction]
        if items:
            rows.append(f'- {label}：{len(items)} 个分范围记录，{sum(r["count"] for r in items)} 次命中。')
    groups = defaultdict(list)
    for record in records:
        groups[record['target_identity']].append(record)
    for target, items in sorted(groups.items(), key=lambda pair: (
            -any(r['direction'] == 'self_to_other' for r in pair[1]),
            -sum(r['count'] for r in pair[1]), pair[0])):
        rows += ['', f'## {cell(items[0]["target_name"])}', '', f'目标账号：`{target}`', '',
                 '| 谁 → 谁 | 叫法 | 类型／方向 | 范围 | 次数／时间 | 原话依据 |',
                 '|---|---|---|---|---|---|']
        for record in sorted(items, key=lambda r: (-r['count'], r['id'])):
            dates = [datetime.fromtimestamp(record[k], timezone.utc).strftime('%Y-%m-%d')
                     if record[k] is not None else '未知'
                     for k in ('first_observed_at', 'last_observed_at')]
            ev = record['evidence'][0]
            rows.append('| ' + ' | '.join(cell(value) for value in (
                f'{record["source_name"]} → {record["target_name"]}', record['alias'],
                f'{USAGE[record["usage"]]}；{DIRECTION[record["direction"]]}',
                record['chat_name'], f'{record["count"]}；{dates[0]}～{dates[1]}',
                f'{ev["text"][:250]}（行 {ev["line"]}）')) + ' |')
    rows += ['', '完整 source_identity / target_identity、chat_id、消息 ID、上下文、确认状态及时间见 review.json 的 context.addresses。',
             '这些记录用于当前审阅，未接入生成器或 Judge；历史评测还必须依赖当时可用的身份与原始消息证据。']
    (output / 'addresses.md').write_text('\n'.join(rows) + '\n')
