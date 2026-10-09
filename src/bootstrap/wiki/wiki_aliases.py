"""Prepare a review-only identity inventory from Wikis and unified chat history.

This module never resolves identities for runtime use or updates source Wikis.
Export display names are retrospective labels, not proof of a historical alias.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


UNCERTAIN = re.compile(r'待验证|未确认|不明确|猜测|推测|可能|不是|两个人|注意|[？?]')
ALIAS_LABEL = re.compile(r'(?:别名(?:/昵称)?|昵称|又名)(?:\*\*)?[：:]\s*')


def key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def clean(value):
    return str(value).replace('**', '').strip(' \t`“”"')


def alias_parts(value):
    """Split explicit alias lists, retaining English names with spaces intact."""
    # Remove each annotation, without losing subsequent aliases on the same line.
    value = re.sub(r'[（(]来源[：:][^）)]*[）)]', '', value)
    value = re.split(r'[（(]来源[：:]', value, maxsplit=1)[0]
    for part in re.split(r'[、/,，;；]', value):
        part = clean(part).rstrip('）)。，')
        if part:
            yield part


def flags_for(alias, raw):
    flags = []
    if UNCERTAIN.search(raw):
        flags.append('uncertain_source')
    if (re.fullmatch(r'[\d\s\-~～至/:年月日.]+', alias) or len(alias) > 45
            or alias.strip('（）()。') in {'暂无', '无', '未知', '待补充'}):
        flags.append('not_a_clean_alias')
    if alias in {'我', '你', '他', '她', '本人', '室友', '老妈', '爸爸', '妈妈', '老婆', '老公'}:
        flags.append('relative_or_contextual_name')
    return flags


def read_sources(wiki_dir, aliases_path):
    claims, documents, unparsed, inputs = {}, [], [], []

    def source(path):
        data = path.read_bytes()
        inputs.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest()})
        return data.decode('utf-8')

    def add(owner, alias, kind, scope, evidence):
        owner, alias = clean(owner), clean(alias)
        if not owner or not alias or owner == alias:
            return
        identity = key([kind, owner, alias, scope])
        item = claims.setdefault(identity, dict(id=identity, kind=kind, owner=owner, alias=alias,
            scope=scope, status='pending_review', evidence=[], flags=[]))
        item['evidence'].append(evidence)
        raw = evidence.get('raw', alias) + ' ' + evidence.get('notes', '')
        item['flags'] = sorted(set(item['flags'] + flags_for(alias, raw)))

    if aliases_path:
        data = json.loads(source(aliases_path))
        for section, kind in [('users', 'person'), ('groups', 'group')]:
            for owner, row in data.get(section, {}).items():
                for alias in row.get('aliases', []):
                    add(owner, alias, kind, '', dict(kind='existing_alias_map', path=str(aliases_path),
                        raw=alias, notes=row.get('notes', ''), location=f'{section}/{owner}'))
    for section, kind in [('users', 'person'), ('groups', 'group')]:
        for path in sorted((wiki_dir / 'alias_suggestions' / section).glob('*.json')):
            row = json.loads(source(path))
            for alias in row.get('aliases', []):
                add(row.get('main_name', path.stem), alias, kind, '', dict(kind='old_suggestion',
                    path=str(path), raw=alias, generated_at=row.get('generated_at')))
        for path in sorted((wiki_dir / section).glob('*.md')):
            lines = source(path).splitlines()
            title = next((clean(x[2:]) for x in lines if x.startswith('# ')), path.stem)
            documents.append(dict(kind=kind, title=title, path=str(path), filename=path.stem))
            heading = ''
            for num, line in enumerate(lines, 1):
                if line.startswith('## '):
                    heading = line[3:].strip()
                ev = dict(kind='wiki_claim', path=str(path), line=num, raw=line)
                label = ALIAS_LABEL.search(line)
                if kind == 'person':
                    # Only the owner's alias fields, never relatives' aliases in prose.
                    direct = re.match(r'^[-*]\s*(?:\*\*)?(?:别名(?:/昵称)?|昵称|又名)(?:\*\*)?[：:]', line)
                    if direct:
                        for alias in alias_parts(line[label.end():]):
                            add(title, alias, kind, '', ev)
                    elif heading in {'别名', '昵称', '别名/昵称'} and line.startswith('- ') and not label:
                        for alias in alias_parts(line[2:]):
                            add(title, alias, kind, '', ev)
                    elif label:
                        unparsed.append(ev)
                else:
                    member = re.match(r'^-\s*\*\*([^*]+)\*\*[（(](?:别名|昵称)[：:]\s*(.*)', line)
                    group_name = re.match(r'^-\s*群(?:聊)?名称[：:]\s*(.*)', line)
                    if member:
                        # Preserve malformed/uncertain claims for review; never resolve them.
                        names = re.split(r'[）)]\s*[：:]', member[2], maxsplit=1)[0]
                        for alias in alias_parts(names):
                            add(member[1], alias, 'person', title, ev)
                    elif group_name:
                        add(title, group_name[1], kind, '', ev)
                    elif label:
                        unparsed.append(ev)
    return list(claims.values()), documents, unparsed, inputs


def account_of(row):
    parts = row['chat_id'].split(':', 2)
    # Do not combine missing-account sources just because their names match.
    return parts[1] if len(parts) == 3 and parts[1] != 'unknown' else 'unknown@' + row['chat_id']


def identity_of(row):
    sender_id = str(row.get('event', {}).get('sender_id') or '')
    if not sender_id:
        return None
    return account_of(row) + '/' + sender_id


def excerpt(row, line):
    return dict(line=line, message_id=row.get('message_id'), timestamp=row.get('timestamp'),
        chat_id=row['chat_id'], chat_name=row.get('chat_name'), sender=row.get('sender'),
        is_self=row.get('is_self'),
        sender_id=row.get('event', {}).get('sender_id'), message_kind=row.get('event', {}).get('kind'),
        text=str(row.get('text', ''))[:650])


def scan_history(path, focus_terms=()):
    identities, groups, quote_targets = {}, {}, defaultdict(list)
    claims, focus, counters = [], {term: dict(count=0, contexts=[]) for term in focus_terms}, Counter()
    previous = defaultdict(lambda: deque(maxlen=10))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for line, data in enumerate(handle, 1):
            digest.update(data)
            row = json.loads(data)
            counters['messages'] += 1
            event, chat_id = row.get('event', {}), row['chat_id']
            ident = identity_of(row)
            # System events may carry the contact ID but aren't authored by that contact.
            if ident and event.get('kind') != 'system':
                item = identities.setdefault(ident, dict(id=ident, account=account_of(row),
                    sender_id=str(event['sender_id']), names={}, chats={}, self_roles=Counter(), messages=0))
                item['messages'] += 1
                item['self_roles'][str(bool(row.get('is_self')))] += 1
                name = str(row.get('sender') or '')
                names = item['names'].setdefault(name, dict(count=0, first_record=row['timestamp'],
                    last_record=row['timestamp'], evidence=[]))
                names['count'] += 1
                names['first_record'] = min(names['first_record'], row['timestamp'])
                names['last_record'] = max(names['last_record'], row['timestamp'])
                if len(names['evidence']) < 2:
                    names['evidence'].append(dict(line=line, message_id=row.get('message_id'), chat_id=chat_id))
                item['chats'][chat_id] = row.get('chat_name', '')
                if event.get('kind') == 'text':
                    intro = re.fullmatch(r'我是群聊["“](.+?)["”]的(.{1,45})', str(row.get('text', '')).strip())
                    if intro:
                        claims.append(dict(id=key(['intro', row.get('message_id'), line]), kind='person',
                            owner=name, alias=clean(intro[2]), scope=intro[1], status='pending_review',
                            bound_identity=ident, flags=[], evidence=[dict(kind='self_introduction',
                            path=str(path), **excerpt(row, line))]))
            elif not ident:
                counters['missing_sender_id'] += 1
            if row.get('chat_type') == 'group':
                group = groups.setdefault(chat_id, dict(id=chat_id, names={}, participants=set(), messages=0))
                group['messages'] += 1
                group['names'][str(row.get('chat_name', ''))] = row.get('source_chat_id')
                if ident and event.get('kind') != 'system':
                    group['participants'].add(ident)
            quote = event.get('quote', {})
            if quote.get('replyToMessageId') and quote.get('quotedSender'):
                quote_targets[(chat_id, 'platform:' + str(quote['replyToMessageId']))].append(
                    (quote, excerpt(row, line)))
            text = str(row.get('text', ''))
            if event.get('kind') in {'text', 'quote'}:
                ev = excerpt(row, line)
                for term, result in focus.items():
                    if term in text:
                        result['count'] += 1
                        if len(result['contexts']) < 12:
                            result['contexts'].append([*previous[chat_id], ev])
                if focus:
                    previous[chat_id].append(ev)
    # Resolve quoted names against the actual referenced message, not the reply author.
    resolved_quotes = set()
    with path.open('rb') as handle:
        for line, data in enumerate(handle, 1):
            row = json.loads(data)
            target = (row['chat_id'], str(row.get('event', {}).get('id', '')))
            if target not in quote_targets:
                continue
            ident = identity_of(row)
            if not ident or row.get('event', {}).get('kind') == 'system':
                continue
            for quote, ev in quote_targets[target]:
                raw = str(quote.get('quotedContent', '')).strip()
                originals = [str(row.get('text', '')).strip(),
                    str(row.get('event', {}).get('original_content', '')).strip()]
                if not raw or raw not in originals:
                    counters['quote_content_mismatch'] += 1
                    continue
                resolved_quotes.add((ev['message_id'], ev['line']))
                alias = clean(quote['quotedSender'])
                if alias == clean(row.get('sender', '')):
                    continue
                claims.append(dict(id=key(['quote', ev['message_id'], line]), kind='person',
                    owner=row.get('sender', ''), alias=alias, scope=row.get('chat_name', ''),
                    bound_identity=ident, status='pending_review', flags=['quoted_name_requires_review'],
                    evidence=[dict(kind='resolved_quote_name', path=str(path), **ev,
                                   referenced_message=excerpt(row, line))]))
    counters['quote_references'] = sum(len(v) for v in quote_targets.values())
    counters['resolved_quote_references'] = len(resolved_quotes)
    for group in groups.values():
        group['participants'] = sorted(group['participants'])
    return dict(identities=list(identities.values()), groups=list(groups.values()),
        history_claims=claims, focus=focus, counts=dict(counters),
        source=dict(path=str(path.resolve()), sha256=digest.hexdigest()))


def build_review(wiki_dir, aliases_path, messages_path, focus_terms=()):
    claims, documents, unparsed, inputs = read_sources(Path(wiki_dir), Path(aliases_path) if aliases_path else None)
    history = scan_history(Path(messages_path), focus_terms)
    # Repeated quote evidence should not create hundreds of identical review rows.
    combined = {c['id']: c for c in claims}
    for claim in history.pop('history_claims'):
        ident = key([claim['kind'], claim['owner'], claim['alias'], claim['scope'], claim['bound_identity']])
        if ident in combined:
            combined[ident]['evidence'].extend(claim['evidence'])
        else:
            combined[ident] = {**claim, 'id': ident}
    claims = list(combined.values())
    by_name, by_group_name = defaultdict(set), defaultdict(set)
    identities = {item['id']: item for item in history['identities']}
    for ident in identities.values():
        for name in ident['names']:
            by_name[name].add(ident['id'])
    for group in history['groups']:
        for name in group['names']:
            by_group_name[name].add(group['id'])
    for claim in claims:
        index = by_name if claim['kind'] == 'person' else by_group_name
        candidates = index[claim['owner']] | index[claim['alias']]
        if claim['scope'] and claim['kind'] == 'person':
            candidates = {ident for ident in candidates if claim['scope'] in identities[ident]['chats'].values()}
        if claim.get('bound_identity'):
            candidates.add(claim['bound_identity'])
        claim['observed_identity_candidates'] = sorted(candidates)
        distinct_ids = ({identities[x]['sender_id'] for x in candidates}
                        if claim['kind'] == 'person' else candidates)
        if len(distinct_ids) > 1:
            claim['flags'].append('multiple_ids_do_not_merge')
        if not candidates:
            claim['flags'].append('no_exact_history_identity')
    alias_owners = defaultdict(set)
    for claim in claims:
        alias_owners[(claim['kind'], claim['alias'], claim['scope'])].add(claim['owner'])
    for claim in claims:
        owners = alias_owners[(claim['kind'], claim['alias'], claim['scope'])]
        if len(owners) > 1:
            claim['flags'].append('multiple_wiki_owners')
            claim['competing_owners'] = sorted(owners)
    grouped_documents = defaultdict(list)
    for doc in documents:
        grouped_documents[(doc['kind'], doc['title'])].append(doc['path'])
    duplicate_titles = [dict(kind=k[0], title=k[1], paths=v) for k, v in grouped_documents.items() if len(v) > 1]
    return dict(schema=1, status='pending_user_review', runtime_usable=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        policy='Review only. No identity merges or Wiki edits. No temporal admissibility for model inputs.',
        inputs=[*inputs, history['source']], documents=documents, duplicate_titles=duplicate_titles,
        claims=claims, unparsed_alias_lines=unparsed, **history,
        summary=dict(**history['counts'], wiki_person_files=sum(d['kind'] == 'person' for d in documents),
            wiki_group_files=sum(d['kind'] == 'group' for d in documents),
            observed_account_identities=len(identities), observed_sender_ids=len({x['sender_id'] for x in identities.values()}),
            observed_groups=len(history['groups']), alias_claims=len(claims),
            flagged_claims=sum(bool(c['flags']) for c in claims),
            duplicate_wiki_titles=len(duplicate_titles), unparsed_alias_lines=len(unparsed)))


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', '<br>')


FLAG_LABELS = dict(uncertain_source='原材料含猜测/否定/待验证', not_a_clean_alias='疑似日期或说明文本',
    relative_or_contextual_name='关系称呼，限上下文', multiple_ids_do_not_merge='涉及多个账号，不自动合并',
    no_exact_history_identity='未精确匹配历史身份', multiple_wiki_owners='别名对应多个 Wiki 人名',
    quoted_name_requires_review='引用名与导出名不同，需确认')


def write_review(review, output):
    """Write a new private draft; refuse overwriting a user's reviewed draft."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    for kind, filename in [('person', 'people.md'), ('group', 'groups.md')]:
        rows = [f'# {"人物" if kind == "person" else "群聊"}别名候选（全部待确认）', '',
            '同名和同别名只用于发现候选，不表示同一人。账号不自动合并；原始证据见 review.json。', '',
            '| 人物/群名 | 别名候选 | 范围 | 历史账号候选 | 备注 | 证据 ID |',
            '|---|---|---|---|---|---|']
        for claim in sorted((c for c in review['claims'] if c['kind'] == kind), key=lambda c: (c['owner'], c['scope'], c['alias'])):
            flags = '；'.join(FLAG_LABELS[x] for x in claim['flags']) or '待用户确认'
            rows.append('| ' + ' | '.join(cell(x) for x in [claim['owner'], claim['alias'], claim['scope'] or '来源未限定群',
                '、'.join(claim['observed_identity_candidates']), flags,
                claim['id'] + f'（{len(claim["evidence"])} 条来源）']) + ' |')
        (output / filename).write_text('\n'.join(rows) + '\n')
    rows = ['# 别名梳理：待确认草案', '', '本目录没有可供模型直接使用的别名映射。确认后再整理 Wiki。', '',
        '- [人物候选表](people.md)', '- [群名候选表](groups.md)', '- [完整来源与历史账号清单](review.json)', '',
        '## 覆盖情况', '']
    rows.extend(f'- {k}: {v}' for k, v in review['summary'].items())
    rows += ['', '导出中的显示名可能是导出时备注；消息时间范围不等于该别名当时已知。',
             'Wiki、旧别名表和旧建议均视为待核实陈述。未解析的自然语言别名段落完整保留在 JSON 中，未假装全部自动理解。',
             '引用名仅在消息 ID 和引用正文都匹配原消息时关联到原发送者；仍须确认命名冲突。', '', '## 同标题 Wiki', '']
    for item in review['duplicate_titles']:
        rows.append(f'- {item["title"]}: ' + '；'.join(item['paths']))
    rows += ['', '## 重点称呼的原始对话证据', '']
    for term, result in review['focus'].items():
        rows += [f'### {term}（命中 {result["count"]} 条，最多展示 12 个片段）', '']
        for context in result['contexts']:
            rows.append(f'**{cell(context[-1]["chat_name"])}**')
            rows.append('')
            for ev in context:
                date = datetime.fromtimestamp(ev['timestamp'], timezone.utc).isoformat()
                rows.append(f'- {date} {cell(ev["sender"])}：{cell(ev["text"])}（原文件行 {ev["line"]}）')
            rows.append('')
    (output / 'README.md').write_text('\n'.join(rows) + '\n')
