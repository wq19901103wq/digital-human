"""Compile curated Wiki claims into private, source-backed review pages.

This compiler renders editorial judgments; it does not infer identities or admit
material to historical experiments. The same plan produces JSON and Markdown.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from .wiki_aliases import cell


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest())


def local_time(timestamp):
    return datetime.fromtimestamp(timestamp, ZoneInfo('Asia/Shanghai')).isoformat() if timestamp is not None else '未知'


def source_locator(item):
    """Project provenance onto identifiers and metadata; never copy source prose."""
    fields = ('kind', 'path', 'line', 'end_line', 'sha256', 'row_sha256', 'excerpt_sha256',
              'message_id', 'claim_id', 'source_ref', 'case_id', 'json_pointer',
              'timestamp', 'observed_at', 'known_at', 'recorded_on', 'source_time_status',
              'speaker_id', 'sender_id', 'chat_id', 'is_self', 'message_kind')
    result = {key: item[key] for key in fields if key in item
              and (item[key] is None or isinstance(item[key], (str, int, float, bool)))}
    for key in ('context_before', 'context_after'):
        if key in item:
            result[key] = [source_locator(row) for row in item[key]]
    return result


def collect_evidence(sources, messages):
    """One scan resolves source locations without exporting message bodies."""
    evidence = {ref: source_locator(item) for ref, item in sources.items()}
    wanted = {}
    for ref, item in evidence.items():
        if item['kind'] == 'raw_message':
            wanted.setdefault(item['message_id'], []).append(ref)
    found, digest = set(), hashlib.sha256()
    with Path(messages).open('rb') as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            row = json.loads(raw)
            mid = row.get('message_id')
            if mid not in wanted:
                continue
            if mid in found:
                raise ValueError(f'duplicate source message ID: {mid}')
            found.add(mid)
            for ref in wanted[mid]:
                evidence[ref].update(path=str(Path(messages).resolve()), line=line,
                    row_sha256=hashlib.sha256(raw).hexdigest(), is_self=row.get('is_self'),
                    observed_at=row.get('timestamp'), speaker_id=row.get('event', {}).get('sender_id'),
                    chat_id=row.get('chat_id'))
    if missing := set(wanted) - found:
        raise ValueError(f'missing source messages: {sorted(missing)}')
    for ref, item in evidence.items():
        if item['kind'] == 'source_wiki':
            raw = Path(item['path']).read_bytes()
            text = raw.decode('utf-8')
            excerpt = sources[ref]['excerpt']
            if not excerpt or excerpt not in text:
                raise ValueError(f'source Wiki excerpt no longer matches: {ref}')
            start = text.index(excerpt)
            item.update(sha256=hashlib.sha256(raw).hexdigest(),
                        path=str(Path(item['path']).resolve()),
                        line=text[:start].count('\n') + 1,
                        end_line=text[:start + len(excerpt)].count('\n') + 1,
                        excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
                        observed_at=None, source_time_status='unknown')
        elif item['kind'] == 'current_conversation':
            item.update(observed_at=None, source_time_status='not_supplied',
                        known_at=item['recorded_on'])
        elif item['kind'] != 'raw_message':
            raise ValueError(f'unsupported source kind: {ref}')
    return evidence, dict(path=str(Path(messages).resolve()), sha256=digest.hexdigest())


def compile_content(plan_path, messages, review_path):
    plan, plan_source = read_json(plan_path)
    review, review_source = read_json(review_path)
    decisions = {d['person_id']: d for d in review['decisions']['decisions'] if d['status'] == 'user_confirmed'}
    identities = deepcopy(plan['identities'])
    for person in identities:
        if person['status'] != 'user_confirmed':
            continue
        decision = decisions.get(person['id'])
        if not decision:
            raise ValueError(f'identity has no user confirmation: {person["id"]}')
        projected = deepcopy(decision)
        decision_index = review['decisions']['decisions'].index(decision)
        projected['source'] = dict(kind='review_decision', **review_source,
                                  json_pointer=f'/decisions/decisions/{decision_index}',
                                  known_at=plan['recorded_on'])
        projected['historical_support'] = [source_locator(row)
                                            for row in decision.get('historical_support', [])]
        person.update(aliases=decision['aliases'], decision=projected)
    evidence, message_source = collect_evidence(plan['evidence_sources'], messages)
    pages = deepcopy(plan['pages'])
    if len({p['id'] for p in pages}) != len(pages):
        raise ValueError('duplicate page IDs')
    for page in pages:
        if not re.fullmatch(r'[a-z0-9_-]+', page['slug']):
            raise ValueError('page slug must be a local filename')
    if len({p['slug'] for p in pages}) != len(pages):
        raise ValueError('duplicate page filenames')
    claims = deepcopy(plan['fact_claims'])
    if len({c['id'] for c in claims}) != len(claims):
        raise ValueError('duplicate claim IDs')
    profiles = deepcopy(plan.get('interaction_profiles', []))
    for claim in [*claims, *profiles]:
        refs = claim['evidence_refs']
        if not refs or set(refs) - evidence.keys():
            raise ValueError(f'missing claim evidence: {claim["id"]}')
        if not claim['page_ids'] or set(claim['page_ids']) - {p['id'] for p in pages}:
            raise ValueError(f'unknown claim page: {claim["id"]}')
        claim.update(observed_at=sorted({evidence[r]['observed_at'] for r in refs
                                        if evidence[r]['observed_at'] is not None}),
                     known_at=plan['recorded_on'], known_at_precision='day',
                     runtime_usable=False, historical_input_status='not_admitted')
        claim.setdefault('valid_from', None)
        claim.setdefault('valid_to', None)
        claim.setdefault('conflicts_with', [])
        claim.setdefault('supersedes', [])
    address_ids = set(plan['address_ids'])
    addresses = [deepcopy(a) for a in review['context']['addresses'] if a['id'] in address_ids]
    if {a['id'] for a in addresses} != address_ids:
        raise ValueError('requested address evidence is missing')
    for address in addresses:
        address['evidence'] = [source_locator(row) for row in address.get('evidence', [])]
        address.update(runtime_usable=False, historical_input_status='not_admitted')
    return dict(schema='wiki_content_v2', status='source_backed_review', provenance_policy='locators_only',
                runtime_usable=False, historical_input_status='not_admitted',
                recorded_on=plan['recorded_on'], identities=identities, pages=pages,
                address_edges=addresses, fact_claims=claims, interaction_profiles=profiles,
                pending_identity_groups=review['decisions'].get('pending_groups', []),
                evidence=evidence, source_manifest=dict(plan=plan_source, review=review_source, messages=message_source),
                summary=dict(pages=len(pages), claims=len(claims), interaction_profiles=len(profiles),
                             addresses=len(addresses), evidence=len(evidence)))


def claim_lines(claim):
    refs = '、'.join(f'[{r}](sources.md#{r})' for r in claim['evidence_refs'])
    times = '、'.join(local_time(t) for t in claim['observed_at']) or '原始时间未提供'
    return [f'### {claim["title"]}', '', claim['description'], '',
            f'- 性质：{claim["claim_type"]}；状态：{claim["status"]}。',
            f'- 证据时间：{times}；本次整理可知日期：{claim["known_at"]}。',
            f'- 适用时间／范围：{claim["validity_note"]}', f'- 依据：{refs}', '']


def source_lines(ref, item):
    lines = [f'<a id="{ref}"></a>', f'## {ref}', '']
    if item['kind'] == 'raw_message':
        lines += [f'- 时间：{local_time(item.get("observed_at"))}',
                  f'- 说话人账号：`{item["speaker_id"]}`；is_self：{item.get("is_self")}',
                  f'- 聊天：`{item.get("chat_id")}`', f'- 消息 ID：`{item["message_id"]}`',
                  f'- 原始位置：`{item["path"]}:{item["line"]}`', '']
    elif item['kind'] == 'source_wiki':
        lines += [f'- 二手摘要位置：`{item["path"]}:{item["line"]}`；原摘要生成时间未知。', '']
    else:
        lines += [f'- 来源：当前用户对话；记录日期 {item["recorded_on"]}；原消息精确时间未提供。',
                  f'- 来源标识：`{item["source_ref"]}`', '']
    return lines


def write_content(content, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'knowledge.json').write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
    index = ['# 人物与群 Wiki：已确认范围', '',
             '本版整理实际称呼、关系、背景及原 Wiki 的纠正。每项均链接原始证据；'
             '待定账号保持分开。时间采用 Asia/Shanghai。', '',
             '**用途：当前背景资料审阅；尚未接入生成器、Judge 或历史评测。** '
             '用户本轮确认与全量历史材料均未获准回填历史题。', '',
             '结构化原件：[knowledge.json](knowledge.json)；[来源定位](sources.md)只保存引用，不复制原文。', '']
    for page in content['pages']:
        index.append(f'- [{page["title"]}]({page["slug"]}.md)：{page["summary"]}')
        rows = [f'# {page["title"]}', '', page['summary'], '',
                '[返回目录](README.md) · [来源证据](sources.md)', '',
                '本页为有来源的背景整理，尚未作为历史实验输入。', '']
        for person in content['identities']:
            if person['id'] != page['id']:
                continue
            rows += ['## 身份范围', '', person['description'], '']
            if person.get('decision'):
                decision = person['decision']
                rows += [f'- 用户已确认：{"、".join(person["aliases"])} → {person["name"]}。',
                         f'- 确认记录：`{decision["source"]["path"]}` '
                         f'JSON Pointer `{decision["source"]["json_pointer"]}`；记录于 {content["recorded_on"]}。',
                         f'- 绑定范围：{decision["scope"]["rule"]}', '']
        rows += ['## 已整理内容', '']
        for claim in [*content['fact_claims'], *content['interaction_profiles']]:
            if page['id'] in claim['page_ids']:
                rows += claim_lines(claim)
        rows += ['## 仍未知／不据此推断', '', *['- ' + s for s in page['unknowns']], '']
        (output / f'{page["slug"]}.md').write_text('\n'.join(rows))
    index += ['', '## 待定身份继续分开', '']
    for group in content['pending_identity_groups']:
        index.append('- ' + ' / '.join(group['names']) + '：不合并账号，不互相继承称呼或背景。')
    index += ['', '## 称呼的使用限制', '',
              '结构化地址边复用原称呼审阅稿；其中次数是规则命中数，不是全部使用频次。'
              '人物页按原句再区分直接称呼、提及、他人用法及梦境。', '',
              '| 说话人 → 对象 | 叫法 | 方向 | 命中数 |', '|---|---|---|---|']
    for a in content['address_edges']:
        index.append('| ' + ' | '.join(map(cell, [f'{a["source_name"]} → {a["target_name"]}',
                                                 a['alias'], a['direction'], str(a['count'])])) + ' |')
    (output / 'README.md').write_text('\n'.join(index) + '\n')
    sources = ['# 来源定位', '', '仅保留原记录定位、时间、说话人及哈希，不复制原文、引用正文或上下文摘录。'
               '原文在原始数据中按需审阅；去掉摘录不代表摘要已通过历史输入准入。', '']
    for ref, item in content['evidence'].items():
        sources += source_lines(ref, item)
    (output / 'sources.md').write_text('\n'.join(sources))
