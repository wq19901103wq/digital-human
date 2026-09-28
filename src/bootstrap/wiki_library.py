"""Resumable, source-located organization of a complete legacy Wiki library.

The library is review material, not an admission path for historical inference.
Document identity and account identity deliberately remain separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import re

from ..iteration.storage import atomic_write, read_json, write_json
from .wiki_content import source_locator

CATEGORIES = {
    'identity': '身份与称呼', 'relationships': '人物关系', 'background': '工作与生活背景',
    'experience': '经历与知识归属', 'entities': '地点与实体归属', 'events': '事件与事项',
    'group_topic': '群与话题背景', 'interaction': '互动习惯与偏好',
}
EMPTY = re.compile(r'^(?:[（(]?(?:暂无|无|未知|不详|待补充|待完善|待填写)[）)]?|[-—/])?[。.]?$')
DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def substantive(line):
    line = re.sub(r'^\s*[-*#]+\s*', '', line).replace('**', '').strip()
    if not line or EMPTY.fullmatch(line):
        return False
    if '：' in line:
        return not EMPTY.fullmatch(line.split('：', 1)[1].strip())
    return True


def inventory(directory, pages=None):
    """Exact byte duplicates only; same display titles never merge identities."""
    docs, seen = [], {}
    selected, found = set(pages or []), set()
    for folder, kind in [('users', 'person'), ('groups', 'conversation'), ('topics', 'topic')]:
        for path in sorted((Path(directory) / folder).glob('*.md')):
            relative = path.relative_to(directory).as_posix()
            if selected and relative not in selected:
                continue
            found.add(relative)
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            locator = dict(kind='source_wiki', path=str(path.resolve()), sha256=sha)
            if (kind, sha) in seen:
                seen[kind, sha]['sources'].append(locator)
                continue
            lines = raw.decode('utf-8').splitlines()
            title = next((s[2:].strip() for s in lines if s.startswith('# ')), path.stem)
            doc = dict(id='wiki-' + digest([folder, path.name, sha])[:20], kind=kind,
                       title=title, filename=path.stem, lines=lines, sources=[locator],
                       empty=not any(substantive(s) for s in lines if not s.startswith('#')))
            docs.append(doc)
            seen[kind, sha] = doc
    if selected - found:
        raise ValueError(f'Wiki pages not found: {sorted(selected - found)}')
    return docs


def units_for(docs, max_chars=10000):
    """Include every source line. Split at line boundaries, never silently truncate."""
    units = []
    for doc in docs:
        if doc['empty']:
            continue
        chunk, size, start, heading = [], 0, 1, ''
        for number, line in enumerate(doc['lines'], 1):
            if chunk and size + len(line) > max_chars:
                units.append(dict(id=f'{doc["id"]}:{start}', page_id=doc['id'], kind=doc['kind'],
                                  title=doc['title'], heading=current_heading, lines=chunk))
                start, chunk, size = number, [], 0
            if not chunk:
                start = number
                current_heading = heading
            chunk.append([number, line])
            size += len(line) + 12
            if line.startswith('## '):
                heading = line[3:]
        if chunk:
            units.append(dict(id=f'{doc["id"]}:{start}', page_id=doc['id'], kind=doc['kind'],
                              title=doc['title'], heading=current_heading, lines=chunk))
    return units


def identity_candidates(doc, review, curated):
    """Return evidence of possible binding; a unique display name isn't proof."""
    if doc['kind'] == 'topic':
        return dict(status='topic', candidates=[])
    direct, names = [], []
    if doc['kind'] == 'person':
        for person in review['identities']:
            candidate = dict(id=person['id'], account=person['account'], sender_id=person['sender_id'])
            if doc['filename'] == person['sender_id']:
                direct.append(candidate)
            elif doc['title'] in person.get('names', {}):
                names.append(candidate)
    else:
        for group in review['groups']:
            if doc['filename'] == group['id'].split(':')[-1]:
                direct.append(dict(id=group['id']))
            elif doc['title'] in group.get('names', {}):
                names.append(dict(id=group['id']))
    # Preserve already reviewed mappings without promoting other matching names.
    curated_ids = []
    for person in curated.get('identities', []):
        name = person['name'].removesuffix('（本人）')
        if doc['kind'] == 'person' and (doc['filename'] == person['id'].split(':')[-1] or doc['title'] == name):
            curated_ids.append(person['id'])
    return dict(status=('curated_reference' if curated_ids else 'account_filename' if direct else
                        'name_candidates' if names else 'unresolved'),
                candidates=direct or names, curated_ids=curated_ids,
                note='显示名匹配只作候选；不同导出账号不合并；旧 Wiki 的别名不自动成为身份确认。')


def temporal_annotation(fact, organized_on):
    """No invented TTL, freshness date or current-state claims for old summaries."""
    dates = []
    for value in fact.get('reported_dates', []):
        if DATE.fullmatch(value):
            date.fromisoformat(value)
            dates.append(value)
    for key in ('valid_from', 'valid_to'):
        value = fact.get(key)
        if value:
            date.fromisoformat(value)
    if fact.get('valid_from') and fact.get('valid_to') and fact['valid_from'] > fact['valid_to']:
        raise ValueError('reversed validity interval')
    kind = fact['temporal_kind']
    state = {'historical': 'historical_only', 'event': 'event_scope_only',
             'mutable': 'current_state_unconfirmed', 'stable': 'reported_stable_fact',
             'unknown': 'applicability_unknown'}[kind]
    if fact.get('valid_to') and fact['valid_to'] < organized_on:
        state = 'ended_interval'
    return dict(temporal_kind=kind, reported_dates=sorted(set(dates)),
                observed_at=[], valid_from=fact.get('valid_from') or None,
                valid_to=fact.get('valid_to') or None, applicability=state,
                last_confirmed_at=None, organized_on=organized_on,
                known_at=None, source_time_status='secondary_summary_dates_only',
                current_state_usable=False,
                note='整理时间不等于确认时间；旧文中的当前/近期相对于原记载，不能相对于今天。')


def locator_for(doc, numbers):
    valid = sorted(set(numbers))
    if not valid or any(type(n) is not int or n < 1 or n > len(doc['lines']) for n in valid):
        raise ValueError('unknown Wiki evidence line')
    source = dict(doc['sources'][0], line=min(valid), end_line=max(valid))
    source['excerpt_sha256'] = digest([doc['lines'][n - 1] for n in valid])
    return source


def compile_library(docs, results, review, curated, refined, manifest, organized_on):
    """Join cached semantic extraction to immutable provenance and prior corrections."""
    content = deepcopy(curated)
    content.update(schema='wiki_library_v1', provenance_policy='locators_only',
                   status='selected_library_review' if manifest.get('pages') else 'full_library_review',
                   organized_on=organized_on, runtime_usable=False,
                   historical_input_status='not_admitted')
    content['source_manifest']['library'] = manifest
    content['library_pages'], content['library_claims'], content['library_gaps'] = [], [], []
    # Raw rule hits include mentions of third parties, not only actual addresses.
    # Preserve curated edges only; extraction is not review of every old rule hit.
    content['address_edges'] = deepcopy(curated.get('address_edges', []))
    for edge in content['address_edges']:
        edge['evidence'] = [source_locator(e) for e in edge.get('evidence', [])]
        edge.update(runtime_usable=False, historical_input_status='not_admitted')
    by_doc, doc_map = defaultdict(list), {d['id']: d for d in docs}
    for result in results:
        for unit in result['units']:
            by_doc[unit['page_id']].append(unit)
    for doc in docs:
        page = {k: deepcopy(v) for k, v in doc.items() if k != 'lines'}
        page['identity_binding'] = identity_candidates(doc, review, curated)
        page['gaps'], page['claim_ids'] = [], []
        if doc['empty']:
            page['gaps'].append('空白模板：没有可整理的实质内容。')
        if page['identity_binding']['status'] in ('name_candidates', 'unresolved'):
            page['gaps'].append('尚缺明确账号绑定；不因名字相同合并或继承背景。')
        for unit in by_doc[doc['id']]:
            page['gaps'].extend(unit['gaps'])
            for fact in unit['facts']:
                locator = locator_for(doc_map[unit['page_id']], fact['lines'])
                ref = 'wiki-' + digest(locator)[:20]
                content['evidence'][ref] = locator
                item = {k: deepcopy(v) for k, v in fact.items() if k not in ('lines', 'reported_dates', 'valid_from', 'valid_to')}
                item.update(id='fact-' + digest([doc['id'], fact])[:24], page_id=doc['id'],
                            subject_id=doc['id'] if fact['subject'] == '@page' else None,
                            evidence_refs=[ref], evidence_level='secondary_wiki_summary',
                            status='wiki_summary_unverified', runtime_usable=False,
                            historical_input_status='not_admitted', **temporal_annotation(fact, organized_on))
                # Old Wiki guesses and Bot-only claims cannot be upgraded by an extractor.
                source = '\n'.join(doc['lines'][n - 1] for n in fact['lines'])
                if re.search(r'待验证|未确认|未否认|推测|猜测|Bot.{0,5}(?:确认|回应|提及|猜)', source, re.I):
                    item['status'] = 'needs_verification'
                if fact['evidence_mode'] in ('inference', 'bot_only', 'uncertain', 'conflict'):
                    item['status'] = 'needs_verification'
                item['temporal_note'] = item.pop('note')
                page['claim_ids'].append(item['id'])
                content['library_claims'].append(item)
        page['gaps'] = sorted(set(page['gaps']))
        page['status'] = 'organized' if page['claim_ids'] else 'no_supported_summary'
        content['library_pages'].append(page)
        for gap in page['gaps']:
            content['library_gaps'].append(dict(page_id=page['id'], description=gap))
    facts = content['library_claims']
    content['summary'] = dict(
        source_files=sum(len(d['sources']) for d in docs), unique_documents=len(docs),
        exact_duplicate_files=sum(len(d['sources']) - 1 for d in docs),
        source_by_kind=dict(Counter(d['kind'] for d in docs for _ in d['sources'])),
        pages_with_content=sum(bool(p['claim_ids']) for p in content['library_pages']),
        empty_documents=sum(d['empty'] for d in docs), structured_summaries=len(facts),
        categories=dict(Counter(f['category'] for f in facts)),
        evidence_modes=dict(Counter(f['evidence_mode'] for f in facts)),
        applicability=dict(Counter(f['applicability'] for f in facts)),
        identity_binding=dict(Counter(p['identity_binding']['status'] for p in content['library_pages'])),
        retained_curated_claims=len(content['fact_claims']), addresses=len(content['address_edges']),
        unverified_summaries=len(facts), new_runtime_admitted_claims=0,
        gaps=len(content['library_gaps']))
    return content


def write_library(content, output):
    """Publish a complete review bundle atomically per file, outside request caches."""
    output = Path(output)
    write_json(output / 'knowledge.json', content)
    write_json(output / 'coverage.json', content['summary'])
    facts = {f['id']: f for f in content['library_claims']}
    index = ['# ' + ('选定 Wiki 整理' if content['source_manifest']['library'].get('pages') else '全量 Wiki 整理'), '',
             '保留既有核对结论，旧 Wiki 新整理内容是二手摘要，不自动成为确认事实或当前状态。',
             '背景实验继续暂停。来源仅含定位与哈希；新材料尚未进入 Gen/Judge。', '',
             '[覆盖及缺口](coverage.md) · [结构化资料](knowledge.json)', '',
             '| 文档 | 类别 | 条目数 | 身份绑定 |', '|---|---|---:|---|']
    for page in content['library_pages']:
        label = page['title'].replace('|', '\\|').replace('\n', ' ')
        index.append(f'| [{label}](pages/{page["id"]}.md) | {page["kind"]} | '
                     f'{len(page["claim_ids"])} | {page["identity_binding"]["status"]} |')
        rows = [f'# {page["title"]}', '',
                '来源为旧 Wiki 二手摘要；保留记载范围，不代表当前已经确认。', '',
                '账号绑定：' + json.dumps(page['identity_binding'], ensure_ascii=False), '']
        for cid in page['claim_ids']:
            f = facts[cid]
            rows.extend([f'## {CATEGORIES[f["category"]]}：{f["predicate"]}', '',
                         f'- 主体：{page["title"] if f["subject"] == "@page" else f["subject"]}',
                         f'- 内容：{f["value"]}', f'- 对象：{f["object_name"] or "未指定"}',
                         f'- 说话人：{f["source_speaker"] or "旧摘要未明确"}；范围：{f["scope"] or "未明确"}',
                         f'- 角色：{json.dumps(f["roles"], ensure_ascii=False)}',
                         f'- 性质：{f["evidence_mode"]}；状态：{f["status"]}',
                         f'- 时间：{f["reported_dates"]}；适用：{f["applicability"]}；'
                         f'有效区间：{f["valid_from"]} 至 {f["valid_to"]}',
                         f'- 限制：{f["limitations"]}', ''])
            for ref in f['evidence_refs']:
                ev = content['evidence'][ref]
                rows.append(f'- 来源：`{ev["path"]}:{ev["line"]}`（至第 {ev["end_line"]} 行）；SHA256 `{ev["sha256"]}`')
            rows.append('')
        rows.extend(['## 缺口', '', *['- ' + s for s in page['gaps']], ''])
        atomic_write(output / 'pages' / f'{page["id"]}.md', '\n'.join(rows))
    atomic_write(output / 'README.md', '\n'.join(index) + '\n')
    summary = ['# 覆盖与缺口', '', '文件整理完成不等于全部事实已确认或可直接用于历史评测。', '',
               '```json', json.dumps(content['summary'], ensure_ascii=False, indent=2), '```', '',
               '## 全库共性缺口', '',
               '- 旧 Wiki 多数没有逐条原聊天消息 ID；本版提供准确 Wiki 文件与行号，不能伪造原消息定位。',
               '- 显示名命中不等于身份确认；待定身份和不同导出账号保持分开。',
               '- 旧摘要里的用户确认、Bot 确认、未否认并不自动获得本次确认权限。',
               '- 可变背景缺最近确认时间；事件保留历史状态，不把整理时间当作信息更新时间。',
               '- 既有人工核对内容继续保留在 fact_claims，新增摘要位于 library_claims，不能覆盖纠正。',
               '- 每页列出内容和具体疑点；空白模板与纯噪声不会计作可用事实。', '']
    atomic_write(output / 'coverage.md', '\n'.join(summary))
