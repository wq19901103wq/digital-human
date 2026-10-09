"""Refine an existing alias inventory into grouped, source-backed confirmation pages."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .wiki_aliases import FLAG_LABELS, alias_parts, cell, clean, key
from .wiki_alias_context import discover_context
from .wiki_addresses import write_addresses


LABELS = {**FLAG_LABELS, 'self_stated_review_required': '本人在聊天中自述称呼，仍待确认',
          'address_target_unconfirmed': '私聊重复称呼；可能提及第三人，指向待确认',
          'generic_address': '泛称或关系称呼，仅限对应说话人及聊天范围'}
SOURCES = dict(existing_alias_map='旧别名表', old_suggestion='旧自动建议', wiki_claim='Wiki',
               self_introduction='自我介绍', resolved_quote_name='匹配引用名',
               context_self_name='聊天自述称呼', private_vocative='私聊重复称呼')
EMPTY = {'', '暂无', '无', '未知', '待补充', '待验证', '未确认', '身份不明', '不详', '我', '你', '他', '她', '本人',
         '别名', '……', '...', '男性', '女性', '志愿者', '账号', '曾改名', '群友称呼', '被@提及'}


def normalized_names(claim):
    """Separate uncertainty annotations from names, without discarding the raw claim."""
    value = clean(claim['alias'])
    # Quoted display names are opaque: a slash may be an actual part of the name.
    literal = any(ev['kind'] in {'resolved_quote_name', 'self_introduction', 'context_self_name', 'private_vocative'}
                  for ev in claim['evidence'])
    if literal:
        yield value, []
        return
    for part in alias_parts(value):
        # The original parser may already have removed a trailing closing bracket.
        notes = re.findall(r'[（(\[]([^）)\]]+)(?:[）)\]]|$)', part)
        name = re.sub(r'[（(\[][^）)\]]*(?:[）)\]]|$)', '', part).strip(' \t（）()[]。，：:')
        name = re.sub(r'^(?:昵称|别称|微信名|花名|本名|主名|曾用群昵称|常用标识)[：:]\s*', '', name)
        yield name, notes


def exclusion_reason(claim, name):
    owner = clean(claim['owner'])
    if owner in EMPTY or name in EMPTY:
        return '空值、占位符或无确定指向的代词'
    if re.match(r'^\d{1,2}:\d{2}|^\d{4}[-年/]\d|^\d{1,2}月\d', owner):
        return '人物字段误提取了时间或记录标题'
    if re.search(r'你已被|移出群聊|撤回了|修改群名|邀请.*加入|对话已记录|滚动摘要', name + owner):
        return '系统通知或摘要标题'
    if name == owner:
        return '与原条目同名，无需重复列为别名'
    literal = any(ev['kind'] in {'resolved_quote_name', 'self_introduction', 'context_self_name', 'private_vocative'}
                  for ev in claim['evidence'])
    if not literal and re.search(r'身份|男性志愿者|协助搬运|后被|疑似|可能|与.+关系|来源[：:]|待确认|待补充|'
                                r'对话中|回应确认|无其他别名|被群友|群友@|备注[：:]|曾戏称|发言时|被@询问|'
                                r'多次出现|群内房间号|同行人|在家庭群|推测|但记录|具体不明确|是.+(?:父|母|姨|舅)', name):
        return '解释句或未解析的人物关系'
    if not literal and (len(name) > 32 or re.match(r'^\d{4}[-年/]\d', name)
                        or re.fullmatch(r'[\d\s\-~～至/:年月日.]+', name)):
        return '日期、数值或长段说明；缺少直接称呼证据'
    return ''


def refine_review(review_path, messages, decisions_path=None, self_account=None):
    raw = Path(review_path).read_bytes()
    review = json.loads(raw)
    decisions = json.loads(Path(decisions_path).read_text()) if decisions_path else {'decisions': []}
    context = discover_context(review, Path(messages), decisions, self_account)
    people, excluded, counts = {}, [], Counter()
    for claim in [*review['claims'], *context['claims']]:
        if claim['kind'] != 'person':
            continue
        for name, notes in normalized_names(claim):
            reason = exclusion_reason(claim, name)
            if reason:
                excluded.append(dict(claim_id=claim['id'], owner=claim['owner'], alias=name,
                                     raw_alias=claim['alias'], reason=reason))
                continue
            counts['retained_assertions'] += 1
            owner = clean(claim['owner'])
            person = people.setdefault(owner, dict(owner=owner, display_group_only=True, aliases={}))
            item = person['aliases'].setdefault(name, dict(alias=name, assertions=[], notes=[], flags=[], sources=[]))
            item['assertions'].append(claim)
            item['notes'] = sorted(set(item['notes'] + notes))
            item['flags'] = sorted(set(item['flags'] + claim.get('flags', [])))
            item['sources'] = sorted(set(item['sources'] + [e['kind'] for e in claim['evidence']]))
    new_pairs = set()
    name_counts = Counter()
    for identity in review['identities']:
        for name, evidence in identity['names'].items():
            name_counts[name] += evidence['count']
    for owner, person in people.items():
        person['observed_messages_by_display_name'] = name_counts[owner]
        for name, item in person['aliases'].items():
            item['new_context_pair'] = all(s in {'context_self_name', 'private_vocative'} for s in item['sources'])
            if item['new_context_pair']:
                new_pairs.add((owner, name))
    counts.update(raw_person_claims=sum(c['kind'] == 'person' for c in review['claims']),
                  messages_scanned=context['counts']['messages_scanned'],
                  grouped_person_entries=len(people), grouped_alias_pairs=sum(len(p['aliases']) for p in people.values()),
                  excluded_fragments=len(excluded), new_context_pairs=len(new_pairs),
                  context_claims=len(context['claims']), weak_context_candidates=len(context['weak_candidates']))
    counts['directed_address_records'] = len(context['addresses'])
    counts['self_to_other_address_records'] = sum(r['direction'] == 'self_to_other' for r in context['addresses'])
    counts['combined_duplicate_assertions'] = counts['retained_assertions'] - counts['grouped_alias_pairs']
    return dict(schema=3, runtime_usable=False, status='pending_user_review',
                source_review=dict(path=str(Path(review_path).resolve()), sha256=hashlib.sha256(raw).hexdigest()),
                decisions=decisions, people=people, excluded=excluded, context=context,
                summary=dict(counts), group_claims=[c for c in review['claims'] if c['kind'] == 'group'])


def example_text(ev):
    if ev.get('timestamp') is not None:
        date = datetime.fromtimestamp(ev['timestamp'], timezone.utc).strftime('%Y-%m-%d')
        return f'{date} {ev.get("chat_name", "")} / {ev.get("sender", "")}：“{ev.get("text", "")[:250]}”（行 {ev.get("line")}）'
    return f'{ev.get("path", "")}:{ev.get("line", ev.get("location", ""))}：{ev.get("raw", "")[:250]}'


def evidence_order(ev):
    return {'context_self_name': 0, 'self_introduction': 1, 'resolved_quote_name': 2,
            'private_vocative': 3, 'existing_alias_map': 4, 'wiki_claim': 5}.get(ev['kind'], 6)


def item_doubts(item):
    flags = [LABELS.get(f, f) for f in item['flags']]
    if any(a.get('preference') == 'disliked' for a in item['assertions']):
        flags.append('本人明确不喜欢此称呼，不能当作推荐称呼')
    return '；'.join(flags) or '尚待用户确认'


def confirmation_page(review, output):
    """Generate a bounded first page; full evidence stays in the per-person pages."""
    rows = ['# 别名重点确认稿', '',
            '本页由清单自动生成；展示顺序不代表身份已确认。完整候选见 [人物全表](people.md)，'
            '点击人物可看原话、账号、聊天范围及疑点。', '', '## 已确认', '']
    for decision in review['decisions'].get('decisions', []):
        rows.append(f'- **{cell("、".join(decision["aliases"]))} → {cell(decision["canonical_name"])}**；'
                    f'仅限 {cell(decision["scope"]["chat_name"])}；{cell(decision["source"]["answer"])}。')
    rows += ['', '## 仍有账号归属冲突', '', '以下各账号保持分开，不因已有 Wiki 记载就合并。', '']
    for group in review['decisions'].get('pending_groups', []):
        names = [f'[{cell(name)}](people/{key(name)}.md)' if name in review['people'] else cell(name)
                 for name in group['names']]
        rows.append('- ' + ' / '.join(names) + '：待确认是否同一人；来源和冲突见对应人物页。')
    rows += ['', '## 聊天中直接说过的称呼', '',
             '自述通常比间接称呼更直接，但也可能是玩笑、曾用名或不喜欢的叫法；以下均待确认。', '',
             '| 原人物条目 | 称呼 | 原话依据 | 疑点 |', '|---|---|---|---|']
    for owner, person in sorted(review['people'].items()):
        for alias, item in sorted(person['aliases'].items()):
            evidence = [ev for a in item['assertions'] for ev in a['evidence'] if ev['kind'] == 'context_self_name']
            if evidence:
                rows.append('| ' + ' | '.join([f'[{cell(owner)}](people/{key(owner)}.md)', cell(alias),
                    cell(example_text(evidence[0])), cell(item_doubts(item))]) + ' |')
    rows += ['', '## 旧别名表中优先审阅的人物', '',
             '选取旧别名表有记录的人物，按同显示名的历史消息数排序，最多 20 项。'
             '这里只合并重复展示，不合并账号；其他人物在完整表中。', '',
             '| 原人物条目 | 旧表候选称呼 | 来源与疑点 |', '|---|---|---|']
    ordered = sorted(review['people'].values(), key=lambda p: (-p['observed_messages_by_display_name'], p['owner']))
    shown = 0
    for person in ordered:
        items = [i for i in person['aliases'].values() if 'existing_alias_map' in i['sources']]
        if not items:
            continue
        flags = sorted({LABELS.get(f, f) for i in items for f in i['flags']})
        owner = person['owner']
        rows.append('| ' + ' | '.join([f'[{cell(owner)}](people/{key(owner)}.md)',
            cell('、'.join(sorted(i['alias'] for i in items))),
            cell('旧别名表；' + ('；'.join(flags) or '尚待确认'))]) + ' |')
        shown += 1
        if shown == 20:
            break
    rows += ['', '[谁怎么称呼谁](addresses.md) · [聊天上下文全部新增及补证](context.md) · [按人物完整确认表](people.md) · [整理数量及边界](README.md)', '',
             '用户确认及导出时显示名不自动代表历史时刻已知；本稿保持 review-only，不接入模型。']
    (output / 'CONFIRM.md').write_text('\n'.join(rows) + '\n')


def write_refined(review, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'people').mkdir()
    (output / 'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    (output / 'decisions.json').write_text(json.dumps(review['decisions'], ensure_ascii=False, indent=2) + '\n')
    intro = ('按原人物条目分组仅为方便审阅，不表示同名账号合并。旧 Wiki 和旧建议不是本轮确认。'
             '引用显示名可能来自导出时备注；记录时间不等于当时已知。身份、范围均保留在逐条来源中。')
    index = ['# 按人物整理的别名确认表', '', intro, '',
             '| 原人物条目 | 候选数 | 别名候选（全部） |', '|---|---:|---|']
    additions = ['# 聊天上下文补充', '', '自述比私聊称呼证据更直接；私聊称呼可能说的是第三人，仍待确认。', '',
                 '| 原人物条目 | 称呼 | 与旧清单关系 | 依据 | 疑点 |', '|---|---|---|---|---|']
    for owner, person in sorted(review['people'].items()):
        slug = key(owner)
        aliases = person['aliases']
        index.append(f'| [{cell(owner)}](people/{slug}.md) | {len(aliases)} | {cell("、".join(sorted(aliases)))} |')
        rows = [f'# {owner}：别名确认稿', '', intro, '', '## 相关用户确认', '']
        related = [d for d in review['decisions'].get('decisions', [])
                   if owner in [d['canonical_name'], *d.get('aliases', [])]]
        for decision in related:
            rows += [f'- **已确认**：{cell("、".join(decision["aliases"]))} → {cell(decision["canonical_name"])}；'
                     f'范围：{cell(decision["scope"]["chat_name"])}；依据：{cell(decision["source"]["answer"])}。']
        if not related:
            rows += ['此人物条目尚无本轮用户确认。']
        for alias, item in sorted(aliases.items()):
            evs = sorted((ev for a in item['assertions'] for ev in a['evidence']), key=evidence_order)
            flags = item_doubts(item)
            rows += ['', f'## {alias}', '', f'- 来源：{cell("、".join(SOURCES.get(s, s) for s in item["sources"]))}',
                     f'- 疑点：{cell(flags or "尚待用户确认")}', f'- 原注释：{cell("；".join(item["notes"]) or "无")}',
                     '- 对应关系及范围（各来源独立保留）：']
            for a in item['assertions']:
                target = a.get('bound_identity') or a.get('suggested_identity')
                binding = '原消息发送者' if a.get('bound_identity') else '推测称呼对象'
                mapping = f'{binding} {target}' if target else '未绑定账号；同名账号不能当作身份确认'
                rows.append(f'  - {cell(a.get("scope_chat_id") or a.get("scope") or "来源未限定范围")}：'
                            f'{cell(mapping)}；原条目 {a["id"]}')
            rows += ['- 代表依据（完整证据和消息 ID 见 review.json）：']
            rows.extend(f'  - {cell(example_text(ev))}' for ev in evs[:3])
            contextual = [ev for ev in evs if ev['kind'] in {'context_self_name', 'private_vocative'}]
            if contextual:
                additions.append('| ' + ' | '.join([f'[{cell(owner)}](people/{slug}.md)', cell(alias),
                    '新增对应关系' if item['new_context_pair'] else '为已有候选补证',
                    cell(example_text(contextual[0])), cell(flags or '待确认')]) + ' |')
        (output / 'people' / f'{slug}.md').write_text('\n'.join(rows) + '\n')
    (output / 'people.md').write_text('\n'.join(index) + '\n')
    (output / 'context.md').write_text('\n'.join(additions) + '\n')
    exclusions = ['# 移出确认表的原始片段', '', '仅清理展示；未删除原始清单与来源。', '',
                  '| 原人物条目 | 原别名 | 原因 | 原条目 ID |', '|---|---|---|---|']
    exclusions += ['| ' + ' | '.join(cell(c[k]) for k in ('owner', 'raw_alias', 'reason', 'claim_id')) + ' |'
                   for c in review['excluded']]
    (output / 'excluded.md').write_text('\n'.join(exclusions) + '\n')
    summary = ['# 别名确认稿（整理版）', '', intro, '',
               '- [先看重点确认稿](CONFIRM.md)', '- [按人物列出的完整确认表](people.md)',
               '- [聊天上下文新增称呼及补证](context.md)', '- [移出的误提取片段及原因](excluded.md)',
               '- [本人如何称呼对方，以及对方如何称呼本人](addresses.md)',
               '- [完整结构化来源与疑点](review.json)', '- [保留的用户确认](decisions.json)', '',
               '## 已确认与待定', '']
    for decision in review['decisions'].get('decisions', []):
        summary += [f'- 已确认：{cell("、".join(decision["aliases"]))} → {cell(decision["canonical_name"])}；'
                    f'范围：{cell(decision["scope"]["chat_name"])}。']
    summary += ['- 其余对应关系均待确认；不继承旧建议的确定性，不因同名合并账号。', '', '## 整理数量', '']
    stat_labels = dict(raw_person_claims='原始人物别名陈述', messages_scanned='补扫历史消息',
        retained_assertions='清理后保留的来源陈述', grouped_person_entries='原人物条目数（非确认人数）',
        grouped_alias_pairs='合并展示后的「人物条目—称呼」关系', excluded_fragments='移出确认表的误提取/同名片段',
        new_context_pairs='聊天补扫新增的「人物条目—称呼」关系', context_claims='聊天补扫保留的分范围陈述',
        weak_context_candidates='仅出现一次的私聊称呼（留档待查）', combined_duplicate_assertions='折叠的重复来源陈述',
        directed_address_records='按说话人、对象、聊天范围和用途区分的称呼记录',
        self_to_other_address_records='其中本人称呼或提及对方的记录（非确认数）')
    summary += [f'- {stat_labels.get(name, name)}：{value:,}' for name, value in review['summary'].items()]
    summary += ['', '重复数按同一原人物条目 + 同一规范化称呼计算；仅合并展示，证据及不同账号/范围不合并。',
                '聊天补充覆盖显式自述称呼和重复私聊称呼。单次私聊称呼留在 JSON 的 weak_candidates 中；'
                '群聊中隐含外号和长篇转述尚未全面语义识别，不能声称已找全。',
                '本稿不接入模型，不修改原 Wiki；新增称呼还可能是玩笑或不喜欢的称呼，需结合所附上下文判断。']
    (output / 'README.md').write_text('\n'.join(summary) + '\n')
    confirmation_page(review, output)
    write_addresses(review, output)
