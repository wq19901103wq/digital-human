"""Reconcile repeated attributes across extraction batches using raw sources."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json

from .wiki_library import digest
from .wiki_library_extract import obj
from .wiki_structured import COMMON, TEXT, array, enum, jobs_for, attribution_issues, require_attribution

FIELDS = {k: v for k, v in COMMON.items() if k != 'fact_ids'}
SCHEMA = obj(dict(reviews=array(obj(dict(record_id=TEXT,
    decision=enum('keep', 'correct', 'omit', 'unresolved'), reason=TEXT,
    value=TEXT, **FIELDS)))))
PROMPT = """核对自动 Wiki 中同一主体同一属性的跨批记录，依据给出的原始聊天逐项纠错。
仅使用输入；不调用工具。自动摘要、重复次数、旧记录的肯定语气均不是证据。
每个 record_id 输出一个决定，不增加记录、不合并不同身份，不改变主体或字段。
keep 保留原记录；correct 修正值、证据性质、归属说明或时期；omit 删除误归属/无依据/纯噪声；
unresolved 保留不能裁定的冲突，但 value 必须写成有保留的表述，mode=uncertain或conflict、polarity=uncertain。
同一字段的多条记录需结合原始证据一起看：先区分时间变化和真正矛盾，不以最新提及覆盖历史。
判断假设、反事实、未实现选项、求职意向与实际经历的区别；不要把一个可能性变成已发生事实。
提问、对方未否认、调侃、复述第三人说法不能当作该主体自述。self_report 必须有该主体的
实际陈述；说话人的原始元数据优先于自动摘要。引文中显示名不能单独证明昵称与账号的绑定。
保留有依据的具体细节；跨记录补证必须列出真正支持新值的原始行号。
valid_from/to 只填明确的有效日期，不拿消息日期当有效日期；不明确填空串。
每条输出 lines 必须来自输入且至少有一条；omit/unresolved 的引用用于解释不能确认的原因。
value/attribution/reason 中文转述，不复制聊天摘录（姓名、称呼、机构名等原子值除外）。
只输出 schema JSON。\n"""


def groups_for(data):
    groups = defaultdict(list)
    for record in data['attributes']:
        groups[(record['subject_id'], record['field'])].append(record)
    return [dict(id=digest(key), lines=sorted({int(n) for r in records for n in r['evidence_refs']}),
                 records=records) for key, records in sorted(groups.items()) if len(records) > 1]


def validate(value, records, rows, data=None):
    expected = {r['id'] for r in records}
    received = [r['record_id'] for r in value['reviews']]
    if set(received) != expected or len(received) != len(expected):
        raise ValueError('every input record needs exactly one review')
    allowed = {r['line'] for r in rows}
    for review in value['reviews']:
        if not review['lines'] or not set(review['lines']) <= allowed:
            raise ValueError(f'invalid source lines for {review["record_id"]}; allowed={sorted(allowed)}')
        if review['decision'] == 'unresolved' and (review['mode'] not in ('uncertain', 'conflict')
                                                  or review['polarity'] != 'uncertain'):
            raise ValueError('unresolved attributes must remain explicitly uncertain')
        for key in ('valid_from', 'valid_to'):
            if review[key] and date.fromisoformat(review[key]).isoformat() != review[key]:
                raise ValueError('effective dates must be exact ISO dates or empty')
        if review['valid_from'] and review['valid_to'] and review['valid_from'] > review['valid_to']:
            raise ValueError('effective interval reversed')
        if review['decision'] != 'omit' and not review['value'].strip():
            raise ValueError('retained attributes need a value')
    if data is not None:
        originals = {r['id']: r for r in records}
        retained = []
        for review in value['reviews']:
            original = originals[review['record_id']]
            if review['decision'] == 'keep':
                retained.append(original)
            elif review['decision'] != 'omit':
                retained.append(dict(original, **{k: review[k] for k in FIELDS}, value=review['value']))
        require_attribution(attribution_issues(dict(entities=data['entities'], attributes=retained),
                                              rows, check_entities=False))


def extract(client, directory, data, groups, rows):
    from .wiki_repair import _cached_call, _render
    records = [r for group in groups for r in group['records']]
    subjects = {r['subject_id'] for r in records}
    payload = dict(subject=data['subject'], entities=[e for e in data['entities'] if e['id'] in subjects],
        records=records, columns=['source_line', 'time', 'sender_id', 'display_name', 'kind', 'text', 'chat_id'],
        messages=[_render(r) + [r['chat_id']] for r in rows])
    return _cached_call(client, directory, PROMPT + json.dumps(payload, ensure_ascii=False), SCHEMA,
                        lambda value: validate(value, records, rows, data))['reviews']


def apply_reviews(data, reviews, rows):
    """Apply model decisions deterministically, retaining lineage and locator-only evidence."""
    by_id = {r['record_id']: r for r in reviews}
    by_line = {r['line']: r for r in rows}
    result, audit = [], []
    for original in data['attributes']:
        review = by_id.get(original['id'])
        if review is None:
            result.append(original)
            continue
        record = original
        refs = sorted(set(map(str, review['lines'])), key=int)
        for ref in refs:
            row = by_line[int(ref)]
            data['evidence'][ref] = {k: v for k, v in row.items() if k not in ('text', 'sender', 'chat_type')} | dict(
                path=data['coverage']['path'], sha256=data['coverage']['sha256'], kind='raw_message')
        if review['decision'] in ('correct', 'unresolved'):
            record = dict(original, **{k: review[k] for k in FIELDS if k != 'lines'}, value=review['value'])
            record.update(evidence_refs=refs, reconciled_from=original['id'],
                chat_ids=sorted({by_line[int(n)]['chat_id'] for n in refs}),
                first_observed_at=min(by_line[int(n)]['timestamp'] for n in refs),
                last_observed_at=max(by_line[int(n)]['timestamp'] for n in refs))
            record['id'] = 'attribute-' + digest({k: v for k, v in record.items() if k != 'id'})[:20]
        if review['decision'] != 'omit':
            result.append(record)
        audit.append(dict(record_id=original['id'], result_id=record['id'] if review['decision'] != 'omit' else '',
                          decision=review['decision'], reason=review['reason'], evidence_refs=refs))
    data['attributes'] = result
    data['attribute_reconciliation'] = dict(scope='repeated_subject_field_only',
        counts=dict(Counter(r['decision'] for r in audit)), reviews=audit,
        unreviewed_attributes=sum(r['id'] not in by_id for r in result if 'reconciled_from' not in r))
    represented = {f for key in ('attributes', 'relations', 'events', 'addresses') for r in data[key] for f in r['fact_ids']}
    for review in data['reviews']:
        if review['disposition'] == 'represented' and review['fact_id'] not in represented:
            review.update(disposition='unresolved', reason='跨批属性复核后没有保留对应记录，原摘要不能作为证据。')
    return data


def reconcile(client, directory, data, rows, workers, progress, status):
    groups = groups_for(data)
    jobs = list(jobs_for(groups, rows, count=4))
    progress.update(stage='reconciling_attributes', reconciliation_batches=len(jobs), reconciliation_completed=0)
    status()
    outputs = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(extract, client, directory, data, gs, rs): i for i, (gs, rs) in enumerate(jobs)}
        for future in as_completed(pending):
            i = pending[future]
            try:
                outputs[i] = future.result()
                progress['reconciliation_completed'] += 1
            except Exception as exc:
                progress['failed'] += 1
                progress['errors'].append(dict(stage='reconciling_attributes', batch=i, error=str(exc)[-1000:]))
            status()
    if progress['failed']:
        return None
    return apply_reviews(data, [r for i in sorted(outputs) for r in outputs[i]], rows)
