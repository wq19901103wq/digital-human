"""Source-grounded revision with evidence for roles and individual claims."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json

from ..iteration.storage import write_json
from . import wiki_revision as revision, wiki_structured as structured
from .wiki_library_extract import obj
from .wiki_repair import _cached_call, _render

SCHEMA = deepcopy(revision.SCHEMA)
DECISION = SCHEMA['properties']['decisions']['items']
ROLE_CHECKS = structured.array(obj(dict(
    role_key=structured.TEXT, described_entity_id=structured.TEXT,
    support=structured.enum('content', 'speaker_metadata', 'uncertain', 'mismatch'),
    lines=structured.array(dict(type='integer')), explanation=structured.TEXT)))
DECISION['properties']['role_checks'] = ROLE_CHECKS
DECISION['required'].append('role_checks')
COMPARISON = obj(dict(present=structured.enum('yes', 'no'), reference=structured.TEXT,
    reference_entity_id=structured.TEXT, support=structured.enum('explicit', 'uncertain', 'not_applicable'),
    lines=structured.array(dict(type='integer'))))
ASSERTION = obj(dict(claim=structured.TEXT, support=structured.enum('content', 'uncertain', 'mismatch'),
    actor_entity_id=structured.TEXT,
    source_basis=structured.enum('self_report', 'explicit_reference', 'context', 'uncertain'),
    lines=structured.LINES, comparison=COMPARISON))
SLOT_CHECKS = structured.array(obj(dict(slot_key=structured.TEXT, assertions=structured.array(ASSERTION))))
DECISION['properties']['slot_checks'] = SLOT_CHECKS
DECISION['required'].append('slot_checks')
SCHEMA['properties']['replacement_checks'] = structured.array(obj(dict(
    record_key=structured.TEXT, role_checks=ROLE_CHECKS, slot_checks=SLOT_CHECKS)))
SCHEMA['required'].append('replacement_checks')
PROMPT = revision.PROMPT + """
本轮必须将消息作者与消息内容描述的人分开核对。role_claims 展开了记录实际绑定的实体；
旧 attribution 文案即使正确，也不能替代 subject_id 等实际结构化角色的核实。
对每条记录，先从原消息独立判断每个角色实际指向谁，再与 role_claims 比较。
role_checks 对输入的每个 role_key 恰好输出一次：described_entity_id 是原文真正描述的
输入实体 ID；不在输入实体中或无法确认则空串。lines 指出支持该角色判断的原消息行。
support=content 表示内容支持该实体承担该角色；speaker_metadata 只证明消息作者；
mismatch 表示内容指向另一个人；uncertain 表示指向不明。explanation 用中文转述依据，
不要复制原文。不得以“由该账号发言”作为属性主体、关系对象或事件参与者的充分依据。
除 speaker_id 外，keep 的每个角色必须有 content 证据且与原绑定一致；仅有作者证据，
或发现主体不一致，必须 correct、unresolved 或 omit。speaker_id 可用作者元数据支持。
correct 的替代同样遵循所述主体：不明身份的第三方用无账号实体，不能转为发言人自述。
role_checks 检查的是输入记录，不是替代记录；其他 decisions/corrections 规则不变。

角色一致仍不足以保留整条记录。content_slots 列出必须逐项核实的语义字段名、正文、时间、细节和无账号实体标签。
attributes.field、events.event_type 和 details/序号/field 都表达事实含义，不是免核实的分类标签。
field 与对应 value 必须作为完整语义一起有据：数值或时长有来源，不代表其所标属性也有来源。
核实 field 时说明原文支持的是哪种属性及其值；核实 value 时也须确认该值属于对应 field。
event_type 须与原文所述事件、description 和 details 一致，不能只因描述有据就保留错误事件类型。
slot_checks 对每个 slot_key 恰好一次，assertions 将该字段内的每个独立动作或断言分别列出，
不得将多人的同话题经历合成一个断言，也不得用一句“整体有依据”代替逐项检查。
每个 assertion.claim 中文转述一件事；lines 仅支持这一件事，不复用整条记录的引用合集。
保留记录的断言来源须包含在该记录 lines 内；实体标签可引用该实体自己的来源。
若需要额外上下文作证，correct 并将相应原消息行纳入替代记录的 lines，不能 keep 旧引用。
actor_entity_id 填这件事真正所属的输入实体 ID（替代检查用 corrections 的 ref），未知留空。
source_basis=self_report 仅用于消息作者自己的事，所列每行的作者账号均应是该 actor；
引文里的“我”不属于外层作者，需用 quoted 语义对应的 explicit_reference；转述也用
explicit_reference 并保留指代依据。context 只用于日期、否定、状态等不独立归属于某人的说明，
不得把某人的动作改成 context 来绕过归属。无法核实 actor 用 uncertain，不能凭已有角色补齐。
support=content 表示该原子断言全部有据；混入错误内容用 mismatch，指向不明用 uncertain。
同字段的原子断言须覆盖该字段全部信息，尤其检查 time_scope 是否偷带其他人的事件。

每个含比较的断言须 comparison.present=yes：reference 转述明确的比较基准，
reference_entity_id 在涉及已知人物时填其 ID/ref，lines 是明确基准的来源。
省略的比较基准不能自动补成发言者、前一个人或本文人物；无法从原文确认则
comparison.support=uncertain，reference/reference_entity_id 留空。若无比较，present=no，
support=not_applicable，其余字段为空。实体 label 中的关系、比较与描述也需要原始依据，
不能凭旧 label 为比较参照作证。保留比较幅度但未知参照时应明确未知，不生成确定的人物绑定。
keep 只允许每项原子断言 content、主体有据且每个比较基准 explicit；否则 correct/unresolved/omit。
可以拆出有依据的部分、删掉无依据修饰，不能因为一个片段正确就保留复合记录。

replacement_checks 对 corrections 各表的每条替代记录恰好一次，record_key 为表名/从0起的序号，
如 events/0。role_checks 按替代记录的实际 subject_ref/object_ref/target_ref、
participants/序号/角色逐一检查，described_entity_id 使用 corrections 实体 ref。
替代的角色不得只有发言者证据，也不能保留已确认错配的参与者。unresolved 可保留 uncertain。
按相同规则检查其 content_slots（字段路径与输入一致，实体用 ref，
无账号实体标签路径为 entities/<ref>/label）。correct 的替代必须全部有据；
unresolved 替代可含 uncertain 断言或未知比较基准，但不能肯定错归。没有替代时返回空数组。
"""


def role_claims(record, entities):
    """Expand exact structural endpoints, independently of attribution prose."""
    refs = [(key, record[key]) for key in ('subject_id', 'object_id', 'target_id', 'speaker_id',
                                        'subject_ref', 'object_ref', 'target_ref')
            if record.get(key)]
    refs += [(f'participants/{i}/{p["role"]}', p.get('entity_id', p.get('entity_ref')))
             for i, p in enumerate(record.get('participants', []))]
    return [dict(role_key=key, entity_id=eid, entity=entities[eid]) for key, eid in refs]


def content_slots(record, entities):
    """Expose semantic keys and values even when endpoint identities are correct."""
    fields = ('field', 'value', 'predicate', 'detail', 'event_type', 'description', 'status', 'time_scope',
              'valid_from', 'valid_to', 'attribution', 'term', 'usage')
    slots = [dict(slot_key=key, value=record[key]) for key in fields if record.get(key)]
    slots.extend(dict(slot_key=f'details/{i}/{key}', value=item[key])
                 for i, item in enumerate(record.get('details', [])) for key in ('field', 'value'))
    refs = {record[key] for key in ('subject_id', 'object_id', 'target_id', 'speaker_id',
            'subject_ref', 'object_ref', 'target_ref') if record.get(key)}
    refs.update(p.get('entity_id', p.get('entity_ref')) for p in record.get('participants', []))
    # Account labels are recovered from sender metadata; unaccounted labels can
    # encode unproven relationship/comparison claims and need their own review.
    slots.extend(dict(slot_key=f'entities/{ref}/label', value=entities[ref]['label'])
                 for ref in sorted(refs) if not entities[ref]['account'])
    return slots


def _validate_slots(checks, record, entities, by_line, *, retained=True, uncertain=False):
    expected = {slot['slot_key'] for slot in content_slots(record, entities)}
    if len(checks) != len(expected) or {c['slot_key'] for c in checks} != expected:
        raise ValueError('every content slot needs exactly one source check')
    bound = {record[key] for key in ('subject_id', 'object_id', 'target_id', 'speaker_id',
        'subject_ref', 'object_ref', 'target_ref') if record.get(key)}
    bound.update(p.get('entity_id', p.get('entity_ref')) for p in record.get('participants', []))
    for check in checks:
        if not check['assertions']:
            raise ValueError('content slots require atomic assertions')
        sources = set(record['lines'])
        if check['slot_key'].startswith('entities/'):
            entity = entities[check['slot_key'].split('/')[1]]
            sources.update(map(int, entity.get('lines', entity.get('evidence_refs', []))))
        for assertion in check['assertions']:
            lines = assertion['lines']
            if not lines or not set(lines) <= by_line.keys() or not assertion['claim'].strip():
                raise ValueError('atomic assertion requires available sources and a claim')
            if retained and not set(lines) <= sources:
                raise ValueError('retained assertion sources must be preserved in the record evidence')
            actor = assertion['actor_entity_id']
            if actor and actor not in entities:
                raise ValueError('atomic assertion refers to an unknown actor')
            if retained and actor and actor not in bound:
                raise ValueError('atomic assertion actor is outside the record roles')
            basis = assertion['source_basis']
            if assertion['support'] == 'content':
                if basis == 'uncertain' or (basis != 'context' and not actor):
                    raise ValueError('supported assertion needs its actual actor and source basis')
                if basis == 'self_report':
                    account = entities[actor]['account']
                    if not account or any(by_line[n]['sender_id'] != account or
                            by_line[n].get('message_kind') == 'forward' for n in lines):
                        raise ValueError('each self-report assertion needs its own author evidence')
            elif retained and (not uncertain or assertion['support'] == 'mismatch'):
                raise ValueError('retained content cannot contain unsupported atomic assertions')
            comparison = assertion['comparison']
            if comparison['reference_entity_id'] and comparison['reference_entity_id'] not in entities:
                raise ValueError('comparison refers to an unknown reference entity')
            if not set(comparison['lines']) <= by_line.keys():
                raise ValueError('comparison requires available source lines')
            if retained and not set(comparison['lines']) <= sources:
                raise ValueError('comparison sources must be preserved in the record evidence')
            if comparison['present'] == 'no':
                if (comparison['support'] != 'not_applicable' or comparison['reference'] or
                        comparison['reference_entity_id'] or comparison['lines']):
                    raise ValueError('absent comparison cannot claim a reference')
            elif comparison['support'] == 'explicit':
                if not comparison['reference'].strip() or not comparison['lines']:
                    raise ValueError('explicit comparison needs a source-grounded reference')
            elif retained and (not uncertain or comparison['support'] != 'uncertain' or
                  comparison['reference'] or comparison['reference_entity_id']):
                raise ValueError('implicit comparison reference must remain unresolved')


def _validate_roles(checks, record, entities, by_line, *, retained=True, uncertain=False):
    expected = {r['role_key']: r['entity_id'] for r in role_claims(record, entities)}
    if len(checks) != len(expected) or {r['role_key'] for r in checks} != set(expected):
        raise ValueError('every structural role needs exactly one source check')
    for check in checks:
        if not check['lines'] or not set(check['lines']) <= by_line.keys():
            raise ValueError('role check requires available source lines')
        actual = check['described_entity_id']
        if actual and actual not in entities:
            raise ValueError('role check refers to unknown input entity')
        if not check['explanation'].strip():
            raise ValueError('role check requires a source explanation')
        if retained:
            allowed = ('content', 'speaker_metadata') if check['role_key'] == 'speaker_id' else ('content',)
            supported = check['support'] in allowed and actual == expected[check['role_key']]
            unresolved = uncertain and check['support'] == 'uncertain' and actual in (
                '', expected[check['role_key']])
            if not (supported or unresolved):
                raise ValueError('retained roles need semantic support, not merely message authorship')
            if not set(check['lines']) <= set(record['lines']):
                raise ValueError('role sources must be preserved in the record evidence')


def validate(value, batch, rows, data):
    revision.validate(value, batch, rows, data['subject'], data)
    entities = {e['id']: e for e in data['entities']}
    originals = {r['id']: r for r in batch}
    by_line = {r['line']: r for r in rows}
    for decision in value['decisions']:
        _validate_roles(decision['role_checks'], originals[decision['record_id']], entities,
                        by_line, retained=decision['action'] == 'keep')
        _validate_slots(decision['slot_checks'], originals[decision['record_id']], entities,
                        by_line, retained=decision['action'] == 'keep')
    corrections = value['corrections']
    replacements = {f'{key}/{i}': record for key in revision.TABLES
                    for i, record in enumerate(corrections[key])}
    checks = value['replacement_checks']
    if len(checks) != len(replacements) or {c['record_key'] for c in checks} != replacements.keys():
        raise ValueError('every replacement needs exactly one content audit')
    replacement_entities = {e['ref']: e for e in corrections['entities']}
    actions = {d['record_id']: d['action'] for d in value['decisions']}
    for check in checks:
        record = replacements[check['record_key']]
        uncertain = actions[record['fact_ids'][0]] == 'unresolved'
        _validate_roles(check['role_checks'], record, replacement_entities, by_line, uncertain=uncertain)
        _validate_slots(check['slot_checks'], record, replacement_entities, by_line,
                        uncertain=uncertain)


def extract(client, cache, data, batch, rows):
    entities = {e['id']: e for e in data['entities']}
    expanded = [dict(r, record_id=r['id'], role_claims=role_claims(r, entities),
                     content_slots=content_slots(r, entities)) for r in batch]
    refs = {c['entity_id'] for r in expanded for c in r['role_claims']}
    payload = dict(subject=data['subject'], records=expanded,
        entities=[e for e in data['entities'] if e['id'] in refs],
        columns=['source_line', 'time', 'sender_id', 'display_name', 'kind', 'text', 'chat_id'],
        messages=[_render(r) + [r['chat_id']] for r in rows])
    return _cached_call(client, cache, PROMPT + json.dumps(payload, ensure_ascii=False), SCHEMA,
                        lambda v: validate(v, batch, rows, data))


def revise(client, directory, data, rows, workers=4):
    jobs = list(structured.jobs_for(revision.records(data), rows, count=12, max_chars=60000))
    progress = dict(stage='semantic_roles_revision', batches=len(jobs), completed=0, failed=0)
    outputs = {}
    write_json(directory / 'progress.json', progress)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(extract, client, directory / 'cache', data, batch, context): i
                   for i, (batch, context) in enumerate(jobs)}
        for future in as_completed(pending):
            outputs[pending[future]] = future.result()
            progress['completed'] += 1
            write_json(directory / 'progress.json', progress)
    result = revision.apply(data, [outputs[i] for i in sorted(outputs)], rows)
    result['semantic_revision']['engine'] = 'roles_claims_v2'
    return result
