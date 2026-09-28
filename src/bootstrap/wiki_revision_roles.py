"""Source-grounded revision with explicit evidence for each semantic role."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json

from ..iteration.storage import write_json
from . import wiki_revision as revision, wiki_structured as structured
from .wiki_library_extract import obj
from .wiki_repair import _cached_call, _render

SCHEMA = deepcopy(revision.SCHEMA)
DECISION = SCHEMA['properties']['decisions']['items']
DECISION['properties']['role_checks'] = structured.array(obj(dict(
    role_key=structured.TEXT, described_entity_id=structured.TEXT,
    support=structured.enum('content', 'speaker_metadata', 'uncertain', 'mismatch'),
    lines=structured.array(dict(type='integer')), explanation=structured.TEXT)))
DECISION['required'].append('role_checks')
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
"""


def role_claims(record, entities):
    """Expand exact structural endpoints, independently of attribution prose."""
    refs = [(key, record[key]) for key in ('subject_id', 'object_id', 'target_id', 'speaker_id')
            if record.get(key)]
    refs += [(f'participants/{i}/{p["role"]}', p['entity_id'])
             for i, p in enumerate(record.get('participants', []))]
    return [dict(role_key=key, entity_id=eid, entity=entities[eid]) for key, eid in refs]


def validate(value, batch, rows, data):
    revision.validate(value, batch, rows, data['subject'], data)
    entities = {e['id']: e for e in data['entities']}
    originals = {r['id']: r for r in batch}
    available = {r['line'] for r in rows}
    for decision in value['decisions']:
        expected = {r['role_key']: r['entity_id']
                    for r in role_claims(originals[decision['record_id']], entities)}
        checks = decision['role_checks']
        if len(checks) != len(expected) or {r['role_key'] for r in checks} != set(expected):
            raise ValueError('every structural role needs exactly one source check')
        for check in checks:
            if not check['lines'] or not set(check['lines']) <= available:
                raise ValueError('role check requires available source lines')
            if check['described_entity_id'] and check['described_entity_id'] not in entities:
                raise ValueError('role check refers to unknown input entity')
            if not check['explanation'].strip():
                raise ValueError('role check requires a source explanation')
            if decision['action'] == 'keep':
                allowed = ('content', 'speaker_metadata') if check['role_key'] == 'speaker_id' else ('content',)
                if (check['support'] not in allowed or
                        check['described_entity_id'] != expected[check['role_key']]):
                    raise ValueError('keep requires semantic role support, not merely message authorship')


def extract(client, cache, data, batch, rows):
    entities = {e['id']: e for e in data['entities']}
    expanded = [dict(r, record_id=r['id'], role_claims=role_claims(r, entities)) for r in batch]
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
    result['semantic_revision']['engine'] = 'roles_v1'
    return result
