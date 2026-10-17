"""Frozen opt-in wire constraints for finite, source-grounded revision retries."""
from copy import deepcopy
import json
from pathlib import Path

from ...config import sha256_file
from ...iteration.storage import read_json, write_json
from .wiki_library_extract import obj, validate_result
from .wiki_repair import _cached_call


CONTRACT = 'revision_wire_v1'
LIMIT = 3
TABLES = ('attributes', 'relations', 'events', 'addresses')
INSTRUCTION = """输出约束版本 revision_wire_v1（仅改变结构，不改变事实核验标准）：
decisions 改为以输入 record_id 为键的对象；role_checks 改为以 role_key 为键的对象；
slot_checks 改为以 slot_key 为键的对象，值直接为该字段的 assertions 数组。按 schema 完整填写。
corrections.entities 分为 e0_entities、e1_entities、other_entities。E0/E1 可省略，
出现时只能是 schema 指定的 person 账号；群或话题不使用 E0。其他实体只用 E2...。
替代记录引用的 ref 必须在本次 corrections 的实体列表声明，绝不能使用输入实体 ID。
corrections 不输出 reviews，由 decisions 和实际替代记录机械映射，不能丢弃有据事实来凑覆盖。
有 replacement_checks 的协议将每个替代的 role_checks/slot_checks 放在该替代的 audit 内，
不输出顶层 replacement_checks；audit 的 key 仍检查该替代的实际角色和全部非空语义字段，
包括 attribution、时间、details 的 field/value 和无账号实体标签，不能照抄原记录的字段清单。
作者证据不等于事实主体证据。content 断言必须给实际 actor 和来源方式；只有日期、状态等
不独立属于某人的说明可用 context，不能用 context 绕过归属。无法核实仍用不确定判断，
不等于已证明事实错误。keep 必须保留原引用，不凭新上下文替原引用背书。
下面原任务的语义、归属及有限纠错规则全部不变；仅以本条说明替换输出结构说明。\n"""


def activate(output, manifest, selected, state, pipeline, client):
    if not selected:
        raise ValueError('revision output contract requires explicit job_ids')
    path = Path(output) / 'revision-output' / CONTRACT / 'manifest.json'
    frozen = dict(schema='wiki_revision_output_v1', contract=CONTRACT,
        batch=dict(path=str(Path(output) / 'manifest.json'),
                   sha256=sha256_file(Path(output) / 'manifest.json')),
        source=manifest['revision']['source'], engine=manifest['revision'].get('engine', 'legacy'),
        config=manifest['config'], client=client.cache_identity(), jobs=sorted(selected),
        pipeline=pipeline, retry_limit=LIMIT)
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError('frozen revision output contract changed')
    else:
        if any(state['jobs'].get(job_id, {}).get('stage') != 'incomplete' for job_id in selected):
            raise ValueError('new revision output contract requires failed jobs only')
        write_json(path, frozen)
    return dict(path=str(path), sha256=sha256_file(path))


class ContractClient:
    def __init__(self, client, binding):
        self.client = client
        self.revision_output_contract = binding

    def __getattr__(self, name):
        return getattr(self.client, name)


def _keyed(schema, keys, field):
    item = deepcopy(schema['items'])
    item['properties'].pop(field)
    item['required'].remove(field)
    return obj({key: deepcopy(item) for key in keys})


def wire_schema(schema, payload):
    wire = deepcopy(schema)
    fields = wire['properties']
    decision = fields['decisions']
    fields['decisions'] = _keyed(decision, [r['record_id'] for r in payload['records']], 'record_id')
    for record in payload['records']:
        checks = fields['decisions']['properties'][record['record_id']]['properties']
        if 'role_checks' in checks:
            checks['role_checks'] = _keyed(checks['role_checks'],
                [c['role_key'] for c in record['role_claims']], 'role_key')
        if 'slot_checks' in checks:
            assertions = checks['slot_checks']['items']['properties']['assertions']
            checks['slot_checks'] = obj({c['slot_key']: deepcopy(assertions)
                                        for c in record['content_slots']})
    corrections = fields['corrections']
    properties = corrections['properties']
    entity = properties.pop('entities')
    properties.pop('reviews')
    lines = [r[0] for r in payload['messages']]
    for ref, account in (('E0', payload['subject']['account']),
                         ('E1', payload['subject']['self_account'])):
        anchors = [r[0] for r in payload['messages'] if r[2] == account]
        available = anchors and (ref != 'E0' or payload['subject'].get('kind', 'person') == 'person')
        partition = deepcopy(entity)
        partition['maxItems'] = 1 if available else 0
        person = partition['items']['properties']
        person['ref'] = dict(type='string', enum=[ref])
        person['kind'] = dict(type='string', enum=['person'])
        person['account'] = dict(type='string', enum=[account])
        if anchors:
            person['lines']['items'] = dict(type='integer', enum=anchors)
        properties[ref.lower() + '_entities'] = partition
    properties['other_entities'] = deepcopy(entity)
    properties['other_entities']['items']['properties']['ref'] = dict(
        type='string', pattern=r'^E(?:[2-9]|[1-9][0-9]+)$')
    audit = fields.pop('replacement_checks', None)
    if audit is not None:
        audit = deepcopy(audit['items'])
        audit['properties'].pop('record_key')
        audit['required'].remove('record_key')
    for table in TABLES:
        record = properties[table]['items']['properties']
        record['fact_ids']['items'] = dict(type='string', enum=[r['record_id'] for r in payload['records']])
        record['fact_ids']['maxItems'] = 1
        if audit is not None:
            record['audit'] = deepcopy(audit)
            properties[table]['items']['required'].append('audit')
    corrections['required'] = list(properties)
    wire['required'] = list(fields)

    def constrain(value):
        if not isinstance(value, dict):
            return
        for key, child in value.get('properties', {}).items():
            if key in ('lines', 'utterance_line'):
                target = child['items'] if key == 'lines' else child
                if 'enum' not in target:
                    target['enum'] = lines
            constrain(child)
        if 'items' in value:
            constrain(value['items'])

    constrain(wire)
    return wire


def normalize(raw, schema):
    validate_result(raw, schema)
    value = deepcopy(raw)
    decisions = []
    for record_id, decision in value['decisions'].items():
        decision['record_id'] = record_id
        if 'role_checks' in decision:
            decision['role_checks'] = [dict(check, role_key=key)
                                       for key, check in decision['role_checks'].items()]
        if 'slot_checks' in decision:
            decision['slot_checks'] = [dict(slot_key=key, assertions=checks)
                                       for key, checks in decision['slot_checks'].items()]
        decisions.append(decision)
    value['decisions'] = decisions
    corrections = value['corrections']
    for key in ('e0_entities', 'e1_entities'):
        if len(corrections[key]) > schema['properties']['corrections']['properties'][key]['maxItems']:
            raise ValueError('unavailable or duplicate anchored person')
    if any(entity['ref'] in ('E0', 'E1') for entity in corrections['other_entities']):
        raise ValueError('other_entities cannot contain E0/E1')
    corrections['entities'] = sum((corrections.pop(key) for key in
        ('e0_entities', 'e1_entities', 'other_entities')), [])
    represented = {rid for table in TABLES for record in corrections[table] for rid in record['fact_ids']}
    corrections['reviews'] = [dict(fact_id=decision['record_id'], reason=decision['reason'],
        disposition='represented' if decision['record_id'] in represented else 'unresolved')
        for decision in decisions if decision['action'] in ('correct', 'unresolved')]
    if 'audit' in schema['properties']['corrections']['properties']['attributes']['items']['properties']:
        value['replacement_checks'] = [dict(record.pop('audit'), record_key=f'{table}/{index}')
            for table in TABLES for index, record in enumerate(corrections[table])]
    return value


def cached_call(client, directory, prompt, schema, parse, payload):
    if not getattr(client, 'revision_output_contract', None):
        return _cached_call(client, directory, prompt, schema, parse)
    previous = _cached_call(client, directory, prompt, schema, parse, cache_only=True)
    if previous is not None:
        return previous
    wire = wire_schema(schema, payload)

    def checked(raw):
        value = normalize(raw, wire)
        validate_result(value, schema)
        parse(value)

    raw = _cached_call(client, directory, INSTRUCTION + prompt, wire, checked)
    return normalize(raw, wire)
