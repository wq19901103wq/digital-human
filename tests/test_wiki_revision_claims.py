"""Semantic role checks cannot treat message authorship as subject evidence."""
from copy import deepcopy
import json

import pytest

from src.bootstrap import wiki_repair as repair, wiki_revision as revision, wiki_revision_claims as roles
from src.bootstrap import wiki_structured as structured
from test_wiki_prompt import knowledge
from test_wiki_repair import Client, row, source
from test_wiki_revision import empty


def slot_checks(slots, line, actor=''):
    return [dict(slot_key=slot['slot_key'], assertions=[dict(claim='这项内容有独立依据。',
        support='content', actor_entity_id=actor, source_basis='explicit_reference' if actor else 'context',
        lines=[line], comparison=dict(present='no', reference='', reference_entity_id='',
                                      support='not_applicable', lines=[]))]) for slot in slots]


class RoleClient(Client):
    def run(self, prompt, schema):
        self.prompts.append(prompt)
        if repair.RETRY_MARKER in prompt:
            prompt = prompt.split(repair.RETRY_MARKER, 1)[1]
        payload = json.loads(prompt[len(roles.PROMPT):])
        return json.dumps(dict(decisions=[dict(record_id=r['id'], action='keep', reason='内容支持。',
            role_checks=[dict(role_key=c['role_key'], described_entity_id=c['entity_id'],
                support='content', lines=[r['lines'][0]], explanation='该角色在内容中明确。')
                         for c in r['role_claims']],
            slot_checks=slot_checks(r['content_slots'], r['lines'][0]))
            for r in payload['records']], corrections=empty(), replacement_checks=[]))


def fixture(tmp_path):
    data = knowledge(tmp_path)
    person = next(e['id'] for e in data['entities'] if e['account'] == 'self')
    event = data['events'][0]
    record = {k: event[k] for k in ('evidence_refs', 'fact_ids', 'mode', 'polarity',
                                   'valid_from', 'valid_to', 'observed_at') if k in event}
    record.update(id='attribute-test', subject_id=person, field='preference', value='某种偏好',
                  mode='self_report', attribution='发言者在评论另一个人的偏好。')
    data['attributes'] = [record]
    rows = revision.read_rows(data['coverage'], 'self')
    client = RoleClient()
    value = roles.extract(client, tmp_path / 'cache', data, revision.records(data), rows)
    return data, rows, value


@pytest.mark.parametrize('mutation', ['speaker_only', 'mismatch', 'uncertain', 'wrong_entity',
                                    'missing', 'duplicate', 'unknown_line'])
def test_keep_requires_each_actual_role_not_just_an_authored_message(tmp_path, mutation):
    data, rows, value = fixture(tmp_path)
    decision = next(d for d in value['decisions'] if d['record_id'] == 'attribute-test')
    check = decision['role_checks'][0]
    if mutation in ('speaker_only', 'mismatch', 'uncertain'):
        check['support'] = 'speaker_metadata' if mutation == 'speaker_only' else mutation
    elif mutation == 'wrong_entity':
        check['described_entity_id'] = next(e['id'] for e in data['entities'] if e['account'] == 'person')
    elif mutation == 'missing':
        decision['role_checks'] = []
    elif mutation == 'duplicate':
        decision['role_checks'].append(deepcopy(check))
    else:
        check['lines'] = [999999]
    with pytest.raises(ValueError):
        roles.validate(value, revision.records(data), rows, data)


def test_mismatched_subject_can_be_omitted_without_rewriting_supported_records(tmp_path):
    data, rows, value = fixture(tmp_path)
    decision = next(d for d in value['decisions'] if d['record_id'] == 'attribute-test')
    decision['action'] = 'omit'
    decision['role_checks'][0].update(support='mismatch', described_entity_id='')
    roles.validate(value, revision.records(data), rows, data)
    result = revision.apply(data, [value], rows)
    assert not result['attributes']
    assert result['events'] == data['events']
    assert result['addresses'] == data['addresses']


def test_role_protocol_has_distinct_cache_and_preserves_kept_records(tmp_path):
    data = knowledge(tmp_path)
    rows = revision.read_rows(data['coverage'], 'self')
    client = RoleClient()
    result = roles.revise(client, tmp_path / 'revision', data, rows)
    assert result['semantic_revision']['engine'] == 'roles_claims_v2'
    assert result['semantic_revision']['counts'] == {'keep': 4}
    for key in revision.TABLES:
        assert result[key] == data[key]
    calls = len(client.prompts)
    assert roles.revise(client, tmp_path / 'revision', data, rows) == result
    assert len(client.prompts) == calls
    assert roles.SCHEMA != revision.SCHEMA
    assert 'role_checks' not in revision.SCHEMA['properties']['decisions']['items']['properties']
    assert '你来定时间' not in json.dumps(result, ensure_ascii=False)


def mixed_event(tmp_path):
    """An old summary combines distinct owners and dates in one topic."""
    path = source(tmp_path, [
        row('group:self:room', 'person', 1000, '我去年修车用了两小时。'),
        row('group:self:room', 'self', 1001, '我预约今天做保养。')])
    rows, coverage = repair.select_history(path, 'person', 'self')
    raw = empty()
    raw['entities'] = [dict(ref='E0', kind='person', label='成员甲', account='person', lines=[1]),
                       dict(ref='E1', kind='person', label='成员乙', account='self', lines=[2])]
    common = dict(time_scope='今天预约；去年维修', valid_from='', valid_to='', temporal_kind='historical',
                  mode='self_report', polarity='affirmed', lines=[1, 2], fact_ids=['F1'],
                  attribution='成员甲讲述保养安排及维修经历。')
    raw['events'] = [dict(common, event_type='vehicle_maintenance', description='预约保养并回顾维修经历',
        status='planned', participants=[dict(entity_ref='E0', role='车主')],
        details=[dict(field='appointment', value='预约今天保养'),
                 dict(field='repair_duration', value='维修耗时两小时')])]
    raw['reviews'] = [dict(fact_id='F1', disposition='represented', reason='旧的复合摘要。')]
    data = structured.compile_records(dict(account='person', self_account='self', name='成员甲'),
        [(raw, {'F1': 'fact-vehicle'})], rows, coverage)
    value = roles.extract(RoleClient(), tmp_path / 'cache', data, revision.records(data), rows)
    return data, rows, value


def assertion_at(value, slot_key):
    return next(c for c in value['decisions'][0]['slot_checks'] if c['slot_key'] == slot_key)['assertions'][0]


@pytest.mark.parametrize('slot_key', ['description', 'details/0/value', 'time_scope'])
@pytest.mark.parametrize('failure', ['wrong_author', 'different_actor', 'unsupported_part'])
def test_one_supported_fragment_cannot_justify_another_event_slot(tmp_path, slot_key, failure):
    data, rows, value = mixed_event(tmp_path)
    # The aggregate self-report guard accepts this record because one author
    # matches. The appointment slot must be checked independently.
    revision.validate(value, revision.records(data), rows, data['subject'], data)
    assertion = assertion_at(value, slot_key)
    person = data['subject']['entity_id']
    other = next(e['id'] for e in data['entities'] if e['account'] == 'self')
    assertion.update(claim='车主预约当日保养。', actor_entity_id=person,
                     source_basis='self_report', lines=[2])
    if failure == 'different_actor':
        assertion['actor_entity_id'] = other
    elif failure == 'unsupported_part':
        assertion.update(support='mismatch', source_basis='uncertain')
    with pytest.raises(ValueError, match='author evidence|outside the record roles|unsupported atomic'):
        roles.validate(value, revision.records(data), rows, data)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'empty', 'unknown_line', 'unknown_actor'])
def test_content_audit_must_cover_each_slot_and_use_real_sources(tmp_path, mutation):
    data, rows, value = mixed_event(tmp_path)
    checks = value['decisions'][0]['slot_checks']
    if mutation == 'missing':
        checks.pop()
    elif mutation == 'duplicate':
        checks[-1] = deepcopy(checks[0])
    elif mutation == 'empty':
        checks[0]['assertions'] = []
    elif mutation == 'unknown_line':
        checks[0]['assertions'][0]['lines'] = [9999]
    else:
        checks[0]['assertions'][0]['actor_entity_id'] = 'invented-actor'
    with pytest.raises(ValueError):
        roles.validate(value, revision.records(data), rows, data)


def comparison_record(tmp_path, *, explicit=False):
    text = '后来认识的队友小三岁。' if not explicit else '新队友比我小三岁。'
    path = source(tmp_path, [row('private:self:person', 'person', 1000, text)])
    rows, coverage = repair.select_history(path, 'person', 'self')
    raw = empty()
    raw['entities'] = [dict(ref='E0', kind='person', label='成员甲', account='person', lines=[1]),
                       dict(ref='E2', kind='person', label='比成员甲小三岁的队友', account='', lines=[1])]
    common = dict(time_scope='往事', valid_from='', valid_to='', temporal_kind='historical',
        mode='reported', polarity='affirmed', lines=[1], fact_ids=['F1'], attribution='成员甲谈起队友。')
    raw['relations'] = [dict(common, subject_ref='E0', predicate='teammate_of', object_ref='E2',
                             detail='队友比成员甲小三岁')]
    raw['reviews'] = [dict(fact_id='F1', disposition='represented', reason='既有关系记录。')]
    data = structured.compile_records(dict(account='person', self_account='self', name='成员甲'),
        [(raw, {'F1': 'fact-team'})], rows, coverage)
    value = roles.extract(RoleClient(), tmp_path / 'cache', data, revision.records(data), rows)
    return data, rows, value


@pytest.mark.parametrize('slot', ['detail', 'entity_label'])
def test_implicit_comparison_baseline_is_not_the_speaker_or_old_entity_label(tmp_path, slot):
    data, rows, value = comparison_record(tmp_path)
    if slot == 'entity_label':
        slot = next(c['slot_key'] for c in value['decisions'][0]['slot_checks']
                    if c['slot_key'].startswith('entities/'))
    assertion_at(value, slot)['comparison'] = dict(present='yes', reference='', reference_entity_id='',
                                                  support='uncertain', lines=[1])
    with pytest.raises(ValueError, match='comparison reference must remain unresolved'):
        roles.validate(value, revision.records(data), rows, data)
    decision = value['decisions'][0]
    decision.update(action='unresolved', reason='关系成立，但年龄比较的基准无法确认。')
    value['corrections']['reviews'] = [dict(fact_id=decision['record_id'], disposition='unresolved',
                                            reason='比较基准不明。')]
    roles.validate(value, revision.records(data), rows, data)
    result = revision.apply(data, [value], rows)
    assert not result['relations']
    assert result['semantic_revision']['counts'] == {'unresolved': 1}
    assert any(r['disposition'] == 'unresolved' for r in result['reviews'])


def test_explicit_comparison_and_third_party_reporting_are_supported(tmp_path):
    data, rows, value = comparison_record(tmp_path, explicit=True)
    person = data['subject']['entity_id']
    teammate = data['relations'][0]['object_id']
    for check in value['decisions'][0]['slot_checks']:
        if check['slot_key'] == 'detail' or check['slot_key'].startswith('entities/'):
            check['assertions'][0].update(actor_entity_id=teammate, source_basis='explicit_reference',
                comparison=dict(present='yes', reference='成员甲本人', reference_entity_id=person,
                                support='explicit', lines=[1]))
    # Reporting a third party is valid; requiring every actor to equal the
    # source author would incorrectly reject this explicit comparison.
    roles.validate(value, revision.records(data), rows, data)


def corrected_event(data, value):
    original = data['events'][0]
    correction = {k: original[k] for k in structured.COMMON if k not in ('lines', 'fact_ids')}
    correction.update(description='曾维修车辆，耗时两小时', event_type='vehicle_repair',
        time_scope='去年', status='completed', attribution='成员甲自述自己的维修经历。',
        lines=[1], fact_ids=[original['id']], participants=[dict(entity_ref='E0', role='车主')],
        details=[dict(field='repair_duration', value='两小时')])
    value['decisions'][0]['action'] = 'correct'
    corrections = value['corrections']
    corrections['entities'] = [dict(ref='E0', kind='person', label='成员甲', account='person', lines=[1])]
    corrections['events'] = [correction]
    corrections['reviews'] = [dict(fact_id=original['id'], disposition='represented', reason='分离两人的经历。')]
    entities = {e['ref']: e for e in corrections['entities']}
    checks = slot_checks(roles.content_slots(correction, entities), 1, 'E0')
    for check in checks:
        check['assertions'][0]['source_basis'] = 'self_report'
    value['replacement_checks'] = [dict(record_key='events/0', slot_checks=checks,
        role_checks=[dict(role_key='participants/0/车主', described_entity_id='E0', support='content',
                          lines=[1], explanation='此人是该次维修的车主。')])]
    return correction


@pytest.mark.parametrize('mutation', ['none', 'missing_audit', 'bad_source', 'unsupported', 'wrong_key',
                                     'lost_source', 'lost_comparison_source'])
def test_corrections_receive_the_same_atomic_source_checks(tmp_path, mutation):
    data, rows, value = mixed_event(tmp_path)
    corrected_event(data, value)
    if mutation == 'missing_audit':
        value['replacement_checks'] = []
    elif mutation == 'wrong_key':
        value['replacement_checks'][0]['record_key'] = 'events/1'
    elif mutation in ('bad_source', 'unsupported', 'lost_source', 'lost_comparison_source'):
        assertion = value['replacement_checks'][0]['slot_checks'][0]['assertions'][0]
        if mutation == 'bad_source':
            assertion['lines'] = [2]
        elif mutation == 'unsupported':
            assertion['support'] = 'uncertain'
        elif mutation == 'lost_source':
            assertion.update(source_basis='explicit_reference', lines=[2])
        else:
            assertion['comparison'] = dict(present='yes', reference='成员甲此前的安排',
                reference_entity_id='E0', support='explicit', lines=[2])
    if mutation != 'none':
        with pytest.raises(ValueError):
            roles.validate(value, revision.records(data), rows, data)
        return
    roles.validate(value, revision.records(data), rows, data)
    result = revision.apply(data, [value], rows)
    assert result['events'][0]['description'] == '曾维修车辆，耗时两小时'
    assert result['events'][0]['time_scope'] == '去年'
    assert result['events'][0]['evidence_refs'] == ['1']


@pytest.mark.parametrize('mutation', ['missing', 'speaker_only', 'mismatch', 'lost_source', 'extra_role'])
def test_replacements_cannot_hide_unsupported_structural_roles(tmp_path, mutation):
    data, rows, value = mixed_event(tmp_path)
    correction = corrected_event(data, value)
    checks = value['replacement_checks'][0]['role_checks']
    if mutation == 'missing':
        checks.clear()
    elif mutation == 'speaker_only':
        checks[0]['support'] = 'speaker_metadata'
    elif mutation == 'mismatch':
        checks[0].update(support='mismatch', described_entity_id='')
    elif mutation == 'lost_source':
        checks[0]['lines'] = [2]
    else:
        value['corrections']['entities'].append(dict(ref='E1', kind='person', label='成员乙',
                                                     account='self', lines=[2]))
        correction['participants'].append(dict(entity_ref='E1', role='预约人'))
        correction['lines'].append(2)
    with pytest.raises(ValueError, match='structural role|semantic support|record evidence'):
        roles.validate(value, revision.records(data), rows, data)


def test_unresolved_replacement_cannot_reintroduce_a_definite_comparison_target(tmp_path):
    data, rows, value = mixed_event(tmp_path)
    corrected = corrected_event(data, value)
    corrected.update(mode='uncertain', polarity='uncertain')
    value['decisions'][0]['action'] = 'unresolved'
    assertion = value['replacement_checks'][0]['slot_checks'][0]['assertions'][0]
    assertion.update(support='uncertain', source_basis='uncertain', actor_entity_id='')
    assertion['comparison'] = dict(present='yes', reference='', reference_entity_id='',
                                  support='uncertain', lines=[1])
    roles.validate(value, revision.records(data), rows, data)
    assertion['comparison']['reference_entity_id'] = 'E0'
    with pytest.raises(ValueError, match='comparison reference must remain unresolved'):
        roles.validate(value, revision.records(data), rows, data)


def duration_record(tmp_path, slot_key, *, incorrect):
    """The duration is sourced; its semantic label can still be wrong."""
    path = source(tmp_path, [row('private:self:person', 'person', 1000, '我去年修车用了两小时。')])
    rows, coverage = repair.select_history(path, 'person', 'self')
    raw = empty()
    raw['entities'] = [dict(ref='E0', kind='person', label='成员甲', account='person', lines=[1])]
    common = dict(time_scope='去年', valid_from='', valid_to='', temporal_kind='historical',
        mode='self_report', polarity='affirmed', lines=[1], fact_ids=['F1'],
        attribution='成员甲自述维修车辆的耗时。')
    if slot_key == 'field':
        collection = 'attributes'
        record = dict(common, subject_ref='E0', field='driving_duration' if incorrect else 'repair_duration',
                      value='两小时')
    else:
        collection = 'events'
        record = dict(common, event_type='driving' if incorrect and slot_key == 'event_type' else 'vehicle_repair',
            description='维修车辆耗时两小时', status='completed',
            participants=[dict(entity_ref='E0', role='车主')],
            details=[dict(field='driving_duration' if incorrect and slot_key == 'details/0/field'
                          else 'repair_duration', value='两小时')])
    raw[collection] = [record]
    raw['reviews'] = [dict(fact_id='F1', disposition='represented', reason='已有耗时记录。')]
    data = structured.compile_records(dict(account='person', self_account='self', name='成员甲'),
        [(raw, {'F1': 'fact-duration'})], rows, coverage)
    value = roles.extract(RoleClient(), tmp_path / 'cache', data, revision.records(data), rows)
    return data, rows, value, raw, collection


@pytest.mark.parametrize('slot_key', ['field', 'event_type', 'details/0/field'])
@pytest.mark.parametrize('replacement', [False, True], ids=['original', 'replacement'])
@pytest.mark.parametrize('audit', ['mismatch', 'missing', 'supported'])
def test_correct_value_does_not_justify_a_wrong_semantic_key(tmp_path, slot_key, replacement, audit):
    data, rows, value, raw, collection = duration_record(tmp_path, slot_key, incorrect=audit != 'supported')
    checks = value['decisions'][0]['slot_checks']
    if replacement:
        original_id = data[collection][0]['id']
        value['decisions'][0]['action'] = 'correct'
        corrections = deepcopy(raw)
        corrections[collection][0]['fact_ids'] = [original_id]
        corrections['reviews'][0]['fact_id'] = original_id
        value['corrections'] = corrections
        entities = {e['ref']: e for e in corrections['entities']}
        record = corrections[collection][0]
        checks = slot_checks(roles.content_slots(record, entities), 1, 'E0')
        value['replacement_checks'] = [dict(record_key=f'{collection}/0', slot_checks=checks,
            role_checks=[dict(role_key=r['role_key'], described_entity_id=r['entity_id'],
                support='content', lines=[1], explanation='原文支持维修车辆的主体。')
                for r in roles.role_claims(record, entities)])]
    # Authorship and the literal duration pass the existing aggregate guards.
    # Only the semantic key is missing support; all value audits stay supported.
    revision.validate(value, revision.records(data), rows, data['subject'], data)
    check = next(c for c in checks if c['slot_key'] == slot_key)
    if audit == 'mismatch':
        check['assertions'][0].update(support='mismatch',
            claim='来源说明的是维修及其耗时，不能据此标为驾驶或驾驶耗时。')
    elif audit == 'missing':
        checks.remove(check)
    if audit != 'supported':
        with pytest.raises(ValueError, match='unsupported atomic assertions|every content slot'):
            roles.validate(value, revision.records(data), rows, data)
        return
    check['assertions'][0]['claim'] = '原文支持维修车辆这一事件及维修耗时两小时。'
    roles.validate(value, revision.records(data), rows, data)
    result = revision.apply(data, [value], rows)[collection][0]
    if collection == 'attributes':
        assert (result['field'], result['value']) == ('repair_duration', '两小时')
    else:
        assert result['event_type'] == 'vehicle_repair'
        assert result['details'] == [dict(field='repair_duration', value='两小时')]
