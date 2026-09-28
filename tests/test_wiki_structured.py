"""Source attribution and normalized export checks, using synthetic people only."""
import json
import xml.etree.ElementTree as ET

import pytest

from src.bootstrap import wiki_repair as repair, wiki_structured as structured
from test_wiki_repair import Client, row, source


META = dict(name='示例人物', account='person', self_account='self')


def fixture(tmp_path):
    path = source(tmp_path, [
        row('private:self:person', 'self', 1000, '阿林，明天见。'),
        row('private:self:person', 'person', 1001, '小周，你来定时间。'),
        row('private:self:person', 'person', 1002, '有人叫我“林老师”，我不喜欢。')])
    rows, coverage = repair.select_history(path, 'person', 'self')
    common = dict(time_scope='历史聊天时', valid_from='', valid_to='', temporal_kind='historical',
                  mode='observed', polarity='affirmed', lines=[1], fact_ids=['F1'],
                  attribution='各自发言中的直接称呼，按消息发送者区分方向。')
    data = dict(entities=[
        dict(ref='E0', kind='person', label='阿林', account='person', lines=[2]),
        dict(ref='E1', kind='person', label='小周', account='self', lines=[1])],
        attributes=[], relations=[], events=[], addresses=[
            dict(common, utterance_line=1, term='阿林', target_ref='E0', usage='direct'),
            dict(common, lines=[2], utterance_line=2, term='小周', target_ref='E1', usage='direct'),
            dict(common, lines=[3], utterance_line=3, term='林老师', target_ref='E0', usage='rejected')],
        reviews=[dict(fact_id='F1', disposition='represented', reason='有各自的消息依据。')])
    return rows, coverage, data


def test_addresses_remain_directed_and_sources_never_copy_text(tmp_path):
    rows, coverage, data = fixture(tmp_path)
    mapping = {'F1': 'fact-1'}
    structured.validate(data, mapping, rows, META)
    compiled = structured.compile_records(META, [(data, mapping)], rows, coverage)
    records = compiled['addresses']
    assert [r['speaker_account'] for r in records] == ['self', 'person', 'person']
    assert records[0]['target_id'] == compiled['subject']['entity_id']
    assert records[1]['target_id'] == records[0]['speaker_id']
    assert records[2]['usage'] == 'rejected'
    assert all('text' not in e for e in compiled['evidence'].values())
    assert '你来定时间' not in json.dumps(compiled, ensure_ascii=False)
    assert compiled['evidence']['1']['path'] == str(coverage['path'])
    assert len(records) == 3  # No inferred reverse address or identity merge.


@pytest.mark.parametrize('mutation', ['wrong_self', 'bad_kind', 'missing_subject', 'unknown_object',
                                    'invented_line', 'missing_review', 'self_as_direct', 'bad_date'])
def test_bad_attribution_and_references_are_rejected(tmp_path, mutation):
    rows, _, data = fixture(tmp_path)
    if mutation == 'wrong_self':
        data['entities'][1]['account'] = 'person'
    elif mutation == 'bad_kind':
        data['entities'][0]['kind'] = 'organization'
    elif mutation in ('missing_subject', 'unknown_object'):
        common = {k: data['addresses'][0][k] for k in structured.COMMON}
        data['relations'] = [dict(common, subject_ref='' if mutation == 'missing_subject' else 'E0',
                                  predicate='friend_of', object_ref='E9', detail='同学')]
    elif mutation == 'invented_line':
        data['addresses'][0]['lines'] = [999]
    elif mutation == 'missing_review':
        data['reviews'] = []
    elif mutation == 'self_as_direct':
        data['addresses'][0]['target_ref'] = 'E1'
    elif mutation == 'bad_date':
        data['addresses'][0]['valid_from'] = '昨天'
    with pytest.raises(ValueError):
        structured.validate(data, {'F1': 'fact-1'}, rows, META)


def test_unresolved_target_and_typed_xml_are_preserved(tmp_path):
    rows, coverage, data = fixture(tmp_path)
    data['addresses'][0].update(target_ref='', usage='uncertain')
    structured.validate(data, {'F1': 'fact-1'}, rows, META)
    compiled = structured.compile_records(META, [(data, {'F1': 'fact-1'})], rows, coverage)
    assert compiled['addresses'][0]['target_id'] == ''
    assert 'target_ref' not in compiled['addresses'][0]
    compiled['subject']['name'] = '林 & <示例>'
    xml = ET.fromstring(structured.to_xml(compiled))
    assert xml.findtext('subject/name') == '林 & <示例>'
    assert xml.findtext('runtime_usable') == 'false'
    assert xml.find('runtime_usable').get('type') == 'boolean'
    assert len(xml.findall('addresses/item')) == 3
    assert xml.find('evidence/field[@name="1"]/line').text == '1'
    structured.export(tmp_path / 'content', compiled)
    assert (tmp_path / 'content/wiki.xml').exists()
    assert '拒绝被称呼' in (tmp_path / 'content/wiki.md').read_text()


def test_structuring_retry_reports_all_bad_citations_and_reuses_success(tmp_path):
    rows, _, data = fixture(tmp_path)

    class Recover(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            result = json.loads(json.dumps(data))
            if repair.RETRY_MARKER not in prompt:
                result['entities'][0]['lines'] = [998]
                result['addresses'][0].update(lines=[999], fact_ids=['F9'])
            else:
                for expected in ('entities[0]', 'addresses[0]', '998', '999', 'F9',
                                 'allowed_source_lines=[1, 2, 3]', "allowed_fact_ids=['F1']",
                                 'Do not substitute unrelated valid lines'):
                    assert expected in prompt
            return json.dumps(result)

    client = Recover()
    facts = [dict(id='fact-original', lines=[1, 2, 3], summary='各自的称呼。')]
    args = (client, tmp_path / 'cache', META, facts, rows)
    result, mapping = structured.extract(*args)
    assert result == data and mapping == {'F1': 'fact-original'}
    assert len(client.prompts) == 2
    structured.extract(*args)
    assert len(client.prompts) == 2


def test_source_windows_do_not_mix_chats_or_distant_time(tmp_path):
    path = source(tmp_path, [row('private:self:person', 'person', 1000),
        row('private:self:person', 'self', 1001), row('private:self:person', 'person', 9999),
        row('group:self:room', 'person', 1000)])
    rows, _ = repair.select_history(path, 'person', 'self')
    facts = [dict(id='one', lines=[1])]
    jobs = list(structured.jobs_for(facts, rows))
    assert [r['line'] for r in jobs[0][1]] == [1, 2]


@pytest.mark.parametrize('usage', ['direct', 'self_reference', 'third_person', 'requested', 'rejected'])
def test_forward_publisher_cannot_be_compiled_as_inner_speaker(tmp_path, usage):
    rows, _, data = fixture(tmp_path)
    rows[0].update(message_kind='forward', text='甲：阿林，明天见。\n乙：小周，你来定。')
    data['addresses'][0].update(usage=usage, mode='reported')
    with pytest.raises(ValueError, match='forwarded address speaker unresolved'):
        structured.validate(data, {'F1': 'fact-1'}, rows, META)


@pytest.mark.parametrize('usage', ['quoted', 'uncertain'])
def test_forward_keeps_reported_or_unknown_roles_and_direct_corroboration(tmp_path, usage):
    rows, _, data = fixture(tmp_path)
    rows[0]['message_kind'] = 'forward'
    data['addresses'][0]['usage'] = usage
    # A separate direct utterance is not disqualified by a forwarded corroboration.
    data['addresses'][1]['lines'].append(1)
    structured.validate(data, {'F1': 'fact-1'}, rows, META)


def test_invalid_old_cache_rereads_source_and_valid_result_needs_no_new_request(tmp_path, monkeypatch):
    rows, _, data = fixture(tmp_path)
    rows[0].update(message_kind='forward', text='甲：阿林，明天见。\n乙：小周，你来定。')

    class Recover(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            result = json.loads(json.dumps(data))
            if repair.RETRY_MARKER in prompt:
                assert 'forwarded address speaker unresolved' in prompt
                assert '甲：阿林，明天见。' in prompt and '乙：小周，你来定。' in prompt
                result['addresses'][0].update(usage='quoted', mode='reported',
                    attribution='外层账号转发甲对阿林的称呼，甲的账号未知。')
            return json.dumps(result)

    client = Recover()
    facts = [dict(id='fact-original', lines=[1, 2, 3], summary='称呼记录。')]
    args = (client, tmp_path / 'cache', META, facts, rows)
    # Simulate the previously accepted cache, without changing its request key.
    with monkeypatch.context() as old:
        old.setattr(structured, 'validate', lambda *a: None)
        structured.extract(*args)
    client.prompts.clear()
    result, _ = structured.extract(*args)
    assert len(client.prompts) == 1 and result['addresses'][0]['usage'] == 'quoted'
    structured.extract(*args)
    assert len(client.prompts) == 1


def test_adding_structured_stage_reuses_all_upstream_requests(tmp_path):
    class StructuredClient(Client):
        def run(self, prompt, schema):
            if not prompt.startswith(structured.PROMPT):
                return super().run(prompt, schema)
            self.prompts.append(prompt)
            payload = json.loads(prompt[len(structured.PROMPT):])
            line = payload['messages'][0][0]
            return json.dumps(dict(entities=[dict(ref='E0', kind='person', label='示例人物',
                account='person', lines=[line])], attributes=[dict(subject_ref='E0', field='occupation',
                value='工程师', time_scope='历史聊天时', valid_from='', valid_to='', temporal_kind='mutable',
                mode='self_report', polarity='affirmed', lines=[line], fact_ids=['F1'],
                attribution='该账号自述其职业。')],
                relations=[], events=[], addresses=[], reviews=[dict(fact_id='F1',
                    disposition='represented', reason='从原始记录拆出单项属性。')]))
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000, 'RAW_PRIVATE_TEXT')])
    client = StructuredClient()
    first, second = tmp_path / 'first', tmp_path / 'second'
    repair.repair(wiki, messages, first, 'person', 'self', {}, client=client)
    client.prompts.clear()
    result = repair.repair(wiki, messages, second, 'person', 'self', {}, client=client,
                           cache_from=first, structured=True)
    assert result['stage'] == 'complete' and len(client.prompts) == 1
    assert client.prompts[0].startswith(structured.PROMPT)
    final = json.loads((second / 'content/knowledge.json').read_text())
    assert final['schema'] == 'wiki_structured_v1'
    assert len(final['attributes']) == 1 and 'sections' not in final
    assert 'RAW_PRIVATE_TEXT' not in json.dumps(final)
    client.prompts.clear()
    repair.repair(wiki, messages, second, 'person', 'self', {}, client=client, structured=True)
    assert not client.prompts


@pytest.mark.parametrize('entity_index', [0, 1])
def test_fixed_subject_and_owner_need_their_own_source_anchor(tmp_path, entity_index):
    rows, coverage, data = fixture(tmp_path)
    data['entities'][entity_index]['lines'] = [1 if entity_index == 0 else 2]
    with pytest.raises(ValueError, match='entity_account_without_source_anchor'):
        structured.validate(data, {'F1': 'fact-1'}, rows, META)
    compiled = structured.compile_records(META, [(data, {'F1': 'fact-1'})], rows, coverage)
    assert structured.attribution_issues(compiled)[0]['reason'] == 'entity_account_without_source_anchor'


@pytest.mark.parametrize('fixed_ref', ['E0', 'E1'])
def test_retries_keep_identity_errors_and_can_omit_unsupported_fixed_people(tmp_path, fixed_ref):
    rows, coverage = repair.select_history(source(tmp_path, [
        row('group:self:room', 'other', 1000, '我明天搬家。')]), 'other', 'self')
    value = dict(entities=[dict(ref='E2', kind='person', label='另一位群成员',
                               account='other', lines=[1])],
        attributes=[], relations=[], addresses=[],
        events=[dict(event_type='move', description='另一位群成员计划搬家。', status='planned',
            participants=[dict(entity_ref='E2', role='搬家者')], details=[],
            time_scope='消息次日', valid_from='', valid_to='', temporal_kind='historical',
            mode='self_report', polarity='affirmed', lines=[1], fact_ids=['F1'],
            attribution='原消息作者自述搬家计划。')],
        reviews=[dict(fact_id='F1', disposition='represented', reason='保留有依据的本人计划。')])

    class Recover(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            result = json.loads(json.dumps(value))
            attempt = len(self.prompts)
            if attempt < 3:
                result['entities'][0].update(ref=fixed_ref, account=(
                    META['account' if fixed_ref == 'E0' else 'self_account'] if attempt == 1 else ''))
                result['events'][0]['participants'][0]['entity_ref'] = fixed_ref
            if attempt == 3:
                feedback, original = prompt.split(repair.RETRY_MARKER, 1)
                assert original == self.prompts[0]
                payload = json.loads(feedback.split('\n', 1)[1])
                assert len(payload['errors']) == 2
                assert 'entity_account_without_source_anchor' in payload['errors'][0]['error']
                assert 'self_report_without_source_author' in payload['errors'][0]['error']
                assert f'{fixed_ref} must retain its supplied account' in payload['errors'][1]['error']
                assert 'E0/E1 are optional' in feedback
                # Keep only the latest failed output; earlier full outputs are not duplicated.
                assert payload['result']['entities'][0]['account'] == ''
                assert all('result' not in error for error in payload['errors'])
            return json.dumps(result)

    client = Recover()
    args = (client, tmp_path / 'cache', META, [dict(id='fact-1', lines=[1], summary='搬家计划')], rows)
    result, mapping = structured.extract(*args)
    assert result == value and len(client.prompts) == 3
    compiled = structured.compile_records(META, [(result, mapping)], rows, coverage)
    assert [e['account'] for e in compiled['entities']] == ['other']
    assert len(compiled['events']) == 1 and not structured.attribution_issues(compiled)
    structured.extract(*args)
    assert len(client.prompts) == 3


@pytest.mark.parametrize('collection', ['attributes', 'relations', 'events'])
def test_self_reports_need_the_actual_subject_author_in_each_record(tmp_path, collection):
    rows, coverage, data = fixture(tmp_path)
    # Identical display names do not make the source accounts interchangeable.
    data['entities'][1]['label'] = data['entities'][0]['label']
    common = {k: data['addresses'][0][k] for k in structured.COMMON}
    common.update(mode='self_report', lines=[1])
    record = {
        'attributes': dict(subject_ref='E0', field='occupation', value='设计师'),
        'relations': dict(subject_ref='E0', predicate='friend_of', object_ref='E1', detail='朋友'),
        'events': dict(event_type='purchase', description='购买文具', status='completed',
                       participants=[dict(entity_ref='E0', role='购买者')], details=[]),
    }[collection]
    data[collection] = [dict(common, **record)]
    with pytest.raises(ValueError, match='self_report_without_source_author'):
        structured.validate(data, {'F1': 'fact-1'}, rows, META)
    compiled = structured.compile_records(META, [(data, {'F1': 'fact-1'})], rows, coverage)
    assert structured.attribution_issues(compiled)[0]['reason'] == 'self_report_without_source_author'
    data[collection][0]['lines'] = [2]
    structured.validate(data, {'F1': 'fact-1'}, rows, META)


def test_reported_third_party_facts_and_unbound_relatives_remain_supported(tmp_path):
    rows, _, data = fixture(tmp_path)
    common = {k: data['addresses'][0][k] for k in structured.COMMON}
    data['entities'].append(dict(ref='E2', kind='person', label='未具名亲友', account='', lines=[1]))
    data['attributes'] = [dict(common, subject_ref='E0', field='occupation', value='设计师', mode='reported'),
                          dict(common, subject_ref='E2', field='occupation', value='编辑', mode='reported')]
    structured.validate(data, {'F1': 'fact-1'}, rows, META)
    rows[0]['message_kind'] = 'forward'
    data['addresses'][0]['usage'] = 'quoted'
    structured.validate(data, {'F1': 'fact-1'}, rows, META)
    data['attributes'][0].update(subject_ref='E1', mode='self_report')
    with pytest.raises(ValueError, match='self_report_without_source_author'):
        structured.validate(data, {'F1': 'fact-1'}, rows, META)


def test_old_attribution_cache_is_rechecked_and_only_invalid_response_is_regenerated(tmp_path, monkeypatch):
    rows, _, data = fixture(tmp_path)

    class Recover(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            result = json.loads(json.dumps(data))
            if repair.RETRY_MARKER not in prompt:
                result['entities'][1]['lines'] = [2]
            else:
                assert 'entity_account_without_source_anchor' in prompt
                assert 'Re-read all supplied raw messages' in prompt
            return json.dumps(result)

    client = Recover()
    args = (client, tmp_path / 'cache', META, [dict(id='fact-1', lines=[1, 2, 3], summary='称呼')], rows)
    with monkeypatch.context() as old:
        old.setattr(structured, 'validate', lambda *a: None)
        structured.extract(*args)
    client.prompts.clear()
    assert structured.extract(*args)[0] == data
    assert len(client.prompts) == 1
    structured.extract(*args)
    assert len(client.prompts) == 1
