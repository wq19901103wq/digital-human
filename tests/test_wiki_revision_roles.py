"""Semantic role checks cannot treat message authorship as subject evidence."""
from copy import deepcopy
import json

import pytest

from src.bootstrap import wiki_revision as revision, wiki_revision_roles as roles
from test_wiki_prompt import knowledge
from test_wiki_repair import Client
from test_wiki_revision import empty


class RoleClient(Client):
    def run(self, prompt, schema):
        self.prompts.append(prompt)
        payload = json.loads(prompt[len(roles.PROMPT):])
        return json.dumps(dict(decisions=[dict(record_id=r['id'], action='keep', reason='内容支持。',
            role_checks=[dict(role_key=c['role_key'], described_entity_id=c['entity_id'],
                support='content', lines=[r['lines'][0]], explanation='该角色在内容中明确。')
                         for c in r['role_claims']]) for r in payload['records']], corrections=empty()))


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
    assert result['semantic_revision']['engine'] == 'roles_v1'
    assert result['semantic_revision']['counts'] == {'keep': 4}
    for key in revision.TABLES:
        assert result[key] == data[key]
    calls = len(client.prompts)
    assert roles.revise(client, tmp_path / 'revision', data, rows) == result
    assert len(client.prompts) == calls
    assert roles.SCHEMA != revision.SCHEMA
    assert 'role_checks' not in revision.SCHEMA['properties']['decisions']['items']['properties']
    assert '你来定时间' not in json.dumps(result, ensure_ascii=False)
