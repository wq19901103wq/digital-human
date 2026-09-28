"""Semantic repairs preserve supported records and expose unresolved ownership."""
from copy import deepcopy
import json

import pytest

from src.bootstrap import wiki_revision as revision, wiki_structured as structured
from src.bootstrap import wiki_delivery, wiki_objects
from test_wiki_prompt import knowledge
from test_wiki_repair import Client, row, source
from src.config import sha256_file


def empty():
    return dict(entities=[], reviews=[], **{key: [] for key in revision.TABLES})


def decisions(data, changes=None, corrections=None):
    changes = changes or {}
    return dict(decisions=[dict(record_id=r['id'], action=changes.get(r['id'], 'keep'),
                               reason='该条归属尚无法确认。' if changes.get(r['id']) == 'unresolved' else '来源支持。')
                           for r in revision.records(data)], corrections=corrections or empty())


class KeepClient(Client):
    def run(self, prompt, schema):
        self.prompts.append(prompt)
        payload = json.loads(prompt[len(revision.PROMPT):])
        return json.dumps(dict(decisions=[dict(record_id=r['id'], action='keep', reason='来源支持。')
                                         for r in payload['records']], corrections=empty()))


def test_all_records_reviewed_with_raw_only_and_exact_cache_resume(tmp_path):
    data = knowledge(tmp_path)
    before = deepcopy(data)
    rows = revision.read_rows(data['coverage'], 'self')
    (tmp_path / 'manual-review.json').write_text('FORBIDDEN_CASE_FEEDBACK')
    client = KeepClient()
    result = revision.revise(client, tmp_path / 'revision', data, rows)
    assert data == before
    assert result['semantic_revision']['counts'] == {'keep': 4}
    for key in revision.TABLES:
        assert result[key] == data[key]
    assert all('FORBIDDEN_CASE_FEEDBACK' not in p for p in client.prompts)
    assert any('你来定时间' in p for p in client.prompts)
    assert '你来定时间' not in json.dumps(result, ensure_ascii=False)
    calls = len(client.prompts)
    assert revision.revise(client, tmp_path / 'revision', data, rows) == result
    assert len(client.prompts) == calls


def test_source_labels_drop_guessed_aliases_without_changing_directed_addresses(tmp_path):
    data = knowledge(tmp_path)
    data['entities'][0].update(label='不属于本人的别名', labels=['不属于本人的别名', '网络泛称'])
    rows = revision.read_rows(data['coverage'], 'self')
    result = revision.apply(data, [decisions(data)], rows)
    assert {e['label'] for e in result['entities']} == {'person', 'self'}
    assert '不属于本人的别名' not in json.dumps(result, ensure_ascii=False)
    assert result['addresses'] == data['addresses']


def test_event_correction_keeps_experience_but_removes_unrelated_participant(tmp_path):
    data = knowledge(tmp_path)
    event = data['events'][0]
    person = data['subject']['entity_id']
    data['entities'].append(dict(id='unrelated', account='', label='隔壁话题', labels=['隔壁话题'],
                                kind='topic', evidence_refs=['1'], identity_status='unresolved_cross_batch'))
    event['participants'].append(dict(entity_id='unrelated', role='participant'))
    corrections = empty()
    corrections['entities'] = [dict(ref='E0', kind='person', account='person', label='人物', lines=[2])]
    corrected = {k: event[k] for k in structured.COMMON if k not in ('lines', 'fact_ids')}
    corrected.update(event_type=event['event_type'], description=event['description'], status=event['status'],
        details=event['details'], participants=[dict(entity_ref='E0', role='traveler')],
        lines=[1], fact_ids=[event['id']])
    corrections['events'] = [corrected]
    corrections['reviews'] = [dict(fact_id=event['id'], disposition='represented', reason='参与者有来源。')]
    value = decisions(data, {event['id']: 'correct'}, corrections)
    rows = revision.read_rows(data['coverage'], 'self')
    revision.validate(value, revision.records(data), rows, data['subject'], data)
    result = revision.apply(data, [value], rows)
    assert result['events'][0]['description'] == '尚未出发'
    assert result['events'][0]['participants'] == [dict(entity_id=person, role='traveler')]
    assert result['events'][0]['fact_ids'] == event['fact_ids']
    assert result['events'][0]['revised_from'] == [event['id']]
    assert result['addresses'] == data['addresses'] and not wiki_delivery.content_issues(result)
    assert 'unrelated' not in {e['id'] for e in result['entities']}


def test_shared_fact_does_not_hide_unresolved_record_in_readable_object(tmp_path):
    data = knowledge(tmp_path)
    record = data['events'][0]
    corrections = empty()
    corrections['reviews'] = [dict(fact_id=record['id'], disposition='unresolved', reason='无法判断。')]
    value = decisions(data, {record['id']: 'unresolved'}, corrections)
    rows = revision.read_rows(data['coverage'], 'self')
    revision.validate(value, revision.records(data), rows, data['subject'], data)
    result = revision.apply(data, [value], rows)
    assert result['reviews'][0]['disposition'] == 'represented'  # Other records share this fact.
    assert any(r['reason'] == '该条归属尚无法确认。' and r['disposition'] == 'unresolved'
               for r in result['reviews'])
    result['subject']['kind'] = 'person'
    result['provenance'] = dict(records={})
    result['coverage']['materials'] = {'material-test': dict(reviews=result['reviews'])}
    result['coverage']['input_records'] = 3
    result['coverage']['records'] = 3
    assert '该条归属尚无法确认。' in wiki_objects.render(result)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'correct_without_record', 'bad_anchor'])
def test_invalid_decisions_are_not_cached_or_exported(tmp_path, mutation):
    data = knowledge(tmp_path)
    value = decisions(data)
    if mutation == 'missing':
        value['decisions'].pop()
    elif mutation == 'duplicate':
        value['decisions'][-1] = value['decisions'][0]
    elif mutation == 'correct_without_record':
        value['decisions'][0]['action'] = 'correct'
        value['corrections']['reviews'] = [dict(fact_id=value['decisions'][0]['record_id'],
                                               disposition='unresolved', reason='无依据。')]
    else:
        data['entities'][0]['evidence_refs'] = ['1']
    rows = revision.read_rows(data['coverage'], 'self')
    with pytest.raises(ValueError):
        revision.validate(value, revision.records(data), rows, data['subject'], data)


def test_cross_batch_duplicates_and_changed_raw_source_fail(tmp_path):
    data = knowledge(tmp_path)
    rows = revision.read_rows(data['coverage'], 'self')
    value = decisions(data)
    with pytest.raises(ValueError, match='all records'):
        revision.apply(data, [value, value], rows)
    path = source(tmp_path, [row('private:self:other', 'other', 9)])
    with pytest.raises(ValueError, match='source changed'):
        revision.read_rows(data['coverage'], 'self')
    assert revision.read_rows(dict(path=str(path), sha256=sha256_file(path)), 'other') == []
