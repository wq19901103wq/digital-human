from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest

from src.infrastructure.storage import read_json
from src.share.wiki import wiki_repair as repair
from src.share.wiki import wiki_reconcile as reconcile
from src.share.wiki import wiki_structured as structured
from src.share.wiki.wiki_library import digest
from test_wiki_reconcile import decision, prepared
from test_wiki_repair import Client, row, source
from test_wiki_structured import META, fixture


class ReferenceClient:
    def __init__(self, value):
        self.value = value
        self.requests = []

    def cache_identity(self):
        return dict(provider='test', model='isolated')

    def run(self, prompt, schema):
        self.requests.append((prompt, schema))
        return json.dumps(self.value)


def test_reference_schema_is_bounded_recursive_and_thread_local():
    original = deepcopy(repair.REREAD_SCHEMA)
    bounds = dict(lines=[41, 9, 41], old_lines=[3], speaker_id=['person'], fact_id=['fact-1'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(
            lambda lines: repair.bounded_reference_schema(repair.REREAD_SCHEMA, bounds | dict(lines=lines)),
            ([41, 9, 41], [72])))
    revision = first['properties']['revisions']['items']['properties']
    fact = revision['replacements']['items']['properties']
    assert revision['fact_id']['enum'] == ['fact-1']
    assert fact['lines']['items']['enum'] == [9, 41]
    assert fact['old_lines']['items']['enum'] == [3]
    assert fact['speaker_id']['enum'] == ['person']
    assert 'enum' not in fact['summary']
    assert second['properties']['additions']['items']['properties']['lines']['items']['enum'] == [72]
    assert repair.REREAD_SCHEMA == original
    assert repair.bounded_reference_schema(repair.REREAD_SCHEMA, dict(lines=[])) == original


def test_reference_contract_reuses_valid_old_cache_without_calls(tmp_path):
    client = ReferenceClient(dict(facts=[]))
    args = (client, tmp_path / 'cache', 'original task', repair.EXTRACT_SCHEMA, lambda value: None)
    repair._cached_call(*args)
    before = list(client.requests)
    assert repair.reference_call(*args, bounds=dict(lines=[1]), contract='bounded_refs_v2') == dict(facts=[])
    assert client.requests == before
    with pytest.raises(ValueError, match='reference contract'):
        repair.reference_call(*args, bounds={}, contract='unknown')


def test_bounded_request_changes_key_without_resetting_old_failure_budget(tmp_path):
    fact = dict(category='background', subject='person', speaker_id='person', summary='历史资料',
                time_scope='historical', mode='self_report', lines=[999], old_lines=[])
    client = ReferenceClient(dict(facts=[fact]))
    directory = tmp_path / 'cache'
    bounds = dict(lines=[41], old_lines=[], speaker_id=['person'])
    args = (client, directory, 'original task', repair.EXTRACT_SCHEMA, lambda value: None)
    bounded = repair.bounded_reference_schema(repair.EXTRACT_SCHEMA, bounds)
    with pytest.raises(ValueError):
        repair._cached_call(client, directory, args[2], bounded, args[4])
    old_key = digest([client.cache_identity(), args[2], bounded])
    failure_path = tmp_path / 'failures' / f'{old_key}-{repair.RETRY_LIMIT}.json'
    failure_bytes = failure_path.read_bytes()
    valid_fact = dict(fact, lines=[41])
    client.value = dict(facts=[valid_fact])
    changed_args = (client, directory, args[2], bounded, args[4])
    assert repair.reference_call(*changed_args, bounds=bounds, contract='bounded_refs_v2') == client.value
    assert len(client.requests) == repair.RETRY_LIMIT + 1
    assert client.requests[-1][0].startswith(repair.REFERENCE_PROMPT)
    assert failure_path.read_bytes() == failure_bytes
    client.value = dict(facts=[fact])
    with pytest.raises(ValueError, match='exhausted'):
        repair._cached_call(*changed_args)
    assert len(client.requests) == repair.RETRY_LIMIT + 1


def test_allowed_speaker_does_not_bypass_cited_author_validation(tmp_path):
    messages = source(tmp_path, [row('private:self:person', 'person', 1000),
                                row('private:self:person', 'self', 1100)])
    selected, _ = repair.select_history(messages, 'person', 'self')
    fact = dict(category='background', subject='person', speaker_id='self', summary='历史资料',
                time_scope='historical', mode='self_report', lines=[1], old_lines=[])
    client = ReferenceClient(dict(facts=[fact]))
    with pytest.raises(ValueError, match='absent from cited messages'):
        repair.extract_batch(client, tmp_path / 'cache', {}, [], selected,
                             reference_contract='bounded_refs_v2')
    assert len(client.requests) == repair.RETRY_LIMIT
    assert client.requests[0][1]['properties']['facts']['items']['properties']['speaker_id']['enum'] == [
        'person', 'self']


def test_reference_upgrade_reuses_complete_repair_requests_and_is_explicit(tmp_path):
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    original, upgraded = tmp_path / 'original', tmp_path / 'upgraded'
    client = Client()
    assert repair.repair(wiki, messages, original, 'person', 'self', {}, client=client)['stage'] == 'complete'
    calls = len(client.prompts)
    assert repair.repair(wiki, messages, upgraded, 'person', 'self', {}, client=client,
                         cache_from=original, reference_contract='bounded_refs_v2')['stage'] == 'complete'
    assert len(client.prompts) == calls
    assert 'reference_contract' not in read_json(original / 'manifest.json')
    assert read_json(upgraded / 'manifest.json')['reference_contract'] == 'bounded_refs_v2'
    with pytest.raises(ValueError, match='inputs or generator changed'):
        repair.repair(wiki, messages, original, 'person', 'self', {}, client=client,
                      reference_contract='bounded_refs_v2')


@pytest.mark.parametrize('source_contract', ['legacy_v1', 'structured_source_v2'])
def test_structured_bounded_identifiers_and_legacy_cache_reuse(tmp_path, source_contract):
    rows, _, value = fixture(tmp_path)
    facts = [dict(id='fact-original', lines=[1, 2, 3], summary='各自的称呼。')]
    client = ReferenceClient(value)
    args = (client, tmp_path / 'bounded', META, facts, rows)
    result, mapping = structured.extract(*args, contract=source_contract,
                                        reference_contract='bounded_refs_v2')
    assert result == value and mapping == {'F1': 'fact-original'}
    prompt, schema = client.requests[0]
    assert prompt.startswith(repair.REFERENCE_PROMPT)
    properties = schema['properties']['addresses']['items']['properties']
    assert properties['lines']['items']['enum'] == [1, 2, 3]
    assert properties['utterance_line']['enum'] == [1, 2, 3]
    assert properties['fact_ids']['items']['enum'] == ['F1']
    assert schema['properties']['reviews']['items']['properties']['fact_id']['enum'] == ['F1']
    legacy = (client, tmp_path / 'legacy', META, facts, rows)
    structured.extract(*legacy)
    calls = len(client.requests)
    assert structured.extract(*legacy, contract=source_contract,
                              reference_contract='bounded_refs_v2')[0] == value
    assert len(client.requests) == calls


def test_reconciliation_bounded_record_ids_reuses_legacy_cache(tmp_path):
    rows, data = prepared(tmp_path)
    groups = reconcile.groups_for(data)
    records = groups[0]['records']
    reviews = [decision(record, 'keep') for record in records]
    client = ReferenceClient(dict(reviews=reviews))
    assert reconcile.extract(client, tmp_path / 'bounded', data, groups, rows,
                             reference_contract='bounded_refs_v2') == reviews
    properties = client.requests[0][1]['properties']['reviews']['items']['properties']
    assert properties['record_id']['enum'] == sorted(record['id'] for record in records)
    assert properties['lines']['items']['enum'] == [1, 2, 3]
    reconcile.extract(client, tmp_path / 'legacy', data, groups, rows)
    calls = len(client.requests)
    assert reconcile.extract(client, tmp_path / 'legacy', data, groups, rows,
                             reference_contract='bounded_refs_v2') == reviews
    assert len(client.requests) == calls


def test_reconciliation_feedback_identifies_exact_missing_and_unexpected_ids(tmp_path):
    rows, data = prepared(tmp_path)
    records = reconcile.groups_for(data)[0]['records']
    reviews = [decision(record, 'keep') for record in records]
    missing = reviews[0]['record_id']
    reviews[0]['record_id'] = 'mistyped-id'
    with pytest.raises(ValueError) as failure:
        reconcile.validate(dict(reviews=reviews), records, rows, data)
    assert missing in str(failure.value)
    assert 'mistyped-id' in str(failure.value)
    assert all(label in str(failure.value) for label in ('expected=', 'received=', 'missing=', 'unexpected='))


def test_structured_feedback_preserves_uncertain_targets_and_requires_real_fact_links(tmp_path):
    rows, _, value = fixture(tmp_path)
    value['addresses'][0].update(target_ref='', usage='direct')
    with pytest.raises(ValueError, match='unresolved target') as failure:
        structured.validate(value, {'F1': 'fact-original'}, rows, META)
    assert all(fragment in str(failure.value) for fragment in
               ('line=1', "target_ref=''", "usage='direct'", 'unrelated known entity'))
    value['addresses'][0]['usage'] = 'uncertain'
    structured.validate(value, {'F1': 'fact-original'}, rows, META)
    value['reviews'].append(dict(fact_id='F2', disposition='represented', reason='无记录链接。'))
    with pytest.raises(ValueError) as failure:
        structured.validate(value, {'F1': 'fact-original', 'F2': 'unlinked'}, rows, META)
    assert 'F2' in str(failure.value) and 'represented' in str(failure.value)
