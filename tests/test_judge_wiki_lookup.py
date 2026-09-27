"""Direct Wiki selection uses accounts and scoped aliases, never similarity search."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from src.generator import history_sources
from src.judge import background_features, feature_inspection, wiki_lookup


@pytest.fixture
def inputs():
    chat = 'group:self:room'
    people = [
        {'id': 'person:self', 'name': '本人', 'status': 'source_account_bound'},
        {'id': 'person:peer', 'name': '同名', 'status': 'source_account_bound'},
        {'id': 'person:friend', 'name': '朋友', 'status': 'user_confirmed',
         'aliases': ['某哥'], 'decision': {'scope': {'chat_id': chat}}},
        {'id': 'person:pending', 'name': '同名', 'status': 'pending', 'aliases': ['待定哥']},
    ]
    pages = [{'id': p['id'], 'title': p['name'], 'summary': 'SECRET_REVIEW'} for p in people]
    pages.append({'id': chat, 'title': '测试群', 'summary': 'SECRET_REVIEW'})
    knowledge = {'identities': people, 'pages': pages, 'fact_claims': [
        {'id': 'work', 'page_ids': ['person:self'], 'subject_id': 'person:self',
         'predicate': 'employment_affiliation', 'value': '公司', 'status': '有据',
         'observed_at': [80], 'scope_chat_id': 'private:self:someone',
         'evidence_refs': ['source-1'], 'description': 'SECRET_REVIEW'},
        {'id': 'address', 'page_ids': ['person:friend', chat], 'subject_id': 'person:peer',
         'object_id': 'person:friend', 'predicate': 'addresses', 'value': '某哥',
         'status': '有据', 'observed_at': [90], 'scope_chat_id': chat},
    ], 'evidence': {'source-1': {'text': 'SECRET_SOURCE', 'quote': 'SECRET_QUOTE'}}}
    case = {'case_id': 'case', 'source_span': {'chat_id': chat},
            'input_cutoff': {'timestamp': 100}, 'context_message_ids': ['m1', 'm2'],
            'human_reply': ['SECRET_ANSWER']}
    blind = {'relationship': 'group', 'context_original': [
        '<messages_to_reply><message role="other">同名：你知道吗</message></messages_to_reply>'],
        'option_A': ['不知道'], 'option_B': ['问某哥'], 'label': 'SECRET_LABEL'}
    accounts = [{'message_id': 'm1', 'sender_id': 'peer', 'sender': '同名', 'is_self': False}]
    return knowledge, case, blind, accounts


def test_direct_pages_use_accounts_and_confirmed_mentions_without_review_prose(inputs):
    knowledge, case, blind, accounts = inputs
    projection = wiki_lookup.diagnostic_projection(*inputs)
    payload = projection['payload']
    assert {p['id'] for p in payload['wiki_pages']} == {
        'person:self', 'person:peer', 'person:friend', 'group:self:room'}
    assert payload['self_person_id'] == 'person:self'
    assert payload['speaker_bindings'][0]['person_id'] == 'person:peer'
    assert payload['facts'][0]['scope_chat_id'] == 'private:self:someone'
    assert payload['facts'][1]['subject_id'] == 'person:peer'
    assert projection['source_refs']['fact:1']['evidence_refs'] == ['source-1']
    assert projection['diagnostic_only'] and not projection['historical_admission']
    assert 'SECRET' not in json.dumps(background_features.build_input(blind, payload))
    swapped = dict(blind, option_A=blind['option_B'], option_B=blind['option_A'])
    assert wiki_lookup.diagnostic_projection(knowledge, case, swapped, accounts) == projection


def test_private_peer_and_self_are_selected_even_without_mentions(inputs):
    knowledge, case, blind, _ = inputs
    case['source_span']['chat_id'] = 'private:self:peer'
    payload = wiki_lookup.diagnostic_projection(knowledge, case, blind, [])['payload']
    assert {p['id'] for p in payload['wiki_pages']} == {'person:self', 'person:peer'}
    assert not payload['identities'][0]['aliases']


def test_scoped_ambiguous_and_pending_aliases_do_not_merge_people(inputs):
    knowledge, case, blind, accounts = inputs
    blind['option_A'] = ['问待定哥']
    duplicate = copy.deepcopy(knowledge['identities'][2])
    duplicate['id'] = 'person:another'
    knowledge['identities'].append(duplicate)
    payload = wiki_lookup.diagnostic_projection(*inputs)['payload']
    assert 'person:friend' not in {p['id'] for p in payload['wiki_pages']}
    assert payload['unresolved'] == [{'alias': '某哥', 'reason': 'ambiguous_confirmed_alias',
                                      'person_ids': ['person:another', 'person:friend']}]
    case['source_span']['chat_id'] = 'group:self:elsewhere'
    payload = wiki_lookup.diagnostic_projection(*inputs)['payload']
    assert all(p['id'] != 'person:friend' for p in payload['wiki_pages'])
    assert not payload['unresolved']


def test_missing_pages_and_unconfirmed_accounts_stay_explicit(inputs):
    knowledge, case, blind, accounts = inputs
    knowledge['pages'] = [p for p in knowledge['pages'] if p['id'] != 'person:peer']
    accounts += [dict(accounts[0], sender_id=None, message_id='m2'),
                 dict(accounts[0], sender_id='pending', message_id='m3'),
                 dict(accounts[0], sender_id=None, is_self=True, message_id='m4')]
    payload = wiki_lookup.diagnostic_projection(*inputs)['payload']
    states = {row['id']: row['status'] for row in payload['lookups']}
    assert states['person:peer'] == 'missing_page'
    assert states['person:pending'] == 'identity_unconfirmed'
    assert payload['speaker_bindings'][-1]['person_id'] == 'person:self'
    assert payload['unresolved'] == [{'sender': '同名', 'reason': 'missing_sender_account'}]


@pytest.mark.parametrize('change', [
    {'observed_at': [80, 101]}, {'observed_at': []},
    {'scope_case_id': 'different'}, {'status': '撤下无依据的职业归属'},
    {'page_ids': ['person:unselected']},
])
def test_unusable_facts_are_not_attached_to_selected_pages(inputs, change):
    knowledge, *_ = inputs
    knowledge['fact_claims'][0].update(change)
    result = wiki_lookup.diagnostic_projection(*inputs)
    assert all(row['claim_id'] != 'work' for row in result['source_refs'].values())


def test_case_scoped_correction_is_separately_reported(inputs):
    knowledge, case, *_ = inputs
    knowledge['fact_claims'][0].update(observed_at=[], scope_case_id=case['case_id'],
                                      status='案例范围内确认')
    result = wiki_lookup.diagnostic_projection(*inputs)
    assert result['case_correction_refs'] == ['fact:1']
    assert not result['historical_admission']


def test_context_accounts_come_from_exact_source_ids_not_names(inputs, tmp_path, monkeypatch):
    _, case, *_ = inputs
    messages = [SimpleNamespace(message_id=mid, sender='同名', is_self=False,
                               event={'sender_id': account})
                for mid, account in [('m1', 'peer'), ('m2', 'different'), ('answer', 'self')]]
    validated = []
    source = SimpleNamespace(validate=lambda row: validated.append(row),
                             chats={case['source_span']['chat_id']: messages})
    monkeypatch.setattr(history_sources, 'load', lambda directory: source)
    result = wiki_lookup.context_accounts(tmp_path, case)
    assert validated == [case]
    assert [row['sender_id'] for row in result] == ['peer', 'different']
    assert 'answer' not in {row['message_id'] for row in result}


def test_default_inspection_uses_direct_wiki_without_memory_search(inputs, tmp_path, monkeypatch):
    knowledge, case, blind, accounts = inputs
    share = tmp_path / 'share'
    (share / 'content').mkdir(parents=True)
    (share / 'content' / 'knowledge.json').write_text(json.dumps(knowledge))
    monkeypatch.setattr(feature_inspection.shares, 'load', lambda ref: {
        'dir': share, 'manifest': {'sha256': 'synthetic'}})
    monkeypatch.setattr(feature_inspection.versions, 'judge_dir', lambda ref: {'config': {'llm': {}}})
    monkeypatch.setattr(feature_inspection.versions, 'data_version_dir', lambda ref: tmp_path)
    monkeypatch.setattr(wiki_lookup, 'context_accounts', lambda directory, row: accounts)
    monkeypatch.setattr(feature_inspection, 'saved_case', lambda *args: {
        'case': case, 'mapping': {'blind_case': blind, 'human_option': 'A', 'candidate_option': 'B'},
        'old_features': None, 'versions': {'data_ref': 'd-0001'}, 'source_trace': 'a' * 32})
    from src.judge import memory_search

    def forbidden(*args, **kwargs):
        pytest.fail('direct Wiki must not plan or execute vector/history searches')

    monkeypatch.setattr(memory_search, 'enrich', forbidden)
    monkeypatch.setattr(memory_search, 'SearchMemory', forbidden)
    payloads = []

    def extract(client, payload):
        payloads.append(payload)
        return {key: {'features': {name: {'state': 'unknown'} for name in background_features.FEATURES}}
                for key in ('option_A', 'option_B')}

    monkeypatch.setattr(background_features, 'extract', extract)
    client = SimpleNamespace(cache_identity=lambda: {'model': 'synthetic'})
    output = tmp_path / 'diagnostic'
    result = feature_inspection.inspect(tmp_path, 'case', 's-0001', 'j-0001', output, client=client)
    assert result['status'] == 'ok' and not result['formal_accuracy_evidence']
    assert list(result['conditions']) == ['context_only', 'with_wiki']
    assert payloads[1]['background']['lookup_mode'] == wiki_lookup.VERSION
    assert 'search_memory' not in payloads[1]
    assert 'SECRET' not in json.dumps(payloads)
    assert not (output / 'cases.jsonl').exists()
