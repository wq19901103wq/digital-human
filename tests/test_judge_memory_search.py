"""Regression boundaries for application-dispatched diagnostic memory search."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import numpy as np

from src import cache, tracing
from src.generator.history import HistoryError
from src.generator.history_sources import Message
from src.judge import background_features as features
from src.judge import memory_search as memory


def message(ident, timestamp, text, *, self=False):
    return Message('本人' if self else '同事', text, self, timestamp, ident, 'group', '测试群', 'room',
                   {'sender_id': 'account' if self else 'peer'})


@pytest.fixture
def search(monkeypatch):
    case = {'case_id': 'sample', 'reply_message_ids': ['answer'],
            'context_message_ids': ['context'], 'input_cutoff': {'timestamp': 100},
            'source_span': {'chat_id': 'group:account:room'},
            'history_excluded_chat_ids': ['group:account:heldout']}
    sources = SimpleNamespace(hashes={'messages.jsonl': 'original-hash'}, checks=0,
        chats={'group:account:room': [
            message('past', 10, '同事 公司'), message('prior', 11, '我的公司', self=True),
            message('context', 98, '公司 当前上文'), message('answer', 99, '公司 当前答案'),
            message('equal', 100, '公司 同秒消息'), message('future', 101, '公司 未来消息')],
            'group:account:heldout': [message('heldout', 50, '公司 留出群')],
            'group:another:room': [message('another', 60, '公司 另一账号')]})
    def validate(row):
        assert row == case
        sources.validated = True
    def check():
        sources.checks += 1
    sources.validate, sources.check = validate, check
    monkeypatch.setattr(memory.history_sources, 'load', lambda path: sources)
    class Backend:
        def identity(self):
            return {'assets': 'test-vectors'}
        def check(self):
            pass
        def score(self, query, texts):
            self.seen = texts
            return np.array([.9] * len(texts)), {'dense_covered': len(texts)}
    result = memory.SearchMemory('frozen', case, {'identities': [], 'facts': []}, backend=Backend())
    assert sources.validated
    return result


def test_neighbor_messages_obey_time_answer_account_and_heldout_boundaries(search):
    result = search.search('公司', top_k=5)
    rows = [m for item in result['history'] for m in item['messages']]
    assert {m['message_id'] for m in rows} == {'past', 'prior'}
    assert len(result['history']) == 1  # Overlapping windows deduplicated.
    assert rows[1]['role'] == 'self'
    assert rows[1]['source_index'] == 1
    assert rows[0]['sender_id'] == 'peer' and rows[0]['person_id'] == 'person:peer'
    assert rows[1]['person_id'] == 'person:account'
    assert search.sources.checks == 2


def test_semantic_route_works_without_literal_match_and_only_scores_eligible_texts(search):
    result = search.search('工作单位')
    assert result['history'][0]['routes'] == ['dense']
    assert search.backend.seen == ['同事 公司', '我的公司']
    assert result['coverage']['dense_covered'] == 2


def test_keyword_route_uses_any_term_and_does_not_import_forbidden_text(search):
    result = search.search('公司 当前答案', top_k=5)
    assert result['history'][0]['routes'] == ['dense', 'keyword']
    assert all('当前答案' not in m['text'] for h in result['history'] for m in h['messages'])


def test_asset_changes_reject_even_a_cached_hit(search, tmp_path, monkeypatch):
    trace = tracing.CaseTrace(tmp_path / 'instance' / 'analysis' / 'memory', search.case, {})
    with trace.operation('search_memory', 'memory', 0, {}):
        search.search('公司')
        def changed():
            raise ValueError('asset changed')
        monkeypatch.setattr(search.backend, 'check', changed)
        with pytest.raises(ValueError, match='asset changed'):
            search.search('公司')
    trace.finish('ok')


def test_retrieval_is_cached_but_checks_sources_even_on_hit(search, tmp_path, monkeypatch):
    directory = tmp_path / 'instance' / 'analysis' / 'diagnostic'
    trace = tracing.CaseTrace(directory, search.case, {'dataset': 'diagnostic'})
    with trace.operation('search_memory', 'memory', 0, {}) as operation:
        first = search.search('公司')
        def unexpected(*args):
            pytest.fail('cached retrieval should not execute again')
        monkeypatch.setattr(search, '_search', unexpected)
        assert search.search('公司') == first
        assert search.sources.checks == 4
        assert any(event['kind'] == 'cache_hit' for event in operation['events'])
        def changed():
            raise HistoryError('历史来源改变')
        monkeypatch.setattr(search.sources, 'check', changed)
        with pytest.raises(HistoryError):
            search.search('公司')
    trace.finish('ok')


def test_retrieval_identity_binds_source_scope_and_wiki(search):
    original = cache.digest(search.identity())
    for field, value in [('excluded', {'other'}), ('blocked_ids', {'extra'}),
                         ('background', {'facts': [{'ref': 'fact:new'}]})]:
        before = getattr(search, field)
        setattr(search, field, value)
        assert cache.digest(search.identity()) != original
        setattr(search, field, before)
    search.sources.hashes['messages.jsonl'] = 'different'
    assert cache.digest(search.identity()) != original


@pytest.mark.parametrize('value', [None, [], {'calls': None}, {'calls': [None]},
    {'calls': [{'query': '公司', 'top_k': 1}] * 4},
    {'calls': [{'query': '公司', 'top_k': True}]},
    {'calls': [{'query': '公司', 'top_k': 6}]},
    {'calls': [{'query': '公司', 'top_k': 1, 'path': '/tmp/answer'}]},
    {'calls': [{'query': '!', 'top_k': 1}]},
    {'calls': [{'query': 'x' * 101, 'top_k': 1}]}])
def test_invalid_tool_arguments_rejected(value):
    with pytest.raises(ValueError):
        memory.parse_plan(json.dumps(value))


def test_tool_planner_only_receives_blind_payload_and_dispatches_allowed_calls(search):
    blind = {'relationship': 'group', 'context_original': ['某人问公司'],
             'option_A': ['问同事'], 'option_B': ['不知道'], 'label': 'SECRET_LABEL'}
    payload = features.build_input(blind, {'facts': [], 'identities': []})
    class Client:
        def run(self, prompt, schema):
            assert 'SECRET_LABEL' not in prompt
            assert 'reply_message_ids' not in prompt and 'human_option' not in prompt
            assert schema['properties']['calls']['type'] == 'array'
            return json.dumps({'calls': [{'query': '公司', 'top_k': 1}]})
    enriched = memory.enrich(Client(), payload, search)
    assert 'search_memory' not in payload
    assert enriched['search_memory'][0]['name'] == 'search_memory'
    ref = enriched['search_memory'][0]['result']['history'][0]['ref']
    value = {option: {'attribution_checks': [],
                     'features': {key: {'state': 'supported', 'evidence_refs': [ref], 'reason': '有据'}
                                  for key in features.FEATURES},
                     'routing': dict.fromkeys(('question_topic', 'question_target',
                                              'delegated_to', 'background_owner'), '未知')}
             for option in ('option_A', 'option_B')}
    value['context_attribution'] = []
    parsed = features.parse_response(json.dumps(value), enriched)
    assert features.categorical_features(parsed)['option_A']['entity_reference'] == 'supported'
    altered = copy.deepcopy(enriched)
    altered['search_memory'] = []
    with pytest.raises(ValueError):
        features.parse_response(json.dumps(value), altered)


def test_empty_plan_needs_no_retrieval(search, monkeypatch):
    class Client:
        def run(self, prompt, schema):
            return '{"calls": []}'
    monkeypatch.setattr(search, 'search', lambda *args: pytest.fail('unnecessary retrieval'))
    assert memory.enrich(Client(), {}, search)['search_memory'] == []
