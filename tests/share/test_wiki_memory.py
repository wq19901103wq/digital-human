"""Memory admits existing Wiki facts by exact account, source and case cutoff."""
import json
from types import SimpleNamespace

import pytest

from src.infrastructure.config import ConfigError
from src.data.history.sources import Message
from src.share.knowledge_view import compact
from src.share.memory.wiki_lookup import WikiLookup, project_structured


@pytest.fixture
def wiki_case(tmp_path):
    chat = 'group:self:room'
    case = {'case_id': 'case', 'chat_type': 'group', 'chat_name': 'room',
            'source_chat_id': 'room', 'source_span': {'chat_id': chat, 'start': 0, 'reply_start': 1},
            'input_cutoff': {'timestamp': 100}, 'reply_message_ids': ['answer'],
            'history_excluded_chat_ids': ['group:self:heldout'],
            'context': [{'sender': '同名', 'text': '聊聊公司', 'is_self': False}],
            'human_reply': ['SECRET_ANSWER']}
    msg = Message('同名', '过去消息', False, 80, 'past', 'group', 'room', 'room', {'sender_id': 'peer'})
    sources = SimpleNamespace(directory=tmp_path, hashes={'messages.jsonl': 'source'},
                              chats={chat: [msg]}, check=lambda: None)
    people = [{'id': 'person:' + name, 'name': '同名', 'status': state}
              for name, state in [('self', 'source_account_bound'), ('peer', 'source_account_bound'),
                                  ('pending', 'pending')]]
    knowledge = {'source_manifest': {'messages': {'sha256': 'source'}}, 'identities': people,
                 'pages': [{'id': p['id'], 'title': p['name'], 'summary': 'SECRET_SUMMARY'} for p in people]
                          + [{'id': chat, 'title': '群', 'summary': 'SECRET_SUMMARY'}],
                 'evidence': {'e': {'kind': 'raw_message', 'message_id': 'past',
                                     'chat_id': chat, 'observed_at': 80}},
                 'fact_claims': [{'id': 'work', 'page_ids': ['person:self'], 'subject_id': 'person:self',
                                 'predicate': 'employment_affiliation', 'value': '公司', 'status': '有据',
                                 'observed_at': [80], 'evidence_refs': ['e'], 'description': '本人公司'}]}
    path = tmp_path / 'content' / 'knowledge.json'
    path.parent.mkdir()
    path.write_text(json.dumps(knowledge))
    return WikiLookup(path, sources), case


def test_account_pages_and_fact_ownership_without_review_summary(wiki_case):
    obj, case = wiki_case
    result = obj.query(case)
    assert {p['id'] for p in result['wiki_pages']} == {'person:self', 'person:peer', 'group:self:room'}
    assert result['self_person_id'] == 'person:self'
    assert result['speaker_bindings'][0]['person_id'] == 'person:peer'
    assert result['facts'][0]['subject_id'] == 'person:self'
    assert result['facts'][0]['description'] == '本人公司'
    assert 'SECRET' not in json.dumps(result)
    case['source_span']['chat_id'] = 'private:self:peer'
    obj.sources.chats['private:self:peer'] = obj.sources.chats['group:self:room']
    assert {p['id'] for p in obj.query(case)['wiki_pages']} == {'person:self', 'person:peer'}


@pytest.mark.parametrize('reason', ['future', 'same_second', 'answer', 'heldout', 'account', 'correction', 'unlocated'])
def test_ineligible_knowledge_never_leaves_memory(wiki_case, reason):
    obj, case = wiki_case
    fact = obj.knowledge['fact_claims'][0]
    if reason in ('future', 'same_second'):
        case['input_cutoff']['timestamp'] = 79 if reason == 'future' else 80
    elif reason == 'answer':
        case['reply_message_ids'].append('past')
    elif reason == 'heldout':
        case['history_excluded_chat_ids'].append('group:self:room')
    elif reason == 'account':
        obj.records['past'] = ('group:another:room', obj.records['past'][1])
        obj.knowledge['evidence']['e']['chat_id'] = 'group:another:room'
    elif reason == 'correction':
        fact['scope_case_id'] = case['case_id']
    else:
        obj.knowledge['evidence']['e']['kind'] = 'source_wiki'
    assert not obj.query(case)['facts']


@pytest.mark.parametrize('reason', ['allowed', 'future', 'same_second', 'answer', 'heldout',
                                   'account', 'sender', 'hash', 'unrelated_page'])
def test_structured_wiki_uses_source_bound_case_projection(wiki_case, reason):
    obj, case = wiki_case
    obj.knowledge = dict(schema='wiki_structured_v1', coverage={'sha256': 'source'},
        subject=dict(account='peer', self_account='self', entity_id='peer-id'),
        entities=[dict(id='peer-id', account='peer', kind='person', label='同名',
                       identity_status='exact_account', evidence_refs=['e'])],
        evidence={'e': dict(kind='raw_message', message_id='past', chat_id='group:self:room',
                            timestamp=80, sha256='source', sender_id='peer', is_self=False)},
        attributes=[dict(id='work', subject_id='peer-id', field='job', value='工程师',
                         evidence_refs=['e'], mode='self_report', polarity='negated',
                         first_observed_at=80, last_observed_at=80)],
        relations=[], events=[], addresses=[])
    if reason in ('future', 'same_second'):
        case['input_cutoff']['timestamp'] = 79 if reason == 'future' else 80
    elif reason == 'answer':
        case['reply_message_ids'].append('past')
    elif reason == 'heldout':
        case['history_excluded_chat_ids'].append('group:self:room')
    elif reason == 'account':
        obj.knowledge['subject']['self_account'] = 'another'
    elif reason == 'sender':
        obj.knowledge['evidence']['e']['sender_id'] = 'another'
    elif reason == 'hash':
        obj.knowledge['evidence']['e']['sha256'] = 'wrong'
    elif reason == 'unrelated_page':
        obj.knowledge['subject']['account'] = 'stranger'
    obj.path.write_text(json.dumps(obj.knowledge))
    restored = WikiLookup(obj.path, obj.sources)
    view = restored.query(case)
    assert bool(view['facts']) == (reason == 'allowed')
    if reason == 'allowed':
        assert view['facts'][0]['subject_id'] == 'peer-id'
        assert view['facts'][0]['polarity'] == 'negated'
        assert next(p for p in view['identities'] if p['role'] == 'self')['account'] == 'self'


@pytest.mark.parametrize('reason', ['future', 'same_second', 'answer', 'heldout', 'missing'])
@pytest.mark.parametrize('role', ['subject_id', 'object_id', 'speaker_id', 'target_id', 'participant'])
def test_prior_fact_cannot_borrow_excluded_identity_binding(wiki_case, reason, role):
    obj, case = wiki_case
    chat = 'group:self:heldout' if reason == 'heldout' else 'group:self:room'
    timestamp = 101 if reason == 'future' else 100 if reason == 'same_second' else 90
    identity = Message('后来确认的名字', '身份来源', True, timestamp, 'identity',
                       'group', 'room', 'room', {'sender_id': 'self'})
    obj.sources.chats.setdefault(chat, []).append(identity)
    if reason == 'answer':
        case['reply_message_ids'].append('identity')
    fact = dict(id='old-fact', field='job', value='过去事实', evidence_refs=['e'])
    fact.update({'participants': [dict(entity_id='bound-later', role='actor')]}
                if role == 'participant' else {role: 'bound-later'})
    obj.knowledge = dict(schema='wiki_structured_v1', coverage={'sha256': 'source'},
        subject=dict(account='peer', self_account='self', entity_id='peer-id'),
        entities=[dict(id='peer-id', account='peer', kind='person', label='原始账号',
                       identity_status='exact_account', evidence_refs=['e']),
                  dict(id='bound-later', account='self', kind='person', label='后来确认的名字',
                       identity_status='exact_account', evidence_refs=[] if reason == 'missing'
                       else ['identity'])],
        evidence={'e': dict(kind='raw_message', message_id='past', chat_id='group:self:room',
                            timestamp=80, sha256='source', sender_id='peer', is_self=False),
                  'identity': dict(kind='raw_message', message_id='identity', chat_id=chat,
                            timestamp=timestamp, sha256='source', sender_id='self', is_self=True)},
        attributes=[fact], relations=[], events=[], addresses=[])
    obj.path.write_text(json.dumps(obj.knowledge))
    restored = WikiLookup(obj.path, obj.sources)
    view = restored.query(case)
    assert not view['facts']
    assert all(e['id'] != 'bound-later' for e in view['identities'])
    assert view['self_person_id'] != 'bound-later'
    assert '后来确认的名字' not in json.dumps(view, ensure_ascii=False)


def test_projection_selects_exact_page_and_retains_uncertainty_with_budget(tmp_path):
    from tests.share.test_wiki_prompt import knowledge
    data = knowledge(tmp_path)
    assert project_structured(data, {'self', 'stranger'}, 'private:self:stranger', lambda _: True) is None
    assert project_structured(data, {'person'}, 'private:other:person', lambda _: True) is None
    result = project_structured(data, {'person'}, 'private:self:person', lambda ref: ref == '1',
                                max_chars=1800)
    assert result['coverage']['eligible'] == 2
    assert result['coverage']['included'] + result['coverage']['omitted_for_budget'] == 2
    assert len(compact(result)) <= 1800
    person = next(entity for entity in result['identities'] if entity['account'] == 'person')
    assert person['name'] == 'person'
    assert all(fact['last_observed_at'] == 1000 for fact in result['facts'])


def test_lookup_rejects_unbound_history(wiki_case):
    lookup, _ = wiki_case
    lookup.sources.hashes['messages.jsonl'] = 'different-source'
    with pytest.raises(ConfigError, match='原始消息来源'):
        WikiLookup(lookup.path, lookup.sources)


@pytest.mark.parametrize('during_query', [False, True])
def test_lookup_rejects_share_changes_before_or_during_query(wiki_case, monkeypatch, during_query):
    lookup, case = wiki_case
    original = lookup._projection

    def changed_query(case):
        result = original(case)
        lookup.path.write_text('{}')
        return result

    if during_query:
        monkeypatch.setattr(lookup, '_projection', changed_query)
    else:
        lookup.path.write_text('{}')
    with pytest.raises(ConfigError, match='Share 内容发生变化'):
        lookup.query(case)


def test_lookup_rechecks_history_after_query(wiki_case, monkeypatch):
    lookup, case = wiki_case
    checks = []

    def changed_history():
        checks.append(True)
        if len(checks) == 2:
            raise ConfigError('history changed')

    monkeypatch.setattr(lookup.sources, 'check', changed_history)
    with pytest.raises(ConfigError, match='history changed'):
        lookup.query(case)
