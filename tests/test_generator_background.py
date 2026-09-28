"""Gen background is selected by account and source time before any model call."""
import json
from types import SimpleNamespace

import pytest

from src.config import ConfigError
from src.generator import background
from src.generator.history_sources import Message
from src.iteration import shares


@pytest.fixture
def setup(tmp_path, monkeypatch):
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
    backend = SimpleNamespace(identity=lambda: {'model': 'synthetic'}, check=lambda: None)
    monkeypatch.setattr(background, 'configured', lambda *args: backend)
    config = {'background': {'policy': background.POLICY, 'memory': {
        'index': 'synthetic', 'model': 'synthetic', 'identity': backend.identity()}}}
    obj = background.Background(config, {'dir': tmp_path}, sources)
    return obj, case, config


def test_account_pages_and_fact_ownership_without_review_summary(setup):
    obj, case, _ = setup
    result = obj.projection(case)
    assert {p['id'] for p in result['wiki_pages']} == {'person:self', 'person:peer', 'group:self:room'}
    assert result['self_person_id'] == 'person:self'
    assert result['speaker_bindings'][0]['person_id'] == 'person:peer'
    assert result['facts'][0]['subject_id'] == 'person:self'
    assert result['facts'][0]['description'] == '本人公司'
    assert 'SECRET' not in json.dumps(result)
    case['source_span']['chat_id'] = 'private:self:peer'
    obj.sources.chats['private:self:peer'] = obj.sources.chats['group:self:room']
    assert {p['id'] for p in obj.projection(case)['wiki_pages']} == {'person:self', 'person:peer'}


@pytest.mark.parametrize('reason', ['future', 'same_second', 'answer', 'heldout', 'account', 'correction', 'unlocated'])
def test_ineligible_background_never_reaches_gen(setup, reason):
    obj, case, _ = setup
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
    assert not obj.projection(case)['facts']


@pytest.mark.parametrize('calls', [[], [{'query': '公司', 'top_k': 1}]])
def test_planner_sees_only_context_and_dispatches_optional_memory(setup, monkeypatch, calls):
    obj, case, _ = setup
    seen = []
    class Client:
        def chat(self, messages, **kwargs):
            assert 'SECRET' not in json.dumps(messages)
            assert 'human_reply' not in json.dumps(messages)
            return json.dumps({'calls': calls})
    class Memory:
        def __init__(self, *args, **kwargs):
            assert calls
        def search(self, **kwargs):
            seen.append(kwargs)
            return {'history': []}
    monkeypatch.setattr(background, 'SearchMemory', Memory)
    message = obj.message(case, Client())
    assert seen == calls
    assert message['role'] == 'system'
    assert 'SECRET' not in message['content']


def test_only_explicit_gen_policy_admits_filtered_runtime(setup, monkeypatch):
    _, _, config = setup
    info = {'id': 's-test'}
    monkeypatch.setattr(shares, 'validate_binding', lambda _: info)
    assert shares.require_runtime(config, consumer='gen') is info
    for consumer in (None, 'judge'):
        with pytest.raises(ConfigError, match='仅可审阅'):
            shares.require_runtime(config, consumer=consumer)
    with pytest.raises(ConfigError, match='仅可审阅'):
        shares.require_runtime({}, consumer='gen')
    monkeypatch.setattr(shares, 'validate_binding', lambda _: None)
    with pytest.raises(ConfigError, match='必须绑定'):
        shares.require_runtime(config, consumer='gen')


@pytest.mark.parametrize('reason', ['allowed', 'future', 'same_second', 'answer', 'heldout',
                                   'account', 'sender', 'hash', 'unrelated_page'])
def test_structured_wiki_uses_source_bound_case_projection(setup, reason):
    obj, case, config = setup
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
    restored = background.Background(config, {'dir': obj.path.parent.parent}, obj.sources)
    view = restored.projection(case)
    assert bool(view['facts']) == (reason == 'allowed')
    if reason == 'allowed':
        assert view['facts'][0]['subject_id'] == 'peer-id'
        assert view['facts'][0]['polarity'] == 'negated'
        assert next(p for p in view['identities'] if p['role'] == 'self')['account'] == 'self'


@pytest.mark.parametrize('reason', ['future', 'same_second', 'answer', 'heldout', 'missing'])
@pytest.mark.parametrize('role', ['subject_id', 'object_id', 'speaker_id', 'target_id', 'participant'])
def test_prior_fact_cannot_borrow_excluded_identity_binding(setup, reason, role):
    obj, case, config = setup
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
    restored = background.Background(config, {'dir': obj.path.parent.parent}, obj.sources)
    view = restored.projection(case)
    assert not view['facts']
    assert all(e['id'] != 'bound-later' for e in view['identities'])
    assert view['self_person_id'] != 'bound-later'
    assert '后来确认的名字' not in json.dumps(view, ensure_ascii=False)
