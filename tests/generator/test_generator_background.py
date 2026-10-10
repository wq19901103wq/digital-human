"""Gen plans optional memory calls and assembles admitted background material."""
import json
from types import SimpleNamespace

import pytest

from src.infrastructure.config import ConfigError
from src.generator.context import background
from src.share import versions as shares
from tests.share.test_wiki_memory import wiki_case


@pytest.fixture
def setup(wiki_case, monkeypatch):
    lookup, case = wiki_case
    backend = SimpleNamespace(identity=lambda: {'model': 'synthetic'}, check=lambda: None)
    monkeypatch.setattr(background, 'configured', lambda *args: backend)
    config = {'background': {'policy': background.POLICY, 'memory': {
        'index': 'synthetic', 'model': 'synthetic', 'identity': backend.identity()}}}
    obj = background.Background(config, {'dir': lookup.path.parent.parent}, lookup.sources)
    return obj, case, config


@pytest.mark.parametrize('calls', [[], [{'query': '公司', 'top_k': 1}]])
def test_planner_and_prompt_use_memory_query_without_reading_answers(setup, monkeypatch, calls):
    obj, case, _ = setup
    expected = obj.wiki.query(case)
    queries, searches = [], []

    def query(received):
        assert received is case
        queries.append(received)
        return expected

    class Client:
        def chat(self, messages, **kwargs):
            assert 'SECRET' not in json.dumps(messages)
            assert 'human_reply' not in json.dumps(messages)
            assert json.loads(messages[1]['content'])['background'] == expected
            return json.dumps({'calls': calls})

    class Memory:
        def __init__(self, directory, received, wiki, *, backend):
            assert calls
            assert directory == obj.sources.directory
            assert received is case and wiki is expected and backend is obj.backend

        def search(self, **kwargs):
            searches.append(kwargs)
            return {'history': []}

    monkeypatch.setattr(obj.wiki, 'query', query)
    monkeypatch.setattr(background, 'SearchMemory', Memory)
    message = obj.message(case, Client())
    assert queries == [case] and searches == calls
    assert message['role'] == 'system'
    assert 'SECRET' not in message['content']
    payload = json.loads(message['content'].removeprefix(background._RULES + '\n'))
    assert payload['wiki'] == expected
    assert payload['search_memory'] == [dict(name='search_memory', arguments=call,
        result={'history': []}) for call in calls]


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


def test_generator_rejects_changed_memory_identity(setup):
    obj, _, config = setup
    config['background']['memory']['identity'] = {'model': 'different'}
    with pytest.raises(ConfigError, match='资产与冻结配置不一致'):
        background.Background(config, {'dir': obj.wiki.path.parent.parent}, obj.sources)


def test_generator_rejects_share_changes_during_planning(setup):
    obj, case, _ = setup

    class Client:
        def chat(self, messages, **kwargs):
            obj.wiki.path.write_text('{}')
            return json.dumps({'calls': []})

    with pytest.raises(ConfigError, match='Share 内容发生变化'):
        obj.message(case, Client())
