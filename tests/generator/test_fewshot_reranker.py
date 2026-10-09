"""Reranking uses eligible history, request caching and source guards end to end."""
import io
import json

import pytest

from src import cache, llm, tracing
from src.config import ConfigError, load_settings
from src.generator.generator import ReplyGenerator
from src.generator.reranker import FewShotReranker
from test_leakage_guards import env as env


@pytest.fixture
def reranked(env, monkeypatch):
    monkeypatch.setenv('RERANK_URL', 'https://example.invalid/api/coding')
    monkeypatch.setenv('RERANK_KEY', 'synthetic')
    calls, answers, hooks = [], [], []
    def send(request, **kwargs):
        body = json.loads(request.data)
        calls.append(body)
        if hooks:
            hooks.pop(0)()
        payload = json.loads(body['messages'][0]['content'])
        raw = answers.pop(0) if answers else json.dumps({'indices': list(reversed(range(payload['select_count'])))})
        return io.BytesIO(json.dumps({'content': [{'type': 'text', 'text': raw}],
            'stop_reason': 'end_turn'}).encode())
    monkeypatch.setattr(llm.urllib.request, 'urlopen', send)
    cfg = {**env.cfg, 'max_shots_per_case': 2, 'retriever': {'enabled': True, 'reranker': {
        'model': 'synthetic-reranker', 'base_url_env': 'RERANK_URL', 'api_key_env': 'RERANK_KEY'}}}
    generator = ReplyGenerator(load_settings(), cfg, env.client, prompt_root=env.gdir,
                                pool_path=env.source.directory / 'fewshot_pool.jsonl')
    return generator, calls, answers, hooks


def build_cached(env, generator, case):
    trace = tracing.CaseTrace(env.root / 'experiments/reranking', case, {'dataset': 'development'})
    with trace.operation('generation', 'candidate', 0, {}):
        result = generator.build_prompt(case)
    trace.finish('ok')
    return result


def test_real_history_filtered_answers_hidden_and_cache_reused(env, reranked):
    generator, calls, _, _ = reranked
    case = env.source.roles['development'][0]
    first = build_cached(env, generator, case)
    assert build_cached(env, generator, case) == first
    assert len(calls) == 1
    wire = json.dumps(calls, ensure_ascii=False)
    for future in env.source.cases[3:]:
        assert all(answer not in wire for answer in future['human_reply'])
    payload = json.loads(calls[0]['messages'][0]['content'])
    assert payload['select_count'] == 2
    assert payload['candidates']
    assert all(set(m) == {'sender', 'text'} for m in payload['current']['context'])
    assert 'human_reply' not in wire and 'reply_message_ids' not in wire
    # A changed current input must not use the previous selection request.
    build_cached(env, generator, env.source.roles['development'][1])
    assert len(calls) == 2


@pytest.mark.parametrize('raw', ['{}', '{"indices":[0,0]}', '{"indices":[-1,0]}',
    '{"indices":[0,999]}', '{"indices":[true,0]}', '{"indices":[0]}', 'not json'])
def test_invalid_selections_never_cached_or_silently_fallback(env, reranked, raw):
    generator, calls, answers, _ = reranked
    answers.extend([raw, raw])
    case = env.source.roles['development'][0]
    with pytest.raises(RuntimeError, match='两次'):
        build_cached(env, generator, case)
    assert len(calls) == 2
    build_cached(env, generator, case)
    assert len(calls) == 3
    build_cached(env, generator, case)
    assert len(calls) == 3


@pytest.mark.parametrize('timing', ['before', 'during', 'after_cached'])
def test_changed_sources_rejected_including_cache_hits(env, reranked, timing):
    generator, calls, _, hooks = reranked
    case = env.source.roles['development'][0]
    source = env.source.directory / 'messages.jsonl'
    def mutate():
        source.write_text(source.read_text() + '\n')
    if timing == 'after_cached':
        build_cached(env, generator, case)
    if timing == 'during':
        hooks.append(mutate)
    else:
        mutate()
    with pytest.raises(ConfigError):
        build_cached(env, generator, case)
    assert len(calls) == (0 if timing == 'before' else 1)
