"""Current-turn selection uses complete eligible history without new LLM calls."""
import json

import pytest

from src.config import ConfigError, load_settings
from src.generator.few_shot import PersonaFewShotRetriever
from src.generator.generator import ReplyGenerator
from src.generator.selection import POLICY, select
from test_leakage_guards import env as env


class Renderer:
    render_selected = staticmethod(PersonaFewShotRetriever.render)


def row(identity, context, reply):
    return {'id': identity, 'context': [context], 'reply': [reply]}


def choose(context, latest, *, count=3, budget=2500):
    return select(context, latest, count=count, budget=budget, retriever=Renderer())


def test_latest_turn_breaks_context_ranking_tie_and_is_deterministic():
    old = row('old-topic', '之前说电影', '挺好看的')
    new = row('current-topic', '后来吃什么', '去吃火锅吧')
    selected, trace = choose([old, new], [new, old], count=1)
    assert selected == [new]
    assert trace['selected'][0]['ranks'] == {'context': 2, 'latest': 1}
    assert choose([old, new], [new, old], count=1) == (selected, trace)


def test_duplicate_content_removed_and_near_duplicate_penalized():
    a = row('a', '周末一起看电影怎么样', '好啊周末见')
    duplicate = {**a, 'id': 'duplicate'}
    near = row('near', '周末一起看电影怎么样呀', '好啊周末见啦')
    different = row('different', '下班吃点啥', '走去吃火锅')
    rows = [a, duplicate, near, different]
    selected, trace = choose(rows, rows)
    assert selected[0] == a and selected[1] == different
    assert duplicate not in selected
    assert {'id': 'duplicate', 'reason': 'duplicate_content'} in trace['skipped']
    assert 'retrieval_features' not in a  # No mutation of source rows.


def test_skip_oversized_example_and_enforce_full_render_budget():
    long = row('long', '超长' * 3000, '长示例')
    a, b = row('a', '今天下雨', '带伞'), row('b', '电影怎么样', '好看')
    selected, trace = choose([long, a, b], [long, a, b], count=2, budget=1000)
    assert selected == [a, b]
    assert trace['skipped'][0] == {'id': 'long', 'reason': 'char_budget'}
    rendered, _ = Renderer.render_selected([a], max_chars=1000)
    assert choose([a], [a], budget=len(rendered) - 1)[0] == []
    assert choose([a], [a], budget=len(rendered))[0] == [a]
    assert choose([a], [a], count=0)[0] == []


def generator(env, selection=POLICY):
    cfg = {**env.cfg, 'retriever': {'enabled': True, 'selection': selection}}
    return ReplyGenerator(load_settings(), cfg, env.client, prompt_root=env.gdir,
                          pool_path=env.source.directory / 'fewshot_pool.jsonl')


def test_both_routes_use_original_history_guard_and_no_llm(env, monkeypatch):
    model = generator(env)
    case = env.source.roles['development'][0]
    calls = []
    retrieve = model._retriever.retrieve
    def record(**kwargs):
        calls.append(kwargs)
        return retrieve(**kwargs)
    monkeypatch.setattr(model._retriever, 'retrieve', record)
    prompt = json.dumps(model.build_prompt(case), ensure_ascii=False)
    assert calls and all(x['history_case'] is case for x in calls)
    assert all(x['exclude_ids'] == {case['case_id']} for x in calls)
    for future in env.source.cases[3:]:
        assert all(answer not in prompt for answer in future['human_reply'])
    assert env.calls == []
    source = env.source.directory / 'messages.jsonl'
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ConfigError):
        model.build_prompt(case)


def test_default_keeps_original_first_three_and_config_must_be_explicit(env, monkeypatch):
    model = generator(env, None)
    rows = [row(str(i), '电影', str(i)) for i in range(5)]
    calls = []
    monkeypatch.setattr(model._retriever, 'retrieve', lambda **kw: calls.append(kw) or rows)
    monkeypatch.setattr(model._retriever, 'render_selected', Renderer.render_selected)
    assert model._style_block(env.source.roles['development'][0]) == Renderer.render_selected(rows[:3])[0]
    assert len(calls) == 1
    with pytest.raises(ConfigError, match='selection'):
        generator(env, 'unknown-policy')
