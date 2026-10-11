"""Per-request training isolation without mutable retriever state or fallback."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from src.generator import generator as runtime
from src.generator.generator import GenerationConstraints, ReplyGenerator
from src.infrastructure.config import ConfigError


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    (tmp_path / 'persona.md').write_text('synthetic persona')
    pool = tmp_path / 'fewshot_pool.jsonl'
    pool.write_text('synthetic pool')
    source = {'valid': True}

    def check():
        if not source['valid']:
            raise ConfigError('source changed')

    class Retriever:
        history_policy = None

        def __init__(self):
            self.fault = None
            self.hook = None
            self.calls = []

        def is_approved(self):
            return True

        def retrieve(self, **kwargs):
            self.calls.append({**kwargs, 'exclude_ids': set(kwargs['exclude_ids'])})
            if self.hook == 'retrieve':
                source['valid'] = False
            if self.fault == 'retrieve':
                raise RuntimeError('retrieval failed')
            ids = ('blocked', 'ok', 'target')
            if self.fault != 'excluded-row':
                ids = [key for key in ids if key not in kwargs['exclude_ids']]
            return [{'id': key, 'context': ['question-' + key], 'reply': ['answer-' + key]}
                    for key in ids]

        def render_selected(self, rows, max_chars):
            if self.hook == 'render':
                source['valid'] = False
            if self.fault == 'render':
                raise RuntimeError('render failed')
            ids = [row['id'] for row in rows]
            if self.fault == 'excluded-render':
                ids.append('blocked')
            return ','.join(ids)[:max_chars], ids

    class Client:
        def __init__(self):
            self.calls = []
            self.hook = None
            self.response = '{"replies":["synthetic reply"]}'

        def cache_identity(self):
            return {'model': 'synthetic'}

        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            if self.hook:
                self.hook()
            return self.response

    retriever, client = Retriever(), Client()
    monkeypatch.setattr(runtime, 'open_session', lambda **kwargs: retriever)
    settings = {'evaluation': {'few_shots_per_case': 3, 'few_shots_char_budget': 1000}}
    generator = ReplyGenerator(settings, {}, client, prompt_root=tmp_path, pool_path=pool)
    case = {'case_id': 'target', 'chat_type': 'private',
            'context': [{'sender': 'other', 'text': 'old'}, {'sender': 'other', 'text': 'latest'}]}
    return SimpleNamespace(generator=generator, retriever=retriever, client=client, source=source,
                           check=check, case=case, settings=settings, root=tmp_path)


def test_constraints_copy_and_freeze_exclusions():
    excluded = {'blocked', 12}
    constraints = GenerationConstraints(excluded, lambda: None)
    excluded.clear()
    assert constraints.excluded_ids == frozenset({'blocked', '12'})
    with pytest.raises(FrozenInstanceError):
        constraints.excluded_ids = frozenset()


@pytest.mark.parametrize('excluded,check', [('id', lambda: None), (b'id', lambda: None),
                                          (None, lambda: None), (set(), None)])
def test_invalid_constraints_rejected(excluded, check):
    with pytest.raises(ConfigError, match='生成约束'):
        GenerationConstraints(excluded, check)


def test_public_generation_merges_case_exclusion_and_keeps_retriever_state(isolated):
    constraints = GenerationConstraints({'blocked'}, isolated.check)
    assert isolated.generator.generate(isolated.case, constraints=constraints)['replies'] == ['synthetic reply']
    assert isolated.retriever.calls[0]['exclude_ids'] == {'blocked', 'target'}
    assert '<style_examples>\nok\n</style_examples>' in isolated.client.calls[0][1]['content']
    assert isolated.generator._retriever is isolated.retriever
    assert not hasattr(isolated.retriever, 'blocked')
    assert not hasattr(isolated.retriever, 'failure')


@pytest.mark.parametrize('fault', ['retrieve', 'render', 'excluded-row', 'excluded-render'])
def test_constrained_retrieval_and_render_fail_closed_before_llm(isolated, fault):
    isolated.retriever.fault = fault
    with pytest.raises(ConfigError, match='禁止降级继续'):
        isolated.generator.generate(isolated.case,
            constraints=GenerationConstraints({'blocked'}, isolated.check))
    assert isolated.client.calls == []


def test_request_constraints_and_failures_do_not_leak_to_next_case(isolated):
    isolated.retriever.fault = 'retrieve'
    with pytest.raises(ConfigError):
        isolated.generator.generate(isolated.case,
            constraints=GenerationConstraints({'blocked'}, isolated.check))
    isolated.retriever.fault = None
    isolated.generator.generate(isolated.case, constraints=GenerationConstraints({'ok'}, isolated.check))
    assert isolated.retriever.calls[-1]['exclude_ids'] == {'ok', 'target'}
    assert '<style_examples>\nblocked\n</style_examples>' in isolated.client.calls[-1][1]['content']
    isolated.generator.generate(isolated.case)
    assert isolated.retriever.calls[-1]['exclude_ids'] == {'target'}


def test_unconstrained_ordinary_generation_retains_no_example_fallback(isolated):
    isolated.retriever.fault = 'retrieve'
    isolated.generator.generate(isolated.case)
    assert '<style_examples>\n(无)\n</style_examples>' in isolated.client.calls[0][1]['content']


@pytest.mark.parametrize('stage', ['before', 'retrieve', 'render', 'model', 'retry', 'cache'])
def test_source_changes_block_requests_results_retries_and_cache_hits(isolated, monkeypatch, stage):
    if stage == 'before':
        isolated.source['valid'] = False
    elif stage in ('retrieve', 'render'):
        isolated.retriever.hook = stage
    elif stage in ('model', 'retry'):
        isolated.client.hook = lambda: isolated.source.update(valid=False)
        if stage == 'retry':
            isolated.client.response = 'invalid JSON'
    else:
        def cached_result(*args):
            isolated.source['valid'] = False
            return {'replies': ['cached']}

        monkeypatch.setattr(runtime.cache, 'memo', cached_result)
    with pytest.raises(ConfigError, match='source changed'):
        isolated.generator.generate(isolated.case,
            constraints=GenerationConstraints({'blocked'}, isolated.check))
    assert len(isolated.client.calls) == (1 if stage in ('model', 'retry') else 0)


def test_disabled_retrieval_still_checks_request_sources(isolated):
    generator = ReplyGenerator(isolated.settings, {'retriever': {'enabled': False}},
                               isolated.client, prompt_root=isolated.root)
    constraints = GenerationConstraints({'blocked'}, isolated.check)
    generator.generate(isolated.case, constraints=constraints)
    assert isolated.retriever.calls == []
    isolated.source['valid'] = False
    with pytest.raises(ConfigError, match='source changed'):
        generator.generate(isolated.case, constraints=constraints)
    assert len(isolated.client.calls) == 1


def test_unconstrained_style_override_keeps_original_signature(isolated):
    class OriginalStyle(ReplyGenerator):
        def _style_block(self, case):
            return 'single example'

    generator = OriginalStyle(isolated.settings, {'retriever': {'enabled': False}},
                              isolated.client, prompt_root=isolated.root)
    generator.generate(isolated.case)
    assert '<style_examples>\nsingle example\n</style_examples>' in isolated.client.calls[0][1]['content']


@pytest.mark.parametrize('constrained', [False, True])
def test_original_source_check_override_preserves_request_constraints(isolated, constrained):
    checks = []

    class OriginalSourceCheck(ReplyGenerator):
        def _check_sources(self):
            checks.append('checked')

    generator = OriginalSourceCheck(isolated.settings, {'retriever': {'enabled': False}},
                                    isolated.client, prompt_root=isolated.root)
    checks.clear()
    constraints = GenerationConstraints({'blocked'}, isolated.check) if constrained else None
    generator.generate(isolated.case, constraints=constraints)
    assert checks and len(isolated.client.calls) == 1
    if constrained:
        isolated.source['valid'] = False
        with pytest.raises(ConfigError, match='source changed'):
            generator.generate(isolated.case, constraints=constraints)
        assert len(isolated.client.calls) == 1
