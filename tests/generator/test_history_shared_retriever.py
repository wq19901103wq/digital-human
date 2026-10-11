"""Sharing an index must preserve per-thread exclusions and reject source changes."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from src.generator.fewshot import shared_history as runtime
from src.generator.generator import GenerationConstraints, ReplyGenerator
from src.infrastructure.config import ConfigError


def fixture(tmp_path, monkeypatch):
    pool = tmp_path / 'fewshot_pool.jsonl'
    report = tmp_path / 'report.json'
    pool.write_text('pool')
    report.write_text('report')
    inputs = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (pool, report)}
    loaded = []

    class Original:
        history_policy = 'complete_before_input_v1'

        def __init__(self, path):
            loaded.append(path)

        def is_approved(self):
            return True

        def retrieve(self, exclude_ids, **kwargs):
            return [{'id': s} for s in ('a', 'b', 'c') if s not in exclude_ids]

        def render_selected(self, rows, max_chars):
            return str(rows)[:max_chars], [r['id'] for r in rows]

    monkeypatch.setattr(runtime.generator, 'open_session', Original)
    return pool, report, inputs, loaded, Original


def test_parallel_workers_and_retry_pools_load_once_preserve_exclusions(tmp_path, monkeypatch):
    pool, _, inputs, loaded, original = fixture(tmp_path, monkeypatch)
    (tmp_path / 'persona.md').write_text('synthetic persona')
    settings = {'evaluation': {'few_shots_per_case': 3, 'few_shots_char_budget': 1000}}
    case = {'case_id': 'c', 'chat_type': 'private', 'context': []}
    with runtime.shared_history_index(inputs):
        def one(blocked):
            generator = ReplyGenerator(settings, {}, None, prompt_root=tmp_path, pool_path=pool)
            constraints = GenerationConstraints({blocked}, lambda: None)
            return generator.build_prompt(case, constraints=constraints)

        for attempt in range(3):
            with ThreadPoolExecutor(max_workers=8) as executor:
                result = list(executor.map(one, ['a', 'b'] * 8))
            for index, messages in enumerate(result):
                selected = 'b' if index % 2 == 0 else 'a'
                blocked = 'a' if index % 2 == 0 else 'b'
                assert str([{'id': selected}]) in messages[1]['content']
                assert f"'id': '{blocked}'" not in messages[1]['content']
                assert "'id': 'c'" not in messages[1]['content']
        assert len(loaded) == 1
    assert runtime.generator.open_session is original


@pytest.mark.parametrize('changed', ['pool', 'report'])
def test_changed_file_rejected_before_original_retrieval(tmp_path, monkeypatch, changed):
    pool, report, inputs, _, original = fixture(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match='文件发生变化'):
        with runtime.shared_history_index(inputs):
            retriever = runtime.generator.open_session(path=pool)
            target = pool if changed == 'pool' else report
            target.write_text('changed')
            monkeypatch.setattr(original, 'retrieve', lambda **kw: pytest.fail('must reject before retrieval'))
            retriever.retrieve(exclude_ids=set())
    assert runtime.generator.open_session is original


def test_wrong_frozen_content_or_render_pool_rejected(tmp_path, monkeypatch):
    pool, report, inputs, loaded, _ = fixture(tmp_path, monkeypatch)
    inputs[str(pool.resolve())] = 'wrong'
    with runtime.shared_history_index(inputs):
        with pytest.raises(ConfigError, match='替换展示库'):
            runtime.generator.open_session(path=pool, render_path=report)
        with pytest.raises(ConfigError, match='冻结内容不符'):
            runtime.generator.open_session(path=pool)
    assert loaded == []


@pytest.mark.parametrize('method', ['retrieve', 'render_selected'])
def test_frozen_retrieval_and_render_calls_overlap(tmp_path, monkeypatch, method):
    pool, _, inputs, loaded, original = fixture(tmp_path, monkeypatch)
    barrier = Barrier(4)
    original_method = getattr(original, method)

    def parallel(instance, *args, **kwargs):
        barrier.wait(timeout=5)
        return original_method(instance, *args, **kwargs)

    monkeypatch.setattr(original, method, parallel)
    with runtime.shared_history_index(inputs):
        instance = runtime.generator.open_session(path=pool)

        def work(number):
            if method == 'retrieve':
                return instance.retrieve(exclude_ids={'b', 'c'})
            return instance.render_selected([{'id': 'a'}], max_chars=100)

        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(work, range(4)))
        assert results == ([[{'id': 'a'}]] * 4 if method == 'retrieve' else
                           [(str([{'id': 'a'}]), ['a'])] * 4)
        assert len(loaded) == 1


@pytest.mark.parametrize('changed', ['pool', 'report'])
@pytest.mark.parametrize('method', ['retrieve', 'render_selected'])
def test_file_changed_during_original_call_is_rejected(tmp_path, monkeypatch, changed, method):
    pool, report, inputs, _, original = fixture(tmp_path, monkeypatch)
    original_method = getattr(original, method)
    target = pool if changed == 'pool' else report

    def changing(instance, *args, **kwargs):
        result = original_method(instance, *args, **kwargs)
        target.write_text('changed during call')
        return result

    monkeypatch.setattr(original, method, changing)
    with runtime.shared_history_index(inputs):
        instance = runtime.generator.open_session(path=pool)
        with pytest.raises(ConfigError, match='文件发生变化'):
            if method == 'retrieve':
                instance.retrieve(exclude_ids=set())
            else:
                instance.render_selected([{'id': 'a'}], max_chars=100)
