"""Sharing an index must preserve per-thread exclusions and reject source changes."""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.generator import shared_history as runtime
from src.iteration.training_operations import ExcludingRetriever
from src.config import ConfigError


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

    monkeypatch.setattr(runtime.generator, 'PersonaFewShotRetriever', Original)
    return pool, report, inputs, loaded, Original


def test_parallel_workers_and_retry_pools_load_once_preserve_exclusions(tmp_path, monkeypatch):
    pool, _, inputs, loaded, original = fixture(tmp_path, monkeypatch)
    with runtime.shared_history_index(inputs):
        def one(blocked):
            retriever = ExcludingRetriever(runtime.generator.PersonaFewShotRetriever(path=pool))
            retriever.blocked = {blocked}
            assert retriever.is_approved()
            rows = retriever.retrieve(exclude_ids={'c'})
            return rows, retriever.render_selected(rows, max_chars=100)

        for attempt in range(3):
            with ThreadPoolExecutor(max_workers=8) as executor:
                result = list(executor.map(one, ['a', 'b'] * 8))
            assert [r[0] for r in result] == [[{'id': 'b'}], [{'id': 'a'}]] * 8
            assert all(row == (str(rows), [r['id'] for r in rows]) for rows, row in result)
        assert len(loaded) == 1
    assert runtime.generator.PersonaFewShotRetriever is original


@pytest.mark.parametrize('changed', ['pool', 'report'])
def test_changed_file_rejected_before_original_retrieval(tmp_path, monkeypatch, changed):
    pool, report, inputs, _, original = fixture(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match='文件发生变化'):
        with runtime.shared_history_index(inputs):
            retriever = runtime.generator.PersonaFewShotRetriever(path=pool)
            target = pool if changed == 'pool' else report
            target.write_text('changed')
            monkeypatch.setattr(original, 'retrieve', lambda **kw: pytest.fail('must reject before retrieval'))
            retriever.retrieve(exclude_ids=set())
    assert runtime.generator.PersonaFewShotRetriever is original


def test_wrong_frozen_content_or_render_pool_rejected(tmp_path, monkeypatch):
    pool, report, inputs, loaded, _ = fixture(tmp_path, monkeypatch)
    inputs[str(pool.resolve())] = 'wrong'
    with runtime.shared_history_index(inputs):
        with pytest.raises(ConfigError, match='替换展示库'):
            runtime.generator.PersonaFewShotRetriever(path=pool, render_path=report)
        with pytest.raises(ConfigError, match='冻结内容不符'):
            runtime.generator.PersonaFewShotRetriever(path=pool)
    assert loaded == []
