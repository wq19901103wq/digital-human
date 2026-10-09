"""data_guard 冻结指纹：检索实现完整清单绑定与续跑失配拒绝。"""
import copy
import json
from pathlib import Path

import pytest

from src.config import ConfigError, sha256_file
from src.iteration import data_guard


def _data_dir(tmp_path):
    data = tmp_path / 'data'
    data.mkdir(parents=True)
    case = {'case_id': 'c-1', 'context': [], 'human_reply': ['回复']}
    (data / 'dev.jsonl').write_text(json.dumps(case, ensure_ascii=False) + '\n', encoding='utf-8')
    (data / 'purposes.json').write_text(json.dumps({
        'history_policy': 'complete_before_input_v1',
        'roles': {'judge_development': {'file': 'dev.jsonl', 'sha256': sha256_file(data / 'dev.jsonl')}},
        'protocol': {'development_start': 1, 'acceptance_start': 2, 'unseen_chat_ids': []},
    }), encoding='utf-8')
    return data, [case]


def _snapshot(tmp_path):
    data, cases = _data_dir(tmp_path)
    return data_guard.generation_snapshot(cases, data, [{'retriever': {'enabled': False}}])


def test_generation_snapshot_binds_complete_fewshot_implementation(tmp_path):
    snapshot = _snapshot(tmp_path)
    expected = {'generator/generator.py', 'generator/history.py', 'iteration/datasets.py',
                'generator/fewshot/session.py', 'generator/fewshot/prepare.py',
                'generator/fewshot/rank.py', 'generator/fewshot/recall_engines.py'}
    assert expected <= set(snapshot['implementation_sha256'])
    root = Path(__file__).resolve().parents[2]
    for name in expected:
        assert snapshot['implementation_sha256'][name] == sha256_file(root / 'src' / name)


def test_changed_rank_implementation_rejects_resume(tmp_path):
    """排序实现（fewshot/rank.py）字节变化 → 实现清单失配 → 禁止复用断点。"""
    current = _snapshot(tmp_path)
    saved = copy.deepcopy(current)
    saved['implementation_sha256']['generator/fewshot/rank.py'] = '0' * 64
    directory = tmp_path / 'experiment'
    directory.mkdir()
    with pytest.raises(ConfigError, match='冻结数据指纹不一致'):
        data_guard.verify(directory, current, expected=saved)


def test_verify_writes_snapshot_once_and_accepts_unchanged_resume(tmp_path):
    current = _snapshot(tmp_path)
    directory = tmp_path / 'experiment'
    directory.mkdir()
    data_guard.verify(directory, current)
    assert json.loads((directory / 'data_snapshot.json').read_text(encoding='utf-8')) == current
    # 已存指纹与当前一致：续跑放行（无 checkpoint 时缺证据才拒绝）。
    data_guard.verify(directory, copy.deepcopy(current))
