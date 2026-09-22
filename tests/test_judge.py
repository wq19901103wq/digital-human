"""Judge pairwise 身份映射测试（SOP §10 防线）。

教训：曾把映射写成 (ai_is=="A")==(not swap)，导致 Judge 判对时记为未识别、
优化方向整个颠倒。此类 bug 单测必须覆盖 swap × 判定方向全组合。
"""
from __future__ import annotations

import json
import random as random_module

import pytest

from src.judge.judge import Judge

CANDIDATE_MARKER = "UNIQUE_CANDIDATE_MARKER_唯一候选标记"


class _StubLLM:
    """确定性假 Judge：按候选标记所在位置回答，pick=True 表示'判对'。"""

    def __init__(self, pick_candidate: bool):
        self._pick = pick_candidate

    def chat(self, messages, json_mode=False):
        prompt = messages[-1]["content"]
        a_block = prompt.split("<reply_A>")[1].split("</reply_A>")[0].strip()
        candidate_at_a = a_block == CANDIDATE_MARKER
        if self._pick:
            ai_is = "A" if candidate_at_a else "B"
        else:
            ai_is = "B" if candidate_at_a else "A"
        return json.dumps({"ai_is": ai_is, "confidence": 0.9})


def _run(swap: bool, pick_candidate: bool) -> bool:
    judge_cfg = {"baseline_id": "judge-test", "mode": "pairwise_llm", "prompt_path": None}
    judge = Judge(judge_cfg, llm=_StubLLM(pick_candidate))
    orig = random_module.Random.random
    random_module.Random.random = lambda self: 0.1 if swap else 0.9  # <0.5 → swap
    try:
        case = {
            "context": [{"sender": "甲", "text": "在吗"}],
            "human_reply": ["在"],
        }
        return judge.is_ai(case, [CANDIDATE_MARKER])
    finally:
        random_module.Random.random = orig


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("pick_candidate", [False, True])
def test_pairwise_identity_mapping(swap: bool, pick_candidate: bool):
    """Judge 判中候选(=pick_candidate)时，is_ai 必须返回 True，与 swap 无关。"""
    assert _run(swap, pick_candidate) is pick_candidate


@pytest.mark.parametrize("swap", [False, True])
def test_pairwise_both_human_is_abstention(swap: bool):
    """both_human（都是真人）= 弃权：对 AI 样本计未识别，对真人样本计未误判（SOP §7.6）。"""

    class _BothHumanLLM:
        def chat(self, messages, json_mode=False):
            return json.dumps({"ai_is": "both_human", "confidence": 0.8})

    judge_cfg = {"baseline_id": "t", "mode": "pairwise_llm", "prompt_path": None}
    judge = Judge(judge_cfg, llm=_BothHumanLLM())
    orig = random_module.Random.random
    random_module.Random.random = lambda self: 0.1 if swap else 0.9
    try:
        case = {
            "context": [{"sender": "甲", "text": "在吗"}],
            "human_reply": ["在"],
        }
        assert judge.is_ai(case, ["UNIQUE_CANDIDATE_MARKER"]) is False
    finally:
        random_module.Random.random = orig
