"""代码固定发言人名册渲染（context_render=speaker_roster_v1）回归。

身份事实（谁发的消息、谁是本人、引用的是谁）由代码从导出记录解析，
不再依赖模型从昵称或“我/你”推断；legacy 渲染字节保持不变。
"""
import hashlib
import json

import pytest

from src.config import sha256_file
from src.iteration import versions
from src.judge import corrected, corrected_v1 as runtime


@pytest.fixture
def case():
    return {"case_id": "case-roster-1", "chat_type": "group", "chat_name": "测试群",
            "source_chat_id": "chat-test",
            "context": [
                {"sender": "严辰", "text": "而我买了冰箱他就猜我喜欢冰箱", "is_self": False,
                 "timestamp": 1633065600},
                {"sender": "王芊", "text": "要让机器理解油烟机和冰箱的关系是件很难的事",
                 "is_self": True, "timestamp": 1633065660},
                {"sender": "严辰", "text": "[引用 王芊：因为目标不一样] 也不知道推送个油烟机",
                 "is_self": False, "timestamp": 1633065720},
                {"sender": "瓜瓜妈", "text": "长残排名榜我应该能进前三", "is_self": False,
                 "timestamp": 1633065780},
            ],
            "human_reply": ["听得了 你吐槽跟我有啥关系"]}


RENDER_CFG = {"context_render": corrected.CONTEXT_RENDER_SPEAKER_ROSTER}


def test_legacy_render_unchanged_by_default(case):
    blind, metadata = corrected.blind_case(case, ["甲"], ["乙"])
    assert len(blind["context_original"]) == 2  # 无名册片段
    assert '我（2021-10-01 13:21）' in blind["context_original"][0]
    assert '严辰（2021-10-01 13:20）' in blind["context_original"][0]
    explicit, metadata2 = corrected.blind_case(case, ["甲"], ["乙"],
                                               {"context_render": corrected.CONTEXT_RENDER_LEGACY})
    assert explicit == blind
    assert metadata2 == metadata


def test_roster_assigns_p1_to_self_regardless_of_order(case):
    blind, _ = corrected.blind_case(case, ["甲"], ["乙"], RENDER_CFG)
    roster = blind["context_original"][0]
    assert roster.startswith("<speaker_roster>")
    assert "P1=本人（王芊）" in roster
    assert "P2=严辰" in roster and "P3=瓜瓜妈" in roster
    # 本人不是首条消息，但编号仍是 P1；他人按出现顺序编号
    history = blind["context_original"][1]
    assert 'P2 严辰（2021-10-01 13:20）' in history
    assert 'P1 王芊（2021-10-01 13:21）' in history
    to_reply = blind["context_original"][2]
    assert 'P2 严辰（2021-10-01 13:22）' in to_reply
    assert 'P3 瓜瓜妈（2021-10-01 13:23）' in to_reply


def test_roster_marks_options_as_sent_by_self(case):
    blind, _ = corrected.blind_case(case, ["甲"], ["乙"], RENDER_CFG)
    assert "选项 A 与选项 B 都是由本人（P1）发送的候选回复" in blind["context_original"][0]


def test_quote_lead_resolved_only_for_roster_names(case):
    blind, _ = corrected.blind_case(case, ["甲"], ["乙"], RENDER_CFG)
    to_reply = blind["context_original"][2]
    # 引用本人名字 → P1；名册外名字保持原样
    assert "[引用 P1（王芊）：因为目标不一样]" in to_reply
    case2 = dict(case)
    case2["context"] = [dict(case["context"][0], text="[引用 陌生人：某句话] 对啊")]
    blind2, _ = corrected.blind_case(case2, ["甲"], ["乙"], RENDER_CFG)
    assert "[引用 陌生人：某句话]" in blind2["context_original"][1]
    case3 = dict(case)
    case3["context"] = [dict(case["context"][0], text="没有引用前缀")]
    blind3, _ = corrected.blind_case(case3, ["甲"], ["乙"], RENDER_CFG)
    assert "没有引用前缀" in blind3["context_original"][1]


def test_roster_fragments_parse_with_frozen_metadata(case):
    blind, metadata = corrected.blind_case(case, ["甲"], ["乙"], RENDER_CFG)
    # 名册片段必须是合法 XML 且不进入冻结时间线（无 message 元素）
    assert runtime.speaker_timeline(blind["context_original"]) == ("P2 严辰", "__self__", "P2 严辰", "P3 瓜瓜妈")
    assert runtime.message_text_timeline(blind["context_original"])[-1].startswith("P3 瓜瓜妈")
    assert metadata["latest_message_has_question_mark"] is False
    assert metadata["recent_speakers"] == ("P3 瓜瓜妈", "P2 严辰", "__self__", "P2 严辰")


def test_roster_render_is_deterministic(case):
    first, _ = corrected.blind_case(case, ["甲"], ["乙"], RENDER_CFG)
    second, _ = corrected.blind_case(json.loads(json.dumps(case)), ["甲"], ["乙"], RENDER_CFG)
    assert first == second


def _bundle(tmp_path, extra_cfg):
    schema = {"known_group_names": [], "known_group_members": []}
    option, names = runtime.vectorize_boolean_option(
        {**dict.fromkeys(runtime.INTEGER_FIELDS, 0),
         **dict.fromkeys(runtime.BOOLEAN_FIELDS, False),
         **{key: choices[0] for key, choices in runtime.ENUM_FIELDS.items()}})
    correction = {"selected_model_name": "symmetric_logistic_refined_boolean_crosses",
                  "feature_system": {"context_schema": schema},
                  "final_model": {"model": "LogisticRegression", "fit_intercept": False,
                                  "feature_names": list(names), "coefficients": [0.] * len(names),
                                  "input_feature_count": len(names)}}
    assets = {"prompt.md": b"<blind_case>{{case}}</blind_case>",
              "correction.json": json.dumps(correction).encode()}
    cfg = {"mode": corrected.MODE, "llm": {"provider": "codex_cli", "model": "test"},
           "correction_threshold": 0.7, "runtime_sha256": sha256_file(corrected.RUNTIME_PATH),
           "assets": {"correction.json": hashlib.sha256(assets["correction.json"]).hexdigest()},
           "prompt_sha256": hashlib.sha256(assets["prompt.md"]).hexdigest(), **extra_cfg}
    jid = versions.create_judge_version(cfg, {"note": "test"}, root=tmp_path, assets=assets)
    return cfg, tmp_path / jid


def _features():
    return {**dict.fromkeys(runtime.INTEGER_FIELDS, 0),
            **dict.fromkeys(runtime.BOOLEAN_FIELDS, False),
            **{key: choices[0] for key, choices in runtime.ENUM_FIELDS.items()}}


class _Client:
    def __init__(self):
        self.prompts = []

    def run(self, prompt, schema=None):
        self.prompts.append(prompt)
        if schema:
            return json.dumps({"option_A": _features(), "option_B": _features()})
        return json.dumps({"human_option": "A", "confidence": 0.9, "reason": "测试"})

    def cache_identity(self):
        return None


def test_judge_threads_config_to_blind_render(case, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "score_formal_judge_pair", lambda *args: 0.9)
    cfg, directory = _bundle(tmp_path, RENDER_CFG)
    client = _Client()
    judge = corrected.CorrectedJudge(cfg, directory, client)
    judge.is_ai(case, ["测试生成回复"])
    assert len(client.prompts) == 2  # 初判 + 特征各一次
    for prompt in client.prompts:
        assert "<speaker_roster>" in prompt
        assert "P1=本人（王芊）" in prompt
        assert "P2=严辰" in prompt


def test_judge_without_render_config_keeps_legacy_blind(case, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "score_formal_judge_pair", lambda *args: 0.9)
    cfg, directory = _bundle(tmp_path, {})
    client = _Client()
    judge = corrected.CorrectedJudge(cfg, directory, client)
    judge.is_ai(case, ["测试生成回复"])
    for prompt in client.prompts:
        assert "<speaker_roster>" not in prompt
