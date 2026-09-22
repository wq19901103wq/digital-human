"""Source-backed speaker roles survive generation without changing old prompts."""
from copy import deepcopy

import pytest

from src import tracing
from src.config import ConfigError, load_settings
from src.generator.generator import ReplyGenerator
from src.generator.history_sources import prompt_case
from src.generator.persona import CONTEXT_IDENTITY_POLICY, PersonaPromptBuilder
from test_leakage_guards import env as env


@pytest.fixture
def prompts(tmp_path):
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "persona.md").write_text("只输出回复 JSON。", encoding="utf-8")
    return directory


def test_legacy_prompt_bytes_remain_unchanged(prompts):
    case = {"context": [{"sender": "小王", "text": "周末见", "is_self": True},
                        {"sender": "小李", "text": "好", "is_self": False}]}
    expected = [
        {"role": "system", "content": "只输出回复 JSON。"},
        {"role": "user", "content":
         "<history>\n小王: 周末见\n小李: 好\n</history>\n"
         "<unread>\n好\n</unread>\n"
         "<style_examples>\n示例内容\n</style_examples>\n"
         "\n<force_reply>评测模式：本题必须给出回复，不得输出空 replies。</force_reply>\n"
         "按 system 中的 output_schema 只输出一个合法 JSON 对象。"},
    ]
    assert PersonaPromptBuilder({}, prompts).build_messages(case, "示例内容", True) == expected
    assert PersonaPromptBuilder({}, prompts, context_identity=None).build_messages(
        case, "示例内容", True) == expected


def test_explicit_identity_uses_typed_flags_and_retains_names(prompts):
    case = {"context": [
        {"sender": "小王", "text": "我有点担心", "is_self": True},
        {"sender": "小王", "text": "没事的", "is_self": False},
        {"sender": "我", "text": "不能按名字认人"},
        {"sender": "未知甲", "text": "字符串不是布尔值", "is_self": "true"},
        {"sender": "未知乙", "text": "数字不是布尔值", "is_self": 1},
    ], "human_reply": ["评测答案，不得出现在输入里"]}
    original = deepcopy(case)
    builder = PersonaPromptBuilder({}, prompts, context_identity=CONTEXT_IDENTITY_POLICY)
    messages = builder.build_messages(prompt_case(case), "原样保留的示例", True)
    content = messages[1]["content"]
    assert "[本人] 小王: 我有点担心" in content
    assert "[其他人] 小王: 没事的" in content
    for sender in ("我", "未知甲", "未知乙"):
        assert f"[身份未标明] {sender}:" in content
    assert content.count("[本人] 小王:") == 1
    assert "原样保留的示例" in content
    assert "评测答案，不得出现在输入里" not in content
    assert messages == builder.build_messages(case, "原样保留的示例", True)
    assert case == original


@pytest.mark.parametrize("policy", [True, False, "unknown", "explicit-self-v2"])
def test_unknown_identity_policy_fails_closed(prompts, policy):
    with pytest.raises(ConfigError, match="context_identity"):
        PersonaPromptBuilder({}, prompts, context_identity=policy)


def test_historical_prompt_uses_roles_after_source_validation(env):
    case = env.source.roles["development"][0]
    original = deepcopy(case)
    config = {**env.cfg, "context_identity": CONTEXT_IDENTITY_POLICY}
    generator = ReplyGenerator(load_settings(), config, env.client, prompt_root=env.gdir,
                               pool_path=env.source.directory / "fewshot_pool.jsonl")
    messages = generator.build_prompt(case)
    history = messages[1]["content"].split("<history>\n", 1)[1].split("\n</history>", 1)[0]
    for message in case["context"]:
        role = "本人" if message["is_self"] else "其他人"
        assert f"[{role}] {message['sender']}: {message['text']}" in history
    assert messages == generator._builder.build_messages(
        prompt_case(case), generator._style_block(case), True)
    tampered = deepcopy(case)
    tampered["context"][0]["is_self"] = not tampered["context"][0]["is_self"]
    with pytest.raises(ConfigError, match="身份|来源"):
        generator.build_prompt(tampered)
    assert case == original and env.calls == []


def test_identity_policy_changes_actual_request_and_cache_key(prompts, tmp_path):
    class Client:
        def __init__(self):
            self.calls = []

        def cache_identity(self):
            return {"model": "identity-test"}

        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            return '{"replies":["好"]}'

    settings = {"evaluation": {"few_shots_per_case": 3, "few_shots_char_budget": 2500}}
    client = Client()
    case = {"case_id": "speaker-role", "context": [
        {"sender": "小王", "text": "担心", "is_self": True},
        {"sender": "小李", "text": "没事", "is_self": False}]}
    for name, identity in (("baseline", None), ("candidate", CONTEXT_IDENTITY_POLICY),
                           ("candidate-resume", CONTEXT_IDENTITY_POLICY)):
        generator = ReplyGenerator(settings, {"retriever": {"enabled": False},
            "context_identity": identity}, client, prompt_root=prompts)
        trace = tracing.CaseTrace(tmp_path / "experiments" / name, case, {"dataset": "development"})
        with trace.operation("generation", "candidate", 0, {"forced_reply": True}) as operation:
            operation["result"] = generator.generate(case)
        trace.finish("ok")
    assert len(client.calls) == 2
    assert "[本人] 小王:" not in client.calls[0][1]["content"]
    assert "[本人] 小王:" in client.calls[1][1]["content"]
