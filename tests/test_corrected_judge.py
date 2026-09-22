"""迁移裁判的通道、资产完整性、双阶段判定与输入来源回归。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.config import ConfigError, sha256_file
from src.iteration import versions
from src.judge import corrected, corrected_v1 as runtime
from src.judge.migration import enrich_cases


def features(value=0):
    return {**dict.fromkeys(runtime.INTEGER_FIELDS, value),
            **dict.fromkeys(runtime.BOOLEAN_FIELDS, False),
            **{key: choices[0] for key, choices in runtime.ENUM_FIELDS.items()}}


@pytest.fixture
def case():
    return {"case_id": "case-1", "chat_type": "group", "chat_name": "测试群",
            "source_chat_id": "chat-test", "source_message_id": "group:test:2",
            "context": [{"sender": "本人昵称", "text": "明天见", "is_self": True, "timestamp": 1789171200},
                        {"sender": "小甲", "text": "几点？", "is_self": False, "timestamp": 1789171260}],
            "human_reply": ["八点"]}


@pytest.fixture
def bundle(tmp_path, case):
    _, metadata = corrected.blind_case(case, ["八点"], ["九点"])
    schema = {"known_group_names": [], "known_group_members": []}
    option, names = runtime.vectorize_boolean_option(features())
    observed, observed_names = runtime.vectorize_observable_option_booleans(["八点"], metadata)
    group, group_names = runtime.vectorize_group_pattern_option_boolean(["八点"], metadata)
    context, context_names = runtime.vectorize_context_booleans(metadata, schema)
    _, expanded = runtime.expand_boolean_option_with_refined_context(
        np.concatenate((option, observed, group)), names + observed_names + group_names, context, context_names)
    correction = {"selected_model_name": "symmetric_logistic_refined_boolean_crosses",
                  "feature_system": {"context_schema": schema},
                  "final_model": {"model": "LogisticRegression", "fit_intercept": False,
                                  "feature_names": expanded, "coefficients": [0.] * len(expanded),
                                  "input_feature_count": len(expanded)}}
    assets = {"prompt.md": b"<blind_case>{{case}}</blind_case>",
              "correction.json": json.dumps(correction).encode()}
    cfg = {"mode": corrected.MODE, "llm": {"provider": "codex_cli", "model": "test"},
           "correction_threshold": 0.7, "runtime_sha256": sha256_file(corrected.RUNTIME_PATH),
           "assets": {"correction.json": hashlib.sha256(assets["correction.json"]).hexdigest()},
           "prompt_sha256": hashlib.sha256(assets["prompt.md"]).hexdigest()}
    jid = versions.create_judge_version(cfg, {"note": "test"}, root=tmp_path, assets=assets)
    return cfg, tmp_path / jid


class Client:
    def __init__(self, option):
        self.option = option
        self.calls = []

    def run(self, prompt, schema=None):
        self.calls.append((prompt, schema))
        if schema:
            return json.dumps({"option_A": features(), "option_B": features()})
        return json.dumps({"human_option": self.option, "confidence": 0.9, "reason": "测试"})


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("base,p,final", [("A", 0.699, "A"), ("A", 0.3, "B"),
                                         ("B", 0.699, "B"), ("B", 0.7, "A"),
                                         ("A", 0.8, "A"), ("B", 0.2, "B")])
def test_two_stage_threshold_and_identity(bundle, case, monkeypatch, swap, base, p, final):
    cfg, directory = bundle
    client = Client(base)
    judge = corrected.CorrectedJudge(cfg, directory, client)
    monkeypatch.setattr(corrected.random, "random", lambda: 0.1 if swap else 0.9)
    monkeypatch.setattr(runtime, "score_formal_judge_pair", lambda *args: p)
    assert judge.is_ai(case, ["测试生成回复"]) == (final != ("A" if swap else "B"))
    assert len(client.calls) == 2
    for prompt, _ in client.calls:
        assert '"human_reply"' not in prompt and '"case_id"' not in prompt
    assert judge.last_verdict["correction_applied"] == (final != base)


def test_context_has_source_roles_and_timeline(case):
    blind, metadata = corrected.blind_case(case, ["甲"], ["乙"])
    assert runtime.speaker_timeline(blind["context_original"]) == ("__self__", "小甲")
    assert '我（2026-09-12 08:00）' in blind["context_original"][0]
    assert metadata["recent_speakers"] == ("小甲", "__self__")
    assert metadata["latest_message_has_question_mark"] is True
    del case["context"][0]["is_self"]
    with pytest.raises(ConfigError, match="旧数据"):
        corrected.blind_case(case, ["甲"], ["乙"])


def test_assets_survive_adopt_and_tampering_fails(bundle, tmp_path):
    cfg, directory = bundle
    new = versions.create_judge_version(cfg, {}, root=tmp_path, source_dir=directory)
    assert (tmp_path / new / "prompt.md").read_bytes() == (directory / "prompt.md").read_bytes()
    assert (tmp_path / new / "correction.json").read_bytes() == (directory / "correction.json").read_bytes()
    artifact = tmp_path / new / "correction.json"
    artifact.chmod(0o644)
    artifact.write_text("{}")
    with pytest.raises(ConfigError, match="资产不匹配"):
        corrected.CorrectedJudge(cfg, tmp_path / new, Client("A"))


def test_missing_snapshot_rejected_before_creating_version(bundle, tmp_path):
    cfg, _ = bundle
    before = set(tmp_path.iterdir())
    with pytest.raises(ConfigError, match="完整的来源资产"):
        versions.create_judge_version(cfg, {}, root=tmp_path)
    assert set(tmp_path.iterdir()) == before


def test_enrichment_preserves_cases_and_checks_source(tmp_path, case):
    messages = [{**m, "chat_id": "group:test", "source_chat_id": "chat-test"} for m in case["context"]]
    messages.append({"chat_id": "group:test", "source_chat_id": "chat-test", "is_self": True, "text": "八点"})
    old = {**case, "context": [{"sender": m["sender"], "text": m["text"]} for m in case["context"]]}
    for filename in ("dev_pool.jsonl", "fixed_test.jsonl"):
        (tmp_path / filename).write_text(json.dumps(old) + "\n")
    enriched = enrich_cases(tmp_path, messages)
    assert enriched["dev_pool.jsonl"] == [case]
    assert json.loads((tmp_path / "dev_pool.jsonl").read_text()) == old
    messages[0]["text"] = "不匹配"
    with pytest.raises(ConfigError, match="上下文不一致"):
        enrich_cases(tmp_path, messages)


def test_feature_format_retry_does_not_change_base(bundle, case):
    class RetryClient(Client):
        def run(self, prompt, schema=None):
            response = super().run(prompt, schema)
            return "bad JSON" if len(self.calls) == 2 else response
    cfg, directory = bundle
    client = RetryClient("A")
    corrected.CorrectedJudge(cfg, directory, client).is_ai(case, ["九点"])
    assert [bool(schema) for _, schema in client.calls] == [False, True, True]


def test_codex_transport_uses_frozen_parameters(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["codex", "--version"]:
            return SimpleNamespace(stdout="codex-cli 0.144.1", returncode=0)
        Path(command[command.index("--output-last-message") + 1]).write_text('{"human_option":"B"}')
        return SimpleNamespace(stdout='{"type":"turn.completed"}\n', stderr="", returncode=0)
    monkeypatch.setattr(corrected.subprocess, "run", run)
    cfg = {"provider": "codex_cli", "model": "gpt-5.6-luna", "reasoning_effort": "low", "timeout_seconds": 360, "codex_cli_version": "0.144.1"}
    client = corrected.CodexJudgeClient(cfg)
    assert json.loads(client.run("盲测"))["human_option"] == "B"
    command, options = calls[-1]
    assert command[command.index("--model") + 1] == cfg["model"]
    assert 'model_reasoning_effort="low"' in command and "--ignore-rules" in command
    assert options["timeout"] == 360


def test_feature_model_override_only_changes_extraction_transport(bundle, case, monkeypatch):
    cfg, directory = bundle
    cfg = {**cfg, 'feature_llm': {'model': 'gpt-5.6-terra', 'reasoning_effort': 'high'}}
    initial = Client('A')
    extractor = Client('B')
    captured = []
    def build(execution):
        captured.append(execution)
        return extractor
    monkeypatch.setattr(corrected, 'CodexJudgeClient', build)
    judge = corrected.CorrectedJudge(cfg, directory, client=initial)
    judge.is_ai(case, ['九点'])
    assert len(initial.calls) == 1 and initial.calls[0][1] is None
    assert len(extractor.calls) == 1 and extractor.calls[0][1]
    assert captured == [{**cfg['llm'], **cfg['feature_llm']}]
    assert judge.config['llm']['model'] == 'test'


def test_normalize_mode_legacy_name():
    from src.judge import LEGACY_CORRECTED_MODE, normalize_mode

    assert normalize_mode(LEGACY_CORRECTED_MODE) == 'corrected_pairwise'
    assert normalize_mode('corrected_pairwise') == 'corrected_pairwise'
    assert normalize_mode('pairwise_llm') == 'pairwise_llm'
    assert normalize_mode(None) is None


def test_build_judge_dispatches_legacy_mode(monkeypatch):
    """历史冻结 config 的旧 mode 名经 normalize_mode 后仍分发到 CorrectedJudge。"""
    from src.judge import corrected
    from src.judge import judge as judge_module
    from src.judge import LEGACY_CORRECTED_MODE

    built = {}

    class FakeCorrectedJudge:
        def __init__(self, config, directory):
            built['config'] = config
            built['dir'] = directory

    monkeypatch.setattr(corrected, 'CorrectedJudge', FakeCorrectedJudge)
    info = {'config': {'mode': LEGACY_CORRECTED_MODE}, 'dir': Path('j-dir')}
    scorer = judge_module.build_judge({}, info)
    assert isinstance(scorer, FakeCorrectedJudge)
    assert built['config'] is info['config'] and built['dir'] == Path('j-dir')


def test_dashboard_pipeline_treats_legacy_mode_as_corrected(bundle):
    """旧 mode 名的冻结版本在界面按 corrected_pairwise 展示，不回落到'机制未记录'。"""
    from src.dashboard import report as dashboard_report
    from src.judge import LEGACY_CORRECTED_MODE

    cfg, directory = bundle
    legacy = {**cfg, 'mode': LEGACY_CORRECTED_MODE}
    for render in (dashboard_report._judge_pipeline,):
        html = render(legacy, directory)
        assert '小模型预测' in html and '机制未记录' not in html
