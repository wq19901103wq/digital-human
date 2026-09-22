"""bootstrap 管线测试：示例数据 → 统计/池/测试集。纯本地，无需网络。"""
from __future__ import annotations

import json
from pathlib import Path

from src.bootstrap import analyze, build_fewshot_pool, build_testsets, ingest
from src.config import load_settings

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "chat_export.sample.jsonl"


def _messages():
    return ingest.load_unified(EXAMPLE)


def test_ingest_validates_schema(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"chat_id": "x"}\n', encoding="utf-8")
    try:
        ingest.load_unified(bad)
    except ValueError as exc:
        assert "缺少字段" in str(exc)
    else:
        raise AssertionError("应当拒绝缺字段的数据")


def test_analyze_basic():
    stats = analyze.analyze(_messages())
    assert stats["total_messages"] > 0
    assert stats["total_self"] > 0
    assert stats["reply_length"]["avg"] > 0
    assert "per_100_chars" in stats["fillers"]
    assert set(stats["by_chat_type"]) == {"group", "private"}
    text = analyze.format_stats(stats)
    assert "平均回复长度" in text


def test_fewshot_pool_rows_loadable(tmp_path):
    report = build_fewshot_pool.build_pool(_messages(), tmp_path / "pool.jsonl")
    assert report["total"] > 0
    row = json.loads((tmp_path / "pool.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["source_provenance"] == "before_automation_cutoff"
    assert isinstance(row["context"], list) and isinstance(row["reply"], list)
    # 检索器认可该池（report.json 存在 → is_approved）
    from src.generator.few_shot import PersonaFewShotRetriever

    retriever = PersonaFewShotRetriever(path=tmp_path / "pool.jsonl")
    assert retriever.is_approved()
    rows = retriever.retrieve(query="晚上聚餐", chat_name="x", is_group=True, limit=3)
    assert rows, "池非空时应当能召回示例"


def test_testsets_disjoint_and_composition(tmp_path):
    settings = load_settings()
    # 小样本数据不足以填满 1000 题：临时调低配置验证机制
    settings["evaluation"]["development"]["total"] = 4
    settings["evaluation"]["development"]["group_ratio"] = 0.5
    settings["evaluation"]["fixed_test"]["total"] = 4
    settings["evaluation"]["fixed_test"]["group_ratio"] = 0.5
    messages = _messages()
    from src.bootstrap.partition import partition

    part = partition(messages, settings, seed=42)
    manifest = build_testsets.build_testsets(
        part.fixed_messages, part.dev_messages, settings,
        tmp_path / "dev.jsonl", tmp_path / "fixed.jsonl"
    )
    dev_ids = {json.loads(l)["source_message_id"] for l in (tmp_path / "dev.jsonl").read_text(encoding="utf-8").splitlines()}
    fixed_ids = {json.loads(l)["source_message_id"] for l in (tmp_path / "fixed.jsonl").read_text(encoding="utf-8").splitlines()}
    assert dev_ids.isdisjoint(fixed_ids), "开发与固定集必须不相交"
    # split-first：各自分区抽题；fixed 只含 fixed 分区的聊天
    fixed_chats = {sid.rpartition(":")[0] for sid in fixed_ids}
    assert fixed_chats <= {str(c) for c in part.fixed_chat_ids}
    assert manifest["fixed_test"]["file_sha256"]
