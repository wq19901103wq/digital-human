"""few-shot 池构建（机制层）：统一消息 → 检索器可用的风格示例池（数据层）。

每行 = 一次真实的「上下文 → 本人回复」，带可信来源标记；
检索器只加载 _is_trusted_human_example 认可的行，防止把自动化后的
bot 输出当成真人风格学（wechat-mac-rpa 的教训）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import ConfigError


def _row_id(chat_id: str, index: int) -> str:
    return hashlib.sha256(f"{chat_id}:{index}".encode("utf-8")).hexdigest()[:16]


def extract_examples(
    messages: list[dict[str, Any]],
    max_context: int = 8,
    protected_indices: dict[str, set[int]] | None = None,
    protect_margin: int | None = None,
) -> list[dict[str, Any]]:
    """滑动窗口提取 QA 对：本人消息 = reply，前面最多 max_context 条 = context。

    protected_indices: 测试集(开发+固定)的 source 位置 {chat_id: {index}}。
    池内与任一受保护位置相距 ≤ margin（默认 max_context+2）的样本一律不入选，
    防止相邻样本的上下文窗口裹进测试题的真人答案（泄漏防护，SOP §9）。
    """
    margin = protect_margin if protect_margin is not None else max_context + 2
    examples = []
    by_chat: dict[str, list[dict[str, Any]]] = {}
    for m in messages:
        by_chat.setdefault(str(m["chat_id"]), []).append(m)

    for chat_id, msgs in by_chat.items():
        protected = protected_indices.get(chat_id, set()) if protected_indices else set()
        for i, m in enumerate(msgs):
            if not m.get("is_self") or not str(m.get("text", "")).strip():
                continue
            if protected and any(abs(i - p) <= margin for p in protected):
                continue
            context_msgs = msgs[max(0, i - max_context) : i]
            if not context_msgs:
                continue
            examples.append(
                {
                    "id": _row_id(chat_id, i),
                    "context": [
                        f"{c.get('sender', '?')}: {c.get('text', '')}" for c in context_msgs
                    ],
                    "context_messages": [
                        {"sender": str(c.get("sender", "")), "text": str(c.get("text", "")),
                         "is_self": bool(c["is_self"]), "timestamp": c["timestamp"]}
                        for c in context_msgs
                    ],
                    "reply": [str(m["text"])],
                    "relationship": str(m.get("chat_type", "private")),
                    "chat_name": str(m.get("chat_name", "")),
                    "source_chat_id": m.get("source_chat_id"),
                    "source_message_id": f"{chat_id}:{i}",
                    # 可信来源标记：导入的历史记录一律视为自动化启用前真人材料
                    "source_provenance": "before_automation_cutoff",
                }
            )
    return examples


def build_pool(
    messages: list[dict[str, Any]],
    out_path: Path,
    max_context: int = 8,
    refreeze: bool = False,
    refreeze_reason: str = "",
    protected_indices: dict[str, set[int]] | None = None,
) -> dict[str, Any]:
    """SOP §1：池一旦生成视为冻结资产。重建会使所有历史基线可比性作废，
    必须显式 refreeze 并记录原因，版本号自动 +1。"""
    old_version = 0
    if out_path.exists():
        if not refreeze:
            raise ConfigError(
                f"few-shot 池已存在（冻结资产）: {out_path}\n"
                "重建将作废全部历史基线的可比性。如确需重建，使用 --refreeze "
                "--refreeze-reason <原因>"
            )
        if not refreeze_reason.strip():
            raise ConfigError("--refreeze 必须提供 --refreeze-reason（写入版本记录）")
        old_report_path = out_path.parent / "report.json"
        if old_report_path.exists():
            try:
                old_version = int(json.loads(old_report_path.read_text(encoding="utf-8")).get("pool_version", 1))
            except (json.JSONDecodeError, ValueError):
                old_version = 1
    examples = extract_examples(
        messages, max_context=max_context, protected_indices=protected_indices
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    # 检索器 is_approved 要求 report.json 且 hash 匹配；
    # review_status 由导入方背书（历史数据视为自动化前真人材料），可人工改为 pending 停用
    report = {
        "total": len(examples),
        "review_status": "approved",
        "examples_sha256": sha,
        "max_context": max_context,
        "pool_version": old_version + 1,
        "refreeze_reason": refreeze_reason or "初始构建",
    }
    (out_path.parent / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已生成 few-shot 池 {out_path}（{len(examples)} 条, sha256={sha[:12]}…）")
    return report
