"""few-shot 池构建（机制层）：统一消息 → 检索器可用的风格示例池（数据层）。

每行 = 一次真实的「上下文 → 本人回复」，带可信来源标记；
检索器只加载 _is_trusted_human_example 认可的行，防止把自动化后的
bot 输出当成真人风格学（wechat-mac 项目的教训）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import ConfigError
from . import conversations


def _row_id(chat_id: str, index: int) -> str:
    return hashlib.sha256(f"{chat_id}:{index}".encode("utf-8")).hexdigest()[:16]


def extract_examples(
    messages: list[dict[str, Any]],
    max_context: int = 8,
    protected_indices: dict[str, set[int]] | None = None,
    protect_margin: int | None = None,
    context_gap_seconds: int = 7200,
    response_gap_seconds: int = 600,
    reply_gap_seconds: int = 120,
    statistics: dict | None = None,
) -> list[dict[str, Any]]:
    """保留近期上文后的完整短连发；跨会话主动发言不作为回复样本。"""
    rule = conversations.policy(context_gap_seconds=context_gap_seconds,
        response_gap_seconds=response_gap_seconds, reply_gap_seconds=reply_gap_seconds)
    if type(max_context) is not int or max_context <= 0:
        raise ConfigError('max_context 必须为正整数')
    stats = statistics if statistics is not None else {}
    def count(key, n=1):
        stats[key] = stats.get(key, 0) + n
    margin = protect_margin if protect_margin is not None else max_context + 2
    examples = []
    by_chat: dict[str, list[dict[str, Any]]] = {}
    for m in messages:
        by_chat.setdefault(str(m["chat_id"]), []).append(m)

    for chat_id, msgs in by_chat.items():
        times = [conversations.timestamp(m) for m in msgs]
        if any(a > b for a, b in zip(times, times[1:])):
            raise ConfigError('历史消息时间倒序')
        protected = protected_indices.get(chat_id, set()) if protected_indices else set()
        i = 0
        while i < len(msgs):
            m = msgs[i]
            if not m.get("is_self") or not str(m.get("text", "")).strip():
                i += 1
                continue
            count('self_runs')
            # Consume the whole self run, but retain only its first time-contiguous
            # burst. A later self message cannot become a reply to oneself.
            j = i + 1
            while j < len(msgs) and msgs[j].get("is_self") and str(msgs[j].get("text", "")).strip():
                j += 1
            end = i + 1
            while end < j and times[end] - times[end-1] <= rule['reply_gap_seconds']:
                end += 1
            burst = msgs[i:end]
            if end < j:
                count('split_self_runs')
                count('unattached_self_messages', j-end)
            if protected and any(abs(k - p) <= margin for p in protected for k in range(i, end)):
                count('protected')
                i = j
                continue
            start = conversations.context_start(msgs, i, max_context, rule)
            context_msgs = msgs[start:i]
            if start > max(0, i - max_context):
                count('trimmed_contexts')
            if not context_msgs:
                count('no_context')
                i = j
                continue
            problems = conversations.issues(context_msgs, burst, rule)
            if problems:
                for problem in problems:
                    count(problem)
                i = j
                continue
            count('retained')
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
                    "reply": [str(x["text"]) for x in burst],
                    "reply_shape": "multi" if len(burst) > 1 else "single",
                    "relationship": str(m.get("chat_type", "private")),
                    "chat_name": str(m.get("chat_name", "")),
                    "source_chat_id": m.get("source_chat_id"),
                    "source_message_id": f"{chat_id}:{i}",
                    # 可信来源标记：导入的历史记录一律视为自动化启用前真人材料
                    "source_provenance": "before_automation_cutoff",
                }
            )
            i = j
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
