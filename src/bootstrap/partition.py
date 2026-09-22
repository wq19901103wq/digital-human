"""数据切分（SOP §9 split-first 原则）：先按会话整块切三个分区，再各自使用。

先划分数据，再学习数据特征。三分区语义（SOP §9.1）：

| 分区 | 用途 | 禁止 |
|------|------|------|
| fixed（测试-固定） | 仅固定测试评测 | 一切学习与训练（人格/场景/池/Judge 训练与评估包） |
| dev（测试-开发）   | 开发 A/B 评测 + Judge 训练/评估包（开发材料，可看答案） | 人格统计、场景生成、few-shot 池 |
| train（训练侧）    | 人格、场景、few-shot 池 | —— |

切分策略（超参 settings.evaluation.split_strategy）：
- by_chat（默认）：整个聊天归且仅归一个分区，边界最干净。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..config import dataset_plan


@dataclass
class Partition:
    fixed_messages: list[dict[str, Any]]
    dev_messages: list[dict[str, Any]]
    train_messages: list[dict[str, Any]]
    fixed_chat_ids: set[str] = field(default_factory=set)
    dev_chat_ids: set[str] = field(default_factory=set)


def partition(messages: list[dict[str, Any]], settings: dict[str, Any], seed: int = 42) -> Partition:
    strategy = settings["evaluation"].get("split_strategy", "by_chat")
    if strategy != "by_chat":
        raise ValueError(f"未知切分策略: {strategy}（当前仅支持 by_chat，SOP §9）")

    rng = random.Random(seed)
    chats: dict[str, list[str]] = {"group": [], "private": []}
    seen: set[tuple[str, str]] = set()
    self_count: dict[str, int] = {}
    for m in messages:
        key = (str(m["chat_type"]), str(m["chat_id"]))
        if key not in seen:
            seen.add(key)
            chats[str(m["chat_type"])].append(str(m["chat_id"]))
        if m.get("is_self") and str(m.get("text", "")).strip():
            self_count[str(m["chat_id"])] = self_count.get(str(m["chat_id"]), 0) + 1

    def _pick(chat_type: str, quota: int, exclude: set[str]) -> set[str]:
        """按该类型候选聊天顺序累积整块聊天，直到本人消息数覆盖配额。"""
        candidates = [c for c in chats.get(chat_type, []) if c not in exclude]
        rng.shuffle(candidates)
        picked: set[str] = set()
        total = 0
        for cid in candidates:
            picked.add(cid)
            # 按 capped 计数：一个聊天最多贡献 4×每聊天题数上限，迫使配额分散到更多聊天
            # 每聊天按 ~1 题上限计数：750 题配额需 ≥30 个群聊聊天，防题海集中
            total += min(self_count.get(cid, 0), 18)
            if total >= quota:
                break
        return picked

    fixed_plan = dataset_plan(settings, "fixed_test")
    dev_plan = dataset_plan(settings, "development")
    fixed_chats = _pick("group", fixed_plan["group"], set()) | _pick("private", fixed_plan["private"], set())
    dev_chats = _pick("group", dev_plan["group"], fixed_chats) | _pick(
        "private", dev_plan["private"], fixed_chats
    )

    def _filter(ids: set[str]) -> list[dict[str, Any]]:
        return [m for m in messages if str(m["chat_id"]) in ids]

    test_ids = fixed_chats | dev_chats
    return Partition(
        fixed_messages=_filter(fixed_chats),
        dev_messages=_filter(dev_chats),
        train_messages=[m for m in messages if str(m["chat_id"]) not in test_ids],
        fixed_chat_ids=fixed_chats,
        dev_chat_ids=dev_chats,
    )
