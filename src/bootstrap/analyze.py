"""聊天风格统计（机制层）：输入统一消息，输出人格/场景生成的统计原料。

统计口径通用：长度分布、连发率、语气词密度、反问率，按群聊/私聊分层。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

FILLER_WORDS = ["哈", "哈哈哈", "吧", "啊", "呢", "嘛", "呀", "哦", "噢", "嗯"]


def analyze(messages: list[dict[str, Any]]) -> dict[str, Any]:
    self_msgs = [m for m in messages if m.get("is_self") and str(m.get("text", "")).strip()]
    lengths = [len(str(m["text"])) for m in self_msgs]
    return {
        "total_messages": len(messages),
        "total_self": len(self_msgs),
        "chats": len({m["chat_id"] for m in messages}),
        "reply_length": _length_stats(lengths),
        "burst": _burst_stats(messages),
        "fillers": _filler_stats(self_msgs),
        "question_rate": _question_rate(self_msgs),
        "by_chat_type": {
            ct: _length_stats(
                [len(str(m["text"])) for m in self_msgs if m["chat_type"] == ct]
            )
            for ct in ("group", "private")
        },
    }


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"avg": 0.0, "short_ratio": 0.0, "mid_ratio": 0.0, "long_ratio": 0.0}
    n = len(lengths)
    short = sum(1 for x in lengths if x <= 10)
    mid = sum(1 for x in lengths if 11 <= x <= 30)
    long_ = n - short - mid
    return {
        "avg": round(sum(lengths) / n, 1),
        "short_ratio": round(short / n, 3),   # ≤10 字
        "mid_ratio": round(mid / n, 3),       # 11-30 字
        "long_ratio": round(long_ / n, 3),    # >30 字
    }


def _burst_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """连发：连续 is_self 消息算一次连发；统计连发消息占比与平均连发长度。"""
    total_self = burst_msgs = 0
    run = 0
    runs: list[int] = []
    for m in messages:
        if m.get("is_self") and str(m.get("text", "")).strip():
            total_self += 1
            run += 1
        else:
            if run:
                runs.append(run)
                if run > 1:
                    burst_msgs += run
            run = 0
    if run:
        runs.append(run)
        if run > 1:
            burst_msgs += run
    return {
        "burst_msg_ratio": round(burst_msgs / total_self, 3) if total_self else 0.0,
        "avg_run_length": round(sum(runs) / len(runs), 2) if runs else 0.0,
    }


def _filler_stats(self_msgs: list[dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    total_chars = 0
    for m in self_msgs:
        text = str(m["text"])
        total_chars += len(text)
        for w in FILLER_WORDS:
            counter[w] += text.count(w)
    return {
        "per_100_chars": {
            w: round(counter[w] / total_chars * 100, 2) if total_chars else 0.0
            for w, _ in counter.most_common(8)
        }
    }


def _question_rate(self_msgs: list[dict[str, Any]]) -> float:
    if not self_msgs:
        return 0.0
    hits = sum(
        1
        for m in self_msgs
        if any(k in str(m["text"]) for k in ("?", "？", "吗", "为啥", "为什么", "怎么", "哪", "多少"))
    )
    return round(hits / len(self_msgs), 3)


def format_stats(stats: dict[str, Any]) -> str:
    """渲染成提示词友好的文本（bootstrap 填充模板用）。"""
    rl = stats["reply_length"]
    bu = stats["burst"]
    fi = stats["fillers"]["per_100_chars"]
    lines = [
        f"- 总消息 {stats['total_messages']}，本人消息 {stats['total_self']}，聊天 {stats['chats']} 个",
        f"- 平均回复长度 {rl['avg']} 字；短/中/长比例 {rl['short_ratio']}/{rl['mid_ratio']}/{rl['long_ratio']}",
        f"- 连发消息占比 {bu['burst_msg_ratio']}，平均连发 {bu['avg_run_length']} 条",
        f"- 反问率 {stats['question_rate']}",
        "- 语气词密度(每百字): " + " > ".join(f"{w}({c})" for w, c in fi.items()),
    ]
    for ct in ("group", "private"):
        s = stats["by_chat_type"][ct]
        lines.append(f"- {'群聊' if ct == 'group' else '私聊'}平均长度 {s['avg']} 字")
    return "\n".join(lines)
