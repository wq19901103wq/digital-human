"""场景提示词生成（机制层）：通用模板 + 私有统计 → private/scenarios/*.md。

已存在的手工调过的场景文件不会被覆盖（除非 --force）——
场景文件是数据层资产，允许人工精调后冻结。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ROOT
from .analyze import format_stats

SCENARIOS = [
    "close_friend",       # 密友
    "friend",             # 朋友（默认）
    "acquaintance",       # 熟人
    "group_chat",         # 群聊
    "receiving_praise",   # 被夸
    "venting",            # 被吐槽
    "receiving_share",    # 被分享
    "answering_question", # 被提问
]

_SCENARIO_NAMES_ZH = {
    "close_friend": "密友私聊",
    "friend": "朋友私聊（默认）",
    "acquaintance": "熟人/泛泛之交",
    "group_chat": "群聊",
    "receiving_praise": "被夸时",
    "venting": "被吐槽/抱怨时",
    "receiving_share": "对方分享内容时",
    "answering_question": "被提问时",
}


def build_scenarios(
    scenarios_dir: Path,
    stats: dict[str, Any],
    llm=None,
    force: bool = False,
) -> list[Path]:
    template = (ROOT / "prompts" / "scenario_generation.template.md").read_text(encoding="utf-8")
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    global_stats = format_stats(stats)
    written = []
    for name in SCENARIOS:
        out = scenarios_dir / f"{name}.md"
        if out.exists() and not force:
            print(f"场景已存在，跳过（--force 可覆盖）: {out}")
            continue
        if llm is not None:
            prompt = (
                template.replace("{{scenario_name}}", _SCENARIO_NAMES_ZH[name])
                .replace("{{global_stats}}", global_stats)
                .replace("{{scenario_stats}}", "(骨架阶段暂无分场景统计，用全局统计)")
            )
            try:
                content = llm.chat([{"role": "user", "content": prompt}])
            except Exception as exc:  # noqa: BLE001
                print(f"场景 {name} LLM 生成失败，使用统计兜底: {exc}")
                content = _fallback(name, global_stats)
        else:
            content = _fallback(name, global_stats)
        out.write_text(content, encoding="utf-8")
        written.append(out)
        print(f"已生成场景 {out}")
    return written


def _fallback(name: str, global_stats: str) -> str:
    return (
        f"# {_SCENARIO_NAMES_ZH[name]}\n\n"
        "> 本文件由 bootstrap 从统计兜底生成；建议人工精调或提供 LLM 重新生成。\n\n"
        "## 全局统计\n" + global_stats + "\n"
    )
