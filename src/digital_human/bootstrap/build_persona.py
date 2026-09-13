"""人格生成（机制层）：模板 + 私有统计/档案 → private/persona.md（数据层）。

档案 profile.json 由用户填写（私有），bootstrap 创建骨架并提示补全。
语气归纳可用 LLM（可选，--no-llm 时用统计摘要兜底）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import ROOT
from .analyze import format_stats

PROFILE_FIELDS = ["name", "occupation", "city", "relationships", "interests"]


def profile_skeleton() -> dict[str, Any]:
    return {k: "" for k in PROFILE_FIELDS}


def ensure_profile(path: Path) -> dict[str, Any]:
    """档案不存在则创建骨架并提示用户填写（档案是数据层，gitignore）。"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profile_skeleton(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已创建档案骨架 {path}，请填写后重新运行 bootstrap")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_tone(self_samples: list[str], llm=None) -> str:
    """从本人真实回复样例归纳语气（可选 LLM；兜底返回统计描述）。"""
    if not llm or not self_samples:
        return ""
    samples = "\n".join(f"- {s}" for s in self_samples[:30])
    prompt = (
        "下面是同一个人真实的聊天回复样例。用 3-5 句话归纳这个人聊天的"
        "自然语气、幽默边界和个人表达习惯，供模仿其风格的生成器使用。"
        "不要列举具体事实或人名。\n" + samples
    )
    try:
        return llm.chat([{"role": "user", "content": prompt}]).strip()
    except Exception as exc:  # noqa: BLE001
        print(f"语气归纳 LLM 失败，使用统计兜底: {exc}")
        return ""


def render_persona(
    profile: dict[str, Any],
    stats_text: str,
    tone_summary: str = "",
) -> str:
    template = (ROOT / "prompts" / "persona.template.md").read_text(encoding="utf-8")
    mapping = {
        "{{真实姓名}}": profile.get("name", ""),
        "{{职业}}": profile.get("occupation", ""),
        "{{城市}}": profile.get("city", ""),
        "{{主要人际关系及偏好称呼}}": profile.get("relationships", ""),
        "{{长期兴趣或专业领域}}": profile.get("interests", ""),
        "{{由数据和样例归纳：自然语气、幽默边界、个人表达习惯}}": tone_summary or "（见场景规则）",
        "{{由数据统计得出：高频语气词及密度，如\"哈 > 吧 > 啊\"，偶尔用不堆砌}}": _fillers_from_stats(stats_text),
        "{{由数据归纳：口头禅清单，偶尔用，不要堆砌}}": "（由场景规则按场景给出）",
    }
    out = template
    for placeholder, value in mapping.items():
        out = out.replace(placeholder, str(value))
    # 长度规则用统计直出
    out = re.sub(
        r"\{\{由数据统计得出[^}]*\}\}",
        _length_rule_from_stats(stats_text),
        out,
        count=1,
    )
    return out


def _length_rule_from_stats(stats_text: str) -> str:
    import re as _re

    m = _re.search(r"平均回复长度 ([\d.]+) 字；短/中/长比例 ([\d.]+)/([\d.]+)/([\d.]+)", stats_text)
    if not m:
        return "日常默认 1 条短句"
    avg, short, mid, _long = m.groups()
    burst = _re.search(r"平均连发 ([\d.]+) 条", stats_text)
    run = burst.group(1) if burst else "1"
    return (
        f"日常默认 1 条短句，平均 {avg} 字；"
        f"短句(≤10字)占比 {short}，确需分层时连发不超过 {run} 条、总 2-3 条"
    )


def _fillers_from_stats(stats_text: str) -> str:
    import re as _re

    m = _re.search(r"语气词密度\(每百字\): (.+)$", stats_text, _re.MULTILINE)
    return m.group(1) if m else "（统计不明显则不用）"


def build_persona(profile_path: Path, out_path: Path, stats: dict, llm=None) -> Path:
    profile = ensure_profile(profile_path)
    missing = [k for k in PROFILE_FIELDS if not profile.get(k)]
    if missing:
        print(f"警告: 档案字段未填写: {missing}（生成结果将留空）")
    stats_text = format_stats(stats)
    tone = summarize_tone(
        [s for s in stats.get("_self_samples", [])],
        llm=llm,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_persona(profile, stats_text, tone), encoding="utf-8")
    print(f"已生成人格文件 {out_path}")
    return out_path
