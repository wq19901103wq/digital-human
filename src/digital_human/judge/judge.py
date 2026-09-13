"""判别器（机制层）：配对盲测（SOP §4）。

Judge 版本 = 判别配置（模型/提示词）。更换模型或提示词时，用开发样本校准
（同一份冻结 pack 上比较新旧版本识别率），作为新评测版本——生成器升级
不再触发任何重训。

build_judge 按版本选择默认 LLM 盲测或 RPA 冻结校正盲测；后者实现见 rpa.py。

识别判定：盲测 A/B（顺序随机、可复现），输出 ai_is ∈ {A, B, both_human}。
both_human = 弃权：AI 样本计未识别，真人样本计未误判（配对判别下唯一的误判口径）。
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from ..config import ConfigError
from ..llm import ChatClient
from .. import tracing

_logger = logging.getLogger("digital_human.judge")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_judge(settings: dict, info: dict):
    """按冻结版本选通道；Codex 裁判不经过生成器的 API 默认配置。"""
    from .rpa import MODE, RpaJudge
    if info["config"].get("mode") == MODE:
        return RpaJudge(info["config"], info["dir"])
    path = info["dir"] / "prompt.md"
    return Judge(info["config"], ChatClient(settings, info["config"]["llm"]),
                 prompt_path=path if path.exists() else None)


class Judge:
    def __init__(self, judge_cfg: dict[str, Any], llm: ChatClient, prompt_path: Path | None = None):
        self._cfg = judge_cfg
        mode = judge_cfg.get("mode", "pairwise_llm")
        if mode != "pairwise_llm":
            raise ConfigError(f"仅支持配对盲测 Judge（SOP §4）: {mode}")
        self._llm = llm
        # 提示词：优先 Judge 版本目录内的快照（随版本冻结）；无快照的旧版本回退全局模板
        template_path = prompt_path or (Path(__file__).resolve().parents[3] / "prompts" / "judge_pairwise.template.md")
        self._template = Path(template_path).read_text(encoding="utf-8")

    @property
    def mode(self) -> str:
        return "pairwise_llm"

    def is_ai(self, case: dict[str, Any], candidate_replies: list[str]) -> bool:
        """True = 判别器指出候选是 AI。真人样本上 True 即误判。"""
        context = "\n".join(
            f"{m.get('sender', '?')}: {m.get('text', '')}" for m in case.get("context", [])
        )
        human_text = "\n".join(str(p) for p in case.get("human_reply", []))
        candidate_text = "\n".join(str(p) for p in candidate_replies)

        # A/B 随机换位并记录 seed，保证可复现
        seed = random.randrange(2**31)
        rng = random.Random(seed)
        swap = rng.random() < 0.5
        reply_a, reply_b = (candidate_text, human_text) if swap else (human_text, candidate_text)
        tracing.note("blind_mapping", {"seed": seed, "candidate_option": "A" if swap else "B",
                                      "human_option": "B" if swap else "A"})

        prompt = (
            self._template.replace("{{context}}", context)
            .replace("{{reply_a}}", reply_a)
            .replace("{{reply_b}}", reply_b)
        )
        raw = self._llm.chat([{"role": "user", "content": prompt}], json_mode=True)
        match = _JSON_RE.search(raw)
        if not match:
            raise RuntimeError(f"Judge 输出无法解析: {raw[:120]}…")
        verdict = json.loads(match.group(0))
        tracing.note("base_verdict", verdict)
        ai_is = str(verdict.get("ai_is", "")).strip()
        # both_human = 认为两边都是真人：对 AI 样本计未识别，对真人样本计未误判。
        if ai_is.lower() in ("both_human", "都是真人"):
            _logger.debug("judge seed=%s swap=%s both_human → not accused", seed, swap)
            return False
        ai_is = ai_is.upper()
        if ai_is not in {"A", "B"}:
            raise RuntimeError(f"Judge 输出非法 ai_is: {ai_is!r}")
        # 映射回真实身份：swap=False 时 candidate 在 B 位，swap=True 时在 A 位。
        identified_ai = (ai_is == "B") == (not swap)
        _logger.debug("judge seed=%s swap=%s ai_is=%s identified=%s", seed, swap, ai_is, identified_ai)
        return identified_ai
