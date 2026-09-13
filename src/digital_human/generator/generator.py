"""C0 生成器（机制层）：人格 + 风格召回 + LLM + JSON 防护。

生成器版本（versions.py）提供配置与自包含提示词快照；本类只负责按配置生成。
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from ..config import ConfigError
from ..llm import ChatClient
from .. import tracing
from .few_shot import PersonaFewShotRetriever
from .persona import PersonaPromptBuilder

_logger = logging.getLogger("digital_human.generator")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ReplyGenerator:
    def __init__(
        self,
        settings: dict[str, Any],
        gen_cfg: dict[str, Any],
        llm: ChatClient | dict[str, ChatClient],
        prompt_root: Path | None = None,
        pool_path: Path | None = None,
    ):
        self._settings = settings
        self._cfg = gen_cfg
        # 单客户端或 {"private":…, "group":…}；按 case 聊天类型选择
        self._llm = llm if isinstance(llm, dict) else {"private": llm, "group": llm}
        self._builder = PersonaPromptBuilder(settings, prompt_root=prompt_root)

        ev = settings["evaluation"]
        self._max_shots = int(gen_cfg.get("max_shots_per_case", ev["few_shots_per_case"]))
        self._budget = int(gen_cfg.get("shots_char_budget", ev["few_shots_char_budget"]))

        # 检索策略已冻结（SOP §1.6）：版本配置只保留启用开关
        switches = gen_cfg.get("retriever", {})
        self._retriever: PersonaFewShotRetriever | None = None
        if switches.get("enabled", True):
            if pool_path is None or not pool_path.exists():
                raise ConfigError(
                    f"配置启用 few-shot 但池文件缺失: {pool_path}（F02：缺失依赖不得静默降级，"
                    f"明确 retriever.enabled=false 才可无池运行）"
                )
            retriever = PersonaFewShotRetriever(path=pool_path)
            if not retriever.is_approved():
                raise ConfigError(
                    f"few-shot 池未通过审批（状态或内容 hash 不符）: {pool_path}"
                )
            self._retriever = retriever

    @property
    def baseline_id(self) -> str:
        return self._cfg["baseline_id"]

    def _style_block(self, case: dict[str, Any]) -> str:
        if self._retriever is None:
            tracing.note("retrieval", {"enabled": False, "rendered": "(无)"})
            return "(无)"
        is_group = case.get("chat_type") == "group"
        messages = case.get("context", [])
        query_text = "\n".join(str(m.get("text", "")) for m in messages[-3:])
        try:
            rows = self._retriever.retrieve(
                query=query_text,
                chat_name=str(case.get("chat_name", "")),
                is_group=is_group,
                limit=self._max_shots * 4,
                exclude_ids={str(case.get("case_id", ""))},
                current_context_messages=[
                    {"sender": str(m.get("sender", "")), "text": str(m.get("text", ""))}
                    for m in messages
                ],
            )[: self._max_shots]
            block, ids = self._retriever.render_selected(rows, max_chars=self._budget)
            tracing.note("retrieval", {"enabled": True, "query": query_text, "selected_ids": ids,
                                      "max_shots": self._max_shots, "char_budget": self._budget, "rendered": block})
            _logger.debug("风格召回 %d 条: %s", len(ids), ids)
            return block
        except Exception as exc:  # noqa: BLE001 - 召回失败降级，不阻断生成
            tracing.note("retrieval", {"enabled": True, "query": query_text, "error": str(exc), "rendered": "(无)"})
            _logger.warning("风格召回失败，降级为无示例: %s", exc)
            return "(无)"

    def build_prompt(self, case: dict[str, Any], forced_reply: bool = True) -> list[dict[str, str]]:
        """组装完整模型输入（触发预检和生成共用，保证比对的输入与真实一致）。"""
        return self._builder.build_messages(case, self._style_block(case), forced_reply)

    def generate(self, case: dict[str, Any], forced_reply: bool = True) -> dict[str, Any]:
        """返回 {"replies": [...], "latency_ms": float}。无效结果重试一次后抛异常（计入失败）。

        有效生成结果（SOP §3.6）：replies 为字符串数组、每项非空、≤3 条；
        强制回复模式下空数组无效。
        """
        messages = self.build_prompt(case, forced_reply)
        client = self._llm["group" if case.get("chat_type") == "group" else "private"]
        started = time.perf_counter()
        raw = client.chat(messages, json_mode=True)
        parsed = self._validate(self._parse(raw), forced_reply)
        tracing.note("validation", {"attempt": 1, "valid": parsed is not None, "parsed": parsed, "raw": raw})
        if parsed is None:
            _logger.warning("生成结果无效，重试一次: %s…", raw[:80])
            raw = client.chat(messages, json_mode=True)
            parsed = self._validate(self._parse(raw), forced_reply)
            tracing.note("validation", {"attempt": 2, "valid": parsed is not None, "parsed": parsed, "raw": raw})
        if parsed is None:
            raise RuntimeError("生成结果两次都无效（JSON 结构/非空/条数校验失败）")
        return {
            "replies": parsed["replies"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    @staticmethod
    def _validate(parsed: dict[str, Any] | None, forced_reply: bool) -> dict[str, Any] | None:
        if not isinstance(parsed, dict):
            return None
        replies = parsed.get("replies")
        if not isinstance(replies, list) or len(replies) > 3:
            return None
        if any(not isinstance(r, str) or not r.strip() for r in replies):
            return None
        if forced_reply and not replies:
            return None
        return {"replies": [r.strip() for r in replies]}

    @staticmethod
    def _parse(raw: str) -> dict[str, Any] | None:
        match = _JSON_RE.search(raw)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
