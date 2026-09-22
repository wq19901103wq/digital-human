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
from .. import cache, tracing
from .few_shot import PersonaFewShotRetriever
from .persona import PersonaPromptBuilder
from .history import HistoryError

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
        self.source_check = None
        self._history_sources = None
        self._material_seal = None
        self._information_end = None
        if pool_path is not None and pool_path.with_name('purposes.json').exists():
            from .history_sources import load
            self._history_sources = load(pool_path.parent)
            from ..iteration.learning_guard import require_materials
            if prompt_root is None:
                raise HistoryError('历史生成必须使用有来源证明的提示词快照')
            from ..iteration.learning_guard import MaterialSeal
            self._material_seal = MaterialSeal(prompt_root)
            require_materials(pool_path.parent.name, [prompt_root])
            self._information_end = json.loads((prompt_root / "learning.json").read_text())["information_end"]

        self._builder = PersonaPromptBuilder(settings, prompt_root=prompt_root,
                                            context_identity=gen_cfg.get("context_identity"))
        self._check_sources()

        ev = settings["evaluation"]
        self._max_shots = int(gen_cfg.get("max_shots_per_case", ev["few_shots_per_case"]))
        self._budget = int(gen_cfg.get("shots_char_budget", ev["few_shots_char_budget"]))

        # Selection policies and model reranking require explicit version config.
        switches = gen_cfg.get("retriever", {})
        if 'source_overlap_policy' in switches:
            from .learned_selection import POLICY, validate_source_overlap_policy
            overlap_policy = switches['source_overlap_policy']
            validate_source_overlap_policy(overlap_policy)
            if overlap_policy is not None and switches.get('learned') != POLICY:
                raise ConfigError('source_overlap_policy requires learned few-shot selection')
        self._learned = None
        if switches.get('learned'):
            from .learned_selection import LearnedSelector, POLICY
            from ..iteration import versions
            if (switches['learned'] != POLICY or switches.get('selection') or switches.get('reranker')
                    or not switches.get('enabled', True) or self._history_sources is None):
                raise ConfigError('learned few-shot selection requires exclusive policy and historical sources')
            self._learned = LearnedSelector(prompt_root / 'ranker',
                versions.PRIVATE / '.cache/fewshot_ranker_features')
        self._selection = switches.get('selection')
        if self._selection:
            from .selection import POLICY
            if self._selection != POLICY or switches.get('reranker') or not switches.get('enabled', True):
                raise ConfigError('few-shot selection 需要有效策略、启用召回，且不能同时使用模型重排')
        self._reranker = None
        if switches.get('reranker'):
            if not switches.get('enabled', True):
                raise ConfigError('few-shot 重排需要启用历史召回')
            from .reranker import FewShotReranker
            self._reranker = FewShotReranker(ChatClient(settings, switches['reranker']))
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
        self._check_sources()

    def _check_sources(self):
        if self.source_check:
            self.source_check()
        if self._material_seal is not None:
            self._material_seal.check()
        if self._history_sources is not None:
            self._history_sources.check()

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
            retrieval = dict(
                chat_name=str(case.get("chat_name", "")),
                is_group=is_group,
                limit=self._max_shots * 4,
                exclude_ids={str(case.get("case_id", ""))},
                current_context_messages=[
                    {"sender": str(m.get("sender", "")), "text": str(m.get("text", ""))}
                    for m in messages
                ],
                **({'history_case': case} if getattr(self._retriever, 'history_policy', None) else {}),
            )
            rows = (self._retriever.retrieve(query=query_text, **retrieval)
                    if self._learned is None else [])
            selection_trace = {}
            if self._learned is not None:
                from .learned_selection import recall
                recalled = recall(self._retriever, case,
                    source_overlap_policy=self._cfg.get('retriever', {}).get('source_overlap_policy'))
                rows = self._learned.select(case, recalled,
                    count=self._max_shots, budget=self._budget, retriever=self._retriever,
                    check=self._check_sources)
            elif self._selection:
                from .selection import select
                latest = str(messages[-1].get('text', '')) if messages else ''
                latest_rows = self._retriever.retrieve(query=latest, **retrieval) if latest != query_text else rows
                rows, selection_trace = select(rows, latest_rows, count=self._max_shots,
                    budget=self._budget, retriever=self._retriever)
            elif self._reranker is not None:
                rows = self._reranker.select(case, rows, count=self._max_shots, budget=self._budget,
                                            retriever=self._retriever, check=self._check_sources)
            else:
                rows = rows[:self._max_shots]
            block, ids = self._retriever.render_selected(rows, max_chars=self._budget)
            tracing.note("retrieval", {"enabled": True, "query": query_text, "selected_ids": ids,
                                      "max_shots": self._max_shots, "char_budget": self._budget,
                                      "rendered": block, **({'selection': selection_trace} if self._selection else {})})
            _logger.debug("风格召回 %d 条: %s", len(ids), ids)
            return block
        except Exception as exc:  # noqa: BLE001 - legacy retrieval retains its original fallback
            if self._learned is not None or self._reranker is not None or self._selection:
                raise
            if isinstance(exc, HistoryError) or getattr(self._retriever, 'history_policy', None):
                raise HistoryError(f'历史召回校验失败，禁止绕过时间限制: {exc}') from exc
            tracing.note("retrieval", {"enabled": True, "query": query_text, "error": str(exc), "rendered": "(无)"})
            _logger.warning("风格召回失败，降级为无示例: %s", exc)
            return "(无)"

    def build_prompt(self, case: dict[str, Any], forced_reply: bool = True) -> list[dict[str, str]]:
        """组装完整模型输入（触发预检和生成共用，保证比对的输入与真实一致）。"""
        if self._history_sources is not None:
            from .history_sources import prompt_case
            self._material_seal.check()
            self._history_sources.validate(case)
            if self._information_end >= case["input_cutoff"]["timestamp"]:
                raise HistoryError("生成器静态学习材料晚于本题输入时间，禁止生成或复用")
            messages = self._builder.build_messages(prompt_case(case), self._style_block(case), forced_reply)
            self._material_seal.check()
            self._history_sources.check()
            return messages
        if 'input_cutoff' in case or 'source_span' in case:
            raise HistoryError('历史题缺少绑定的原始数据集，禁止降级为普通生成')
        return self._builder.build_messages(case, self._style_block(case), forced_reply)

    def generate(self, case: dict[str, Any], forced_reply: bool = True) -> dict[str, Any]:
        """返回 {"replies": [...], "latency_ms": float}。无效结果重试一次后抛异常（计入失败）。

        有效生成结果（SOP §3.6）：replies 为字符串数组、每项非空、≤3 条；
        强制回复模式下空数组无效。
        """
        messages = self.build_prompt(case, forced_reply)
        self._check_sources()
        client = self._llm["group" if case.get("chat_type") == "group" else "private"]
        identity = cache.client_identity(client)
        started = time.perf_counter()
        result = cache.memo("generation", {"client": identity, "messages": messages,
                            "forced_reply": forced_reply, "parser": cache.code_digest(__file__)} if identity else None,
                            lambda: self._generate(messages, client, forced_reply))
        self._check_sources()
        return {**result, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}

    def _generate(self, messages, client, forced_reply):
        started = time.perf_counter()
        def request():
            self._check_sources()
            with cache.validation(lambda raw: self._validate(self._parse(raw), forced_reply) is not None):
                raw = client.chat(messages, json_mode=True)
            self._check_sources()
            return raw
        raw = request()
        parsed = self._validate(self._parse(raw), forced_reply)
        tracing.note("validation", {"attempt": 1, "valid": parsed is not None, "parsed": parsed, "raw": raw})
        if parsed is None:
            _logger.warning("生成结果无效，重试一次: %s…", raw[:80])
            raw = request()
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
