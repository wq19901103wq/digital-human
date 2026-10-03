"""fewshot.prepare：一次性准备层（PreparedPool）。

唯一执行池文件解析、report.json/来源核验、内容哈希与逐行特征物化的入口；
逐题路径（session/rank/引擎）只读消费本模块产物，不做全量加载、内容哈希
或索引构建。

语义与旧 src/generator/few_shot.py::PersonaFewShotRetriever 的
``__init__ + is_approved + _load + _load_render_rows`` 合计等价，验证失败的
fail-closed 语义原样保留。与旧实现的差异（设计 §4 生命周期措辞）：
普通池只在 open_session/PreparedPool.load 边界准备一次，池文件在会话期
间的变化不在逐题路径重新加载（历史池的来源 stamp 变化仍由 check() 立即拒绝）。
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from ...history.sources import load as load_history_sources
from . import rank

_logger = logging.getLogger("src.generator.fewshot")

_TRUSTED_PROVENANCE = {"explicit_human_marker", "before_automation_cutoff"}

# 旧 PersonaFewShotRetriever.__init__ 的冻结策略默认值（SOP §1.6）。
_DEFAULT_FLAGS = dict(
    enable_clarification=False,
    enable_precise_situation=True,
    prefer_same_chat_after_situation=False,
    prefer_same_chat_after_exact_move=False,
    prioritize_latest_turn=False,
    avoid_concrete_examples_for_sparse_query=False,
    prefer_response_move_before_relevance=False,
    prefer_latest_turn_form=True,
    prefer_group_participant_structure=False,
    prefer_multi_message_reply_coverage=False,
    increase_semantic_topic_weight=False,
    enable_information_gap_response_move=True,
)


def _is_trusted_human_example(row: dict[str, Any]) -> bool:
    return row.get("source_provenance") in _TRUSTED_PROVENANCE


class PreparedPool:
    """一次性准备的全部状态持有者；load 之后逐题路径只读。

    内部用普通 dict/list 持有（不做深拷贝防御）；属性以下划线开头，
    包外请只读访问（rows/history_policy/flags 等只读属性见下）。
    """

    def __init__(self, path: Path, *, render_path: Path | None = None,
                 flags: dict[str, bool] | None = None):
        self.path = Path(path)
        self.render_path = Path(render_path) if render_path is not None else None
        merged = dict(_DEFAULT_FLAGS)
        if flags:
            unknown = set(flags) - set(_DEFAULT_FLAGS)
            if unknown:
                raise ValueError(f'未知 fewshot 旗标: {sorted(unknown)}')
            merged.update(flags)
        self.flags = merged
        for key, value in merged.items():
            setattr(self, key, value)
        self.history_policy: str | None = None
        self.history_sources = None
        self.report: dict[str, Any] | None = None
        self.examples_sha256: str | None = None
        self._mtime_ns = -1
        self._rows: list[dict[str, Any]] = []
        self._rows_by_group: dict[bool, list[dict[str, Any]]] = {}
        self._rows_by_group_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        self._rows_by_group_precise_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        self._rows_by_group_latest_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        self._latest_situations_by_id: dict[str, set[str]] = {}
        self._latest_profiles_by_id: dict[str, dict[str, str]] = {}
        self._latest_turn_forms_by_id: dict[str, str] = {}
        self._group_participant_profiles_by_id: dict[str, tuple[bool, bool, bool, bool, bool]] = {}
        self._terms_by_id: dict[str, Counter] = {}
        # 旧逐题 ``(row_id, gap_move)`` 缓存语义固化到准备期：
        # 处境标签只依赖行与实例旗标，按行缓存（首个同 ID 行生效，与旧缓存一致）；
        # 处境画像依赖 gap_move，两版都在 load 时算好。
        self._row_situation_tag_by_id: dict[str, set[str]] = {}
        self._row_situation_profile_by_id: dict[tuple[str, bool], dict[str, str]] = {}
        self._render_mtime_ns = -1
        self._render_rows_by_id: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path, *, render_path: Path | None = None,
             flags: dict[str, bool] | None = None) -> "PreparedPool":
        """一次性准备：来源绑定（旧 is_approved 语义）+ 池加载与物化（旧 _load）。"""
        pool = cls(path, render_path=render_path, flags=flags)
        pool._bind_history()
        pool._load()
        return pool

    # -- 只读访问 --------------------------------------------------------
    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    def check(self) -> None:
        """现 _history_sources.check() 语义：历史来源 stamp 变化立即拒绝；普通池无操作。"""
        if self.history_sources is not None:
            self.history_sources.check()

    def is_approved(self) -> bool:
        """秒级审批判定：基于 load 时物化的 report/哈希与行信任标记。

        判定条件与旧 is_approved 逐条一致；差异在于报告与池哈希在 load 时
        物化（旧实现每次调用重读 report.json 并对整个池重算 sha256）。
        """
        if self.history_sources is not None:
            self.history_sources.check()
        report = self.report
        if report is None:
            return False
        if report.get("review_status") != "approved" or not report.get("examples_sha256"):
            return False
        if not self.examples_sha256 or report["examples_sha256"] != self.examples_sha256:
            return False
        if self.history_sources is not None:
            for row in self._rows:
                self.history_sources.validate(row, example=True)
        return bool(self._rows) and all(_is_trusted_human_example(row) for row in self._rows)

    # -- 旧 is_approved 的来源绑定（含 report 读取与哈希比对）------------
    def _bind_history(self) -> None:
        report_path = self.path.with_name("report.json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.report = report
        if report.get("review_status") != "approved" or not report.get("examples_sha256"):
            return
        self.history_policy = report.get('history_policy')
        if self.history_policy or self.path.with_name('purposes.json').exists():
            # 沿用旧语义：来源核验失败直接抛错（fail-closed），不做任何回退。
            self.history_sources = load_history_sources(self.path.parent)
            self.history_policy = self.history_sources.purpose['history_policy']
        try:
            self.examples_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError:
            self.examples_sha256 = None

    # -- 旧 _load_render_rows -------------------------------------------
    def _load_render_rows(self) -> dict[str, dict[str, Any]]:
        if self.render_path is None:
            return {}
        try:
            mtime_ns = self.render_path.stat().st_mtime_ns
        except OSError:
            return {}
        rows_by_id: dict[str, dict[str, Any]] = {}
        try:
            for line in self.render_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("id") and _is_trusted_human_example(row):
                    rows_by_id[str(row["id"])] = row
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("persona few-shot 展示库加载失败: %s", exc)
            return {}
        self._render_mtime_ns = mtime_ns
        self._render_rows_by_id = rows_by_id
        return rows_by_id

    # -- 旧 _load + 逐行处境/画像缓存物化 --------------------------------
    def _load(self) -> list[dict[str, Any]]:
        try:
            mtime_ns = self.path.stat().st_mtime_ns
        except OSError:
            return []
        rows = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if self.history_sources is not None:
                    self.history_sources.validate(row, example=True)
                if (
                    row.get("id")
                    and isinstance(row.get("context"), list)
                    and isinstance(row.get("reply"), list)
                    and _is_trusted_human_example(row)
                ):
                    rows.append(row)
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("persona few-shot 加载失败: %s", exc)
            return []
        self._mtime_ns = mtime_ns
        self._rows = rows
        rows_by_group: dict[bool, list[dict[str, Any]]] = {False: [], True: []}
        rows_by_group_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        rows_by_group_precise_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        rows_by_group_latest_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        latest_situations_by_id: dict[str, set[str]] = {}
        latest_profiles_by_id: dict[str, dict[str, str]] = {}
        latest_turn_forms_by_id: dict[str, str] = {}
        group_participant_profiles_by_id: dict[str, tuple[bool, bool, bool, bool, bool]] = {}
        render_rows = self._load_render_rows()
        terms_by_id: dict[str, Counter] = {}
        row_situation_tag_by_id: dict[str, set[str]] = {}
        row_situation_profile_by_id: dict[tuple[str, bool], dict[str, str]] = {}
        for row in rows:
            group = row.get("relationship") == "group"
            rows_by_group[group].append(row)
            identity = str(row["id"])
            row_tags = rank._row_situation_tags(row, self.enable_clarification)
            # 逐题缓存首个同 ID 行生效：与旧按题缓存（题内首次计算结果驻留）一致。
            if identity not in row_situation_tag_by_id:
                row_situation_tag_by_id[identity] = row_tags
            for gap_move in (False, True):
                if (identity, gap_move) not in row_situation_profile_by_id:
                    row_situation_profile_by_id[(identity, gap_move)] = rank._row_situation_profile(
                        row, self.enable_clarification, gap_move)
            # 桶成员资格按行自身标签（与旧 _load 逐行计算一致，重复 ID 各按各行）。
            for situation in row_tags:
                rows_by_group_situation.setdefault((group, situation), []).append(row)
            precise_situation = rank._precise_situation("\n".join(str(value) for value in row.get("context") or []))
            if precise_situation:
                rows_by_group_precise_situation.setdefault((group, precise_situation), []).append(row)
            row_messages = [str(value).strip() for value in row.get("context") or [] if str(value).strip()]
            latest_text = row_messages[-1] if row_messages else ""
            latest_situations = rank._situation_tags(
                latest_text,
                enable_clarification=self.enable_clarification,
            )
            latest_profile = rank._situation_profile(
                latest_text,
                enable_clarification=self.enable_clarification,
                enable_information_gap_response_move=self.enable_information_gap_response_move,
            )
            if self.enable_clarification and rank._is_clarification_reply(row):
                latest_situations.add("needs_clarification")
                latest_profile["expected_response_move"] = "clarify_detail"
            latest_situations_by_id[identity] = latest_situations
            latest_profiles_by_id[identity] = latest_profile
            latest_turn_forms_by_id[identity] = rank._latest_turn_form(latest_text)
            named_row = render_rows.get(identity, row)
            group_participant_profiles_by_id[identity] = rank._group_participant_profile(
                list(named_row.get("context_messages") or [])
            )
            for situation in latest_situations:
                rows_by_group_latest_situation.setdefault((group, situation), []).append(row)
            terms_by_id[identity] = Counter(rank._terms(" ".join(row["context"])))
        self._rows_by_group = rows_by_group
        self._rows_by_group_situation = rows_by_group_situation
        self._rows_by_group_precise_situation = rows_by_group_precise_situation
        self._rows_by_group_latest_situation = rows_by_group_latest_situation
        self._latest_situations_by_id = latest_situations_by_id
        self._latest_profiles_by_id = latest_profiles_by_id
        self._latest_turn_forms_by_id = latest_turn_forms_by_id
        self._group_participant_profiles_by_id = group_participant_profiles_by_id
        self._terms_by_id = terms_by_id
        self._row_situation_tag_by_id = row_situation_tag_by_id
        self._row_situation_profile_by_id = row_situation_profile_by_id
        return rows
