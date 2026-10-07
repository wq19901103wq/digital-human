"""fewshot.prepare：一次性准备层（PreparedPool）。

唯一执行池文件解析、report.json/来源核验、内容哈希与逐行特征物化的入口；
逐题路径（session/rank/引擎）只读消费本模块产物，不做全量加载、内容哈希
或索引构建。

语义与旧 src/generator/few_shot.py::PersonaFewShotRetriever 的
``__init__ + is_approved + _load + _load_render_rows`` 合计等价，验证失败的
fail-closed 语义原样保留。与旧实现的差异（设计 §4 生命周期措辞）：
普通池只在 open_session/PreparedPool.load 边界准备一次，池文件在会话期
间的变化不在逐题路径重新加载（历史池的来源 stamp 变化仍由 check() 立即拒绝）。

I4：处境标签/画像不在准备期按 ID 物化——重复 ID 时旧类是打分循环内惰性
缓存（同 ID 行第一次进入循环的那行参与计算并驻留、跨查询），标签键 id、
画像键 (id, gap_move)；该缓存现住在 RecallSession 上（见 session.py）。

I10：load 完成后 PreparedPool 深只读——行 tuple 化并逐行包 MappingProxyType，
桶/特征/画像/flags 等映射包 MappingProxyType；任何属性赋值抛 TypeError。
行内嵌套结构递归冻结：dict（含 Counter 等 dict 子类）包只读映射视图、
list 包 FrozenList（list 子类，支持取值/迭代/in 与 list 相等语义，改/增/删
一律 TypeError）、set 包 frozenset；解冻见 thaw_value——召回返回时已递归
解冻为全新普通 list/dict，调用方拿到的与池内部零共享。

I9：历史池存在有效磁盘索引时，open_session 先探测索引绑定（代码摘要/
flags/内容哈希），有效则本模块 hydrate_from_index 从 sqlite 恢复行定位与桶，
正文和词表保留按需读取的深只读视图，不全量解码；来源验证由绑定内容哈希
保证，恢复后仍调 check() 做 stamp 轻量检查；无效/缺失才走内存 _load。
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ...history.sources import load as load_history_sources
from . import rank

_logger = logging.getLogger("src.generator.fewshot")


class FrozenList(list):
    """只读 list（I10）：保持 list 子类身份（isinstance/相等/序列协议与旧实现一致），
    任何修改操作抛 TypeError。防正常接口的意外写入，不防 list.append(x) 式主动绕行。"""

    _MESSAGE = 'fewshot 冻结列表禁止修改'

    def __init__(self, items=()):
        list.__init__(self, items)

    def _readonly(self, *args, **kwargs):
        raise TypeError(self._MESSAGE)

    __setitem__ = _readonly
    __delitem__ = _readonly
    __iadd__ = _readonly
    __imul__ = _readonly
    append = _readonly
    extend = _readonly
    insert = _readonly
    pop = _readonly
    remove = _readonly
    clear = _readonly
    sort = _readonly
    reverse = _readonly


class FrozenDict(dict):
    """只读 dict（I10）：保持 dict 子类身份（rank.py 的 isinstance(explicit, dict)
    显式画像判定不受影响），任何修改操作抛 TypeError。"""

    _MESSAGE = 'fewshot 冻结映射禁止修改'

    def _readonly(self, *args, **kwargs):
        raise TypeError(self._MESSAGE)

    __setitem__ = _readonly
    __delitem__ = _readonly
    __ior__ = _readonly
    update = _readonly
    setdefault = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly


# 标量/不可变类型快速通道：冻结递归对叶子节点一律原样返回，
# 用 type() 精确匹配避免逐节点多次 isinstance 调用（10 万行规模加载期基准）。
_IMMUTABLE_TYPES = (str, int, float, bool, bytes, type(None))


class _FrozenMapping(Mapping):
    """按访问冻结磁盘映射的值，不枚举或物化未访问的正文与特征。"""

    __slots__ = ('_mapping',)

    def __init__(self, mapping):
        self._mapping = mapping

    def __getitem__(self, key):
        return _freeze_value(self._mapping[key])

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)


def _freeze_value(value: Any) -> Any:
    """递归冻结行内嵌套结构：普通 dict→FrozenDict、其他 dict 子类→只读映射视图
    （Counter 等计数视图零拷贝直接包装，计数值为不可变 int）、list→FrozenList、
    set→frozenset、tuple 递归重建；标量原样返回。"""
    if type(value) in _IMMUTABLE_TYPES:
        return value
    if type(value) is dict:
        return FrozenDict({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, dict):
        # Counter 等计数映射：计数值为 int，零拷贝只读视图即可。
        return MappingProxyType(value)
    if isinstance(value, list):
        return FrozenList([_freeze_value(item) for item in value])
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, Mapping):
        return _FrozenMapping(value)
    return value


def thaw_value(value: Any) -> Any:
    """冻结值的递归解冻（I10 返回边界）：MappingProxyType/FrozenDict→dict、
    FrozenList→list、frozenset→set、tuple 递归重建；产物为全新普通容器，
    与池内部零共享。"""
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [thaw_value(item) for item in value]
    if isinstance(value, (frozenset, set)):
        return {thaw_value(item) for item in value}
    if isinstance(value, tuple):
        return tuple(thaw_value(item) for item in value)
    return value

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
    """一次性准备的全部状态持有者；load 之后逐题路径只读（I10 深只读）。

    load/hydrate 完成后结构冻结：属性赋值抛 TypeError，行与映射均为
    MappingProxyType/tuple。构建期（_load/hydrate 内部）在冻结前完成全部
    可变操作。
    """

    def __init__(self, path: Path, *, render_path: Path | None = None,
                 flags: dict[str, bool] | None = None):
        self._frozen = False
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
        self._row_ids_by_group: dict[bool, frozenset[str]] = {}
        self._minimum_identity_length = 0
        self._rows_by_group_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        self._rows_by_group_precise_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        self._rows_by_group_latest_situation: dict[tuple[bool, str], list[dict[str, Any]]] = {}
        self._latest_situations_by_id: dict[str, set[str]] = {}
        self._latest_profiles_by_id: dict[str, dict[str, str]] = {}
        self._latest_turn_forms_by_id: dict[str, str] = {}
        self._group_participant_profiles_by_id: dict[str, tuple[bool, bool, bool, bool, bool]] = {}
        self._terms_by_id: dict[str, Counter] = {}
        self._render_mtime_ns = -1
        self._render_rows_by_id: dict[str, dict[str, Any]] = {}

    # -- I10 深只读 --------------------------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_frozen', False):
            raise TypeError('PreparedPool 已冻结（深只读）：禁止属性赋值')
        object.__setattr__(self, name, value)

    def _freeze(self) -> None:
        """构建完成后递归冻结全部内部结构（行 tuple + 只读包装，I10 深只读）。

        行内嵌套 dict/list/set 经 _freeze_value 递归冻结；桶成员与 _rows 持有
        同一批行代理（id(row) 缓存保证身份一致，_build_examples 的
        ``member is row`` 判定不受影响）。
        """
        frozen_rows: dict[int, Any] = {}

        def freeze_row(row):
            proxy = frozen_rows.get(id(row))
            if proxy is None:
                proxy = MappingProxyType(_freeze_value(row))
                frozen_rows[id(row)] = proxy
            return proxy

        object.__setattr__(self, '_rows', tuple(freeze_row(row) for row in self._rows))
        for name in ('_rows_by_group', '_rows_by_group_situation',
                     '_rows_by_group_precise_situation', '_rows_by_group_latest_situation'):
            mapping = getattr(self, name)
            object.__setattr__(self, name, MappingProxyType({
                key: FrozenList(freeze_row(member) for member in members)
                for key, members in mapping.items()
            }))
        object.__setattr__(self, '_row_ids_by_group', MappingProxyType({
            key: frozenset(str(row['id']) for row in members)
            for key, members in self._rows_by_group.items()
        }))
        object.__setattr__(self, '_minimum_identity_length', min(
            (len(identity) for identities in self._row_ids_by_group.values() for identity in identities),
            default=0))
        for name in ('_latest_situations_by_id', '_latest_profiles_by_id', '_latest_turn_forms_by_id',
                     '_group_participant_profiles_by_id', '_terms_by_id'):
            mapping = getattr(self, name)
            frozen = ({key: _freeze_value(value) for key, value in mapping.items()}
                      if isinstance(mapping, dict) else _FrozenMapping(mapping))
            object.__setattr__(self, name, MappingProxyType(frozen))
        object.__setattr__(self, '_render_rows_by_id', MappingProxyType({
            key: MappingProxyType(_freeze_value(row)) for key, row in self._render_rows_by_id.items()
        }))
        object.__setattr__(self, 'flags', MappingProxyType(self.flags))
        object.__setattr__(self, '_frozen', True)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path, *, render_path: Path | None = None,
             flags: dict[str, bool] | None = None) -> "PreparedPool":
        """一次性准备：来源绑定（旧 is_approved 语义）+ 池加载与物化（旧 _load）。"""
        pool = cls(path, render_path=render_path, flags=flags)
        pool._bind_history()
        pool._load()
        pool._freeze()
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
        for row in rows:
            group = row.get("relationship") == "group"
            rows_by_group[group].append(row)
            identity = str(row["id"])
            # I4：处境标签/画像不在准备期物化（重复 ID 语义见模块文档）。
            row_tags = rank._row_situation_tags(row, self.enable_clarification)
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
        return rows

    def hydrate_from_index(self, reader) -> None:
        """索引绑定已由 database() 校验（代码摘要/flags/来源内容哈希），此处
        恢复行定位/桶及小特征（按 position 顺序）；正文和词表按需解码。

        语义与适配器 disk_history.load_examples 的恢复后半段一致，重复 ID
        保持末行覆盖（latest_* / terms 均 ORDER BY position 后写）。恢复后
        调 check() 做来源 stamp 轻量检查，随后冻结（I10）。
        """
        from .. import disk_history

        if self.history_sources is None:
            raise ValueError('仅历史策略池可从磁盘索引恢复')
        connection = reader.connection
        rows = []
        identities = set()
        rows_by_group = {False: [], True: []}
        restored_buckets = {
            name: {} for name in disk_history.BUCKETS}
        for offset, size, header, stored_buckets in connection.execute(
                'SELECT offset,size,header,buckets FROM examples ORDER BY position'):
            row = disk_history.Row(reader, offset, size, json.loads(header))
            rows.append(row)
            identities.add(str(row['id']))
            group = row.get('relationship') == 'group'
            rows_by_group[group].append(row)
            for name, keys in zip(disk_history.BUCKETS, json.loads(stored_buckets)):
                for key in keys:
                    restored_buckets[name].setdefault((group, key), []).append(row)
        latest_situations_by_id: dict[str, set[str]] = {}
        latest_profiles_by_id: dict[str, dict[str, str]] = {}
        latest_turn_forms_by_id: dict[str, str] = {}
        group_participant_profiles_by_id: dict[str, tuple[bool, bool, bool, bool, bool]] = {}
        for (identity, situations, profile, turn_form, participants) in connection.execute(
                'SELECT id, latest_situations, latest_profile, latest_turn_form, '
                'group_participant_profile FROM examples ORDER BY position'):
            latest_situations_by_id[identity] = set(json.loads(situations))
            latest_profiles_by_id[identity] = json.loads(profile)
            latest_turn_forms_by_id[identity] = turn_form
            group_participant_profiles_by_id[identity] = tuple(json.loads(participants))
        self._mtime_ns = self.path.stat().st_mtime_ns
        self._rows = rows
        self._rows_by_group = rows_by_group
        self._rows_by_group_situation = restored_buckets[disk_history.BUCKETS[0]]
        self._rows_by_group_precise_situation = restored_buckets[disk_history.BUCKETS[1]]
        self._rows_by_group_latest_situation = restored_buckets[disk_history.BUCKETS[2]]
        self._latest_situations_by_id = latest_situations_by_id
        self._latest_profiles_by_id = latest_profiles_by_id
        self._latest_turn_forms_by_id = latest_turn_forms_by_id
        self._group_participant_profiles_by_id = group_participant_profiles_by_id
        self._terms_by_id = disk_history.FeatureMap(reader, frozenset(identities), '_terms_by_id')
        self.check()
        self._freeze()
