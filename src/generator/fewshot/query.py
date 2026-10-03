"""fewshot.query：逐题输入包 QueryContext。

调用方提供的原始参数 + RecallSession.retrieve 计算的查询级派生特征。
score_and_select 只读消费本对象；retrieve 会把排除集（含历史资格并集）
解析结果写回 ``exclude_ids``（等价旧 retrieve 内对 exclude_ids 参数的遮蔽）。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryContext:
    # —— 调用方原始参数（与旧 retrieve 签名一致）——
    query: str
    chat_name: str
    is_group: bool
    limit: int = 8
    relationship: str | None = None
    chat_id: str | None = None
    interaction_context: str = ""
    exclude_ids: set[str] | None = None
    use_situation: bool = True
    current_context_messages: list[dict[str, str]] | None = None
    diversity_prefix_size: int | None = None
    include_retrieval_features: bool = False
    history_case: dict | None = None

    # —— retrieve 内计算的查询级派生特征（与旧实现公式逐字一致）——
    query_terms: Counter = field(default_factory=Counter)
    query_messages: list[str] = field(default_factory=list)
    query_has_multiple_messages: bool = False
    situation_query: str = ""
    query_situations: set[str] = field(default_factory=set)
    query_profile: dict[str, str] = field(default_factory=dict)
    query_latest_form: str = ""
    query_group_participant_profile: tuple[bool, bool, bool, bool, bool] = (False, False, False, False, False)
    precise_situation: str = ""
    current_chat_id: str = ""

    # —— retrieve 内计算的候选池状态（trace 字段；旧 698-701/709-745 语义）——
    candidate_pool_situation_filtered: bool = False
    candidate_pool_precise_situation: bool = False
    candidate_pool_exhaustion_fallback: bool = False
    candidate_pool_sparse_portable: bool = False
