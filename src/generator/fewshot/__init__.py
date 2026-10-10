"""fewshot 包：新代码的唯一 few-shot 检索实现（见 docs/superpowers/specs/2026-10-03-retrieval-modularization-design.md）。

结构：prepare（一次性 PreparedPool）/ query（逐题输入包）/
recall_engines（逐题资格与词项特征引擎）/ rank（逐题纯函数打分）/
session（RecallSession 逐题会话 + open_session 工厂）；
features / crosses / boosting（现用示例评分）；training（离线学习）。
"""
from .prepare import PreparedPool
from .query import QueryContext
from .rank import (_PROFILE_SITUATIONS, _SKILL_SITUATIONS, _chat_id, _group_participant_profile,
                   _has_context_bound_reply, _has_information_gap, _has_portable_sparse_reply,
                   _has_strong_concrete_payload, _is_brief_exclamation, _is_clarification_reply,
                   _is_information_gap_reply, _is_sparse_turn, _latest_turn_form, _needs_clarification,
                   _precise_situation, _profile_compatibility, _row_situation_profile,
                   _row_situation_tags, _situation_profile, _situation_tags, _terms)
from .recall_engines import InMemoryRecallEngine, RecallEngine, SqliteIndexedRecallEngine
from .session import RecallSession, open_session

__all__ = ['PreparedPool', 'QueryContext', 'RecallEngine', 'RecallSession',
           'InMemoryRecallEngine', 'SqliteIndexedRecallEngine', 'open_session',
           '_PROFILE_SITUATIONS', '_SKILL_SITUATIONS', '_chat_id', '_group_participant_profile',
           '_has_context_bound_reply', '_has_information_gap', '_has_portable_sparse_reply',
           '_has_strong_concrete_payload', '_is_brief_exclamation', '_is_clarification_reply',
           '_is_information_gap_reply', '_is_sparse_turn', '_latest_turn_form', '_needs_clarification',
           '_precise_situation', '_profile_compatibility', '_row_situation_profile',
           '_row_situation_tags', '_situation_profile', '_situation_tags', '_terms']


def __getattr__(name):
    if name == 'logistic':
        from .training import logistic
        return logistic
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
