"""fewshot 包：新代码的唯一 few-shot 检索实现（见 docs/superpowers/specs/2026-10-03-retrieval-modularization-design.md）。

目标结构：prepare（一次性 PreparedPool）/ sources / recall / rank（逐题纯函数）/ render / session。
本阶段仅迁入 rank 纯函数层；__init__ 暂只 re-export rank 的名，后续阶段随包扩充。
"""
from .rank import (_PROFILE_SITUATIONS, _SKILL_SITUATIONS, _chat_id, _group_participant_profile,
                   _has_context_bound_reply, _has_information_gap, _has_portable_sparse_reply,
                   _has_strong_concrete_payload, _is_brief_exclamation, _is_clarification_reply,
                   _is_information_gap_reply, _is_sparse_turn, _latest_turn_form, _needs_clarification,
                   _precise_situation, _profile_compatibility, _row_situation_profile,
                   _row_situation_tags, _situation_profile, _situation_tags, _terms)

__all__ = ['_PROFILE_SITUATIONS', '_SKILL_SITUATIONS', '_chat_id', '_group_participant_profile',
           '_has_context_bound_reply', '_has_information_gap', '_has_portable_sparse_reply',
           '_has_strong_concrete_payload', '_is_brief_exclamation', '_is_clarification_reply',
           '_is_information_gap_reply', '_is_sparse_turn', '_latest_turn_form', '_needs_clarification',
           '_precise_situation', '_profile_compatibility', '_row_situation_profile',
           '_row_situation_tags', '_situation_profile', '_situation_tags', '_terms']
