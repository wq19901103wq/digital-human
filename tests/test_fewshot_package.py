"""fewshot.rank 纯函数层搬迁 parity：与旧 few_shot 模块零行为变化。

旧模块 src.generator.few_shot 保持逐字节不动（继续服务适配器层与旧快照），
新包 src.generator.fewshot.rank 是同一批纯函数的逐字副本。本文件锁住两者
在任何输入下结果完全相等。
"""
import inspect
from collections import Counter

import pytest

from leakage_support import historical
from src.generator import few_shot
from src.generator.fewshot import rank

MOVED_NAMES = ['_SKILL_SITUATIONS', '_PROFILE_SITUATIONS', '_chat_id', '_terms', '_is_sparse_turn',
               '_is_brief_exclamation', '_latest_turn_form', '_group_participant_profile',
               '_has_strong_concrete_payload', '_has_context_bound_reply', '_has_portable_sparse_reply',
               '_needs_clarification', '_situation_tags', '_precise_situation', '_situation_profile',
               '_is_clarification_reply', '_has_information_gap', '_is_information_gap_reply',
               '_row_situation_profile', '_profile_compatibility', '_row_situation_tags']

QUERY_TEXTS = [
    "在吗",
    "卧槽",
    "有些词我不知道该怎么回",
    "你今天怎么这么牛逼",
    "周末一起出来玩吗？",
    "帮我带杯咖啡",
    "烦死了今天又被领导批了",
    "https://example.com 看这个",
    "你可真行，又装起来了",
    "人呢？到哪了",
    "我昨天刚面的，太坑了",
    "听朋友说他人品不太行",
    "到底是吃火锅还是吃烧烤？",
    "为什么你每次都这样啊",
    "羡慕你们这些大佬",
    "我到了，是3号楼吗？",
    "还行",
    "好的",
    "在忙，先撤了去睡觉",
    "建立上下文的铺垫句要超过三十个字符才行，这里再补一句\n人呢",
    "",
    "？？？",
]

INTERACTION_CONTEXTS = ["", "handling_praise", "answering_questions", "receiving_share", "unknown_skill"]

MESSAGE_LISTS = [
    [],
    [{"sender": "我", "text": "在吗"}, {"sender": "小明", "text": "在"}],
    [{"sender": "小明", "text": "周末出来玩"}, {"sender": "小红", "text": "@小李 来吗"}],
    [{"sender": "我", "text": "走吧"}, {"sender": "小明", "text": "来了"}, {"sender": "小红", "text": "等等我"}],
    [{"sender": "", "text": ""}, {"sender": "我", "text": "ok"}, {"sender": "小李", "text": "收到"}],
    [{"text": "没 sender 的消息"}, {"sender": "我", "text": "嗯"}],
]

SYNTHETIC_ROWS = [
    {"id": "explicit-tags", "context": ["在吗"], "reply": ["在呢"],
     "situation_tags": ["praise", "teasing"]},
    {"id": "explicit-profile", "context": ["问个问题"], "reply": ["比如说？"],
     "situation_profile": {"expected_response_move": "direct_answer", "praise_target": "listener"}},
    {"id": "semantic-act", "context": ["吐槽一下"], "reply": ["抱抱"],
     "semantic_profile": {"incoming_act": "handling_vent"}},
    {"id": "clarify-reply", "context": ["有些词我不知道怎么回"], "reply": ["比如哪个？"]},
    {"id": "gap-reply", "context": ["听朋友说他技术不太行"], "reply": ["真的假的？谁说的"]},
    {"id": "multi-context", "context": ["第一句建立上下文", "第二句补充信息达到三十个字符以上才行哦",
                                        "周末一起出来吃吗"], "reply": ["好啊"]},
    {"id": "empty", "context": [], "reply": []},
    {"id": "non-string", "context": ["看电影", 123, None], "reply": [456, "嗯"]},
]


@pytest.mark.parametrize('name', MOVED_NAMES)
def test_moved_names_source_identical(name):
    old_value = getattr(few_shot, name)
    new_value = getattr(rank, name)
    if inspect.isfunction(old_value):
        assert inspect.getsource(new_value) == inspect.getsource(old_value)
    else:
        assert new_value == old_value


@pytest.mark.parametrize('text', QUERY_TEXTS)
@pytest.mark.parametrize('interaction_context', INTERACTION_CONTEXTS)
@pytest.mark.parametrize('enable_clarification', [False, True])
def test_query_level_parity(text, interaction_context, enable_clarification):
    assert rank._situation_tags(text, interaction_context, enable_clarification) == \
        few_shot._situation_tags(text, interaction_context, enable_clarification)
    for gap_move in (False, True):
        assert rank._situation_profile(text, interaction_context, enable_clarification, gap_move) == \
            few_shot._situation_profile(text, interaction_context, enable_clarification, gap_move)
    assert rank._precise_situation(text) == few_shot._precise_situation(text)
    assert rank._terms(text) == few_shot._terms(text)
    assert rank._latest_turn_form(text) == few_shot._latest_turn_form(text)


@pytest.mark.parametrize('messages', MESSAGE_LISTS)
def test_group_participant_profile_parity(messages):
    assert rank._group_participant_profile(messages) == few_shot._group_participant_profile(messages)


@pytest.mark.parametrize('chat_name', ["", "例子", "某群聊", "chat with spaces"])
def test_chat_id_parity(chat_name):
    assert rank._chat_id(chat_name) == few_shot._chat_id(chat_name)


@pytest.mark.parametrize('row', SYNTHETIC_ROWS)
@pytest.mark.parametrize('enable_clarification', [False, True])
def test_row_level_parity(row, enable_clarification):
    assert rank._row_situation_tags(dict(row), enable_clarification) == \
        few_shot._row_situation_tags(dict(row), enable_clarification)
    for gap_move in (False, True):
        query_profile = few_shot._situation_profile("\n".join(str(value) for value in row.get("context") or []),
                                                    enable_clarification=enable_clarification,
                                                    enable_information_gap_response_move=gap_move)
        new_profile = rank._row_situation_profile(dict(row), enable_clarification, gap_move)
        old_profile = few_shot._row_situation_profile(dict(row), enable_clarification, gap_move)
        assert new_profile == old_profile
        assert rank._profile_compatibility(query_profile, new_profile) == \
            few_shot._profile_compatibility(query_profile, old_profile)


def _pool_rows(tmp_path):
    return historical(tmp_path / 'data').examples


def test_pool_row_level_parity(tmp_path):
    for row in _pool_rows(tmp_path) + SYNTHETIC_ROWS:
        assert rank._row_situation_tags(row, False) == few_shot._row_situation_tags(row, False)
        assert rank._row_situation_tags(row, True) == few_shot._row_situation_tags(row, True)
        assert rank._precise_situation("\n".join(str(value) for value in row.get("context") or [])) == \
            few_shot._precise_situation("\n".join(str(value) for value in row.get("context") or []))
        for gap_move in (False, True):
            assert rank._row_situation_profile(row, False, gap_move) == \
                few_shot._row_situation_profile(row, False, gap_move)


def test_retriever_materialized_values_match_rank(tmp_path):
    """旧类 _load() 物化的逐行画像必须与 rank 纯函数直接计算结果一致。"""
    source = historical(tmp_path / 'data')
    retriever = few_shot.PersonaFewShotRetriever(source.directory / 'fewshot_pool.jsonl')
    rows = retriever._load()
    assert rows
    for row in rows:
        identity = str(row["id"])
        context = [str(value) for value in row.get("context") or []]
        latest_text = context[-1] if context else ""
        assert Counter(rank._terms(" ".join(context))) == retriever._terms_by_id[identity]
        assert rank._latest_turn_form(latest_text) == retriever._latest_turn_forms_by_id[identity]
        assert rank._situation_tags(latest_text, enable_clarification=retriever.enable_clarification) == \
            retriever._latest_situations_by_id[identity]
        assert rank._situation_profile(
            latest_text,
            enable_clarification=retriever.enable_clarification,
            enable_information_gap_response_move=retriever.enable_information_gap_response_move,
        ) == retriever._latest_profiles_by_id[identity]
        assert rank._group_participant_profile(list(row.get("context_messages") or [])) == \
            retriever._group_participant_profiles_by_id[identity]
