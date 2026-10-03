"""fewshot.rank 纯函数层搬迁 parity：与旧 few_shot 模块零行为变化。

活树 src/generator/few_shot.py 已删除（模块化重构收尾）；旧行为参照改由
认证归档 a5118947… 的副本提供（tests/_reference_few_shot.py 按路径加载，
字节 pin 强校验）。新包 src.generator.fewshot.rank 是同一批纯函数的逐字
副本。本文件锁住两者在任何输入下结果完全相等。
"""
import inspect
from collections import Counter

import pytest

from _reference_few_shot import few_shot
from leakage_support import historical
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


# ---------------------------------------------------------------------------
# RecallSession parity：新会话（内存/索引双引擎）vs 旧 PersonaFewShotRetriever
# 输出逐字节一致（retrieve 全参数矩阵 / render_selected / is_approved）。
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from src.generator import disk_history
from src.generator.fewshot import (InMemoryRecallEngine, RecallSession,
                                   SqliteIndexedRecallEngine, open_session)


def _example(identity, context, reply, *, group=False, chat_id='chat-a',
             reply_shape='single', semantic_profile=None, situation_tags=None,
             situation_profile=None, provenance='before_automation_cutoff'):
    row = {
        'id': identity,
        'context': [f'朋友: {text}' for text in context],
        'context_messages': [{'sender': '朋友', 'text': text} for text in context],
        'reply': list(reply),
        'relationship': 'group' if group else 'private',
        'chat_id': chat_id,
        'chat_name': '某群' if group else '某友',
        'reply_shape': reply_shape,
        'source_provenance': provenance,
    }
    if semantic_profile:
        row['semantic_profile'] = semantic_profile
    if situation_tags:
        row['situation_tags'] = situation_tags
    if situation_profile:
        row['situation_profile'] = situation_profile
    return row


def _synthetic_rows():
    return [
        _example('praise-1', ['你今天怎么这么牛逼'], ['少来，别捧了'], reply_shape='laugh',
                 semantic_profile={'incoming_act': 'handling_praise', 'response_move': 'deflect'}),
        _example('praise-2', ['羡慕你们这些大佬'], ['我们也就混口饭']),
        _example('question-1', ['晚上吃火锅还是烧烤？'], ['还是火锅吧']),
        _example('question-2', ['为什么你每次都迟到'], ['路上太堵了']),
        _example('vent-1', ['烦死了今天又被领导批了'], ['抱抱，辛苦了'],
                 semantic_profile={'incoming_act': 'handling_vent', 'response_move': 'comfort'}),
        _example('share-1', ['https://example.com 看这个'], ['哈哈这也能行'],
                 semantic_profile={'incoming_act': 'receiving_share', 'response_move': 'comment'}),
        _example('tease-1', ['你可真行，又装起来了'], ['哈哈哪有'],
                 semantic_profile={'incoming_act': 'hostile_teasing', 'response_move': 'playful_counter'}),
        _example('invite-1', ['周末一起出来玩吗？'], ['来啊，去哪']),
        _example('presence-1', ['在吗'], ['在呢']),
        _example('sparse-ok', ['好的'], ['嗯']),
        _example('gap-1', ['听朋友说他技术不太行'], ['真的假的？谁说的']),
        _example('multi-1', ['第一句铺垫一下', '第二句再补充一点信息', '所以你怎么看'], ['我觉得还行', '你呢'],
                 reply_shape='multi'),
        _example('loc-1', ['人呢？到哪了'], ['快到了']),
        _example('group-1', ['群里说这个'], ['看到了'], group=True, chat_id='chat-g'),
        _example('group-2', ['@小李 你来吗'], ['来了'], group=True, chat_id='chat-g'),
        _example('untrusted-1', ['这条不应入库'], ['嗯'], provenance='model_generated'),
        # 重复 ID：两行内容不同，验证首个同 ID 行缓存语义与桶成员资格。
        _example('dup-1', ['重复编号第一行'], ['第一版回复']),
        _example('dup-1', ['重复编号第二行内容不一样'], ['第二版回复'], reply_shape='laugh'),
    ]


PARITY_QUERIES = [
    '在吗',
    '卧槽',
    '你今天怎么这么牛逼',
    '晚上一起吃火锅吗？',
    '烦死了今天好累',
    'https://foo.bar 看这个',
    '你可真行',
    '好的',
    '听同学说他水平不太行',
    '人呢？到哪了',
    '第一句铺垫\n第二句补充\n周末一起出来玩吗',
    '',
]

# (use_situation, prioritize_latest_turn, include_retrieval_features,
#  diversity_prefix_size, exclude_ids, limit)
PARITY_COMBOS = [
    (True, False, False, None, None, 8),
    (True, False, True, None, None, 8),
    (False, False, False, None, None, 8),
    (True, True, False, None, None, 8),
    (True, True, True, None, None, 8),
    (False, True, False, None, None, 5),
    (True, False, False, 1, None, 8),
    (True, False, False, 2, None, 3),
    (True, False, True, 2, {'praise-1', 'presence-1'}, 8),
    (True, True, False, None, {'question-1'}, 4),
    (True, False, False, None, {'gap-1', 'loc-1', 'dup-1'}, 8),
]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return Path(path)


@pytest.fixture
def synthetic_pool(tmp_path):
    return _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _synthetic_rows())


def _old_retriever(pool_path, *, render_path=None, prioritize_latest_turn=False):
    retriever = few_shot.PersonaFewShotRetriever(path=pool_path, render_path=render_path)
    retriever.prioritize_latest_turn = prioritize_latest_turn
    return retriever


def _new_session(pool_path, *, render_path=None, prioritize_latest_turn=False):
    flags = {'prioritize_latest_turn': True} if prioritize_latest_turn else None
    return open_session(pool_path, render_path=render_path, flags=flags,
                        index_policy='use_if_present')


@pytest.mark.parametrize('query', PARITY_QUERIES)
@pytest.mark.parametrize('is_group', [False, True])
@pytest.mark.parametrize('combo', PARITY_COMBOS)
def test_recall_session_retrieve_parity(synthetic_pool, query, is_group, combo):
    use_situation, prioritize, with_features, prefix, excluded, limit = combo
    old = _old_retriever(synthetic_pool, prioritize_latest_turn=prioritize)
    new = _new_session(synthetic_pool, prioritize_latest_turn=prioritize)
    kwargs = dict(
        chat_name='某群' if is_group else '某友', is_group=is_group, limit=limit,
        relationship='group' if is_group else 'private',
        exclude_ids=set(excluded) if excluded else None,
        use_situation=use_situation, diversity_prefix_size=prefix,
        include_retrieval_features=with_features,
        current_context_messages=[{'sender': '我', 'text': '走吧'}, {'sender': '朋友', 'text': '来了'}],
    )
    expected = old.retrieve(query=query, **kwargs)
    actual = new.retrieve(query=query, **kwargs)
    assert actual == expected
    if with_features:
        assert all('retrieval_features' in row for row in actual) == \
            all('retrieval_features' in row for row in expected)
    # 返回边界（设计 §4）：嵌套修改不污染后续召回。
    if actual:
        actual[0].setdefault('context', []).append('污染')
        again = new.retrieve(query=query, **kwargs)
        assert again == expected


def test_recall_session_is_approved_parity(tmp_path):
    rows = _synthetic_rows()
    pool = _write_jsonl(tmp_path / 'approved' / 'pool.jsonl', rows)
    import hashlib
    report = {'review_status': 'approved',
              'examples_sha256': hashlib.sha256(pool.read_bytes()).hexdigest()}
    pool.with_name('report.json').write_text(json.dumps(report), encoding='utf-8')
    old = few_shot.PersonaFewShotRetriever(path=pool)
    new = open_session(pool, index_policy='use_if_present')
    assert old.is_approved() == new.is_approved() is True
    # 审批状态降级 → 两侧一致 False。
    pool.with_name('report.json').write_text(json.dumps({'review_status': 'pending'}),
                                             encoding='utf-8')
    old2 = few_shot.PersonaFewShotRetriever(path=pool)
    new2 = open_session(pool, index_policy='use_if_present')
    assert old2.is_approved() is False
    assert new2.is_approved() is False
    # 内容哈希不符 → 两侧一致 False（新会话在 load 时绑定来源，判定仍一致）。
    pool.with_name('report.json').write_text(json.dumps(report), encoding='utf-8')
    with pool.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(_example('extra-1', ['追加一行'], ['嗯']), ensure_ascii=False) + '\n')
    old3 = few_shot.PersonaFewShotRetriever(path=pool)
    new3 = open_session(pool, index_policy='use_if_present')
    assert old3.is_approved() is False
    assert new3.is_approved() is False


def test_recall_session_render_selected_parity(tmp_path):
    pool_path = _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _synthetic_rows())
    render_rows = []
    for row in _synthetic_rows()[:6]:
        render_rows.append({
            'id': row['id'], 'source_provenance': 'explicit_human_marker',
            'context_messages': [{'sender': '朋友', 'text': f"渲染版-{row['context_messages'][0]['text']}"}],
            'reply_messages': [{'sender': '我', 'text': '渲染版回复'}],
        })
    render_path = _write_jsonl(tmp_path / 'plain' / 'render.jsonl', render_rows)
    old = _old_retriever(pool_path, render_path=render_path)
    new = _new_session(pool_path, render_path=render_path)
    rows = old.retrieve(query='在吗', chat_name='某友', is_group=False, limit=4)
    assert rows
    for max_chars in (200, 800, 2500):
        assert new.render_selected(rows, max_chars=max_chars) == \
            old.render_selected(rows, max_chars=max_chars)
    assert RecallSession.render(rows, max_chars=2500) == \
        few_shot.PersonaFewShotRetriever.render(rows, max_chars=2500)


def _history_fixture(tmp_path):
    return historical(tmp_path / 'data')


def _history_kwargs(case, size):
    messages = case['context']
    return dict(
        query='\n'.join(str(m.get('text', '')) for m in messages[-size:]),
        chat_name=str(case.get('chat_name', '')),
        is_group=case.get('chat_type') == 'group',
        limit=12,
        exclude_ids={str(case.get('case_id', ''))},
        history_case=case,
        current_context_messages=[{'sender': str(m.get('sender', '')),
                                   'text': str(m.get('text', ''))} for m in messages],
    )


def _assert_history_parity(old, new, fixture):
    assert old.is_approved() == new.is_approved() is True
    for case in fixture.cases:
        for size in (3, 1):
            kwargs = _history_kwargs(case, size)
            expected = old.retrieve(**kwargs)
            actual = new.retrieve(**kwargs)
            assert actual == expected
            traced = new.retrieve(**{**kwargs, 'include_retrieval_features': True})
            traced_old = old.retrieve(**{**kwargs, 'include_retrieval_features': True})
            assert traced == traced_old
    # 无 history_case 的历史召回必须失败关闭（两侧一致）。
    from src.generator.history import HistoryError
    with pytest.raises(HistoryError):
        old.retrieve(query='问题', chat_name='例子', is_group=False)
    with pytest.raises(HistoryError):
        new.retrieve(query='问题', chat_name='例子', is_group=False)


def test_recall_session_history_pool_in_memory_parity(tmp_path):
    fixture = _history_fixture(tmp_path)
    pool = fixture.directory / 'fewshot_pool.jsonl'
    old = few_shot.PersonaFewShotRetriever(pool)
    old.is_approved()
    new = open_session(pool, index_policy='use_if_present')
    assert isinstance(new._engine, InMemoryRecallEngine)
    _assert_history_parity(old, new, fixture)


def test_recall_session_history_pool_indexed_parity(tmp_path):
    fixture = _history_fixture(tmp_path)
    pool = fixture.directory / 'fewshot_pool.jsonl'
    # 索引干净缺失时 use_if_present → 内存引擎；同一池随后构建索引。
    in_memory = open_session(pool, index_policy='use_if_present')
    assert isinstance(in_memory._engine, InMemoryRecallEngine)
    indexed = open_session(pool, index_policy='build')
    assert isinstance(indexed._engine, SqliteIndexedRecallEngine)
    # 索引引擎与内存引擎逐题输出一致。
    for case in fixture.cases:
        for size in (3, 1):
            kwargs = _history_kwargs(case, size)
            assert indexed.retrieve(**kwargs) == in_memory.retrieve(**kwargs)
            assert indexed.retrieve(**{**kwargs, 'include_retrieval_features': True}) == \
                in_memory.retrieve(**{**kwargs, 'include_retrieval_features': True})
    # 旧 AST 补丁路径（disk_storage）与两个新引擎一致。
    with disk_history.disk_storage(few_shot):
        old = few_shot.PersonaFewShotRetriever(pool)
        old.is_approved()
        for case in fixture.cases[:3]:
            for size in (3, 1):
                kwargs = _history_kwargs(case, size)
                assert indexed.retrieve(**kwargs) == [dict(row) for row in old.retrieve(**kwargs)]
    indexed.close()


def test_open_session_index_policy_matrix(tmp_path):
    fixture = _history_fixture(tmp_path)
    pool = fixture.directory / 'fewshot_pool.jsonl'
    # 干净缺失 + use_if_present → 内存引擎（不落盘）。
    session = open_session(pool, index_policy='use_if_present')
    assert isinstance(session._engine, InMemoryRecallEngine)
    # build → 构建并挂索引引擎；二次打开秒级复用。
    built = open_session(pool, index_policy='build')
    assert isinstance(built._engine, SqliteIndexedRecallEngine)
    built.close()
    reopened = open_session(pool, index_policy='use_if_present')
    assert isinstance(reopened._engine, SqliteIndexedRecallEngine)
    reopened.close()
    # 校验和不符（sqlite 在、manifest 被篡改）→ 拒绝。
    cache = tmp_path / '.cache' / 'disk-history'
    manifest = next(cache.glob('*.json'))
    manifest.write_text(json.dumps({'identity': 'x', 'sha256': 'y'}), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum mismatch'):
        open_session(pool, index_policy='build')
    # 发布不完整（sqlite 缺失、manifest 保留）→ 拒绝。
    target = next(cache.glob('*.sqlite3'))
    target.unlink()
    with pytest.raises(ValueError, match='incomplete'):
        open_session(pool, index_policy='build')
    # 未知策略 → 拒绝。
    with pytest.raises(ValueError, match='索引策略'):
        open_session(pool, index_policy='bogus')


def test_open_session_plain_pool_never_indexes(tmp_path):
    pool = synthetic_pool = _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _synthetic_rows())
    session = open_session(pool, index_policy='build')
    assert isinstance(session._engine, InMemoryRecallEngine)
    assert not (tmp_path / '.cache').exists()


def test_session_exposes_controlled_identity_attributes_for_adapter_cache_key(tmp_path):
    """vars() 公共身份面与旧 PersonaFewShotRetriever 一致：适配器层 retrieval_cache
    的 vars(retriever._original) 键扫描对 open_session 产物继续显式稳定（设计 §6）。"""
    pool_path = _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _synthetic_rows())
    session = open_session(pool_path, flags=dict(enable_clarification=True),
                           index_policy='use_if_present')
    try:
        old = {name for name in vars(few_shot.PersonaFewShotRetriever(pool_path))
               if not name.startswith('_')}
        public = {name: value for name, value in vars(session).items()
                  if not name.startswith('_')}
        assert set(public) == old
        assert public['path'] == Path(pool_path) and public['render_path'] is None
        assert public['history_policy'] is None and session._history_sources is None
        assert public['enable_clarification'] is True
        assert public['enable_information_gap_response_move'] is True
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 修复批次 2（2026-10-03 codex 复审）：
# I4 重复 ID 的惰性画像绑定语义 / I9 索引水合路径 / I10 PreparedPool 深只读。
# ---------------------------------------------------------------------------

from src.generator.fewshot import prepare


def _duplicate_id_rows():
    """同 ID 两行、情境标签与画像均不同（显式 tags/profile），见 I4。"""
    return [
        _example('dup-i4', ['普通陈述一句'], ['嗯'], reply_shape='single',
                 situation_tags=['praise'],
                 situation_profile={'expected_response_move': 'deflect_or_playfully_accept'}),
        # 桶只命中第二行：唯一带 question 标签的同 ID 行。
        _example('dup-i4', ['另一个普通陈述'], ['好吧'], reply_shape='laugh',
                 situation_tags=['question'],
                 situation_profile={'expected_response_move': 'direct_answer'}),
        _example('other-i4', ['周末一起出来玩吗？'], ['来啊']),
    ]


def test_duplicate_id_lazy_cache_matches_archive_second_row_bucket(tmp_path):
    """I4：情境桶只选中第二条同 ID 行时，惰性缓存（按“第一次进入循环的行”
    计算并驻留、跨查询）与归档旧类逐一相等；连续 retrieve 驻留生效后仍相等。"""
    pool_path = _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _duplicate_id_rows())
    old = _old_retriever(pool_path)
    new = _new_session(pool_path)
    # 两个 question 查询 + 两个带信息缺口的 question 查询（覆盖 gap_move
    # 两版画像键 (id, False)/(id, True) 的首次计算与二次驻留）。
    requests = [
        dict(query='这是为什么啊', chat_name='某友', is_group=False, limit=4),
        dict(query='到底怎么弄呢？', chat_name='某友', is_group=False, limit=4),
        dict(query='听同学说他水平不太行，为什么啊', chat_name='某友', is_group=False, limit=4),
        dict(query='听朋友说他技术不太行，到底为什么', chat_name='某友', is_group=False, limit=4),
    ]
    for request in requests:
        expected = old.retrieve(**request)
        actual = new.retrieve(**request)
        assert actual == expected
        dup_rows = [row for row in actual if row['id'] == 'dup-i4']
        assert len(dup_rows) == 1 and dup_rows[0]['reply'] == ['好吧']  # 桶只命中第二行
    # 驻留确实生效：标签键 id、画像键 (id, gap_move) 均已缓存。
    assert 'dup-i4' in new._row_situation_tag_cache
    assert ('dup-i4', False) in new._row_situation_profile_cache
    assert ('dup-i4', True) in new._row_situation_profile_cache


def test_prepared_pool_is_deep_read_only(tmp_path):
    """I10：池属性赋值/行修改/映射写入一律抛错或被拒。"""
    pool_path = _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _synthetic_rows())
    session = _new_session(pool_path)
    pool = session.pool
    assert isinstance(pool.rows, tuple)
    with pytest.raises(TypeError, match='深只读'):
        pool.history_policy = 'complete_before_input_v1'
    with pytest.raises(TypeError, match='深只读'):
        pool.enable_clarification = True
    with pytest.raises(TypeError, match='深只读'):
        pool._frozen = False
    row = pool.rows[0]
    with pytest.raises(TypeError):
        row['id'] = 'changed'
    with pytest.raises(TypeError):
        row['injected'] = True
    with pytest.raises(AttributeError):
        row.update({'id': 'changed'})
    with pytest.raises(TypeError):
        pool._terms_by_id['x'] = Counter()
    with pytest.raises(TypeError):
        pool._rows_by_group[False] = []
    with pytest.raises(TypeError):
        pool.flags['enable_clarification'] = False
    with pytest.raises(AttributeError):
        row.injected = True


def test_prepared_pool_nested_structures_are_deep_read_only(tmp_path):
    """I10 深只读：行内嵌套 dict/list、桶成员列表、特征 Counter 的写/增/删全抛错。"""
    pool_path = _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _synthetic_rows())
    pool = _new_session(pool_path).pool
    row = pool.rows[0]
    # 行内嵌套 list：改/增/删一律 TypeError。
    with pytest.raises(TypeError, match='冻结列表'):
        row['reply'].clear()
    with pytest.raises(TypeError, match='冻结列表'):
        row['context'].append('注入')
    with pytest.raises(TypeError, match='冻结列表'):
        row['context'][0] = '改写'
    with pytest.raises(TypeError, match='冻结列表'):
        del row['reply'][0]
    with pytest.raises(TypeError, match='冻结列表'):
        row['context'] += ['绕过 iadd']
    # 行内嵌套 dict（isinstance dict 语义保留，但写入被拒）。
    profiled = next(r for r in pool.rows if 'semantic_profile' in r)
    assert isinstance(profiled['semantic_profile'], dict)
    with pytest.raises(TypeError, match='冻结映射'):
        profiled['semantic_profile']['incoming_act'] = 'tampered'
    with pytest.raises(TypeError, match='冻结映射'):
        profiled['semantic_profile'].update({'incoming_act': 'tampered'})
    with pytest.raises(TypeError, match='冻结映射'):
        profiled['semantic_profile'].pop('incoming_act')
    # 桶成员与 _rows 是同一批行代理：经桶写同样被拒，桶本身也不可增删。
    bucket_row = pool._rows_by_group[False][0]
    with pytest.raises(TypeError):
        bucket_row['id'] = 'changed'
    with pytest.raises(TypeError, match='冻结列表'):
        pool._rows_by_group[False].append({'id': '注入'})
    with pytest.raises(TypeError, match='冻结列表'):
        pool._rows_by_group_situation[next(iter(pool._rows_by_group_situation))].pop()
    # 特征 Counter（只读映射视图）：外层与内层写入均拒绝。
    terms = pool._terms_by_id[str(row['id'])]
    with pytest.raises(TypeError):
        terms['注入词'] = 99
    with pytest.raises(AttributeError):
        terms.update({'注入词': 99})
    with pytest.raises(TypeError):
        pool._terms_by_id['注入id'] = Counter()
    with pytest.raises(TypeError):
        pool._latest_profiles_by_id[str(row['id'])]['expected_response_move'] = 'tampered'
    # 冻结不破坏正常只读语义（list/dict 身份与相等）。
    assert isinstance(row['context'], list) and isinstance(row['reply'], list)
    assert row['reply'] == [value for value in row['reply']]


def test_retrieve_returns_fully_thawed_plain_containers(tmp_path):
    """I10 返回边界：retrieve 返回递归解冻后的普通 list/dict，修改不回流污染共享池。"""
    pool_path = _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _synthetic_rows())
    session = _new_session(pool_path)
    request = dict(query='在吗', chat_name='某友', is_group=False, limit=4)
    rows = session.retrieve(**request)
    assert rows and all(type(row) is dict for row in rows)
    for row in rows:
        assert all(type(value) in (dict, list) or not isinstance(value, (dict, list))
                   for value in row.values())
        assert all(type(value) is dict for value in row.values() if isinstance(value, dict))
    tampered = session.retrieve(**request)
    tampered[0]['context'].append('随意修改')
    tampered[0]['reply'].clear()
    tampered[0].setdefault('semantic_profile', {})['incoming_act'] = 'tampered'
    expected = _new_session(pool_path).retrieve(**request)
    assert session.retrieve(**request) == expected


def test_returned_row_nested_mutation_does_not_leak_between_retrieves(tmp_path):
    """I10：两次 retrieve 之间经返回行修改嵌套 list/dict 不影响第二次结果。"""
    pool_path = _write_jsonl(tmp_path / 'plain' / 'pool.jsonl', _synthetic_rows())
    session = _new_session(pool_path)
    request = dict(query='在吗', chat_name='某友', is_group=False, limit=4)
    expected = session.retrieve(**request)
    assert expected
    tampered = session.retrieve(**request)
    tampered[0]['context'].append('污染嵌套列表')
    tampered[0]['reply'].clear()
    if 'semantic_profile' in tampered[0]:
        tampered[0]['semantic_profile']['incoming_act'] = 'tampered'
    again = session.retrieve(**request)
    assert again == expected
    # 新会话（缓存冷启动）与驻留缓存会话一致：惰性缓存不引入漂移。
    fresh = _new_session(pool_path)
    assert fresh.retrieve(**request) == expected


def test_open_session_second_open_hydrates_from_index_without_memory_load(tmp_path, monkeypatch):
    """I9：二次 open_session 走水合路径——不执行内存 _load（单遍 sqlite 恢复），
    水合状态与内存装载逐字节一致。"""
    fixture = _history_fixture(tmp_path)
    pool_path = fixture.directory / 'fewshot_pool.jsonl'
    first = open_session(pool_path, index_policy='build')
    assert isinstance(first._engine, SqliteIndexedRecallEngine)
    first.close()
    calls = []
    original_load = prepare.PreparedPool._load

    def spy(self):
        calls.append(self.path)
        return original_load(self)

    monkeypatch.setattr(prepare.PreparedPool, '_load', spy)
    second = open_session(pool_path, index_policy='use_if_present')
    try:
        assert isinstance(second._engine, SqliteIndexedRecallEngine)
        assert calls == []  # 水合路径：内存装载未被调用
        memory = prepare.PreparedPool.load(pool_path)
        assert [dict(row) for row in second.pool.rows] == [dict(row) for row in memory.rows]
        for name in disk_history.FEATURES:
            assert dict(getattr(second.pool, name)) == dict(getattr(memory, name))
        for name in disk_history.BUCKETS:
            assert ({key: [dict(row) for row in members]
                     for key, members in getattr(second.pool, name).items()}
                    == {key: [dict(row) for row in members]
                        for key, members in getattr(memory, name).items()})
        # 水合池逐题输出与内存池一致。
        memory_session = RecallSession(memory, InMemoryRecallEngine())
        for case in fixture.cases[:2]:
            for size in (3, 1):
                kwargs = _history_kwargs(case, size)
                assert second.retrieve(**kwargs) == memory_session.retrieve(**kwargs)
    finally:
        second.close()


def test_hydrated_pool_retrieve_matches_memory_pool(tmp_path):
    """I9：水合会话与内存会话逐题输出（含 trace/render）一致。"""
    fixture = _history_fixture(tmp_path)
    pool_path = fixture.directory / 'fewshot_pool.jsonl'
    open_session(pool_path, index_policy='build').close()
    hydrated = open_session(pool_path, index_policy='use_if_present')
    memory_session = open_session(pool_path, index_policy='use_if_present')
    # use_if_present 且索引已存在 → 水合；临时移除索引强制内存装载对照。
    assert isinstance(hydrated._engine, SqliteIndexedRecallEngine)
    cache = tmp_path / '.cache' / 'disk-history'
    saved = tmp_path / 'index-saved'
    saved.mkdir()
    for path in cache.iterdir():
        path.rename(saved / path.name)
    try:
        memory_session = open_session(pool_path, index_policy='use_if_present')
        assert isinstance(memory_session._engine, InMemoryRecallEngine)
    finally:
        for path in saved.iterdir():
            path.rename(cache / path.name)
    try:
        for case in fixture.cases:
            for size in (3, 1):
                kwargs = _history_kwargs(case, size)
                assert hydrated.retrieve(**kwargs) == memory_session.retrieve(**kwargs)
                assert hydrated.retrieve(**{**kwargs, 'include_retrieval_features': True}) == \
                    memory_session.retrieve(**{**kwargs, 'include_retrieval_features': True})
                rows = hydrated.retrieve(**kwargs)
                assert hydrated.render_selected(rows) == memory_session.render_selected(rows)
    finally:
        hydrated.close()
        memory_session.close()
