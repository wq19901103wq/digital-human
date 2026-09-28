"""Source identities never broaden the historical generator input boundary."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.config import ConfigError, load_settings
from src.generator.generator import ReplyGenerator
from src.generator.history_sources import Message
from src.generator.source_projection import POLICY, project_validated, validate_policy
from test_leakage_guards import env as env


def row(name, text, *, account=None, self_=False, platform=None, quote=None):
    event = {'sender_id': account, 'id': 'platform:' + str(platform) if platform else 'fallback:1'}
    if quote is not None:
        event['quote'] = quote
    return Message(name, text, self_, 1, 'message-id', 'group', '群', 'chat', event)


def project(rows, *, before=(), after=()):
    source = SimpleNamespace(chats={'chat': [*before, *rows, *after]})
    case = {'source_span': {'chat_id': 'chat', 'start': len(before),
                           'reply_start': len(before) + len(rows)},
            'human_reply': ['不能进入输入的答案'], 'labels': ['不能进入输入的评分']}
    saved = deepcopy(case)
    result = project_validated(case, source)
    assert case == saved
    assert '不能进入输入' not in result.history + result.identity_rule + result.unread
    return result


def test_account_identity_beats_names_and_preserves_absent_self():
    result = project([
        row('同名', '第一位', account='account-a'),
        row('同名', '第二位', account='account-b'),
        row('改名了', '仍是第一位', account='account-a'),
    ])
    assert '[M1 | P2] 同名: 第一位' in result.history
    assert '[M2 | P3] 同名: 第二位' in result.history
    assert '[M3 | P2] 改名了: 仍是第一位' in result.history
    assert 'P1：本人，本次回复者；当前片段无本人发言' in result.identity_rule
    assert '["同名", "改名了"]' in result.identity_rule
    assert 'account-a' not in str(result) and 'account-b' not in str(result)
    assert result.stats['known_account_messages'] == 3


def test_unknown_accounts_are_not_name_aliases_and_self_is_typed():
    result = project([
        row('本人', '同名不是本人'),
        row('本人', '来源仍不明', account='unknown'),
        row('本人', '字符串不代表本人', self_='true'),
        row('本人', '数字不代表本人', self_=1),
        row('新显示名', '可靠的本人标志', self_=True),
    ])
    for number in range(1, 5):
        assert f'[M{number} | P{number + 1}]' in result.history
    assert '[M5 | P1] 新显示名: 可靠的本人标志' in result.history
    assert result.stats['unknown_account_messages'] == 5


def test_quote_links_only_exact_unique_earlier_platform_id_in_input_span():
    result = project([
        row('现在的名字', '原话', account='account-a', platform='10'),
        row('回复者', '[引用 旧显示名：原话] 好', account='account-b', platform='11',
            quote={'replyToMessageId': 10, 'quotedSender': '旧显示名',
                   'quotedContent': '不得额外注入的元数据文本'}),
        row('回复者', '片段之外的引用', account='account-b', platform='12',
            quote={'replyToMessageId': '9'}),
        row('回复者', '引用未来', account='account-b', platform='13',
            quote={'replyToMessageId': '14'}),
        row('另一位', '未来消息', account='account-c', platform='14'),
        row('重复记录', '重复甲', platform='15'),
        row('重复记录', '重复乙', platform='15'),
        row('回复者', '目标重复', platform='16', quote={'replyToMessageId': '15'}),
    ], before=[row('之前', '片段外内容', platform='9')],
       after=[row('本人', '未来真人答案', self_=True, platform='20')])
    assert '[M2 | P3 | 引用 M1（P2）] 回复者: [引用 旧显示名：原话] 好' in result.history
    assert result.stats['quote_messages'] == 4 and result.stats['linked_quotes'] == 1
    assert result.history.count(' | 引用 ') == 1
    for hidden in ('片段外内容', '未来真人答案', '不得额外注入的元数据文本'):
        assert hidden not in result.history + result.identity_rule


def test_semantic_subjects_mentions_and_conditions_remain_original_text():
    rows = [row('本人', '他就喜欢这样说\n我又考砸了', self_=True, platform='1'),
            row('朋友', '@本人 18 点来吗', account='friend', platform='2'),
            row('本人', '18 点来不了，12 点可以', self_=True, platform='3'),
            row('朋友', '那就 12 点', account='friend', platform='4')]
    result = project(rows)
    assert result.history == '\n'.join(
        f'[M{i + 1} | {"P1" if m.is_self else "P2"}] {m.sender}: {m.text}'
        for i, m in enumerate(rows))
    assert result.unread == '[M4 | P2] 朋友: 那就 12 点'
    assert '不直接判定引用、转述或模仿中的事实主体' in result.identity_rule
    assert result.stats['linked_quotes'] == 0


@pytest.mark.parametrize('policy', [True, False, '', 'unknown', 'source_participants_v2'])
def test_unknown_policy_fails_closed(policy):
    with pytest.raises(ConfigError, match='context_projection'):
        validate_policy(policy)


def test_real_source_guard_precedes_projection_and_no_answer_enters_request(env):
    case = env.source.roles['development'][0]
    generator = ReplyGenerator(load_settings(), {**env.cfg, 'context_projection': POLICY},
        env.client, prompt_root=env.gdir, pool_path=env.source.directory / 'fewshot_pool.jsonl')
    messages = generator.build_prompt(case)
    content = messages[1]['content']
    assert '<source_participants>' in content
    assert 'P1：本人，本次回复者；当前片段无本人发言' in content
    assert '[M1 | P2]' in content
    for text in case['human_reply']:
        assert text not in content
    forged = deepcopy(case)
    forged['context'][0]['is_self'] = True
    with pytest.raises(ConfigError, match='身份|来源'):
        generator.build_prompt(forged)
    assert env.calls == []


def test_projection_cannot_run_without_sources_or_mix_legacy_identity(env):
    with pytest.raises(ConfigError, match='context_projection'):
        ReplyGenerator(load_settings(), {'context_projection': POLICY},
            env.client, prompt_root=env.gdir)
    with pytest.raises(ConfigError, match='exclusive rendering'):
        ReplyGenerator(load_settings(), {**env.cfg, 'context_projection': POLICY,
            'context_identity': 'explicit-self-v1'}, env.client, prompt_root=env.gdir,
            pool_path=env.source.directory / 'fewshot_pool.jsonl')


def test_candidate_request_is_distinct_and_resume_reuses_it(env):
    from src import tracing
    case = env.source.roles['development'][0]
    for name, policy in [('baseline', None), ('candidate', POLICY), ('resume', POLICY)]:
        generator = ReplyGenerator(load_settings(), {**env.cfg, 'context_projection': policy},
            env.client, prompt_root=env.gdir, pool_path=env.source.directory / 'fewshot_pool.jsonl')
        trace = tracing.CaseTrace(env.root / 'experiments' / name, case, {'dataset': 'development'})
        with trace.operation('generation', 'candidate', 0, {'forced_reply': True}) as operation:
            operation['result'] = generator.generate(case)
        trace.finish('ok')
    assert len(env.calls) == 2
    assert '<source_participants>' not in env.calls[0][1][1]['content']
    assert '<source_participants>' in env.calls[1][1][1]['content']
