"""Review drafts must preserve uncertainty and bind quotes to the original sender."""
import json

import pytest

from src.bootstrap.wiki_aliases import alias_parts, build_review, write_review
from src.bootstrap.wiki_alias_context import authored_text, self_names, vocative_name
from src.bootstrap.wiki_alias_review import exclusion_reason, normalized_names, refine_review, write_refined
from src.bootstrap.wiki_content import compile_content, write_content


def fixture_sources(tmp_path, rows):
    wiki = tmp_path / 'wiki'
    (wiki / 'users').mkdir(parents=True)
    (wiki / 'groups').mkdir()
    messages = tmp_path / 'messages.jsonl'
    messages.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return wiki, messages


def message(mid, sender, sid, text, *, chat='group:account:g1', kind='text', quote=None):
    return dict(chat_id=chat, chat_type=chat.split(':')[0], chat_name='测试群',
        sender=sender, timestamp=100, text=text, message_id=mid, is_self=False,
        event=dict(id='platform:' + mid, sender_id=sid, kind=kind, **({'quote': quote} if quote else {})))


def test_source_annotations_do_not_hide_other_aliases():
    assert list(alias_parts('小蓝（来源：群/昨日）、Blue Fox（来源：其他群/今天）')) == ['小蓝', 'Blue Fox']


def test_quote_is_bound_to_original_author_and_exact_content(tmp_path):
    original = message('1', '蓝色', 'person-a', '周末见')
    reply = message('2', '绿色', 'person-b', '好的', kind='quote', quote={
        'replyToMessageId': '1', 'quotedSender': '小蓝', 'quotedContent': '周末见'})
    mismatch = message('3', '绿色', 'person-b', '你好', kind='quote', quote={
        'replyToMessageId': '1', 'quotedSender': '错误别名', 'quotedContent': '并非原文'})
    wiki, messages = fixture_sources(tmp_path, [original, reply, mismatch])
    review = build_review(wiki, None, messages)
    assert len(review['claims']) == 1
    claim = review['claims'][0]
    assert (claim['owner'], claim['alias'], claim['bound_identity']) == ('蓝色', '小蓝', 'account/person-a')
    assert claim['evidence'][0]['referenced_message']['message_id'] == '1'
    assert review['counts']['quote_content_mismatch'] == 1
    assert not review['runtime_usable']


def test_same_display_name_does_not_merge_accounts_or_people(tmp_path):
    wiki, messages = fixture_sources(tmp_path, [
        message('1', '蓝色', 'person-a', '早'), message('2', '蓝色', 'person-b', '晚'),
        message('3', '蓝色', 'person-a', '好', chat='group:another:g2')])
    (wiki / 'users' / 'blue.md').write_text('# 蓝色\n- 别名：小蓝\n')
    review = build_review(wiki, None, messages)
    assert len(review['identities']) == 3
    assert 'multiple_ids_do_not_merge' in review['claims'][0]['flags']


def test_uncertain_wiki_noise_and_notes_are_retained(tmp_path):
    wiki, messages = fixture_sources(tmp_path, [])
    (wiki / 'users' / 'a.md').write_text('# 蓝色\n- **别名**：小蓝（未确认）\n')
    (wiki / 'groups' / 'a.md').write_text('# 测试群\n- 群聊名称：2021-01-01 ~ 2021-02-01\n')
    (wiki / 'groups' / 'b.md').write_text('# 测试群\n')
    aliases = tmp_path / 'aliases.json'
    aliases.write_text(json.dumps({'users': {'蓝色': {'aliases': ['Blue'], 'notes': '可能是同一人'}}}))
    review = build_review(wiki, aliases, messages)
    assert len(review['duplicate_titles']) == 1
    assert all('uncertain_source' in c['flags'] for c in review['claims'] if c['kind'] == 'person')
    assert 'not_a_clean_alias' in next(c for c in review['claims'] if c['kind'] == 'group')['flags']


def test_intro_system_exclusion_and_review_only_output(tmp_path):
    wiki, messages = fixture_sources(tmp_path, [
        message('1', '蓝色', 'person-a', '我是群聊"测试群"的小蓝'),
        message('2', '系统', 'system-id', '你好', kind='system')])
    original = messages.read_bytes()
    review = build_review(wiki, None, messages, ['小蓝'])
    assert len(review['identities']) == 1
    assert review['claims'][0]['bound_identity'] == 'account/person-a'
    assert review['focus']['小蓝']['count'] == 1
    output = tmp_path / 'review'
    write_review(review, output)
    assert json.loads((output / 'review.json').read_text())['status'] == 'pending_user_review'
    assert messages.read_bytes() == original
    with pytest.raises(FileExistsError):
        write_review(review, output)


def test_context_names_separate_preferences_quotes_and_commands():
    assert list(self_names('可以叫我 蓝蓝 or 阿蓝')) == [('蓝蓝', 'self_stated'), ('阿蓝', 'self_stated')]
    assert list(self_names('嗯嗯，我叫蓝天，你可以叫我小蓝蓝[偷笑]')) == [('小蓝蓝', 'self_stated')]
    assert list(self_names('我是产品专家（大家叫我小蓝或者小天）')) == [('小蓝', 'reported'), ('小天', 'reported')]
    assert list(self_names('我不喜欢别人叫我蓝老板')) == [('蓝老板', 'disliked')]
    for text in ['我朋友叫我早睡！', '叫我update', '到了叫我', '我妈叫我来监工',
                 '大家都叫我买那个房子', '还好没叫我阿姨', '我们管家也老叫我姐',
                 '叫我报计算机', '叫我们早晚各一次', '叫我开窗', '叫我不要有所期待',
                 '叫我谢谢你哈哈哈', '叫我继续抱起来', '叫我提交凭证', '叫我干嘛',
                 '叫我爱豆签总发，他最近天天赚大钱']:
        assert not list(self_names(text))
    row = message('1', '绿', 'b', '好的[引用 蓝：可以叫我小蓝]', kind='quote')
    assert not list(self_names(authored_text(row)))
    row['text'] = '他说：“叫我小蓝”'
    assert not list(self_names(authored_text(row)))


def test_vocatives_and_literal_names_do_not_include_descriptions():
    assert vocative_name('谢谢蓝哥，下次见') == '蓝哥'
    assert vocative_name('好吧王老师，我认栽') == '王老师'
    assert vocative_name('哈哈王老师，您这串消息我看到了') == '王老师'
    for word in ['小红书', '老地方', '小米', '老多了', '小心', '阿里啊']:
        assert vocative_name(word) is None
    quoted = dict(alias='蓝色(Blue)/2', evidence=[dict(kind='resolved_quote_name')])
    assert list(normalized_names(quoted)) == [('蓝色(Blue)/2', [])]
    annotation = dict(alias='小蓝（未确认）', evidence=[dict(kind='wiki_claim')])
    assert list(normalized_names(annotation)) == [('小蓝', ['未确认'])]
    assert list(self_names('叫我艺涵大美女就行了')) == [('艺涵大美女', 'self_stated')]
    assert vocative_name('叫我小蓝就行了，绿总') == '绿总'
    for text in ['王总。。。', '王总', '还是老了', '蓝哥，', '老地方，见']:
        assert vocative_name(text) is None


def test_vocative_candidates_exclude_ordinary_nouns_and_predicates():
    for text in ['小区，楼，街道都没有封', '老房子，现在不行了', '各种生活费，阿姨费',
                 '嗯，我一般是大仓位，小波动', '小组长，算不上啥领导',
                 '小学老师，现在想想啊，有些素质差的', '高中老师！好厉害',
                 '搜噶，都是大哥', '等我去外企生了二胎，就去当老师',
                 '不是蓝总，只有一个', '说大老板，不是直接的老板']:
        assert vocative_name(text) is None
    assert vocative_name('收到，谢谢蓝哥') == '蓝哥'
    assert vocative_name('恭喜蓝总，入住新家') == '蓝总'


def test_wiki_metadata_is_not_an_alias_but_literal_names_are_preserved():
    wiki = dict(owner='蓝', alias='花名：Blue', evidence=[dict(kind='wiki_claim')])
    assert list(normalized_names(wiki)) == [('Blue', [])]
    for name in ['2024-01-14等多次出现', '某人的同行人', '群友称呼', '别名', '志愿者']:
        assert exclusion_reason(wiki, name)
    quoted = dict(owner='蓝', alias='某人的同行人', evidence=[dict(kind='resolved_quote_name')])
    assert not exclusion_reason(quoted, quoted['alias'])


def test_addresses_preserve_direction_and_do_not_turn_requests_or_echoes_into_usage(tmp_path):
    chat = 'private:account:other'
    rows = [message('1', '蓝', 'self', '绿总，你好', chat=chat),
            message('2', '绿', 'other', '蓝总，早', chat=chat),
            message('3', '蓝', 'self', '蓝总。。。', chat=chat),
            message('4', '蓝', 'self', '叫我小蓝就行了，绿总', chat=chat),
            message('5', '蓝', 'self', '我不喜欢别人叫我蓝老板', chat=chat),
            message('6', '蓝', 'self', '同事都叫我阿蓝', chat=chat),
            message('7', '蓝', 'self', '他说：“绿总，你好”', chat=chat),
            message('8', '蓝', 'self', '同事都叫我阿蓝，你可以叫我小蓝', chat=chat)]
    for row in rows:
        row['is_self'] = row['event']['sender_id'] == 'self'
    wiki, messages = fixture_sources(tmp_path, rows)
    source = tmp_path / 'original'
    write_review(build_review(wiki, None, messages), source)
    refined = refine_review(source / 'review.json', messages)
    edges = refined['context']['addresses']
    calls = [r for r in edges if r['usage'] == 'direct_address_candidate']
    assert {(r['alias'], r['direction'], r['count']) for r in calls} == {
        ('绿总', 'self_to_other', 2), ('蓝总', 'other_to_self', 1)}
    assert {r['usage'] for r in edges if r['alias'] == '小蓝'} == {'requested_name'}
    assert next(r['count'] for r in edges if r['alias'] == '小蓝') == 2
    assert {r['usage'] for r in edges if r['alias'] == '蓝老板'} == {'disliked_name'}
    assert {r['usage'] for r in edges if r['alias'] == '阿蓝'} == {'reported_name'}
    assert all(r['direction'] == 'naming_statement' for r in edges if r not in calls)
    assert all(not r['runtime_usable'] and r['historical_input_status'] == 'not_admitted' for r in edges)
    green = next(r for r in calls if r['alias'] == '绿总')
    assert green['source_identity'] == 'account/self'
    assert green['target_identity'] == 'account/other'
    assert green['binding'] == 'private_addressee_unconfirmed'
    assert green['evidence'][0]['is_self'] is True
    output = tmp_path / 'refined'
    write_refined(refined, output)
    assert '本人→对方' in (output / 'addresses.md').read_text()


def test_mixed_naming_statements_keep_each_clause_usage():
    assert list(self_names('同事都叫我阿蓝，你可以叫我小蓝')) == [
        ('阿蓝', 'reported'), ('小蓝', 'self_stated')]
    assert list(self_names('大家叫我小蓝，你可以叫我小蓝')) == [
        ('小蓝', 'reported'), ('小蓝', 'self_stated')]


def test_export_self_roles_are_not_merged_across_accounts(tmp_path):
    rows = [message('1', '本人', 'self-a', '蓝哥，你好', chat='private:account:a'),
            message('2', '蓝', 'a', '你好', chat='private:account:a'),
            message('3', '另一个账号', 'self-b', '绿哥，你好', chat='private:another:b'),
            message('4', '绿', 'b', '你好', chat='private:another:b')]
    rows[0]['is_self'] = rows[2]['is_self'] = True
    wiki, messages = fixture_sources(tmp_path, rows)
    source = tmp_path / 'original'
    write_review(build_review(wiki, None, messages), source)
    refined = refine_review(source / 'review.json', messages, self_account='account')
    edges = refined['context']['addresses']
    assert {(r['alias'], r['direction']) for r in edges} == {
        ('蓝哥', 'self_to_other'), ('绿哥', 'other_account')}
    other = next(r for r in edges if r['alias'] == '绿哥')
    assert other['source_is_self'] is None and other['source_export_is_self'] is True
    unspecified = refine_review(source / 'review.json', messages)
    assert all(r['direction'] == 'other_account' for r in unspecified['context']['addresses'])


def test_confirmed_alias_use_is_speaker_specific_scoped_and_not_backdated(tmp_path):
    chat = 'group:account:g1'
    rows = [message('1', '蓝', 'a', '我是群聊"测试群"的Blue'),
            message('2', '本人', 'self', '分享蓝哥的账号'),
            message('3', '绿', 'b', '蓝弟弟好厉害'),
            message('4', '本人', 'self', '分享蓝哥的账号', chat='group:account:g2')]
    rows[1]['is_self'] = rows[3]['is_self'] = True
    wiki, messages = fixture_sources(tmp_path, rows)
    source = tmp_path / 'original'
    write_review(build_review(wiki, None, messages), source)
    decision = dict(decisions=[dict(id='confirmed-now', canonical_name='蓝', person_id='person-a',
        aliases=['蓝哥', '蓝弟弟'], status='user_confirmed', observed_identity_records=['account/a'],
        scope=dict(chat_id=chat, chat_name='测试群'), source=dict(answer='确认蓝这一组'))])
    decisions = source / 'decisions.json'
    decisions.write_text(json.dumps(decision))
    refined = refine_review(source / 'review.json', messages, decisions)
    edges = refined['context']['addresses']
    assert {(r['source_identity'], r['alias'], r['direction']) for r in edges} == {
        ('account/self', '蓝哥', 'self_to_other'), ('account/b', '蓝弟弟', 'other_to_other')}
    assert all(r['usage'] == 'scoped_name_use' and r['chat_id'] == chat for r in edges)
    assert all(r['identity_decision_id'] == 'confirmed-now' for r in edges)
    assert all(r['historical_input_status'] == 'not_admitted' for r in edges)


def test_refinement_cleanup_context_and_preserve_confirmation(tmp_path):
    chat = 'private:account:a'
    wiki, messages = fixture_sources(tmp_path, [
        message('1', '蓝', 'a', '可以叫我 小蓝 or 阿蓝', chat=chat),
        message('2', '绿', 'b', '蓝哥，你好', chat=chat),
        message('3', '绿', 'b', '蓝哥，明天见', chat=chat),
        message('4', '绿', 'b', '小绿，明天见', chat=chat),
        message('5', '同名', 'c', '我是群聊"测试群"的2501'),
        message('6', '同名', 'd', '我是群聊"测试群"的2501')])
    (wiki / 'users' / 'blue.md').write_text('# 蓝\n- 别名：小蓝、阿蓝[待验证]\n')
    (wiki / 'groups' / 'blue.md').write_text('# 测试群\n- **蓝**（别名：小蓝）：说明\n')
    (wiki / 'users' / 'noise.md').write_text('# 09:55\n- 别名：滚动摘要\n')
    (wiki / 'users' / 'noise2.md').write_text('# 绿\n- 别名：（暂无、后被某人@为小绿\n')
    original = build_review(wiki, None, messages)
    source = tmp_path / 'original'
    write_review(original, source)
    decision = dict(decisions=[dict(canonical_name='蓝', aliases=['小蓝'],
        scope=dict(chat_name='测试群'), source=dict(answer='确认小蓝，其余待定'))])
    decisions = source / 'decisions.json'
    decisions.write_text(json.dumps(decision))
    refined = refine_review(source / 'review.json', messages, decisions)
    assert refined['decisions'] == decision
    assert not refined['runtime_usable']
    assert '09:55' not in refined['people']
    blue = refined['people']['蓝']['aliases']
    assert len(blue['小蓝']['assertions']) == 3
    assert blue['阿蓝']['notes'] == ['待验证']
    assert not blue['阿蓝']['new_context_pair']
    vocative = blue['蓝哥']['assertions'][0]
    assert vocative['suggested_identity'] == 'account/a'
    assert 'bound_identity' not in vocative
    assert vocative['evidence_count'] == 2
    assert len(refined['context']['weak_candidates']) == 1
    # Numeric room nicknames with direct evidence remain, without merging same-name accounts.
    assertions = refined['people']['同名']['aliases']['2501']['assertions']
    assert {a['bound_identity'] for a in assertions} == {'account/c', 'account/d'}
    output = tmp_path / 'refined'
    write_refined(refined, output)
    assert '其余对应关系均待确认' in (output / 'README.md').read_text()
    assert '新增对应关系' in (output / 'context.md').read_text()
    assert '小蓝' in (output / 'CONFIRM.md').read_text()
    assert json.loads((source / 'review.json').read_text()) == original
    with pytest.raises(FileExistsError):
        write_refined(refined, output)
    messages.write_text(messages.read_text() + json.dumps(message('7', '新人', 'z', '来了')) + '\n')
    with pytest.raises(ValueError, match='历史文件与原清单不一致'):
        refine_review(source / 'review.json', messages, decisions)


def content_fixture(tmp_path):
    _, messages = fixture_sources(tmp_path, [message('m1', '蓝', 'a', '梦见绿是同事')])
    review = tmp_path / 'refined.json'
    review.write_text(json.dumps(dict(decisions=dict(decisions=[], pending_groups=[
        dict(names=['绿', 'Green'], action='keep_accounts_unmerged')]), context=dict(addresses=[]))))
    plan = dict(recorded_on='2026-09-23', identities=[], address_ids=[],
        pages=[dict(id='p1', slug='blue', title='蓝', summary='来源支持的内容', unknowns=['职业未知'])],
        evidence_sources=dict(dream=dict(kind='raw_message', message_id='m1'),
            correction=dict(kind='current_conversation', source_ref='user:1', recorded_on='2026-09-23', text='这是梦')),
        fact_claims=[dict(id='c1', title='不建立同事关系', subject_id='p1', predicate='dream',
            value='梦见绿是同事', page_ids=['p1'], evidence_refs=['dream', 'correction'],
            claim_type='原文观察', status='梦境', description='只描述梦境', validity_note='仅该次聊天')])
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    return plan, plan_path, messages, review


def test_wiki_content_preserves_sources_pending_identities_and_current_knowledge_time(tmp_path):
    _, plan_path, messages, review = content_fixture(tmp_path)
    original = messages.read_bytes()
    content = compile_content(plan_path, messages, review)
    claim = content['fact_claims'][0]
    assert claim['observed_at'] == [100]
    assert claim['known_at'] == '2026-09-23'
    assert claim['historical_input_status'] == 'not_admitted'
    assert claim['valid_from'] is None
    assert content['evidence']['dream']['speaker_id'] == 'a'
    assert content['evidence']['dream']['message_id'] == 'm1'
    assert content['evidence']['dream']['line'] == 1
    assert 'message' not in content['evidence']['dream']
    assert 'text' not in content['evidence']['correction']
    assert content['provenance_policy'] == 'locators_only'
    assert content['pending_identity_groups'][0]['action'] == 'keep_accounts_unmerged'
    output = tmp_path / 'content'
    write_content(content, output)
    assert '不建立同事关系' in (output / 'blue.md').read_text()
    assert '不合并账号' in (output / 'README.md').read_text()
    assert '梦见绿是同事' not in (output / 'sources.md').read_text()
    assert str(messages.resolve()) in (output / 'sources.md').read_text()
    assert json.loads((output / 'knowledge.json').read_text())['runtime_usable'] is False
    assert messages.read_bytes() == original
    with pytest.raises(FileExistsError):
        write_content(content, output)


def test_wiki_sources_do_not_copy_bodies_in_nested_provenance(tmp_path):
    plan, plan_path, messages, review_path = content_fixture(tmp_path)
    body, quote, context, wiki_body, answer = [f'PRIVATE_SOURCE_{n}' for n in range(5)]
    row = message('m1', '蓝', 'a', body, quote={'quotedContent': quote})
    messages.write_text(json.dumps(row))
    wiki = tmp_path / 'old.md'
    wiki.write_text('标题\n' + wiki_body + '\n')
    plan['evidence_sources']['wiki'] = dict(kind='source_wiki', path=str(wiki), excerpt=wiki_body)
    plan['evidence_sources']['dream']['unrecognized_body'] = quote
    plan['evidence_sources']['correction']['text'] = answer
    plan['identities'] = [dict(id='p1', status='user_confirmed', name='蓝', description='账号身份')]
    plan['address_ids'] = ['a1']
    plan_path.write_text(json.dumps(plan))
    review = dict(decisions=dict(decisions=[dict(id='d1', person_id='p1', status='user_confirmed',
        aliases=['小蓝'], source=dict(answer=answer), scope=dict(rule='仅本群'),
        historical_support=[dict(message_id='m1', timestamp=100, text=body)])]),
        context=dict(addresses=[dict(id='a1', source_name='甲', target_name='蓝', alias='小蓝',
            direction='self_to_other', count=1, evidence=[dict(message_id='m1', path=str(messages),
                line=1, text=body, event=row['event'],
                context_before=[dict(message_id='m0', timestamp=99, text=context)])])]))
    review_path.write_text(json.dumps(review))
    content = compile_content(plan_path, messages, review_path)
    source = content['evidence']['wiki']
    assert source['line'] == 2 and source['end_line'] == 2
    assert source['excerpt_sha256'] and source['sha256']
    decision = content['identities'][0]['decision']
    assert decision['source']['json_pointer'] == '/decisions/decisions/0'
    assert decision['historical_support'] == [dict(message_id='m1', timestamp=100)]
    assert content['address_edges'][0]['evidence'][0]['context_before'] == [
        dict(message_id='m0', timestamp=99)]
    output = tmp_path / 'content'
    write_content(content, output)
    exported = '\n'.join(p.read_text() for p in output.iterdir())
    assert all(secret not in exported for secret in (body, quote, context, wiki_body, answer))
    assert content['runtime_usable'] is False


@pytest.mark.parametrize('error', ['missing_message', 'missing_ref', 'unconfirmed_identity', 'changed_wiki'])
def test_wiki_content_rejects_missing_provenance_and_unapproved_identity_merge(tmp_path, error):
    plan, plan_path, messages, review = content_fixture(tmp_path)
    if error == 'missing_message':
        plan['evidence_sources']['dream']['message_id'] = 'absent'
    elif error == 'missing_ref':
        plan['fact_claims'][0]['evidence_refs'] = ['absent']
    elif error == 'unconfirmed_identity':
        plan['identities'] = [dict(id='p1', status='user_confirmed')]
    else:
        wiki = tmp_path / 'changed.md'
        wiki.write_text('修改后的摘要')
        plan['evidence_sources']['wiki'] = dict(kind='source_wiki', path=str(wiki), excerpt='原来的摘要')
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        compile_content(plan_path, messages, review)
