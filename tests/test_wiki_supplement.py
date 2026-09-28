"""Grounded gap completion: identity, raw selection, scope records and recovery."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.bootstrap import wiki_batch, wiki_prompt, wiki_scope as scope, wiki_source_index as sources
from src.bootstrap import wiki_supplement as supplement
from src.config import sha256_file
from src.iteration.storage import read_json
from test_wiki_repair import row, source


def indexed(tmp_path, rows):
    messages = source(tmp_path, rows)
    path = tmp_path / 'index.sqlite3'
    sha = sha256_file(messages)
    sources.build(messages, path, 'self', sha)
    return sources.SourceIndex(path, messages, sha)


def test_topic_anchors_require_raw_author_and_bounded_same_exporter_context(tmp_path):
    text = '一种有详细说明的测试运动方式，保留完整原始文本。'
    rows = [dict(row('group:self:room', 'peer', 1000, text), sender='甲', chat_name='运动群'),
            dict(row('group:self:room', 'other', 1100, '附近的上下文'), chat_name='运动群'),
            dict(row('group:self:room', 'peer', 2000, '时间太远'), chat_name='运动群'),
            dict(row('group:other:room', 'peer', 1000, text), sender='甲', chat_name='运动群'),
            dict(row('group:self:else', 'peer', 1000, text), sender='甲', chat_name='另一个群')]
    index = indexed(tmp_path, rows)
    try:
        selection = sources.select_topic(index, ['# 运动', '- [1970-01-01] 甲 @ 运动群: ', text[:20]])
        assert selection['lines'] == [1, 2]
        assert selection['anchors'][0]['source']['line'] == 1
        assert selection['unmatched_old_lines'] == []
        assert text not in json.dumps(selection, ensure_ascii=False)
        assert text.encode() not in (tmp_path / 'index.sqlite3').read_bytes()
        wrong_author = sources.select_topic(index, [f'- [1970-01-01] 乙 @ 运动群: {text}'])
        assert wrong_author['lines'] == [] and wrong_author['unmatched_old_lines'] == [1]
    finally:
        index.close()


def test_ambiguous_person_needs_distinct_dated_utterances_and_never_uses_quoted_author(tmp_path):
    a, b = '这是一段独特的原始发言', '另一个明确的本人原话'
    rows = [dict(row('private:self:p1', 'p1', 1000, a), sender='同名'),
            dict(row('private:self:p1', 'p1', 1001, b), sender='同名'),
            dict(row('private:self:p2', 'p2', 1002, '收到[引用 同名：' + a + ']'), sender='同名')]
    index = indexed(tmp_path, rows)
    lines = ['## 说过的话', f'- “{a}”（来源：私聊/1970-01-01）', f'- “{b}”（来源：私聊/1970-01-01）']
    try:
        proof = sources.resolve_person(index, lines, ['p1', 'p2'])
        assert proof['account'] == 'p1'
        assert [e['line'] for e in proof['evidence']] == [1, 2]
        assert 'text' not in proof['evidence'][0]
        repeated = sources.resolve_person(index, [lines[0], lines[1], lines[1]], ['p1', 'p2'])
        assert repeated['account'] == ''
    finally:
        index.close()


def test_fused_person_page_stays_unresolved(tmp_path):
    a, b = '第一位用户的独特发言', '第二位用户的独特发言'
    index = indexed(tmp_path, [row('private:self:p1', 'p1', 1000, a), row('private:self:p2', 'p2', 1001, b)])
    try:
        proof = sources.resolve_person(index, ['## 说过的话', f'- “{a}”（来源：私聊/1970-01-01）',
            f'- “{b}”（来源：私聊/1970-01-01）'], ['p1', 'p2'])
        assert proof['account'] == '' and proof['binding'] == 'mixed_raw_accounts'
        assert proof['supported_accounts'] == ['p1', 'p2']
    finally:
        index.close()


def test_shared_utterance_keeps_page_ambiguous_despite_other_unique_anchors(tmp_path):
    texts = ['第一段明确的本人发言', '第二段明确的本人发言', '多人都说过的同一句话']
    rows = [row('private:self:p1', 'p1', 1000 + i, text) for i, text in enumerate(texts)]
    rows.append(row('private:self:p2', 'p2', 1005, texts[-1]))
    index = indexed(tmp_path, rows)
    try:
        proof = sources.resolve_person(index, ['## 说过的话'] + [
            f'- “{text}”（来源：私聊/1970-01-01）' for text in texts], ['p1', 'p2'])
        assert proof['account'] == '' and proof['binding'] == 'ambiguous_utterance_authorship'
        assert proof['ambiguous_old_lines'] == [4]
        assert {e['line'] for e in proof['evidence']} == {1, 2, 3, 4}
    finally:
        index.close()


def test_group_exact_id_overrides_names_and_other_exporters_are_excluded(tmp_path):
    rows = [dict(row('group:self:123@chatroom', 'peer', 1000), chat_name='同名群'),
            dict(row('group:self:456@chatroom', 'peer', 1000), chat_name='同名群'),
            dict(row('group:other:123@chatroom', 'peer', 1000), chat_name='同名群')]
    index = indexed(tmp_path, rows)
    try:
        exact = sources.select_group(index, ['- 群聊 ID：123@chatroom'], '同名群', ['123_chatroom'])
        assert exact['chat_id'] == 'group:self:123@chatroom' and exact['selected_rows'] == 1
        ambiguous = sources.select_group(index, [], '同名群', [])
        assert ambiguous['binding'] == 'ambiguous_group'
        assert len(ambiguous['candidates']) == 2
    finally:
        index.close()


class ScopeClient:
    def __init__(self):
        self.prompts = []

    def cache_identity(self):
        return {'model': 'synthetic-scope'}

    def run(self, prompt, schema):
        self.prompts.append(prompt)
        if prompt.startswith(scope.EXTRACT_PROMPT):
            data = json.loads(prompt[len(scope.EXTRACT_PROMPT):])
            return json.dumps(dict(facts=[dict(category='group_topic', subject='群', speaker_id='peer',
                summary='成员讨论运动安排。', time_scope='当时', mode='observed',
                lines=[data['messages'][0][0]], old_lines=[2])]))
        data = json.loads(prompt[len(scope.STRUCTURE_PROMPT):])
        meta = data['scope']
        common = dict(time_scope='当时', valid_from='', valid_to='', temporal_kind='historical',
            mode='observed', polarity='affirmed', lines=[data['messages'][0][0]],
            fact_ids=[data['facts'][0]['id']], attribution='消息中成员讨论的主题。')
        return json.dumps(dict(entities=[dict(ref='S0', kind=meta['kind'], label=meta['name'],
            account='', lines=common['lines']), dict(ref='E2', kind='person', label='同名者',
            account='', lines=common['lines'])],
            attributes=[dict(common, subject_ref='S0', field='discussion', value='运动安排')],
            relations=[], events=[], addresses=[],
            reviews=[dict(fact_id=data['facts'][0]['id'], disposition='represented', reason='有原始证据')]))


def test_scope_two_pass_cache_nonperson_root_and_locator_only_exports(tmp_path):
    index = indexed(tmp_path, [row('group:self:room', 'peer', 1000, 'RAW_SCOPE_MESSAGE')])
    wiki = tmp_path / 'scope.md'
    wiki.write_text('# 运动群\n- 运动安排\n')
    job = dict(id='page', title='运动群', kind='conversation', chat_id='group:self:room',
        self_account='self', wiki=dict(path=str(wiki)), sources=[dict(path=str(wiki), sha256=sha256_file(wiki))])
    output = tmp_path / 'out'
    client = ScopeClient()
    (tmp_path / 'review.json').write_text('FORBIDDEN_HUMAN_REVIEW')
    try:
        rows = index.rows(chat_id=job['chat_id'])
        coverage = dict(path=index.path, sha256=index.sha256)
        assert scope.generate(job, rows, coverage, output, {}, client=client)['stage'] == 'complete'
        assert len(client.prompts) == 2
        assert all('RAW_SCOPE_MESSAGE' in p and 'FORBIDDEN_HUMAN_REVIEW' not in p for p in client.prompts)
        data = read_json(output / 'content/knowledge.json')
        root = next(e for e in data['entities'] if e['id'] == data['subject']['entity_id'])
        assert root['kind'] == 'group' and not root['account']
        assert data['attributes'][0]['subject_id'] == root['id']
        assert 'RAW_SCOPE_MESSAGE' not in json.dumps(data)
        assert data['coverage']['legacy_lines_without_supported_fact'] == []
        assert not data['runtime_usable']
        assert wiki_prompt.export(output / 'content/knowledge.json', tmp_path / 'prompt')['records'] == 1
        scope.generate(job, rows, coverage, output, {}, client=client)
        assert len(client.prompts) == 2
        facts = read_json(output / 'facts.json')
        meta = data['subject']
        value, mapping = scope.structure(client, output / 'cache', meta, facts, rows)
        merged = scope.compile_records(meta, [(value, mapping), (value, mapping)], rows, coverage)
        assert sum(e['kind'] == 'group' for e in merged['entities']) == 1
        assert sum(e['kind'] == 'person' for e in merged['entities']) == 2
        assert len(merged['attributes']) == 1
        invalid = deepcopy(value)
        invalid['entities'][0]['kind'] = 'person'
        with pytest.raises(ValueError, match='S0'):
            scope.validate_structure(invalid, mapping, rows, meta)
    finally:
        index.close()


def test_supplement_resume_and_frozen_source_checks(tmp_path, monkeypatch):
    library = tmp_path / 'wiki'
    (library / 'groups').mkdir(parents=True)
    (library / 'topics').mkdir()
    (library / 'groups/123@chatroom.md').write_text('# 群\n- 群聊 ID：123@chatroom\n')
    (library / 'topics/运动.md').write_text('# 运动\n- [1970-01-01] peer @ 群: 原始消息中的运动安排\n')
    messages = source(tmp_path, [dict(row('group:self:123@chatroom', 'peer', 1000, '原始消息中的运动安排'), chat_name='群')])
    parent, output = tmp_path / 'parent', tmp_path / 'supplement'
    wiki_batch.prepare(library, messages, parent, 'self', {})
    summary = supplement.prepare(parent, output)
    assert summary['jobs'] == 2 and summary['kinds'] == {'conversation': 1, 'topic': 1}
    calls = []
    real_generate = scope.generate
    def fake(*args, **kwargs):
        calls.append(args[0]['id'])
        if len(calls) == 1:
            return dict(stage='incomplete', errors=['temporary failure'])
        return real_generate(*args, **kwargs, client=ScopeClient())
    monkeypatch.setattr(scope, 'generate', fake)
    assert supplement.run(output, max_jobs=1)['counts']['complete'] == 1
    assert supplement.run(output)['stage'] == 'complete'
    assert len(calls) == 3
    supplement.run(output)
    assert len(calls) == 3
    assert wiki_batch.status(output)['counts']['complete'] == 2
    messages.write_text(messages.read_text() + '\n')
    with pytest.raises(ValueError, match='messages changed'):
        supplement.run(output)


def test_shared_document_keeps_content_subject_and_publisher_roles_in_cards(tmp_path):
    index = indexed(tmp_path, [row('group:self:room', 'peer', 1000,
        '转发一份设备维护说明：每月检查一次滤网。')])
    meta = dict(name='资料群', kind='group', scope_id='group:self:room', account='', self_account='self')
    common = dict(time_scope='当时', valid_from='', valid_to='', temporal_kind='historical',
        mode='reported', polarity='affirmed', lines=[1], fact_ids=['F1'],
        attribution='成员转发的设备说明包含维护建议，未证明其亲自执行。')
    value = dict(entities=[
        dict(ref='S0', kind='group', label='资料群', account='', lines=[1]),
        dict(ref='E2', kind='person', label='发布者', account='peer', lines=[1]),
        dict(ref='E3', kind='object', label='设备维护说明', account='', lines=[1])],
        attributes=[dict(common, subject_ref='E3', field='maintenance_interval', value='每月检查滤网')],
        relations=[], addresses=[], events=[dict(common, mode='observed', event_type='分享资料',
            description='成员在群内转发设备维护说明。', status='completed', details=[],
            participants=[dict(entity_ref='E2', role='发布者'), dict(entity_ref='E3', role='传播内容'),
                          dict(entity_ref='S0', role='发布群聊')])],
        reviews=[dict(fact_id='F1', disposition='represented', reason='传播行为及资料内容有原始依据')])
    try:
        rows, mapping = index.rows(chat_id=meta['scope_id']), {'F1': 'fact-1'}
        scope.validate_structure(value, mapping, rows, meta)
        data = scope.compile_records(meta, [(value, mapping)], rows, dict(path=index.path, sha256=index.sha256))
        cards = list(wiki_prompt.packets(data))
        facts = [fact for card in cards for fact in card['facts']]
        entities = {e['id']: e for card in cards for e in card['identities']}
        attribute = next(f for f in facts if f['kind'] == 'attribute')
        assert entities[attribute['subject_id']]['kind'] == 'object'
        assert not entities[attribute['subject_id']]['account']
        assert attribute['mode'] == 'reported'
        event = next(f for f in facts if f['kind'] == 'event')
        roles = {p['role']: entities[p['entity_id']] for p in event['participants']}
        assert roles['发布者']['account'] == 'peer'
        assert roles['传播内容']['id'] == attribute['subject_id']
        assert roles['发布群聊']['id'] == cards[0]['subject_id']
        assert event['mode'] == 'observed'
    finally:
        index.close()
