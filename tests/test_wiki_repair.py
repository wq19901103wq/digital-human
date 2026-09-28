import json
from pathlib import Path

import pytest

from src.bootstrap import wiki_repair as repair


def row(chat, sender, timestamp, text='raw history', kind='text'):
    return dict(chat_id=chat, chat_type=chat.split(':')[0], sender=sender, is_self=sender == 'self',
                timestamp=timestamp, text=text, event=dict(sender_id=sender, kind=kind),
                message_id=f'{chat}:{timestamp}')


def source(tmp_path, rows):
    path = tmp_path / 'messages.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def test_selection_uses_account_and_separates_chat_and_exporter(tmp_path):
    path = source(tmp_path, [row('private:self:person', 'self', 1000),
        row('private:self:person', 'person', 1100), row('private:self:other', 'other', 1100),
        row('group:self:room', 'other', 1000), row('group:self:room', 'person', 1100),
        row('group:self:room', 'other', 9000), row('group:someone:room', 'person', 1100),
        row('group:self:unrelated', 'other', 1100)])
    selected, coverage = repair.select_history(path, 'person', 'self')
    assert [r['line'] for r in selected] == [4, 5, 1, 2]
    assert coverage['private_rows'] == 2 and coverage['group_rows'] == 2
    assert not coverage['nickname_only_group_mentions_included']


def test_batches_cover_messages_without_truncation_or_cross_chat_overlap(tmp_path):
    path = source(tmp_path, [row('private:self:person', 'person', 1000+i, 'x'*300) for i in range(20)]
                  + [row('group:self:room', 'person', 1200)])
    selected, _ = repair.select_history(path, 'person', 'self')
    batches = repair.make_batches(selected, max_chars=2000, overlap=2)
    assert {r['line'] for batch in batches for r in batch} == set(range(1, 22))
    assert all(len({r['chat_id'] for r in batch}) == 1 for batch in batches)
    assert all(r['text'] == 'x'*300 for batch in batches for r in batch if r['line'] <= 20)


class Client:
    def __init__(self):
        self.prompts = []

    def cache_identity(self):
        return {'provider': 'test', 'model': 'isolated'}

    def run(self, prompt, schema):
        self.prompts.append(prompt)
        if repair.RETRY_MARKER in prompt:
            prompt = prompt.split(repair.RETRY_MARKER, 1)[1]
        prompt = prompt.replace(repair.MENTION_PROMPT, '')
        if prompt.startswith(repair.CONSOLIDATE_PROMPT):
            data = json.loads(prompt[len(repair.CONSOLIDATE_PROMPT):])
            return json.dumps(dict(items=[dict(category=k, **item)
                for k, items in data['sections'].items() for item in items]))
        if prompt.startswith(repair.REREAD_PROMPT):
            data = json.loads(prompt[len(repair.REREAD_PROMPT):])
            assert '有日期的工作经历。' in data['draft_wiki']
            return json.dumps(dict(revisions=[dict(fact_id=f['id'], action='retain',
                reason='原聊天仍支持。', replacements=[]) for f in data['initial_facts']], additions=[]))
        if prompt.startswith(repair.EXTRACT_PROMPT):
            data = json.loads(prompt[len(repair.EXTRACT_PROMPT):])
            fact = dict(category='background', subject='person', speaker_id='person',
                summary='在消息时期就职于一家机构。', time_scope='historical', mode='self_report',
                lines=[data['messages'][0][0]], old_lines=[3])
            return json.dumps({'facts': [fact]})
        data = json.loads(prompt[len(repair.SECTION_PROMPT):])
        ids = [f['id'] for f in data['facts']]
        return json.dumps(dict(items=[dict(text='有日期的工作经历。', fact_ids=ids)] if ids else [],
            changes=[dict(old_line=line, action='correct', reason='原始聊天支持修订。', fact_ids=ids)
                     for line in data['review_lines']]))


def test_source_only_cached_generation_and_locator_only_export(tmp_path):
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000, 'RAW_PRIVATE_TEXT')])
    (tmp_path / 'review.json').write_text('FORBIDDEN_CURATED_CASE_CORRECTION')
    client = Client()
    output = tmp_path / 'output'
    result = repair.repair(wiki, messages, output, 'person', 'self', {}, client=client, workers=2)
    assert result['stage'] == 'complete'
    # Unchanged final sections are satisfied by the first draft's exact-input cache.
    assert len(client.prompts) == 11
    assert result['reread_completed'] == 1
    assert all('FORBIDDEN_CURATED_CASE_CORRECTION' not in p for p in client.prompts)
    assert 'RAW_PRIVATE_TEXT' in client.prompts[0]
    knowledge = json.loads((output / 'content/knowledge.json').read_text())
    assert knowledge['evidence']['1']['message_id']
    assert 'RAW_PRIVATE_TEXT' not in json.dumps(knowledge)
    assert 'text' not in knowledge['evidence']['1']
    assert not knowledge['runtime_usable']
    assert 'RAW_PRIVATE_TEXT' not in client.prompts[-1]
    assert any('RAW_PRIVATE_TEXT' in p for p in client.prompts if p.startswith(repair.REREAD_PROMPT))
    client.prompts.clear()
    again = repair.repair(wiki, messages, output, 'person', 'self', {}, client=client)
    assert again['stage'] == 'complete' and client.prompts == []
    wiki.write_text(wiki.read_text() + '\n- changed input')
    with pytest.raises(ValueError, match='inputs or generator changed'):
        repair.repair(wiki, messages, output, 'person', 'self', {}, client=client)


def test_cannot_invent_source_lines(tmp_path):
    class Invent(Client):
        def run(self, prompt, schema):
            value = json.loads(super().run(prompt, schema))
            value['facts'][0]['lines'] = [999]
            return json.dumps(value)
    rows, _ = repair.select_history(source(tmp_path, [row('private:self:person', 'person', 1000)]),
                                   'person', 'self')
    with pytest.raises(ValueError, match='outside its batch'):
        repair.extract_batch(Invent(), tmp_path / 'cache', {}, [(3, 'old')], rows)
    assert not list((tmp_path / 'cache').glob('*.json'))


def test_missing_old_claim_review_does_not_export_success(tmp_path):
    class Missing(Client):
        def run(self, prompt, schema):
            if repair.RETRY_MARKER in prompt:
                prompt = prompt.split(repair.RETRY_MARKER, 1)[1]
            if prompt.startswith(repair.SECTION_PROMPT):
                return json.dumps(dict(items=[], changes=[]))
            return super().run(prompt, schema)
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- old claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    output = tmp_path / 'output'
    result = repair.repair(wiki, messages, output, 'person', 'self', {}, client=Missing())
    assert result['stage'] == 'incomplete'
    assert not (output / 'content/wiki.md').exists()


def test_reread_replaces_wrong_first_pass_fact_before_final_wiki(tmp_path):
    class Correct(Client):
        def run(self, prompt, schema):
            if prompt.startswith(repair.REREAD_PROMPT):
                data = json.loads(prompt[len(repair.REREAD_PROMPT):])
                old = data['initial_facts'][0]
                new = {k: v for k, v in old.items() if k != 'id'}
                new['summary'] = '工作属于对话者，非本文人物。'
                new['subject'] = 'other'
                return json.dumps(dict(revisions=[dict(fact_id=old['id'], action='correct',
                    reason='回读识别出主体错误。', replacements=[new])], additions=[]))
            return super().run(prompt, schema)
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    output = tmp_path / 'output'
    result = repair.repair(wiki, messages, output, 'person', 'self', {}, client=Correct())
    knowledge = json.loads((output / 'content/knowledge.json').read_text())
    assert result['stage'] == 'complete' and result['reread_actions'] == {'correct': 1}
    assert knowledge['facts'][0]['subject'] == 'other'
    assert not any(f['summary'] == '在消息时期就职于一家机构。' for f in knowledge['facts'])


def test_incomplete_reread_does_not_deliver_first_draft(tmp_path):
    class Skip(Client):
        def run(self, prompt, schema):
            if repair.RETRY_MARKER in prompt:
                prompt = prompt.split(repair.RETRY_MARKER, 1)[1]
            if prompt.startswith(repair.REREAD_PROMPT):
                return json.dumps(dict(revisions=[], additions=[]))
            return super().run(prompt, schema)
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    output = tmp_path / 'output'
    result = repair.repair(wiki, messages, output, 'person', 'self', {}, client=Skip())
    assert result['stage'] == 'incomplete'
    assert (output / 'intermediate/draft.md').exists()
    assert not (output / 'content/wiki.md').exists()


def test_invalid_citation_is_retried_with_diagnostic_and_success_is_cached(tmp_path):
    class Recover(Client):
        def run(self, prompt, schema):
            value = json.loads(super().run(prompt, schema))
            if repair.RETRY_MARKER not in prompt:
                value['facts'][0]['lines'] = [999]
            else:
                assert 'lines=[999]' in prompt
            return json.dumps(value)
    rows, _ = repair.select_history(source(tmp_path, [row('private:self:person', 'person', 1000)]),
                                   'person', 'self')
    client = Recover()
    args = (client, tmp_path / 'cache', {}, [(3, 'old')], rows)
    assert repair.extract_batch(*args)['facts'][0]['lines'] == [1]
    assert len(client.prompts) == 2
    assert repair.extract_batch(*args)['facts'][0]['lines'] == [1]
    assert len(client.prompts) == 2
    assert len(list((tmp_path / 'failures').glob('*.json'))) == 1


def test_reuses_completed_requests_after_retry_implementation_changes(tmp_path):
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    client = Client()
    first = tmp_path / 'first'
    repair.repair(wiki, messages, first, 'person', 'self', {}, client=client)
    manifest = json.loads((first / 'manifest.json').read_text())
    manifest['generator_sha256'] = 'earlier-implementation'
    (first / 'manifest.json').write_text(json.dumps(manifest))
    client.prompts.clear()
    result = repair.repair(wiki, messages, tmp_path / 'second', 'person', 'self', {},
                           client=client, cache_from=first)
    assert result['stage'] == 'complete' and client.prompts == []


def test_speaker_diagnostic_identifies_the_fact_and_actual_source_authors(tmp_path):
    rows, _ = repair.select_history(source(tmp_path, [
        row('private:self:person', 'self', 1000),
        row('private:self:person', 'person', 1001)]), 'person', 'self')
    fact = dict(category='identity', subject='person', speaker_id='person',
                summary='对方这样称呼本文人物。', time_scope='historical', mode='reported',
                lines=[1], old_lines=[])
    with pytest.raises(ValueError, match='not the Wiki subject') as error:
        repair.validate_fact_sources(fact, rows, set())
    assert fact['summary'] in str(error.value)
    assert "{1: 'self'}" in str(error.value)
    fact['speaker_id'] = 'self'
    repair.validate_fact_sources(fact, rows, set())


def test_section_retry_identifies_bad_references_and_preserves_exact_fact_ids(tmp_path):
    class RecoverSection(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            if repair.RETRY_MARKER not in prompt:
                return json.dumps(dict(items=[dict(text='有依据的经历。', fact_ids=['typo'])],
                                       changes=[]))
            assert "invalid_ids=['typo']" in prompt
            assert "allowed_ids=['fact-valid']" in prompt
            assert '有依据的经历。' in prompt
            return json.dumps(dict(items=[dict(text='有依据的经历。', fact_ids=['fact-valid'])],
                                   changes=[]))
    client = RecoverSection()
    facts = [dict(id='fact-valid', category='experience', old_lines=[], summary='有依据的经历。')]
    result = repair.generate_section(client, tmp_path / 'cache', {}, [], facts, {}, 'experience')
    assert result['items'][0]['fact_ids'] == ['fact-valid']
    assert len(client.prompts) == 2
    repair.generate_section(client, tmp_path / 'cache', {}, [], facts, {}, 'experience')
    assert len(client.prompts) == 2


def test_reread_retry_identifies_missing_and_mistyped_fact_ids(tmp_path):
    class RecoverReview(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            retried = repair.RETRY_MARKER in prompt
            if retried:
                assert "missing_ids=['fact-original']" in prompt
                assert "unknown_ids=['fact-typo']" in prompt
            return json.dumps(dict(revisions=[dict(
                fact_id='fact-original' if retried else 'fact-typo', action='retain',
                reason='原聊天仍支持。', replacements=[])], additions=[]))
    rows, _ = repair.select_history(source(tmp_path, [row('private:self:person', 'person', 1000)]),
                                   'person', 'self')
    client = RecoverReview()
    facts = [dict(id='fact-original', summary='有依据的经历。')]
    args = (client, tmp_path / 'cache', {}, [], rows, facts, '自动初稿')
    result = repair.reread_batch(*args)
    assert result['revisions'][0]['fact_id'] == 'fact-original'
    assert len(client.prompts) == 2
    repair.reread_batch(*args)
    assert len(client.prompts) == 2


def test_consolidation_checks_references_and_preserves_old_claim_decisions(tmp_path):
    class Merge(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            if repair.RETRY_MARKER not in prompt:
                return json.dumps(dict(items=[dict(category='background', text='合并后的经历。',
                                                   fact_ids=['unknown'])]))
            assert "invalid_ids=['unknown']" in prompt
            return json.dumps(dict(items=[dict(category='background', text='合并后的经历。',
                                               fact_ids=['F1', 'F2'])]))
    facts = [dict(id='fact-a', category='background', summary='同一经历的主体和时间。'),
             dict(id='fact-b', category='events', summary='同一经历的其他细节。')]
    sections = {k: dict(items=[], changes=[]) for k in repair.CATEGORIES}
    sections['background'] = dict(items=[dict(text='重复一。', fact_ids=['fact-a'])],
        changes=[dict(old_line=2, action='correct', reason='有依据。', fact_ids=['fact-a'])])
    sections['events']['items'] = [dict(text='重复二。', fact_ids=['fact-b'])]
    result = repair.consolidate_sections(Merge(), tmp_path / 'cache', {}, facts, sections)
    assert result['events']['items'] == []
    assert result['background']['items'] == [dict(text='合并后的经历。', fact_ids=['fact-a', 'fact-b'])]
    assert result['background']['changes'] == sections['background']['changes']


def test_adding_consolidation_reuses_all_completed_upstream_requests(tmp_path):
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    client = Client()
    first = tmp_path / 'first'
    repair.repair(wiki, messages, first, 'person', 'self', {}, client=client)
    manifest = json.loads((first / 'manifest.json').read_text())
    manifest.pop('consolidation_sha256')
    (first / 'manifest.json').write_text(json.dumps(manifest))
    for path in (first / 'cache').glob('*.json'):
        result = json.loads(path.read_text())['result']
        if 'items' in result and 'changes' not in result:
            path.unlink()
    client.prompts.clear()
    result = repair.repair(wiki, messages, tmp_path / 'second', 'person', 'self', {},
                           client=client, cache_from=first)
    assert result['stage'] == 'complete'
    assert len(client.prompts) == 1 and client.prompts[0].startswith(repair.CONSOLIDATE_PROMPT)
