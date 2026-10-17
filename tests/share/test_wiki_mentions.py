"""Incremental mention retrieval keeps raw attribution and existing work intact."""
import json

import pytest

from src.share.wiki import wiki_mentions as mentions
from src.share.wiki import wiki_repair as repair
from test_wiki_repair import Client, row, source


class MentionClient(Client):
    def run(self, prompt, schema):
        value = json.loads(super().run(prompt, schema))
        if repair.MENTION_PROMPT in prompt and prompt.startswith(repair.EXTRACT_PROMPT):
            value['facts'][0].update(speaker_id='other', mode='reported',
                summary='另一位聊天参与者提及此人的到访计划。')
        return json.dumps(value)


@pytest.mark.parametrize('stage', ['extract', 'reread'])
def test_mention_failures_preserve_request_key_and_tail(tmp_path, monkeypatch, stage):
    key = 'a' * 64
    def fail(*args, **kwargs):
        raise repair.request_error(key, 'failure ' * 200)
    monkeypatch.setattr(repair, 'extract_batch', fail if stage == 'extract' else
                        lambda *args, **kwargs: dict(facts=[]))
    monkeypatch.setattr(repair, 'reread_batch', fail)
    progress = dict(failed=0, errors=[])
    result = mentions.extend_facts(object(), tmp_path, {}, [], [dict(line=1, timestamp=1,
        sender_id='other', sender='other', message_kind='text', text='test', chat_id='other')],
        dict(policy='test', anchors=[], terms=[]), {}, 1, 1000, progress, lambda: None)
    assert result is None
    assert progress['errors'] == [dict(stage=f'mentions_{stage}', batch=0,
        error=('failure ' * 200)[-800:], request_sha256=key)]


def test_reconciliation_failures_preserve_request_key_and_tail(tmp_path, monkeypatch):
    from src.share.wiki import wiki_reconcile
    key = 'b' * 64
    def fail(*args, **kwargs):
        raise repair.request_error(key, 'failure ' * 200)
    monkeypatch.setattr(wiki_reconcile, 'groups_for', lambda data: [{}])
    monkeypatch.setattr(wiki_reconcile, 'jobs_for', lambda *args, **kwargs: [([], [])])
    monkeypatch.setattr(wiki_reconcile, 'extract', fail)
    progress = dict(failed=0, errors=[])
    result = wiki_reconcile.reconcile(object(), tmp_path, {}, [], 1, progress, lambda: None)
    assert result is None
    assert progress['errors'] == [dict(stage='reconciling_attributes', batch=0,
        error=('failure ' * 200)[-1000:], request_sha256=key)]


def test_legacy_recall_names_do_not_include_people_from_biography():
    names = mentions.legacy_names('# 阿林\n- 别名：林老师、小林\n## 经历\n- 同事张三\n'
                                  '## 别名\n- A_Lin、阿林\n', '阿林')
    assert set(names) == {'阿林', '林老师', '小林', 'A_Lin'}


def test_mentions_include_absent_person_but_preserve_ambiguity_and_exporter(tmp_path):
    records = [row('private:self:person', 'person', 1000),
        row('group:self:room', 'other', 1000, '阿林在旅行'),
        row('group:self:room', 'self', 1001, '下周回来吗'),
        row('group:self:room', 'other', 9000, '不相关'),
        row('group:another:room', 'other', 1000, '阿林在旅行'),
        row('group:self:room2', 'homonym', 1000, '今天下雨'),
        row('group:self:room3', 'other', 1000, 'xalicey'),
        row('group:self:room4', 'other', 1000, '@ALICE 明天见')]
    records[5]['sender'] = '阿林'
    path = source(tmp_path, records)
    base, _ = repair.select_history(path, 'person', 'self')
    selected, report = mentions.select_mentions(path, base, 'person', 'self',
                                                '- 别名：Alice', '阿林')
    assert {r['line'] for r in selected} == {2, 3, 8}
    assert report['added_rows'] == 3
    assert report['anchors'][0]['identity_status'] == 'unverified_mention'
    assert report['anchors'][0]['ambiguous_terms'] == ['阿林']
    assert '"text"' not in json.dumps(report)
    assert '阿林在旅行' not in json.dumps(report, ensure_ascii=False)


def test_already_covered_mention_windows_do_not_trigger_duplicate_calls(tmp_path):
    path = source(tmp_path, [row('group:self:room', 'person', 1000),
        row('group:self:room', 'other', 1001, '阿林'),
        row('group:self:room', 'other', 1002, '阿林')])
    base, _ = repair.select_history(path, 'person', 'self')
    selected, report = mentions.select_mentions(path, base, 'person', 'self', '', '阿林')
    assert selected == [] and report['added_rows'] == 0


def test_other_private_mentions_preserve_speaker_chat_and_coverage(tmp_path):
    path = source(tmp_path, [row('private:self:person', 'person', 1000),
        row('private:self:other', 'other', 2000, '阿林明天来'),
        row('private:self:other', 'self', 2001, '在哪见'),
        row('private:another:other', 'other', 2000, '阿林明天来')])
    base, _ = repair.select_history(path, 'person', 'self')
    selected, report = mentions.select_mentions(path, base, 'person', 'self', '', '阿林')
    assert {r['line'] for r in selected} == {2, 3}
    assert selected[0]['sender_id'] == 'other'
    assert all(r['chat_type'] == 'private' for r in selected)
    assert report['added_rows_by_chat_type'] == {'private': 2}
    wiki = tmp_path / '阿林.md'
    wiki.write_text('# 阿林\n## 基本信息\n- old claim\n')
    output = tmp_path / 'expanded'
    result = repair.repair(wiki, path, output, 'person', 'self', {}, client=MentionClient(), include_mentions=True)
    assert result['stage'] == 'complete'
    coverage = json.loads((output / 'content/knowledge.json').read_text())['coverage']
    assert coverage['target_private_rows'] == 1
    assert coverage['private_rows'] == 3
    assert coverage['group_rows'] == 0
    assert coverage['mention_added_rows_by_chat_type'] == {'private': 2}


def test_incremental_two_pass_generation_reuses_all_base_calls(tmp_path):
    wiki = tmp_path / '阿林.md'
    wiki.write_text('# 阿林\n## 基本信息\n- old claim\n')
    path = source(tmp_path, [row('private:self:person', 'person', 1000),
                            row('group:self:room', 'other', 2000, '阿林明天来')])
    client = Client()
    original = tmp_path / 'original'
    repair.repair(wiki, path, original, 'person', 'self', {}, client=client)
    original_prompts = set(client.prompts)
    client.prompts.clear()

    client = MentionClient()
    output = tmp_path / 'expanded'
    result = repair.repair(wiki, path, output, 'person', 'self', {}, client=client,
                           cache_from=original, include_mentions=True)
    assert result['stage'] == 'complete'
    assert result['mention_completed'] == result['mention_reread_completed'] == 1
    assert original_prompts.isdisjoint(client.prompts)
    assert any(repair.MENTION_PROMPT in p for p in client.prompts)
    data = json.loads((output / 'content/knowledge.json').read_text())
    assert data['coverage']['mention_added_rows'] == 1
    assert data['coverage']['nickname_only_group_mentions_included']
    assert any(f['speaker_id'] == 'other' and f['mode'] == 'reported' for f in data['facts'])
    client.prompts.clear()
    repair.repair(wiki, path, output, 'person', 'self', {}, client=client, include_mentions=True)
    assert client.prompts == []


def test_explicit_reread_contract_reaches_base_and_mention_windows(tmp_path):
    wiki = tmp_path / '阿林.md'
    wiki.write_text('# 阿林\n## 基本信息\n- old claim\n')
    path = source(tmp_path, [row('private:self:person', 'person', 1000),
                            row('group:self:room', 'other', 2000, '阿林明天来')])
    client = MentionClient()
    result = repair.repair(wiki, path, tmp_path / 'expanded', 'person', 'self', {},
        client=client, include_mentions=True, reread_contract='reread_evidence_v2')
    assert result['stage'] == 'complete'
    rereads = [prompt for prompt in client.prompts
               if prompt.startswith(repair.REREAD_EVIDENCE_PROMPT)]
    assert len(rereads) == 2
    assert sum(repair.MENTION_PROMPT in prompt for prompt in rereads) == 1
    for prompt in rereads:
        payload = json.loads(prompt[len(repair.REREAD_EVIDENCE_PROMPT):].replace(repair.MENTION_PROMPT, ''))
        assert payload['source_contract']['message_lines'] == sorted({row[0] for row in payload['messages']})
        assert payload['source_contract']['old_wiki_lines'] == [1, 2, 3]
        assert payload['source_contract']['initial_fact_ids'] == [fact['id'] for fact in payload['initial_facts']]
