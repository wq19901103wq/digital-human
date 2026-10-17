import json
from pathlib import Path

import pytest

from src.share.wiki import wiki_repair as repair


def row(chat, sender, timestamp, text='raw history', kind='text'):
    return dict(chat_id=chat, chat_type=chat.split(':')[0], sender=sender, is_self=sender == 'self',
                timestamp=timestamp, text=text, event=dict(sender_id=sender, kind=kind),
                message_id=f'{chat}:{timestamp}')


def source(tmp_path, rows):
    path = tmp_path / 'messages.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def test_longer_client_timeout_preserves_request_identity_and_successful_wiki_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.judge import corrected
    config = dict(provider='codex_cli', model='unchanged', reasoning_effort='low',
                  codex_cli_version='test-version', timeout_seconds=360)
    monkeypatch.setattr(corrected.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='codex test-version'))
    original = corrected.CodexJudgeClient(config)
    extended = corrected.CodexJudgeClient(dict(config, timeout_seconds=900))
    assert original.cache_identity() == extended.cache_identity()
    result = dict(value='cached')
    calls = []
    monkeypatch.setattr(original, 'run', lambda *args:
                        calls.append(args) or json.dumps(result))
    monkeypatch.setattr(extended, 'run', lambda *args: pytest.fail('successful request must be reused'))
    schema = dict(type='object', properties=dict(value=dict(type='string')),
                  required=['value'], additionalProperties=False)
    parse = lambda value: value['value']
    assert repair._cached_call(original, tmp_path, 'prompt', schema, parse) == result
    assert repair._cached_call(extended, tmp_path, 'prompt', schema, parse) == result
    assert len(calls) == 1
    assert config['timeout_seconds'] == 360


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
        prompt = prompt.replace(repair.REREAD_EVIDENCE_PROMPT, repair.REREAD_PROMPT, 1)
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


def test_explicit_reread_contract_changes_only_reread_requests(tmp_path):
    from src.infrastructure.storage import read_json
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    original, upgraded = tmp_path / 'original', tmp_path / 'upgraded'
    client = Client()
    assert repair.repair(wiki, messages, original, 'person', 'self', {}, client=client,
                         workers=2)['stage'] == 'complete'
    legacy_manifest = read_json(original / 'manifest.json')
    original_calls = list(client.prompts)
    assert 'reread_contract' not in legacy_manifest
    assert repair.repair(wiki, messages, upgraded, 'person', 'self', {}, client=client,
        workers=2, cache_from=original, reread_contract='reread_evidence_v2')['stage'] == 'complete'
    new_calls = client.prompts[len(original_calls):]
    assert len(new_calls) == 1
    assert new_calls[0].startswith(repair.REREAD_EVIDENCE_PROMPT)
    payload = json.loads(new_calls[0][len(repair.REREAD_EVIDENCE_PROMPT):])
    assert payload['source_contract'] == dict(version='reread_evidence_v2',
        message_lines=[1], old_wiki_lines=[1, 2, 3],
        initial_fact_ids=[fact['id'] for fact in payload['initial_facts']])
    assert 'lines 仅引用 messages' in repair.REREAD_EVIDENCE_PROMPT
    assert 'old_lines 仅引用 old_wiki' in repair.REREAD_EVIDENCE_PROMPT
    assert '完整 id' in repair.REREAD_EVIDENCE_PROMPT
    upgraded_manifest = read_json(upgraded / 'manifest.json')
    assert upgraded_manifest['reread_contract'] == 'reread_evidence_v2'
    assert upgraded_manifest['prompts_sha256'] != legacy_manifest['prompts_sha256']
    with pytest.raises(ValueError, match='inputs or generator changed'):
        repair.repair(wiki, messages, original, 'person', 'self', {}, client=client,
                      reread_contract='reread_evidence_v2')


def test_reread_contract_does_not_relabel_invalid_source_references(tmp_path):
    class InvalidClient(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            return json.dumps(dict(revisions=[], additions=[dict(category='background',
                subject='person', speaker_id='person', summary='有证据的历史资料。',
                time_scope='historical', mode='self_report', lines=[41], old_lines=[41])]))
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    selected, _ = repair.select_history(messages, 'person', 'self')
    selected[0]['line'] = 41
    client = InvalidClient()
    kwargs = dict(reread_contract='reread_evidence_v2')
    with pytest.raises(ValueError, match='old_lines'):
        repair.reread_batch(client, tmp_path / 'cache', {}, [[3, 'old claim']], selected,
                            [], '', **kwargs)
    assert len(client.prompts) == repair.RETRY_LIMIT
    with pytest.raises(ValueError, match='exhausted'):
        repair.reread_batch(client, tmp_path / 'cache', {}, [[3, 'old claim']], selected,
                            [], '', **kwargs)
    assert len(client.prompts) == repair.RETRY_LIMIT


def test_cache_transfer_preserves_exhausted_unchanged_request_budget(tmp_path):
    class InvalidExtraction(Client):
        def run(self, prompt, schema):
            value = json.loads(super().run(prompt, schema))
            if 'facts' in value:
                value['facts'][0]['old_lines'] = [999]
            return json.dumps(value)
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    client = InvalidExtraction()
    original, upgraded = tmp_path / 'original', tmp_path / 'upgraded'
    assert repair.repair(wiki, messages, original, 'person', 'self', {},
                         client=client)['stage'] == 'incomplete'
    assert len(client.prompts) == repair.RETRY_LIMIT
    failures = {path.name: path.read_bytes() for path in (original / 'failures').glob('*.json')}
    assert len(failures) == repair.RETRY_LIMIT
    assert repair.repair(wiki, messages, upgraded, 'person', 'self', {}, client=client,
        cache_from=original, reread_contract='reread_evidence_v2')['stage'] == 'incomplete'
    assert len(client.prompts) == repair.RETRY_LIMIT
    assert {path.name: path.read_bytes() for path in (upgraded / 'failures').glob('*.json')} == failures


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


def feedback_fixture(tmp_path):
    schema = repair.obj(dict(value=repair.TEXT))
    prompt = 'Only the frozen source is evidence.'
    def parse(value):
        if value['value'] != 'supported':
            raise ValueError('missing source: expected=source-1')
    return tmp_path / 'cache', prompt, schema, parse


def exhausted_correction_fixture(tmp_path):
    from src.infrastructure.storage import write_json
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Invalid(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            return json.dumps(dict(value='unsupported'))
    client = Invalid()
    with pytest.raises(ValueError, match='missing source'):
        repair._cached_call(client, directory, prompt, schema, parse)
    key = repair.digest([client.cache_identity(), prompt, schema])
    failure = tmp_path / 'failures' / f'{key}-3.json'
    selected = dict(file=f'failures/{failure.name}',
                    source=dict(path=str(failure), sha256=repair.sha256_file(failure)))
    write_json(tmp_path / 'manifest.json', dict(correction=dict(contract=repair.CORRECTION_POLICY,
        limit=repair.RETRY_LIMIT, requests={key: selected})))
    return directory, prompt, schema, parse, client, key


def test_explicit_correction_has_finite_budget_and_keeps_unselected_requests_exhausted(tmp_path):
    from src.infrastructure.storage import read_json, write_json
    directory, prompt, schema, parse, client, key = exhausted_correction_fixture(tmp_path)
    old = {path: path.read_bytes() for path in (tmp_path / 'failures').glob('*.json')}
    manifest = read_json(tmp_path / 'manifest.json')
    write_json(tmp_path / 'manifest.json', dict(correction=manifest['correction'] | dict(requests={})))
    with pytest.raises(ValueError, match='unchanged request exhausted'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 3
    write_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='missing source'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 6
    assert all(request.startswith('纠错版本 source_feedback_v2') for request in client.prompts[3:])
    for attempt in range(4, 7):
        record = read_json(tmp_path / 'failures' / f'{key}-{attempt}.json')
        assert record['retry_policy'] == repair.CORRECTION_POLICY
        assert record['policy_attempt'] == attempt - 3 and record['attempt'] == attempt
    with pytest.raises(ValueError, match='unchanged request exhausted'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 6
    assert all(path.read_bytes() == content for path, content in old.items())


def test_explicit_correction_preserves_allowance_after_transport_and_reuses_success(tmp_path):
    from src.infrastructure.storage import read_json
    directory, prompt, schema, parse, _, key = exhausted_correction_fixture(tmp_path)
    class Recover(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            if len(self.prompts) == 1:
                raise transport_error()
            return json.dumps(dict(value='supported'))
    client = Recover()
    with pytest.raises(type(transport_error())):
        repair._cached_call(client, directory, prompt, schema, parse)
    record = read_json(tmp_path / 'failures' / f'{key}-4.json')
    assert record['retry_policy'] == repair.CORRECTION_POLICY and record['policy_attempt'] == 0
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert len(client.prompts) == 2 and 'missing source' in client.prompts[-1]


@pytest.mark.parametrize('mutation', ['bytes', 'symlink', 'file', 'contract', 'limit'])
def test_explicit_correction_rejects_changed_failure_seal_before_request(tmp_path, mutation):
    from src.infrastructure.storage import read_json, write_json
    directory, prompt, schema, parse, client, key = exhausted_correction_fixture(tmp_path)
    manifest = read_json(tmp_path / 'manifest.json')
    failure = tmp_path / 'failures' / f'{key}-3.json'
    if mutation == 'bytes':
        failure.write_text(failure.read_text() + '\n')
    elif mutation == 'symlink':
        saved = tmp_path / 'saved.json'
        failure.rename(saved)
        failure.symlink_to(saved)
    elif mutation == 'file':
        manifest['correction']['requests'][key]['file'] = f'../failures/{failure.name}'
    else:
        manifest['correction'][mutation] = 'changed'
    write_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='correction'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 3


def transport_error():
    from src.judge.codex_transport import CodexTransportError
    return CodexTransportError(dict(exit_code=1, terminal_events=['turn.failed'],
        signals=['rate_limit'], terminal_signals=['rate_limit'], stderr_signals=[]))


def test_transport_failures_leave_format_budget_and_original_request_unchanged(tmp_path):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Recover(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            if len(self.prompts) <= 4:
                raise transport_error()
            return '{"value": "supported"}'
    client = Recover()
    key = repair.digest([client.cache_identity(), prompt, schema])
    saved_bytes = {}
    for attempt in range(1, 5):
        with pytest.raises(type(transport_error())):
            repair._cached_call(client, directory, prompt, schema, parse)
        assert len(client.prompts) == attempt
        saved = tmp_path / 'failures' / f'{key}-{attempt}.json'
        record = json.loads(saved.read_text())
        assert record['attempt'] == attempt and record['policy_attempt'] == 0
        assert record['retry_budget'] == 'format_only_v1'
        assert record['failure_kind'] == 'transport' and record['errors'] == []
        saved_bytes[saved] = saved.read_bytes()
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert client.prompts == [prompt] * 5
    assert json.loads((directory / f'{key}.json').read_text())['attempts'] == 5
    assert all(path.read_bytes() == content for path, content in saved_bytes.items())


def test_service_blocked_without_request_creates_no_failure_record(tmp_path):
    from src.judge.codex_transport import CodexServiceBlocked
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Blocked(Client):
        def run(self, request, schema):
            raise CodexServiceBlocked(transport_error().details)
    with pytest.raises(CodexServiceBlocked):
        repair._cached_call(Blocked(), directory, prompt, schema, parse)
    assert not list(tmp_path.rglob('*.json'))


def test_resume_after_transport_failure_keeps_only_real_content_feedback(tmp_path):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Recover(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            if len(self.prompts) == 2:
                raise transport_error()
            return json.dumps(dict(value='unsupported' if len(self.prompts) == 1 else 'supported'))
    client = Recover()
    with pytest.raises(type(transport_error())):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 2
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert len(client.prompts) == 3 and client.prompts.count(prompt) == 1
    assert 'expected=source-1' in client.prompts[-1] and 'unsupported' in client.prompts[-1]
    assert 'Codex transport diagnostics' not in client.prompts[-1]
    assert client.prompts[-1] == client.prompts[-2]
    records = [json.loads(path.read_text()) for path in sorted((tmp_path / 'failures').glob('*.json'))]
    assert [record['policy_attempt'] for record in records] == [1, 1]


@pytest.mark.parametrize('wrapper', ['', 'Codex Judge 执行失败（exit 1）：stream disconnected\n\n',
                                     'Codex Judge 执行失败（exit 1）：\n'])
def test_legacy_transport_history_recovers_only_documented_format_budget(tmp_path, wrapper):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Invalid(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            assert 'unsupported' in request and 'Codex transport diagnostics' not in request
            return '{"value": "unsupported"}'
    client = Invalid()
    key = repair.digest([client.cache_identity(), prompt, schema])
    failures = tmp_path / 'failures'
    failures.mkdir()
    errors = []
    for attempt in range(1, 4):
        error = 'missing source: expected=source-1' if attempt == 2 else str(transport_error())
        if attempt != 2:
            error = wrapper + error
        errors.append(dict(attempt=attempt, error=error))
        (failures / f'{key}-{attempt}.json').write_text(json.dumps(dict(attempt=attempt,
            policy_attempt=attempt, retry_policy=repair.RETRY_POLICY, error=error,
            result=dict(value='unsupported') if attempt == 2 else None, errors=list(errors))))
    before = {path: path.read_bytes() for path in failures.glob('*.json')}
    with pytest.raises(ValueError, match='expected=source-1'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 2
    latest = json.loads((failures / f'{key}-5.json').read_text())
    assert latest['attempt'] == 5 and latest['policy_attempt'] == repair.RETRY_LIMIT
    with pytest.raises(ValueError, match='unchanged request exhausted'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 2
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('attempt,error,result', [
    (4, str(transport_error()), None),
    (3, 'Codex transport diagnostics: {malformed', None),
    (3, 'Codex transport diagnostics: {}', None),
    (3, 'untrusted text\n\n' + str(transport_error()), None),
    (3, 'Codex Judge 执行失败（exit 2）：stderr\n\n' + str(transport_error()), None),
    (3, 'Codex Judge 执行失败（exit 1）：stderr\n\nCodex transport diagnostics: {malformed', None),
    (3, 'Codex Judge 执行失败（exit 1）：stderr\n\n' + str(transport_error()) + '\nextra', None),
    (3, 'Codex Judge 执行失败（exit 2）：\n' + str(transport_error()), None),
    (3, 'Codex Judge 执行失败（exit 1）：\nCodex transport diagnostics: {malformed', None),
    (3, 'Codex Judge 执行失败（exit 1）：\n' + str(transport_error()) + '\nextra', None),
    (3, 'Codex Judge 执行失败（exit 1）：' + str(transport_error()), None),
    (3, 'Codex Judge 执行失败（exit 1）：stderr\n' + str(transport_error()), None),
    (3, 'Codex Judge 执行失败（exit 1）：\n' + str(transport_error()), dict(value='unsupported')),
    (3, 'Codex Judge 执行失败（exit 1）：stderr\n\n' + str(transport_error()), dict(value='unsupported')),
    (3, str(transport_error()), dict(value='unsupported')),
])
def test_unknown_or_incomplete_transport_history_does_not_reset_budget(tmp_path, attempt, error, result):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    client = Client()
    key = repair.digest([client.cache_identity(), prompt, schema])
    failures = tmp_path / 'failures'
    failures.mkdir()
    saved = failures / f'{key}-{attempt}.json'
    saved.write_text(json.dumps(dict(attempt=attempt, error=error, result=result)))
    before = saved.read_bytes()
    with pytest.raises(ValueError, match='unchanged request exhausted'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert client.prompts == [] and saved.read_bytes() == before


@pytest.mark.parametrize('wrapper', ['Codex Judge 执行失败（exit 1）：stream disconnected\n\n',
                                    'Codex Judge 执行失败（exit 1）：\n'])
def test_wrapped_transport_history_resumes_original_request_without_content_feedback(tmp_path, wrapper):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Recover(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            return '{"value": "supported"}'
    client = Recover()
    key = repair.digest([client.cache_identity(), prompt, schema])
    failures = tmp_path / 'failures'
    failures.mkdir()
    for attempt in range(1, 4):
        error = wrapper + str(transport_error())
        (failures / f'{key}-{attempt}.json').write_text(json.dumps(dict(
            attempt=attempt, policy_attempt=attempt, error=error, result=None)))
    before = {path: path.read_bytes() for path in failures.glob('*.json')}
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert client.prompts == [prompt]
    assert json.loads((directory / f'{key}.json').read_text())['attempts'] == 4
    assert all(path.read_bytes() == content for path, content in before.items())


def test_valid_saved_content_is_reused_after_later_transport_failure(tmp_path):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    client = Client()
    key = repair.digest([client.cache_identity(), prompt, schema])
    failures = tmp_path / 'failures'
    failures.mkdir()
    (failures / f'{key}-3.json').write_text(json.dumps(dict(attempt=3, policy_attempt=3,
        retry_budget='format_only_v1', error='old check failed', result=dict(value='supported'))))
    (failures / f'{key}-4.json').write_text(json.dumps(dict(attempt=4, policy_attempt=3,
        retry_budget='format_only_v1', error=str(transport_error()), result=None)))
    before = {path: path.read_bytes() for path in failures.glob('*.json')}
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert client.prompts == []
    assert json.loads((directory / f'{key}.json').read_text())['attempts'] == 4
    assert all(path.read_bytes() == content for path, content in before.items())


def test_saved_legacy_failure_is_rechecked_and_resume_starts_with_feedback(tmp_path):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Recover(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            assert request.endswith(repair.RETRY_MARKER + prompt)
            assert 'expected=source-1' in request
            assert 'unsupported' in request
            return '{"value": "supported"}'
    client = Recover()
    key = repair.digest([client.cache_identity(), prompt, schema])
    failures = tmp_path / 'failures'
    failures.mkdir()
    legacy = failures / f'{key}-1.json'
    legacy.write_text(json.dumps(dict(attempt=1, error='generic failure',
        result=dict(value='unsupported'), errors=[dict(attempt=1, error='generic failure')])))
    before = legacy.read_bytes()
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert legacy.read_bytes() == before and len(client.prompts) == 1
    assert json.loads((directory / f'{key}.json').read_text())['attempts'] == 2
    repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 1


@pytest.mark.parametrize('record', [
    dict(attempt=3),
    dict(attempt=4, retry_policy=repair.RETRY_POLICY, policy_attempt=1),
    dict(attempt=1, retry_policy=repair.RETRY_POLICY, policy_attempt=3),
])
def test_exhausted_saved_failure_never_resets_budget(tmp_path, record):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    client = Client()
    key = repair.digest([client.cache_identity(), prompt, schema])
    failures = tmp_path / 'failures'
    failures.mkdir()
    saved = failures / f'{key}-{record["attempt"]}.json'
    saved.write_text(json.dumps(dict(record, error='generic failure',
                                    result=dict(value='unsupported'))))
    before = saved.read_bytes()
    with pytest.raises(ValueError, match='unchanged request exhausted'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert client.prompts == []
    assert saved.read_bytes() == before
    assert list(failures.glob('*.json')) == [saved]
    assert not list(directory.glob('*.json'))


def test_saved_legacy_failure_consumes_remaining_request_budget(tmp_path):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Invalid(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            return '{"value": "unsupported"}'
    client = Invalid()
    key = repair.digest([client.cache_identity(), prompt, schema])
    failures = tmp_path / 'failures'
    failures.mkdir()
    legacy = failures / f'{key}-1.json'
    legacy.write_text(json.dumps(dict(attempt=1, error='generic failure',
                                     result=dict(value='unsupported'))))
    before = legacy.read_bytes()
    with pytest.raises(ValueError, match='expected=source-1'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == repair.RETRY_LIMIT - 1
    assert all(request.endswith(repair.RETRY_MARKER + prompt) for request in client.prompts)
    latest = json.loads((failures / f'{key}-3.json').read_text())
    assert latest['attempt'] == latest['policy_attempt'] == repair.RETRY_LIMIT
    with pytest.raises(ValueError, match='unchanged request exhausted'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == repair.RETRY_LIMIT - 1
    assert legacy.read_bytes() == before


def test_supported_saved_failure_is_reused_after_budget_exhausted(tmp_path):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    client = Client()
    key = repair.digest([client.cache_identity(), prompt, schema])
    failures = tmp_path / 'failures'
    failures.mkdir()
    saved = failures / f'{key}-3.json'
    saved.write_text(json.dumps(dict(attempt=3, error='old check failed',
                                    result=dict(value='supported'))))
    before = saved.read_bytes()
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert client.prompts == []
    assert saved.read_bytes() == before
    assert json.loads((directory / f'{key}.json').read_text())['attempts'] == 3


def test_exhausted_feedback_is_persistent_and_changed_requests_have_separate_budgets(tmp_path):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Invalid(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            return '{"value": "unsupported"}'
    client = Invalid()
    with pytest.raises(ValueError, match='expected=source-1'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == repair.RETRY_LIMIT
    assert client.prompts[0] == prompt
    assert len(set(client.prompts)) == repair.RETRY_LIMIT
    before = {path: path.read_bytes() for path in (tmp_path / 'failures').glob('*.json')}
    with pytest.raises(ValueError, match='unchanged request exhausted'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == repair.RETRY_LIMIT
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list(directory.glob('*.json'))
    with pytest.raises(ValueError, match='expected=source-1'):
        repair._cached_call(client, directory, prompt + 'changed input', schema, parse)
    assert len(client.prompts) == 2 * repair.RETRY_LIMIT
    assert client.prompts[repair.RETRY_LIMIT] == prompt + 'changed input'


def test_interrupted_correction_preserves_feedback_and_remaining_request_budget(tmp_path):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Interrupted(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            if len(self.prompts) == 2:
                raise TimeoutError('transport timed out')
            return json.dumps(dict(value='unsupported' if len(self.prompts) == 1 else 'supported'))
    client = Interrupted()
    with pytest.raises(TimeoutError):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 2
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    assert len(client.prompts) == 3 and client.prompts.count(prompt) == 1
    assert 'transport timed out' in client.prompts[-1]
    assert len(list((tmp_path / 'failures').glob('*.json'))) == 2


def test_cache_validation_rejection_preserves_raw_response_for_feedback(tmp_path):
    from src import cache
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    class Validating(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            text = json.dumps(dict(value='unsupported' if len(self.prompts) == 1 else 'supported'))
            if not cache.response_valid(text):
                raise RuntimeError('transport rejected the response')
            return text
    client = Validating()
    with pytest.raises(RuntimeError, match='transport rejected'):
        repair._cached_call(client, directory, prompt, schema, parse)
    assert len(client.prompts) == 1
    assert repair._cached_call(client, directory, prompt, schema, parse) == dict(value='supported')
    failure = json.loads(next((tmp_path / 'failures').glob('*.json')).read_text())
    assert failure['result'] == dict(value='unsupported')
    assert 'unsupported' in client.prompts[-1]


@pytest.mark.parametrize('update', ['FEEDBACK_PIPELINE_UPDATES', 'TRANSPORT_IMPLEMENTATION_UPDATES',
                                    'TRANSPORT_DIAGNOSTIC_IMPLEMENTATION_UPDATES',
                                    'TRANSPORT_WRAPPER_IMPLEMENTATION_UPDATES'])
def test_source_feedback_upgrade_cannot_silently_resume_old_generator(tmp_path, update):
    from src.share.wiki import wiki_batch
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000)])
    output = tmp_path / 'output'
    client = Client()
    assert repair.repair(wiki, messages, output, 'person', 'self', {}, client=client)['stage'] == 'complete'
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['generator_sha256'] = getattr(wiki_batch, update)['wiki_repair.py'][0]
    manifest_path.write_text(json.dumps(manifest))
    before = manifest_path.read_bytes()
    calls = len(client.prompts)
    with pytest.raises(ValueError, match='inputs or generator changed'):
        repair.repair(wiki, messages, output, 'person', 'self', {}, client=client)
    assert len(client.prompts) == calls
    assert manifest_path.read_bytes() == before
    manifest['generator_sha256'] = 'future-unapproved-implementation'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='inputs or generator changed'):
        repair.repair(wiki, messages, output, 'person', 'self', {}, client=client)
    assert len(client.prompts) == calls


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


@pytest.mark.parametrize('action', ['retain', 'correct'])
@pytest.mark.parametrize('resolution', ['omit', 'unresolved'])
def test_section_retry_identifies_unsupported_old_claim_without_declaring_it_false(
        tmp_path, action, resolution):
    unsupported = dict(old_line=20, action=action, reason='没有可核实的关系记录。', fact_ids=[])
    class ResolveSection(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            if repair.RETRY_MARKER not in prompt:
                return json.dumps(dict(items=[], changes=[unsupported]))
            assert 'retaining/correcting a claim requires chat evidence' in prompt
            assert json.dumps(unsupported, ensure_ascii=False) in prompt
            assert 'use omit or unresolved' in prompt
            assert 'does not mean the old claim is false' in prompt
            assert 'Do not manufacture a fact id or attach an unrelated fact' in prompt
            return json.dumps(dict(items=[], changes=[dict(unsupported, action=resolution)]))
    client = ResolveSection()
    args = (client, tmp_path / 'cache', {}, [], [], {20: dict(category='background')}, 'background')
    result = repair.generate_section(*args)
    assert result['changes'] == [dict(unsupported, action=resolution)]
    assert len(client.prompts) == 2
    assert repair.generate_section(*args) == result
    assert len(client.prompts) == 2


def test_section_feedback_upgrade_uses_only_remaining_correction_attempt(tmp_path):
    from src.infrastructure.storage import read_json, write_json
    directory = tmp_path / 'cache'
    unsupported = dict(old_line=20, action='retain', reason='没有可核实的关系记录。', fact_ids=[])
    class ResolveSection(Client):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            assert 'offending_change=' in prompt and json.dumps(unsupported, ensure_ascii=False) in prompt
            return json.dumps(dict(items=[], changes=[dict(unsupported, action='unresolved')]))
    client = ResolveSection()
    payload = dict(subject={}, section=repair.CATEGORIES['background'], old_wiki=[],
                   review_lines=[20], facts=[])
    prompt = repair.SECTION_PROMPT + json.dumps(payload, ensure_ascii=False)
    key = repair.digest([client.cache_identity(), prompt, repair.SECTION_SCHEMA])
    original = tmp_path / 'failures' / f'{key}-3.json'
    failure = dict(attempt=3, error='retaining/correcting a claim requires chat evidence',
                   result=dict(items=[], changes=[unsupported]), errors=[],
                   retry_policy=repair.RETRY_POLICY, policy_attempt=repair.RETRY_LIMIT)
    write_json(original, failure)
    write_json(tmp_path / 'manifest.json', dict(correction=dict(contract=repair.CORRECTION_POLICY,
        limit=repair.RETRY_LIMIT, requests={key: dict(file=f'failures/{original.name}',
            source=dict(path=str(original), sha256=repair.sha256_file(original)))})))
    for attempt in (4, 5):
        write_json(tmp_path / 'failures' / f'{key}-{attempt}.json', dict(failure,
            attempt=attempt, retry_policy=repair.CORRECTION_POLICY, policy_attempt=attempt-3))
    write_json(tmp_path / 'failures' / f'{key}-6.json', dict(attempt=6,
        error=str(transport_error()), result=None, errors=[], retry_policy=repair.CORRECTION_POLICY,
        policy_attempt=2, retry_budget='format_only_v1'))
    before = {path: path.read_bytes() for path in (tmp_path / 'failures').glob('*.json')}
    manifest_bytes = (tmp_path / 'manifest.json').read_bytes()
    args = (client, directory, {}, [], [], {20: dict(category='background')}, 'background')
    result = repair.generate_section(*args)
    assert result['changes'][0]['action'] == 'unresolved' and len(client.prompts) == 1
    assert read_json(directory / f'{key}.json')['attempts'] == 7
    assert repair.generate_section(*args) == result and len(client.prompts) == 1
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (tmp_path / 'manifest.json').read_bytes() == manifest_bytes


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
