"""Same-page completion preserves correction evidence and downstream scope."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.infrastructure.config import sha256_file
from src.infrastructure.storage import read_json, write_json
from src.judge import corrected
from src.share.wiki import wiki_batch, wiki_cached, wiki_completion as completion
from src.share.wiki import wiki_constraints as constraints, wiki_continuation as continuation
from src.share.wiki import wiki_reconcile, wiki_repair as repair, wiki_structured
from src.share.wiki.wiki_library import digest
from test_wiki_constraints import Client, change, section_task
from test_wiki_continuation import correction_batch


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('completion regression must not make live model requests')
    monkeypatch.setattr(corrected.CodexJudgeClient, '__init__', forbidden)
    monkeypatch.setattr(corrected.CodexJudgeClient, '_run', forbidden)
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', forbidden)


def downstream(subject, stage='consolidating'):
    subject = deepcopy(subject)
    if stage == 'consolidating':
        payload = dict(subject=subject, categories=repair.CATEGORIES,
            sections={category: [] for category in repair.CATEGORIES}, facts=[dict(id='F1')])
        template, schema = repair.CONSOLIDATE_PROMPT, repair.CONSOLIDATE_SCHEMA
        bounds = dict(fact_ids=['F1'])
    else:
        rows = [[1, 'synthetic time', subject['account'], 'Person', 'text', 'synthetic message', 'chat']]
        payload = dict(subject=subject, columns=completion.COLUMNS, messages=rows)
        if stage == 'structuring':
            payload.update(facts=[dict(id='F1')], source_contract=wiki_structured.source_contract(subject,
                [dict(line=1, sender_id=subject['account'], message_kind='text')]))
            template, schema = wiki_structured.source_prompt('structured_source_v2'), wiki_structured.SCHEMA
            bounds = dict(lines=[1], utterance_line=[1], fact_id=['F1'], fact_ids=['F1'])
        else:
            subject['entity_id'] = 'entity-' + digest([subject['self_account'], subject['account']])[:20]
            payload.update(entities=[], records=[dict(id='record-1')])
            template, schema = wiki_reconcile.PROMPT, wiki_reconcile.SCHEMA
            bounds = dict(lines=[1], record_id=['record-1'])
    prompt = repair.REFERENCE_PROMPT + template + json.dumps(payload, ensure_ascii=False)
    return prompt, repair.bounded_reference_schema(schema, bounds)


def corrected_page(tmp_path):
    source, knowledge = correction_batch(tmp_path)
    original = read_json(source / 'manifest.json')
    original['pipeline'] = wiki_batch.pipeline()
    write_json(source / 'manifest.json', original)
    job = next(job for job in original['jobs'] if job['id'] == 'rejected')
    prompt, schema, _ = section_task()
    key = digest([Client().cache_identity(), prompt, schema])
    selected = dict(rejected=dict(verified_sources=wiki_cached._sources(job, original, None),
        requests={key: dict(plan=constraints.wire_plan(prompt, schema)['seal'],
            source=continuation.locator(Path(job['output']) / 'failures' / f'{"a" * 64}-3.json'))}))
    output = tmp_path / 'corrected'
    def finalize(manifest):
        constraints._finalize(manifest, source, output, selected,
            source / 'grant.json', source / 'ledger.json')
    continuation.prepare(source, output, _finalize=finalize)
    manifest = read_json(output / 'manifest.json')
    policy = next(job for job in manifest['jobs'] if job['id'] == 'rejected')['correction']
    ledger = read_json(source / 'ledger.json')
    ledger['used']['rejected'] = constraints.LIMIT
    ledger['seal'] = digest({name: value for name, value in ledger.items() if name != 'seal'})
    write_json(source / 'ledger.json', ledger)
    directory = output / 'jobs/rejected'
    write_json(directory / 'cache' / f'{key}.json', dict(request_sha256=key,
        retry_policy=constraints.CONTRACT, policy_attempt=constraints.LIMIT, attempt=9,
        retry_budget='all_invocations_v1', grant_sha256=policy['grant']['sha256'],
        wire_plan=policy['requests'][key]['plan'], result=dict(items=[], changes=[change()])))
    subject = dict(name=job['title'], account=job['account'], self_account=job['self_account'])
    consolidate, consolidate_schema = downstream(subject)
    blocked_key = digest([Client().cache_identity(), consolidate, consolidate_schema])
    progress = dict(stage='incomplete', batches=1, completed=1, reread_completed=1,
        sections_completed=len(repair.CATEGORIES), mention_batches=0, mention_completed=0,
        mention_reread_completed=0, errors=[dict(stage='consolidating', error=completion.BLOCKED,
            request_sha256=blocked_key)])
    write_json(directory / 'progress.json', progress)
    write_json(directory / 'manifest.json', dict(subject=subject, config=manifest['config'],
        correction=policy, wiki=job['wiki'], messages=manifest['messages'],
        generator_sha256=manifest['pipeline']['wiki_repair.py']))
    state = read_json(source / 'progress.json')
    state['stage'] = 'incomplete'
    write_json(output / 'progress.json', state)
    return output, knowledge, key


def prepared_completion(tmp_path):
    source, knowledge, key = corrected_page(tmp_path)
    output = tmp_path / 'completed'
    continuation.prepare(source, output, completion_contract=completion.CONTRACT, job_ids=['rejected'])
    manifest = read_json(output / 'manifest.json')
    permit = completion.validate_batch(manifest, output)
    return source, output, knowledge, key, manifest, permit


def test_prepare_archives_correction_and_preserves_all_progress_and_evidence(tmp_path):
    source, knowledge, key = corrected_page(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'completed'
    result = continuation.prepare(source, output, completion_contract=completion.CONTRACT, job_ids=['rejected'])
    manifest = read_json(output / 'manifest.json')
    permit = completion.validate_batch(manifest, output)
    old_manifest = read_json(source / 'manifest.json')
    assert result['selected_jobs'] == 1 and result['completion_contract'] == completion.CONTRACT
    assert result['reused'] == 1 and result['total'] == 4
    assert read_json(output / 'progress.json')['jobs'] == read_json(source / 'progress.json')['jobs']
    assert all(path.read_bytes() == value for path, value in before.items())
    assert 'correction' not in manifest and manifest['correction_history'][-1] == old_manifest['correction']
    assert manifest['config'] == old_manifest['config'] and manifest['messages'] == old_manifest['messages']
    assert manifest['gaps'] == old_manifest['gaps']
    selected = next(job for job in manifest['jobs'] if job['id'] == 'rejected')
    assert 'correction' not in selected and selected['correction_history'][-1] == permit['correction']
    assert [job['id'] for job in manifest['jobs'] if job.get('completion')] == ['rejected']
    reused = next(job for job in manifest['jobs'] if job['id'] == 'page')
    assert reused['reuse'] == continuation.locator(knowledge)
    for relative, checksum in permit['inherited'].items():
        assert sha256_file(output / 'jobs/rejected' / relative) == checksum
    assert key in permit['correction']['requests']


def test_completion_preserves_source_review_instead_of_restoring_complete(tmp_path, monkeypatch):
    source, _, _ = corrected_page(tmp_path)
    monkeypatch.setattr(continuation.wiki_delivery, 'inspect_job',
        lambda *args: dict(content_issues=['synthetic source mismatch']))
    output = tmp_path / 'completed'
    result = continuation.prepare(source, output,
        completion_contract=completion.CONTRACT, job_ids=['rejected'])
    state = read_json(output / 'progress.json')
    manifest = read_json(output / 'manifest.json')
    assert result['source_review'] == 1 and result['reused'] == 0
    assert state['counts'] == dict(incomplete=4)
    assert state['jobs']['page']['stage'] == 'incomplete'
    assert 'source review' in state['jobs']['page']['error']
    assert not any(job.get('reuse') for job in manifest['jobs'])
    completion.validate_batch(manifest, output)


@pytest.mark.parametrize('jobs', [None, [], ['unknown'], ['transport'], ['rejected', 'unknown']])
def test_selection_requires_exactly_the_corrected_incomplete_page(tmp_path, jobs):
    source, _, _ = corrected_page(tmp_path)
    with pytest.raises(ValueError):
        continuation.prepare(source, tmp_path / 'completed', completion_contract=completion.CONTRACT, job_ids=jobs)


@pytest.mark.parametrize('option', ['parent', 'reread_contract', 'structured_contract',
    'reference_contract', 'correction_contract', 'request_locations'])
def test_completion_cannot_expand_contracts_or_correction_allowances(tmp_path, option):
    with pytest.raises(ValueError, match='other contracts or allowances'):
        continuation.prepare(tmp_path / 'source', tmp_path / 'completed',
            completion_contract=completion.CONTRACT, job_ids=['rejected'], **{option: 'changed'})


@pytest.mark.parametrize('mutation', ['unsealed', 'unused', 'success_missing', 'unsuccessful',
    'upstream', 'additional_error', 'different_block', 'config', 'source', 'pipeline'])
def test_incomplete_or_changed_source_evidence_cannot_authorize_completion(tmp_path, mutation):
    source, _, key = corrected_page(tmp_path)
    directory = source / 'jobs/rejected'
    if mutation in ('unsealed', 'unused'):
        path = tmp_path / 'batch/ledger.json'
        value = read_json(path)
        value['used']['rejected'] = 2
        if mutation == 'unused':
            value['seal'] = digest({name: item for name, item in value.items() if name != 'seal'})
    elif mutation in ('success_missing', 'unsuccessful'):
        path = directory / 'cache' / f'{key}.json'
        if mutation == 'success_missing':
            path.unlink()
        else:
            value = read_json(path)
            value['policy_attempt'] = 2
    elif mutation in ('upstream', 'additional_error', 'different_block'):
        path = directory / 'progress.json'
        value = read_json(path)
        if mutation == 'upstream':
            value['reread_completed'] = 0
        elif mutation == 'additional_error':
            value['errors'].append(dict(stage='structuring', error='semantic error'))
        else:
            value['errors'][0]['error'] = 'service failure'
    else:
        path = directory / 'manifest.json'
        value = read_json(path)
        if mutation == 'config':
            value['config']['model'] = 'different'
        elif mutation == 'source':
            value['messages']['sha256'] = 'changed'
        else:
            value['generator_sha256'] = 'changed'
    if path.exists():
        write_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        continuation.prepare(source, tmp_path / 'completed', completion_contract=completion.CONTRACT,
            job_ids=['rejected'])


@pytest.mark.parametrize('mutation', ['permit', 'manifest', 'ledger', 'source_success',
    'cache_missing', 'cache_changed', 'failure_missing', 'failure_changed', 'provenance'])
def test_sealed_completion_rejects_tampering_before_any_request(tmp_path, monkeypatch, mutation):
    source, output, _, key, manifest, permit = prepared_completion(tmp_path)
    if mutation == 'manifest':
        manifest['jobs'][0]['account'] = 'other'
    elif mutation == 'provenance':
        monkeypatch.setattr(completion, 'provenance', lambda: dict(changed=True))
    else:
        if mutation == 'permit':
            path = Path(manifest['completion']['permit']['path'])
        elif mutation == 'ledger':
            path = Path(permit['ledger']['path'])
        elif mutation == 'source_success':
            path = source / 'jobs/rejected/cache' / f'{key}.json'
        elif mutation.startswith('cache'):
            path = output / 'jobs/rejected/cache' / f'{key}.json'
        else:
            path = next((output / 'jobs/rejected/failures').glob('*.json'))
        if mutation.endswith('missing'):
            path.unlink()
        else:
            write_json(path, dict(changed=True))
    with pytest.raises((ValueError, FileNotFoundError)):
        completion.validate_batch(manifest, output)


@pytest.mark.parametrize('stage', ['consolidating', 'structuring', 'reconciling'])
@pytest.mark.parametrize('feedback', [False, True])
def test_client_allows_only_same_page_downstream_schema_and_standard_feedback(tmp_path, stage, feedback):
    _, _, _, _, _, permit = prepared_completion(tmp_path)
    client = Client()
    wrapped = completion.CompletionClient(client, permit)
    prompt, schema = downstream(permit['subject'], stage)
    if feedback:
        prompt = wiki_cached.RETRY_PREFIX + json.dumps(dict(error='synthetic invalid reference')) + repair.RETRY_MARKER + prompt
    wrapped.run(prompt, schema)
    assert wrapped.cache_identity() == client.cache_identity()
    assert client.calls == [(prompt, schema)]


@pytest.mark.parametrize('mutation', ['other_page', 'other_template', 'unbounded_schema',
    'changed_consolidation', 'wrong_columns', 'wrong_sources'])
def test_client_blocks_scope_or_schema_changes_without_model_calls(tmp_path, mutation):
    _, _, _, _, _, permit = prepared_completion(tmp_path)
    client = Client()
    prompt, schema = downstream(permit['subject'], 'structuring')
    if mutation == 'other_page':
        prompt, schema = downstream(permit['subject'] | dict(account='other'), 'structuring')
    elif mutation == 'other_template':
        prompt = prompt.replace(wiki_structured.source_prompt('structured_source_v2'), repair.EXTRACT_PROMPT)
    elif mutation == 'unbounded_schema':
        schema = wiki_structured.SCHEMA
    elif mutation == 'changed_consolidation':
        prompt, schema = downstream(permit['subject'])
        prompt = prompt.replace('"F1"', '"F2"')
    elif mutation == 'wrong_columns':
        prompt = prompt.replace('"source_line"', '"other_line"')
    else:
        prefix = repair.REFERENCE_PROMPT + wiki_structured.source_prompt('structured_source_v2')
        payload = json.loads(prompt[len(prefix):])
        payload['source_contract']['authors'] = []
        prompt = prefix + json.dumps(payload, ensure_ascii=False)
    with pytest.raises(ValueError, match='outside selected page downstream completion'):
        completion.CompletionClient(client, permit).run(prompt, schema)
    assert not client.calls


def test_successful_correction_is_reused_but_never_reissued(tmp_path):
    _, output, _, key, _, permit = prepared_completion(tmp_path)
    client = Client()
    wrapped = completion.CompletionClient(client, permit)
    prompt, schema, _ = section_task()
    result = repair._cached_call(wrapped, output / 'jobs/rejected/cache', prompt, schema, lambda value: None)
    assert result == dict(items=[], changes=[change()]) and not client.calls
    with pytest.raises(ValueError, match='outside selected page downstream completion'):
        wrapped.run(prompt, schema)
    assert not client.calls and key == digest([wrapped.cache_identity(), prompt, schema])


@pytest.mark.parametrize('options', [dict(attempts=2), dict(request_timeout_seconds=361)])
def test_run_cannot_change_timeout_or_outer_attempt_budget(tmp_path, options):
    _, output, _, _, _, _ = prepared_completion(tmp_path)
    with pytest.raises(ValueError, match='preserves timeout'):
        wiki_batch.run(output, **options)


def test_batch_runs_only_authorized_page_with_existing_parallel_budget_and_preserves_other_gaps(tmp_path, monkeypatch):
    source, output, knowledge, _, _, permit = prepared_completion(tmp_path)
    before = read_json(source / 'progress.json')['jobs']
    calls, configs = [], []
    def make_client(config):
        configs.append(config)
        return Client()
    monkeypatch.setattr(corrected, 'CodexJudgeClient', make_client)
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', lambda: None)
    def finish(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(directory.name)
        assert 'correction' not in kwargs
        assert kwargs['workers'] == 8
        assert kwargs['client'].permit == permit
        kwargs['client'].run(*downstream(permit['subject']))
        wiki_structured.export(directory / 'content', read_json(knowledge))
        return dict(stage='complete')
    monkeypatch.setattr(repair, 'repair', finish)
    state = wiki_batch.run(output, object_workers=4, request_workers=8, attempts=1,
        request_timeout_seconds=360, profile_requests=True)
    assert calls == ['rejected'] and configs == [read_json(source / 'manifest.json')['config']]
    assert state['counts'] == dict(complete=2, incomplete=2)
    assert state['object_workers'] == 4 and state['stage_workers'] == 8
    assert state['jobs']['page'] == before['page']
    for name in ('transport', 'unknown'):
        assert state['jobs'][name] == before[name]
    assert completion.sealed_read(permit['ledger'])['used']['rejected'] == constraints.LIMIT
    wiki_batch.run(output, object_workers=4, request_workers=8, attempts=1, profile_requests=True)
    assert calls == ['rejected']
