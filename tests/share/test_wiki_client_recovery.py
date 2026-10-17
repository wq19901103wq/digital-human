"""Client recovery preserves evidence and grants no new content correction budget."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from src.infrastructure.storage import read_json, write_json
from src.judge import corrected
from src.share.wiki import wiki_batch, wiki_continuation, wiki_repair, wiki_resumption, wiki_structured
from src.share.wiki import wiki_client_recovery as recovery
from src.share.wiki.wiki_library import digest
from test_wiki_continuation import correction_batch


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('client recovery regression must not make live model requests')
    monkeypatch.setattr(corrected.CodexJudgeClient, '__init__', forbidden)
    monkeypatch.setattr(corrected.CodexJudgeClient, '_run', forbidden)
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', forbidden)


def client_batch(tmp_path):
    batch, knowledge = correction_batch(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    manifest.update(pipeline=wiki_batch.pipeline(), correction=dict(contract='source_feedback_v2',
        limit=3, selected_jobs=0, selected_requests=0))
    write_json(batch / 'manifest.json', manifest)
    error = 'Command synthetic-client timed out after 900 seconds'
    directory = batch / 'jobs/unknown'
    key = 'c' * 64
    for path in (directory / 'failures').glob('*.json'):
        path.unlink()
    for attempt, result, message in ((1, dict(reviews=[]), 'rejected output'), (2, None, error)):
        write_json(directory / 'failures' / f'{key}-{attempt}.json', dict(attempt=attempt,
            request_sha256=key, retry_policy='source_feedback_v1', policy_attempt=attempt,
            retry_budget='format_only_v1', error=message, result=result,
            errors=[dict(attempt=attempt, error=message)]))
    errors = [dict(stage='structuring', batch=0, error=error, request_sha256=key)]
    write_json(directory / 'progress.json', dict(stage='incomplete', errors=errors))
    state = read_json(batch / 'progress.json')
    state['jobs']['unknown'] = dict(stage='incomplete', attempts=2,
        error=f'RuntimeError: repair incomplete: {errors}'[-1600:])
    write_json(batch / 'progress.json', state)
    return batch, knowledge


def prepared_recovery(tmp_path):
    source, knowledge = client_batch(tmp_path)
    output = tmp_path / 'recovery'
    result = wiki_continuation.prepare(source, output,
        client_recovery_contract=recovery.CONTRACT, job_ids=['unknown'])
    manifest = read_json(output / 'manifest.json')
    permit, claimed = recovery.validate_batch(manifest, output)
    assert not claimed
    return source, output, knowledge, result, manifest, permit


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for kind in ('cache', 'failures') for path in (directory / kind).glob('*.json')}


def located_client_batch(tmp_path, monkeypatch):
    from src.share.wiki import wiki_cached
    source, knowledge = client_batch(tmp_path)
    manifest = read_json(source / 'manifest.json')
    job = next(item for item in manifest['jobs'] if item['id'] == 'unknown')
    directory = Path(job['output'])
    errors = read_json(directory / 'progress.json')['errors']
    first = errors[0]
    first.pop('request_sha256')
    errors.append(first | dict(batch=1))
    schema = dict(type='object', required=['answer'], properties=dict(answer=dict(type='string')))
    client = wiki_cached.RequestLocator(manifest['config'], directory)
    keys = [digest([client.cache_identity(), f'task-{batch}', schema]) for batch in range(2)]
    for path in sorted((directory / 'failures').glob('*.json')):
        record = read_json(path)
        for key in keys:
            write_json(path.with_name(path.name.replace('c' * 64, key)),
                       record | dict(request_sha256=key))
        path.unlink()
    write_json(directory / 'progress.json', dict(stage='incomplete', errors=errors))
    state = read_json(source / 'progress.json')
    state['jobs']['unknown']['error'] = f'RuntimeError: repair incomplete: {errors}'[-1600:]
    write_json(source / 'progress.json', state)
    def replay(wiki, messages, staged, account, self_account, config, **kwargs):
        records = []
        for batch in range(2):
            try:
                wiki_repair._cached_call(kwargs['client'], staged / 'cache', f'task-{batch}',
                    schema, lambda value: None)
            except ValueError as exc:
                records.append(wiki_repair.failure_record(exc, stage='structuring', batch=batch))
        return dict(stage='incomplete', errors=records)
    monkeypatch.setattr(wiki_repair, 'repair', replay)
    locations = tmp_path / 'locations'
    wiki_cached.locate_requests(source, locations, ['unknown'])
    receipt = read_json(locations / 'unknown.json')
    return source, knowledge, locations, receipt


def test_exact_offline_locations_disambiguate_all_saved_errors_without_mutation(tmp_path, monkeypatch):
    source, _, locations, receipt = located_client_batch(tmp_path, monkeypatch)
    before = snapshot(source / 'jobs/unknown')
    progress = (source / 'jobs/unknown/progress.json').read_bytes()
    output = tmp_path / 'recovery'
    result = wiki_continuation.prepare(source, output,
        client_recovery_contract=recovery.CONTRACT, job_ids=['unknown'], request_locations=locations)
    manifest = read_json(output / 'manifest.json')
    permit, claimed = recovery.validate_batch(manifest, output)
    assert result['selected_jobs'] == 1 and not claimed
    proof = permit['proofs']['unknown']
    assert proof['request_location']['path'] == str(locations / 'unknown.json')
    assert {item['request_sha256'] for item in proof['failure_evidence']} == set(receipt['result']['requests'])
    assert all(item['kind'] == 'legacy_client_timeout' for item in proof['failure_evidence'])
    assert snapshot(source / 'jobs/unknown') == before
    assert (source / 'jobs/unknown/progress.json').read_bytes() == progress
    receipt['result']['errors'].pop()
    write_json(locations / 'unknown.json', receipt)
    with pytest.raises(ValueError, match='sealed evidence changed'):
        recovery.validate_batch(manifest, output)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate', 'digest', 'model_requests',
    'boolean_requests', 'job', 'stage', 'explicit_key', 'progress', 'history', 'source', 'cache',
    'result', 'verified_sources', 'source_files', 'swapped_keys', 'prompt', 'schema'])
def test_recovery_rejects_changed_or_incomplete_location_evidence_before_grant(tmp_path, monkeypatch, mutation):
    source, _, locations, receipt = located_client_batch(tmp_path, monkeypatch)
    result = receipt['result']
    directory = source / 'jobs/unknown'
    if mutation == 'missing':
        result['errors'].pop()
    elif mutation == 'extra':
        result['requests']['a' * 64] = {}
    elif mutation == 'duplicate':
        result['errors'][1] = result['errors'][0]
    elif mutation == 'digest':
        result['errors'][0]['error'] = 'changed'
    elif mutation in ('model_requests', 'boolean_requests'):
        result['model_requests'] = 1 if mutation == 'model_requests' else False
    elif mutation == 'job':
        result['job_id'] = 'other'
    elif mutation == 'stage':
        result['stage'] = 'complete'
    elif mutation == 'swapped_keys':
        first, second = result['errors']
        first['request_sha256'], second['request_sha256'] = (
            second['request_sha256'], first['request_sha256'])
    elif mutation in ('prompt', 'schema'):
        result['requests'][next(iter(result['requests']))][f'{mutation}_sha256'] = 'b' * 64
    elif mutation in ('explicit_key', 'progress'):
        saved = read_json(directory / 'progress.json')
        saved['errors'][0].update(request_sha256='d' * 64) if mutation == 'explicit_key' else saved.update(changed=True)
        write_json(directory / 'progress.json', saved)
    elif mutation in ('history', 'result'):
        key = next(iter(result['requests']))
        path = directory / 'failures' / f'{key}-2.json'
        saved = read_json(path)
        saved['error' if mutation == 'history' else 'result'] = 'changed'
        write_json(path, saved)
    elif mutation == 'source':
        Path(receipt['source_job']['wiki']['path']).write_text('changed')
    elif mutation == 'cache':
        key = next(iter(result['requests']))
        write_json(directory / 'cache' / f'{key}.json', dict(result={}))
    elif mutation == 'verified_sources':
        receipt['verified_sources'] = {}
    else:
        receipt['source_files'] = {}
    if mutation != 'digest':
        receipt['result_sha256'] = digest(result)
    write_json(locations / 'unknown.json', receipt)
    with pytest.raises(ValueError):
        wiki_continuation.prepare(source, tmp_path / 'recovery',
            client_recovery_contract=recovery.CONTRACT, job_ids=['unknown'], request_locations=locations)
    assert not (source / recovery.GRANT).exists()


def test_recovery_with_missing_location_receipt_does_not_grant(tmp_path):
    source, _ = client_batch(tmp_path)
    with pytest.raises(FileNotFoundError):
        wiki_continuation.prepare(source, tmp_path / 'recovery',
            client_recovery_contract=recovery.CONTRACT, job_ids=['unknown'], request_locations=tmp_path / 'missing')
    assert not (source / recovery.GRANT).exists()


def test_legacy_error_requires_trusted_location_even_when_failure_text_is_unique(tmp_path):
    source, _ = client_batch(tmp_path)
    path = source / 'jobs/unknown/progress.json'
    progress = read_json(path)
    progress['errors'][0].pop('request_sha256')
    write_json(path, progress)
    state = read_json(source / 'progress.json')
    state['jobs']['unknown']['error'] = f'RuntimeError: repair incomplete: {progress["errors"]}'[-1600:]
    write_json(source / 'progress.json', state)
    with pytest.raises(ValueError, match='missing request key requires offline location evidence'):
        recovery.prepare(source, tmp_path / 'recovery', ['unknown'])
    assert not (source / recovery.GRANT).exists()


def test_preserves_all_successes_failure_histories_and_global_correction(tmp_path):
    source, output, _, result, manifest, permit = prepared_recovery(tmp_path)
    assert result['reused'] == 1 and result['selected_jobs'] == 1
    assert manifest['correction'] == read_json(source / 'manifest.json')['correction']
    assert read_json(output / 'progress.json')['jobs'] == read_json(source / 'progress.json')['jobs']
    for job_id in ('rejected', 'transport', 'unknown'):
        assert snapshot(source / 'jobs' / job_id) == snapshot(output / 'jobs' / job_id)
        assert read_json(source / 'jobs' / job_id / 'progress.json') == read_json(
            output / 'jobs' / job_id / 'progress.json')
    assert permit['job_ids'] == ['unknown'] and permit['limit'] == 1
    assert not any(job.get('correction') for job in manifest['jobs'])


@pytest.mark.parametrize('selection', [[], None, ['page'], ['rejected'], ['transport'],
    ['missing'], ['unknown', 'unknown'], ['unknown'] * 6])
def test_rejects_invalid_selection_and_non_client_failures(tmp_path, selection):
    source, _ = client_batch(tmp_path)
    with pytest.raises(ValueError):
        recovery.prepare(source, tmp_path / 'recovery', selection)
    assert not (source / recovery.GRANT).exists()


@pytest.mark.parametrize('option', ['parent', 'reread_contract', 'structured_contract',
    'reference_contract', 'correction_contract', 'completion_contract'])
def test_recovery_cannot_upgrade_other_contracts(tmp_path, option):
    with pytest.raises(ValueError, match='other contracts or allowances'):
        wiki_continuation.prepare(tmp_path / 'missing', tmp_path / 'output',
            client_recovery_contract=recovery.CONTRACT, job_ids=['unknown'], **{option: 'changed'})


@pytest.mark.parametrize('mutation', ['success', 'history_missing', 'unknown_error', 'state_mismatch'])
def test_requires_null_result_contiguous_history_and_matching_saved_errors(tmp_path, mutation):
    source, _ = client_batch(tmp_path)
    directory = source / 'jobs/unknown'
    if mutation == 'success':
        write_json(directory / 'cache' / f'{"c" * 64}.json', dict(result={}))
    elif mutation == 'history_missing':
        (directory / 'failures' / f'{"c" * 64}-1.json').unlink()
    elif mutation == 'unknown_error':
        path = directory / 'failures' / f'{"c" * 64}-2.json'
        write_json(path, read_json(path) | dict(error='unknown'))
    else:
        path = source / 'progress.json'
        value = read_json(path)
        value['jobs']['unknown']['error'] = 'mismatch'
        write_json(path, value)
    with pytest.raises(ValueError, match='evidence'):
        recovery.prepare(source, tmp_path / 'recovery', ['unknown'])


@pytest.mark.parametrize('mutation', ['missing_record', 'missing_result', 'non_null_result', 'success_cache'])
def test_mixed_failures_require_complete_null_result_transport_evidence(tmp_path, mutation):
    from test_wiki_repair import transport_error
    source, _ = client_batch(tmp_path)
    directory = source / 'jobs/unknown'
    key, error = 'd' * 64, str(transport_error())
    path = directory / 'progress.json'
    progress = read_json(path)
    progress['errors'].append(dict(stage='structuring', batch=1, error=error, request_sha256=key))
    write_json(path, progress)
    state = read_json(source / 'progress.json')
    state['jobs']['unknown']['error'] = f'RuntimeError: repair incomplete: {progress["errors"]}'[-1600:]
    write_json(source / 'progress.json', state)
    record = dict(attempt=1, request_sha256=key, error=error, result=None)
    if mutation == 'missing_result':
        record.pop('result')
    elif mutation == 'non_null_result':
        record['result'] = dict(reviews=[])
    elif mutation == 'success_cache':
        write_json(directory / 'cache' / f'{key}.json', dict(result={}))
    if mutation != 'missing_record':
        write_json(directory / 'failures' / f'{key}-1.json', record)
    with pytest.raises(ValueError, match='evidence'):
        recovery.prepare(source, tmp_path / 'recovery', ['unknown'])
    assert not (source / recovery.GRANT).exists()


def test_grant_is_unique_across_ancestor_branches_and_cannot_be_dropped(tmp_path):
    root, _ = client_batch(tmp_path)
    first, second, output = (tmp_path / name for name in ('first', 'second', 'recovery'))
    wiki_continuation.prepare(root, first)
    wiki_continuation.prepare(root, second)
    for batch in (first, second):
        state = read_json(batch / 'progress.json')
        state['jobs']['unknown'] = read_json(root / 'progress.json')['jobs']['unknown']
        write_json(batch / 'progress.json', state)
        write_json(batch / 'jobs/unknown/progress.json', read_json(root / 'jobs/unknown/progress.json'))
    recovery.prepare(first, output, ['unknown'])
    for batch in (root, first, second, output):
        with pytest.raises(ValueError):
            recovery.prepare(batch, tmp_path / f'duplicate-{batch.name}', ['unknown'])
    for prepare in (wiki_continuation.prepare, wiki_resumption.prepare):
        with pytest.raises(ValueError, match='original output'):
            prepare(output, tmp_path / 'new-continuation')


@pytest.mark.parametrize('mutation', ['config', 'job_scope', 'source', 'source_state',
    'progress', 'failure', 'cache', 'grant', 'ledger', 'missing_marker'])
def test_tampering_fails_closed_before_repair(tmp_path, monkeypatch, mutation):
    source, output, _, _, manifest, _ = prepared_recovery(tmp_path)
    if mutation in ('config', 'job_scope', 'missing_marker'):
        if mutation == 'config':
            manifest['config']['model'] = 'different'
        elif mutation == 'job_scope':
            manifest['jobs'][-1]['account'] = 'different'
        else:
            manifest.pop('client_recovery')
        write_json(output / 'manifest.json', manifest)
    else:
        paths = dict(source=source / 'manifest.json', source_state=source / 'progress.json',
            progress=source / 'jobs/unknown/progress.json',
            failure=output / 'jobs/unknown/failures' / f'{"c" * 64}-1.json',
            cache=output / 'jobs/unknown/cache/success.json',
            grant=source / recovery.GRANT, ledger=source / recovery.LEDGER)
        write_json(paths[mutation], dict(changed=True))
    monkeypatch.setattr(wiki_repair, 'repair', lambda *args, **kwargs: pytest.fail('unsealed repair'))
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        wiki_batch.run(output, attempts=1)


def test_claim_is_atomic_persistent_and_failure_or_interrupt_cannot_reissue(tmp_path, monkeypatch):
    _, output, _, _, manifest, permit = prepared_recovery(tmp_path)
    def attempt():
        try:
            recovery.claim(permit, 'unknown')
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _: attempt(), range(4))) == 1
    assert recovery.validate_batch(manifest, output)[1] == {'unknown'}
    with pytest.raises(ValueError):
        recovery.claim(permit, 'rejected')
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', lambda: None)
    assert wiki_batch.run(output, attempts=1)['counts']['complete'] == 1


@pytest.mark.parametrize('options', [dict(attempts=2), dict(attempts=1, retry_transport_only=True),
    dict(attempts=1, skip_incomplete=True), dict(attempts=1, cached_only=True)])
def test_rejects_outer_retry_and_mode_expansion(tmp_path, options):
    _, output, _, _, _, _ = prepared_recovery(tmp_path)
    with pytest.raises(ValueError, match='one live repair attempt'):
        wiki_batch.run(output, **options)


def test_runs_only_selected_unclaimed_page_with_long_timeout_and_original_recipe(tmp_path, monkeypatch):
    source, output, knowledge, _, manifest, _ = prepared_recovery(tmp_path)
    configs, calls = [], []
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', lambda: None)
    def client(config):
        configs.append(config)
        return object()
    monkeypatch.setattr(corrected, 'CodexJudgeClient', client)
    def finish(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(directory.name)
        assert config == manifest['config']
        assert 'correction' not in kwargs and kwargs['workers'] == 8
        assert kwargs['reference_contract'] == 'bounded_refs_v2'
        assert recovery.validate_batch(manifest, output)[1] == {'unknown'}
        wiki_structured.export(directory / 'content', read_json(knowledge))
        return dict(stage='complete')
    monkeypatch.setattr(wiki_repair, 'repair', finish)
    before = snapshot(output / 'jobs/unknown')
    state = wiki_batch.run(output, object_workers=4, request_workers=8, attempts=1,
        request_timeout_seconds=1800, profile_requests=True)
    assert state['counts'] == dict(complete=2, incomplete=2)
    assert calls == ['unknown']
    assert configs == [manifest['config'] | dict(timeout_seconds=1800)]
    assert state['jobs']['unknown']['attempts'] == 3
    assert state['jobs']['rejected'] == read_json(source / 'progress.json')['jobs']['rejected']
    assert snapshot(output / 'jobs/unknown') == before
    wiki_batch.run(output, object_workers=4, request_workers=8, attempts=1)
    assert calls == ['unknown']


def test_service_precheck_does_not_consume_an_opportunity(tmp_path, monkeypatch):
    _, output, _, _, manifest, _ = prepared_recovery(tmp_path)
    def unavailable():
        raise RuntimeError('service unavailable before repair')
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', unavailable)
    with pytest.raises(RuntimeError, match='unavailable'):
        wiki_batch.run(output, attempts=1)
    assert recovery.validate_batch(manifest, output)[1] == set()


def test_existing_rejected_output_allowance_is_not_reset(tmp_path):
    _, output, _, _, _, _ = prepared_recovery(tmp_path)
    class Client:
        def cache_identity(self):
            return dict(model='synthetic')
        def run(self, prompt, schema):
            pytest.fail('exhausted format budget must not call a model')
    def reject(value):
        raise ValueError('rejected output')
    client = Client()
    prompt, schema = 'original request', dict(type='object')
    key = digest([client.cache_identity(), prompt, schema])
    directory = output / 'jobs/unknown'
    for attempt in range(1, wiki_repair.RETRY_LIMIT + 1):
        write_json(directory / 'failures' / f'{key}-{attempt}.json', dict(attempt=attempt,
            request_sha256=key, policy_attempt=attempt, retry_budget='format_only_v1',
            retry_policy=wiki_repair.RETRY_POLICY, error='rejected output', result={'wrong': True}))
    with pytest.raises(ValueError, match='exhausted'):
        wiki_repair._cached_call(client, directory / 'cache', prompt, schema, reject)
