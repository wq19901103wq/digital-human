"""Continuation preserves source inventory/artifacts and reuses only safe work."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import pytest

from src.share.wiki import wiki_batch
from src.share.wiki import wiki_cached
from src.share.wiki import wiki_continuation as continuation
from src.share.wiki import wiki_constraints
from src.share.wiki import wiki_delivery
from src.share.wiki.wiki_library import digest
from src.share.wiki import wiki_prompt
from src.share.wiki import wiki_repair
from src.share.wiki import wiki_structured
from src.share.wiki import wiki_supplement
from src.infrastructure.storage import LockBusy
from src.infrastructure.storage import file_lock
from src.infrastructure.storage import read_json
from src.infrastructure.storage import write_json
from test_wiki_delivery import batch_fixture


def prepared(tmp_path):
    batch, knowledge = batch_fixture(tmp_path)
    wiki = tmp_path / 'old.md'
    wiki.write_text('# synthetic old wiki\n')
    manifest = read_json(batch / 'manifest.json')
    manifest.update(messages=continuation.locator(tmp_path / 'messages.jsonl'),
        self_account='self', config={}, max_chars=120000, pipeline=wiki_batch.pipeline())
    manifest['jobs'][0].update(sources=[continuation.locator(wiki)], wiki=continuation.locator(wiki))
    write_json(batch / 'manifest.json', manifest)
    return batch, knowledge


def correction_batch(tmp_path):
    from test_wiki_repair import transport_error
    batch, knowledge = prepared(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    manifest.update(pipeline=manifest['pipeline'] | continuation.CORRECTION_BASE,
        reread_contract='reread_evidence_v2', structured_contract='structured_source_v2',
        reference_contract='bounded_refs_v2', config=dict(provider='codex_cli', model='unchanged',
            reasoning_effort='low', codex_cli_version='synthetic-version', timeout_seconds=360))
    state = read_json(batch / 'progress.json')
    for name, error, result in (
            ('rejected', 'review disposition disagrees with records: F1', dict(reviews=[])),
            ('transport', str(transport_error()), None), ('unknown', 'unclassified client failure', None)):
        job = manifest['jobs'][0] | dict(id=name, output=str(batch / 'jobs' / name))
        manifest['jobs'].append(job)
        directory = Path(job['output'])
        key = {'rejected': 'a', 'transport': 'b', 'unknown': 'c'}[name] * 64
        for attempt in range(1, 4):
            write_json(directory / 'failures' / f'{key}-{attempt}.json', dict(attempt=attempt,
                request_sha256=key, retry_policy='source_feedback_v1', retry_budget='format_only_v1',
                policy_attempt=attempt if result is not None else 0, error=error, result=result,
                errors=[dict(attempt=attempt, error=error)]))
        write_json(directory / 'cache/success.json', dict(result={'existing': True}))
        errors = [dict(stage='structuring', batch=0, error=error)]
        write_json(directory / 'progress.json', dict(stage='incomplete', errors=errors))
        state['jobs'][name] = dict(stage='incomplete', error=f'RuntimeError: repair incomplete: {errors}'[-1600:])
    write_json(batch / 'manifest.json', manifest)
    write_json(batch / 'progress.json', state)
    return batch, knowledge


def located_correction_batch(tmp_path, *, current_pipeline=False):
    batch, knowledge = correction_batch(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    if current_pipeline:
        manifest['pipeline'] = wiki_batch.pipeline()
        write_json(batch / 'manifest.json', manifest)
    job = next(item for item in manifest['jobs'] if item['id'] == 'rejected')
    directory = Path(job['output'])
    prompt = 'synthetic correction request'
    schema = dict(type='object', properties=dict(reviews=dict(
        type='array', items=dict(type='object', properties={}))))
    key = digest([wiki_cached.CachedClient(manifest['config'], directory).cache_identity(), prompt, schema])
    for attempt in range(1, 4):
        path = directory / 'failures' / f'{"a" * 64}-{attempt}.json'
        record = read_json(path)
        write_json(directory / 'failures' / f'{key}-{attempt}.json', record | dict(request_sha256=key))
        write_json(directory / 'failures' / f'{"d" * 64}-{attempt}.json',
                   record | dict(request_sha256='d' * 64))
        path.unlink()
    sources = wiki_cached._sources(job, manifest, None)
    evidence = dict(kind='output_rejected', history_complete=True, policy_attempt=3,
        request_sha256=key, failure_file=f'failures/{key}-3.json', attempt=3)
    receipt = dict(schema='wiki_request_location_v2', source_job=job,
        source_manifest_sha256=digest(manifest), verified_sources=sources,
        provenance=wiki_constraints.provenance(),
        fingerprint=wiki_cached._fingerprint(directory, job, manifest, sources),
        result=dict(requests={key: dict(prompt_sha256=digest(prompt), schema_sha256=digest(schema))},
                    failure_evidence=[evidence]))
    diagnostic = tmp_path / 'diagnostic'
    write_json(diagnostic / 'rejected.json', receipt)
    return batch, knowledge, diagnostic, receipt, prompt, schema


@pytest.mark.parametrize('current_pipeline', [False, True])
def test_located_correction_selects_exact_request_despite_ambiguous_old_error(tmp_path, current_pipeline):
    batch, knowledge, diagnostic, receipt, _, _ = located_correction_batch(
        tmp_path, current_pipeline=current_pipeline)
    manifest, prior = read_json(batch / 'manifest.json'), read_json(batch / 'progress.json')
    job = receipt['source_job']
    assert continuation.correction_requests(job, prior['jobs']['rejected']) == {}
    before = {path: path.read_bytes() for path in batch.rglob('*.json')}
    output = tmp_path / 'continued'
    result = continuation.prepare(batch, output, correction_contract='source_feedback_v2',
                                  request_locations=diagnostic, job_ids=['rejected'])
    assert result['selected_jobs'] == result['selected_requests'] == 1
    current = read_json(output / 'manifest.json')
    assert {item['id'] for item in current['jobs']} == {item['id'] for item in manifest['jobs']}
    assert current['config'] == manifest['config'] and current['gaps'] == manifest['gaps']
    assert read_json(output / 'progress.json')['jobs'] == prior['jobs']
    assert all(path.read_bytes() == content for path, content in before.items())
    selected = next(item for item in current['jobs'] if item['id'] == 'rejected')
    assert set(selected['correction']['requests']) == set(receipt['result']['requests'])
    assert selected['correction']['limit'] == 3
    assert all('correction' not in item for item in current['jobs'] if item['id'] != 'rejected')
    completed = next(item for item in current['jobs'] if item['id'] == 'page')
    assert completed['reuse'] == continuation.locator(knowledge)
    for name in ('cache', 'failures'):
        for path in (Path(job['output']) / name).glob('*.json'):
            assert (Path(selected['output']) / name / path.name).read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match='unused registered'):
        continuation.prepare(batch, tmp_path / 'again', correction_contract='source_feedback_v2',
                             request_locations=diagnostic, job_ids=['rejected'])
    with pytest.raises(ValueError, match='unused registered'):
        continuation.prepare(output, tmp_path / 'reset', correction_contract='source_feedback_v2',
                             request_locations=diagnostic, job_ids=['rejected'])


def test_located_correction_budget_remains_three_and_resume_cannot_reset_it(tmp_path):
    batch, _, diagnostic, _, prompt, schema = located_correction_batch(tmp_path)
    output = tmp_path / 'continued'
    continuation.prepare(batch, output, correction_contract='source_feedback_v2',
                         request_locations=diagnostic, job_ids=['rejected'])
    manifest = read_json(output / 'manifest.json')
    selected = next(item for item in manifest['jobs'] if item['id'] == 'rejected')
    directory = Path(selected['output'])
    write_json(directory / 'manifest.json', dict(correction=selected['correction']))
    class Client(wiki_cached.CachedClient):
        calls = 0

        def run(self, request, schema=None):
            self.calls += 1
            return json.dumps(dict(reviews=[]))

    def reject(raw):
        raise ValueError('synthetic content rejected')

    client = Client(manifest['config'], directory)
    with pytest.raises(ValueError, match='synthetic content rejected'):
        wiki_repair._cached_call(client, directory / 'cache', prompt, schema, reject)
    assert client.calls == 3
    resumed = Client(manifest['config'], directory)
    with pytest.raises(ValueError, match='exhausted 3 correction attempts'):
        wiki_repair._cached_call(resumed, directory / 'cache', prompt, schema, reject)
    assert resumed.calls == 0
    failures = [read_json(path) for path in (directory / 'failures').glob('*.json')]
    assert sorted(record['policy_attempt'] for record in failures
                  if record['retry_policy'] == 'source_feedback_v2') == [1, 2, 3]


@pytest.mark.parametrize('mutation', ['receipt', 'provenance', 'input', 'files', 'evidence',
    'unknown', 'multiple', 'policy', 'result', 'key', 'attempt', 'complete', 'missing_job'])
def test_located_correction_rejects_stale_or_unproven_selection_before_publication(tmp_path, mutation):
    batch, _, diagnostic, receipt, _, _ = located_correction_batch(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    directory = Path(receipt['source_job']['output'])
    evidence = receipt['result']['failure_evidence'][0]
    path = directory / evidence['failure_file']
    job_id = 'rejected'
    if mutation == 'receipt':
        receipt['source_manifest_sha256'] = 'changed'
    elif mutation == 'provenance':
        receipt['provenance'] = dict(changed=True)
    elif mutation == 'input':
        Path(manifest['messages']['path']).write_text('changed source\n')
    elif mutation == 'files':
        write_json(directory / 'cache/success.json', dict(changed=True))
    elif mutation == 'evidence':
        evidence['history_complete'] = False
    elif mutation == 'unknown':
        evidence['kind'] = 'unknown'
    elif mutation == 'multiple':
        receipt['result']['requests']['d' * 64] = dict(prompt_sha256='other')
    elif mutation in ('policy', 'result', 'key', 'attempt'):
        field, value = {'policy': ('retry_policy', 'source_feedback_v2'),
            'result': ('result', None), 'key': ('request_sha256', 'd' * 64),
            'attempt': ('attempt', 2)}[mutation]
        write_json(path, read_json(path) | {field: value})
        receipt['fingerprint'] = wiki_cached._fingerprint(directory, receipt['source_job'],
                                                        manifest, receipt['verified_sources'])
    else:
        job_id = 'page' if mutation == 'complete' else 'missing'
    write_json(diagnostic / 'rejected.json', receipt)
    output = tmp_path / 'continued'
    with pytest.raises(ValueError):
        continuation.prepare(batch, output, correction_contract='source_feedback_v2',
                             request_locations=diagnostic, job_ids=[job_id])
    assert not (output / 'manifest.json').exists()
    assert not (output / 'progress.json').exists()
    assert not (batch / '.correction-source_feedback_v2.json').exists()


@pytest.mark.parametrize('options', [dict(request_locations='diagnostic'),
    dict(job_ids=['rejected']), dict(request_locations='diagnostic', job_ids=['rejected']),
    dict(correction_contract='source_feedback_v2', request_locations='diagnostic', job_ids=[]),
    dict(correction_contract='source_feedback_v2', request_locations='diagnostic',
         job_ids=['rejected', 'rejected'])])
def test_located_correction_requires_explicit_contract_receipt_and_unique_jobs(tmp_path, options):
    batch, _ = correction_batch(tmp_path)
    with pytest.raises(ValueError, match='request locations'):
        continuation.prepare(batch, tmp_path / 'continued', **options)
    assert not (batch / '.correction-source_feedback_v2.json').exists()


def test_explicit_correction_continuation_preserves_inventory_and_runs_only_selected_jobs(tmp_path, monkeypatch):
    batch, knowledge = correction_batch(tmp_path)
    manifest, prior = read_json(batch / 'manifest.json'), read_json(batch / 'progress.json')
    existing = {path: path.read_bytes() for path in batch.rglob('*.json')}
    output = tmp_path / 'continued'
    result = continuation.prepare(batch, output, correction_contract='source_feedback_v2')
    assert result == dict(total=4, reused=1, source_review=0, resumed=3, pending=0,
        cached_requests=3, failed_requests=9, selected_jobs=1, selected_requests=1)
    current = read_json(output / 'manifest.json')
    assert current['jobs'][0]['id'] == 'rejected'
    assert current['config'] == manifest['config'] and current['messages'] == manifest['messages']
    assert current['gaps'] == manifest['gaps'] and current['pipeline'] == wiki_batch.pipeline()
    assert read_json(output / 'progress.json')['jobs'] == prior['jobs']
    assert all(path.read_bytes() == content for path, content in existing.items())
    for job in current['jobs']:
        old = next(item for item in manifest['jobs'] if item['id'] == job['id'])
        assert all(job[key] == old[key] for key in ('account', 'binding', 'sources', 'wiki'))
        if job['id'] == 'page':
            assert job['reuse'] == continuation.locator(knowledge)
            continue
        assert ('correction' in job) is (job['id'] == 'rejected')
        assert (Path(job['output']) / 'progress.json').read_bytes() == (
            Path(old['output']) / 'progress.json').read_bytes()
        assert all((Path(job['output']) / path.relative_to(old['output'])).read_bytes() == path.read_bytes()
                   for path in Path(old['output']).glob('failures/*.json'))
    calls = []
    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(directory.name)
        assert config == manifest['config']
        assert kwargs['correction'] == current['jobs'][0]['correction']
        wiki_structured.export(directory / 'content', read_json(knowledge))
        return dict(stage='complete')
    monkeypatch.setattr(wiki_batch.wiki_repair, 'repair', repair)
    result = wiki_batch.run(output, object_workers=4, attempts=1)
    assert calls == ['rejected']
    assert result['counts'] == dict(complete=2, incomplete=2)
    assert all(result['jobs'][name] == prior['jobs'][name] for name in ('transport', 'unknown'))
    wiki_batch.run(output, object_workers=4, attempts=1)
    assert calls == ['rejected']
    with pytest.raises(ValueError, match='unused registered'):
        continuation.prepare(batch, tmp_path / 'again', correction_contract='source_feedback_v2')
    with pytest.raises(ValueError, match='unused registered'):
        continuation.prepare(output, tmp_path / 'reset', correction_contract='source_feedback_v2')


@pytest.mark.parametrize('mutation', ['pipeline', 'contract', 'policy', 'grant', 'mixed'])
def test_correction_upgrade_rejects_unregistered_or_repeated_allowances(tmp_path, mutation):
    batch, _ = correction_batch(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    if mutation == 'pipeline':
        manifest['pipeline']['wiki_mentions.py'] = 'unknown'
    elif mutation == 'contract':
        manifest['reference_contract'] = 'legacy_v1'
    elif mutation == 'policy':
        failure = batch / 'jobs/rejected/failures' / ('a' * 64 + '-3.json')
        write_json(failure, read_json(failure) | dict(retry_policy='source_feedback_v2'))
    elif mutation == 'grant':
        write_json(batch / '.correction-source_feedback_v2.json', dict(existing=True))
    write_json(batch / 'manifest.json', manifest)
    kwargs = dict(reference_contract='bounded_refs_v2') if mutation == 'mixed' else {}
    with pytest.raises(ValueError, match='correction'):
        continuation.prepare(batch, tmp_path / 'continued', correction_contract='source_feedback_v2', **kwargs)
    assert not (tmp_path / 'continued/manifest.json').exists()


@pytest.mark.parametrize('issue', ['forward', 'entity_anchor', 'self_report'])
@pytest.mark.parametrize('reread_contract', [None, 'reread_evidence_v2'])
@pytest.mark.parametrize('structured_contract', [None, 'structured_source_v2'])
@pytest.mark.parametrize('reference_contract', [None, 'bounded_refs_v2'])
def test_continuation_reuses_checked_pages_and_only_retries_missing_or_unsafe(
        tmp_path, monkeypatch, issue, reread_contract, structured_contract, reference_contract):
    batch, source = prepared(tmp_path)
    manifest, state = read_json(batch / 'manifest.json'), read_json(batch / 'progress.json')
    data = read_json(source)
    if issue == 'forward':
        data['evidence']['1']['message_kind'] = 'forward'
    elif issue == 'entity_anchor':
        data['entities'][0]['evidence_refs'] = ['1']
    else:
        data['events'][0].update(mode='self_report', evidence_refs=['1'])
        data['events'][0]['participants'] = [dict(entity_id=data['subject']['entity_id'], role='发言者')]
    for name in ('bad', 'interrupted', 'pending'):
        job = dict(manifest['jobs'][0], id=name, output=str(batch / 'jobs' / name))
        manifest['jobs'].append(job)
        if name == 'pending':
            continue
        write_json(Path(job['output']) / 'cache/exact-request.json', dict(result={'cached': name}))
        write_json(Path(job['output']) / 'failures/exhausted-1.json', dict(error='source mismatch'))
        if name == 'interrupted':
            state['jobs'][name] = dict(stage='running')
            continue
        wiki_structured.export(Path(job['output']) / 'content', data)
        knowledge = Path(job['output']) / 'content/knowledge.json'
        exported = wiki_prompt.export(knowledge, batch / 'prompt' / name)
        state['jobs'][name] = dict(stage='complete', knowledge=str(knowledge),
            sha256=continuation.locator(knowledge)['sha256'],
            records=exported['records'], packets=exported['packets'])
    write_json(batch / 'manifest.json', manifest)
    write_json(batch / 'progress.json', state)
    before = {p: p.read_bytes() for p in batch.rglob('*') if p.is_file()}
    output = tmp_path / 'continued'
    result = continuation.prepare(batch, output, reread_contract=reread_contract,
                                  structured_contract=structured_contract, reference_contract=reference_contract)
    assert result == dict(total=4, reused=1, source_review=1, resumed=1, pending=1,
                         cached_requests=2, failed_requests=2)
    new = read_json(output / 'manifest.json')
    assert new['gaps'] == manifest['gaps'] and new['jobs'][0]['id'] == 'bad'
    assert new.get('reread_contract', 'legacy_v1') == (reread_contract or 'legacy_v1')
    assert new.get('structured_contract', 'legacy_v1') == (structured_contract or 'legacy_v1')
    assert new.get('reference_contract', 'legacy_v1') == (reference_contract or 'legacy_v1')
    assert new['pipeline'] == wiki_batch.pipeline()
    assert new['continuation']['previous_pipeline'] == manifest['pipeline']
    for job in new['jobs']:
        old = next(j for j in manifest['jobs'] if j['id'] == job['id'])
        assert all(job[k] == old[k] for k in ('account', 'binding', 'sources', 'wiki'))
    assert (output / 'jobs/bad/cache/exact-request.json').read_bytes() == (
        batch / 'jobs/bad/cache/exact-request.json').read_bytes()
    assert (output / 'jobs/bad/failures/exhausted-1.json').read_bytes() == (
        batch / 'jobs/bad/failures/exhausted-1.json').read_bytes()
    assert all(p.read_bytes() == content for p, content in before.items())
    assert wiki_delivery.inspect([output])['summary']['counts'] == {'pending': 3, 'complete': 1}

    calls = []
    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(directory.name)
        assert kwargs['reread_contract'] == (reread_contract or 'legacy_v1')
        assert kwargs['structured_contract'] == (structured_contract or 'legacy_v1')
        assert kwargs['reference_contract'] == (reference_contract or 'legacy_v1')
        # The batch passes the untouched raw inputs, never delivery review text.
        assert Path(wiki) == tmp_path / 'old.md' and Path(messages) == tmp_path / 'messages.jsonl'
        wiki_structured.export(directory / 'content', read_json(source))
        return dict(stage='complete')
    monkeypatch.setattr(wiki_batch.wiki_repair, 'repair', repair)
    assert wiki_batch.run(output)['stage'] == 'complete'
    assert calls == ['bad', 'interrupted', 'pending']
    wiki_batch.run(output)
    assert len(calls) == 3
    assert wiki_delivery.require_valid([output], require_complete=True)['stage'] == 'complete'
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('mutation', ['unknown_contract', 'supplement', 'already_v2',
                                     'mixed_pipeline', 'unregistered_pipeline'])
@pytest.mark.parametrize('contract_name, contract', [('reread_contract', 'reread_evidence_v2'),
                                                   ('structured_contract', 'structured_source_v2'),
                                                   ('reference_contract', 'bounded_refs_v2')])
def test_explicit_contract_upgrade_rejects_unsupported_contracts_and_pipelines(
        tmp_path, mutation, contract_name, contract):
    batch, _ = prepared(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    if mutation == 'unknown_contract':
        contract = 'unknown'
    elif mutation == 'supplement':
        manifest.update(schema='wiki_supplement_v1', pipeline=wiki_supplement.pipeline())
    elif mutation == 'already_v2':
        manifest[contract_name] = contract
    elif mutation == 'mixed_pipeline':
        manifest['pipeline'] = wiki_batch.legacy_reread_pipeline(
            wiki_batch.previous_failure_location_pipeline(wiki_batch.pipeline())
            | continuation.CORRECTION_BASE)
        manifest['pipeline']['wiki_mentions.py'] = wiki_batch.pipeline()['wiki_mentions.py']
    else:
        manifest['pipeline']['wiki_reconcile.py'] = 'unregistered'
    write_json(batch / 'manifest.json', manifest)
    output = tmp_path / 'continued'
    with pytest.raises(ValueError, match='upgrade requires'):
        continuation.prepare(batch, output, **{contract_name: contract})
    assert not (output / 'manifest.json').exists()


@pytest.mark.parametrize('mutation', ['pipeline', 'raw', 'wiki', 'artifact'])
def test_continuation_rejects_nonvalidation_changes_or_corruption(tmp_path, mutation):
    batch, source = prepared(tmp_path)
    if mutation == 'pipeline':
        manifest = read_json(batch / 'manifest.json')
        manifest['pipeline']['wiki_repair.py'] = 'unregistered-change'
        write_json(batch / 'manifest.json', manifest)
    else:
        path = {'raw': tmp_path / 'messages.jsonl', 'wiki': tmp_path / 'old.md', 'artifact': source}[mutation]
        path.write_text('changed')
    with pytest.raises(ValueError):
        continuation.prepare(batch, tmp_path / 'continued')
    assert not (tmp_path / 'continued/manifest.json').exists()


def test_live_source_cannot_be_continued(tmp_path):
    batch, _ = prepared(tmp_path)
    with file_lock(batch / '.lock', blocking=False):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(continuation.prepare, batch, tmp_path / 'continued')
            with pytest.raises(LockBusy):
                future.result()


def test_supplement_continuation_binds_new_parent_and_preserves_gap_accounting(tmp_path, monkeypatch):
    batch, _ = prepared(tmp_path)
    new_parent = tmp_path / 'new-parent'
    continuation.prepare(batch, new_parent)
    supplement = tmp_path / 'supplement'
    supplement.mkdir()
    old = read_json(batch / 'manifest.json')
    index = tmp_path / 'index.sqlite3'
    index.write_bytes(b'synthetic index')
    manifest = deepcopy(old)
    manifest.update(schema='wiki_supplement_v1', pipeline=wiki_supplement.pipeline(),
        parent=continuation.locator(batch / 'manifest.json'), index=continuation.locator(index))
    manifest['jobs'][0]['id'] = 'gap'
    manifest['jobs'][0]['reuse'] = continuation.locator(batch / 'jobs/page/content/knowledge.json')
    manifest['gaps'] = []
    manifest['summary']['gaps'] = {}
    wiki_prompt.export(manifest['jobs'][0]['reuse']['path'], supplement / 'prompt/gap')
    state = read_json(batch / 'progress.json')
    state['jobs'] = {'gap': state['jobs']['page']}
    write_json(supplement / 'manifest.json', manifest)
    write_json(supplement / 'progress.json', state)
    new_supplement = tmp_path / 'new-supplement'
    with pytest.raises(ValueError, match='does not descend'):
        continuation.prepare(supplement, new_supplement, parent=batch)
    assert continuation.prepare(supplement, new_supplement, parent=new_parent)['reused'] == 1
    value = wiki_delivery.inspect([new_parent, new_supplement])
    assert value['summary']['counts'] == {'complete': 2} and value['gaps'] == []
    class Index:
        def __init__(self, *a):
            pass
        def close(self):
            pass
    monkeypatch.setattr(wiki_supplement.wiki_source_index, 'SourceIndex', Index)
    def no_generation(*a, **k):
        raise AssertionError('completed scope must not be generated again')
    monkeypatch.setattr(wiki_supplement.wiki_scope, 'generate', no_generation)
    monkeypatch.setattr(wiki_supplement.wiki_repair, 'repair', no_generation)
    assert wiki_supplement.run(new_supplement)['stage'] == 'complete'
