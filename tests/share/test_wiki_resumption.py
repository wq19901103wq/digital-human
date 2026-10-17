"""Ordinary continuations retain failure state without widening model allowances."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from src.infrastructure.storage import LockBusy, file_lock, read_json, write_json
from src.judge import corrected
from src.share.wiki import wiki_batch, wiki_completion, wiki_continuation, wiki_delivery
from src.share.wiki import wiki_prompt, wiki_resumption as resumption, wiki_structured
from test_wiki_completion import prepared_completion
from test_wiki_continuation import correction_batch


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('resumption regression must not make live model requests')
    monkeypatch.setattr(corrected.CodexJudgeClient, '__init__', forbidden)
    monkeypatch.setattr(corrected.CodexJudgeClient, '_run', forbidden)
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', forbidden)


def ordinary_batch(tmp_path):
    batch, knowledge = correction_batch(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    manifest['pipeline'] = wiki_batch.pipeline()
    write_json(batch / 'manifest.json', manifest)
    with file_lock(batch / '.lock'):
        pass
    return batch, knowledge


def snapshot(directory):
    return {path: path.read_bytes() for path in directory.rglob('*') if path.is_file()}


def test_preserves_failed_state_progress_and_budgets_without_live_requests(tmp_path):
    source, _ = ordinary_batch(tmp_path)
    before = snapshot(source)
    output = tmp_path / 'continued'
    result = resumption.prepare(source, output)
    assert result == dict(total=4, reused=1, source_review=0, resumed=3, pending=0,
                         cached_requests=3, failed_requests=9, restored_jobs=0)
    state = read_json(output / 'progress.json')
    assert state['jobs'] == read_json(source / 'progress.json')['jobs']
    for name in ('rejected', 'transport', 'unknown'):
        original = source / 'jobs' / name
        target = output / 'jobs' / name
        for path in original.rglob('*.json'):
            assert (target / path.relative_to(original)).read_bytes() == path.read_bytes()
    assert snapshot(source) == before
    assert wiki_delivery.inspect([output])['summary']['counts'] == {'complete': 1, 'incomplete': 3}


def test_restores_multilevel_omitted_state_and_prefers_latest_progress(tmp_path, monkeypatch):
    source, _ = ordinary_batch(tmp_path)
    first, second, output = (tmp_path / name for name in ('first', 'second', 'continued'))
    wiki_continuation.prepare(source, first)
    first_state = read_json(first / 'progress.json')
    first_state['jobs']['unknown'] = dict(stage='running', error='new interruption')
    write_json(first / 'progress.json', first_state)
    newer = dict(stage='incomplete', errors=[dict(stage='rereading', error='new interruption')])
    write_json(first / 'jobs/unknown/progress.json', newer)
    write_json(first / 'jobs/unknown/cache/new-success.json', dict(result={'new': True}))
    wiki_continuation.prepare(first, second)
    (second / 'jobs/transport/cache/success.json').unlink()
    (second / 'jobs/transport/failures' / f'{"b" * 64}-1.json').unlink()
    before = snapshot(tmp_path)
    calls = []
    locator = wiki_continuation.locator
    def counted(path):
        calls.append(Path(path))
        return locator(path)
    monkeypatch.setattr(wiki_continuation, 'locator', counted)
    result = resumption.prepare(second, output)
    assert result['restored_jobs'] == result['resumed'] == 3
    assert result['pending'] == 0 and result['cached_requests'] == 4 and result['failed_requests'] == 9
    state = read_json(output / 'progress.json')
    assert state['jobs']['unknown']['stage'] == 'incomplete'
    assert state['jobs']['unknown']['error'] == 'new interruption'
    assert state['jobs']['transport'] == read_json(source / 'progress.json')['jobs']['transport']
    assert read_json(output / 'jobs/unknown/progress.json') == newer
    assert read_json(output / 'jobs/transport/progress.json') == read_json(source / 'jobs/transport/progress.json')
    receipts = read_json(output / 'manifest.json')['continuation']['state_sources']
    assert receipts['unknown']['state'] == locator(first / 'progress.json')
    assert receipts['transport']['progress'] == locator(source / 'jobs/transport/progress.json')
    assert calls.count(source / 'progress.json') == 1
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('mutation', ['hash', 'input', 'inventory', 'gaps', 'cycle', 'lost_complete'])
def test_history_rejects_unsealed_changed_or_incompatible_inputs(tmp_path, mutation):
    source, _ = ordinary_batch(tmp_path)
    continued = tmp_path / 'legacy'
    wiki_continuation.prepare(source, continued)
    original = read_json(source / 'manifest.json')
    current = read_json(continued / 'manifest.json')
    if mutation == 'hash':
        original['self_account'] = 'changed'
    elif mutation == 'input':
        original['config']['model'] = 'changed'
    elif mutation == 'inventory':
        original['jobs'][1]['account'] = 'changed'
    elif mutation == 'gaps':
        original['gaps'] = []
    elif mutation == 'cycle':
        current['continuation']['source'] = dict(path=str(continued / 'manifest.json'), sha256='cycle')
        write_json(continued / 'manifest.json', current)
    else:
        state = read_json(continued / 'progress.json')
        state['jobs'].pop('page')
        write_json(continued / 'progress.json', state)
    if mutation in ('hash', 'input', 'inventory', 'gaps'):
        write_json(source / 'manifest.json', original)
        if mutation != 'hash':
            current['continuation']['source'] = wiki_continuation.locator(source / 'manifest.json')
            write_json(continued / 'manifest.json', current)
    output = tmp_path / 'continued'
    with pytest.raises(ValueError):
        resumption.prepare(continued, output)
    assert not (output / 'manifest.json').exists()


def test_conflicting_failure_budget_never_publishes_a_manifest(tmp_path):
    source, _ = ordinary_batch(tmp_path)
    continued = tmp_path / 'legacy'
    wiki_continuation.prepare(source, continued)
    failure = continued / 'jobs/transport/failures' / f'{"b" * 64}-1.json'
    saved = read_json(failure)
    write_json(failure, saved | dict(policy_attempt=99))
    output = tmp_path / 'continued'
    with pytest.raises(ValueError, match='conflicting failure budgets'):
        resumption.prepare(continued, output)
    assert not (output / 'manifest.json').exists()


def test_transport_resume_does_not_run_unknown_or_rejected_outputs(tmp_path, monkeypatch):
    source, knowledge = ordinary_batch(tmp_path)
    output = tmp_path / 'continued'
    resumption.prepare(source, output)
    before = snapshot(source)
    calls = []
    def repair(wiki, messages, directory, account, self_account, config, **options):
        calls.append(directory.name)
        assert config == read_json(source / 'manifest.json')['config']
        assert options['workers'] == 8
        wiki_structured.export(directory / 'content', read_json(knowledge))
        return dict(stage='complete')
    monkeypatch.setattr(wiki_batch.wiki_repair, 'repair', repair)
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', lambda *args, **kwargs: None)
    monkeypatch.setattr(corrected.CodexJudgeClient, '__init__', lambda *args, **kwargs: None)
    result = wiki_batch.run(output, workers=4, object_workers=4, request_workers=8,
                           attempts=1, retry_transport_only=True)
    assert calls == ['transport'] and result['counts'] == {'complete': 2, 'incomplete': 2}
    assert snapshot(source) == before
    assert (output / 'jobs/rejected/failures' / f'{"a" * 64}-3.json').exists()
    assert wiki_delivery.inspect([output])['summary']['counts'] == {'complete': 2, 'incomplete': 2}


@pytest.mark.parametrize('surface', ['source', 'output', 'ancestor'])
def test_resumption_obeys_all_batch_locks(tmp_path, surface):
    source, _ = ordinary_batch(tmp_path)
    ancestor = source
    if surface == 'ancestor':
        source = tmp_path / 'legacy'
        wiki_continuation.prepare(ancestor, source)
    output = tmp_path / 'continued'
    directory = dict(source=source, output=output, ancestor=ancestor)[surface]
    with file_lock(directory / '.lock'), ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(resumption.prepare, source, output)
        with pytest.raises(LockBusy):
            task.result(timeout=10)
    assert not (output / 'manifest.json').exists()


@pytest.mark.parametrize('artifact', ['jobs', 'prompt', 'progress.json'])
def test_output_with_unpublished_artifacts_is_not_merged(tmp_path, artifact):
    source, _ = ordinary_batch(tmp_path)
    output = tmp_path / 'continued'
    if artifact.endswith('.json'):
        write_json(output / artifact, {})
    else:
        (output / artifact).mkdir(parents=True)
    with pytest.raises(ValueError, match='unfinished artifacts'):
        resumption.prepare(source, output)
    assert not (output / 'manifest.json').exists()


@pytest.mark.parametrize('mutation', ['active_correction', 'orphan_completion'])
def test_no_implicit_allowance_upgrade(tmp_path, mutation):
    source, _ = ordinary_batch(tmp_path)
    manifest = read_json(source / 'manifest.json')
    if mutation == 'active_correction':
        manifest['jobs'][1]['correction'] = dict(contract='source_feedback_v2')
    else:
        manifest['jobs'][1]['completion'] = dict(contract=wiki_completion.CONTRACT)
    write_json(source / 'manifest.json', manifest)
    output = tmp_path / 'continued'
    with pytest.raises(ValueError):
        resumption.prepare(source, output)
    assert not (output / 'manifest.json').exists()


@pytest.mark.parametrize('mutation', ['unfinished', 'invalid_content', 'changed_permit', 'none'])
def test_completed_page_permit_is_checked_before_retirement(tmp_path, mutation):
    source, batch, knowledge, _, manifest, permit = prepared_completion(tmp_path)
    if mutation != 'unfinished':
        data = read_json(knowledge)
        data['subject']['name'] = next(job['title'] for job in manifest['jobs'] if job['id'] == 'rejected')
        if mutation == 'invalid_content':
            data['entities'][0]['evidence_refs'] = ['1']
        wiki_structured.export(batch / 'jobs/rejected/content', data)
        completed = batch / 'jobs/rejected/content/knowledge.json'
        exported = wiki_prompt.export(completed, batch / 'prompt/rejected')
        state = read_json(batch / 'progress.json')
        state['jobs']['rejected'] = dict(stage='complete', knowledge=str(completed),
            sha256=wiki_continuation.locator(completed)['sha256'],
            records=exported['records'], packets=exported['packets'])
        write_json(batch / 'progress.json', state)
    if mutation == 'changed_permit':
        path = Path(manifest['completion']['permit']['path'])
        write_json(path, read_json(path) | dict(output='different'))
    before = snapshot(tmp_path)
    output = tmp_path / 'continued'
    if mutation != 'none':
        with pytest.raises(ValueError):
            resumption.prepare(batch, output)
        assert not (output / 'manifest.json').exists()
        return
    result = resumption.prepare(batch, output)
    resumed = read_json(output / 'manifest.json')
    assert 'completion' not in resumed and 'correction' not in resumed
    assert resumed['completion_history'][-1] == manifest['completion']
    assert not any(job.get('completion') or job.get('correction') for job in resumed['jobs'])
    assert result['reused'] == 2 and result['restored_jobs'] == 0
    assert wiki_delivery.inspect([output])['summary']['counts'] == {'complete': 2, 'incomplete': 2}
    assert wiki_completion.validate_batch(manifest, batch) == permit
    assert all(path.read_bytes() == content for path, content in before.items())


def test_explicit_contract_paths_remain_delegated(tmp_path, monkeypatch):
    calls = []
    def prepare(*args, **options):
        calls.append((args, deepcopy(options)))
        return dict(delegated=True)
    monkeypatch.setattr(wiki_continuation, 'prepare', prepare)
    options = dict(completion_contract=wiki_completion.CONTRACT, job_ids=['page'])
    assert resumption.prepare(tmp_path / 'source', tmp_path / 'output', **options) == {'delegated': True}
    assert calls == [((tmp_path / 'source', tmp_path / 'output'), options)]
