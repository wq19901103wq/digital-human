"""Library coverage, ambiguous names, exact source binding and crash recovery."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event, Lock

import pytest

from src.share.wiki import wiki_batch as batch
from src.infrastructure.storage import LockBusy
from src.infrastructure.storage import read_json
from src.infrastructure.storage import write_json
from test_wiki_repair import row, source


@pytest.mark.parametrize('command,wait,request_workers,retry_transport_only,retry_signals', [
    ('run', False, None, False, None), ('run', True, None, False, None), ('run', False, 8, False, None),
    ('run', False, 8, True, None), ('run', False, 8, True, ['rate_limit']),
    ('revise', False, None, False, None)])
def test_entrypoint_preflights_transport_before_resuming_original_batch(
        tmp_path, monkeypatch, command, wait, request_workers, retry_transport_only, retry_signals):
    from scripts import batch_wiki as entrypoint
    from src.judge import codex_transport
    from src.share.wiki import wiki_revision_batch
    output = tmp_path / 'instances' / 'batch'
    output.mkdir(parents=True)
    manifest = json.dumps({'schema': 'wiki_batch_v1', 'llm_config': {'model': 'gpt-5.6-luna'}})
    (output / 'manifest.json').write_text(manifest)
    calls = []
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(codex_transport, 'settings', lambda: calls.append('preflight'))
    monkeypatch.setattr(codex_transport, 'install', lambda **kwargs: calls.append(('install', kwargs)))
    backend = batch if command == 'run' else wiki_revision_batch
    monkeypatch.setattr(backend, 'run', lambda directory, **kwargs:
                        calls.append(('run', directory, kwargs)) or {'stage': 'complete'})
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', command, '--output', str(output),
                        '--workers', '2', '--object-workers', '2', '--max-jobs', '1',
                        *(['--request-workers', str(request_workers)] if request_workers is not None else []),
                        *(['--retry-transport-only'] if retry_transport_only else []),
                        *(['--retry-transport-signal', 'rate_limit'] if retry_signals else []),
                        *(['--wait'] if wait else [])])
    assert entrypoint.main() == 0
    assert calls[0] == 'preflight'
    assert calls[1][0] == 'install' and calls[2][0] == 'run'
    assert calls[1][1]['expected_root'] == tmp_path
    assert calls[1][1]['feature_format_retry'] is False
    assert calls[1][1]['stop_on_service_failure'] is True
    assert calls[1][1]['receipt_directory'] == output / 'codex_transport_overrides'
    assert len(calls[1][1]['provenance']['wiki_manifest_sha256']) == 64
    assert calls[2][1] == output and calls[2][2]['max_jobs'] == 1
    if command == 'run':
        assert calls[2][2]['wait'] is wait
        assert calls[2][2].get('retry_transport_only', False) is retry_transport_only
        assert calls[2][2].get('retry_transport_signals') == retry_signals
        if request_workers is None:
            assert 'request_workers' not in calls[2][2]
        else:
            assert calls[2][2]['request_workers'] == request_workers
    assert (output / 'manifest.json').read_text() == manifest


@pytest.mark.parametrize('option', [['--request-workers', '8'], ['--retry-transport-only'],
                                    ['--retry-transport-only', '--retry-transport-signal', 'rate_limit'],
                                    ['--job-id', 'selected']])
def test_entrypoint_rejects_shared_requests_for_supplement_before_installation(tmp_path, monkeypatch, option):
    from scripts import batch_wiki as entrypoint
    output = tmp_path / 'instances' / 'batch'
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text('{"schema": "wiki_supplement_v1"}')
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda *args: pytest.fail('no installation'))
    monkeypatch.setattr(entrypoint.wiki_supplement, 'run', lambda *args, **kwargs: pytest.fail('no run'))
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', 'run', '--output', str(output), *option])
    with pytest.raises(SystemExit) as error:
        entrypoint.main()
    assert error.value.code == 2


def test_entrypoint_rejects_signal_filter_without_transport_only_before_installation(tmp_path, monkeypatch):
    from scripts import batch_wiki as entrypoint
    output = tmp_path / 'instances' / 'batch'
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text('{"schema": "wiki_batch_v1"}')
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda *args: pytest.fail('no installation'))
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', 'run', '--output', str(output),
                                              '--retry-transport-signal', 'rate_limit'])
    with pytest.raises(SystemExit) as error:
        entrypoint.main()
    assert error.value.code == 2


def test_entrypoint_configuration_failure_never_starts_batch(tmp_path, monkeypatch):
    from scripts import batch_wiki as entrypoint
    from src.judge import codex_transport
    output = tmp_path / 'instances' / 'batch'
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text('{"schema": "wiki_batch_v1"}')
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    def unavailable():
        raise RuntimeError('configuration unavailable')
    monkeypatch.setattr(codex_transport, 'settings', unavailable)
    monkeypatch.setattr(codex_transport, 'install', lambda **kwargs: pytest.fail('no installation'))
    monkeypatch.setattr(batch, 'run', lambda *args, **kwargs: pytest.fail('no batch execution'))
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', 'run', '--output', str(output)])
    with pytest.raises(RuntimeError, match='configuration unavailable'):
        entrypoint.main()


def test_entrypoint_status_does_not_install_transport(tmp_path, monkeypatch):
    from scripts import batch_wiki as entrypoint
    output = tmp_path / 'instances' / 'batch'
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda *args: pytest.fail('read-only status'))
    monkeypatch.setattr(batch, 'status', lambda directory: {'stage': 'interrupted'})
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', 'status', '--output', str(output)])
    assert entrypoint.main() == 0


@pytest.mark.parametrize('job_ids', [None, ['first', 'second']])
def test_entrypoint_passes_runtime_timeout_and_profile_without_changing_manifest(tmp_path, monkeypatch,
                                                                               job_ids):
    from scripts import batch_wiki as entrypoint
    output = tmp_path / 'instances' / 'batch'
    output.mkdir(parents=True)
    manifest = '{"schema": "wiki_batch_v1", "config": {"timeout_seconds": 360}}'
    (output / 'manifest.json').write_text(manifest)
    calls = []
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda directory: calls.append(directory))
    monkeypatch.setattr(batch, 'run', lambda directory, **kwargs:
                        calls.append(kwargs) or {'stage': 'complete'})
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', 'run', '--output', str(output),
        '--request-workers', '8', '--request-timeout-seconds', '900', '--profile-requests',
        *[argument for job_id in job_ids or [] for argument in ('--job-id', job_id)]])
    assert entrypoint.main() == 0
    assert calls[0] == output
    assert calls[1]['request_workers'] == 8
    assert calls[1]['request_timeout_seconds'] == 900 and calls[1]['profile_requests'] is True
    assert calls[1].get('job_ids') == job_ids
    assert (output / 'manifest.json').read_text() == manifest


@pytest.mark.parametrize('schema,options', [
    ('wiki_batch_v1', ['--request-timeout-seconds', '0']),
    ('wiki_batch_v1', ['--request-timeout-seconds', '-1']),
    ('wiki_supplement_v1', ['--request-timeout-seconds', '900']),
    ('wiki_supplement_v1', ['--profile-requests'])])
def test_entrypoint_rejects_invalid_runtime_options_before_install(tmp_path, monkeypatch, schema, options):
    from scripts import batch_wiki as entrypoint
    output = tmp_path / 'instances' / 'batch'
    output.mkdir(parents=True)
    write_json(output / 'manifest.json', dict(schema=schema))
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda *args: pytest.fail('no installation'))
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', 'run', '--output', str(output), *options])
    with pytest.raises(SystemExit) as error:
        entrypoint.main()
    assert error.value.code == 2


def test_entrypoint_profile_is_offline_and_filters_the_requested_run(tmp_path, monkeypatch):
    from scripts import batch_wiki as entrypoint
    from src.share.wiki import wiki_job_pool
    output = tmp_path / 'instances' / 'batch'
    calls = []
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda *args: pytest.fail('no installation'))
    monkeypatch.setattr(wiki_job_pool, 'request_profile', lambda directory, **kwargs:
                        calls.append((directory, kwargs)) or dict(summary=dict(requests=0)))
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', 'profile', '--output', str(output),
                                              '--run-id', 'selected-run'])
    assert entrypoint.main() == 0
    assert calls == [(output, dict(run_id='selected-run'))]


def inputs(tmp_path):
    library = tmp_path / 'wiki'
    for folder in ('users', 'groups', 'topics'):
        (library / folder).mkdir(parents=True)
    for folder, name in [('users', '甲'), ('users', '同名'), ('users', 'peer2'),
                         ('users', '无来源'), ('groups', '群'), ('topics', '话题')]:
        (library / folder / f'{name}.md').write_text(f'# {name}\n- 一条旧摘要。\n')
    rows = [dict(row('private:self:peer1', 'peer1', 1000), sender='甲', chat_name='甲'),
            dict(row('private:self:peer2', 'peer2', 1001), sender='同名', chat_name='同名'),
            dict(row('private:self:peer3', 'peer3', 1002), sender='同名', chat_name='同名'),
            dict(row('private:other:peer4', 'peer4', 1003), sender='甲', chat_name='甲')]
    messages = source(tmp_path, rows)
    output = tmp_path / 'batch'
    batch.prepare(library, messages, output, 'self', {})
    return library, messages, output


def test_inventory_names_are_leads_and_ambiguous_accounts_never_merge(tmp_path):
    _, _, output = inputs(tmp_path)
    manifest = read_json(output / 'manifest.json')
    assert manifest['summary']['files'] == 6 and len(manifest['jobs']) == 2
    jobs = {j['account']: j for j in manifest['jobs']}
    assert jobs['peer1']['binding'] == 'unverified_name_candidate'
    assert jobs['peer2']['binding'] == 'account_filename'
    assert jobs['peer1']['binding_evidence'][0]['sender_id'] == 'peer1'
    ambiguous = next(g for g in manifest['gaps'] if g['reason'] == 'ambiguous_name')
    assert ambiguous['candidates'] == ['peer2', 'peer3']
    assert manifest['summary']['gaps']['requires_group_or_topic_pipeline'] == 2
    assert 'raw history' not in json.dumps(manifest)


@pytest.mark.parametrize('binding', sorted(batch.VERIFIED_BINDINGS))
def test_batch_prioritizes_confirmed_bindings_stably_without_changing_manifest(tmp_path, monkeypatch,
                                                                             binding):
    _, _, output = inputs(tmp_path)
    manifest = read_json(output / 'manifest.json')
    candidate = next(job for job in manifest['jobs'] if job['binding'] == 'unverified_name_candidate')
    confirmed = next(job for job in manifest['jobs'] if job['binding'] == 'account_filename')
    manifest['jobs'] = [dict(candidate, id='candidate-first'),
                        dict(confirmed, id='confirmed-first', binding=binding),
                        dict(candidate, id='candidate-second'),
                        dict(confirmed, id='confirmed-second', binding=binding)]
    write_json(output / 'manifest.json', manifest)
    frozen = (output / 'manifest.json').read_bytes()
    admitted = []

    def schedule(jobs, execute, **kwargs):
        admitted.extend(job['id'] for job in jobs)
        kwargs['save']()

    monkeypatch.setattr(batch.wiki_job_pool, 'run_jobs', schedule)
    batch.run(output, workers=2, object_workers=4)
    assert admitted == ['confirmed-first', 'confirmed-second', 'candidate-first', 'candidate-second']
    assert (output / 'manifest.json').read_bytes() == frozen


def pre_feedback_pipeline(current):
    return current | {
        'wiki_repair.py': '113b3c3445698773eb571286f6dcba9a1be03456cc0af4e141c7a0fcc5478de8',
        'wiki_structured.py': '4809954a35b934dc678be21d2a20f570a61cb4b9d0aa628584ed7c667c7c9620',
        'wiki_mentions.py': '012dd09f76df3166ba5f754ff1027fa2c276fc163b587a4a713312f2a3b9ec3b',
        'wiki_reconcile.py': '46c2c5d2e5bb4f584baeeca835e4c14ba2b0817cd7ae46021f381f64eb76eccf',
    }


def registered_pipeline(current=None):
    from src.share.wiki import wiki_continuation
    current = current or batch.pipeline()
    current = batch.previous_failure_location_pipeline(current, required_names=current) or current
    return current | wiki_continuation.CORRECTION_BASE


def test_source_feedback_runtime_cannot_silently_accept_registered_old_pipeline():
    previous = registered_pipeline()
    assert previous != batch.pipeline()
    assert not batch.compatible_pipeline(previous)


@pytest.mark.parametrize('pipeline_kind', ['person', 'supplement', 'repair'])
def test_section_evidence_diagnostic_alias_preserves_exact_frozen_dependencies(pipeline_kind):
    from src.share.wiki import wiki_supplement
    current = (wiki_supplement.pipeline() if pipeline_kind == 'supplement' else batch.pipeline())
    if pipeline_kind == 'repair':
        current = {'wiki_repair.py': current['wiki_repair.py']}
    required_names = frozenset(current)
    frozen = batch.previous_section_evidence_pipeline(current, required_names=required_names)
    previous = batch.previous_failure_location_pipeline(current, required_names=required_names)
    assert frozen == previous | {'wiki_repair.py': batch.SECTION_EVIDENCE_DIAGNOSTIC_UPDATES['wiki_repair.py'][0]}
    assert batch.compatible_pipeline(current, current, required_names=required_names)
    assert batch.compatible_pipeline(frozen, current, required_names=required_names)
    location = batch.previous_request_location_pipeline(frozen, required_names=required_names)
    assert batch.compatible_pipeline(location, current, required_names=required_names)
    for name in current:
        assert not batch.compatible_pipeline(frozen | {name: 'changed'}, current, required_names=required_names)
        assert not batch.compatible_pipeline(frozen, current | {name: 'future'}, required_names=required_names)
        partial = {key: value for key, value in current.items() if key != name}
        assert batch.previous_section_evidence_pipeline(partial, required_names=required_names) is None
        assert not batch.compatible_pipeline(frozen, partial, required_names=required_names)
        assert not batch.compatible_pipeline(partial, current, required_names=required_names)
    assert not batch.compatible_pipeline(frozen | {'unexpected.py': 'changed'}, current,
                                         required_names=required_names)


@pytest.mark.parametrize('pipeline_kind', ['person', 'supplement', 'repair'])
def test_failure_location_alias_requires_exact_dependencies(pipeline_kind):
    from src.share.wiki import wiki_supplement
    current = wiki_supplement.pipeline() if pipeline_kind == 'supplement' else batch.pipeline()
    if pipeline_kind == 'repair':
        current = {'wiki_repair.py': current['wiki_repair.py']}
    required_names = frozenset(current)
    frozen = batch.previous_failure_location_pipeline(current, required_names=required_names)
    assert frozen is not None and batch.compatible_pipeline(frozen, current, required_names=required_names)
    for name in current:
        assert not batch.compatible_pipeline(frozen | {name: 'changed'}, current, required_names=required_names)
        assert not batch.compatible_pipeline(frozen, current | {name: 'changed'}, required_names=required_names)
        partial = {key: value for key, value in current.items() if key != name}
        assert batch.previous_failure_location_pipeline(partial, required_names=required_names) is None
    if len(current) > 1:
        for name in batch.FAILURE_LOCATION_DIAGNOSTIC_UPDATES.keys() & current.keys():
            assert not batch.compatible_pipeline(frozen | {name: current[name]}, current,
                                                 required_names=required_names)


@pytest.mark.parametrize('pipeline_kind', ['person', 'supplement'])
def test_feedback_alias_requires_exact_dependency_bytes_and_complete_key_set(pipeline_kind):
    from src.share.wiki import wiki_supplement
    current = registered_pipeline(batch.pipeline() if pipeline_kind == 'person' else wiki_supplement.pipeline())
    required_names = frozenset(current)
    frozen = pre_feedback_pipeline(current)
    before = dict(frozen)
    assert batch.compatible_pipeline(frozen, current, required_names=required_names)
    assert batch.compatible_pipeline(current, current, required_names=required_names)
    for name in current:
        assert not batch.compatible_pipeline(frozen | {name: 'changed'}, current, required_names=required_names)
        assert not batch.compatible_pipeline({key: value for key, value in frozen.items() if key != name}, current,
                                             required_names=required_names)
        assert not batch.compatible_pipeline(frozen, current | {name: 'future-change'}, required_names=required_names)
        missing_current = {key: value for key, value in current.items() if key != name}
        missing_frozen = {key: value for key, value in frozen.items() if key != name}
        assert not batch.compatible_pipeline(frozen, missing_current, required_names=required_names)
        assert not batch.compatible_pipeline(missing_frozen, missing_current, required_names=required_names)
        assert batch.previous_feedback_pipeline(missing_current, required_names=required_names) is None
    assert not batch.compatible_pipeline(frozen | {'unexpected.py': 'changed'}, current, required_names=required_names)
    assert not batch.compatible_pipeline(frozen, current | {'unexpected.py': 'changed'}, required_names=required_names)
    assert frozen == before


def test_feedback_alias_rejects_partial_dependency_snapshots_by_default():
    current = batch.pipeline()
    frozen = pre_feedback_pipeline(current)
    partial_current = {'wiki_repair.py': current['wiki_repair.py']}
    partial_frozen = {'wiki_repair.py': frozen['wiki_repair.py']}
    assert batch.previous_feedback_pipeline(partial_current) is None
    assert not batch.compatible_pipeline(partial_frozen, partial_current)
    assert not batch.compatible_pipeline(partial_current, partial_current)


def test_reread_implementation_alias_requires_a_complete_legacy_snapshot():
    current = registered_pipeline()
    legacy = batch.legacy_reread_pipeline(current)
    assert legacy is not None and batch.compatible_pipeline(legacy, current)
    for name in batch.REREAD_IMPLEMENTATION_UPDATES:
        assert not batch.compatible_pipeline(legacy | {name: current[name]}, current)
    assert not batch.compatible_pipeline(legacy | {'wiki_reconcile.py': 'changed'}, current)


def test_reference_alias_requires_exact_complete_snapshot_and_rejects_mixed_bytes():
    current = registered_pipeline()
    previous = batch.previous_reference_pipeline(current)
    assert previous == {name: pair[0] for name, pair in batch.REFERENCE_IMPLEMENTATION_UPDATES.items()}
    assert batch.compatible_pipeline(previous, current)
    for name in current:
        assert not batch.compatible_pipeline(previous | {name: current[name]}, current)
        assert not batch.compatible_pipeline(previous, current | {name: 'future'})
        partial = {key: value for key, value in current.items() if key != name}
        assert batch.previous_reference_pipeline(partial) is None
    assert not batch.compatible_pipeline(previous | {'unexpected.py': 'unknown'}, current)
    subset = {'wiki_repair.py': current['wiki_repair.py']}
    old_subset = batch.previous_reference_pipeline(subset, required_names=subset)
    assert batch.compatible_pipeline(old_subset, subset, required_names=subset)
    assert not batch.compatible_pipeline(old_subset, subset)


@pytest.mark.parametrize('object_workers', [1, 3])
@pytest.mark.parametrize('pre_feedback', [False, True])
def test_batch_retries_failures_and_resume_skips_completed_requests(tmp_path, monkeypatch,
                                                                  object_workers, pre_feedback):
    if pre_feedback:
        frozen_runtime = registered_pipeline()
        monkeypatch.setattr(batch, 'pipeline', lambda: frozen_runtime)
    _, _, output = inputs(tmp_path)
    manifest_path = output / 'manifest.json'
    if pre_feedback:
        manifest = read_json(manifest_path)
        manifest['pipeline'] = pre_feedback_pipeline(manifest['pipeline'])
        write_json(manifest_path, manifest)
    frozen = manifest_path.read_bytes()
    calls = []
    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(account)
        if len(calls) == 1:
            return dict(stage='incomplete', errors=['temporary'])
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')
    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    state = batch.run(output, object_workers=object_workers, max_jobs=1)
    assert state['stage'] == 'incomplete' and state['counts']['complete'] == 1
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0] == 'peer2'
    state = batch.run(output, object_workers=object_workers)
    assert state['stage'] == 'complete' and state['counts']['complete'] == 2
    assert len(calls) == 3
    batch.run(output, object_workers=object_workers)
    assert len(calls) == 3
    assert manifest_path.read_bytes() == frozen
    completed = next(iter(state['jobs'].values()))
    Path(completed['knowledge']).write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        batch.run(output, object_workers=object_workers)


def test_batch_refuses_changed_raw_source(tmp_path):
    _, messages, output = inputs(tmp_path)
    messages.write_text(messages.read_text() + '\n')
    with pytest.raises(ValueError, match='raw source changed'):
        batch.run(output)


@pytest.mark.parametrize('object_workers', [1, 4])
def test_batch_job_selection_preserves_unselected_gaps_and_reuses_completed_work(
        tmp_path, monkeypatch, object_workers):
    _, _, output = inputs(tmp_path)
    manifest_path = output / 'manifest.json'
    frozen = manifest_path.read_bytes()
    jobs = {job['account']: job for job in read_json(manifest_path)['jobs']}
    selected, unselected = jobs['peer1']['id'], jobs['peer2']['id']
    state = dict(jobs={})
    prior = dict(stage='incomplete', account='peer2', attempts=3, errors=['unresolved source'])
    state['jobs'][unselected] = prior
    write_json(output / 'progress.json', state)
    failure = Path(jobs['peer1']['output']) / 'failures/old.json'
    write_json(failure, dict(attempt=1, policy_attempt=1, retry_budget='format_only_v1'))
    old_failure = failure.read_bytes()
    calls = []

    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        assert failure.read_bytes() == old_failure
        calls.append(account)
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')

    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    state = batch.run(output, object_workers=object_workers, job_ids=[selected, selected], max_jobs=1)
    assert state['stage'] == 'incomplete' and state['counts'] == dict(complete=1, incomplete=1)
    assert state['jobs'][unselected] == prior and state['selected_jobs'] == [selected]
    assert batch.status(output)['selected_jobs'] == [selected]
    batch.run(output, object_workers=object_workers, job_ids=[selected])
    assert calls == ['peer1']
    assert failure.read_bytes() == old_failure and manifest_path.read_bytes() == frozen
    Path(state['jobs'][selected]['knowledge']).write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        batch.run(output, job_ids=[unselected])
    assert calls == ['peer1']


@pytest.mark.parametrize('job_ids,diagnostic', [([], 'job_ids must not be empty'),
                                              (['missing'], 'unknown Wiki jobs')])
def test_batch_rejects_invalid_job_selection_without_changing_progress(tmp_path, monkeypatch,
                                                                     job_ids, diagnostic):
    _, _, output = inputs(tmp_path)
    write_json(output / 'progress.json', dict(stage='incomplete', jobs={}))
    prior = (output / 'progress.json').read_bytes()
    monkeypatch.setattr(batch.wiki_repair, 'repair', lambda *args, **kwargs: pytest.fail('no repair'))
    with pytest.raises(ValueError, match=diagnostic):
        batch.run(output, job_ids=job_ids)
    assert (output / 'progress.json').read_bytes() == prior


@pytest.mark.parametrize('options', [dict(skip_incomplete=True), dict(retry_transport_only=True)])
def test_batch_job_selection_cannot_bypass_retry_filters(tmp_path, monkeypatch, options):
    _, _, output = inputs(tmp_path)
    selected = read_json(output / 'manifest.json')['jobs'][0]['id']
    state = dict(jobs={})
    prior = dict(stage='incomplete', attempts=3, errors=['unresolved source'])
    state['jobs'][selected] = prior
    write_json(output / 'progress.json', state)
    monkeypatch.setattr(batch.wiki_repair, 'repair', lambda *args, **kwargs: pytest.fail('no repair'))
    state = batch.run(output, job_ids=[selected], **options)
    assert state['jobs'][selected] == prior
    assert state['counts'] == dict(incomplete=1)


def test_waiting_batch_reads_latest_progress_and_retries_only_unfinished_jobs(tmp_path, monkeypatch):
    _, _, output = inputs(tmp_path)
    jobs = read_json(output / 'manifest.json')['jobs']
    entered = Event()
    original_lock = batch.file_lock
    calls = []

    @contextmanager
    def observed_lock(path, *, blocking):
        assert blocking is True
        entered.set()
        with original_lock(path, blocking=blocking):
            yield

    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append((account, kwargs['workers']))
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')

    monkeypatch.setattr(batch, 'file_lock', observed_lock)
    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with original_lock(output / '.lock', blocking=False):
            waiting = pool.submit(batch.run, output, wait=True, workers=2, object_workers=4)
            assert entered.wait(10), 'successor did not try to acquire the batch lock'
            assert not waiting.done()
            completed = jobs[0]
            knowledge = Path(completed['output']) / 'content/knowledge.json'
            repair(None, None, Path(completed['output']), completed['account'],
                   completed['self_account'], {}, workers=2)
            calls.clear()
            write_json(output / 'progress.json', dict(stage='incomplete', jobs={
                completed['id']: dict(stage='complete', knowledge=str(knowledge),
                                      sha256=batch.sha256_file(knowledge)),
                jobs[1]['id']: dict(stage='incomplete', error='temporary')}))
        state = waiting.result(timeout=10)
    assert state['stage'] == 'complete' and state['counts'] == {'complete': 2}
    assert state['object_workers'] == 4 and state['workers'] == 2
    assert calls == [(jobs[1]['account'], 2)]


@pytest.mark.parametrize('object_workers', [1, 3])
def test_skip_incomplete_preserves_failure_and_runs_pending_without_charging_job_limit(tmp_path, monkeypatch,
                                                                                     object_workers):
    _, _, output = inputs(tmp_path)
    calls = []

    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(account)
        if len(calls) == 1:
            return dict(stage='incomplete', errors=['semantic gap'])
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')

    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    failed = batch.run(output, attempts=1, max_jobs=1)
    failed_id = next(iter(failed['jobs']))
    failure = dict(failed['jobs'][failed_id])
    state = batch.run(output, object_workers=object_workers, max_jobs=1, skip_incomplete=True)
    assert state['stage'] == 'incomplete'
    assert state['counts'] == {'incomplete': 1, 'complete': 1}
    assert state['jobs'][failed_id] == failure
    assert calls[0] != calls[1]
    batch.run(output, skip_incomplete=True)
    assert len(calls) == 2
    assert batch.run(output)['stage'] == 'complete'
    assert len(calls) == 3 and calls[0] == calls[2]
    completed = state['jobs'][next(job_id for job_id in state['jobs'] if job_id != failed_id)]
    Path(completed['knowledge']).write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        batch.run(output, skip_incomplete=True)


def test_status_distinguishes_live_job_from_interrupted_saved_work(tmp_path, monkeypatch):
    _, _, output = inputs(tmp_path)
    assert batch.status(output)['counts']['pending'] == 2
    jobs = read_json(output / 'manifest.json')['jobs']
    write_json(output / 'progress.json', dict(stage='running', active_job=jobs[0]['id'],
        jobs={j['id']: dict(stage='running') for j in jobs}))
    monkeypatch.setattr(batch, 'locked', lambda _: True)
    live = batch.status(output)
    assert live['running'] and live['counts']['running'] == 1
    assert live['counts']['interrupted'] == 1
    assert live['active']['id'] == jobs[0]['id']
    assert live['active']['progress'] == {}
    monkeypatch.setattr(batch, 'locked', lambda _: False)
    stopped = batch.status(output)
    assert stopped['stage'] == 'interrupted' and stopped['active'] is None
    assert stopped['counts']['running'] == 0 and stopped['counts']['interrupted'] == 2


@pytest.mark.parametrize('stage,completed,reread_completed', [
    ('mentions_extract', 3, 0),
    ('mentions_reread', 4, 2),
])
def test_status_reports_each_active_jobs_mention_progress(
        tmp_path, monkeypatch, stage, completed, reread_completed):
    _, _, output = inputs(tmp_path)
    jobs = read_json(output / 'manifest.json')['jobs']
    active_ids = [job['id'] for job in jobs]
    write_json(output / 'progress.json', dict(stage='running', active_job=active_ids[0],
        active_jobs=active_ids, jobs={job_id: dict(stage='running') for job_id in active_ids}))
    details = {}
    for index, job in enumerate(jobs):
        detail = dict(stage=stage, completed=1, batches=1, failed=0,
            reread_completed=1, mention_batches=4, mention_completed=completed - index,
            mention_reread_completed=max(0, reread_completed - index), updated_at=10 + index)
        details[job['id']] = detail
        write_json(Path(job['output']) / 'progress.json', detail)
    monkeypatch.setattr(batch, 'locked', lambda _: True)
    live = batch.status(output)
    assert live['active']['progress'] == details[active_ids[0]]
    assert {job['id']: job['progress'] for job in live['active_jobs']} == details
    monkeypatch.setattr(batch, 'locked', lambda _: False)
    stopped = batch.status(output)
    assert stopped['active'] is None and stopped['active_jobs'] == []
    assert stopped['counts']['interrupted'] == len(jobs)


@pytest.mark.parametrize('request_workers', [None, 8])
def test_batch_runs_bounded_objects_under_one_lock_and_reports_each_active_job(
        tmp_path, monkeypatch, request_workers):
    from src.judge import corrected
    library, messages, _ = inputs(tmp_path)
    (library / 'users/peer3.md').write_text('# 第三位\n- 一条旧摘要。\n')
    output = tmp_path / 'parallel'
    batch.prepare(library, messages, output, 'self', {})
    manifest = (output / 'manifest.json').read_bytes()
    entered, release, guard = Event(), Event(), Lock()
    active, calls, peak = set(), [], 0
    clients = []

    class Client:
        def __init__(self, config):
            assert config == {}
            clients.append(self)

        def cache_identity(self):
            return dict(model='unchanged')

        def run(self, prompt, schema=None):
            return 'unchanged'

    monkeypatch.setattr(corrected, 'CodexJudgeClient', Client)

    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        nonlocal peak
        with guard:
            active.add(account)
            calls.append((account, kwargs['workers'], kwargs['client']))
            peak = max(peak, len(active))
            if len(active) == 2:
                entered.set()
        assert release.wait(10), 'coordinator did not release active objects'
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        with guard:
            active.remove(account)
        return dict(stage='complete')

    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(batch.run, output, object_workers=2, workers=7,
                              request_workers=request_workers)
        try:
            assert entered.wait(10), 'independent objects did not overlap'
            progress = read_json(output / 'progress.json')
            live = batch.status(output)
            assert len(progress['active_jobs']) == 2
            assert progress['active_job'] in progress['active_jobs']
            assert live['running'] and live['counts']['running'] == 2
            assert live['counts']['pending'] == 1
            assert {j['id'] for j in live['active_jobs']} == set(progress['active_jobs'])
            assert live['object_workers'] == 2 and live['workers'] == 7
            assert live['request_workers'] == request_workers
            assert live['stage_workers'] == (request_workers or 7)
            with pytest.raises(LockBusy):
                batch.run(output, object_workers=2)
        finally:
            release.set()
        state = running.result(timeout=10)
    assert state['stage'] == 'complete' and state['counts']['complete'] == 3
    assert peak == 2 and [workers for _, workers, _ in calls] == [request_workers or 7] * 3
    if request_workers is None:
        assert clients == [] and all(client is None for _, _, client in calls)
    else:
        assert len(clients) == 3
        assert all(client.client in clients for _, _, client in calls)
        assert all(client.slots is calls[0][2].slots for _, _, client in calls)
        assert all(client.cache_identity() == dict(model='unchanged') for _, _, client in calls)
    assert state['active_jobs'] == [] and state['active_job'] is None
    batch.run(output, object_workers=3, request_workers=request_workers)
    assert len(calls) == 3
    assert len(clients) == (3 if request_workers else 0)
    assert (output / 'manifest.json').read_bytes() == manifest


@pytest.mark.parametrize('request_workers', [None, 8])
@pytest.mark.parametrize('profile_requests', [False, True])
def test_runtime_timeout_and_profile_preserve_frozen_inputs_and_complete_reuse(
        tmp_path, monkeypatch, request_workers, profile_requests):
    from src.judge import corrected
    from src.share.wiki import wiki_job_pool
    library, messages, _ = inputs(tmp_path)
    output = tmp_path / 'runtime'
    config = dict(timeout_seconds=360, model='unchanged')
    batch.prepare(library, messages, output, 'self', config)
    frozen = (output / 'manifest.json').read_bytes()
    clients, calls = [], []

    class Client:
        def __init__(self, request_config):
            assert request_config == dict(config, timeout_seconds=900)
            clients.append(self)

        def cache_identity(self):
            return dict(model='unchanged')

        def run(self, prompt, schema=None):
            return 'unchanged'

    monkeypatch.setattr(corrected, 'CodexJudgeClient', Client)

    def repair(wiki, messages, directory, account, self_account, frozen_config, **kwargs):
        assert frozen_config == config
        client = kwargs['client']
        assert client.cache_identity() == dict(model='unchanged')
        assert kwargs['workers'] == (request_workers or 2)
        write_json(directory / 'progress.json', dict(stage='structuring'))
        assert client.run('private') == 'unchanged'
        calls.append(client)
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')

    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    state = batch.run(output, workers=2, object_workers=2, request_workers=request_workers,
                      request_timeout_seconds=900, profile_requests=profile_requests)
    assert state['counts'] == dict(complete=2)
    assert state['request_timeout_seconds'] == 900
    assert state['effective_request_timeout_seconds'] == 900
    assert state['profile_requests'] is profile_requests
    assert bool(state['profile_run_id']) is profile_requests
    live = batch.status(output)
    for field in ('request_timeout_seconds', 'effective_request_timeout_seconds',
                  'profile_requests', 'profile_run_id'):
        assert live[field] == state[field]
    assert len(clients) == len(calls) == 2
    if request_workers:
        assert calls[0].slots is calls[1].slots
    profile = wiki_job_pool.request_profile(output)
    assert profile['summary']['requests'] == (2 if profile_requests else 0)
    if profile_requests:
        assert profile['run_ids'] == [state['profile_run_id']]
        assert profile['by_stage']['structuring']['success'] == 2
    batch.run(output, request_timeout_seconds=1200, profile_requests=True)
    assert len(clients) == len(calls) == 2
    assert (output / 'manifest.json').read_bytes() == frozen


@pytest.mark.parametrize('timeout', [0, -1, True, 900.0, '900'])
def test_runtime_timeout_requires_a_positive_integer_before_loading_batch(tmp_path, timeout):
    with pytest.raises(ValueError, match='request timeout must be a positive integer'):
        batch.run(tmp_path / 'missing', request_timeout_seconds=timeout)


def test_service_block_preserves_parallel_success_and_resumes_only_missing_bodies(tmp_path, monkeypatch):
    from src.judge import codex_transport
    library, messages, _ = inputs(tmp_path)
    (library / 'users/peer3.md').write_text('# 第三位\n- 一条旧摘要。\n')
    output = tmp_path / 'parallel'
    batch.prepare(library, messages, output, 'self', {})
    manifest_path = output / 'manifest.json'
    frozen = manifest_path.read_bytes()
    jobs = {job['account']: job for job in read_json(manifest_path)['jobs']}
    overlap, blocked = Barrier(2), Event()
    details = dict(exit_code=1, terminal_signals=['rate_limit'])
    calls = []

    def available():
        if blocked.is_set():
            raise codex_transport.CodexServiceBlocked(details)

    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append((account, kwargs['workers']))
        if len(calls) <= 2:
            overlap.wait(timeout=10)
            if account == 'peer2':
                blocked.set()
                return dict(stage='incomplete', errors=['transport failed'])
            assert blocked.wait(10)
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')

    monkeypatch.setattr(codex_transport, 'check_available', available)
    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    state = batch.run(output, object_workers=2, workers=2, attempts=3)
    assert state['stage'] == 'blocked' and state['pid'] is None
    assert state['blocker'] == details
    assert state['active_jobs'] == [] and state['active_job'] is None
    failed, succeeded, pending = [jobs[account]['id'] for account in ('peer2', 'peer3', 'peer1')]
    assert state['jobs'][failed]['stage'] == 'interrupted'
    assert state['jobs'][failed]['attempts'] == 1
    assert state['jobs'][succeeded]['stage'] == 'complete'
    assert pending not in state['jobs']
    assert batch.status(output)['counts']['pending'] == 1
    knowledge = Path(state['jobs'][succeeded]['knowledge'])
    original_body = knowledge.read_bytes()
    blocked.clear()
    recovered = batch.run(output, object_workers=2, workers=2, attempts=3)
    assert recovered['stage'] == 'complete' and recovered['pid'] is None
    assert 'blocker' not in recovered and recovered['counts'] == {'complete': 3}
    assert recovered['jobs'][failed]['attempts'] == 2
    assert recovered['jobs'][succeeded] == state['jobs'][succeeded]
    assert knowledge.read_bytes() == original_body
    assert sorted(account for account, _ in calls) == ['peer1', 'peer2', 'peer2', 'peer3']
    assert all(workers == 2 for _, workers in calls)
    assert manifest_path.read_bytes() == frozen


@pytest.mark.parametrize('repair_sha', [
    'ab53268eaf06c574e20f4c8c9c3d76812f7d8db0c82f247100b8f8b766653072',
    '8073f66bb240174c99ea276c087997df898953c91359529c6811a197f5955cdc'])
def test_transport_compatibility_requires_complete_exact_registered_bindings(repair_sha):
    current = registered_pipeline()
    frozen = (batch.previous_xml_export_pipeline(current) or current) | {'wiki_repair.py': repair_sha}
    assert batch.compatible_pipeline(frozen, current)
    assert frozen in (batch.previous_transport_pipeline(current), batch.previous_transport_diagnostic_pipeline(current))
    for name in current:
        assert not batch.compatible_pipeline(frozen | {name: 'changed'}, current)
        assert not batch.compatible_pipeline({key: value for key, value in frozen.items() if key != name}, current)
        assert not batch.compatible_pipeline(frozen, current | {name: 'changed'})
    assert not batch.compatible_pipeline(frozen | {'unexpected.py': 'changed'}, current)
    assert not batch.compatible_pipeline(frozen, current | {'unexpected.py': 'changed'})


@pytest.mark.parametrize('pipeline_kind', ['person', 'supplement'])
def test_xml_export_compatibility_preserves_exact_frozen_dependencies(pipeline_kind):
    from src.share.wiki import wiki_supplement
    current = registered_pipeline(batch.pipeline() if pipeline_kind == 'person' else wiki_supplement.pipeline())
    required_names = frozenset(current)
    frozen = batch.previous_xml_export_pipeline(current, required_names=required_names)
    wrapper = batch.previous_transport_wrapper_pipeline(current, required_names=required_names) or current
    assert frozen == wrapper | {name: pair[0] for name, pair in batch.XML_EXPORT_IMPLEMENTATION_UPDATES.items()}
    assert batch.compatible_pipeline(frozen, current, required_names=required_names)
    for name in current:
        assert not batch.compatible_pipeline(frozen | {name: 'changed'}, current, required_names=required_names)
        assert not batch.compatible_pipeline(frozen, current | {name: 'future'}, required_names=required_names)
        partial = {key: value for key, value in current.items() if key != name}
        assert batch.previous_xml_export_pipeline(partial, required_names=required_names) is None
    reference = batch.previous_reference_pipeline(current, required_names=required_names)
    assert reference is not None and batch.compatible_pipeline(reference, current, required_names=required_names)
    assert not batch.compatible_pipeline(reference | {'wiki_structured.py': current['wiki_structured.py']},
                                         current, required_names=required_names)
    assert not batch.compatible_pipeline(frozen | {'unexpected.py': 'changed'}, current,
                                         required_names=required_names)
    subset = {'wiki_repair.py': current['wiki_repair.py']}
    assert batch.previous_xml_export_pipeline(subset, required_names=subset) is None
    diagnostic = batch.previous_transport_diagnostic_pipeline(subset, required_names=subset)
    assert diagnostic == {'wiki_repair.py': batch.TRANSPORT_DIAGNOSTIC_IMPLEMENTATION_UPDATES['wiki_repair.py'][0]}
    assert batch.compatible_pipeline(diagnostic, subset, required_names=subset)


@pytest.mark.parametrize('pipeline_kind', ['person', 'supplement'])
def test_transport_wrapper_compatibility_rejects_mixed_or_changed_frozen_dependencies(pipeline_kind):
    from src.share.wiki import wiki_supplement
    current = registered_pipeline(batch.pipeline() if pipeline_kind == 'person' else wiki_supplement.pipeline())
    required_names = frozenset(current)
    frozen = batch.previous_transport_wrapper_pipeline(current, required_names=required_names)
    assert frozen == current | {name: pair[0] for name, pair in batch.TRANSPORT_WRAPPER_IMPLEMENTATION_UPDATES.items()}
    assert batch.compatible_pipeline(frozen, current, required_names=required_names)
    xml_export = batch.previous_xml_export_pipeline(current, required_names=required_names)
    assert not batch.compatible_pipeline(xml_export | {'wiki_repair.py': current['wiki_repair.py']}, current,
                                         required_names=required_names)
    for name in current:
        assert not batch.compatible_pipeline(frozen | {name: 'changed'}, current, required_names=required_names)
        assert not batch.compatible_pipeline(frozen, current | {name: 'future'}, required_names=required_names)
        partial = {key: value for key, value in current.items() if key != name}
        assert batch.previous_transport_wrapper_pipeline(partial, required_names=required_names) is None
    assert not batch.compatible_pipeline(frozen | {'unexpected.py': 'changed'}, current,
                                         required_names=required_names)


@pytest.mark.parametrize('object_workers', [1, 3])
@pytest.mark.parametrize('wrapper', ['Codex Judge 执行失败（exit 1）：stderr\n\n',
                                    'Codex Judge 执行失败（exit 1）：\n'])
def test_transport_only_resume_preserves_semantic_failures_and_reuses_completed_pages(tmp_path, monkeypatch,
                                                                                    object_workers, wrapper):
    from test_wiki_repair import transport_error
    _, _, output = inputs(tmp_path)
    jobs = read_json(output / 'manifest.json')['jobs']
    errors = [dict(stage='extracting', batch=0, error=wrapper + str(transport_error()))]
    semantic_errors = errors + [dict(stage='structuring', batch=0, error='unknown entity ref: E1')]
    state = dict(jobs={})
    for job, failures in zip(jobs, (errors, semantic_errors)):
        write_json(Path(job['output']) / 'progress.json', dict(stage='incomplete', errors=failures))
        state['jobs'][job['id']] = dict(stage='incomplete', attempts=1,
            error=f'RuntimeError: repair incomplete: {failures}'[-1600:])
    write_json(output / 'progress.json', state)
    retained = dict(state['jobs'][jobs[1]['id']])
    calls = []
    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(account)
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')
    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    result = batch.run(output, retry_transport_only=True, object_workers=object_workers, attempts=1, max_jobs=1)
    assert result['counts'] == {'complete': 1, 'incomplete': 1}
    assert result['jobs'][jobs[1]['id']] == retained
    assert calls == [jobs[0]['account']]
    batch.run(output, retry_transport_only=True)
    assert calls == [jobs[0]['account']]
    Path(result['jobs'][jobs[0]['id']]['knowledge']).write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        batch.run(output, retry_transport_only=True)


@pytest.mark.parametrize('progress,outer_error', [
    ({}, None),
    (dict(stage='incomplete', errors=[]), 'RuntimeError: repair incomplete: []'),
    (dict(stage='incomplete', errors=['unknown']), None),
    (dict(stage='incomplete', errors=[dict(error='Command timed out')]), None),
    (dict(stage='incomplete', errors=[dict(error='Codex transport diagnostics: {malformed')]), None),
])
def test_transport_only_resume_rejects_unknown_or_missing_failure_evidence(tmp_path, progress, outer_error):
    write_json(tmp_path / 'progress.json', progress)
    assert not batch._transport_incomplete(dict(output=str(tmp_path)), dict(error=outer_error))


def test_transport_only_resume_requires_current_matching_failure_record(tmp_path):
    from test_wiki_repair import transport_error
    errors = [dict(error=str(transport_error()))]
    write_json(tmp_path / 'progress.json', dict(stage='incomplete', errors=errors))
    assert not batch._transport_incomplete(dict(output=str(tmp_path)), dict(error='new unknown failure'))
    assert batch._transport_incomplete(dict(output=str(tmp_path)),
                                      dict(error=f'RuntimeError: repair incomplete: {errors}'[-1600:]))


@pytest.mark.parametrize('signals,non_transport,reason,eligible', [
    (['rate_limit'], None, 'transport_only', ['rate_limit']),
    (['stream_disconnected'], None, 'transport_only', ['stream_disconnected']),
    (['rate_limit', 'stream_disconnected'], None, 'transport_only', []),
    ([], None, 'transport_only', []),
    ([None], None, 'transport_only', []),
    ([['rate_limit']], None, 'transport_only', []),
    (['rate_limit'], 'unknown entity ref: E1', 'non_transport_failure', []),
    (['rate_limit'], 'Codex transport diagnostics: {malformed', 'non_transport_failure', []),
    (['rate_limit'], 'unchanged request exhausted 3 correction attempts', 'non_transport_failure', []),
])
def test_status_reports_full_failure_evidence_without_admitting_unparsed_errors(
        tmp_path, monkeypatch, signals, non_transport, reason, eligible):
    from src.judge.codex_transport import CodexTransportError
    from test_wiki_repair import transport_error
    _, _, output = inputs(tmp_path)
    job = read_json(output / 'manifest.json')['jobs'][0]
    diagnostic = str(CodexTransportError(transport_error().details | dict(terminal_signals=signals)))
    errors = [dict(stage='extracting', batch=1, error=diagnostic)]
    if non_transport is not None:
        errors.insert(0, dict(stage='structuring', batch=0, error=non_transport))
    write_json(Path(job['output']) / 'progress.json', dict(stage='incomplete', errors=errors))
    prior = dict(stage='incomplete', error=f'RuntimeError: repair incomplete: {errors}'[-1600:])
    write_json(output / 'progress.json', dict(stage='incomplete', jobs={job['id']: prior}))
    saved = (output / 'progress.json').read_bytes()
    monkeypatch.setattr(batch.codex_transport, 'check_available', lambda: pytest.fail('read-only status'))
    value = batch.status(output)
    assert value['failure_summary'] == dict(jobs=1, reasons={reason: 1},
        recovered_jobs=0, recovered_errors=0,
        retry_transport_signals={signal: int(signal in eligible)
                                 for signal in sorted(batch.RETRY_TRANSPORT_SIGNALS)})
    detail = value['incomplete_jobs'][0]
    assert detail['id'] == job['id'] and detail['title'] == job['title']
    assert detail['reason'] == reason and detail['retry_transport_signals'] == eligible
    assert detail['terminal_signals'] == sorted({signal for signal in signals if isinstance(signal, str)})
    assert detail['errors'] == ([dict(stage='structuring', batch=0, error=non_transport)]
                                if non_transport is not None else [])
    for signal in batch.RETRY_TRANSPORT_SIGNALS:
        assert batch._transport_incomplete(job, prior, retry_signals={signal}) is (signal in eligible)
    assert (output / 'progress.json').read_bytes() == saved


@pytest.mark.parametrize('detail,error,reason', [
    ({}, None, 'missing_failure_evidence'),
    (dict(stage='incomplete', errors=[]), 'RuntimeError: repair incomplete: []', 'missing_failure_evidence'),
    (dict(stage='incomplete', errors=[dict(error='unknown')]), 'stale failure', 'failure_evidence_mismatch'),
])
def test_status_reports_missing_or_mismatched_failure_evidence(tmp_path, detail, error, reason):
    _, _, output = inputs(tmp_path)
    job = read_json(output / 'manifest.json')['jobs'][0]
    write_json(Path(job['output']) / 'progress.json', detail)
    write_json(output / 'progress.json', dict(jobs={job['id']: dict(stage='incomplete', error=error)}))
    value = batch.status(output)
    assert value['failure_summary']['reasons'] == {reason: 1}
    assert value['incomplete_jobs'][0]['retry_transport_signals'] == []
    assert value['incomplete_jobs'][0]['errors'] == []


@pytest.mark.parametrize('error,result,legacy,wrapped,kind', [
    ('review disposition disagrees with records: F9', {'reviews': []}, False, False, 'output_rejected'),
    ('fact speaker absent from cited messages', {'facts': []}, True, True, 'output_rejected'),
    ("Command '['codex', 'exec']' timed out after 360 seconds", None, False, False,
     'legacy_client_timeout'),
    ('Codex Judge 执行失败（exit 1）：missing field base_instructions', None, False, False,
     'legacy_client_failure'),
    ('unknown failure', None, False, False, 'unknown'),
])
def test_status_diagnoses_existing_failure_records_without_resuming_or_writing(
        tmp_path, monkeypatch, error, result, legacy, wrapped, kind):
    _, _, output = inputs(tmp_path)
    job = read_json(output / 'manifest.json')['jobs'][0]
    directory = Path(job['output'])
    key = 'a' * 64
    record = dict(attempt=1, error=error, result=result, errors=[dict(error=error)],
                  retry_budget='format_only_v1', policy_attempt=3)
    if not legacy:
        record['request_sha256'] = key
    write_json(directory / 'failures' / f'{key}-1.json', record)
    current_error = ('unchanged request exhausted 3 correction attempts; last error: ' + error
                     if wrapped else error)
    errors = [dict(stage='structuring', batch=0, error=current_error)]
    write_json(directory / 'progress.json', dict(stage='incomplete', errors=errors))
    prior = dict(stage='incomplete', error=f'RuntimeError: repair incomplete: {errors}'[-1600:])
    write_json(output / 'progress.json', dict(jobs={job['id']: prior}))
    snapshot = {path: path.read_bytes() for path in output.rglob('*.json')}
    monkeypatch.setattr(batch.codex_transport, 'check_available', lambda: pytest.fail('read-only status'))
    monkeypatch.setattr(batch.wiki_repair, 'repair', lambda *args, **kwargs: pytest.fail('no generation'))
    value = batch.status(output)
    detail = value['incomplete_jobs'][0]
    evidence, = detail['failure_evidence']
    assert detail['reason'] == 'non_transport_failure'
    assert detail['retry_transport_signals'] == []
    assert evidence['kind'] == kind
    assert evidence['request_sha256'] == key
    assert evidence['request_sha256_source'] == ('filename' if legacy else 'record')
    assert evidence['failure_file'] == f'failures/{key}-1.json'
    assert evidence['result_present'] is (result is not None)
    assert evidence['history_complete'] and evidence['attempt'] == 1
    assert evidence['policy_attempt'] == 3 and evidence['exhausted_wrapper'] is wrapped
    assert not evidence['cache_file_present']
    assert value['failure_evidence_summary']['error_kinds'] == {kind: 1}
    assert {path: path.read_bytes() for path in output.rglob('*.json')} == snapshot


@pytest.mark.parametrize('change,reason', [
    ('duplicate', 'ambiguous_failure_record'), ('stale', 'missing_failure_record'),
    ('wrong_key', 'invalid_failure_history'), ('missing', 'missing_failure_record'),
    ('symlink', 'invalid_failure_history'), ('bad_json', 'invalid_failure_history'),
])
def test_failure_evidence_fails_closed_on_ambiguous_stale_or_invalid_records(tmp_path, change, reason):
    key = 'a' * 64
    record = dict(request_sha256=key, attempt=1, result={'facts': []}, error='output rejected')
    path = tmp_path / 'failures' / f'{key}-1.json'
    write_json(path, record)
    if change == 'duplicate':
        write_json(path.with_name('b' * 64 + '-1.json'), record | dict(request_sha256='b' * 64))
    elif change == 'stale':
        write_json(path.with_name(f'{key}-2.json'), record | dict(attempt=2, error='new failure'))
    elif change == 'wrong_key':
        write_json(path, record | dict(request_sha256='b' * 64))
    elif change == 'missing':
        path.unlink()
    elif change == 'symlink':
        original = tmp_path / 'original.json'
        path.rename(original)
        path.symlink_to(original)
    else:
        path.write_text('{')
    item = dict(stage='structuring', batch=0, error='output rejected')
    evidence = batch._failure_evidence(tmp_path, item, batch._failure_records(tmp_path, allow_legacy=True))
    assert evidence['kind'] == 'unknown' and evidence['reason'] == reason
    assert 'request_sha256' not in evidence and 'result_present' not in evidence


def test_failure_evidence_matches_bounded_tail_and_reports_cache_presence_not_validity(tmp_path):
    item, record, path, prior = save_truncated_transport(tmp_path)
    record.update(retry_budget='format_only_v1', policy_attempt=1)
    write_json(path, record)
    cache_file = tmp_path / 'cache' / f'{record["request_sha256"]}.json'
    cache_file.parent.mkdir()
    cache_file.write_text('{')
    detail = batch._incomplete_failure(dict(output=str(tmp_path)), prior, inspect_records=True)
    evidence, = detail['failure_evidence']
    assert detail['reason'] == 'non_transport_failure'
    assert evidence['kind'] == 'recorded_transport_failure'
    assert evidence['match'] == 'tail' and evidence['cache_file_present']
    assert not batch._retryable_failure(detail)
    assert 'cache_valid' not in evidence


def test_legacy_failure_evidence_does_not_relax_transport_recovery(tmp_path):
    item, record, path, prior = save_truncated_transport(tmp_path)
    record.pop('request_sha256')
    write_json(path, record)
    detail = batch._incomplete_failure(dict(output=str(tmp_path)), prior, inspect_records=True)
    assert detail['reason'] == 'non_transport_failure'
    assert not batch._transport_incomplete(dict(output=str(tmp_path)), prior)
    assert detail['failure_evidence'][0]['kind'] == 'recorded_transport_failure'
    assert detail['failure_evidence'][0]['request_sha256_source'] == 'filename'


def test_exhausted_failure_evidence_follows_latest_rejected_output_not_later_transport(tmp_path):
    from test_wiki_repair import transport_error
    key = 'a' * 64
    rejected = dict(request_sha256=key, attempt=1, result={'facts': []}, error='output rejected')
    write_json(tmp_path / 'failures' / f'{key}-1.json', rejected)
    write_json(tmp_path / 'failures' / f'{key}-2.json', rejected | dict(
        attempt=2, result=None, error=str(transport_error())))
    item = dict(stage='structuring', error=(
        'unchanged request exhausted 3 correction attempts; last error: output rejected'))
    evidence = batch._failure_evidence(tmp_path, item, batch._failure_records(tmp_path))
    assert evidence['kind'] == 'output_rejected' and evidence['attempt'] == 1
    assert evidence['latest_recorded_attempt'] == 2 and evidence['exhausted_wrapper']


@pytest.mark.parametrize('wrapped', [False, True])
def test_failure_evidence_matches_complete_singleton_check_without_changing_failure(tmp_path, wrapped):
    key = 'a' * 64
    error = 'account must match the supplied account; source unchanged'
    record = dict(request_sha256=key, attempt=1, result={'entities': []}, error=error,
                  retry_policy='source_feedback_v1', policy_attempt=3)
    path = tmp_path / 'failures' / f'{key}-1.json'
    write_json(path, record)
    before = path.read_bytes()
    current = 'mechanical checks failed: ' + json.dumps([dict(record='entities[1]', error=error)])
    if wrapped:
        current = 'unchanged request exhausted 3 correction attempts; last error: ' + current
    item = dict(stage='structuring', batch=34, error=current)
    evidence = batch._failure_evidence(tmp_path, item, batch._failure_records(tmp_path))
    assert evidence['kind'] == 'output_rejected'
    assert evidence['request_sha256'] == key
    assert evidence['match'] == 'singleton_mechanical_check'
    assert evidence['validation_record'] == 'entities[1]'
    assert evidence['exhausted_wrapper'] is wrapped
    assert evidence['policy_attempt'] == 3 and evidence['retry_policy'] == 'source_feedback_v1'
    assert path.read_bytes() == before
    assert not list((tmp_path / 'cache').glob('*'))
    detail = dict(reason='non_transport_failure', diagnostics=[], failure_evidence=[evidence])
    assert not batch._retryable_failure(detail)


@pytest.mark.parametrize('change', [
    'different_error', 'multiple_checks', 'extra_field', 'empty_record', 'wrong_record_type',
    'incomplete_json', 'missing_result', 'transport', 'stale', 'duplicate',
])
def test_singleton_failure_evidence_rejects_lossy_or_ambiguous_matches(tmp_path, change):
    key = 'a' * 64
    record = dict(request_sha256=key, attempt=1, result={'entities': []}, error='output rejected')
    path = tmp_path / 'failures' / f'{key}-1.json'
    checks = [dict(record='entities[1]', error=record['error'])]
    if change == 'different_error':
        checks[0]['error'] += ' changed'
    elif change == 'multiple_checks':
        checks.append(dict(record='reviews', error='another failure'))
    elif change == 'extra_field':
        checks[0]['extra'] = 'unrecognized'
    elif change == 'empty_record':
        checks[0]['record'] = ''
    elif change == 'wrong_record_type':
        checks[0]['record'] = 1
    elif change == 'missing_result':
        record.pop('result')
    elif change == 'transport':
        record['result'] = None
    write_json(path, record)
    if change == 'stale':
        write_json(path.with_name(f'{key}-2.json'), record | dict(attempt=2, error='new failure'))
    elif change == 'duplicate':
        write_json(path.with_name('b' * 64 + '-1.json'), record | dict(request_sha256='b' * 64))
    current = 'mechanical checks failed: ' + json.dumps(checks)
    if change == 'incomplete_json':
        current = current[:-1]
    item = dict(stage='structuring', error=current)
    evidence = batch._failure_evidence(tmp_path, item, batch._failure_records(tmp_path))
    assert evidence['kind'] == 'unknown'
    assert evidence['reason'] == ('ambiguous_failure_record' if change == 'duplicate'
                                  else 'missing_failure_record')
    assert 'request_sha256' not in evidence


@pytest.mark.parametrize('terminal_signals,retry_signals,expected', [
    (['rate_limit'], {'rate_limit'}, True),
    (['stream_disconnected'], {'rate_limit'}, False),
    (['rate_limit', 'stream_disconnected'], {'rate_limit'}, False),
    (['rate_limit', 'stream_disconnected'], {'rate_limit', 'stream_disconnected'}, True),
    ([], {'rate_limit'}, False),
    ([None], {'rate_limit'}, False),
    ([['rate_limit']], {'rate_limit'}, False),
])
def test_transport_signal_filter_uses_all_nonempty_terminal_signals(tmp_path, terminal_signals,
                                                                  retry_signals, expected):
    from test_wiki_repair import transport_error
    details = transport_error().details | dict(terminal_signals=terminal_signals,
        signals=['rate_limit', 'model_catalog_error'], stderr_signals=['model_catalog_error'])
    errors = [dict(error='Codex Judge 执行失败（exit 1）：\nCodex transport diagnostics: ' + json.dumps(details))]
    write_json(tmp_path / 'progress.json', dict(stage='incomplete', errors=errors))
    prior = dict(error=f'RuntimeError: repair incomplete: {errors}'[-1600:])
    assert batch._transport_incomplete(dict(output=str(tmp_path)), prior, retry_signals=retry_signals) is expected
    errors.append(dict(error='unsupported entity attribution'))
    write_json(tmp_path / 'progress.json', dict(stage='incomplete', errors=errors))
    prior['error'] = f'RuntimeError: repair incomplete: {errors}'[-1600:]
    assert not batch._transport_incomplete(dict(output=str(tmp_path)), prior, retry_signals=retry_signals)


def test_transport_signal_resume_preserves_unrecovered_jobs_and_completed_content(tmp_path, monkeypatch):
    from test_wiki_repair import transport_error
    _, _, output = inputs(tmp_path)
    jobs = read_json(output / 'manifest.json')['jobs']
    state = dict(jobs={})
    for job, signal in zip(jobs, ('stream_disconnected', 'rate_limit')):
        details = transport_error().details | dict(signals=[signal], terminal_signals=[signal])
        errors = [dict(batch=0, error='Codex Judge 执行失败（exit 1）：\nCodex transport diagnostics: '
                       + json.dumps(details))]
        write_json(Path(job['output']) / 'progress.json', dict(stage='incomplete', errors=errors))
        state['jobs'][job['id']] = dict(stage='incomplete', attempts=1,
            error=f'RuntimeError: repair incomplete: {errors}'[-1600:])
    write_json(output / 'progress.json', state)
    retained = dict(state['jobs'][jobs[0]['id']])
    calls = []
    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(account)
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')
    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    options = dict(retry_transport_only=True, retry_transport_signals=['rate_limit'], object_workers=3, attempts=1)
    result = batch.run(output, **options)
    assert result['counts'] == {'incomplete': 1, 'complete': 1}
    assert result['retry_transport_signals'] == ['rate_limit']
    assert result['jobs'][jobs[0]['id']] == retained
    assert calls == [jobs[1]['account']]
    body = Path(result['jobs'][jobs[1]['id']]['knowledge'])
    before = body.read_bytes()
    batch.run(output, **options)
    assert calls == [jobs[1]['account']] and body.read_bytes() == before


def save_truncated_transport(directory, stage='mentions_reread', key='a' * 64):
    from test_wiki_repair import transport_error
    full = 'Codex Judge 执行失败（exit 1）：' + 'stderr ' * 200 + '\n\n' + str(transport_error())
    limit = 800 if stage in ('mentions_extract', 'mentions_reread') else (
        1000 if stage in ('structuring', 'reconciling_attributes') else 600)
    item = dict(batch=0, error=full[-limit:])
    if stage is not None:
        item['stage'] = stage
    record = dict(request_sha256=key, attempt=1, result=None, error=full,
                  errors=[dict(attempt=1, error=full)], retry_policy='source_feedback_v1', policy_attempt=1)
    path = directory / 'failures' / f'{key}-1.json'
    write_json(path, record)
    write_json(directory / 'progress.json', dict(stage='incomplete', errors=[item]))
    prior = dict(stage='incomplete', attempts=1, error=f'RuntimeError: repair incomplete: {[item]}'[-1600:])
    return item, record, path, prior


@pytest.mark.parametrize('stage', [None, 'extracting', 'drafting', 'writing_final', 'rereading',
    'consolidating', 'mentions_extract', 'mentions_reread', 'structuring', 'reconciling_attributes'])
def test_transport_tail_recovery_uses_unique_same_object_complete_evidence(tmp_path, stage):
    item, record, path, prior = save_truncated_transport(tmp_path, stage)
    before = path.read_bytes(), (tmp_path / 'progress.json').read_bytes()
    detail = batch._incomplete_failure(dict(output=str(tmp_path)), prior)
    assert detail['reason'] == 'transport_only'
    assert batch._retryable_failure(detail, {'rate_limit'})
    assert not batch._retryable_failure(detail, {'stream_disconnected'})
    assert detail['recovered_errors'] == [{name: item[name] for name in ('stage', 'batch') if name in item}
        | dict(request_sha256=record['request_sha256'], attempt=1, file=f'failures/{path.name}')]
    assert (path.read_bytes(), (tmp_path / 'progress.json').read_bytes()) == before


@pytest.mark.parametrize('change', [
    {'request_sha256': 'b' * 64}, {'attempt': 2}, {'attempt': True}, {'result': {}},
    {'errors': None}, {'errors': [dict(error='unknown entity ref: E1')]},
    {'failure_kind': 'format'}, {'transport': {}}, {'retry_policy': 'unknown'},
    {'policy_attempt': 4}, {'policy_attempt': True},
    {'retry_budget': 'unknown'}, {'retry_budget': 'format_only_v1'},
    {'retry_budget': 'format_only_v1', 'policy_attempt': 3},
])
def test_transport_tail_recovery_rejects_inconsistent_or_semantic_records(tmp_path, change):
    _, record, path, prior = save_truncated_transport(tmp_path)
    write_json(path, record | change)
    assert not batch._transport_incomplete(dict(output=str(tmp_path)), prior)


def test_transport_tail_recovery_accepts_current_zero_format_budget_without_writing(tmp_path):
    _, record, path, prior = save_truncated_transport(tmp_path)
    diagnostic = batch.wiki_repair._transport_diagnostic(record['error'])
    write_json(path, record | dict(errors=[], failure_kind='transport', transport=diagnostic,
                                  retry_budget='format_only_v1', policy_attempt=0))
    before = path.read_bytes()
    assert batch._transport_incomplete(dict(output=str(tmp_path)), prior)
    assert path.read_bytes() == before


@pytest.mark.parametrize('attempts,expected', [([1, 2, 3], True), ([1, 3], False),
                                            ([2, 3], False), ([3], False)])
@pytest.mark.parametrize('current_budget', [False, True])
def test_transport_tail_recovery_requires_complete_request_history(tmp_path, attempts, expected,
                                                                 current_budget):
    _, record, path, prior = save_truncated_transport(tmp_path)
    path.unlink()
    if current_budget:
        record.update(retry_budget='format_only_v1', policy_attempt=0)
    for attempt in attempts:
        error = record['error'] if attempt == 3 else record['error'].replace('rate_limit', 'stream_disconnected')
        write_json(path.with_name(f"{record['request_sha256']}-{attempt}.json"), record | dict(
            attempt=attempt, error=error, errors=[dict(attempt=attempt, error=error)]))
    assert batch._transport_incomplete(dict(output=str(tmp_path)), prior) is expected


@pytest.mark.parametrize('failure', ['missing', 'bad_json', 'wrong_name', 'duplicate', 'stale',
                                    'other_object', 'symlink', 'history_format'])
def test_transport_tail_recovery_fails_closed_on_missing_or_ambiguous_history(tmp_path, failure):
    _, record, path, prior = save_truncated_transport(tmp_path)
    if failure == 'missing':
        path.unlink()
    elif failure == 'bad_json':
        path.write_text('{')
    elif failure == 'wrong_name':
        path.rename(path.with_name('unknown.json'))
    elif failure == 'duplicate':
        write_json(path.with_name('b' * 64 + '-1.json'), record | dict(request_sha256='b' * 64))
    elif failure in ('stale', 'history_format'):
        write_json(path.with_name('a' * 64 + '-2.json'), record | dict(attempt=2,
            error='unknown entity ref: E1', errors=[dict(error='unknown entity ref: E1')]))
        if failure == 'history_format':
            write_json(path.with_name('a' * 64 + '-3.json'), record | dict(attempt=3))
    elif failure in ('other_object', 'symlink'):
        other = tmp_path / 'other-object' / path.name
        other.parent.mkdir()
        path.rename(other)
        if failure == 'symlink':
            path.symlink_to(other)
    assert not batch._transport_incomplete(dict(output=str(tmp_path)), prior)


@pytest.mark.parametrize('failure', ['wrong_stage', 'short_tail', 'changed_tail', 'outer_mismatch',
                                    'missing_result', 'wrong_exit', 'extra_semantic', 'budget_wrapper'])
def test_transport_tail_recovery_does_not_relax_wrapper_or_progress_contract(tmp_path, failure):
    item, record, path, prior = save_truncated_transport(tmp_path)
    if failure == 'wrong_stage':
        item['stage'] = 'unknown'
    elif failure == 'short_tail':
        item['error'] = item['error'][1:]
    elif failure == 'changed_tail':
        item['error'] = '!' + item['error'][1:]
    elif failure == 'missing_result':
        record.pop('result')
    elif failure == 'wrong_exit':
        record['error'] = record['error'].replace('exit 1', 'exit 2')
    elif failure == 'extra_semantic':
        record['error'] += '\nunknown entity ref: E1'
        item['error'] = record['error'][-800:]
    elif failure == 'budget_wrapper':
        record.update(error='unchanged request exhausted 3 correction attempts; last error: ' + record['error'])
    write_json(path, record)
    write_json(tmp_path / 'progress.json', dict(stage='incomplete', errors=[item]))
    prior['error'] = 'stale' if failure == 'outer_mismatch' else (
        f'RuntimeError: repair incomplete: {[item]}'[-1600:])
    assert not batch._transport_incomplete(dict(output=str(tmp_path)), prior)


def test_status_recovers_tails_once_per_incomplete_object_and_preserves_all_evidence(tmp_path, monkeypatch):
    _, _, output = inputs(tmp_path)
    job = read_json(output / 'manifest.json')['jobs'][0]
    directory = Path(job['output'])
    item, _, _, _ = save_truncated_transport(directory)
    errors = [item, item | dict(batch=1)]
    write_json(directory / 'progress.json', dict(stage='incomplete', errors=errors))
    prior = dict(stage='incomplete', error=f'RuntimeError: repair incomplete: {errors}'[-1600:])
    write_json(output / 'progress.json', dict(jobs={job['id']: prior}))
    snapshot = {path: path.read_bytes() for path in output.rglob('*.json')}
    calls = []
    original = batch._failure_records
    monkeypatch.setattr(batch, '_failure_records', lambda directory: calls.append(directory) or original(directory))
    monkeypatch.setattr(batch.codex_transport, 'check_available', lambda: pytest.fail('read-only status'))
    value = batch.status(output)
    assert calls == [directory]
    assert value['failure_summary']['recovered_jobs'] == 1
    assert value['failure_summary']['recovered_errors'] == 2
    assert value['incomplete_jobs'][0]['retry_transport_signals'] == ['rate_limit']
    assert {path: path.read_bytes() for path in output.rglob('*.json')} == snapshot


@pytest.mark.parametrize('object_workers', [1, 3])
def test_recovered_tail_resume_reuses_successes_and_preserves_format_history(tmp_path, monkeypatch, object_workers):
    from test_wiki_repair import Client, feedback_fixture
    _, _, output = inputs(tmp_path)
    jobs = read_json(output / 'manifest.json')['jobs']
    directory = Path(jobs[0]['output'])
    cache, prompt, schema, parse = feedback_fixture(directory)
    client = Client()
    monkeypatch.setattr(client, 'run', lambda request, schema: client.prompts.append(request)
                        or '{"value": "supported"}')
    key = batch.wiki_repair.digest([client.cache_identity(), prompt, schema])
    _, _, path, prior = save_truncated_transport(directory, key=key)
    before = path.read_bytes()
    success_key = batch.wiki_repair.digest([client.cache_identity(), 'cached source', schema])
    success = cache / f'{success_key}.json'
    write_json(success, dict(request_sha256=success_key, attempts=1, result=dict(value='supported')))
    success_before = success.read_bytes()
    write_json(Path(jobs[1]['output']) / 'progress.json', dict(stage='incomplete', errors=[dict(error='unknown')]))
    retained = dict(stage='incomplete', attempts=1, error="RuntimeError: repair incomplete: [{'error': 'unknown'}]")
    write_json(output / 'progress.json', dict(jobs={jobs[0]['id']: prior, jobs[1]['id']: retained}))
    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        for request in ('cached source', prompt):
            assert batch.wiki_repair._cached_call(client, cache, request, schema, parse) == dict(value='supported')
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')
    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    options = dict(retry_transport_only=True, retry_transport_signals=['rate_limit'],
                   object_workers=object_workers, attempts=1)
    result = batch.run(output, **options)
    assert result['counts'] == {'complete': 1, 'incomplete': 1}
    assert result['jobs'][jobs[1]['id']] == retained
    body = Path(result['jobs'][jobs[0]['id']]['knowledge'])
    body_before = body.read_bytes()
    batch.run(output, **options)
    assert client.prompts == [prompt]
    assert path.read_bytes() == before and success.read_bytes() == success_before
    assert body.read_bytes() == body_before


@pytest.mark.parametrize('transport_only,signals', [
    (False, ['rate_limit']), (True, []), (True, ['unknown']), (True, 'rate_limit'), (True, [None]), (True, 1),
])
def test_batch_rejects_invalid_retry_signal_configuration(tmp_path, transport_only, signals):
    with pytest.raises(ValueError, match='retry_transport_signals'):
        batch.run(tmp_path, retry_transport_only=transport_only, retry_transport_signals=signals)


def test_status_marks_all_saved_active_jobs_interrupted_after_coordinator_exits(tmp_path, monkeypatch):
    _, _, output = inputs(tmp_path)
    jobs = read_json(output / 'manifest.json')['jobs']
    ids = [job['id'] for job in jobs]
    write_json(output / 'progress.json', dict(stage='running', active_job=ids[0], active_jobs=ids,
        object_workers=2, workers=4, jobs={job_id: dict(stage='running') for job_id in ids}))
    monkeypatch.setattr(batch, 'locked', lambda _: True)
    assert batch.status(output)['counts']['running'] == 2
    monkeypatch.setattr(batch, 'locked', lambda _: False)
    stopped = batch.status(output)
    assert stopped['stage'] == 'interrupted' and stopped['counts']['interrupted'] == 2
    assert stopped['active_jobs'] == [] and stopped['active'] is None


def test_batch_refuses_zero_object_workers(tmp_path):
    with pytest.raises(ValueError, match='object_workers'):
        batch.run(tmp_path, object_workers=0)


def test_batch_refuses_invalid_shared_request_limit_before_opening_batch(tmp_path):
    with pytest.raises(ValueError, match='request workers'):
        batch.run(tmp_path, request_workers=0)
