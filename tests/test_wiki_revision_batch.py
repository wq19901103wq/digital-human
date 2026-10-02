"""Revision batches follow frozen originals and overlay them without double counting."""
from copy import deepcopy
from pathlib import Path
from threading import Barrier

import pytest

from src.bootstrap import wiki_delivery as delivery, wiki_revision_batch as revision_batch
from src.iteration.storage import read_json, write_json
from test_wiki_delivery import batch_fixture
from test_wiki_revision import KeepClient


def prepared(tmp_path):
    source, knowledge = batch_fixture(tmp_path)
    manifest = read_json(source / 'manifest.json')
    data = read_json(knowledge)
    manifest.update(messages=data['coverage'], self_account='self', config={})
    manifest['jobs'][0]['binding'] = 'raw_member_snapshot'
    write_json(source / 'manifest.json', manifest)
    output = tmp_path / 'revision'
    revision_batch.prepare(source, output)
    return source, output, knowledge


def test_run_reuses_completion_preserves_original_and_overlays_one_object(tmp_path):
    source, output, knowledge = prepared(tmp_path)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    client = KeepClient()
    state = revision_batch.run(output, client=client)
    assert state['stage'] == 'complete'
    assert state['jobs']['page']['revision_source']['path'] == str(knowledge)
    assert state['jobs']['page']['revision_counts'] == {'keep': 4}
    calls = len(client.prompts)
    assert revision_batch.run(output, client=client)['stage'] == 'complete'
    assert len(client.prompts) == calls
    assert all(p.read_bytes() == content for p, content in before.items())
    value = delivery.inspect([source, output])
    assert value['summary']['total_jobs'] == 1 and len(value['gaps']) == 1
    assert value['summary']['records'] == 4 and value['summary']['revisions'] == {'complete': 1}
    assert value['objects'][0]['body_status'] == 'complete'
    assert value['jobs'][0]['batch'] == str(output)
    assert value['jobs'][0]['original_batch'] == str(source)
    catalog = tmp_path / 'catalog'
    delivery.write_catalog(value, catalog)
    delivery.require_catalog(catalog, [source, output])


def test_pending_revision_archives_old_body_and_does_not_reuse_its_cards(tmp_path):
    source, output, _ = prepared(tmp_path)
    value = delivery.inspect([source, output])
    assert value['summary']['total_jobs'] == 1 and value['summary']['records'] == 0
    assert value['jobs'][0]['stage'] == 'revision_pending'
    assert not value['jobs'][0]['reuse_eligible']
    assert value['objects'][0]['body_status'] == 'in_progress'
    catalog = tmp_path / 'catalog'
    delivery.write_catalog(value, catalog)
    assert '尚未出发' not in (catalog / 'objects' / (value['objects'][0]['id'] + '.md')).read_text()
    delivery.require_catalog(catalog, [source, output])


@pytest.mark.parametrize('mutation', ['manifest', 'knowledge', 'binding'])
def test_saved_revision_catalog_checks_original_binding(tmp_path, mutation):
    source, output, knowledge = prepared(tmp_path)
    revision_batch.run(output, client=KeepClient())
    catalog = tmp_path / 'catalog'
    delivery.write_catalog(delivery.inspect([source, output]), catalog)
    if mutation == 'manifest':
        manifest = read_json(output / 'manifest.json')
        manifest['jobs'][0]['account'] = 'different'
        write_json(output / 'manifest.json', manifest)
    elif mutation == 'knowledge':
        knowledge.write_text('{}')
    else:
        progress = read_json(source / 'progress.json')
        progress['jobs']['page']['stage'] = 'incomplete'
        write_json(source / 'progress.json', progress)
    with pytest.raises(ValueError, match='changed|matches completed original'):
        delivery.require_catalog(catalog, [source, output])


def test_catalog_remains_valid_when_original_progress_adds_unrelated_state(tmp_path):
    source, output, _ = prepared(tmp_path)
    revision_batch.run(output, client=KeepClient())
    catalog = tmp_path / 'catalog'
    delivery.write_catalog(delivery.inspect([source, output]), catalog)
    progress = read_json(source / 'progress.json')
    progress['jobs']['new-progress-only'] = dict(stage='running')
    write_json(source / 'progress.json', progress)
    delivery.require_catalog(catalog, [source, output])


@pytest.mark.parametrize('mutation', ['manifest', 'pipeline', 'knowledge'])
def test_mutated_inputs_cannot_reuse_revision(tmp_path, mutation):
    source, output, knowledge = prepared(tmp_path)
    client = KeepClient()
    revision_batch.run(output, client=client)
    if mutation == 'manifest':
        manifest = read_json(source / 'manifest.json')
        manifest['jobs'][0]['account'] = 'changed'
        write_json(source / 'manifest.json', manifest)
    elif mutation == 'pipeline':
        manifest = read_json(output / 'manifest.json')
        manifest['pipeline'] = {}
        write_json(output / 'manifest.json', manifest)
    else:
        knowledge.write_text('{}')
    if mutation == 'knowledge':
        assert delivery.inspect([source, output])['jobs'][0]['stage'] == 'invalid'
    calls = len(client.prompts)
    with pytest.raises(ValueError):
        revision_batch.run(output, client=client)
    assert len(client.prompts) == calls
    if mutation == 'knowledge':
        assert read_json(output / 'progress.json')['stage'] == 'incomplete'


def test_source_completion_withdrawal_blocks_resume_without_discarding_valid_cache(tmp_path):
    source, output, _ = prepared(tmp_path)
    client = KeepClient()
    completed = revision_batch.run(output, client=client)
    calls = len(client.prompts)
    original = read_json(source / 'progress.json')
    changed = deepcopy(original)
    changed['jobs']['page']['stage'] = 'incomplete'
    write_json(source / 'progress.json', changed)
    with pytest.raises(ValueError, match='matches completed original'):
        revision_batch.run(output, client=client)
    failed = read_json(output / 'progress.json')
    assert failed['stage'] == 'incomplete'
    assert failed['jobs'] == completed['jobs']
    assert len(client.prompts) == calls
    write_json(source / 'progress.json', original)
    recovered = revision_batch.run(output, client=client)
    assert recovered['stage'] == 'complete' and 'error' not in recovered
    assert len(client.prompts) == calls


@pytest.mark.parametrize('mutation', ['source_changed', 'source_missing', 'config',
                                      'job_changed', 'job_missing'])
def test_invalid_frozen_inputs_clear_completion_and_restoration_reuses_cache(tmp_path, mutation):
    source, output, _ = prepared(tmp_path)
    client = KeepClient()
    completed = revision_batch.run(output, client=client)
    calls = len(client.prompts)
    if mutation.startswith('source_'):
        path = source / 'manifest.json'
    elif mutation.startswith('job_'):
        path = output / 'jobs' / 'page' / 'manifest.json'
    else:
        path = output / 'manifest.json'
    original = path.read_bytes()
    if mutation.endswith('missing'):
        path.unlink()
    else:
        changed = read_json(path)
        changed['config'] = {'model': 'different-model'}
        write_json(path, changed)
    with pytest.raises((OSError, ValueError)):
        revision_batch.run(output, client=client)
    failed = read_json(output / 'progress.json')
    assert failed['stage'] == 'incomplete' and failed['error']
    assert failed['jobs'] == completed['jobs']
    assert len(client.prompts) == calls
    path.write_bytes(original)
    recovered = revision_batch.run(output, client=client)
    assert recovered['stage'] == 'complete' and 'error' not in recovered
    assert len(client.prompts) == calls


def test_overlay_rejects_missing_original_duplicate_revision_and_changed_identity(tmp_path):
    source, output, _ = prepared(tmp_path)
    with pytest.raises(ValueError, match='original batch'):
        delivery.inspect([output])
    other = tmp_path / 'other-revision'
    revision_batch.prepare(source, other)
    with pytest.raises(ValueError, match='one revision'):
        delivery.inspect([source, output, other])
    manifest = read_json(output / 'manifest.json')
    manifest['jobs'][0]['account'] = 'other'
    write_json(output / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='identity'):
        delivery.inspect([source, output])


def test_follow_waits_for_new_source_completion_and_finishes(tmp_path, monkeypatch):
    source, output, _ = prepared(tmp_path)
    finished = read_json(source / 'progress.json')
    write_json(source / 'progress.json', dict(stage='running', jobs={}))
    calls = []
    monkeypatch.setattr(revision_batch.wiki_batch, 'status', lambda _: dict(running=not calls))
    def complete(delay):
        calls.append(delay)
        write_json(source / 'progress.json', finished)
    monkeypatch.setattr(revision_batch.time, 'sleep', complete)
    assert revision_batch.run(output, client=KeepClient(), follow=True)['stage'] == 'complete'
    assert calls == [30]


@pytest.mark.parametrize('kind', ['group', 'topic'])
def test_group_and_topic_revision_export_keep_their_nonperson_subject(tmp_path, kind):
    source, output, knowledge = prepared(tmp_path)
    data = deepcopy(read_json(knowledge))
    data['subject'].update(kind=kind, account='', scope_id='group:self:room')
    job = dict(read_json(output / 'manifest.json')['jobs'][0], account='')
    state = revision_batch.export(output, job, data)
    value = delivery.inspect_job(output, job, state)
    assert value['records'] == 4
    revised = Path(state['knowledge'])
    assert read_json(revised)['subject']['kind'] == kind
    assert '\n类型：' + kind + '\n' in revised.with_name('wiki.md').read_text()
    assert read_json(knowledge)['subject'].get('kind') != kind


@pytest.mark.parametrize('engine', ['roles_v1', 'roles_claims_v2'])
def test_role_revision_follows_prior_revision_and_checks_the_entire_source_chain(tmp_path, engine):
    if engine == 'roles_v1':
        from test_wiki_revision_roles import RoleClient
    else:
        from test_wiki_revision_claims import RoleClient
    source, prior, knowledge = prepared(tmp_path)
    revision_batch.run(prior, client=KeepClient())
    next_batch = tmp_path / 'role-revision'
    revision_batch.prepare(prior, next_batch, engine=engine)
    pending = delivery.inspect([source, prior, next_batch])
    assert pending['summary']['records'] == 0
    assert pending['jobs'][0]['stage'] == 'revision_pending'
    state = revision_batch.run(next_batch, client=RoleClient())
    assert state['stage'] == 'complete'
    prior_knowledge = read_json(prior / 'progress.json')['jobs']['page']['knowledge']
    assert state['jobs']['page']['revision_source']['path'] == prior_knowledge
    value = delivery.inspect([source, prior, next_batch])
    assert value['summary']['total_jobs'] == 1
    assert value['summary']['records'] == 4
    assert value['jobs'][0]['batch'] == str(next_batch)
    assert value['jobs'][0]['original_batch'] == str(source)
    catalog = tmp_path / 'catalog'
    delivery.write_catalog(value, catalog)
    delivery.require_catalog(catalog, [source, prior, next_batch])
    knowledge.write_text('{}')
    assert delivery.inspect([source, prior, next_batch])['jobs'][0]['stage'] == 'invalid'
    with pytest.raises(ValueError, match='changed'):
        delivery.require_catalog(catalog, [source, prior, next_batch])


def test_legacy_runner_compatibility_does_not_alias_semantic_engine_changes(tmp_path):
    frozen = revision_batch.pipeline() | {'wiki_revision_batch.py': revision_batch.LEGACY_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, 'legacy')
    assert not revision_batch.compatible_pipeline(frozen, 'roles_v1')
    frozen['wiki_revision.py'] = 'changed'
    assert not revision_batch.compatible_pipeline(frozen, 'legacy')


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1'])
def test_source_validation_compatibility_preserves_only_exact_generation_inputs(engine):
    frozen = revision_batch.pipeline(engine) | {
        'wiki_revision_batch.py': revision_batch.SOURCE_CHECK_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine)
    frozen['wiki_revision.py'] = 'changed'
    assert not revision_batch.compatible_pipeline(frozen, engine)


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1', 'roles_claims_v2'])
def test_claims_dispatch_compatibility_cannot_change_engine_or_source_bytes(engine):
    frozen = revision_batch.pipeline(engine) | {
        'wiki_revision_batch.py': revision_batch.CLAIMS_DISPATCH_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine) == (engine != 'roles_claims_v2')
    backend = Path(revision_batch.engine_for(engine).__file__).name
    frozen[backend] = 'changed'
    assert not revision_batch.compatible_pipeline(frozen, engine)


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1', 'roles_claims_v2'])
def test_parallel_runner_alias_preserves_all_semantic_bindings(engine):
    frozen = revision_batch.pipeline(engine) | {
        'wiki_revision_batch.py': revision_batch.SERIAL_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine)
    for name in frozen:
        assert not revision_batch.compatible_pipeline(frozen | {name: 'changed'}, engine)


def prepared_people(tmp_path, count=3):
    from src.bootstrap import wiki_repair, wiki_structured
    from test_wiki_structured import META, fixture
    from test_wiki_repair import row, source as write_source
    _, _, template = fixture(tmp_path)
    messages = write_source(tmp_path, [
        message for i in range(count) for message in (
            row(f'private:self:person-{i}', 'self', 1000, '阿林，明天见。'),
            row(f'private:self:person-{i}', f'person-{i}', 1001, '小周，你来定时间。'),
            row(f'private:self:person-{i}', f'person-{i}', 1002, '有人叫我“林老师”，我不喜欢。'))])
    source, output = tmp_path / 'source', tmp_path / 'parallel-revision'
    jobs, progress = [], {}
    for i in range(count):
        account = f'person-{i}'
        rows, coverage = wiki_repair.select_history(messages, account, 'self')
        data = deepcopy(template)
        data['entities'][0]['account'] = account
        for record in data['entities'] + data['addresses']:
            record['lines'] = [line + 3 * i for line in record['lines']]
        for record in data['addresses']:
            record['utterance_line'] += 3 * i
        compiled = wiki_structured.compile_records(dict(META, account=account),
            [(data, {'F1': 'fact-1'})], rows, coverage)
        job = dict(id=account, title=account, kind='person', account=account,
            self_account='self', binding='raw_member_snapshot', output=str(source / 'jobs' / account))
        jobs.append(job)
        progress[account] = revision_batch.export(source, job, compiled)
    write_json(source / 'manifest.json', dict(schema='wiki_batch_v1', jobs=jobs, gaps=[],
        messages=coverage, self_account='self', config={}, summary={}))
    write_json(source / 'progress.json', dict(stage='complete', jobs=progress, total=count))
    revision_batch.prepare(source, output)
    return source, output


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1', 'roles_claims_v2'])
def test_incomplete_scheduling_alias_requires_every_semantic_dependency(engine):
    frozen = revision_batch.pipeline(engine) | {
        'wiki_revision_batch.py': revision_batch.OBJECT_POOL_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine)
    for name in frozen:
        assert not revision_batch.compatible_pipeline(frozen | {name: 'changed'}, engine)


def test_skip_incomplete_advances_new_and_interrupted_jobs_without_retrying_failure(tmp_path, monkeypatch):
    _, output = prepared_people(tmp_path)
    original = revision_batch.wiki_revision.revise

    def fail_first(*args):
        if args[2]['subject']['account'] == 'person-0':
            raise ValueError('unresolved evidence')
        return original(*args)

    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', fail_first)
    client = KeepClient()
    failed = revision_batch.run(output, client=client, attempts=1, max_jobs=1)
    failure = deepcopy(failed['jobs']['person-0'])
    assert failure['stage'] == 'incomplete'
    advanced = revision_batch.run(output, client=client, max_jobs=1, skip_incomplete=True)
    assert advanced['counts'] == {'incomplete': 1, 'complete': 1}
    assert advanced['jobs']['person-0'] == failure
    advanced['jobs']['person-2'] = dict(stage='running', account='person-2', attempts=3)
    write_json(output / 'progress.json', advanced)
    resumed = revision_batch.run(output, client=client, skip_incomplete=True)
    assert resumed['counts'] == {'incomplete': 1, 'complete': 2}
    assert resumed['jobs']['person-2']['attempts'] == 4
    assert resumed['jobs']['person-0'] == failure
    assert resumed['stage'] == 'incomplete' and resumed['skip_incomplete']
    calls = len(client.prompts)
    revision_batch.run(output, client=client, skip_incomplete=True)
    assert len(client.prompts) == calls
    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', original)
    retried = revision_batch.run(output, client=client, attempts=1)
    assert retried['stage'] == 'complete'
    assert retried['jobs']['person-0']['attempts'] == failure['attempts'] + 1
    assert not retried['skip_incomplete']


def test_parallel_revision_overlaps_distinct_people_limits_admission_and_reuses(tmp_path, monkeypatch):
    _, output = prepared_people(tmp_path)
    original = revision_batch.wiki_revision.revise
    barrier = Barrier(2, timeout=10)
    row_ids, snapshots = [], []
    def parallel(*args):
        row_ids.append(id(args[3]))
        barrier.wait()
        snapshots.append(read_json(output / 'progress.json')['active_jobs'])
        return original(*args)
    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', parallel)
    client = KeepClient()
    result = revision_batch.run(output, client=client, object_workers=2, workers=1, max_jobs=2)
    assert result['counts'] == {'complete': 2} and result['stage'] == 'incomplete'
    assert result['active_jobs'] == [] and result['active_job'] is None
    assert all(set(active) == {'person-0', 'person-1'} for active in snapshots)
    assert len(set(row_ids)) == 1  # Shared raw source, no full-history copy per worker.
    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', original)
    finished = revision_batch.run(output, client=client, object_workers=2, workers=1)
    assert finished['stage'] == 'complete' and finished['counts'] == {'complete': 3}
    calls = len(client.prompts)
    revision_batch.run(output, client=client, object_workers=2)
    assert len(client.prompts) == calls


def test_parallel_revision_isolates_failure_and_rechecks_source_before_export(tmp_path, monkeypatch):
    source, output = prepared_people(tmp_path, 2)
    original = revision_batch.wiki_revision.revise
    barrier = Barrier(2, timeout=10)
    def change_source(*args):
        barrier.wait()
        if args[2]['subject']['account'] == 'person-0':
            saved = read_json(source / 'progress.json')['jobs']['person-0']
            Path(saved['knowledge']).write_text('{}')
        return original(*args)
    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', change_source)
    state = revision_batch.run(output, client=KeepClient(), object_workers=2, workers=1)
    assert state['counts'] == {'complete': 1, 'incomplete': 1}
    assert 'revision source changed' in state['jobs']['person-0']['error']
    assert not (output / 'jobs/person-0/content/knowledge.json').exists()
    assert state['jobs']['person-1']['stage'] == 'complete'


@pytest.mark.parametrize('completed', [False, True])
def test_changed_ancestor_blocks_chained_generation_and_reuse(tmp_path, completed):
    from test_wiki_revision_roles import RoleClient
    source, prior, knowledge = prepared(tmp_path)
    revision_batch.run(prior, client=KeepClient())
    output = tmp_path / 'role-revision'
    revision_batch.prepare(prior, output, engine='roles_v1')
    client = RoleClient()
    if completed:
        revision_batch.run(output, client=client)
    calls = len(client.prompts)
    knowledge.write_text('{}')
    if completed:
        with pytest.raises(ValueError, match='original knowledge changed'):
            revision_batch.run(output, client=client)
    else:
        result = revision_batch.run(output, client=client)
        assert result['jobs']['page']['stage'] == 'incomplete'
        assert 'original knowledge changed' in result['jobs']['page']['error']
    assert read_json(output / 'progress.json')['stage'] == 'incomplete'
    assert len(client.prompts) == calls


def test_ancestor_changed_during_generation_cannot_be_exported_as_complete(tmp_path):
    from test_wiki_revision_roles import RoleClient
    source, prior, knowledge = prepared(tmp_path)
    revision_batch.run(prior, client=KeepClient())
    output = tmp_path / 'role-revision'
    revision_batch.prepare(prior, output, engine='roles_v1')

    class MutatingClient(RoleClient):
        def run(self, prompt, schema):
            result = super().run(prompt, schema)
            knowledge.write_text('{}')
            return result

    result = revision_batch.run(output, client=MutatingClient())
    assert result['stage'] == 'incomplete'
    job = result['jobs']['page']
    assert job['stage'] == 'incomplete' and 'original knowledge changed' in job['error']
    assert 'knowledge' not in job
