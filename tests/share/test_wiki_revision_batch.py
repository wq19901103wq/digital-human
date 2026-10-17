"""Revision batches follow frozen originals and overlay them without double counting."""
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Event

import pytest

from src.share.wiki import wiki_delivery as delivery
from src.share.wiki import wiki_revision_batch as revision_batch
from src.infrastructure.storage import read_json
from src.infrastructure.storage import write_json
from test_wiki_delivery import batch_fixture
from test_wiki_revision import KeepClient


def prepared(tmp_path, *, engine='legacy'):
    source, knowledge = batch_fixture(tmp_path)
    manifest = read_json(source / 'manifest.json')
    data = read_json(knowledge)
    manifest.update(messages=data['coverage'], self_account='self', config={})
    manifest['jobs'][0]['binding'] = 'raw_member_snapshot'
    write_json(source / 'manifest.json', manifest)
    output = tmp_path / 'revision'
    revision_batch.prepare(source, output, engine=engine)
    return source, output, knowledge


HISTORICAL_SEMANTIC_PIPELINE = {
    'wiki_repair.py': 'dd1a8fdad053d40ae186245d6093b47f2244ceab3a775cf0b67831892fb69351',
    'wiki_mentions.py': 'f55ef49dea4f750bacd8bd5791cf4e07440bb28fd5c5e00f1b7c2a437ae331b3',
    'wiki_reconcile.py': '46c2c5d2e5bb4f584baeeca835e4c14ba2b0817cd7ae46021f381f64eb76eccf',
    'wiki_structured.py': '3e6f9d331fc15e2b5e19eb721e2a10b80083a0269bca2ef25d665d25e7fee7ef',
    'wiki_revision.py': '0867ea792f816341947cd7726799a711dbff77f220652b09a54a4d61b37ef466',
    'wiki_prompt.py': '5f06f1b0782e5dbb9f5820ddcbed11053a11c72a8795c736fa70d7b641e3bb3c',
}
TARGET_SEMANTIC_PIPELINE = {
    'wiki_repair.py': '113b3c3445698773eb571286f6dcba9a1be03456cc0af4e141c7a0fcc5478de8',
    'wiki_mentions.py': '012dd09f76df3166ba5f754ff1027fa2c276fc163b587a4a713312f2a3b9ec3b',
    'wiki_reconcile.py': '46c2c5d2e5bb4f584baeeca835e4c14ba2b0817cd7ae46021f381f64eb76eccf',
    'wiki_structured.py': '4809954a35b934dc678be21d2a20f570a61cb4b9d0aa628584ed7c667c7c9620',
    'wiki_revision.py': '1984dc367ed08984f8939b55d666637065bd4cbd9097f8737eb52a7ef43a96bf',
    'wiki_prompt.py': '7be0cb6b366819bc3e5062780abed3c0b6b83b1e208cec535e75516b5c7cc088',
    'knowledge_view.py': 'b365c63f58137c9efd7537d39c30148367af8b7994400e8d818ab6a388741e44',
}
FEEDBACK_SEMANTIC_PIPELINE = TARGET_SEMANTIC_PIPELINE | {
    'wiki_repair.py': '509566dd8d22a67cbf93da00d192944f025cce581fb4d3d995fc164ce363d1d4',
    'wiki_structured.py': 'c00354d8384e078a2348e3466d067a9f05818f8f1c24000e4ff34ce20f7e03c3',
    'wiki_revision.py': '9f4992d9829ab0047830912183633f23964b7dd64804d8a968607398b715aa06',
}
FEEDBACK_ENGINES = {
    'legacy': {},
    'roles_v1': {'wiki_revision_roles.py':
        '040aa3920b8841f02de4d32ddb37bb240e4914ed1b211f1fdb1cc7fd79030e79'},
    'roles_claims_v2': {'wiki_revision_claims.py':
        '7df5a6cb1b121cf2e44fc71e4b44e65460ff0039c679833272762df228b96138'},
}
DIAGNOSTIC_ENGINES = FEEDBACK_ENGINES | {
    'roles_claims_v2': {'wiki_revision_claims.py':
        'edc21bed46700e81917351f3bf320a0d6462c96ee3c3c2b9fa2057dbcf0e1eb9'},
}
REREAD_SEMANTIC_PIPELINE = FEEDBACK_SEMANTIC_PIPELINE | {
    'wiki_repair.py': '34ff485f3fcb6f6e6c257a8082088eac2344be5e38f7e61a206cc8bcc56347a4',
    'wiki_mentions.py': 'f3975a5ca52a736fa39e63701f3ea2dd96417a40af177b73eb8d929b888a55da',
}
STRUCTURED_SEMANTIC_PIPELINE = REREAD_SEMANTIC_PIPELINE | {
    'wiki_repair.py': '64a6b79311739b76fcbd61cbd3a97af735db442ca12752d47c2748b7e2b24e22',
    'wiki_structured.py': '63922879a56e8ff465907a7ff5b2520cd5ac4e7390eee15fd6deefdba12cf90e',
}
REFERENCE_SEMANTIC_PIPELINE = STRUCTURED_SEMANTIC_PIPELINE | {
    'wiki_repair.py': 'ab53268eaf06c574e20f4c8c9c3d76812f7d8db0c82f247100b8f8b766653072',
    'wiki_mentions.py': 'e3fe3552562f9330a33120b177e287aaba204c0972918bb7746dfb40fd8b64e5',
    'wiki_structured.py': '1f3cc787ade5806a305c4b9493c9d82ebf8dca10c051d6ccfc89c9cfd4acf810',
    'wiki_reconcile.py': '4663c7db102c33d8287ae4505fb77196d4dd833af46c876c659ef7ec9206aa05',
}
SERVICE_SEMANTIC_PIPELINE = REFERENCE_SEMANTIC_PIPELINE | {
    'wiki_repair.py': '8073f66bb240174c99ea276c087997df898953c91359529c6811a197f5955cdc',
}
CURRENT_SEMANTIC_PIPELINE = SERVICE_SEMANTIC_PIPELINE | {
    'wiki_repair.py': '06e9f7b9b39de02d8e302b8bc05a974d6c5b3a0e8763f9e9674540c5e440c540',
    'wiki_mentions.py': '1c95b8b04955395bdf24b6ae7242ca9514ff649a2550d94aaa61e55036673a06',
    'wiki_reconcile.py': '8cf651176335ad04bd07d0780436aadb86175b73309e6dfb7228547cac2ca86e',
    'wiki_structured.py': '12a395f3658420e16601de84f027a44ee601f508f0d0baf188bd6778068c0bb7',
    'wiki_prompt.py': '932f859684a6aa9297f8643b3f44d90ca0e4b5145e2ca786229fc4a9f75ea51d',
}
HISTORICAL_ENGINES = {
    'legacy': ('d6db8d19f19e2f57091a39ca58302c28e0d42a6920bf7d73bfba51a4ac556641', {}, {}),
    'roles_v1': ('03b4d8731658131b7af3e1e33894494bc4c2af09c0c81f907529acc68ecc476a',
        {'wiki_revision_roles.py': '6a896619d5abc21623d71a014a0b5827ee13a61a00d0c49a7a63f38f5ecfd216'},
        {'wiki_revision_roles.py': '040aa3920b8841f02de4d32ddb37bb240e4914ed1b211f1fdb1cc7fd79030e79'}),
    'roles_claims_v2': ('0f465169abc187cd0eeec4e351b9946dd73952e72bbdd1b9eb89d63b3ec3e68f',
        {'wiki_revision_claims.py': '058b60a4115b56cbb679a2ffbb249ff68a149fcfc3168dd196822aa0d36d209d'},
        {'wiki_revision_claims.py': '03fa67062f8f9c4f8072802c22a7f634ccf8c6ed4d8a0752f4ebd48a3bbb11d2'}),
}


def historical_pipeline(engine, snapshot):
    runner, old_engine, new_engine = HISTORICAL_ENGINES[engine]
    if snapshot == 'pre_migration':
        return HISTORICAL_SEMANTIC_PIPELINE | old_engine | {'wiki_revision_batch.py': runner}
    if snapshot == 'pre_feedback':
        return TARGET_SEMANTIC_PIPELINE | new_engine | {
            'wiki_revision_batch.py': '8b085de3e13d715542139ba8ab77895c22d4333ac8ee88264a8cb9af05b86b42'}
    if snapshot == 'pre_diagnostic':
        return FEEDBACK_SEMANTIC_PIPELINE | FEEDBACK_ENGINES[engine] | {
            'wiki_revision_batch.py': 'd9d2be4619c3f356a13a8c7a2f34a9b8de0d8c63fbfb07bbc377941cc803da0e'}
    if snapshot == 'pre_structured':
        return REREAD_SEMANTIC_PIPELINE | DIAGNOSTIC_ENGINES[engine] | {
            'wiki_revision_batch.py': '14d041408aefd8ea5e53b568495a96800b6628b9c10d6ddc25cae6c5be70e495'}
    if snapshot == 'pre_service':
        return REFERENCE_SEMANTIC_PIPELINE | DIAGNOSTIC_ENGINES[engine] | {
            'wiki_revision_batch.py': '88ef382af16752572b954afaf4419a5d08c65ff80ff165b31cb86ce81b139747'}
    return {name: digest for name, digest in (TARGET_SEMANTIC_PIPELINE | new_engine).items()
            if name != 'knowledge_view.py'} | {
        'wiki_revision_batch.py': 'dfa96ff82a3cdb66e279204d9ff1b65a1a32cd63395bbca96b6590ab537aea36'}


def revision_client(engine):
    if engine == 'roles_v1':
        from test_wiki_revision_roles import RoleClient
        return RoleClient()
    if engine == 'roles_claims_v2':
        from test_wiki_revision_claims import RoleClient
        return RoleClient()
    return KeepClient()


@pytest.mark.parametrize('engine', HISTORICAL_ENGINES)
@pytest.mark.parametrize('snapshot', ['pre_migration', 'unsealed_view', 'pre_feedback',
                                      'pre_diagnostic', 'pre_structured', 'pre_service'])
def test_exact_historical_migration_accepts_only_unchanged_complete_snapshots(engine, snapshot):
    frozen = historical_pipeline(engine, snapshot)
    before = deepcopy(frozen)
    current = revision_batch.previous_output_pipeline(revision_batch.pipeline(engine))
    assert {name: digest for name, digest in current.items() if name != 'wiki_revision_batch.py'} == (
        CURRENT_SEMANTIC_PIPELINE | DIAGNOSTIC_ENGINES[engine])
    assert revision_batch.compatible_pipeline(frozen, engine)
    assert frozen == before
    for name in frozen:
        assert not revision_batch.compatible_pipeline(frozen | {name: 'changed'}, engine)
        assert not revision_batch.compatible_pipeline(
            {key: value for key, value in frozen.items() if key != name}, engine)
    assert not revision_batch.compatible_pipeline(frozen | {'unexpected.py': 'changed'}, engine)
    for other in HISTORICAL_ENGINES:
        if other != engine:
            assert not revision_batch.compatible_pipeline(frozen, other)


@pytest.mark.parametrize('engine', HISTORICAL_ENGINES)
@pytest.mark.parametrize('snapshot', ['pre_migration', 'unsealed_view', 'pre_feedback',
                                      'pre_diagnostic', 'pre_structured', 'current'])
def test_pre_service_runner_rejects_mixed_semantic_snapshots(engine, snapshot):
    frozen = revision_batch.pipeline(engine) if snapshot == 'current' else historical_pipeline(engine, snapshot)
    frozen = frozen | {'wiki_revision_batch.py': revision_batch.PRE_SERVICE_RUNNER_SHA256}
    assert not revision_batch.compatible_pipeline(frozen, engine)


@pytest.mark.parametrize('engine', HISTORICAL_ENGINES)
@pytest.mark.parametrize('snapshot', ['pre_migration', 'unsealed_view', 'pre_feedback', 'pre_diagnostic'])
def test_pre_structured_runner_rejects_other_historical_semantic_snapshots(engine, snapshot):
    frozen = historical_pipeline(engine, snapshot) | {
        'wiki_revision_batch.py': revision_batch.PRE_STRUCTURED_RUNNER_SHA256}
    assert not revision_batch.compatible_pipeline(frozen, engine)


@pytest.mark.parametrize('engine', HISTORICAL_ENGINES)
@pytest.mark.parametrize('snapshot', ['pre_migration', 'unsealed_view', 'pre_feedback',
                                      'pre_diagnostic', 'pre_structured', 'pre_service'])
def test_migration_refuses_future_semantic_changes_and_missing_card_binding(engine, snapshot, monkeypatch):
    frozen = historical_pipeline(engine, snapshot)
    current = revision_batch.pipeline(engine)
    for name in TARGET_SEMANTIC_PIPELINE | HISTORICAL_ENGINES[engine][2]:
        monkeypatch.setattr(revision_batch, 'pipeline', lambda _: current | {name: 'changed'})
        assert not revision_batch.compatible_pipeline(frozen, engine)
        monkeypatch.setattr(revision_batch, 'pipeline',
            lambda _: {key: value for key, value in current.items() if key != name})
        assert not revision_batch.compatible_pipeline(frozen, engine)
    monkeypatch.setattr(revision_batch, 'pipeline', lambda _: current | {'unexpected.py': 'changed'})
    assert not revision_batch.compatible_pipeline(frozen, engine)


@pytest.mark.parametrize('engine', HISTORICAL_ENGINES)
def test_new_revision_requires_explicit_shared_card_binding(engine):
    current = revision_batch.pipeline(engine)
    assert 'knowledge_view.py' in current
    current.pop('knowledge_view.py')
    assert not revision_batch.compatible_pipeline(current, engine)


@pytest.mark.parametrize('engine', HISTORICAL_ENGINES)
def test_reread_migration_requires_the_complete_registered_binding(engine):
    current = revision_batch.pipeline(engine)
    legacy = FEEDBACK_SEMANTIC_PIPELINE | DIAGNOSTIC_ENGINES[engine] | {
        'wiki_revision_batch.py': current['wiki_revision_batch.py']}
    assert revision_batch.compatible_pipeline(legacy, engine)
    for name in ('wiki_repair.py', 'wiki_mentions.py'):
        assert not revision_batch.compatible_pipeline(legacy | {name: current[name]}, engine)


@pytest.mark.parametrize('engine', HISTORICAL_ENGINES)
@pytest.mark.parametrize('snapshot', ['pre_migration', 'unsealed_view', 'pre_feedback',
                                      'pre_diagnostic', 'pre_structured', 'pre_service'])
@pytest.mark.parametrize('interrupted', [False, True])
def test_historical_resume_reuses_valid_requests_and_preserves_every_input_manifest(
        tmp_path, monkeypatch, engine, snapshot, interrupted):
    source, output, knowledge = prepared(tmp_path, engine=engine)
    manifest_path = output / 'manifest.json'
    manifest = read_json(manifest_path)
    manifest['pipeline'] = historical_pipeline(engine, snapshot)
    write_json(manifest_path, manifest)
    job_path = Path(manifest['jobs'][0]['output']) / 'manifest.json'
    write_json(job_path, dict(source=revision_batch.locator(knowledge),
        pipeline=manifest['pipeline'], config=manifest['config']))
    before = {path: path.read_bytes() for path in (manifest_path, job_path, source / 'manifest.json')}
    original_export = revision_batch.export
    if interrupted:
        def fail_export(*args):
            raise OSError('interrupted after cached semantic decisions')
        monkeypatch.setattr(revision_batch, 'export', fail_export)
    client = revision_client(engine)
    first = revision_batch.run(output, client=client, attempts=1)
    assert first['stage'] == ('incomplete' if interrupted else 'complete')
    calls = len(client.prompts)
    assert calls > 0
    monkeypatch.setattr(revision_batch, 'export', original_export)
    resumed = revision_batch.run(output, client=client, attempts=1)
    assert resumed['stage'] == 'complete' and resumed['counts'] == {'complete': 1}
    assert len(client.prompts) == calls
    assert all(path.read_bytes() == content for path, content in before.items())
    inspected = delivery.inspect([source, output])
    assert inspected['objects'][0]['body_status'] == 'complete'
    assert inspected['jobs'][0]['reuse_eligible'] and inspected['summary']['records'] == 4


@pytest.mark.parametrize('engine', HISTORICAL_ENGINES)
@pytest.mark.parametrize('mutation', ['manifest', 'knowledge', 'raw_messages'])
def test_historical_completion_cannot_bypass_changed_sources(tmp_path, engine, mutation):
    source, output, knowledge = prepared(tmp_path, engine=engine)
    manifest_path = output / 'manifest.json'
    manifest = read_json(manifest_path)
    manifest['pipeline'] = historical_pipeline(engine, 'pre_migration')
    write_json(manifest_path, manifest)
    client = revision_client(engine)
    revision_batch.run(output, client=client)
    calls = len(client.prompts)
    if mutation == 'manifest':
        changed = read_json(source / 'manifest.json')
        changed['jobs'][0]['account'] = 'changed'
        write_json(source / 'manifest.json', changed)
    elif mutation == 'knowledge':
        knowledge.write_text('{}')
    else:
        changed_path = Path(manifest['messages']['path'])
        changed_path.write_bytes(changed_path.read_bytes() * 2)
    with pytest.raises(ValueError):
        revision_batch.run(output, client=client)
    assert len(client.prompts) == calls
    assert read_json(output / 'progress.json')['stage'] == 'incomplete'


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
    frozen = revision_batch.previous_output_pipeline(revision_batch.pipeline()) | {
        'wiki_revision_batch.py': revision_batch.LEGACY_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, 'legacy')
    assert not revision_batch.compatible_pipeline(frozen, 'roles_v1')
    frozen['wiki_revision.py'] = 'changed'
    assert not revision_batch.compatible_pipeline(frozen, 'legacy')


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1'])
def test_source_validation_compatibility_preserves_only_exact_generation_inputs(engine):
    frozen = revision_batch.previous_output_pipeline(revision_batch.pipeline(engine)) | {
        'wiki_revision_batch.py': revision_batch.SOURCE_CHECK_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine)
    frozen['wiki_revision.py'] = 'changed'
    assert not revision_batch.compatible_pipeline(frozen, engine)


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1', 'roles_claims_v2'])
def test_claims_dispatch_compatibility_cannot_change_engine_or_source_bytes(engine):
    frozen = revision_batch.previous_output_pipeline(revision_batch.pipeline(engine)) | {
        'wiki_revision_batch.py': revision_batch.CLAIMS_DISPATCH_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine) == (engine != 'roles_claims_v2')
    backend = Path(revision_batch.engine_for(engine).__file__).name
    frozen[backend] = 'changed'
    assert not revision_batch.compatible_pipeline(frozen, engine)


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1', 'roles_claims_v2'])
def test_parallel_runner_alias_preserves_all_semantic_bindings(engine):
    frozen = revision_batch.previous_output_pipeline(revision_batch.pipeline(engine)) | {
        'wiki_revision_batch.py': revision_batch.SERIAL_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine)
    for name in frozen:
        assert not revision_batch.compatible_pipeline(frozen | {name: 'changed'}, engine)


def prepared_people(tmp_path, count=3):
    from src.share.wiki import wiki_repair
    from src.share.wiki import wiki_structured
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
    frozen = revision_batch.previous_output_pipeline(revision_batch.pipeline(engine)) | {
        'wiki_revision_batch.py': revision_batch.OBJECT_POOL_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine)
    for name in frozen:
        assert not revision_batch.compatible_pipeline(frozen | {name: 'changed'}, engine)


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1', 'roles_claims_v2'])
def test_selection_alias_requires_every_semantic_dependency(engine):
    frozen = revision_batch.previous_output_pipeline(revision_batch.pipeline(engine)) | {
        'wiki_revision_batch.py': revision_batch.PRE_SELECTION_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine)
    for name in frozen:
        assert not revision_batch.compatible_pipeline(frozen | {name: 'changed'}, engine)


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1', 'roles_claims_v2'])
def test_connection_runner_alias_preserves_all_semantic_bindings(engine):
    frozen = revision_batch.previous_output_pipeline(revision_batch.pipeline(engine)) | {
        'wiki_revision_batch.py': revision_batch.PRE_CONNECTION_RUNNER_SHA256}
    assert revision_batch.compatible_pipeline(frozen, engine)
    for name in frozen:
        assert not revision_batch.compatible_pipeline(frozen | {name: 'changed'}, engine)


def test_selected_revision_retries_connection_only_and_reuses_completed_work(tmp_path, monkeypatch):
    from src.share.wiki import wiki_connection_retry as connection
    from src.judge.codex_transport import CodexTransportError
    _, output = prepared_people(tmp_path)
    original = revision_batch.wiki_revision.revise
    manifest = (output / 'manifest.json').read_bytes()
    state = read_json(output / 'progress.json')
    state['jobs']['person-0'] = dict(stage='incomplete', error='unresolved evidence')
    write_json(output / 'progress.json', state)
    preserved = deepcopy(state['jobs']['person-0'])
    calls, sleeps = [], []
    details = dict(exit_code=1, terminal_events=['turn.failed'], signals=['stream_disconnected'],
                   terminal_signals=['stream_disconnected'], stderr_signals=[])

    class Disconnect(KeepClient):
        def run(self, prompt, schema=None):
            calls.append((prompt, schema))
            if len(calls) == 1:
                raise CodexTransportError(details)
            return super().run(prompt, schema)

    def revise(client, directory, data, rows, workers):
        assert isinstance(client, connection.ConnectionRetryClient)
        if directory.name == 'person-2':
            raise ValueError('unresolved evidence')
        return original(client, directory, data, rows, workers)

    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', revise)
    monkeypatch.setattr(connection.time, 'sleep', sleeps.append)
    client = Disconnect()
    options = dict(client=client, attempts=1, job_ids=['person-1', 'person-2'], connection_retry_seconds=60)
    result = revision_batch.run(output, **options)
    assert result['counts'] == {'incomplete': 2, 'complete': 1}
    assert result['jobs']['person-0'] == preserved
    assert result['jobs']['person-1']['attempts'] == 1
    assert result['jobs']['person-2']['attempts'] == 1
    assert calls[0] == calls[1] and sleeps == [60]
    assert (output / 'manifest.json').read_bytes() == manifest
    status = connection.status(output, result['connection_retry_run_id'], running=False)
    assert status['total_retries'] == 1 and not status['waiting']
    count = len(calls)
    revision_batch.run(output, **(options | dict(job_ids=['person-1'])))
    assert len(calls) == count


@pytest.mark.parametrize('options', [dict(connection_retry_seconds=0),
    dict(connection_retry_seconds=True), dict(attempts=2), dict(job_ids=None), dict(skip_incomplete=True)])
def test_invalid_revision_connection_mode_fails_before_batch_reads(tmp_path, options):
    with pytest.raises(ValueError, match='connection retry requires'):
        revision_batch.run(tmp_path / 'missing',
            **(dict(attempts=1, job_ids=['selected'], connection_retry_seconds=60) | options))


def test_selected_retries_preserve_other_failures_and_run_in_parallel(tmp_path, monkeypatch):
    _, output = prepared_people(tmp_path)
    original = revision_batch.wiki_revision.revise

    def fail(*args):
        raise ValueError('unresolved evidence')

    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', fail)
    client = KeepClient()
    failed = revision_batch.run(output, client=client, attempts=1)
    preserved = deepcopy(failed['jobs']['person-0'])
    barrier = Barrier(2, timeout=10)
    snapshots = []

    def parallel(*args):
        barrier.wait()
        snapshots.append(read_json(output / 'progress.json')['active_jobs'])
        return original(*args)

    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', parallel)
    selected = ['person-2', 'person-1', 'person-2']
    result = revision_batch.run(output, client=client, attempts=1, object_workers=2,
                                job_ids=selected)
    assert result['counts'] == {'incomplete': 1, 'complete': 2}
    assert result['stage'] == 'incomplete' and result['total'] == 3
    assert result['jobs']['person-0'] == preserved
    assert result['selected_jobs'] == ['person-1', 'person-2']
    assert all(set(active) == {'person-1', 'person-2'} for active in snapshots)
    calls = len(client.prompts)
    revision_batch.run(output, client=client, job_ids=selected)
    assert len(client.prompts) == calls


@pytest.mark.parametrize('selected', [[], ['unknown'], ['person-0', 'unknown']])
def test_invalid_selection_preserves_progress_and_makes_no_requests(tmp_path, selected):
    _, output = prepared_people(tmp_path)
    previous = (output / 'progress.json').read_bytes()
    client = KeepClient()
    with pytest.raises(ValueError, match='job_ids must not be empty|unknown revision jobs'):
        revision_batch.run(output, client=client, job_ids=selected)
    assert (output / 'progress.json').read_bytes() == previous
    assert not client.prompts


@pytest.mark.parametrize('wait', [False, True])
def test_revision_wait_uses_batch_lock_and_keeps_unselected_jobs_pending(tmp_path, monkeypatch, wait):
    _, output = prepared_people(tmp_path)
    original = revision_batch.file_lock
    locks = []

    def lock(path, *, blocking):
        locks.append((path, blocking))
        return original(path, blocking=blocking)

    monkeypatch.setattr(revision_batch, 'file_lock', lock)
    result = revision_batch.run(output, client=KeepClient(), job_ids=['person-1'], wait=wait)
    assert locks == [(output / '.lock', wait)]
    assert result['counts'] == {'complete': 1}
    assert result['stage'] == 'incomplete' and result['total'] == 3
    assert set(result['jobs']) == {'person-1'}


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


def test_pre_service_revision_block_preserves_parallel_success_and_resumes_only_missing(tmp_path, monkeypatch):
    from src.judge import codex_transport
    source, output = prepared_people(tmp_path)
    manifest_path = output / 'manifest.json'
    manifest = read_json(manifest_path)
    manifest['pipeline'] = historical_pipeline('legacy', 'pre_service')
    write_json(manifest_path, manifest)
    original_progress = read_json(source / 'progress.json')
    frozen = {path: path.read_bytes() for path in (manifest_path, source / 'manifest.json',
        source / 'progress.json', *(Path(job['knowledge']) for job in original_progress['jobs'].values()))}
    overlap, blocked = Barrier(2), Event()
    details = dict(exit_code=1, terminal_signals=['rate_limit'])
    original = revision_batch.wiki_revision.revise
    calls = []

    def available():
        if blocked.is_set():
            raise codex_transport.CodexServiceBlocked(details)

    def revise(client, directory, data, rows, workers):
        account = data['subject']['account']
        calls.append((account, workers))
        if len(calls) <= 2:
            overlap.wait(timeout=10)
            if account == 'person-0':
                blocked.set()
                raise codex_transport.CodexTransportError(details)
            assert blocked.wait(10)
        return original(client, directory, data, rows, workers)

    monkeypatch.setattr(codex_transport, 'check_available', available)
    monkeypatch.setattr(revision_batch.wiki_revision, 'revise', revise)
    client = KeepClient()
    state = revision_batch.run(output, client=client, object_workers=2, workers=2, attempts=3)
    assert state['stage'] == 'blocked' and state['pid'] is None
    assert state['blocker'] == details
    assert state['active_jobs'] == [] and state['active_job'] is None
    assert state['jobs']['person-0']['stage'] == 'interrupted'
    assert state['jobs']['person-0']['attempts'] == 1
    assert state['jobs']['person-1']['stage'] == 'complete'
    assert 'person-2' not in state['jobs']
    knowledge = Path(state['jobs']['person-1']['knowledge'])
    original_body = knowledge.read_bytes()
    job_manifest = Path(manifest['jobs'][0]['output']) / 'manifest.json'
    frozen[job_manifest] = job_manifest.read_bytes()
    blocked.clear()
    recovered = revision_batch.run(output, client=client, object_workers=2, workers=2, attempts=3)
    assert recovered['stage'] == 'complete' and recovered['pid'] is None
    assert 'blocker' not in recovered and recovered['counts'] == {'complete': 3}
    assert recovered['jobs']['person-0']['attempts'] == 2
    assert recovered['jobs']['person-1'] == state['jobs']['person-1']
    assert knowledge.read_bytes() == original_body
    assert sorted(account for account, _ in calls) == ['person-0', 'person-0', 'person-1', 'person-2']
    assert all(workers == 2 for _, workers in calls)
    assert all(path.read_bytes() == content for path, content in frozen.items())


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
