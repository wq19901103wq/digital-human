"""Delivery catalogs distinguish actual reusable results, candidates and gaps."""
from copy import deepcopy
import json

import pytest

from src.bootstrap import wiki_delivery as delivery, wiki_prompt, wiki_structured
from src.config import sha256_file
from src.iteration.storage import read_json, write_json
from test_wiki_prompt import knowledge


def batch_fixture(tmp_path, *, reused=False, data=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = knowledge(tmp_path) if data is None else data
    batch = tmp_path / 'batch'
    directory = tmp_path / 'reused' if reused else batch / 'jobs/page'
    wiki_structured.export(directory / 'content', data)
    source = directory / 'content/knowledge.json'
    exported = wiki_prompt.export(source, batch / 'prompt/page')
    job = dict(id='page', title='旧名字候选', kind='person', account='person', self_account='self',
        output=str(batch / 'jobs/page'), binding='unverified_name_candidate')
    if reused:
        job['reuse'] = dict(path=str(source), sha256=sha256_file(source))
    gap = dict(id='gap', title='同名页', kind='person', reason='ambiguous_name', candidates=['p1', 'p2'])
    write_json(batch / 'manifest.json', dict(schema='wiki_batch_v1', jobs=[job], gaps=[gap],
        summary=dict(gaps={'ambiguous_name': 1})))
    write_json(batch / 'progress.json', dict(stage='complete', jobs={'page': dict(stage='complete',
        knowledge=str(source), sha256=sha256_file(source), records=exported['records'], packets=exported['packets'])}))
    return batch, source


def test_reused_outputs_are_linked_without_copying_facts_or_confirming_identity(tmp_path):
    batch, source = batch_fixture(tmp_path, reused=True)
    before = source.read_bytes()
    value = delivery.inspect([batch])
    assert value['stage'] == 'complete' and value['summary']['counts'] == {'complete': 1}
    assert value['summary']['records'] == 4 and value['runtime_usable'] is False
    page = value['jobs'][0]
    assert page['binding'] == 'unverified_name_candidate'
    assert page['files']['knowledge.json']['path'] == str(source)
    output = tmp_path / 'catalog'
    delivery.write_catalog(value, output)
    obj = value['objects'][0]
    assert obj['title'] == 'person' and obj['readable_bound_jobs'] == 0
    assert 'objects/' + obj['id'] + '.md' in (output / 'index.md').read_text()
    assert '../reused/content/wiki.md' in (output / 'materials.md').read_text()
    object_text = (output / 'objects' / (obj['id'] + '.md')).read_text()
    assert '../../reused/content/wiki.md' in object_text
    assert '旧资料归属待核实' in object_text and 'unverified_name_candidate' in object_text
    assert '同名页' in (output / 'identity-review.md').read_text()
    assert '旧名字候选' in (output / 'identity-review.md').read_text()
    assert '尚未出发' not in (output / 'catalog.json').read_text()
    assert source.read_bytes() == before


def test_catalog_recognizes_every_active_object_and_keeps_stale_jobs_interrupted(tmp_path, monkeypatch):
    batch, _ = batch_fixture(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    manifest['jobs'] += [dict(manifest['jobs'][0], id=name) for name in ('second', 'stale')]
    write_json(batch / 'manifest.json', manifest)
    write_json(batch / 'progress.json', dict(stage='running', jobs={
        name: dict(stage='running') for name in ('page', 'second', 'stale')}))
    active = [dict(id=name) for name in ('page', 'second')]
    monkeypatch.setattr(delivery.wiki_batch, 'status', lambda _: dict(stage='running',
        running=True, active=active[0], active_jobs=active))
    value = delivery.inspect([batch])
    assert {j['id']: j['stage'] for j in value['jobs']} == {
        'page': 'running', 'second': 'running', 'stale': 'interrupted'}


def test_catalog_groups_by_scoped_account_not_files_names_or_friendship():
    base = dict(batch='legacy', id='one', kind='person', self_account='self', account='p1',
        title='同名', binding='account_filename', stage='complete', reuse_eligible=True, records=4)
    jobs = [base, dict(base, id='two', title='旧称', binding='unverified_name_candidate'),
        dict(base, batch='members', id='member', binding='raw_member_snapshot'),
        dict(base, id='other-person', account='p2'),
        dict(base, id='other-owner', self_account='other')]
    group = dict(base, kind='conversation', account='', chat_id='group:self:room',
        binding='exact_group_id', id='group')
    jobs += [group, dict(group, id='renamed-group', title='改过名的群'),
             dict(group, id='other-room-owner', self_account='other', chat_id='group:other:room'),
             dict(base, id='topic', kind='topic', account='')]
    before = deepcopy(jobs)
    catalog = delivery.object_catalog(jobs)
    assert catalog['summary']['person']['total'] == 3
    assert catalog['summary']['group']['total'] == 2
    assert catalog['summary']['total'] == 5
    assert catalog['summary']['topics'] == dict(total=1, readable=1, empty=0)
    person = next(o for o in catalog['objects'] if o['kind'] == 'person'
                  and o['self_account'] == 'self' and o['account'] == 'p1')
    assert person['total_jobs'] == person['readable_jobs'] == 3
    assert person['readable_bound_jobs'] == 2 and person['all_jobs_complete']
    assert person['jobs'] == [dict(batch=j['batch'], id=j['id']) for j in jobs[:3]]
    assert jobs == before  # Indexing must not rewrite facts or upgrade candidate bindings.


def test_object_coverage_excludes_unusable_results_and_unscoped_identities():
    base = dict(batch='batch', id='candidate', kind='person', self_account='self', account='p1',
        title='未确认的名字', binding='unverified_name_candidate', stage='complete', reuse_eligible=True, records=4)
    jobs = [base, dict(base, id='pending', binding='raw_member_snapshot', stage='pending', reuse_eligible=False),
        dict(base, id='invalid', stage='invalid', reuse_eligible=False),
        dict(base, id='review', stage='needs_review', reuse_eligible=False),
        dict(base, id='no-account', account=''), dict(base, id='no-owner', self_account=''),
        dict(base, id='mismatched-group', kind='conversation', chat_id='group:other:room'),
        dict(base, id='no-group', kind='conversation', chat_id='private:self:person')]
    value = delivery.object_catalog(jobs)
    assert value['summary']['person'] == dict(total=1, readable=1,
        readable_with_verified_binding=0, all_jobs_complete=0, body_complete=0,
        body_in_progress=1, body_evidence_gap=0, with_empty_materials=0,
        identity_only=0, with_identity_review=1)
    assert value['summary']['unresolved_jobs'] == 4
    assert {r['id'] for r in value['unresolved']} == {
        'no-account', 'no-owner', 'mismatched-group', 'no-group'}
    obj = value['objects'][0]
    assert obj['title'] == 'p1' and obj['readable_jobs'] == 1 and obj['total_jobs'] == 4
    assert not obj['all_jobs_complete'] and not obj['runtime_usable']


@pytest.mark.parametrize('bound_stages,candidate_stage,expected', [
    ([], 'complete', 'identity_pending'),
    ([], 'pending', 'identity_pending'),
    (['complete'], 'pending', 'complete'),
    (['complete', 'complete'], 'complete', 'complete'),
    (['complete', 'pending'], 'complete', 'in_progress'),
    (['needs_review'], 'complete', 'in_progress'),
    (['invalid'], 'complete', 'in_progress'),
])
def test_body_status_tracks_all_verified_materials_separately_from_candidates(
        bound_stages, candidate_stage, expected):
    base = dict(batch='batch', kind='person', self_account='self', account='person', title='未确认名字', records=4)
    jobs = [dict(base, id='candidate', binding='unverified_name_candidate', stage=candidate_stage,
                 reuse_eligible=candidate_stage == 'complete')]
    jobs += [dict(base, id=str(i), binding='raw_member_snapshot', stage=stage,
                  reuse_eligible=stage == 'complete') for i, stage in enumerate(bound_stages)]
    value = delivery.object_catalog(jobs)
    obj = value['objects'][0]
    assert obj['body_status'] == expected
    assert obj['identity_review_jobs'] == 1 and obj['bound_jobs'] == len(bound_stages)
    assert obj['readable_bound_jobs'] == bound_stages.count('complete')
    stats = value['summary']['person']
    assert stats['body_complete'] + stats['body_in_progress'] + stats['identity_only'] == 1
    assert stats['with_identity_review'] == 1
    if not bound_stages:
        assert obj['title'] == 'person' and stats['body_complete'] == 0


@pytest.mark.parametrize('kind,binding,chat_id', [
    ('person', 'raw_member_snapshot', ''), ('group', 'exact_group_id', 'group:self:room')])
@pytest.mark.parametrize('other_records,other_stage,expected', [
    (0, 'complete', 'evidence_gap'), (4, 'complete', 'complete'), (0, 'pending', 'in_progress')])
def test_empty_materials_are_gaps_not_readable_bodies(kind, binding, chat_id,
        other_records, other_stage, expected):
    base = dict(batch='batch', id='empty', kind=kind, binding=binding, chat_id=chat_id,
        self_account='self', account='person' if kind == 'person' else '', title='已核实对象',
        stage='complete', reuse_eligible=True, records=0)
    jobs = [base, dict(base, id='other', records=other_records, stage=other_stage,
                      reuse_eligible=other_stage == 'complete')]
    value = delivery.object_catalog(jobs)
    obj = value['objects'][0]
    assert obj['body_status'] == expected
    assert obj['readable_bound_jobs'] == int(other_records > 0)
    assert obj['all_jobs_complete'] == (other_stage == 'complete')
    assert obj['empty_bound_jobs'] == 1 + int(other_stage == 'complete' and not other_records)
    stats = value['summary'][kind]
    assert sum(stats[key] for key in ('body_complete', 'body_in_progress',
                                     'body_evidence_gap', 'identity_only')) == 1
    assert stats['with_empty_materials'] == 1


def test_empty_topics_are_not_counted_as_readable():
    job = dict(batch='batch', id='topic', kind='topic', stage='complete', reuse_eligible=True, records=0)
    value = delivery.object_catalog([job, dict(job, id='readable', records=2)])
    assert value['summary']['topics'] == dict(total=2, readable=1, empty=1)


def test_index_includes_unstarted_objects_without_calling_candidates_completed_bodies(tmp_path):
    batch, _ = batch_fixture(tmp_path)
    manifest = read_json(batch / 'manifest.json')
    manifest['jobs'].append(dict(manifest['jobs'][0], id='pending', account='pending-person',
                                binding='raw_member_snapshot', title='未整理人物'))
    write_json(batch / 'manifest.json', manifest)
    output = tmp_path / 'catalog'
    value = delivery.inspect([batch])
    delivery.write_catalog(value, output)
    index = (output / 'index.md').read_text().split('## 群')[0]
    assert '### 正文完成（0）' in index and '### 仍在整理（1）' in index
    assert 'pending-person' in index.split('### 仍在整理（1）')[1].split('### 身份待核实')[0]
    assert 'person' in index.split('### 身份待核实（1）')[1]
    assert '旧名字候选' not in index
    delivery.require_catalog(output, [batch])


def test_cross_batch_object_entry_reuses_sources_and_keeps_candidate_material_separate(tmp_path):
    legacy, source = batch_fixture(tmp_path / 'legacy', reused=True)
    members, member_source = batch_fixture(tmp_path / 'members', reused=True)
    manifest = read_json(members / 'manifest.json')
    manifest['jobs'][0].update(binding='raw_member_snapshot', title='已确认成员')
    manifest['jobs'].append(dict(manifest['jobs'][0], id='pending', title='未完成材料'))
    write_json(members / 'manifest.json', manifest)
    before = {p: p.read_bytes() for p in (source, member_source)}
    value = delivery.inspect([legacy, members])
    assert len(value['objects']) == 1
    obj = value['objects'][0]
    assert obj['title'] == '已确认成员'
    assert obj['total_jobs'] == 3 and obj['readable_jobs'] == 2 and obj['readable_bound_jobs'] == 1
    assert not obj['all_jobs_complete']
    output = tmp_path / 'catalog'
    delivery.write_catalog(value, output)
    assert len(list((output / 'objects').glob('*.md'))) == 1
    text = (output / 'objects' / (obj['id'] + '.md')).read_text()
    assert '../../members/reused/content/wiki.md' in text.split('## 旧资料归属待核实')[0]
    assert '../../legacy/reused/content/wiki.md' in text.split('## 旧资料归属待核实')[1]
    assert '未完成材料' in text and 'pending' in text
    assert '**仍在整理**' in text and '1 / 2 份' in text and '另有 1 份材料' in text
    assert all(p.read_bytes() == original for p, original in before.items())


@pytest.mark.parametrize('mutation', ['knowledge', 'cards', 'rehash_cards', 'sources', 'markdown', 'xml', 'count'])
def test_corrupted_completed_outputs_are_excluded_and_check_fails(tmp_path, mutation):
    batch, source = batch_fixture(tmp_path)
    cache = {}
    assert delivery.inspect([batch], cache=cache)['summary']['counts'] == {'complete': 1}
    if mutation == 'knowledge':
        source.write_text('{}')
    elif mutation in ('cards', 'rehash_cards'):
        path = batch / 'prompt/page/cards.jsonl'
        path.write_text('{}\n')
        if mutation == 'rehash_cards':
            manifest = read_json(batch / 'prompt/page/manifest.json')
            manifest['files']['cards.jsonl'] = sha256_file(path)
            write_json(batch / 'prompt/page/manifest.json', manifest)
    elif mutation == 'sources':
        path = batch / 'prompt/page/sources.json'
        path.write_text('{}')
    elif mutation in ('markdown', 'xml'):
        source.with_name('wiki.md' if mutation == 'markdown' else 'wiki.xml').write_text('changed')
    else:
        state = read_json(batch / 'progress.json')
        state['jobs']['page']['records'] += 1
        write_json(batch / 'progress.json', state)
    value = delivery.inspect([batch], cache=cache)
    assert value['stage'] == 'incomplete' and value['jobs'][0]['stage'] == 'invalid'
    assert value['summary']['records'] == 0
    with pytest.raises(ValueError):
        delivery.require_valid([batch])


def test_supplement_replaces_parent_gaps_and_group_name_candidates_stay_separate(tmp_path):
    batch, _ = batch_fixture(tmp_path)
    supplement = tmp_path / 'supplement'
    manifest = dict(schema='wiki_supplement_v1',
        parent=dict(path=str(batch / 'manifest.json'), sha256=sha256_file(batch / 'manifest.json')),
        jobs=[dict(id='gap', title='核实后的人物', kind='person', account='p1',
                   binding='verified_utterance_anchors', output=str(supplement / 'jobs/gap')),
              dict(id='group', title='群候选', kind='conversation', account='', chat_id='group:self:room',
                   binding='unique_group_name_candidate', output=str(supplement / 'jobs/group'))],
        gaps=[], summary=dict(gaps={}))
    write_json(supplement / 'manifest.json', manifest)
    value = delivery.inspect([batch, supplement])
    assert value['gaps'] == [] and value['summary']['counts'] == {'complete': 1, 'pending': 2}
    assert value['stage'] == 'incomplete'
    delivery.write_catalog(value, tmp_path / 'catalog')
    identities = (tmp_path / 'catalog/identity-review.md').read_text()
    assert '核实后的人物' not in identities and 'group:self:room' in identities
    assert value['summary']['objects']['unresolved_jobs'] == 2
    coverage = (tmp_path / 'catalog/coverage.md').read_text()
    assert '无法定位聊天对象' in coverage and 'group:self:room' in coverage
    with pytest.raises(ValueError, match='not complete'):
        delivery.require_valid([batch, supplement], require_complete=True)
    other = tmp_path / 'old-supplement'
    write_json(other / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='one supplement'):
        delivery.inspect([batch, supplement, other])


def test_follow_writes_final_catalog_and_stops_when_batches_finish(tmp_path, monkeypatch):
    batch, _ = batch_fixture(tmp_path)
    finished = delivery.inspect([batch])
    running = deepcopy(finished)
    running['stage'] = 'incomplete'
    running['batches'][0]['running'] = True
    values = iter([running, finished])
    monkeypatch.setattr(delivery, 'inspect', lambda *a, **k: next(values))
    sleeps = []
    monkeypatch.setattr(delivery.time, 'sleep', sleeps.append)
    assert delivery.deliver([batch], tmp_path / 'catalog', follow=True)['stage'] == 'complete'
    assert sleeps == [30]
    assert read_json(tmp_path / 'catalog/catalog.json')['stage'] == 'complete'


def test_unified_check_reports_incomplete_separately_from_invalid_artifacts(tmp_path, capsys):
    from scripts.check import main
    batch, _ = batch_fixture(tmp_path)
    assert main(['wiki', '--batch', str(batch), '--require-complete']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'passed' and report['model_requests_prohibited']
    state = read_json(batch / 'progress.json')
    state['stage'] = 'running'
    state['jobs']['page']['stage'] = 'running'
    write_json(batch / 'progress.json', state)
    assert main(['wiki', '--batch', str(batch), '--require-complete']) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'failed'


@pytest.mark.parametrize('usage', ['direct', 'self_reference', 'third_person', 'requested', 'rejected'])
def test_forwarded_address_roles_are_isolated_without_changing_completed_outputs(tmp_path, capsys, usage):
    from scripts.check import main
    data = knowledge(tmp_path)
    address = data['addresses'][0]
    address['usage'] = usage
    address['mode'] = 'reported'  # Reported does not resolve the inner speaker.
    data['evidence'][str(address['utterance_line'])]['message_kind'] = 'forward'
    batch, source = batch_fixture(tmp_path, data=data)
    before = {path: path.read_bytes() for path in batch.rglob('*') if path.is_file()}
    cache = {}
    value = delivery.inspect([batch], cache=cache)
    assert delivery.inspect([batch], cache=cache)['summary'] == value['summary']
    page = value['jobs'][0]
    assert page['stage'] == 'needs_review' and page['verified']
    assert not page['reuse_eligible'] and not page['runtime_usable']
    assert page['files']['knowledge.json']['path'] == str(source)
    assert page['content_issues'] == [dict(record_id=address['id'],
        reason='forwarded_address_speaker_unresolved', evidence_refs=['1'],
        usage=usage, speaker_account='self')]
    assert value['summary']['records'] == value['summary']['packets'] == 0
    assert value['summary']['review_records'] == 4 and value['summary']['content_issues'] == 1
    output = tmp_path / 'catalog'
    delivery.write_catalog(value, output)
    assert '旧名字候选' not in (output / 'index.md').read_text()
    review = (output / 'content-review.md').read_text()
    assert '旧名字候选' in review and address['id'] in review
    assert '尚未出发' not in review
    assert 'needs_review' in (output / 'coverage.md').read_text()
    assert main(['wiki', '--batch', str(batch)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert 'content needs review' in report['checks'][0]['error']
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('usage', ['quoted', 'uncertain'])
def test_forwarded_quoted_or_uncertain_addresses_do_not_assert_a_direct_speaker(tmp_path, usage):
    data = knowledge(tmp_path)
    data['addresses'][0]['usage'] = usage
    data['evidence']['1']['message_kind'] = 'forward'
    batch, _ = batch_fixture(tmp_path, data=data)
    page = delivery.inspect([batch])['jobs'][0]
    assert page['stage'] == 'complete' and page['reuse_eligible']
    assert page['content_issues'] == [] and not page['runtime_usable']


def test_forwarded_corroboration_does_not_override_the_actual_utterance_source(tmp_path):
    data = knowledge(tmp_path)
    data['evidence']['4'] = dict(data['evidence']['1'], line=4, message_kind='forward')
    for address in data['addresses']:
        address['evidence_refs'].append('4')
    batch, _ = batch_fixture(tmp_path, data=data)
    assert delivery.inspect([batch])['summary']['counts'] == {'complete': 1}


@pytest.mark.parametrize('issue', ['entity_account_without_source_anchor', 'self_report_without_source_author'])
def test_wrong_account_attribution_is_excluded_from_reusable_catalog(tmp_path, issue):
    data = knowledge(tmp_path)
    if issue == 'entity_account_without_source_anchor':
        data['entities'][0]['evidence_refs'] = ['1']
    else:
        data['events'][0].update(mode='self_report', evidence_refs=['1'])
    batch, source = batch_fixture(tmp_path, data=data)
    before = source.read_bytes()
    value = delivery.inspect([batch])
    page = value['jobs'][0]
    assert page['stage'] == 'needs_review' and not page['reuse_eligible']
    assert page['content_issues'][0]['reason'] == issue
    assert value['summary']['records'] == 0
    delivery.write_catalog(value, tmp_path / 'catalog')
    assert issue in (tmp_path / 'catalog/content-review.md').read_text()
    assert source.read_bytes() == before
