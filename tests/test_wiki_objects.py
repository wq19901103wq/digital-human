"""Object composition preserves attribution, provenance and unresolved identities."""
from copy import deepcopy
import json

import pytest

from src.bootstrap import wiki_delivery as delivery, wiki_objects, wiki_prompt
from src.iteration.storage import read_json, write_json
from test_wiki_delivery import batch_fixture
from test_wiki_prompt import knowledge


def verified_batch(path, data=None):
    batch, source = batch_fixture(path, data=data)
    manifest = read_json(batch / 'manifest.json')
    manifest['jobs'][0].update(binding='account_filename', title='已核实人物')
    write_json(batch / 'manifest.json', manifest)
    return batch, source


def composed(batches):
    value = delivery.inspect(batches)
    assert value['summary']['counts'] == {'complete': len(batches)}
    assert len(value['objects']) == 1
    return value, wiki_objects.compose(value['objects'][0], value['jobs'])


def test_same_source_copies_do_not_become_independent_evidence(tmp_path):
    one, source_one = verified_batch(tmp_path / 'one')
    two, source_two = verified_batch(tmp_path / 'two')
    before = {p: p.read_bytes() for p in (source_one, source_two)}
    value, data = composed([one, two])
    assert data['coverage']['input_records'] == 8
    assert data['coverage']['records'] == 4 and data['coverage']['unique_evidence'] == 3
    assert all(len(e['paths']) == 2 for e in data['evidence'].values())
    assert len(data['entities']) == 2
    assert all(len(origins) == 2 for origins in data['provenance']['records'].values())
    assert data == wiki_objects.compose(value['objects'][0], list(reversed(value['jobs'])))
    output = tmp_path / 'catalog'
    delivery.write_catalog(value, output)
    report = delivery.require_catalog(output, [one, two])
    assert report['objects'] == {'person': 1} and report['records'] == 4
    obj = value['objects'][0]
    text = (output / 'objects' / (obj['id'] + '.md')).read_text()
    assert '尚未出发' in text and '计划中' in text and '否定' in text
    assert '〔person〕' in text and '〔self〕' in text
    assert all(p.read_bytes() == old for p, old in before.items())


def test_equal_local_line_numbers_in_different_histories_remain_distinct(tmp_path):
    one, _ = verified_batch(tmp_path / 'one')
    data = knowledge(tmp_path)
    for evidence in data['evidence'].values():
        evidence['sha256'] = 'different-source-file'
    two, _ = verified_batch(tmp_path / 'two', data)
    _, merged = composed([one, two])
    assert merged['coverage']['unique_evidence'] == 6
    assert merged['coverage']['records'] == 8
    addresses = merged['addresses']
    assert len({r['utterance_ref'] for r in addresses}) == 6
    assert {r['utterance_line'] for r in addresses} == {1, 2, 3}
    assert all(merged['evidence'][r['utterance_ref']]['line'] == r['utterance_line'] for r in addresses)


def test_unresolved_names_stay_local_and_differing_claims_keep_roles_and_dates(tmp_path):
    data = knowledge(tmp_path)
    unknown = dict(id='local-name', kind='person', account='', label='同名', labels=['同名'],
        evidence_refs=['2'], identity_status='unresolved_cross_batch')
    data['entities'].append(unknown)
    data['events'][0]['participants'].append(dict(entity_id='local-name', role='邀请者'))
    common = {k: data['events'][0][k] for k in (
        'time_scope', 'valid_from', 'valid_to', 'temporal_kind', 'mode', 'polarity',
        'attribution', 'chat_ids', 'evidence_refs', 'fact_ids', 'first_observed_at', 'last_observed_at')}
    data['attributes'] = [dict(common, id='local-attribute', subject_id=data['subject']['entity_id'],
        field='preference', value='计划去甲地', mode='plan', valid_from='2020-01-01', valid_to='2020-02-01')]
    one, _ = verified_batch(tmp_path / 'one', data)
    second = deepcopy(data)
    second['attributes'][0].update(value='未确认去甲地', mode='uncertain', polarity='uncertain',
        valid_from='2021-01-01', valid_to='2021-02-01')
    second['events'][0].update(status='unknown', mode='uncertain')
    two, _ = verified_batch(tmp_path / 'two', second)
    _, merged = composed([one, two])
    assert len([e for e in merged['entities'] if not e['account']]) == 2
    assert len([e for e in merged['entities'] if e['account'] == 'person']) == 1
    assert {r['mode'] for r in merged['attributes']} == {'plan', 'uncertain'}
    assert {r['valid_from'] for r in merged['attributes']} == {'2020-01-01', '2021-01-01'}
    invitees = {p['entity_id'] for r in merged['events'] for p in r['participants'] if p['role'] == '邀请者'}
    assert len(invitees) == 2
    assert len(merged['coverage']['materials']) == 2
    assert all(len(material['reviews']) == 1 for material in merged['coverage']['materials'].values())
    cards = [json.loads(line) for line in wiki_objects.artifacts(merged)['cards.jsonl'].splitlines()]
    assert sum(len(c['facts']) for c in cards) == merged['coverage']['records']
    assert all(wiki_prompt.entity_refs(f) <= {e['id'] for e in c['identities']} for c in cards for f in c['facts'])
    assert '未确认实体' in wiki_objects.render(merged)
    assert '你来定时间' not in json.dumps(merged, ensure_ascii=False)


def test_candidate_materials_do_not_enter_verified_object_content(tmp_path):
    one, _ = verified_batch(tmp_path / 'one')
    candidate = knowledge(tmp_path)
    candidate['events'][0]['description'] = '候选旧资料中尚未核实的事件'
    two, _ = batch_fixture(tmp_path / 'two', data=candidate)
    value, data = composed([one, two])
    assert data['coverage']['input_records'] == 4
    assert '候选旧资料中尚未核实的事件' not in json.dumps(data, ensure_ascii=False)
    value['jobs'][0]['stage'] = 'needs_review'
    data = wiki_objects.compose(value['objects'][0], value['jobs'])
    assert data['coverage']['records'] == 0


def test_empty_completed_review_remains_a_linked_gap_without_object_body(tmp_path):
    empty = knowledge(tmp_path)
    for table in wiki_prompt.TABLES:
        empty[table] = []
    batch, source = verified_batch(tmp_path / 'empty', empty)
    before = source.read_bytes()
    value, data = composed([batch])
    assert not wiki_objects.eligible(value['jobs'][0])
    assert data['coverage']['records'] == 0 and data['coverage']['materials'] == {}
    obj = value['objects'][0]
    assert obj['body_status'] == 'evidence_gap' and obj['all_jobs_complete']
    output = tmp_path / 'catalog'
    delivery.write_catalog(value, output)
    report = delivery.require_catalog(output, [batch])
    assert report['objects'] == {} and report['records'] == 0
    assert not (output / 'objects' / obj['id'] / 'knowledge.json').exists()
    page = (output / 'objects' / (obj['id'] + '.md')).read_text()
    assert '无可用正文' in page and '已处理，无有效记录' in page
    assert '../../empty/batch/jobs/page/content/wiki.md' in page
    assert '已处理但无有效正文' in (output / 'coverage.md').read_text()
    assert source.read_bytes() == before
    assert read_json(source)['reviews'] == empty['reviews']


def test_group_materials_share_only_the_explicit_group_subject(tmp_path):
    batches = []
    for name in ('one', 'two'):
        data = knowledge(tmp_path)
        data['subject'].update(name='群旧称' if name == 'one' else '群新称', kind='group',
            account='', scope_id='group:self:room')
        data['entities'][0].update(kind='group', account='', identity_status='explicit_scope')
        batch, source = batch_fixture(tmp_path / name, data=data)
        manifest = read_json(batch / 'manifest.json')
        manifest['jobs'][0].update(kind='conversation', account='', chat_id='group:self:room',
            binding='exact_group_id', title=data['subject']['name'])
        write_json(batch / 'manifest.json', manifest)
        markdown = source.with_name('wiki.md')
        markdown.write_text(markdown.read_text().replace('\n账号：\n', '\n类型：group\n', 1))
        batches.append(batch)
    value, merged = composed(batches)
    assert value['objects'][0]['kind'] == 'group'
    assert merged['coverage']['records'] == 4
    assert len([e for e in merged['entities'] if e['kind'] == 'group']) == 1
    assert merged['subject']['scope_id'] == 'group:self:room'
    assert '群号：room' in wiki_objects.render(merged)
    wrong = dict(value['objects'][0], account='other-room')
    with pytest.raises(ValueError, match='identity differs'):
        wiki_objects.compose(wrong, value['jobs'])


@pytest.mark.parametrize('field,value', [('self_account', 'different-owner'), ('account', 'different-person')])
def test_mismatched_identity_rejected(tmp_path, field, value):
    batch, _ = verified_batch(tmp_path / 'one')
    snapshot = delivery.inspect([batch])
    obj = dict(snapshot['objects'][0], **{field: value})
    with pytest.raises(ValueError, match='differs from material'):
        wiki_objects.compose(obj, snapshot['jobs'])


@pytest.mark.parametrize('artifact', ['knowledge.json', 'wiki.md', 'wiki.xml', 'cards.jsonl', 'sources.json', 'entry', 'index'])
def test_unified_check_detects_changed_object_exports(tmp_path, capsys, artifact):
    from scripts.check import main
    batch, _ = verified_batch(tmp_path / 'one')
    value = delivery.inspect([batch])
    output = tmp_path / 'catalog'
    delivery.write_catalog(value, output)
    obj = value['objects'][0]
    command = ['wiki', '--batch', str(batch), '--catalog', str(output)]
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)['model_requests_prohibited']
    path = output / 'objects' / obj['id'] / artifact
    if artifact == 'entry':
        path = output / 'objects' / (obj['id'] + '.md')
    elif artifact == 'index':
        path = output / 'index.md'
    path.write_text('changed')
    assert main(command) == 1
    expected = 'object index differs from saved coverage' if artifact == 'index' else 'differs from verified materials'
    assert expected in json.loads(capsys.readouterr().out)['checks'][0]['error']
