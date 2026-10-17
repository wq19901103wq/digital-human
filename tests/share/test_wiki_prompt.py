"""Compact cards preserve ownership/status and make coverage loss explicit."""
import json

import pytest

from src.share import knowledge_view
from src.share.wiki import wiki_library
from src.share.wiki import wiki_prompt as prompt
from src.share.wiki import wiki_structured as structured
from test_wiki_structured import META, fixture


def knowledge(tmp_path):
    rows, coverage, data = fixture(tmp_path)
    common = {k: data['addresses'][0][k] for k in structured.COMMON}
    data['events'] = [dict(common, event_type='trip', description='尚未出发', status='planned',
        mode='plan', polarity='negated', participants=[dict(entity_ref='E0', role='traveler')],
        details=[dict(field='destination', value='示例城市')])]
    return structured.compile_records(META, [(data, {'F1': 'fact-1'})], rows, coverage)


def test_packets_preserve_every_record_and_direction_without_raw_excerpts(tmp_path):
    data = knowledge(tmp_path)
    cards = list(prompt.packets(data, max_chars=1600))
    facts = [f for card in cards for f in card['facts']]
    assert len(facts) == 4 and len({f['ref'] for f in facts}) == 4
    event = next(f for f in facts if f['kind'] == 'event')
    assert event['mode'] == 'plan' and event['polarity'] == 'negated' and event['status'] == 'planned'
    assert event['participants'][0]['role'] == 'traveler'
    addresses = [f for f in facts if f['kind'] == 'address']
    assert addresses[0]['target_id'] == addresses[1]['speaker_id']
    assert addresses[0]['speaker_id'] == addresses[1]['target_id']
    assert addresses[2]['usage'] == 'rejected'
    for card in cards:
        ids = {e['id'] for e in card['identities']}
        assert card['self_person_id'] in ids
        assert all(knowledge_view.entity_refs(f) <= ids for f in card['facts'])
    source = tmp_path / 'knowledge.json'
    source.write_text(json.dumps(data))
    report = prompt.export(source, tmp_path / 'export', max_chars=1600)
    assert report['records'] == 4 and not report['runtime_usable']
    assert '你来定时间' not in (tmp_path / 'export/cards.jsonl').read_text()
    assert 'path' not in (tmp_path / 'export/cards.jsonl').read_text()
    assert json.loads((tmp_path / 'export/sources.json').read_text())['evidence']['1']['line'] == 1


def test_shared_card_keeps_the_existing_self_identity_digest(tmp_path):
    data = knowledge(tmp_path)
    account = data['subject']['self_account']
    data['entities'] = [entity for entity in data['entities'] if entity.get('account') != account]
    result = knowledge_view.card(data, [])
    assert result['self_person_id'] == 'entity-' + wiki_library.digest([account, account])[:20]
    assert next(entity for entity in result['identities'] if entity['role'] == 'self')['account'] == account


def test_migrated_packet_preserves_historical_compact_card_bytes():
    data = dict(schema='wiki_structured_v1', subject=dict(entity_id='person', self_account='self'),
        entities=[dict(id='person', kind='person', account='person', label='对方',
                       identity_status='exact_account'),
                  dict(id='self', kind='person', account='self', label='本人',
                       identity_status='exact_account'),
                  dict(id='candidate', kind='person', account='', label='同名候选',
                       identity_status='unresolved_cross_batch')],
        attributes=[], relations=[], addresses=[], events=[dict(id='event', subject_id='person',
            event_type='trip', description='尚未出发', status='planned',
            participants=[dict(entity_id='person', role='traveler'),
                          dict(entity_id='candidate', role='uncertain')],
            details=[dict(field='destination', value='示例城市')], mode='plan', polarity='negated',
            attribution='', valid_to=None, chat_ids=[], evidence_refs=['source-1'], fact_ids=['fact-1'])])
    expected = (
        '{"schema":"wiki_prompt_v1","subject_id":"person","self_person_id":"self",'
        '"identities":[{"id":"person","kind":"person","account":"person","name":"对方",'
        '"role":"other","identity_status":"exact_account"},'
        '{"id":"self","kind":"person","account":"self","name":"本人",'
        '"role":"self","identity_status":"exact_account"},'
        '{"id":"candidate","kind":"person","account":"","name":"同名候选",'
        '"role":"other","identity_status":"unresolved_cross_batch"}],'
        '"facts":[{"ref":"event","kind":"event","subject_id":"person","event_type":"trip",'
        '"description":"尚未出发","status":"planned",'
        '"participants":[{"entity_id":"person","role":"traveler"},'
        '{"entity_id":"candidate","role":"uncertain"}],'
        '"details":[{"field":"destination","value":"示例城市"}],"mode":"plan","polarity":"negated"}]}'
    )
    assert [prompt.compact(card) for card in prompt.packets(data)] == [expected]


def reference_packets(data, max_chars):
    current = []
    for record in knowledge_view.records(data):
        if current and len(prompt.compact(prompt.card(data, current + [record]))) > max_chars:
            yield prompt.card(data, current)
            current = []
        current.append(record)
    if current:
        yield prompt.card(data, current)


@pytest.mark.parametrize('max_chars', [1000, 1600, 4096, 16000, 1000000])
@pytest.mark.parametrize('self_present', [False, True])
def test_packet_lengths_match_existing_boundaries_with_unicode_and_new_identities(
        tmp_path, max_chars, self_present):
    data = knowledge(tmp_path)
    if not self_present:
        data['entities'] = [entity for entity in data['entities']
                            if entity.get('account') != data['subject']['self_account']]
    data['entities'].extend(dict(id=f'extra-{index}', kind='person', account='',
        label='候选\\"\n' + str(index), identity_status='unresolved_cross_batch', evidence_refs=[])
        for index in range(30))
    event = data['events'][0]
    data['events'] = [dict(event, id=f'event-{index}', description='行程\\"\n' * (index % 9 + 1),
        participants=[dict(entity_id=f'extra-{index % 30}', role='uncertain'),
                      dict(entity_id='missing', role='uncertain')]) for index in range(100)]
    data['events'].insert(3, dict(event, id='oversized', description='大段记录' * 10000))
    expected = [prompt.compact(item) for item in reference_packets(data, max_chars)]
    assert [prompt.compact(item) for item in prompt.packets(data, max_chars)] == expected


def test_packet_sizing_does_not_reserialize_growing_cards(tmp_path, monkeypatch):
    data = knowledge(tmp_path)
    event = data['events'][0]
    data['events'] = [dict(event, id=f'event-{index}') for index in range(200)]
    calls, original = [], prompt.card

    def counted_card(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(prompt, 'card', counted_card)
    cards = list(prompt.packets(data, max_chars=1000000))
    assert len(cards) == 1 and len(cards[0]['facts']) == 203
    assert calls == [0, 203, 203]


def test_unchanged_export_preserves_file_fingerprints_and_bytes(tmp_path, monkeypatch):
    source, output = tmp_path / 'knowledge.json', tmp_path / 'export'
    source.write_text(json.dumps(knowledge(tmp_path)))
    report = prompt.export(source, output, max_chars=1600)
    before = {path: (path.read_bytes(), path.stat()) for path in output.iterdir()}
    monkeypatch.setattr(prompt, 'atomic_write', lambda *args: pytest.fail('unchanged export wrote a file'))
    assert prompt.export(source, output, max_chars=1600) == report
    for path, (content, stamp) in before.items():
        assert path.read_bytes() == content
        assert (path.stat().st_mtime_ns, path.stat().st_ctime_ns, path.stat().st_size) == (
            stamp.st_mtime_ns, stamp.st_ctime_ns, stamp.st_size)


@pytest.mark.parametrize('name', ['cards.jsonl', 'sources.json', 'manifest.json'])
@pytest.mark.parametrize('mutation', ['remove', 'change'])
def test_export_repairs_missing_or_changed_projections(tmp_path, name, mutation):
    source, output = tmp_path / 'knowledge.json', tmp_path / 'export'
    source.write_text(json.dumps(knowledge(tmp_path)))
    report = prompt.export(source, output)
    before = {path: path.read_bytes() for path in output.iterdir()}
    path = output / name
    if mutation == 'remove':
        path.unlink()
    else:
        path.write_text('{}')
    assert prompt.export(source, output) == report
    assert all(path.read_bytes() == content for path, content in before.items())


def test_export_rebuilds_for_changed_source_or_packet_size(tmp_path):
    source, output = tmp_path / 'knowledge.json', tmp_path / 'export'
    data = knowledge(tmp_path)
    source.write_text(json.dumps(data))
    first = prompt.export(source, output, max_chars=1600)
    data['events'][0]['description'] = '已出发' * 200
    source.write_text(json.dumps(data))
    second = prompt.export(source, output, max_chars=1600)
    third = prompt.export(source, output, max_chars=1000000)
    assert second['source_sha256'] != first['source_sha256']
    assert second['files']['cards.jsonl'] != first['files']['cards.jsonl']
    assert third['source_sha256'] == second['source_sha256']
    assert third['max_chars'] == 1000000 and third['packets'] == 1
