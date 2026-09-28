"""Compact cards preserve ownership/status and make coverage loss explicit."""
import json

from src.bootstrap import wiki_prompt as prompt, wiki_structured as structured
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
        assert all(prompt.entity_refs(f) <= ids for f in card['facts'])
    source = tmp_path / 'knowledge.json'
    source.write_text(json.dumps(data))
    report = prompt.export(source, tmp_path / 'export', max_chars=1600)
    assert report['records'] == 4 and not report['runtime_usable']
    assert '你来定时间' not in (tmp_path / 'export/cards.jsonl').read_text()
    assert 'path' not in (tmp_path / 'export/cards.jsonl').read_text()
    assert json.loads((tmp_path / 'export/sources.json').read_text())['evidence']['1']['line'] == 1


def test_projection_selects_exact_page_and_retains_uncertainty_with_budget(tmp_path):
    data = knowledge(tmp_path)
    assert prompt.project(data, {'self', 'stranger'}, 'private:self:stranger', lambda _: True) is None
    assert prompt.project(data, {'person'}, 'private:other:person', lambda _: True) is None
    result = prompt.project(data, {'person'}, 'private:self:person', lambda r: r == '1', max_chars=1800)
    assert result['coverage']['eligible'] == 2
    assert result['coverage']['included'] + result['coverage']['omitted_for_budget'] == 2
    assert len(prompt.compact(result)) <= 1800
    person = next(e for e in result['identities'] if e['account'] == 'person')
    assert person['name'] == 'person'  # Name evidence is not prior.
    assert all(f['last_observed_at'] == 1000 for f in result['facts'])
