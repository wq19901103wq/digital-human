"""Wiki organization must preserve scope, source boundaries and temporal uncertainty."""
import json

import pytest

from src.bootstrap.wiki_library import inventory, temporal_annotation, units_for
from src.bootstrap.wiki_library_extract import organize_library, parse_result


def test_selected_pages_do_not_extract_the_whole_library(tmp_path):
    root = tmp_path / 'wiki'
    (root / 'users').mkdir(parents=True)
    (root / 'groups').mkdir()
    (root / 'users' / 'a.md').write_text('# A\n- 旧职业\n')
    (root / 'users' / 'b.md').write_text('# B\n- 不应加载\n')
    (root / 'groups' / 'a.md').write_text('# A\n- 群聊\n')
    docs = inventory(root, ['users/a.md'])
    assert len(docs) == 1
    assert docs[0]['kind'] == 'person'
    assert docs[0]['title'] == 'A'
    with pytest.raises(ValueError, match='not found'):
        inventory(root, ['users/missing.md'])


def test_chunks_keep_all_lines_and_starting_section():
    doc = dict(id='a', kind='person', title='A', empty=False,
               lines=['## 旧工作', '一' * 20, '二' * 20, '## 家庭', '三' * 20])
    units = units_for([doc], max_chars=65)
    assert [tuple(row) for u in units for row in u['lines']] == list(enumerate(doc['lines'], 1))
    assert units[0]['heading'] == ''
    assert units[1]['heading'] == '旧工作'


def test_organization_does_not_refresh_old_facts_or_invent_expiry():
    base = dict(temporal_kind='mutable', reported_dates=['2021-03-01'], valid_from=None, valid_to=None)
    result = temporal_annotation(base, '2026-09-27')
    assert result['applicability'] == 'current_state_unconfirmed'
    assert result['last_confirmed_at'] is None
    assert result['valid_to'] is None
    assert not result['current_state_usable']
    assert temporal_annotation(dict(base, temporal_kind='historical'), '2026-09-27')['applicability'] == 'historical_only'
    with pytest.raises(ValueError, match='reversed'):
        temporal_annotation(dict(base, valid_from='2025-01-01', valid_to='2024-01-01'), '2026-09-27')


def test_extractor_cannot_silently_omit_pages_or_cite_unseen_lines():
    batch = [dict(id='u', page_id='p', lines=[[4, 'x']])]
    with pytest.raises(ValueError, match='every input unit'):
        parse_result('{"units": []}', batch)
    fact = dict(subject='@page', predicate='工作', value='曾在甲公司', object_name='甲公司',
                source_speaker='', scope='', category='background', evidence_mode='reported',
                temporal_kind='historical', reported_dates=[], valid_from=None, valid_to=None,
                roles=[], limitations='', lines=[5])
    with pytest.raises(ValueError, match='outside its input'):
        parse_result(json.dumps(dict(units=[dict(id='u', facts=[fact], gaps=[])])), batch)


def test_completed_extraction_is_reused_and_sources_have_no_prose(tmp_path, monkeypatch):
    root = tmp_path / 'wiki'
    (root / 'users').mkdir(parents=True)
    (root / 'users' / 'a.md').write_text('# A\n- 旧工作 PRIVATE-SOURCE\n')
    review = tmp_path / 'review.json'
    review.write_text(json.dumps(dict(identities=[], groups=[])))
    refined = tmp_path / 'refined.json'
    refined.write_text(json.dumps(dict(context=dict(addresses=[dict(id='unreviewed-hit')]))))
    curated = tmp_path / 'curated.json'
    curated.write_text(json.dumps(dict(source_manifest={}, evidence={}, identities=[], fact_claims=[], address_edges=[])))
    calls = []

    class Client:
        def __init__(self, config):
            pass

        def run(self, prompt, schema):
            from src.bootstrap.wiki_library_extract import PROMPT
            calls.append(prompt)
            units = json.loads(prompt[len(PROMPT):])
            fact = dict(subject='@page', predicate='旧工作', value='过去工作信息', object_name='',
                        source_speaker='', scope='', category='background', evidence_mode='summary_assertion',
                        temporal_kind='historical', reported_dates=[], valid_from=None, valid_to=None,
                        roles=[], limitations='', lines=[2])
            return json.dumps(dict(units=[dict(id=u['id'], facts=[fact], gaps=[]) for u in units]))

    monkeypatch.setattr('src.judge.corrected.CodexJudgeClient', Client)
    out = tmp_path / 'out'
    args = (root, out, review, refined, curated, {})
    result = organize_library(*args, pages=['users/a.md'])
    assert result['structured_summaries'] == 1
    assert organize_library(*args, pages=['users/a.md']) == result
    assert len(calls) == 1
    content = json.loads((out / 'content' / 'knowledge.json').read_text())
    assert content['address_edges'] == []  # Unreviewed rule hits are not endorsed.
    assert not content['runtime_usable']
    assert 'PRIVATE-SOURCE' not in json.dumps(content)
    assert all('line' in e and 'sha256' in e for e in content['evidence'].values())
    (root / 'users' / 'a.md').write_text('# A\n- Changed\n')
    with pytest.raises(ValueError, match='inputs changed'):
        organize_library(*args, pages=['users/a.md'])
