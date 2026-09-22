from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import pytest
pytest.importorskip('torch')  # 可选重依赖
pytest.importorskip('xgboost')  # 可选重依赖
from src.bootstrap.history import write_rows
from src.config import ConfigError, sha256_file
from src.generator.few_shot import PersonaFewShotRetriever
from src.generator.history_sources import digest
from src.generator.ranker_expansion import cap_rows, isolated_rows, prepare, read_inventory
from src.generator.ranker_expansion.reuse import import_labels, reuse_source
from src.generator.ranker_samples import prepare as original_prepare
from src.generator.ranker_labels import _save, _saved
from src.iteration.storage import read_json, write_json
from tests.leakage_support import historical, mechanical
from tests.test_ranker_samples import teacher_at
from tests.test_ranker_labels import row


def setup_batch(tmp_path):
    fixture = historical(tmp_path / 'history')
    path = fixture.directory / 'gen_learning.jsonl'
    write_rows(path, fixture.cases[:3])
    purposes = read_json(fixture.directory / 'purposes.json')
    purposes['roles']['gen_learning']['sha256'] = sha256_file(path)
    write_json(fixture.directory / 'purposes.json', purposes)
    teacher = teacher_at(tmp_path, fixture, 0)
    generator = tmp_path / 'generator'
    mechanical(generator)
    write_json(generator / 'config.json', dict(max_shots_per_case=3, shots_char_budget=2500))
    old = tmp_path / 'old'
    original_prepare(fixture.directory, teacher, generator, old, progress=lambda *a, **k: None)
    return fixture, teacher, generator, old


def test_expansion_deduplicates_roles_reuses_candidates_and_resumes(tmp_path, monkeypatch):
    fixture, teacher, generator, old = setup_batch(tmp_path)
    output = tmp_path / 'expanded'
    args = (fixture.directory, teacher, generator, output)
    kw = dict(observations=3, reuse_inventory=old, progress=lambda *a, **k: None)
    calls = []
    original = PersonaFewShotRetriever.retrieve
    def retrieve(self, *a, **k):
        calls.append(k['history_case']['case_id'])
        assert k['history_case']['case_id'] != fixture.cases[1]['case_id']
        return original(self, *a, **k)
    monkeypatch.setattr(PersonaFewShotRetriever, 'retrieve', retrieve)
    result = prepare(*args, **kw)
    assert result['candidate_observations'] == 3
    assert result['targets'] == 2 and result['shortfall'] == 0
    assert result['capacity']['union_targets'] == 3
    assert result['capacity']['excluded_targets'] == 1
    assert result['reused_candidate_observations'] == 1
    _, records = read_inventory(output)
    ids = [(r['target_id'], e['example']['id']) for r in records for e in r['candidates']]
    assert len(ids) == len(set(ids)) == 3
    assert result['split']['fit']['observations'] == 1
    assert result['split']['validation']['observations'] == 2
    assert len(calls) == 2
    assert prepare(*args, **kw) == result
    assert len(calls) == 2
    with pytest.raises(ConfigError, match='frozen artifact changed'):
        prepare(*args, **{**kw, 'observations': 4})


def test_shortfall_is_not_padded_with_duplicate_observations(tmp_path):
    fixture, teacher, generator, old = setup_batch(tmp_path)
    result = prepare(fixture.directory, teacher, generator, tmp_path / 'expanded',
                     observations=100, reuse_inventory=old, progress=lambda *a, **k: None)
    assert result['status'] == 'inventory_shortfall'
    assert result['candidate_observations'] == 3 and result['shortfall'] == 97


def test_combined_split_purges_old_fit_material_against_new_validation():
    rows = [row(i) for i in range(10)]
    rows[0]['target']['source_span']['end_timestamp'] = 800
    selected, audit = isolated_rows(rows)
    assert '0' not in {r['target_id'] for r in selected}
    assert audit['purged_during_expansion'][0]['target_id'] == '0'
    assert sum(len(r['candidates']) for r in cap_rows(selected, 5)) == 5


def batches(tmp_path):
    old_row = row(1, candidates=1)
    new_row = {**deepcopy(old_row), 'split': 'validation', 'payload_sha256': 'new-row'}
    manifest = dict(policy={'draw': 0}, data='d', teacher='j', generator='g', clients={'model': 'same'},
                    teacher_client={'model': 'same'}, label_usage='supervision', inputs={'/runtime': 'same'})
    old = SimpleNamespace(inventory=tmp_path / 'old', output=tmp_path / 'old' / 'labeling',
        manifest={**manifest, 'original_manifest_sha256': 'old'}, items=[(old_row, old_row['candidates'][0])],
        states={}, seal=SimpleNamespace(check=lambda: None))
    new = SimpleNamespace(inventory=tmp_path / 'new', output=tmp_path / 'new' / 'labeling',
        manifest={**manifest, 'original_manifest_sha256': 'new'}, items=[(new_row, new_row['candidates'][0])],
        states={}, seal=SimpleNamespace(check=lambda: None))
    identity = digest([old_row['target_id'], old_row['candidates'][0]['example']['id']])
    binding = digest([digest(old.manifest), old_row['payload_sha256'], old_row['candidates'][0]])
    path = old.output / 'observations' / (identity + '.json')
    _save(path, dict(binding=binding, status='complete', generation={'replies': ['生成']},
                    identified_ai=False, z=1, split='fit', target_id='1', example_id='e0'))
    old.states[identity] = _saved(path, binding)
    return new, old, identity


def test_reuse_preserves_label_and_trace_provenance_but_updates_split(tmp_path):
    new, old, identity = batches(tmp_path)
    assert import_labels(new, old)['complete'] == 1
    value = new.states[identity]
    assert value['z'] == 1 and value['split'] == 'validation'
    assert value['reused_from']['original_split'] == 'fit'
    assert old.states[identity]['split'] == 'fit'
    assert import_labels(new, old)['already_present'] == 1


def test_expanded_batch_cannot_skip_or_change_reuse_source(tmp_path):
    source = tmp_path / 'old'
    inventory = tmp_path / 'new'
    assert reuse_source(inventory) is None
    write_json(inventory / 'expansion_plan.json', {'reuse_inventory': str(source)})
    assert reuse_source(inventory) == source
    assert reuse_source(inventory, source) == source
    with pytest.raises(ConfigError, match='复用来源'):
        reuse_source(inventory, tmp_path / 'other')


def test_label_queue_waits_for_completed_inventory(tmp_path, monkeypatch):
    from scripts import label_fewshot_ranker as cli
    sleeps = []
    def finish(seconds):
        sleeps.append(seconds)
        assert read_json(tmp_path / 'labeling' / 'queue.json')['status'] == 'waiting_for_inventory'
        write_json(tmp_path / 'inventory.json', {})
        write_json(tmp_path / 'report.json', {})
    monkeypatch.setattr(cli.time, 'sleep', finish)
    cli.wait_for_inventory(tmp_path)
    cli.wait_for_inventory(tmp_path)
    assert sleeps == [30]


@pytest.mark.parametrize('change', ['client', 'runtime', 'content'])
def test_reuse_rejects_changed_semantic_conditions(tmp_path, change):
    new, old, _ = batches(tmp_path)
    if change == 'client':
        new.manifest['clients'] = {'model': 'changed'}
    elif change == 'runtime':
        new.manifest['inputs'] = {'/runtime': 'changed'}
    else:
        new.items[0][0]['target']['chat_type'] = 'group'
    with pytest.raises(ConfigError, match='变化'):
        import_labels(new, old)
