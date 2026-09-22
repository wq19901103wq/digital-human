"""The rerun adapter must preserve source checks at request and resume boundaries."""
import json
from types import SimpleNamespace

import pytest

from scripts.legacy import rerun_guarded_history_judge as job
from src import cache
from src.config import ConfigError, sha256_file
from src.iteration.storage import write_json


@pytest.mark.parametrize('when', ['before', 'during'])
def test_preflight_initial_blocks_source_mutation(tmp_path, when):
    source = tmp_path / 'source'
    source.write_text('clean')
    seal = job.guard.RunSeal(files=[source])
    calls = []

    def request(prompt):
        calls.append(prompt)
        source.write_text('changed during request')
        return json.dumps({'human_option': 'A', 'reason': 'ok'})

    if when == 'before':
        source.write_text('changed before request')
    with pytest.raises(ConfigError, match='冻结输入变化'):
        job.initial_request(SimpleNamespace(run=request), 'blind input', seal.check)
    assert len(calls) == (when == 'during')


def test_prepare_will_not_resign_old_completed_outputs(tmp_path):
    write_json(tmp_path / 'generations.json', {'entries': {'x': {'status': 'ok'}}})
    with pytest.raises(ConfigError, match='不能补签'):
        job.prepare(tmp_path)
    assert not (tmp_path / 'spec.json').exists()


def evaluation_fixture(tmp_path, monkeypatch):
    study = tmp_path / 'study'
    spec = {'data_ref': 'd', 'generator_ref': 'g', 'dataset': 'training',
            'purpose_snapshot': {}, 'protocol': {'flip_extra_rounds': 2}}
    write_json(study / 'spec.json', spec)
    recipe = {**spec, 'dataset': 'development'}
    write_json(study / 'development/recipe.json', recipe)
    case = {'case_id': 'one', 'human_reply': ['human']}
    entry = {'status': 'ok', 'input_sha256': cache.digest(case), 'replies': ['ai'],
             'output_sha256': cache.digest(['ai'])}
    generated = {'identity': cache.digest(recipe), 'entries': {'one': entry}}
    monkeypatch.setattr(job.versions, 'PRIVATE', tmp_path / 'private')
    monkeypatch.setattr(job.versions, 'data_version_dir', lambda ref: tmp_path / 'data')
    monkeypatch.setattr(job.versions, 'generator_dir', lambda ref: tmp_path / 'generator')
    monkeypatch.setattr(job.versions, 'judge_dir', lambda ref: {'dir': tmp_path / ref, 'config': {}})
    monkeypatch.setattr(job.datasets, 'assert_pack', lambda *args: None)
    monkeypatch.setattr(job.guard, 'verify_generation', lambda recipe, cases: SimpleNamespace(check=lambda: None))
    monkeypatch.setattr(job.guard, 'require_materials', lambda *args: {'verified': True})
    monkeypatch.setattr(job.gates, 'materials', lambda spec: {'frozen': True})
    return study, spec, case, generated


def test_new_evaluation_binds_generation_and_learning_before_save(tmp_path, monkeypatch):
    study, spec, case, generated = evaluation_fixture(tmp_path, monkeypatch)
    binds = []

    def bind(frozen):
        assert not (job.versions.PRIVATE / 'experiments' / frozen['id'] / 'spec.json').exists()
        pack = job.base.read(job.versions.PRIVATE / 'judge_eval' / frozen['pack_ref'] / 'pack.json')
        assert pack['generation_proof']['recipe'] == {**spec, 'dataset': 'development'}
        assert pack['generation_proof']['rows_sha256'] == cache.digest(pack['rows'])
        binds.append(True)
        frozen['learning_snapshot'] = {'verified': True}

    monkeypatch.setattr(job.guard, 'bind', bind)
    target = job.make_evaluation.__wrapped__(study, spec, {'luna': 'j-luna', 'sol': 'j-sol'}, [case], generated)
    frozen = job.base.read(target / 'spec.json')
    assert binds == [True]
    assert frozen['learning_snapshot'] == {'verified': True}
    assert frozen['material_repair']['adoption_allowed'] is False
    assert frozen['protocol']['flip_extra_rounds'] == 2
    before = sha256_file(target / 'spec.json')
    monkeypatch.setattr(job.guard, 'verify_repair', lambda spec: None)
    monkeypatch.setattr(job.guard, 'verify', lambda spec: (_ for _ in ()).throw(ConfigError('unproven')))
    with pytest.raises(ConfigError, match='unproven'):
        job.make_evaluation.__wrapped__(study, spec, {'luna': 'j-luna', 'sol': 'j-sol'}, [case], generated)
    assert sha256_file(target / 'spec.json') == before
    assert binds == [True]


@pytest.mark.parametrize('change', ['recipe', 'identity', 'reply'])
def test_changed_generation_cannot_create_pack(tmp_path, monkeypatch, change):
    study, spec, case, generated = evaluation_fixture(tmp_path, monkeypatch)
    if change == 'recipe':
        write_json(study / 'development/recipe.json', {**spec, 'dataset': 'fixed_test'})
    elif change == 'identity':
        generated['identity'] = 'old'
    else:
        generated['entries']['one']['replies'] = ['changed']
    with pytest.raises(ConfigError):
        job.make_evaluation.__wrapped__(study, spec, {'luna': 'j-luna', 'sol': 'j-sol'}, [case], generated)
    assert not list(job.versions.PRIVATE.rglob('pack.json'))
