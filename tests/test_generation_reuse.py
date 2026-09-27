"""Saved generation reuse binds inputs, never selects by Judge outcome."""
from copy import deepcopy
import json

import pytest

from src.config import ConfigError
from src.iteration import generation_reuse as reuse


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse.versions, 'PRIVATE', tmp_path)
    # Historical executor authentication is independently covered by runtime tests.
    monkeypatch.setattr(reuse.runtime, 'verify_historical_implementation', lambda *a: None)
    snapshot = dict(data_ref='d-1', sources={'messages': 'hash'}, purposes={'dev': 'hash'},
                    materials={'g-1': {'config': 'hash'}}, guard_code={'code': 'hash'})
    recipe = dict(dataset='development', data_ref='d-1', generator_ref='g-1',
                  learning_snapshot=deepcopy(snapshot))
    spec = dict(kind='gen_ab', dataset='development', data_ref='d-1',
                baseline_ref='g-0', candidate_ref='g-1', judge_ref='j-1',
                protocol={'force_reply': True}, learning_snapshot=deepcopy(snapshot))
    case = dict(case_id='case-1', context=['history'], human_reply=['answer'])
    directory = tmp_path/'experiments'/'source'
    save(directory/'spec.json', spec)
    save(directory/'runtime.json', {'frozen': True})
    trace = dict(case_id='case-1', case=case, trace_ref='first', started_at=1,
        status='failed', versions={k: spec[k] for k in
            ('kind', 'dataset', 'data_ref', 'baseline_ref', 'candidate_ref', 'judge_ref')},
        operations=[dict(kind='generation', branch='candidate', round=0, status='ok',
            input={'forced_reply': True}, result={'replies': ['generated']}),
            dict(kind='judge', status='failed')])
    save(directory/'traces/first.json', trace)
    later = deepcopy(trace)
    later.update(trace_ref='later', started_at=2, status='ok')
    later['operations'][0]['result']['replies'] = ['later generation']
    save(directory/'traces/later.json', later)
    return recipe, case, spec, trace, directory


def test_earliest_generation_reused_even_if_judge_failed(source):
    recipe, case, _, _, _ = source
    missing = {**case, 'case_id': 'missing'}
    recipe['generation_reuse'] = reuse.prepare(recipe, [case, missing], 'source')
    rows, _ = reuse.load(recipe, [case, missing])
    assert list(rows) == ['case-1']
    assert rows['case-1']['result']['replies'] == ['generated']
    reuse.verify_rows(recipe, [case, missing], [{**case, 'ai_replies': ['generated']}])
    with pytest.raises(ConfigError, match='复用回复被替换'):
        reuse.verify_rows(recipe, [case, missing], [{**case, 'ai_replies': ['altered']}])


@pytest.mark.parametrize('damage', ['case', 'version', 'branch', 'round', 'forced'])
def test_exact_generation_conditions_required(source, damage):
    _, case, spec, trace, _ = source
    trace = deepcopy(trace)
    if damage == 'case':
        trace['case']['context'] = ['different']
    elif damage == 'version':
        trace['versions']['candidate_ref'] = 'g-other'
    else:
        op = trace['operations'][0]
        if damage == 'forced':
            op['input']['forced_reply'] = False
        else:
            op[damage] = 'baseline' if damage == 'branch' else 1
    with pytest.raises(ConfigError):
        reuse.operation(trace, spec, 'candidate', case)


@pytest.mark.parametrize('name', ['spec.json', 'runtime.json', 'traces/first.json'])
def test_pinned_source_cannot_change(source, name):
    recipe, case, _, _, directory = source
    recipe['generation_reuse'] = reuse.prepare(recipe, [case], 'source')
    path = directory/name
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ConfigError):
        reuse.load(recipe, [case])


def test_changed_data_or_materials_not_reusable(source):
    recipe, case, _, _, _ = source
    recipe['learning_snapshot']['materials']['g-1']['config'] = 'changed'
    with pytest.raises(ConfigError):
        reuse.prepare(recipe, [case], 'source')
