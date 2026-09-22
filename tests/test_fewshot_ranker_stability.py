"""Repeat-label orchestration uses distinct draws and durable checkpoints."""
from types import SimpleNamespace

import pytest

from src.config import ConfigError, sha256_file
from src.generator.fewshot_ranker import stability
from src.generator.history_sources import digest
from src.generator import ranker_labels as labels
from src.iteration.storage import read_json, write_json
from tests.test_ranker_labels import row


def fixture(tmp_path, monkeypatch):
    inventory = tmp_path/'inventory'
    settings = tmp_path/'settings.yaml'
    settings.write_text('fixture')
    manifest = dict(data='d', teacher='j', generator='g',
        inputs={str(settings): sha256_file(settings)}, clients={'private': 'fake'}, teacher_client='fake')
    write_json(inventory/'labeling/manifest.json', manifest)
    groups, observations = [], []
    original_generator = SimpleNamespace(generate_one=lambda *args: {'replies': ['original']})
    for i in (1, 2):
        sample = row(i)
        sample['split'] = 'fit' if i == 1 else 'validation'
        write_json(inventory/'targets'/f'{i}.json', sample)
        indices = []
        for j, entry in enumerate(sample['candidates']):
            labels.label_one(inventory/'labeling', manifest, sample, entry, original_generator,
                SimpleNamespace(is_ai=lambda *args, j=j: j == 1), lambda: None)
            indices.append(len(observations))
            observations.append({'z': int(j == 0)})
        groups.append(dict(target_id=str(i), split=sample['split'], candidates=sample['candidates'], indices=indices))
    write_json(tmp_path/'g/config.json', {'llm': {'timeout_seconds': 20}})
    write_json(tmp_path/'j/config.json', {})
    calls = dict(generated=[], judged=[], epochs=[])

    class Generator:
        def __init__(self, *args):
            pass

        def generate_one(self, case, example):
            calls['generated'].append((case['case_id'], example['id']))
            return {'replies': ['repeat']}

    class Judge:
        feature_client = SimpleNamespace(cache_identity=lambda: 'fake')

        def is_ai(self, case, replies):
            calls['judged'].append(replies)
            if calls.pop('fail_once', False):
                raise TimeoutError('judge retry')
            return case['case_id'] == '2'

    monkeypatch.setattr(stability, 'settings_path', lambda: settings)
    monkeypatch.setattr(stability, 'load_settings', lambda: {})
    monkeypatch.setattr(stability.versions, 'switch_instance', lambda *args: None)
    monkeypatch.setattr(stability, 'build_clients', lambda *args: {'private': SimpleNamespace(cache_identity=lambda: 'fake')})
    monkeypatch.setattr(stability, 'build_judge', lambda *args: Judge())
    monkeypatch.setattr(stability, 'SingleExampleGenerator', Generator)
    monkeypatch.setattr(stability, 'load_dataset', lambda *args: (groups, observations, {}, {'source': 'frozen'}))
    return (inventory, tmp_path/'d', tmp_path/'j', tmp_path/'g', 'test'), calls, settings


def test_repeat_namespaces_positive_cohort_resume_and_original_unchanged(tmp_path, monkeypatch):
    args, calls, settings = fixture(tmp_path, monkeypatch)
    originals = {p: p.read_bytes() for p in args[0].rglob('*.json')}
    batch = stability.StabilityBatch(*args, tmp_path/'out', 'regenerate', 2)
    assert len(batch.cohort) == 2
    assert len({digest(m) for _, m in batch.draws.values()}) == 2
    result = batch.run(workers=1)
    assert result['status'] == 'complete' and result['labeled'] == 4
    assert result['methods']['regenerate']['new_positive_count_histogram'] == {'0': 1, '1': 0, '2': 1}
    assert result['methods']['regenerate']['split_histograms']['validation'] == {'0': 1, '1': 0, '2': 0}
    assert len(calls['generated']) == len(calls['judged']) == 4
    assert all(example == 'e0' for _, example in calls['generated'])
    resumed = stability.StabilityBatch(*args, tmp_path/'out', 'regenerate', 2).run(workers=1)
    assert resumed['labeled'] == 4 and len(calls['generated']) == len(calls['judged']) == 4
    assert all(p.read_bytes() == value for p, value in originals.items())
    settings.write_text('changed')
    with pytest.raises(ConfigError, match='冻结输入变化'):
        batch.report()
    with pytest.raises(ConfigError, match='conditions changed'):
        stability.StabilityBatch(*args, tmp_path/'other', 'regenerate', 2)


def test_repeat_failure_reuses_generation_and_excludes_incomplete_sample(tmp_path, monkeypatch):
    args, calls, _ = fixture(tmp_path, monkeypatch)
    batch = stability.StabilityBatch(*args, tmp_path/'out', 'regenerate', 2)
    calls['fail_once'] = True
    key, failed = batch._one(batch.items[0])
    batch.states[key] = failed
    assert failed['status'] == 'failed' and failed['failed_stage'] == 'judge'
    value = batch.report()['methods']['regenerate']
    assert value['fully_retested_samples'] == 0 and value['new_positive_count_histogram'] == {'0': 0, '1': 0, '2': 0}
    completed = batch.run(workers=1)
    assert completed['labeled'] == 4 and len(calls['generated']) == 4 and len(calls['judged']) == 5


def test_fixed_reply_mode_never_regenerates_and_uses_distinct_namespace(tmp_path, monkeypatch):
    args, calls, _ = fixture(tmp_path, monkeypatch)
    batch = stability.StabilityBatch(*args, tmp_path/'out', 'both', 1)
    assert len({digest(m) for _, m in batch.draws.values()}) == 2
    tasks = [task for task in batch.items if task[0][0] == 'judge']
    for task in tasks:
        key, value = batch._one(task)
        batch.states[key] = value
    assert calls['generated'] == [] and calls['judged'] == [['original'], ['original']]
    assert batch.report()['methods']['judge']['fully_retested_samples'] == 2
    with pytest.raises(ConfigError, match='must not overwrite'):
        stability.StabilityBatch(*args, args[0]/'labeling/overlap', 'judge', 1)


def test_stability_transport_override_is_recorded_separately(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    before = (tmp_path/'g/config.json').read_bytes()
    result = stability.execute(*args, stability_output=tmp_path/'out', run=True,
                              workers=1, min_generation_timeout_seconds=120)
    assert result['status'] == 'complete'
    assert read_json(tmp_path/'out/transport.json')['effective_generation_timeouts'] == {'default': 120}
    assert (tmp_path/'g/config.json').read_bytes() == before
