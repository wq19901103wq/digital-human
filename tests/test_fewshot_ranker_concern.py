"""Independent concern evidence, frozen-cache reuse and fixed-recipe deployment."""
from contextlib import nullcontext
from copy import deepcopy
import json

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('xgboost')
from src.config import ConfigError, sha256_file
from src.generator import learned_sources, ranker_report
from src.generator.fewshot_ranker import concern, concern_comparison as comparison, supplemental
from src.generator.fewshot_ranker.features import context_view
from src.generator.history_sources import digest
from src.iteration.storage import read_json, write_json
from tests.test_fewshot_ranker_actions import control_model, frozen_control
from tests.test_fewshot_ranker_lr import feature_row, frozen_source
from tests.test_fewshot_ranker_training import record, synthetic


class Client:
    calls = 0
    def cache_identity(self):
        return {'model': 'test'}
    def run(self, prompt, schema):
        self.calls += 1
        return json.dumps(dict(self_concern_state='concern', later_incoming_basis='reassurance_or_opinion',
                               self_positions=[0], incoming_positions=[1]))


def concerned_record():
    value = record(10)
    value['context'][0].update(is_self=True, text='我还是担心钱不够')
    value['context'][1]['text'] = '会好的'
    return value


def test_concern_tasks_are_independent_answer_blind_and_shared():
    target, example = concerned_record(), dict(record(), id='e')
    groups = [dict(target_id='a', target=target, candidates=[dict(example=example)]),
              dict(target_id='b', target=deepcopy(target), candidates=[dict(example=example)])]
    tasks, refs = concern.prepare(groups, {'model': 'test'})
    assert len(tasks) == 2 and refs[('a', 'e')] == refs[('b', 'e')]
    assert all('TARGET_SECRET' not in task['prompt'] and 'GENERATED_SECRET' not in task['prompt']
               and '"reply"' not in task['prompt'] for task in tasks.values())
    before = deepcopy(tasks)
    target.update(human_reply=['changed future'], z=0)
    example['reply'] = ['changed historical response']
    assert concern.prepare(groups, {'model': 'test'})[0] == before
    target['context'][0]['timestamp'] = 999999
    with pytest.raises(ConfigError, match='cutoff'):
        concern.prepare(groups, {'model': 'test'})


def test_concern_evidence_validates_speaker_order_and_cache_integrity(tmp_path):
    client = Client()
    key, request = concern.task(context_view(concerned_record()), client.cache_identity())
    value = supplemental.extract_one(tmp_path, key, request, client)
    assert supplemental.extract_one(tmp_path, key, request, client) == value and client.calls == 1
    with pytest.raises(ConfigError, match='speaker or position'):
        concern.validate(dict(value, self_positions=[1]), request)
    with pytest.raises(ConfigError, match='speaker or position'):
        concern.validate(dict(value, self_positions=[0, 0]), request)
    assert all('uniqueItems' not in request['schema']['properties'][name] for name in concern.POSITIONS)
    with pytest.raises(ConfigError, match='corresponding evidence'):
        concern.validate(dict(value, self_positions=[]), request)
    with pytest.raises(ConfigError, match='corresponding evidence'):
        concern.validate(dict(value, self_concern_state='unstated'), request)
    saved = read_json(tmp_path/(key+'.json'))
    saved['features']['self_concern_state'] = 'reassured'
    write_json(tmp_path/(key+'.json'), saved)
    with pytest.raises(ConfigError, match='cache binding'):
        supplemental.cached(tmp_path, key, request)


def test_absent_self_requires_no_llm_and_run_resumes(tmp_path):
    client = Client()
    key, request = concern.task(context_view(record()), client.cache_identity())
    values = supplemental.run({key: request}, tmp_path/'cache', tmp_path/'run', client, workers=1)
    assert client.calls == 0 and values[key]['self_concern_state'] == 'unstated'
    assert supplemental.run({key: request}, tmp_path/'cache', tmp_path/'run', client, workers=1) == values
    assert read_json(tmp_path/'run/feature_progress.json')['reused'] == 1


def test_invalid_supplement_is_preserved_for_diagnosis_and_never_reused(tmp_path):
    class InvalidClient(Client):
        def run(self, prompt, schema):
            return json.dumps(dict(self_concern_state='unstated',
                later_incoming_basis='reassurance_or_opinion', self_positions=[], incoming_positions=[1]))
    client = InvalidClient()
    key, request = concern.task(context_view(concerned_record()), client.cache_identity())
    for _ in range(2):
        with pytest.raises(ConfigError, match='must follow'):
            supplemental.extract_one(tmp_path, key, request, client)
        assert supplemental.cached(tmp_path, key, request) is None
    records = list((tmp_path/'failures'/key).glob('*.json'))
    assert len(records) == 2
    for path in records:
        failure = read_json(path)
        assert failure['key'] == failure['request_sha256'] == digest(request)
        assert json.loads(failure['raw'])['self_concern_state'] == 'unstated'
        assert failure['payload_sha256'] == digest({k: v for k, v in failure.items() if k != 'payload_sha256'})
    valid = Client()
    assert supplemental.extract_one(tmp_path, key, request, valid)['self_concern_state'] == 'concern'
    assert valid.calls == 1 and supplemental.cached(tmp_path, key, request) is not None


def test_bounded_correction_has_its_own_cache_identity_and_strict_evidence(tmp_path):
    class CorrectingClient(Client):
        prompts = []
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return json.dumps(dict(self_concern_state='unstated',
                    later_incoming_basis='concrete_condition', self_positions=[], incoming_positions=[1]))
            return super().run(prompt, schema)
    client = CorrectingClient()
    key, request = concern.task(context_view(concerned_record()), client.cache_identity())
    value = supplemental.extract_one(tmp_path, key, request, client)
    assert value['self_concern_state'] == 'concern'
    assert len(client.prompts) == 2 and client.prompts[1].startswith(request['prompt'])
    assert not (tmp_path/(key+'.json')).exists()
    correction = read_json(supplemental.repair_path(tmp_path, key))
    assert correction['key'] != key and correction['parent_key'] == key
    assert correction['key'] == digest(supplemental.repair_request(request,
        correction['failed_raw'], correction['validation_error']))
    assert supplemental.extract_one(tmp_path, key, request, client) == value and len(client.prompts) == 2
    correction['features']['incoming_positions'] = [0]
    correction['payload_sha256'] = digest({k: v for k, v in correction.items() if k != 'payload_sha256'})
    write_json(supplemental.repair_path(tmp_path, key), correction)
    with pytest.raises(ConfigError, match='speaker or position'):
        supplemental.cached(tmp_path, key, request)


def test_concern_eight_fields_exclude_positions_and_fit_does_not_read_outer_labels():
    groups, rows, observations = synthetic()
    original = comparison.feature_rows([{**feature_row(), **row} for row in rows], True)
    control, _, detail = control_model(groups, observations, original)
    value = dict(self_concern_state='concern', later_incoming_basis='concrete_condition',
                 self_positions=[999], incoming_positions=[1000])
    expanded = [concern.expand(row, value, value) for row in original]
    assert len(set(expanded[0])-set(original[0])) == 8
    assert not any('positions' in key for key in expanded[0])
    model, scores, fitted = comparison.fit_selected(groups, observations, expanded, control, detail,
                                                     transform=concern.TRANSFORM)
    changed = deepcopy(observations)
    for row in changed:
        if row['split'] == 'validation':
            row['z'] = 1-row['z']
    second, _, other = comparison.fit_selected(groups, changed, expanded, control, detail,
                                                transform=concern.TRANSFORM)
    assert model == second and fitted == other
    assert model['recipe'] == control['recipe'] and model['rounds'] == control['rounds']
    assert model['feature_transform'] == concern.TRANSFORM and 'identity=15' not in model['vocabulary']
    from src.generator.fewshot_ranker.boosting import score_document
    np.testing.assert_allclose(score_document(model, expanded), scores)


@pytest.mark.parametrize('enabled', [False, True])
def test_selector_scores_training_features_and_reuses_both_caches(tmp_path, monkeypatch, enabled):
    from src.generator import learned_selection as selection
    from src.generator.fewshot_ranker import boosting, extraction, training
    from tests.test_fewshot_selection import Renderer

    groups, rows, observations = synthetic()
    original = comparison.feature_rows([{**feature_row(), **row} for row in rows], True)
    model, _, detail = control_model(groups, observations, original)
    supplement = dict(self_concern_state='concern', later_incoming_basis='reassurance_or_opinion',
                      self_positions=[0], incoming_positions=[1])
    if enabled:
        expanded = [concern.expand(row, supplement, supplement) for row in original]
        model, _, _ = comparison.fit_selected(groups, observations, expanded, model, detail,
                                               transform=concern.TRANSFORM)
    calls = []
    class ServingClient(Client):
        def __init__(self, config):
            pass
        def run(self, prompt, schema):
            calls.append(prompt)
            return (super().run(prompt, schema) if prompt.startswith(concern.INSTRUCTION) else
                    json.dumps({field: 'unknown' for field in schema['properties']}))
    monkeypatch.setattr(selection, 'CodexJudgeClient', ServingClient)
    write_json(tmp_path/'model.json', model)
    write_json(tmp_path/'provenance.json', dict(feature_identity={'model': 'test'}, feature_config={}))
    selector = selection.LearnedSelector(tmp_path, tmp_path/'cache')
    target = dict(concerned_record(), case_id='target')
    examples = [dict(record(i), id=f'example-{i}') for i in (1, 2)]
    checks = []
    selected = selector.select(target, examples, count=2, budget=2500, retriever=Renderer(),
                               check=lambda: checks.append(True))
    tasks, refs = selector.tasks(target, examples)
    values = {key: supplemental.cached(selector.cache, key, task) for key, task in tasks.items()}
    plain_groups = [dict(target_id='target', target=target,
                        candidates=[dict(example=row) for row in examples])]
    _, plain_refs = extraction.prepare(plain_groups, {'model': 'test'})
    expected = comparison.feature_rows(training.assemble(plain_groups, plain_refs, values), True)
    if enabled:
        expected = [concern.expand(row, *(values[k] for k in refs[('target', example['id'])][3:]))
                    for row, example in zip(expected, examples)]
    actual = selection.feature_rows(target, examples, refs, values, transform=model['feature_transform'])
    assert actual == expected
    assert selected == selection.choose(examples, boosting.score_document(model, expected), Renderer(),
                                         count=2, budget=2500)
    assert sum(prompt.startswith(concern.INSTRUCTION) for prompt in calls) == int(enabled)
    count = len(calls)
    assert selector.select(target, examples, count=2, budget=2500, retriever=Renderer(),
                           check=lambda: checks.append(True)) == selected
    assert len(calls) == count and len(checks) == 4


def test_concern_comparison_seals_inputs_and_reuses_complete_run(tmp_path, monkeypatch):
    source = frozen_source(tmp_path, monkeypatch)
    _, groups, observations, summary, _, _ = ranker_report.load_completed(source)
    for group in groups:
        group['candidates'] = [dict(example=dict(record(), id=observations[i]['example_id']))
                               for i in group['indices']]
    monkeypatch.setattr(ranker_report, 'load_dataset', lambda path: (groups, observations, summary,
                        {'source': 'frozen_test_inventory'}))
    manifest = read_json(source/'manifest.json')
    manifest['features'] = dict(client=Client().cache_identity())
    write_json(source/'manifest.json', manifest)
    matrix = read_json(source/'feature_matrix.json')
    matrix['manifest_sha256'] = digest(manifest)
    write_json(source/'feature_matrix.json', matrix)
    write_json(source/'completed.json', dict(manifest_sha256=digest(manifest), artifacts={
        name: sha256_file(source/name) for name in ('predictions.json', 'feature_matrix.json')}))
    write_json(source/'transport.json', dict(config={}))
    monkeypatch.setattr(comparison, 'CodexJudgeClient', lambda config: Client())
    monkeypatch.setattr(comparison.versions, 'switch_instance', lambda instance: None)
    monkeypatch.setattr(comparison.control, 'job', lambda output: nullcontext())
    reference, output = tmp_path/'reference', tmp_path/'concern'
    control = frozen_control(source, reference)
    control_hash = sha256_file(control/'model.json')
    args = dict(feature_cache=tmp_path/'cache', instance='test', workers=1)
    result = comparison.run(source, reference, output, **args)
    assert result['status'] == 'complete' and result['llm_contexts'] == 0
    assert result['supplemental_contexts'] == len(groups) and len(result['added_fields']) == 8
    assert sha256_file(control/'model.json') == control_hash
    model = output/comparison.METHOD
    parent, child = read_json(output/'manifest.json'), read_json(model/'manifest.json')
    assert learned_sources.deployment_recipe(parent, child, reference, model, set()) == concern.TRANSFORM
    monkeypatch.setattr(comparison, 'fit_selected', lambda *a, **k: pytest.fail('must reuse fit'))
    assert comparison.run(source, reference, output, **args) == result
    values = read_json(output/'supplemental_values.json')
    values.pop(next(iter(values)))
    write_json(output/'supplemental_values.json', values)
    with pytest.raises(ConfigError, match='coverage differs'):
        learned_sources.verify_concern(parent, output, set())
    with pytest.raises(ConfigError, match='artifacts changed'):
        comparison.run(source, reference, output, **args)
