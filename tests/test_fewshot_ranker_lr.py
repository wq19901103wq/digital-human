from copy import deepcopy
from itertools import permutations

import numpy as np
import pytest

import pytest
pytest.importorskip('torch')  # 可选重依赖
pytest.importorskip('xgboost')  # 可选重依赖
from src.config import ConfigError, sha256_file
from src.generator import ranker_report
from src.generator.fewshot_ranker import comparison, crosses, features, logistic, schema, training
from src.generator.history_sources import digest
from src.iteration.storage import read_json, write_json
from tests.test_fewshot_ranker_training import record, synthetic


def feature_row():
    target, example = record(20), record(1)
    context = {key: 'unknown' for key in schema.CONTEXT}
    reply = {key: 'unknown' for key in schema.REPLY}
    return features.combine(target, example, {**context, **features.context_local(target)},
        {**context, **features.context_local(example, example=True)}, {**reply, **features.reply_local(example)})


def test_crosses_use_only_cached_categories_without_mutating_source():
    row = feature_row()
    row.update({'target.needs_empathy': 'yes', 'example_reply.acknowledges_own_emotion': 'yes',
                'target.environment': 'unknown', 'example_context.environment': 'unknown'})
    saved = deepcopy(row)
    expanded = crosses.expand(row)
    assert row == saved and all(isinstance(v, str) for v in expanded.values())
    assert expanded['cross_v2.needs_empathy_x_reply_acknowledges_own_emotion'] == '["yes","yes"]'
    assert expanded['cross_v2.equal_context_environment'] == 'unknown'
    assert all(expanded[k] == v for k, v in row.items())
    del row['target.needs_empathy']
    with pytest.raises(ConfigError, match='Missing categorical'):
        crosses.expand(row)


def test_lr_fit_only_vocabulary_outer_label_isolation_and_serialized_scores():
    groups, rows, obs = synthetic()
    saved, scores, report = logistic.train(groups, obs, rows)
    assert saved['C'] in logistic.RECIPE['c_values']
    assert 'identity=15' not in saved['vocabulary']
    changed = deepcopy(obs)
    for row in changed:
        if row['split'] == 'validation':
            row['z'] = 1-row['z']
    second, _, _ = logistic.train(groups, changed, rows)
    assert saved == second
    np.testing.assert_allclose(logistic.score_document(saved, rows), scores)
    assert training.metrics(obs, scores)['group_auc_macro'] == 1
    # Features constant across a target's candidates cannot influence pairwise LR.
    assert all(saved['weights'][index] == 0 for key, index in saved['vocabulary'].items()
               if key.startswith('identity='))
    assert report['internal_split']['rule'].endswith('no_outer_validation_tuning')
    a, b = logistic.score_document(saved, [{'quality': '1'}, {'quality': '0'}])
    assert a > b and (a-b) == -(b-a)
    with pytest.raises(ConfigError, match='categorical strings'):
        logistic.encoder([{'quality': 1}], [0])


def test_available_positive_counts_include_all_negative_and_all_positive_contexts():
    obs = [dict(target_id=t, z=z) for t, labels in
           [('zero', [0, 0]), ('mixed', [1, 0]), ('all', [1, 1])] for z in labels]
    value = training.metrics(obs, [0]*6)
    assert value['contexts'] == 3 and value['contexts_with_positive'] == 2
    assert value['contexts_without_positive'] == 1 and value['contexts_all_positive'] == 1
    assert value['mixed_contexts'] == 1 and value['positive_count_histogram'] == {'0': 1, '1': 1, '2': 1}
    assert value['top1_positive_yield'] == .5 and value['top1_positive_given_available'] == .75
    assert value['positive_context_fraction'] == pytest.approx(2/3)
    assert value['candidate_count_histogram'] == {'2': 3}
    assert value['topk']['1']['hit_rate'] == value['top1_positive_yield']
    assert value['topk']['3']['hit_rate'] == pytest.approx(2/3)
    assert value['topk']['3']['mean_positive_count'] == 1
    assert value['topk']['5']['contexts_with_fewer_than_k_candidates'] == 3
    assert value['topk']['5']['hit_rate_given_available'] == 1


@pytest.mark.parametrize('labels,scores', [
    ([0, 1, 1, 0, 0], [0]*5),
    ([0, 1, 0, 1, 0], [3, 2, 2, 2, 1]),
    ([0, 0, 0, 0, 0], [5, 4, 3, 2, 1]),
    ([1, 1, 1, 1, 1], [5, 4, 3, 2, 1]),
    ([1, 0], [0, 0]),
])
def test_topk_tie_expectations_equal_enumerated_valid_rankings(labels, scores):
    orderings = [p for p in permutations(range(len(labels)))
                if all(scores[a] >= scores[b] for a, b in zip(p, p[1:]))]
    for k in (1, 3, 5):
        counts = [sum(labels[i] for i in p[:k]) for p in orderings]
        result = training.topk_expectation(labels, scores, k)
        assert result['hit_probability'] == pytest.approx(np.mean([c > 0 for c in counts]))
        assert result['positive_count'] == pytest.approx(np.mean(counts))
        assert result['selected_count'] == min(k, len(labels))


def test_topk_baseline_requires_frozen_choices_and_labels_with_same_context_denominator():
    obs = [dict(target_id=t, example_id=str(i), z=z) for t, labels in
           [('a', [0, 1, 0, 1, 0]), ('b', [1, 0, 0]), ('c', [1, 0, 0])]
           for i, z in enumerate(labels)]
    baseline = dict(a=['0', '2', '4'], b=['0', '1'], c=['0', 'missing', '2'])
    result = ranker_report.topk_comparison(obs, [r['z'] for r in obs], baseline)
    assert result['1']['compared_contexts'] == 3
    assert result['3']['compared_contexts'] == 1
    assert result['3']['excluded'] == dict(baseline_has_fewer_than_k_choices=1,
                                          baseline_topk_not_fully_labeled=1)
    assert result['3']['model']['hit_rate'] == 1
    assert result['3']['model']['mean_positive_count'] == 2
    assert result['3']['baseline']['hit_rate'] == 0
    assert result['3']['baseline']['contexts_with_positive'] == 1
    assert result['5']['compared_contexts'] == 0
    assert result['5']['baseline']['hit_rate'] is None


def frozen_source(tmp_path, monkeypatch):
    groups, rows, obs = synthetic()
    rows = [{**feature_row(), **row} for row in rows]
    inventory, source = tmp_path/'inventory', tmp_path/'source'
    binding = {'source': 'frozen_test_inventory'}
    manifest = dict(inventory=str(inventory), dataset=binding)
    predicted = [dict(r, logit=float(r['z'])) for r in obs]
    write_json(source/'manifest.json', manifest)
    write_json(source/'predictions.json', dict(rows=predicted))
    write_json(source/'feature_matrix.json', dict(manifest_sha256=digest(manifest), rows=rows))
    write_json(source/'completed.json', dict(manifest_sha256=digest(manifest), artifacts={
        name: sha256_file(source/name) for name in ('predictions.json', 'feature_matrix.json')}))
    for group in groups:
        write_json(inventory/'targets'/(group['target_id']+'.json'), dict(baseline_ids=['0']))
    monkeypatch.setattr(ranker_report, 'load_dataset', lambda path: (groups, obs, {}, binding))
    return source


def test_completed_matrix_runs_offline_reuses_results_and_rejects_changed_source(tmp_path, monkeypatch):
    from src.judge.corrected import CodexJudgeClient
    def forbidden(*args, **kwargs):
        raise AssertionError('offline comparison must not extract or relabel')
    monkeypatch.setattr(CodexJudgeClient, 'run', forbidden)
    source = frozen_source(tmp_path, monkeypatch)
    output = tmp_path/'comparison'
    value = comparison.run(source, output)
    assert value['new_model_requests'] == 0
    assert set(value['methods']) == {'dnn_saved', 'lr_existing', 'lr_expanded'}
    assert value['methods']['lr_expanded']['validation']['selection']['compared_contexts'] == 5
    assert value['methods']['lr_existing']['validation']['metrics']['contexts_with_positive'] == 5
    assert value['added_fields'] and 'Positive candidates per context' in (output/'REPORT.md').read_text()
    assert 'Top1 / Top3 / Top5' in (output/'REPORT.md').read_text()
    for result in value['methods'].values():
        assert set(result['validation']['metrics']['topk']) == {'1', '3', '5'}
    monkeypatch.setattr(logistic, 'train', forbidden)
    assert comparison.run(source, output) == value
    matrix = read_json(source/'feature_matrix.json')
    matrix['rows'][0]['target.needs_empathy'] = 'yes'
    write_json(source/'feature_matrix.json', matrix)
    with pytest.raises(ConfigError, match='Completed training artifacts changed'):
        comparison.run(source, tmp_path/'changed')
