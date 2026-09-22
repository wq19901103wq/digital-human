"""Ordinal labels, loss ordering, isolation and cached comparison regressions."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.config import ConfigError, sha256_file
from src.generator.fewshot_ranker import (
    boosting, graded_comparison as comparison, graded_labels, graded_metrics,
    logistic, neural_models, stability, training,
)
from src.generator.history_sources import digest
from src.iteration.storage import read_json, write_json
from tests.test_fewshot_ranker_architectures import mixer_recipe
from tests.test_fewshot_ranker_lr import feature_row, frozen_source
from tests.test_fewshot_ranker_stability import fixture
from tests.test_fewshot_ranker_training import synthetic


def test_strict_ordinal_pairs_skip_ties_and_balance_contexts():
    obs = [dict(z=z) for z in (3, 2, 2, 1, 0, 1, 0)]
    groups = [dict(indices=list(range(5))), dict(indices=[5, 6])]
    pairs, weights, mixed = training.pairs(groups, obs)
    assert mixed == 2 and len(pairs) == 10
    assert all(obs[p]['z'] > obs[n]['z'] for p, n in pairs)
    assert (1, 2) not in map(tuple, pairs)
    assert weights[:9].sum() == pytest.approx(1) and weights[-1] == 1
    binary = [dict(z=int(r['z'] > 0)) for r in obs]
    bp, bw, _ = training.pairs(groups, binary)
    assert bp.tolist() == [[0, 4], [1, 4], [2, 4], [3, 4], [5, 6]]
    assert bw.tolist() == [.25, .25, .25, .25, 1]


def completed_repeats(tmp_path, monkeypatch):
    args, calls, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path/'repeats'
    stability.StabilityBatch(*args, output, 'regenerate', 2).run(workers=1)
    groups, obs, _, binding = stability.load_dataset(args[0])
    for group in groups:
        for index, entry in zip(group['indices'], group['candidates']):
            obs[index].update(target_id=group['target_id'], split=group['split'],
                              example_id=entry['example']['id'])
    return args[0], output, binding, groups, obs, calls


def test_completed_repeat_grades_preserve_zero_trial_count_without_calls(tmp_path, monkeypatch):
    inventory, output, binding, groups, obs, calls = completed_repeats(tmp_path, monkeypatch)
    before = deepcopy(calls)
    labels, report, _ = graded_labels.load(inventory, output, binding, groups, obs)
    assert [r['grade'] for r in labels] == [3, 0, 1, 0]
    assert [r['trial_count'] for r in labels] == [3, 1, 3, 1]
    assert report['grade_counts']['all'] == {'3': 1, '0': 2, '1': 1}
    assert labels[1]['repeat_labels'] == [] and calls == before
    assert all('grade' not in row for row in obs)
    with pytest.raises(ConfigError, match='bound to this dataset'):
        graded_labels.load(inventory, output, {'source': 'foreign'}, groups, obs)


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'foreign'])
def test_incomplete_or_wrong_repeat_evidence_is_rejected(tmp_path, monkeypatch, damage):
    inventory, output, binding, groups, obs, _ = completed_repeats(tmp_path, monkeypatch)
    path = next((output/'regenerate/draw-1/observations').glob('*.json'))
    if damage == 'missing':
        path.unlink()
    else:
        value = read_json(path)
        value['binding'] = 'foreign'
        if damage == 'foreign':
            value['payload_sha256'] = digest({k: v for k, v in value.items() if k != 'payload_sha256'})
        write_json(path, value)
    with pytest.raises(ConfigError):
        graded_labels.load(inventory, output, binding, groups, obs)


def test_graded_metrics_are_tie_aware_and_keep_zero_contexts():
    labels = [dict(target_id=t, grade=g) for t, values in [('a', [3, 2, 1, 0]), ('b', [0, 0])]
              for g in values]
    perfect = graded_metrics.evaluate(labels, [3, 2, 1, 0, 0, 0])
    assert perfect['ordered_pair_accuracy_macro'] == perfect['ndcg']['3'] == 1
    assert perfect['positive_only_pair_accuracy_macro'] == 1
    assert perfect['thresholds']['3']['topk']['1']['hit_rate'] == .5
    assert perfect['thresholds']['3']['topk']['1']['hit_rate_given_available'] == 1
    assert perfect['mean_grade_at_k']['1'] == 1.5
    tied = graded_metrics.evaluate(labels, [0]*6)
    assert tied['ordered_pair_accuracy_macro'] == tied['positive_only_pair_accuracy_macro'] == .5
    assert tied['thresholds']['3']['topk']['1']['hit_rate'] == .125
    assert tied['thresholds']['3']['topk']['1']['hit_rate_given_available'] == .25
    assert tied['mean_rank_by_grade']['3']['mean_rank'] == 2.5
    assert tied['mean_grade_at_k']['1'] == .75
    delta = graded_metrics.compare(tied, perfect)
    assert delta['1']['hit_grade_ge_3']['mean_delta'] == .375


def test_threshold_report_shows_each_auc_and_both_denominators():
    labels = [dict(target_id=t, grade=g) for t, values in [('a', [3, 2, 1, 0]), ('b', [0, 0])]
              for g in values]
    before = graded_metrics.evaluate(labels, [0]*6)
    after = graded_metrics.evaluate(labels, [3, 2, 1, 0, 0, 0])
    value = dict(labels=dict(grade_counts={}), methods=dict(model=dict(before=before, after=after)), seconds=0)
    text = comparison.render(value)
    assert text.count('ROC-AUC before → after') == text.count('PR-AUC (AP) before → after') == 3
    assert text.count('Denominator: 2 contexts.') == text.count('Denominator: 1 contexts.') == 3
    assert '3 positive rows (50.00%)' in text and '2 positive rows (33.33%)' in text
    assert '1 positive rows (16.67%)' in text
    assert '| model | 12.50% → 50.00%' in text
    assert '| model | 25.00% → 100.00%' in text
    zero = graded_metrics.evaluate([dict(target_id='z', grade=0)], [0])
    value['methods']['model'] = dict(before=zero, after=zero)
    text = comparison.render(value)
    assert text.count('Denominator: 0 contexts.') == 3
    assert '| model | N/A → N/A | N/A → N/A | N/A → N/A |' in text
    assert '| model | 0.00% → 0.00% | 0.00% → 0.00% | 0.00% → 0.00% |' in text


def control_model(groups, rows, obs, architecture):
    fit = [g for g in groups if g['split'] == 'fit']
    if architecture == 'xgb':
        recipe = dict(boosting.recipes()[0], max_rounds=2, min_child_weight=0)
        encoder = logistic.encoder(rows, [i for g in fit for i in g['indices']])
        model, _ = boosting.fit(encoder.transform(rows), obs, fit, [], recipe, 2)
        value = dict(kind='fewshot_pairwise_xgboost_v1', vocabulary=encoder.vocabulary_,
                     booster=json.loads(model.save_raw(raw_format='json').decode()),
                     feature_names=encoder.get_feature_names_out().tolist())
        saved = dict(selected_recipe=recipe, selected_rounds=2)
    else:
        recipe = mixer_recipe() if architecture == 'mixer' else dict(training.RECIPE, hidden=[8])
        model, _ = training.fit(rows, obs, fit, [], recipe, epochs=1)
        value = neural_models.document(model)
        saved = dict(selected_recipe=recipe, selected_epochs=1)
    value['feature_transform'] = dict(chat_id_pair=False)
    return value, saved


@pytest.mark.parametrize('architecture', ['dnn', 'mixer', 'xgb'])
def test_fixed_recipe_reproduces_binary_control_and_ignores_outer_grades(architecture):
    groups, raw, obs = synthetic()
    rows = comparison.feature_rows([{**feature_row(), **r} for r in raw], False)
    saved, detail = control_model(groups, rows, obs, architecture)
    binary = [dict(r, grade=r['z']) for r in obs]
    value, _, _ = comparison.fit_selected(groups, binary, rows, saved, detail)
    assert {k: v for k, v in value.items() if k != 'supervision'} == saved
    grades = [dict(r, grade=3*r['z']) for r in obs]
    value, scores, fit = comparison.fit_selected(groups, grades, rows, saved, detail)
    changed = [dict(r, grade=3-r['grade']) if r['split'] == 'validation' else r for r in grades]
    second, second_scores, second_fit = comparison.fit_selected(groups, changed, rows, saved, detail)
    assert value == second and fit == second_fit
    np.testing.assert_array_equal(scores, second_scores)
    assert all('grade' not in r for r in obs)


def test_graded_workflow_reuses_controls_and_completed_methods(tmp_path, monkeypatch):
    from src.judge.corrected import CodexJudgeClient
    from src.llm import ChatClient

    def forbidden(*args, **kwargs):
        raise AssertionError('no refitting or LLM calls for completed work')

    monkeypatch.setattr(CodexJudgeClient, 'run', forbidden)
    monkeypatch.setattr(ChatClient, 'chat', forbidden)
    source = frozen_source(tmp_path, monkeypatch)
    reference, output = tmp_path/'reference', tmp_path/'graded'
    manifest, groups, obs, _, _, _ = comparison.load_completed(source)
    rows = comparison.feature_rows(read_json(source/'feature_matrix.json')['rows'], False)
    model, detail = control_model(groups, rows, obs, 'dnn')
    binding = dict(kind='cached_fewshot_architecture_comparison', source_manifest_sha256=digest(manifest),
                   source_completed_sha256=sha256_file(source/'completed.json'), dataset=manifest['dataset'])
    write_json(reference/'manifest.json', binding)
    names = []
    for name, value in [('model', model), ('training', detail), ('predictions', read_json(source/'predictions.json'))]:
        path = f'dnn_semantic/{name}.json'
        write_json(reference/path, value)
        names.append(path)
    comparison.seal(reference, binding, names)
    monkeypatch.setattr(comparison, 'METHODS', ('dnn_semantic',))
    labels = [dict(r, grade=3*r['z']) for r in obs]
    monkeypatch.setattr(graded_labels, 'load', lambda *args: (labels, dict(grade_counts={}), {}))
    result = comparison.run(source, reference, tmp_path/'repeats', output)
    assert result['status'] == 'complete' and result['new_model_requests'] == 0
    assert 'All-three-positive Hit@K' in (output/'REPORT.md').read_text()
    monkeypatch.setattr(comparison, 'fit_selected', forbidden)
    assert comparison.run(source, reference, tmp_path/'repeats', output) == result
    hashes = {p: sha256_file(p) for p in output.rglob('*.json')}
    hashes[output/'REPORT.md'] = sha256_file(output/'REPORT.md')
    with monkeypatch.context() as reporting:
        reporting.setattr(graded_metrics, 'evaluate', forbidden)
        first_report = comparison.report(output)
        assert first_report['new_model_requests'] == first_report['new_training_runs'] == 0
        assert comparison.report(output) == first_report
        from scripts.report_fewshot_ranker import main
        main(['--training-output', str(output)])
    assert all(sha256_file(p) == sha for p, sha in hashes.items())
    assert 'PR-AUC (AP)' in (output/'THRESHOLD_REPORT.md').read_text()
    (output/'completed.json').unlink()
    with pytest.raises(ConfigError, match='must be complete'):
        comparison.report(output)
    (output/'report.json').unlink()
    resumed = comparison.run(source, reference, tmp_path/'repeats', output)
    assert resumed['methods'] == result['methods']
    write_json(reference/'dnn_semantic/training.json', dict(detail, changed=True))
    with pytest.raises(ConfigError, match='artifacts changed'):
        comparison.run(source, reference, tmp_path/'repeats', output)
