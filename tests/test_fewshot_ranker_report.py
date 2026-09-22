import pytest

from src.generator.ranker_report import selection_comparison, render
from src.generator.history import HistoryError


def test_actual_baseline_order_not_union_rank_and_missing_choice_is_not_substituted():
    observations = [dict(target_id='a', example_id='union_first', recall_rank=1, z=1),
                    dict(target_id='a', example_id='existing_first', recall_rank=2, z=0),
                    dict(target_id='b', example_id='available', recall_rank=1, z=1),
                    dict(target_id='c', example_id='available', recall_rank=1, z=0)]
    baseline = dict(a=['existing_first', 'union_first'], b=['missing', 'available'], c=[])
    summary, rows = selection_comparison(observations, [2, 1, 2, 1], baseline)
    assert summary['total_contexts'] == 3 and summary['compared_contexts'] == 1
    assert summary['excluded'] == dict(baseline_first_example_not_labeled=1, no_frozen_baseline_example=1)
    assert summary['model_top1_positive_rate'] == 1
    assert summary['baseline_top1_positive_rate'] == 0
    assert summary['net_expected_positive_selections'] == summary['expected_wins'] == 1
    assert rows[0]['baseline_example_id'] == 'existing_first'


def test_same_context_tie_expectations_and_zero_positive_contexts():
    observations = [dict(target_id='a', example_id=str(i), z=z) for i, z in enumerate([1, 0])] + [
        dict(target_id='b', example_id=str(i), z=z) for i, z in enumerate([0, 0])]
    first, details = selection_comparison(observations, [0]*4, dict(a=['0'], b=['0']))
    assert first['compared_contexts'] == 2
    assert first['model_top1_positive_rate'] == .25
    assert first['baseline_top1_positive_rate'] == .5
    assert first['expected_losses'] == .5 and first['expected_ties'] == 1.5
    assert first['positive_rate_difference'] == -.25
    assert first['net_expected_positive_selections'] == -.5
    assert first['random_top1_positive_rate'] == .25
    assert first['observed_oracle_positive_rate'] == .5
    second, _ = selection_comparison(observations, [0]*4, dict(a=['0'], b=['0']))
    assert first == second
    metrics = dict(observations=4, contexts=2, roc_auc=.5, pr_auc_average_precision=.25, group_auc_macro=.5)
    text = render(dict(splits={'validation': dict(model_metrics=metrics, selection=first, contexts=details)}))
    assert '25.000%' in text and '50.000%' in text and 'union recall rank' in text


def test_rejects_nonfinite_predictions():
    with pytest.raises(HistoryError, match='Invalid prediction'):
        selection_comparison([dict(target_id='a', example_id='e', z=1)], [float('nan')], {'a': ['e']})
