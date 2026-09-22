"""Validate new policy comparison planning and independent policy semantics."""
import copy
import pytest
from scripts.legacy import finish_pairwise_policy_comparison as job
from scripts.legacy import verify_pairwise_policy_comparison as audit
from scripts.legacy.verify_pairwise_tuning import count


def test_pure_policy_ignores_initial_below_hybrid_threshold_and_tie_is_a():
    draw = {'initial': {'human_option': 'B'}, 'mapping': {'human_option': 'A'}}
    assert audit.decision(.6, draw, 'pure')['identified_ai'] is True
    assert audit.decision(.6, draw, 'hybrid')['identified_ai'] is False
    assert audit.decision(.5, draw, 'pure')['human_option'] == 'A'
    with pytest.raises(ValueError):
        audit.decision(.6, draw, 'bad')


def test_cross_arm_union_and_records_keep_rounds_and_votes_separate():
    rows = [{'case_id': 'different'}, {'case_id': 'same'}]
    b = dict(arm='luna', recipe='lr_l2_c1', policy='pure')
    c = dict(arm='sol', recipe='lr_l2_c0.1', policy='pure')
    comparison = dict(baseline=b, candidate=c)
    initial = lambda side, cid: cid == 'same' or side['arm'] == 'sol'
    needed = job.needed_rounds(rows, [comparison, comparison], initial)
    assert needed == {(a, 'different', r) for a in ('luna', 'sol') for r in (1, 2)}
    draws, predictions = {}, {}
    for row in rows:
        cid = row['case_id']
        for side in (b, c):
            for rnd in range(3):
                draws[side['arm'], cid, rnd] = dict(reference=dict(arm=side['arm'], round=rnd))
                predictions[job.prediction_key(side, cid, rnd)] = {'identified_ai': initial(side, cid)}
    records = job.build_records(rows, comparison, predictions, draws)
    assert count(records)['net_win_confirmed'] == 1
    assert len(records[1]['draws']) == 1
    for rnd, pair in enumerate(records[0]['draws']):
        assert pair['baseline'] == dict(arm='luna', round=rnd)
        assert pair['candidate'] == dict(arm='sol', round=rnd)
    broken = copy.deepcopy(records)
    broken[0]['draws'].pop()
    with pytest.raises(ValueError):
        count(broken)
    del predictions[job.prediction_key(c, 'different', 2)]
    with pytest.raises(KeyError):
        job.build_records(rows, comparison, predictions, draws)
