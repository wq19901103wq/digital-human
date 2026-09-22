import copy

import pytest

from scripts.legacy.finish_guarded_history_audit import checked_legacy_view


def fixture():
    cases = [{'case_id': 'a', 'familiarity': 'familiar', 'chat_type': 'group'},
             {'case_id': 'b', 'familiarity': 'familiar', 'chat_type': 'group'},
             {'case_id': 'c', 'familiarity': 'unseen_holdout', 'chat_type': 'private'}]
    records = {'a': {'status': 'ok', 'baseline_correct': True, 'candidate_correct': False},
               'b': {'status': 'failed'},
               'c': {'status': 'ok', 'baseline_correct': False, 'candidate_correct': True}}
    state = {'metrics': {'pairs': 2, 'unknown_extension': 9, 'cohorts': {
        'familiar/group': {'attempted': 2, 'pairs': 1, 'failures': 1,
                           'identified_baseline': 1, 'identified_candidate': 0},
        'unseen_holdout/private': {'attempted': 1, 'pairs': 1, 'failures': 0,
                                   'identified_baseline': 0, 'identified_candidate': 1}}}}
    return state, cases, records


def test_preserves_all_other_fields_and_original_state():
    state, cases, records = fixture()
    before = copy.deepcopy(state)
    view, groups = checked_legacy_view(state, cases, records)
    assert state == before
    assert groups == state['metrics']['cohorts']
    assert view == {'metrics': {'pairs': 2, 'unknown_extension': 9}}


@pytest.mark.parametrize('field', ['attempted', 'pairs', 'failures', 'identified_baseline', 'identified_candidate'])
def test_rejects_changed_cohort_field(field):
    state, cases, records = fixture()
    state['metrics']['cohorts']['familiar/group'][field] += 1
    with pytest.raises(ValueError, match='cohorts differ'):
        checked_legacy_view(state, cases, records)


def test_rejects_changed_group_membership():
    state, cases, records = fixture()
    cases[0]['familiarity'] = 'unseen_holdout'
    with pytest.raises(ValueError, match='cohorts differ'):
        checked_legacy_view(state, cases, records)


def test_rejects_missing_and_duplicate_cases():
    state, cases, records = fixture()
    for changed in (cases[:-1], cases + [cases[0]]):
        with pytest.raises(ValueError, match='missing, duplicate, or extra'):
            checked_legacy_view(state, changed, records)


def test_rejects_invalid_vote():
    state, cases, records = fixture()
    records['a']['baseline_correct'] = 'yes'
    with pytest.raises(ValueError, match='invalid vote'):
        checked_legacy_view(state, cases, records)
