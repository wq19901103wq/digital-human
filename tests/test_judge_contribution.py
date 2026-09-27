"""Semantic correction boundaries, blinded inputs and cached-response validation."""
from copy import deepcopy
import json

import pytest

from src.config import ConfigError
from src.judge import contribution as c

PAYLOAD = {'context': [{'ref': 'context:0:0', 'role': 'other'}]}


def evidence():
    option = dict(contribution_types=['generic_agreement'], semantic_increment='none',
        reply_target_refs=['context:0:0'], need_checks=[dict(need_id='n1', satisfaction='unmet',
            confidence='high', reason='does not answer')], empty_response='generic_agreement_only',
        legitimate_alternative=False, diagnostics={key: dict(state='unknown', confidence='low',
            basis='insufficient', plausible_alternative=False, context_refs=[],
            reason='unknown') for key in c.DIAGNOSTICS}, reason='unmet need')
    good = deepcopy(option)
    good.update(contribution_types=['answer'], semantic_increment='task', empty_response='no')
    good['need_checks'][0]['satisfaction'] = 'met'
    return dict(context_events=[], needs=[dict(id='n1', kind='information', target='self', explicit=True,
        status='open', confidence='high', context_refs=['context:0:0'], description='question')],
        option_A=option, option_B=good)


def test_symmetric_correction_and_trace():
    parsed = evidence()
    c.parse_response(json.dumps(parsed), PAYLOAD)
    score, audit = c.decide(.8, parsed, PAYLOAD)
    assert score == 0 and audit['triggered'] and audit['base_probability_a'] == .8
    parsed['option_A'], parsed['option_B'] = parsed['option_B'], parsed['option_A']
    assert c.decide(.2, parsed, PAYLOAD)[0] == 1


@pytest.mark.parametrize('change', [dict(target='other'), dict(target='group'),
    dict(explicit=False), dict(status='resolved'), dict(status='changed'), dict(confidence='low'),
    dict(kind='empathy'), dict(kind='discussion'), dict(kind='confirmation'),
    dict(context_refs=[]), dict(context_refs=['missing'])])
def test_ambiguous_or_non_self_or_closed_needs_do_not_override(change):
    parsed = evidence()
    parsed['needs'][0].update(change)
    assert c.decide(.8, parsed, PAYLOAD)[0] == .8


@pytest.mark.parametrize('kind', ['answer', 'state', 'refusal', 'uncertainty', 'action',
    'clarification', 'confirmation', 'empathy', 'opinion', 'new_topic', 'unknown'])
def test_valid_contributions_protect_even_with_conflicting_empty_flag(kind):
    parsed = evidence()
    parsed['option_A']['contribution_types'] = [kind]
    assert c.decide(.8, parsed, PAYLOAD)[0] == .8


@pytest.mark.parametrize('change', [dict(legitimate_alternative=True),
    dict(semantic_increment='social'), dict(semantic_increment='unknown'), dict(empty_response='no')])
def test_earlier_topic_or_social_purpose_protected(change):
    parsed = evidence()
    parsed['option_A'].update(change)
    assert c.decide(.8, parsed, PAYLOAD)[0] == .8


@pytest.mark.parametrize('satisfaction', ['partial', 'legitimate_nonanswer', 'unknown', 'met'])
def test_partial_or_uncertain_response_protected(satisfaction):
    parsed = evidence()
    parsed['option_A']['need_checks'][0]['satisfaction'] = satisfaction
    assert c.decide(.8, parsed, PAYLOAD)[0] == .8


def test_neither_satisfies_need_or_different_need_preserves_base():
    parsed = evidence()
    parsed['option_B'] = deepcopy(parsed['option_A'])
    assert c.decide(.8, parsed, PAYLOAD)[0] == .8
    parsed = evidence()
    parsed['option_B']['need_checks'][0]['need_id'] = 'n2'
    assert c.decide(.8, parsed, PAYLOAD)[0] == .8


@pytest.mark.parametrize('damage', ['refs', 'enum', 'missing', 'extra', 'need', 'duplicate', 'problem'])
def test_invalid_cached_evidence_rejected(damage):
    parsed = evidence()
    if damage == 'refs':
        parsed['needs'][0]['context_refs'] = ['option_B']
    elif damage == 'enum':
        parsed['option_A']['semantic_increment'] = 'lots'
    elif damage == 'missing':
        del parsed['needs']
    elif damage == 'extra':
        parsed['winner'] = 'B'
    elif damage == 'need':
        parsed['option_A']['need_checks'][0]['need_id'] = 'invented'
    elif damage == 'duplicate':
        parsed['needs'].append(deepcopy(parsed['needs'][0]))
    else:
        parsed['option_A']['diagnostics']['pragmatic_fit']['state'] = 'problem'
    with pytest.raises(ValueError):
        c.parse_response(json.dumps(parsed), PAYLOAD)


def joint_evidence(feature):
    parsed = evidence()
    parsed['needs'] = []
    for option in ('option_A', 'option_B'):
        parsed[option]['need_checks'] = []
    parsed['option_A']['diagnostics'][feature].update(state='problem', confidence='high',
        basis='direct_conflict', context_refs=['context:0:0'])
    return parsed


@pytest.mark.parametrize('feature', c.DIAGNOSTICS)
def test_grounded_joint_dimensions_participate_symmetrically(feature):
    parsed = joint_evidence(feature)
    c.parse_response(json.dumps(parsed), PAYLOAD)
    score, audit = c.decide(.8, parsed, PAYLOAD)
    assert score == 0 and audit['flags']['option_A'][0]['feature'] == feature
    parsed['option_A'], parsed['option_B'] = parsed['option_B'], parsed['option_A']
    assert c.decide(.2, parsed, PAYLOAD)[0] == 1


@pytest.mark.parametrize('change', [dict(confidence='medium'), dict(confidence='low'),
    dict(state='unknown'), dict(basis='insufficient'), dict(basis='compatible'),
    dict(plausible_alternative=True), dict(context_refs=[]), dict(context_refs=['missing'])])
def test_ambiguous_or_ungrounded_joint_dimensions_do_not_override(change):
    parsed = joint_evidence('place_event_binding')
    parsed['option_A']['diagnostics']['place_event_binding'].update(change)
    assert c.decide(.8, parsed, PAYLOAD)[0] == .8


def test_inference_scope_can_flag_invalid_inference_but_other_dimensions_cannot():
    for feature in ('inference_scope', 'pragmatic_fit'):
        parsed = joint_evidence(feature)
        parsed['option_A']['diagnostics'][feature]['basis'] = 'invalid_inference'
        assert c.decide(.8, parsed, PAYLOAD)[0] == (0 if feature == 'inference_scope' else .8)


def test_problems_on_both_options_preserve_base_even_across_feature_families():
    parsed = evidence()  # A has an unmet need; B has a grounded event conflict.
    parsed['option_B']['diagnostics']['event_participant_direction'].update(
        state='problem', confidence='high', basis='direct_conflict', context_refs=['context:0:0'])
    score, audit = c.decide(.8, parsed, PAYLOAD)
    assert score == .8 and not audit['triggered']
    assert audit['flags']['option_A'] and audit['flags']['option_B']


@pytest.mark.parametrize('damage', [None, 'empty_id', 'duplicate', 'missing_refs', 'foreign_refs'])
def test_context_events_require_unique_ids_and_grounded_refs(damage):
    parsed = evidence()
    event = dict(id='e1', action='pick up', actor='P2', recipient='P3', place='unknown',
        place_role='pickup', condition='now', confidence='high', context_refs=['context:0:0'])
    parsed['context_events'] = [event]
    if damage == 'empty_id':
        event['id'] = ''
    elif damage == 'duplicate':
        parsed['context_events'].append(deepcopy(event))
    elif damage == 'missing_refs':
        event['context_refs'] = []
    elif damage == 'foreign_refs':
        event['context_refs'] = ['option_B']
    if damage:
        with pytest.raises(ValueError):
            c.parse_response(json.dumps(parsed), PAYLOAD)
    else:
        assert c.parse_response(json.dumps(parsed), PAYLOAD)['context_events'] == [event]


def test_allowlisted_inputs_and_policy():
    payload = c.build_input(dict(relationship='group', option_A=['A'], option_B=['B'],
        context_original=['<context><message role="self">me</message></context>'],
        human_option='B', case_id='secret', human_reply='secret', labels=[0, 1]))
    assert set(payload) == {'schema', 'relationship', 'context', 'option_A', 'option_B'}
    assert 'secret' not in c.prompt_for(payload)
    c.validate_policy(c.POLICY)
    with pytest.raises(ConfigError):
        c.validate_policy('unknown')


def test_extraction_failure_does_not_fallback():
    class Broken:
        def run(self, *args):
            return '{}'
    with pytest.raises(ValueError):
        c.extract(Broken(), PAYLOAD)
