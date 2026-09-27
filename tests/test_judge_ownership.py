"""General attribution correction; no diagnostic case IDs or names."""
from copy import deepcopy

import pytest

from src.config import ConfigError
from src.judge.ownership import POLICY, decide, validate_policy
from src.judge.background_features import build_input


def evidence():
    return {'option_A': {'attribution_checks': [{
        'claim_mode': 'self_continuation', 'context_owner': 'other',
        'alignment': 'owner_shift', 'context_refs': ['context:0:0']}]},
        'option_B': {'attribution_checks': []}}


PAYLOAD = {'context': [{'ref': 'context:0:0', 'role': 'other'},
                       {'ref': 'context:0:1', 'role': 'self'}]}


def test_correction_is_symmetric_and_keeps_base_evidence():
    parsed = evidence()
    score, audit = decide(.8, parsed, PAYLOAD)
    assert score == 0 and audit['triggered'] and audit['base_probability_a'] == .8
    flipped = {'option_A': parsed['option_B'], 'option_B': parsed['option_A']}
    assert decide(.2, flipped, PAYLOAD)[0] == 1


@pytest.mark.parametrize('change', [
    {'claim_mode': 'quotation'}, {'claim_mode': 'new_claim'},
    {'claim_mode': 'new_commitment'}, {'context_owner': 'mixed'},
    {'context_owner': 'unknown'}, {'alignment': 'stance_reversal'},
    {'context_refs': []}, {'context_refs': ['missing']},
    {'context_refs': ['context:0:0', 'context:0:1']},
])
def test_ambiguous_or_different_error_preserves_base(change):
    parsed = evidence()
    parsed['option_A']['attribution_checks'][0].update(change)
    assert decide(.8, parsed, PAYLOAD)[0] == .8


def test_both_flags_preserve_base():
    parsed = evidence()
    parsed['option_B'] = deepcopy(parsed['option_A'])
    assert decide(.8, parsed, PAYLOAD)[0] == .8


def test_input_excludes_answers_metadata_and_distinguishes_roles():
    payload = build_input({'relationship': 'group', 'option_A': ['A'], 'option_B': ['B'],
        'human_option': 'B', 'case_id': 'diagnostic', 'correct_answer': 'secret',
        'context_original': ['<context><message role="other">Someone</message>'
                             '<message role="self">Me</message></context>']},
        {'identities': [], 'facts': []})
    assert not {'human_option', 'case_id', 'correct_answer'} & payload.keys()
    assert [r['role'] for r in payload['context']] == ['other', 'self']
    validate_policy(POLICY)
    with pytest.raises(ConfigError):
        validate_policy('case_specific')
