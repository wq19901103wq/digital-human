"""Versioned context-only ownership correction on the frozen GBDT scorer."""
from __future__ import annotations

from .. import tracing
from ..config import ConfigError
from . import background_features
from .corrected import CodexJudgeClient
from .gbdt import GBDTJudge


POLICY = 'context_owner_shift_v1'


def validate_policy(value):
    if value != POLICY:
        raise ConfigError('未知的发言归属纠正规则')


def decide(probability_a, parsed, payload):
    """Only unambiguous theft of another sender's prior statement can override.

    This is a hypothesis to evaluate, not proof of an error. Quotation, new
    commitments, mixed/unknown ownership and dual flags preserve the base score.
    Neither case IDs nor speaker names nor answer origins participate.
    """
    roles = {row['ref']: row['role'] for row in payload['context']}
    flags = {}
    for option in ('option_A', 'option_B'):
        flags[option] = [item for item in parsed[option]['attribution_checks']
            if item['claim_mode'] == 'self_continuation'
            and item['context_owner'] == 'other' and item['alignment'] == 'owner_shift'
            and item['context_refs']
            and all(roles.get(ref) == 'other' for ref in item['context_refs'])]
    a, b = bool(flags['option_A']), bool(flags['option_B'])
    result = (0. if a else 1.) if a != b else probability_a
    return result, {'policy': POLICY, 'base_probability_a': probability_a,
                    'probability_a': result, 'triggered': a != b, 'flags': flags}


class OwnershipJudge(GBDTJudge):
    def __init__(self, config, version_dir, *args, **kwargs):
        validate_policy(config.get('ownership_correction'))
        super().__init__(config, version_dir, *args, **kwargs)
        self.ownership_client = CodexJudgeClient({**config['llm'],
            **config.get('feature_llm', {}), 'reasoning_effort': 'medium'})

    def probability_a(self, features, blind, metadata):
        base = super().probability_a(features, blind, metadata)
        payload = background_features.build_input(blind, {'identities': [], 'facts': []})
        self._check_sources()
        # Extraction failures remain failed observations for retry, never silent
        # fallbacks that could change the evaluated population or hide outages.
        parsed = background_features.extract(self.ownership_client, payload)
        self._check_sources()
        result, evidence = decide(base, parsed, payload)
        tracing.note('ownership_features', {'input': payload, 'parsed': parsed,
            'client': self.ownership_client.cache_identity(), 'decision': evidence})
        return result
