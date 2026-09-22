"""Frozen LR dataset training evidence tools. Extracted from scripts/legacy/verify_lr_dataset_training.py."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

from .evidence import SavedCache, digest, require
from .. import corrected_v1 as rt


class Evidence:
    def __init__(self):
        self.store = SavedCache()
        self.counts = Counter()
        self.hashes = {}

    def document(self, path):
        blob = path.read_bytes()
        self.hashes[str(path.resolve())] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    def cache_entry(self, event, cid, round_index):
        entry = self.store.get(event['data']['key'])
        # A failed request need not have committed a cache entry.
        if entry is None and event['kind'] == 'cache_lookup':
            return None
        require(entry is not None, f'{cid}: missing successful cache entry')
        origin = entry['origin']
        require((str(origin['case_id']), origin['round']) == (str(cid), round_index),
                f'{cid}: cache reused across cases or independent rounds')
        require(entry['layer'] == event['data']['layer'], f'{cid}: cache layer mismatch')
        partition_key = ('development' if origin['dataset'] == 'development'
                         else f"{origin['dataset']}:{origin['experiment_id']}")
        expected = digest({'schema': 1, 'layer': entry['layer'], 'identity': entry['identity'],
                           'partition': partition_key, 'case_id': origin['case_id'], 'round': round_index,
                           'sample_epoch': origin['sample_epoch'], 'epoch': origin['epoch']})
        require(expected == entry['key'], f'{cid}: cache content identity mismatch')
        self.counts['cache_entries_checked'] += 1
        return entry

    def entries(self, op, cid):
        return [entry for e in op.get('events', []) if e['kind'] in ('cache_hit', 'cache_lookup')
                if (entry := self.cache_entry(e, cid, op['round'])) is not None]

    def feature_request(self, op, cid, blind, features, config):
        prompt, schema = rt.feature_extractor_prompt(blind), rt.feature_response_json_schema()
        verified = 0
        for event in op.get('events', []):
            if event['kind'] != 'codex':
                continue
            request = event['request']
            require(request == {'prompt': prompt, 'schema': schema, 'config': config},
                    f'{cid}: actual feature prompt/config/schema differs')
            if event['status'] == 'ok':
                try:
                    parsed = rt.parse_feature_response(event['response']['text'])
                except ValueError:
                    continue
                if parsed == features:
                    verified += 1
        for entry in self.entries(op, cid):
            identity = entry['identity']
            if entry['layer'] != 'llm_request' or identity.get('prompt') != prompt:
                continue
            require(identity['schema'] == schema, f'{cid}: cached feature schema differs')
            require(all(identity['client'][k] == config[k] for k in
                        ('provider', 'model', 'reasoning_effort', 'codex_cli_version')),
                    f'{cid}: cached feature configuration differs')
            require(rt.parse_feature_response(entry['value']) == features, f'{cid}: cached feature value differs')
            verified += 1
        require(verified > 0, f'{cid}: no actual or cached feature request evidence')
        self.counts['feature_requests_verified'] += 1


def vector(model, features, blind, metadata):
    """Assemble the frozen feature definition; do not use the fitting driver."""
    a, b, feature_names = rt.vectorize_boolean_options(features)
    options = []
    for name, base in (('option_A', a), ('option_B', b)):
        observed, observed_names = rt.vectorize_observable_option_booleans(blind[name], metadata)
        group, group_names = rt.vectorize_group_pattern_option_boolean(blind[name], metadata)
        context, context_names = rt.vectorize_context_booleans(metadata, model.context_schema)
        expanded, names = rt.expand_boolean_option_with_refined_context(
            np.concatenate((base, observed, group)), [*feature_names, *observed_names, *group_names],
            context, context_names)
        require(tuple(names) == model.feature_names, 'frozen feature names/order changed')
        options.append(expanded)
    result = options[0] - options[1]
    require(result.shape == (524,) and np.isfinite(result).all(), 'invalid 524-dimensional vector')
    return result
