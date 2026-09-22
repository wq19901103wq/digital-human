"""Checked, request-free replay of saved development Judge draws.

Only observations are reused. Both frozen scorers run again under a newly
registered experiment; historical metrics and adoption flags are never imported.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import cache, tracing
from ..config import sha256_file, valid_name
from ..judge import corrected_v1 as rt
from ..judge.corrected import CorrectedJudge, blind_case
from . import versions
from .training_evidence import Evidence, require


def _client_matches(identity, config):
    return all(identity.get(k) == config.get(k) for k in
               ('provider', 'model', 'reasoning_effort', 'codex_cli_version', 'model_revision'))


def initial_request(op, cid, prompt, response, config, evidence, visited=None):
    visited = set() if visited is None else visited
    verified = False
    for event in op['events']:
        if event['kind'] == 'codex' and event['request']['prompt'] == prompt:
            require(event['request'] == dict(prompt=prompt, schema=None, config=config),
                    'Saved initial request configuration differs')
            if event['status'] == 'ok':
                try:
                    verified |= rt._parse_api_judge_response(event['response']['text']) == response
                except ValueError:
                    pass
    for entry in evidence.entries(op, cid):
        identity = entry['identity']
        if entry['layer'] not in ('llm_request', 'judge_initial') or identity.get('prompt') != prompt:
            continue
        require(_client_matches(identity['client'], config), 'Saved initial client differs')
        if entry['layer'] == 'llm_request':
            require(identity['schema'] is None and rt._parse_api_judge_response(entry['value']) == response,
                    'Saved initial response differs')
            verified = True
        else:
            require(entry['value'] == response, 'Saved initial decision differs')
            origin = entry['origin']
            key = (origin['experiment_path'], origin['trace_ref'], origin['operation_index'])
            if key in visited:
                continue
            visited.add(key)
            trace = evidence.document(Path(origin['experiment_path']) / 'traces' / (origin['trace_ref'] + '.json'))
            producer = trace['operations'][origin['operation_index']]
            require(str(trace['case_id']) == str(cid) and producer['round'] == op['round'],
                    'Initial cache producer belongs to another case or round')
            initial_request(producer, cid, prompt, response, config, evidence, visited)
            verified = True
    require(verified, 'No actual initial request evidence')


def verify_draw(case, info, reference, round_index, evidence):
    require(type(round_index) is int and round_index in (0, 1, 2)
            and reference['round'] == round_index, 'Saved draw round differs')
    path = Path(reference['path'])
    trace = evidence.document(path)
    require(sha256_file(path) == reference['sha256'] and trace['case'] == case and trace['status'] in ('ok', 'failed'),
            'Saved trace content, case or status differs')
    index = reference['operation']
    require(type(index) is int and 0 <= index < len(trace['operations']), 'Invalid saved operation')
    op = trace['operations'][index]
    entries, results, visited = [], [], set()
    while True:
        require(op['kind'] == 'judge' and op['status'] == 'ok' and op['round'] == round_index
                and op['input']['candidate_replies'] == case['ai_replies'], 'Saved operation differs')
        saved = [e for e in evidence.entries(op, case['case_id']) if e['layer'] == 'judge']
        require(len(saved) == 1, 'Missing whole-Judge provenance')
        entry = saved[0]
        identity = entry['identity']
        require(identity['config'] == info['config'] and identity['candidate_replies'] == case['ai_replies']
                and identity['case'] == {k: case.get(k) for k in
                    ('context', 'human_reply', 'chat_type', 'source_chat_id', 'chat_name')},
                'Saved Judge configuration or case differs')
        entries.append(entry)
        results.append(op['result']['identified_ai'])
        if any(e['kind'] == 'blind_mapping' for e in op['events']):
            break
        origin = entry['origin']
        key = (origin['experiment_path'], origin['trace_ref'], origin['operation_index'])
        require(key not in visited and origin['round'] == round_index, 'Circular or cross-round Judge cache')
        visited.add(key)
        producer = evidence.document(Path(origin['experiment_path']) / 'traces' / (origin['trace_ref'] + '.json'))
        # A later candidate/round may have failed after this operation committed.
        # Reuse the successful operation, never the enclosing case's failed result.
        require(producer['case'] == case and producer['status'] in ('ok', 'failed'),
                'Judge producer case or status differs')
        op = producer['operations'][origin['operation_index']]
    mappings = [e['data'] for e in op['events'] if e['kind'] == 'blind_mapping']
    features = [e['data']['parsed'] for e in op['events'] if e['kind'] == 'features' and e['data'].get('valid')]
    initials = [e['data'] for e in op['events'] if e['kind'] == 'base_verdict']
    require(len(mappings) == len(initials) == 1 and features, 'Saved draw observations missing')
    mapping, features, initial = mappings[0], features[-1], initials[0]
    human = mapping['human_option']
    require(human in ('A', 'B') and mapping['candidate_option'] == ('B' if human == 'A' else 'A'),
            'Saved option mapping differs')
    a, b = ((case['human_reply'], case['ai_replies']) if human == 'A'
            else (case['ai_replies'], case['human_reply']))
    blind, metadata = blind_case(case, a, b)
    require(mapping['blind_case'] == blind and cache.digest(mapping['context_metadata']) == cache.digest(metadata),
            'Saved context or options differ')
    prompt = rt.feature_extractor_prompt(blind)
    feature_op = {**op, 'events': [e for e in op['events'] if e['kind'] != 'codex'
                                  or e['request']['prompt'] == prompt]}
    cfg = {**info['config']['llm'], **info['config'].get('feature_llm', {})}
    evidence.feature_request(feature_op, case['case_id'], blind, features, cfg)
    for entry in evidence.entries(feature_op, case['case_id']):
        if entry['layer'] == 'llm_request' and entry['identity'].get('prompt') == prompt:
            require(_client_matches(entry['identity']['client'], cfg), 'Saved feature client differs')
    initial_prompt = (info['dir'] / 'prompt.md').read_text().replace('{{case}}', json.dumps(blind, ensure_ascii=False))
    initial_request(op, case['case_id'], initial_prompt, initial, info['config']['llm'], evidence)
    model = rt.load_formal_judge(info['dir'] / 'correction.json')
    probability = rt.score_formal_judge_pair(model, features, a, b, metadata)
    small = 'A' if probability >= .5 else 'B'
    corrected = small != initial['human_option'] and max(probability, 1-probability) >= info['config']['correction_threshold']
    final = small if corrected else initial['human_option']
    hit = final == human
    for entry in entries:
        verdict = entry['value']['last_verdict']
        require(abs(probability - verdict['small_model_probability_a']) <= 1e-12
                and verdict['human_option'] == final and verdict['correction_applied'] == corrected
                and entry['value']['identified_ai'] is hit, 'Saved hybrid verdict cannot be reproduced')
    require(all(value is hit for value in results), 'Saved operation result differs')
    return dict(mapping=mapping, features=features, initial=initial,
                provenance=dict(source='verified_saved_draw', reference=reference))


def source_info(value):
    """Observation provenance is independent of the current comparison baseline."""
    info = versions.judge_dir(valid_name(value['source_judge']))
    config = info['config']
    require(config == value['source_config'], 'Saved source configuration differs')
    require(config.get('decision_policy', 'hybrid') == 'hybrid', 'Replay source must be hybrid')
    require(sha256_file(Path(rt.__file__)) == config['runtime_sha256'], 'Saved source runtime differs')
    for name, expected in {**config['assets'], 'prompt.md': config['prompt_sha256']}.items():
        require(Path(name).name == name and sha256_file(info['dir'] / name) == expected,
                'Saved source asset differs: ' + name)
    return info


def manifest(path, pack, baseline, candidate):
    """Bind the full same-condition observation index before writing a spec."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    require(value.get('schema') == 1 and value.get('dataset') == 'development'
            and value['data_ref'] == pack['data_ref'] and value['generator_ref'] == pack['c0_gen_version']
            and value['cases_sha256'] == cache.digest(pack['rows']),
            'Saved draws do not match the development data and generator')
    source = source_info(value)
    configs = [{**info['config']['llm'], **info['config'].get('feature_llm', {})}
               for info in (source, baseline, candidate)]
    require(configs[0] == configs[1] == configs[2], 'Replay feature configuration differs')
    require(baseline['config'].get('decision_policy', 'hybrid') in ('hybrid', 'lr_only', 'gbdt_only', 'score_fusion', 'embedding_only'),
            'Unsupported replay baseline')
    if baseline['config'].get('decision_policy', 'hybrid') == 'hybrid':
        require(baseline['config'] == source['config'], 'Replay hybrid baseline differs from source')
    require(candidate['config'].get('decision_policy') in ('lr_only', 'gbdt_only', 'score_fusion', 'embedding_only'), 'Unsupported replay candidate')
    ids = {str(row['case_id']) for row in pack['rows']}
    require(len(ids) == len(pack['rows']) and set(value['draws']) == ids, 'Saved draw case membership differs')
    for draws in value['draws'].values():
        require('0' in draws and set(draws) <= {'0', '1', '2'}, 'Initial draw missing or unknown round')
    frozen = dict(path=str(path), sha256=sha256_file(path))
    for name, expected in value['inputs'].items():
        require(sha256_file(Path(name)) == expected, 'Saved observation input changed: ' + name)
    require(value['inputs'] and value.get('cache_entries'), 'Saved draw evidence hashes missing')
    return frozen, value


class BoundEvidence(Evidence):
    def __init__(self, hashes, cache_entries):
        super().__init__()
        self.expected = hashes
        self.cache_entries = cache_entries

    def document(self, path):
        value = super().document(path)
        key = str(path.resolve())
        require(self.hashes[key] == self.expected.get(key), 'Unbound or changed saved request evidence')
        return value


    def cache_entry(self, event, cid, round_index):
        entry = super().cache_entry(event, cid, round_index)
        if entry is not None:
            require(cache_entry_digest(entry) == self.cache_entries.get(entry['key']),
                    'Unbound or changed saved cache evidence')
        return entry


def cache_entry_digest(entry):
    return cache.digest({k: entry[k] for k in ('key', 'layer', 'identity', 'value', 'origin')})


class SavedDrawReplay:
    def __init__(self, frozen, pack, baseline, candidate, *, supplement_missing=False):
        require(sha256_file(Path(frozen['path'])) == frozen['sha256'], 'Replay manifest changed')
        actual, self.manifest = manifest(frozen['path'], pack, baseline, candidate)
        require(actual == frozen, 'Replay manifest identity differs')
        self.rows = {str(row['case_id']): row for row in pack['rows']}
        from .learning_guard import MaterialSeal
        self.info, self.memo = source_info(self.manifest), {}
        self.source_seal = MaterialSeal(self.info['dir'])
        self.supplement = None
        if supplement_missing:
            require(all(info['config'].get('decision_policy') in
                        ('lr_only', 'gbdt_only', 'score_fusion', 'embedding_only')
                        for info in (baseline, candidate)), 'Supplement requires two feature-only scorers')
            from ..judge.lr_retrain import FeatureReplay
            self.supplement = FeatureReplay(None, None, self.info['config'], pack['rows'])

    def get(self, case, round_index, client, source_check=None):
        self.source_seal.check()
        if source_check:
            source_check()
        require(type(round_index) is int and round_index in (0, 1, 2), 'Invalid independent round')
        cid = str(case['case_id'])
        require(case == self.rows.get(cid), 'Replay sample content changed')
        key = cid, round_index
        if key not in self.memo:
            index = self.manifest['draws'][cid].get(str(round_index))
            if index is None and self.supplement is not None and round_index in (1, 2):
                draw = self.supplement.get(case, round_index, client, source_check)
                self.source_seal.check()
                return draw
            require(index is not None, 'Required independent saved round missing; live fallback prohibited')
            evidence = BoundEvidence(self.manifest['inputs'], self.manifest['cache_entries'])
            try:
                saved = evidence.document(Path(index['path']))
                study = evidence.document(Path(index['study']))
                require(study['data_ref'] == self.manifest['data_ref']
                        and study['generator_ref'] == self.manifest['generator_ref'], 'Draw study versions differ')
                require(saved['identity'] == cache.digest(dict(case=case, config=self.info['config'],
                    round=round_index, study=sha256_file(Path(index['study'])))), 'Saved draw identity differs')
                self.memo[key] = verify_draw(case, self.info, saved['reference'], round_index, evidence)
            finally:
                evidence.store.db.close()
        if source_check:
            source_check()
        self.source_seal.check()
        return self.memo[key]


class HybridReplayJudge(CorrectedJudge):
    def __init__(self, *args, replay, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay = replay

    def is_ai(self, case, candidate_replies):
        self._stage('replaying')
        scope = cache._scope.get()
        require(scope is not None and candidate_replies == case['ai_replies'], 'Replay needs the exact case and round scope')
        draw = self.replay.get(case, scope['context']['round'], self.feature_client, self._check_sources)
        mapping, features, initial = draw['mapping'], draw['features'], draw['initial']
        blind, metadata = mapping['blind_case'], mapping['context_metadata']
        probability = rt.score_formal_judge_pair(self.model, features, blind['option_A'], blind['option_B'], metadata)
        small = 'A' if probability >= .5 else 'B'
        corrected = small != initial['human_option'] and max(probability, 1-probability) >= self.config['correction_threshold']
        final = small if corrected else initial['human_option']
        self.last_verdict = dict(base=initial, small_model_probability_a=probability,
            correction_applied=corrected, human_option=final, candidate_option=mapping['candidate_option'])
        tracing.note('blind_mapping', mapping)
        tracing.note('features', dict(valid=True, parsed=features))
        tracing.note('base_verdict', initial)
        tracing.note('feature_reuse', draw['provenance'])
        tracing.note('correction', {**self.last_verdict, 'identified_ai': final == mapping['human_option']})
        self._check_sources()
        return final == mapping['human_option']


def scorers(frozen, pack, baseline, candidate, *, supplement_missing=False):
    from ..judge.gbdt import GBDTJudge
    from ..judge.fusion import FusionJudge
    from ..judge.lr_retrain import LRJudge
    from ..judge.embedding_runtime import EmbeddingJudge
    replay = SavedDrawReplay(frozen, pack, baseline, candidate, supplement_missing=supplement_missing)
    classes = dict(hybrid=HybridReplayJudge, lr_only=LRJudge, gbdt_only=GBDTJudge,
                   score_fusion=FusionJudge, embedding_only=EmbeddingJudge)
    return tuple(classes[info['config'].get('decision_policy', 'hybrid')](
        info['config'], info['dir'], replay=replay) for info in (baseline, candidate))
