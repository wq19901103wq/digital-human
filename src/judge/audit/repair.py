"""Repaired-history Judge reconstruction audit. Extracted from scripts/legacy/verify_repaired_history_judge.py."""
from __future__ import annotations

import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .evidence import digest, require
from .history import generation_cache
from .dataset import Evidence
from .matching import source_text_matches
from ...config import sha256_file
from ...iteration import datasets, versions
from ...iteration.learning_guard import require_materials
from ...judge import corrected_v1 as rt
from ...judge.corrected import blind_case

ROOT = Path(__file__).resolve().parents[3]


def lines(path):
    require(path.name != 'fixed_test.jsonl', 'fixed answers must remain unopened')
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sources(directory, spec, evidence):
    for name, expected in spec['inputs'].items():
        require(sha256_file(Path(name)) == expected, f'frozen input changed: {name}')
    data = versions.data_version_dir(spec['data_ref'])
    require(datasets.snapshot(data) == spec['purpose_snapshot'], 'purpose manifest changed')
    selected = evidence.document(directory / 'sources.json')
    train, dev = selected['train'], selected['development']
    require(len(train) == 1200 and len(dev) == 1000, 'formal sample counts changed')
    require({r['case_id']: r for r in train} == {r['case_id']: r for r in lines(data / 'judge_training.jsonl')},
            'training cases differ from the original purpose')
    require(dev == lines(data / 'judge_dev_pool.jsonl'), 'development cases/order changed')
    old = evidence.document(directory.parent / '20260914-d0011-sol-high-train1200/sources.json')
    require(train == old['cases'], 'original training order changed')
    purpose = evidence.document(data / 'purposes.json')['protocol']
    cutoff, heldout = purpose['development_start'], set(purpose['unseen_chat_ids'])
    history = defaultdict(list)
    with (data / 'messages.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            history[row['chat_id']].append(row)
    learned, tested = set(), set()
    for role, cases, ids in [('training', train, learned), ('development', dev, tested)]:
        for case in cases:
            span = case['source_span']
            start, middle, end = [span[k] for k in ('start', 'reply_start', 'end')]
            rows = history[span['chat_id']]
            require(0 <= start < middle < end <= len(rows), 'invalid source offsets')
            require(case['context_message_ids'] == [r['message_id'] for r in rows[start:middle]]
                    and case['reply_message_ids'] == [r['message_id'] for r in rows[middle:end]], 'message binding differs')
            require(len(case['human_reply']) == end-middle and all(r['is_self'] for r in rows[middle:end]),
                    'human bubble/source mismatch')
            require(all(source_text_matches(t, r['text']) for t, r in zip(case['human_reply'], rows[middle:end])),
                    'target text differs from source')
            require(len(case['context']) == middle-start and all(
                source_text_matches(c['text'], r['text']) and all(c[k] == r[k] for k in ('sender', 'is_self', 'timestamp'))
                for c, r in zip(case['context'], rows[start:middle])), 'context provenance differs')
            if role == 'training':
                require(rows[end-1]['timestamp'] < cutoff and span['chat_id'] not in heldout, 'future/heldout training')
            else:
                require(rows[start]['timestamp'] >= cutoff, 'development before learning cutoff')
            ids.update(r['message_id'] for r in rows[start:end])
            evidence.counts[role + '_source_spans'] += 1
    require(not learned & tested, 'learning/evaluation full-message overlap')
    audit = evidence.document(directory / 'source_audit.json')
    reference = evidence.document(directory / 'reference.json')
    require(not reference['facts'] and len(reference['examples']) == len(audit['reference_spans']) == 7,
            'unverified static material retained')
    for example, span in zip(reference['examples'], audit['reference_spans']):
        rows = history[span['chat_id']][span['start']:span['end']]
        texts = example['context'] + example['human_reply']
        require(len(rows) == len(texts) and all(source_text_matches(t, r['text']) for t, r in zip(texts, rows)),
                'reference source mismatch')
        require(span['message_ids'] == [r['message_id'] for r in rows] and
                rows[-1]['timestamp'] < cutoff and span['chat_id'] not in heldout and
                not {r['message_id'] for r in rows} & tested, 'reference future/heldout/answer overlap')
        evidence.counts['static_reference_spans'] += 1
    names, members = Counter(), Counter()
    for case in train:
        _, meta = blind_case(case, [], [])
        if meta['relationship'] == 'group':
            names[str(meta['group_name'])] += 1
            members.update((str(meta['source_chat_id']), str(s)) for s in meta['recent_speakers'][:3] if s != '__self__')
    template = evidence.document(directory / 'model_template.json')
    expected = {'known_group_names': sorted(n for n, c in names.items() if c >= 5),
                'known_group_members': [list(m) for m, c in sorted(members.items()) if c >= 8]}
    require(template['feature_system']['context_schema'] == expected, 'category vocabulary leaks outside training')
    require(not np.any(template['final_model']['coefficients']), 'old LR weights retained')
    require_materials(spec['data_ref'], [versions.generator_dir(spec['generator_ref'])])
    require(spec['protocol']['flip_extra_rounds'] == 2 and spec['scoring']['correction_threshold'] == .7
            and spec['scoring']['decision_policy'] == 'hybrid' and spec['adoption_allowed'] is False,
            'scoring/scope changed')
    return selected


def vector(model, features, blind, metadata):
    a, b, names = rt.vectorize_boolean_options(features)
    options = []
    for option, base in [('option_A', a), ('option_B', b)]:
        observed, observed_names = rt.vectorize_observable_option_booleans(blind[option], metadata)
        group, group_names = rt.vectorize_group_pattern_option_boolean(blind[option], metadata)
        ctx, ctx_names = rt.vectorize_context_booleans(metadata, model.context_schema)
        value, expanded_names = rt.expand_boolean_option_with_refined_context(
            np.concatenate((base, observed, group)), [*names, *observed_names, *group_names], ctx, ctx_names)
        require(tuple(expanded_names) == model.feature_names, 'feature schema/order differs')
        options.append(value)
    result = options[0] - options[1]
    require(result.shape == (len(model.feature_names),) and np.isfinite(result).all(), 'invalid feature vector')
    return result


def training(directory, spec, train, generated, arm, evidence, complete):
    target = directory / arm
    labels = ['A'] * 600 + ['B'] * 600
    random.Random(42).shuffle(labels)
    rows = []
    for case, human in zip(train, labels):
        cid = case['case_id']
        if cid not in generated:
            continue
        a, b = ((case['human_reply'], generated[cid]['replies']) if human == 'A' else
                (generated[cid]['replies'], case['human_reply']))
        blind, meta = blind_case(case, a, b)
        rows.append({'case_id': cid, 'split': 'train', 'human_option': human, 'source_case_id': cid,
            'source_chat_id': case['source_chat_id'], 'source_message_id': case['source_message_id'],
            'metadata': meta, 'blind': blind})
    by_id = {r['case_id']: r for r in rows}
    checkpoint = evidence.document(target / ('features.json' if complete else 'preflight.json'))
    if complete:
        require(digest(evidence.document(target / 'training.json')['rows']) == digest(rows) and len(rows) == 1200,
                'training labels/order differ')
        fspec = evidence.document(target / 'spec.json')
        require(fspec == {**spec, 'id': directory.name + '-' + arm, 'feature_config': spec['feature_configs'][arm],
                          'training_sha256': digest(rows)}, 'feature run spec differs')
        require(checkpoint['identity'] == digest({'training': digest(rows), 'config': spec['feature_configs'][arm],
            'runtime': sha256_file(ROOT / 'src/judge/corrected_v1.py'),
            'extractor': sha256_file(ROOT / 'src/judge/lr_retrain.py')}), 'feature checkpoint identity differs')
        require(set(checkpoint['entries']) == set(by_id), 'missing or extra training features')
    else:
        require(checkpoint['identity'] == digest(spec) and set(checkpoint['entries']) == {r['case_id'] for r in train[:2]},
                'preflight conditions changed')
    model = rt.load_formal_judge(directory / 'model_template.json')
    vectors = {}
    with generation_cache(evidence, target):
        for cid, entry in checkpoint['entries'].items():
            row = by_id[cid]
            require(entry['status'] == 'ok' and entry['input_sha256'] == digest(row)
                    and entry['features_sha256'] == digest(entry['features']), 'feature content changed')
            trace = evidence.document(target / 'traces' / (entry['trace_ref'] + '.json'))
            require(trace['status'] == 'ok' and trace['case']['blind'] == row['blind'] and len(trace['operations']) == 1,
                    'feature trace mismatch')
            evidence.feature_request(trace['operations'][0], cid, row['blind'], entry['features'], spec['feature_configs'][arm])
            vectors[cid] = vector(model, entry['features'], row['blind'], row['metadata'])
            evidence.counts[arm + '_training_features'] += 1
    if not complete:
        return
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        fitted = LogisticRegression(**spec['recipe']).fit(csr_matrix(np.stack([vectors[r['case_id']] for r in rows])),
                                                         np.array([r['human_option'] == 'A' for r in rows]))
    artifact = evidence.document(target / 'correction.json')
    delta = float(np.max(np.abs(fitted.coef_[0] - artifact['final_model']['coefficients'])))
    require(delta < 1e-8 and artifact['training']['features_sha256'] == sha256_file(target / 'features.json')
            and artifact['training']['evaluation_used_for_fit'] is False, 'refit cannot reproduce saved weights')
    ref = evidence.document(target / 'candidate.json')['judge_ref']
    info = versions.judge_dir(ref)
    for name in ('prompt.md', 'profile.json', 'reference.json'):
        require(sha256_file(info['dir'] / name) == sha256_file(directory / name), 'candidate static assets changed')
    require(sha256_file(info['dir'] / 'correction.json') == sha256_file(target / 'correction.json'), 'candidate weights changed')
    require(info['config']['llm'] == spec['scoring']['initial'] and info['config']['feature_llm'] == spec['feature_configs'][arm]
            and info['config']['correction_threshold'] == .7 and info['config'].get('decision_policy') != 'lr_only',
            'hybrid configuration differs')
    return {'judge_ref': ref, 'training_cases': len(rows), 'dimensions': len(model.feature_names), 'coefficient_max_difference': delta}


def initial_request(op, cid, prompt, response, config, evidence, visited=None):
    visited = set() if visited is None else visited
    verified = 0
    for event in op['events']:
        if event['kind'] == 'codex' and event['request']['prompt'] == prompt:
            require(event['request'] == {'prompt': prompt, 'schema': None, 'config': config}, 'initial request differs')
            if event['status'] == 'ok':
                try:
                    parsed = rt._parse_api_judge_response(event['response']['text'])
                except ValueError:
                    continue
                if parsed == response:
                    verified += 1
    for entry in evidence.entries(op, cid):
        identity = entry['identity']
        if entry['layer'] == 'llm_request' and identity.get('prompt') == prompt:
            require(identity['schema'] is None and all(identity['client'][k] == config[k] for k in
                ('provider', 'model', 'reasoning_effort', 'codex_cli_version')), 'initial cached request differs')
            require(rt._parse_api_judge_response(entry['value']) == response, 'initial cached response differs')
            verified += 1
        if entry['layer'] == 'judge_initial' and identity.get('prompt') == prompt:
            require(entry['value'] == response and all(identity['client'][k] == config[k] for k in
                ('provider', 'model', 'reasoning_effort', 'codex_cli_version')), 'initial decision cache differs')
            # A parsed decision alone is insufficient: follow its producing request.
            origin = entry['origin']
            key = (origin['experiment_path'], origin['trace_ref'], origin['operation_index'])
            if key not in visited:
                visited.add(key)
                trace = evidence.document(Path(origin['experiment_path']) / 'traces' / (origin['trace_ref'] + '.json'))
                producer = trace['operations'][origin['operation_index']]
                require(str(trace['case_id']) == str(cid) and producer['round'] == op['round'],
                        'initial cache producer belongs to another case/round')
                # The producer can reference its own cache entry; visited breaks that cycle.
                initial_request(producer, cid, prompt, response, config, evidence, visited)
                verified += 1
    require(verified > 0, 'no actual initial prompt evidence')
    evidence.counts['initial_requests_verified'] += 1
