"""Verify a saved paired GBDT against an already reconstructed training source.

No evaluation data, model requests, new recipes or output files are used here.
The original LR source audit must run first: its feature requests, examples and
train-only vocabulary are the learning inputs of the derived scorer as well.
"""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from ..cache import digest
from ..config import sha256_file
from ..judge import corrected_v1 as rt
from ..judge.gbdt import GBDTScorer, option_vectors
from . import runtime, versions
from .training_evidence import read, require


def _name(value):
    require(isinstance(value, str) and value not in ('', '.', '..')
            and Path(value).name == value, 'GBDT source identifier is invalid')
    return value


def _refit(recipe, a, b, y):
    from xgboost import XGBRanker
    cfg = {k: v for k, v in recipe.items() if k not in ('id', 'kind')}
    require(set(cfg) <= {'objective', 'n_estimators', 'max_depth', 'learning_rate',
        'reg_lambda', 'reg_alpha', 'min_child_weight', 'subsample', 'colsample_bytree',
        'tree_method', 'n_jobs', 'random_state'} and cfg.get('objective') == 'rank:pairwise'
        and cfg.get('n_jobs') == 1 and type(cfg.get('random_state')) is int,
        'GBDT recipe is not a reproducible paired fit')
    x = np.stack((a, b), axis=1).reshape(-1, a.shape[1])
    labels = np.stack((y, 1. - y), axis=1).ravel()
    fitted = XGBRanker(**cfg).fit(x, labels, qid=np.repeat(np.arange(len(a)), 2))
    return {'schema': 1, 'scoring': 'sigmoid(score(A)-score(B))', 'recipe': recipe,
        'dimensions': a.shape[1], 'parameters': {
            'booster_json': fitted.get_booster().save_raw(raw_format='json').decode()},
        'diagnostics': {'converged': True, 'iterations': cfg['n_estimators']}}


def verify(directory, record, original, study, source_spec, arm):
    """Called only after learning_guard reconstructed the original clean study."""
    from .learning_guard import RunSeal
    descriptor = read(directory / 'scorer_source.json')
    require(set(descriptor) == {'schema', 'kind', 'study', 'arm', 'recipe_id'}
            and descriptor['schema'] == 1 and descriptor['kind'] == 'paired_gbdt_v1'
            and descriptor['arm'] == arm, 'GBDT source descriptor differs')
    sweep = versions.PRIVATE / 'judge_training' / _name(descriptor['study'])
    recipe_id = _name(descriptor['recipe_id'])
    target = sweep / arm
    model_path = target / 'models' / (recipe_id + '.json')
    receipt_path = target / 'receipts' / (recipe_id + '.json')
    manifest_path = target / 'training_source.json'
    source_record = read(original / 'learning.json')
    additions = {name: sha256_file(directory / name) for name in ('scorer.json', 'scorer_source.json')}
    require(record['asset_files'] == {**source_record['asset_files'], **additions},
            'GBDT inherited learning assets differ')
    require({k: v for k, v in record.items() if k not in ('asset_files', 'evidence_files')} ==
            {k: v for k, v in source_record.items() if k not in ('asset_files', 'evidence_files')},
            'GBDT learning cutoff or policy differs')
    cfg, old_cfg = read(directory / 'config.json'), read(original / 'config.json')
    expected = {**old_cfg, 'decision_policy': 'gbdt_only', 'scorer_file': 'scorer.json',
        'assets': {**old_cfg['assets'], **additions, 'learning.json': sha256_file(directory / 'learning.json')}}
    require(cfg == expected, 'GBDT feature configuration differs from reconstructed source')
    spec = read(sweep / 'spec.json')
    paths = [study / arm / name for name in
             ('spec.json', 'training.json', 'features.json', 'correction.json', 'candidate.json')]
    proof_paths = [sweep / 'spec.json', manifest_path, model_path, receipt_path,
                   study / 'model_template.json', *paths, *runtime.verify_inputs(spec['inputs'])]
    seal = RunSeal([directory, original], proof_paths)
    require(spec['kind'] == 'paired_classifier_sweep' and spec['dataset'] == 'development'
            and spec['adoption_allowed'] is False
            and Path(spec['source']).resolve() == study.resolve()
            and spec['source_sha256'] == sha256_file(study / 'spec.json'),
            'GBDT sweep source differs from reconstructed study')
    for key in ('data_ref', 'generator_ref', 'feature_configs', 'training_total'):
        require(spec[key] == source_spec[key], 'GBDT training conditions differ: ' + key)
    require(arm in spec['arms'] and spec['packages'] == {
        p: importlib.metadata.version(p) for p in spec['packages']}, 'GBDT dependencies or arm differ')
    recipes = [r for r in spec['recipes'] if r['id'] == recipe_id]
    require(len(recipes) == 1 and recipes[0]['kind'] == 'gbdt', 'GBDT recipe missing or ambiguous')
    recipe = recipes[0]
    model = read(model_path)
    require(model['recipe'] == recipe and sha256_file(directory / 'scorer.json') == sha256_file(model_path),
            'GBDT candidate weights differ from saved model')
    GBDTScorer(model)  # Reject incompatible booster versions before refitting.
    rows = read(study / arm / 'training.json')['rows']
    entries = read(study / arm / 'features.json')['entries']
    require(len(rows) == source_spec['training_total'] and all(r['split'] == 'train' for r in rows),
            'GBDT matrix contains non-training rows')
    template = rt.load_formal_judge(study / 'model_template.json')
    pairs = [option_vectors(template, entries[r['case_id']]['features'], r['blind'], r['metadata']) for r in rows]
    a, b = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
    y = np.array([r['human_option'] == 'A' for r in rows], dtype=float)
    manifest = {'study_sha256': sha256_file(sweep / 'spec.json'), 'arm': arm,
        'inputs': {str(p.resolve()): sha256_file(p) for p in paths},
        'training_matrix_sha256': digest({'case_ids': [r['case_id'] for r in rows],
                                          'a': a.tolist(), 'b': b.tolist(), 'y': y.tolist()}),
        'dimensions': a.shape[1], 'training_total': len(rows), 'evaluation_used_for_fit': False}
    require(read(manifest_path) == manifest, 'GBDT training matrix provenance differs')
    require(read(receipt_path) == {'identity': digest({'training': manifest, 'recipe': recipe}),
                                  'model_sha256': sha256_file(model_path)}, 'GBDT model receipt differs')
    require(model == _refit(recipe, a, b, y), 'GBDT saved trees cannot be reproduced from training features')
    required = {**source_record['evidence_files'], **{str(p.resolve()): sha256_file(p) for p in proof_paths}}
    require(record['evidence_files'] == required, 'GBDT learning evidence is incomplete or changed')
    seal.check()
