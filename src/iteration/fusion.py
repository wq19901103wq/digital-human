"""Build and reconstruct fixed GBDT/LR fusion using training-only scales.

No evaluation answers or model requests are used. Evaluation and promotion
remain ordinary registered experiments with their independent confirmations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import sha256_file
from ..judge import corrected_v1 as rt
from ..judge.gbdt import GBDTScorer, option_vectors
from . import gbdt_evidence, versions
from .training_evidence import read, require


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def _parent(ref, policy):
    require(isinstance(ref, str) and Path(ref).name == ref and ref not in ('', '.', '..'),
            'Fusion parent identifier is invalid')
    info = versions.judge_dir(ref)
    directory, cfg = info['dir'], info['config']
    require(cfg.get('decision_policy') == policy, 'Fusion parent policy differs')
    record = read(directory / 'learning.json')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()
              and p.name not in ('config.json', 'meta.json', 'learning.json')}
    require(actual == set(record['asset_files']), 'Fusion parent learning assets differ')
    for name, expected in record['asset_files'].items():
        require(Path(name).name == name and not (directory / name).is_symlink()
                and sha256_file(directory / name) == expected, 'Fusion parent asset changed')
    require(cfg['prompt_sha256'] == sha256_file(directory / 'prompt.md')
            and all(sha256_file(directory / n) == h for n, h in cfg['assets'].items()),
            'Fusion parent configuration assets differ')
    return info, record


def _document(gbdt, lr, data_ref, study, spec, arm):
    rows_path, features_path = (study / arm / n for n in ('training.json', 'features.json'))
    rows, entries = read(rows_path)['rows'], read(features_path)['entries']
    require(spec['data_ref'] == data_ref and len(rows) == spec['training_total']
            and len({r['case_id'] for r in rows}) == len(rows)
            and all(r['split'] == 'train' for r in rows), 'Fusion scales require complete training rows only')
    model = rt.load_formal_judge(lr['dir'] / 'correction.json')
    scorer = GBDTScorer(read(gbdt['dir'] / 'scorer.json'))
    pairs = [option_vectors(model, entries[r['case_id']]['features'], r['blind'], r['metadata']) for r in rows]
    a, b = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
    margins = {'gbdt': scorer.score(a) - scorer.score(b),
               'lr': (a @ model.coefficients) - (b @ model.coefficients)}
    scales = {k: float(np.sqrt(np.mean(v ** 2))) for k, v in margins.items()}
    require(all(np.isfinite(v) and v > 0 for v in scales.values()), 'Fusion training scale is zero or invalid')
    return {'schema': 1, 'kind': 'gbdt_lr_score_fusion_v1',
        'parents': {'gbdt': gbdt['id'], 'lr': lr['id']}, 'weights': {'gbdt': .8, 'lr': .2},
        'normalization': {'method': 'training_pair_margin_rms', 'scales': scales,
            'data_ref': data_ref, 'study': study.name, 'arm': arm, 'total': len(rows),
            'training_sha256': sha256_file(rows_path), 'features_sha256': sha256_file(features_path)}}


def _assemble(gbdt, lr, record, document):
    cfg = gbdt['config']
    assets = {n: (gbdt['dir'] / n).read_bytes() for n in ['prompt.md', *cfg['assets']]}
    assets['fusion.json'] = _bytes(document)
    addition = {'fusion.json': hashlib.sha256(assets['fusion.json']).hexdigest()}
    parent_paths = [p for info in (gbdt, lr) for p in info['dir'].rglob('*')
                    if p.is_file() and p.name != 'meta.json']
    learning = {**record, 'asset_files': {**record['asset_files'], **addition},
        'evidence_files': {**record['evidence_files'],
            **{str(p.resolve()): sha256_file(p) for p in parent_paths}}}
    assets['learning.json'] = _bytes(learning)
    config = {**cfg, 'decision_policy': 'score_fusion', 'fusion_file': 'fusion.json',
        'assets': {**cfg['assets'], **addition, 'learning.json': hashlib.sha256(assets['learning.json']).hexdigest()}}
    return config, assets


def verify(directory, record, original, study, source_spec, arm):
    """Validate parents against the already reconstructed clean LR training."""
    from .learning_guard import RunSeal
    document = read(directory / 'fusion.json')
    parents = document.get('parents', {})
    require(set(parents) == {'gbdt', 'lr'}, 'Fusion parents differ')
    gbdt, tree_record = _parent(parents['gbdt'], 'gbdt_only')
    lr, lr_record = _parent(parents['lr'], 'lr_only')
    seal = RunSeal([directory, gbdt['dir'], lr['dir'], original, study])
    old_cfg, old_record = read(original / 'config.json'), read(original / 'learning.json')
    require(lr['config'] == {**old_cfg, 'decision_policy': 'lr_only'} and lr_record == old_record,
            'Fusion LR parent differs from reconstructed training')
    gbdt_evidence.verify(gbdt['dir'], tree_record, original, study, source_spec, arm)
    expected = _document(gbdt, lr, source_spec['data_ref'], study, source_spec, arm)
    require(document == expected, 'Fusion weights or training-only scales differ')
    config, assets = _assemble(gbdt, lr, tree_record, expected)
    require(read(directory / 'config.json') == config and record == json.loads(assets['learning.json']),
            'Fusion configuration or learning evidence differs')
    require(all((directory / n).read_bytes() == payload for n, payload in assets.items()),
            'Fusion inherited assets differ')
    seal.check()


def create(gbdt_ref, lr_ref, data_ref, change):
    """Create an immutable candidate; never evaluate or move version pointers."""
    from .learning_guard import RunSeal, require_materials
    gbdt, record = _parent(gbdt_ref, 'gbdt_only')
    lr, _ = _parent(lr_ref, 'lr_only')
    origin = read(gbdt['dir'] / 'source_judge.json')
    name, arm = origin['study'], origin['arm']
    require(all(isinstance(n, str) and Path(n).name == n and n not in ('', '.', '..')
                for n in (name, arm)), 'Fusion training origin is invalid')
    study = versions.PRIVATE / 'judge_training' / name
    seal = RunSeal([gbdt['dir'], lr['dir'], study])
    require_materials(data_ref, [gbdt['dir'], lr['dir']])
    spec = read(study / 'spec.json')
    original = versions.judge_dir(read(study / arm / 'candidate.json')['judge_ref'])['dir']
    require(lr['config'] == {**read(original / 'config.json'), 'decision_policy': 'lr_only'}
            and read(lr['dir'] / 'learning.json') == read(original / 'learning.json'),
            'Fusion LR parent differs from reconstructed training')
    document = _document(gbdt, lr, data_ref, study, spec, arm)
    config, assets = _assemble(gbdt, lr, record, document)
    seal.check()
    return versions.create_judge_version(config, {'change': change, 'fusion': document}, assets=assets)
