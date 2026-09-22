"""Score fusion and train-only provenance, without any model requests."""
import copy
import hashlib
import json
import shutil

import numpy as np
import pytest
from scipy.special import expit

from src.config import ConfigError
from src.iteration import fusion, learning_guard, versions
from src.judge import corrected
from src.judge.fusion import FusionJudge, FusionScorer
from src.judge.gbdt import GBDTScorer
from src.judge.judge import build_judge
from test_gbdt_evidence import trained, candidate, save, resign
from test_gbdt_judge import document, saved, offline_clients
from test_corrected_judge import bundle, case, features


def recipe():
    return {'schema': 1, 'kind': 'gbdt_lr_score_fusion_v1', 'weights': {'gbdt': .8, 'lr': .2},
            'normalization': {'method': 'training_pair_margin_rms', 'scales': {'gbdt': 2., 'lr': 3.}}}


def test_option_score_fusion_and_swap():
    tree = GBDTScorer(document(1))
    scorer = FusionScorer(tree, [2.], recipe())
    a, b = np.array([[1.], [0.]]), np.array([[2.], [1.]])
    expected = expit(.8 / 2 * (tree.score(a) - tree.score(b)) + .2 / 3 * (a[:, 0] - b[:, 0]) * 2)
    assert np.allclose(scorer.probability_a(a, b), expected)
    assert np.allclose(scorer.probability_a(a, b) + scorer.probability_a(b, a), 1)
    assert not np.allclose(expected, expit(scorer.score(a - b)))
    for value in (0, -1, float('nan'), float('inf'), True):
        changed = recipe()
        changed['normalization']['scales']['lr'] = value
        with pytest.raises(ConfigError):
            FusionScorer(tree, [2.], changed)


def test_dispatch_and_asset_binding(saved, case):
    cfg, directory = saved
    assets = {n: (directory / n).read_bytes() for n in ['prompt.md', *cfg['assets']]}
    assets['fusion.json'] = json.dumps(recipe()).encode()
    cfg = {**cfg, 'decision_policy': 'score_fusion', 'fusion_file': 'fusion.json',
           'assets': {**cfg['assets'], 'fusion.json': hashlib.sha256(assets['fusion.json']).hexdigest()}}
    jid = versions.create_judge_version(cfg, {}, root=directory.parent, assets=assets)
    directory = directory.parent / jid
    judge = build_judge({}, {'config': cfg, 'dir': directory})
    assert type(judge) is FusionJudge
    blind, metadata = corrected.blind_case(case, ['八点'], ['九点'])
    assert np.isfinite(judge.probability_a({'option_A': features(), 'option_B': features(2)}, blind, metadata))
    with pytest.raises(ConfigError, match='已绑定'):
        FusionJudge({**cfg, 'fusion_file': '../fusion.json'}, directory)
    (directory / 'fusion.json').chmod(0o644)
    (directory / 'fusion.json').write_text('{}')
    with pytest.raises(ConfigError, match='学习材料变化'):
        judge.is_ai(case, ['九点'])
    with pytest.raises(ConfigError, match='资产不匹配'):
        FusionJudge(cfg, directory)


@pytest.fixture
def parents(candidate):
    c = candidate
    cfg = fusion.read(c['directory'] / 'config.json')
    tree = versions.create_judge_version(cfg, {}, source_dir=c['directory'])
    cfg = {**fusion.read(c['original'] / 'config.json'), 'decision_policy': 'lr_only'}
    lr = versions.create_judge_version(cfg, {}, source_dir=c['original'])
    return c, tree, lr


def test_create_and_reconstruct_without_requests_or_pointer_changes(parents, monkeypatch):
    c, tree, lr = parents
    monkeypatch.setattr(corrected.CodexJudgeClient, 'run', lambda *a, **kw: pytest.fail('no model requests'))
    before = versions.POINTERS_PATH.read_bytes()
    jid = fusion.create(tree, lr, c['source_spec']['data_ref'], 'fixed fusion')
    directory = versions.judge_dir(jid)['dir']
    assert learning_guard.require_materials(c['source_spec']['data_ref'], [directory])['promotion_eligible']
    normalization = fusion.read(directory / 'fusion.json')['normalization']
    assert normalization['total'] == c['source_spec']['training_total']
    assert normalization['method'] == 'training_pair_margin_rms'
    assert versions.POINTERS_PATH.read_bytes() == before


@pytest.mark.parametrize('part', ['scales', 'weights', 'parent_config', 'evidence'])
def test_resigned_fusion_forgery_rejected(parents, tmp_path, part):
    c, tree, lr = parents
    jid = fusion.create(tree, lr, c['source_spec']['data_ref'], 'fixed fusion')
    directory = tmp_path / 'forged'
    shutil.copytree(versions.judge_dir(jid)['dir'], directory)
    for path in directory.iterdir():
        path.chmod(0o644)
    if part in ('scales', 'weights'):
        doc = fusion.read(directory / 'fusion.json')
        if part == 'scales':
            doc['normalization']['scales']['lr'] *= 2
        else:
            doc['weights'] = {'gbdt': .2, 'lr': .8}
        save(directory / 'fusion.json', doc)
    elif part == 'parent_config':
        path = versions.judge_dir(lr)['dir'] / 'config.json'
        path.chmod(0o644)
        cfg = fusion.read(path)
        cfg['feature_llm']['model'] = 'changed'
        save(path, cfg)
    else:
        doc = fusion.read(directory / 'learning.json')
        doc['evidence_files'].pop(str((versions.judge_dir(lr)['dir'] / 'config.json').resolve()))
        save(directory / 'learning.json', doc)
    resign({**c, 'directory': directory})
    with pytest.raises(ConfigError):
        learning_guard.require_materials(c['source_spec']['data_ref'], [directory])


def test_nontraining_normalization_rejected(parents, tmp_path):
    c, tree, lr = parents
    study = tmp_path / 'invalid-training'
    shutil.copytree(c['study'] / c['arm'], study / c['arm'])
    path = study / c['arm'] / 'training.json'
    doc = fusion.read(path)
    doc['rows'][0]['split'] = 'development'
    path.chmod(0o644)
    save(path, doc)
    with pytest.raises(ConfigError, match='training rows only'):
        fusion._document(versions.judge_dir(tree), versions.judge_dir(lr),
                         c['source_spec']['data_ref'], study, c['source_spec'], c['arm'])


def test_cli_forwards_frozen_parent_ids(monkeypatch):
    from scripts import evaluate_judge
    calls = []
    monkeypatch.setattr(fusion, 'create', lambda *args: calls.append(args) or 'candidate')
    monkeypatch.setattr(evaluate_judge.sys, 'argv', ['evaluate_judge.py', 'fuse', '--gbdt', 'tree',
        '--lr', 'linear', '--data', 'data', '--change', 'fixed fusion'])
    evaluate_judge.main()
    assert calls == [('tree', 'linear', 'data', 'fixed fusion')]
