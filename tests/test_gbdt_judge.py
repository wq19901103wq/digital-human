"""Scoring, replay and fail-closed asset checks for the GBDT runtime adapter."""
import hashlib
import json

import numpy as np
import pytest
from scipy.special import expit

from src import cache
from src.config import ConfigError
from src.iteration import versions
from src.judge import gbdt, corrected, corrected_v1 as rt
from src.judge.judge import build_judge
from src.judge.lr_retrain import LRJudge
from test_corrected_judge import bundle, case, features

xgb = pytest.importorskip('xgboost')


@pytest.fixture(autouse=True)
def offline_clients(monkeypatch):
    class NoRequest:
        def __init__(self, config):
            self.config = config

        def run(self, *args, **kwargs):
            pytest.fail('runtime tests must not request an LLM')
    monkeypatch.setattr(corrected, 'CodexJudgeClient', NoRequest)


def document(dimensions):
    x = np.zeros((4, dimensions))
    x[:, 0] = [0., 1., 0., 1.]
    recipe = dict(kind='gbdt', objective='rank:pairwise', n_estimators=2,
                  max_depth=1, min_child_weight=0, reg_lambda=0, n_jobs=1, random_state=0)
    ranker = xgb.XGBRanker(**{k: v for k, v in recipe.items() if k != 'kind'})
    ranker.fit(x, [0, 1, 0, 1], qid=[0, 0, 1, 1])
    return dict(schema=1, scoring='sigmoid(score(A)-score(B))', recipe=recipe,
                dimensions=dimensions, parameters={
                    'booster_json': ranker.get_booster().save_raw(raw_format='json').decode()},
                diagnostics={'converged': True})


@pytest.fixture
def saved(bundle):
    cfg, directory = bundle
    model = rt.load_formal_judge(directory / 'correction.json')
    payload = json.dumps(document(len(model.feature_names))).encode()
    assets = {name: (directory / name).read_bytes() for name in ['prompt.md', *cfg['assets']]}
    assets['scorer.json'] = payload
    cfg = {**cfg, 'decision_policy': 'gbdt_only', 'scorer_file': 'scorer.json',
           'assets': {**cfg['assets'], 'scorer.json': hashlib.sha256(payload).hexdigest()}}
    jid = versions.create_judge_version(cfg, {}, root=directory.parent, assets=assets)
    return cfg, directory.parent / jid


def test_nonlinear_options_scored_before_difference():
    scorer = gbdt.GBDTScorer(document(1))
    a, b = np.array([[1.], [0.]]), np.array([[2.], [1.]])
    p = scorer.probability_a(a, b)
    assert p[0] == .5  # 1 and 2 land in the same leaf.
    assert p[1] < .5
    assert np.allclose(p, expit(scorer.score(a) - scorer.score(b)))
    assert not np.allclose(p, expit(scorer.score(a - b)))
    assert np.allclose(p + scorer.probability_a(b, a), 1)


def test_dispatch_and_frozen_feature_order(saved, case):
    cfg, directory = saved
    judge = build_judge({}, {'config': cfg, 'dir': directory})
    assert isinstance(judge, gbdt.GBDTJudge)
    blind, metadata = corrected.blind_case(case, ['八点'], ['九点'])
    parsed = {'option_A': features(), 'option_B': features(2)}
    a, b = gbdt.option_vectors(judge.model, parsed, blind, metadata)
    expected = float(expit(judge.scorer.score(a[None])[0] - judge.scorer.score(b[None])[0]))
    assert judge.probability_a(parsed, blind, metadata) == expected
    # Same input vocabulary as the existing LR, including observable/context crosses.
    from src.judge.lr_retrain import difference_vector
    assert np.array_equal(a - b, difference_vector(judge.model, parsed, blind, metadata))
    from dataclasses import replace
    changed = replace(judge.model, feature_names=tuple(reversed(judge.model.feature_names)))
    with pytest.raises(ConfigError, match='特征定义'):
        gbdt.option_vectors(changed, parsed, blind, metadata)


@pytest.mark.parametrize('human', ['A', 'B'])
def test_replay_preserves_rounds_without_llm_requests(saved, case, human):
    cfg, directory = saved
    case = {**case, 'ai_replies': ['九点']}
    a, b = (case['human_reply'], case['ai_replies']) if human == 'A' else (case['ai_replies'], case['human_reply'])
    blind, metadata = corrected.blind_case(case, a, b)
    mapping = dict(human_option=human, candidate_option='B' if human == 'A' else 'A',
                   blind_case=blind, context_metadata=metadata)
    class NoRequest:
        def run(self, *args, **kwargs):
            pytest.fail('a saved feature draw must not request an LLM')
    class Replay:
        rounds = []
        def get(self, row, round_index, client, source_check):
            source_check()
            assert row == case
            self.rounds.append(round_index)
            return dict(mapping=mapping, features={'option_A': features(), 'option_B': features(2)})
    replay = Replay()
    judge = gbdt.GBDTJudge(cfg, directory, client=NoRequest(), replay=replay)
    for round_index in range(3):
        token = cache._scope.set({'context': {'round': round_index}})
        try:
            actual = judge.is_ai(case, case['ai_replies'])
            chosen = 'A' if judge.last_verdict['small_model_probability_a'] >= .5 else 'B'
            assert actual is (chosen == human)
            assert judge.last_verdict['decision_policy'] == 'gbdt_only'
            with pytest.raises(ConfigError, match='候选回复'):
                judge.is_ai(case, ['changed'])
        finally:
            cache._scope.reset(token)
    assert replay.rounds == [0, 1, 2]
    with pytest.raises(ConfigError, match='独立轮次'):
        judge.is_ai(case, case['ai_replies'])


def test_asset_tampering_rejected_before_request(saved, case):
    cfg, directory = saved
    judge = gbdt.GBDTJudge(cfg, directory)
    target = directory / 'scorer.json'
    target.chmod(0o644)
    target.write_text('{}')
    with pytest.raises(ConfigError, match='学习材料变化'):
        judge.is_ai(case, ['九点'])
    with pytest.raises(ConfigError, match='资产不匹配'):
        gbdt.GBDTJudge(cfg, directory)


@pytest.mark.parametrize('name', ['../scorer.json', '/scorer.json', 'missing.json', '.', None])
def test_unbound_model_rejected(saved, name):
    cfg, directory = saved
    with pytest.raises(ConfigError, match='已绑定的资产'):
        gbdt.GBDTJudge({**cfg, 'scorer_file': name}, directory)


@pytest.mark.parametrize('change', [
    {'schema': 2}, {'scoring': 'score(A-B)'}, {'dimensions': True}, {'dimensions': 2},
    {'recipe': {'kind': 'dnn'}}, {'parameters': {'booster_json': '{}'}},
])
def test_invalid_model_rejected(change):
    with pytest.raises(ConfigError):
        gbdt.GBDTScorer({**document(1), **change})


def test_booster_objective_cannot_be_relabelled():
    stored = document(1)
    booster = json.loads(stored['parameters']['booster_json'])
    booster['learner']['objective'] = {'name': 'reg:squarederror', 'reg_loss_param': {'scale_pos_weight': '1'}}
    stored['parameters']['booster_json'] = json.dumps(booster)
    with pytest.raises(ConfigError, match='配对目标'):
        gbdt.GBDTScorer(stored)


def test_booster_dependency_change_rejected(monkeypatch):
    stored = document(1)
    monkeypatch.setattr(xgb, '__version__', '0.0.0')
    with pytest.raises(ConfigError, match='XGBoost 版本不一致'):
        gbdt.GBDTScorer(stored)


@pytest.mark.parametrize('x', [np.zeros((1, 2)), [[float('nan')]], [], [1.]])
def test_invalid_input_rejected(x):
    scorer = gbdt.GBDTScorer(document(1))
    with pytest.raises(ConfigError):
        scorer.score(x)
    with pytest.raises(ConfigError):
        scorer.probability_a([[0.]], [[0.], [1.]])


def test_lr_dispatch_and_scoring_unchanged(bundle, case):
    cfg, directory = bundle
    judge = build_judge({}, {'config': {**cfg, 'decision_policy': 'lr_only'}, 'dir': directory})
    assert type(judge) is LRJudge
    blind, metadata = corrected.blind_case(case, ['八点'], ['九点'])
    assert judge.probability_a({'option_A': features(), 'option_B': features()}, blind, metadata) == .5
    assert judge.decision_policy == 'lr_only'
