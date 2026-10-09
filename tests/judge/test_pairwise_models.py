import json

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.config import ConfigError
from src.judge import pairwise_models as pm


@pytest.fixture
def pairs():
    rng = np.random.default_rng(13)
    a, b = rng.integers(0, 2, (2, 160, 4)).astype(float)
    y = (a[:, 0] + a[:, 1] * .7 > b[:, 0] + b[:, 1] * .7).astype(int)
    return a, b, y


def test_lr_reproduces_original_pairwise_recipe(pairs):
    a, b, y = pairs
    recipe = pm.recipes()[0]
    cfg = {k: v for k, v in recipe.items() if k not in ('id', 'kind')}
    expected = LogisticRegression(**cfg).fit(csr_matrix(a - b), y)
    actual = pm.fit(a, b, y, recipe)
    np.testing.assert_allclose(actual.parameters['coefficients'], expected.coef_[0], atol=1e-12)
    np.testing.assert_allclose(actual.probability_a(a, b), expected.predict_proba(csr_matrix(a - b))[:, 1], atol=1e-12)


@pytest.mark.parametrize('kind', ['lr', 'gbdt', 'dnn'])
def test_option_scoring_symmetry_and_roundtrip(pairs, kind):
    if kind in ('gbdt', 'dnn'):
        pytest.importorskip('xgboost' if kind == 'gbdt' else 'torch')  # 可选重依赖
    a, b, y = pairs
    recipe = next(r for r in pm.recipes() if r['kind'] == kind)
    with threadpool_limits(limits=1):
        scorer = pm.fit(a, b, y, recipe)
        copy = pm.Scorer.from_document(json.loads(json.dumps(scorer.document(), allow_nan=False)))
        p = scorer.probability_a(a, b)
        np.testing.assert_allclose(p, copy.probability_a(a, b), atol=1e-12)
        np.testing.assert_allclose(p + scorer.probability_a(b, a), 1., atol=1e-12)
        np.testing.assert_allclose(scorer.probability_a(a, a), .5)
        assert np.mean((p >= .5) == y) > .75


def test_nonlinear_scorer_distinguishes_equal_input_differences():
    scorer = pm.Scorer({'kind': 'dnn'}, 1, {'layers': [
        {'weight': [[1.]], 'bias': [0.]}, {'weight': [[1.]], 'bias': [0.]}]}, {})
    a, b = np.array([[1.], [3.]]), np.array([[0.], [2.]])
    np.testing.assert_equal(a - b, [[1.], [1.]])
    p = scorer.probability_a(a, b)
    assert abs(p[0] - p[1]) > .1
    np.testing.assert_allclose(p, expit(np.tanh(a[:, 0]) - np.tanh(b[:, 0])))


def test_shared_network_gradient():
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=(2, 5, 2))
    y, sizes = np.array([0., 1., 1., 0., 1.]), [2, 3, 2, 1]
    count = 2 * 3 + 3 + 3 * 2 + 2 + 2
    theta = rng.normal(0, .2, count)
    fn = lambda p: pm.network_loss_gradient(p, a, b, y, sizes, .1)
    assert check_grad(lambda p: fn(p)[0], lambda p: fn(p)[1], theta) < 1e-6
    reverse = pm.network_loss_gradient(theta, b, a, 1. - y, sizes, .1)
    np.testing.assert_allclose(fn(theta)[0], reverse[0], atol=1e-12)
    np.testing.assert_allclose(fn(theta)[1], reverse[1], atol=1e-12)


def test_grid_has_unique_ids_and_no_evaluation_stopping():
    grid = pm.recipes()
    assert len(grid) == len({r['id'] for r in grid}) == 14
    assert {r['kind'] for r in grid} == {'lr', 'gbdt', 'dnn'}
    assert all('early_stopping_rounds' not in r and 'eval_set' not in r for r in grid)


def test_bad_matrices_and_intercept_rejected(pairs):
    a, b, y = pairs
    with pytest.raises(ConfigError):
        pm.fit(a, b[:-1], y, pm.recipes()[0])
    with pytest.raises(ConfigError):
        pm.fit(a, b, y, {**pm.recipes()[0], 'fit_intercept': True})
    with pytest.raises(ConfigError):
        pm.fit(a, b, np.full_like(y, 2), pm.recipes()[0])

