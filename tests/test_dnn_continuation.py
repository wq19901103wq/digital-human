import copy

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts.legacy import continue_pairwise_dnn as job
from scripts.legacy.finalize_dnn_checkpoints import validate_segment
from src.config import ConfigError
from src.judge import pairwise_models as pm


def test_continuation_reuses_weights_and_reduces_same_objective():
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(2, 60, 3))
    y = (a[:, 0] > b[:, 0]).astype(float)
    recipe = dict(id='fixture', kind='dnn', hidden=[4, 2], l2=.01,
                  max_iter=1, gtol=1e-6, ftol=1e-10, random_state=0)
    with threadpool_limits(limits=1):
        initial = pm.fit(a, b, y, recipe).document()
        untouched = copy.deepcopy(initial)
        limited = job.optimize(initial, a, b, y, budget=1)
        assert not limited['diagnostics']['converged']
        assert limited['diagnostics']['stop_reason'].endswith('ITERATIONS REACHED LIMIT')
        final = job.optimize(initial, a, b, y, budget=2000)
        assert initial == untouched
        assert final == job.optimize(initial, a, b, y, budget=2000)
        assert final['diagnostics']['converged']
        assert final['diagnostics']['initial_objective'] == initial['diagnostics']['objective']
        assert final['diagnostics']['objective'] < initial['diagnostics']['objective']
        assert final['recipe'] == initial['recipe']
        scorer = pm.Scorer.from_document(final)
        np.testing.assert_allclose(scorer.probability_a(a, b) + scorer.probability_a(b, a), 1.)
        shapes = [3, 4, 2, 1]
        recovered = pm._unpack(job.pack(final), shapes)
        for (w, bias), saved in zip(recovered, final['parameters']['layers']):
            np.testing.assert_array_equal(w, saved['weight'])
            np.testing.assert_array_equal(bias, saved['bias'])
    invalid = copy.deepcopy(initial)
    invalid['parameters']['layers'][-1]['bias'] = [1.]
    with pytest.raises(ConfigError):
        job.pack(invalid)
    with pytest.raises(ConfigError):
        job.optimize(initial, a[:, :2], b[:, :2], y)
    saved = dict(identity='fixture', previous_model_digest=job.cache.digest(initial), segment=1,
                 document=final, document_sha256=job.cache.digest(final))
    assert validate_segment(saved, 'fixture', initial, 1) == final
    with pytest.raises(ConfigError):
        validate_segment(saved, 'different-data-or-config', initial, 1)
    altered = copy.deepcopy(saved)
    altered['document']['parameters']['layers'][0]['weight'][0][0] += .1
    with pytest.raises(ConfigError):
        validate_segment(altered, 'fixture', initial, 1)
