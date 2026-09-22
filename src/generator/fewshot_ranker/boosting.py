"""Categorical XGBoost with context-balanced pairwise logistic supervision.

Trees score each candidate independently. A custom objective sums the same
positive-negative pair loss as the neural/LR comparisons. A diagonal upper
bound (twice the pair curvature) replaces its coupled Hessian for tree fitting.
"""
import json

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction import DictVectorizer
from threadpoolctl import threadpool_limits
import xgboost as xgb

from ..history_sources import require
from . import logistic, training


def recipes():
    return [dict(max_depth=depth, min_child_weight=child, reg_lambda=decay,
        eta=.05, subsample=.8, colsample_bytree=.8, alpha=0., max_rounds=250,
        patience=20, seed=20260920, inner_fraction=.2)
        for depth, child, decay in [(2, 1, 1), (2, 5, 10), (4, 1, 1),
                                    (4, 5, 10), (6, 1, 1), (6, 5, 10)]]


def problem(groups, observations):
    indices = [i for group in groups for i in group['indices']]
    mapping = {index: offset for offset, index in enumerate(indices)}
    pairs, weights, count = training.pairs(groups, observations)
    local = np.asarray([[mapping[p], mapping[n]] for p, n in pairs])
    return indices, local, weights, count


def derivatives(scores, pairs, weights):
    p, n = pairs.T
    probability = expit(scores[p]-scores[n])
    gradient, hessian = np.zeros(len(scores)), np.zeros(len(scores))
    contribution = (probability-1)*weights
    curvature = 2*probability*(1-probability)*weights
    np.add.at(gradient, p, contribution)
    np.add.at(gradient, n, -contribution)
    np.add.at(hessian, p, curvature)
    np.add.at(hessian, n, curvature)
    return gradient, hessian


def pair_loss(scores, pairs, weights, count):
    return float(np.sum(np.logaddexp(0, -(scores[pairs[:, 0]]-scores[pairs[:, 1]]))*weights)/count)


def fit(x, observations, groups, valid, recipe, rounds=None):
    indices, pairs, weights, count = problem(groups, observations)
    dtrain = xgb.DMatrix(x[indices], nthread=1)
    parameters = {key: recipe[key] for key in ('max_depth', 'min_child_weight', 'reg_lambda',
        'eta', 'subsample', 'colsample_bytree', 'alpha', 'seed')}
    parameters.update(tree_method='hist', nthread=1, base_score=0., disable_default_eval_metric=1)
    options, history = {}, {}
    if valid:
        vi, vp, vw, vc = problem(valid, observations)
        options = dict(evals=[(xgb.DMatrix(x[vi], nthread=1), 'inner')],
            custom_metric=lambda predicted, _: ('pairwise_loss', pair_loss(predicted, vp, vw, vc)),
            early_stopping_rounds=recipe['patience'], maximize=False, evals_result=history)
    model = xgb.train(parameters, dtrain, num_boost_round=rounds or recipe['max_rounds'],
        obj=lambda predicted, _: derivatives(predicted, pairs, weights), verbose_eval=False, **options)
    if valid:
        selected_rounds = model.best_iteration+1
        model = model[:selected_rounds]
    else:
        selected_rounds = rounds or recipe['max_rounds']
    return model, dict(mixed_contexts=count, pairs=len(pairs), selected_rounds=selected_rounds,
                       history=history)


@threadpool_limits.wrap(limits=1)
def score_document(value, rows):
    require(value['kind'] == 'fewshot_pairwise_xgboost_v1' and value['xgboost_version'] == xgb.__version__,
            'XGBoost schema or runtime differs')
    vectorizer = DictVectorizer(sparse=True, sort=True)
    vectorizer.vocabulary_, vectorizer.feature_names_ = value['vocabulary'], value['feature_names']
    model = xgb.Booster(params={'nthread': 1})
    model.load_model(bytearray(json.dumps(value['booster']).encode()))
    return model.predict(xgb.DMatrix(vectorizer.transform(rows), nthread=1), output_margin=True)


@threadpool_limits.wrap(limits=1)
def train(groups, observations, rows, candidates):
    require(candidates and len({r['inner_fraction'] for r in candidates}) == 1,
            'XGBoost tuning requires one inner split')
    fit_groups, valid_groups, split = training.inner_split(groups, candidates[0]['inner_fraction'])
    encoder = logistic.encoder(rows, [i for g in fit_groups for i in g['indices']])
    x = encoder.transform(rows)
    vp, vw, vc = training.pairs(valid_groups, observations)
    matrix = xgb.DMatrix(x, nthread=1)
    trials = []
    for index, recipe in enumerate(candidates):
        model, detail = fit(x, observations, fit_groups, valid_groups, recipe)
        predicted = model.predict(matrix, output_margin=True)
        loss = pair_loss(predicted, vp, vw, vc)
        require(np.isfinite(loss), 'Nonfinite XGBoost tuning loss')
        trials.append(dict(index=index, recipe=recipe, inner_pairwise_loss=loss, **detail))
        print(f'XGBoost trial {index+1}/{len(candidates)}: inner loss={loss:.6f}, '
              f'rounds={detail["selected_rounds"]}', flush=True)
    chosen = min(trials, key=lambda trial: (trial['inner_pairwise_loss'], trial['index']))
    all_fit = [group for group in groups if group['split'] == 'fit']
    encoder = logistic.encoder(rows, [i for g in all_fit for i in g['indices']])
    x = encoder.transform(rows)
    model, detail = fit(x, observations, all_fit, [], chosen['recipe'], chosen['selected_rounds'])
    value = dict(schema=1, kind='fewshot_pairwise_xgboost_v1', xgboost_version=xgb.__version__,
        recipe=chosen['recipe'], rounds=chosen['selected_rounds'],
        encoding='fit_only_categorical_one_hot; sparse_absence_is_missing; no_ordinal_IDs',
        objective='per_context_pairwise_logistic; diagonal_hessian_upper_bound',
        vocabulary={key: int(index) for key, index in encoder.vocabulary_.items()},
        feature_names=encoder.get_feature_names_out().tolist(),
        booster=json.loads(model.save_raw(raw_format='json').decode()))
    predicted = model.predict(xgb.DMatrix(x, nthread=1), output_margin=True)
    require(bool(np.isfinite(predicted).all()), 'Nonfinite XGBoost predictions')
    return value, predicted, dict(internal_split=split, trials=trials, selected_trial=chosen['index'],
        selected_recipe=chosen['recipe'], selected_rounds=chosen['selected_rounds'], fitting=detail,
        fields=len(rows[0]), one_hot_dimensions=x.shape[1],
        selection='minimum inner pairwise loss; no outer validation tuning')
