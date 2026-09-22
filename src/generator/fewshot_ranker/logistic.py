"""Sparse categorical LR: candidate scores, context-balanced pairwise supervision."""
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ..history_sources import require
from .training import inner_split, pairs

RECIPE = dict(c_values=[.01, .1, 1., 10.], inner_fraction=.2, penalty='l2',
    solver='liblinear', fit_intercept=False, max_iter=2000, tolerance=1e-6,
    seed=20260920, encoding='categorical_one_hot_fit_only',
    objective='per_context_mean_logistic_loss_on_score_positive_minus_score_negative')


def encoder(rows, indices):
    require(rows and all(isinstance(v, str) for row in rows for v in row.values()),
            'LR requires categorical strings; no continuous or ordinal numeric bypass')
    value = DictVectorizer(sparse=True, sort=True)
    value.fit([rows[i] for i in indices])
    return value


def fit(x, observations, groups, c, recipe):
    pair_indices, weights, mixed = pairs(groups, observations)
    difference = x[pair_indices[:, 0]] - x[pair_indices[:, 1]]
    # Mirrored labels give sklearn both classes. Their half weights preserve
    # exactly one unit of total supervision for each mixed-label context.
    examples = sparse.vstack((difference, -difference), format='csr')
    labels = np.r_[np.ones(len(weights)), np.zeros(len(weights))]
    model = LogisticRegression(C=c, solver=recipe['solver'], fit_intercept=False,
        max_iter=recipe['max_iter'], tol=recipe['tolerance'], random_state=recipe['seed'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(examples, labels, sample_weight=np.r_[weights, weights]/2)
    return model, dict(mixed_contexts=mixed, pairs=len(pair_indices),
                       iterations=int(model.n_iter_[0]))


def score_document(value, rows):
    vectorizer = DictVectorizer(sparse=True, sort=True)
    vectorizer.vocabulary_ = value['vocabulary']
    vectorizer.feature_names_ = value['feature_names']
    return np.asarray(vectorizer.transform(rows) @ np.asarray(value['weights'])).ravel()


def train(groups, observations, rows, recipe=None):
    recipe = dict(RECIPE if recipe is None else recipe)
    require(recipe['c_values'] and all(c > 0 for c in recipe['c_values']), 'Invalid LR regularization grid')
    train_groups, valid_groups, split = inner_split(groups, recipe['inner_fraction'])
    fit_indices = [i for g in train_groups for i in g['indices']]
    vectorizer = encoder(rows, fit_indices)
    x = vectorizer.transform(rows)
    p, w, mixed = pairs(valid_groups, observations)
    trials = []
    for c in recipe['c_values']:
        model, detail = fit(x, observations, train_groups, c, recipe)
        scores = model.decision_function(x)
        loss = float(np.sum(np.logaddexp(0, -(scores[p[:, 0]]-scores[p[:, 1]]))*w)/mixed)
        trials.append(dict(C=c, inner_pairwise_loss=loss, **detail))
    chosen = min(trials, key=lambda item: (item['inner_pairwise_loss'], item['C']))
    all_fit = [g for g in groups if g['split'] == 'fit']
    vectorizer = encoder(rows, [i for g in all_fit for i in g['indices']])
    x = vectorizer.transform(rows)
    model, detail = fit(x, observations, all_fit, chosen['C'], recipe)
    value = dict(schema=1, kind='fewshot_pairwise_categorical_lr', recipe=recipe, C=chosen['C'],
        vocabulary={k: int(v) for k, v in vectorizer.vocabulary_.items()},
        feature_names=vectorizer.get_feature_names_out().tolist(), weights=model.coef_[0].tolist(),
        intercept=0., unseen_category='zero contribution', fields=len(rows[0]))
    predicted = model.decision_function(x)
    require(bool(np.isfinite(predicted).all()), 'Nonfinite LR scores')
    return value, predicted, dict(internal_split=split, trials=trials, selected_C=chosen['C'],
        fitting=detail, fields=len(rows[0]), one_hot_dimensions=x.shape[1],
        vocabulary='fit only', model_scores='uncalibrated logits; sigmoid(score(A)-score(B))')
