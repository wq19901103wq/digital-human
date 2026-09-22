"""Bounded DNN/DCN tuning on the chronological inner fit split only."""
import numpy as np

from .neural_models import document
from ..history_sources import require
from . import training


def recipes(cross_layers):
    return [dict(training.RECIPE, hidden=hidden, normalization=norm, dropout=dropout,
                 weight_decay=decay, cross_layers=cross_layers, max_epochs=30, patience=4)
            for hidden, norm, dropout, decay in
            [([64], 'layer', .2, .001), ([128, 64], 'layer', .3, .01),
             ([64, 32], 'batch', .2, .001)]]


def train(groups, observations, rows, candidates):
    require(candidates and len({r['inner_fraction'] for r in candidates}) == 1,
            'Neural tuning must use one common inner split')
    fit, valid, split = training.inner_split(groups, candidates[0]['inner_fraction'])
    p, w, count = training.pairs(valid, observations)
    trials = []
    for index, recipe in enumerate(candidates):
        model, detail = training.fit(rows, observations, fit, valid, recipe)
        predicted = training.scores(model, training.encode_rows(model.encoder, rows))
        loss = float(np.sum(np.logaddexp(0, -(predicted[p[:, 0]]-predicted[p[:, 1]]))*w)/count)
        require(np.isfinite(loss), 'Nonfinite inner tuning loss')
        trials.append(dict(index=index, recipe=recipe, inner_pairwise_loss=loss, **detail))
        print(f'neural trial {index+1}/{len(candidates)}: inner loss={loss:.6f}, '
              f'epochs={detail["selected_epochs"]}', flush=True)
    chosen = min(trials, key=lambda r: (r['inner_pairwise_loss'], r['index']))
    all_fit = [g for g in groups if g['split'] == 'fit']
    model, detail = training.fit(rows, observations, all_fit, [], chosen['recipe'],
                                 epochs=chosen['selected_epochs'])
    predicted = training.scores(model, training.encode_rows(model.encoder, rows))
    return document(model), predicted, dict(internal_split=split, trials=trials,
        selected_trial=chosen['index'], selected_epochs=chosen['selected_epochs'],
        selected_recipe=chosen['recipe'], fitting=detail,
        selection='minimum inner pairwise loss; no outer validation tuning')
