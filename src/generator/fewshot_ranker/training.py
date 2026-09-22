"""Context-balanced pairwise fitting and held-out ranking evaluation."""
from collections import Counter, defaultdict
import copy
import math

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

from ...judge.embedding import fit_encoder
from ...iteration.storage import write_json, write_once_json, atomic_write
from ..history_sources import require
from .features import combine, context_local, reply_local
from .neural_models import create, document

RECIPE = dict(embedding_dim=8, hidden=[128, 64], normalization='layer', activation='silu',
              dropout=.1, cross_layers=0, learning_rate=.001, weight_decay=.0001,
              batch_size=256, max_epochs=50, patience=5, seed=20260920, inner_fraction=.2)


def assemble(groups, refs, values):
    rows = []
    for group in groups:
        target = group['target']
        for entry in group['candidates']:
            example = entry['example']
            t, e, r = (values[key] for key in refs[(group['target_id'], example['id'])])
            rows.append(combine(target, example, {**t, **context_local(target)},
                {**e, **context_local(example, example=True)}, {**r, **reply_local(example)}))
    return rows


def inner_split(groups, fraction):
    fitting = sorted((g for g in groups if g['split'] == 'fit'),
                     key=lambda g: (g['target']['input_cutoff']['timestamp'], g['target_id']))
    require(len(fitting) >= 5, 'Too few contexts for chronological early stopping')
    cutoff = fitting[max(1, int(len(fitting)*(1-fraction)))]['target']['input_cutoff']['timestamp']
    validation = [g for g in fitting if g['target']['input_cutoff']['timestamp'] >= cutoff]
    answer_ids = {mid for g in validation for mid in g['target']['reply_message_ids']}
    train, purged = [], []
    for group in fitting:
        if group in validation:
            continue
        materials = [group['target'], *[c['example'] for c in group['candidates']]]
        if any(m['source_span']['end_timestamp'] >= cutoff or answer_ids.intersection(
                m['context_message_ids'] + m['reply_message_ids']) for m in materials):
            purged.append(group['target_id'])
        else:
            train.append(group)
    require(train and validation, 'Empty inner temporal split')
    return train, validation, dict(cutoff=cutoff, fitting_contexts=len(train),
        validation_contexts=len(validation), purged_contexts=purged,
        rule='earlier_fit_material_only; no_outer_validation_tuning')


def pairs(groups, observations):
    """All strict within-context preferences; equal labels provide no pair."""
    result, weights, mixed = [], [], 0
    for group in groups:
        ordered = [(p, n) for p in group['indices'] for n in group['indices']
                   if observations[p]['z'] > observations[n]['z']]
        if not ordered:
            continue
        mixed += 1
        count = len(ordered)
        result.extend(ordered)
        weights.extend([1/count]*count)
    require(mixed > 0, 'No mixed-label contexts available for pairwise fitting or early stopping')
    return np.asarray(result, dtype=np.int64), np.asarray(weights, dtype=np.float32), mixed


def encode_rows(encoder, rows):
    return np.asarray([[encoder['vocabularies'][key].get(row[key], 0) for key in encoder['fields']]
                       for row in rows], dtype=np.int64)


def scores(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model.score(torch.as_tensor(x[start:start+512], dtype=torch.long)).numpy()
                               for start in range(0, len(x), 512)])


def fit(rows, observations, train_groups, validation_groups, recipe, *, epochs=None):
    indices = [i for g in train_groups for i in g['indices']]
    encoder = fit_encoder([(rows[i], rows[i]) for i in indices])
    x = torch.as_tensor(encode_rows(encoder, rows), dtype=torch.long)
    pair_indices, weights, mixed = pairs(train_groups, observations)
    validation = pairs(validation_groups, observations) if validation_groups else None
    torch.set_num_threads(1)
    torch.manual_seed(recipe['seed'])
    torch.use_deterministic_algorithms(True)
    model = create(encoder, recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    rng = np.random.default_rng(recipe['seed'])
    # Sum of weights is the number of mixed contexts. Fixed scaling makes the
    # shuffled minibatch estimator unbiased for the mean of per-context losses.
    weight_tensor = torch.from_numpy(weights * len(weights)/mixed)
    best_loss, best_epoch, best_state, stale, history = math.inf, 0, None, 0, []
    for epoch in range(1, (epochs or recipe['max_epochs'])+1):
        model.train()
        order = rng.permutation(len(pair_indices))
        loss_sum = 0.
        for offset in range(0, len(order), recipe['batch_size']):
            ix = order[offset:offset+recipe['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            difference = model(x[pair_indices[ix]])
            losses = nn.functional.softplus(-difference)
            loss = (losses * weight_tensor[ix]).mean()
            require(bool(torch.isfinite(loss)), 'Nonfinite ranker training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += loss.item()*len(ix)
        item = dict(epoch=epoch, pairwise_train_loss=loss_sum/len(order))
        if validation:
            p, w, count = validation
            predicted = scores(model, x.numpy())
            val_loss = float(np.sum(np.logaddexp(0, -(predicted[p[:, 0]]-predicted[p[:, 1]]))*w)/count)
            item['inner_validation_pairwise_loss'] = val_loss
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(item)
        if validation and stale >= recipe['patience']:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    return model, dict(epochs_run=epoch, selected_epochs=best_epoch, mixed_contexts=mixed,
                       pairs=len(pair_indices), history=history, fields=len(encoder['fields']),
                       parameters=sum(p.numel() for p in model.parameters()))


def topk_expectation(labels, scores, k):
    """Uniform sampling at the cutoff tie; labels never break score ties."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    require(k > 0 and len(labels) == len(scores) > 0 and np.isfinite(scores).all()
            and np.isin(labels, [0, 1]).all(), 'Invalid TopK inputs')
    count = min(k, len(labels))
    cutoff = np.sort(scores)[-count]
    above, tied = scores > cutoff, scores == cutoff
    certain = int(labels[above].sum())
    take, size, positive = count-int(above.sum()), int(tied.sum()), int(labels[tied].sum())
    hit = 1. if certain else 1-math.comb(size-positive, take)/math.comb(size, take)
    return dict(hit_probability=hit, positive_count=certain+take*positive/size,
                selected_count=count)


def summarize_topk(values, k, positive_contexts):
    hits = sum(v['hit_probability'] for v in values)
    count = sum(v['positive_count'] for v in values)
    return dict(contexts=len(values), contexts_with_positive=positive_contexts,
        contexts_with_fewer_than_k_candidates=sum(v['selected_count'] < k for v in values),
        expected_hit_contexts=hits, hit_rate=hits/len(values) if values else None,
        hit_rate_given_available=hits/positive_contexts if positive_contexts else None,
        mean_positive_count=count/len(values) if values else None,
        mean_positive_count_given_available=count/positive_contexts if positive_contexts else None)


def metrics(observations, predictions):
    y, s = np.asarray([r['z'] for r in observations]), np.asarray(predictions, dtype=float)
    require(len(y) == len(s) and len(y) > 0 and bool(np.isfinite(s).all())
            and bool(np.isin(y, [0, 1]).all()), 'Invalid evaluation predictions')
    groups = defaultdict(list)
    for i, row in enumerate(observations):
        groups[row['target_id']].append(i)
    aucs, pair_wins, pair_count, top, ndcg1, ndcg3 = [], 0., 0, [], [], []
    positive_groups, all_positive_groups, positive_counts = 0, 0, Counter()
    topk = {k: [] for k in (1, 3, 5)}
    for indices in groups.values():
        labels, score = y[indices], s[indices]
        for k, values in topk.items():
            values.append(topk_expectation(labels, score, k))
        positive_counts[int(labels.sum())] += 1
        all_positive_groups += int(bool(np.all(labels == 1)))
        top.append(float(labels[score == score.max()].mean()))  # Expected value for ties.
        if labels.sum():
            positive_groups += 1
            if len(indices) == 1:
                ndcg1.append(1.)
                ndcg3.append(1.)
            else:
                ndcg1.append(float(ndcg_score(labels[None, :], score[None, :], k=1)))
                ndcg3.append(float(ndcg_score(labels[None, :], score[None, :], k=3)))
        if 0 < labels.sum() < len(labels):
            differences = score[labels == 1][:, None]-score[labels == 0][None, :]
            wins = float(np.sum(differences > 0) + .5*np.sum(differences == 0))
            aucs.append(wins/differences.size)
            pair_wins += wins
            pair_count += differences.size
    both = len(set(y)) == 2
    return dict(observations=len(y), contexts=len(groups), positive=int(y.sum()),
        positive_rate=float(y.mean()), roc_auc=float(roc_auc_score(y, s)) if both else None,
        pr_auc_average_precision=float(average_precision_score(y, s)) if y.sum() else None,
        group_auc_macro=float(np.mean(aucs)) if aucs else None,
        pair_accuracy=pair_wins/pair_count if pair_count else None, directed_pairs=pair_count,
        mixed_contexts=len(aucs), top1_positive_yield=float(np.mean(top)),
        contexts_with_positive=positive_groups, contexts_without_positive=len(groups)-positive_groups,
        contexts_all_positive=all_positive_groups,
        positive_context_fraction=positive_groups/len(groups),
        positive_count_histogram={str(k): v for k, v in sorted(positive_counts.items())},
        candidate_count_histogram={str(k): v for k, v in sorted(Counter(map(len, groups.values())).items())},
        topk={str(k): summarize_topk(v, k, positive_groups) for k, v in topk.items()},
        top1_positive_given_available=float(sum(top)/positive_groups) if positive_groups else None,
        ndcg_at_1=float(np.mean(ndcg1)) if ndcg1 else None,
        ndcg_at_3=float(np.mean(ndcg3)) if ndcg3 else None,
        ndcg_contexts=positive_groups, ndcg_policy='exclude_zero_positive_contexts; average_ties',
        ranking_ties='expected_uniform_selection')


def train(groups, observations, rows, output, recipe=None):
    recipe = copy.deepcopy(recipe or RECIPE)
    write_once_json(output / 'recipe.json', recipe)
    inner_train, inner_valid, split_report = inner_split(groups, recipe['inner_fraction'])
    _, selection = fit(rows, observations, inner_train, inner_valid, recipe)
    all_fit = [g for g in groups if g['split'] == 'fit']
    model, fitting = fit(rows, observations, all_fit, [], recipe, epochs=selection['selected_epochs'])
    document_value = document(model)
    write_json(output / 'model.json', document_value)
    x = encode_rows(model.encoder, rows)
    predictions = scores(model, x)
    results = {}
    for split in ('fit', 'validation'):
        indices = [i for i, row in enumerate(observations) if row['split'] == split]
        subset = [observations[i] for i in indices]
        results[split] = dict(dnn=metrics(subset, predictions[indices]),
            recall_order=metrics(subset, [-r['recall_rank'] for r in subset]),
            random_expected=metrics(subset, np.zeros(len(subset))))
    write_json(output / 'predictions.json', dict(rows=[dict(**r, logit=float(p))
                                                        for r, p in zip(observations, predictions)]))
    value = dict(status='complete', recipe=recipe, internal_split=split_report,
        epoch_selection=selection, fitting=fitting, metrics=results,
        label_meaning='1 = frozen teacher did not identify single-example generation as AI',
        scope='offline label ranking; not independent judge evaluation or generator uplift',
        model_scores='uncalibrated logits; pairwise sigmoid(score(A)-score(B))',
        feature_vocabulary='fit only; outer validation never used for early stopping or vocabulary',
        model_path=str(output / 'model.json'))
    write_json(output / 'report.json', value)
    lines = ['# Few-shot ranker offline training', '',
        'Validation uses the original chronological, context-grouped split. Labels are frozen Judge proxy supervision.', '',
        '| Split / method | N | ROC-AUC | PR-AUC (AP) | Group AUC | Top1 positive | NDCG@3 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split, methods in results.items():
        for name, m in methods.items():
            def number(key):
                return 'N/A' if m[key] is None else f'{m[key]:.6f}'
            lines.append(f'| {split} / {name} | {m["observations"]} | ' + ' | '.join(number(k) for k in
                ('roc_auc', 'pr_auc_average_precision', 'group_auc_macro', 'top1_positive_yield', 'ndcg_at_3')) + ' |')
    lines += ['', f'Selected epochs: {selection["selected_epochs"]}. All fields use 8-dimensional embeddings.',
              'Group AUC uses mixed-label contexts. NDCG excludes zero-positive contexts; Top1 includes every context.',
              'Recall baseline orders only the observed candidates, not the entire recall pool. No production change.']
    atomic_write(output / 'REPORT.md', '\n'.join(lines)+'\n')
    return value
