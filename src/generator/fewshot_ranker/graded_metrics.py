"""Tie-aware grade ranking metrics, including all-zero contexts in Hit@K."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import ndcg_score

from ..history_sources import require
from .training import metrics, topk_expectation


def evaluate(labels, scores):
    scores = np.asarray(scores, dtype=float)
    require(len(labels) == len(scores) > 0 and np.isfinite(scores).all()
            and all(r['grade'] in (0, 1, 2, 3) for r in labels), 'Invalid graded metric inputs')
    groups = defaultdict(list)
    for i, row in enumerate(labels):
        groups[row['target_id']].append(i)
    thresholds = {str(g): metrics([dict(r, z=int(r['grade'] >= g)) for r in labels], scores)
                  for g in (1, 2, 3)}
    contexts, ranks, ndcg = [], defaultdict(list), {k: [] for k in (1, 3, 5)}
    for target, indices in groups.items():
        grades = np.asarray([labels[i]['grade'] for i in indices])
        s = scores[indices]
        wins, count, positive_wins, positive_count = 0., 0, 0., 0
        for j, grade in enumerate(grades):
            ranks[str(grade)].append(float(1+sum(s > s[j])+.5*(sum(s == s[j])-1)))
            lower = s[grades < grade]
            wins += float(sum(s[j] > lower)+.5*sum(s[j] == lower))
            count += len(lower)
            positive_lower = s[(grades > 0) & (grades < grade)]
            positive_wins += float(sum(s[j] > positive_lower)+.5*sum(s[j] == positive_lower))
            positive_count += len(positive_lower)
        top = {}
        for k in (1, 3, 5):
            # A grade equals the sum of its three threshold indicators.
            levels = [topk_expectation((grades >= g).astype(int), s, k) for g in (1, 2, 3)]
            top[str(k)] = dict(mean_grade=sum(v['positive_count'] for v in levels)/levels[0]['selected_count'],
                hit_by_grade={str(g): levels[g-1]['hit_probability'] for g in (1, 2, 3)})
            if grades.max() > 0:
                ndcg[k].append(1. if len(grades) == 1 else float(ndcg_score(grades[None, :], s[None, :], k=k)))
        contexts.append(dict(target_id=target, topk=top, ordered_pairs=count,
            pair_accuracy=wins/count if count else None, positive_only_pairs=positive_count,
            positive_only_pair_accuracy=positive_wins/positive_count if positive_count else None))
    pair_values = [r['pair_accuracy'] for r in contexts if r['ordered_pairs']]
    positive_pair_values = [r['positive_only_pair_accuracy'] for r in contexts if r['positive_only_pairs']]
    return dict(thresholds=thresholds, contexts=contexts,
        ordered_pair_accuracy_macro=float(np.mean(pair_values)) if pair_values else None,
        contexts_with_ordered_pairs=len(pair_values),
        positive_only_pair_accuracy_macro=float(np.mean(positive_pair_values)) if positive_pair_values else None,
        contexts_with_positive_grade_pairs=len(positive_pair_values),
        mean_rank_by_grade={g: dict(samples=len(v), mean_rank=float(np.mean(v))) for g, v in sorted(ranks.items())},
        mean_grade_at_k={str(k): float(np.mean([r['topk'][str(k)]['mean_grade'] for r in contexts])) for k in ndcg},
        ndcg={str(k): float(np.mean(v)) if v else None for k, v in ndcg.items()},
        ndcg_policy='linear grade gain; exclude all-zero contexts; average ties',
        ranking_ties='uniform expected selection and mean ranks')


def compare(before, after):
    """Paired context bootstrap, descriptive development uncertainty only."""
    require([r['target_id'] for r in before['contexts']] == [r['target_id'] for r in after['contexts']],
            'Graded comparisons require identical contexts')
    deltas = {}
    rng = np.random.default_rng(20260920)
    for k in ('1', '3', '5'):
        fields = {'mean_grade': lambda r: r['topk'][k]['mean_grade']}
        fields.update({f'hit_grade_ge_{g}': lambda r, g=g: r['topk'][k]['hit_by_grade'][g] for g in ('1', '2', '3')})
        deltas[k] = {}
        for name, field in fields.items():
            d = np.asarray([field(b)-field(a) for a, b in zip(before['contexts'], after['contexts'])])
            sampled = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
            deltas[k][name] = dict(mean_delta=float(d.mean()), sum_context_delta=float(d.sum()),
                paired_context_bootstrap_95_interval=np.quantile(sampled, [.025, .975]).tolist())
    return deltas
