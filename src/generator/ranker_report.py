"""Report held-out single-example selection against the frozen existing selector."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import ROOT, sha256_file
from ..iteration.storage import atomic_write, file_lock, read_json, write_once_json
from .fewshot_ranker.dataset import load_dataset
from .fewshot_ranker.training import metrics, summarize_topk, topk_expectation
from .history_sources import digest, require


def selection_comparison(observations, predictions, baseline_ids):
    """Compare identical contexts; never replace an unobserved baseline choice."""
    require(len(observations) == len(predictions), 'Prediction count differs from observations')
    by_target = defaultdict(list)
    for row, score in zip(observations, predictions):
        require(np.isfinite(score) and row['z'] in (0, 1), 'Invalid prediction or binary label')
        by_target[row['target_id']].append(dict(row, logit=float(score)))
    details, excluded = [], Counter()
    for identity, rows in by_target.items():
        choices = baseline_ids[identity]
        lookup = {r['example_id']: r for r in rows}
        require(len(lookup) == len(rows), 'Duplicate observed candidate')
        if not choices:
            excluded['no_frozen_baseline_example'] += 1
            continue
        if choices[0] not in lookup:
            excluded['baseline_first_example_not_labeled'] += 1
            continue
        best = max(r['logit'] for r in rows)
        selected = [r for r in rows if r['logit'] == best]
        model_yield = float(np.mean([r['z'] for r in selected]))
        baseline_yield = lookup[choices[0]]['z']
        details.append(dict(target_id=identity, candidates=len(rows),
            baseline_example_id=choices[0], model_example_ids=[r['example_id'] for r in selected],
            baseline_z=baseline_yield, model_expected_z=model_yield,
            random_expected_z=float(np.mean([r['z'] for r in rows])),
            any_positive=int(any(r['z'] for r in rows)),
            delta=model_yield-baseline_yield))
    summary = dict(total_contexts=len(by_target), compared_contexts=len(details),
        excluded=dict(excluded), baseline='first ID in frozen baseline_ids, no fallback',
        candidate_scope='same observed labeled candidates for each context',
        ties='uniform expectation among equal model scores',
        scope='single-example labels; not the combined outcome of the online three-example selection')
    if details:
        deltas = np.asarray([r['delta'] for r in details])
        rng = np.random.default_rng(20260920)
        bootstrap = rng.choice(deltas, size=(2000, len(details)), replace=True).mean(axis=1)
        summary.update(
            model_top1_positive_rate=float(np.mean([r['model_expected_z'] for r in details])),
            baseline_top1_positive_rate=float(np.mean([r['baseline_z'] for r in details])),
            random_top1_positive_rate=float(np.mean([r['random_expected_z'] for r in details])),
            observed_oracle_positive_rate=float(np.mean([r['any_positive'] for r in details])),
            expected_wins=float(sum(r['model_expected_z'] for r in details if r['baseline_z'] == 0)),
            expected_losses=float(sum(1-r['model_expected_z'] for r in details if r['baseline_z'] == 1)),
            expected_ties=float(sum(r['model_expected_z'] if r['baseline_z'] else
                                    1-r['model_expected_z'] for r in details)),
            net_expected_positive_selections=float(deltas.sum()),
            positive_rate_difference=float(deltas.mean()),
            paired_context_bootstrap_95_interval=np.quantile(bootstrap, [.025, .975]).tolist(),
            bootstrap_resamples=2000, bootstrap_seed=20260920)
    return summary, details


def topk_comparison(observations, predictions, baseline_ids):
    """Only compare K choices when the frozen selector actually supplies K labels."""
    require(len(observations) == len(predictions), 'Prediction count differs from observations')
    groups = defaultdict(list)
    for row, score in zip(observations, predictions):
        groups[row['target_id']].append(dict(row, logit=float(score)))
    result = {}
    for k in (1, 3, 5):
        model, baseline, excluded, positive_contexts = [], [], Counter(), 0
        for identity, rows in groups.items():
            choices = baseline_ids[identity]
            lookup = {r['example_id']: r for r in rows}
            require(len(lookup) == len(rows) and len(set(choices)) == len(choices), 'Duplicate candidate')
            if len(choices) < k:
                excluded['baseline_has_fewer_than_k_choices'] += 1
                continue
            if any(cid not in lookup for cid in choices[:k]):
                excluded['baseline_topk_not_fully_labeled'] += 1
                continue
            model.append(topk_expectation([r['z'] for r in rows], [r['logit'] for r in rows], k))
            baseline.append(topk_expectation([lookup[cid]['z'] for cid in choices[:k]], [0]*k, k))
            positive_contexts += int(any(r['z'] for r in rows))
        result[str(k)] = dict(total_contexts=len(groups), compared_contexts=len(model),
            excluded=dict(excluded), model=summarize_topk(model, k, positive_contexts),
            baseline=summarize_topk(baseline, k, positive_contexts))
    return result


def render(value):
    lines = ['# Few-shot ranker: validation and existing-selector comparison', '',
        'The outer validation split was not used for fitting, vocabulary or epoch selection.',
        'Positive means the frozen Judge did not identify the single-example generation as AI.', '',
        '| Split | Observations | Contexts | ROC-AUC | PR-AUC (AP) | Within-context AUC |',
        '|---|---:|---:|---:|---:|---:|']
    def number(value):
        return 'N/A' if value is None else f'{value:.6f}'
    for split, result in value['splits'].items():
        m = result['model_metrics']
        lines.append(f'| {split} | {m["observations"]} | {m["contexts"]} | ' +
                     ' | '.join(number(m[k]) for k in
                                ('roc_auc', 'pr_auc_average_precision', 'group_auc_macro')) + ' |')
    lines += ['', '## Same-context Top1 selection', '',
        'Existing selection uses the first ID in each frozen baseline_ids record. It is not union recall rank.',
        'Both methods are evaluated on the same contexts; missing baseline labels are excluded without substitution.', '',
        '| Split | Compared / total | DNN positive | Existing positive | Random expected | Wins / losses / ties | Net positives |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split, result in value['splits'].items():
        s = result['selection']
        prefix = f'| {split} | {s["compared_contexts"]} / {s["total_contexts"]} | '
        if not s['compared_contexts']:
            lines.append(prefix + 'N/A | N/A | N/A | N/A | N/A |')
            continue
        rates = [f'{100*s[k]:.3f}%' for k in
                 ('model_top1_positive_rate', 'baseline_top1_positive_rate', 'random_top1_positive_rate')]
        counts = ' / '.join(f'{s[k]:g}' for k in ('expected_wins', 'expected_losses', 'expected_ties'))
        lines.append(prefix + ' | '.join(rates) + f' | {counts} | {s["net_expected_positive_selections"]:+g} |')
    for split, result in value['splits'].items():
        s = result['selection']
        lines += ['', f'{split} excluded contexts: {s["excluded"]}.']
        if s['compared_contexts']:
            low, high = s['paired_context_bootstrap_95_interval']
            lines.append(f'Paired positive-rate difference: {100*s["positive_rate_difference"]:+.3f} pp; '
                         f'95% context-bootstrap interval [{100*low:+.3f}, {100*high:+.3f}] pp.')
    lines += ['', 'Tied model scores use uniform expected selection, so counts may be fractional.',
        'Fit metrics are descriptive; validation metrics determine the held-out conclusion.',
        'These are frozen proxy labels for one example at a time. This does not measure joint use of three '
        'examples, live generator uplift, or satisfy a production promotion gate.',
        'The existing selector has no fitted comparable probability score; its direct comparison is paired Top1 outcome.',
        'No training, resampling, relabeling or model requests are performed by this report.']
    return '\n'.join(lines) + '\n'


def load_completed(training_output):
    """Reuse one bound artifact loader for reports and offline model comparisons."""
    output = Path(training_output).resolve()
    manifest = read_json(output / 'manifest.json')
    completed = read_json(output / 'completed.json')
    require(completed['manifest_sha256'] == digest(manifest) and all(
        sha256_file(output / name) == checksum for name, checksum in completed['artifacts'].items()),
        'Completed training artifacts changed')
    inventory = Path(manifest['inventory'])
    groups, observations, summary, binding = load_dataset(inventory)
    require(binding == manifest['dataset'], 'Training dataset differs from bound inventory')
    predicted = read_json(output / 'predictions.json')['rows']
    require(len(predicted) == len(observations) and all(
        {k: v for k, v in row.items() if k != 'logit'} == observation
        for row, observation in zip(predicted, observations)), 'Predictions differ from bound observations')
    baseline_ids = {identity: read_json(inventory / 'targets' / (identity + '.json'))['baseline_ids']
                    for identity in {r['target_id'] for r in observations}}
    return manifest, groups, observations, summary, predicted, baseline_ids


def report(training_output):
    output = Path(training_output).resolve()
    with file_lock(output / '.selection_report.lock', blocking=False):
        manifest, _, observations, _, predicted, baseline_ids = load_completed(output)
        splits = {}
        for split in ('fit', 'validation'):
            subset = [r for r in predicted if r['split'] == split]
            scores = [r['logit'] for r in subset]
            summary, details = selection_comparison(subset, scores, baseline_ids)
            splits[split] = dict(model_metrics=metrics(subset, scores), selection=summary, contexts=details)
        value = dict(schema=1, kind='frozen_single_example_selection_report',
            training_manifest_sha256=digest(manifest), completed_sha256=sha256_file(output / 'completed.json'),
            reporting_sources={name: sha256_file(ROOT / name) for name in
                ('src/generator/ranker_report.py', 'scripts/report_fewshot_ranker.py',
                 'src/generator/fewshot_ranker/dataset.py', 'src/generator/fewshot_ranker/training.py')},
            splits=splits, label_meaning='1 = frozen teacher did not identify single-example generation as AI',
            scope='offline single-example label ranking only; no production adoption')
        write_once_json(output / 'selection_report.json', value)
        atomic_write(output / 'SELECTION_REPORT.md', render(value))
        return {split: {k: v for k, v in result.items() if k != 'contexts'} for split, result in splits.items()}
