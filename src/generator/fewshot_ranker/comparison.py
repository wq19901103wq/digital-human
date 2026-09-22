"""Reusable offline model comparison on an immutable completed feature matrix."""
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sklearn

from ...config import ROOT, sha256_file
from ...iteration.storage import atomic_write, file_lock, read_json, write_once_json
from ..history_sources import digest, require
from ..ranker_report import load_completed, selection_comparison, topk_comparison
from . import crosses, logistic
from .training import metrics


def evaluate(observations, scores, baseline_ids):
    results = {}
    for split in ('fit', 'validation'):
        indices = [i for i, r in enumerate(observations) if r['split'] == split]
        subset = [observations[i] for i in indices]
        predicted = [float(scores[i]) for i in indices]
        comparison, contexts = selection_comparison(subset, predicted, baseline_ids)
        results[split] = dict(metrics=metrics(subset, predicted), selection=comparison, contexts=contexts,
            topk_comparison=topk_comparison(subset, predicted, baseline_ids))
    return results


def render(value):
    lines = ['# Few-shot LR comparison (cached features)', '',
        'Same original chronological fit/validation groups and frozen single-example labels. '
        'C is selected only on a chronological inner split of fit. No LLM requests.', '',
        '| Method | ROC-AUC | PR-AUC (AP) | Within-context AUC | Top1 positive | Wins / losses | Net vs existing |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for name, result in value['methods'].items():
        m, s = result['validation']['metrics'], result['validation']['selection']
        selected = ('N/A | N/A | N/A' if not s['compared_contexts'] else
            f'{s["model_top1_positive_rate"]:.2%} | {s["expected_wins"]:g} / '
            f'{s["expected_losses"]:g} | {s["net_expected_positive_selections"]:+g}')
        lines.append(f'| {name} | ' + ' | '.join(number(m[k]) for k in
            ('roc_auc', 'pr_auc_average_precision', 'group_auc_macro')) + f' | {selected} |')
    original = value['methods']['dnn_saved']['validation']
    m, s = original['metrics'], original['selection']
    lines += ['', f'Existing selector Top1 positive: {s.get("baseline_top1_positive_rate", 0):.2%}. '
        f'Compared {s["compared_contexts"]}/{s["total_contexts"]} contexts; excluded: {s["excluded"]}.', '',
        f'Validation: {m["contexts"]} contexts / {m["observations"]} observations / {m["positive"]} positives.',
        f'At least one positive: {m["contexts_with_positive"]} ({m["positive_context_fraction"]:.2%}); '
        f'zero positives: {m["contexts_without_positive"]}; all positive: {m["contexts_all_positive"]}.', '',
        '| Positive candidates per context | Contexts |', '|---|---:|']
    for count, contexts in m['positive_count_histogram'].items():
        lines.append(f'| {count} | {contexts} |')
    lines += ['', '## Top1 / Top3 / Top5', '',
        'Hit@K means at least one positive among the first K individually labeled examples. '
        'Ties use uniform expected selection. All-negative contexts stay in the overall denominator.',
        f'Observed candidates per context: {m["candidate_count_histogram"]}.', '',
        '| Method | K | Contexts | Expected hit contexts | Hit@K | Hit@K given any positive | Mean positive count@K |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def topk_row(name, k, item):
        return (f'| {name} | {k} | {item["contexts"]} | {item["expected_hit_contexts"]:g} | ' +
            ' | '.join(number(item[key]) for key in
                ('hit_rate', 'hit_rate_given_available', 'mean_positive_count')) + ' |')
    for name, result in value['methods'].items():
        for k, item in result['validation']['metrics']['topk'].items():
            lines.append(topk_row(name, k, item))
    for k, item in original['topk_comparison'].items():
        lines.append(topk_row('existing_selector', k, item['baseline']))
    for k, item in value['random_expected']['validation']['topk'].items():
        lines.append(topk_row('random_expected', k, item))
    lines += ['', 'Frozen selector comparison (same eligible contexts, no fill-in choices):']
    for k, item in original['topk_comparison'].items():
        lines.append(f'- K={k}: {item["compared_contexts"]}/{item["total_contexts"]}; '
                     f'excluded: {item["excluded"]}.')
    lines.append('When fewer than K candidates exist, model metrics use all available candidates; '
                 'frozen-selector comparison requires all K choices and labels. Per-method paired subsets are in report.json.')
    lines += ['', '## LR tuning', '']
    for name, detail in value['training'].items():
        lines.append(f'- {name}: selected C={detail["selected_C"]:g}; '
            f'{detail["fields"]} categorical fields; {detail["one_hot_dimensions"]} one-hot dimensions.')
        lines.append('  Inner pairwise losses: ' + ', '.join(
            f'C={r["C"]:g}: {r["inner_pairwise_loss"]:.6f}' for r in detail['trials']))
    lines += ['', '## Added categorical crosses', '', *['- '+name for name in value['added_fields']], '',
        'Positive means frozen Judge did not recognize a reply generated with one example as AI. '
        'This is offline selection of observed examples, not evidence for jointly using multiple examples online.',
        'All-negative contexts remain in overall Top1 results. Conditional Top1 is separately reported.',
        'LR scores are uncalibrated ranking logits. Ties use uniform expected selection.',
        'No production version is changed.']
    return '\n'.join(lines)+'\n'


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source != output, 'Offline comparison requires a separate output directory')
    started = time.monotonic()
    with file_lock(output / '.run.lock', blocking=False):
        manifest, groups, observations, summary, dnn, baseline_ids = load_completed(source)
        matrix = read_json(source / 'feature_matrix.json')
        rows = matrix['rows']
        require(matrix['manifest_sha256'] == digest(manifest) and len(rows) == len(observations),
                'Feature matrix differs from bound observations')
        expanded = [crosses.expand(row) for row in rows]
        source_names = [*Path(__file__).parent.glob('*.py'), ROOT/'src/generator/ranker_report.py',
                        ROOT/'scripts/train_fewshot_ranker.py']
        sources = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_names}
        binding = dict(schema=1, kind='cached_fewshot_lr_comparison', source=str(source),
            source_manifest_sha256=digest(manifest), source_completed_sha256=sha256_file(source/'completed.json'),
            dataset=manifest['dataset'], recipe=logistic.RECIPE, crosses=crosses.VERSION, runtime=sources,
            libraries=dict(python=platform.python_version(), numpy=np.__version__,
                           sklearn=sklearn.__version__, scipy=scipy.__version__))
        write_once_json(output/'manifest.json', binding)
        completed = read_json(output/'completed.json', default=None)
        if completed:
            require(completed['manifest_sha256'] == digest(binding) and all(
                sha256_file(output/name) == v for name, v in completed['artifacts'].items()),
                'Completed offline comparison artifacts changed')
            return read_json(output/'report.json')
        methods = {'dnn_saved': evaluate(observations, [r['logit'] for r in dnn], baseline_ids)}
        random_expected = {}
        for split in ('fit', 'validation'):
            subset = [r for r in observations if r['split'] == split]
            random_expected[split] = metrics(subset, np.zeros(len(subset)))
        training, artifacts = {}, []
        for name, features in (('lr_existing', rows), ('lr_expanded', expanded)):
            directory = output/name
            model, predicted, detail = logistic.train(groups, observations, features)
            write_once_json(directory/'model.json', model)
            write_once_json(directory/'predictions.json', dict(rows=[dict(r, logit=float(p))
                for r, p in zip(observations, predicted)]))
            training[name] = detail
            methods[name] = evaluate(observations, predicted, baseline_ids)
            artifacts += [f'{name}/model.json', f'{name}/predictions.json']
        require(all(sha256_file(ROOT/name) == v for name, v in sources.items()),
                'Comparison code changed during fitting')
        value = dict(status='complete', manifest_sha256=digest(binding), samples=summary,
            methods=methods, random_expected=random_expected, training=training,
            added_fields=sorted(set(expanded[0])-set(rows[0])),
            seconds=time.monotonic()-started, new_model_requests=0,
            scope='offline frozen-label ranking only; no production adoption')
        write_once_json(output/'report.json', value)
        atomic_write(output/'REPORT.md', render(value))
        artifacts += ['report.json', 'REPORT.md']
        write_once_json(output/'completed.json', dict(manifest_sha256=digest(binding),
            artifacts={name: sha256_file(output/name) for name in artifacts}))
        return value
