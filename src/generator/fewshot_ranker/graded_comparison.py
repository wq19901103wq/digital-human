"""Matched ordinal-versus-binary training using frozen features and recipes."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import platform
import time

import numpy as np
import torch
import xgboost as xgb
from threadpoolctl import threadpool_limits

from ...config import ROOT, sha256_file
from ...iteration.storage import atomic_write, file_lock, read_json, write_once_json
from ..history_sources import digest, require
from ..ranker_report import load_completed
from . import boosting, graded_labels, graded_metrics, logistic, neural_models, training
from .architecture_comparison import feature_rows
from .identity_comparison import completed, predictions, seal

METHODS = ('dnn_semantic', 'dnn_chat_pair', 'rankmixer_chat_pair', 'xgb_chat_pair')


@threadpool_limits.wrap(limits=1)
def fit_selected(groups, labels, rows, saved_model, saved_training):
    """Freeze the old selected recipe/epochs; outer validation never affects fit."""
    obs = [dict(r, z=r['grade']) for r in labels]
    fit_groups = [g for g in groups if g['split'] == 'fit']
    recipe = saved_training['selected_recipe']
    if saved_model['kind'] == 'fewshot_pairwise_xgboost_v1':
        encoder = logistic.encoder(rows, [i for g in fit_groups for i in g['indices']])
        x = encoder.transform(rows)
        model, detail = boosting.fit(x, obs, fit_groups, [], recipe, saved_training['selected_rounds'])
        value = deepcopy(saved_model)
        value.update(booster=json.loads(model.save_raw(raw_format='json').decode()),
            vocabulary={k: int(v) for k, v in encoder.vocabulary_.items()},
            feature_names=encoder.get_feature_names_out().tolist())
        predicted = model.predict(xgb.DMatrix(x, nthread=1), output_margin=True)
    else:
        model, detail = training.fit(rows, obs, fit_groups, [], recipe,
                                    epochs=saved_training['selected_epochs'])
        value = neural_models.document(model)
        predicted = training.scores(model, training.encode_rows(model.encoder, rows))
    value['feature_transform'] = saved_model['feature_transform']
    value['supervision'] = 'within_context_ordinal_3_gt_2_gt_1_gt_0; equal_pairs_skipped'
    pair_indices, _, _ = training.pairs(fit_groups, obs)
    detail['grade_pair_counts'] = dict(Counter(f'{obs[p]["z"]}>{obs[n]["z"]}' for p, n in pair_indices))
    return value, predicted, dict(recipe=recipe, fitting=detail,
        control_epochs=saved_training.get('selected_epochs'), control_rounds=saved_training.get('selected_rounds'),
        policy='reuse control recipe and stopping; fit contexts only; no outer validation tuning',
        loss='mean_over_contexts(mean_over_strict_pairs(softplus(-(score(high)-score(low)))))')


def render(value):
    counts = value['labels']['grade_counts']
    percent = lambda v: 'N/A' if v is None else f'{v:.2%}'
    number = lambda v: 'N/A' if v is None else f'{v:.4f}'
    lines = ['# Graded few-shot pairwise comparison', '',
        'Same frozen contexts, features, chronological split, model recipes, seeds and epochs/rounds. '
        'Only supervision changes: 3 > 2 > 1 > 0. Each context has equal total pair weight. '
        'Binary controls reuse their saved predictions. No new LLM requests.', '',
        '| Split | Grade 0 | Grade 1 | Grade 2 | Grade 3 |', '|---|---:|---:|---:|---:|']
    for split, c in counts.items():
        lines.append('| '+split+' | '+' | '.join(str(c.get(str(g), 0)) for g in range(4))+' |')
    for threshold, title in [('1', 'Original-positive Hit@K (grade ≥1)'),
                             ('2', 'Repeat-supported Hit@K (grade ≥2)'),
                             ('3', 'All-three-positive Hit@K (grade 3)')]:
        first = next(iter(value['methods'].values()))['before']['thresholds'][threshold]
        lines += ['', f'## {title}', '',
            f'Validation: {first["observations"]} rows / {first["contexts"]} contexts; '
            f'{first["positive"]} positive rows ({percent(first["positive_rate"])}); '
            f'{first["contexts_with_positive"]} contexts contain a qualifying example '
            f'({percent(first["positive_context_fraction"])} coverage).', '',
            '### AUC and PR-AUC', '',
            'PR-AUC uses Average Precision (AP); the random-score AP reference is the positive row fraction. '
            'Group AUC averages only contexts containing both classes.', '',
            '| Model | ROC-AUC before → after | PR-AUC (AP) before → after | Group AUC before → after |',
            '|---|---:|---:|---:|']
        for name, result in value['methods'].items():
            a, b = (result[side]['thresholds'][threshold] for side in ('before', 'after'))
            lines.append('| '+name+' | '+' | '.join(number(a[k])+' → '+number(b[k]) for k in
                ('roc_auc', 'pr_auc_average_precision', 'group_auc_macro'))+' |')
        for field, heading, denominator in (
            ('hit_rate', 'Hit@K over all contexts', first['contexts']),
            ('hit_rate_given_available', 'Hit@K given a qualifying example exists', first['contexts_with_positive'])):
            lines += ['', f'### {heading}', '', f'Denominator: {denominator} contexts.', '',
            '| Model | Top1 before → after | Top3 before → after | Top5 before → after |',
            '|---|---:|---:|---:|']
            for name, result in value['methods'].items():
                a, b = (result[side]['thresholds'][threshold]['topk'] for side in ('before', 'after'))
                lines.append('| '+name+' | '+' | '.join(
                    percent(a[k][field])+' → '+percent(b[k][field]) for k in ('1', '3', '5'))+' |')
    lines += ['', '## Grade ordering', '',
        '| Model | Mean Top1 grade before → after | NDCG@3 before → after | Pair accuracy before → after | Positive-only pair accuracy before → after | Grade 3 mean rank before → after |',
        '|---|---:|---:|---:|---:|---:|']
    for name, result in value['methods'].items():
        columns = []
        for field in (lambda r: r['mean_grade_at_k']['1'], lambda r: r['ndcg']['3'],
                      lambda r: r['ordered_pair_accuracy_macro'],
                      lambda r: r['positive_only_pair_accuracy_macro'],
                      lambda r: r['mean_rank_by_grade'].get('3', {}).get('mean_rank')):
            columns.append(number(field(result['before']))+' → '+number(field(result['after'])))
        lines.append('| '+name+' | '+' | '.join(columns)+' |')
    lines += ['', '## Interpretation', '',
        'Unconditional Hit@K includes all validation contexts and means at least one individually qualifying example '
        'among K; it does not measure joint K-shot generation. Score ties use uniform expectations. '
        'Conditional Hit@K excludes contexts without an example meeting that threshold; undefined metrics are N/A. '
        'Thresholds change both positive prevalence and available-context denominators, so compare models within a threshold. '
        'NDCG uses linear grade gain and excludes all-zero contexts. Lower mean rank is better.',
        'Grade 0 is the user-approved lowest level, observed 0/1; grades 1–3 are observed 1/3, 2/3, 3/3. '
        'Original negatives were not retested. Grades are ordinal supervision, not calibrated success probabilities. '
        'Repeats regenerated replies and judged them, so variation includes both generator and Judge.',
        'Validation has already been used for development. Results do not establish independent test or online gains. '
        'No model promotion. Paired context deltas and bootstrap intervals are saved in report.json.', '',
        f'Local execution: {value["seconds"]:.1f}s; new LLM requests: 0.']
    return '\n'.join(lines)+'\n'


def report(output):
    """Refresh a derived view from sealed metrics without fitting or modifying evidence."""
    output = Path(output).resolve()
    with file_lock(output/'.report_queue.lock', blocking=False):
        binding = read_json(output/'manifest.json')
        require(binding.get('kind') == 'cached_fewshot_graded_comparison' and completed(output, binding),
                'Graded comparison must be complete before reporting')
        value = read_json(output/'report.json')
        require(value['status'] == 'complete' and value['manifest_sha256'] == digest(binding),
                'Graded report does not match completed comparison')
        source_sha = sha256_file(output/'report.json')
        path = output/'THRESHOLD_REPORT.md'
        atomic_write(path, render(value)+'\nDerived from sealed report.json SHA256: '+source_sha+'\n')
        return dict(status='complete', report=str(path), source_report_sha256=source_sha,
                    new_model_requests=0, new_training_runs=0)


def run(source, reference, stability, output):
    source, reference, stability, output = map(lambda p: Path(p).resolve(), (source, reference, stability, output))
    require(output not in (source, reference, stability), 'Use a new graded comparison output')
    started = time.monotonic()
    with file_lock(output/'.run.lock', blocking=False):
        manifest, groups, obs, summary, _, _ = load_completed(source)
        matrix = read_json(source/'feature_matrix.json')
        require(matrix['manifest_sha256'] == digest(manifest) and len(matrix['rows']) == len(obs),
                'Feature matrix differs from frozen observations')
        ref = read_json(reference/'manifest.json')
        require(ref['kind'] == 'cached_fewshot_architecture_comparison'
                and ref['source_manifest_sha256'] == digest(manifest)
                and ref['source_completed_sha256'] == sha256_file(source/'completed.json')
                and ref['dataset'] == manifest['dataset'] and completed(reference, ref),
                'Architecture controls are not bound to this source')
        labels, label_binding, watched = graded_labels.load(manifest['inventory'], stability,
                                                           manifest['dataset'], groups, obs)
        source_files = [*Path(__file__).parent.glob('*.py'), ROOT/'src/generator/ranker_report.py',
                        ROOT/'src/judge/embedding.py', ROOT/'scripts/train_fewshot_ranker.py']
        watched.update({p: sha256_file(p) for p in source_files})
        for directory in (source, reference):
            for name in ('manifest.json', 'completed.json'):
                watched[directory/name] = sha256_file(directory/name)
            watched.update({directory/name: sha for name, sha in read_json(directory/'completed.json')['artifacts'].items()})
        binding = dict(schema=1, kind='cached_fewshot_graded_comparison', source=str(source),
            source_manifest_sha256=digest(manifest), source_completed_sha256=sha256_file(source/'completed.json'),
            reference=str(reference), reference_completed_sha256=sha256_file(reference/'completed.json'),
            dataset=manifest['dataset'], labels=label_binding, methods=list(METHODS),
            policy='fixed control recipes and stopping; graded context-balanced pairs; no outer tuning',
            runtime={str(p.relative_to(ROOT)): watched[p] for p in source_files},
            libraries=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, xgboost=xgb.__version__))
        write_once_json(output/'manifest.json', binding)
        if completed(output, binding):
            return read_json(output/'report.json')
        write_once_json(output/'labels.json', dict(binding=label_binding, rows=labels))
        vi = [i for i, r in enumerate(obs) if r['split'] == 'validation']
        validation = [labels[i] for i in vi]
        methods, artifacts = {}, ['labels.json']
        print('Loaded frozen features and completed repeat labels; zero LLM requests', flush=True)
        for name in METHODS:
            control = reference/name
            saved_model, saved_training = (read_json(control/f'{f}.json') for f in ('model', 'training'))
            transform = saved_model['feature_transform']
            rows = feature_rows(matrix['rows'], transform['chat_id_pair'])
            directory = output/name
            method_binding = dict(parent_sha256=digest(binding), method=name,
                control_model_sha256=sha256_file(control/'model.json'),
                control_training_sha256=sha256_file(control/'training.json'))
            write_once_json(directory/'manifest.json', method_binding)
            if not completed(directory, method_binding):
                print(f'Fitting graded {name} with saved recipe/stopping', flush=True)
                model, scores, detail = fit_selected(groups, labels, rows, saved_model, saved_training)
                write_once_json(directory/'model.json', model)
                write_once_json(directory/'training.json', detail)
                write_once_json(directory/'predictions.json', dict(rows=[dict(r, logit=float(s)) for r, s in zip(obs, scores)]))
                seal(directory, method_binding, ['model.json', 'training.json', 'predictions.json'])
            before, after = (graded_metrics.evaluate(validation, np.asarray(predictions(p, obs))[vi])
                             for p in (control, directory))
            methods[name] = dict(before=before, after=after, deltas=graded_metrics.compare(before, after))
            artifacts += [f'{name}/{f}' for f in ('manifest.json', 'completed.json', 'model.json', 'training.json', 'predictions.json')]
            print(f'Completed graded {name}', flush=True)
        require(all(sha256_file(p) == sha for p, sha in watched.items()), 'Comparison inputs changed during fitting')
        value = dict(status='complete', manifest_sha256=digest(binding), samples=summary,
            labels=label_binding, methods=methods, seconds=time.monotonic()-started, new_model_requests=0,
            scope='offline development label comparison; no production adoption')
        write_once_json(output/'report.json', value)
        atomic_write(output/'REPORT.md', render(value))
        seal(output, binding, [*artifacts, 'report.json', 'REPORT.md'])
        return value
