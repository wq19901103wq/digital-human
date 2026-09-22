"""Compare categorical neural and boosted rankers on frozen cached features."""
from pathlib import Path
import platform
import time

import numpy as np
import torch
import xgboost

from ...config import ROOT, sha256_file
from ...iteration.storage import atomic_write, file_lock, read_json, write_once_json
from ..history_sources import digest, require
from ..ranker_report import load_completed
from . import boosting, crosses, identity_crosses, neural_sweep
from .comparison import evaluate
from .identity_comparison import completed, predictions, seal


NEURAL_RECIPES = {
    'dnn_semantic': neural_sweep.recipes(0),
    'dnn_chat_pair': neural_sweep.recipes(0),
    'rankmixer_chat_pair': [dict(recipe, architecture='rankmixer', token_width=width,
        mixer_layers=layers, ffn_expansion=2)
        for recipe, width, layers in zip(neural_sweep.recipes(0), [32, 64, 32], [1, 1, 2])],
}
XGB_RECIPES = boosting.recipes()
CONTROLS = ('lr_semantic_tuned', 'lr_response_need', 'lr_behavior_id', 'dnn_behavior_id')


def feature_rows(raw, chat_pair):
    rows = [crosses.expand(row) for row in raw]
    if chat_pair:
        for row in rows:
            row['id_cross.chat_id'] = identity_crosses.pair(row, 'target.chat_id', 'example_context.chat_id')
    return rows


def render(value):
    original = value['methods']['dnn_saved']['validation']
    m = original['metrics']
    lines = ['# Few-shot architecture comparison', '',
        f'Validation: {m["observations"]} rows / {m["contexts"]} contexts / {m["positive"]} positives. '
        f'{m["contexts_with_positive"]} contexts have positives, {m["contexts_without_positive"]} have none.', '',
        '| Method | AUC | AP | Group AUC | Hit@1 | Hit@3 | Hit@5 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    number = lambda v: 'N/A' if v is None else f'{v:.4f}'
    percent = lambda v: 'N/A' if v is None else f'{v:.2%}'
    for name, result in value['methods'].items():
        metrics = result['validation']['metrics']
        cells = [number(metrics[k]) for k in ('roc_auc', 'pr_auc_average_precision', 'group_auc_macro')]
        cells += [percent(metrics['topk'][k]['hit_rate']) for k in ('1', '3', '5')]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    baseline = original['topk_comparison']
    lines += ['| existing_selector | — | — | — | ' + ' | '.join(
        percent(baseline[k]['baseline']['hit_rate']) for k in ('1', '3', '5')) + ' |', '',
        'Hit@K counts contexts with at least one individually positive example among K. '
        'All-negative contexts remain in the denominator; ties use uniform expected selection. '
        'This does not evaluate joint K-shot generation.', '', '## Inner-fit tuning', '']
    for name, detail in value['training'].items():
        lines.append(f'- {name}: selected trial {detail["selected_trial"]}; '
                     f'inner loss {min(t["inner_pairwise_loss"] for t in detail["trials"]):.6f}.')
        for trial in detail['trials']:
            r = trial['recipe']
            config = (f'depth={r["max_depth"]}, child={r["min_child_weight"]}, lambda={r["reg_lambda"]}, '
                      f'rounds={trial["selected_rounds"]}' if 'max_depth' in r else
                      f'hidden={r["hidden"]}, norm={r["normalization"]}, dropout={r["dropout"]}, '
                      f'decay={r["weight_decay"]}, epochs={trial["selected_epochs"]}' +
                      (f', token width={r["token_width"]}, mixer layers={r["mixer_layers"]}'
                       if 'token_width' in r else ''))
            lines.append(f'  - {trial["index"]}: {config}; loss={trial["inner_pairwise_loss"]:.6f}.')
    lines += ['', f'Inner-fit selected method: **{value["inner_selected_method"]}**.', '',
        '## Scope', '',
        'All neural fields are categorical, embedding size 8, with no numeric bypass. '
        'Chat-pair models add the explicit target-chat × example-chat categorical ID. '
        'RankMixer is a small dense adaptation using four domain tokens, token permutation, '
        'per-token FFNs and a DNN head; no sparse MoE. XGBoost uses fit-only categorical one-hot inputs.',
        'All methods score each example separately and optimize within-context logit differences. '
        'Hyperparameters, vocabulary and stopping use chronological inner fit only. '
        'Saved controls are reused without refitting. No new feature extraction or LLM requests.',
        'The outer validation set has been used before: these are development comparisons, '
        'not independent test results or production adoption. Original single-draw labels are unchanged. '
        'Per-context outcomes, conditional TopK and uncertainty are in report.json.']
    return '\n'.join(lines)+'\n'


def run(source, reference, output):
    source, reference, output = (Path(p).resolve() for p in (source, reference, output))
    require(output not in (source, reference), 'Use a new architecture output')
    started = time.monotonic()
    with file_lock(output/'.run.lock', blocking=False):
        manifest, groups, observations, summary, dnn, baseline = load_completed(source)
        matrix = read_json(source/'feature_matrix.json')
        require(matrix['manifest_sha256'] == digest(manifest) and len(matrix['rows']) == len(observations),
                'Feature matrix differs from frozen observations')
        ref = read_json(reference/'manifest.json')
        require(ref['kind'] == 'cached_fewshot_feature_comparison' and
                ref['source_manifest_sha256'] == digest(manifest) and
                ref['source_completed_sha256'] == sha256_file(source/'completed.json') and
                ref['dataset'] == manifest['dataset'] and completed(reference, ref),
                'Saved controls are not bound to this source')
        source_names = [*Path(__file__).parent.glob('*.py'), ROOT/'src/generator/ranker_report.py',
                        ROOT/'src/judge/embedding.py', ROOT/'scripts/train_fewshot_ranker.py']
        sources = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_names}
        binding = dict(schema=1, kind='cached_fewshot_architecture_comparison', source=str(source),
            source_manifest_sha256=digest(manifest), source_completed_sha256=sha256_file(source/'completed.json'),
            reference=str(reference), reference_completed_sha256=sha256_file(reference/'completed.json'),
            dataset=manifest['dataset'], neural_recipes=NEURAL_RECIPES, xgb_recipes=XGB_RECIPES,
            runtime=sources, libraries=dict(python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__, xgboost=xgboost.__version__))
        write_once_json(output/'manifest.json', binding)
        if completed(output, binding):
            return read_json(output/'report.json')
        methods = {'dnn_saved': evaluate(observations, [r['logit'] for r in dnn], baseline)}
        for name in CONTROLS:
            methods[name+'_saved'] = evaluate(observations, predictions(reference/name, observations), baseline)
        print('Loaded frozen features and controls; no LLM requests', flush=True)
        details, artifacts = {}, []
        for name in [*NEURAL_RECIPES, 'xgb_chat_pair']:
            directory = output/name
            transform = dict(semantic=crosses.VERSION, chat_id_pair=name != 'dnn_semantic',
                             identity_crosses=identity_crosses.VERSION)
            method_binding = dict(parent_sha256=digest(binding), method=name, feature_transform=transform)
            write_once_json(directory/'manifest.json', method_binding)
            if not completed(directory, method_binding):
                print(f'Fitting {name}', flush=True)
                rows = feature_rows(matrix['rows'], transform['chat_id_pair'])
                model, scores, detail = (boosting.train(groups, observations, rows, XGB_RECIPES)
                    if name.startswith('xgb') else neural_sweep.train(groups, observations, rows, NEURAL_RECIPES[name]))
                model['feature_transform'] = transform
                write_once_json(directory/'model.json', model)
                write_once_json(directory/'training.json', detail)
                write_once_json(directory/'predictions.json', dict(rows=[dict(r, logit=float(s))
                    for r, s in zip(observations, scores)]))
                seal(directory, method_binding, ['model.json', 'training.json', 'predictions.json'])
            details[name] = read_json(directory/'training.json')
            methods[name] = evaluate(observations, predictions(directory, observations), baseline)
            artifacts += [f'{name}/{f}' for f in
                          ('manifest.json', 'completed.json', 'model.json', 'training.json', 'predictions.json')]
            print(f'Completed {name}', flush=True)
        require(all(sha256_file(ROOT/name) == sha for name, sha in sources.items()),
                'Comparison code changed during fitting')
        inner_loss = {name: min(t['inner_pairwise_loss'] for t in detail['trials'])
                      for name, detail in details.items()}
        value = dict(status='complete', manifest_sha256=digest(binding), samples=summary,
            methods=methods, training=details, inner_losses=inner_loss,
            inner_selected_method=min(inner_loss, key=lambda n: (inner_loss[n], n)),
            seconds=time.monotonic()-started, new_model_requests=0,
            scope='offline development comparison; original labels; no production adoption')
        write_once_json(output/'report.json', value)
        atomic_write(output/'REPORT.md', render(value))
        seal(output, binding, [*artifacts, 'report.json', 'REPORT.md'])
        return value
