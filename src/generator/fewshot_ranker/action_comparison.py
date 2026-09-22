"""Fixed-recipe XGB comparison adding only cached reply-action crosses."""
from copy import deepcopy
from pathlib import Path
import json
import platform
import time

import numpy as np
import xgboost as xgb
from threadpoolctl import threadpool_limits

from ...config import ROOT, sha256_file
from ...iteration.storage import atomic_write, file_lock, read_json, write_once_json
from ..history_sources import digest, require
from ..ranker_report import load_completed
from . import action_crosses, boosting, logistic
from .architecture_comparison import feature_rows
from .comparison import evaluate
from .identity_comparison import completed, predictions, seal

KIND = 'cached_fewshot_action_comparison'
METHOD = 'xgb_chat_pair'
CONTROL_TRANSFORM = {k: v for k, v in action_crosses.TRANSFORM.items() if k != 'reply_action_crosses'}


@threadpool_limits.wrap(limits=1)
def fit_selected(groups, observations, rows, saved_model, saved_training):
    """Keep the control's binary labels, recipe and rounds; fit contexts only."""
    require(saved_model['kind'] == 'fewshot_pairwise_xgboost_v1' and
            not saved_model.get('supervision') and
            saved_model['feature_transform'] == CONTROL_TRANSFORM and
            all(r['z'] in (0, 1) for r in observations), 'Expected frozen binary XGB control')
    require(saved_training['selected_recipe'] == saved_model['recipe'] and
            saved_training['selected_rounds'] == saved_model['rounds'],
            'Control model and training recipe differ')
    fit_groups = [g for g in groups if g['split'] == 'fit']
    encoder = logistic.encoder(rows, [i for g in fit_groups for i in g['indices']])
    x = encoder.transform(rows)
    model, detail = boosting.fit(x, observations, fit_groups, [], saved_model['recipe'], saved_model['rounds'])
    value = deepcopy(saved_model)
    value.update(booster=json.loads(model.save_raw(raw_format='json').decode()),
        vocabulary={k: int(v) for k, v in encoder.vocabulary_.items()},
        feature_names=encoder.get_feature_names_out().tolist(), feature_transform=action_crosses.TRANSFORM)
    scores = model.predict(xgb.DMatrix(x, nthread=1), output_margin=True)
    require(bool(np.isfinite(scores).all()), 'Nonfinite action-comparison scores')
    return value, scores, dict(selected_recipe=saved_model['recipe'], selected_rounds=saved_model['rounds'],
        fitting=detail, fields=len(rows[0]), one_hot_dimensions=x.shape[1],
        supervision='original_binary_single_draw',
        selection='reuse control recipe and rounds; fit contexts only; no outer tuning')


def control_binding(source, manifest, reference):
    ref = read_json(reference/'manifest.json')
    require(ref['kind'] == 'cached_fewshot_architecture_comparison' and
            ref['source_manifest_sha256'] == digest(manifest) and
            ref['source_completed_sha256'] == sha256_file(source/'completed.json') and
            ref['dataset'] == manifest['dataset'] and completed(reference, ref),
            'Action control is not bound to this frozen source')
    control = reference/METHOD
    child = read_json(control/'manifest.json')
    require(child['parent_sha256'] == digest(ref) and child['method'] == METHOD and
            child['feature_transform'] == CONTROL_TRANSFORM and completed(control, child),
            'Action control completion differs')
    return control


def render(value):
    lines = ['# Cached reply-action cross comparison', '',
        'Only 19 categorical crosses change. Original binary labels, fit/validation contexts, '
        'XGB recipe, seed and stopping rounds are held constant. Saved control scores are reused.', '',
        '| Model | AUC | PR-AUC (AP) | Group AUC | Hit@1 | Hit@3 | Hit@5 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    number = lambda v: 'N/A' if v is None else f'{v:.4f}'
    for name, result in value['methods'].items():
        m = result['validation']['metrics']
        cells = [number(m[k]) for k in ('roc_auc', 'pr_auc_average_precision', 'group_auc_macro')]
        cells += [number(m['topk'][k]['hit_rate']) for k in ('1', '3', '5')]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Hit@K counts any individually positive example among K, including all-negative '
        'contexts in the denominator. Ties use uniform expectations. This is an offline development '
        'comparison on previously used validation data, not a generator development or adoption result.', '',
        f'Local training: {value["seconds"]:.1f}s. New LLM requests: 0.', '', 'Added fields:', '',
        *['- '+name for name in value['added_fields']]]
    return '\n'.join(lines)+'\n'


def run(source, reference, output):
    source, reference, output = (Path(p).resolve() for p in (source, reference, output))
    require(output not in (source, reference), 'Use a new action comparison output')
    started = time.monotonic()
    with file_lock(output/'.run.lock', blocking=False):
        manifest, groups, observations, summary, _, baseline = load_completed(source)
        matrix = read_json(source/'feature_matrix.json')
        require(matrix['manifest_sha256'] == digest(manifest) and len(matrix['rows']) == len(observations),
                'Feature matrix differs from frozen observations')
        control = control_binding(source, manifest, reference)
        source_files = [*Path(__file__).parent.glob('*.py'), ROOT/'src/generator/ranker_report.py',
                        ROOT/'src/judge/embedding.py', ROOT/'scripts/train_fewshot_ranker.py']
        runtime = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files}
        watched = {p: sha256_file(p) for p in source_files}
        for directory in (source, reference, control):
            for name in ('manifest.json', 'completed.json'):
                watched[directory/name] = sha256_file(directory/name)
            watched.update({directory/name: sha for name, sha in
                            read_json(directory/'completed.json')['artifacts'].items()})
        binding = dict(schema=1, kind=KIND, source=str(source), reference=str(reference),
            source_manifest_sha256=digest(manifest), source_completed_sha256=sha256_file(source/'completed.json'),
            reference_completed_sha256=sha256_file(reference/'completed.json'), dataset=manifest['dataset'],
            control_model_sha256=sha256_file(control/'model.json'),
            control_training_sha256=sha256_file(control/'training.json'),
            feature_transform=action_crosses.TRANSFORM, runtime=runtime,
            policy='only reply-action crosses change; original binary labels; fixed control recipe and rounds',
            libraries=dict(python=platform.python_version(), numpy=np.__version__, xgboost=xgb.__version__))
        write_once_json(output/'manifest.json', binding)
        if completed(output, binding):
            return read_json(output/'report.json')
        raw = feature_rows(matrix['rows'], True)
        rows = [action_crosses.expand(r) for r in raw]
        directory = output/METHOD
        method_binding = dict(parent_sha256=digest(binding), method=METHOD,
            feature_transform=action_crosses.TRANSFORM,
            control_model_sha256=binding['control_model_sha256'],
            control_training_sha256=binding['control_training_sha256'])
        write_once_json(directory/'manifest.json', method_binding)
        if not completed(directory, method_binding):
            print('Fitting XGB with cached action crosses; same labels, recipe and rounds; zero LLM requests', flush=True)
            saved_model, saved_training = (read_json(control/f'{name}.json') for name in ('model', 'training'))
            model, scores, detail = fit_selected(groups, observations, rows, saved_model, saved_training)
            write_once_json(directory/'model.json', model)
            write_once_json(directory/'training.json', detail)
            write_once_json(directory/'predictions.json', dict(rows=[dict(r, logit=float(s))
                for r, s in zip(observations, scores)]))
            seal(directory, method_binding, ['model.json', 'training.json', 'predictions.json'])
        methods = {name: evaluate(observations, predictions(path, observations), baseline)
                   for name, path in (('control_saved', control), ('reply_action_crosses', directory))}
        require(all(sha256_file(p) == sha for p, sha in watched.items()), 'Comparison inputs changed during fitting')
        value = dict(status='complete', manifest_sha256=digest(binding), samples=summary, methods=methods,
            training=read_json(directory/'training.json'), added_fields=sorted(set(rows[0])-set(raw[0])),
            seconds=time.monotonic()-started, new_model_requests=0,
            scope='offline binary-label ranking comparison; generator benefit requires formal development')
        write_once_json(output/'report.json', value)
        atomic_write(output/'REPORT.md', render(value))
        artifacts = [f'{METHOD}/{name}' for name in
                     ('manifest.json', 'completed.json', 'model.json', 'training.json', 'predictions.json')]
        seal(output, binding, [*artifacts, 'report.json', 'REPORT.md'])
        return value
