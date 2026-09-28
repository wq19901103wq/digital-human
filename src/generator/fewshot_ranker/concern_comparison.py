"""Frozen-recipe comparison with separately extracted self-concern context fields."""
from collections import Counter
from pathlib import Path
import os
import platform
import time

import numpy as np
import xgboost as xgb

from ...config import ROOT, sha256_file
from ...iteration import control, versions
from ...iteration.storage import atomic_write, file_lock, read_json, write_json, write_once_json
from ...judge.corrected import CodexJudgeClient
from ..history_sources import digest, require
from ..ranker_report import load_completed
from . import concern, supplemental
from .action_comparison import METHOD, control_binding, fit_selected
from .architecture_comparison import feature_rows
from .comparison import evaluate
from .identity_comparison import completed, predictions, seal

KIND = 'supplemental_fewshot_concern_comparison'


def render(value):
    lines = ['# Independent self-concern comparison', '',
        'Target and example contexts are extracted independently. Only 8 categorical fields change. '
        'Original binary labels, chronological split, XGB recipe, seed and rounds are retained.', '',
        '| Model | AUC | AP | Group AUC | Hit@1 | Hit@3 | Hit@5 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    number = lambda v: 'N/A' if v is None else f'{v:.4f}'
    for name, result in value['methods'].items():
        m = result['validation']['metrics']
        cells = [number(m[k]) for k in ('roc_auc', 'pr_auc_average_precision', 'group_auc_macro')]
        cells += [number(m['topk'][k]['hit_rate']) for k in ('1', '3', '5')]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', f'Unique supplemental contexts: {value["supplemental_contexts"]}; '
        f'LLM-derived contexts: {value["llm_contexts"]}; structurally absent self: {value["local_contexts"]}.', '',
        'Hit@K uses separately labeled examples and includes all-negative contexts. This is an offline '
        'development comparison on previously used validation data. Generator benefit requires the '
        'formal development and fixed tests. Evidence positions are stored for audit, never ranker features.', '',
        'Added fields:', '', *['- '+name for name in value['added_fields']]]
    return '\n'.join(lines)+'\n'


def run(source, reference, output, *, feature_cache, instance, workers=16, passes=3):
    source, reference, output, feature_cache = (Path(p).resolve() for p in
                                              (source, reference, output, feature_cache))
    require(output not in (source, reference), 'Use a new concern comparison output')
    require(1 <= workers <= 16 and passes >= 1, 'Invalid concern extraction limits')
    versions.switch_instance(instance)
    started = time.monotonic()
    with file_lock(output/'.run.lock', blocking=False), control.job(output):
        def state(status, **kw):
            write_json(output/'state.json', dict(status=status, pid=os.getpid(), updated_at=time.time(), **kw))
        try:
            state('preparing')
            manifest, groups, observations, summary, _, baseline = load_completed(source)
            matrix = read_json(source/'feature_matrix.json')
            require(matrix['manifest_sha256'] == digest(manifest) and len(matrix['rows']) == len(observations),
                    'Feature matrix differs from frozen observations')
            control_dir = control_binding(source, manifest, reference)
            config = read_json(source/'transport.json')['config']
            client = CodexJudgeClient(config)
            require(client.cache_identity() == manifest['features']['client'],
                    'Concern extraction must retain the original feature client')
            tasks, refs = concern.prepare(groups, client.cache_identity())
            source_files = [*Path(__file__).parent.glob('*.py'), ROOT/'src/generator/ranker_report.py',
                            ROOT/'src/judge/embedding.py', ROOT/'scripts/train_fewshot_ranker.py']
            runtime = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files}
            watched = {p: sha256_file(p) for p in source_files}
            for directory in (source, reference, control_dir):
                for name in ('manifest.json', 'completed.json'):
                    watched[directory/name] = sha256_file(directory/name)
                watched.update({directory/name: sha for name, sha in
                                read_json(directory/'completed.json')['artifacts'].items()})
            watched[source/'transport.json'] = sha256_file(source/'transport.json')
            binding = dict(schema=1, kind=KIND, source=str(source), reference=str(reference),
                source_manifest_sha256=digest(manifest), source_completed_sha256=sha256_file(source/'completed.json'),
                reference_completed_sha256=sha256_file(reference/'completed.json'), dataset=manifest['dataset'],
                control_model_sha256=sha256_file(control_dir/'model.json'),
                control_training_sha256=sha256_file(control_dir/'training.json'),
                feature_transform=concern.TRANSFORM, runtime=runtime,
                features=dict(client=client.cache_identity(), request_keys_sha256=digest(sorted(tasks)), count=len(tasks)),
                policy='only independent self-concern supplement changes; original labels; fixed control recipe and rounds',
                libraries=dict(python=platform.python_version(), numpy=np.__version__, xgboost=xgb.__version__))
            write_once_json(output/'manifest.json', binding)
            if completed(output, binding):
                state('complete', reused=True)
                return read_json(output/'report.json')
            write_once_json(output/'supplemental_tasks.json', tasks)
            state('extracting', samples=summary, feature_tasks=len(tasks))
            values = supplemental.run(tasks, feature_cache, output, client, workers=workers, passes=passes)
            write_once_json(output/'supplemental_values.json', values)
            raw = feature_rows(matrix['rows'], True)
            rows = [concern.expand(row, *(values[key] for key in refs[(obs['target_id'], obs['example_id'])]))
                    for row, obs in zip(raw, observations)]
            write_once_json(output/'feature_matrix.json', dict(manifest_sha256=digest(binding), rows=rows))
            directory = output/METHOD
            method_binding = dict(parent_sha256=digest(binding), method=METHOD, feature_transform=concern.TRANSFORM,
                control_model_sha256=binding['control_model_sha256'],
                control_training_sha256=binding['control_training_sha256'])
            write_once_json(directory/'manifest.json', method_binding)
            state('training', samples=summary, feature_tasks=len(tasks))
            if not completed(directory, method_binding):
                model, scores, detail = fit_selected(groups, observations, rows,
                    read_json(control_dir/'model.json'), read_json(control_dir/'training.json'), transform=concern.TRANSFORM)
                write_once_json(directory/'model.json', model)
                write_once_json(directory/'training.json', detail)
                write_once_json(directory/'predictions.json', dict(rows=[dict(r, logit=float(s))
                    for r, s in zip(observations, scores)]))
                seal(directory, method_binding, ['model.json', 'training.json', 'predictions.json'])
            methods = {name: evaluate(observations, predictions(path, observations), baseline)
                       for name, path in (('control_saved', control_dir), ('self_concern', directory))}
            require(all(sha256_file(p) == sha for p, sha in watched.items()), 'Concern inputs changed during comparison')
            local = sum(concern.local_value(request) is not None for request in tasks.values())
            value = dict(status='complete', manifest_sha256=digest(binding), samples=summary, methods=methods,
                training=read_json(directory/'training.json'), added_fields=sorted(set(rows[0])-set(raw[0])),
                seconds=time.monotonic()-started, supplemental_contexts=len(tasks),
                local_contexts=local, llm_contexts=len(tasks)-local,
                feature_distribution={k: dict(Counter(v[k] for v in values.values())) for k in concern.DEFINITIONS},
                scope='offline binary-label ranking comparison; generator benefit requires formal development')
            write_once_json(output/'report.json', value)
            atomic_write(output/'REPORT.md', render(value))
            artifacts = [f'{METHOD}/{name}' for name in
                         ('manifest.json', 'completed.json', 'model.json', 'training.json', 'predictions.json')]
            seal(output, binding, [*artifacts, 'report.json', 'REPORT.md', 'supplemental_tasks.json',
                                   'supplemental_values.json', 'feature_matrix.json'])
            state('complete', samples=summary, feature_tasks=len(tasks))
            return value
        except BaseException as exc:
            state('needs_attention', error_type=type(exc).__name__, error=str(exc))
            raise
