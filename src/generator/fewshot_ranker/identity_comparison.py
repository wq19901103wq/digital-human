"""Cached behavioral/identity ablations and model tuning, using frozen controls."""
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sklearn
import torch

from ...config import ROOT, sha256_file
from ...iteration.storage import atomic_write, file_lock, read_json, write_once_json
from ..history_sources import digest, require
from ..ranker_report import load_completed
from . import crosses, discovery, identity_crosses, logistic, neural_sweep
from .comparison import evaluate

LR_RECIPE = dict(logistic.RECIPE, c_values=[.0003, .001, .003, .01, .03, .1, 1.])
NEURAL_RECIPES = {'dnn_behavior_id': neural_sweep.recipes(0), 'dcn_behavior_id': neural_sweep.recipes(2)}


def completed(directory, binding):
    value = read_json(directory/'completed.json', default=None)
    if value:
        require(value['manifest_sha256'] == digest(binding) and all(
            sha256_file(directory/name) == checksum for name, checksum in value['artifacts'].items()),
            'Completed comparison artifacts changed')
    return value


def seal(directory, binding, names):
    write_once_json(directory/'completed.json', dict(manifest_sha256=digest(binding),
        artifacts={name: sha256_file(directory/name) for name in names}))


def predictions(directory, observations):
    saved = read_json(directory/'predictions.json')['rows']
    require(len(saved) == len(observations) and all(
        {k: v for k, v in row.items() if k != 'logit'} == obs
        for row, obs in zip(saved, observations)), 'Control predictions differ from frozen observations')
    values = [r['logit'] for r in saved]
    require(bool(np.isfinite(values).all()), 'Nonfinite saved scores')
    return values


def render(value):
    original = value['methods']['dnn_saved']['validation']
    m = original['metrics']
    lines = ['# Few-shot behavior / identity / model comparison', '',
        'Frozen samples, labels, feature extraction and chronological split. Zero new LLM requests. '
        'All hyperparameters and stopping epochs selected inside fit, before outer validation reporting.', '',
        f'Validation: {m["contexts"]} contexts / {m["observations"]} rows / {m["positive"]} positives. '
        f'{m["contexts_with_positive"]} contexts have a positive; {m["contexts_without_positive"]} have none.', '',
        '| Method | AUC | AP | Group AUC | Hit@1 | Hit@3 | Hit@5 | Net Top1 vs selector |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    number = lambda v: 'N/A' if v is None else f'{v:.4f}'
    percent = lambda v: 'N/A' if v is None else f'{v:.2%}'
    for name, result in value['methods'].items():
        metrics, selection = result['validation']['metrics'], result['validation']['selection']
        cells = [number(metrics[k]) for k in ('roc_auc', 'pr_auc_average_precision', 'group_auc_macro')]
        cells += [percent(metrics['topk'][k]['hit_rate']) for k in ('1', '3', '5')]
        net = selection.get('net_expected_positive_selections')
        cells += ['N/A' if net is None else f'{net:+g}']
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    baseline = original['topk_comparison']
    lines.append('| existing_selector | — | — | — | ' + ' | '.join(
        percent(baseline[k]['baseline']['hit_rate']) for k in ('1', '3', '5')) + ' | 0 |')
    lines += ['', 'Hit@K: at least one positive among K separately labeled examples; '
        'all-negative contexts remain in the denominator. Ties use uniform expected selection. '
        'Existing selector Top5 is unavailable when its frozen output contains only three choices.', '',
        '| Method | Conditional Hit@1 | Conditional Hit@3 | Conditional Hit@5 | Mean positives@3 | Mean positives@5 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, result in value['methods'].items():
        top = result['validation']['metrics']['topk']
        lines.append('| ' + name + ' | ' + ' | '.join(
            [percent(top[k]['hit_rate_given_available']) for k in ('1', '3', '5')] +
            [number(top[k]['mean_positive_count']) for k in ('3', '5')]) + ' |')
    lines += ['', '## Tuning (inner fit only)', '']
    for name, detail in value['training'].items():
        if 'selected_C' in detail:
            lines.append(f'- {name}: C={detail["selected_C"]:g}, {detail["fields"]} fields, '
                         f'{detail["one_hot_dimensions"]} one-hot dimensions.')
            lines += [f'  - C={t["C"]:g}, inner loss={t["inner_pairwise_loss"]:.6f}' for t in detail['trials']]
        else:
            lines.append(f'- {name}: trial {detail["selected_trial"]}, {detail["selected_epochs"]} refit epochs.')
            for trial in detail['trials']:
                r = trial['recipe']
                lines.append(f'  - trial {trial["index"]}: hidden={r["hidden"]}, norm={r["normalization"]}, '
                    f'dropout={r["dropout"]}, decay={r["weight_decay"]}, cross layers={r["cross_layers"]}, '
                    f'inner loss={trial["inner_pairwise_loss"]:.6f}, epochs={trial["selected_epochs"]}.')
    lines += ['', f'Chosen using inner fit only: **{value["inner_selected_method"]}**. '
              'All outer validation metrics are reported, including unsuccessful ablations.', '',
              '## Behavioral feature exploration', '',
              'Each family is added separately to the same tuned semantic LR control. '
              'behavior_all combines all four; behavior_id also adds ID/style interactions.', '',
              '| Family | Fields | Fields ≥50% known in fit | Fields varying within ≥10 mixed fit contexts |',
              '|---|---:|---:|---:|']
    profile = value['feature_profile']
    for family, count in profile['family_sizes'].items():
        items = [v for k, v in profile['fields'].items() if k.startswith(f'discovery.{family}.')]
        lines.append(f'| {family} | {count} | {sum(v["known_fraction"] >= .5 for v in items)} | '
                     f'{sum(v["mixed_contexts_with_distinct_values"] >= 10 for v in items)} |')
    lines += ['', 'Field-level fit coverage and support counts are in report.json. '
              'Unavailable reply timing stays unknown; no future or label-derived interaction history is used.', '',
        '## Identity coverage', '',
        '| Field | Fit categories | Validation rows seen in fit | Seen in mixed-label fit | Validation unknown |',
        '|---|---:|---:|---:|---:|']
    for key, item in value['identity_coverage'].items():
        if key == 'validation_contexts':
            lines.append(f'\nFamiliar target chats: {item["familiar_target_chat"]}/{item["total"]} validation contexts.\n')
        else:
            lines.append(f'| {key} | {item["fit_unique"]} | {item["validation_known_in_fit"]}/{item["validation_rows"]} | '
                         f'{item["validation_seen_in_mixed_fit"]} | {item["validation_unknown"]} |')
    lines += ['## Scope', '',
        'Chat ID denotes a conversation; sender IDs are chat-scoped exporter identities, not global person IDs. '
        'Unseen categories have zero contribution. No label-derived history features are added.',
        'LR ablations share the same C grid: semantic crosses; + ID pairs; + ID/style pairs. '
        'DNN/DCN compare architectures on the same combined behavior and ID/style inputs; '
        'every input uses embedding dimension 8.',
        'These are development comparisons on an already-used outer validation set. '
        'The report includes all methods; it does not turn the best outer metric into an independent test.',
        'Frozen labels measure generation with one example. Hit@3/5 does not measure jointly generating '
        'with three/five examples. No production change. Per-context outcomes and uncertainty are in report.json.']
    return '\n'.join(lines)+'\n'


def run(source, reference, output):
    source, reference, output = (Path(p).resolve() for p in (source, reference, output))
    require(output not in (source, reference), 'Use a new output for the feature comparison')
    started = time.monotonic()
    with file_lock(output/'.run.lock', blocking=False):
        manifest, groups, observations, summary, dnn, baseline = load_completed(source)
        matrix = read_json(source/'feature_matrix.json')
        require(matrix['manifest_sha256'] == digest(manifest) and len(matrix['rows']) == len(observations),
                'Feature matrix differs from bound observations')
        ref = read_json(reference/'manifest.json')
        require(ref['kind'] == 'cached_fewshot_lr_comparison' and
                ref['source_manifest_sha256'] == digest(manifest) and
                ref['source_completed_sha256'] == sha256_file(source/'completed.json') and
                ref['dataset'] == manifest['dataset'] and completed(reference, ref),
                'Saved LR controls are not bound to this source')
        source_names = [*Path(__file__).parent.glob('*.py'), ROOT/'src/generator/ranker_report.py',
                        ROOT/'src/judge/embedding.py', ROOT/'scripts/train_fewshot_ranker.py']
        sources = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_names}
        binding = dict(schema=1, kind='cached_fewshot_feature_comparison', source=str(source),
            source_manifest_sha256=digest(manifest), source_completed_sha256=sha256_file(source/'completed.json'),
            reference=str(reference), reference_completed_sha256=sha256_file(reference/'completed.json'),
            dataset=manifest['dataset'], lr_recipe=LR_RECIPE, neural_recipes=NEURAL_RECIPES,
            crosses=identity_crosses.VERSION, discovery=discovery.VERSION, runtime=sources,
            libraries=dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__))
        write_once_json(output/'manifest.json', binding)
        if completed(output, binding):
            return read_json(output/'report.json')
        methods = {'dnn_saved': evaluate(observations, [r['logit'] for r in dnn], baseline)}
        for name in ('lr_existing', 'lr_expanded'):
            methods[name+'_saved'] = evaluate(observations, predictions(reference/name, observations), baseline)
        print('Loaded frozen dataset and three saved controls; no feature extraction', flush=True)
        raw = matrix['rows']
        semantic = [crosses.expand(row) for row in raw]
        additions = [discovery.fields(row) for row in raw]
        trials = [('lr_semantic_tuned', 'semantic', ()), ('lr_id_pairs', 'id_pairs', ()),
                  ('lr_id_style', 'id_style', ())]
        trials += [('lr_'+family, 'semantic', (family,)) for family in discovery.FAMILIES]
        trials += [('lr_behavior_all', 'semantic', discovery.FAMILIES),
                   ('lr_behavior_id', 'id_style', discovery.FAMILIES)]
        trials += [(name, 'id_style', discovery.FAMILIES) for name in NEURAL_RECIPES]
        details, artifacts = {}, []
        for name, base, families in trials:
            directory = output/name
            transform = dict(base=base, families=list(families), discovery=discovery.VERSION,
                             identity_crosses=identity_crosses.VERSION, semantic=crosses.VERSION)
            method_binding = dict(parent_sha256=digest(binding), method=name, feature_transform=transform)
            write_once_json(directory/'manifest.json', method_binding)
            if not completed(directory, method_binding):
                print(f'Fitting {name}', flush=True)
                prefixes = tuple(f'discovery.{family}.' for family in families)
                rows = [dict(row if base == 'semantic' else identity_crosses.expand(row, style=base == 'id_style'),
                             **{k: v for k, v in extra.items() if k.startswith(prefixes)})
                        for row, extra in zip(semantic, additions)]
                if name.startswith('lr_'):
                    model, scores, detail = logistic.train(groups, observations, rows, LR_RECIPE)
                else:
                    model, scores, detail = neural_sweep.train(groups, observations, rows, NEURAL_RECIPES[name])
                model['feature_transform'] = transform
                write_once_json(directory/'model.json', model)
                write_once_json(directory/'training.json', detail)
                write_once_json(directory/'predictions.json', dict(rows=[dict(r, logit=float(s))
                    for r, s in zip(observations, scores)]))
                seal(directory, method_binding, ['model.json', 'training.json', 'predictions.json'])
                del rows
            details[name] = read_json(directory/'training.json')
            methods[name] = evaluate(observations, predictions(directory, observations), baseline)
            artifacts += [f'{name}/{f}' for f in
                          ('manifest.json', 'completed.json', 'model.json', 'training.json', 'predictions.json')]
            print(f'Completed {name}', flush=True)
        require(all(sha256_file(ROOT/name) == checksum for name, checksum in sources.items()),
                'Comparison code changed during fitting')
        inner_loss = {name: min(t['inner_pairwise_loss'] for t in detail['trials'])
                      for name, detail in details.items()}
        value = dict(status='complete', manifest_sha256=digest(binding), samples=summary,
            methods=methods, training=details,
            inner_selected_method=min(inner_loss, key=lambda name: (inner_loss[name], name)),
            inner_losses=inner_loss, feature_profile=discovery.profile(groups, observations, additions),
            identity_coverage=identity_crosses.coverage(groups, observations,
                [identity_crosses.expand(row) for row in raw]),
            added_fields=sorted((set(identity_crosses.expand(raw[0], style=True))-set(semantic[0])) | set(additions[0])),
            seconds=time.monotonic()-started,
            new_model_requests=0, scope='offline development comparison; no production adoption')
        write_once_json(output/'report.json', value)
        atomic_write(output/'REPORT.md', render(value))
        seal(output, binding, [*artifacts, 'report.json', 'REPORT.md'])
        return value
