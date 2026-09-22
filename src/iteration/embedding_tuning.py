"""Resumable embedding search; one shared audit, no LLM requests or new samples."""
from __future__ import annotations

import concurrent.futures
import itertools
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from .. import cache
from ..config import ConfigError, load_settings, sha256_file, valid_name
from ..judge import embedding as net, corrected_v1 as rt
from ..judge.gbdt import GBDTScorer, option_vectors
from . import comparisons, draw_replay, experiment, learning_guard, record_contract, task_state, versions
from .storage import file_lock, read_json, write_json, write_once_json

SEEDS = (17, 29, 43)


def recipes():
    """Matched architecture/norm comparisons, then reproducible optimizer exploration."""
    common = dict(activation='silu', dropout=.1, learning_rate=.001, weight_decay=.01,
                  batch_size=128, max_epochs=120, patience=18)
    result = [dict(common, hidden=hidden, cross_layers=cross, normalization=norm)
              for hidden, cross, norm in itertools.product(
                  ([32], [64, 32], [128, 64, 32]), (0, 2), ('none', 'batch', 'layer'))]
    rng = np.random.default_rng(20260917)
    for base in list(result):
        result.append(dict(base,
            activation=str(rng.choice(['relu', 'silu', 'gelu'])),
            dropout=float(rng.choice([0., .15, .3])),
            learning_rate=float(rng.choice([.0003, .001, .003])),
            weight_decay=float(rng.choice([.001, .03, .1])),
            batch_size=int(rng.choice([64, 256]))))
    return {f'e8-{i+1:02d}': value for i, value in enumerate(result)}


def split_training(rows, source_rows):
    """Last 20% of existing training pairs select heads; formal dev stays untouched."""
    source = {r['case_id']: r for r in source_rows}
    if len(source) != len(source_rows) or any(r['case_id'] not in source for r in rows):
        raise ConfigError('Training split lacks original sample timestamps')
    ordered = sorted(range(len(rows)), key=lambda i: (
        source[rows[i]['case_id']]['source_span']['end_timestamp'], rows[i]['case_id']))
    count = max(1, int(len(rows) * .2))
    if len(rows) < 10:
        raise ConfigError('Too few training pairs for an internal split')
    return np.asarray(ordered[:-count]), np.asarray(ordered[-count:])


def bind_batch(directory, spec):
    """Same ID never silently reuses changed data/features/parameters/runtime."""
    path = directory / 'spec.json'
    if path.exists() and read_json(path) != spec:
        raise ConfigError('Embedding batch inputs changed; use a new study ID')
    write_once_json(path, spec)


def shared_learning_snapshot(pack, baseline):
    """Existing provenance guards run once per batch, never per trial."""
    learning_guard.verify_pack(pack, 'development')
    return learning_guard.snapshot(pack['data_ref'], [baseline['dir']], 'judge_development')


def _job(payload):
    key, encoder, recipe, pairs, labels, train, validation, seed, epochs = payload
    model, result = net.fit(encoder, recipe, pairs, labels, train, validation, seed, epochs)
    return key, dict(result, model=net.document(model))


def train_jobs(directory, jobs, workers, check):
    """Shared matrices are prepared once; workers never audit or call LLMs."""
    pending, results = [], {}
    for job in jobs:
        key = job[0]
        path = directory / (key + '.json')
        if path.exists():
            result = read_json(path)
            if result.get('job_sha256') != cache.digest(dict(key=key, encoder=job[1], recipe=job[2],
                    seed=job[7], epochs=job[8])):
                raise ConfigError('Training checkpoint identity differs')
            if result.get('model_sha256') != cache.digest(result['model']):
                raise ConfigError('Training checkpoint model changed')
            results[key] = result
        else:
            pending.append(job)
    if not pending:
        return results
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(_job, job): job for job in pending}
        while futures:
            done, _ = concurrent.futures.wait(futures, timeout=20,
                                              return_when=concurrent.futures.FIRST_COMPLETED)
            check()
            for future in done:
                job = futures.pop(future)
                key, result = future.result()
                result['job_sha256'] = cache.digest(dict(key=key, encoder=job[1], recipe=job[2],
                    seed=job[7], epochs=job[8]))
                result['model_sha256'] = cache.digest(result['model'])
                write_once_json(directory / (key + '.json'), result)
                results[key] = result
                print(json.dumps(dict(job=key, epoch=result['best_epoch'],
                    completed=len(results), total=len(jobs)), ensure_ascii=False), flush=True)
            task_state.update(directory.parent, directory.name, completed=len(results), total=len(jobs))
    return results


def _initial_summary(records):
    if len({r['case_id'] for r in records}) != len(records):
        raise ConfigError('Duplicate formal comparison cases')
    return dict(total=len(records), baseline_correct=sum(r['baseline_correct'] for r in records),
        candidate_correct=sum(r['candidate_correct'] for r in records),
        initial_net=sum(int(r['candidate_correct']) - int(r['baseline_correct']) for r in records),
        disagreements=sum(r['baseline_correct'] != r['candidate_correct'] for r in records),
        pending_confirmation=sum(r['status'] == 'failed' for r in records))


def comparison_records(rows, index, baseline_hits, hits, probability, extra):
    """Never substitute initial observations for missing independent rounds."""
    if extra != 2:
        raise ConfigError('Saved embedding comparison requires two independent extra rounds')
    records = []
    for case in rows:
        cid = case['case_id']
        ix = index[cid, 0]
        b, c = bool(baseline_hits[ix]), bool(hits[ix])
        row = dict(case_id=cid, status='ok', baseline_correct=b, candidate_correct=c,
                   candidate_probability_a=float(probability[ix]))
        if b != c:
            indices = [index.get((cid, rnd)) for rnd in range(extra + 1)]
            if any(i is None for i in indices):
                row.update(status='failed', reason='Initial scoring complete; independent saved rounds missing',
                           missing_rounds=[r for r, i in enumerate(indices) if i is None])
            else:
                bv, cv = [bool(baseline_hits[i]) for i in indices], [bool(hits[i]) for i in indices]
                row.update(flip_verified=True, baseline_votes=bv, candidate_votes=cv,
                    baseline_identified_final=sum(bv) > len(bv)/2,
                    candidate_identified_final=sum(cv) > len(cv)/2)
        records.append(row)
    return records


def formal_comparisons(directory, spec, fitted, encoder, baseline, pack, check):
    """All models consume the exact same verified saved observations."""
    protocol = spec['protocol']
    targets = {}
    for key, fitted_result in sorted(fitted.items()):
        targets[key] = comparisons.register(f'embedding-{directory.name}-{key}',
            data_ref=spec['data_ref'], pack_ref=spec['pack_ref'],
            comparison={'baseline': {'judge_ref': baseline['id'], 'config_sha256': spec['baseline_sha256']},
                'candidate': {'kind': 'embedding_dnn', 'study': directory.name, 'trial': key,
                    'model_sha256': fitted_result['model_sha256'], 'recipe': fitted_result['model']['recipe']}},
            change='All-category embedding size 8; shared option scorer; cached Luna features',
            protocol=protocol, provenance={'mode': 'registered_embedding_tuning',
                'training_spec_sha256': sha256_file(directory / 'spec.json'),
                'saved_draws': spec['saved_draws'], 'selection': 'training_internal_only'})
    frozen = {'path': spec['saved_draws'], 'sha256': spec['saved_draws_sha256']}
    # Replay validates observations independently of candidate type. This loader
    # binds their feature channel to the actual baseline; DNN uses those same fields.
    replay = draw_replay.SavedDrawReplay(frozen, pack, baseline, baseline)
    scorer = GBDTScorer(read_json(baseline['dir'] / 'scorer.json'))
    legacy_encoder = rt.load_formal_judge(baseline['dir'] / 'correction.json')
    data, baseline_hits, human, index = [], [], [], {}
    task_state.update(directory, 'preparing_shared_development', completed=0, total=len(pack['rows']))
    for i, case in enumerate(pack['rows']):
        for rnd in sorted(map(int, replay.manifest['draws'][case['case_id']])):
            draw = replay.get(case, rnd, None)
            mapping = draw['mapping']
            blind, metadata = mapping['blind_case'], mapping['context_metadata']
            a, b = option_vectors(legacy_encoder, draw['features'], blind, metadata)
            hit = bool((scorer.probability_a(a[None, :], b[None, :])[0] >= .5)
                       == (mapping['human_option'] == 'A'))
            index[case['case_id'], rnd] = len(data)
            data.append(tuple(net.categories(draw['features'][side], blind[side], metadata)
                              for side in ('option_A', 'option_B')))
            baseline_hits.append(hit)
            human.append(mapping['human_option'] == 'A')
        if (i+1) % 100 == 0:
            check()
            task_state.update(directory, 'preparing_shared_development', completed=i+1, total=len(pack['rows']))
            print(f'shared development prepared {i+1}/{len(pack["rows"])}', flush=True)
    encoded, human = net.encode(encoder, data), np.asarray(human)
    output = []
    torch.set_num_threads(1)
    for key, result in sorted(fitted.items()):
        check()
        model = net.restore(result['model'])
        probability = net.predict(model, encoded)
        hits = (probability >= .5) == human
        records = comparison_records(pack['rows'], index, baseline_hits, hits, probability,
                                     protocol['flip_extra_rounds'])
        target = targets[key]
        if experiment.state_of(target).get('status') != 'finished':
            with record_contract.writer(target) as append:
                for row in records:
                    append(row)
        initial = _initial_summary(records)
        write_once_json(target / 'initial.json', initial)
        item = dict(trial=key, experiment=target.name, initial=initial,
                    recipe=result['model']['recipe'], seed=result['seed'])
        if not initial['pending_confirmation']:
            item['confirmed'] = comparisons.complete(target)
        else:
            ok = [r for r in records if r['status'] == 'ok']
            metrics = dict(attempted=len(records), pairs=len(ok), failures=len(records)-len(ok),
                identified_baseline=sum(r['baseline_correct'] for r in ok),
                identified_candidate=sum(r['candidate_correct'] for r in ok))
            record_contract.validate_completion(target, metrics, complete=False)
            write_json(target / 'state.json', dict(status='paused', phase='awaiting_independent_evidence',
                reason='完整初测已保存；缺少部分分歧题的独立观察，不发起新请求，不据初测晋级',
                metrics=metrics, initial=initial, verdict='confirmation_pending'))
        output.append(item)
        print(json.dumps(dict(trial=key, formal_initial=initial), ensure_ascii=False), flush=True)
    return output


def run(study, source, arm, pack_ref, saved_draws, workers=3):
    directory = versions.PRIVATE / 'judge_training' / valid_name(study)
    with file_lock(directory / '.run.lock', blocking=False):
        try:
            return _run(directory, source, arm, pack_ref, saved_draws, workers)
        except BaseException as exc:
            task_state.update(directory, 'interrupted', status='interrupted', error=str(exc))
            raise


def _run(directory, source, arm, pack_ref, saved_draws, workers):
    source = versions.PRIVATE / 'judge_training' / valid_name(source)
    arm_path = source / valid_name(arm)
    root_spec, source_spec = read_json(source / 'spec.json'), read_json(arm_path / 'spec.json')
    rows, entries = read_json(arm_path / 'training.json')['rows'], read_json(arm_path / 'features.json')['entries']
    pointers = versions.load_pointers()
    # Freeze the original comparison baseline across resume, even if pointers move.
    old = read_json(directory / 'spec.json', default={})
    baseline = versions.judge_dir(old.get('baseline', pointers['production_judge']))
    if baseline['config'].get('decision_policy') != 'gbdt_only':
        raise ConfigError('This batch compares against a frozen GBDT baseline')
    feature_config = {**baseline['config']['llm'], **baseline['config'].get('feature_llm', {})}
    if feature_config != source_spec['feature_config']:
        raise ConfigError('Training and baseline feature extraction conditions differ')
    pack_path = versions.PRIVATE / 'judge_eval' / valid_name(pack_ref) / 'pack.json'
    pack = read_json(pack_path)
    if pack['data_ref'] != root_spec['data_ref'] or pack['c0_gen_version'] != root_spec['generator_ref']:
        raise ConfigError('Training and development data/generator conditions differ')
    material_seal = learning_guard.RunSeal(directories=[
        versions.data_version_dir(pack['data_ref']),
        versions.generator_dir(pack['c0_gen_version']), baseline['dir']])
    learning_snapshot = shared_learning_snapshot(pack, baseline)
    material_seal.check()
    grid = recipes()
    spec = dict(schema=1, kind='embedding_dnn_tuning', dataset='training', id=directory.name,
        source=str(arm_path), data_ref=root_spec['data_ref'], generator_ref=root_spec['generator_ref'],
        training_total=len(rows), embedding_dim=8, numeric_bypass=False, arm=arm,
        baseline=baseline['id'], baseline_sha256=sha256_file(baseline['dir'] / 'config.json'),
        pack_ref=pack_ref, pack_sha256=sha256_file(pack_path), saved_draws=str(Path(saved_draws).resolve()),
        saved_draws_sha256=sha256_file(Path(saved_draws)),
        inputs={str(p): sha256_file(p) for p in (arm_path / 'training.json', arm_path / 'features.json',
            arm_path / 'spec.json', source / 'sources.json', Path(net.__file__), Path(__file__))},
        torch_version=torch.__version__, numpy_version=np.__version__, recipes=grid, seeds=list(SEEDS),
        selection={'split': 'last_20_percent_of_original_training_by_time', 'metric': 'mean_internal_logloss',
            'stability_top': 6, 'formal_top': 4, 'primary_seed': SEEDS[0],
            'scope': 'classifier selection on cached features; not an end-to-end temporal holdout'},
        protocol=experiment._protocol_snapshot(load_settings()), llm_requests_allowed=False,
        learning_snapshot=learning_snapshot)
    bind_batch(directory, spec)
    task_state.update(directory, 'preparing_shared_training', total=len(rows))
    seal = learning_guard.verify_training(arm_path, source_spec, rows, entries, complete=True)
    local_seal = learning_guard.RunSeal(files=[*map(Path, spec['inputs']), pack_path,
        Path(saved_draws), *[baseline['dir'] / n for n in ['config.json', *baseline['config']['assets']]]])
    def check():
        seal.check()
        local_seal.check()
        material_seal.check()
    raw = net.raw_pairs(rows, entries)
    train, validation = split_training(rows, read_json(source / 'sources.json')['train'])
    labels = np.asarray([r['human_option'] == 'A' for r in rows])
    encoder = net.fit_encoder([raw[i] for i in train])
    pairs = net.encode(encoder, raw)
    write_once_json(directory / 'split.json', dict(training=[rows[i]['case_id'] for i in train],
        internal_validation=[rows[i]['case_id'] for i in validation], encoder=encoder))
    def job(key, recipe, seed, *, all_rows=False, epochs=None):
        return (key, full_encoder if all_rows else encoder, recipe, full_pairs if all_rows else pairs,
                labels, np.arange(len(rows)) if all_rows else train,
                np.asarray([], dtype=int) if all_rows else validation, seed, epochs)
    broad = train_jobs(directory / 'search', [job(k + '-s17', r, 17) for k, r in grid.items()], workers, check)
    top = sorted(grid, key=lambda k: (broad[k+'-s17']['validation']['logloss'], k))[:6]
    stability = train_jobs(directory / 'stability', [job(k+f'-s{s}', grid[k], s)
                           for k in top for s in SEEDS[1:]], workers, check)
    ranked = sorted(top, key=lambda k: (np.mean([broad[k+'-s17']['validation']['logloss'],
        *[stability[k+f'-s{s}']['validation']['logloss'] for s in SEEDS[1:]]]), k))[:4]
    selected = dict(configurations=ranked, primary=ranked[0]+'-s17',
        epochs={k: int(np.median([broad[k+'-s17']['best_epoch'],
                 *[stability[k+f'-s{s}']['best_epoch'] for s in SEEDS[1:]]])) for k in ranked})
    # Selection frozen before any formal development scoring or observations.
    write_once_json(directory / 'selection.json', selected)
    full_encoder = net.fit_encoder(raw)
    full_pairs = net.encode(full_encoder, raw)
    fitted = train_jobs(directory / 'refit', [job(k+f'-s{s}', grid[k], s, all_rows=True,
                        epochs=selected['epochs'][k]) for k in ranked for s in SEEDS], workers, check)
    results = formal_comparisons(directory, spec, fitted, full_encoder, baseline, pack, check)
    check()
    report = dict(study=directory.name, baseline=baseline['id'], data_ref=spec['data_ref'],
        generator_ref=spec['generator_ref'], training_total=len(rows), embedding_dim=8,
        search_runs=len(broad), stability_runs=len(stability), refit_runs=len(fitted),
        selected=selected, comparisons=results, llm_requests=0,
        stage='formal_initial_complete', adoption_allowed=False)
    write_once_json(directory / 'results.json', report)
    lines = ['# Embedding DNN tuning', '',
        f"Baseline: {baseline['id']}; data: {spec['data_ref']}; generator: {spec['generator_ref']}",
        '', 'Selection uses training-internal logloss; all formal rates below are INITIAL, not confirmed gains.',
        '', '| Trial | Baseline | Candidate | Initial net | Missing confirmation cases |',
        '| --- | ---: | ---: | ---: | ---: |']
    for item in sorted(results, key=lambda x: x['trial']):
        m = item['initial']
        lines.append(f"| {item['trial']} | {m['baseline_correct']}/{m['total']} | "
                     f"{m['candidate_correct']}/{m['total']} | {m['initial_net']:+d} | {m['pending_confirmation']} |")
    (directory / 'report.md').write_text('\n'.join(lines)+'\n')
    task_state.update(directory, 'formal_initial_complete', status='finished',
        completed=len(results), total=len(results),
        confirmation_pending=any(r['initial']['pending_confirmation'] for r in results))
    return report
