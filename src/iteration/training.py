"""Instance-configured history training, from material construction to audited candidates."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter

import numpy as np

from .. import cache, tracing
from ..config import ConfigError, ROOT, load_settings, settings_path, sha256_file
from ..generator.history_sources import load
from ..judge import lr_retrain as lr, corrected_v1 as rt
from ..judge.corrected import CodexJudgeClient, MODE, RUNTIME_PATH, blind_case
from . import branches, control, datasets, experiment, learning_guard, runtime, task_state, versions
from . import training_evidence as audit, training_operations as operations
from .storage import file_lock, write_json, read_json as read, write_once_json as save_once

TEMPLATES = ROOT / 'prompts/mechanical'
PERSONA = (TEMPLATES / 'persona.md').read_text()
SCENARIOS = {p.name: p.read_text() for p in (TEMPLATES / 'scenarios').glob('*.md')}
DEFAULT_RECIPE = dict(fit_intercept=False, C=1., penalty='l2', solver='lbfgs',
                      max_iter=5000, tol=1e-4, random_state=0)


def blob(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode()


def hashes(paths):
    return {str(p.resolve()): sha256_file(p) for p in paths}


def mechanical_generator(data_ref, config, recipe='mechanical'):
    """Create a source-free generator snapshot; retrieval still uses verified history."""
    if recipe not in learning_guard.GENERATOR_RECIPES:
        raise ConfigError('unknown reviewed generator recipe')
    assets = learning_guard.GENERATOR_RECIPES[recipe]
    sources = {name: (ROOT / 'prompts' / recipe / name if name == 'persona.md'
                      else TEMPLATES / name) for name in assets}
    if any(not p.is_file() or sha256_file(p) != assets[name] for name, p in sources.items()):
        raise ConfigError('reviewed generator instruction recipe changed')
    identity = config if recipe == 'mechanical' else {'config': config, 'recipe': recipe}
    directory = versions.PRIVATE / 'material_recipes' / cache.digest(identity)
    directory.mkdir(parents=True, exist_ok=True)
    for name, p in sources.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != p.read_bytes():
            raise ConfigError('mechanical instruction recipe changed')
        target.write_bytes(p.read_bytes())
    save_once(directory / 'learning.json', {'schema': 2, 'verified': True, 'information_end': 0,
        'asset_files': assets,
        'evidence_files': hashes([directory / 'persona.md', *(directory / 'scenarios').glob('*.md')])})
    return versions.create_generator_version(config, data_ref, source_dir=directory)


def mechanical_judge(config):
    from ..judge.judge import DEFAULT_TEMPLATE_PATH
    learning_guard.require_default_judge_template(DEFAULT_TEMPLATE_PATH)
    directory = versions.PRIVATE / 'material_recipes' / cache.digest({'judge': config})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'prompt.md'
    if path.exists() and path.read_bytes() != DEFAULT_TEMPLATE_PATH.read_bytes():
        raise ConfigError('generic judge instructions changed')
    path.write_bytes(DEFAULT_TEMPLATE_PATH.read_bytes())
    record = {'schema': 2, 'verified': True, 'information_end': 0,
              'asset_files': {'prompt.md': learning_guard.GENERIC_JUDGE_TEMPLATE}, 'evidence_files': hashes([path])}
    return versions.create_judge_version({'mode': 'pairwise_llm', 'llm': config,
        'prompt_sha256': sha256_file(path)}, {'origin': 'generic_instructions'},
        assets={'prompt.md': path.read_bytes(), 'learning.json': blob(record)})


def context_schema(cases):
    names, members = Counter(), Counter()
    for case in cases:
        _, meta = blind_case(case, [], [])
        if meta['relationship'] == 'group':
            names[str(meta['group_name'])] += 1
            members.update((str(meta['source_chat_id']), str(s)) for s in meta['recent_speakers'][:3] if s != '__self__')
    return {'known_group_names': sorted(n for n, c in names.items() if c >= 5),
            'known_group_members': [list(m) for m, c in sorted(members.items()) if c >= 8]}


def model_template(cases, recipe=None):
    recipe = recipe or DEFAULT_RECIPE
    schema = context_schema(cases)
    dummy = {**dict.fromkeys(rt.INTEGER_FIELDS, 0), **dict.fromkeys(rt.BOOLEAN_FIELDS, False),
             **{k: v[0] for k, v in rt.ENUM_FIELDS.items()}}
    _, names = rt.vectorize_boolean_option(dummy)
    option_names = [*names, *rt.OBSERVABLE_OPTION_BOOLEAN_NAMES, *rt.GROUP_PATTERN_OPTION_BOOLEAN_NAMES]
    _, meta = blind_case(cases[0], [], [])
    context, context_names = rt.vectorize_context_booleans(meta, schema)
    vector, names = rt.expand_boolean_option_with_refined_context(np.zeros(len(option_names)), option_names,
                                                                  context, context_names)
    return {'schema_version': 1, 'selected_model_name': 'symmetric_logistic_refined_boolean_crosses',
        'feature_system': {'context_schema': schema, 'option_feature_names': option_names,
                          'context_feature_names': context_names, 'linear_input_feature_count': len(names)},
        'final_model': {'model': 'LogisticRegression', 'input_feature_count': len(names),
                        'feature_names': names, 'coefficients': vector.tolist(), **recipe}}


def clean_prompt(reference, profile):
    prompt = rt.build_shared_single_case_prompt({}, reference, profile)
    prompt = prompt.replace('<blind_case>\n{}\n</blind_case>', '<blind_case>\n{{case}}\n</blind_case>')
    return prompt.replace('参考资料来自与测试聊天隔离的 2026-04-01 前真人聊天，以及用户确认的本人事实。',
        '参考示例仅来自本次开发时间边界之前的可核验历史聊天，不含留出聊天或额外个人事实。')


@versions.transaction
def submit(name, plan):
    branches._name(name)
    required = {'data_ref', 'generator_ref', 'feature_configs', 'scoring', 'change'}
    if not required <= plan.keys() or not str(plan['change']).strip():
        raise ConfigError('training requires explicit data, generator, feature models, scoring and human change')
    if not plan['feature_configs']:
        raise ConfigError('at least one feature arm is required')
    for arm in plan['feature_configs']:
        branches._name(arm)
        if arm in {'development', 'comparison', 'generator_source'}:
            raise ConfigError('reserved training arm name')
    recipe = {**DEFAULT_RECIPE, **plan.get('recipe', {})}
    if set(recipe) != set(DEFAULT_RECIPE) or recipe['fit_intercept'] is not False:
        raise ConfigError('unsupported fitting recipe; the stored model has no intercept')
    if plan.get('branch_name') and plan.get('candidate_arm') not in plan['feature_configs']:
        raise ConfigError('branch submission requires the human-selected candidate arm')
    directory = versions.PRIVATE / 'judge_training' / name
    if directory.exists():
        raise ConfigError('study already exists; resume the same frozen study')
    directory.mkdir(parents=True)
    save_once(directory / 'proposal.json', {**plan, 'recipe': recipe})
    runtime.freeze(directory)
    task_state.update(directory, 'queued', status='queued')
    return directory


def prepare(directory):
    if (directory / 'spec.json').exists():
        spec = read(directory / 'spec.json')
        runtime.verify_inputs(spec['inputs'])
        return spec
    plan = read(directory / 'proposal.json')
    data_ref = plan['data_ref']
    data = versions.data_version_dir(data_ref)
    history = load(data)
    train, dev = history.role('judge_training'), history.role('judge_development')
    if len(train) < 2 or not dev:
        raise ConfigError('training needs at least two cases and a disjoint development set')
    # Representative preflight cases are followed by the remaining frozen order.
    first = [next(c for c in train if c['chat_type'] == kind) for kind in sorted({c['chat_type'] for c in train})]
    train = first + [c for c in train if c['case_id'] not in {r['case_id'] for r in first}]
    learning_guard.require_materials(data_ref, [versions.generator_dir(plan['generator_ref'])])
    reference_ids = plan.get('reference_case_ids', [])
    known = {c['case_id']: c for c in history.role('gen_learning') + train}
    if len(reference_ids) != len(set(reference_ids)) or not set(reference_ids) <= known.keys():
        raise ConfigError('static reference IDs must belong to verified learning purposes')
    examples, spans = [], []
    for cid in reference_ids:
        c = known[cid]
        span = c['source_span']
        examples.append({'relationship': c['chat_type'], 'context': [m['text'] for m in c['context']],
                         'human_reply': c['human_reply']})
        spans.append({**span, 'message_ids': c['context_message_ids'] + c['reply_message_ids'],
                      'information_end': span['end_timestamp']})
    reference = {'facts': [], 'examples': examples}
    information_end = max(c['source_span']['end_timestamp'] for c in train + [known[c] for c in reference_ids])
    save_once(directory / 'sources.json', {'train': train, 'development': dev})
    save_once(directory / 'reference.json', reference)
    profile_path = directory / 'profile.json'
    profile_path.write_bytes((TEMPLATES / 'profile.json').read_bytes())
    if sha256_file(profile_path) != learning_guard.GENERIC_PROFILE:
        raise ConfigError('unreviewed generic profile')
    save_once(directory / 'model_template.json', model_template(train, plan['recipe']))
    (directory / 'prompt.md').write_text(clean_prompt(reference, read(profile_path)))
    save_once(directory / 'source_audit.json', {'reference_spans': spans,
        'training_information_end': information_end, 'data_ref': data_ref})
    material_files = [p for p in directory.glob('*') if p.is_file() and p.name not in
                      ('state.json', 'control.json', 'runtime.json', 'scheduler.log')]
    gen = versions.generator_dir(plan['generator_ref'])
    inputs = hashes([*history.files, *material_files, *[p for p in gen.rglob('*') if p.is_file()],
                     *[p for p in runtime.files(ROOT) if p.suffix == '.py'], settings_path()])
    spec = {'id': directory.name, 'kind': 'history_lr_training', 'dataset': 'training',
        'data_ref': data_ref, 'generator_ref': plan['generator_ref'], 'source_judge': plan.get('source_judge'),
        'training_total': len(train), 'evaluation_total': len(dev), 'inputs': inputs,
        'purpose_snapshot': datasets.snapshot(data), 'recipe': plan['recipe'],
        'feature_configs': plan['feature_configs'], 'scoring': plan['scoring'],
        'information_end': information_end, 'protocol': experiment._protocol_snapshot(load_settings()),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'adoption_allowed': False}
    runtime.archive_inputs(inputs)
    save_once(directory / 'spec.json', spec)
    evidence = audit.Evidence()
    try:
        audit.sources(directory, spec, evidence)
    finally:
        evidence.store.db.close()
    return spec


def generate(directory, spec, rows, workers, limit=0):
    snapshot = {'cases': rows, 'retrieval_excluded_ids_by_chat':
                {c['source_message_id'].rpartition(':')[0]: [] for c in rows}}
    return operations.generate(directory, spec, snapshot, workers, limit)


def preflight(directory, spec, train):
    generated = generate(directory, spec, train, 2, 2)
    rows = [r for r in audit.training_rows(train, generated) if r['case_id'] in {c['case_id'] for c in train[:2]}]
    if len(rows) != 2:
        raise ConfigError('generation preflight incomplete')
    for arm, config in spec['feature_configs'].items():
        target = directory / arm
        path = target / 'preflight.json'
        value = read(path) if path.exists() else {'identity': cache.digest(spec), 'entries': {}}
        if value['identity'] != cache.digest(spec):
            raise ConfigError('preflight identity changed')
        for row in rows:
            if row['case_id'] in value['entries']:
                continue
            runtime.verify_inputs(spec['inputs'])
            trace = tracing.CaseTrace(target, {'case_id': row['case_id'], 'blind': row['blind']}, spec)
            with trace.operation('feature_extraction', 'training', 0, {}):
                features = lr.extract(CodexJudgeClient(config), row['blind'], lambda: runtime.verify_inputs(spec['inputs']))
            trace.finish('ok')
            value['entries'][row['case_id']] = {'status': 'ok', 'input_sha256': cache.digest(row),
                'trace_ref': trace.ref, 'features': features, 'features_sha256': cache.digest(features)}
            write_json(path, value)
    evidence = audit.Evidence()
    try:
        selected = audit.sources(directory, spec, evidence)
        pool = {r['id']: r for r in audit.lines(versions.data_version_dir(spec['data_ref']) / 'fewshot_pool.jsonl')}
        with audit.generation_cache(evidence, directory):
            outputs = audit.generations(directory, spec, selected['train'], pool, evidence, False)
        for arm in spec['feature_configs']:
            audit.training(directory, spec, train, outputs, arm, evidence, False)
    finally:
        evidence.store.db.close()
    write_json(directory / 'preflight_audit.json', {'status': 'passed', 'spec_sha256': sha256_file(directory / 'spec.json')})


def train_arm(directory, spec, arm, rows, workers):
    target = directory / arm
    fspec = {**spec, 'id': directory.name + '-' + arm, 'feature_config': spec['feature_configs'][arm],
             'training_sha256': cache.digest(rows)}
    save_once(target / 'spec.json', fspec)
    save_once(target / 'training.json', {'rows': rows})
    if not (target / 'features.json').exists():
        checkpoint = audit.checkpoint(target, fspec, rows)
        checkpoint['entries'] = read(target / 'preflight.json')['entries']
        write_json(target / 'features.json', checkpoint)
    if not operations.train_features(target, fspec, rows, workers, 0):
        raise ConfigError('features incomplete; fitting a smaller set is forbidden')
    artifact = target / 'correction.json'
    if not artifact.exists():
        task_state.update(target, 'fitting', total=len(rows))
        fitted = lr.fit(rows, audit.checkpoint(target, fspec, rows)['entries'], directory / 'model_template.json')
        result = read(directory / 'model_template.json')
        result['final_model']['coefficients'] = fitted.coef_[0].tolist()
        result['training'] = {'cases': len(rows), 'training_sha256': cache.digest(rows),
            'features_sha256': sha256_file(target / 'features.json'), 'evaluation_used_for_fit': False,
            'data_ref': spec['data_ref'], 'feature_config': fspec['feature_config'], 'recipe': spec['recipe']}
        save_once(artifact, result)
    if not (target / 'candidate.json').exists():
        assets = {name: (directory / name).read_bytes() for name in ('prompt.md', 'profile.json', 'reference.json')}
        assets['correction.json'] = artifact.read_bytes()
        assets['source_judge.json'] = blob({'origin': 'history_lr_v1', 'study': directory.name, 'arm': arm})
        assets['learning.json'] = blob({'schema': 2, 'verified': True, 'information_end': spec['information_end'],
            'asset_files': {k: hashlib.sha256(v).hexdigest() for k, v in assets.items()},
            'evidence_files': hashes([directory / 'spec.json', directory / 'source_audit.json', directory / 'sources.json',
                                      target / 'training.json', target / 'features.json', artifact])})
        cfg = {'mode': MODE, 'llm': spec['scoring']['initial'], 'feature_llm': fspec['feature_config'],
            'runtime_sha256': sha256_file(RUNTIME_PATH), 'correction_threshold': spec['scoring']['correction_threshold'],
            'decision_policy': spec['scoring']['decision_policy'],
            'assets': {k: hashlib.sha256(v).hexdigest() for k, v in assets.items() if k != 'prompt.md'},
            'prompt_sha256': hashlib.sha256(assets['prompt.md']).hexdigest()}
        ref = versions.create_judge_version(cfg, {'origin': 'history_lr_v1', 'study': directory.name}, assets=assets)
        save_once(target / 'candidate.json', {'judge_ref': ref, 'artifact_sha256': sha256_file(artifact)})
    ref = read(target / 'candidate.json')['judge_ref']
    learning_guard.require_materials(spec['data_ref'], [versions.judge_dir(ref)['dir']])
    task_state.update(target, 'finished', status='finished', candidate_ref=ref)
    return ref


def pending():
    return [{'kind': 'training', 'id': p.parent.name} for p in sorted((versions.PRIVATE / 'judge_training').glob('*/proposal.json'))
            if read(p.parent / 'state.json').get('status') not in ('finished', 'needs_attention')]


def run(name, workers=2):
    directory = versions.PRIVATE / 'judge_training' / branches._name(name)
    with file_lock(directory / '.run.lock', blocking=False), control.job(directory):
        try:
            runtime.require_current(directory)
            # A cache store must exist even before the independent read-only preflight audit.
            cache.get_store(versions.PRIVATE / '.cache')
            task_state.update(directory, 'preparing')
            spec = prepare(directory)
            selected = read(directory / 'sources.json')
            task_state.update(directory, 'preflight')
            preflight(directory, spec, selected['train'])
            task_state.update(directory, 'generating')
            outputs = generate(directory, spec, selected['train'], workers)
            rows = audit.training_rows(selected['train'], outputs)
            if len(rows) != spec['training_total']:
                raise ConfigError('training generation incomplete')
            refs = {}
            for arm in spec['feature_configs']:
                task_state.update(directory, 'training', current_arm=arm)
                refs[arm] = train_arm(directory, spec, arm, rows, workers)
            task_state.update(directory, 'generating_development')
            dev_spec = {**spec, 'dataset': 'development'}
            dev = generate(directory / 'development', dev_spec, selected['development'], workers)
            if any(dev['entries'].get(c['case_id'], {}).get('status') != 'ok' for c in selected['development']):
                raise ConfigError('development generation incomplete')
            pack_rows = [{**c, 'ai_replies': dev['entries'][c['case_id']]['replies']} for c in selected['development']]
            pack_ref = 'pack-calibration-' + name
            save_once(versions.PRIVATE / 'judge_eval' / pack_ref / 'pack.json', {
                'data_ref': spec['data_ref'], 'c0_gen_version': spec['generator_ref'], 'rows': pack_rows,
                'generation_proof': learning_guard.pack_proof(dev_spec, pack_rows)})
            task_state.update(directory, 'independent_audit')
            for ref in refs.values():
                learning_guard.require_materials(spec['data_ref'], [versions.judge_dir(ref)['dir']])
            evidence = audit.Evidence()
            try:
                pool = {r['id']: r for r in audit.lines(versions.data_version_dir(spec['data_ref']) / 'fewshot_pool.jsonl')}
                with audit.generation_cache(evidence, directory / 'development'):
                    audit.generations(directory / 'development', dev_spec, selected['development'], pool, evidence, True)
            finally:
                evidence.store.db.close()
            save_once(directory / 'audit.json', {'status': 'passed', 'candidates': refs,
                'spec_sha256': sha256_file(directory / 'spec.json'), 'development_pack': pack_ref})
            plan = read(directory / 'proposal.json')
            if plan.get('branch_name'):
                try:
                    receipt = directory / 'branch_submission.json'
                    if not receipt.exists():
                        proposal = branches.submit(plan['branch_name'], 'judge', plan['change'],
                            candidate_ref=refs[plan['candidate_arm']], development_pack=pack_ref,
                            validation_pack=plan.get('validation_pack'))
                        save_once(receipt, {'branch': plan['branch_name'], 'revision': proposal['revision'],
                                            'candidate_ref': refs[plan['candidate_arm']]})
                    elif read(receipt)['candidate_ref'] != refs[plan['candidate_arm']]:
                        raise ConfigError('prior branch handoff differs from the frozen candidate')
                except ConfigError as exc:
                    task_state.update(directory, 'baseline_review', status='needs_attention',
                                      reason=str(exc), candidates=refs, audit='passed')
                    return
            task_state.update(directory, 'finished', status='finished', candidates=refs, adopted=False)
        except control.StopRequested:
            raise
        except BaseException as exc:
            task_state.update(directory, 'stopped', status='stopped', error=f'{type(exc).__name__}: {exc}')
            raise
