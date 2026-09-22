"""Offline integration acceptance: all source guards are real, only LLMs are fake.

Uses synthetic raw chat archives in tmp_path. Never reads real fixed answers,
fits a real model, calls a provider or changes production pointers.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from leakage_support import historical, mechanical
from src.iteration import training_operations as features_job
from src.iteration import training_operations as generation_job
from src import cache, llm, tracing
from src.bootstrap import history
from src.config import ConfigError, load_settings, sha256_file
from src.generator.generator import ReplyGenerator
from src.generator.history_sources import load
from src.judge import judge as judge_module, lr_retrain
from src.iteration import experiment, learning_guard as guard, promote, runner, versions
from src.iteration.storage import write_json


@pytest.fixture
def env(tmp_path, monkeypatch):
    default_template = tmp_path / 'judge_pairwise.template.md'
    default_template.write_bytes(judge_module.DEFAULT_TEMPLATE_PATH.read_bytes())
    monkeypatch.setattr(judge_module, 'DEFAULT_TEMPLATE_PATH', default_template)
    root = tmp_path / 'instance'
    for name, value in {'PRIVATE': root, 'DATA_ROOT': root / 'data',
                        'GEN_ROOT': root / 'generators', 'JUDGE_ROOT': root / 'judges',
                        'POINTERS_PATH': root / 'pointers.json'}.items():
        monkeypatch.setattr(versions, name, value)
    monkeypatch.setattr(experiment, 'EXP_ROOT', root / 'experiments')
    source = historical(root / 'data' / 'd-test')
    gdir, jdir = root / 'generators/g-0001', root / 'judges/j-0001'
    mechanical(gdir)
    # Synthetic pairwise Judge has only reviewed mechanical assets. Real Judge
    # prompt construction/scoring is exercised; no learned model is fabricated.
    mechanical(jdir)
    cfg = {'llm': {'model': 'fake-base'}, 'retriever': {'enabled': True}}
    write_json(gdir / 'config.json', {**cfg, 'data_version': 'd-test'})
    write_json(jdir / 'config.json', {'mode': 'pairwise_llm', 'llm': {'model': 'fake-judge'}})
    write_json(jdir / 'meta.json', {})
    write_json(root / 'pointers.json', {'data': 'd-test', 'production_gen': 'g-0001',
        'iteration_gen': 'g-0001', 'production_judge': 'j-0001', 'iteration_judge': 'j-0001'})
    calls, hook = [], []

    class Client:
        def __init__(self, model):
            self.model = model

        def cache_identity(self):
            return {'model': self.model, 'offline': True}

        def chat(self, messages, **kw):
            calls.append((self.model, messages))
            if hook:
                hook.pop(0)()
            return json.dumps({'ai_is': 'A'} if self.model == 'fake-judge' else
                              {'replies': ['模拟回复-' + self.model]})

    monkeypatch.setattr(llm, 'build_clients', lambda settings, cfg: Client(cfg['model']))
    monkeypatch.setattr(judge_module, 'ChatClient', lambda settings, cfg: Client(cfg['model']))
    return SimpleNamespace(root=root, source=source, gdir=gdir, jdir=jdir, cfg=cfg,
                           calls=calls, hook=hook, client=Client('fake-base'), template=default_template)


def model(env):
    return ReplyGenerator(load_settings(), env.cfg, env.client,
        prompt_root=env.gdir, pool_path=env.source.directory / 'fewshot_pool.jsonl')


@pytest.mark.parametrize('recipe', ['conversation', 'conversation_partner', 'conversation_rhythm'])
def test_reviewed_conversation_recipe_preserves_config_and_guards(env, recipe):
    from src.iteration import branches, training
    pointers = (env.root / 'pointers.json').read_bytes()
    proposal = branches.submit('conversation', 'gen', 'generic conversation action',
                               prompt_recipe=recipe)
    info = versions.load_generator(proposal['candidate_ref'])
    assert proposal['development_ref'] == 'g-0001'
    assert info['config']['llm'] == env.cfg['llm']
    assert info['config']['retriever'] == env.cfg['retriever']
    assert info['dir'].joinpath('persona.md').read_bytes() != env.gdir.joinpath('persona.md').read_bytes()
    for name in ('group_chat.md', 'friend.md'):
        assert (info['dir'] / 'scenarios' / name).read_bytes() == (env.gdir / 'scenarios' / name).read_bytes()
    guard.require_materials('d-test', [info['dir']])
    assert training.mechanical_generator('d-test', env.cfg, recipe) == info['id']
    generator = ReplyGenerator(load_settings(), info['config'], env.client,
        prompt_root=info['dir'], pool_path=env.source.directory / 'fewshot_pool.jsonl')
    case = env.source.roles['development'][0]
    prompt = json.dumps(generator.build_prompt(case), ensure_ascii=False)
    assert 'conversation_action' in prompt
    if recipe == 'conversation_partner':
        assert '<conversation_partner>' in prompt
    elif recipe == 'conversation_rhythm':
        assert '<message_rhythm>' in prompt
    assert all(answer not in prompt for answer in case['human_reply'])
    assert (env.root / 'pointers.json').read_bytes() == pointers
    assert not env.calls


@pytest.mark.parametrize('recipe', ['conversation', 'conversation_partner', 'conversation_rhythm'])
def test_conversation_recipe_cannot_be_resigned_with_answers(env, recipe):
    from src.iteration import training
    ref = training.mechanical_generator('d-test', env.cfg, recipe)
    directory = versions.generator_dir(ref)
    directory.chmod(0o755)
    persona = directory / 'persona.md'
    persona.chmod(0o644)
    persona.write_text(persona.read_text() + '\nfuture answer')
    record = json.loads((directory / 'learning.json').read_text())
    record['asset_files']['persona.md'] = sha256_file(persona)
    record['evidence_files'] = {str(persona): sha256_file(persona)}
    (directory / 'learning.json').chmod(0o644)
    write_json(directory / 'learning.json', record)
    with pytest.raises(ConfigError):
        guard.require_materials('d-test', [directory])
    assert not env.calls


def test_unknown_or_judge_prompt_recipe_rejected(env):
    from src.iteration import branches, training
    with pytest.raises(ConfigError, match='unknown reviewed'):
        training.mechanical_generator('d-test', env.cfg, '../unknown')
    with pytest.raises(ConfigError, match='generator branch'):
        branches.submit('invalid', 'judge', 'not a generator', prompt_recipe='conversation')
    assert not env.calls


def create(env):
    return experiment.create_gen_experiment('development', '离线保护验收',
        {'llm': {'model': 'fake-candidate'}}, limit=2, exp_id='guard-acceptance')


@pytest.mark.parametrize('field', ['input_cutoff', 'source_span', 'context', 'human_reply',
    'context_message_ids', 'reply_message_ids', 'case_id', 'content_sha256', 'unread', 'scenario'])
def test_forged_case_rejected_before_request(env, field):
    generator = model(env)
    case = copy.deepcopy(env.source.roles['development'][0])
    if field == 'input_cutoff':
        case[field]['timestamp'] = 999999
    elif field == 'source_span':
        case[field]['end_timestamp'] = 0
    elif field == 'context':
        case[field][-1]['text'] = 'future answer'
    elif field in ('human_reply', 'context_message_ids', 'reply_message_ids'):
        case[field] = ['forged']
    else:
        case[field] = 'forged'
    with pytest.raises(ConfigError):
        generator.generate(case)
    assert env.calls == []


def test_historical_case_cannot_downgrade_to_ordinary_generation(env):
    generator = ReplyGenerator(load_settings(), {'retriever': {'enabled': False}},
                              env.client, prompt_root=env.gdir)
    with pytest.raises(ConfigError, match='禁止降级'):
        generator.generate(env.source.roles['development'][0])
    assert env.calls == []


@pytest.mark.parametrize('field', ['timestamp', 'context', 'reply_shape', 'truncated_bubble'])
def test_pool_self_reported_approval_cannot_hide_forged_content(env, field):
    rows = copy.deepcopy(env.source.examples)
    if field == 'truncated_bubble':
        rows[0]['reply'].pop()
        rows[0]['reply_message_ids'].pop()
        rows[0]['source_span']['end'] -= 1
        rows[0]['source_span']['end_timestamp'] -= 1
    else:
        rows[0][field] = 0 if field == 'timestamp' else 'forged'
    path = env.source.directory / 'fewshot_pool.jsonl'
    history.write_rows(path, rows)
    report = json.loads((path.parent / 'report.json').read_text())
    report['examples_sha256'] = sha256_file(path)
    write_json(path.parent / 'report.json', report)
    with pytest.raises(ConfigError):
        model(env)
    assert env.calls == []


def test_past_examples_outside_training_allowed_current_future_excluded(env):
    generator = model(env)
    case = env.source.roles['development'][1]
    prompt = json.dumps(generator.build_prompt(case), ensure_ascii=False)
    # The first dev answer is observable before the second dev input, despite
    # not being a training row. It is therefore a valid retrieval example.
    previous = env.source.roles['development'][0]
    assert previous['case_id'] not in {r['case_id'] for r in env.source.roles['judge_training']}
    assert previous['human_reply'][0] in prompt
    for future in env.source.cases[4:]:
        assert all(answer not in prompt for answer in future['human_reply'])
    assert 'human_reply' not in prompt and 'human_option' not in prompt


@pytest.mark.parametrize('change', ['unknown_model', 'unknown_persona', 'nested_config', 'future'])
def test_verified_flag_cannot_certify_arbitrary_assets(env, change):
    record = json.loads((env.gdir / 'learning.json').read_text())
    if change == 'future':
        record['information_end'] = 130000
    else:
        name = {'unknown_model': 'model.json', 'unknown_persona': 'persona.md',
                'nested_config': 'scenarios/config.json'}[change]
        path = env.gdir / name
        path.write_text('unknown vocabulary / weights / future fact')
        record['asset_files'][name] = sha256_file(path)
    write_json(env.gdir / 'learning.json', record)
    with pytest.raises(ConfigError):
        create(env)
    assert env.calls == []
    assert not (env.root / 'experiments/guard-acceptance/spec.json').exists()


@pytest.mark.parametrize('change', ['role', 'pool', 'persona', 'evidence', 'template'])
@pytest.mark.parametrize('entry', ['resume', 'finished', 'promote'])
def test_unchanged_ids_cannot_reuse_changed_sources(env, change, entry):
    directory = create(env)
    runner.run_gen_experiment(directory)
    saved = (directory / 'cases.jsonl').read_bytes()
    pointers = (env.root / 'pointers.json').read_bytes()
    count = len(env.calls)
    if entry == 'resume':
        write_json(directory / 'state.json', {'status': 'interrupted'})
    if change == 'role':
        path = env.source.directory / 'dev_pool.jsonl'
        rows = json.loads('[' + ','.join(path.read_text().splitlines()) + ']')
        rows[0]['context'][-1]['text'] = 'new content with same ID'
        history.write_rows(path, rows)
    elif change == 'pool':
        path = env.source.directory / 'fewshot_pool.jsonl'
        path.write_text(path.read_text() + '\n')
    elif change == 'persona':
        (env.gdir / 'persona.md').write_text('future fact')
    elif change == 'template':
        env.template.write_text('unverified default prompt with answers')
    else:
        (env.gdir.parent / 'g-0001-source.json').write_text('changed evidence')
    with pytest.raises(ConfigError):
        if entry == 'promote':
            promote.promote_gen(directory.name)
        else:
            runner.run_gen_experiment(directory)
    assert len(env.calls) == count
    assert (directory / 'cases.jsonl').read_bytes() == saved
    assert (env.root / 'pointers.json').read_bytes() == pointers


@pytest.mark.parametrize('change', ['persona', 'evidence', 'raw', 'template', 'nested_meta'])
def test_mutation_during_request_aborts_without_valid_checkpoint(env, change):
    directory = create(env)
    path = {'persona': env.gdir / 'persona.md',
            'evidence': env.gdir.parent / 'g-0001-source.json',
            'raw': env.source.directory / 'messages.jsonl', 'template': env.template,
            'nested_meta': env.gdir / 'scenarios/meta.json'}[change]
    env.hook.append(lambda: path.write_text((path.read_text() if path.exists() else '') + '\nchanged'))
    with pytest.raises(ConfigError, match='变化'):
        runner.run_gen_experiment(directory)
    assert len(env.calls) == 1
    assert not (directory / 'cases.jsonl').read_text().strip()
    traces = [json.loads(p.read_text()) for p in (directory / 'traces').glob('*.json')]
    assert traces and all(t['status'] == 'interrupted' for t in traces)


def test_unchanged_resume_finished_and_cache_keep_valid_results(env):
    directory = create(env)
    runner.run_gen_experiment(directory)
    saved = (directory / 'cases.jsonl').read_bytes()
    count = len(env.calls)
    assert count > 0
    runner.run_gen_experiment(directory)  # finished shortcut must still validate
    write_json(directory / 'state.json', {'status': 'interrupted'})
    runner.run_gen_experiment(directory)  # full checkpoint resume
    assert len(env.calls) == count and (directory / 'cases.jsonl').read_bytes() == saved
    # Different experiment/branch, identical conditions: round zero is reused;
    # supplementary rounds 1 and 2 each cause their own independent request.
    generator, case = model(env), env.source.roles['development'][0]
    for exp in ('next', 'next-again'):
        for round_index in (0, 1, 2):
            trace = tracing.CaseTrace(env.root / 'experiments' / exp, case,
                {'dataset': 'development', 'data_ref': 'd-test'})
            with trace.operation('generation', 'candidate', round_index, {}):
                generator.generate(case)
            trace.finish('ok')
    assert len(env.calls) == count + 2


def test_generation_checkpoint_cannot_skip_source_validation(env):
    case = env.source.roles['judge_training'][0]
    spec = {'id': 'training', 'data_ref': 'd-test', 'generator_ref': 'g-0001', 'dataset': 'training'}
    spec['learning_snapshot'] = guard.snapshot('d-test', [env.gdir], 'judge_training')
    directory = env.root / 'judge_training/training'
    value = {'identity': cache.digest(spec), 'entries': {case['case_id']: {
        'status': 'ok', 'replies': ['saved'], 'input_sha256': cache.digest(case),
        'output_sha256': cache.digest(['saved'])}}}
    write_json(directory / 'generations.json', value)
    assert generation_job.generate(directory, spec, {'cases': [case]}, 1) == value
    old = (directory / 'generations.json').read_bytes()
    (env.gdir / 'persona.md').write_text('unverified future facts')
    with pytest.raises(ConfigError):
        generation_job.generate(directory, spec, {'cases': [case]}, 1)
    assert env.calls == [] and (directory / 'generations.json').read_bytes() == old


def test_unknown_training_origin_rejected_before_client_or_fitter(env, monkeypatch):
    monkeypatch.setattr(features_job, 'CodexJudgeClient', lambda *a: pytest.fail('client must not be built'))
    directory = env.root / 'judge_training/unknown'
    directory.mkdir(parents=True)
    with pytest.raises(ConfigError):
        features_job.train_features(directory, {'kind': 'legacy'}, [], 1, 0)
    import sklearn.linear_model
    monkeypatch.setattr(sklearn.linear_model.LogisticRegression, 'fit',
                        lambda *a, **kw: pytest.fail('fitter must not run'))
    with pytest.raises(ConfigError, match='旧模型模板'):
        lr_retrain.fit([{'split': 'train'} for _ in range(1200)], {}, env.jdir / 'model.json')


def test_development_checks_never_parse_fixed_answers(env, monkeypatch):
    from pathlib import Path
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path.name == 'fixed_test.jsonl':
            pytest.fail('fixed answers must remain unopened')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    directory = create(env)
    guard.verify(experiment.spec_of(directory))
    runner.run_gen_experiment(directory)


def test_unknown_default_prompt_rejected_at_creation(env):
    env.template.write_text('unverified future personal facts / reference answers')
    with pytest.raises(ConfigError, match='默认 Judge 提示词'):
        create(env)
    assert env.calls == []
    assert not (env.root / 'experiments/guard-acceptance/spec.json').exists()
