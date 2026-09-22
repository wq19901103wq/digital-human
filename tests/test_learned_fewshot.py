"""Learned selection preserves frozen features, budgets and resumable comparisons."""
from copy import deepcopy
import json

import pytest

import pytest
pytest.importorskip('torch')  # 可选重依赖
pytest.importorskip('xgboost')  # 可选重依赖
from src.config import ConfigError
from src.generator import learned_selection as selection
from src.generator import learned_sources
from src.generator.fewshot_ranker import crosses, extraction, identity_crosses, training
from src.iteration import branches, learned_gen, versions
from src.iteration.storage import read_json, write_json
from test_fewshot_ranker_training import record
from test_fewshot_selection import Renderer
from test_branches import tree as tree, isolated_legacy_provenance as isolated_legacy_provenance


def example(identity, i=1):
    return dict(record(i), id=identity)


def test_serving_reads_frozen_transport_config(tmp_path, monkeypatch):
    identity = {'model': 'frozen'}
    config = {'provider': 'codex_cli', 'timeout_seconds': 360}
    seen = []
    class Client:
        def __init__(self, value):
            seen.append(value)
        def cache_identity(self):
            return identity
    monkeypatch.setattr(selection, 'CodexJudgeClient', Client)
    write_json(tmp_path / 'model.json', {})
    write_json(tmp_path / 'provenance.json', dict(feature_identity=identity,
        feature_config={'config': config, 'workers': 16}))
    selection.LearnedSelector(tmp_path, tmp_path / 'cache')
    assert seen == [config]


def test_runtime_evidence_relocation_keeps_hashes_and_data_paths(tmp_path, monkeypatch):
    from src.config import sha256_file
    source = tmp_path / 'original/src/scorer.py'
    source.parent.mkdir(parents=True)
    source.write_text('frozen')
    checksum = sha256_file(source)
    monkeypatch.setattr(learned_sources, 'ROOT', tmp_path / 'snapshot')
    expected = {'evidence_files': {str(tmp_path / 'snapshot/src/scorer.py'): checksum,
                                   '/data/samples.json': 'data'}}
    proof = {'evidence_files': {str(source): checksum, '/data/samples.json': 'data'}}
    assert learned_sources.source_evidence_view(expected, proof, ['src/scorer.py']) == proof
    source.write_text('changed')
    with pytest.raises(ConfigError, match='changed or missing'):
        learned_sources.source_evidence_view(expected, proof, ['src/scorer.py'])
    with pytest.raises(ConfigError, match='Invalid ranker runtime'):
        learned_sources.source_evidence_view(expected, proof, ['../samples.json'])


def test_deployment_pins_inference_without_rerunning_historical_training(monkeypatch):
    runtime = {name: 'frozen' for name in learned_sources.SERVING_RUNTIME}
    runtime['scripts/train_fewshot_ranker.py'] = 'old-training'
    monkeypatch.setattr(learned_sources, 'sha256_file', lambda path: 'frozen')
    assert learned_sources.serving_runtime({'runtime': runtime}) == {
        name: 'frozen' for name in learned_sources.SERVING_RUNTIME}
    runtime['src/generator/fewshot_ranker/schema.py'] = 'changed'
    with pytest.raises(ConfigError, match='feature/scoring runtime changed'):
        learned_sources.serving_runtime({'runtime': runtime})


@pytest.mark.parametrize('evaluation_data', ['original-data', 'new-data'])
def test_verification_reconstructs_original_data_before_current_guard(tmp_path, monkeypatch, evaluation_data):
    from src.config import sha256_file
    from src.iteration import learning_guard, material_compatibility
    deployed = tmp_path / 'generator'
    model = tmp_path / 'training/model'
    write_json(model.parent / 'manifest.json', dict(runtime={}))
    write_json(deployed / 'ranker/model.json', dict(weights=[1]))
    proof = dict(model_directory=str(model), data_ref='original-data', evidence_files={},
                 model_sha256=sha256_file(deployed / 'ranker/model.json'), information_end=20)
    proof_path = deployed / 'ranker/provenance.json'
    write_json(proof_path, proof)
    record = dict(asset_files=dict(next(iter(learning_guard.GENERATOR_RECIPES.values()))),
                  information_end=20, evidence_files={})
    calls = []
    def reconstruct(directory, data, evidence):
        calls.append(('reconstruct', data))
        assert directory == str(model)
        assert evidence == proof['evidence_files']
        return proof
    def current(data, directory, loader):
        calls.append(('current', data))
        assert directory == deployed and loader is material_compatibility.ranker_sources
    monkeypatch.setattr(learned_sources, 'reconstruct', reconstruct)
    monkeypatch.setattr(material_compatibility, 'require_current', current)
    before = proof_path.read_bytes()
    learned_sources.verify(evaluation_data, deployed, record)
    assert calls == [('reconstruct', 'original-data')] + (
        [('current', 'new-data')] if evaluation_data != 'original-data' else [])
    assert proof_path.read_bytes() == before
    write_json(deployed / 'ranker/model.json', dict(weights=[2]))
    with pytest.raises(ConfigError, match='differs from completed source'):
        learned_sources.verify(evaluation_data, deployed, record)


def test_ranked_selection_budget_ties_and_duplicates():
    a, b = example('a'), example('b', 2)
    b['reply'] = ['different']
    duplicate = dict(a, id='copy')
    oversized = example('oversized')
    oversized['reply'] = ['x' * 5000]
    rows = [b, oversized, duplicate, a]
    chosen = selection.choose(rows, [1, 3, 1, 1], Renderer(), count=3, budget=1000)
    assert chosen == [a, b]
    block, _ = Renderer.render_selected([a])
    assert selection.choose([a], [1], Renderer(), count=1, budget=len(block)-1) == []
    with pytest.raises(ConfigError, match='Nonfinite'):
        selection.choose([a], [float('nan')], Renderer(), count=1, budget=1000)


def test_recall_reuses_guard_and_strict_history(monkeypatch):
    target = dict(record(10), case_id='target')
    a, b, same_second = example('a'), example('b', 2), example('same')
    same_second['source_span']['end_timestamp'] = target['input_cutoff']['timestamp']
    calls = []
    class Retriever:
        def retrieve(self, **kwargs):
            calls.append(kwargs)
            return [a, b, same_second] if len(calls) == 1 else [b, a]
    monkeypatch.setattr(selection, 'eligible', lambda row, case: True)
    assert selection.recall(Retriever(), target) == [a, b]
    assert all(c['history_case'] is target and c['limit'] == 12 and
               c['exclude_ids'] == {'target'} for c in calls)
    monkeypatch.setattr(selection, 'eligible', lambda row, case: False)
    with pytest.raises(ConfigError, match='ineligible'):
        selection.recall(Retriever(), target)


def test_recall_source_overlap_filters_complete_examples_without_more_recall(monkeypatch):
    target = dict(record(10), case_id='target')
    overlapping, same_text, shared_context = [example(name, i) for i, name in
        enumerate(('overlapping', 'same-text', 'shared-context'), 1)]
    overlapping['reply_message_ids'] = ['earlier-bubble', *target['context_message_ids']]
    shared_context['context_message_ids'] = list(target['context_message_ids'])
    calls = []
    class Retriever:
        def retrieve(self, **kwargs):
            calls.append(kwargs)
            return [overlapping, same_text] if len(calls) % 2 else [shared_context, overlapping]
    monkeypatch.setattr(selection, 'eligible', lambda row, case: True)
    original = selection.recall(Retriever(), target)
    calls.clear()
    retained = selection.recall(Retriever(), target,
        source_overlap_policy=selection.SOURCE_OVERLAP_POLICY)
    assert retained == [row for row in original if row['id'] != 'overlapping']
    assert set(row['id'] for row in retained) == {'same-text', 'shared-context'}
    assert len(calls) == 2 and all(call['limit'] == 12 for call in calls)
    assert overlapping['reply'] == same_text['reply'] == ['yes', 'detail']
    # Retained rows have identical extraction keys and requests: no new feature cache identity.
    def tasks(rows):
        return extraction.prepare([dict(target_id='target', target=target,
            candidates=[dict(example=row) for row in rows])], {'model': 'frozen'})
    before_tasks, before_refs = tasks(original)
    after_tasks, after_refs = tasks(retained)
    assert all(before_tasks[key] == value for key, value in after_tasks.items())
    assert all(before_refs[key] == value for key, value in after_refs.items())
    # The policy never weakens the existing answer/future guard, even for rows it would remove.
    monkeypatch.setattr(selection, 'eligible', lambda row, case: row is not overlapping)
    with pytest.raises(ConfigError, match='ineligible'):
        selection.recall(Retriever(), target,
            source_overlap_policy=selection.SOURCE_OVERLAP_POLICY)


def test_recall_source_overlap_policy_rejects_unknown_configuration():
    class Retriever:
        def retrieve(self, **kwargs):
            pytest.fail('Invalid policy must fail before recall')
    with pytest.raises(ConfigError, match='Unsupported learned source overlap policy'):
        selection.recall(Retriever(), {}, source_overlap_policy='typo')


@pytest.mark.parametrize('switches, message', [
    ({'learned': selection.POLICY, 'source_overlap_policy': 'typo'},
     'Unsupported learned source overlap policy'),
    ({'source_overlap_policy': selection.SOURCE_OVERLAP_POLICY},
     'source_overlap_policy requires learned few-shot selection'),
])
def test_generator_rejects_invalid_source_overlap_configuration(monkeypatch, switches, message):
    from src.generator import generator
    monkeypatch.setattr(generator, 'PersonaPromptBuilder', lambda *a, **kw: object())
    settings = {'evaluation': {'few_shots_per_case': 3, 'few_shots_char_budget': 2500}}
    with pytest.raises(ConfigError, match=message):
        generator.ReplyGenerator(settings, {'retriever': switches}, llm=None)


@pytest.mark.parametrize('policy', [None, selection.SOURCE_OVERLAP_POLICY])
def test_generator_passes_source_overlap_policy_to_shared_recall(monkeypatch, policy):
    from src.generator.generator import ReplyGenerator
    target, rows = dict(record(10), case_id='target'), [example('a')]
    generator = ReplyGenerator.__new__(ReplyGenerator)
    generator._cfg = {'retriever': {} if policy is None else {'source_overlap_policy': policy}}
    generator._retriever = Renderer()
    generator._max_shots, generator._budget = 3, 2500
    generator._selection, generator._reranker = None, None
    generator._check_sources = lambda: None
    calls = []
    def recall(retriever, case, *, source_overlap_policy):
        assert retriever is generator._retriever and case is target
        calls.append(source_overlap_policy)
        return rows
    class Selector:
        def select(self, case, recalled, *, count, budget, retriever, check):
            assert case is target and recalled is rows
            assert count == 3 and budget == 2500 and retriever is generator._retriever
            check()
            return recalled
    generator._learned = Selector()
    monkeypatch.setattr(selection, 'recall', recall)
    assert generator._style_block(target) == Renderer.render_selected(rows, max_chars=2500)[0]
    assert calls == [policy]


def test_serving_features_match_training_and_ignore_target_answers():
    target, rows = dict(record(10), case_id='target'), [example('a'), example('b', 2)]
    groups = [dict(target_id='target', target=target, candidates=[{'example': r} for r in rows])]
    tasks, refs = extraction.prepare(groups, {'model': 'frozen'})
    assert len(tasks) == 5  # One target, two independent sides per historical example.
    assert 'TARGET_SECRET' not in json.dumps(tasks) and 'GENERATED_SECRET' not in json.dumps(tasks)
    values = {k: {field: 'unknown' for field in v['schema']['properties']} for k, v in tasks.items()}
    expected = []
    for row in training.assemble(groups, refs, values):
        row = crosses.expand(row)
        row['id_cross.chat_id'] = identity_crosses.pair(row, 'target.chat_id', 'example_context.chat_id')
        expected.append(row)
    assert selection.feature_rows(target, rows, refs, values) == expected
    changed = deepcopy(target)
    changed.update(human_reply=['other answer'], generated_reply=['other generation'], z=0)
    assert selection.feature_rows(changed, rows, refs, values) == expected


def test_serving_action_crosses_require_explicit_model_transform(monkeypatch):
    from src.generator.fewshot_ranker import action_crosses
    target, rows = dict(record(10), case_id='target'), [example('a')]
    tasks, refs = extraction.prepare([dict(target_id='target', target=target,
        candidates=[dict(example=row) for row in rows])], {'model': 'frozen'})
    values = {key: {field: 'unknown' for field in task['schema']['properties']}
              for key, task in tasks.items()}
    old = selection.feature_rows(target, rows, refs, values)
    calls = []
    def expand(row):
        calls.append(row)
        return {**row, 'action_cross.test': 'added'}
    monkeypatch.setattr(action_crosses, 'expand', expand)
    assert selection.feature_rows(target, rows, refs, values,
        transform=learned_sources.TRANSFORM) == old
    assert calls == []
    assert selection.feature_rows(target, rows, refs, values,
        transform={**learned_sources.TRANSFORM, 'reply_action_crosses': action_crosses.VERSION}) == [
            {**row, 'action_cross.test': 'added'} for row in old]
    assert calls == old  # The new crosses follow, and retain, the existing semantic/identity crosses.
    with pytest.raises(ConfigError, match='Unsupported reply action feature transform'):
        selection.feature_rows(target, rows, refs, values,
            transform={'reply_action_crosses': 'unknown'})


@pytest.mark.parametrize('approved', [True, False])
@pytest.mark.parametrize('workers', [16, 48])
@pytest.mark.parametrize('source_policy', [None, selection.SOURCE_OVERLAP_POLICY])
def test_precompute_initializes_history_filter_before_recall(tmp_path, monkeypatch, approved, workers,
                                                          source_policy):
    from types import SimpleNamespace
    from src.iteration.parallel import completed_map
    events = []
    class Retriever:
        def __init__(self, **kwargs):
            pass
        def is_approved(self):
            events.append('history_ready')
            return approved
    selector = SimpleNamespace(cache=tmp_path, client=None, tasks=lambda *a: ({}, {}))
    def recall(retriever, case, *, source_overlap_policy):
        assert events == ['history_ready']
        assert source_overlap_policy == source_policy
        events.append('recall')
        return []
    monkeypatch.setattr(learned_gen, 'PersonaFewShotRetriever', Retriever)
    monkeypatch.setattr(learned_gen, 'LearnedSelector', lambda *a: selector)
    monkeypatch.setattr(learned_gen, 'recall', recall)
    monkeypatch.setattr(learned_gen.experiment, 'spec_of', lambda *a: {'data_ref': 'd-0001'})
    monkeypatch.setattr(versions, 'data_version_dir', lambda *a: tmp_path)
    monkeypatch.setattr(versions, 'generator_dir', lambda *a: tmp_path)
    monkeypatch.setattr(versions, 'load_generator', lambda *a: {
        'config': {'retriever': {'source_overlap_policy': source_policy}}})
    monkeypatch.setattr(learned_gen.datasets, 'rows_for', lambda *a: [{}])
    def extract(*args, **kwargs):
        assert kwargs['workers'] == workers
        assert sorted(completed_map(lambda x: x, range(50), workers)) == list(range(50))
        events.append('extract')
    monkeypatch.setattr(learned_gen.extraction, 'run', extract)
    monkeypatch.setattr(learned_gen.control, 'policy', lambda: {'max_active_requests': workers})
    if approved:
        learned_gen.precompute(tmp_path, 'g-test', tmp_path, workers)
        assert events == ['history_ready', 'recall', 'extract']
        progress = learned_gen.status(tmp_path)['recall']
        assert progress['phase'] == 'complete'
        assert progress['completed'] == progress['total'] == 1
    else:
        with pytest.raises(ConfigError, match='not approved'):
            learned_gen.precompute(tmp_path, 'g-test', tmp_path, workers)
        assert events == ['history_ready']
    with pytest.raises(ValueError, match='workers'):
        list(completed_map(str, [1], 17))
    with pytest.raises(ConfigError, match='concurrency limit'):
        learned_gen.precompute(tmp_path, 'g-test', tmp_path, workers + 1)


def test_explicit_worker_limit_is_parallel_and_restored():
    from threading import Barrier, get_ident
    from src.iteration.parallel import completed_map, worker_limit
    barrier = Barrier(17)
    def work(item):
        barrier.wait(timeout=10)
        return get_ident()
    with pytest.raises(RuntimeError, match='restore'):
        with worker_limit(17):
            assert len(set(completed_map(work, range(17), 17))) == 17
            raise RuntimeError('restore')
    with pytest.raises(ValueError, match='workers'):
        list(completed_map(str, [1], 17))


@pytest.mark.usefixtures('isolated_legacy_provenance')
def test_frozen_candidate_resume_reuses_version_and_formal_stage(tree, monkeypatch, tmp_path):
    base = versions.load_pointers()['production_gen']
    cfg = deepcopy(versions.load_generator(base)['config'])
    cfg['retriever'] = {'enabled': True, 'learned': selection.POLICY}
    candidate = versions.create_generator_version(cfg, 'd-0001')
    def forbidden(*args, **kwargs):
        pytest.fail('A frozen candidate must not be redeployed or versioned')
    monkeypatch.setattr(learned_gen.learned_sources, 'deploy', forbidden)
    monkeypatch.setattr(versions, 'create_generator_version', forbidden)
    args = dict(candidate=candidate, base=base, name='frozen', data='d-0001', stage='fixed_test')
    output = tmp_path / 'workflow'
    assert learned_gen.prepare(output, **args) == candidate
    assert learned_gen.prepare(output, **args) == candidate
    state = read_json(tree / 'branches/frozen/state.json')
    proposal = read_json(tree / 'branches/frozen/revisions/v-0001.json')
    assert state['stage_limit'] == 'fixed_test'
    assert proposal['candidate_ref'] == candidate and proposal['development_ref'] == base
    assert proposal['basis']['data'] == 'd-0001'
    assert len(list((tree / 'branches/frozen/revisions').glob('*.json'))) == 1
    for changes in ({'base': candidate}, {'data': 'd-0002'}, {'stage': 'development'},
                    {'candidate': base}, {'model': tmp_path / 'model'}):
        with pytest.raises(ConfigError):
            learned_gen.prepare(output, **{**args, **changes})
    with pytest.raises(ConfigError, match='Production baseline changed'):
        learned_gen.prepare(tmp_path / 'other', **{**args, 'name': 'other', 'base': candidate})
    # A completed run can still be reopened after its production promotion.
    versions.save_pointers({**versions.load_pointers(), 'production_gen': candidate})
    assert learned_gen.prepare(output, **args) == candidate


@pytest.mark.usefixtures('isolated_legacy_provenance')
def test_workflow_resume_keeps_one_revision_and_versions_ranker(tree, monkeypatch, tmp_path):
    base = versions.load_pointers()['production_gen']
    basis = branches.basis()
    write_json(tree / 'branches/source/state.json', {'development': {'candidate_ref': base, 'basis': basis}})
    source = versions.generator_dir(base)
    def deploy(model, base_dir, data, output):
        import shutil
        output.mkdir(parents=True, exist_ok=True)
        for path in source.rglob('*'):
            dest = output / path.relative_to(source)
            if path.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                shutil.copyfile(path, dest)
        write_json(output / 'ranker/model.json', {'model': 'trained'})
        write_json(output / 'ranker/provenance.json', {'proof': 'synthetic'})
        return output
    monkeypatch.setattr(learned_gen.learned_sources, 'deploy', deploy)
    monkeypatch.setattr(branches, '_accepted_source', lambda *a: {'candidate_ref': base, 'basis': basis})
    args = dict(model=tmp_path / 'model', base=base, source_branch='source', name='learned', data='d-0001')
    first = learned_gen.prepare(tmp_path / 'workflow', **args)
    assert learned_gen.prepare(tmp_path / 'workflow', **args) == first
    assert len(list((tree / 'branches/learned/revisions').glob('*.json'))) == 1
    assert read_json(versions.generator_dir(first) / 'ranker/model.json') == {'model': 'trained'}
    cfg = versions.load_generator(first)['config']
    assert cfg['llm'] == versions.load_generator(base)['config']['llm']
    assert cfg['retriever'] == {'enabled': True, 'learned': selection.POLICY}
    branches.submit('unrelated', 'gen', 'separate experiment', overrides={'llm': {'model': 'other'}})
    jobs = branches.advance('learned')
    assert jobs and all('learned' in job['id'] for job in jobs)
    assert read_json(tree / 'branches/unrelated/state.json')['rounds'] == []
    with pytest.raises(ConfigError, match='baseline changed'):
        learned_gen.prepare(tmp_path / 'other', **{**args, 'base': first})
