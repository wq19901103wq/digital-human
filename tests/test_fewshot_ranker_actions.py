"""Cached action crosses retain binary supervision and deployment provenance."""
from copy import deepcopy

import numpy as np
import pytest

from src.config import ConfigError, sha256_file
from src.generator import learned_sources, ranker_report
from src.generator.fewshot_ranker import action_comparison as comparison, action_crosses, boosting
from src.generator.fewshot_ranker.architecture_comparison import feature_rows
from src.generator.history_sources import digest
from src.iteration.storage import read_json, write_json
from tests.test_fewshot_ranker_lr import feature_row, frozen_source
from tests.test_fewshot_ranker_training import synthetic


def control_model(groups, observations, rows):
    model, scores, detail = boosting.train(groups, observations, rows,
        [dict(boosting.recipes()[0], max_rounds=2, min_child_weight=0)])
    model['feature_transform'] = comparison.CONTROL_TRANSFORM
    return model, scores, detail


def frozen_control(source, reference):
    manifest, groups, observations, _, _, _ = ranker_report.load_completed(source)
    rows = feature_rows(read_json(source/'feature_matrix.json')['rows'], True)
    model, scores, detail = control_model(groups, observations, rows)
    binding = dict(kind='cached_fewshot_architecture_comparison',
        source_manifest_sha256=digest(manifest),
        source_completed_sha256=sha256_file(source/'completed.json'), dataset=manifest['dataset'])
    write_json(reference/'manifest.json', binding)
    child = reference/comparison.METHOD
    child_binding = dict(parent_sha256=digest(binding), method=comparison.METHOD,
                         feature_transform=comparison.CONTROL_TRANSFORM)
    write_json(child/'manifest.json', child_binding)
    write_json(child/'model.json', model)
    write_json(child/'training.json', detail)
    write_json(child/'predictions.json', dict(rows=[dict(r, logit=float(s))
        for r, s in zip(observations, scores)]))
    names = ['model.json', 'training.json', 'predictions.json']
    comparison.seal(child, child_binding, names)
    comparison.seal(reference, binding,
        [f'{comparison.METHOD}/{name}' for name in [*names, 'manifest.json', 'completed.json']])
    return child


def test_action_crosses_are_nineteen_local_categorical_fields():
    row = feature_row()
    row.update({'target.last_self_action': 'agree', 'example_reply.first_bubble_action': 'ask',
                'target.needs_empathy': 'yes', 'example_context.needs_empathy': 'no'})
    before = deepcopy(row)
    expanded = action_crosses.expand(row)
    assert row == before and all(expanded[k] == v for k, v in row.items())
    assert len(set(expanded)-set(row)) == 19
    assert all(isinstance(v, str) for v in expanded.values())
    assert expanded['action_cross.last_self_action_x_reply_first_bubble_action'] == '["agree","ask"]'
    assert expanded['action_cross.equal_context_needs_empathy'] == 'False'
    assert expanded['action_cross.equal_context_self_stance'] == 'unknown'
    row['target.last_self_action'] = 'unregistered_category'
    with pytest.raises(ConfigError, match='invalid reply-action'):
        action_crosses.expand(row)
    del row['target.last_self_action']
    with pytest.raises(ConfigError, match='Missing or invalid'):
        action_crosses.expand(row)


def test_action_fit_preserves_recipe_rounds_binary_labels_and_outer_isolation():
    groups, rows, observations = synthetic()
    original = feature_rows([{**feature_row(), **row} for row in rows], True)
    control, _, control_detail = control_model(groups, observations, original)
    expanded = [action_crosses.expand(row) for row in original]
    model, scores, detail = comparison.fit_selected(groups, observations, expanded, control, control_detail)
    assert model['recipe'] == control['recipe'] and model['rounds'] == control['rounds']
    assert not model.get('supervision') and detail['supervision'] == 'original_binary_single_draw'
    assert model['feature_transform'] == action_crosses.TRANSFORM
    assert 'identity=15' not in model['vocabulary']
    np.testing.assert_allclose(boosting.score_document(model, expanded), scores)
    changed = deepcopy(observations)
    for row in changed:
        if row['split'] == 'validation':
            row['z'] = 1-row['z']
    other, _, other_detail = comparison.fit_selected(groups, changed, expanded, control, control_detail)
    assert model == other and detail == other_detail
    with pytest.raises(ConfigError, match='binary XGB'):
        comparison.fit_selected(groups, observations, expanded,
                                dict(control, supervision='graded'), control_detail)
    with pytest.raises(ConfigError, match='recipe differ'):
        comparison.fit_selected(groups, observations, expanded, control,
                                dict(control_detail, selected_rounds=999))


def test_action_run_reuses_saved_control_and_resume_without_llm(tmp_path, monkeypatch):
    from src.judge.corrected import CodexJudgeClient
    from src.llm import ChatClient
    def forbidden(*args, **kwargs):
        raise AssertionError('cached action comparison must not request LLMs or refit saved work')
    monkeypatch.setattr(CodexJudgeClient, 'run', forbidden)
    monkeypatch.setattr(ChatClient, 'chat', forbidden)
    source = frozen_source(tmp_path, monkeypatch)
    reference, output = tmp_path/'reference', tmp_path/'actions'
    control = frozen_control(source, reference)
    control_hash = sha256_file(control/'model.json')
    result = comparison.run(source, reference, output)
    assert result['status'] == 'complete' and result['new_model_requests'] == 0
    assert len(result['added_fields']) == 19
    assert sha256_file(control/'model.json') == control_hash
    model = output/comparison.METHOD
    parent, child = read_json(output/'manifest.json'), read_json(model/'manifest.json')
    assert learned_sources.deployment_recipe(parent, child, reference, model, set()) == action_crosses.TRANSFORM
    tampered = dict(child, control_model_sha256='changed')
    with pytest.raises(ConfigError, match='control artifact differs'):
        learned_sources.deployment_recipe(parent, tampered, reference, model, set())
    changed = read_json(model/'model.json')
    write_json(model/'model.json', dict(changed, rounds=999))
    with pytest.raises(ConfigError, match='changed labels, recipe or stopping'):
        learned_sources.deployment_recipe(parent, child, reference, model, set())
    write_json(model/'model.json', changed)
    monkeypatch.setattr(comparison, 'fit_selected', forbidden)
    assert comparison.run(source, reference, output) == result
    (output/'completed.json').unlink()
    (output/'report.json').unlink()
    assert comparison.run(source, reference, output)['methods'] == result['methods']
    matrix = read_json(source/'feature_matrix.json')
    matrix['rows'][0]['target.needs_empathy'] = 'yes'
    write_json(source/'feature_matrix.json', matrix)
    with pytest.raises(ConfigError, match='Completed training artifacts changed'):
        comparison.run(source, reference, tmp_path/'changed')


def test_archived_training_code_keeps_old_proof_and_action_serving_is_strict(tmp_path, monkeypatch):
    from src.iteration import runtime, versions
    monkeypatch.setattr(learned_sources, 'ROOT', tmp_path/'repo')
    monkeypatch.setattr(versions, 'PRIVATE', tmp_path/'private')
    names = [*learned_sources.SERVING_RUNTIME, 'scripts/train_fewshot_ranker.py',
             'src/generator/fewshot_ranker/action_crosses.py']
    bound = {}
    for name in names:
        path = learned_sources.ROOT/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen '+name)
        bound[name] = sha256_file(path)
    runtime.archive_inputs({str(learned_sources.ROOT/name): checksum for name, checksum in bound.items()})
    (learned_sources.ROOT/'scripts/train_fewshot_ranker.py').write_text('new training CLI')
    manifest = dict(runtime=bound)
    evidence = {str(learned_sources.ROOT/name): value for name, value in bound.items()}
    # Old manifests capture training time; deployment evidence captures the
    # exact later reconstruction. Training itself is never rerun on deploy.
    manifest['runtime'] = dict(bound, **{'scripts/train_fewshot_ranker.py': 'unavailable-training-time'})
    logical, paths = learned_sources.historical_runtime(manifest, evidence)
    assert logical[str(learned_sources.ROOT/'scripts/train_fewshot_ranker.py')] == bound['scripts/train_fewshot_ranker.py']
    assert versions.PRIVATE/'evidence_blobs'/bound['scripts/train_fewshot_ranker.py'] in paths
    assert learned_sources.serving_runtime(manifest) == {k: bound[k] for k in learned_sources.SERVING_RUNTIME}
    manifest['feature_transform'] = action_crosses.TRANSFORM
    assert len(learned_sources.serving_runtime(manifest)) == len(learned_sources.SERVING_RUNTIME)+1
    (learned_sources.ROOT/'src/generator/fewshot_ranker/action_crosses.py').write_text('changed serving')
    with pytest.raises(ConfigError, match='feature/scoring runtime changed'):
        learned_sources.serving_runtime(manifest)


def test_deploy_from_existing_ranker_preserves_prompts_and_original_dataset(tmp_path, monkeypatch):
    from src.iteration import learning_guard
    base, output = tmp_path/'g29', tmp_path/'candidate'
    model = tmp_path/'comparison/xgb_chat_pair'
    source, inventory = tmp_path/'source', tmp_path/'inventory'
    base.mkdir()
    (base/'prompt.md').write_text('unchanged generation instructions')
    prompts = {'prompt.md': sha256_file(base/'prompt.md')}
    write_json(base/'ranker/model.json', {'old': True})
    write_json(base/'ranker/provenance.json', {'old': True})
    assets = {**prompts, **{name: sha256_file(base/name) for name in
                          ('ranker/model.json', 'ranker/provenance.json')}}
    write_json(base/'learning.json', dict(asset_files=assets, information_end=20, evidence_files={}))
    write_json(model.parent/'manifest.json', dict(source=str(source)))
    write_json(source/'manifest.json', dict(inventory=str(inventory)))
    write_json(inventory/'labeling/manifest.json', dict(data='d-original'))
    write_json(model/'model.json', {'new': True})
    proof = dict(data_ref='d-original', information_end=20, evidence_files={})
    seen = []
    def reconstruct(directory, data):
        assert directory == model and data == 'd-original'
        return proof
    class Seal:
        def __init__(self, directory):
            assert directory == base
        def check(self):
            seen.append('seal')
    monkeypatch.setattr(learning_guard, 'MaterialSeal', Seal)
    monkeypatch.setattr(learning_guard, 'GENERATOR_RECIPES', {'approved': prompts})
    monkeypatch.setattr(learning_guard, 'require_materials', lambda data, dirs: seen.append((data, dirs)))
    monkeypatch.setattr(learned_sources, 'reconstruct', reconstruct)
    assert learned_sources.deploy(model, base, 'd-current', output) == output
    assert (output/'prompt.md').read_bytes() == (base/'prompt.md').read_bytes()
    assert read_json(output/'ranker/model.json') == {'new': True}
    assert read_json(output/'ranker/provenance.json') == proof
    assert seen == [('d-current', [base]), 'seal', ('d-current', [output])]
    assert set(read_json(output/'learning.json')['asset_files']) == set(assets)
