from copy import deepcopy

import numpy as np
import pytest

import pytest
pytest.importorskip('torch')  # 可选重依赖
pytest.importorskip('xgboost')  # 可选重依赖
from src.config import ConfigError
from src.generator.fewshot_ranker import (
    comparison, discovery, identity_comparison, identity_crosses, logistic, neural_sweep, training,
)
from src.iteration.storage import read_json, write_json
from src.judge.embedding import restore
from tests.test_fewshot_ranker_lr import feature_row, frozen_source
from tests.test_fewshot_ranker_training import synthetic


def test_categorical_feature_families_use_cached_inputs_without_mutation():
    row = feature_row()
    row.update({'target.needs_empathy': 'yes', 'example_reply.opening_style': 'comfort',
                'target.chat_id': 'one|two', 'example_context.chat_id': 'three'})
    before = deepcopy(row)
    expanded = identity_crosses.expand(row, style=True)
    additions = discovery.fields(row)
    assert row == before and all(isinstance(v, str) for v in {**expanded, **additions}.values())
    assert additions['discovery.response_need.target.needs_empathy_x_example_reply.opening_style'] == '["yes","comfort"]'
    assert expanded['id_cross.chat_id'] == '["one|two","three"]'
    assert expanded['id_cross.chat_id'] != identity_crosses.pair(
        {'a': 'one', 'b': 'two|three'}, 'a', 'b')
    assert identity_crosses.pair({'a': 'unknown', 'b': 'unknown'}, 'a', 'b') == 'unknown'
    assert len(additions) == sum(len(v) for v in discovery.SPECS.values())
    assert set(discovery.fields(row, ['rhythm'])) == {k for k in additions if '.rhythm.' in k}
    # Outcome-like extra inputs cannot affect the deterministic FG.
    row.update(label='1', target_answer='hidden', generated_reply='hidden')
    assert discovery.fields(row) == additions
    with pytest.raises(ConfigError, match='Missing categorical'):
        identity_crosses.pair({'a': 1, 'b': 'valid'}, 'a', 'b')


def test_feature_profile_and_identity_support_do_not_use_outer_labels():
    groups, rows, obs = synthetic()
    rows = [{**feature_row(), **r} for r in rows]
    expanded = [identity_crosses.expand(r) for r in rows]
    additions = [discovery.fields(r) for r in rows]
    profile = discovery.profile(groups, obs, additions)
    support = identity_crosses.coverage(groups, obs, expanded)
    changed = deepcopy(obs)
    changed_additions = deepcopy(additions)
    for i, row in enumerate(changed):
        if row['split'] == 'validation':
            row['z'] = 1-row['z']
            changed_additions[i] = {k: 'outer-only' for k in additions[i]}
    assert discovery.profile(groups, changed, changed_additions) == profile
    assert identity_crosses.coverage(groups, changed, expanded) == support
    assert profile['fit_contexts'] == 10
    assert support['validation_contexts']['total'] == 5


@pytest.mark.parametrize('cross_layers', [0, 2])
def test_neural_sweep_isolates_outer_labels_and_restores_scores(cross_layers):
    groups, rows, obs = synthetic()
    recipes = [dict(r, hidden=[8], max_epochs=2, patience=1)
               for r in neural_sweep.recipes(cross_layers)[::2]]
    saved, predicted, detail = neural_sweep.train(groups, obs, rows, recipes)
    changed = deepcopy(obs)
    for row in changed:
        if row['split'] == 'validation':
            row['z'] = 1-row['z']
    second, _, second_detail = neural_sweep.train(groups, changed, rows, recipes)
    assert saved == second and detail == second_detail
    assert saved['encoder']['embedding_dim'] == 8
    assert saved['encoder']['numeric_bypass'] is False
    model = restore(saved)
    np.testing.assert_allclose(training.scores(model, training.encode_rows(model.encoder, rows)), predicted)
    assert all('15' not in values for values in saved['encoder']['vocabularies'].values())


def test_feature_comparison_reuses_controls_and_resumes_without_model_requests(tmp_path, monkeypatch):
    from src.judge.corrected import CodexJudgeClient
    from src.llm import ChatClient

    def forbidden(*args, **kwargs):
        raise AssertionError('completed artifacts must be reused without model requests or training')

    monkeypatch.setattr(CodexJudgeClient, 'run', forbidden)
    monkeypatch.setattr(ChatClient, 'chat', forbidden)
    source = frozen_source(tmp_path, monkeypatch)
    reference, output = tmp_path/'reference', tmp_path/'features'
    controls = comparison.run(source, reference)
    monkeypatch.setattr(comparison, 'run', forbidden)
    monkeypatch.setattr(identity_comparison, 'LR_RECIPE', dict(logistic.RECIPE, c_values=[.001, .01]))
    monkeypatch.setattr(identity_comparison, 'NEURAL_RECIPES', {
        name: [dict(recipes[0], hidden=[8], max_epochs=1, patience=1)]
        for name, recipes in identity_comparison.NEURAL_RECIPES.items()})
    value = identity_comparison.run(source, reference, output)
    assert value['new_model_requests'] == 0
    assert len(value['training']) == 11
    assert value['methods']['lr_existing_saved'] == controls['methods']['lr_existing']
    assert value['methods']['lr_expanded_saved'] == controls['methods']['lr_expanded']
    assert set(value['feature_profile']['family_sizes']) == set(discovery.FAMILIES)
    assert 'Behavioral feature exploration' in (output/'REPORT.md').read_text()
    assert value['inner_selected_method'] == min(value['inner_losses'], key=lambda k: (value['inner_losses'][k], k))
    monkeypatch.setattr(logistic, 'train', forbidden)
    monkeypatch.setattr(neural_sweep, 'train', forbidden)
    assert identity_comparison.run(source, reference, output) == value
    # A partial run containing completed methods must resume without refitting.
    (output/'completed.json').unlink()
    (output/'report.json').unlink()
    assert identity_comparison.run(source, reference, output)['methods'] == value['methods']
    matrix = read_json(source/'feature_matrix.json')
    matrix['rows'][0]['target.needs_empathy'] = 'yes'
    write_json(source/'feature_matrix.json', matrix)
    with pytest.raises(ConfigError, match='Completed training artifacts changed'):
        identity_comparison.run(source, reference, tmp_path/'changed')
