from copy import deepcopy

import numpy as np
import pytest
torch = pytest.importorskip('torch')

from src.config import ConfigError, sha256_file
from src.generator.fewshot_ranker import (
    architecture_comparison as comparison, boosting, neural_models, neural_sweep, training,
)
from src.generator.history_sources import digest
from src.iteration.storage import read_json, write_json
from tests.test_fewshot_ranker_lr import feature_row, frozen_source
from tests.test_fewshot_ranker_training import synthetic


def mixer_recipe():
    return dict(neural_sweep.recipes(0)[0], architecture='rankmixer', token_width=8,
                mixer_layers=1, ffn_expansion=2, hidden=[8], max_epochs=2, patience=1)


def test_rankmixer_roundtrip_swap_and_outer_label_isolation():
    groups, rows, observations = synthetic()
    rows = comparison.feature_rows([{**feature_row(), **row} for row in rows], True)
    value, predicted, detail = neural_sweep.train(groups, observations, rows, [mixer_recipe()])
    changed = deepcopy(observations)
    for row in changed:
        if row['split'] == 'validation':
            row['z'] = 1-row['z']
    second, _, second_detail = neural_sweep.train(groups, changed, rows, [mixer_recipe()])
    assert value == second and detail == second_detail
    model = neural_models.restore(value)
    x = torch.as_tensor(training.encode_rows(model.encoder, rows))
    np.testing.assert_allclose(training.scores(model, x.numpy()), predicted)
    with torch.no_grad():
        pair = x[:2].unsqueeze(0)
        torch.testing.assert_close(model(pair), -model(pair.flip(1)), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(model(pair), model.score(x[:1])-model.score(x[1:2]),
                                   atol=1e-7, rtol=1e-5)
    assert value['encoder']['embedding_dim'] == 8 and value['encoder']['numeric_bypass'] is False
    assert all(layer.embedding_dim == 8 for layer in model.embeddings)
    assert '15' not in value['encoder']['vocabularies']['identity']
    with pytest.raises(ConfigError, match='categorical IDs'):
        model.score(x.float())
    shaped = torch.arange(32).reshape(1, 4, 8)
    assert torch.equal(neural_models.token_mix(neural_models.token_mix(shaped)), shaped)
    assert not torch.equal(neural_models.token_mix(shaped), shaped)


def test_xgb_gradient_matches_pairwise_loss_and_curvature_is_upper_bound():
    scores = np.asarray([.3, -.4, 1.2])
    pairs = np.asarray([[0, 1], [2, 1]])
    weights = np.asarray([.5, .5])
    gradient, curvature = boosting.derivatives(scores, pairs, weights)
    eps = 1e-5
    numerical = []
    hessian = np.zeros((3, 3))
    for i in range(3):
        change = np.eye(3)[i]*eps
        numerical.append((boosting.pair_loss(scores+change, pairs, weights, 1)-
                          boosting.pair_loss(scores-change, pairs, weights, 1))/(2*eps))
        hessian[:, i] = (boosting.derivatives(scores+change, pairs, weights)[0]-
                         boosting.derivatives(scores-change, pairs, weights)[0])/(2*eps)
    np.testing.assert_allclose(gradient, numerical, atol=1e-9)
    assert np.linalg.eigvalsh(np.diag(curvature)-hessian).min() > -1e-9
    assert gradient.sum() == pytest.approx(0)


def test_xgb_roundtrip_and_outer_label_isolation():
    groups, rows, observations = synthetic()
    recipes = [dict(r, max_rounds=5, patience=2, min_child_weight=0)
               for r in boosting.recipes()[:2]]
    value, scores, detail = boosting.train(groups, observations, rows, recipes)
    changed = deepcopy(observations)
    for row in changed:
        if row['split'] == 'validation':
            row['z'] = 1-row['z']
    second, _, second_detail = boosting.train(groups, changed, rows, recipes)
    assert value == second and detail == second_detail
    assert 'identity=15' not in value['vocabulary']
    np.testing.assert_allclose(boosting.score_document(value, rows), scores)
    assert training.metrics(observations, scores)['group_auc_macro'] == 1


def test_architecture_comparison_reuses_controls_and_partial_completion(tmp_path, monkeypatch):
    from src.judge.corrected import CodexJudgeClient
    from src.llm import ChatClient

    def forbidden(*args, **kwargs):
        raise AssertionError('saved work must be reused without LLM calls or refitting')

    monkeypatch.setattr(CodexJudgeClient, 'run', forbidden)
    monkeypatch.setattr(ChatClient, 'chat', forbidden)
    source = frozen_source(tmp_path, monkeypatch)
    reference, output = tmp_path/'reference', tmp_path/'comparison'
    manifest = read_json(source/'manifest.json')
    binding = dict(kind='cached_fewshot_feature_comparison', source_manifest_sha256=digest(manifest),
                   source_completed_sha256=sha256_file(source/'completed.json'), dataset=manifest['dataset'])
    write_json(reference/'manifest.json', binding)
    names = []
    for name in comparison.CONTROLS:
        path = f'{name}/predictions.json'
        write_json(reference/path, read_json(source/'predictions.json'))
        names.append(path)
    comparison.seal(reference, binding, names)
    monkeypatch.setattr(comparison, 'NEURAL_RECIPES', {
        'dnn_semantic': [dict(neural_sweep.recipes(0)[0], hidden=[8], max_epochs=1)],
        'rankmixer_chat_pair': [mixer_recipe()],
    })
    monkeypatch.setattr(comparison, 'XGB_RECIPES', [dict(boosting.recipes()[0], max_rounds=2)])
    result = comparison.run(source, reference, output)
    assert result['new_model_requests'] == 0 and result['status'] == 'complete'
    assert len(result['training']) == 3
    for name in comparison.CONTROLS:
        assert result['methods'][name+'_saved'] == result['methods']['dnn_saved']
    assert 'Hit@5' in (output/'REPORT.md').read_text()
    monkeypatch.setattr(neural_sweep, 'train', forbidden)
    monkeypatch.setattr(boosting, 'train', forbidden)
    assert comparison.run(source, reference, output) == result
    (output/'completed.json').unlink()
    (output/'report.json').unlink()
    assert comparison.run(source, reference, output)['methods'] == result['methods']
    matrix = read_json(source/'feature_matrix.json')
    matrix['rows'][0]['target.needs_empathy'] = 'yes'
    write_json(source/'feature_matrix.json', matrix)
    with pytest.raises(ConfigError, match='Completed training artifacts changed'):
        comparison.run(source, reference, tmp_path/'changed')
