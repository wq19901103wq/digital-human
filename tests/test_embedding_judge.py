"""Reusable embedding input, scoring, selection and cached comparison regressions."""
import copy

import numpy as np
import pytest
torch = pytest.importorskip('torch')

from src import cache
from src.config import ConfigError
from src.iteration import embedding_tuning as tuning
from src.iteration.storage import write_json
from src.judge import embedding as net
from test_corrected_judge import features


def inputs():
    a = net.categories(features(0), ['hello'], {'recent_speakers': ['alice', '__self__']})
    b = net.categories(features(3), ['yes', 'ok?'], {'recent_speakers': ['alice', '__self__']})
    raw = [(a, b), (b, a)] * 8
    encoder = net.fit_encoder(raw)
    return encoder, net.encode(encoder, raw), raw


def recipe(norm='none'):
    return dict(tuning.recipes()['e8-01'], hidden=[8], normalization=norm,
                cross_layers=2, max_epochs=3, patience=2)


def test_all_fields_embed_and_unknown_does_not_learn():
    encoder, x, raw = inputs()
    model = net.Ranker(encoder, recipe())
    assert all(e.embedding_dim == 8 for e in model.embeddings)
    changed = copy.deepcopy(raw[:1])
    changed[0][0]['speaker_0'] = 'unseen-evaluation-person'
    i = encoder['fields'].index('speaker_0')
    assert net.encode(encoder, changed)[0, 0, i] == 0
    assert 'unseen-evaluation-person' not in encoder['vocabularies']['speaker_0']
    with pytest.raises(ConfigError):
        model.score(torch.as_tensor(x[0], dtype=torch.float32))
    for update in ({'embedding_dim': 16}, {'numeric_bypass': True}):
        with pytest.raises(ConfigError):
            net.Ranker(dict(encoder, **update), recipe())


def test_categories_ignore_labels_ids_and_preserve_recent_order():
    m = {'recent_speakers': ['latest', 'previous', '__self__', 'older', 'fifth', 'sixth']}
    a = net.categories(features(), ['yes'], m)
    b = net.categories(features(), ['yes'], dict(m, case_id='secret', human_option='B', split='test'))
    assert a == b
    assert a['speaker_0'] == 'latest' and a['speaker_4'] == 'fifth'
    assert a['speaker_role_2'] == 'self'
    assert all(isinstance(v, str) for v in a.values())


@pytest.mark.parametrize('norm', ['none', 'batch', 'layer'])
def test_shared_scores_swap_serialization_and_bn_eval(norm):
    encoder, x, _ = inputs()
    model = net.Ranker(encoder, recipe(norm))
    if norm == 'batch':
        model.train()
        model(torch.as_tensor(x))
    p = net.predict(model, x)
    assert np.allclose(p + net.predict(model, x[:, ::-1].copy()), 1, atol=1e-6)
    assert np.allclose(p[:1], net.predict(model, x[:1]), atol=1e-6)
    restored = net.restore(net.document(model))
    assert np.allclose(p, net.predict(restored, x), atol=1e-7)


def test_fit_is_repeatable_and_training_only():
    encoder, x, _ = inputs()
    labels = np.asarray([True, False] * 8)
    args = (encoder, recipe(), x, labels, np.arange(12), np.arange(12, 16), 17)
    a, ra = net.fit(*args)
    b, rb = net.fit(*args)
    assert ra == rb and net.document(a) == net.document(b)
    assert ra['best_epoch'] <= 3
    full, result = net.fit(encoder, recipe(), x, labels, np.arange(16), [], 17, epochs=2)
    assert result['validation'] is None and result['epochs_run'] == 2


def test_grid_and_time_split():
    grid = tuning.recipes()
    assert len(grid) == len({cache.digest(v) for v in grid.values()}) == 36
    assert {v['normalization'] for v in grid.values()} == {'none', 'batch', 'layer'}
    rows = [{'case_id': str(i)} for i in range(10)]
    sources = [dict(r, source_span={'end_timestamp': 10-i}) for i, r in enumerate(rows)]
    train, val = tuning.split_training(rows, sources)
    assert list(val) == [1, 0] and set(train).isdisjoint(val)
    with pytest.raises(ConfigError):
        tuning.split_training(rows, sources[:-1])


def test_shared_preparation_preserves_existing_learning_guards(monkeypatch, tmp_path):
    calls = []
    pack = {'data_ref': 'd-0011'}
    monkeypatch.setattr(tuning.learning_guard, 'verify_pack',
                        lambda p, role: calls.append(('pack', p, role)))
    def snapshot(data, directories, role):
        calls.append(('materials', data, directories, role))
        return {'validated': True}
    monkeypatch.setattr(tuning.learning_guard, 'snapshot', snapshot)
    assert tuning.shared_learning_snapshot(pack, {'dir': tmp_path}) == {'validated': True}
    assert calls == [('pack', pack, 'development'),
                     ('materials', 'd-0011', [tmp_path], 'judge_development')]
    def reject(*args):
        raise ConfigError('learning materials contain future information')
    monkeypatch.setattr(tuning.learning_guard, 'snapshot', reject)
    with pytest.raises(ConfigError):
        tuning.shared_learning_snapshot(pack, {'dir': tmp_path})


def test_changed_batch_and_checkpoint_rejected_and_resume_reused(tmp_path, monkeypatch):
    tuning.bind_batch(tmp_path, {'features_sha256': 'original'})
    tuning.bind_batch(tmp_path, {'features_sha256': 'original'})
    with pytest.raises(ConfigError):
        tuning.bind_batch(tmp_path, {'features_sha256': 'changed'})
    encoder, x, _ = inputs()
    r = recipe()
    model = net.document(net.Ranker(encoder, r))
    job = ('trial', encoder, r, x, np.ones(16), np.arange(12), np.arange(12,16), 17, None)
    result = {'model': model, 'model_sha256': cache.digest(model),
              'job_sha256': cache.digest(dict(key='trial', encoder=encoder, recipe=r, seed=17, epochs=None))}
    path = tmp_path / 'search' / 'trial.json'
    write_json(path, result)
    monkeypatch.setattr(tuning.concurrent.futures, 'ProcessPoolExecutor',
                        lambda **kw: pytest.fail('completed work must not start workers'))
    assert tuning.train_jobs(path.parent, [job], 3, lambda: None) == {'trial': result}
    result['model']['scoring'] = 'changed'
    write_json(path, result)
    with pytest.raises(ConfigError):
        tuning.train_jobs(path.parent, [job], 3, lambda: None)


def test_missing_independent_rounds_do_not_count_as_confirmation():
    rows = [{'case_id': 'a'}, {'case_id': 'b'}]
    index = {('a', 0): 0, ('b', 0): 1}
    records = tuning.comparison_records(rows, index, [False, True], [True, True], [.8, .8], 2)
    assert records[0]['status'] == 'failed' and records[0]['missing_rounds'] == [1, 2]
    assert 'flip_verified' not in records[0]
    summary = tuning._initial_summary(records)
    assert summary['total'] == 2 and summary['initial_net'] == 1 and summary['pending_confirmation'] == 1
    index.update({('a', 1): 2, ('a', 2): 3})
    records = tuning.comparison_records(rows, index, [False, True, True, True],
                                        [True, True, False, False], [.8, .8, .2, .2], 2)
    assert records[0]['baseline_identified_final'] is True
    assert records[0]['candidate_identified_final'] is False
    assert tuning._initial_summary(records)['pending_confirmation'] == 0
    with pytest.raises(ConfigError):
        tuning.comparison_records(rows, index, [], [], [], 0)
