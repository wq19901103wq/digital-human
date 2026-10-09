"""Frozen primary reconstruction uses original training features, never evaluation labels."""
import json

import numpy as np
import pytest

pytest.importorskip('torch')  # 可选重依赖：缺省时整模块跳过
from src import cache
from src.config import ConfigError, sha256_file
from src.iteration import embedding_adoption as adoption
from src.iteration.storage import write_json
from src.judge import embedding as net
from test_embedding_judge import recipe
from test_corrected_judge import features, bundle, case


@pytest.fixture
def study(tmp_path):
    source, directory = tmp_path / 'source', tmp_path / 'study'
    arm = source / 'luna'
    rows = [dict(case_id=str(i), split='train', human_option='A' if i % 2 else 'B',
                 blind={'option_A': ['yes'], 'option_B': ['ok', 'why?']}, metadata={})
            for i in range(16)]
    entries = {r['case_id']: {'features': {'option_A': features(0), 'option_B': features(3)}} for r in rows}
    write_json(arm / 'training.json', {'rows': rows})
    write_json(arm / 'features.json', {'entries': entries})
    r = recipe('batch')
    spec = dict(kind='embedding_dnn_tuning', dataset='training', embedding_dim=8, numeric_bypass=False,
        source=str(arm), arm='luna', torch_version=net.torch.__version__, numpy_version=np.__version__,
        training_total=16, inputs={str(p): sha256_file(p) for p in arm.iterdir()},
        selection={'primary_seed': 17}, recipes={'e8-01': r})
    write_json(directory / 'spec.json', spec)
    write_json(directory / 'selection.json', dict(primary='e8-01-s17', configurations=['e8-01'], epochs={'e8-01': 2}))
    raw = net.raw_pairs(rows, entries)
    encoder = net.fit_encoder(raw)
    model, result = net.fit(encoder, r, net.encode(encoder, raw),
        np.asarray([v['human_option'] == 'A' for v in rows]), np.arange(16), [], 17, 2)
    doc = net.document(model)
    write_json(directory / 'refit/e8-01-s17.json', {**result, 'model': doc,
        'model_sha256': cache.digest(doc),
        'job_sha256': cache.digest(dict(key='e8-01-s17', encoder=encoder, recipe=r, seed=17, epochs=2))})
    return directory, source, doc


def test_frozen_primary_reconstructs_exactly(study):
    directory, source, doc = study
    actual, _, key = adoption.selected_model(directory, source, 'luna')
    assert actual == doc and key == 'e8-01-s17'


def test_runtime_dispatch_uses_saved_embedding_weights(study, bundle, case, monkeypatch):
    from src.iteration import versions
    from src.judge import corrected
    from src.judge.judge import build_judge
    from src.judge.embedding_runtime import EmbeddingJudge
    class NoRequest:
        def __init__(self, config):
            self.config = config
        def run(self, *args, **kwargs):
            pytest.fail('no LLM request')
    monkeypatch.setattr(corrected, 'CodexJudgeClient', NoRequest)
    _, _, doc = study
    cfg, directory = bundle
    assets = {n: (directory / n).read_bytes() for n in ['prompt.md', *cfg['assets']]}
    assets['embedding.json'] = adoption._bytes(doc)
    cfg = {**cfg, 'decision_policy': 'embedding_only', 'embedding_file': 'embedding.json',
           'assets': {**cfg['assets'], 'embedding.json': adoption.hashlib.sha256(assets['embedding.json']).hexdigest()}}
    jid = versions.create_judge_version(cfg, {}, root=directory.parent, assets=assets)
    scorer = build_judge({}, dict(config=cfg, dir=directory.parent / jid))
    assert isinstance(scorer, EmbeddingJudge)
    blind, metadata = corrected.blind_case(case, ['八点'], ['九点'])
    parsed = dict(option_A=features(0), option_B=features(3))
    pair = tuple(net.categories(parsed[k], blind[k], metadata) for k in ('option_A', 'option_B'))
    expected = float(net.predict(net.restore(doc), net.encode(doc['encoder'], [pair]))[0])
    assert scorer.probability_a(parsed, blind, metadata) == expected


def test_resigned_weights_rejected(study):
    directory, source, _ = study
    path = directory / 'refit/e8-01-s17.json'
    result = json.loads(path.read_text())
    key = next(k for k in result['model']['weights'] if k.endswith('weight'))
    result['model']['weights'][key][0][0] += 1
    result['model_sha256'] = cache.digest(result['model'])
    write_json(path, result)
    with pytest.raises(ConfigError, match='cannot be reconstructed'):
        adoption.selected_model(directory, source, 'luna')


@pytest.mark.parametrize('replies', [('ok', 'ok'), ('收到了', '收到')])
def test_identical_encoded_inputs_tie_exactly(study, case, monkeypatch, replies):
    from src.judge.corrected import blind_case
    from src.judge.embedding_runtime import EmbeddingJudge
    _, _, doc = study
    scorer = object.__new__(EmbeddingJudge)
    scorer.network = net.restore(doc)
    blind, metadata = blind_case(case, [replies[0]], [replies[1]])
    parsed = dict(option_A=features(0), option_B=features(0))
    pair = tuple(net.categories(parsed[k], blind[k], metadata) for k in ('option_A', 'option_B'))
    encoded = net.encode(doc['encoder'], [pair])
    assert np.array_equal(encoded[0, 0], encoded[0, 1])
    monkeypatch.setattr(net, 'predict', lambda *args: np.asarray([0.49999994]))
    assert scorer.probability_a(parsed, blind, metadata) == 0.5


def test_distinct_encoded_inputs_keep_network_probability(study, case, monkeypatch):
    from src.judge.corrected import blind_case
    from src.judge.embedding_runtime import EmbeddingJudge
    _, _, doc = study
    scorer = object.__new__(EmbeddingJudge)
    scorer.network = net.restore(doc)
    blind, metadata = blind_case(case, ['yes'], ['ok', 'why?'])
    parsed = dict(option_A=features(0), option_B=features(3))
    calls = []
    def predict(model, encoded):
        assert not np.array_equal(encoded[0, 0], encoded[0, 1])
        calls.append(encoded)
        return np.asarray([0.49999994])
    monkeypatch.setattr(net, 'predict', predict)
    assert scorer.probability_a(parsed, blind, metadata) == 0.49999994
    assert len(calls) == 1


@pytest.mark.parametrize('resign', [False, True])
def test_nontraining_or_changed_inputs_rejected(study, resign):
    directory, source, _ = study
    path = source / 'luna/training.json'
    rows = json.loads(path.read_text())
    rows['rows'][0]['split'] = 'development'
    write_json(path, rows)
    if resign:
        spec = json.loads((directory / 'spec.json').read_text())
        spec['inputs'][str(path)] = sha256_file(path)
        write_json(directory / 'spec.json', spec)
    with pytest.raises(ConfigError):
        adoption.selected_model(directory, source, 'luna')
