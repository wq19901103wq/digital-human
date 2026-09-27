"""Text-vector reuse contract; all retrieval assets here are synthetic."""
import pickle

import numpy as np
import pytest

from src.judge import memory_dense as dense
from src.judge.background_features import response_schema


class Encoder:
    def __init__(self, *args):
        pass

    def encode(self, texts):
        output = np.zeros((len(texts), 512), dtype=np.float32)
        for i, text in enumerate(texts):
            output[i, sum(map(ord, text)) % 512] = 1
        return output


@pytest.fixture
def assets(tmp_path):
    index = tmp_path / 'trusted.pkl'
    texts = ['历史甲', '历史乙', '答案', '将来']
    # Old sender, timestamp and even context text must never reach retrieval.
    value = {'version': 'message_level_v2_dense',
             'messages': [{'text': text, 'sender': 'WRONG', 'timestamp': 999999,
                           'context': 'DO NOT IMPORT'} for text in texts],
             'embeddings': Encoder().encode(texts)}
    index.write_bytes(pickle.dumps(value))
    (tmp_path / 'model.onnx').write_bytes(b'test weights')
    (tmp_path / 'tokenizer.json').write_text('{}')
    return index, tmp_path


def test_only_exact_text_vectors_are_reused_and_coverage_is_explicit(assets):
    backend = dense.DenseMemory(*assets, encoder_factory=Encoder)
    scores, coverage = backend.score('答案', [' 历史甲 ', '缺少的文本'])
    assert len(scores) == 2 and scores[0] == 0 and np.isnan(scores[1])
    assert coverage['eligible_messages'] == 2
    assert coverage['dense_covered'] == coverage['keyword_only'] == 1
    assert not hasattr(backend, 'messages')  # Old attribution metadata discarded.
    assert coverage['encoder_probe_min_cosine'] == 1


def test_changed_assets_and_encoder_mismatch_are_rejected(assets):
    index, directory = assets
    class WrongEncoder(Encoder):
        def encode(self, texts):
            return np.roll(super().encode(texts), 1, axis=1)
    with pytest.raises(ValueError, match='不兼容'):
        dense.DenseMemory(*assets, encoder_factory=WrongEncoder).score('历史', ['历史甲'])
    backend = dense.DenseMemory(*assets, encoder_factory=Encoder)
    before = backend.identity()
    (directory / 'tokenizer.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='已改变'):
        backend.score('历史', ['历史甲'])
    assert dense.DenseMemory(*assets, encoder_factory=Encoder).identity() != before


def test_corrupt_vector_index_rejected(assets):
    index, _ = assets
    with index.open('rb') as stream:
        value = pickle.load(stream)
    value['embeddings'][0, 0] = np.nan
    index.write_bytes(pickle.dumps(value))
    with pytest.raises(ValueError, match='归一化'):
        dense.DenseMemory(*assets, encoder_factory=Encoder).score('历史', ['历史甲'])


def test_missing_configuration_never_silently_downgrades(monkeypatch):
    monkeypatch.delenv('WECHAT_HISTORY_INDEX_PATH', raising=False)
    monkeypatch.delenv('WECHAT_BGE_MODEL_PATH', raising=False)
    with pytest.raises(ValueError, match='显式配置'):
        dense.configured()


def test_context_attribution_schema_only_accepts_input_context_refs():
    schema = response_schema({'context': [{'ref': 'context:1', 'role': 'other'}]})
    alternative, = schema['properties']['context_attribution']['items']['anyOf']
    refs = alternative['properties']['evidence_refs']
    assert refs['items']['enum'] == ['context:1']
