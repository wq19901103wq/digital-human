"""Reuse trusted local RPA text vectors, never its identities or context windows.

Only exact message-text matches can reuse an embedding. SearchMemory supplies
the already eligible frozen messages; unknown texts retain keyword coverage.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import pickle
from threading import RLock

import numpy as np

from .. import cache
from ..config import sha256_file
from ..generator.history_sources import stamp


class BGEEncoder:
    def __init__(self, model, tokenizer):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        self.tokenizer.enable_truncation(max_length=200)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                           providers=['CPUExecutionProvider'])

    def encode(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        inputs = {'input_ids': np.array([row.ids for row in encoded], dtype=np.int64),
                  'attention_mask': np.array([row.attention_mask for row in encoded], dtype=np.int64)}
        vectors = self.session.run(None, inputs)[0][:, 0, :].astype(np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


class DenseMemory:
    def __init__(self, index, model_dir, *, encoder_factory=BGEEncoder):
        self.index = Path(index).absolute()
        directory = Path(model_dir).absolute()
        model = directory / 'model_optimized.onnx'
        if not model.exists():
            model = directory / 'model.onnx'
        tokenizer = directory / 'tokenizer.json'
        self.assets = {'index': self.index, 'model': model, 'tokenizer': tokenizer}
        self.stamps = {key: stamp(path) for key, path in self.assets.items()}
        self.hashes = {key: sha256_file(path) for key, path in self.assets.items()}
        self.encoder_factory = encoder_factory
        self.encoder = None
        self.vectors = None
        self.text_to_row = {}
        self.metadata = {}
        self._lock = RLock()
        self.check()

    def check(self):
        if self.stamps != {key: stamp(path) for key, path in self.assets.items()}:
            raise ValueError('记忆索引或编码模型已改变；请重新绑定后检索')

    def identity(self):
        return {'version': 'rpa_bge_text_reuse_v1', 'assets': self.hashes,
                'implementation': cache.code_digest(__file__),
                'encoding': 'CLS/L2;max_length=200;no_query_prefix',
                'alignment': 'exact_stripped_text;frozen_source_metadata_only'}

    def _load(self):
        with self._lock:
            self._load_once()

    def _load_once(self):
        if self.vectors is not None:
            return
        # Explicit operator-supplied trusted local pickle, never a model/tool path.
        with self.index.open('rb') as stream:
            value = pickle.load(stream)
        messages, vectors = value.get('messages'), value.get('embeddings')
        if (not isinstance(messages, list) or not messages or
                not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or
                vectors.shape != (len(messages), 512) or vectors.dtype != np.float32 or
                not all(isinstance(m, dict) and isinstance(m.get('text'), str) for m in messages)):
            raise ValueError('不支持的 RPA 消息向量索引格式')
        for start in range(0, len(vectors), 4096):
            block = vectors[start:start + 4096]
            if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=.002):
                raise ValueError('RPA 向量必须为有限的归一化文本向量')
        encoder = self.encoder_factory(self.assets['model'], self.assets['tokenizer'])
        # Detect a different model or context-encoded index before reuse.
        probes = np.unique(np.linspace(0, len(messages) - 1, min(5, len(messages)), dtype=int))
        probe_vectors = encoder.encode([messages[i]['text'].strip() for i in probes])
        similarity = np.sum(probe_vectors * vectors[probes], axis=1)
        if not np.isfinite(similarity).all() or np.min(similarity) < .995:
            raise ValueError(f'索引与纯文本 BGE 编码不兼容: min_cosine={float(np.min(similarity)):.6f}')
        self.check()
        self.text_to_row = {m['text'].strip(): i for i, m in enumerate(messages)}
        self.metadata = {'index_version': value.get('version'), 'index_messages': len(messages),
                         'unique_texts': len(self.text_to_row),
                         'encoder_probe_min_cosine': float(np.min(similarity))}
        self.encoder, self.vectors = encoder, vectors

    def score(self, query, eligible_texts):
        self.check()
        self._load()
        indices = np.array([self.text_to_row.get(text.strip(), -1) for text in eligible_texts])
        positions = np.flatnonzero(indices >= 0)
        scores = np.full(len(indices), np.nan, dtype=np.float32)
        with self._lock:
            query_vector = self.encoder.encode([query])[0]
        # No future/excluded records compete for top-k. Bound temporary copies.
        for start in range(0, len(positions), 4096):
            batch = positions[start:start + 4096]
            scores[batch] = self.vectors[indices[batch]] @ query_vector
        self.check()
        return scores, {**self.metadata, 'eligible_messages': len(indices),
                        'dense_covered': len(positions), 'keyword_only': len(indices) - len(positions)}


@lru_cache(maxsize=2)
def _load_configured(index, model):
    return DenseMemory(index, model)


def configured(index=None, model=None):
    index = index or os.environ.get('WECHAT_HISTORY_INDEX_PATH')
    model = model or os.environ.get('WECHAT_BGE_MODEL_PATH')
    if not index or not model:
        raise ValueError('with_memory 需要显式配置 --memory-index 和 --memory-model，或对应 WECHAT 环境变量')
    backend = _load_configured(str(Path(index).absolute()), str(Path(model).absolute()))
    backend.check()
    return backend
