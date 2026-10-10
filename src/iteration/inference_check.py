"""Read-only saved-model readiness, isolated from training and model requests."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np

from src.infrastructure.config import ConfigError, sha256_file
from src.infrastructure.storage import read_json

KINDS = ('judge-gbdt', 'fewshot-xgboost', 'judge-embedding')


def verify(model_path, kind, *, progress=None):
    if kind not in KINDS:
        raise ConfigError('不支持的在线评分类型')
    path = Path(model_path)
    digest = sha256_file(path)
    document = read_json(path)
    notify = progress or (lambda stage: None)
    notify('importing_runtime')
    if kind == 'judge-gbdt':
        runtime = import_module('src.judge.gbdt')
        notify('loading_saved_model')
        scorer = runtime.GBDTScorer(document)
        notify('predicting')
        values = scorer.score(np.zeros((2, scorer.dimensions)))
    elif kind == 'fewshot-xgboost':
        runtime = import_module('src.generator.fewshot.boosting')
        notify('loading_and_predicting')
        values = runtime.score_document(document, [{}, {}])
    else:
        runtime = import_module('src.judge.embedding')
        notify('loading_saved_model')
        runtime.torch.set_num_threads(1)
        model = runtime.restore(document)
        notify('predicting')
        pairs = np.zeros((2, 2, len(model.encoder['fields'])), dtype=np.int64)
        values = runtime.predict(model, pairs)
    values = np.asarray(values)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ConfigError('在线评分返回的形状或数值无效')
    if sha256_file(path) != digest:
        raise ConfigError('在线评分检查过程中模型资产变化')
    return dict(kind=kind, model_sha256=digest, predictions=2, training=False,
                external_model_requests=0, source_provenance_verified=False,
                performance_claim=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=KINDS)
    parser.add_argument('--model', required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify(args.model, args.kind,
                    progress=lambda stage: print('stage: ' + stage, file=sys.stderr, flush=True))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
