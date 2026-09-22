"""Frozen, train-normalized fusion of independent GBDT and LR option scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from ..config import ConfigError
from .gbdt import GBDTJudge, option_vectors


class FusionScorer:
    def __init__(self, gbdt, coefficients, document):
        if (document.get('schema') != 1 or document.get('kind') != 'gbdt_lr_score_fusion_v1'
                or document.get('weights') != {'gbdt': .8, 'lr': .2}
                or document.get('normalization', {}).get('method') != 'training_pair_margin_rms'):
            raise ConfigError('融合评分配方不符')
        scales = document['normalization'].get('scales', {})
        if (set(scales) != {'gbdt', 'lr'} or any(type(v) not in (int, float)
                or not np.isfinite(v) or v <= 0 for v in scales.values())):
            raise ConfigError('融合评分训练尺度无效')
        self.gbdt = gbdt
        self.coefficients = np.asarray(coefficients, dtype=float)
        if (self.coefficients.shape != (gbdt.dimensions,)
                or not np.isfinite(self.coefficients).all()):
            raise ConfigError('融合 LR 系数与特征维度不符')
        self.scales = scales

    def score(self, x):
        x = np.asarray(x, dtype=float)
        tree = self.gbdt.score(x)  # Also validates dimensions and finite inputs.
        value = .8 * tree / self.scales['gbdt'] + .2 * (x @ self.coefficients) / self.scales['lr']
        if not np.isfinite(value).all():
            raise ConfigError('融合评分数值无效')
        return value

    def probability_a(self, a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.ndim != 2 or a.shape != b.shape:
            raise ConfigError('融合配对选项矩阵形状不一致')
        return expit(self.score(a) - self.score(b))


class FusionJudge(GBDTJudge):
    decision_policy = 'score_fusion'

    def __init__(self, config, version_dir, *args, **kwargs):
        filename = config.get('fusion_file')
        if (not isinstance(filename, str) or Path(filename).name != filename
                or filename in ('', '.', '..') or filename not in config.get('assets', {})):
            raise ConfigError('融合配方必须是冻结版本中已绑定的资产')
        super().__init__(config, version_dir, *args, **kwargs)
        try:
            document = json.loads((Path(version_dir) / filename).read_text())
        except (OSError, ValueError) as exc:
            raise ConfigError('融合评分配方无法读取') from exc
        self.fusion = FusionScorer(self.scorer, self.model.coefficients, document)
        self._check_sources()

    def probability_a(self, features, blind, metadata):
        a, b = option_vectors(self.model, features, blind, metadata)
        return float(self.fusion.probability_a(a[None, :], b[None, :])[0])
