"""Frozen GBDT inference over the existing paired feature extraction channel.

This module does not fit models or grant learning provenance or adoption. Those
checks remain mandatory at the experiment and promotion entry points.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from ..config import ConfigError
from . import corrected_v1 as rt
from .lr_retrain import LRJudge


def option_vectors(model, features, blind, metadata):
    """Expand each option independently, preserving the frozen vocabulary order."""
    a, b, names = rt.vectorize_boolean_options(features)
    context, context_names = rt.vectorize_context_booleans(metadata, model.context_schema)
    result = []
    for name, value in (('option_A', a), ('option_B', b)):
        observed, observed_names = rt.vectorize_observable_option_booleans(blind[name], metadata)
        group, group_names = rt.vectorize_group_pattern_option_boolean(blind[name], metadata)
        vector, expanded_names = rt.expand_boolean_option_with_refined_context(
            np.concatenate((value, observed, group)), [*names, *observed_names, *group_names],
            context, context_names)
        if tuple(expanded_names) != model.feature_names or not np.isfinite(vector).all():
            raise ConfigError('选项向量与冻结特征定义不符')
        result.append(vector)
    return tuple(result)


class GBDTScorer:
    """Read the saved ranking booster; compute sigmoid(score(A) - score(B))."""
    def __init__(self, document):
        if (not isinstance(document, dict) or document.get('schema') != 1
                or document.get('scoring') != 'sigmoid(score(A)-score(B))'):
            raise ConfigError('GBDT 模型格式或评分方式不符')
        recipe = document.get('recipe', {})
        dimensions = document.get('dimensions')
        parameters = document.get('parameters', {})
        if (not isinstance(recipe, dict) or recipe.get('kind') != 'gbdt'
                or recipe.get('objective') != 'rank:pairwise'
                or type(dimensions) is not int or dimensions <= 0
                or not isinstance(parameters, dict)
                or not isinstance(parameters.get('booster_json'), str)):
            raise ConfigError('GBDT 配方、维度或权重格式无效')
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ConfigError('GBDT Judge 需要可选依赖 xgboost>=2.0') from exc
        try:
            saved_version = json.loads(parameters['booster_json'])['version']
            current_version = [int(part) for part in xgb.__version__.split('.')]
        except (ValueError, KeyError, TypeError) as exc:
            raise ConfigError('GBDT 权重缺少可核验的 XGBoost 版本') from exc
        if saved_version != current_version:
            raise ConfigError('GBDT 权重与当前 XGBoost 版本不一致；须使用原版本')
        self.dimensions = dimensions
        self._xgb = xgb
        self._booster = xgb.Booster(params={'nthread': 1})
        try:
            self._booster.load_model(bytearray(parameters['booster_json'].encode()))
            config = json.loads(self._booster.save_config())
            objective = config['learner']['objective']['name']
        except (ValueError, KeyError, xgb.core.XGBoostError) as exc:
            raise ConfigError('GBDT 保存的权重无法加载') from exc
        if self._booster.num_features() != dimensions or objective != 'rank:pairwise':
            raise ConfigError('GBDT 实际模型与声明的维度或配对目标不符')

    def score(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not len(x) or x.shape[1] != self.dimensions or not np.isfinite(x).all():
            raise ConfigError('GBDT 推理特征维度或数值无效')
        value = self._booster.predict(self._xgb.DMatrix(x, nthread=1), output_margin=True)
        if value.shape != (len(x),) or not np.isfinite(value).all():
            raise ConfigError('GBDT 模型分数无效')
        return value.astype(float)

    def probability_a(self, a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.ndim != 2 or a.shape != b.shape:
            raise ConfigError('GBDT 配对选项矩阵形状不一致')
        return expit(self.score(a) - self.score(b))


class GBDTJudge(LRJudge):
    """Pure GBDT decisions with the same feature replay and independent rounds as LR."""
    decision_policy = 'gbdt_only'

    def __init__(self, config, version_dir, *args, **kwargs):
        filename = config.get('scorer_file')
        if (config.get('decision_policy') != self.decision_policy or not isinstance(filename, str)
                or Path(filename).name != filename or filename in ('', '.', '..')
                or filename not in config.get('assets', {})):
            raise ConfigError('GBDT 评分模型必须是冻结版本中已绑定的资产')
        super().__init__(config, version_dir, *args, **kwargs)
        try:
            document = json.loads((Path(version_dir) / filename).read_text())
        except (OSError, ValueError) as exc:
            raise ConfigError('GBDT 评分模型无法读取') from exc
        self.scorer = GBDTScorer(document)
        if self.scorer.dimensions != len(self.model.feature_names):
            raise ConfigError('GBDT 模型维度与冻结特征定义不符')
        self._check_sources()

    def probability_a(self, features, blind, metadata):
        a, b = option_vectors(self.model, features, blind, metadata)
        return float(self.scorer.probability_a(a[None, :], b[None, :])[0])
