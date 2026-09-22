"""Train shared option scorers on paired labels; never feed differences to a network.

Source authorization belongs to the experiment driver. This module has no data,
LLM, development-label, or production-pointer access.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..config import ConfigError
from . import corrected_v1 as rt


def option_vectors(model, features, blind, metadata):
    a, b, names = rt.vectorize_boolean_options(features)
    context, context_names = rt.vectorize_context_booleans(metadata, model.context_schema)
    result = []
    for name, value in (("option_A", a), ("option_B", b)):
        observed, observed_names = rt.vectorize_observable_option_booleans(blind[name], metadata)
        group, group_names = rt.vectorize_group_pattern_option_boolean(blind[name], metadata)
        vector, expanded_names = rt.expand_boolean_option_with_refined_context(
            np.concatenate((value, observed, group)), [*names, *observed_names, *group_names],
            context, context_names)
        if tuple(expanded_names) != model.feature_names or not np.isfinite(vector).all():
            raise ConfigError("选项向量与冻结特征定义不符")
        result.append(vector)
    return tuple(result)


def recipes():
    """A bounded grid frozen before observing this sweep's development results."""
    lr = dict(kind="lr", penalty="l2", C=1., solver="lbfgs", fit_intercept=False,
              max_iter=5000, tol=1e-4, random_state=0)
    result = [{"id": "lr_l2_c1", **lr}]
    result += [{"id": f"lr_l2_c{c:g}", **lr, "C": c} for c in (.1, .3, 3., 10.)]
    result.append({"id": "lr_l1_c1", **lr, "penalty": "l1", "solver": "liblinear"})
    for depth in (2, 3):
        for regularization in (1., 10.):
            result.append(dict(id=f"gbdt_d{depth}_l2_{regularization:g}", kind="gbdt",
                objective="rank:pairwise", n_estimators=200, max_depth=depth,
                learning_rate=.05, reg_lambda=regularization, reg_alpha=0.,
                min_child_weight=1., subsample=1., colsample_bytree=1.,
                tree_method="hist", n_jobs=1, random_state=0))
    for hidden in ((32, 16), (64, 32)):
        for l2 in (.001, .01):
            result.append(dict(id=f"dnn_{hidden[0]}_{hidden[1]}_l2_{l2:g}", kind="dnn",
                hidden=list(hidden), l2=l2, max_iter=500, gtol=1e-6, ftol=1e-10, random_state=0))
    return result


def _matrices(a, b, y=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or not all(a.shape) or not np.isfinite([a, b]).all():
        raise ConfigError("配对特征须为同形状、有限值的两个矩阵")
    if y is None:
        return a, b
    y = np.asarray(y, dtype=float)
    if y.shape != (len(a),) or not np.isin(y, (0., 1.)).all() or len(np.unique(y)) != 2:
        raise ConfigError("训练标签须为包含两个类别的配对 0/1 标签")
    return a, b, y


def _unpack(theta, sizes):
    layers, offset = [], 0
    for i, (left, right) in enumerate(zip(sizes, sizes[1:])):
        count = left * right
        weight = theta[offset:offset + count].reshape(left, right)
        offset += count
        # The final shared output bias cancels in the logit difference.
        bias = theta[offset:offset + right] if i < len(sizes) - 2 else np.zeros(right)
        offset += right if i < len(sizes) - 2 else 0
        layers.append((weight, bias))
    if offset != len(theta):
        raise ConfigError("网络参数形状不符")
    return layers


def _forward(x, layers):
    activations = [x]
    for i, (weight, bias) in enumerate(layers):
        value = activations[-1] @ weight + bias
        activations.append(np.tanh(value) if i < len(layers) - 1 else value)
    return activations


def network_loss_gradient(theta, a, b, y, sizes, l2):
    layers = _unpack(theta, sizes)
    left, right = _forward(a, layers), _forward(b, layers)
    margin = (left[-1] - right[-1]).ravel()
    loss = np.mean(np.logaddexp(0., margin) - y * margin)
    loss += .5 * l2 * sum(np.square(weight).sum() for weight, _ in layers)
    derivative = ((expit(margin) - y) / len(y))[:, None]
    gradients = [[np.zeros_like(w), np.zeros_like(bias)] for w, bias in layers]
    for activations, delta in ((left, derivative), (right, -derivative)):
        for i in range(len(layers) - 1, -1, -1):
            gradients[i][0] += activations[i].T @ delta
            gradients[i][1] += delta.sum(axis=0)
            if i:
                delta = (delta @ layers[i][0].T) * (1. - np.square(activations[i]))
    packed = []
    for i, ((weight, _), (dw, db)) in enumerate(zip(layers, gradients)):
        packed.append((dw + l2 * weight).ravel())
        if i < len(layers) - 1:
            packed.append(db)
    return float(loss), np.concatenate(packed)


@dataclass
class Scorer:
    recipe: dict
    dimensions: int
    parameters: dict
    diagnostics: dict

    def score(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dimensions or not np.isfinite(x).all():
            raise ConfigError("推理特征维度或数值无效")
        kind = self.recipe["kind"]
        if kind == "lr":
            value = x @ np.asarray(self.parameters["coefficients"])
        elif kind == "dnn":
            layers = [(np.asarray(v["weight"]), np.asarray(v["bias"])) for v in self.parameters["layers"]]
            value = _forward(x, layers)[-1].ravel()
        elif kind == "gbdt":
            import xgboost as xgb
            if not hasattr(self, "_booster"):
                self._booster = xgb.Booster(params={"nthread": 1})
                self._booster.load_model(bytearray(self.parameters["booster_json"].encode()))
            value = self._booster.predict(xgb.DMatrix(x, nthread=1), output_margin=True)
        else:
            raise ConfigError("未知配对模型")
        if value.shape != (len(x),) or not np.isfinite(value).all():
            raise ConfigError("模型分数无效")
        return value.astype(float)

    def probability_a(self, a, b):
        a, b = _matrices(a, b)
        return expit(self.score(a) - self.score(b))

    def document(self):
        return dict(schema=1, scoring="sigmoid(score(A)-score(B))", recipe=self.recipe,
                    dimensions=self.dimensions, parameters=self.parameters, diagnostics=self.diagnostics)

    @classmethod
    def from_document(cls, value):
        if value.get("schema") != 1 or value.get("scoring") != "sigmoid(score(A)-score(B))":
            raise ConfigError("模型格式或评分方式不符")
        return cls(value["recipe"], value["dimensions"], value["parameters"], value["diagnostics"])


def fit(a, b, y, recipe):
    a, b, y = _matrices(a, b, y)
    cfg = {k: v for k, v in recipe.items() if k not in ("id", "kind")}
    kind = recipe["kind"]
    if kind == "lr":
        from scipy.sparse import csr_matrix
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        if cfg.get("fit_intercept") is not False:
            raise ConfigError("配对线性模型不能含不对称截距")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(**cfg).fit(csr_matrix(a - b), y)
        params = {"coefficients": model.coef_[0].tolist()}
        diagnostics = dict(converged=True, iterations=int(model.n_iter_[0]))
    elif kind == "gbdt":
        from xgboost import XGBRanker
        if cfg.get("objective") != "rank:pairwise":
            raise ConfigError("GBDT 必须分别给选项打分并使用配对排序目标")
        # Each query contains exactly two options; a positive label means human.
        x = np.stack((a, b), axis=1).reshape(-1, a.shape[1])
        labels = np.stack((y, 1. - y), axis=1).ravel()
        model = XGBRanker(**cfg).fit(x, labels, qid=np.repeat(np.arange(len(a)), 2))
        params = {"booster_json": model.get_booster().save_raw(raw_format="json").decode()}
        diagnostics = dict(converged=True, iterations=cfg["n_estimators"])
    elif kind == "dnn":
        sizes = [a.shape[1], *cfg["hidden"], 1]
        if any(type(n) is not int or n <= 0 for n in sizes) or cfg["l2"] < 0:
            raise ConfigError("网络宽度或正则项无效")
        rng, chunks = np.random.default_rng(cfg["random_state"]), []
        for i, (left, right) in enumerate(zip(sizes, sizes[1:])):
            chunks.append(rng.normal(0., np.sqrt(2. / (left + right)), size=left * right))
            if i < len(sizes) - 2:
                chunks.append(np.zeros(right))
        result = minimize(network_loss_gradient, np.concatenate(chunks),
            args=(a, b, y, sizes, cfg["l2"]), method="L-BFGS-B", jac=True,
            options={"maxiter": cfg["max_iter"], "gtol": cfg["gtol"], "ftol": cfg["ftol"], "maxls": 40})
        # A predeclared epoch budget is not development-based early stopping.
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or result.status not in (0, 1):
            raise ConfigError(f"网络优化失败：{result.message}")
        params = {"layers": [dict(weight=w.tolist(), bias=bias.tolist()) for w, bias in _unpack(result.x, sizes)]}
        diagnostics = dict(converged=bool(result.success), iterations=int(result.nit),
                           objective=float(result.fun), stop_reason=str(result.message))
    else:
        raise ConfigError("未知训练模型")
    scorer = Scorer(json.loads(json.dumps(recipe)), a.shape[1], params, diagnostics)
    scorer.probability_a(a, b)  # Validate serialized inference before returning it.
    return scorer
