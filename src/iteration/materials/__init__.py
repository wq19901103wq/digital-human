"""材料验证包：seals（逐题印章）+ validation（来源核验与验证复用，唯一实现模块）。

函数一律经模块属性访问（materials.validation.<fn>），保证 monkeypatch 生效点唯一；
本包仅 re-export 类与常量（设计 §4 单实现原则）。
"""
from .seals import MaterialSeal, RunSeal
from .validation import (CONVERSATION_ASSETS, GENERATOR_RECIPES, GENERIC_JUDGE_TEMPLATE,
                         GENERIC_PROFILE, MECHANICAL_ASSETS)

__all__ = ['CONVERSATION_ASSETS', 'GENERATOR_RECIPES', 'GENERIC_JUDGE_TEMPLATE', 'GENERIC_PROFILE',
           'MaterialSeal', 'MECHANICAL_ASSETS', 'RunSeal']
