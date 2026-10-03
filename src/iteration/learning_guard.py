"""兼容层：实现已迁入 src/iteration/materials/（seals.py / validation.py）。

保留原因：
- 钉死依赖：src/judge/corrected.py 的冻结证明字节绑定第 234 行
  `from ..iteration.learning_guard import MaterialSeal`，本文件因此必须存在。
- 历史调用方兼容：旧代码/测试可能经本层引用历史公共名。

新代码一律 `from src.iteration.materials import seals / validation`。
下列函数 re-export 仅为兼容；monkeypatch 请指向 materials.validation（或 materials.seals）
的真实属性——新调用方全部经模块属性访问函数，patch 生效点唯一。
"""
from .materials.seals import MaterialSeal, RunSeal
from .materials.validation import (CONVERSATION_ASSETS, GENERATOR_RECIPES, GENERIC_JUDGE_TEMPLATE,
    GENERIC_PROFILE, MECHANICAL_ASSETS, _directories, _reconstructed_judge, _require, bind,
    execution_seal, pack_proof, require_default_judge_template, require_materials, same_materials,
    snapshot, verify, verify_features, verify_fit, verify_generation, verify_pack, verify_repair,
    verify_training)

__all__ = ['CONVERSATION_ASSETS', 'GENERATOR_RECIPES', 'GENERIC_JUDGE_TEMPLATE', 'GENERIC_PROFILE',
           'MaterialSeal', 'MECHANICAL_ASSETS', 'RunSeal', '_directories', '_reconstructed_judge',
           '_require', 'bind', 'execution_seal', 'pack_proof', 'require_default_judge_template',
           'require_materials', 'same_materials', 'snapshot', 'verify', 'verify_features',
           'verify_fit', 'verify_generation', 'verify_pack', 'verify_repair', 'verify_training']
