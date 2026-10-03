"""旧 few_shot.py 行为参照加载器：从认证归档 a5118947… 按路径 importlib 加载。

模块化重构（2026-10-03）删除活树 src/generator/few_shot.py 后，parity/适配器
测试的旧行为参照统一由本模块提供。来源与字节身份：

- 来源归档：instances/qianwang/runtimes/a5118947…（9月29 日冻结 runtime，
  归档自带旧实现，旧代进程经 sys.path 快照优先解析到它）。
- 归档 src/generator/few_shot.py 的 sha256 与归档 manifest 钉死值一致
  （ARCHIVED_FEW_SHOT_SHA256，加载时强校验，篡改即拒绝）。
- 模块名自定义（archived_reference_few_shot），不与 src.generator.fewshot
  包冲突；归档文件仅依赖标准库，加载无需额外 sys.path 引导。
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

ARCHIVE_SHA256 = 'a5118947eb888be5d46901f2e5334ce14fc16a932e8a503e5cc595a30d0ac1e4'
ARCHIVED_FEW_SHOT_SHA256 = ('fef16b699e3be2c9ce56aa1c3de8a05a2d0b852a32457abda8edb452'
                            'fba0efd9')


def archive_root(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return root / 'instances' / 'qianwang' / 'runtimes' / ARCHIVE_SHA256


def archived_few_shot_path(root: Path | None = None) -> Path:
    return archive_root(root) / 'src' / 'generator' / 'few_shot.py'


def load_reference_few_shot(root: Path | None = None):
    """按路径加载归档旧 few_shot 模块，并以标准包名 src.generator.few_shot 注册。

    标准包名让模块内函数级相对 import（.history / .history_sources）经
    src.generator 包解析（history.py 与归档字节一致；history_sources 走活树
    shim，与 src.history.sources 同源），同时让当前树进程内的
    `from src.generator import few_shot`（旧装配入口与适配器测试）继续工作。
    加载前强校验字节 pin。
    """
    path = archived_few_shot_path(root)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != ARCHIVED_FEW_SHOT_SHA256:
        raise AssertionError(f'归档 few_shot.py 字节与参照 pin 不符：{actual}')
    name = 'src.generator.few_shot'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 进程内单例：parity 参照类与旧模块属性被多处 monkeypatch，必须共享同一模块对象。
few_shot = load_reference_few_shot()
