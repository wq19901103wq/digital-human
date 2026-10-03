"""旧 few_shot.py 行为参照加载器：归档优先，入库夹具兜底（干净 checkout / CI 可收集）。

模块化重构（2026-10-03）删除活树 src/generator/few_shot.py 后，parity/适配器
测试的旧行为参照统一由本模块提供。来源与字节身份：

- 首选来源：instances/qianwang/runtimes/a5118947…（9月29 日冻结 runtime 归档，
  旧代进程经 sys.path 快照优先解析到它）。归档存在时按其路径加载并强校验字节。
- 兜底来源：tests/_reference_cache/few_shot.py（入库物化副本，归档 few_shot.py
  的源码固化文件，sha256 同一 pin；缺失时由 tests/_reference_few_shot_source.py
  的 SOURCE 现场重建）。归档缺失（干净 checkout / CI）时退回该真实文件，
  保证本文件 import 不依赖本机私有实例目录，且模块 __file__ 指向真实路径
  （inspect.getsource 与磁盘适配器 checksum(module.__file__) 可用）。
- 归档 src/generator/few_shot.py 的 sha256 与归档 manifest 钉死值一致
  （ARCHIVED_FEW_SHOT_SHA256，两种来源都强校验，篡改即拒绝）。
- 模块名自定义注册（archived_reference_few_shot），不与 src.generator.fewshot
  包冲突；归档文件仅依赖标准库，加载无需额外 sys.path 引导。
- 归档字节冒烟在文件尾（默认 skip，本机 DH_ARCHIVE_TESTS=1 且归档存在时运行）。
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


def cached_few_shot_path(root: Path | None = None) -> Path:
    """兜底物化副本：真实文件，模块 __file__ 指向它（getsource/checksum 可用）。"""
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    return base / '_reference_cache' / 'few_shot.py'


def _bundled_source() -> str:
    """入库夹具（tests/_reference_few_shot_source.py）：归档源码固化副本。"""
    from _reference_few_shot_source import ARCHIVED_FEW_SHOT_SHA256 as bundled_pin
    from _reference_few_shot_source import SOURCE

    if hashlib.sha256(SOURCE.encode('utf-8')).hexdigest() != bundled_pin:
        raise AssertionError('入库参照夹具 few_shot 源码与 pin 不符')
    return SOURCE


def load_reference_few_shot(root: Path | None = None):
    """加载旧 few_shot 模块，并以标准包名 src.generator.few_shot 注册。

    归档存在 → 按路径加载归档字节；归档缺失 → 退回入库夹具源码（模块内函数级
    相对 import 经 src.generator 包解析，两种来源执行语义一致）。两种来源都先
    强校验字节 pin，再注册模块名——标准包名让 `from src.generator import
    few_shot`（旧装配入口与适配器测试）继续工作。

    root 仅供测试注入私有树根；默认取仓库根。
    """
    name = 'src.generator.few_shot'
    path = archived_few_shot_path(root)
    if path.is_file():
        source = path.read_text(encoding='utf-8')
        actual = hashlib.sha256(source.encode('utf-8')).hexdigest()
        if actual != ARCHIVED_FEW_SHOT_SHA256:
            raise AssertionError(f'归档 few_shot.py 字节与参照 pin 不符：{actual}')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    source = _bundled_source()
    actual = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if actual != ARCHIVED_FEW_SHOT_SHA256:
        raise AssertionError(f'入库参照 few_shot.py 字节与 pin 不符：{actual}')
    path = cached_few_shot_path(root)
    if not path.is_file():
        # 物化到真实文件：模块 __file__ 必须是磁盘路径，inspect.getsource 与
        # 磁盘适配器 checksum(module.__file__) 依赖它；占位字符串会炸。
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVED_FEW_SHOT_SHA256:
        raise AssertionError(f'物化参照 few_shot.py 字节与 pin 不符：{path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 进程内单例：parity 参照类与旧模块属性被多处 monkeypatch，必须共享同一模块对象。
few_shot = load_reference_few_shot()


# ---------------------------------------------------------------------------
# 归档冒烟：默认 skip；仅本机归档存在且显式 DH_ARCHIVE_TESTS=1 时运行。
# ---------------------------------------------------------------------------
def test_archive_reference_smoke():
    import os

    import pytest

    if not archived_few_shot_path().is_file():
        pytest.skip('本机认证归档不存在（干净 checkout / CI）')
    if os.environ.get('DH_ARCHIVE_TESTS') != '1':
        pytest.skip('归档冒烟为本地显式测试：DH_ARCHIVE_TESTS=1 开启')
    assert hashlib.sha256(archived_few_shot_path().read_bytes()).hexdigest() == \
        ARCHIVED_FEW_SHOT_SHA256
    previous = sys.modules.get('src.generator.few_shot')
    module = load_reference_few_shot()
    try:
        assert hasattr(module, 'PersonaFewShotRetriever')
        assert callable(module.PersonaFewShotRetriever)
    finally:
        if previous is not None:
            sys.modules['src.generator.few_shot'] = previous
