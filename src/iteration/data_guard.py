"""续跑前核对冻结数据；不更换样本、不删除断点、不替代计算层缓存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.infrastructure.config import ConfigError
from src.infrastructure.config import sha256_file
from src.infrastructure.storage import write_json
from . import datasets


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


# 检索实现的完整实现清单：排序/召回语义由这四件共同决定，指纹必须全量绑定
# （只绑 session.py 时改动 rank.py 等不会使冻结指纹失配，断点会被脏实现续跑）。
RETRIEVAL_IMPLEMENTATION = tuple('generator/fewshot/' + name for name in
                                 ('session.py', 'prepare.py', 'rank.py', 'recall_engines.py'))

# 过渡兼容：清单扩展前的旧形态实现键集（few_shot.py 单件时代）。旧断点/旧包
# 按归档字节核验（runtime 归档才是旧运行的身份），不得要求重跑已完成题目；
# 旧形态只认 LEGACY_IMPLEMENTATION_KEYS 的三种键集，不做模糊匹配。
LEGACY_IMPLEMENTATION = ('generator/generator.py', 'generator/few_shot.py',
                         'generator/history.py', 'iteration/datasets.py')

# 清单逐代扩展过程中出现过的旧形态：fewshot/session.py 替换 few_shot.py（四件
# 键集），以及只绑 session.py 单件的形态。实现件全部按归档字节核验。
LEGACY_IMPLEMENTATION_SESSION = ('generator/generator.py', 'generator/history.py',
                                 'iteration/datasets.py', 'generator/fewshot/session.py')
LEGACY_IMPLEMENTATION_KEYS = {frozenset(LEGACY_IMPLEMENTATION),
                              frozenset(LEGACY_IMPLEMENTATION_SESSION),
                              frozenset(('generator/fewshot/session.py',))}

# 旧形态实现键 → runtime 归档相对名（含历史 src/digital_human/ 前缀的回退匹配）。
_LEGACY_ARCHIVE_NAMES = {
    'generator/generator.py': 'src/generator/generator.py',
    'generator/few_shot.py': 'src/generator/few_shot.py',
    'generator/history.py': 'src/generator/history.py',
    'generator/fewshot/session.py': 'src/generator/fewshot/session.py',
    'iteration/datasets.py': 'src/iteration/datasets.py',
}


def _is_legacy_shape(snapshot: dict) -> bool:
    implementation = snapshot.get('implementation_sha256')
    return isinstance(implementation, dict) and frozenset(implementation) in \
        LEGACY_IMPLEMENTATION_KEYS


def _legacy_implementation(directory: Path, keys) -> dict:
    """旧形态快照的 implementation_sha256：全部实现件一律按 runtime 归档字节
    核验（活树实现已演进/已删除，现值不再是旧运行的身份）；归档缺失或归档中
    没有该文件时按失配处理（禁止续跑）。"""
    from . import runtime
    try:
        archive = runtime.verify(directory / 'runtime.json', check_environment=False)
        files = json.loads((archive / 'manifest.json').read_text(encoding='utf-8'))['files']
    except (OSError, KeyError, json.JSONDecodeError, ConfigError) as exc:
        raise ConfigError('旧形态数据指纹缺少可核验的运行归档证据（runtime.json 缺失、'
                          '无效或已变化），无法按归档字节确认旧实现，禁止续跑') from exc
    result = {}
    for name in keys:
        relative = _LEGACY_ARCHIVE_NAMES.get(name)
        if relative is None:
            raise ConfigError(f'旧形态数据指纹含未知实现键 {name}，无法按归档字节确认，禁止续跑')
        archived = next((files[candidate] for candidate in
                         (relative, relative.replace('src/', 'src/digital_human/', 1))
                         if candidate in files), None)
        if archived is None:
            raise ConfigError('旧形态数据指纹缺少归档证据：runtime 归档未包含 '
                              f'{relative}，无法确认旧实现，禁止续跑')
        result[name] = archived
    return result


def generation_snapshot(cases: list[dict], data_dir: Path, configs: list[dict]) -> dict:
    ids = [str(case["case_id"]) for case in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ConfigError("实验数据为空或存在重复 case_id，禁止按 ID 续跑")
    snapshot = {"schema": 1, "cases_sha256": _digest(cases), "case_count": len(cases)}
    snapshot.update(datasets.snapshot(data_dir))
    if 'purposes_sha256' in snapshot:
        source = Path(__file__).parents[1]
        snapshot['implementation_sha256'] = {name: sha256_file(source / name) for name in
            ('generator/generator.py', 'generator/history.py', 'iteration/datasets.py',
             *RETRIEVAL_IMPLEMENTATION)}
    # 检索关闭时池不参与计算；开启时，池内容及审批条件都属于输入依赖。
    if any(config.get("retriever", {}).get("enabled", True) for config in configs):
        pool = data_dir / "fewshot_pool.jsonl"
        snapshot["fewshot_pool_sha256"] = sha256_file(pool) if pool.exists() else None
        report = data_dir / "report.json"
        approval = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        snapshot["fewshot_approval_sha256"] = _digest({
            key: approval.get(key) for key in ("review_status", "examples_sha256")})
    return snapshot


def pack_snapshot(sha256: str) -> dict:
    return {"schema": 1, "pack_sha256": sha256}


def verify(directory: Path, current: dict, *, expected: dict | None = None,
           has_checkpoint: bool = False) -> None:
    """调用方持有运行锁。创建时快照优先；旧任务仅在未产出前建立快照。

    独立文件保存首次运行的指纹，避免事后把当前输入补签成旧结果的来源。
    旧任务已有完整 pack 指纹时可据此验证；缺乏证据的旧断点只保留，不猜测。
    """
    path = directory / "data_snapshot.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    for original in (expected, saved):
        if original is None:
            continue
        # 过渡兼容：旧形态快照与“旧形态 current”（全部实现件取归档字节）比较；
        # 数据类输入（池/cases/purposes）仍按活树现值。新形态照旧与 current 比较。
        candidate = {**current, 'implementation_sha256':
                     _legacy_implementation(directory, original['implementation_sha256'])} \
            if _is_legacy_shape(original) else current
        if original != candidate:
            changed = sorted(key for key in original.keys() | candidate.keys()
                             if original.get(key) != candidate.get(key))
            raise ConfigError(f"冻结数据指纹不一致（{', '.join(changed)}），禁止复用旧断点或改变本轮样本；"
                              "请恢复原始数据后续跑，变更数据应使用新版本、新实验")
    if expected is None and saved is None and has_checkpoint:
        raise ConfigError("旧断点缺少可验证的数据指纹，无法确认原始输入；已保留旧结果，禁止按 case_id 直接复用")
    if saved is None:
        write_json(path, current)


def has_results(directory: Path) -> bool:
    checkpoint = directory / "cases.jsonl"
    return (checkpoint.exists() and bool(checkpoint.read_text(encoding="utf-8").strip())) or \
        any((directory / "traces").glob("*.json"))
