"""续跑前核对冻结数据；不更换样本、不删除断点、不替代计算层缓存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import ConfigError, sha256_file
from .storage import write_json
from . import datasets


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def generation_snapshot(cases: list[dict], data_dir: Path, configs: list[dict]) -> dict:
    ids = [str(case["case_id"]) for case in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ConfigError("实验数据为空或存在重复 case_id，禁止按 ID 续跑")
    snapshot = {"schema": 1, "cases_sha256": _digest(cases), "case_count": len(cases)}
    snapshot.update(datasets.snapshot(data_dir))
    if 'purposes_sha256' in snapshot:
        source = Path(__file__).parents[1]
        snapshot['implementation_sha256'] = {name: sha256_file(source / name) for name in
            ('generator/generator.py', 'generator/few_shot.py', 'generator/history.py', 'iteration/datasets.py')}
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
        if original is not None and original != current:
            changed = sorted(key for key in original.keys() | current.keys()
                             if original.get(key) != current.get(key))
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
