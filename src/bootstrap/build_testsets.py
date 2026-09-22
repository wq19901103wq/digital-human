"""开发/固定测试集构建（机制层）：从各自分区的消息抽题，带锁定与防重叠。

SOP §9：fixed/dev 是两个独立分区（由 partition.py 整块切出），本模块
直接从各自分区抽题，不跨分区重采样——会话隔离在切分层保证，这里不再打乱。

- 构成按 settings 的 total + group_ratio 截断到可用量
- 固定集先写、开发集后写，两者按 source_message_id 严格不相交
- 固定集写 manifest（case ID 清单 + SHA-256，frozen=true），封存后只回总分
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ..config import ConfigError, dataset_plan
from .build_fewshot_pool import extract_examples


def load_protected_indices(*jsonl_paths: Path) -> dict[str, set[int]]:
    """从开发/固定测试集收集受保护的 source 位置 {chat_id: {index}}（SOP §9.2 邻接保护）。"""
    protected: dict[str, set[int]] = {}
    for path in jsonl_paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sid = str(json.loads(line).get("source_message_id", ""))
            if ":" not in sid:
                continue
            chat_id, _, index = sid.rpartition(":")
            try:
                protected.setdefault(chat_id, set()).add(int(index))
            except ValueError:
                continue
    return protected


def _next_manifest_version(manifest_path: Path) -> int:
    try:
        return int(json.loads(manifest_path.read_text(encoding="utf-8")).get("version", 1)) + 1
    except (OSError, json.JSONDecodeError, ValueError):
        return 1


def _to_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["id"],
        "chat_type": row["relationship"],
        "chat_name": row["chat_name"],
        "source_chat_id": row.get("source_chat_id"),
        "context": [
            dict(cm)
            for cm in row.get("context_messages", [])
        ],
        "human_reply": row["reply"],
        "source_message_id": row["source_message_id"],
    }


def _write_jsonl(cases: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_testsets(
    fixed_messages: list[dict[str, Any]],
    dev_messages: list[dict[str, Any]],
    settings: dict[str, Any],
    dev_path: Path,
    fixed_path: Path,
    seed: int = 42,
    refreeze: bool = False,
    refreeze_reason: str = "",
) -> dict[str, Any]:
    """SOP §1.3：测试集为冻结资产。已存在时必须 refreeze + 原因，版本 +1。"""
    if (dev_path.exists() or fixed_path.exists()) and not refreeze:
        raise ConfigError(
            f"测试集已存在（冻结资产）: {dev_path} / {fixed_path}\n"
            "重建将使历史 A/B 结果不可比。如确需重建，使用 --refreeze "
            "--refreeze-reason <原因>"
        )
    if refreeze and not refreeze_reason.strip():
        raise ConfigError("--refreeze 必须提供 --refreeze-reason（写入 manifest）")

    manifest: dict[str, Any] = {
        "seed": seed,
        "frozen": True,
        "version": _next_manifest_version(fixed_path.parent / "testsets_manifest.json"),
        "refreeze_reason": refreeze_reason or "初始构建",
        "split": "by_chat：fixed/dev 分区由 partition.py 整块切出，构建不跨分区重采样",
    }

    for dataset, messages, path in (
        ("fixed_test", fixed_messages, fixed_path),
        ("development", dev_messages, dev_path),
    ):
        plan = dataset_plan(settings, dataset)
        examples = extract_examples(messages)
        rng = random.Random(seed)
        rng.shuffle(examples)
        group_pool = [e for e in examples if e["relationship"] == "group"]
        private_pool = [e for e in examples if e["relationship"] != "group"]
        # 每聊天题数上限（settings.max_cases_per_chat，默认 25）：1000 题应来自 40+ 聊天，
        # 防同一聊天内上下文前缀重叠的题海支配识别率指标
        cap = int(settings["evaluation"].get("max_cases_per_chat", 25))
        per_chat: dict[str, int] = {}

        def _take(pool: list, quota: int) -> list:
            out = []
            for e in pool:
                if len(out) >= quota:
                    break
                cid = str(e.get("source_chat_id") or str(e.get("source_message_id", "")).split(":")[0])
                if per_chat.get(cid, 0) >= cap:
                    continue
                per_chat[cid] = per_chat.get(cid, 0) + 1
                out.append(e)
            return out

        group_sel = _take(group_pool, plan["group"])
        private_sel = _take(private_pool, plan["private"])
        # 群聊/私聊交错 + 播种打乱：任何"取前/尾 N 条"的抽样都不偏向单一类型
        selected = [c for pair in zip(group_sel, private_sel) for c in pair]
        selected += group_sel[len(private_sel):] + private_sel[len(group_sel):]
        rng.shuffle(selected)
        if len(selected) < plan["total"]:
            print(
                f"警告: {dataset} 可用样本不足（{len(selected)}/{plan['total']}），"
                "请补充数据或调低 config/settings.yaml 的 total"
            )
        cases = [_to_case(e) for e in selected]
        file_sha = _write_jsonl(cases, path)
        ids_sha = hashlib.sha256(
            "\n".join(sorted(c["case_id"] for c in cases)).encode("utf-8")
        ).hexdigest()
        manifest[dataset] = {
            "path": str(path),
            "total": len(cases),
            "group": sum(1 for c in cases if c["chat_type"] == "group"),
            "private": sum(1 for c in cases if c["chat_type"] != "group"),
            "file_sha256": file_sha,
            "ids_sha256": ids_sha,
        }
        print(f"已生成 {dataset}: {manifest[dataset]}")

    manifest_path = fixed_path.parent / "testsets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest 已写入 {manifest_path}（fixed_test 已冻结，勿手工改动）")
    return manifest
