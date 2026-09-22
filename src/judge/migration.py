"""已有实例补齐 来源裁判 输入；只增加来源元数据，不重新抽题。"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from ..config import ConfigError, sha256_file
from ..iteration import versions
from .corrected import blind_case


def enrich_cases(data_dir: Path, messages: list[dict]) -> dict[str, list[dict]]:
    by_chat = defaultdict(list)
    for message in messages:
        by_chat[message["chat_id"]].append(message)
    result = {}
    for filename in ("dev_pool.jsonl", "fixed_test.jsonl"):
        cases = [json.loads(line) for line in (data_dir / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        for case in cases:
            chat, _, position = case["source_message_id"].rpartition(":")
            index = int(position)
            rows = by_chat[chat]
            width = len(case["context"])
            if index < width or index >= len(rows) or [rows[index]["text"]] != case["human_reply"] or not rows[index]["is_self"]:
                raise ConfigError(f"无法核对原始导出中的题目来源：{case['case_id']}")
            context = rows[index - width:index]
            if [{"sender": m["sender"], "text": m["text"]} for m in context] != [
                    {"sender": m["sender"], "text": m["text"]} for m in case["context"]]:
                raise ConfigError(f"原始导出与冻结上下文不一致：{case['case_id']}")
            case["context"] = [{key: m[key] for key in ("sender", "text", "is_self", "timestamp")} for m in context]
            case["source_chat_id"] = rows[index]["source_chat_id"]
            blind_case(case, case["human_reply"], ["迁移输入校验"])
        result[filename] = cases
    return result


def snapshot_enriched_data(old_id: str, cases: dict[str, list[dict]]) -> str:
    old = versions.data_version_dir(old_id)
    manifest = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    manifest["metadata_migration"] = {"source_data": old_id, "reason": "补齐 来源裁判 所需的本人角色、时间及来源聊天 ID；题目、答案与分集不变"}
    new_id = versions.create_data_version(None, manifest)
    new = versions.data_version_dir(new_id)
    # copyfile 不继承冻结文件的只读权限；原版本保持原样。
    shutil.copytree(old, new, dirs_exist_ok=True, copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns("manifest.json"))
    new.chmod(0o755)
    for filename, rows in cases.items():
        path = new / filename
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = json.loads((new / "manifest.json").read_text(encoding="utf-8"))
    for key, filename in [("development", "dev_pool.jsonl"), ("fixed_test", "fixed_test.jsonl")]:
        manifest["testsets"][key].update(path=str(new / filename), file_sha256=sha256_file(new / filename))
    (new / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tests_manifest = new / "testsets_manifest.json"
    if tests_manifest.exists():
        tests = json.loads(tests_manifest.read_text(encoding="utf-8"))
        for key in ("development", "fixed_test"):
            tests[key].update(manifest["testsets"][key])
        tests_manifest.write_text(json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8")
    versions.finalize_data_version(new_id)
    return new_id
