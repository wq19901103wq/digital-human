#!/usr/bin/env python3
"""本地查询实验历史和缓存（不发起模型请求、不修改实验或生产指针）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="幂等导入历史记录，仅建立查询索引")
    sub.add_parser("stats")
    reuse = sub.add_parser("reuse-blind-orders", help="复用重建包的既有盲序；保留已分配顺序，不发起模型请求")
    reuse.add_argument("--experiment", required=True)
    entries = sub.add_parser("entries", help="按模型、实验和样本查缓存键；get 返回完整请求与结果")
    entries.add_argument("--layer")
    entries.add_argument("--model")
    entries.add_argument("--experiment")
    entries.add_argument("--case-id")
    entries.add_argument("--limit", type=int, default=20)
    history = sub.add_parser("history")
    history.add_argument("--experiment")
    history.add_argument("--case-id")
    history.add_argument("--data-ref")
    history.add_argument("--version")
    history.add_argument("--limit", type=int, default=20)
    get = sub.add_parser("get")
    get.add_argument("key")
    args = parser.parse_args()
    if Path(args.instance).name != args.instance or args.instance in {".", ".."}:
        parser.error("实例名不能包含路径")
    from src.iteration import versions
    instance = versions.switch_instance(args.instance)
    if not instance.is_dir():
        parser.error("实例不存在")
    store = cache.get_store(instance / ".cache")
    if args.command == "index":
        result = cache.import_history(instance)
    elif args.command == "stats":
        result = store.stats()
    elif args.command == "reuse-blind-orders":
        from src.config import sha256_file
        if Path(args.experiment).name != args.experiment or args.experiment in {".", ".."}:
            parser.error("实验名不能包含路径")
        spec = json.loads((instance / "experiments" / args.experiment / "spec.json").read_text())
        pack_root = instance / "judge_eval"
        pack_path = (pack_root / spec["pack_ref"] / "pack.json").resolve()
        if not pack_path.is_relative_to(pack_root.resolve()) or sha256_file(pack_path) != spec["pack_sha256"]:
            parser.error("评测包与冻结实验不一致")
        result = store.reuse_pack_blind_orders(json.loads(pack_path.read_text())["rows"])
    elif args.command == "get":
        result = store.get(args.key)
    elif args.command == "entries":
        result = store.entries(layer=args.layer, model=args.model, experiment_id=args.experiment,
                               case_id=args.case_id, limit=args.limit)
    else:
        result = store.history(experiment_id=args.experiment, case_id=args.case_id,
                               data_ref=args.data_ref, version=args.version, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
