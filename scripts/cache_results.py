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
