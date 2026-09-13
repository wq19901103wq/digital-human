#!/usr/bin/env python3
"""晋升：校验实验结论 → 候选版本转正 → 切指针（SOP §6/§7）。

用法：
  python scripts/promote.py gen --exp <实验id>     # 开发/固定生成器实验
  python scripts/promote.py judge --exp <实验id>   # Judge 评估实验（首验或比较晋升）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.digital_human.iteration import promote, versions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="晋升（唯一入口）")
    parser.add_argument("--instance", default=None, help="数字人实例名（默认 env DH_INSTANCE 或 default）")
    sub = parser.add_subparsers(dest="kind", required=True)
    g = sub.add_parser("gen", help="生成器实验晋升")
    g.add_argument("--exp", required=True)
    j = sub.add_parser("judge", help="Judge 评估实验转正（establish/compare 之后）")
    j.add_argument("--exp", required=True)
    args = parser.parse_args()
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")

    prev_prod = versions.load_pointers()["production_gen"]
    pointers = promote.promote_gen(args.exp) if args.kind == "gen" else promote.promote_judge(args.exp)
    print("指针:", pointers)
    if args.kind == "gen" and pointers["production_gen"] != prev_prod:
        print("注意：生产生成器已更新；Judge 无需重训（对抗绑定已移除，SOP §4），历史识别率按 Judge 版本断代")


if __name__ == "__main__":
    main()
