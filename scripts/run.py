#!/usr/bin/env python3
"""实验唯一运行入口（SOP §2）：新建自动预检，已有按 ID 恢复。

  # 新建（自动预检：准入/候选版本/diff/one-shot）并运行
  python scripts/run.py --dataset development --change "唯一改动" --override k=v [--limit 5]

  # 在原执行器下恢复已有实验；代码变化后由调度器使用冻结快照恢复
  python scripts/run.py --exp <实验id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import parse_overrides  # noqa: E402
from src.iteration import experiment, runner, versions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="实验唯一入口：新建自动预检 / 已有按 ID 恢复")
    parser.add_argument("--instance", default=None, help="数字人实例名（默认 env DH_INSTANCE 或 default）")
    parser.add_argument("--exp", default=None, help="恢复已有实验（与新建参数互斥）")
    parser.add_argument("--dataset", choices=["development", "fixed_test"])
    parser.add_argument('--data', help='在指定数据版本上比较；不改变当前数据或模型指针')
    parser.add_argument('--purpose', choices=['gen_optimization', 'development', 'fixed_test'],
                        help='显式选择用途清单；Gen 优化结果不能晋升')
    parser.add_argument('--against-production', action='store_true',
                        help='在原开发集比较锁定开发版与生产版，核验固定轮准入')
    parser.add_argument("--change", default=None, help="本轮唯一改动（新建必填）")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None, help="冒烟只跑前 N 题（仅 development）")
    parser.add_argument("--create-only", action="store_true", help="只预检建实验不运行")
    parser.add_argument('--workers', type=int, default=None, help='Judge 同时处理题数（默认沿用实验配置）')
    args = parser.parse_args()
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")

    if args.exp:
        if args.data or args.purpose or args.against_production:
            parser.error('恢复实验时不能更换数据版本或用途')
        exp_dir = experiment.load_experiment(args.exp)
        spec = experiment.spec_of(exp_dir)
        print(f"恢复实验 {spec['id']}（{experiment.state_of(exp_dir).get('status', 'running')}）")
        if args.create_only:
            print("（--create-only 仅对新建有意义；恢复模式直接运行）")
            return
        if spec["kind"] == experiment.KIND_JUDGE_EVAL:
            runner.run_judge_experiment(exp_dir, workers=args.workers)
            state = experiment.state_of(exp_dir)
            print(f"\n结论: {state.get('verdict')} — {state.get('reason', '')}")
            if state.get("verdict") == "adopt":
                print(f"晋升: python scripts/promote.py judge --exp {exp_dir.name}")
            return
    else:
        if not args.dataset or not args.change:
            raise SystemExit("新建实验必须提供 --dataset 和 --change；恢复用 --exp <id>")
        overrides = parse_overrides(args.override)
        exp_dir = experiment.create_gen_experiment(
            dataset=args.dataset, single_change=args.change,
            overrides=overrides, limit=args.limit, data_ref=args.data, purpose=args.purpose,
            against_production=args.against_production,
        )
        spec = experiment.spec_of(exp_dir)
        print(f"预检通过，实验已创建: {exp_dir}")
        print(f"  角色 {spec['role']} | 基线 {spec['baseline_ref']} | 候选 {spec['candidate_ref']} "
              f"| Judge {spec['judge_ref']} | diff {spec['config_diff']}")
        if args.create_only:
            return

    if experiment.state_of(exp_dir).get('status') != 'finished':
        runner.run_gen_experiment(exp_dir)
    state = experiment.state_of(exp_dir)
    print(f"\n结论: {state.get('verdict')} — {state.get('reason', '')}")
    print(f"账单: {exp_dir}/bill.md | 后台: python scripts/serve_dashboard.py")
    if not spec.get("against_production") and state.get("verdict") in ("merge_to_iteration_baseline", "adopt"):
        print(f"晋升: python scripts/promote.py gen --exp {exp_dir.name}")


if __name__ == "__main__":
    main()
