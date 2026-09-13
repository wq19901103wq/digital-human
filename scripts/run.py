#!/usr/bin/env python3
"""实验唯一运行入口（SOP §2）：新建自动预检，已有按 ID 恢复。

  # 新建（自动预检：准入/候选版本/diff/one-shot）并运行
  python scripts/run.py --dataset development --change "唯一改动" --override k=v [--limit 5]

  # 恢复已有实验（成功题不重抽；改代码/配置后同一入口续跑）
  python scripts/run.py --exp <实验id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.digital_human.iteration import experiment, runner, versions  # noqa: E402


def _parse_override(raw: str) -> dict:
    key, sep, value = raw.partition("=")
    if not key or not sep:
        raise SystemExit(f"--override 格式错误: {raw}（应为 key=value）")
    parsed: object
    if value.lower() in ("true", "false"):
        parsed = value.lower() == "true"
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)  # 温度等小数参数
            except ValueError:
                parsed = value
    parts = key.split(".")
    out: dict = {}
    cursor = out
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = parsed
    return out


def _deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def main() -> None:
    parser = argparse.ArgumentParser(description="实验唯一入口：新建自动预检 / 已有按 ID 恢复")
    parser.add_argument("--instance", default=None, help="数字人实例名（默认 env DH_INSTANCE 或 default）")
    parser.add_argument("--exp", default=None, help="恢复已有实验（与新建参数互斥）")
    parser.add_argument("--dataset", choices=["development", "fixed_test"])
    parser.add_argument("--change", default=None, help="本轮唯一改动（新建必填）")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None, help="冒烟只跑前 N 题（仅 development）")
    parser.add_argument("--create-only", action="store_true", help="只预检建实验不运行")
    args = parser.parse_args()
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")

    if args.exp:
        exp_dir = experiment.load_experiment(args.exp)
        spec = experiment.spec_of(exp_dir)
        print(f"恢复实验 {spec['id']}（{experiment.state_of(exp_dir).get('status', 'running')}）")
        if args.create_only:
            print("（--create-only 仅对新建有意义；恢复模式直接运行）")
            return
        if spec["kind"] == experiment.KIND_JUDGE_EVAL:
            runner.run_judge_experiment(exp_dir)
            state = experiment.state_of(exp_dir)
            print(f"\n结论: {state.get('verdict')} — {state.get('reason', '')}")
            if state.get("verdict") == "adopt":
                print(f"晋升: python scripts/promote.py judge --exp {exp_dir.name}")
            return
    else:
        if not args.dataset or not args.change:
            raise SystemExit("新建实验必须提供 --dataset 和 --change；恢复用 --exp <id>")
        overrides: dict = {}
        for raw in args.override:
            _deep_merge(overrides, _parse_override(raw))
        exp_dir = experiment.create_gen_experiment(
            dataset=args.dataset, single_change=args.change,
            overrides=overrides, limit=args.limit,
        )
        spec = experiment.spec_of(exp_dir)
        print(f"预检通过，实验已创建: {exp_dir}")
        print(f"  角色 {spec['role']} | 基线 {spec['baseline_ref']} | 候选 {spec['candidate_ref']} "
              f"| Judge {spec['judge_ref']} | diff {spec['config_diff']}")
        if args.create_only:
            return

    runner.run_gen_experiment(exp_dir)
    state = experiment.state_of(exp_dir)
    print(f"\n结论: {state.get('verdict')} — {state.get('reason', '')}")
    print(f"账单: {exp_dir}/bill.md | 后台: python scripts/serve_dashboard.py")
    if state.get("verdict") in ("merge_to_iteration_baseline", "adopt"):
        print(f"晋升: python scripts/promote.py gen --exp {exp_dir.name}")


if __name__ == "__main__":
    main()
