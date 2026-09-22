#!/usr/bin/env python3
"""晋升：校验实验结论 → 候选版本转正 → 切指针（SOP §6/§7）。

用法：
  python scripts/promote.py gen --exp <实验id>     # 开发/固定生成器实验
  python scripts/promote.py judge --exp <实验id>   # Judge 评估实验（首验或比较晋升）
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def review_policy():
    """Use the revised policy with the experiment's verified evidence executor."""
    from src import iteration
    for name in ('protocol', 'promote'):
        qualified = f'src.iteration.{name}'
        source = ROOT / 'src/iteration' / f'{name}.py'
        spec = importlib.util.spec_from_file_location(qualified, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        setattr(iteration, name, module)
        spec.loader.exec_module(module)
    return iteration.promote


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="晋升（唯一入口）")
    parser.add_argument("--instance", default=None, help="数字人实例名（默认 env DH_INSTANCE 或 default）")
    sub = parser.add_subparsers(dest="kind", required=True)
    g = sub.add_parser("gen", help="生成器实验晋升")
    g.add_argument("--exp", required=True)
    j = sub.add_parser("judge", help="Judge 评估实验转正（establish/compare 之后）")
    j.add_argument("--exp", required=True)
    for command in (g, j):
        command.add_argument('--threshold-reason', default=None,
            help='明确修订门槛时：按当前配置复核已有固定成绩并采用，记录用户决定，保留原结论')
        command.add_argument('--evidence-runtime', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.evidence_runtime:
        if args.threshold_reason is None:
            parser.error('冻结来源复核仅用于明确修订门槛')
        sys.path.insert(0, str(Path(args.evidence_runtime).resolve()))
    from src.config import ConfigError
    from src.iteration import experiment, promote, runtime, versions
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")

    if args.threshold_reason is not None:
        directory = experiment.load_experiment(args.exp)
        descriptor = directory / 'runtime.json'
        if descriptor.exists() or args.evidence_runtime:
            snapshot = runtime.verify(descriptor)
            if args.evidence_runtime:
                if (snapshot.resolve() != Path(args.evidence_runtime).resolve() or
                        Path(runtime.__file__).resolve() != snapshot / 'src/iteration/runtime.py'):
                    raise ConfigError('门槛复核必须使用此实验的原冻结执行器')
                promote = review_policy()
            else:
                # A separate process prevents the current runner from contaminating
                # historical evidence checks. No evaluation or model request runs.
                command = [sys.executable, str(Path(__file__).resolve())]
                if args.instance:
                    command += ['--instance', args.instance]
                command += [args.kind, '--exp', args.exp, '--threshold-reason', args.threshold_reason,
                            '--evidence-runtime', str(snapshot)]
                subprocess.run(command, check=True)
                return

    prev_prod = versions.load_pointers()["production_gen"]
    promote_fn = promote.promote_gen if args.kind == 'gen' else promote.promote_judge
    pointers = promote_fn(args.exp, threshold_reason=args.threshold_reason)
    print("指针:", pointers)
    if args.kind == "gen" and pointers["production_gen"] != prev_prod:
        print("注意：生产生成器已更新；Judge 无需重训（对抗绑定已移除，SOP §4），历史识别率按 Judge 版本断代")


if __name__ == "__main__":
    main()
