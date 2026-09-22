#!/usr/bin/env python3
"""Judge 评估包管理（SOP §7）。

用法：
  python scripts/evaluate_judge.py build --workers 2
      # 冻结评估包：当前生产基线生成 AI 回复（需 LLM）
  python scripts/evaluate_judge.py compare --pack <pack_id> --change "换 judge 模型" \
      --override llm.model=<新模型>
      # 候选 = 新 Judge 版本，同 pack 与现用比较识别率；通过则 promote.py judge --exp <id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import parse_overrides  # noqa: E402
from src.iteration import branch_packs, experiment, runner, versions  # noqa: E402


def cmd_build(args: argparse.Namespace) -> None:
    ref = branch_packs.prepare_initial(args.kind, args.sample)
    print(f"回复包已创建: {ref}；可用 iterate_branches.py worker --kind pack --job {ref} 恢复")
    branch_packs.build(ref, getattr(args, 'workers', 1))
    print(f"评估包已冻结: {versions.PRIVATE / 'judge_eval' / ref / 'pack.json'}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Judge 校准：候选 = 新 Judge 版本（--override 改模型/提示词），同一冻结 pack
    上与现用版本比较识别率（SOP §4.3）。通过后 promote.py judge --exp 转正。"""
    overrides = parse_overrides(args.override)
    dataset = "fixed_test" if "validation" in args.pack else "development"
    if dataset == "fixed_test" and overrides:
        raise SystemExit("固定轮禁止 override：候选 = 开发 Judge 基线锁定快照")
    exp_dir = experiment.create_judge_eval_experiment(dataset, args.pack, overrides,
                                                       change=args.change or str(overrides),
                                                       against_production=args.against_production,
                                                       candidate_ref=getattr(args, 'candidate', None),
                                                       saved_draw_manifest=getattr(args, 'saved_draw_manifest', None),
                                                       supplement_missing=getattr(args, 'supplement_missing_rounds', None))
    print(f"校准实验已创建: {exp_dir}")
    if experiment.state_of(exp_dir).get('status') != 'finished':
        runner.run_judge_experiment(exp_dir, workers=args.workers)
    state = experiment.state_of(exp_dir)
    print(f"结论: {state.get('verdict')} — {state.get('reason')}")
    if state.get("verdict") == "adopt":
        print(f"下一步: python scripts/promote.py judge --exp {exp_dir.name}")


def cmd_fuse(args: argparse.Namespace) -> None:
    from src.iteration.fusion import create
    print(create(args.gbdt, args.lr, args.data, args.change))


def cmd_embedding(args: argparse.Namespace) -> None:
    from src.iteration.embedding_adoption import create
    print(create(args.study, args.change))


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge 评估包")
    parser.add_argument("--instance", default=None, help="数字人实例名（默认 env DH_INSTANCE 或 default）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="构建冻结评估包（需 LLM）")
    b.add_argument("--kind", choices=["calibration", "validation"], default="calibration",
                   help="calibration=完整开发包；固定包由分支绑定验收批次后构建")
    b.add_argument("--sample", type=int, default=None,
                   help="可选的规模断言；省略时使用完整冻结清单，不重新抽样")
    b.add_argument("--workers", type=int, default=1)
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("compare", help="现用 vs 候选（新 Judge 版本）同包校准")
    c.add_argument("--pack", required=True)
    c.add_argument('--candidate', help='已冻结 Judge 版本；只用于开发轮，不能叠加 override')
    c.add_argument('--saved-draw-manifest', help='复用已核验的开发观察；仍核验独立轮次，禁止实时请求回退')
    c.add_argument('--supplement-missing-rounds', action='store_true',
                   help='显式允许为两个纯特征判别器共享补齐缺失的 r1/r2；已有观察不重抽')
    c.add_argument('--workers', type=int, default=1)
    c.add_argument('--against-production', action='store_true',
                   help='在原开发包比较锁定开发 Judge 与生产 Judge，复用完整直接比较')
    c.add_argument("--change", default=None, help="校准说明（改了什么）")
    c.add_argument("--override", action="append", default=[], help="候选 Judge 配置改动，如 llm.model=xxx")
    c.set_defaults(func=cmd_compare)
    f = sub.add_parser('fuse', help='冻结 80% GBDT + 20% LR 融合候选；仅使用训练特征确定尺度')
    f.add_argument('--gbdt', required=True)
    f.add_argument('--lr', required=True)
    f.add_argument('--data', required=True)
    f.add_argument('--change', required=True)
    f.set_defaults(func=cmd_fuse)
    e = sub.add_parser('embedding', help='冻结训练内部已选定的主候选；不重新搜索超参')
    e.add_argument('--study', required=True)
    e.add_argument('--change', required=True)
    e.set_defaults(func=cmd_embedding)
    args = parser.parse_args()
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")
    args.func(args)


if __name__ == "__main__":
    main()
