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
    ref = branch_packs.prepare_initial(args.kind, args.sample, args.reuse_generations)
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


def cmd_features(args: argparse.Namespace) -> None:
    from src.judge.feature_inspection import inspect
    inspect(Path(args.experiment), args.case, args.share, args.judge, Path(args.output),
            branch=args.branch, round_index=args.round, conditions=args.condition,
            memory_index=args.memory_index, memory_model=args.memory_model,
            reasoning_effort=args.reasoning_effort, extraction_mode=args.extraction_mode)
    print(f'特征诊断完成（不计入正式识别率）: {Path(args.output) / "report.json"}')


def cmd_ownership_report(args: argparse.Namespace) -> None:
    import json
    from src.judge.ownership_report import build_report
    directory = versions.PRIVATE / 'experiments' / args.experiment
    family = args.feature_family
    report = build_report(directory, family + '_features')
    path = directory / (family + '_report.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items()
                      if k != 'flagged_or_overridden_cases'}, ensure_ascii=False, indent=2))
    print(f'纠正统计（仅读已保存实录）: {path}')


def cmd_contribution(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from src.judge.feature_inspection import inspect_contribution
    def one(case_id):
        output = Path(args.output) / case_id
        inspect_contribution(Path(args.experiment), case_id, args.judge, output,
                             branch=args.branch, round_index=args.round)
        print(f'特征诊断完成（不计入正式识别率）: {output / "report.json"}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, dict.fromkeys(args.case)))


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
    b.add_argument('--reuse-generations', help='复用同数据同生成器的已有开发生成实录，只生成缺项')
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
    f = sub.add_parser('features', help='复用已有 case，诊断有无 Wiki 的背景/逻辑特征（仅诊断）')
    f.add_argument('--experiment', required=True)
    f.add_argument('--case', required=True)
    f.add_argument('--share', required=True, help='供诊断的已冻结共享资料版本')
    f.add_argument('--judge', required=True, help='沿用该版本的特征抽取模型配置')
    f.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'],
                   help='仅覆盖此次诊断的思考深度，记录到结果与请求缓存；不修改冻结 Judge')
    f.add_argument('--extraction-mode', choices=['joint', 'separate_context'], default='joint',
                   help='separate_context 先独立缓存纯上文归属，再只读检查回复；默认沿用一步抽取')
    f.add_argument('--branch', choices=['baseline', 'candidate'], default='candidate')
    f.add_argument('--round', type=int, default=0)
    f.add_argument('--output', required=True, help='可续跑的诊断输出目录')
    f.add_argument('--memory-index', help='可信本地 RPA 消息 BGE pickle 索引；也可用 WECHAT_HISTORY_INDEX_PATH')
    f.add_argument('--memory-model', help='本地 BGE ONNX 模型目录；也可用 WECHAT_BGE_MODEL_PATH')
    f.add_argument('--condition', action='append', choices=['context_only', 'with_wiki', 'with_background', 'with_memory'],
                   help='默认对比 context_only/with_wiki：按账号和已确认别名直接读取 Wiki；其余为旧诊断路径')
    f.set_defaults(func=cmd_features)
    o = sub.add_parser('ownership-report', help='从已有正式实录汇总归属纠正与真人误伤；不请求模型')
    o.add_argument('--experiment', required=True, help='Judge 实验 ID')
    o.add_argument('--feature-family', choices=['ownership', 'contribution'], default='ownership')
    o.set_defaults(func=cmd_ownership_report)
    f = sub.add_parser('contribution-features', help='复用已存盲测回复诊断信息贡献，不重新生成')
    f.add_argument('--experiment', required=True)
    f.add_argument('--case', action='append', required=True)
    f.add_argument('--judge', required=True)
    f.add_argument('--output', required=True)
    f.add_argument('--branch', choices=['baseline', 'candidate'], default='candidate')
    f.add_argument('--round', type=int, default=0)
    f.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    f.set_defaults(func=cmd_contribution)
    args = parser.parse_args()
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")
    args.func(args)


if __name__ == "__main__":
    main()
