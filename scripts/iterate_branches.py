#!/usr/bin/env python3
"""多分支迭代：提交提案 → 冒烟/开发/固定验收 → 推全 → 其他分支重测。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.iteration import branches, experiment, runner, scheduler, versions, jobs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    submit = sub.add_parser("submit", help="新分支或已有分支的新提案；只冻结，不启动任务")
    submit.add_argument("--name", required=True)
    submit.add_argument("--kind", choices=["gen", "judge"], required=True)
    submit.add_argument("--change", required=True)
    candidate = submit.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--overrides", help="JSON 对象，如 '{\"llm\":{\"model\":\"m2\"}}'")
    candidate.add_argument("--candidate", help="已有不可变候选版本 ID")
    candidate.add_argument("--prompt-recipe", choices=['mechanical', 'conversation',
                                                      'conversation_partner', 'conversation_rhythm',
                                                      'conversation_rhythm_partner'],
                           help="使用已审阅的通用生成指令，保留分支当前开发版配置")
    submit.add_argument("--development-pack")
    submit.add_argument("--validation-pack")
    submit.add_argument("--saved-draw-manifest", help="复用已核验的开发观察；缺失轮次不请求模型")
    submit.add_argument("--from-branch", help="继承同条件下已通过的分支开发版作为对照和改动起点")
    run = sub.add_parser("run", help="启动调度；分支通过固定验收后自动推全")
    run.add_argument("--max-experiments", type=int, default=2)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument('--timeout-seconds', type=float, help='提高 ChatClient 等待下限，保留冻结请求与断点')
    run.add_argument("--once", action="store_true", help="只推进/调度一次，已启动的子进程继续运行")
    sub.add_parser("status", help="只读查看各分支及任务状态")
    limit = sub.add_parser('limit', help='持久设置分支自动推进的阶段上限')
    limit.add_argument('--name', required=True)
    limit.add_argument('--stage', choices=['development', 'fixed_test'], required=True)
    for command in ("pause", "resume"):
        sub.add_parser(command).add_argument("--name", required=True)
    retry = sub.add_parser("retry", help="重置耗尽的调度重试次数，逐题断点保留")
    retry.add_argument("--kind", choices=list(jobs.FOLDERS), required=True)
    retry.add_argument("--job", required=True)
    recovery = sub.add_parser("recover-pack", help="从冻结实录恢复未落盘的终止格式失败；不覆盖成功结果")
    recovery.add_argument("--job", required=True)
    worker = sub.add_parser("worker", help="调度器的独立进程入口")
    worker.add_argument("--kind", choices=list(jobs.FOLDERS), required=True)
    worker.add_argument("--job", required=True)
    worker.add_argument("--workers", type=int, default=4)
    worker.add_argument("--timeout-seconds", type=float, help="仅提高请求等待上限，保留冻结代码与已有结果")
    cancel = sub.add_parser("cancel", help="取消指定任务；已用验收批次不会退回")
    cancel.add_argument("--kind", choices=list(jobs.FOLDERS), required=True)
    cancel.add_argument("--job", required=True)
    train = sub.add_parser("train", help="提交人工配置的训练流程 JSON")
    train.add_argument("--name", required=True)
    train.add_argument("--plan", required=True)
    sealed = sub.add_parser("seal", help="在任何固定验收前封存批次")
    sealed.add_argument("--data", required=True)
    sealed.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.instance:
        branches._name(args.instance)
        versions.switch_instance(args.instance)
    if args.command == "submit":
        overrides = json.loads(args.overrides) if args.overrides else None
        if overrides is not None and not isinstance(overrides, dict):
            parser.error("--overrides 必须是 JSON 对象")
        result = branches.submit(args.name, args.kind, args.change, overrides=overrides,
                                 candidate_ref=args.candidate, development_pack=args.development_pack,
                                 validation_pack=args.validation_pack,
                                 saved_draw_manifest=args.saved_draw_manifest,
                                 prompt_recipe=args.prompt_recipe, from_branch=args.from_branch)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "run":
        scheduler.Scheduler(max_experiments=args.max_experiments, workers=args.workers,
                            max_attempts=args.max_attempts, timeout_seconds=args.timeout_seconds).run(once=args.once)
    elif args.command == 'limit':
        branches.set_stage_limit(args.name, args.stage)
    elif args.command == "status":
        print(json.dumps({"production": branches.basis(), "branches": branches.status(),
                          "jobs": scheduler.jobs_status()}, ensure_ascii=False, indent=2))
    elif args.command in {"pause", "resume"}:
        branches.set_paused(args.name, args.command == "pause")
    elif args.command == "retry":
        scheduler.retry(args.kind, args.job)
    elif args.command == 'recover-pack':
        from src.iteration.branch_packs import recover_terminal_failures
        print(json.dumps(recover_terminal_failures(args.job), ensure_ascii=False))
    elif args.command == 'cancel':
        from src.iteration import control
        branches._name(args.job)
        control.cancel(jobs.directory({'kind': args.kind, 'id': args.job}))
    elif args.command == 'train':
        from src.iteration import training
        print(training.submit(args.name, json.loads(Path(args.plan).read_text())))
    elif args.command == 'seal':
        from src.iteration import acceptance
        print(json.dumps(acceptance.seal(args.data, args.batch_size)))
    else:
        from src.iteration import control, runtime
        branches._name(args.job)
        directory = jobs.directory({'kind': args.kind, 'id': args.job})
        if args.timeout_seconds is not None:
            from src.iteration.pack_transport import launch
            raise SystemExit(launch(directory, instance=versions.PRIVATE.name, job=args.job,
                                    workers=args.workers, seconds=args.timeout_seconds, kind=args.kind))
        runtime.require_current(directory)
        with control.job(directory):
            work(args)


def work(args):
    if args.kind == 'training':
        from src.iteration import training
        training.run(args.job, args.workers)
    elif args.kind == "pack":
        from src.iteration.branch_packs import build
        build(args.job, args.workers)
    else:
        branches._name(args.job)
        directory = experiment.load_experiment(args.job)
        if experiment.spec_of(directory)["kind"] == experiment.KIND_JUDGE_EVAL:
            runner.run_judge_experiment(directory, workers=args.workers)
        else:
            runner.run_gen_experiment(directory, workers=args.workers)


if __name__ == "__main__":
    main()
