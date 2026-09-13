#!/usr/bin/env python3
"""Judge 评估包管理（SOP §7）。

用法：
  python scripts/evaluate_judge.py build --sample 500
      # 冻结评估包：当前生产基线生成 AI 回复（需 LLM）
  python scripts/evaluate_judge.py compare --pack <pack_id> --change "换 judge 模型" \
      --override llm.model=<新模型>
      # 候选 = 新 Judge 版本，同 pack 与现用比较识别率；通过则 promote.py judge --exp <id>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.digital_human.config import ConfigError, load_settings, resolve  # noqa: E402
from src.digital_human.generator.generator import ReplyGenerator  # noqa: E402
from src.digital_human.iteration import experiment, runner, versions  # noqa: E402
from src.digital_human.llm import ChatClient  # noqa: E402

from src.digital_human.llm import build_clients  # noqa: E402


def _stratified_sample(cases: list, sample: int, group_ratio: float, seed: int = 42) -> list:
    """从整个开发集按聊天类型分层抽样（不用尾部切片——顺序无关，SOP §4.3）。"""
    import random as _random
    rng = _random.Random(seed)
    by_type = {"group": [c for c in cases if c.get("chat_type") == "group"],
               "private": [c for c in cases if c.get("chat_type") != "group"]}
    group_n = max(1, round(sample * group_ratio)) if by_type["group"] else 0
    private_n = max(1, sample - group_n) if by_type["private"] else 0
    rng.shuffle(by_type["group"])
    rng.shuffle(by_type["private"])
    return by_type["group"][:group_n] + by_type["private"][:private_n]


def cmd_build(args: argparse.Namespace) -> None:
    settings = load_settings()
    pointers = versions.load_pointers()
    production = versions.load_generator(pointers["production_gen"])
    data_dir = versions.data_version_dir(pointers["data"])
    if args.sample < 2:
        raise ConfigError("--sample 至少 2（按聊天类型分层抽样，单边至少 1 条）")
    source_file = "dev_pool.jsonl" if args.kind == "calibration" else "fixed_test.jsonl"
    ratio_key = "development" if args.kind == "calibration" else "fixed_test"
    pool = [json.loads(l) for l in (data_dir / source_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    ratio = float(settings["evaluation"][ratio_key]["group_ratio"])
    cases = _stratified_sample(pool, args.sample, ratio)
    if not cases:
        raise ConfigError("校准包为空：开发集无可抽样本")
    g = sum(1 for c in cases if c.get("chat_type") == "group")
    print(f"用生产基线 {production['id']} 为 {len(cases)} 条{args.kind}包题生成 AI 回复（群聊 {g} / 私聊 {len(cases) - g}）…")
    gen = ReplyGenerator(settings, production["config"],
                          build_clients(settings, production["config"]["llm"]),
                          prompt_root=production["dir"], pool_path=data_dir / "fewshot_pool.jsonl")
    rows = []
    for i, case in enumerate(cases):
        rows.append({
            "case_id": case["case_id"], "chat_type": case.get("chat_type"),
            "chat_name": case.get("chat_name"), "source_chat_id": case.get("source_chat_id"),
            "context": case.get("context"), "human_reply": case.get("human_reply"),
            "ai_replies": gen.generate(case)["replies"],
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cases)}")
    pack_id = time.strftime(f"pack-{args.kind}-%Y%m%d-%H%M%S")
    pack_dir = versions.PRIVATE / "judge_eval" / pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "pack.json").write_text(json.dumps({
        "pack_id": pack_id, "c0_gen_version": production["id"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"评估包已冻结: {pack_dir}/pack.json")


def cmd_compare(args: argparse.Namespace) -> None:
    """Judge 校准：候选 = 新 Judge 版本（--override 改模型/提示词），同一冻结 pack
    上与现用版本比较识别率（SOP §4.3）。通过后 promote.py judge --exp 转正。"""
    overrides: dict = {}
    for raw in args.override:
        key, _, value = raw.partition("=")
        parts = key.split(".")
        cursor = overrides
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        try:
            parsed: object = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
        cursor[parts[-1]] = parsed
    dataset = "fixed_test" if "validation" in args.pack else "development"
    if dataset == "fixed_test" and overrides:
        raise SystemExit("固定轮禁止 override：候选 = 开发 Judge 基线锁定快照")
    exp_dir = experiment.create_judge_eval_experiment(dataset, args.pack, overrides,
                                                       change=args.change or str(overrides))
    print(f"校准实验已创建: {exp_dir}")
    runner.run_judge_experiment(exp_dir)
    state = experiment.state_of(exp_dir)
    print(f"结论: {state.get('verdict')} — {state.get('reason')}")
    if state.get("verdict") == "adopt":
        print(f"下一步: python scripts/promote.py judge --exp {exp_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge 评估包")
    parser.add_argument("--instance", default=None, help="数字人实例名（默认 env DH_INSTANCE 或 default）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="构建冻结评估包（需 LLM）")
    b.add_argument("--kind", choices=["calibration", "validation"], default="calibration",
                   help="calibration=开发轮（dev 抽样）；validation=固定轮（fixed 抽样，one-shot）")
    b.add_argument("--sample", type=int, default=500)
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("compare", help="现用 vs 候选（新 Judge 版本）同包校准")
    c.add_argument("--pack", required=True)
    c.add_argument("--change", default=None, help="校准说明（改了什么）")
    c.add_argument("--override", action="append", default=[], help="候选 Judge 配置改动，如 llm.model=xxx")
    c.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")
    args.func(args)


if __name__ == "__main__":
    main()
