#!/usr/bin/env python3
"""bootstrap：原始聊天记录 → 数据版本 + 初始生成器/Judge 版本 + 指针。

每次运行创建**新的数据版本目录**（旧版本原地保留、永远只读）；
--refreeze 语义 = 新建版本并切换指针（历史实验引用旧版本，可复现）。

用法：
  python scripts/bootstrap.py --data data/chat_export.jsonl [--model <生成模型>]
      [--judge-model <模型>] [--adopt]
      [--no-llm] [--force] [--refreeze-reason <原因>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.digital_human.bootstrap import (  # noqa: E402
    analyze,
    build_fewshot_pool,
    build_persona,
    build_scenarios,
    build_testsets,
    ingest,
    partition,
)
from src.digital_human.config import ConfigError, load_settings, resolve  # noqa: E402
from src.digital_human.iteration import versions  # noqa: E402
from src.digital_human.llm import ChatClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="从聊天数据创建数据版本与初始基线")
    parser.add_argument("--instance", default=None, help="数字人实例名（默认 env DH_INSTANCE 或 default）")
    parser.add_argument("--data", required=True, help="统一 jsonl 聊天记录路径")
    parser.add_argument("--model", default="", help="生成模型名")
    parser.add_argument("--judge-model", default="", help="Judge 模型名（默认与 --model 相同）")
    parser.add_argument("--no-llm", action="store_true", help="不调 LLM，统计兜底")
    parser.add_argument("--force", action="store_true", help="覆盖场景文件（默认保留人工精调）")
    parser.add_argument("--refreeze-reason", default="", help="重建原因（写入 manifest）")
    parser.add_argument("--adopt", action="store_true",
                        help="采用新数据：为当前配置新建生成器/Judge 版本（新快照）并切换全部指针；\n"
                             "默认只创建数据版本，指针不动（使用新数据另行选择）")
    args = parser.parse_args()
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")

    settings = load_settings()
    messages = ingest.load_chat_export(Path(args.data))
    print(f"载入 {len(messages)} 条消息")

    part = partition.partition(messages, settings)
    print(
        f"切分完成: fixed {len(part.fixed_chat_ids)} / dev {len(part.dev_chat_ids)} 个聊天，"
        f"训练消息 {len(part.train_messages)} 条（split-first，三分区）"
    )

    stats = analyze.analyze(part.train_messages)  # 只用训练侧（SOP §9）
    stats["_self_samples"] = [str(m["text"]) for m in part.train_messages if m.get("is_self")][:50]
    print("统计: " + analyze.format_stats(stats).replace("\n", " | "))

    llm = None
    if not args.no_llm and args.model:
        try:
            llm = ChatClient(settings, {"model": args.model, "timeout_seconds": 120})
        except RuntimeError as exc:
            print(f"LLM 不可用（{exc}），改用统计兜底")

    # ---- 数据版本目录 ----
    prev_data = None
    try:
        prev_data = versions.current_data_version()
    except ConfigError:
        pass
    vid = versions.create_data_version(None, {
        "seed": 42,
        "split_strategy": settings["evaluation"].get("split_strategy", "by_chat"),
        "source_messages": len(messages),
        "refreeze_reason": args.refreeze_reason or "初始构建",
        "partitions": {
            "fixed_chats": len(part.fixed_chat_ids),
            "dev_chats": len(part.dev_chat_ids),
            "train_messages": len(part.train_messages),
        },
    })
    d = versions.data_version_dir(vid)
    print(f"数据版本 {vid} 已创建: {d}")

    # 人格工作区：上一版本的人工精调 carry over；但档案更新后用 LLM 重新生成
    # （carry 版只作无 LLM 时的兜底），否则 profile.json 改了人格也不变。
    (d / "scenarios").mkdir(exist_ok=True)
    carried = False
    if prev_data is not None and (prev_data / "persona.md").exists():
        import shutil
        shutil.copyfile(prev_data / "persona.md", d / "persona.md")
        carried = True
        if not any((d / "scenarios").iterdir()):
            shutil.copytree(prev_data / "scenarios", d / "scenarios", dirs_exist_ok=True)
        print("人格/场景已从上一数据版本带入（场景保留人工精调）")
    if llm is not None:
        build_persona.build_persona(
            versions.PRIVATE / "config" / "profile.json", d / "persona.md", stats, llm=llm
        )
    elif not carried:
        build_persona.build_persona(
            versions.PRIVATE / "config" / "profile.json", d / "persona.md", stats, llm=None
        )
    elif carried:
        print("（无 LLM：人格沿用上一版本；提供 LLM 后重跑可刷新）")
    build_scenarios.build_scenarios(d / "scenarios", stats, llm=llm, force=args.force)

    ts_manifest = build_testsets.build_testsets(
        part.fixed_messages, part.dev_messages, settings,
        d / "dev_pool.jsonl", d / "fixed_test.jsonl",
    )
    pool_report = build_fewshot_pool.build_pool(part.train_messages, d / "fewshot_pool.jsonl")

    # manifest 补测试集/池统计
    manifest_path = d / "manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["testsets"] = {k: ts_manifest[k] for k in ("fixed_test", "development")}
    manifest["fewshot_pool"] = pool_report
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    versions.finalize_data_version(vid)
    print(f"manifest 已写入 {manifest_path}")

    # ---- 初始生成器 / Judge 版本：仅首次初始化或显式 --adopt 时创建并切指针 ----
    first_run = not versions.POINTERS_PATH.exists()
    if args.adopt or first_run:
        minimum = settings["evaluation"].get("pool_min_total", 5000)
        pool_n = int(manifest.get("fewshot_pool", {}).get("total", 0))
        if pool_n < minimum:
            raise SystemExit(f"准入拒绝：池 {pool_n} 条 < {minimum}；不初始化指针，先补数据")
    if not (first_run or args.adopt):
        print(f"数据版本 {vid} 已就绪（指针未动）。采用新数据: "
              f"python scripts/bootstrap.py --data <同一或新导出> --adopt")
        return

    llm_cfg = {"base_url_env": settings["llm"]["base_url_env"],
               "api_key_env": settings["llm"]["api_key_env"], "timeout_seconds": 60}
    if first_run:
        gen_cfg = {
            "llm": {**llm_cfg, "model": args.model or "PLEASE_SET_MODEL"},
            "max_shots_per_case": settings["evaluation"]["few_shots_per_case"],
            "shots_char_budget": settings["evaluation"]["few_shots_char_budget"],
            "retriever": {"enabled": True},
        }
        judge_cfg = {"mode": "pairwise_llm",
                     "llm": {**llm_cfg, "model": args.judge_model or args.model or "PLEASE_SET_MODEL"}}
    else:
        # --adopt：沿用现有版本配置，对新数据做快照
        prev = versions.load_pointers()
        gen_cfg = versions.load_generator(prev["production_gen"])["config"]
        judge_cfg = versions.judge_dir(prev["production_judge"])["config"]
        print(f"--adopt：沿用 g/j 配置，对新数据 {vid} 重建快照版本")

    gid = versions.create_generator_version(gen_cfg, vid)
    jid = versions.create_judge_version(
        judge_cfg, meta={"note": "bootstrap 初始版本" if first_run else f"adopt {vid}"},
        source_dir=None if first_run else versions.judge_dir(prev["production_judge"])["dir"])

    pointers = versions.load_pointers() if versions.POINTERS_PATH.exists() else {}
    pointers.update({"data": vid, "production_gen": gid, "iteration_gen": gid,
                     "production_judge": jid, "iteration_judge": jid})
    versions.save_pointers(pointers)
    print(f"指针: data={vid} production_gen={gid} iteration_gen={gid} production_judge={jid} iteration_judge={jid}")

    print("\nbootstrap 完成。下一步:")
    print("  1. 填写 private/config/profile.json 后重跑以补全人格（人工精调会被带入新版本）")
    print("  2. python scripts/run.py --dataset development --change <唯一改动> --override k=v --limit 5  # 冒烟")
    print("  3. python scripts/serve_dashboard.py 查看后台")


if __name__ == "__main__":
    main()
