#!/usr/bin/env python3
"""从外部来源部署（wechat-mac 项目）迁移数据与实验配置（一次性导入，机制层无改动）。

产物 = 新数据版本 + 初始生成器/Judge 版本 + 指针（同 bootstrap，但数据来自 wechat 资产）：
- 测试集：从 WeFlow 导出按 split-first 现切（不走 bootstrap 的统计再生成）
- few-shot 池：直接采用 wechat 已策划的 80,530 条（sha 校验 + approved 报告）
- 人格/场景：wechat 的 persona.md + 已有场景文件
- 生成器配置：C0 已采用配置（私聊 deepseek-v4-flash / 群聊 doubao-seed-2.1-turbo）

用法：
  python scripts/import_wechat.py --exports <wechat>/data/exports \
      --pool <wechat>/data/reports/private/situation_few_shot_library_v1/persona_examples.jsonl \
      --persona <wechat>/data/persona.md [--scenarios <dir>] [--adopt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEPLOYMENT = min(ROOT.parent.glob('wechat-mac-*/'), key=lambda p: len(p.name), default=ROOT.parent / 'wechat-mac-missing')
sys.path.insert(0, str(ROOT))

from src.bootstrap import build_testsets, ingest, partition  # noqa: E402
from src.config import ConfigError, load_settings  # noqa: E402
from src.iteration import versions  # noqa: E402
from src.judge.corrected import prepare_import  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="来源部署资产迁移")
    parser.add_argument("--instance", default=None, help="数字人实例名（默认 env DH_INSTANCE 或 default）")
    parser.add_argument("--source-root", default=str(SOURCE_DEPLOYMENT), help="来源部署根目录")
    parser.add_argument("--judge-only", action="store_true", help="已有实例对齐来源裁判，并补齐现有题目的来源元数据")
    parser.add_argument("--exports", help="WeFlow 导出目录（含 b/ 与 main/）；judge-only 默认取来源项目 data/exports")
    parser.add_argument("--pool", help="已策划 few-shot 池 jsonl")
    parser.add_argument("--persona", help="wechat persona.md")
    parser.add_argument("--scenarios", default=None,
                        help="场景目录：支持平铺 *.md，或 skills/<名>/SKILL.md 结构（自动展平为 <名>.md）；"
                             "默认取 wechat 的 prompts/scenarios")
    parser.add_argument("--c0-config", default=None,
                        help="wechat config/current_c0.json：读取私聊/群聊模型与超时（不硬编码）")
    parser.add_argument("--adopt", action="store_true", help="切换指针（同 bootstrap --adopt）")
    args = parser.parse_args()
    if getattr(args, "instance", None):
        versions.switch_instance(args.instance)
        print(f"实例: {args.instance}")

    source_root = Path(args.source_root).resolve()
    judge_cfg, judge_meta, judge_assets = prepare_import(source_root)
    if args.judge_only:
        from src.judge.migration import enrich_cases, snapshot_enriched_data
        from src.judge.corrected import CorrectedJudge
        from src.dashboard import report
        pointers = versions.load_pointers()
        messages = ingest.load_weflow_dir(Path(args.exports) if args.exports else source_root / "data/exports")
        enriched = enrich_cases(versions.data_version_dir(pointers["data"]), messages)
        print(f"已核对来源裁判 {judge_meta['source_baseline_id']} 及 {sum(map(len, enriched.values()))} 题的来源元数据")
        if not args.adopt:
            print("只核对；加 --adopt 后创建快照并切换当前实例")
            return
        with tempfile.TemporaryDirectory() as directory:
            # 指针切换前验证完整资产、运行时版本及 524 维模型。
            jid = versions.create_judge_version(judge_cfg, judge_meta, assets=judge_assets, root=Path(directory))
            CorrectedJudge(judge_cfg, Path(directory) / jid)
        current = versions.current_judge("production")
        same_data = all(rows == [json.loads(line) for line in (versions.data_version_dir(pointers["data"]) / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
                        for filename, rows in enriched.items())
        if same_data and current["config"] == judge_cfg and pointers["iteration_judge"] == current["id"]:
            print(f"已对齐来源裁判：{current['id']}，无需重复创建版本")
            return
        vid = snapshot_enriched_data(pointers["data"], enriched)
        jid = versions.create_judge_version(judge_cfg, judge_meta, assets=judge_assets)
        backup = versions.PRIVATE / f"pointers.before-{jid}.json"
        shutil.copyfile(versions.POINTERS_PATH, backup)
        pointers.update(data=vid, production_judge=jid, iteration_judge=jid)
        versions.save_pointers(pointers)
        report.refresh_dashboard(versions.PRIVATE / "experiments", ROOT / "dashboard" / versions.PRIVATE.name)
        report.write_instance_index(ROOT / "instances", ROOT / "dashboard")
        print(f"已对齐: data={vid}, production_judge=iteration_judge={jid}；原生成器版本不变；指针备份 {backup}")
        return
    if not all((args.exports, args.pool, args.persona)):
        parser.error("完整迁移必须提供 --exports、--pool 和 --persona")

    settings = load_settings()
    messages = ingest.load_weflow_dir(Path(args.exports))
    print(f"载入 {len(messages)} 条文本消息")
    part = partition.partition(messages, settings)
    print(f"切分: fixed {len(part.fixed_chat_ids)} / dev {len(part.dev_chat_ids)} 聊天, "
          f"train {len(part.train_messages)} 条消息")

    vid = versions.create_data_version(None, {
        "seed": 42, "split_strategy": "by_chat",
        "source": "wechat-mac migration",
        "source_messages": len(messages),
        "partitions": {"fixed_chats": len(part.fixed_chat_ids),
                       "dev_chats": len(part.dev_chat_ids),
                       "train_messages": len(part.train_messages)},
    })
    d = versions.data_version_dir(vid)
    (d / "scenarios").mkdir(exist_ok=True)
    shutil.copyfile(args.persona, d / "persona.md")
    scen_src = Path(args.scenarios) if args.scenarios else None
    if scen_src is None:
        candidate = source_root / "prompts/scenarios"
        if candidate.exists():
            scen_src = candidate
    if scen_src is not None and scen_src.exists():
        flat = list(scen_src.glob("*.md"))
        if flat:
            shutil.copytree(scen_src, d / "scenarios", dirs_exist_ok=True)
        else:  # skills/<名>/SKILL.md 结构 → 展平并映射到生成器读取的标准名
            SKILL_NAME_MAP = {"group_banter": "group_chat", "casual_chat": "friend",
                              "close_friends": "close_friend", "acquaintances": "acquaintance",
                              "praise": "receiving_praise", "vent": "venting",
                              "share": "receiving_share", "question": "answering_question"}
            for skill in scen_src.iterdir():
                md = skill / "SKILL.md"
                if skill.is_dir() and md.exists():
                    target = SKILL_NAME_MAP.get(skill.name, skill.name)
                    shutil.copyfile(md, d / "scenarios" / f"{target}.md")
        print(f"场景已导入: {scen_src} → {d / 'scenarios'}")

    testsets = build_testsets.build_testsets(
        part.fixed_messages, part.dev_messages, settings,
        d / "dev_pool.jsonl", d / "fixed_test.jsonl",
    )
    # 池：采用 wechat 策划池，按**来源会话**排除——回复时间戳反查导出消息所属 chat_id，
    # 属于本次 fixed/dev 聊天的池行整行剔除（多回复样本也覆盖；不依赖文本匹配）。
    ts2chats: dict[int, set] = {}
    for msg in messages:
        ts2chats.setdefault(int(msg["timestamp"]), set()).add(msg["chat_id"])
    test_chats = part.fixed_chat_ids | part.dev_chat_ids
    pool_rows, kept, leaked = [], 0, 0
    with open(args.pool, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ts = row.get("timestamp")
            chats = ts2chats.get(int(ts), set()) if ts else set()
            if not chats:
                leaked += 1  # 无法定位来源 → 拒绝入库（F03：不静默放行）
                continue
            if chats & test_chats:
                leaked += 1
                continue
            pool_rows.append(line if line.endswith("\n") else line + "\n")
            kept += 1
    print(f"池过滤：保留 {kept} 条，剔除来源属于测试分区的 {leaked} 条（按会话归属）")
    pool_dst = d / "fewshot_pool.jsonl"
    pool_dst.write_text("".join(pool_rows), encoding="utf-8")
    pool_sha = hashlib.sha256(pool_dst.read_bytes()).hexdigest()
    (d / "report.json").write_text(json.dumps({
        "review_status": "approved",
        "examples_sha256": pool_sha,
        "total": kept,
        "source": "wechat-mac situation_few_shot_library_v1（已策划、来源核验）",
        "filtered_against": "本次 fixed/dev 划分（文本匹配）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    manifest["testsets"] = {k: testsets[k] for k in ("fixed_test", "development")}
    manifest["fewshot_pool"] = {"total": kept, "sha256": pool_sha}
    (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    versions.finalize_data_version(vid)
    print(f"数据版本 {vid}: 池 {kept} 条（按本次划分过滤后）, fixed {testsets['fixed_test']['total']}, dev {testsets['development']['total']}")

    # 采用前准入：构成与配额不符、池不足，一律拒绝切指针（直接从落盘文件计数，与 manifest 结构解耦）
    def _actual(path: Path) -> dict:
        cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return {"total": len(cases), "group": sum(1 for c in cases if c.get("chat_type") == "group")}

    for ds in ("fixed_test", "development"):
        plan = {"total": settings["evaluation"][ds]["total"],
                "group": round(settings["evaluation"][ds]["total"] * settings["evaluation"][ds]["group_ratio"])}
        actual = _actual(d / ("fixed_test.jsonl" if ds == "fixed_test" else "dev_pool.jsonl"))
        if actual != plan:
            raise SystemExit(f"准入拒绝：{ds} 构成 {actual} ≠ 要求 {plan}；不 --adopt，先修数据")
    if kept < settings["evaluation"].get("pool_min_total", 5000):
        raise SystemExit(f"准入拒绝：过滤后池 {kept} 条 < pool_min_total；不 --adopt")
    if not args.adopt:
        print("未 --adopt：指针未动。确认后重跑加 --adopt 建初始版本并切指针。")
        return

    if versions.POINTERS_PATH.exists():
        versions._write_json(versions.PRIVATE / 'data_preparations' / f'{vid}.json', {
            'data_ref': vid, 'previous_baseline': versions.load_pointers(),
            'status': 'prepared', 'reason': '数据已准备；模型与生产组合须经独立评测采用'})
        print(f'{vid} 已准备；现有模型和基线未变化。数据变更不再隐式采用新生成器或 Judge。')
        return

    # 生成器配置：首次初始化才读取旧模型配置并创建模型版本。
    c0 = {}
    c0_path = Path(args.c0_config) if args.c0_config else source_root / "config/current_c0.json"
    if c0_path.exists():
        c0 = json.loads(c0_path.read_text(encoding="utf-8"))
        print(f"C0 配置来源: {c0_path}")
    elif args.c0_config:
        raise SystemExit(f"显式传入的 --c0-config 不存在: {c0_path}（拒绝静默用默认值改变超时/模型）")
    else:
        raise SystemExit("未找到来源项目 current_c0.json，请显式传 --c0-config（不允许静默改变超时/模型）")
    def _mcfg(section: str, default_model: str) -> dict:
        s = c0.get(section) or {}
        return {"model": s.get("model", default_model),
                "timeout_seconds": s.get("timeout_seconds", 60)}
    gen_cfg = {
        "llm": {"private": _mcfg("private", "deepseek-v4-flash"),
                "group": _mcfg("group", "doubao-seed-2.1-turbo")},
        # 与 bootstrap 同源：条数/预算读 settings，不在脚本里硬编码
        "max_shots_per_case": settings["evaluation"]["few_shots_per_case"],
        "shots_char_budget": settings["evaluation"]["few_shots_char_budget"],
        "retriever": {"enabled": True},
    }
    if c0:
        print("连接提示：生成器沿用来源项目的 Anthropic 通道。请在项目 .env 中设置同一通道的"
              " DH_LLM_BASE_URL / DH_LLM_API_KEY；Coding Plan 的模型名不能直接套用普通 /api/v3 端点。"
              "\nJudge 使用已冻结的 Codex CLI 配置。先跑 --limit 5 验证整条调用链。")
    gid = versions.create_generator_version(gen_cfg, vid)
    jid = versions.create_judge_version(
        judge_cfg, meta=judge_meta, assets=judge_assets,
    )
    pointers = versions.load_pointers() if versions.POINTERS_PATH.exists() else {}
    pointers.update({"data": vid, "production_gen": gid, "iteration_gen": gid,
                     "production_judge": jid, "iteration_judge": jid})
    versions.save_pointers(pointers)
    print(f"指针: data={vid} production_gen={gid} iteration_gen={gid} production_judge={jid} iteration_judge={jid}")


if __name__ == "__main__":
    main()
