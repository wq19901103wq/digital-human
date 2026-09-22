"""实验（SOP §3/§5/§8）：唯一执行状态源。

实验目录 experiments/<e-id>/：
- spec.json   创建时冻结：kind / dataset / change / baseline_ref / candidate_ref /
              judge_ref / data_ref / config_diff / fingerprint
- state.json  唯一状态：running → finished + verdict + metrics
- cases.jsonl 逐题明细（checkpoint）
- candidate/  实验内候选版本目录（自包含快照，随实验归档）

one-shot、续跑、审计都从实验目录取数——没有第二份状态需要保持一致。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..config import ConfigError, dataset_plan, load_settings
from ..generator.generator import ReplyGenerator
from . import acceptance, data_guard, datasets, gates, protocol, versions, record_contract
from .storage import read_json as _read_json, write_json, write_once_json

_write_json = write_json  # 兼容旧脚本（历史调用名）

EXP_ROOT = versions.PRIVATE / "experiments"  # 兼容测试补丁；内部一律走 versions.PRIVATE
KIND_GEN_AB = "gen_ab"
KIND_JUDGE_EVAL = "judge_eval"

def write_spec(exp_dir: Path, spec: dict) -> None:
    """Freeze the executor explicitly before publishing an experiment contract."""
    from . import runtime
    record_contract.require_location(exp_dir, spec)
    spec['record_contract'] = record_contract.plan(record_contract.planned_rows(spec))
    runtime.freeze(exp_dir)
    write_once_json(exp_dir / 'spec.json', spec)


def load_experiment(exp_id: str) -> Path:
    d = record_contract.directory(exp_id)
    if not (d / "spec.json").exists():
        raise ConfigError(f"实验不存在: {d}")
    return d


def spec_of(exp_dir: Path) -> dict[str, Any]:
    return _read_json(exp_dir / "spec.json")


def state_of(exp_dir: Path) -> dict[str, Any]:
    p = exp_dir / "state.json"
    return _read_json(p) if p.exists() else {"status": "running"}


def finish(exp_dir: Path, metrics: dict[str, Any], verdict: dict[str, Any]) -> None:
    """唯一收尾入口。experiment_incomplete 不算完赛：保持 running 状态，
    恢复时继续循环（失败题可重试，成功题不重抽）。"""
    incomplete = verdict["verdict"] == "experiment_incomplete"
    spec = record_contract.validate_completion(exp_dir, metrics, complete=not incomplete)
    previous = state_of(exp_dir)
    if previous.get('status') == 'finished':
        if previous.get('metrics') != metrics or previous.get('verdict') != verdict['verdict']:
            raise ConfigError('已完成实验的指标和结论不可改写')
        return
    if verdict['verdict'] in ('adopt', 'merge_to_iteration_baseline'):
        if spec.get('comparison'):
            raise ConfigError('专项比较不能保存为基线晋级结论')
        expected = protocol.decide(metrics, spec['dataset'], spec['protocol'],
                                   higher_is_better=spec['kind'] == KIND_JUDGE_EVAL)
        if expected['verdict'] != verdict['verdict']:
            raise ConfigError('晋级结论不满足冻结协议门槛')
    if not incomplete:
        acceptance.transition(spec_of(exp_dir), "finished")
    progress = state_of(exp_dir).get("progress")
    if progress:
        progress.update(status="stopped" if incomplete else "finished", phase="finished",
                        updated_at=time.time(), ended_at=time.time())
    write_json(
        exp_dir / "state.json",
        {
            "status": "running" if incomplete else "finished",
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
            "metrics": metrics,
            "finished_at": None if incomplete else time.strftime("%Y-%m-%d %H:%M:%S"),
            **({"progress": progress} if progress else {}),
        },
    )
    if not incomplete:
        from . import measurements
        measurements.record_experiment_safely(exp_dir)  # 派生层写入，失败不阻断收尾


def record_smoke_recovery(failed_dir: Path, verified_dir: Path, reason: str) -> None:
    """关联已成功的新配置检查；旧失败证据和指标保持原样。仅适用于冒烟。"""
    old, new = spec_of(failed_dir), spec_of(verified_dir)
    state, verified = state_of(failed_dir), state_of(verified_dir)
    if failed_dir.parent.resolve() != verified_dir.parent.resolve() or failed_dir == verified_dir:
        raise ConfigError("修复验证必须来自同一实例的另一任务")
    if not old.get("smoke") or not new.get("smoke") or old.get("kind") != new.get("kind"):
        raise ConfigError("只能关联同类型冒烟，不能以连接检查替代正式评测")
    if state.get("verdict") != "experiment_incomplete" or verified.get("status") != "finished":
        raise ConfigError("原任务必须失败，验证任务必须已完成")
    if verified.get("metrics", {}).get("failures") != 0:
        raise ConfigError("验证任务仍有失败，不能标记修复")
    before, _ = protocol.summarize_final_records(failed_dir / "cases.jsonl")
    after, _ = protocol.summarize_final_records(verified_dir / "cases.jsonl")
    failed = [row for row in before.values() if row.get("status") == "failed"]
    if not failed or len(after) != verified.get("metrics", {}).get("attempted"):
        raise ConfigError("验证记录不完整")
    def evidence(row):
        return {"context": [{"sender": m.get("sender"), "text": m.get("text")} for m in row.get("context", [])],
                "human_reply": row.get("human_reply")}
    for row in failed:
        resolved = after.get(str(row["case_id"]), {})
        if resolved.get("status") != "ok" or evidence(resolved) != evidence(row):
            raise ConfigError("验证任务未覆盖原失败题或题目内容已变化")
    state["resolution"] = {"verified_by": new["id"], "reason": reason,
                           "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(failed_dir / "state.json", state)


# ---------- 准入（SOP §1.2/§1.5/§3.3） ----------

def _protocol_snapshot(settings: dict[str, Any]) -> dict[str, Any]:
    """实验输入快照：协议超参随实验冻结，运行/决策不再读当前 settings（SOP §2）。"""
    ev, ad = settings["evaluation"], settings["evaluation"]["adoption"]
    return {
        "flip_extra_rounds": ev["flip_verification"]["extra_rounds_per_version"] if ev["flip_verification"]["enabled"] else 0,
        "force_reply": ev["force_reply"],
        "max_failure_rate": ev.get("max_failure_rate", 0.01),
        "dev_min_net_win_rate": ad["dev_min_net_win_rate"],
        "gate_schema": 3,
        "fixed_entry_min_net_win_rate": ad.get("fixed_entry_min_net_win_rate", ad["formal_min_net_win_rate"]),
        "formal_min_net_win_rate": ad["formal_min_net_win_rate"],
        "judge_min_gain_rate": ad.get("judge_min_gain_rate", 0.0),
    }


def _verify_composition(cases: list[dict[str, Any]], dataset: str) -> None:
    settings = load_settings()
    plan = dataset_plan(settings, dataset)
    actual = {
        "total": len(cases),
        "group": sum(1 for c in cases if c.get("chat_type") == "group"),
    }
    expected = {"total": plan["total"], "group": plan["group"]}
    if actual != expected:
        raise ConfigError(
            f"数据集构成未达准入: 实际 {actual} ≠ 要求 {expected}"
            "（数据不足：补充后新建数据版本，或调低 settings 的 total）"
        )


def _verify_pool_minimum(data_dir: Path) -> None:
    report_path = data_dir / "report.json"
    total = 0
    if report_path.exists():
        try:
            total = int(_read_json(report_path).get("total", 0))
        except ValueError:
            total = 0
    minimum = int(load_settings()["evaluation"].get("pool_min_total", 5000))
    if total < minimum:
        raise ConfigError(
            f"few-shot 池仅 {total} 条 < {minimum}（SOP §1.4，settings.pool_min_total）：补充数据后新建数据版本"
        )


def _verify_trigger(baseline_gen: ReplyGenerator, cand_gen: ReplyGenerator,
                    diff: list[str], cases: list[dict[str, Any]]) -> None:
    """分层取样（每类最多 3 题），至少 1 题输入有差异；纯模型改动跳过。"""
    non_llm = [d for d in diff if not d.startswith("llm.")]
    if not non_llm:
        return
    sample: list[dict[str, Any]] = []
    for chat_type in ("group", "private"):
        sample.extend([c for c in cases if c.get("chat_type") == chat_type][:3])
    sample = sample[:6] or cases[:3]
    differing = sum(
        1 for case in sample
        if json.dumps(baseline_gen.build_prompt(case), sort_keys=True, ensure_ascii=False)
        != json.dumps(cand_gen.build_prompt(case), sort_keys=True, ensure_ascii=False)
    )
    if differing == 0:
        raise ConfigError(
            f"触发预检失败：{len(sample)} 条分层样本输入全部相同，feature 未触发。先修实验。"
        )


# ---------- one-shot（SOP §8.3）：从实验目录查询，无独立 ledger ----------

def _matching_experiments(kind: str, **filters: Any) -> list[tuple[Path, dict, dict]]:
    out = []
    exp_root = versions.PRIVATE / "experiments"
    if not exp_root.exists():
        return out
    for d in sorted(exp_root.iterdir()):
        if not (d / "spec.json").exists():
            continue
        spec = spec_of(d)
        if spec.get("kind") != kind:
            continue
        if any(spec.get(k) != v for k, v in filters.items()):
            continue
        out.append((d, spec, state_of(d)))
    return out


def check_one_shot(candidate_fp: str, exp_id: str, kind: str = KIND_GEN_AB) -> None:
    """同一候选指纹：完赛(adopt/reject)永久拒绝；进行中/可恢复必须恢复原实验。"""
    for d, spec, state in _matching_experiments(kind, dataset="fixed_test"):
        if spec.get("candidate_fingerprint") != candidate_fp:
            # 老规格的指纹包含 Judge 来源 meta；按候选内容回算，来源变化不能绕过 one-shot。
            ref = spec.get("candidate_ref")
            if not ref:
                continue
            info = versions.load_generator(ref) if kind == KIND_GEN_AB else versions.judge_dir(ref)
            if _candidate_fingerprint(info["config"], info["dir"]) != candidate_fp:
                continue
        other_id = spec["id"]
        status = state.get("status")
        verdict = state.get("verdict", "")
        if status == "finished" and verdict in ("adopt", "reject"):
            raise ConfigError(
                f"该候选已于实验 {other_id} 完赛（{verdict}）：SOP §8 one-shot。"
                "有异议→开发侧出新候选，或申请数据版本更新。"
            )
        if other_id != exp_id:
            raise ConfigError(
                f"该候选已有实验 {other_id}；必须恢复原实验续跑（保留成功样本），"
                "确需重开请先废弃该实验目录"
            )


# ---------- 创建 ----------

def _asset_diff(baseline_dir: Path, data_dir: Path) -> list[str]:
    """资产内容差异（SOP §2）：基线版本快照 vs 数据版本工作区。
    提示词/场景迭代不改配置，diff 断言必须能看到内容变化。"""
    from ..config import sha256_file
    diffs = []
    for name in ("persona.md",):
        b, w = baseline_dir / name, data_dir / name
        if b.exists() and w.exists() and sha256_file(b) != sha256_file(w):
            diffs.append(f"asset:{name}: 内容已更新（基线快照 ≠ 当前工作区）")
    bdir, wdir = baseline_dir / "scenarios", data_dir / "scenarios"
    if bdir.exists() and wdir.exists():
        bnames = {f.name for f in bdir.glob("*.md")}
        wnames = {f.name for f in wdir.glob("*.md")}
        for n in sorted(wnames - bnames):
            diffs.append(f"asset:scenarios/{n}: 新增")
        for n in sorted(bnames - wnames):
            diffs.append(f"asset:scenarios/{n}: 删除")
        for n in sorted(bnames & wnames):
            if sha256_file(bdir / n) != sha256_file(wdir / n):
                diffs.append(f"asset:scenarios/{n}: 内容已更新")
    return diffs


def _fingerprint(core: dict[str, Any], diff: list[str]) -> str:
    payload = json.dumps({**core, "config_diff": diff}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _candidate_fingerprint(candidate_config: dict[str, Any], data_dir: Path | None = None) -> str:
    """候选内容指纹 = 行为配置 + 人格/场景内容。
    data_dir 传版本目录时经 version_payload 统一读取（快照即唯一依据）；传工作区用于
    开发轮新快照（创建前工作区即快照内容）。改场景=新候选，不受旧 one-shot 影响（SOP §2.5）。"""
    if data_dir is not None:
        payload = versions.version_payload(data_dir)
        payload.pop("meta", None)  # 训练/提交来源不属于模型行为，换来源仍是同一候选。
        payload["config"] = candidate_config
    else:
        payload = {"config": candidate_config}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    import copy

    out = copy.deepcopy(base)
    for key, value in overrides.items():
        # 双模型归一：对单模型基线调 llm.private.x / llm.group.x 时，
        # 先把基线展开为双模型（两类共用原配置），再递归合并——避免丢 model/超时
        if key == "llm" and isinstance(value, dict) and "model" not in value \
                and "private" not in value and "group" not in value \
                and isinstance(out.get("llm"), dict) and ("private" in out["llm"] or "group" in out["llm"]):
            # 双模型基线上的公共参数（temperature 等）：两侧都应用，并回写顶层便于 diff
            for sub in ("private", "group"):
                if isinstance(out["llm"].get(sub), dict):
                    out["llm"][sub].update(value)
            out["llm"].update(value)
            continue
        if key == "llm" and isinstance(value, dict) and "model" in value \
                and isinstance(out.get("llm"), dict) and ("private" in out["llm"] or "group" in out["llm"]):
            # 双模型基线上 llm.model=新模型：两侧模型都替换；同传的其余键（温度等）也应用到两侧
            for sub in ("private", "group"):
                if isinstance(out["llm"].get(sub), dict):
                    out["llm"][sub]["model"] = value["model"]
                    for k, v in value.items():
                        if k != "model":
                            out["llm"][sub][k] = v
            out["llm"].update({k: v for k, v in value.items() if k != "model"})
            continue
        if key == "llm" and isinstance(value, dict) and ("private" in value or "group" in value) \
                and isinstance(out.get("llm"), dict) and "model" in out["llm"]:
            out["llm"] = {"private": dict(out["llm"]), "group": dict(out["llm"])}
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@versions.transaction
def create_gen_experiment(
    dataset: str,
    single_change: str,
    overrides: dict[str, Any] | None = None,
    exp_id: str | None = None,
    limit: int | None = None,
    data_ref: str | None = None,
    purpose: str | None = None,
    against_production: bool = False,
) -> Path:
    """创建生成器 A/B 实验：预检、准入、候选版本、one-shot 一次完成。"""
    if exp_id is not None:
        record_contract.directory(exp_id)
    if not single_change.strip():
        raise ConfigError("single_change 必须非空：每轮只改一个方向")
    if dataset not in ("development", "fixed_test"):
        raise ConfigError(f"dataset 必须是 development/fixed_test: {dataset}")
    if limit is not None and dataset == "fixed_test":
        raise ConfigError("固定集禁止冒烟：接线验证只在 development 上做（SOP §8.2）")
    overrides = overrides or {}
    if against_production and (dataset != 'development' or overrides or limit is not None or
                               purpose not in (None, 'development')):
        raise ConfigError('生产对比只使用完整开发集和已锁定开发版，禁止 override/冒烟/更换用途')
    purpose = purpose or dataset
    if purpose not in ({'development', 'gen_optimization'} if dataset == 'development' else {'fixed_test'}):
        raise ConfigError('数据用途与实验阶段不匹配')

    pointers = versions.load_pointers()
    if data_ref is not None:
        pointers = {**pointers, 'data': data_ref}
    data_dir = versions.data_version_dir(pointers["data"])
    judge = versions.current_judge("production")
    production = versions.load_generator(pointers["production_gen"])
    iteration = versions.load_generator(pointers["iteration_gen"])
    asset_dir = data_dir if (data_dir / 'persona.md').exists() else iteration['dir']
    from . import learning_guard
    learning_guard.require_materials(pointers['data'], [production['dir'], iteration['dir'],
                                                       asset_dir, judge['dir']])

    if dataset == "development" and not against_production:
        baseline = iteration
        cand_cfg = _merge(iteration["config"], overrides)
        role = "iteration vs iteration+change"
        candidate_source = iteration["id"]
    else:
        if overrides:
            raise ConfigError("固定测试的候选 = 开发基线锁定快照，禁止 override（SOP §6）")
        baseline = production
        cand_cfg = iteration["config"]
        role = "production vs iteration(locked)"
        candidate_source = iteration["id"]

    diff = protocol.config_diff(baseline["config"], cand_cfg)
    if not diff:
        # 提示词/场景迭代：配置不变但工作区内容相对基线快照有变化 → 合法唯一改动
        diff = _asset_diff(baseline["dir"], iteration['dir'] if against_production or dataset == 'fixed_test' else asset_dir)
    if not diff and limit is None:
        raise ConfigError("候选与基线行为配置、人格/场景内容完全相同：没有可迭代的改动")
    if not diff:
        # 冒烟允许无改动：纯接线验证（SOP §2.2），diff 如实标注
        diff = ["(冒烟接线验证：无配置/内容改动)"]

    frozen_protocol = _protocol_snapshot(load_settings())
    entry = None
    if against_production:
        reused = gates.reusable_comparison(KIND_GEN_AB, pointers['data'], baseline['id'],
            iteration['id'], frozen_protocol, judge_ref=judge['id'])
        if reused:
            return reused
    if dataset == 'fixed_test':
        entry = gates.require_fixed_entry(KIND_GEN_AB, pointers['data'], baseline['id'],
            iteration['id'], frozen_protocol, judge_ref=judge['id'])
    exp_id = exp_id or f"{dataset}-{time.strftime('%Y%m%d-%H%M%S')}"
    receipt = None
    if dataset == 'fixed_test' and acceptance.exists(pointers['data']):
        receipt = acceptance.claim({'id': exp_id, 'kind': KIND_GEN_AB, 'data_ref': pointers['data'],
            'baseline_ref': baseline['id'], 'candidate_ref': candidate_source, 'judge_ref': judge['id'],
            'protocol': frozen_protocol})
        cases = acceptance.rows(receipt)
    else:
        cases_path = datasets.case_path(data_dir, purpose)
        cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        cases = cases[:limit]
    else:
        _verify_composition(cases, dataset)
        _verify_pool_minimum(data_dir)
    data_snapshot = data_guard.generation_snapshot(
        cases[:5] if limit == 0 else cases, data_dir, [baseline["config"], cand_cfg])

    exp_id = exp_id or f"{dataset}-{time.strftime('%Y%m%d-%H%M%S')}"
    exp_dir = record_contract.directory(exp_id)
    if exp_dir.exists():
        raise ConfigError(f"实验已存在: {exp_dir}；恢复运行直接用 run 命令 --exp {exp_id}")

    if dataset == "development" and not against_production:
        # 候选 = 普通生成器版本（创建即得版本号；晋升只切指针，无拷贝、无第二身份）
        cand_ref = versions.create_generator_version(cand_cfg, pointers["data"], source_dir=asset_dir)
        cand_fp = _candidate_fingerprint(cand_cfg, versions.generator_dir(cand_ref))
    else:
        cand_ref = candidate_source  # 固定轮候选 = 开发基线版本本身
        # 指纹必须取候选版本的实际快照：改工作区但未创建新候选 ≠ 新候选（SOP §2.5）
        cand_ver = versions.load_generator(cand_ref)
        cand_fp = _candidate_fingerprint(cand_ver["config"], cand_ver["dir"])

    if dataset == "fixed_test":
        check_one_shot(cand_fp, exp_id)  # 模型调用前；失败不留孤儿目录
        datasets.acceptance_available(data_dir, exp_id)

    source_audit = datasets.static_sources(data_dir, [baseline['dir'], versions.generator_dir(cand_ref),
                                                     judge['dir']], dataset)
    if purpose == 'gen_optimization':
        source_audit.update(promotion_eligible=False, reason='Gen 优化题用于调整策略，不能代替开发验证')
    if dataset == 'fixed_test' and source_audit.get('promotion_eligible') is False:
        raise ConfigError('静态学习来源尚未核验，禁止消耗独立验收题；先做开发诊断')

    exp_dir.mkdir(parents=True)

    spec = {
        "id": exp_id,
        "kind": KIND_GEN_AB,
        "dataset": dataset,
        "purpose": purpose,
        "single_change": single_change.strip(),
        "role": role,
        "baseline_ref": baseline["id"],
        "candidate_ref": cand_ref,
        "candidate_source_ref": candidate_source,
        "candidate_fingerprint": cand_fp,
        "judge_ref": judge["id"],
        "data_ref": pointers["data"],
        "data_snapshot": data_snapshot,
        "source_audit": source_audit,
        "config_diff": diff,
        "smoke": limit is not None,
        "smoke_limit": limit,
        "protocol": frozen_protocol,
        "against_production": against_production,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if receipt:
        spec['acceptance'] = receipt
    spec['evaluation_materials'] = gates.materials(spec)
    learning_guard.bind(spec)
    if entry:
        spec['fixed_entry'] = entry
    spec["fingerprint"] = _fingerprint(
        {k: spec[k] for k in ("kind", "dataset", "single_change", "baseline_ref",
                              "judge_ref", "data_ref", "candidate_fingerprint")},
        diff,
    )
    write_spec(exp_dir, spec)
    return exp_dir


@versions.transaction
def create_judge_eval_experiment(
    dataset: str,
    pack_id: str,
    judge_overrides: dict[str, Any] | None,
    change: str,
    exp_id: str | None = None,
    candidate_ref: str | None = None,
    against_production: bool = False,
    saved_draw_manifest: str | None = None,
    supplement_missing: bool = False,
) -> Path:
    """Judge 实验，与生成器完全同构（SOP §4）：
    - development：基线=iteration_judge，候选=新 Judge 版本（overrides 非空），校准包
    - fixed_test：基线=production_judge，候选=iteration_judge 锁定快照（overrides 必空），
      验证包 one-shot
    """
    if exp_id is not None:
        record_contract.directory(exp_id)
    if dataset not in ("development", "fixed_test"):
        raise ConfigError(f"dataset 必须是 development/fixed_test: {dataset}")
    if supplement_missing and (dataset != 'development' or not saved_draw_manifest):
        raise ConfigError('Missing-round supplementation requires saved development draws')
    judge_overrides = judge_overrides or {}
    pointers = versions.load_pointers()
    if against_production and (dataset != 'development' or judge_overrides or candidate_ref):
        raise ConfigError('生产对比只允许开发集与已锁定开发 Judge，禁止 override/指定其他候选')
    frozen_protocol = _protocol_snapshot(load_settings())
    entry = None
    if dataset == 'fixed_test':
        if candidate_ref:
            raise ConfigError('固定验收必须使用已锁定的开发 Judge')
        if judge_overrides:
            raise ConfigError('固定轮候选 = 开发 Judge 基线锁定快照，禁止 override（SOP §4）')
        entry = gates.require_fixed_entry(KIND_JUDGE_EVAL, pointers['data'], pointers['production_judge'],
                                         pointers['iteration_judge'], frozen_protocol)
    pack_dir = versions.PRIVATE / "judge_eval" / pack_id
    if not (pack_dir / "pack.json").exists():
        raise ConfigError(f"评估包不存在: {pack_dir}")
    pack_bytes = (pack_dir / "pack.json").read_bytes()
    pack = json.loads(pack_bytes)
    if pack.get('data_ref'):
        pointers = {**pointers, 'data': pack['data_ref']}
    data_dir = versions.data_version_dir(pointers['data'])
    datasets.assert_pack(data_dir, pack, 'judge_development' if dataset == 'development' else 'fixed_test')
    from . import learning_guard
    learning_guard.verify_pack(pack, dataset)
    learning_guard.require_materials(pointers['data'], [versions.judge_dir(ref)['dir'] for ref in
        {pointers['production_judge'], pointers['iteration_judge'], candidate_ref or pointers['iteration_judge']}])
    if pack.get("c0_gen_version") != pointers["production_gen"]:
        raise ConfigError(
            f"评估包由 {pack.get('c0_gen_version')} 生成，当前生产 {pointers['production_gen']}；"
            "请重新 build（SOP §4：校准的是对当前生成器的识别能力）")
    if dataset == "fixed_test" and "validation" not in pack_id:
        raise ConfigError("Judge 固定轮必须使用 validation 包（fixed 抽样）；校准包只属于开发轮（SOP §4）")

    if dataset == "development":
        baseline = versions.current_judge("production" if against_production else "iteration")
        if against_production:
            candidate_ref = pointers['iteration_judge']
        cand_cfg = _merge(baseline["config"], judge_overrides)
        if candidate_ref:
            if judge_overrides:
                raise ConfigError('已冻结的 Judge 候选不能再叠加 override')
            cand_cfg = versions.judge_dir(candidate_ref)['config']
        # 提示词迭代：全局模板若已更新，候选的 prompt_sha256 取当前模板（版本创建时快照新模板）
        from ..config import ROOT as _ROOT, sha256_file as _sha
        from ..judge import normalize_mode
        imported = normalize_mode(cand_cfg.get("mode", "pairwise_llm")) != "pairwise_llm"
        if not imported and not candidate_ref:
            cand_cfg["prompt_sha256"] = _sha(_ROOT / "prompts" / "judge_pairwise.template.md")
        diff = protocol.config_diff(baseline["config"], cand_cfg)
        if not diff:
            raise ConfigError("候选 Judge 与基线配置相同：没有可校准的改动")
        cand_ref = candidate_ref or versions.create_judge_version(
            cand_cfg, meta={"origin": "calibration", "changed": diff},
            source_dir=baseline["dir"] if imported else None,
        )
        role = "production_judge vs iteration_judge(locked)" if against_production else "iteration_judge vs iteration_judge+change"
        if against_production:
            reused = gates.reusable_comparison(KIND_JUDGE_EVAL, pointers['data'], baseline['id'],
                cand_ref, frozen_protocol, development_pack=pack_id)
            if reused:
                return reused
        cand_fp = None
    else:
        if candidate_ref:
            raise ConfigError('固定验收必须使用已锁定的开发 Judge')
        if judge_overrides:
            raise ConfigError("固定轮候选 = 开发 Judge 基线锁定快照，禁止 override（SOP §4）")
        baseline = versions.current_judge("production")
        cand_ref = pointers["iteration_judge"]
        diff = protocol.config_diff(baseline["config"], versions.current_judge("iteration")["config"])
        if not diff:
            raise ConfigError("开发 Judge 与生产 Judge 相同：没有可验证的改动")
        role = "production_judge vs iteration_judge(locked)"
        cand_fp = _candidate_fingerprint(versions.current_judge("iteration")["config"],
                                       versions.current_judge("iteration")["dir"])

    replay_binding = None
    if saved_draw_manifest is not None:
        if dataset != 'development':
            raise ConfigError('Saved development draws cannot enter fixed acceptance')
        from .draw_replay import manifest
        replay_binding, _ = manifest(saved_draw_manifest, pack, baseline, versions.judge_dir(cand_ref))
    receipt = pack.get('acceptance') if dataset == 'fixed_test' else None
    if receipt:
        if exp_id is not None and exp_id != receipt['experiment_id']:
            raise ConfigError('validation pack belongs to another acceptance experiment')
        exp_id = receipt['experiment_id']
    exp_id = exp_id or f"judge-{dataset}-{time.strftime('%Y%m%d-%H%M%S')}"
    if dataset == "fixed_test":
        check_one_shot(cand_fp, exp_id, kind=KIND_JUDGE_EVAL)
        datasets.acceptance_available(data_dir, exp_id)
    source_audit = datasets.static_sources(data_dir, [baseline['dir'], versions.judge_dir(cand_ref)['dir'],
        versions.generator_dir(pack['c0_gen_version'])], dataset)
    if dataset == 'fixed_test' and source_audit.get('promotion_eligible') is False:
        raise ConfigError('静态学习来源尚未核验，禁止消耗独立验收题')
    exp_dir = record_contract.directory(exp_id)
    if exp_dir.exists():
        raise ConfigError(f"实验已存在: {exp_dir}；恢复运行用 run.py --exp {exp_id}")
    exp_dir.mkdir(parents=True)

    spec = {
        "id": exp_id,
        "kind": KIND_JUDGE_EVAL,
        "dataset": dataset,
        "pack_ref": pack_id,
        "pack_sha256": hashlib.sha256(pack_bytes).hexdigest(),
        "purpose_snapshot": datasets.snapshot(data_dir),
        "source_audit": source_audit,
        "single_change": change,
        "role": role,
        "baseline_ref": baseline["id"],
        "candidate_ref": cand_ref,
        "candidate_fingerprint": cand_fp,
        "judge_ref": baseline["id"],
        "data_ref": pointers["data"],
        "config_diff": diff,
        "protocol": frozen_protocol,
        "against_production": against_production,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if replay_binding:
        spec['saved_draw_replay'] = replay_binding
        if supplement_missing:
            from .draw_replay import SavedDrawReplay
            SavedDrawReplay(replay_binding, pack, baseline, versions.judge_dir(cand_ref),
                            supplement_missing=True)
            spec['supplement_missing_rounds'] = True
    if receipt:
        spec['acceptance'] = receipt
        spec['generator_ref'] = pack['c0_gen_version']
        acceptance.rows(receipt, spec)
    spec['evaluation_materials'] = gates.materials(spec)
    learning_guard.bind(spec)
    if entry:
        if entry['data_ref'] != spec['data_ref']:
            raise ConfigError('固定包与开发准入证据数据版本不一致')
        spec['fixed_entry'] = entry
    write_spec(exp_dir, spec)
    return exp_dir
