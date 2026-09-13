"""生成器 A/B 跑题循环（SOP §3/§4）：从实验取规格，跑完交回 experiment.finish。

本模块只做一件事：把 spec 里的 Baseline/Candidate 跑完 1000 题并统计。
预检/准入/one-shot 在 experiment.create；晋升在 promote；状态在 experiment.finish。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import ConfigError, load_settings
from ..generator.generator import ReplyGenerator
from ..judge.judge import Judge, build_judge
from . import experiment, protocol, report, versions
from .progress import track_run

_logger = logging.getLogger("digital_human.runner")


@track_run
def run_gen_experiment(exp_dir: Path, *, _progress=None) -> Path:
    spec = experiment.spec_of(exp_dir)
    if spec["kind"] != experiment.KIND_GEN_AB:
        raise ConfigError(f"不是生成器实验: {spec['kind']}")
    state = experiment.state_of(exp_dir)
    if state.get("status") == "finished":
        _logger.info("实验已完赛(%s)，直接出报告", state.get("verdict"))
        report.write_run(exp_dir)
        return exp_dir

    settings = load_settings()
    data_dir = versions.data_version_dir(spec["data_ref"])
    baseline = versions.load_generator(spec["baseline_ref"])
    candidate = versions.load_generator(spec["candidate_ref"])
    judge_info = versions.judge_dir(spec["judge_ref"])

    cases_path = data_dir / ("dev_pool.jsonl" if spec["dataset"] == "development" else "fixed_test.jsonl")
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    proto = spec.get("protocol") or experiment._protocol_snapshot(settings)  # 快照优先，不回读当前 settings
    if spec.get("smoke"):
        cases = cases[: int(spec.get("smoke_limit") or 5)]  # --limit N 按 N 执行（SOP §2.2）
    _progress.stage("preparing", total=len(cases))

    from ..llm import build_clients
    baseline_gen = ReplyGenerator(settings, baseline["config"], build_clients(settings, baseline["config"]["llm"]),
                                  prompt_root=baseline["dir"], pool_path=data_dir / "fewshot_pool.jsonl")
    cand_gen = ReplyGenerator(settings, candidate["config"], build_clients(settings, candidate["config"]["llm"]),
                              prompt_root=candidate["dir"], pool_path=data_dir / "fewshot_pool.jsonl")
    judge = build_judge(settings, judge_info)

    # 触发预检：续跑时跳过（候选已验证过且 cases 已有记录说明能跑）
    cases_jsonl = exp_dir / "cases.jsonl"
    if not spec.get("smoke") and not cases_jsonl.exists():
        _progress.stage("preflight")
        experiment._verify_trigger(baseline_gen, cand_gen, spec["config_diff"], cases)

    extra_rounds = proto["flip_extra_rounds"]

    # 恢复：只跳过成功题；失败题最终状态为 failed，必须可重试（SOP §2.4）
    final, _ = _final_records(cases_jsonl)
    done = {cid: r for cid, r in final.items() if r.get("status") == "ok"}
    with cases_jsonl.open("a", encoding="utf-8") as fout:
        for case_index, case in enumerate(cases, 1):
            cid = str(case["case_id"])
            if cid in done:
                continue
            _progress.stage("preparing", case_index=case_index, branch="", round=0)
            _progress.start_case(case)
            record: dict[str, Any] = {
                "case_id": cid,
                "chat_type": case.get("chat_type"),
                "context": case.get("context"),
                "human_reply": case.get("human_reply"),
                "status": "ok",
            }
            try:
                base = _progress.generate(baseline_gen, case, "baseline", proto["force_reply"])
                cand = _progress.generate(cand_gen, case, "candidate", proto["force_reply"])
                record.update(
                    baseline_replies=base["replies"], candidate_replies=cand["replies"],
                    baseline_latency_ms=base["latency_ms"], candidate_latency_ms=cand["latency_ms"],
                    identified_baseline=_progress.judge(judge, case, base["replies"], "baseline"),
                    identified_candidate=_progress.judge(judge, case, cand["replies"], "candidate"),
                    flip_verified=None,
                )
                if spec.get("smoke"):
                    record["flip_verification_skipped"] = "smoke"
                if not spec.get("smoke") and record["identified_baseline"] != record["identified_candidate"] and extra_rounds:
                    b_votes = [record["identified_baseline"]]
                    c_votes = [record["identified_candidate"]]
                    for round_index in range(1, extra_rounds + 1):
                        b_reply = _progress.generate(baseline_gen, case, "baseline", round=round_index)
                        b_votes.append(_progress.judge(judge, case, b_reply["replies"], "baseline", round_index))
                        c_reply = _progress.generate(cand_gen, case, "candidate", round=round_index)
                        c_votes.append(_progress.judge(judge, case, c_reply["replies"], "candidate", round_index))
                    record["flip_verified"] = {
                        "baseline_votes": b_votes,
                        "candidate_votes": c_votes,
                        "baseline_identified_final": sum(b_votes) > len(b_votes) / 2,
                        "candidate_identified_final": sum(c_votes) > len(c_votes) / 2,
                    }
            except Exception as exc:  # noqa: BLE001 - 单题失败豁免（SOP §4）
                record["status"] = "failed"
                record["reason"] = str(exc)[:200]
                _logger.warning("单题失败 %s: %s", cid, exc)
            _progress.finish_case(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            _progress.completed(record)
            done[cid] = record if record["status"] == "ok" else done.get(cid)
            _logger.info("[%s] 进度 %d/%d", spec["id"], len(done), len(cases))

    _progress.stage("summarizing", branch="", round=0)
    final_records, retries = _final_records(cases_jsonl)
    ok_records = [r for r in final_records.values() if r.get("status") != "failed"]
    failures = len(final_records) - len(ok_records)
    metrics = _metrics(ok_records, attempted=len(cases), failures=failures, retries=retries)
    verdict = protocol.decide(metrics, spec["dataset"], proto)
    experiment.finish(exp_dir, metrics, verdict)
    report.write_run(exp_dir)
    report.refresh_dashboard(versions.PRIVATE / "experiments", versions.PRIVATE.parent.parent / "dashboard" / versions.PRIVATE.name)
    report.write_instance_index(versions.PRIVATE.parent, versions.PRIVATE.parent.parent / "dashboard")
    return exp_dir


def _final_records(cases_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """每题取最后一条记录为最终状态；重试次数 = 总行数 - 题数（SOP §4）。"""
    final: dict[str, dict[str, Any]] = {}
    lines = 0
    if cases_path.exists():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            lines += 1
            rec = json.loads(line)
            final[str(rec["case_id"])] = rec
    return final, max(0, lines - len(final))


def _metrics(records: list[dict[str, Any]], attempted: int, failures: int, retries: int = 0) -> dict[str, Any]:
    """净胜只计翻转补验确认且非 contested 的题；失败率分母 = 整轮每题。"""
    def _count(pred) -> int:
        return sum(1 for r in records if pred(r))

    pairs = len(records)
    identified_b = _count(lambda r: r["identified_baseline"])
    identified_c = _count(lambda r: r["identified_candidate"])
    wins = _count(lambda r: r["identified_baseline"] and not r["identified_candidate"])
    losses = _count(lambda r: r["identified_candidate"] and not r["identified_baseline"])
    ties = pairs - wins - losses

    wins_c = losses_c = contested = 0
    for r in records:
        v = r.get("flip_verified")
        if v is None:
            continue
        b, c = v["baseline_identified_final"], v["candidate_identified_final"]
        if b == c:
            contested += 1
        elif b:
            wins_c += 1
        else:
            losses_c += 1

    def _rate(n: int, subset: int | None = None) -> float:
        denom = subset if subset else pairs
        return round(n / denom, 4) if denom else 0.0

    group_n = _count(lambda r: r.get("chat_type") == "group")
    return {
        "attempted": attempted,
        "failures": failures,
        "retries": retries,
        "failure_rate": round(failures / attempted, 4) if attempted else 0.0,
        "pairs": pairs,
        "identified_baseline": identified_b,
        "identified_candidate": identified_c,
        "identification_rate_baseline": _rate(identified_b),
        "identification_rate_candidate": _rate(identified_c),
        "identification_rate_baseline_group": _rate(
            _count(lambda r: r.get("chat_type") == "group" and r["identified_baseline"]), group_n
        ),
        "identification_rate_candidate_group": _rate(
            _count(lambda r: r.get("chat_type") == "group" and r["identified_candidate"]), group_n
        ),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_win_raw": wins - losses,
        "wins_confirmed": wins_c,
        "losses_confirmed": losses_c,
        "contested": contested,
        "net_win_confirmed": wins_c - losses_c,
        "sign_p": protocol.sign_test_two_sided(wins_c, losses_c),
        "latency": _latency_summary(records),
    }


def _latency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        return round(s[min(len(s) - 1, int(len(s) * p))], 1)

    b = [r.get("baseline_latency_ms", 0.0) for r in records]
    c = [r.get("candidate_latency_ms", 0.0) for r in records]
    return {
        "baseline_p50": _pct(b, 0.5),
        "baseline_p95": _pct(b, 0.95),
        "candidate_p50": _pct(c, 0.5),
        "candidate_p95": _pct(c, 0.95),
    }


# ---------- Judge 评估执行（SOP §7） ----------

@track_run
def run_judge_experiment(exp_dir: Path, *, _progress=None) -> Path:
    """Judge 校准（SOP §4.3）：同一冻结 pack 上现用 vs 候选版本，只比识别率。
    候选识别率不低于现用（不更差）→ adopt。无真人vs真人试验、无训练、无绑定。"""
    spec = experiment.spec_of(exp_dir)
    if spec["kind"] != experiment.KIND_JUDGE_EVAL:
        raise ConfigError(f"不是 Judge 校准实验: {spec['kind']}")
    state = experiment.state_of(exp_dir)
    if state.get("status") == "finished":
        report.write_run(exp_dir)
        return exp_dir

    pack = json.loads(
        (versions.PRIVATE / "judge_eval" / spec["pack_ref"] / "pack.json").read_text(encoding="utf-8")
    )
    if not pack.get("rows"):
        raise ConfigError("校准包为空（--sample 0 或无可用样本）：拒绝比较，防止 0≥0 误判通过")
    _progress.stage("preparing", total=len(pack["rows"]))
    base_info = versions.judge_dir(spec["baseline_ref"])
    cand_info = versions.judge_dir(spec["candidate_ref"])

    base_scorer = build_judge(load_settings(), base_info)
    cand_scorer = build_judge(load_settings(), cand_info)

    # 每题两版各自判定：识别数 + 配对净胜，喂给与生成器同一个 decide()（与生成器同构）
    # 逐题 checkpoint：恢复时跳过已成功的题，不重复调用（SOP §2.4）
    cases_path = exp_dir / "cases.jsonl"
    done: dict[str, dict] = {}
    if cases_path.exists():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[str(r["case_id"])] = r
    base_correct: list[bool] = []
    cand_correct: list[bool] = []
    with cases_path.open("a", encoding="utf-8") as fout:
        for case_index, row in enumerate(pack["rows"], 1):
            cid = str(row["case_id"])
            if cid in done and done[cid].get("status") != "failed":
                base_correct.append(bool(done[cid]["baseline_correct"]))
                cand_correct.append(bool(done[cid]["candidate_correct"]))
                continue
            case = row
            _progress.stage("preparing", case_index=case_index, branch="", round=0)
            _progress.start_case(case)
            record: dict = {"case_id": cid}
            try:
                # 初测 + 翻转补测都属单题调用：任一失败按 SOP §2.4 豁免，不中断整轮（#3）
                b = bool(_progress.judge(base_scorer, case, row["ai_replies"], "baseline"))
                c = bool(_progress.judge(cand_scorer, case, row["ai_replies"], "candidate"))
                record["baseline_correct"] = b
                record["candidate_correct"] = c
                if b != c:
                    # 翻转补测与初测写在同一条记录里：记录落盘即完成，恢复不会读到半条
                    extra = int((spec.get("protocol") or {}).get("flip_extra_rounds", 2))
                    b_votes, c_votes = [b], [c]
                    for round_index in range(1, extra + 1):
                        b_votes.append(bool(_progress.judge(base_scorer, case, row["ai_replies"], "baseline", round_index)))
                        c_votes.append(bool(_progress.judge(cand_scorer, case, row["ai_replies"], "candidate", round_index)))
                    record["flip_verified"] = True
                    record["baseline_identified_final"] = sum(b_votes) > len(b_votes) / 2
                    record["candidate_identified_final"] = sum(c_votes) > len(c_votes) / 2
            except Exception as exc:  # noqa: BLE001
                record = {"case_id": cid, "status": "failed", "reason": str(exc)[:200]}
            _progress.finish_case(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            _progress.completed(record)
    # 按 case_id 从最终记录汇总（#2：失败题排除后不再与题目错位）
    _progress.stage("summarizing", branch="", round=0)
    done_rows, _ = _final_records(exp_dir / "cases.jsonl")
    total = len(pack["rows"])
    failures = 0
    base_hits = cand_hits = n_ok = 0
    raw_wins = raw_losses = raw_ties = 0
    wins = losses = contested = 0
    for row in pack["rows"]:
        r = done_rows.get(str(row["case_id"]), {})
        if r.get("status") == "failed":
            failures += 1
            continue
        b, c = bool(r.get("baseline_correct")), bool(r.get("candidate_correct"))
        n_ok += 1
        base_hits += b
        cand_hits += c
        if b == c:
            raw_ties += 1
            continue
        if c:
            raw_wins += 1
        else:
            raw_losses += 1
        if not r.get("flip_verified"):
            if c:
                wins += 1
            else:
                losses += 1
            continue
        bf, cf = bool(r["baseline_identified_final"]), bool(r["candidate_identified_final"])
        if bf == cf:
            contested += 1
        elif bf:
            losses += 1
        else:
            wins += 1
    failure_rate = round(failures / total, 4) if total else 0.0
    _, retries = _final_records(exp_dir / "cases.jsonl")
    metrics = {
        "pairs": n_ok,
        "identified_baseline": base_hits,    # 对 judge 实验 = 正确识别数（decide 复用字段）
        "identified_candidate": cand_hits,
        "wins": raw_wins, "losses": raw_losses, "ties": raw_ties,
        "wins_confirmed": wins, "losses_confirmed": losses,
        "contested": contested,
        "net_win_confirmed": wins - losses,
        "sign_p": protocol.sign_test_two_sided(wins, losses), "n": total,
        "failure_rate": failure_rate, "failures": failures, "attempted": total, "retries": retries,
        "identification_rate_baseline": round(base_hits / n_ok, 4) if n_ok else 0.0,
        "identification_rate_candidate": round(cand_hits / n_ok, 4) if n_ok else 0.0,
    }
    # judge 的优化方向是识别率更高：decide() 的「严格下降」对 baseline 而言即候选更多
    verdict = protocol.decide(metrics, "development" if spec.get("dataset") != "fixed_test" else "fixed_test",
                              spec.get("protocol") or experiment._protocol_snapshot(load_settings()),
                              higher_is_better=True)
    experiment.finish(exp_dir, metrics, verdict)
    report.write_run(exp_dir)
    report.refresh_dashboard(versions.PRIVATE / "experiments", versions.PRIVATE.parent.parent / "dashboard" / versions.PRIVATE.name)
    report.write_instance_index(versions.PRIVATE.parent, versions.PRIVATE.parent.parent / "dashboard")
    return exp_dir


def _score_pack(pack: dict, scorer: Judge) -> dict[str, float]:
    hits = 0
    for row in pack["rows"]:
        hits += scorer.is_ai(row, row["ai_replies"])
    n = len(pack["rows"]) or 1
    return {"identification_rate": round(hits / n, 4), "n": len(pack["rows"])}
