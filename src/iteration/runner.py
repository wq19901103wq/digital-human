"""生成器 A/B 跑题循环（SOP §3/§4）：从实验取规格，跑完交回 experiment.finish。

本模块执行 spec 冻结的全部 Baseline/Candidate 题目并统计。
预检/准入/one-shot 在 experiment.create；晋升在 promote；状态在 experiment.finish。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from threading import local

from ..config import ConfigError, load_settings
from ..generator.generator import ReplyGenerator
from ..judge.judge import Judge, build_judge
from . import data_guard, experiment, protocol, versions
from ..dashboard import report
from .progress import ParallelCaseProgress, track_run
from .parallel import completed_map
from . import datasets, record_contract

_logger = logging.getLogger("digital_human.runner")


@track_run
def run_gen_experiment(exp_dir: Path, *, workers=None, _progress=None) -> Path:
    spec = experiment.spec_of(exp_dir)
    data_dir = versions.data_version_dir(spec['data_ref'])
    if (data_dir / 'purposes.json').exists():
        from ..config import sha256_file
        from ..generator.shared_history import shared_history_index
        inputs = {str(path.resolve()): sha256_file(path)
                  for path in (data_dir / 'fewshot_pool.jsonl', data_dir / 'report.json')}
        # Reuse the pinned, serialized retriever across both sides and workers.
        # Full provenance/resume checks still run before any component is built.
        with shared_history_index(inputs):
            return _run_gen_experiment(exp_dir, workers=workers, _progress=_progress)
    return _run_gen_experiment(exp_dir, workers=workers, _progress=_progress)


def _run_gen_experiment(exp_dir: Path, *, workers=None, _progress=None) -> Path:
    spec = experiment.spec_of(exp_dir)
    if spec["kind"] != experiment.KIND_GEN_AB:
        raise ConfigError(f"不是生成器实验: {spec['kind']}")
    from .learning_guard import execution_seal
    seal = execution_seal(spec)
    state = experiment.state_of(exp_dir)
    if state.get("status") == "finished":
        _logger.info("实验已完赛(%s)，直接出报告", state.get("verdict"))
        report.write_run(exp_dir)
        return exp_dir
    from . import measurements
    measurements.apply_replay_safely(exp_dir)  # 同对复用：重放已完成题，只跑缺的

    from .acceptance import transition
    transition(spec, 'opened')
    settings = load_settings()
    data_dir = versions.data_version_dir(spec["data_ref"])
    baseline = versions.load_generator(spec["baseline_ref"])
    candidate = versions.load_generator(spec["candidate_ref"])
    judge_info = versions.judge_dir(spec["judge_ref"])

    cases = datasets.rows_for(spec)
    proto = spec.get("protocol") or experiment._protocol_snapshot(settings)  # 快照优先，不回读当前 settings
    if spec.get("smoke"):
        cases = cases[: int(spec.get("smoke_limit") or 5)]  # --limit N 按 N 执行（SOP §2.2）
    data_guard.verify(exp_dir, data_guard.generation_snapshot(cases, data_dir,
                      [baseline["config"], candidate["config"]]),
                      expected=spec.get("data_snapshot"), has_checkpoint=data_guard.has_results(exp_dir))
    workers = (spec.get('execution') or {}).get('workers', 1) if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ConfigError('workers 必须为 1–16 的整数')
    _progress.stage('parallel' if workers > 1 else 'preparing', total=len(cases), workers=workers)

    from ..llm import build_clients
    worker_state = local()

    def components():
        if not hasattr(worker_state, 'components'):
            generators = [ReplyGenerator(settings, info['config'], build_clients(settings, info['config']['llm']),
                          prompt_root=info['dir'], pool_path=data_dir / 'fewshot_pool.jsonl')
                          for info in (baseline, candidate)]
            worker_state.components = (*generators, build_judge(settings, judge_info))
            for component in worker_state.components:
                component.source_check = seal.check
            seal.check()
        return worker_state.components

    # 触发预检：续跑时跳过（候选已验证过且 cases 已有记录说明能跑）
    cases_jsonl = exp_dir / "cases.jsonl"
    if not spec.get("smoke") and not cases_jsonl.exists():
        _progress.stage("preflight")
        baseline_gen, cand_gen, _ = components()
        experiment._verify_trigger(baseline_gen, cand_gen, spec["config_diff"], cases)

    extra_rounds = proto["flip_extra_rounds"]

    def evaluate(item):
        seal.check()
        case_index, case = item
        cid = str(case['case_id'])
        progress = ParallelCaseProgress(_progress, case_index) if workers > 1 else _progress
        progress.stage('preparing', case_index=case_index, branch='', round=0)
        progress.start_case(case)
        record: dict[str, Any] = {
            'case_id': cid, 'chat_type': case.get('chat_type'),
            'context': case.get('context'), 'human_reply': case.get('human_reply'), 'status': 'ok',
        }
        try:
            baseline_gen, cand_gen, judge = components()
            base = progress.generate(baseline_gen, case, 'baseline', proto['force_reply'])
            cand = progress.generate(cand_gen, case, 'candidate', proto['force_reply'])
            record.update(
                baseline_replies=base['replies'], candidate_replies=cand['replies'],
                baseline_latency_ms=base['latency_ms'], candidate_latency_ms=cand['latency_ms'],
                identified_baseline=progress.judge(judge, case, base['replies'], 'baseline'),
                identified_candidate=progress.judge(judge, case, cand['replies'], 'candidate'),
                flip_verified=None,
            )
            if spec.get('smoke'):
                record['flip_verification_skipped'] = 'smoke'
            if not spec.get('smoke') and record['identified_baseline'] != record['identified_candidate'] and extra_rounds:
                b_votes = [record['identified_baseline']]
                c_votes = [record['identified_candidate']]
                for round_index in range(1, extra_rounds + 1):
                    b_reply = progress.generate(baseline_gen, case, 'baseline', round=round_index)
                    b_votes.append(progress.judge(judge, case, b_reply['replies'], 'baseline', round_index))
                    c_reply = progress.generate(cand_gen, case, 'candidate', round=round_index)
                    c_votes.append(progress.judge(judge, case, c_reply['replies'], 'candidate', round_index))
                record['flip_verified'] = {
                    'baseline_votes': b_votes, 'candidate_votes': c_votes,
                    'baseline_identified_final': sum(b_votes) > len(b_votes) / 2,
                    'candidate_identified_final': sum(c_votes) > len(c_votes) / 2,
                }
        except ConfigError:
            progress.trace.finish('interrupted')
            raise
        except Exception as exc:  # 单题失败保留原重试及计分口径。
            record['status'] = 'failed'
            record['reason'] = str(exc)[:200]
            _logger.warning('单题失败 %s: %s', cid, exc)
        except BaseException:
            progress.trace.finish('interrupted')
            raise
        seal.check()
        progress.finish_case(record)
        return record

    # 恢复：只跳过成功题；失败题最终状态为 failed，必须可重试（SOP §2.4）
    final, _ = _final_records(cases_jsonl)
    done = {cid: r for cid, r in final.items() if r.get('status') == 'ok'}
    pending = [(i, case) for i, case in enumerate(cases, 1) if str(case['case_id']) not in done]
    _progress.stage('parallel' if workers > 1 else 'preparing')
    with record_contract.writer(exp_dir) as append:
        for record in completed_map(evaluate, pending, workers):
            seal.check()
            append(record)
            _progress.completed(record)
            if record['status'] == 'ok':
                done[record['case_id']] = record
            _logger.info("[%s] 进度 %d/%d", spec["id"], len(done), len(cases))

    _progress.stage("summarizing", branch="", round=0)
    final_records, retries = _final_records(cases_jsonl)
    ok_records = [r for r in final_records.values() if r.get("status") != "failed"]
    failures = len(final_records) - len(ok_records)
    metrics = _metrics(ok_records, attempted=len(cases), failures=failures, retries=retries)
    if any('familiarity' in c for c in cases):
        metrics['cohorts'] = datasets.cohort_metrics(cases, final_records)
    verdict = protocol.decide(metrics, spec["dataset"], proto)
    seal.check()
    experiment.finish(exp_dir, metrics, verdict)
    report.publish(exp_dir)
    return exp_dir


def _final_records(cases_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    return protocol.summarize_final_records(cases_path)


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
def run_judge_experiment(exp_dir: Path, *, workers=None, _progress=None) -> Path:
    """Judge 校准（SOP §4.3）：同一冻结 pack 上现用 vs 候选版本，只比识别率。
    候选识别率不低于现用（不更差）→ adopt。无真人vs真人试验、无训练、无绑定。"""
    spec = experiment.spec_of(exp_dir)
    if spec["kind"] != experiment.KIND_JUDGE_EVAL:
        raise ConfigError(f"不是 Judge 校准实验: {spec['kind']}")
    from .learning_guard import verify_repair
    verify_repair(spec)
    from .learning_guard import execution_seal
    seal = execution_seal(spec)
    state = experiment.state_of(exp_dir)
    if state.get("status") == "finished":
        report.write_run(exp_dir)
        return exp_dir
    from . import measurements
    measurements.apply_replay_safely(exp_dir)  # 同对复用：重放已完成题，只跑缺的

    from hashlib import sha256
    pack_bytes = (versions.PRIVATE / "judge_eval" / spec["pack_ref"] / "pack.json").read_bytes()
    data_guard.verify(exp_dir, data_guard.pack_snapshot(sha256(pack_bytes).hexdigest()),
                      expected=data_guard.pack_snapshot(spec["pack_sha256"]) if spec.get("pack_sha256") else None,
                      has_checkpoint=data_guard.has_results(exp_dir))
    pack = json.loads(pack_bytes)
    if spec.get('purpose_snapshot'):
        data_dir = versions.data_version_dir(spec['data_ref'])
        if datasets.snapshot(data_dir) != spec['purpose_snapshot']:
            raise ConfigError('Judge 用途清单或时间协议发生变化')
        datasets.assert_pack(data_dir, pack, 'judge_development' if spec['dataset'] == 'development' else 'fixed_test')
    observation_pack = pack
    if spec.get("smoke") and spec.get("smoke_limit"):
        pack = {**pack, "rows": pack["rows"][:int(spec["smoke_limit"])]}
    if not pack.get("rows"):
        raise ConfigError("校准包为空（--sample 0 或无可用样本）：拒绝比较，防止 0≥0 误判通过")
    workers = (spec.get('execution') or {}).get('workers', 1) if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ConfigError('workers 必须为 1–16 的整数')
    ids = [str(row['case_id']) for row in pack['rows']]
    if len(set(ids)) != len(ids):
        raise ConfigError('评估包存在重复 case_id')
    _progress.stage('parallel' if workers > 1 else 'preparing', total=len(pack['rows']), workers=workers)
    base_info = versions.judge_dir(spec['baseline_ref'])
    cand_info = versions.judge_dir(spec['candidate_ref'])
    if spec.get('lr_retrain'):
        from ..judge import lr_retrain
        if sha256(Path(lr_retrain.__file__).read_bytes()).hexdigest() != spec['lr_retrain']['implementation_sha256']:
            raise ConfigError('重训对比实现已变化，拒绝混用旧结果')
    from .acceptance import transition
    transition(spec, 'opened')
    settings = load_settings()
    worker_state = local()

    def evaluate(item):
        seal.check()
        case_index, row = item
        progress = ParallelCaseProgress(_progress, case_index) if workers > 1 else _progress
        progress.stage('preparing', case_index=case_index, branch='', round=0)
        progress.start_case(row)
        record = {'case_id': str(row['case_id']), 'status': 'ok'}
        try:
            if row.get('generation_status') == 'failed':
                error = row['generation_error']
                raise RuntimeError(f"评估包生成失败：{error['message']}")
            if not hasattr(worker_state, 'scorers'):
                if spec.get('saved_draw_replay'):
                    if spec.get('lr_retrain') or spec['dataset'] != 'development':
                        raise ConfigError('Saved draw replay is only valid for ordinary development')
                    from .draw_replay import scorers
                    worker_state.scorers = scorers(spec['saved_draw_replay'], observation_pack, base_info, cand_info,
                        supplement_missing=spec.get('supplement_missing_rounds', False))
                elif spec.get('lr_retrain'):
                    from ..judge.lr_retrain import FeatureReplay, LRJudge
                    source_dir = None
                    source_spec = None
                    source_config = {'llm': spec['lr_retrain'].get('feature_config', {})}
                    if spec['lr_retrain'].get('source_experiment'):
                        source_dir = versions.PRIVATE / 'experiments' / spec['lr_retrain']['source_experiment']
                        source_spec = experiment.spec_of(source_dir)
                        if sha256((source_dir / 'spec.json').read_bytes()).hexdigest() != spec['lr_retrain']['source_spec_sha256']:
                            raise ConfigError('重训对比的来源实验已变化')
                        source_config = versions.judge_dir(source_spec['candidate_ref'])['config']
                    if not source_config['llm']:
                        raise ConfigError('重训配对实验缺少冻结特征配置')
                    feature_config = {**source_config['llm'], **source_config.get('feature_llm', {})}
                    for info in (base_info, cand_info):
                        cfg = info['config']
                        if (cfg.get('decision_policy') != 'lr_only' or
                                {**cfg['llm'], **cfg.get('feature_llm', {})} != feature_config):
                            raise ConfigError('重训配对实验必须使用相同特征配置和全 LR 判别')
                    replay = FeatureReplay(source_dir, source_spec, source_config, pack['rows'])
                    worker_state.scorers = tuple(LRJudge(info['config'], info['dir'], replay=replay)
                                                 for info in (base_info, cand_info))
                else:
                    worker_state.scorers = (build_judge(settings, base_info), build_judge(settings, cand_info))
                for scorer in worker_state.scorers:
                    scorer.source_check = seal.check
                seal.check()
            base_scorer, cand_scorer = worker_state.scorers
            b = bool(progress.judge(base_scorer, row, row['ai_replies'], 'baseline'))
            c = bool(progress.judge(cand_scorer, row, row['ai_replies'], 'candidate'))
            record.update(baseline_correct=b, candidate_correct=c)
            extra = int((spec.get('protocol') or {}).get('flip_extra_rounds', 2))
            if b != c and extra and not spec.get('smoke'):
                b_votes, c_votes = [b], [c]
                for round_index in range(1, extra + 1):
                    b_votes.append(bool(progress.judge(base_scorer, row, row['ai_replies'], 'baseline', round_index)))
                    c_votes.append(bool(progress.judge(cand_scorer, row, row['ai_replies'], 'candidate', round_index)))
                record.update(flip_verified=True, baseline_votes=b_votes, candidate_votes=c_votes,
                    baseline_identified_final=sum(b_votes) > len(b_votes) / 2,
                    candidate_identified_final=sum(c_votes) > len(c_votes) / 2)
        except ConfigError:
            progress.trace.finish('interrupted')
            raise
        except Exception as exc:  # 单题失败仍可续跑；有效率分母不含失败题。
            record = {'case_id': str(row['case_id']), 'status': 'failed', 'reason': str(exc)[:200]}
            if row.get('generation_trace_ref'):
                record['generation_trace_ref'] = row['generation_trace_ref']
        except BaseException:
            progress.trace.finish('interrupted')
            raise
        seal.check()
        progress.finish_case(record)
        return record

    cases_path = exp_dir / 'cases.jsonl'
    done, _ = _final_records(cases_path)
    pending = [(i, row) for i, row in enumerate(pack['rows'], 1)
               if str(row['case_id']) not in done or done[str(row['case_id'])].get('status') == 'failed']
    with record_contract.writer(exp_dir) as append:
        for record in completed_map(evaluate, pending, workers):
            seal.check()
            append(record)
            _progress.completed(record)
            done[record['case_id']] = record
            _logger.info('[%s] 进度 %d/%d', spec['id'], len(done), len(pack['rows']))
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
    if any('familiarity' in c for c in pack['rows']):
        metrics['cohorts'] = datasets.cohort_metrics(pack['rows'], done_rows, judge=True)
    verdict = protocol.decide(metrics, "development" if spec.get("dataset") != "fixed_test" else "fixed_test",
                              spec.get("protocol") or experiment._protocol_snapshot(load_settings()),
                              higher_is_better=True)
    seal.check()
    experiment.finish(exp_dir, metrics, verdict)
    report.publish(exp_dir)
    return exp_dir


def _score_pack(pack: dict, scorer: Judge) -> dict[str, float]:
    hits = 0
    for row in pack["rows"]:
        hits += scorer.is_ai(row, row["ai_replies"])
    n = len(pack["rows"]) or 1
    return {"identification_rate": round(hits / n, 4), "n": len(pack["rows"])}
