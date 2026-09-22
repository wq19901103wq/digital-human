"""晋升（SOP §3/§4）：唯一入口，晋升 = 校验实验结论 → 切指针。

- gen dev 实验通过 → iteration_gen = 候选版本（版本已存在，无拷贝）
- gen formal 实验通过 → production_gen = 候选版本（= 实验时的开发基线）
- judge 开发晋级 → iteration_judge；固定验收晋级 → production_judge

禁止手工编辑 pointers.json / 版本目录（不可变）。
"""
from __future__ import annotations

import json

from math import ceil

from ..config import ConfigError, load_settings, sha256_file
from . import experiment, gates, protocol, versions
from .storage import read_json, write_once_json

DEVELOPMENT_PASS = "merge_to_iteration_baseline"
FORMAL_PASS = "adopt"


def threshold_review(directory):
    """Read a separate policy decision without rewriting the frozen experiment."""
    receipt = read_json(directory / 'threshold_review.json', default={})
    if receipt and any(sha256_file(directory / name) != receipt['evidence_sha256'][name]
                       for name in ('spec.json', 'state.json')):
        raise ConfigError('门槛复核的原实验记录已变化')
    return receipt


def _review_thresholds(directory, spec, state, reason):
    if not reason or not reason.strip():
        raise ConfigError('调整已完成实验的采用门槛必须记录用户决定')
    if (spec.get('dataset') != 'fixed_test' or spec.get('branch_round') or
            spec['protocol'].get('gate_schema', 1) < 2 or not spec.get('record_contract')):
        raise ConfigError('门槛复核仅支持有完整契约和准入凭证的独立固定实验')
    current = experiment._protocol_snapshot(load_settings())
    # Only acceptance thresholds change; requests, independent votes and failure limits stay frozen.
    revised = {**spec['protocol'], 'gate_schema': 3}
    for key in ('dev_min_net_win_rate', 'fixed_entry_min_net_win_rate', 'formal_min_net_win_rate'):
        revised[key] = current[key]
    result = protocol.decide(state['metrics'], 'fixed_test', revised,
                             higher_is_better=spec['kind'] == experiment.KIND_JUDGE_EVAL)
    receipt = dict(schema=1, experiment_id=spec['id'], reason=reason,
                   original_verdict=state['verdict'], original_protocol=spec['protocol'],
                   protocol=revised, decision=result,
                   evidence_sha256={name: sha256_file(directory / name)
                                    for name in ('spec.json', 'state.json', 'cases.jsonl')})
    existing = threshold_review(directory)
    if existing and existing != receipt:
        raise ConfigError('此实验已有不同的门槛复核决定，禁止覆盖')
    return receipt


def _record_review(directory, spec, receipt):
    if not receipt:
        return
    # The ordinary promotion path has already verified the original entry receipt.
    entry = spec['fixed_entry']
    needed = max(1, ceil(receipt['protocol']['fixed_entry_min_net_win_rate'] * entry['pairs']))
    if entry['net_win_confirmed'] < needed:
        raise ConfigError('原开发对比未达到新的固定轮准入门槛')
    write_once_json(directory / 'threshold_review.json', receipt)


def _require_finished(exp_id: str) -> tuple[object, dict, dict]:
    exp_dir = experiment.load_experiment(exp_id)
    spec = experiment.spec_of(exp_dir)
    gates.require_promotable(spec)
    gates.verify_learning(spec)
    state = experiment.state_of(exp_dir)
    if state.get("status") != "finished":
        raise ConfigError(f"实验未完赛: {exp_id}（status={state.get('status', 'running')}）")
    if spec.get('record_contract'):
        from .record_contract import validate_completion
        validate_completion(exp_dir, state.get('metrics', {}))
    return exp_dir, spec, state


@versions.transaction
def promote_gen(exp_id: str, *, threshold_reason: str | None = None) -> dict[str, str]:
    exp_dir, spec, state = _require_finished(exp_id)
    if spec["kind"] != experiment.KIND_GEN_AB:
        raise ConfigError(f"不是生成器实验: {spec['kind']}")
    review = _review_thresholds(exp_dir, spec, state, threshold_reason) if threshold_reason is not None else None
    if spec.get("branch_round"):
        from .branches import promote_experiment
        return promote_experiment(exp_id)
    verdict = review['decision']['verdict'] if review else state["verdict"]
    pointers = versions.load_pointers()
    if spec["data_ref"] != pointers["data"] or spec["judge_ref"] != pointers["production_judge"]:
        raise ConfigError("数据或生产 Judge 已前进：实验已过期，必须重测")
    if review and threshold_review(exp_dir) == review and pointers['production_gen'] == spec['candidate_ref']:
        return pointers

    if spec["dataset"] == "development":
        if spec.get('against_production'):
            raise ConfigError('生产对比只用于固定轮准入，不重复晋级开发版')
        if verdict != DEVELOPMENT_PASS:
            raise ConfigError(f"开发实验决定为 {verdict}，要求 {DEVELOPMENT_PASS}；禁止后门晋升")
        if pointers["iteration_gen"] != spec["baseline_ref"]:
            raise ConfigError(
                f"开发基线已前进（{pointers['iteration_gen']} ≠ 实验冻结的 {spec['baseline_ref']}）："
                "实验已过期，禁止覆盖"
            )
        versions.generator_dir(spec["candidate_ref"])  # 版本必须存在
        if spec['protocol'].get('gate_schema', 1) >= 2:
            gates.require_development(exp_dir)
        pointers["iteration_gen"] = spec["candidate_ref"]
        versions.save_pointers(pointers)
        return pointers

    if verdict != FORMAL_PASS:
        raise ConfigError(f"固定实验决定为 {verdict}，要求 {FORMAL_PASS}；生产指针保持原版本")
    if pointers["production_gen"] != spec["baseline_ref"]:
        raise ConfigError(
            f"当前生产基线 {pointers['production_gen']} ≠ 实验冻结的 {spec['baseline_ref']}：已过期"
        )
    if pointers["iteration_gen"] != spec["candidate_ref"]:
        raise ConfigError(
            f"开发基线已前进（{pointers['iteration_gen']} ≠ 实验候选 {spec['candidate_ref']}）："
            "该候选必须用其原实验晋升，或废弃后重测"
        )
    if spec['protocol'].get('gate_schema', 1) >= 2:
        gates.verify_fixed_receipt(spec)
    _record_review(exp_dir, spec, review)
    pointers["production_gen"] = spec["candidate_ref"]
    versions.save_pointers(pointers)
    return pointers


@versions.transaction
def promote_judge(exp_id: str, *, threshold_reason: str | None = None) -> dict[str, str]:
    """与 promote_gen 同构：dev 通过 → iteration_judge；fixed 双条件通过 → production_judge。"""
    exp_dir, spec, state = _require_finished(exp_id)
    if spec["kind"] != experiment.KIND_JUDGE_EVAL:
        raise ConfigError(f"不是 Judge 实验: {spec['kind']}")
    review = _review_thresholds(exp_dir, spec, state, threshold_reason) if threshold_reason is not None else None
    if spec.get("branch_round"):
        from .branches import promote_experiment
        return promote_experiment(exp_id)
    dataset = spec.get("dataset", "development")
    need = DEVELOPMENT_PASS if dataset == "development" else FORMAL_PASS
    verdict = review['decision']['verdict'] if review else state['verdict']
    if verdict != need:
        raise ConfigError(f"Judge 实验决定为 {verdict}，要求 {need}；禁止后门晋升")
    pointers = versions.load_pointers()
    pack = json.loads((versions.PRIVATE / "judge_eval" / spec["pack_ref"] / "pack.json").read_text())
    if spec["data_ref"] != pointers["data"] or pack["c0_gen_version"] != pointers["production_gen"]:
        raise ConfigError("数据或生产生成器已前进：Judge 实验已过期，必须重测")
    if review and threshold_review(exp_dir) == review and pointers['production_judge'] == spec['candidate_ref']:
        return pointers
    if dataset == "development":
        if spec.get('against_production'):
            raise ConfigError('生产对比只用于固定轮准入，不重复晋级开发版')
        if pointers["iteration_judge"] != spec["baseline_ref"]:
            raise ConfigError(
                f"开发 Judge 已前进（{pointers['iteration_judge']} ≠ 实验冻结的 {spec['baseline_ref']}）：实验已过期"
            )
        versions.judge_dir(spec["candidate_ref"])
        if spec['protocol'].get('gate_schema', 1) >= 2:
            gates.require_development(exp_dir)
        pointers["iteration_judge"] = spec["candidate_ref"]
    else:
        if pointers["production_judge"] != spec["baseline_ref"]:
            raise ConfigError(f"当前生产 Judge {pointers['production_judge']} ≠ 实验基线 {spec['baseline_ref']}：已过期")
        if pointers["iteration_judge"] != spec["candidate_ref"]:
            raise ConfigError(
                f"开发 Judge 已前进（{pointers['iteration_judge']} ≠ 实验候选 {spec['candidate_ref']}）：禁止覆盖"
            )
        if spec['protocol'].get('gate_schema', 1) >= 2:
            gates.verify_fixed_receipt(spec)
        _record_review(exp_dir, spec, review)
        pointers["production_judge"] = spec["candidate_ref"]
    versions.save_pointers(pointers)
    return pointers
