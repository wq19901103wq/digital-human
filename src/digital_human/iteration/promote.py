"""晋升（SOP §3/§4）：唯一入口，晋升 = 校验实验结论 → 切指针。

- gen dev 实验通过 → iteration_gen = 候选版本（版本已存在，无拷贝）
- gen formal 实验通过 → production_gen = 候选版本（= 实验时的开发基线）
- judge 校准实验通过 → judge = 候选 Judge 版本

禁止手工编辑 pointers.json / 版本目录（不可变）。
"""
from __future__ import annotations

from ..config import ConfigError
from . import experiment, versions

DEVELOPMENT_PASS = "merge_to_iteration_baseline"
FORMAL_PASS = "adopt"


def _require_finished(exp_id: str) -> tuple[object, dict, dict]:
    exp_dir = experiment.load_experiment(exp_id)
    spec = experiment.spec_of(exp_dir)
    state = experiment.state_of(exp_dir)
    if spec.get("smoke"):
        raise ConfigError("冒烟实验不得晋升")
    if state.get("status") != "finished":
        raise ConfigError(f"实验未完赛: {exp_id}（status={state.get('status', 'running')}）")
    return exp_dir, spec, state


def promote_gen(exp_id: str) -> dict[str, str]:
    exp_dir, spec, state = _require_finished(exp_id)
    verdict = state["verdict"]
    pointers = versions.load_pointers()

    if spec["dataset"] == "development":
        if verdict != DEVELOPMENT_PASS:
            raise ConfigError(f"开发实验决定为 {verdict}，要求 {DEVELOPMENT_PASS}；禁止后门晋升")
        if pointers["iteration_gen"] != spec["baseline_ref"]:
            raise ConfigError(
                f"开发基线已前进（{pointers['iteration_gen']} ≠ 实验冻结的 {spec['baseline_ref']}）："
                "实验已过期，禁止覆盖"
            )
        versions.generator_dir(spec["candidate_ref"])  # 版本必须存在
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
    pointers["production_gen"] = spec["candidate_ref"]
    versions.save_pointers(pointers)
    return pointers


def promote_judge(exp_id: str) -> dict[str, str]:
    """与 promote_gen 同构：dev 通过 → iteration_judge；fixed 双条件通过 → production_judge。"""
    exp_dir, spec, state = _require_finished(exp_id)
    if spec["kind"] != experiment.KIND_JUDGE_EVAL:
        raise ConfigError(f"不是 Judge 实验: {spec['kind']}")
    dataset = spec.get("dataset", "development")
    need = DEVELOPMENT_PASS if dataset == "development" else FORMAL_PASS
    if state["verdict"] != need:
        raise ConfigError(f"Judge 实验决定为 {state['verdict']}，要求 {need}；禁止后门晋升")
    pointers = versions.load_pointers()
    if dataset == "development":
        if pointers["iteration_judge"] != spec["baseline_ref"]:
            raise ConfigError(
                f"开发 Judge 已前进（{pointers['iteration_judge']} ≠ 实验冻结的 {spec['baseline_ref']}）：实验已过期"
            )
        versions.judge_dir(spec["candidate_ref"])
        pointers["iteration_judge"] = spec["candidate_ref"]
    else:
        if pointers["production_judge"] != spec["baseline_ref"]:
            raise ConfigError(f"当前生产 Judge {pointers['production_judge']} ≠ 实验基线 {spec['baseline_ref']}：已过期")
        if pointers["iteration_judge"] != spec["candidate_ref"]:
            raise ConfigError(
                f"开发 Judge 已前进（{pointers['iteration_judge']} ≠ 实验候选 {spec['candidate_ref']}）：禁止覆盖"
            )
        pointers["production_judge"] = spec["candidate_ref"]
    versions.save_pointers(pointers)
    return pointers
