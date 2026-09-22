"""对抗迭代协议（机制层）：统计检验与采用决策（纯逻辑）。

预检/准入/实验管理在 experiment.py；版本与指针在 versions.py；晋升在 promote.py。
"""
from __future__ import annotations

from math import ceil, comb
from pathlib import Path
from typing import Any

from ..config import load_settings


def config_diff(base: dict[str, Any], cand: dict[str, Any], prefix: str = "") -> list[str]:
    """计算 Baseline→Candidate 的配置 diff（点路径列表）。

    SOP: 每轮账单必须包含且仅包含这一个 diff；diff 为空 = 没有改动，拒绝运行。
    """
    diffs: list[str] = []
    for key in sorted(set(base) | set(cand)):
        path = f"{prefix}.{key}" if prefix else key
        bv, cv = base.get(key), cand.get(key)
        if isinstance(bv, dict) and isinstance(cv, dict):
            diffs.extend(config_diff(bv, cv, path))
        elif bv != cv:
            diffs.append(f"{path}: {bv!r} -> {cv!r}")
    return diffs


def sign_test_two_sided(wins: int, losses: int) -> float:
    """配对翻转的双侧精确符号检验 p 值。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def decide(metrics: dict[str, Any], dataset: str, protocol: dict[str, Any],
           higher_is_better: bool = False) -> dict[str, Any]:
    """按实验快照的协议超参（experiment.spec["protocol"]）给出决定。

    口径统一（SOP §2/§3）：净胜一律用翻转补验后的确认净胜，
    固定轮另外保留初测识别数严格改善的条件。
    """
    max_failure = float(protocol.get("max_failure_rate", 0.01))
    verdict: dict[str, Any] = {"dataset": dataset, "metrics": metrics}

    # 失败率门槛优先：数据不完整时任何结论都无效
    if metrics.get("failure_rate", 0.0) >= max_failure:
        verdict["verdict"] = "experiment_incomplete"
        verdict["reason"] = (
            f"失败率 {metrics['failure_rate']:.2%} ≥ 上限 {max_failure:.0%}"
            f"（失败 {metrics['failures']}/{metrics['attempted']} 题）；"
            "按 SOP 整轮结果不可用，修复后从检查点重跑"
        )
        return verdict

    if dataset == "development":
        net = metrics["net_win_confirmed"]
        # 旧实验按冻结门槛重现；新协议明确区分开发晋级与固定轮准入。
        schema = protocol.get("gate_schema", 1)
        if schema >= 3:
            dev_need = max(1, ceil(protocol["dev_min_net_win_rate"] * metrics["pairs"]))
        elif schema == 2:
            dev_need = 1
        else:
            dev_need = max(1, round(protocol["dev_min_net_win_rate"] * metrics["pairs"]))
        if net >= dev_need:
            verdict["verdict"] = "merge_to_iteration_baseline"
            verdict["reason"] = (
                f"翻转验证后净胜 {net}/{metrics['pairs']}，达到门槛 +{dev_need}，并入开发基线"
            )
        elif metrics["losses_confirmed"] > metrics["wins_confirmed"]:
            verdict["verdict"] = "reject"
            verdict["reason"] = (
                f"Candidate 净负 "
                f"{metrics['wins_confirmed']}-{metrics['losses_confirmed']}，淘汰"
            )
        else:
            verdict["verdict"] = "observe"
            verdict["reason"] = "未达到并入门槛且未净负，继续观察"
    else:  # fixed_test：双条件（SOP §3），正式失败生产指针不动
        # 生成器：识别数下降=更像人；Judge：识别数上升=更强（SOP §4 方向镜像）
        improved = (metrics["identified_candidate"] > metrics["identified_baseline"]) if higher_is_better \
            else (metrics["identified_candidate"] < metrics["identified_baseline"])
        formal_need = max(1, ceil(protocol["formal_min_net_win_rate"] * metrics["pairs"]))
        enough = metrics["net_win_confirmed"] >= formal_need
        if improved and enough:
            verdict["verdict"] = "adopt"
            verdict["reason"] = (
                f"识别数 {metrics['identified_baseline']}→{metrics['identified_candidate']} "
                f"严格{'上升' if higher_is_better else '下降'}，且确认净胜 {metrics['net_win_confirmed']:+d} ≥ "
                f"+{formal_need}（{protocol['formal_min_net_win_rate'] * 100:g}%）；生产指针可升级"
            )
        else:
            verdict["verdict"] = "reject"
            missing = []
            if not improved:
                missing.append(
                    f"识别数未严格{'上升' if higher_is_better else '下降'}（{metrics['identified_baseline']}→{metrics['identified_candidate']}）"
                )
            if not enough:
                missing.append(
                    f"确认净胜 {metrics['net_win_confirmed']:+d} < +{formal_need}"
                )
            verdict["reason"] = "；".join(missing) + "；生产指针保持原版本"
    return verdict


def summarize_final_records(cases_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """每题取最终记录 + 重试次数——生成器/Judge/渲染三方唯一入口（收敛项）。"""
    from .journal import records
    final: dict[str, dict[str, Any]] = {}
    rows = records(cases_path)
    for row in rows:
        final[str(row['case_id'])] = row
    return final, max(0, len(rows) - len(final))
