#!/usr/bin/env python3
"""P0 统计验证（设计 §9 P0）：量化配对同测相关性与 judge 抽样噪声。

方法学（实现评审修订）：初测判定被请求缓存去重、完全确定，不能反映 judge
噪声；**补验轮是独立新抽样**，本脚本全部基于 flip_verified 的逐票数据：

  (a) 同实验两侧补验票的逐题相关 ρ（配对同测的正相关证据）；
  (b) 方差膨胀：bootstrap 估计——同实验配对票组合的净胜率方差 vs
      跨实验单侧拼票组合的净胜率方差（归一化为净胜率，评审 P2-8）；
  (c) judge 抽样噪声：同一 (实验,题,侧) 的多轮补验票自身的一致性
      （相邻轮判定翻转率）。

分组一律用指纹（借 measurements 公共库），不用裸 ref（评审 P2-7）。
输出 JSON 到 instances/<名>/analysis/measure-stats-<日期>.json。
用法: python scripts/measure_stats.py --instance example-agent
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.iteration import measurements, versions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    args = parser.parse_args()
    versions.switch_instance(args.instance)
    instance_dir = versions.PRIVATE

    # 解析：复用 measurements 公共库（含 flip 首票一致性过滤与指纹），
    # 实验 -> pair_fp -> {case -> {side -> [verify votes]}}
    exp_rows = defaultdict(lambda: defaultdict(dict))
    exp_pair = {}
    exp_root = instance_dir / 'experiments'
    for exp_dir in sorted(exp_root.iterdir()):
        if not (exp_dir / 'cases.jsonl').is_file():
            continue
        try:
            rows = list(measurements.iter_side_measurements(exp_dir))
        except Exception:
            continue
        if not rows:
            continue
        pair_fp = rows[0]['pair_fp']
        exp_pair[exp_dir.name] = pair_fp
        votes = defaultdict(dict)
        for r in rows:
            if r['round_kind'] != 'verify':
                continue
            votes[r['case_id']].setdefault(r['side_ref'], []).append(int(r['verdict']))
        for case, sides in votes.items():
            if all(len(v) >= 2 for v in sides.values()):
                exp_rows[exp_dir.name][case] = sides

    # 指纹分组（同 pair_fp 才能拼/bootstrap）
    groups = defaultdict(list)
    for exp_name, pair_fp in exp_pair.items():
        if exp_rows.get(exp_name):
            groups[pair_fp].append(exp_name)

    # (a) 实验内两侧补验票相关（按题：两侧多数判定是否一致 → phi 过于稀疏，
    # 改用“两侧各轮判定向量”的逐题一致率与列联相关）
    agree_rates, phis = [], []
    for exp_name, cases in exp_rows.items():
        both = {c: s for c, s in cases.items() if len(s) == 2}
        if len(both) < 30:
            continue
        a, b = [], []
        for sides in both.values():
            vals = list(sides.values())
            a.append(sum(vals[0]) > len(vals[0]) / 2)
            b.append(sum(vals[1]) > len(vals[1]) / 2)
        agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        agree_rates.append({'experiment': exp_name, 'n': len(a), 'majority_agree': round(agree, 4)})
        phis.append(_phi([int(x) for x in a], [int(x) for x in b]))
    phis = [p for p in phis if p is not None]

    # (b) 净胜率方差：同实验配对 vs bootstrap 跨实验拼侧
    rng = random.Random(42)
    within_rates, cross_rates = [], []
    for members in groups.values():
        for exp_name in members:
            rates = _net_rates(exp_rows[exp_name])
            if rates:
                within_rates.append(rates['paired'])
        if len(members) >= 2:
            for _ in range(200):
                ex = rng.choice(members)
                ey = rng.choice(members)
                if ex == ey:
                    continue
                cx = exp_rows[ex]
                cy = exp_rows[ey]
                merged = {}
                for case in set(cx) | set(cy):
                    merged[case] = {}
                    if case in cx and 'baseline' in cx[case]:
                        merged[case]['baseline'] = cx[case]['baseline']
                    if case in cy and 'candidate' in cy[case]:
                        merged[case]['candidate'] = cy[case]['candidate']
                rates = _net_rates(merged)
                if rates:
                    cross_rates.append(rates['paired'])
    var_within, var_cross = _var(within_rates), _var(cross_rates)
    inflation = (var_cross / var_within) if (var_within and var_cross is not None) else None

    # (c) judge 抽样噪声：同 (exp,case,side) 多轮补验票的翻转率
    flips = total_pairs = 0
    for cases in exp_rows.values():
        for sides in cases.values():
            for votes in sides.values():
                for x, y in zip(votes, votes[1:]):
                    total_pairs += 1
                    flips += int(bool(x) != bool(y))
    flip_rate = (flips / total_pairs) if total_pairs else None

    report = {
        'instance': args.instance,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'methodology': 'flip-vote based（初测受请求缓存去重影响，不反映抽样噪声）',
        'experiments_with_verified_rounds': len(exp_rows),
        'within_experiment': {
            'majority_agree_mean': round(sum(a['majority_agree'] for a in agree_rates) / len(agree_rates), 4) if agree_rates else None,
            'phi_mean': round(sum(phis) / len(phis), 4) if phis else None,
            'by_experiment': agree_rates[:40],
            'paired_net_win_rate_variance': var_within,
            'samples': len(within_rates),
        },
        'cross_experiment_shuffle': {
            'paired_net_win_rate_variance': var_cross,
            'samples': len(cross_rates),
            'variance_inflation_ratio': round(inflation, 3) if inflation else None,
        },
        'judge_sampling_noise': {
            'adjacent_verify_vote_flip_rate': round(flip_rate, 4) if flip_rate else None,
            'vote_pairs': total_pairs,
        },
        'pass_criterion': '膨胀 ≤ 1.10 且补验票翻转率有上界（如 ≤0.35）方可放行单侧跨实验补票',
    }
    out_dir = instance_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'measure-stats-{time.strftime("%Y%m%d")}.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('within_experiment',)},
                     ensure_ascii=False, indent=2))
    print(f'\n一致率={report["within_experiment"]["majority_agree_mean"]} '
          f'ρ={report["within_experiment"]["phi_mean"]} '
          f'膨胀={report["cross_experiment_shuffle"]["variance_inflation_ratio"]} '
          f'翻转率={report["judge_sampling_noise"]["adjacent_verify_vote_flip_rate"]}')
    print(f'报告: {out}')


def _net_rates(cases: dict) -> dict | None:
    """逐题比较两侧补验多数判定 → 净胜率（胜−负）/分歧题数。"""
    wins = losses = 0
    for sides in cases.values():
        if len(sides) != 2:
            continue
        vals = list(sides.values())
        b = sum(vals[0]) > len(vals[0]) / 2
        c = sum(vals[1]) > len(vals[1]) / 2
        if b == c:
            continue
        if b:
            wins += 1
        else:
            losses += 1
    n = wins + losses
    return {'paired': (wins - losses) / n} if n >= 10 else None


def _phi(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    na, nb = sum(a), sum(b)
    if n < 30 or na in (0, n) or nb in (0, n):
        return None
    both = sum(1 for x, y in zip(a, b) if x and y)
    cov = both / n - (na / n) * (nb / n)
    var = ((na / n) * (1 - na / n) * (nb / n) * (1 - nb / n)) ** 0.5
    return cov / var if var else None


def _var(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    return round(sum((v - mean) ** 2 for v in values) / (n - 1), 5)


if __name__ == '__main__':
    main()
