#!/usr/bin/env python3
"""测量库回填：扫描实例已完成实验，把 cases.jsonl 解析为按侧测量写入库。

幂等：已入库的实验自动跳过（按行比对）；与运行时写入共用 measurements 公共库。
用法：
  python scripts/backfill_measurements.py --instance example-agent [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.iteration import measurements, versions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--dry-run', action='store_true', help='只解析统计，不写库')
    parser.add_argument('--fix-traces', action='store_true',
                        help='把 cases 引用但本实验缺失、存在于其他实验的调用实录随迁归位')
    args = parser.parse_args()

    versions.switch_instance(args.instance)
    instance_dir = versions.PRIVATE
    exp_root = instance_dir / 'experiments'
    if not exp_root.is_dir():
        raise SystemExit(f'实例不存在: {instance_dir}')

    db = None if args.dry_run else measurements.connect(instance_dir)
    total_new = total_seen = skipped = failed = 0
    for exp_dir in sorted(exp_root.iterdir()):
        if not (exp_dir / 'spec.json').is_file():
            continue
        state_p = exp_dir / 'state.json'
        if not state_p.is_file() or json.loads(state_p.read_text()).get('status') != 'finished':
            skipped += 1  # 未完成实验的测量不入库（source_status 语义）
            continue
        try:
            seen = len(list(measurements.iter_side_measurements(exp_dir)))
            total_seen += seen
            if args.dry_run:
                print(f'[dry] {exp_dir.name}: {seen} 条')
                continue
            new = measurements.record_experiment(exp_dir, db)
            if new == 0 and seen:
                skipped += 1
            total_new += new
            print(f'{exp_dir.name}: 新增 {new} / 解析 {seen}')
        except Exception as exc:  # noqa: BLE001 - 单实验失败不阻断整体回填
            failed += 1
            print(f'{exp_dir.name}: 失败 {type(exc).__name__}: {exc}', file=sys.stderr)
    if db:
        db.close()
    if args.fix_traces:
        print(f'trace 归位: 随迁 {fix_traces(instance_dir)} 个实录文件')
    print(f'\n合计: 解析 {total_seen} 条, 新写入 {total_new}, 已存在跳过 {skipped} 实验, 失败 {failed}')


def fix_traces(instance_dir: Path) -> int:
    """重放/复用产生的跨实验 trace 引用随迁：保持实验是自足审计单元。"""
    exps = instance_dir / 'experiments'
    index: dict[str, str] = {}
    for d in sorted(exps.iterdir()):
        tdir = d / 'traces'
        if tdir.is_dir():
            for f in tdir.glob('*.json'):
                index.setdefault(f.stem, d.name)
    copied = 0
    for d in sorted(exps.iterdir()):
        cases = d / 'cases.jsonl'
        if not cases.is_file():
            continue
        tdir = d / 'traces'
        tdir.mkdir(exist_ok=True)
        for line in cases.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            ref = json.loads(line).get('trace_ref')
            if not ref or Path(str(ref)).name != str(ref):
                continue
            dst = tdir / f'{ref}.json'
            src_exp = index.get(ref)
            if not dst.exists() and src_exp and src_exp != d.name:
                src = exps / src_exp / 'traces' / f'{ref}.json'
                if src.is_file():
                    dst.write_bytes(src.read_bytes())
                    copied += 1
    return copied


if __name__ == '__main__':
    main()
