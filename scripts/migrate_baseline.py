#!/usr/bin/env python3
"""Prepare/apply a source-verified baseline migration, separate from promotion."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.iteration import baseline, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='核验并保存迁移凭证，不切换基线')
    for name in ('data', 'generator', 'judge', 'reason'):
        prepare.add_argument('--' + name, required=True)
    data = sub.add_parser('prepare-data', help='复用完整数据验收报告；仅迁移数据，不验收或改变模型')
    for name in ('data', 'report', 'reason'):
        data.add_argument('--' + name, required=True)
    apply = sub.add_parser('apply', help='重新核验凭证后原子切换整组基线；不声称性能提升')
    apply.add_argument('--receipt', required=True)
    args = parser.parse_args()
    versions.switch_instance(args.instance)
    if args.command == 'prepare':
        result = str(baseline.prepare(args.data, args.generator, args.judge, reason=args.reason))
    elif args.command == 'prepare-data':
        result = str(baseline.prepare_data(args.data, args.report, reason=args.reason))
    else:
        result = baseline.apply(args.receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
