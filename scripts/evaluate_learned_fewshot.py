#!/usr/bin/env python3
"""Evaluate a learned selector against a development or production generator baseline."""
import argparse
import json
import os
from pathlib import Path
import sys

# XGBoost model deserialization uses OpenMP before its saved nthread setting
# takes effect. Keep local numeric work bounded; request workers remain parallel.
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.iteration import learned_gen, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'start', 'status'])
    parser.add_argument('--instance', required=True)
    parser.add_argument('--output', type=Path, required=True)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--model', type=Path)
    selector.add_argument('--candidate', help='Reuse an immutable learned generator version')
    parser.add_argument('--base')
    parser.add_argument('--source-branch')
    parser.add_argument('--name')
    parser.add_argument('--data')
    parser.add_argument('--stage', choices=['development', 'fixed_test'], default='development',
                        help='Advance through this stage using the existing promotion gates')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--feature-workers', type=int,
                        help='Feature extraction concurrency, bounded by the instance resource policy')
    parser.add_argument('--timeout', type=float, default=360)
    args = parser.parse_args()
    versions.switch_instance(args.instance)
    if args.command == 'status':
        result = learned_gen.status(args.output)
    else:
        if not all((args.base, args.name, args.data)) or not (args.model or args.candidate):
            parser.error('run/start requires base, name, data and either model or candidate')
        if (args.model and not args.source_branch) or (args.candidate and args.source_branch):
            parser.error('source-branch is required only with model; candidate compares against production')
        if not 1 <= args.workers <= 16 or not 0 < args.timeout <= 3600:
            parser.error('workers must be within 1–16; timeout must be within (0, 3600]')
        if args.command == 'start':
            result = learned_gen.launch(args.output, ['run', *sys.argv[2:]])
        else:
            options = vars(args).copy()
            for key in ('command', 'instance', 'output'):
                options.pop(key)
            result = learned_gen.run(args.output, **options)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
