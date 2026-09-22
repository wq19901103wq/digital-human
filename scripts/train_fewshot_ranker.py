#!/usr/bin/env python3
"""Extract independent cached features, train a few-shot ranker, and report held-out AUC."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generator.fewshot_ranker.pipeline import execute, status
from src.iteration.storage import file_lock, locked, write_json


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status', 'compare-lr', 'compare-id', 'compare-features', 'compare-models', 'compare-grades', 'compare-actions'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--source', type=Path, help='Completed training output; reuse its features and labels')
    parser.add_argument('--reference', type=Path, help='Saved controls: compare-lr output for features, compare-features output for models')
    parser.add_argument('--stability', type=Path, help='Completed two-draw positive regenerate retests for compare-grades')
    parser.add_argument('--feature-config', type=Path)
    parser.add_argument('--feature-cache', type=Path)
    parser.add_argument('--instance')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'compare-grades':
        if not args.source or not args.reference or not args.stability or args.background:
            parser.error('compare-grades requires --source, --reference, --stability; runs locally in foreground')
        from src.generator.fewshot_ranker.graded_comparison import run
        result = run(args.source, args.reference, args.stability, args.output)
        print(json.dumps(dict(status=result['status'], output=str(args.output), seconds=result['seconds'],
                              new_model_requests=result['new_model_requests']), indent=2))
        return
    if args.command in ('compare-id', 'compare-features', 'compare-models', 'compare-actions'):
        if not args.source or not args.reference or args.background:
            parser.error('feature comparison requires --source and --reference and runs locally in the foreground')
        if args.command == 'compare-actions':
            from src.generator.fewshot_ranker.action_comparison import run
        elif args.command == 'compare-models':
            from src.generator.fewshot_ranker.architecture_comparison import run
        else:
            from src.generator.fewshot_ranker.identity_comparison import run
        result = run(args.source, args.reference, args.output)
        print(json.dumps(dict(status=result['status'], output=str(args.output), seconds=result['seconds'],
                              new_model_requests=result['new_model_requests']), ensure_ascii=False, indent=2))
        return
    if args.command == 'compare-lr':
        if not args.source or args.background:
            parser.error('compare-lr requires --source and runs locally in the foreground')
        from src.generator.fewshot_ranker.comparison import run
        result = run(args.source, args.output)
        print(json.dumps(dict(status=result['status'], output=str(args.output), seconds=result['seconds'],
            validation={name: {k: v for k, v in item['validation'].items() if k != 'contexts'}
                        for name, item in result['methods'].items()}), ensure_ascii=False, indent=2))
        return
    if args.command == 'status':
        print(json.dumps(status(args.output), ensure_ascii=False, indent=2))
        return
    if not all((args.inventory, args.feature_config, args.feature_cache, args.instance)):
        parser.error('run requires --inventory, --feature-config, --feature-cache and --instance')
    if not 1 <= args.workers <= 16 or args.passes < 1:
        parser.error('workers must be 1–16; passes must be positive')
    if args.background:
        with file_lock(args.output / '.launch.lock', blocking=False):
            if locked(args.output / '.run.lock'):
                parser.error('this run is already active')
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       *[v for v in argv if v != '--background']]
            with (args.output / 'run.log').open('a') as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            write_json(args.output / 'process.json', dict(pid=process.pid, launched_at=time.time(), command=command))
            for _ in range(50):
                if locked(args.output / '.run.lock') or process.poll() is not None:
                    break
                time.sleep(.1)
            print(json.dumps(dict(pid=process.pid, output=str(args.output),
                                  running=process.poll() is None, exit_code=process.poll()), ensure_ascii=False))
        return
    options = vars(args)
    options.pop('command')
    options.pop('background')
    options.pop('source')
    options.pop('reference')
    options.pop('stability')
    print(json.dumps(execute(**options), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
