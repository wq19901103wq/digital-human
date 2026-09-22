#!/usr/bin/env python3
"""Prepare/resume a source-bound single-shot ranker inventory without LLM calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generator.ranker_samples import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--generator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observations', type=int, help='Expand to this many distinct context/example observations')
    parser.add_argument('--reuse-inventory', type=Path, help='Preserve candidates and labels from an existing batch')
    parser.add_argument('--reuse-preparation', type=Path,
                        help='Reuse retrieval checkpoints from a stopped expansion with a different sample target')
    args = parser.parse_args()
    if args.observations is not None:
        if not args.reuse_inventory:
            parser.error('--observations requires --reuse-inventory')
        from src.generator.ranker_expansion import prepare as expand
        if args.reuse_preparation:
            from src.generator.ranker_preparation import seed_checkpoints
            seed_checkpoints(args.reuse_preparation, args.output,
                             observations=args.observations, reuse_inventory=args.reuse_inventory)
        result = expand(args.data, args.teacher, args.generator, args.output,
                        observations=args.observations, reuse_inventory=args.reuse_inventory)
    else:
        if args.reuse_inventory or args.reuse_preparation:
            parser.error('--reuse-inventory/--reuse-preparation requires --observations')
        result = prepare(args.data, args.teacher, args.generator, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
