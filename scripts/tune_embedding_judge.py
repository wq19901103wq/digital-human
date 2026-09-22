#!/usr/bin/env python3
"""Reuse clean cached features for an entire resumable embedding DNN search."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Set before importing numerical libraries. PyTorch and XGBoost share this
# process during batch preparation; avoid competing OpenMP pools on macOS.
# Experiment parallelism is controlled separately by --workers.
os.environ.setdefault('OMP_NUM_THREADS', '1')


def main():
    from src.iteration import embedding_tuning, versions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--arm', default='luna')
    parser.add_argument('--pack', required=True)
    parser.add_argument('--saved-draws', required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    versions.switch_instance(args.instance)
    result = embedding_tuning.run(args.study, args.source, args.arm, args.pack,
                                  args.saved_draws, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
