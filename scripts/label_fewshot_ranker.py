#!/usr/bin/env python3
"""Prepare, label, or resume a frozen single-example ranker inventory."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import sha256_file
from src.generator.ranker_labels import LabelBatch
from src.iteration.storage import file_lock, write_json


def wait_for_inventory(inventory):
    """Queue behind candidate preparation without making model requests."""
    inventory = Path(inventory)
    while not all((inventory / name).exists() for name in ('report.json', 'inventory.json')):
        state = dict(status='waiting_for_inventory', inventory=str(inventory.resolve()), updated_at=time.time())
        write_json(inventory / 'labeling' / 'queue.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        time.sleep(30)


def execute(inventory, data, teacher, generator, instance, *, run=False, workers=16,
            passes=2, min_generation_timeout_seconds=None, reuse_inventory=None, wait_for_reuse=False,
            wait_for_preparation=False):
    """Keep transport waits separate from frozen model inputs and saved labels."""
    with file_lock(inventory / 'labeling' / '.run.lock', blocking=False):
        if wait_for_preparation:
            wait_for_inventory(inventory)
        batch = LabelBatch(inventory, data, teacher, generator, instance)
        if (inventory / 'expansion_plan.json').exists():
            from src.generator.ranker_expansion.reuse import reuse_source
            reuse_inventory = reuse_source(inventory, reuse_inventory)
        if reuse_inventory is not None:
            from src.generator.ranker_expansion.reuse import import_when_idle
            import_when_idle(batch, reuse_inventory, data, teacher, generator, instance, wait=wait_for_reuse)
        if not run:
            return batch.report()
        write_json(batch.output / 'queue.json', dict(status='ready', updated_at=time.time()))
        llm = batch.gen_cfg['llm']
        configs = {key: llm[key] for key in ('private', 'group') if key in llm}
        configs = configs or {'default': llm}
        original = {key: cfg.get('timeout_seconds', 60) for key, cfg in configs.items()}
        if min_generation_timeout_seconds is not None:
            for cfg in configs.values():
                cfg['timeout_seconds'] = max(float(cfg.get('timeout_seconds', 60)),
                                            min_generation_timeout_seconds)
        write_json(batch.output / 'transport.json', dict(
            original_generation_timeouts=original,
            effective_generation_timeouts={key: cfg.get('timeout_seconds', 60)
                                           for key, cfg in configs.items()},
            entrypoint_sha256=sha256_file(Path(__file__)),
            policy='transport_wait_only_request_body_and_cache_identity_unchanged'))
        return batch.run(workers, passes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--generator', type=Path, required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--run', action='store_true', help='Explicitly enable generation and labeling requests')
    parser.add_argument('--stability-output', type=Path, help='Repeat originally positive labels into a separate output')
    parser.add_argument('--stability-mode', choices=['regenerate', 'judge', 'both'], default='regenerate')
    parser.add_argument('--additional-draws', type=int, default=2)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--passes', type=int, default=2, help='Retry only unfinished observations on subsequent passes')
    parser.add_argument('--reuse-inventory', type=Path, help='Import condition-identical successes from an older batch')
    parser.add_argument('--wait-for-reuse', action='store_true', help='Queue until the source labeling runner is idle')
    parser.add_argument('--wait-for-preparation', action='store_true', help='Queue until the inventory is prepared')
    parser.add_argument('--min-generation-timeout-seconds', type=float,
                        help='Raise transport waits without changing frozen requests or redoing successful labels')
    args = parser.parse_args()
    if args.wait_for_preparation and not args.run:
        parser.error('--wait-for-preparation requires --run')
    if args.wait_for_reuse and (not args.reuse_inventory or not args.run):
        parser.error('--wait-for-reuse requires --reuse-inventory and --run')
    if args.min_generation_timeout_seconds is not None and not 0 < args.min_generation_timeout_seconds < float('inf'):
        parser.error('--min-generation-timeout-seconds must be finite and positive')
    options = vars(args)
    if args.stability_output:
        if args.additional_draws < 1 or any((args.reuse_inventory, args.wait_for_reuse, args.wait_for_preparation)):
            parser.error('stability requires positive additional draws and a completed source inventory')
        from src.generator.fewshot_ranker.stability import execute as repeat
        result = repeat(**options)
        result = {k: v for k, v in result.items() if k != 'methods'}
    else:
        for key in ('stability_output', 'stability_mode', 'additional_draws'):
            options.pop(key)
        result = execute(**options)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
