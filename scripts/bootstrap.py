#!/usr/bin/env python3
"""Build a source-verified instance from explicit data and policy; no model calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import history, ingest  # noqa: E402
from src.config import ConfigError, sha256_file  # noqa: E402
from src.iteration import baseline, training, versions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--data', type=Path, required=True, help='normalized chat JSONL')
    parser.add_argument('--policy', type=Path, help='defaults to the instance data_policy.json')
    parser.add_argument('--model', required=True, help='explicit generator model')
    parser.add_argument('--judge-model', help='optional source-free pairwise judge model')
    parser.add_argument('--initialize', action='store_true', help='initialize an empty instance with an audited baseline receipt')
    args = parser.parse_args()
    versions.switch_instance(args.instance)
    if args.initialize and (versions.POINTERS_PATH.exists() or not args.judge_model):
        raise ConfigError('initialization requires an empty instance and an explicit judge model; use baseline migration for an existing instance')
    policy_path = args.policy or versions.PRIVATE / 'data_policy.json'
    policy = json.loads(policy_path.read_text())
    digest = sha256_file(args.data)
    messages = ingest.load_chat_export(args.data)
    examples = history.examples(messages)
    roles, protocol = history.plan(examples, **policy)
    if sha256_file(args.data) != digest:
        raise ConfigError('source changed while importing')
    data_ref = history.publish(messages, examples, roles, protocol, {str(args.data.resolve()): digest})
    gen_ref = training.mechanical_generator(data_ref, {'llm': {'model': args.model}, 'retriever': {'enabled': True}})
    result = {'data_ref': data_ref, 'generator_ref': gen_ref, 'baseline_changed': False}
    if args.judge_model:
        judge_ref = training.mechanical_judge({'model': args.judge_model})
        result['judge_ref'] = judge_ref
        if args.initialize:
            receipt = baseline.prepare(data_ref, gen_ref, judge_ref, reason='explicit source-verified instance initialization')
            baseline.apply(receipt.stem)
            result.update(baseline_changed=True, migration_receipt=receipt.stem)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
