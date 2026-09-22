#!/usr/bin/env python3
"""Audit/build global history and purpose manifests without adopting or training."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import conversations, history, ingest, import_coverage  # noqa: E402
from src.config import ConfigError, sha256_file  # noqa: E402
from src.iteration import versions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--policy', type=Path, help='实例数据策略，默认 instances/<instance>/data_policy.json')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--exports', type=Path)
    source.add_argument('--source-data', help='复用已有版本的冻结原始消息，仅重新切分')
    source.add_argument('--reimport-data', help='从旧版清单指定且哈希一致的原始导出补全消息类型')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--development-start', type=int)
    parser.add_argument('--acceptance-start', type=int)
    parser.add_argument('--max-context', type=int, default=8, help='每题上文窗口（条）')
    parser.add_argument('--context-gap-seconds', type=int, default=7200)
    parser.add_argument('--response-gap-seconds', type=int, default=600)
    parser.add_argument('--reply-gap-seconds', type=int, default=120)
    args = parser.parse_args()
    versions.switch_instance(args.instance)
    if args.source_data:
        archive = versions.data_version_dir(args.source_data) / 'messages.jsonl'
        source_paths = [archive]
    elif args.reimport_data:
        source_paths = [Path(p) for p in import_coverage.frozen_exports(versions.data_version_dir(args.reimport_data))]
    else:
        source_paths = sorted(args.exports.rglob('*.json'))
    source_files = {str(p.resolve()): sha256_file(p) for p in source_paths}
    import_statistics = {}
    messages = (ingest.load_unified(archive) if args.source_data else
                ingest.load_weflow_files(source_paths, statistics=import_statistics))
    segmentation = conversations.policy(context_gap_seconds=args.context_gap_seconds,
        response_gap_seconds=args.response_gap_seconds, reply_gap_seconds=args.reply_gap_seconds)
    statistics = {}
    rows = history.examples(messages, max_context=args.max_context, segmentation=segmentation, statistics=statistics)
    print(json.dumps({'source_messages': len(messages), 'examples': len(rows),
                      'reply_segmentation': segmentation, 'construction_counts': statistics,
                      'types': dict(Counter(r['relationship'] for r in rows))}, ensure_ascii=False), flush=True)
    policy_path = args.policy or versions.PRIVATE / 'data_policy.json'
    policy = {k: v for k, v in json.loads(policy_path.read_text()).items() if not isinstance(v, str)}
    for name in ('development_start', 'acceptance_start'):
        if getattr(args, name) is not None:
            policy[name] = getattr(args, name)
    roles, protocol = history.plan(rows, segmentation=segmentation, **policy)
    protocol['construction_counts'] = statistics
    protocol['max_context'] = args.max_context  # 构建参数入冻结协议，可溯源
    if args.source_data:
        protocol['source_data_ref'] = args.source_data
    if args.reimport_data:
        protocol['reimport_data_ref'] = args.reimport_data
    if import_statistics:
        protocol['message_ingestion'] = import_statistics
    print(json.dumps({'protocol': {k: v for k, v in protocol.items() if k != 'unseen_chat_ids'},
                      'roles': {k: len(v) for k, v in roles.items()}}, ensure_ascii=False), flush=True)
    current_paths = source_paths if (args.source_data or args.reimport_data) else sorted(args.exports.rglob('*.json'))
    if source_files != {str(p.resolve()): sha256_file(p) for p in current_paths}:
        raise ConfigError('构建期间来源文件变化')
    if args.build:
        print('Created', history.publish(messages, rows, roles, protocol, source_files), flush=True)


if __name__ == '__main__':
    main()
