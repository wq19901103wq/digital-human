#!/usr/bin/env python3
"""Freeze, inspect and independently bind shared Wiki versions to Gen/Judge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.iteration import shares, versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    actions = parser.add_subparsers(dest='action', required=True)
    freeze = actions.add_parser('freeze')
    freeze.add_argument('--source', type=Path, required=True)
    actions.add_parser('list')
    show = actions.add_parser('show')
    show.add_argument('ref')
    bind = actions.add_parser('bind')
    bind.add_argument('--kind', choices=['gen', 'judge'], required=True)
    bind.add_argument('--model', required=True)
    bind.add_argument('--share', required=True)
    configure = actions.add_parser('configure-gen', help='冻结按账号 Wiki 和按需 search_memory 生成器候选')
    configure.add_argument('--model', required=True)
    configure.add_argument('--share', required=True)
    configure.add_argument('--memory-index', type=Path, required=True)
    configure.add_argument('--memory-model', type=Path, required=True)
    args = parser.parse_args(argv)
    versions.switch_instance(args.instance)
    if args.action == 'freeze':
        ref = shares.create(args.source)
        result = shares.load(ref)['manifest']
    elif args.action == 'show':
        result = shares.load(args.ref)['manifest']
    elif args.action == 'bind':
        result = {'kind': args.kind, 'version': shares.bind_model(args.kind, args.model, args.share),
                  'share_ref': args.share, 'pointers_changed': False, 'runtime_usable': False}
    elif args.action == 'configure-gen':
        from src.generator.background import configure
        result = {'kind': 'gen', 'version': configure(args.model, args.share,
                  args.memory_index, args.memory_model), 'share_ref': args.share,
                  'pointers_changed': False, 'runtime_policy': 'account_wiki_memory_v1'}
    else:
        result = shares.inventory()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
