#!/usr/bin/env python3
"""Correct one legacy person Wiki using only its text and raw account history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap.wiki_repair import repair  # noqa: E402
from src.config import load_settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wiki', required=True, type=Path)
    parser.add_argument('--messages', required=True, type=Path)
    parser.add_argument('--account', required=True)
    parser.add_argument('--self-account', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--llm-config', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-chars', type=int, default=120000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--cache-from', type=Path, help='Reuse exact-input successful requests from an earlier run')
    parser.add_argument('--structured', action='store_true', help='Reread fact sources into atomic records and XML')
    parser.add_argument('--include-mentions', action='store_true',
                        help='Also reread private/group windows mentioning legacy names or account display names')
    parser.add_argument('--reconcile-attributes', action='store_true',
                        help='Reconcile repeated subject/attribute records against their combined raw evidence')
    args = parser.parse_args()
    output = args.output.resolve()
    if not any(output.is_relative_to(ROOT / folder) for folder in ('instances', 'private')):
        parser.error('private Wiki output must stay in instances/ or private/')
    if output.is_relative_to(args.wiki.resolve().parent):
        parser.error('write a new artifact outside the original Wiki directory')
    load_settings()
    saved = json.loads(args.llm_config.read_text())
    llm = saved.get('feature_llm') or saved.get('llm') or saved.get('extraction', {}).get('config') or saved
    # Whitelist transport settings; model configs must not smuggle personas or prior reviews.
    config = {key: llm[key] for key in ('provider', 'model', 'reasoning_effort', 'timeout_seconds',
                                      'codex_cli_version')}
    result = repair(args.wiki, args.messages, output, args.account, args.self_account, config,
                    workers=args.workers, max_chars=args.batch_chars, prepare_only=args.prepare_only,
                    cache_from=args.cache_from, structured=args.structured, include_mentions=args.include_mentions,
                    reconcile_attributes=args.reconcile_attributes)
    return 0 if result['stage'] in ('prepared', 'complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
