#!/usr/bin/env python3
"""Prepare private alias inventories or source-backed Wiki pages without changing source Wikis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap.wiki_aliases import build_review, write_review  # noqa: E402
from src.bootstrap.wiki_alias_review import refine_review, write_refined  # noqa: E402
from src.bootstrap.wiki_content import compile_content, write_content  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--wiki-dir', type=Path)
    mode.add_argument('--refine-from', type=Path, help='Reuse an existing review.json and supplement contextual names')
    mode.add_argument('--content-plan', type=Path, help='Render curated claims with source locators, without copied evidence text')
    mode.add_argument('--library', type=Path, help='Organize all users/groups/topics Wikis; resumable cached extraction')
    parser.add_argument('--identity-review', type=Path, help='Existing original review with account inventory')
    parser.add_argument('--curated', type=Path, help='Existing curated knowledge.json; preserve verified corrections')
    parser.add_argument('--llm-config', type=Path, help='Reuse feature_llm/llm from an existing model config')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--max-batches', type=int, help='Bound work per invocation; successful batches are resumed')
    parser.add_argument('--reuse-from', action='append', type=Path, default=[], help='Reuse exact extraction units from a previous library job; repeatable')
    parser.add_argument('--page', action='append', default=[], help='With --library, select an exact relative path such as users/name.md; repeatable')
    parser.add_argument('--review', type=Path, help='Existing refined review.json for --content-plan')
    parser.add_argument('--decisions', type=Path, help='Preserve existing explicit user decisions when refining')
    parser.add_argument('--aliases', type=Path)
    parser.add_argument('--messages', required=True, type=Path)
    parser.add_argument('--self-account', help='Source account whose is_self represents the subject; never merge self roles across exports')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--focus', action='append', default=[], help='Include bounded raw-chat excerpts for this term')
    args = parser.parse_args()
    output = args.output.resolve()
    if not any(output.is_relative_to(ROOT / private) for private in ('instances', 'private')):
        parser.error('包含私有聊天内容的草案只能写到 instances/ 或 private/ 下')
    if output.exists() and not args.library:
        parser.error('输出目录已存在；使用新目录以免覆盖用户审阅')
    if args.wiki_dir and output.is_relative_to(args.wiki_dir.resolve()):
        parser.error('草案不能写入原 Wiki')
    if args.library:
        if not all((args.identity_review, args.review, args.curated, args.llm_config)):
            parser.error('--library requires --identity-review, --review, --curated, --llm-config')
        if output.is_relative_to(args.library.resolve()):
            parser.error('整理结果不能写入原 Wiki')
        from src.bootstrap.wiki_library_extract import organize_library
        from src.config import load_settings
        load_settings()
        config = json.loads(args.llm_config.read_text())
        summary = organize_library(args.library, output, args.identity_review, args.review, args.curated,
                                   config.get('feature_llm') or config['llm'], workers=args.workers,
                                   inventory_only=args.inventory_only, max_batches=args.max_batches,
                                   pages=args.page, reuse_from=args.reuse_from)
        print(json.dumps(dict(output=str(output), **summary), ensure_ascii=False))
        return
    if args.content_plan:
        if not args.review:
            parser.error('--content-plan requires --review')
        review = compile_content(args.content_plan, args.messages, args.review)
        write_content(review, output)
    elif args.refine_from:
        decision_path = args.decisions or args.refine_from.parent / 'decisions.json'
        review = refine_review(args.refine_from, args.messages, decision_path if decision_path.exists() else None,
                               self_account=args.self_account)
        write_refined(review, output)
    else:
        review = build_review(args.wiki_dir, args.aliases, args.messages, args.focus)
        write_review(review, output)
    print(json.dumps(dict(output=str(output), **review['summary']), ensure_ascii=False))


if __name__ == '__main__':
    main()
