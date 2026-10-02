#!/usr/bin/env python3
"""Prepare, resume or inspect an account-scoped legacy Wiki repair batch."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import wiki_batch, wiki_continuation, wiki_delivery, wiki_supplement  # noqa: E402
from src.config import load_settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--wiki', required=True, type=Path)
    prepare.add_argument('--messages', required=True, type=Path)
    prepare.add_argument('--self-account', required=True)
    prepare.add_argument('--llm-config', required=True, type=Path)
    prepare.add_argument('--batch-chars', type=int, default=120000)
    prepare.add_argument('--reuse', action='append', default=[], type=Path)
    supplement = commands.add_parser('prepare-supplement')
    supplement.add_argument('--parent', required=True, type=Path)
    continuation = commands.add_parser('prepare-continuation', help='保留清单并复用校验改进前的有效产物与缓存')
    continuation.add_argument('--source', required=True, type=Path)
    continuation.add_argument('--parent', type=Path, help='补充批次对应的新人物批次')
    revision = commands.add_parser('prepare-revision', help='保留原批次，新增全记录回源语义修订')
    revision.add_argument('--source', required=True, type=Path)
    revision.add_argument('--engine', choices=['legacy', 'roles_v1', 'roles_claims_v2'], default='legacy')
    members = commands.add_parser('prepare-members', help='由原始群成员表和聊天 API 导出补建人物页')
    members.add_argument('--members', required=True, type=Path)
    members.add_argument('--page', action='append', required=True, type=Path)
    members.add_argument('--self-account', required=True)
    members.add_argument('--llm-config', required=True, type=Path)
    members.add_argument('--non-friends-only', action='store_true')
    members.add_argument('--priority-account', action='append', default=[])
    members.add_argument('--batch-chars', type=int, default=120000)
    run = commands.add_parser('run')
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--object-workers', type=int, default=1, help='同时整理的聊天对象数')
    run.add_argument('--attempts', type=int, default=2)
    run.add_argument('--max-jobs', type=int)
    run.add_argument('--skip-incomplete', action='store_true', help='保留失败缺口，先推进未执行和中断的对象')
    revise = commands.add_parser('revise', help='修订已完成内容；复用有效记录和修订缓存')
    revise.add_argument('--workers', type=int, default=4)
    revise.add_argument('--object-workers', type=int, default=1, help='同时修订的聊天对象数')
    revise.add_argument('--attempts', type=int, default=2)
    revise.add_argument('--max-jobs', type=int)
    revise.add_argument('--follow', action='store_true')
    revise.add_argument('--skip-incomplete', action='store_true', help='保留失败缺口，先推进未执行和中断的对象')
    status = commands.add_parser('status')
    catalog = commands.add_parser('catalog', help='核对已完成产物并生成可浏览目录')
    catalog.add_argument('--batch', action='append', required=True, type=Path)
    catalog.add_argument('--follow', action='store_true', help='随正在运行的批次自动更新目录')
    for command in (prepare, supplement, continuation, revision, revise, members, run, status, catalog):
        command.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not any(output.is_relative_to(ROOT / part) for part in ('instances', 'private')):
        parser.error('private Wiki output must stay in instances/ or private/')
    load_settings()
    if args.command in ('prepare', 'prepare-members'):
        if args.command == 'prepare' and output.is_relative_to(args.wiki.resolve()):
            parser.error('write outside original Wiki library')
        saved = json.loads(args.llm_config.read_text())
        llm = (saved.get('feature_llm') or saved.get('llm') or saved.get('config')
               or saved.get('extraction', {}).get('config') or saved)
        config = {k: llm[k] for k in ('provider', 'model', 'reasoning_effort', 'timeout_seconds',
                                     'codex_cli_version')}
        if args.command == 'prepare-members':
            from src.bootstrap import wiki_members
            value = wiki_members.prepare(args.members, args.page, output, args.self_account, config,
                non_friends_only=args.non_friends_only, priority_accounts=args.priority_account,
                max_chars=args.batch_chars)
        else:
            value = wiki_batch.prepare(args.wiki, args.messages, output, args.self_account, config,
                                      max_chars=args.batch_chars, reuse=args.reuse)
    elif args.command == 'prepare-supplement':
        value = wiki_supplement.prepare(args.parent, output)
    elif args.command == 'prepare-continuation':
        value = wiki_continuation.prepare(args.source, output, parent=args.parent)
    elif args.command == 'prepare-revision':
        from src.bootstrap import wiki_revision_batch
        value = wiki_revision_batch.prepare(args.source, output, engine=args.engine)
    elif args.command == 'revise':
        from src.bootstrap import wiki_revision_batch
        value = wiki_revision_batch.run(output, workers=args.workers, attempts=args.attempts,
                                       object_workers=args.object_workers,
                                       max_jobs=args.max_jobs, follow=args.follow,
                                       skip_incomplete=args.skip_incomplete)
        value = {k: v for k, v in value.items() if k != 'jobs'}
    elif args.command == 'status':
        value = wiki_batch.status(output)
    elif args.command == 'catalog':
        value = wiki_delivery.deliver(args.batch, output, follow=args.follow)
        value = dict(stage=value['stage'], **value['summary'])
    else:
        manifest = json.loads((output / 'manifest.json').read_text())
        backend = wiki_supplement if manifest['schema'] == 'wiki_supplement_v1' else wiki_batch
        value = backend.run(output, workers=args.workers, object_workers=args.object_workers,
                            attempts=args.attempts, max_jobs=args.max_jobs,
                            skip_incomplete=args.skip_incomplete)
        value = {k: v for k, v in value.items() if k != 'jobs'}
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return 0 if args.command == 'status' or value.get('stage', 'complete') == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
