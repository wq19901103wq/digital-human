#!/usr/bin/env python3
"""Prepare, resume or inspect an account-scoped legacy Wiki repair batch."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.share.wiki import wiki_batch
from src.share.wiki import wiki_continuation
from src.share.wiki import wiki_delivery
from src.share.wiki import wiki_supplement# noqa: E402
from src.infrastructure.config import load_settings# noqa: E402


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
    continuation.add_argument('--reread-contract', choices=['reread_evidence_v2'],
                              help='显式升级回读的消息行号与旧稿行号契约，保留未变请求的缓存和失败预算')
    continuation.add_argument('--structured-contract', choices=['structured_source_v2'],
                              help='显式提供结构化作者锚点，复用通过校验的旧输出并保留原失败预算')
    continuation.add_argument('--reference-contract', choices=['bounded_refs_v2'],
                              help='显式按批约束来源及记录编号，复用合格旧请求且不重置失败预算')
    continuation.add_argument('--correction-contract', choices=['source_feedback_v2', 'source_constraints_v3'],
                              help='仅为已耗尽额度的拒绝输出启用集中纠错；有限新额度，旧记录不变')
    continuation.add_argument('--request-locations', type=Path,
                              help='复用离线定位凭据，精确选择已耗尽额度的拒绝请求，不重置已有纠错额度')
    continuation.add_argument('--completion-contract', choices=['page_completion_v1'],
                              help='仅续补选中页的后续阶段，要求已成功纠错且不重置旧额度')
    continuation.add_argument('--client-recovery-contract', choices=['client_recovery_v1'],
                              help='显式选最多五个客户端失败页；每页一次恢复，保留旧纠错预算和成功缓存')
    continuation.add_argument('--job-id', action='append', default=[])
    revision = commands.add_parser('prepare-revision', help='保留原批次，新增全记录回源语义修订')
    revision.add_argument('--source', required=True, type=Path)
    revision.add_argument('--engine', choices=['legacy', 'roles_v1', 'roles_claims_v2'], default='legacy')
    members = commands.add_parser('prepare-members', help='由原始群成员表和聊天 API 导出补建人物页')
    members.add_argument('--members', required=True, type=Path)
    members.add_argument('--page', action='append', required=True, type=Path)
    members.add_argument('--self-account', required=True)
    members.add_argument('--llm-config', required=True, type=Path)
    members.add_argument('--non-friends-only', action='store_true')
    members.add_argument('--account', action='append', default=[], help='仅准备指定原始成员账号，可重复')
    members.add_argument('--priority-account', action='append', default=[])
    members.add_argument('--batch-chars', type=int, default=120000)
    run = commands.add_parser('run')
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--object-workers', type=int, default=1, help='同时整理的聊天对象数')
    run.add_argument('--request-workers', type=int,
                     help='人物批次共享请求上限；各对象可借用空闲槽位，不改变模型或请求缓存')
    run.add_argument('--request-timeout-seconds', type=int,
                     help='仅覆盖人物批次本次运行的单客户端请求超时，不改冻结配置或成功缓存')
    run.add_argument('--profile-requests', action='store_true',
                     help='记录人物批次共享槽排队与客户端调用耗时，不记录提示词或原文')
    run.add_argument('--connection-retry-seconds', type=int,
                     help='明确授权后仅续补指定连接缺口；断流或客户端超时按此间隔自动重试原请求')
    run.add_argument('--cached-only', action='store_true',
                     help='仅用原流水线复核历史输出；不发模型请求，不改原失败预算，未变缺口不重复复核')
    run.add_argument('--attempts', type=int, default=2)
    run.add_argument('--max-jobs', type=int)
    run.add_argument('--job-id', action='append', help='人物批次只续补指定对象，可重复；未选缺口和原失败预算保留')
    retry = run.add_mutually_exclusive_group()
    retry.add_argument('--skip-incomplete', action='store_true', help='保留失败缺口，先推进未执行和中断的对象')
    retry.add_argument('--retry-transport-only', action='store_true',
                       help='人物批次仅续补有完整传输诊断的失败项，保留语义或未知失败缺口')
    run.add_argument('--retry-transport-signal', action='append', choices=sorted(wiki_batch.RETRY_TRANSPORT_SIGNALS),
                     help='与 --retry-transport-only 联用；仅补终端信号全部在已恢复列表中的对象，可重复')
    run.add_argument('--wait', action='store_true', help='等待当前批次结束后读取最新进度并续跑')
    revise = commands.add_parser('revise', help='修订已完成内容；复用有效记录和修订缓存')
    revise.add_argument('--workers', type=int, default=4)
    revise.add_argument('--object-workers', type=int, default=1, help='同时修订的聊天对象数')
    revise.add_argument('--attempts', type=int, default=2)
    revise.add_argument('--max-jobs', type=int)
    revise.add_argument('--follow', action='store_true')
    revise.add_argument('--skip-incomplete', action='store_true', help='保留失败缺口，先推进未执行和中断的对象')
    revise.add_argument('--job-id', action='append', help='只续跑指定对象，可重复；未选缺口保留，完成项仍复用')
    revise.add_argument('--connection-retry-seconds', type=int,
                        help='仅对选定修订中的纯断流或客户端超时按间隔重试原请求；需 --attempts 1')
    revise.add_argument('--output-contract', choices=['revision_wire_v1'],
                        help='显式冻结选定失败页的新输出约束；保留旧预算及有效缓存，每个新请求仍最多三次纠错')
    revise.add_argument('--wait', action='store_true', help='等待当前修订结束后读取最新进度并续跑')
    status = commands.add_parser('status')
    profile = commands.add_parser('profile', help='离线汇总请求分段耗时；不调用模型')
    profile.add_argument('--run-id', help='仅汇总指定运行；省略时汇总所有已记录运行')
    locate = commands.add_parser('locate-requests', help='零模型请求重放指定对象，定位实际失败请求')
    locate.add_argument('--source', required=True, type=Path)
    locate.add_argument('--job-id', action='append', required=True)
    locate.add_argument('--workers', type=int, default=8)
    locate.add_argument('--object-workers', type=int, default=4)
    catalog = commands.add_parser('catalog', help='核对已完成产物并生成可浏览目录')
    catalog.add_argument('--batch', action='append', required=True, type=Path)
    catalog.add_argument('--follow', action='store_true', help='随正在运行的批次自动更新目录')
    for command in (prepare, supplement, continuation, revision, revise, members, run, status, profile, locate, catalog):
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
            from src.share.wiki import wiki_members
            value = wiki_members.prepare(args.members, args.page, output, args.self_account, config,
                non_friends_only=args.non_friends_only, priority_accounts=args.priority_account,
                accounts=args.account, max_chars=args.batch_chars)
        else:
            value = wiki_batch.prepare(args.wiki, args.messages, output, args.self_account, config,
                                      max_chars=args.batch_chars, reuse=args.reuse)
    elif args.command == 'prepare-supplement':
        value = wiki_supplement.prepare(args.parent, output)
    elif args.command == 'prepare-continuation':
        from src.share.wiki import wiki_resumption
        value = wiki_resumption.prepare(args.source, output, parent=args.parent,
                                         reread_contract=args.reread_contract,
                                         structured_contract=args.structured_contract,
                                         reference_contract=args.reference_contract,
                                         correction_contract=args.correction_contract,
                                         completion_contract=args.completion_contract,
                                         client_recovery_contract=args.client_recovery_contract,
                                         request_locations=args.request_locations, job_ids=args.job_id)
    elif args.command == 'prepare-revision':
        from src.share.wiki import wiki_revision_batch
        value = wiki_revision_batch.prepare(args.source, output, engine=args.engine)
    elif args.command == 'revise':
        from src.share.wiki import wiki_revision_batch
        if args.connection_retry_seconds is not None and (args.connection_retry_seconds < 1
                or not args.job_id or args.attempts != 1 or args.skip_incomplete):
            parser.error('--connection-retry-seconds requires a positive interval, --job-id, --attempts 1 and no --skip-incomplete')
        install_transport(output)
        value = wiki_revision_batch.run(output, workers=args.workers, attempts=args.attempts,
                                       object_workers=args.object_workers,
                                       max_jobs=args.max_jobs, follow=args.follow,
                                       skip_incomplete=args.skip_incomplete,
                                       job_ids=args.job_id, wait=args.wait,
                                       connection_retry_seconds=args.connection_retry_seconds,
                                       output_contract=args.output_contract)
        value = {k: v for k, v in value.items() if k != 'jobs'}
    elif args.command == 'status':
        value = wiki_batch.status(output)
    elif args.command == 'profile':
        from src.share.wiki import wiki_job_pool
        value = wiki_job_pool.request_profile(output, run_id=args.run_id)
    elif args.command == 'locate-requests':
        from src.share.wiki import wiki_cached
        value = wiki_cached.locate_requests(args.source, output, args.job_id,
            workers=args.workers, object_workers=args.object_workers)
    elif args.command == 'catalog':
        value = wiki_delivery.deliver(args.batch, output, follow=args.follow)
        value = dict(stage=value['stage'], **value['summary'])
    else:
        manifest = json.loads((output / 'manifest.json').read_text())
        backend = wiki_supplement if manifest['schema'] == 'wiki_supplement_v1' else wiki_batch
        if args.job_id is not None and backend is wiki_supplement:
            parser.error('--job-id currently supports person Wiki batches only')
        if (args.request_workers is not None or args.request_timeout_seconds is not None or
                args.profile_requests or args.retry_transport_only or args.retry_transport_signal or args.cached_only
                or args.connection_retry_seconds is not None
                ) and backend is wiki_supplement:
            parser.error('request runtime options currently support person Wiki batches only')
        if args.retry_transport_signal and not args.retry_transport_only:
            parser.error('--retry-transport-signal requires --retry-transport-only')
        if args.request_timeout_seconds is not None and args.request_timeout_seconds < 1:
            parser.error('--request-timeout-seconds must be positive')
        if args.connection_retry_seconds is not None and (args.connection_retry_seconds < 1
                or args.attempts != 1 or not args.job_id or args.cached_only or args.skip_incomplete
                or args.retry_transport_only or args.retry_transport_signal):
            parser.error('--connection-retry-seconds requires a positive interval, --job-id and --attempts 1')
        if args.cached_only and (args.skip_incomplete or args.retry_transport_only or args.retry_transport_signal
                or args.request_workers is not None or args.request_timeout_seconds is not None or args.profile_requests):
            parser.error('--cached-only cannot use live request or retry options')
        if not args.cached_only:
            install_transport(output)
        request_options = ({'request_workers': args.request_workers}
                           if args.request_workers is not None else {})
        if args.request_timeout_seconds is not None:
            request_options['request_timeout_seconds'] = args.request_timeout_seconds
        if args.profile_requests:
            request_options['profile_requests'] = True
        if args.connection_retry_seconds is not None:
            request_options['connection_retry_seconds'] = args.connection_retry_seconds
        if args.retry_transport_only:
            request_options['retry_transport_only'] = True
        if args.retry_transport_signal:
            request_options['retry_transport_signals'] = args.retry_transport_signal
        if args.cached_only:
            request_options['cached_only'] = True
        if args.job_id is not None:
            request_options['job_ids'] = args.job_id
        value = backend.run(output, workers=args.workers, object_workers=args.object_workers,
                            attempts=args.attempts, max_jobs=args.max_jobs,
                            skip_incomplete=args.skip_incomplete, wait=args.wait, **request_options)
        value = {k: v for k, v in value.items() if k != 'jobs'}
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return 0 if args.command in ('status', 'profile') or value.get('stage', 'complete') == 'complete' else 1


def install_transport(output):
    from src.judge import codex_transport
    from src.infrastructure.config import sha256_file
    codex_transport.settings()
    codex_transport.install(expected_root=ROOT, feature_format_retry=False, stop_on_service_failure=True,
        receipt_directory=output / 'codex_transport_overrides', provenance={
            'bootstrap_sha256': sha256_file(Path(__file__)),
            'wiki_manifest_sha256': sha256_file(output / 'manifest.json')})


if __name__ == '__main__':
    raise SystemExit(main())
