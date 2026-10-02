#!/usr/bin/env python3
"""统一校验入口：框架回归、Judge 材料/已存观察、正式开发证据。不会运行实验或晋级。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Report:
    def __init__(self, mode):
        self.value = {'schema': 1, 'mode': mode, 'checks': []}

    def run(self, name, function):
        print(f'check: {name}', file=sys.stderr, flush=True)
        started = time.monotonic()
        item = {'name': name}
        try:
            item.update(status='passed', detail=function())
        except Exception as exc:
            item.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        item['seconds'] = round(time.monotonic() - started, 3)
        self.value['checks'].append(item)
        return item['status'] == 'passed'

    def failure(self, name, exc):
        self.value['checks'].append(dict(name=name, status='failed',
                                         error=f'{type(exc).__name__}: {exc}'))

    def finish(self):
        self.value['status'] = ('passed' if self.value['checks'] and
            all(x['status'] == 'passed' for x in self.value['checks']) else 'failed')
        return self.value


def command(argv):
    result = subprocess.run([sys.executable, *argv], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    if result.returncode:
        raise RuntimeError(f'exit={result.returncode}\n{result.stdout}')
    return result.stdout.strip()


def check_code(args, report):
    staged = ['--staged'] if args.staged else []
    history = ['--history'] if args.history else []
    # Keep one implementation of every rule; aggregate failures instead of hiding
    # all checks after the first failure. Experiment checks never run this suite.
    checks = [('rules', ['scripts/check_rules.py', *staged]),
              ('docs', ['scripts/check_docs.py'])]
    if args.public or args.history:
        checks.append(('public', ['scripts/check_public.py', *staged, *history]))
    for name, argv in checks:
        report.run(name, lambda argv=argv: command(argv))
    if args.full or args.tests:
        report.run('regression', lambda: command(['-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                                  *(args.tests or [])]))
    if args.demo_output:
        report.run('offline_demo', lambda: command(['scripts/offline_demo.py', '--output', args.demo_output]))


@contextmanager
def offline():
    from src.judge.corrected import CodexJudgeClient
    from src.llm import ChatClient
    def forbidden(*args, **kwargs):
        raise RuntimeError('校验禁止请求大模型；缺失证据必须由正式实验流程处理')
    with patch.object(CodexJudgeClient, 'run', forbidden), patch.object(ChatClient, 'chat', forbidden), \
            patch.object(ChatClient, '_send', forbidden):
        yield


def check_replay(path, pack, baseline, candidate):
    from src.iteration import draw_replay
    binding, value = draw_replay.manifest(path, pack, baseline, candidate)
    replay = draw_replay.SavedDrawReplay(binding, pack, baseline, candidate)
    count = 0
    for case in pack['rows']:
        for rnd in value['draws'][str(case['case_id'])]:
            replay.get(case, int(rnd), None)
            count += 1
    return dict(manifest=binding, cases=len(pack['rows']), saved_draws=count,
                note='核验已有观察；正式运行器仍按分歧要求独立补验，缺失则拒绝，无实时请求回退')


def check_judge(args, report):
    from src.config import ConfigError, valid_name
    from src.iteration import datasets, learning_guard as guard, versions
    path = versions.PRIVATE / 'judge_eval' / valid_name(args.pack) / 'pack.json'
    pack = json.loads(path.read_text())
    pointers = versions.load_pointers()
    baseline = versions.judge_dir(args.baseline or pointers['iteration_judge'])
    candidate = versions.judge_dir(args.candidate)
    data = versions.data_version_dir(pack['data_ref'])
    dirs = [versions.generator_dir(pack['c0_gen_version']), baseline['dir'], candidate['dir']]
    seal = guard.RunSeal([data, *dirs], [path, versions.POINTERS_PATH])
    report.value['selection'] = dict(data=pack['data_ref'], generator=pack['c0_gen_version'],
                                     baseline=baseline['id'], candidate=candidate['id'])
    report.value['current_pointers'] = pointers

    def check_pack():
        datasets.assert_pack(data, pack, 'judge_development')
        guard.verify_pack(pack, 'development')
        return {'cases': len(pack['rows'])}

    report.run('development_pack', check_pack)
    def learning_sources():
        from src import cache
        proof = guard.snapshot(pack['data_ref'], dirs, 'judge_development')
        return dict(snapshot_sha256=cache.digest(proof), materials=list(proof['materials']),
                    guard_code=proof['guard_code'])
    report.run('learning_sources', learning_sources)
    def current_context():
        expected = dict(data=pack['data_ref'], production_gen=pack['c0_gen_version'], iteration_judge=baseline['id'])
        differences = {k: dict(current=pointers.get(k), required=v) for k, v in expected.items() if pointers.get(k) != v}
        if differences:
            raise ConfigError('材料可以核验，但当前基线尚未衔接：' + json.dumps(differences, ensure_ascii=False))
        # Ordinary creation also validates the production Judge, if different.
        if pointers['production_judge'] != baseline['id']:
            guard.require_materials(pack['data_ref'], [versions.judge_dir(pointers['production_judge'])['dir']])
        return expected
    report.run('current_baseline_context', current_context)
    if args.saved_draw_manifest:
        report.run('saved_observations', lambda: check_replay(args.saved_draw_manifest, pack, baseline, candidate))
    report.run('unchanged_inputs', seal.check)


def check_experiment(args, report):
    from src.config import ConfigError
    from src.iteration import experiment, gates, protocol, record_contract, versions
    if not args.exp:
        from scripts.check_rules import check_instance
        def registry():
            findings = check_instance(versions.PRIVATE)
            if findings:
                raise ConfigError('\n'.join(findings))
            return '已存结果登记检查通过'
        report.run('registered_results', registry)
        return
    directory = experiment.load_experiment(args.exp)
    def evidence():
        spec, metrics = gates.evidence(directory)
        record_contract.validate_completion(directory, experiment.state_of(directory).get('metrics', {}))
        decision = protocol.decide(metrics, 'development', spec['protocol'])
        report.value['metrics'] = metrics
        report.value['development_decision'] = decision
        if args.gate == 'fixed-entry':
            return gates.require_fixed_entry(spec['kind'], spec['data_ref'], spec['baseline_ref'],
                spec['candidate_ref'], spec['protocol'], judge_ref=spec.get('judge_ref'),
                development_pack=spec.get('pack_ref'), evidence_id=directory.name)
        if decision['verdict'] != 'merge_to_iteration_baseline':
            raise ConfigError(decision['reason'])
        return {'evidence_passed': True, 'note': '只核验开发证据；实际晋级仍由 promote 重新核验当前指针'}
    report.run(args.gate, evidence)
    if getattr(args, 'reply_model_report', False):
        from src.iteration.reply_model_report import summarize
        report.run('reply_model_comparison', lambda: summarize(directory,
            verified_metrics=report.value.get('metrics')))


def check_data(args, report):
    from src.bootstrap import conversations, data_quality, import_coverage
    from src.iteration import versions
    overrides = {k: getattr(args, k) for k in
        ('context_gap_seconds', 'response_gap_seconds', 'reply_gap_seconds') if getattr(args, k) is not None}
    report.value['data_quality'] = {}
    for ref in args.data:
        def inspect(ref=ref):
            directory = versions.data_version_dir(ref)
            binding = None if args.import_coverage_only else data_quality.acceptance_binding(directory)
            result = (import_coverage.audit(directory) if args.import_coverage_only else
                data_quality.audit(directory,
                    segmentation=conversations.policy(**overrides) if overrides else None))
            report.value['data_quality'][ref] = result
            if binding is not None:
                if binding != data_quality.acceptance_binding(directory):
                    from src.config import ConfigError
                    raise ConfigError('数据内容或验收代码在检查过程中变化')
                result['acceptance_binding'] = binding
            data_quality.require_passed(result)
            return dict(data_ref=ref, passed=True)
        report.run('data:' + ref, inspect)


def check_materials(args, report):
    from src.config import ConfigError
    from src.iteration import material_compatibility, versions
    def inspect():
        value = material_compatibility.inspect(versions.data_version_dir(args.data),
            versions.judge_dir(args.judge)['dir'], versions.generator_dir(args.generator))
        report.value['material_compatibility'] = value
        if value['source_conflicts']:
            raise ConfigError('学习来源与新版开发用途冲突；详见 material_compatibility.summary')
        return '来源记录未见冲突；不替代模型重建和正式实验保护'
    report.run('material_source_compatibility', inspect)
    if args.reconstruct:
        from src.iteration import gbdt_evidence, learning_guard, pack_transport, runtime
        def reconstruct():
            with pack_transport.archived_gbdt_paths(gbdt_evidence, runtime):
                return learning_guard.require_materials(args.data,
                    [versions.judge_dir(args.judge)['dir'], versions.generator_dir(args.generator)])
        report.run('material_reconstruction', reconstruct)


def check_dashboard(args, report):
    from src.dashboard.diagnostics import inspect, inspect_access
    from src.iteration import versions
    if args.base_url:
        report.run('dashboard_access', lambda: inspect_access(args.base_url, args.instance, args.share,
                                                             download=args.download))
        return
    report.run('dashboard_render', lambda: inspect(versions.PRIVATE))


def check_retrieval(args, report):
    from src.iteration.retrieval_profile import inspect
    if not report.run('frozen_retrieval_profile', lambda: inspect(args.instance, args.exp,
            cases=args.cases, repeats=args.repeats, materials=args.materials)):
        return
    if args.materials:
        result = report.value['checks'][-1]['detail']['materials']
        if result['status'] == 'failed':
            report.failure('material_reuse', RuntimeError(result['error']))
        else:
            report.run('material_reuse', lambda: result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='可选 JSON 报告路径（不可覆盖既有文件）')
    sub = parser.add_subparsers(dest='mode', required=True)
    code = sub.add_parser('code', help='改框架后：规则、文档及所选回归；公开发布检查按需启用')
    scope = code.add_mutually_exclusive_group()
    scope.add_argument('--staged', action='store_true')
    scope.add_argument('--history', action='store_true', help='显式检查公开发布边界及 Git 历史')
    code.add_argument('--public', action='store_true', help='公开发布时检查导出边界')
    tests = code.add_mutually_exclusive_group()
    tests.add_argument('--tests', nargs='+', help='相关测试文件；省略不运行 pytest')
    tests.add_argument('--full', action='store_true', help='运行全部离线回归')
    code.add_argument('--demo-output', help='另运行合成演示；输出目录须为空')
    wiki = sub.add_parser('wiki', help='核对 Wiki 批次产物及已知转发称呼归属问题；不生成事实')
    wiki.add_argument('--batch', action='append', required=True, type=Path)
    wiki.add_argument('--require-complete', action='store_true', help='批次仍有未完成任务时检查失败')
    wiki.add_argument('--catalog', type=Path, help='同时核对已交付对象正文及 JSON、XML、离线卡片')
    judge = sub.add_parser('judge', help='开发轮材料和已存观察；不创建评测、不请求、不切基线')
    judge.add_argument('--instance', required=True)
    judge.add_argument('--pack', required=True)
    judge.add_argument('--baseline', help='默认当前开发 Judge；显式指定可核验待迁移对照')
    judge.add_argument('--candidate', required=True)
    judge.add_argument('--saved-draw-manifest')
    exp = sub.add_parser('experiment', help='复核已完成正式开发实验及分层门槛')
    exp.add_argument('--instance', required=True)
    exp.add_argument('--exp', help='正式开发实验 ID；省略则检查实例历史结果登记')
    exp.add_argument('--gate', choices=['development', 'fixed-entry'], default='development')
    exp.add_argument('--reply-model-report', action='store_true',
                     help='附加群/私回复模型质量和耗时汇总；区分真实调用与缓存历史，不改变晋级门槛')
    data = sub.add_parser('data', help='只读检查数据来源、会话切分及用途隔离；只输出聚合统计')
    data.add_argument('--instance', required=True)
    data.add_argument('--data', nargs='+', required=True)
    data.add_argument('--import-coverage-only', action='store_true',
                      help='只核对冻结导出清单到消息归档的全类型覆盖，不重复样本检查')
    for name in ('context-gap-seconds', 'response-gap-seconds', 'reply-gap-seconds'):
        data.add_argument('--' + name, type=int, help='覆盖审计阈值，不修改冻结数据')
    materials = sub.add_parser('materials', help='数据变更后的已绑定学习来源诊断；不授权评测')
    materials.add_argument('--instance', required=True)
    materials.add_argument('--data', required=True)
    materials.add_argument('--judge', required=True)
    materials.add_argument('--generator', required=True)
    materials.add_argument('--reconstruct', action='store_true',
                           help='同时使用正式运行器的保护重建材料；无模型请求')
    dashboard = sub.add_parser('dashboard', help='只读分析首页及实时接口渲染耗时、体积与调用热点')
    dashboard.add_argument('--instance', required=True)
    dashboard.add_argument('--base-url', help='只读检查部署后的匿名拒绝及登录访问；远程必须为 HTTPS')
    dashboard.add_argument('--share', nargs='+', default=[], help='需核验页面及 XML 下载的 Share 版本')
    dashboard.add_argument('--download', choices=['knowledge.json', 'wiki.xml'], default='knowledge.json',
                           help='Share 下载文件；XML 快照使用 wiki.xml')
    retrieval = sub.add_parser('retrieval', help='离线对比冻结召回与索引召回；无模型请求、无实验写入')
    retrieval.add_argument('--instance', required=True)
    retrieval.add_argument('--exp', required=True)
    retrieval.add_argument('--cases', type=int, default=3)
    retrieval.add_argument('--repeats', type=int, default=1)
    retrieval.add_argument('--materials', action='store_true', help='附加跨上下文材料校验凭据复用耗时')
    args = parser.parse_args(argv)
    if args.mode == 'experiment' and args.gate == 'fixed-entry' and not args.exp:
        parser.error('--gate fixed-entry 需要 --exp 正式开发实验 ID')
    if args.output and Path(args.output).exists():
        parser.error('报告已存在；使用新路径，禁止覆盖历史产物')
    report = Report(args.mode)
    if args.mode == 'code':
        check_code(args, report)
    elif args.mode == 'wiki':
        from src.bootstrap.wiki_delivery import require_valid
        with offline():
            report.run('wiki_artifacts', lambda: require_valid(args.batch,
                require_complete=args.require_complete, catalog=args.catalog))
        report.value.update(evaluation_started=False, model_requests_prohibited=True)
    else:
        from src.iteration import versions
        from src.config import ConfigError
        before = None
        try:
            versions.switch_instance(args.instance)
            before = versions.POINTERS_PATH.read_bytes()
            with offline():
                {'judge': check_judge, 'experiment': check_experiment, 'data': check_data,
                 'materials': check_materials, 'dashboard': check_dashboard,
                 'retrieval': check_retrieval}[args.mode](args, report)
        except Exception as exc:
            report.failure('validation_setup', exc)
        finally:
            if before is not None:
                def unchanged():
                    if versions.POINTERS_PATH.read_bytes() != before:
                        raise ConfigError('校验期间基线指针发生变化，请重试')
                    return True
                report.run('unchanged_pointers', unchanged)
        report.value.update(evaluation_started=False, model_requests_prohibited=True)
    value = report.finish()
    if args.output:
        from src.iteration.storage import write_once_json
        write_once_json(Path(args.output), value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return int(value['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
