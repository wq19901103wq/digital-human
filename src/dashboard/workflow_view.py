"""Read-only overview of branches, training stages and concurrent evaluations."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

from ..iteration import task_state, jobs
from . import components as report

PHASES = {
    'needs_attention': '需要处理', 'cancelled': '已取消', 'deadline_reached': '已到截止时间',
    'request_budget_exhausted': '调用额度已耗尽', 'cost_budget_exhausted': '成本预留额度已耗尽',
    'baseline_review': '候选已训练，等待基线兼容处理', 'generating_development': '生成开发回复',
    'queued': '等待调度', 'preparing': '准备材料', 'prepared': '材料已准备',
    'waiting_for_source_acceptance': '等待来源验收', 'trained': '训练完成',
    'runtime_unavailable': '运行快照不可用',
    'preflight': '小样本预检', 'preflight_passed': '预检通过',
    'generating': '生成训练或评测回复', 'training_and_development': '训练与开发回复并行构建',
    'generated': '回复已生成', 'superseded_by_temporal_split': '已被新的数据划分替代',
    'extracting': '抽取训练特征', 'fitting': '拟合小模型', 'training': '训练模型',
    'evaluating': '开发评测', 'independent_audit': '独立核验',
    'finished': '已完成', 'interrupted': '已中断', 'stopped': '已停止',
    'smoke': '冒烟检查', 'development': '开发评测', 'entry': '与当前生产基线直接比较',
    'fixed_test': '固定验收', 'promoted': '已采用', 'rejected': '未通过',
    'development_accepted': '保留开发版，等待人工提交下一改动',
    'blocked': '等待处理阻塞', 'conflict': '等待处理合并冲突', 'superseded': '基线已更新，等待重测',
    'integrated': '改动已包含在当前基线', 'retry_exhausted': '重试次数已耗尽',
    'submitted': '已提交', 'waiting_retry': '等待重试', 'launch_failed': '启动失败',
}


def _read(path):
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (ValueError, OSError):
        return {'status': 'unreadable', 'phase': '状态暂时无法读取'}


def _label(value):
    return PHASES.get(value, value or '尚未开始')


def _safe_ref(value):
    return isinstance(value, str) and re.fullmatch(r'[\w-]+', value)


def _state(directory):
    return task_state.resolve(_read(directory / 'state.json'))


def _progress(value):
    total = value.get('total')
    done = value.get('successful', value.get('completed'))
    if isinstance(total, int) and isinstance(done, int):
        if done > total:
            return f'旧记录计数未区分阶段（成功 {done}，总量 {total}）'
        return f'成功 {done} / {total} · 失败 {value.get("failed", 0)}'
    return ''


def _status(value):
    if value.get('status') in ('interrupted', 'stopped', 'paused', 'unreadable', 'needs_attention', 'queued'):
        return {'paused': '已暂停'}.get(value['status'], _label(value['status']))
    return _label(value.get('phase', value.get('status', 'queued')))


def _training(instance, runs=()):
    rows = []
    directory_paths = {p.parent for p in (instance / 'judge_training').glob('*/spec.json')} | {p.parent for p in (instance / 'judge_training').glob('*/proposal.json')}
    paths = sorted([d / ('spec.json' if (d / 'spec.json').exists() else 'proposal.json') for d in directory_paths],
                   key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in paths:
        directory, spec = path.parent, _read(path)
        state = _state(directory)
        children = []
        for child in sorted(directory.glob('*/state.json')):
            value = _state(child.parent)
            name = '开发回复' if child.parent.name == 'development' else child.parent.name
            children.append(f'<li><strong>{report._e(name)}</strong> · {report._e(_status(value))}'
                            f' · {report._e(_progress(value))}</li>')
        evidence = state.get('evaluation_experiment') or spec.get('evaluation_experiment')
        link = ''
        if _safe_ref(evidence) and (instance / 'experiments' / evidence / 'spec.json').is_file():
            link = f'<a href="/instances/{quote(instance.name)}/experiments/{quote(evidence)}/index.html">查看评测结果</a>'
        linked = [r for r in runs if r['spec'].get('provenance', {}).get('study') == directory.name]
        if linked:
            link += '<details><summary>查看 ' + str(len(linked)) + ' 项比较结果</summary><ul>' + ''.join(
                '<li><a href="' + report._run_url(r) + '">' + report._e(r['spec']['provenance']['comparison_id']) + '</a></li>'
                for r in linked) + '</ul></details>'
        # Legacy producers may have written outside experiments. Surface every
        # missing result automatically; only the checked importer may register it.
        saved = {p.parent.name for p in directory.glob('comparisons/*/cases.jsonl')}
        registered = {r['spec']['provenance'].get('comparison_id') for r in linked}
        missing = sorted(saved - registered)
        if missing:
            link += ('<details class="error-box"><summary>' + str(len(missing))
                     + ' 项历史比较尚未入表</summary><p>此任务的历史表尚不完整；'
                     '需通过现有历史导入入口补录，不能把缺失记录解释为没有结果。</p><ul>'
                     + ''.join('<li>' + report._e(name) + '</li>' for name in missing)
                     + '</ul></details>')
        versions = ' · '.join(report._version_link(instance, folder, spec.get(key)) for folder, key in
                              [('data', 'data_ref'), ('generators', 'generator_ref'), ('judges', 'source_judge')])
        message = state.get('interruption_reason') or state.get('reason') or state.get('error', '')
        if spec.get('adoption_allowed') is False:
            message += ' 本轮只比较效果，不切换基线。'
        rows.append(f'<tr><td><strong>{report._e(directory.name)}</strong><div>{versions}</div></td>'
                    f'<td>{report._e(_status(state))}<div class="help">{report._e(message)}</div></td>'
                    f'<td>{report._e(_progress(state))}<ul>{"".join(children)}</ul>{link}</td></tr>')
    if not rows:
        return ''
    return ('<h3>训练任务</h3><p class="help">准备材料 → 小样本预检 → 生成回复 → 抽特征 → 拟合 → 开发评测 → 独立核验。'
            '可并行的子任务分别显示进度；状态结合生产进程是否存活判断。</p>'
            '<div class="table-wrap"><table><thead><tr><th>任务 / 版本</th><th>当前步骤</th><th>子任务进度</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table></div>')


def _branches(instance):
    rows = []
    for path in sorted((instance / 'branches').glob('*/state.json')):
        state, directory = _read(path), path.parent
        rid = (state.get('rounds') or [None])[-1]
        record = _read(directory / 'rounds' / (rid + '.json')) if _safe_ref(rid) else {}
        phase = '已暂停' if state.get('paused') else _label(record.get('phase', 'queued'))
        links = []
        for stage, suffix in [('smoke', 'smoke'), ('development', 'dev'), ('entry', 'entry'), ('fixed_test', 'fixed')]:
            eid = f'branch-{directory.name}-{rid}-{suffix}'
            if (instance / 'experiments' / eid / 'spec.json').is_file():
                links.append(f'<a href="/instances/{quote(instance.name)}/experiments/{quote(eid)}/index.html">{_label(stage)}</a>')
        rows.append(f'<tr><td>{report._e(directory.name)} · {report._e(rid or "待建立首轮")}</td>'
                    f'<td>{report._e(phase)}<div class="help">{report._e(record.get("reason", ""))}</div></td>'
                    f'<td>{" · ".join(links) or "尚未创建评测"}</td></tr>')
    if not rows:
        return '<p class="help">尚未向多实验调度器提交分支。候选方向由人提出，提交后按流程执行。</p>'
    return ('<h3>候选分支</h3><p class="help">人工提交 → 冒烟 → 开发 → 当前生产基线对比 → 固定验收 → 采用 / 保留 / 淘汰。</p>'
            '<div class="table-wrap"><table><thead><tr><th>分支 / 轮次</th><th>当前步骤 / 阻塞原因</th><th>各阶段记录</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table></div>')


def _running(instance, runs, render_stage):
    entries = []
    for run in runs:
        value = run['state'].get('progress') or {}
        if run['state'].get('status') != 'finished' and task_state.active(value):
            entries.append(f'<details data-detail-key="running-{report._e(run["spec"]["id"])}" open>'
                           f'<summary><a href="{report._run_url(run)}">{report._e(report._title(run["spec"]))}</a></summary>'
                           + (render_stage(run) if render_stage else '') + '</details>')
    for path in sorted((instance / 'scheduling').glob('*.json')):
        metadata = _read(path)
        if metadata.get('job'):
            metadata = jobs.snapshot(metadata['job'], metadata, instance=instance)
        if metadata.get('status') in ('retry_exhausted', 'launch_failed', 'needs_attention'):
            entries.append(f'<p>{report._e((metadata.get("job") or {}).get("id", path.stem))} · '
                           f'{report._e(_label(metadata["status"]))} · 尝试 {metadata.get("attempts", 0)} 次 · '
                           f'{report._e(_label(metadata.get("reason")))} {report._e(metadata.get("error", ""))}</p>')
    return '<h3>正在执行的评测</h3>' + ''.join(entries) if entries else ''


def html(instance: Path, runs=(), *, render_stage=None):
    return ('<section class="panel" id="workflows"><div class="section-head"><h2>实验流程与并发任务</h2>'
            '<span class="pill">实时同步</span></div>' + _training(instance, runs) + _branches(instance)
            + _running(instance, runs, render_stage) + _resources(instance) + '</section>')


def _resources(instance):
    usage = _read(instance / 'resource_usage.json')
    policy = _read(instance / 'resource_policy.json')
    rows = []
    if policy or usage:
        rows.append(f"调用 {usage.get('requests', 0)} / {policy.get('max_requests', '不限')}；"
                    f"成本预留 {usage.get('reserved_cost_units', 0)} / {policy.get('max_cost_units', '不限')}；"
                    f"并发上限 {policy.get('max_active_requests', 16)}")
    for path in sorted((instance / 'acceptance').glob('*/manifest.json')):
        manifest = _read(path)
        ledger = _read(path.parent / 'ledger.json')
        total, used = len(manifest.get('batches', [])), len(ledger.get('claims', {}))
        rows.append(f"{path.parent.name} 固定验收批次：已消耗 {used} / {total}，剩余 {total - used}")
    return '<h3>资源与验收批次</h3>' + ''.join(f'<p>{report._e(row)}</p>' for row in rows) if rows else ''
