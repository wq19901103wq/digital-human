"""实验与实例的只读报告；读取规格、状态、逐题结果及发送时保存的调用实录。"""
from __future__ import annotations

import difflib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .. import tracing
from ..iteration import experiment, gates, promote, protocol
from ..judge import normalize_mode
from ..iteration.progress import is_active
from ..iteration.storage import atomic_write

from .components import _e, _version_url, _version_link, _pre, _details, _badge, _title, _run_url

_UI = Path(__file__).resolve().parent
_LABELS = {
    'merge_to_iteration_baseline': ('开发评测达标', 'success'),
    'adopt': ('固定验收通过', 'success'),
    'reject': ('未达到采用条件', 'danger'),
    'observe': ('改善不足，继续观察', 'warning'),
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def _phase(spec: dict) -> tuple[str, str]:
    if spec.get('comparison'):
        return '专项判别比较', 'comparison'
    if spec.get('smoke'):
        return '连接检查', 'smoke'
    if spec.get('material_repair') or spec.get('lr_retrain'):
        return '重训控制实验', 'development'
    if spec.get('dataset') == 'fixed_test':
        return '固定验收', 'fixed'
    if spec.get('purpose') == 'gen_optimization':
        return '生成器策略优化', 'development'
    return ('裁判抽样校准' if spec.get('kind') == 'judge_eval' else '生成器开发评测'), 'development'


def _status(spec: dict, state: dict) -> tuple[str, str]:
    if state.get('status') in ('paused', 'stopped', 'interrupted', 'queued'):
        return {'paused': '已暂停', 'stopped': '已停止', 'interrupted': '已中断', 'queued': '等待执行'}[state['status']], 'neutral'
    if (state.get('resolution') or {}).get('verified_by'):
        return '历史失败 · 新配置已验证', 'neutral'
    if is_active(state.get('progress') or {}):
        return '运行中', 'success'
    if state.get('verdict') == 'experiment_incomplete':
        return '执行失败，待重试', 'danger'
    if state.get('status') != 'finished':
        return '尚未完成', 'neutral'
    if spec.get('smoke'):
        return '连接检查通过', 'success'
    if gates.diagnostic_reason(spec):
        return '诊断对比完成', 'neutral'
    if spec.get('kind') == 'judge_eval' and state.get('verdict') == 'merge_to_iteration_baseline':
        return '本轮校准达标', 'success'
    return _LABELS.get(state.get('verdict'), ('已结束，暂无结论', 'neutral'))


def _reason(spec: dict, state: dict) -> str:
    if is_active(state.get('progress') or {}):
        return '任务正在运行，以下识别率来自已完成的有效题，结束后才形成结论。'
    resolution = state.get('resolution') or {}
    if resolution.get('verified_by'):
        return str(resolution.get('reason') or '同一组失败题已由新任务验证通过；原始失败结果保留。')
    if spec.get('smoke') and state.get('status') == 'finished':
        return '本次连接检查已完成，不形成效果改善或版本采用结论。'
    if state.get('status') == 'finished' and gates.diagnostic_reason(spec):
        return gates.diagnostic_reason(spec)
    reason = str(state.get('reason') or '暂无最终结论；这里不代表后台进程一定仍在运行。')
    if state.get('verdict') == 'merge_to_iteration_baseline':
        reason = reason.replace('，并入开发基线', '，满足开发基线晋升条件')
    # 历史账单沿用生成器措辞；裁判的优化方向是识别数上升。原文仍保留在诊断区。
    return reason.replace('严格下降', '严格上升') if spec.get('kind') == 'judge_eval' else reason


def _page(title: str, body: str, instance: str = '', live_url: str = '') -> str:
    nav = (f'<a href="/dashboard/{quote(instance, safe="")}/index.html">实例概览</a>'
           f'<a href="/dashboard/{quote(instance, safe="")}/versions/index.html">版本档案</a>') if instance else ''
    updated = datetime.now().strftime('%m-%d %H:%M')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)} · 数字人工作台</title>
<style>{(_UI / 'dashboard.css').read_text(encoding='utf-8')}</style></head><body data-live-url="{_e(live_url)}">
<header class="topbar"><div class="topbar-inner"><a class="brand" href="/dashboard/index.html"><span class="brand-mark">DH</span>数字人工作台</a>
<nav class="topnav" aria-label="主导航">{nav}<a href="/dashboard/index.html">全部实例</a><span class="pill">{_e(instance or '总览')}</span></nav></div></header>
<main class="page">{('<div class="live-bar"><span data-live-status>正在连接实时状态…</span><button class="button" data-reload>立即刷新</button></div>' if live_url else '')}{body}<footer class="footer">报告生成于 {updated} · 实时状态见页面上方</footer></main>
<script>{(_UI / 'dashboard.js').read_text(encoding='utf-8')}</script></body></html>'''


def _stat(label: str, value: Any, foot: str) -> str:
    return f'<div class="stat"><div class="stat-label">{_e(label)}</div><strong class="stat-value">{_e(value)}</strong><div class="stat-foot">{_e(foot)}</div></div>'


def _load_run(exp_dir: Path, include_records: bool = True) -> dict:
    spec, state = experiment.spec_of(exp_dir), experiment.state_of(exp_dir)
    review = promote.threshold_review(exp_dir)
    if review:
        state = {**state, 'verdict': review['decision']['verdict'],
                 'reason': '按修订门槛复核：' + review['decision']['reason'] +
                           '；原协议结论保留为 ' + review['original_verdict'],
                 'threshold_review': review}
    # 固定集报告只读汇总，不能把答案藏进折叠区或原始 JSON。
    records = {}
    if include_records and spec.get('dataset') != 'fixed_test':
        records, _ = protocol.summarize_final_records(exp_dir / 'cases.jsonl')
    metrics = state.get('metrics') or {}
    progress = state.get('progress') or {}
    if progress and state.get('status') != 'finished':
        metrics = {**progress.get('counts', {}), 'attempted': progress.get('total', 0)}
    attempted = metrics.get('attempted', metrics.get('n', spec.get('smoke_limit') or len(records)))
    failures = metrics.get('failures', sum(r.get('status') == 'failed' for r in records.values()))
    pairs = metrics.get('pairs', sum(r.get('status') != 'failed' for r in records.values()))
    return {'spec': spec, 'state': state, 'metrics': metrics, 'records': list(records.values()),
            'attempted': attempted, 'failures': failures, 'pairs': pairs, 'dir': exp_dir,
            'instance_dir': exp_dir.parent.parent}


def _runs(exp_root: Path) -> list[dict]:
    items = [_load_run(p.parent, include_records=False) for p in exp_root.glob('*/spec.json')]
    return sorted(items, key=lambda r: (r['spec'].get('created', ''), r['spec']['id']), reverse=True)


def _current_run(runs: list) -> dict:
    return next((r for r in runs if is_active(r['state'].get('progress') or {})), runs[0])


def _error_html(run: dict) -> str:
    records = run['records']
    if not records and run['failures'] and run['spec'].get('dataset') != 'fixed_test':
        final, _ = protocol.summarize_final_records(run['dir'] / 'cases.jsonl')
        records = list(final.values())
    counts = Counter(str(r.get('reason') or '失败原因未记录') for r in records if r.get('status') == 'failed')
    if not counts:
        return ''
    reason, n = counts.most_common(1)[0]
    other = f'；另外 {len(counts) - 1} 种原因可在逐题详情查看' if len(counts) > 1 else ''
    return f'<div class="error-box"><h3>主要失败原因 · {n} 题{other}</h3>{_pre(reason)}</div>'


def _next_step(run: dict) -> str:
    spec, state = run['spec'], run['state']
    if is_active(state.get('progress') or {}):
        return '任务正在执行，进度与识别率自动更新；完成后再查看正式结论。'
    if (state.get('resolution') or {}).get('verified_by'):
        return '同一组题目已在新配置下完成生成和评分，修复验证见关联任务。'
    if state.get('verdict') == 'experiment_incomplete':
        return '下一步：根据失败原因修复执行问题，再续跑这次任务。已有成功题会保留。'
    if state.get('status') != 'finished':
        return '下一步：查看执行进度；若任务已停止，可从已有记录继续运行。'
    if state.get('threshold_review'):
        return '门槛修订沿用原实验成绩和独立补验，实际采用状态见版本指针；原始结论保留。'
    if spec.get('smoke'):
        return '下一步：连接检查已通过，可以准备完整开发评测。'
    if spec.get('material_repair') or spec.get('lr_retrain'):
        return '下一步：选定候选后，与当前已晋级基线直接比较；本轮控制实验不能替代正式晋级评测。'
    if spec.get('source_audit', {}).get('promotion_eligible') is False:
        return ('下一步：用开发验证题单比较选定策略。' if spec.get('purpose') == 'gen_optimization'
                else '下一步：核验模型学习来源，当前诊断结果不能晋升。')
    if spec.get('against_production'):
        return '下一步：核验生产基线直接比较是否满足固定验收准入；本轮不重复晋级开发版。'
    if spec.get('branch_round') and state.get('verdict') in ('adopt', 'merge_to_iteration_baseline'):
        return '分支调度器将核对当前全量后推进；基线已变化时须重新评测。'
    if state.get('verdict') in ('adopt', 'merge_to_iteration_baseline'):
        return '下一步：核对评测结果与版本差异，再执行对应的版本晋升。'
    return '下一步：查看逐题差异，在开发侧调整候选后再评测。'


def _model_rows(config: dict) -> str:
    llm = config.get('llm') or {}
    models = [(label, (llm.get(key) or {}).get('model')) for key, label in [('private', '私聊'), ('group', '群聊')] if isinstance(llm.get(key), dict)]
    if not models:
        models = [('模型', llm.get('model') or '未记录')]
    return ''.join(f'<div class="model-row"><span>{label}</span><strong>{_e(model)}</strong></div>' for label, model in models)


def _judge_pipeline(config: dict, directory: Path) -> str:
    mode = normalize_mode(config.get('mode'))
    llm = config.get('llm') or {}
    model_name = llm.get('model', '模型未记录')
    if mode != 'corrected_pairwise':
        if mode != 'pairwise_llm':
            return f'<div class="judge-pipeline"><strong>裁判机制未记录</strong><p>模型：{_e(model_name)}。快照不足，无法确认评分流程或是否包含小模型。</p></div>'
        return f'<div class="judge-pipeline"><strong>单一大模型配对盲测</strong><p>{_e(model_name)} 对真人与 AI 回复做配对判断。该版本未配置特征提取与小模型校正。</p></div>'
    model = (_json(directory / 'correction.json').get('final_model') or {})
    features = model.get('input_feature_count', '—')
    threshold = float(config.get('correction_threshold', 0.7)) * 100
    feature_llm = {**llm, **config.get('feature_llm', {})}
    split = bool(config.get('feature_llm'))
    if split:
        first_steps = f'''<li><b>① 大模型初判</b><span>{_e(model_name)} · {_e(llm.get('reasoning_effort', '未记录'))}</span><p>对两边回复做初始配对判断。</p></li>
<li><b>② 大模型抽特征</b><span>{_e(feature_llm.get('model'))} · {_e(feature_llm.get('reasoning_effort', '未记录'))}</span><p>抽取结构化特征，交给固定的小模型。</p></li>'''
    else:
        first_steps = f'''<li><b>① 大模型初判 + 抽特征</b><span>{_e(model_name)} · {_e(llm.get('reasoning_effort', '未记录'))}</span><p>先判断真人回复，再单独调用抽取两边回复的结构化特征。</p></li>'''
    return f'''<div class="judge-pipeline"><strong>大模型抽特征 + 小模型预测</strong>
<ol class="mechanism-steps{' split-models' if split else ''}">{first_steps}
<li><b>{'③' if split else '②'} 小模型预测</b><span>本地逻辑回归 · {_e(features)} 维</span><p>使用冻结权重计算哪一边是真人的概率。</p></li>
<li><b>{'④' if split else '③'} 合并为最终判断</b><span>改判阈值 {threshold:g}%</span><p>小模型与初判不同时，置信度 ≥ {threshold:g}% 才改判；否则保留大模型初判。</p></li></ol>
<p class="help">大模型通过 {_e(llm.get('provider', '未记录通道'))} 调用；小模型在本地执行。此处展示评测推理流程，不是重新训练小模型。</p></div>'''


def _generator_pipeline(config: dict) -> str:
    enabled = (config.get('retriever') or {}).get('enabled', True)
    shots = config.get('max_shots_per_case', '未记录')
    budget = config.get('shots_char_budget', '未记录')
    style = f'风格召回：每题最多 {_e(shots)} 条示例，字数上限 {_e(budget)}。' if enabled else '本版本关闭风格示例召回。'
    return '<div class="generator-pipeline"><strong>人格提示词 + 场景规则 + 聊天上下文 → 生成回复</strong><p>' + style + ' 按私聊 / 群聊选择模型，检查返回的回复格式。</p></div>'


def _baseline_pointers(instance_dir: Path) -> dict:
    pointers = _json(instance_dir / 'pointers.json')
    pointers.setdefault('production_judge', pointers.get('judge'))
    for kind in ('gen', 'judge'):
        pointers.setdefault('iteration_' + kind, pointers.get('production_' + kind))
    return pointers


def _branch_baselines(instance_dir: Path) -> str:
    """各分支保留的开发版（分支基线）：生产指针之外、晋升之前的候选位置。
    stage_limit 挡住的分支明确标出解除命令，不再静默停车。"""
    rows = []
    for path in sorted((instance_dir / 'branches').glob('*/state.json')):
        state = _json(path)
        accepted = state.get('development') or {}
        candidate = accepted.get('candidate_ref')
        if not candidate:
            continue
        if state.get('stage_limit') == 'development':
            note = (f'<span class="help">🔒 仅开发比较，固定准入/验收暂停 · 解除: '
                    f'iterate_branches limit --stage fixed_test {path.parent.name} 后重新提交</span>')
        else:
            note = '<span class="help">推进中（达标后自动进入固定准入）</span>'
        folder = 'judges' if str(candidate).startswith('j-') else 'generators'
        rows.append(f'<tr><td>{_e(path.parent.name)}</td>'
                    f'<td>{_version_link(instance_dir, folder, candidate)}</td><td>{note}</td></tr>')
    if not rows:
        return ''
    return ('<h3>分支基线 · 各分支保留的开发版</h3><p class="help">分支基线在每次开发达标时前进；'
            '生产基线只在推全时前进。分支基线 ≠ 已晋升，推全后两者合并。</p>'
            '<div class="table-wrap"><table><thead><tr><th>分支</th><th>分支基线</th><th>推进状态</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table></div>')


def _baseline_summary(instance_dir: Path) -> str:
    pointers = _baseline_pointers(instance_dir)
    prepared = [p.parent.name for p in sorted((instance_dir / 'data').glob('*/purposes.json'), reverse=True)
                if p.parent.name != pointers.get('data')]
    prepared_html = ('<p>已准备数据（未切换基线）：' + '、'.join(
        _version_link(instance_dir, 'data', ref) for ref in prepared[:3]) + '</p>') if prepared else ''
    rows = []
    for label, kind, folder in [('生成器', 'gen', 'generators'), ('Judge', 'judge', 'judges')]:
        rows.append(f'<tr><th scope="row">{label}</th>'
                    f'<td>{_version_link(instance_dir, folder, pointers.get("production_" + kind))}</td>'
                    f'<td>{_version_link(instance_dir, folder, pointers.get("iteration_" + kind))}</td></tr>')
    return ('<section class="panel baseline-summary" id="baselines"><h2>当前基线</h2>'
            '<p>当前数据：' + _version_link(instance_dir, 'data', pointers.get('data')) + '</p>'
            + prepared_html +
            '<div class="table-wrap"><table><thead><tr><th>对象</th><th>当前生产基线</th><th>当前开发基线</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table></div>'
            '<p class="help">标准单线开发评测：创建时的开发基线 vs 候选；固定验收：创建时的生产基线 vs 开发候选。'
            '多分支迭代各轮以当时的生产基线为起点。实验开始后对照版本冻结，基线变化需新建对比，历史结果不改写。</p>'
            '<p class="help">创建版本、训练、重试和评测完成均不会自动切换基线；只有晋级才会更新对应基线。'
            '显式初始化/迁移另行记录，不属于实验晋级。历史方案复核、重训控制实验使用的实验对照不代表已晋级基线。</p>'
            + _branch_baselines(instance_dir) + '</section>')


def _baseline_context(run: dict, *, compact: bool = False) -> str:
    spec, instance = run['spec'], run['instance_dir']
    if spec.get('comparison'):
        return '<div class="baseline-context"><strong>专项判别比较</strong><p>' + _e(gates.diagnostic_reason(spec)) + '</p></div>'
    pointers = _baseline_pointers(instance)
    judge = spec.get('kind') == 'judge_eval'
    kind, folder = ('judge', 'judges') if judge else ('gen', 'generators')
    production = spec.get('dataset') == 'fixed_test' or bool(spec.get('branch_round')) or bool(spec.get('against_production'))
    key = ('production_' if production else 'iteration_') + kind
    label = '当前生产基线' if production else '当前开发基线'
    current = pointers.get(key)
    if spec.get('purpose') == 'gen_optimization':
        purpose = '生成器优化材料对比'
    elif spec.get('lr_retrain') or spec.get('material_repair'):
        purpose = '重训控制实验'
    elif spec.get('rerun'):
        purpose = '历史方案复核'
    elif spec.get('branch_round'):
        purpose = '多分支迭代'
    elif spec.get('against_production'):
        purpose = '生产基线直接比较'
    else:
        purpose = '连接检查' if spec.get('smoke') else '固定验收' if production else '标准开发评测'
    same = bool(current) and spec.get('baseline_ref') == current
    if not current:
        relation = label + '未设置'
    elif same:
        relation = '本轮对照与' + label + '一致'
    elif spec.get('candidate_ref') == current:
        relation = '本轮候选现为' + label
    else:
        relation = '本轮对照不是' + label
    if compact:
        relation = ('当前基线未设置' if not current else '与当前基线一致' if same
                    else '候选现为当前基线' if spec.get('candidate_ref') == current else '非当前基线对照')
    body = (f'<div class="baseline-context"><strong>{purpose}</strong>'
            f'<p>{relation}</p><p>{label}：{_version_link(instance, folder, current)}</p>')
    different = []
    if pointers.get('data') and spec.get('data_ref') != pointers['data']:
        different.append('评测数据与当前不同')
    if judge:
        pack_ref = spec.get('pack_ref')
        pack = _json(instance / 'judge_eval' / pack_ref / 'pack.json') if pack_ref else {}
        if pack.get('c0_gen_version') and pointers.get('production_gen') and pack['c0_gen_version'] != pointers['production_gen']:
            different.append('评估包的生成器与当前生产基线不同')
    elif pointers.get('production_judge') and spec.get('judge_ref') != pointers['production_judge']:
        different.append('评分 Judge 与当前生产基线不同')
    if different:
        body += '<p class="help">' + ('评测条件与当前不同。' if compact else '；'.join(different) + '。') + '</p>'
    if gates.diagnostic_reason(spec):
        body += '<p class="help">' + _e(gates.diagnostic_reason(spec)) + '</p>'
    elif not compact and (not same or different):
        body += '<p class="help">结果仅对本轮冻结对照有效，不能直接视为对当前系统的提升。</p>'
    return body + '</div>'


_STAGES = {'parallel': '多题并行评分', 'preparing': '准备数据与版本', 'preflight': '检查实验改动',
           'generating': '生成回复', 'judging': '大模型初判', 'extracting': '大模型抽取特征',
           'predicting': '小模型预测与校正', 'checkpoint': '保存本题结果',
           'summarizing': '汇总结果', 'finished': '任务已结束', 'interrupted': '执行已中断'}


def _pack_preparation_html(instance_dir: Path) -> str:
    paths = list((instance_dir / 'judge_eval').glob('pack-calibration-*/progress.json'))
    if not paths:
        return ''
    progress = _json(max(paths, key=lambda path: path.stat().st_mtime_ns))
    if progress.get('status') in ('finished', 'paused'):
        return ''
    title = '正在生成 Judge 评估包' if is_active(progress) else '评估包准备已停止，可续建'
    return (f'<section class="panel"><h2>{title} · {progress.get("total", 0)} 题</h2>'
            f'<p>已生成 {progress.get("completed", 0)} / {progress.get("total", 0)} 题 · 并行 {progress.get("workers", 1)} 题</p>'
            f'<p>后续抽特征模型：{_e(", ".join(progress.get("models", [])))} / {_e(progress.get("effort", ""))}</p>'
            '<p class="help">完整评估包生成后自动开始评分。此处是生成进度，识别率在评分开始后更新。</p></section>')


def _stage_html(run: dict) -> str:
    state = run['state']
    progress = state.get('progress') or {}
    active = is_active(progress)
    phase = progress.get('phase', '')
    if state.get('status') == 'finished':
        title = '任务已结束'
    elif active:
        title = _STAGES.get(phase, '执行中')
    elif phase == 'finished' and state.get('verdict') == 'experiment_incomplete':
        title = '本轮已结束，有失败题待重试'
    elif progress:
        title = '执行已停止 · 最后阶段：' + _STAGES.get(phase, phase)
    else:
        title = '没有正在执行的进程记录'
    branch = {'baseline': '对照', 'candidate': '候选'}.get(progress.get('branch'), '')
    supplement = f'补测第 {progress["round"]} 轮 · ' if progress.get('round') else ''
    case_index = progress.get('case_index', 0)
    detail = f'第 {case_index} / {run["attempted"]} 题 · {supplement}{branch}' if active and case_index else ''
    active_cases = progress.get('active_cases') or {}
    if active and progress.get('workers', 1) > 1:
        detail = f"并行上限 {progress['workers']} 题 · 正在处理 {len(active_cases)} 题"
    clock = f'<span data-stage-clock="{progress["phase_started_at"]}"></span>' if active and progress.get('phase_started_at') else ''
    finished = run['pairs'] + run['failures']
    ratio = min(100, finished / run['attempted'] * 100) if run['attempted'] else 0
    body = f'''<div class="execution"><div class="section-head"><h3>当前阶段 · {_e(title)}</h3>{clock}</div>
<p>{_e(detail)}</p><div class="progress" role="progressbar" aria-label="题目处理进度" aria-valuemin="0" aria-valuemax="{run['attempted']}" aria-valuenow="{finished}"><span style="width:{ratio:.1f}%"></span></div>
<div class="progress-copy"><span>已处理 {finished} / {run['attempted']} 题</span><span>有效 {run['pairs']} · 失败 {run['failures']}</span></div>'''
    if active and active_cases:
        body += '<ul>'
        for item in sorted(active_cases.values(), key=lambda item: item['case_index']):
            label = {'baseline': '对照', 'candidate': '候选'}.get(item.get('branch'), '')
            repeat = f" · 补测第 {item['round']} 轮" if item.get('round') else ''
            body += f"<li>第 {item['case_index']} 题 · {_e(label)} · {_e(_STAGES.get(item['phase'], item['phase']))}{repeat}</li>"
        body += '</ul>'
    if active:
        steps = [('preparing', '准备'), ('generating', '生成'), ('judging', '初判'),
                 ('extracting', '抽特征'), ('predicting', '小模型'), ('checkpoint', '记录'), ('summarizing', '汇总')]
        body += '<div class="stage-steps">' + ''.join(f'<span class="{"current" if key == phase else ""}">{label}</span>' for key, label in steps) + '</div>'
    timings = progress.get('timings') or {}
    if timings:
        body += _details('已记录的阶段耗时', '<p class="help">本次运行累计耗时；包含已完成的等待与重试。正在执行的步骤耗时见上方。</p>' + ''.join(f'<p>{_e(_STAGES.get(key, key))}：{seconds:.1f} 秒</p>' for key, seconds in timings.items() if seconds >= .01))
    elif not active:
        body += '<p class="help">历史任务未记录细分阶段耗时；后续运行将自动记录。</p>'
    return body + '</div>'


def _rate(hits: int | None, total: int) -> str:
    return f'{hits / total:.1%}' if total and hits is not None else '—'


def _rate_cards(run: dict) -> str:
    m, n = run['metrics'], run['pairs']
    judge = run['spec'].get('kind') == 'judge_eval'
    name = '初测正确识别率' if judge else '初测 AI 识别率'
    body = '<div class="rate-grid">'
    for branch, label in [('baseline', '对照'), ('candidate', '候选')]:
        hits = m.get('identified_' + branch)
        value = _rate(hits, n)
        foot = f'{hits} / {n} 道有效题' if n and hits is not None else '尚无有效判定'
        body += _stat(label + name, value, foot)
    processed = n + run['failures']
    rate = _rate(run['failures'], processed)
    body += _stat('执行失败率', rate, f'{run["failures"]} / {processed} 道已处理题') + '</div>'
    rule = '裁判识别越高越好。' if judge else '生成器被识别为 AI 越低越好。'
    body += f'<p class="help">{rule}初测指首次完整评分（来源裁判含小模型校正）；失败题不计入，翻转补测另计。</p>'
    if run['spec'].get('purpose') == 'gen_optimization':
        body += '<p class="notice">策略优化：本题单用于调整生成器，选定策略后须使用开发验证题单比较，不能直接晋升。</p>'
    elif run['spec'].get('source_audit', {}).get('promotion_eligible') is False:
        body += '<p class="notice">诊断对比：旧模型或提示词的学习来源尚未通过时间核验，不能据此晋升或宣称独立验收。</p>'
    cohorts = m.get('cohorts', {})
    if cohorts:
        labels = {'familiar': '熟悉对象', 'unseen_holdout': '未见对象（整聊天留出）', 'observed_first': '观测首轮'}
        body += '<div class="table-wrap"><table><thead><tr><th>场景</th><th>有效 / 计划</th><th>失败</th><th>对照识别率</th><th>候选识别率</th></tr></thead><tbody>'
        for key, group in cohorts.items():
            familiarity, kind = key.split('/')
            label = labels.get(familiarity, familiarity) + (' · 群聊' if kind == 'group' else ' · 私聊')
            body += (f'<tr><td>{_e(label)}</td><td>{group["pairs"]} / {group["attempted"]}</td>'
                     f'<td>{group["failures"]}</td><td>{_rate(group["identified_baseline"], group["pairs"])}</td>'
                     f'<td>{_rate(group["identified_candidate"], group["pairs"])}</td></tr>')
        body += '</tbody></table></div>'
    if run['spec'].get('smoke'):
        body += f'<p class="notice">连接检查 · {run["attempted"]} 题小样本，仅供查看调用结果，不代表完整评测效果。</p>'
    elif judge:
        body += f'<p class="notice">本次为 {run["attempted"]} 题裁判抽样校准。识别率仅代表该冻结评估包；不能代替完整固定验收。</p>'
    elif run['state'].get('status') != 'finished':
        body += '<p class="notice">阶段性结果，随有效题增加而变化，尚不能作为采用结论。</p>'
    return body


def _previous_task(run: dict) -> dict | None:
    order = (run['spec'].get('created', ''), run['spec']['id'])
    for item in _runs(run['dir'].parent):
        if item['spec'].get('kind') == run['spec'].get('kind') and (item['spec'].get('created', ''), item['spec']['id']) < order:
            return item
    return None


def _delta_table(rows: list) -> str:
    if not rows:
        return ''
    return '<div class="table-wrap"><table class="change-table"><thead><tr><th>改动项</th><th>之前</th><th>之后</th></tr></thead><tbody>' + ''.join(
        f'<tr><td>{_e(r["label"])}</td><td>{_e(r["before"])}</td><td>{_e(r["after"])}</td></tr>' for r in rows) + '</tbody></table></div>'


def _delta_html(title: str, delta: dict) -> str:
    heading = f'<h4>{_e(title)} · {_e(delta["before"])} → {_e(delta["after"])}</h4>'
    if not delta['available']:
        return heading + '<p class="notice">版本快照缺失或不完整，无法确认具体差异。</p>'
    if delta['changes']:
        content = _delta_table(delta['changes'])
    else:
        content = '<p>模型行为配置、提示词与快照资产一致。</p>'
    if delta['metadata']:
        content += '<p class="help">以下是快照标记变化，评测输入以任务绑定的数据版本为准：</p>' + _delta_table(delta['metadata'])
    return heading + content


def _local_delta(run: dict) -> dict:
    from .changes import snapshot_delta
    spec = run['spec']
    return snapshot_delta(run['instance_dir'], 'judges' if spec.get('kind') == 'judge_eval' else 'generators',
                          spec.get('baseline_ref', ''), spec.get('candidate_ref', ''))


def _change_summary(run: dict) -> str:
    if run['spec'].get('comparison'):
        return ' → '.join(_comparison_label(run['spec']['comparison'][side]) for side in ('baseline', 'candidate'))
    delta = _local_delta(run)
    if not delta['available']:
        return '缺少版本快照，改动待核对'
    if not delta['changes']:
        return '两版策略相同 · 连接验证' if run['spec'].get('smoke') else '两版策略相同'
    return '；'.join(r['label'] + '：' + r['before'] + ' → ' + r['after'] for r in delta['changes'])


def _iteration_html(run: dict) -> str:
    from .changes import snapshot_delta
    spec, instance = run['spec'], run['instance_dir']
    if spec.get('comparison'):
        return _comparison_configuration(spec)
    judge = spec.get('kind') == 'judge_eval'
    subject = '裁判' if judge else '生成器'
    body = '<div class="iteration-change"><h3>这次迭代了什么</h3>'
    body += f'<p class="help">任务说明（创建时填写）：{_e(_title(spec))}</p>'
    body += _delta_html(f'本轮对比 · 对照{subject}与候选{subject}', _local_delta(run))
    body += f'<p class="help">本轮两版共用数据 {_version_link(instance, "data", spec.get("data_ref"))}。'
    if not judge:
        body += f'评分裁判固定为 {_version_link(instance, "judges", spec.get("judge_ref"))}。'
    body += '</p>'
    previous = _previous_task(run)
    if previous:
        old = previous['spec']
        body += f'<h4>相较上一条同类任务，运行配置有什么变化</h4><p class="help">按创建时间比较：<a href="{_run_url(previous)}">{_e(old["id"])}</a> → {_e(spec["id"])}。这不是本轮候选的改动。</p>'
        folder = 'judges' if judge else 'generators'
        body += _delta_html(f'{subject}实验对照', snapshot_delta(instance, folder, old.get('baseline_ref', ''), spec.get('baseline_ref', '')))
        if not judge:
            body += _delta_html('评分裁判', snapshot_delta(instance, 'judges', old.get('judge_ref', ''), spec.get('judge_ref', '')))
        old_data, new_data = old.get('data_ref'), spec.get('data_ref')
        if old_data != new_data:
            body += _delta_table([{'label': '任务绑定的数据版本', 'before': old_data or '未记录', 'after': new_data or '未记录'}])
            migration = (_json(instance / 'data' / str(new_data) / 'manifest.json').get('metadata_migration') or {})
            if migration.get('source_data') == old_data and migration.get('reason'):
                body += '<p>数据版本说明：' + _e(migration['reason']) + '</p>'
    else:
        body += '<p class="help">这是首条同类任务，没有更早的任务可作运行配置对比。</p>'
    body += '<p class="help">以上按保存的版本配置和文件内容核对。代码修复、外部接口或环境变量值没有随任务保存，不能由模型版本差异推断。</p>'
    body += _details('创建时记录的原始改动说明', _pre(spec.get('config_diff') or []))
    return body + '</div>'


def _latest(run: dict) -> str:
    spec, state = run['spec'], run['state']
    label, tone = _status(spec, state)
    return f'''<section class="panel"><div class="section-head"><h2><span class="section-index">01</span>最近任务</h2>{_badge(label, tone)}</div>
<div class="meta"><span>{_e(_phase(spec)[0])}</span><span>·</span><span>{_e(spec.get('created', ''))}</span></div>
<h3 class="latest-title">{_e(_title(spec))}</h3>{_baseline_context(run)}<p class="change-summary">{_e(_change_summary(run))}</p>{_stage_html(run)}{_rate_cards(run)}{_iteration_html(run)}
{_error_html(run)}<p class="next-step">{_e(_next_step(run))}</p><a class="button primary" href="{_run_url(run)}">查看任务与逐题详情 →</a></section>'''


def _list_controls(search_label: str, options: list[tuple[str, str]],
                   *filter_groups: list[tuple[str, str]]) -> str:
    """搜索框 + 多个筛选下拉；每个附加组用 data-filter-attr 指定行数据属性。"""
    attrs = ('status', 'object', 'size')
    toolbar = (f'<div class="toolbar"><input type="search" aria-label="{_e(search_label)}" placeholder="{_e(search_label)}">'
               f'<select aria-label="筛选类型">'
               + ''.join(f'<option value="{_e(value)}">{_e(label)}</option>' for value, label in options)
               + '</select>')
    for attr, group in zip(attrs, filter_groups):
        opts = ''.join(f'<option value="{_e(value)}">{_e(label)}</option>' for value, label in group)
        toolbar += f'<select data-filter-attr="{attr}" aria-label="筛选">{opts}</select>'
    return toolbar + '</div>'


def _pager() -> str:
    return '<p class="empty" data-no-results hidden>没有符合条件的记录。</p><div class="pager"><span data-list-count aria-live="polite"></span><div class="pager-actions"><button class="button" data-previous>上一页</button><span data-page-label></span><button class="button" data-next>下一页</button></div></div>'


def _dashboard_intro(runs: list, instance: str) -> str:
    valid = [r for r in runs if not r['spec'].get('smoke') and r['state'].get('status') == 'finished' and r['state'].get('verdict') != 'experiment_incomplete' and r['pairs'] > 0]
    if not runs:
        headline, description = '准备开始第一次检查', '当前还没有实验记录。先用少量题目确认生成与评分流程能正常完成。'
    elif is_active(_current_run(runs)['state'].get('progress') or {}):
        headline, description = '任务正在运行', '下方实时显示当前阶段、已处理题数和有效题的识别率。'
    elif not valid and runs[0]['spec'].get('smoke') and runs[0]['state'].get('status') == 'finished':
        headline, description = '连接与流程检查已通过', '当前配置已完成生成、裁判评分与结果记录；下一步可以准备完整开发评测。连接检查不代表效果改善。'
    elif not valid:
        headline = '连接与流程检查尚未通过' if all(r['spec'].get('smoke') for r in runs) and not any(r['state'].get('status') == 'finished' for r in runs) else '尚无完整评测结论'
        description = '先确认任务能正常完成，再判断候选是否更好。连接检查不计入模型效果结论。'
    else:
        headline, description = '查看当前配置与最近评测', f'已有 {len(valid)} 次完成的有效评测，样本规模与具体结论请查看对应实验；不同数据、基线或裁判的结果不能直接排名。'
    hero = f'<div class="hero"><div class="hero-copy"><p class="eyebrow">数字人 · {_e(instance)}</p><h1>{headline}</h1><p class="muted">{description}</p></div></div>'
    failed = [r for r in runs if r['state'].get('verdict') == 'experiment_incomplete']
    resolved = sum(bool((r['state'].get('resolution') or {}).get('verified_by')) for r in failed)
    stats = f'<div class="summary-counts"><span>任务记录 <strong>{len(runs)}</strong></span><span>有效评测 <strong>{len(valid)}</strong></span><span>待处理失败 <strong>{len(failed) - resolved}</strong></span><span class="help">保留历史失败 {len(failed)} 条，其中 {resolved} 条已由新配置验证</span></div>'
    return hero + stats


def _initial_net(run: dict) -> int | None:
    """Initial net uses the same counts as initial accuracy, never confirmed votes."""
    baseline = run['metrics'].get('identified_baseline')
    candidate = run['metrics'].get('identified_candidate')
    if not run['pairs'] or baseline is None or candidate is None:
        return None
    return candidate - baseline if run['spec'].get('kind') == 'judge_eval' else baseline - candidate


def _history_rates(run: dict) -> str:
    n = run['pairs']
    body = ''
    for branch, label in [('baseline', '对照'), ('candidate', '候选')]:
        hits = run['metrics'].get('identified_' + branch)
        count = f'{hits}/{n}' if n and hits is not None else '无有效判定'
        body += f'<div class="history-rate"><span>{label}</span><strong>{_rate(hits, n)}</strong><small>{_e(count)}</small></div>'
    direction = '正确识别 · 越高越好' if run['spec'].get('kind') == 'judge_eval' else '被识别为 AI · 越低越好'
    body += f'<div class="table-sub">{direction}</div>'
    net = _initial_net(run)
    body += '<div class="history-initial-net">初测净胜 <strong>' + (f'{net:+d}' if net is not None else '—') + '</strong></div>'
    if n and run['state'].get('status') != 'finished':
        body += '<div class="table-sub">阶段性结果</div>'
    return body


def _history_versions(run: dict) -> str:
    spec = run['spec']
    if spec.get('comparison'):
        sides = ''.join(f'<dt>{label}</dt><dd>{_e(_comparison_label(spec["comparison"][side]))}</dd>'
                        for side, label in [('baseline', '本轮对照'), ('candidate', '本轮候选')])
        return ('<dl class="history-versions">' + sides + '<dt>数据</dt><dd>'
                + _version_link(run['instance_dir'], 'data', spec.get('data_ref'))
                + '</dd><dt>生成器</dt><dd>'
                + _version_link(run['instance_dir'], 'generators', spec.get('generator_ref')) + '</dd></dl>')
    folder = 'judges' if spec.get('kind') == 'judge_eval' else 'generators'

    def version(ref, kind=folder):
        return _version_link(run['instance_dir'], kind, ref)

    comparison = (f'<span class="version-comparison"><small>本轮对照</small>{version(spec.get("baseline_ref"))}</span>'
                  f'<span class="version-comparison"><small>本轮候选</small>{version(spec.get("candidate_ref"))}</span>')
    if spec.get('kind') == 'judge_eval':
        pack_ref = spec.get('pack_ref')
        pack = _json(run['instance_dir'] / 'judge_eval' / pack_ref / 'pack.json') if pack_ref else {}
        generator, judge = version(pack.get('c0_gen_version'), 'generators'), comparison
    else:
        generator, judge = comparison, version(spec.get('judge_ref'), 'judges')
    entries = [('数据版本', version(spec.get('data_ref'), 'data')), ('生成器', generator), ('Judge', judge)]
    return '<dl class="history-versions">' + ''.join(
        f'<dt>{label}</dt><dd>{value}</dd>' for label, value in entries) + '</dl>'


def _history_net(run: dict) -> str:
    m = run['metrics']
    net = m.get('net_win_confirmed')
    if not run['pairs'] or net is None:
        return '—'
    detail = (f"补验后：胜 {m['wins_confirmed']} / 负 {m['losses_confirmed']}"
              if 'wins_confirmed' in m and 'losses_confirmed' in m else '')
    if m.get('contested') is not None:
        detail += f" · 打平 {m['contested']}"
    body = f'<strong>{net:+d}</strong><div class="table-sub">{_e(detail)}</div>'
    initial = _initial_net(run)
    if initial is not None and initial * net < 0:
        body += '<div class="table-sub">初测与补验方向不同</div>'
    return body


def _history_conclusion(run: dict) -> str:
    """Display stored decisions; use the same frozen gate for diagnostic numbers."""
    spec, state = run['spec'], run['state']
    if spec.get('smoke'):
        return '连接检查，不参与晋级'
    if state.get('status') != 'finished' or state.get('verdict') == 'experiment_incomplete' or not run['pairs']:
        return '结果未完整，暂无结论'
    diagnostic = gates.diagnostic_reason(spec)
    verdict = state.get('verdict')
    if diagnostic:
        try:
            verdict = protocol.decide(run['metrics'], spec.get('dataset', 'development'),
                                      spec['protocol'], higher_is_better=spec.get('kind') == 'judge_eval')['verdict']
        except (KeyError, TypeError, ValueError):
            return '诊断记录，缺少完整门槛依据'
    if spec.get('dataset') == 'fixed_test':
        text, tone = ('固定验收达标', 'success') if verdict == 'adopt' else ('固定验收未达标', 'danger')
    elif verdict == 'merge_to_iteration_baseline':
        text, tone = ('开发收益达标', 'success') if diagnostic else ('开发晋级达标', 'success')
    elif verdict in ('reject', 'observe'):
        text, tone = '开发未达标', 'neutral'
    else:
        text, tone = '暂无有效晋级结论', 'neutral'
    detail = '历史比较；数值达标与实际采用分开记录' if spec.get('comparison') else diagnostic
    if state.get('threshold_review'):
        review = state['threshold_review']
        detail = ('按修订门槛 ' + format(review['protocol']['formal_min_net_win_rate'], '.1%') +
                  ' 复核；原协议结论：' + review['original_verdict'])
    if spec.get('against_production'):
        detail = '用于固定轮准入，不重复晋级开发版'
    return _badge(text, tone) + (f'<div class="table-sub">{_e(detail)}</div>' if detail else '')


def _branch_development(run: dict) -> str | None:
    spec = run['spec']
    branch = str(spec.get('branch_round', '')).split('/')[0]
    if not branch or branch in ('.', '..'):
        return None
    state = _json(run['instance_dir'] / 'branches' / branch / 'state.json')
    accepted = state.get('development') or {}
    ptr = _json(run['instance_dir'] / 'pointers.json')
    current_basis = {key: ptr.get(key) for key in ('data', 'production_gen', 'production_judge')}
    if (accepted.get('candidate_ref') == spec.get('candidate_ref')
            and accepted.get('experiment_id') and accepted.get('basis') == current_basis):
        return branch
    return None


def _history_adoption(run: dict) -> str:
    spec = run['spec']
    ptr = _json(run['instance_dir'] / 'pointers.json')
    suffix = 'judge' if spec.get('kind') == 'judge_eval' else 'gen'

    def roles(ref):
        if not ref:
            return []
        return [label for key, label in [('production_', '当前生产版'), ('iteration_', '当前开发版')]
                if ptr.get(key + suffix) == ref]

    current = roles(spec.get('candidate_ref'))
    branch = _branch_development(run)
    if branch:
        current.append(f'分支 {branch} 当前开发版')
    if current:
        text = '候选：' + '、'.join(current)
    elif (spec.get('branch_round') and
          ptr.get('branch_promotions', {}).get(spec['id']) == spec.get('candidate_ref')):
        text = '本轮曾推全，当前版本已前进'
    elif spec.get('comparison'):
        text = '本比较未绑定候选版本，未执行采用'
    else:
        text = '候选：当前未采用'
    baseline = roles(spec.get('baseline_ref'))
    if baseline:
        text += '；对照：' + '、'.join(baseline)
    return _e(text)


def _status_bucket(spec: dict, state: dict) -> str:
    """列表状态筛选桶：running / passed / failed / done（已完成其他）。"""
    if is_active(state.get('progress') or {}) or state.get('status') != 'finished':
        return 'running'
    if state.get('verdict') == 'experiment_incomplete' and not (state.get('resolution') or {}).get('verified_by'):
        return 'failed'
    if state.get('verdict') in ('adopt', 'merge_to_iteration_baseline'):
        return 'passed'
    return 'done'


def _promoted(run: dict, ptr: dict) -> bool:
    """已晋升 = 指针切换或推全凭证等硬证据；数值达标（verdict）不算晋升。"""
    spec = run['spec']
    ref = spec.get('candidate_ref')
    if not ref or spec.get('comparison'):
        return False
    if ptr.get('branch_promotions', {}).get(spec.get('id')) == ref:
        return True
    return any(ptr.get(key) == ref for key in
               ('production_gen', 'iteration_gen', 'production_judge', 'iteration_judge'))


def _history_html(runs: list) -> str:
    rows = []
    ptr = _json(runs[0]['instance_dir'] / 'pointers.json') if runs else {}
    for run in runs:
        spec, state = run['spec'], run['state']
        label, tone = _status(spec, state)
        phase, key = _phase(spec)
        bucket = _status_bucket(spec, state)
        promoted = _promoted(run, ptr)
        change = _details('具体改动与基线关系', _baseline_context(run, compact=True) + _iteration_html(run), key='task-change-' + spec['id'])
        title = (_change_summary(run) if spec.get('comparison') else _title(spec))
        processed = run['pairs'] + run['failures']
        failure_rate = _rate(run['failures'], processed)
        promoted_badge = _badge('已晋升', 'success') if promoted else ''
        rows.append(f'''<tr data-item data-kind="{key}" data-status="{bucket}" data-promoted="{'true' if promoted else 'false'}"
 data-object="{'judge' if spec.get('kind') == 'judge_eval' else 'gen'}" data-size="{'full' if run['attempted'] >= 50 else 'small'}">
<td><a class="task-link" href="{_run_url(run)}">{_e(title)}</a><div class="table-sub">{_e(spec.get("created", ""))} · {_e(phase)}</div>{_history_versions(run)}{change}</td>
<td>{_history_rates(run)}</td><td>{_history_net(run)}{_history_conclusion(run)}</td>
<td>{promoted_badge}{_badge(label, tone)}<div class="table-sub">有效 {run["pairs"]} / {run["attempted"]}<br>失败 {run["failures"]} · {failure_rate}</div>{_history_adoption(run)}</td></tr>''')
    history = '<section id="history" class="panel" data-list data-page-size="10"><div class="section-head"><h2><span class="section-index">02</span>实验记录</h2><span class="help">按时间倒序 · 点击任务进入流程与逐题详情</span></div>'
    history += _list_controls('搜索任务名称、时间或版本', [('all', '全部任务'), ('development', '开发评测'), ('fixed', '固定验收'), ('comparison', '专项比较'), ('smoke', '连接检查')],
                              [('all', '全部状态'), ('running', '运行中'), ('passed', '达标'), ('done', '未达标及其他'), ('failed', '执行失败'), ('promoted', '已晋升')],
                              [('all', '全部对象'), ('gen', '生成器'), ('judge', '裁判')],
                              [('all', '全部规模'), ('full', '完整评测'), ('small', '小样本（<50 题）')])
    history += ('<label class="help" style="margin-left:10px"><input type="checkbox" data-show-smoke> '
                '显示连接检查</label>'
                '<label class="help" style="margin-left:10px"><input type="checkbox" data-show-comparison> '
                '显示专项比较</label>')
    history += '<p class="help">所有净胜均为候选相对本轮对照，正数表示候选更好。本轮对照是该实验记录的比较对象，不一定是当前基线。初测识别率和初测净胜来自同一批有效题的首次判定；Judge 初测净胜＝候选正确数−对照正确数，生成器方向相反。</p>'
    history += '<p class="help">补验确认净胜＝初测分歧题经补验后的胜题−负题，按冻结协议汇总双方多数票；补验后打平不计胜负。因此补验净胜可能与初测方向不同，不能用初测识别率之差解释。晋级结论按当时冻结协议显示；历史专项比较单列开发收益是否达标，补录不会自动采用。实际采用读取当前指针及分支开发版记录。点击版本号和详情查看依据。</p>'
    history += '<div class="table-wrap"><table class="history"><thead><tr><th>任务 · 比较对象 · 版本</th><th>初测识别率 · 初测净胜</th><th>确认净胜 · 晋级结论</th><th>执行状态 · 实际采用（当前）</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>' + _pager() + '</section>'
    return history


def _region(name: str, body: str) -> str:
    return f'<div data-live-region="{name}">{body}</div>'


def dashboard_html(exp_root: Path) -> str:
    """主页 = 当前基线 + 实验记录（含状态筛选）；流程与机制点进实验/版本查看。"""
    runs = _runs(exp_root)
    instance = exp_root.parent
    name = instance.name
    intro = _region('intro', _dashboard_intro(runs, name))
    overview = (_region('configuration', _baseline_summary(instance))
                + _region('history', _history_html(runs)))
    return _page(name, intro + overview + _workflows_region(instance, runs), name,
                 f'/api/live/{quote(name, safe="")}')


def _workflows_region(instance: Path, runs: list) -> str:
    """并行训练与分支工作流：主页默认折叠，保留实时 region。"""
    from . import workflow_view
    return _region('workflows', '<details class="panel"><summary><h2>并行训练与分支工作流（点开查看）</h2></summary>'
                   + _pack_preparation_html(instance) + workflow_view.html(instance, runs, render_stage=_stage_html)
                   + '</details>')


def refresh_dashboard(exp_root: Path, dashboard_dir: Path) -> None:
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(dashboard_dir / 'index.html', dashboard_html(exp_root))


def write_data_browser(instance_dir: Path, data_ref: str, dashboard_dir: Path) -> Path:
    """数据版本浏览器：渲染开发集逐题（上下文+真人回复，可点看）。
    开发集可审计在线看；固定集维持 SOP §5 线下审计，不在线渲染。"""
    data_dir = instance_dir / "data" / data_ref
    out = dashboard_dir / f"data-{data_ref}.html"
    rows = []
    dev = data_dir / "dev_pool.jsonl"
    cases = []
    if dev.exists():
        for line in dev.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    for i, c in enumerate(cases, 1):
        chat = "".join(
            f'<div class="chat-line"><span class="sender">{_e(m.get("sender", "?"))}</span>'
            f'<span class="message">{_e(m.get("text", ""))}</span></div>'
            for m in c.get("context", []))
        body = (f'<div class="case-id">Case ID：{_e(c["case_id"])} · {len(c.get("context", []))} 条上下文</div>'
                f'<h3>聊天上下文</h3><div class="chat">{chat or "<p class=muted>无上下文</p>"}</div>'
                f'<div class="reply human"><div class="reply-label">真人回复</div>'
                + "".join(f"<p>{_e(r)}</p>" for r in (c.get("human_reply") or []))
                + "</div>")
        rows.append(f'<details class="case"><summary><span class="case-summary">'
                    f'<span class="case-title">第 {i} 题 · {_e(c.get("chat_type", ""))}</span>'
                    f'<span class="case-preview">{_e(" / ".join(c.get("human_reply") or [])[:60])}</span>'
                    f'</span></summary><div class="detail-body">{body}</div></details>')
    head = (f'<p class="eyebrow">DATA</p><h1>数据版本 {_e(data_ref)} · 开发集浏览</h1>'
            f'<p class="muted">共 {len(cases)} 题（群聊 {sum(1 for c in cases if c.get("chat_type") == "group")} / 私聊 '
            f'{sum(1 for c in cases if c.get("chat_type") != "group")}）。固定集不在线展示（SOP §5）。</p>')
    page = _page(f"数据 {data_ref}", head + '<section class="panel" data-list data-page-size="20">' + "".join(rows) + "</section>",
                 instance=instance_dir.name)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(out, page)
    return out


def write_instance_index(instances_dir: Path, dashboard_dir: Path) -> None:
    cards = []
    for path in sorted(instances_dir.glob('*/pointers.json')):
        instance_dir = path.parent
        runs = _runs(instance_dir / 'experiments')
        status = _badge(*_status(runs[0]['spec'], runs[0]['state'])) if runs else _badge('暂无任务')
        latest = _title(runs[0]['spec']) if runs else '先完成一次连接检查'
        cards.append(f'<section class="panel"><div class="section-head"><h2>{_e(instance_dir.name)}</h2>{status}</div><p>{_e(latest)}</p><p class="help">{len(runs)} 条任务记录</p><a class="button primary" href="/dashboard/{quote(instance_dir.name, safe="")}/index.html">进入实例 →</a></section>')
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    body = '<p class="eyebrow">WORKSPACE</p><h1>数字人实例</h1><p class="muted">选择一个实例，查看配置、执行情况与评测记录。</p><div class="version-grid">' + (''.join(cards) or '<p class="empty">暂无实例。</p>') + '</div>'
    atomic_write(dashboard_dir / 'index.html', _page('全部实例', body))


def _identified(value: Any, judge: bool) -> str:
    if value is None:
        return '未完成判定'
    if judge:
        return '正确识别出 AI' if value else '未识别出 AI'
    return '被识别为 AI' if value else '未被识别为 AI'


def _comparison(b: Any, c: Any, judge: bool) -> tuple[str, str]:
    if b is None or c is None:
        return '判定未完成', 'pending'
    if b == c:
        return '两版结果一致', 'tie'
    better = bool(c) if judge else not bool(c)
    return (('候选识别更准' if judge else '候选更像真人'), 'win') if better else (('候选识别更差' if judge else '候选更易被识别'), 'loss')


def _case_result(row: dict, judge: bool) -> tuple[str, str]:
    if row.get('status') == 'failed':
        return '执行失败', 'failed'
    flip = row.get('flip_verified')
    if flip:
        final = row if judge else flip
        return _comparison(final.get('baseline_identified_final'), final.get('candidate_identified_final'), judge)
    keys = ('baseline_correct', 'candidate_correct') if judge else ('identified_baseline', 'identified_candidate')
    return _comparison(row.get(keys[0]), row.get(keys[1]), judge)


def _reply(label: str, replies: list | None, css: str = '', result: str = '') -> str:
    content = ''.join(f'<p>{_e(reply)}</p>' for reply in replies or []) or '<p class="muted">未记录回复</p>'
    footer = f'<p class="reply-result">初测：{_e(result)}</p>' if result else ''
    return f'<div class="reply {css}"><div class="reply-label">{_e(label)}</div>{content}{footer}</div>'


def _case_html(row: dict, index: int, judge: bool, raw: list[dict], trace_base: str = '') -> str:
    anchor = 'case-' + quote(str(row['case_id']), safe='')
    links = (f'<span class="case-actions"><a data-case-link href="#{anchor}" '
             f'aria-label="打开第 {index} 题的直达链接">题目链接</a>'
             '<button type="button" class="button" data-copy-case-link aria-live="polite">复制链接</button></span>')
    label, key = _case_result(row, judge)
    tone = {'failed': 'danger', 'loss': 'warning', 'win': 'success'}.get(key, 'neutral')
    # 徽章永远是补验后最终判定；与回复区“初测”脚注并列时容易误读，在徽章后显式标注。
    flip_note = '<small>（补验后）</small>' if isinstance(row.get('flip_verified'), dict) else ''
    context = row.get('context') or []
    preview = str(context[-1].get('text', '')) if context else '展开查看输入、回复和判定'
    kind = {'private': '私聊', 'group': '群聊'}.get(row.get('chat_type'), '裁判评测' if judge else '对话')
    def message_html(m):
        role = '本人 · 被模仿对象' if m.get('is_self') is True else ('对方' if m.get('is_self') is False else '角色未记录')
        stamp = m.get('timestamp')
        if isinstance(stamp, (int, float)):
            seconds = stamp / 1000 if stamp > 10_000_000_000 else stamp
            stamp = datetime.fromtimestamp(seconds, ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        return f'<div class="chat-line"><span class="sender">{_e(m.get("sender", "未记录"))}<small>{_e(role)}</small></span><div class="message">{_e(m.get("text", ""))}<small>{_e(stamp) if stamp else ""}</small></div></div>'
    chat = ''.join(message_html(m) for m in context)
    body = f'<div class="case-id">Case ID：{_e(row["case_id"])} · 保存记录 {len(raw)} 条</div><h3>聊天上下文</h3><div class="chat">{chat or "<p class=muted>该记录没有保存上下文。</p>"}</div>'
    if context and context[-1].get('is_self') is True:
        body += '<p class="notice">数据边界异常：上下文最后一条来自本人。当前样本可能把本人的连续回复拆成了新问题，需要核对待回复消息；这不是清洗已通过的样本。</p>'
    body += _details('样本字段与来源信息', _pre({k: v for k, v in row.items() if k not in {'trace_ref'}}))
    from .trace_view import slot
    attempts = [r for r in raw if r.get('trace_ref')]
    if row.get('trace_ref') and not any(r['trace_ref'] == row['trace_ref'] for r in attempts):
        attempts.append(row)
    if trace_base and attempts:
        for n, attempt in enumerate(reversed(attempts)):
            title = '实际提示词与调用过程' if n == 0 else f'此前执行的调用实录 · {len(attempts) - n}'
            body += slot(trace_base + '/' + quote(attempt['trace_ref'], safe=''), title)
    elif row.get('draws'):
        body += _details('复用的抽特征调用来源',
            '<p class="help">本项使用已有调用产物计算分类器预测。下面保留原调用引用、轮次和内容哈希；两侧预测见样本字段，不代表重新调用大模型。</p>' + _pre(row['draws']))
    else:
        body += _details('实际提示词与调用过程 · 历史未记录', '<p class="notice">此历史任务没有保存实际调用实录，无法还原当时完整提示词、A/B 换位、模型返回正文和抽取特征。页面下方的版本提示词是模板，不能当作本题实际输入。新的执行会从发送请求前开始记录。</p>')
    if row.get('status') == 'failed':
        body += '<div class="error-box"><h3>本题失败原因</h3>' + _pre(row.get('reason') or '未记录') + '</div>'
    body += _reply('真人实际回复', row.get('human_reply'), 'human')
    keys = ('baseline_correct', 'candidate_correct') if judge else ('identified_baseline', 'identified_candidate')
    b, c = row.get(keys[0]), row.get(keys[1])
    if judge:
        body += _reply('交给两版裁判识别的 AI 回复', row.get('ai_replies'))
        body += '<div class="reply-grid">' + _reply('对照裁判', [_identified(b, True)]) + _reply('候选裁判', [_identified(c, True)], 'candidate') + '</div>'
    else:
        body += '<div class="reply-grid">' + _reply('对照生成器回复', row.get('baseline_replies'), result=_identified(b, False)) + _reply('候选生成器回复', row.get('candidate_replies'), 'candidate', _identified(c, False)) + '</div>'
    flip = row.get('flip_verified')
    verification = '<h3>补测与最终判定</h3>'
    if row.get('status') == 'failed':
        verification += '<p>本题执行未完成，不计入有效结果。</p>'
    elif row.get('flip_verification_skipped') == 'smoke':
        verification += '<p>连接检查只验证生成和评分，不做胜负补测，不形成效果结论。</p>'
    elif flip:
        verification += f'<p>初测：{_e(_comparison(b, c, judge)[0])}；最终：{_e(label)}。</p>'
        if key == 'tie':
            verification += '<p>补测后两版结果一致，本题不计入确认净胜。</p>'
        final = row if judge else flip
        verification += '<dl><dt>对照最终判定</dt><dd>' + _e(_identified(final.get('baseline_identified_final'), judge)) + '</dd><dt>候选最终判定</dt><dd>' + _e(_identified(final.get('candidate_identified_final'), judge)) + '</dd></dl>'
        if isinstance(flip, dict):
            for k, title in [('baseline_votes', '对照历次判定'), ('candidate_votes', '候选历次判定')]:
                if flip.get(k):
                    verification += f'<p>{title}（含初测）：{_e(" → ".join(_identified(v, judge) for v in flip[k]))}</p>'
    else:
        verification += f'<p>未进行补测。初测：{_e(_comparison(b, c, judge)[0])}。</p>'
    body += '<div class="verification">' + verification + '</div>'
    timing = {label: row[k] for k, label in [('baseline_latency_ms', '对照耗时（毫秒）'), ('candidate_latency_ms', '候选耗时（毫秒）')] if k in row}
    if timing:
        body += '<p class="help">' + ' · '.join(f'{k}：{_e(v)}' for k, v in timing.items()) + '</p>'
    body += _details('查看原始记录与重试历史', _pre(raw))
    return f'<details class="case" data-item data-case-id="{_e(row["case_id"])}" data-kind="{key}" id="{anchor}"><summary><span class="case-summary"><span class="case-title">第 {index} 题 · {kind}</span><span class="case-preview">{_e(preview)}</span></span>{_badge(label, tone)}{flip_note}{links}</summary><div class="detail-body">{body}</div></details>'


def _cases_html(run: dict) -> str:
    if run['spec'].get('dataset') == 'fixed_test':
        return '<section class="panel" id="cases"><h2>逐题详情</h2><p class="notice">固定验收仅展示汇总结果。固定集答案不在开发后台展开。</p></section>'
    records = list(run['records'])
    progress = run['state'].get('progress') or {}
    pending_refs = [progress.get('trace_ref')] + [item.get('trace_ref') for item in (progress.get('active_cases') or {}).values()]
    for pending in pending_refs:
        if pending and not any(r.get('trace_ref') == pending for r in records):
            doc = tracing.read(run['dir'], pending)
            row = {**doc['case'], 'trace_ref': pending, 'status': 'pending'}
            records = [r for r in records if str(r['case_id']) != str(row['case_id'])] + [row]
    if not records:
        return '<section class="panel" id="cases"><h2>逐题详情</h2><p class="empty">尚未产生逐题记录。</p></section>'
    histories: dict[str, list] = {}
    cases_file = run['dir'] / 'cases.jsonl'
    for line in (cases_file.read_text(encoding='utf-8') if cases_file.exists() else '').splitlines():
        if line.strip():
            row = json.loads(line)
            histories.setdefault(str(row['case_id']), []).append(row)
    judge = run['spec'].get('kind') == 'judge_eval'
    inputs = {}
    if judge and run['spec'].get('pack_ref'):
        pack = _json(run['instance_dir'] / 'judge_eval' / run['spec']['pack_ref'] / 'pack.json')
        inputs = {str(r['case_id']): r for r in pack.get('rows', [])}
    elif run['spec'].get('data_ref'):
        source = run['instance_dir'] / 'data' / run['spec']['data_ref'] / 'dev_pool.jsonl'
        if source.exists():
            inputs = {str(r['case_id']): r for r in map(json.loads, filter(str.strip, source.read_text(encoding='utf-8').splitlines()))}
    ordinals = {cid: index for index, cid in enumerate(inputs, 1)}
    if ordinals:
        records.sort(key=lambda row: ordinals.get(str(row['case_id']), len(ordinals) + 1))
    options = [('all', '全部结果'), ('failed', '执行失败'), ('win', '候选更好'), ('loss', '候选更差'), ('tie', '两版一致'), ('pending', '判定未完成')]
    body = '<section class="panel" id="cases" data-list data-page-size="20"><div class="section-head"><h2>逐题详情<span class="count">' + str(len(records)) + ' 题</span></h2><span class="help">点击每题展开</span></div>'
    body += '<p class="help">展开每题查看消息角色、来源、实际提示词、模型返回、裁判特征和补测过程。点击“复制链接”可分享具体题目，打开链接会自动定位并展开。长调用记录点击后加载；历史任务未保存的细节会明确标注。</p>'
    body += _list_controls('搜索聊天内容、回复或 case ID', options)
    for index, row in enumerate(records, 1):
        merged = {**inputs.get(str(row['case_id']), {}), **row}
        trace_base = f'/api/trace/{quote(run["instance_dir"].name, safe="")}/{quote(run["spec"]["id"], safe="")}'
        body += _case_html(merged, ordinals.get(str(row['case_id']), index), judge, histories.get(str(row['case_id']), []), trace_base)
    return body + _pager() + '</section>'


def _version_html(instance_dir: Path, ref: str, judge: bool, title: str) -> str:
    directory = instance_dir / ('judges' if judge else 'generators') / ref
    config = _json(directory / 'config.json')
    if not config:
        return _details(title, '<p class="muted">版本快照不存在。</p>')
    body = f'<p><a class="button" href="{_version_url(instance_dir, "judges" if judge else "generators", ref)}">打开版本详情 · {_e(ref)}</a></p>'
    body += _model_rows(config) + (_judge_pipeline(config, directory) if judge else _generator_pipeline(config)) + _details('完整行为配置', _pre(config))
    for filename, label in [('persona.md', '人格与回复规则'), ('prompt.md', '裁判提示词')]:
        path = directory / filename
        if path.exists():
            body += _details(label, _pre(path.read_text(encoding='utf-8')))
    for path in sorted((directory / 'scenarios').glob('*.md')):
        body += _details('场景规则 · ' + path.stem, _pre(path.read_text(encoding='utf-8')))
    if normalize_mode(config.get('mode')) == 'corrected_pairwise':
        body += _details('导入来源配置', _pre(_json(directory / 'source_judge.json')))
        body += _details('来源裁判规则', _pre(_json(directory / 'profile.json')))
        body += _details('裁判参考资料', _pre(_json(directory / 'reference.json')))
        correction = _json(directory / 'correction.json')
        body += _details('冻结校正模型与特征', _pre({'final_model': correction.get('final_model'), 'feature_system': correction.get('feature_system')}))
    meta = _json(directory / 'meta.json')
    if meta:
        body += _details('版本来源', _pre(meta))
    return _details(f'{title} · {ref}', body, key='version-' + title + '-' + ref)


def _configuration_html(run: dict) -> str:
    spec, instance = run['spec'], run['instance_dir']
    if spec.get('comparison'):
        return '<section class="panel" id="configuration"><h2>本次比较配置</h2>' + _comparison_configuration(spec) + '</section>'
    judge = spec.get('kind') == 'judge_eval'
    b, c = str(spec.get('baseline_ref', '')), str(spec.get('candidate_ref', ''))
    folder = instance / ('judges' if judge else 'generators')
    body = '<section class="panel" id="configuration"><h2>本次改动与版本快照</h2><p class="help">这里展示实验创建时使用的版本，可能与当前生产版本不同。</p>' + _iteration_html(run)
    if judge:
        body += '<div class="mechanism-grid">' + ''.join(f'<div><h3>{label} · {_e(ref)}</h3>' + _judge_pipeline(_json(folder / ref / 'config.json'), folder / ref) + '</div>' for ref, label in [(b, '对照裁判'), (c, '候选裁判')]) + '</div>'
    elif spec.get('judge_ref'):
        path = instance / 'judges' / spec['judge_ref']
        body += f'<h3>本轮评分机制 · {_e(spec["judge_ref"])}</h3>' + _judge_pipeline(_json(path / 'config.json'), path)
    body += '<div class="version-grid">' + _version_html(instance, b, judge, '对照版本') + _version_html(instance, c, judge, '候选版本') + '</div>'
    assets = {p.relative_to(folder / ref) for ref in [b, c] if ref for p in (folder / ref).rglob('*.md')}
    diffs = []
    for rel in sorted(assets):
        bp, cp = folder / b / rel, folder / c / rel
        old = bp.read_text(encoding='utf-8').splitlines() if bp.exists() else []
        new = cp.read_text(encoding='utf-8').splitlines() if cp.exists() else []
        diff = '\n'.join(difflib.unified_diff(old, new, fromfile=f'对照/{rel}', tofile=f'候选/{rel}', lineterm=''))
        if diff:
            diffs.append(_details(str(rel), _pre(diff)))
    body += _details('人格、提示词与场景的逐行差异', ''.join(diffs) or '<p class="muted">两版文本内容一致。</p>')
    if not judge and spec.get('judge_ref'):
        body += _version_html(instance, spec['judge_ref'], True, '本轮评分裁判')
    return body + '</section>'


def _comparison_label(arm):
    def compact(value):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):  # 扫描臂直接存参数字典的形态：禁止 str(dict) 进标题
            return ' '.join(f'{k}={v}' for k, v in value.items())
        return str(value)
    policy = {'pure': '纯分类器', 'hybrid': '初判与分类器合并'}.get(arm.get('policy'), arm.get('policy', ''))
    named = ' · '.join(compact(v) for v in (arm.get('arm'), arm.get('recipe'), policy) if v)
    if named:
        return named
    # 无 arm/recipe 的扫描臂：用参数紧凑摘要，禁止把原始 dict 序列化进标题
    params = arm.get('parameters') or {}
    return ' · '.join(f'{k}={v}' for k, v in params.items()) or '未命名比较臂'


def _comparison_configuration(spec):
    body = '<div class="version-grid">'
    for side, label in [('baseline', '本轮对照'), ('candidate', '本轮候选')]:
        arm = spec['comparison'][side]
        body += '<div><h3>' + label + '</h3><p>' + _e(_comparison_label(arm)) + '</p>' + _pre(arm.get('parameters', {})) + '</div>'
    body += '</div>'
    if spec.get('provenance', {}).get('mode') == 'historical_import':
        body += '<p class="help">由原始逐题结果核验后补录，保留来源哈希。补录不代表重新运行或重新认证原训练过程。</p>'
    return body


def _metrics_html(run: dict) -> str:
    spec, state, m = run['spec'], run['state'], run['metrics']
    body = '<section class="panel" id="results"><h2>识别率与评测结果</h2>' + _rate_cards(run)
    if not run['pairs']:
        return body + '<div class="empty"><strong>暂无有效评测结果</strong>有效题数为 0，无法计算可比较的识别率与净胜。</div></section>'
    if spec.get('smoke'):
        return body + '</section>'
    judge = spec.get('kind') == 'judge_eval'
    heading = '正确识别数' if judge else '被识别为 AI 的题数'
    rule = '裁判识别正确的题数越多越好。' if judge else '生成器被识别为 AI 的题数越少越好。'
    if state.get('status') != 'finished' or state.get('verdict') == 'experiment_incomplete':
        body += '<p class="notice">本轮尚未完成，以下为已保存的部分结果，不能作为采用结论。</p>'
    body += f'<p class="help">{rule}确认净胜根据补测后的最终胜负计算。</p><div class="metric-grid">'
    for label, value in [(f'对照{heading}', f'{m.get("identified_baseline", "—")} / {run["pairs"]}'), (f'候选{heading}', f'{m.get("identified_candidate", "—")} / {run["pairs"]}'), ('确认净胜', f'{m["net_win_confirmed"]:+d}' if 'net_win_confirmed' in m else '—')]:
        body += f'<div class="metric"><span>{_e(label)}</span><strong>{_e(value)}</strong></div>'
    body += '</div><p class="help">确认胜 ' + _e(m.get('wins_confirmed')) + ' · 确认负 ' + _e(m.get('losses_confirmed')) + ' · 补测后打平 ' + _e(m.get('contested')) + '</p>'
    if state.get('status') == 'finished' and state.get('verdict') != 'experiment_incomplete' and m.get('sign_p') is not None:
        body += _details('统计细节', f'<p>符号检验 p 值：{m["sign_p"]:.4f}</p><p class="help">这是统计参考值，不是成功率；版本采用以本轮协议和正式结论为准。</p>')
    return body + '</section>'


def _adoption_note(run: dict) -> str:
    spec = run['spec']
    if gates.diagnostic_reason(spec):
        return '不参与版本采用。' + gates.diagnostic_reason(spec)
    if spec.get('against_production'):
        return '本轮只用于固定验收准入，不重复晋级开发版，也不切换生产基线。'
    ptr = _json(run['instance_dir'] / 'pointers.json')
    if spec.get('branch_round'):
        if ptr.get('branch_promotions', {}).get(spec['id']) == spec.get('candidate_ref'):
            return '本轮已推全；后续全量版本可能继续前进。'
        if spec.get('production_basis') != {key: ptr.get(key) for key in
                                            ('data', 'production_gen', 'production_judge')}:
            return '本轮基线已过期；结果保留，不能直接推全。'
        if spec.get('dataset') == 'development':
            branch = _branch_development(run)
            if branch:
                return f'候选已保留为分支 {branch} 的当前开发版；相对生产版达到固定准入门槛后再进入固定验收。'
            return '开发达标后由调度器保留分支开发版；相对生产版达到固定准入门槛后再进入固定验收。'
        return '本轮尚未推全，以调度器核对后的采用记录为准。'
    key = ('production_' if spec.get('dataset') == 'fixed_test' else 'iteration_') + ('judge' if spec.get('kind') == 'judge_eval' else 'gen')
    if ptr.get(key) and ptr[key] == spec.get('candidate_ref'):
        return '当前对应指针已指向本次候选。'
    return '评测结论与版本采用分开记录；当前对应指针未指向本次候选。'


def _render_bill(spec: dict, state: dict) -> str:
    label, _ = _status(spec, state)
    return '\n'.join([f'# 实验记录 · {spec["id"]}', '', f'任务：{_title(spec)}', f'阶段：{_phase(spec)[0]}', f'状态：{label}', f'说明：{_reason(spec, state)}', f'实验对照：{spec.get("baseline_ref")} / 候选：{spec.get("candidate_ref")} / 裁判：{spec.get("judge_ref")} / 数据：{spec.get("data_ref")}', '', '原始指标（诊断用；无有效样本时的零值不代表模型效果）：', json.dumps(state.get('metrics') or {}, ensure_ascii=False, indent=2), ''])


def _run_overview(run: dict) -> str:
    spec, state, instance = run['spec'], run['state'], run['instance_dir']
    label, tone = _status(spec, state)
    hero = f'<div class="hero"><div class="hero-copy"><p class="eyebrow">{_e(instance.name)} / {_e(_phase(spec)[0])}</p><h1>{_e(_title(spec))}</h1><div class="meta"><span>{_e(spec.get("created", ""))}</span><code>{_e(spec["id"])}</code></div></div>{_badge(label, tone)}</div>'
    nav = '<nav class="report-nav" aria-label="实验详情导航"><a href="#configuration">本次改动与机制</a><a href="#results">结果概览</a><a href="#cases">逐题查看</a><a href="#diagnostics">运行详情</a></nav>'
    stats = '<div class="stats">' + _stat('计划题数', run['attempted'], '本次任务的题目数量') + _stat('有效完成', run['pairs'], '已完成生成与评分') + _stat('执行失败', run['failures'], '失败题可修复后续跑') + '</div>'
    recovery = state.get('resolution') or {}
    recovery_link = ''
    if recovery.get('verified_by'):
        href = f'/instances/{quote(instance.name, safe="")}/experiments/{quote(str(recovery["verified_by"]), safe="")}/index.html'
        recovery_link = f'<a class="button primary" href="{href}">查看修复后的同题验证 →</a>'
    summary = f'<section class="panel"><div class="outcome"><div><h2>{_e(label)}</h2><p>{_e(_reason(spec, state))}</p></div></div>{recovery_link}{_error_html(run)}<p class="next-step">{_e(_next_step(run))}</p><p class="decision-note">{_e(_adoption_note(run))}</p></section>'
    baseline = '<section class="panel experiment-baselines"><h2>本轮对照与当前基线</h2>' + _history_versions(run) + _baseline_context(run) + '</section>'
    return hero + nav + baseline + _stage_html(run) + stats + summary


def _render_run_html(run: dict) -> str:
    spec, state, instance = run['spec'], run['state'], run['instance_dir']
    data = _json(instance / 'data' / str(spec.get('data_ref', '')) / 'manifest.json')
    diagnostics = '<section class="panel" id="diagnostics"><h2>运行详情</h2><p class="help">规格、状态与数据说明直接在本页查看。原始状态中的默认数值不代表有效评测结果。</p>'
    diagnostics += _details('实验规格与评测协议', _pre(spec)) + _region('state', _details('原始运行状态与指标', _pre(state))) + _details('数据版本说明', _pre(data)) + '</section>'
    body = _region('summary', _run_overview(run)) + _configuration_html(run) + _region('metrics', _metrics_html(run)) + _region('cases', _cases_html(run)) + diagnostics
    return _page(_title(spec), body, instance.name, f'/api/live/{quote(instance.name, safe="")}/{quote(spec["id"], safe="")}')


def live_payload(instance_dir: Path, run_id: str = '', *, cases_revision: str = '', config_revision: str = '') -> dict:
    """仅返回页面所需内容；固定轮从不读取逐题明细。"""
    if run_id:
        directory = instance_dir / 'experiments' / run_id
        run = _load_run(directory, include_records=False)
        regions = {'summary': _run_overview(run), 'metrics': _metrics_html(run),
                   'state': _details('原始运行状态与指标', _pre(run['state']))}
        case_file = directory / 'cases.jsonl'
        revision = str(case_file.stat().st_mtime_ns) if case_file.exists() else 'none'
        revision += ':' + str((run['state'].get('progress') or {}).get('trace_ref') or '')
        revision += ':' + ','.join(sorted(item.get('trace_ref', '') for item in
                                         ((run['state'].get('progress') or {}).get('active_cases') or {}).values()))
        if revision != cases_revision:
            regions['cases'] = _cases_html(_load_run(directory))
        return {'regions': regions, 'cases_revision': revision}
    runs = _runs(instance_dir / 'experiments')
    regions = {'intro': _dashboard_intro(runs, instance_dir.name), 'history': _history_html(runs)}
    regions['workflows'] = _workflows_region(instance_dir, runs)
    pointer_file = instance_dir / 'pointers.json'
    revision = str(pointer_file.stat().st_mtime_ns) if pointer_file.exists() else 'none'
    if revision != config_revision:
        regions['configuration'] = _baseline_summary(instance_dir)
    return {'regions': regions, 'config_revision': revision}


def write_run(exp_dir: Path) -> None:
    run = _load_run(exp_dir)
    (exp_dir / 'bill.md').write_text(_render_bill(run['spec'], run['state']), encoding='utf-8')
    (exp_dir / 'index.html').write_text(_render_run_html(run), encoding='utf-8')


def publish(exp_dir: Path) -> None:
    """Refresh all derived reports from one experiment's owning instance."""
    instance = exp_dir.parent.parent
    dashboard = instance.parent.parent / 'dashboard'
    write_run(exp_dir)
    refresh_dashboard(instance / 'experiments', dashboard / instance.name)
    write_instance_index(instance.parent, dashboard)
