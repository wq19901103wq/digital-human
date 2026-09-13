"""实验与实例的只读报告；读取规格、状态、逐题结果及发送时保存的调用实录。"""
from __future__ import annotations

import difflib
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .. import tracing
from . import experiment, protocol
from .progress import is_active

_UI = Path(__file__).resolve().parents[1] / 'dashboard'
_LABELS = {
    'merge_to_iteration_baseline': ('开发评测达标', 'success'),
    'adopt': ('固定验收通过', 'success'),
    'reject': ('未达到采用条件', 'danger'),
    'observe': ('改善不足，继续观察', 'warning'),
}


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else '—'), quote=True)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def _pre(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f'<pre>{_e(text)}</pre>'


def _details(title: str, body: str, opened: bool = False, *, key: str = '') -> str:
    identity = f' data-detail-key="{_e(key)}"' if key else ''
    return f'<details{identity}{" open" if opened else ""}><summary>{_e(title)}</summary><div class="detail-body">{body}</div></details>'


def _badge(text: str, tone: str = 'neutral') -> str:
    return f'<span class="badge {tone}">{_e(text)}</span>'


def _phase(spec: dict) -> tuple[str, str]:
    if spec.get('smoke'):
        return '连接检查', 'smoke'
    if spec.get('dataset') == 'fixed_test':
        return '固定验收', 'fixed'
    return ('裁判抽样校准' if spec.get('kind') == 'judge_eval' else '生成器开发评测'), 'development'


def _title(spec: dict) -> str:
    return str(spec.get('single_change') or '未填写任务说明').removeprefix('冒烟：').removeprefix('冒烟:').strip()


def _status(spec: dict, state: dict) -> tuple[str, str]:
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
    reason = str(state.get('reason') or '暂无最终结论；这里不代表后台进程一定仍在运行。')
    if state.get('verdict') == 'merge_to_iteration_baseline':
        reason = reason.replace('，并入开发基线', '，满足开发基线晋升条件')
    # 历史账单沿用生成器措辞；裁判的优化方向是识别数上升。原文仍保留在诊断区。
    return reason.replace('严格下降', '严格上升') if spec.get('kind') == 'judge_eval' else reason


def _page(title: str, body: str, instance: str = '', live_url: str = '') -> str:
    nav = f'<a href="/dashboard/{quote(instance, safe="")}/index.html">实例概览</a>' if instance else ''
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


def _run_url(run: dict) -> str:
    return f'/instances/{quote(run["instance_dir"].name, safe="")}/experiments/{quote(run["spec"]["id"], safe="")}/index.html'


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
    if spec.get('smoke'):
        return '下一步：连接检查已通过，可以准备完整开发评测。'
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
    mode = config.get('mode')
    llm = config.get('llm') or {}
    model_name = llm.get('model', '模型未记录')
    if mode != 'rpa_corrected_pairwise':
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


def _configuration(instance_dir: Path) -> str:
    ptr = _json(instance_dir / 'pointers.json')
    parts = []
    for label, prod, dev, folder in [('回复生成器', 'production_gen', 'iteration_gen', 'generators'), ('评分裁判', 'production_judge', 'iteration_judge', 'judges')]:
        current = ptr.get(prod, '')
        if prod == 'production_judge' and not current:
            current = ptr.get('judge', '')
        cfg = _json(instance_dir / folder / str(current) / 'config.json')
        draft = ptr.get(dev, current)
        note = '开发版与生产版一致' if draft == current else f'开发版：{draft}'
        detail = _generator_pipeline(cfg) if folder == 'generators' else _judge_pipeline(cfg, instance_dir / folder / str(current))
        if folder == 'judges' and cfg.get('mode') == 'rpa_corrected_pairwise':
            llm = cfg.get('llm') or {}
            note += f' · Codex CLI · 推理 {llm.get("reasoning_effort")} · {llm.get("timeout_seconds")} 秒'
        detail += _version_html(instance_dir, str(current), folder == 'judges', '展开配置、提示词与来源')
        models = _model_rows(cfg) if folder == 'generators' else ''
        parts.append(f'<div class="config-block {"judge-config" if folder == "judges" else "generator-config"}"><div class="config-title"><h3>{label}</h3><code>{_e(current or "未设置")}</code></div><p class="help">{_e(note)}</p>{detail}<div class="config-models">{models}</div></div>')
    data = ptr.get('data', '')
    manifest = _json(instance_dir / 'data' / str(data) / 'manifest.json')
    total = (manifest.get('fewshot_pool') or {}).get('total')
    pool = f'{total:,} 条风格示例' if isinstance(total, int) else '风格池数量未记录'
    sets = manifest.get('testsets') or {}
    sizes = ' · '.join(f'{label} {sets[key]["total"]:,} 题' for key, label in [('development', '开发评测'), ('fixed_test', '固定验收')] if key in sets)
    parts.append(f'<div class="config-block"><div class="config-title"><span>评测数据与风格池</span><code>{_e(data or "未设置")}</code></div><p>{_e(pool)}</p><p class="help">{_e(sizes)}</p></div>')
    return '<section class="panel mechanism-panel" id="mechanism"><div class="section-head"><h2>当前系统怎样生成和评分</h2><span class="pill">当前配置 · 生产版本</span></div><div class="mechanism-grid">' + parts[1] + parts[0] + '</div>' + parts[2] + '</section>'


_STAGES = {'preparing': '准备数据与版本', 'preflight': '检查实验改动',
           'generating': '生成回复', 'judging': '大模型初判', 'extracting': '大模型抽取特征',
           'predicting': '小模型预测与校正', 'checkpoint': '保存本题结果',
           'summarizing': '汇总结果', 'finished': '任务已结束', 'interrupted': '执行已中断'}


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
    branch = {'baseline': '基线', 'candidate': '候选'}.get(progress.get('branch'), '')
    supplement = f'补测第 {progress["round"]} 轮 · ' if progress.get('round') else ''
    case_index = progress.get('case_index', 0)
    detail = f'第 {case_index} / {run["attempted"]} 题 · {supplement}{branch}' if active and case_index else ''
    clock = f'<span data-stage-clock="{progress["phase_started_at"]}"></span>' if active and progress.get('phase_started_at') else ''
    finished = run['pairs'] + run['failures']
    ratio = min(100, finished / run['attempted'] * 100) if run['attempted'] else 0
    body = f'''<div class="execution"><div class="section-head"><h3>当前阶段 · {_e(title)}</h3>{clock}</div>
<p>{_e(detail)}</p><div class="progress" role="progressbar" aria-label="题目处理进度" aria-valuemin="0" aria-valuemax="{run['attempted']}" aria-valuenow="{finished}"><span style="width:{ratio:.1f}%"></span></div>
<div class="progress-copy"><span>已处理 {finished} / {run['attempted']} 题</span><span>有效 {run['pairs']} · 失败 {run['failures']}</span></div>'''
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
    for branch, label in [('baseline', '基线'), ('candidate', '候选')]:
        hits = m.get('identified_' + branch)
        value = _rate(hits, n)
        foot = f'{hits} / {n} 道有效题' if n and hits is not None else '尚无有效判定'
        body += _stat(label + name, value, foot)
    processed = n + run['failures']
    rate = _rate(run['failures'], processed)
    body += _stat('执行失败率', rate, f'{run["failures"]} / {processed} 道已处理题') + '</div>'
    rule = '裁判识别越高越好。' if judge else '生成器被识别为 AI 越低越好。'
    body += f'<p class="help">{rule}初测指首次完整评分（RPA 包含小模型校正）；失败题不计入，翻转补测另计。</p>'
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
    from ..dashboard.changes import snapshot_delta
    spec = run['spec']
    return snapshot_delta(run['instance_dir'], 'judges' if spec.get('kind') == 'judge_eval' else 'generators',
                          spec.get('baseline_ref', ''), spec.get('candidate_ref', ''))


def _change_summary(run: dict) -> str:
    delta = _local_delta(run)
    if not delta['available']:
        return '缺少版本快照，改动待核对'
    if not delta['changes']:
        return '两版策略相同 · 连接验证' if run['spec'].get('smoke') else '两版策略相同'
    return '；'.join(r['label'] + '：' + r['before'] + ' → ' + r['after'] for r in delta['changes'])


def _iteration_html(run: dict) -> str:
    from ..dashboard.changes import snapshot_delta
    spec, instance = run['spec'], run['instance_dir']
    judge = spec.get('kind') == 'judge_eval'
    subject = '裁判' if judge else '生成器'
    body = '<div class="iteration-change"><h3>这次迭代了什么</h3>'
    body += f'<p class="help">任务说明（创建时填写）：{_e(_title(spec))}</p>'
    body += _delta_html(f'本轮对比 · 基线{subject}与候选{subject}', _local_delta(run))
    body += f'<p class="help">本轮两版共用数据 {_e(spec.get("data_ref") or spec.get("pack_ref") or "未记录")}。'
    if not judge:
        body += f'评分裁判固定为 {_e(spec.get("judge_ref", "未记录"))}。'
    body += '</p>'
    previous = _previous_task(run)
    if previous:
        old = previous['spec']
        body += f'<h4>相较上一条同类任务，运行配置有什么变化</h4><p class="help">按创建时间比较：<a href="{_run_url(previous)}">{_e(old["id"])}</a> → {_e(spec["id"])}。这不是本轮候选的改动。</p>'
        folder = 'judges' if judge else 'generators'
        body += _delta_html(f'{subject}基线', snapshot_delta(instance, folder, old.get('baseline_ref', ''), spec.get('baseline_ref', '')))
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
<h3 class="latest-title">{_e(_title(spec))}</h3><p class="change-summary">{_e(_change_summary(run))}</p>{_stage_html(run)}{_rate_cards(run)}{_iteration_html(run)}
{_error_html(run)}<p class="next-step">{_e(_next_step(run))}</p><a class="button primary" href="{_run_url(run)}">查看任务与逐题详情 →</a></section>'''


def _list_controls(search_label: str, options: list[tuple[str, str]]) -> str:
    opts = ''.join(f'<option value="{_e(value)}">{_e(label)}</option>' for value, label in options)
    return f'<div class="toolbar"><input type="search" aria-label="{_e(search_label)}" placeholder="{_e(search_label)}"><select aria-label="筛选类型">{opts}</select></div>'


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


def _history_rates(run: dict) -> str:
    n = run['pairs']
    body = ''
    for branch, label in [('baseline', '基线'), ('candidate', '候选')]:
        hits = run['metrics'].get('identified_' + branch)
        count = f'{hits}/{n}' if n and hits is not None else '无有效判定'
        body += f'<div class="history-rate"><span>{label}</span><strong>{_rate(hits, n)}</strong><small>{_e(count)}</small></div>'
    direction = '正确识别 · 越高越好' if run['spec'].get('kind') == 'judge_eval' else '被识别为 AI · 越低越好'
    body += f'<div class="table-sub">{direction}</div>'
    if n and run['state'].get('status') != 'finished':
        body += '<div class="table-sub">阶段性结果</div>'
    return body


def _history_html(runs: list) -> str:
    rows = []
    for run in runs:
        spec, state = run['spec'], run['state']
        label, tone = _status(spec, state)
        phase, key = _phase(spec)
        change = f'<p class="change-summary">{_e(_change_summary(run))}</p>' + _details('展开具体改动', _iteration_html(run), key='task-change-' + spec['id'])
        processed = run['pairs'] + run['failures']
        failure_rate = _rate(run['failures'], processed)
        rows.append(f'''<tr data-item data-kind="{key}">
<td><a class="task-link" href="{_run_url(run)}">{_e(_title(spec))}</a><div class="table-sub">{_e(spec.get("created", ""))}</div></td>
<td>{_history_rates(run)}</td><td>{change}</td><td>{_e(phase)}</td><td>{_badge(label, tone)}</td>
<td>{run["pairs"]} / {run["attempted"]}<div class="table-sub">执行失败率 {failure_rate}<br>失败 {run["failures"]} / 已处理 {processed} 题</div></td>
<td><a href="{_run_url(run)}">查看详情</a></td></tr>''')
    history = '<section id="history" class="panel" data-list data-page-size="10"><div class="section-head"><h2><span class="section-index">03</span>实验历史</h2><span class="help">按时间倒序</span></div>'
    history += _list_controls('搜索任务名称或时间', [('all', '全部任务'), ('smoke', '连接检查'), ('development', '开发评测'), ('fixed', '固定验收')])
    history += '<p class="help">识别率按初测有效题计算（RPA 包含小模型校正），失败题不计入，翻转补测另计；执行失败率按已处理题计算。运行中的结果自动更新。</p>'
    history += '<div class="table-wrap"><table class="history"><thead><tr><th>任务</th><th>初测识别率 · 有效样本</th><th>具体改动</th><th>阶段</th><th>状态</th><th>有效 / 总题数</th><th></th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>' + _pager() + '</section>'
    return history


def _region(name: str, body: str) -> str:
    return f'<div data-live-region="{name}">{body}</div>'


def dashboard_html(exp_root: Path) -> str:
    runs = _runs(exp_root)
    instance = exp_root.parent.name
    intro = _region('intro', _dashboard_intro(runs, instance))
    task = _latest(_current_run(runs)) if runs else '<section class="panel"><h2>最近任务</h2><p class="empty">还没有任务记录。</p></section>'
    overview = _region('configuration', _configuration(exp_root.parent)) + _region('task', task)
    return _page(instance, intro + overview + _region('history', _history_html(runs)), instance,
                 f'/api/live/{quote(instance, safe="")}')


def refresh_dashboard(exp_root: Path, dashboard_dir: Path) -> None:
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    (dashboard_dir / 'index.html').write_text(dashboard_html(exp_root), encoding='utf-8')


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
    (dashboard_dir / 'index.html').write_text(_page('全部实例', body), encoding='utf-8')


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
    label, key = _case_result(row, judge)
    tone = {'failed': 'danger', 'loss': 'warning', 'win': 'success'}.get(key, 'neutral')
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
    from ..dashboard.trace_view import slot
    attempts = [r for r in raw if r.get('trace_ref')]
    if row.get('trace_ref') and not any(r['trace_ref'] == row['trace_ref'] for r in attempts):
        attempts.append(row)
    if trace_base and attempts:
        for n, attempt in enumerate(reversed(attempts)):
            title = '实际提示词与调用过程' if n == 0 else f'此前执行的调用实录 · {len(attempts) - n}'
            body += slot(trace_base + '/' + quote(attempt['trace_ref'], safe=''), title)
    else:
        body += _details('实际提示词与调用过程 · 历史未记录', '<p class="notice">此历史任务没有保存实际调用实录，无法还原当时完整提示词、A/B 换位、模型返回正文和抽取特征。页面下方的版本提示词是模板，不能当作本题实际输入。新的执行会从发送请求前开始记录。</p>')
    if row.get('status') == 'failed':
        body += '<div class="error-box"><h3>本题失败原因</h3>' + _pre(row.get('reason') or '未记录') + '</div>'
    body += _reply('真人实际回复', row.get('human_reply'), 'human')
    keys = ('baseline_correct', 'candidate_correct') if judge else ('identified_baseline', 'identified_candidate')
    b, c = row.get(keys[0]), row.get(keys[1])
    if judge:
        body += _reply('交给两版裁判识别的 AI 回复', row.get('ai_replies'))
        body += '<div class="reply-grid">' + _reply('基线裁判', [_identified(b, True)]) + _reply('候选裁判', [_identified(c, True)], 'candidate') + '</div>'
    else:
        body += '<div class="reply-grid">' + _reply('基线生成器回复', row.get('baseline_replies'), result=_identified(b, False)) + _reply('候选生成器回复', row.get('candidate_replies'), 'candidate', _identified(c, False)) + '</div>'
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
        verification += '<dl><dt>基线最终判定</dt><dd>' + _e(_identified(final.get('baseline_identified_final'), judge)) + '</dd><dt>候选最终判定</dt><dd>' + _e(_identified(final.get('candidate_identified_final'), judge)) + '</dd></dl>'
        if isinstance(flip, dict):
            for k, title in [('baseline_votes', '基线历次判定'), ('candidate_votes', '候选历次判定')]:
                if flip.get(k):
                    verification += f'<p>{title}（含初测）：{_e(" → ".join(_identified(v, judge) for v in flip[k]))}</p>'
    else:
        verification += f'<p>未进行补测。初测：{_e(_comparison(b, c, judge)[0])}。</p>'
    body += '<div class="verification">' + verification + '</div>'
    timing = {label: row[k] for k, label in [('baseline_latency_ms', '基线耗时（毫秒）'), ('candidate_latency_ms', '候选耗时（毫秒）')] if k in row}
    if timing:
        body += '<p class="help">' + ' · '.join(f'{k}：{_e(v)}' for k, v in timing.items()) + '</p>'
    body += _details('查看原始记录与重试历史', _pre(raw))
    return f'<details class="case" data-item data-case-id="{_e(row["case_id"])}" data-kind="{key}"><summary><span class="case-summary"><span class="case-title">第 {index} 题 · {kind}</span><span class="case-preview">{_e(preview)}</span></span>{_badge(label, tone)}</summary><div class="detail-body">{body}</div></details>'


def _cases_html(run: dict) -> str:
    if run['spec'].get('dataset') == 'fixed_test':
        return '<section class="panel" id="cases"><h2>逐题详情</h2><p class="notice">固定验收仅展示汇总结果。固定集答案不在开发后台展开。</p></section>'
    records = list(run['records'])
    pending = (run['state'].get('progress') or {}).get('trace_ref')
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
    options = [('all', '全部结果'), ('failed', '执行失败'), ('win', '候选更好'), ('loss', '候选更差'), ('tie', '两版一致'), ('pending', '判定未完成')]
    body = '<section class="panel" id="cases" data-list data-page-size="20"><div class="section-head"><h2>逐题详情<span class="count">' + str(len(records)) + ' 题</span></h2><span class="help">点击每题展开</span></div>'
    body += '<p class="help">展开每题查看消息角色、来源、实际提示词、模型返回、裁判特征和补测过程。长调用记录点击后加载；历史任务未保存的细节会明确标注。</p>'
    body += _list_controls('搜索聊天内容、回复或 case ID', options)
    for index, row in enumerate(records, 1):
        merged = {**inputs.get(str(row['case_id']), {}), **row}
        trace_base = f'/api/trace/{quote(run["instance_dir"].name, safe="")}/{quote(run["spec"]["id"], safe="")}'
        body += _case_html(merged, index, judge, histories.get(str(row['case_id']), []), trace_base)
    return body + _pager() + '</section>'


def _version_html(instance_dir: Path, ref: str, judge: bool, title: str) -> str:
    directory = instance_dir / ('judges' if judge else 'generators') / ref
    config = _json(directory / 'config.json')
    if not config:
        return _details(title, '<p class="muted">版本快照不存在。</p>')
    body = _model_rows(config) + (_judge_pipeline(config, directory) if judge else _generator_pipeline(config)) + _details('完整行为配置', _pre(config))
    for filename, label in [('persona.md', '人格与回复规则'), ('prompt.md', '裁判提示词')]:
        path = directory / filename
        if path.exists():
            body += _details(label, _pre(path.read_text(encoding='utf-8')))
    for path in sorted((directory / 'scenarios').glob('*.md')):
        body += _details('场景规则 · ' + path.stem, _pre(path.read_text(encoding='utf-8')))
    if config.get('mode') == 'rpa_corrected_pairwise':
        body += _details('RPA 原始已采用配置', _pre(_json(directory / 'source_judge.json')))
        body += _details('RPA 裁判规则', _pre(_json(directory / 'profile.json')))
        body += _details('裁判参考资料', _pre(_json(directory / 'reference.json')))
        correction = _json(directory / 'correction.json')
        body += _details('冻结校正模型与特征', _pre({'final_model': correction.get('final_model'), 'feature_system': correction.get('feature_system')}))
    meta = _json(directory / 'meta.json')
    if meta:
        body += _details('版本来源', _pre(meta))
    return _details(f'{title} · {ref}', body, key='version-' + title + '-' + ref)


def _configuration_html(run: dict) -> str:
    spec, instance = run['spec'], run['instance_dir']
    judge = spec.get('kind') == 'judge_eval'
    b, c = str(spec.get('baseline_ref', '')), str(spec.get('candidate_ref', ''))
    folder = instance / ('judges' if judge else 'generators')
    body = '<section class="panel" id="configuration"><h2>本次改动与版本快照</h2><p class="help">这里展示实验创建时使用的版本，可能与当前生产版本不同。</p>' + _iteration_html(run)
    if judge:
        body += '<div class="mechanism-grid">' + ''.join(f'<div><h3>{label} · {_e(ref)}</h3>' + _judge_pipeline(_json(folder / ref / 'config.json'), folder / ref) + '</div>' for ref, label in [(b, '基线裁判'), (c, '候选裁判')]) + '</div>'
    elif spec.get('judge_ref'):
        path = instance / 'judges' / spec['judge_ref']
        body += f'<h3>本轮评分机制 · {_e(spec["judge_ref"])}</h3>' + _judge_pipeline(_json(path / 'config.json'), path)
    body += '<div class="version-grid">' + _version_html(instance, b, judge, '基线版本') + _version_html(instance, c, judge, '候选版本') + '</div>'
    assets = {p.relative_to(folder / ref) for ref in [b, c] if ref for p in (folder / ref).rglob('*.md')}
    diffs = []
    for rel in sorted(assets):
        bp, cp = folder / b / rel, folder / c / rel
        old = bp.read_text(encoding='utf-8').splitlines() if bp.exists() else []
        new = cp.read_text(encoding='utf-8').splitlines() if cp.exists() else []
        diff = '\n'.join(difflib.unified_diff(old, new, fromfile=f'基线/{rel}', tofile=f'候选/{rel}', lineterm=''))
        if diff:
            diffs.append(_details(str(rel), _pre(diff)))
    body += _details('人格、提示词与场景的逐行差异', ''.join(diffs) or '<p class="muted">两版文本内容一致。</p>')
    if not judge and spec.get('judge_ref'):
        body += _version_html(instance, spec['judge_ref'], True, '本轮评分裁判')
    return body + '</section>'


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
    for label, value in [(f'基线{heading}', f'{m.get("identified_baseline", "—")} / {run["pairs"]}'), (f'候选{heading}', f'{m.get("identified_candidate", "—")} / {run["pairs"]}'), ('确认净胜', f'{m["net_win_confirmed"]:+d}' if 'net_win_confirmed' in m else '—')]:
        body += f'<div class="metric"><span>{_e(label)}</span><strong>{_e(value)}</strong></div>'
    body += '</div><p class="help">确认胜 ' + _e(m.get('wins_confirmed')) + ' · 确认负 ' + _e(m.get('losses_confirmed')) + ' · 补测后打平 ' + _e(m.get('contested')) + '</p>'
    if state.get('status') == 'finished' and state.get('verdict') != 'experiment_incomplete' and m.get('sign_p') is not None:
        body += _details('统计细节', f'<p>符号检验 p 值：{m["sign_p"]:.4f}</p><p class="help">这是统计参考值，不是成功率；版本采用以本轮协议和正式结论为准。</p>')
    return body + '</section>'


def _adoption_note(run: dict) -> str:
    spec = run['spec']
    if spec.get('smoke'):
        return '连接检查不参与版本晋升。'
    ptr = _json(run['instance_dir'] / 'pointers.json')
    key = ('production_' if spec.get('dataset') == 'fixed_test' else 'iteration_') + ('judge' if spec.get('kind') == 'judge_eval' else 'gen')
    if ptr.get(key) and ptr[key] == spec.get('candidate_ref'):
        return '当前对应指针已指向本次候选。'
    return '评测结论与版本采用分开记录；当前对应指针未指向本次候选。'


def _render_bill(spec: dict, state: dict) -> str:
    label, _ = _status(spec, state)
    return '\n'.join([f'# 实验记录 · {spec["id"]}', '', f'任务：{_title(spec)}', f'阶段：{_phase(spec)[0]}', f'状态：{label}', f'说明：{_reason(spec, state)}', f'基线：{spec.get("baseline_ref")} / 候选：{spec.get("candidate_ref")} / 裁判：{spec.get("judge_ref")} / 数据：{spec.get("data_ref")}', '', '原始指标（诊断用；无有效样本时的零值不代表模型效果）：', json.dumps(state.get('metrics') or {}, ensure_ascii=False, indent=2), ''])


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
    return hero + nav + _stage_html(run) + stats + summary


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
        if revision != cases_revision:
            regions['cases'] = _cases_html(_load_run(directory))
        return {'regions': regions, 'cases_revision': revision}
    runs = _runs(instance_dir / 'experiments')
    regions = {'intro': _dashboard_intro(runs, instance_dir.name), 'history': _history_html(runs)}
    regions['task'] = _latest(_current_run(runs)) if runs else '<section class="panel"><h2>最近任务</h2><p class="empty">还没有任务记录。</p></section>'
    pointer_file = instance_dir / 'pointers.json'
    revision = str(pointer_file.stat().st_mtime_ns) if pointer_file.exists() else 'none'
    if revision != config_revision:
        regions['configuration'] = _configuration(instance_dir)
    return {'regions': regions, 'config_revision': revision}


def write_run(exp_dir: Path) -> None:
    run = _load_run(exp_dir)
    (exp_dir / 'bill.md').write_text(_render_bill(run['spec'], run['state']), encoding='utf-8')
    (exp_dir / 'index.html').write_text(_render_run_html(run), encoding='utf-8')
