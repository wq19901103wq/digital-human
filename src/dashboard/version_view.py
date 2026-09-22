"""只读版本档案：冻结文件、分页数据与有记录的实验用途。"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from . import report
from ..iteration import datasets
from .changes import snapshot_delta

KINDS = {'data': '数据版本', 'generators': '生成器快照', 'judges': 'Judge 版本'}
DATASETS = {'development': ('开发集', 'dev_pool.jsonl'), 'fewshot': ('风格示例池', 'fewshot_pool.jsonl')}
DATASETS.update({key: value for key, value in datasets.ROLES.items() if key not in ('development', 'fixed_test')})
PAGE_SIZE = 20


def _directory(instance: Path, kind: str, ref: str) -> Path:
    if kind not in KINDS or not re.fullmatch(r'[\w-]+', ref):
        raise FileNotFoundError('版本不存在')
    directory = instance / kind / ref
    if not directory.resolve().is_relative_to(instance.resolve()):
        raise PermissionError('版本路径越界')
    marker = 'manifest.json' if kind == 'data' else 'config.json'
    if directory.is_symlink() or (directory / marker).is_symlink():
        raise PermissionError('版本快照不能通过符号链接读取')
    if not directory.is_dir() or not (directory / marker).is_file():
        raise FileNotFoundError('版本快照缺失')
    return directory


def _files(directory: Path, kind: str) -> dict[str, Path]:
    if kind == 'data':
        candidates = [directory / name for name in ('manifest.json', 'purposes.json', 'report.json', 'stats.json', 'persona.md')]
    else:
        candidates = [p for p in directory.iterdir() if p.suffix in {'.json', '.md'}]
    candidates += list((directory / 'scenarios').glob('*.md'))
    return {p.relative_to(directory).as_posix(): p for p in sorted(candidates)
            if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(directory.resolve())}


def _current(instance: Path, kind: str, ref: str) -> str:
    pointers = report._baseline_pointers(instance)
    keys = {'data': [('data', '当前数据')],
            'generators': [('production_gen', '当前生产基线'), ('iteration_gen', '当前开发基线')],
            'judges': [('production_judge', '当前生产基线'), ('iteration_judge', '当前开发基线')]}
    return ' / '.join(label for key, label in keys[kind] if pointers.get(key) == ref) or '非当前基线快照'


def _generator_summary(instance: Path) -> str:
    specs = [report._json(p) for p in (instance / 'experiments').glob('*/spec.json')]
    gen = [s for s in specs if s.get('kind') == 'gen_ab']
    smoke = sum(bool(s.get('smoke')) for s in gen)
    dev = sum(not s.get('smoke') and s.get('dataset') == 'development' for s in gen)
    fixed = sum(not s.get('smoke') and s.get('dataset') == 'fixed_test' for s in gen)
    count = len(list((instance / 'generators').glob('*/config.json')))
    return (f'<div class="notice"><p>已保存 {count} 个生成器快照；关联生成器实验中，'
            f'连接检查 {smoke} 次、完整开发评测 {dev} 次、固定验收 {fixed} 次。</p>'
            '<p>旧流程曾为无改动检查及换数据重复创建快照。现在行为内容相同会复用版本，构建来源单独记录。编号不是能力提升次数；'
            '是否通过评测、是否采用，要分别查看实验结果与采用记录。</p></div>')


def index_html(instance: Path) -> str:
    body = '<p class="eyebrow">VERSIONS</p><h1>版本档案</h1>' + report._baseline_summary(instance) + _generator_summary(instance)
    for kind, label in KINDS.items():
        marker = 'manifest.json' if kind == 'data' else 'config.json'
        rows = []
        for path in sorted((instance / kind).glob('*/' + marker), reverse=True):
            ref = path.parent.name
            _directory(instance, kind, ref)
            cfg = report._json(path)
            if kind == 'data':
                info = report._e('全量历史 · 按用途清单 · 全局时间边界' if cfg.get('schema') == 2 else
                                cfg.get('refreeze_reason') or cfg.get('source') or '来源未记录')
            elif kind == 'generators':
                info = '材料来自 ' + report._version_link(instance, 'data', cfg.get('data_version'))
            else:
                from ..judge import normalize_mode
                info = report._e(cfg.get('decision_policy') or normalize_mode(cfg.get('mode')) or '模式未记录')
            rows.append(f'<tr><td>{report._version_link(instance, kind, ref)}</td>'
                        f'<td>{report._e(_current(instance, kind, ref))}</td><td>{info}</td></tr>')
        body += (f'<section class="panel"><h2>{label} · {len(rows)}</h2><div class="table-wrap"><table>'
                 '<thead><tr><th>版本</th><th>当前用途</th><th>保存的来源 / 配置</th></tr></thead><tbody>'
                 + ''.join(rows) + '</tbody></table></div></section>')
    return report._page('版本档案', body, instance.name)


def _usage(instance: Path, kind: str, ref: str) -> str:
    rows = []
    for run in report._runs(instance / 'experiments'):
        spec = run['spec']
        judge = spec.get('kind') == 'judge_eval'
        roles = []
        if kind == 'data' and spec.get('data_ref') == ref:
            roles.append('评测数据')
        elif kind == ('judges' if judge else 'generators'):
            roles += [label for key, label in [('baseline_ref', '实验对照'), ('candidate_ref', '候选')]
                      if spec.get(key) == ref]
        elif kind == 'judges' and spec.get('judge_ref') == ref:
            roles.append('评分裁判')
        elif kind == 'generators' and judge and spec.get('pack_ref'):
            pack = report._json(instance / 'judge_eval' / spec['pack_ref'] / 'pack.json')
            if pack.get('c0_gen_version') == ref:
                roles.append('Judge 评估包的回复生成器')
        if roles:
            rows.append(f'<tr><td><a href="{report._run_url(run)}">{report._e(report._title(spec))}</a></td>'
                        f'<td>{report._e(" / ".join(roles))}</td><td>{report._phase(spec)[0]}</td>'
                        f'<td>{report._version_link(instance, "data", spec.get("data_ref"))}</td>'
                        f'<td>{report._badge(*report._status(spec, run["state"]))}</td></tr>')
    body = '<section class="panel"><h2>关联实验</h2>'
    if rows:
        body += ('<div class="table-wrap"><table><thead><tr><th>任务</th><th>用途</th><th>阶段</th>'
                 '<th>评测数据</th><th>结果</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
    else:
        body += '<p class="help">未找到引用此版本的实验记录；不能据此推断其创建原因或是否曾被采用。</p>'
    return body + '</section>'


def _generator_evidence(instance: Path, ref: str) -> str:
    refs = sorted(p.parent.name for p in (instance / 'generators').glob('*/config.json'))
    same = []
    for other in refs:
        if other == ref:
            continue
        delta = snapshot_delta(instance, 'generators', other, ref)
        if delta['available'] and not delta['changes']:
            same.append(report._version_link(instance, 'generators', other))
    body = '<section class="panel"><h2>快照差异与用途</h2>' + _generator_summary(instance)
    builds = [report._json(p) for p in sorted((instance / 'generator_builds').glob('*.json'))]
    builds = [b for b in builds if b.get('generator_ref') == ref]
    if builds:
        body += '<h3>构建来源记录</h3><p>下列构建使用同一行为版本，数据变化不会重复编号。</p><ul>'
        body += ''.join('<li>' + report._e(b.get('created', '')) + ' · '
                        + report._version_link(instance, 'data', b.get('data_ref')) + '</li>' for b in builds)
        body += '</ul>'
    if same:
        body += ('<p>行为配置、人格与场景内容相同的其他快照：' + '、'.join(same)
                 + '。数据绑定可能不同；这不表示风格池或评测输入也相同。</p>')
    position = refs.index(ref)
    if position:
        previous = refs[position - 1]
        delta = snapshot_delta(instance, 'generators', previous, ref)
        body += ('<p>与相邻编号 ' + report._version_link(instance, 'generators', previous)
                 + ' 的实际文件比较（相邻编号不代表父子关系）：</p>')
        body += report._delta_html('相邻快照差异', delta)
    body += '<p class="help">没有保存创建原因的版本，仅展示可核对的文件差异和实验引用。</p></section>'
    return body


def _data_records(directory: Path, base_url: str, view: str, page: int) -> str:
    if view not in DATASETS:
        raise PermissionError('此数据分区不在线展示')
    if not 1 <= page <= 1_000_000:
        raise ValueError('页码须为正整数')
    label, filename = DATASETS[view]
    path = directory / filename
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise PermissionError('数据文件路径越界')
    if not path.is_file():
        return f'<section class="panel"><h2>{label}</h2><p>此版本未保存该文件。</p></section>'
    start = (page - 1) * PAGE_SIZE
    purpose = datasets.manifest(directory)
    public_before = purpose.get('protocol', {}).get('public_history_before')
    rows = []
    with path.open(encoding='utf-8') as stream:
        index = 0
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if view == 'fewshot' and public_before is not None:
                if row.get('source_span', {}).get('end_timestamp', float('inf')) >= public_before:
                    continue
            if index >= start:
                rows.append(row)
                if len(rows) > PAGE_SIZE:
                    break
            index += 1
    body = f'<section class="panel" id="data-records"><h2>{label} · 实际记录</h2><p class="help">每页 {PAGE_SIZE} 条，按冻结文件原序显示。展开可查看完整原始字段。</p>'
    if view == 'fewshot' and public_before is not None:
        body += '<p class="notice">候选库包含全量历史；此处只展示开发窗口之前的可公开材料。后续历史由回放按题目时间提供，验收答案不批量公开。</p>'
    for offset, row in enumerate(rows[:PAGE_SIZE], start + 1):
        context = ''
        for message in row.get('context', []):
            sender, text = (message.get('sender', '?'), message.get('text', '')) if isinstance(message, dict) else ('', message)
            context += f'<div class="chat-line"><span class="sender">{report._e(sender)}</span><span class="message">{report._e(text)}</span></div>'
        replies = row.get('human_reply', row.get('reply', row.get('response', '未记录')))
        body += (f'<article class="version-record"><h3>第 {offset} 条 · {report._e(row.get("case_id", row.get("id", "")))}</h3>'
                 + '<div class="chat">' + context + '</div><h4>真人回复 / 风格示例</h4>' + report._pre(replies)
                 + report._details('完整原始记录', report._pre(row)) + '</article>')
    if not rows:
        body += '<p class="help">本页没有记录。</p>'
    body += '<nav class="pager" aria-label="数据分页">'
    if page > 1:
        body += f'<a class="button" href="{base_url}?{urlencode({"view": view, "page": page - 1})}">上一页</a>'
    body += f'<span>第 {page} 页</span>'
    if len(rows) > PAGE_SIZE:
        body += f'<a class="button" href="{base_url}?{urlencode({"view": view, "page": page + 1})}">下一页</a>'
    body += (f'<form action="{base_url}" method="get" class="version-page-form"><input type="hidden" name="view" value="{view}">'
             f'<label>跳至第 <input aria-label="页码" type="number" name="page" min="1" max="1000000" value="{page}"> 页</label>'
             '<button class="button" type="submit">跳转</button></form></nav></section>')
    return body


def _purposes(instance: Path, directory: Path, url: str) -> str:
    info = datasets.manifest(directory)
    if not info:
        return ''
    body = ('<section class="panel"><h2>数据用途与时间边界</h2>'
            '<p>全量历史只存一份；学习、优化、拟合、验证和验收分别保存清单。示例按每题输入时刻过滤。</p>'
            '<p>评测包含 80% 熟悉对象时间回放、20% 未见对象整聊天留出；留出条件单独标记，不声称为自然首次接触。</p>'
            '<div class="table-wrap"><table><thead><tr><th>用途</th><th>题数</th><th>群聊 / 私聊</th><th>聊天数</th><th>范围</th></tr></thead><tbody>')
    for role in datasets.ROLES:
        entry = info['roles'].get(role)
        if not entry:
            continue
        title = report._e(entry['label'])
        if not entry.get('sealed'):
            title = f'<a href="{url}?{urlencode({"view": role})}#data-records">{title}</a>'
        types = entry.get('chat_types', {})
        body += (f'<tr><td>{title}</td><td>{entry["total"]}</td><td>{types.get("group", 0)} / {types.get("private", 0)}</td>'
                 f'<td>{entry.get("chats", "—")}</td><td>{"封存，仅统计" if entry.get("sealed") else "可查看实际样本"}</td></tr>')
    body += '</tbody></table></div><p>人格与场景属于生成器资产；本数据版本不附带、不重建生成器。</p>'
    clock = timezone(timedelta(hours=8))
    times = [info['protocol'].get(k) for k in ('development_start', 'acceptance_start')]
    if all(isinstance(t, (int, float)) for t in times):
        start, end = [datetime.fromtimestamp(t, clock).strftime('%Y-%m-%d %H:%M') for t in times]
        body += f'<p>全局时间（北京时间）：学习材料早于 {start}；开发窗口 {start} 至 {end}；验收从 {end} 开始。</p>'
    audit = report._json(instance / 'data_preparations' / f'{directory.name}-audit.json')
    if audit.get('status') == 'passed':
        body += f'<p>数据来源审计通过：{audit.get("messages_verified", 0):,} 条消息、{audit.get("examples_verified", 0):,} 条完整示例。</p>'
    body += '<p class="notice">当前旧模型和提示词的学习来源仍须核验。数据边界通过不代表模型已通过正式验收。</p>'
    if info['protocol'].get('development_roles_share_messages'):
        body += '<p class="notice">Gen 与 Judge 共用开发验证题，用于选择候选；两次结果不代表独立验证。固定验收片段另行隔离。</p>'
    jobs = []
    for path in sorted((instance / 'judge_training').glob('*/spec.json')):
        spec = report._json(path)
        if spec.get('data_ref') == directory.name:
            from ..iteration.task_state import read as task_state
            state = task_state(path.parent)
            from .workflow_view import _status
            phase = _status(state)
            jobs.append(f'<li>{report._e(path.parent.name)} · {report._e(phase)} · '
                        + report._version_link(instance, 'generators', spec.get('generator_ref')) + ' · '
                        + report._version_link(instance, 'judges', spec.get('source_judge')) + '</li>')
    body += '<h3>Judge 训练任务</h3>' + ('<ul>' + ''.join(jobs) + '</ul>' if jobs else '<p>暂无引用该数据版本的训练任务。</p>')
    return body + '</section>'


def detail_html(instance: Path, kind: str, ref: str, query: dict) -> str:
    directory = _directory(instance, kind, ref)
    files = _files(directory, kind)
    url = report._version_url(instance, kind, ref)
    marker = 'manifest.json' if kind == 'data' else 'config.json'
    cfg = report._json(directory / marker)
    body = (f'<p class="eyebrow">{report._e(instance.name)} / VERSIONS</p><h1>{KINDS[kind]} · {report._e(ref)}</h1>'
            f'<p>{report._badge(_current(instance, kind, ref))}</p>'
            '<p class="help">以下内容直接读取此版本保存的文件；关联实验和当前用途按现有记录显示。</p>')
    body += report._baseline_summary(instance)
    if kind == 'generators':
        body += '<p>人格与场景材料来自 ' + report._version_link(instance, 'data', cfg.get('data_version')) + '；实际评测数据见关联实验。</p>'
    if kind == 'data':
        body += _purposes(instance, directory, url)
        sets = cfg.get('testsets', {})
        body += '<div class="stats">' + report._stat('开发集', sets.get('development', {}).get('total', '未记录'), 'manifest 中记录的题数')
        body += report._stat('固定验收集', sets.get('fixed_test', {}).get('total', '未记录'), '只展示统计与文件指纹，答案不在线展示')
        body += report._stat('风格示例池', cfg.get('fewshot_pool', {}).get('total', '未记录'), 'manifest 中记录的示例数') + '</div>'
        body += '<nav class="report-nav" aria-label="版本数据内容">' + ''.join(
            f'<a href="{url}?{urlencode({"view": view})}#data-records">{label}</a>' for view, (label, filename) in DATASETS.items()
            if (directory / filename).is_file()) + '</nav>'
    selected = query.get('file', [marker])[0]
    if selected not in files:
        raise PermissionError('此文件不在版本可查看内容中')
    body += '<section class="panel"><h2>冻结文件</h2><nav class="report-nav" aria-label="版本文件">'
    body += ''.join(f'<a href="{url}?{urlencode({"file": name})}" aria-current="{"page" if name == selected else "false"}">{report._e(name)}</a>' for name in files)
    raw = files[selected].read_bytes()
    body += (f'</nav><h3>{report._e(selected)}</h3><p class="help">{len(raw):,} 字节 · SHA-256 '
             f'<code>{hashlib.sha256(raw).hexdigest()}</code></p>' + report._pre(raw.decode('utf-8')) + '</section>')
    if kind == 'data':
        body += _data_records(directory, url, query.get('view', ['development'])[0], int(query.get('page', ['1'])[0]))
    elif kind == 'generators':
        body += _generator_evidence(instance, ref)
    body += _usage(instance, kind, ref)
    return report._page(f'{KINDS[kind]} {ref}', body, instance.name)
