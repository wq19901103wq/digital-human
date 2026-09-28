"""Read-only Share bindings and paginated, manifest-listed knowledge browsing."""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import quote, urlencode

from . import report

COLLECTIONS = {
    'pages': '人物、群与话题', 'fact_claims': '事实与关系',
    'identities': '身份', 'address_edges': '定向称呼',
    'interaction_profiles': '互动习惯', 'pending_identity_groups': '待确认身份',
    'library_pages': '整理的 Wiki', 'library_claims': 'Wiki 摘要',
    'library_gaps': '资料缺口', 'attributes': '人物属性', 'relations': '人物与实体关系',
    'events': '经历与事件', 'addresses': '称呼记录', 'entities': '人物与实体',
    'reviews': '事实处理记录', 'evidence': '来源定位',
}
PAGE_SIZE = 20
VALUE_LABELS = {
    'usage': dict(direct='直接称呼', self_reference='自称', third_person='提及第三人',
                  quoted='引用称呼', requested='希望被称呼', rejected='拒绝被称呼', uncertain='对象待确认'),
    'mode': dict(self_report='自述', reported='转述', observed='观察', inferred='推断', plan='计划',
                 joke='玩笑', uncertain='待确认', conflict='冲突'),
    'polarity': dict(affirmed='肯定', negated='否定', uncertain='不确定'),
    'temporal_kind': dict(stable='稳定属性', mutable='可能变化', historical='历史状态', event='事件', unknown='未明确'),
    'status': dict(planned='计划中', ongoing='进行中', completed='已完成', cancelled='已取消', unknown='未确认'),
    'disposition': dict(represented='已结构化', omit='不纳入', unresolved='未解决'),
}


def _config(instance, folder, ref):
    if not ref or not re.fullmatch(r'[\w-]+', str(ref)):
        return {}
    path = instance / folder / ref / 'config.json'
    if not path.resolve().is_relative_to(instance.resolve()) or any(
            p.is_symlink() for p in (path, path.parent, path.parent.parent)):
        return {}
    return report._json(path)


def binding_html(instance: Path, folder: str, ref: str | None) -> str:
    config = _config(instance, folder, ref)
    share = config.get('share_ref')
    if not share:
        return '<span class="help">未绑定</span>'
    result = report._version_link(instance, 'share', share)
    digest = config.get('share_sha256')
    if digest:
        result += f' <small title="SHA-256 {report._e(digest)}">{report._e(digest[:12])}</small>'
    else:
        result += ' <span class="help">未记录摘要</span>'
    return result


def current_bindings(instance: Path, ref: str) -> list[str]:
    pointers = report._baseline_pointers(instance)
    result = []
    for role, label in [('production', '生产'), ('iteration', '开发')]:
        for kind, folder in [('gen', 'generators'), ('judge', 'judges')]:
            model = pointers.get(f'{role}_{kind}')
            if _config(instance, folder, model).get('share_ref') == ref:
                result.append(f'{label} {kind} {model}')
    return result


def _files(directory, manifest):
    files = {'manifest.json': directory / 'manifest.json'}
    for name in manifest.get('files', {}):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise PermissionError('Share 文件路径越界')
        path = directory / 'content' / relative
        if not path.resolve().is_relative_to(directory.resolve()) or any(
                p.is_symlink() for p in (path, *path.parents) if p != directory.parent):
            raise PermissionError('Share 文件不能通过符号链接读取')
        if path.is_file():
            files['content/' + name] = path
    return files


def _rows(data, collection):
    value = data.get(collection, [])
    if isinstance(value, dict):
        return [dict(row, id=key) for key, row in value.items()]
    return value


def _entity_name(data, eid):
    entity = next((e for e in data.get('entities', []) if e['id'] == eid), {})
    label = entity.get('label') or eid or '待确认'
    if entity.get('account') and entity['account'] == data.get('subject', {}).get('self_account'):
        label += '（本人）'
    return label


def _record(row, data, url):
    title = (row.get('title') or row.get('name') or row.get('label') or row.get('field')
             or row.get('term') or row.get('event_type') or row.get('predicate') or row.get('id') or '记录')
    body = '<article class="version-record"><h3>' + report._e(title) + '</h3>'
    for key in ('subject_id', 'object_id', 'speaker_id', 'target_id'):
        if key in row:
            label = dict(subject_id='主体', object_id='客体', speaker_id='说话人', target_id='称呼对象')[key]
            body += f'<p><strong>{label}：</strong>{report._e(_entity_name(data, row[key]))}</p>'
    for key in ('summary', 'description', 'value', 'detail', 'reason'):
        if row.get(key):
            body += '<p>' + report._e(row[key]) + '</p>'
    for participant in row.get('participants', []):
        body += ('<p><strong>' + report._e(participant['role']) + '：</strong>'
                 + report._e(_entity_name(data, participant['entity_id'])) + '</p>')
    for detail in row.get('details', []):
        body += '<p><strong>' + report._e(detail['field']) + '：</strong>' + report._e(detail['value']) + '</p>'
    metadata = [('subject', '主体'), ('object_name', '对象'), ('account', '账号'),
                ('usage', '称呼用法'), ('time_scope', '时期'), ('mode', '依据性质'), ('polarity', '肯定或否定'),
                ('attribution', '归属依据'),
                ('temporal_kind', '时间性质'), ('disposition', '处理结果'),
                ('status', '状态'), ('validity_note', '适用范围'), ('reported_dates', '记载日期'),
                ('valid_from', '生效时间'), ('valid_to', '结束时间'), ('applicability', '时间适用性'),
                ('limitations', '限制'), ('unknowns', '待确认'), ('gaps', '缺口')]
    body += ''.join(f'<p><strong>{label}：</strong>{report._e(VALUE_LABELS.get(key, {}).get(str(row[key]), row[key]))}</p>'
                    for key, label in metadata if row.get(key))
    claims = [claim for collection in ('fact_claims', 'library_claims') for claim in data.get(collection, [])
              if row.get('id') and (row['id'] in claim.get('page_ids', []) or row['id'] == claim.get('page_id'))]
    if claims:
        body += report._details(f'展开本页内容（{len(claims)} 项）',
                                ''.join(_record(c, {}, url) for c in claims))
    refs = row.get('evidence_refs', [])
    if refs:
        body += '<p>来源：' + '、'.join(
            f'<a href="{url}?{urlencode({"view": "evidence", "q": ref})}#share-records">{report._e(ref)}</a>'
            for ref in refs) + '</p>'
    return body + report._details('完整字段', report._pre(row)) + '</article>'


def detail_html(instance: Path, directory: Path, query: dict) -> str:
    manifest = report._json(directory / 'manifest.json')
    files = _files(directory, manifest)
    data = report._json(files['content/knowledge.json']) if 'content/knowledge.json' in files else {}
    ref = directory.name
    url = report._version_url(instance, 'share', ref)
    body = f'<p class="eyebrow">SHARE</p><h1>Share · {report._e(ref)}</h1>'
    body += '<p>当前基线绑定：' + report._e(' / '.join(current_bindings(instance, ref)) or '无') + '</p>'
    body += '<p class="help">共享资料版本独立保存；Gen 与 Judge 可各自绑定不同版本。下方展示此快照的实际内容。</p>'
    if manifest.get('runtime_usable') is False:
        body += '<p class="notice">资料登记状态：审阅材料；登记本身不代表已进入模型输入，实际使用须查看模型配置。</p>'
    body += '<p>内容摘要：<code>' + report._e(manifest.get('sha256')) + '</code></p>'
    downloads = []
    for name, label in [('content/wiki.xml', '下载完整 XML'), ('content/knowledge.json', '下载完整 JSON')]:
        if name in files:
            href = '/instances/' + '/'.join(quote(part, safe='') for part in
                (instance.name, 'share', ref, *Path(name).parts))
            downloads.append(f'<a class="button" href="{href}" download>{label}</a>')
    if downloads:
        body += '<nav class="report-nav">' + ''.join(downloads) + '</nav>'
    collection = query.get('view', [next((key for key in COLLECTIONS if data.get(key)), 'pages')])[0]
    if collection not in COLLECTIONS:
        raise ValueError('未知 Share 资料类别')
    page = int(query.get('page', ['1'])[0])
    if not 1 <= page <= 1_000_000:
        raise ValueError('页码须为正整数')
    search = query.get('q', [''])[0].strip()
    body += '<nav class="report-nav" aria-label="Share 资料类别">'
    for key, label in COLLECTIONS.items():
        if data.get(key):
            body += f'<a href="{url}?{urlencode({"view": key})}#share-records">{label} · {len(data[key])}</a>'
    body += '</nav><section class="panel" id="share-records"><h2>' + COLLECTIONS[collection] + '</h2>'
    body += (f'<form action="{url}" method="get"><input type="hidden" name="view" value="{collection}">'
             f'<label>搜索 <input name="q" value="{report._e(search)}" placeholder="姓名、内容或来源 ID"></label>'
             '<button class="button" type="submit">查询</button></form>')
    rows = _rows(data, collection)
    if search:
        matching = {e['id'] for e in data.get('entities', []) if search.casefold() in str(e).casefold()}
        rows = [row for row in rows if search.casefold() in str(row).casefold()
                or any(eid in str(row) for eid in matching)]
    body += f'<p>共 {len(rows)} 条 · 第 {page} 页 · 每页 {PAGE_SIZE} 条</p>'
    selected = rows[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]
    body += ''.join(_record(row, data, url) for row in selected) or '<p>本页无记录。</p>'
    body += '<nav class="pager">'
    for target, label in [(page - 1, '上一页'), (page + 1, '下一页')]:
        if target >= 1 and (target < page or page * PAGE_SIZE < len(rows)):
            body += f'<a class="button" href="{url}?{urlencode({"view": collection, "q": search, "page": target})}#share-records">{label}</a>'
    body += '</nav></section><section class="panel"><h2>冻结资料文件</h2><nav class="report-nav">'
    body += ''.join(f'<a href="{url}?{urlencode({"file": name})}#share-file">{report._e(name)}</a>' for name in files)
    body += '</nav>'
    if 'file' in query:
        filename = query['file'][0]
        if filename not in files:
            raise PermissionError('此文件不在 Share 清单中')
        with files[filename].open('r', encoding='utf-8') as stream:
            text = stream.read(128001)
        body += '<div id="share-file"><h3>' + report._e(filename) + '</h3>' + report._pre(text[:128000])
        if len(text) > 128000:
            body += '<p>文件预览显示前 128,000 字符；完整结构化内容可通过上方分类分页浏览。</p>'
        body += '</div>'
    body += '</section>'
    return report._page(f'Share {ref}', body, instance.name)
