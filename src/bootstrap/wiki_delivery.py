"""Read-only artifact checks and an automatically refreshed Wiki batch catalog.

The catalog indexes existing results; it never generates facts, resolves names,
or admits offline cards into model prompts.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

from ..config import sha256_file
from ..iteration.storage import atomic_write, file_lock, read_json, write_json
from . import wiki_batch, wiki_objects, wiki_prompt, wiki_structured
from .wiki_library import digest


VERIFIED_BINDINGS = wiki_objects.VERIFIED_BINDINGS
BODY_LABELS = dict(complete='正文完成', in_progress='仍在整理', evidence_gap='无可用正文',
                   identity_pending='身份待核实')


def object_catalog(jobs):
    """Group references by scoped account IDs, never merge facts or display names."""
    objects, topics, unresolved = {}, [], []
    for job in jobs:
        ref = dict(batch=job['batch'], id=job['id'])
        if job['kind'] == 'topic':
            topics.append(ref)
            continue
        owner = job.get('self_account', '')
        account = job.get('account', '')
        parts = job.get('chat_id', '').split(':', 2)
        if job['kind'] == 'person' and owner and account:
            key = ('person', owner, account)
        elif (job['kind'] in ('group', 'conversation') and owner and len(parts) == 3
              and parts[0] == 'group' and parts[1] == owner and parts[2]):
            key = ('group', owner, parts[2])
        else:
            unresolved.append(ref)
            continue
        obj = objects.setdefault(key, dict(id='object-' + digest(key)[:20], kind=key[0],
            self_account=owner, account=key[2], jobs=[]))
        obj['jobs'].append(job)
    result = []
    for key in sorted(objects):
        obj = objects[key]
        members = obj['jobs']
        completed = [j for j in members if wiki_objects.completed(j)]
        readable = [j for j in members if wiki_objects.readable(j)]
        bound = [j for j in readable if j['binding'] in VERIFIED_BINDINGS]
        completed_bound = [j for j in completed if j['binding'] in VERIFIED_BINDINGS]
        bound_jobs = sum(j['binding'] in VERIFIED_BINDINGS for j in members)
        if not bound_jobs:
            body_status = 'identity_pending'
        elif len(completed_bound) < bound_jobs:
            body_status = 'in_progress'
        else:
            body_status = 'complete' if bound else 'evidence_gap'
        # A candidate document's title is not an established name of this account.
        obj.update(title=bound[0]['title'] if bound else obj['account'],
            counts=dict(Counter(j['stage'] for j in members)), total_jobs=len(members),
            readable_jobs=len(readable), readable_bound_jobs=len(bound),
            completed_bound_jobs=len(completed_bound),
            empty_jobs=len(completed) - len(readable),
            empty_bound_jobs=len(completed_bound) - len(bound),
            bound_jobs=bound_jobs, body_status=body_status,
            identity_review_jobs=len(members) - bound_jobs,
            all_jobs_complete=len(completed) == len(members),
            jobs=[dict(batch=j['batch'], id=j['id']) for j in members], runtime_usable=False)
        result.append(obj)
    summary = {}
    for kind in ('person', 'group'):
        selected = [o for o in result if o['kind'] == kind]
        summary[kind] = dict(total=len(selected),
            readable=sum(o['readable_jobs'] > 0 for o in selected),
            readable_with_verified_binding=sum(o['readable_bound_jobs'] > 0 for o in selected),
            body_complete=sum(o['body_status'] == 'complete' for o in selected),
            body_in_progress=sum(o['body_status'] == 'in_progress' for o in selected),
            body_evidence_gap=sum(o['body_status'] == 'evidence_gap' for o in selected),
            with_empty_materials=sum(o['empty_jobs'] > 0 for o in selected),
            identity_only=sum(o['body_status'] == 'identity_pending' for o in selected),
            with_identity_review=sum(o['identity_review_jobs'] > 0 for o in selected),
            all_jobs_complete=sum(o['all_jobs_complete'] for o in selected))
    summary.update(total=len(result), unresolved_jobs=len(unresolved),
        topics=dict(total=len(topics), readable=sum(j['kind'] == 'topic' and
                    wiki_objects.readable(j) for j in jobs),
            empty=sum(j['kind'] == 'topic' and wiki_objects.completed(j)
                      and not wiki_objects.readable(j) for j in jobs)))
    return dict(objects=result, topics=topics, unresolved=unresolved, summary=summary)


def content_issues(data):
    """Check source roles and account anchors without rewriting any facts."""
    issues = wiki_structured.attribution_issues(data)
    for record in data['addresses']:
        ref = str(record['utterance_line'])
        evidence = data['evidence'][ref]
        if wiki_structured.address_needs_source_review(record['usage'], evidence.get('message_kind')):
            issues.append(dict(record_id=record['id'],
                reason='forwarded_address_speaker_unresolved', evidence_refs=[ref],
                usage=record['usage'], speaker_account=record['speaker_account']))
    return issues


def inspect_job(batch, job, saved):
    """Check the saved completion and all its projections without modifying them."""
    expected = Path(job.get('reuse', {}).get('path') or Path(job['output']) / 'content/knowledge.json')
    source = Path(saved['knowledge'])
    if source.resolve() != expected.resolve() or sha256_file(source) != saved['sha256']:
        raise ValueError('completed knowledge path or hash changed')
    data = read_json(source)
    if (data['subject']['account'] != job['account'] or
            data['subject']['self_account'] != job['self_account']):
        raise ValueError('knowledge account differs from frozen job')
    export = batch / 'prompt' / job['id']
    manifest = read_json(export / 'manifest.json')
    if (manifest['schema'] != 'wiki_prompt_export_v1' or
            Path(manifest['source']).resolve() != source.resolve() or
            manifest['source_sha256'] != saved['sha256'] or
            manifest.get('runtime_usable') is not False or
            manifest.get('historical_input_status') != 'requires_case_projection'):
        raise ValueError('prompt export binding or admission status changed')
    for name in ('cards.jsonl', 'sources.json'):
        if sha256_file(export / name) != manifest['files'][name]:
            raise ValueError(f'{name} hash differs from export manifest')
    cards = [json.loads(line) for line in (export / 'cards.jsonl').read_text().splitlines()]
    if cards != list(wiki_prompt.packets(data, manifest['max_chars'])):
        raise ValueError('cards differ from complete structured records')
    records = wiki_prompt.records(data)
    if any(item['records'] != len(records) or item['packets'] != len(cards)
           for item in (saved, manifest)):
        raise ValueError('saved record or packet count differs from artifacts')
    sources = read_json(export / 'sources.json')
    expected_sources = dict(records={r['id']: dict(evidence_refs=r['evidence_refs'],
        fact_ids=r.get('fact_ids', [])) for r in records},
        entities={e['id']: e['evidence_refs'] for e in data['entities']}, evidence=data['evidence'])
    if sources != expected_sources:
        raise ValueError('source sidecar differs from structured evidence')
    rendered = wiki_structured.render(data)
    if data['subject'].get('kind') in ('group', 'topic'):
        rendered = rendered.replace('\n账号：\n', '\n类型：' + data['subject']['kind'] + '\n', 1)
        rendered = rendered.replace('## 人物属性\n', '## 群与话题属性\n', 1)
    if (source.with_name('wiki.md').read_text() != rendered or
            source.with_name('wiki.xml').read_text() != wiki_structured.to_xml(data)):
        raise ValueError('Markdown or XML differs from structured knowledge')
    files = {name: source.with_name(name) for name in ('knowledge.json', 'wiki.md', 'wiki.xml')}
    files.update({name: export / name for name in ('cards.jsonl', 'sources.json', 'manifest.json')})
    return dict(records=len(records), packets=len(cards), content_issues=content_issues(data),
        files={name: dict(path=str(path.resolve()), sha256=sha256_file(path)) for name, path in files.items()})


def _checked(batch, job, saved, cache):
    source = Path(saved['knowledge'])
    paths = [source.with_name(n) for n in ('knowledge.json', 'wiki.md', 'wiki.xml')]
    paths += [batch / 'prompt' / job['id'] / n for n in ('manifest.json', 'cards.jsonl', 'sources.json')]
    fingerprint = (json.dumps([job, saved], sort_keys=True), tuple(wiki_batch._stamp(p) for p in paths))
    key = str(batch), job['id']
    if key not in cache or cache[key][0] != fingerprint:
        result = inspect_job(batch, job, saved)
        if fingerprint[1] != tuple(wiki_batch._stamp(p) for p in paths):
            raise ValueError('artifacts changed while being checked')
        cache[key] = fingerprint, result
    return cache[key][1]


def _revision_sources(manifests):
    revisions = {}
    for path, manifest in manifests.items():
        if not manifest.get('revision'):
            continue
        binding = manifest['revision']['source']
        source = str(Path(binding['path']).resolve().parent)
        if source not in manifests or sha256_file(Path(binding['path'])) != binding['sha256']:
            raise ValueError('revision requires its unchanged original batch')
        if source in revisions:
            raise ValueError('choose one revision per original batch')
        original = {j['id']: j for j in manifests[source]['jobs']}
        revised = {j['id']: j for j in manifest['jobs']}
        if original.keys() != revised.keys() or any(
                any(original[j].get(k) != revised[j].get(k) for k in
                    ('kind', 'account', 'self_account', 'chat_id', 'binding')) for j in original):
            raise ValueError('revision inventory or identity differs from original')
        revisions[source] = path
    for source in revisions:
        seen, cursor = set(), source
        while cursor in revisions:
            if cursor in seen:
                raise ValueError('cyclic revision sources')
            seen.add(cursor)
            cursor = revisions[cursor]
    return revisions


def _overlay_revisions(jobs, revisions):
    lookup = {(j['batch'], j['id']): j for j in jobs}
    selected, counts = [], Counter()
    for job in jobs:
        if job['batch'] in revisions.values():
            continue
        revision = revisions.get(job['batch'])
        if revision:
            while revision in revisions:
                revision = revisions[revision]
            revised = lookup[revision, job['id']]
            counts[revised['stage']] += 1
            if revised['stage'] in ('complete', 'invalid', 'needs_review'):
                job = dict(revised, original_batch=job['batch'])
            else:
                job = dict(job, revision_batch=revision, revision_stage=revised['stage'])
                if job['stage'] == 'complete':
                    # Old content remains archived; an opted-in revision must finish
                    # before it can enter the current object body or prompt cards.
                    job.update(stage='revision_' + revised['stage'], reuse_eligible=False)
                if revised.get('error'):
                    job['error'] = revised['error']
        selected.append(job)
    return selected, dict(counts)


def _require_revision_original(manifest, job_id, binding, seen=None):
    seen = set() if seen is None else seen
    source = Path(manifest['revision']['source']['path']).parent
    if source in seen:
        raise ValueError('cyclic revision source binding')
    seen.add(source)
    if sha256_file(source / 'manifest.json') != manifest['revision']['source']['sha256']:
        raise ValueError('revision source manifest changed')
    original = read_json(source / 'progress.json')['jobs'][job_id]
    if (original['stage'] != 'complete' or binding !=
            dict(path=original['knowledge'], sha256=original['sha256'])):
        raise ValueError('revision no longer matches completed original')
    if sha256_file(Path(original['knowledge'])) != original['sha256']:
        raise ValueError('revision original knowledge changed')
    parent = read_json(source / 'manifest.json')
    if parent.get('revision'):
        _require_revision_original(parent, job_id, original.get('revision_source'), seen)


def inspect(batches, *, cache=None):
    """Only verified completed pages are deliverable; retain every other job/gap."""
    cache = {} if cache is None else cache
    manifests = {str(Path(p).resolve()): read_json(Path(p) / 'manifest.json') for p in batches}
    if len(manifests) != len(batches):
        raise ValueError('duplicate batch path')
    revisions = _revision_sources(manifests)
    replaced = {}
    for path, manifest in manifests.items():
        if manifest['schema'] not in ('wiki_batch_v1', 'wiki_supplement_v1'):
            raise ValueError('unsupported Wiki batch schema')
        parent = manifest.get('parent')
        if parent:
            parent_dir = str(Path(parent['path']).resolve().parent)
            if sha256_file(Path(parent['path'])) != parent['sha256']:
                raise ValueError('supplement parent changed')
            if parent_dir in replaced:
                raise ValueError('choose one supplement per parent; do not mix replaced batches')
            replaced[parent_dir] = {j['id'] for j in manifest['jobs'] + manifest['gaps']}
    jobs, gaps, summaries = [], [], []
    for path, manifest in manifests.items():
        batch = Path(path)
        progress = read_json(batch / 'progress.json', default={})
        status = wiki_batch.status(batch)
        active = status.get('active_jobs', [status['active']] if status['active'] else [])
        active_ids = {item['id'] for item in active}
        summaries.append(dict(path=path, manifest_sha256=sha256_file(batch / 'manifest.json'), **status))
        for job in manifest['jobs']:
            saved = progress.get('jobs', {}).get(job['id'], {})
            stage = saved.get('stage', 'pending')
            if stage == 'running' and (not status['running'] or job['id'] not in active_ids):
                stage = 'interrupted'
            item = {k: job[k] for k in ('id', 'title', 'kind', 'account', 'binding')}
            item['self_account'] = job.get('self_account', manifest.get('self_account', ''))
            item['chat_id'] = job.get('chat_id', '')
            item.update(batch=path, stage=stage, runtime_usable=False, reuse_eligible=False)
            if stage == 'complete':
                try:
                    if manifest.get('revision'):
                        _require_revision_original(manifest, job['id'], saved.get('revision_source'))
                        item['revision_source'] = saved['revision_source']
                    item.update(_checked(batch, job, saved, cache), verified=True)
                    if item['content_issues']:
                        item['stage'] = 'needs_review'
                    else:
                        item['reuse_eligible'] = True
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    item.update(stage='invalid', verified=False, error=f'{type(exc).__name__}: {exc}')
            elif saved.get('error'):
                item['error'] = saved['error']
            jobs.append(item)
        if not manifest.get('revision'):
            gaps.extend(dict(g, batch=path) for g in manifest['gaps']
                        if g['id'] not in replaced.get(path, set()))
    jobs, revision_counts = _overlay_revisions(jobs, revisions)
    counts = dict(Counter(j['stage'] for j in jobs))
    reusable = [j for j in jobs if j['reuse_eligible']]
    review = [j for j in jobs if j['stage'] == 'needs_review']
    complete = counts.get('complete', 0) == len(jobs) and all(b['stage'] == 'complete' for b in summaries)
    grouped = object_catalog(jobs)
    return dict(schema='wiki_batch_delivery_v1', updated_at=datetime.now(timezone.utc).isoformat(),
        stage='complete' if complete else 'incomplete', runtime_usable=False,
        batches=summaries, jobs=jobs, gaps=gaps, objects=grouped['objects'],
        topics=grouped['topics'], unresolved_jobs=grouped['unresolved'],
        summary=dict(total_jobs=len(jobs), counts=counts, coverage_gaps=dict(Counter(g['reason'] for g in gaps)),
            objects=grouped['summary'], revisions=revision_counts,
            records=sum(j['records'] for j in reusable), packets=sum(j['packets'] for j in reusable),
            review_records=sum(j['records'] for j in review),
            content_issues=sum(len(j['content_issues']) for j in review)))


def require_valid(batches, *, require_complete=False, catalog=None):
    value = inspect(batches)
    invalid = [dict(batch=j['batch'], id=j['id'], error=j['error']) for j in value['jobs'] if j['stage'] == 'invalid']
    if invalid:
        raise ValueError(json.dumps(invalid, ensure_ascii=False))
    review = [dict(batch=j['batch'], id=j['id'], content_issues=j['content_issues'])
              for j in value['jobs'] if j['stage'] == 'needs_review']
    if review:
        raise ValueError('Wiki content needs review: ' + json.dumps(review, ensure_ascii=False))
    if require_complete and value['stage'] != 'complete':
        raise ValueError('Wiki generation is not complete: ' + json.dumps(value['summary'], ensure_ascii=False))
    objects = require_catalog(catalog, batches) if catalog is not None else None
    return dict(**value['summary'], stage=value['stage'], runtime_usable=False,
                object_artifacts=objects)


def require_catalog(output, batches):
    """Verify the delivered snapshot against its immutable source artifacts."""
    output = Path(output)
    value = read_json(output / 'catalog.json')
    if {b['path'] for b in value['batches']} != {str(Path(b).resolve()) for b in batches}:
        raise ValueError('catalog batch selection differs from requested batches')
    manifests = {}
    for batch in value['batches']:
        path = Path(batch['path']) / 'manifest.json'
        if sha256_file(path) != batch['manifest_sha256']:
            raise ValueError('catalog frozen batch manifest changed')
        manifests[batch['path']] = read_json(path)
    _revision_sources(manifests)
    for job in value['jobs']:
        manifest = manifests[job['batch']]
        if manifest.get('revision') and job['stage'] == 'complete':
            _require_revision_original(manifest, job['id'], job.get('revision_source'))
    grouped = object_catalog(value['jobs'])
    if grouped['objects'] != value['objects'] or grouped['summary'] != value['summary']['objects']:
        raise ValueError('catalog object coverage differs from saved jobs')
    if (output / 'index.md').read_text() != '\n'.join(_object_index(value, output)) + '\n':
        raise ValueError('object index differs from saved coverage')
    lookup = {(j['batch'], j['id']): j for j in value['jobs']}
    checked, records = Counter(), 0
    for obj in value['objects']:
        members = [lookup[r['batch'], r['id']] for r in obj['jobs']]
        rendered, files = _object_page(obj, members, output)
        page = output / 'objects' / (obj['id'] + '.md')
        if page.read_text() != rendered:
            raise ValueError('object reading entry differs from verified materials: ' + obj['id'])
        for name, content in files.items():
            if (output / 'objects' / obj['id'] / name).read_text() != content:
                raise ValueError('object artifact differs from verified materials: ' + obj['id'] + '/' + name)
        if files:
            checked[obj['kind']] += 1
            records += json.loads(files['manifest.json'])['records']
    return dict(updated_at=value['updated_at'], objects=dict(checked), records=records,
                coverage=value['summary']['objects'], runtime_usable=False)


def _cell(value):
    return str(value).replace('|', '／').replace('\n', ' ').replace('\r', ' ')


def _link(output, path, label):
    return '[' + label + '](' + quote(os.path.relpath(path, output), safe='/') + ')'


def _files(job, output):
    return ' · '.join(_link(output, job['files'][name]['path'], label) for name, label in
        [('wiki.md', '阅读'), ('knowledge.json', 'JSON'), ('wiki.xml', 'XML'),
         ('cards.jsonl', '卡片'), ('sources.json', '来源定位')])


def _object_page(obj, members, output):
    path = output / 'objects' / (obj['id'] + '.md')
    lines = ['# ' + _cell(obj['title']), '', '账号／群号：' + _cell(obj['account']), '',
        '资料所属本人账号：' + _cell(obj['self_account']), '',
        '**' + BODY_LABELS[obj['body_status']] + '**：'
        f"本批归属明确的资料已处理 {obj['completed_bound_jobs']} / {obj['bound_jobs']} 份，"
        f"其中 {obj['empty_bound_jobs']} 份无有效记录；"
        f"另有 {obj['identity_review_jobs']} 份材料的身份或归属待核实。"
        '正文完成指本批明确归属资料处理完且有有效记录，不表示全部历史或语义均已核实。', '',
        f"可读资料 {obj['readable_jobs']} / {obj['total_jobs']} 份；"
        f"其中 {obj['readable_bound_jobs']} 份归属明确，合入下方正文。"
        '同一来源不重复计证；不同表述保留原有时期与性质，不自动判为互相印证。'
        '旧资料归属待核实的部分单列在末尾。', '']
    files = {}
    if obj['readable_bound_jobs']:
        data = wiki_objects.compose(obj, members)
        files = wiki_objects.artifacts(data)
        lines += [' · '.join(_link(path.parent, path.with_suffix('') / name, label)
            for name, label in [('knowledge.json', '合并 JSON'), ('wiki.xml', 'XML'),
                               ('cards.jsonl', '离线卡片'), ('sources.json', '来源定位')]), '',
            f"合并后 {data['coverage']['records']} 条记录，"
            f"引用 {data['coverage']['unique_evidence']} 条去重来源。"
            '这是现有已完成资料的整理，尚未进入生产 prompt。', '',
            # Keep one title and account header in the object entry.
            '\n'.join(files['wiki.md'].splitlines()[3:]), '']
    else:
        lines += ['当前没有可合入正文的已完成、归属明确的材料。', '']
    for bound, heading in [(True, '归属有明确依据的资料'), (False, '旧资料归属待核实')]:
        lines += ['', '## ' + heading, '', '| 来源标题 | 状态 | 定位依据 | 文件 |',
                  '| --- | --- | --- | --- |']
        for job in members:
            if (job['binding'] in VERIFIED_BINDINGS) != bound:
                continue
            links = _files(job, path.parent) if job['stage'] == 'complete' else '待完成或待检查'
            status = '已处理，无有效记录；待核实项见原文' if (wiki_objects.completed(job)
                and not wiki_objects.readable(job)) else job['stage']
            lines.append('| ' + ' | '.join([_cell(job['title']), status, job['binding'], links]) + ' |')
    return '\n'.join(lines) + '\n', files


def _object_index(value, output):
    """One reading entry per account; uncertain legacy bindings stay marked."""
    lookup = {(j['batch'], j['id']): j for j in value['jobs']}
    summary = value['summary']['objects']
    text = ['# 按聊天对象查看 Wiki', '', '更新时间：' + value['updated_at'], '',
        '同一账号或群号只计一个对象；人物包含未加好友的群成员。话题另列。', '',
        '正文完成：本批归属明确的材料全部处理完且有有效记录。仍在整理：存在未完成或未通过检查的明确归属材料。'
        '无可用正文：明确归属的材料已处理完，但没有有效记录，保留资料缺口。'
        '身份待核实：该对象目前只有候选材料，尚无可确认归属的正文。'
        '这些状态不表示覆盖了全部历史或核实了全部语义。', '',
        '有正文的对象也可能另有身份待核实材料；候选资料始终单列，不合入正文。'
        '“含身份待核实材料”和“含空材料”可与前面状态重叠，不另行相加。', '',
        '| 对象 | 总数（含候选） | 正文完成 | 仍在整理 | 无可用正文 | 仅身份待核实 | 含身份待核实材料 | 含空材料 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for kind, label in [('person', '人物'), ('group', '群')]:
        stats = summary[kind]
        text.append(f"| {label} | {stats['total']} | {stats['body_complete']} | "
                    f"{stats['body_in_progress']} | {stats['body_evidence_gap']} | {stats['identity_only']} | "
                    f"{stats['with_identity_review']} | {stats['with_empty_materials']} |")
    text += ['', '以上仅统计已定位到账号／群号的任务，含尚待确认的候选关联；无法定位的资料见缺口清单。', '',
        '[身份待核实](identity-review.md) · [内容待核实](content-review.md) · '
        '[资料缺口与未完成任务](coverage.md) · [原始整理任务](materials.md) · [机器索引](catalog.json)', '']
    if value['summary'].get('revisions'):
        text += ['回源修订：' + '、'.join(f'{k} {v}' for k, v in value['summary']['revisions'].items()) + '。'
                 '已完成修订替代对应旧稿，只计一次；等待修订的旧稿保留存档，暂不合入当前正文。', '']
    for kind, label in [('person', '人物'), ('group', '群')]:
        text += ['', '## ' + label]
        for status, heading in BODY_LABELS.items():
            selected = [o for o in value['objects'] if o['kind'] == kind and o['body_status'] == status]
            text += ['', f'### {heading}（{len(selected)}）', '',
                '| 聊天对象 | 账号／群号 | 有效正文材料 / 明确归属总数 | 身份待核实材料 | 原材料状态 |',
                '| --- | --- | --- | --- | --- |']
            for obj in selected:
                path = output / 'objects' / (obj['id'] + '.md')
                text.append('| ' + ' | '.join([_link(output, path, _cell(obj['title'])),
                    _cell(obj['account']), f"{obj['readable_bound_jobs']} / {obj['bound_jobs']}",
                    str(obj['identity_review_jobs']),
                    '、'.join(f'{stage} {count}' for stage, count in sorted(obj['counts'].items()))]) + ' |')
    text += ['', '## 话题', '', f"有有效正文 {summary['topics']['readable']} / {summary['topics']['total']} 项；"
             f"另有 {summary['topics']['empty']} 项已处理但无有效记录。", '',
             '| 话题 | 状态 | 文件 |', '| --- | --- | --- |']
    for ref in value['topics']:
        job = lookup[ref['batch'], ref['id']]
        text.append('| ' + ' | '.join([_cell(job['title']), job['stage'],
            _files(job, output) if job['stage'] == 'complete' else '待完成']) + ' |')
    return text


def _write_objects(value, output):
    lookup = {(j['batch'], j['id']): j for j in value['jobs']}
    for obj in value['objects']:
        path = output / 'objects' / (obj['id'] + '.md')
        members = [lookup[r['batch'], r['id']] for r in obj['jobs']]
        rendered, files = _object_page(obj, members, output)
        wiki_objects.write(files, path.with_suffix(''))
        if not path.exists() or path.read_text() != rendered:
            atomic_write(path, rendered)


def write_catalog(value, output):
    output = Path(output)
    _write_objects(value, output)
    summary = value['summary']
    complete = [j for j in value['jobs'] if j['stage'] == 'complete']
    text = ['# Wiki 原始整理任务', '', '[按聊天对象查看](index.md)', '', '更新时间：' + value['updated_at'], '',
        f"已核对且未命中已知待核实规则：{len(complete)} / {summary['total_jobs']} 份材料"
        f"（其中 {sum(not wiki_objects.readable(j) for j in complete)} 份无有效记录）；"
        f"{summary['records']} 条结构化记录；{summary['packets']} 个卡片包。", '',
        f"另有 {summary['counts'].get('needs_review', 0)} 页因内容归属待核实单列，"
        f"涉及 {summary['content_issues']} 条记录；这些页面未计入以上可复用数量。", '',
        '批次状态：' + value['stage'] + '。记录与各格式一致性已核对；不代表语义无误或身份全部确认。', '',
        '卡片供离线复用，仍须按题筛选时间与来源后才能进入 prompt。', '',
        '[内容待核实](content-review.md) · [身份待核实](identity-review.md) · '
        '[资料缺口及未完成任务](coverage.md) · [机器索引](catalog.json)', '',
        '| 页面 | 类型 | 身份定位 | 记录 / 包 | 文件 |', '| --- | --- | --- | --- | --- |']
    for job in complete:
        links = _files(job, output)
        text.append('| ' + ' | '.join([_cell(job['title']), job['kind'], job['binding'],
            f"{job['records']} / {job['packets']}", links]) + ' |')
    review = ['# 内容待核实', '',
        '以下页面已生成且格式一致，但命中了来源角色检查，暂不作为可复用成果。原始产物保持不变。', '',
        '检查人物账号的发言锚点、自述主体与原始作者、转发中的内外层称呼。'
        '需回读来源后再决定归属；通过这些必要检查不代表语义完全正确。', '',
        '| 页面 | 记录 ID | 原始行号 | 用法／性质 | 归属账号 | 原因 |', '| --- | --- | --- | --- | --- | --- |']
    for job in value['jobs']:
        for issue in job.get('content_issues', []):
            links = _link(output, job['files']['wiki.md']['path'], _cell(job['title']))
            links += ' · ' + _link(output, job['files']['sources.json']['path'], '来源定位')
            review.append('| ' + links + ' | ' + ' | '.join(map(_cell, [issue['record_id'],
                ', '.join(issue['evidence_refs']), issue.get('usage', issue.get('mode', '')),
                issue.get('speaker_account', ', '.join(issue.get('accounts', []))), issue['reason']])) + ' |')
    identities = ['# 身份待核实', '', '以下状态直接来自冻结清单；目录不推断或合并身份。', '',
        '## 同名或归属证据不足', '', '| 页面 | 原因 | 候选账号 |', '| --- | --- | --- |']
    ambiguous = [g for g in value['gaps'] if g.get('candidates')]
    for gap in ambiguous:
        identities.append('| ' + ' | '.join(map(_cell, [gap['title'], gap['reason'],
            ', '.join(gap['candidates'])])) + ' |')
    identities += ['', '## 仅由唯一显示名定位的候选', '',
        '唯一名字匹配仍不是身份确认；此处包括尚未生成的候选任务。', '',
        '| 页面 | 候选账号或群定位 | 状态 |', '| --- | --- | --- |']
    for job in value['jobs']:
        if job['binding'] in ('unverified_name_candidate', 'unique_group_name_candidate'):
            identities.append('| ' + ' | '.join(map(_cell, [job['title'],
                job['account'] or job['chat_id'] or job['binding'], job['stage']])) + ' |')
    coverage = ['# 覆盖与未完成项', '', '任务完成与资料覆盖分开统计，缺口不会记作生成成功。', '',
        '| 缺口原因 | 材料数 |', '| --- | --- |']
    coverage += [f'| {reason} | {count} |' for reason, count in summary['coverage_gaps'].items()]
    unresolved = {(r['batch'], r['id']) for r in value['unresolved_jobs']}
    coverage += ['', '## 无法定位聊天对象的任务', '',
        '账号或所属本人账号缺失、群号范围不一致的任务不计入对象覆盖；仅保留原始定位。', '',
        '| 材料 | 本人账号 | 对方账号／群定位 | 状态 |', '| --- | --- | --- | --- |']
    coverage += ['| ' + ' | '.join(map(_cell, [j['title'], j['self_account'],
        j['account'] or j['chat_id'], j['stage']])) + ' |'
        for j in value['jobs'] if (j['batch'], j['id']) in unresolved]
    coverage += ['', '## 已处理但无有效正文', '',
        '以下材料已完成整理，但没有可复用记录；审阅结果和待核实原因保留在原材料中，不计为可读正文。', '',
        '| 材料 | 类型 | 定位依据 | 审阅结果 |', '| --- | --- | --- | --- |']
    coverage += ['| ' + ' | '.join([_cell(j['title']), j['kind'], j['binding'], _files(j, output)]) + ' |'
        for j in value['jobs'] if wiki_objects.completed(j) and not wiki_objects.readable(j)]
    coverage += ['', '## 未完成任务', '', '| 页面 | 状态 | 错误（如有） |', '| --- | --- | --- |']
    coverage += ['| ' + ' | '.join(map(_cell, [j['title'], j['stage'], j.get('error') or
        ', '.join(sorted({i['reason'] for i in j.get('content_issues', [])}))])) + ' |'
                 for j in value['jobs'] if j['stage'] != 'complete']
    coverage += ['', '## 资料缺口', '', '| 页面 | 原因 |', '| --- | --- |']
    coverage += ['| ' + _cell(g['title']) + ' | ' + g['reason'] + ' |' for g in value['gaps']]
    for name, lines in [('content-review.md', review), ('identity-review.md', identities),
                        ('coverage.md', coverage), ('materials.md', text),
                        ('index.md', _object_index(value, output))]:
        atomic_write(output / name, '\n'.join(lines) + '\n')
    write_json(output / 'catalog.json', value)


def deliver(batches, output, *, follow=False, interval=30):
    """Refresh as running batches finish; stop explicitly if generation stops."""
    output = Path(output).resolve()
    if any(output == Path(p).resolve() or output in Path(p).resolve().parents for p in batches):
        raise ValueError('catalog must use a separate output directory')
    with file_lock(output / '.lock', blocking=False):
        cache = {}
        while True:
            value = inspect(batches, cache=cache)
            write_catalog(value, output)
            print(json.dumps(dict(stage=value['stage'], **value['summary']), ensure_ascii=False), flush=True)
            if not follow or not any(b['running'] for b in value['batches']):
                return value
            time.sleep(interval)
