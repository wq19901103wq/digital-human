"""Complete frozen Wiki inventory gaps without modifying an active person batch."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import os
import time

from ..config import sha256_file
from ..iteration.storage import file_lock, read_json, write_json
from . import wiki_batch, wiki_prompt, wiki_repair, wiki_scope, wiki_source_index
from .wiki_library import inventory


def pipeline():
    return wiki_batch.pipeline() | {Path(module.__file__).name: sha256_file(Path(module.__file__))
        for module in (wiki_scope, wiki_source_index, wiki_prompt)}


def prepare(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    with file_lock(output / '.lock', blocking=False):
        if (output / 'manifest.json').exists():
            raise ValueError('supplement already prepared; use run to resume')
        original = read_json(parent / 'manifest.json')
        if original['schema'] != 'wiki_batch_v1':
            raise ValueError('supplement requires an original account Wiki batch')
        messages = original['messages']
        source_stamp = wiki_batch._stamp(messages['path'])
        docs = {d['id']: d for d in inventory(original['wiki_root'])}
        index_path = output / 'source-index.sqlite3'
        wiki_source_index.build(messages['path'], index_path, original['self_account'], messages['sha256'])
        index = wiki_source_index.SourceIndex(index_path, messages['path'], messages['sha256'])
        jobs, gaps, identities = [], [], []
        try:
            for gap in original['gaps']:
                doc = docs.get(gap['id'])
                if not doc or doc['sources'] != gap['sources']:
                    raise ValueError('legacy Wiki changed since original inventory')
                base = dict(id=doc['id'], title=doc['title'], kind=doc['kind'], sources=doc['sources'])
                if gap['reason'] == 'ambiguous_name':
                    proof = wiki_source_index.resolve_person(index, doc['lines'], gap['candidates'])
                    identities.append(dict(base, **proof))
                    if not proof['account']:
                        gaps.append(dict(base, reason=proof['binding'], candidates=proof['candidates']))
                        continue
                    selection = dict(binding=proof['binding'], binding_evidence=proof['evidence'],
                        selected_rows=index.db.execute('SELECT COUNT(*) FROM messages WHERE sender_id=?',
                                                      (proof['account'],)).fetchone()[0])
                    account = proof['account']
                elif gap['reason'] == 'requires_group_or_topic_pipeline':
                    if doc['kind'] == 'topic':
                        selection = wiki_source_index.select_topic(index, doc['lines'])
                        selection['selected_rows'] = len(selection['lines'])
                    else:
                        selection = wiki_source_index.select_group(index, doc['lines'], doc['title'],
                            [Path(s['path']).stem for s in doc['sources']])
                    if not selection.get('selected_rows'):
                        gaps.append(dict(base, reason=('no_grounded_topic_anchors' if doc['kind'] == 'topic'
                                                       else selection['binding']), selection=selection))
                        continue
                    account = ''
                else:
                    gaps.append(gap)
                    continue
                jobs.append(dict(base, **selection, account=account, self_account=original['self_account'],
                    wiki=doc['sources'][0], output=str(output / 'jobs' / doc['id'])))
        finally:
            index.close()
        if wiki_batch._stamp(messages['path']) != source_stamp:
            raise ValueError('raw source changed during supplement preparation')
        jobs.sort(key=lambda j: (j['selected_rows'], j['id']))
        manifest = dict(schema='wiki_supplement_v1',
            parent=dict(path=str(parent / 'manifest.json'), sha256=sha256_file(parent / 'manifest.json')),
            messages=messages, self_account=original['self_account'], config=original['config'],
            max_chars=original['max_chars'], pipeline=pipeline(),
            index=dict(path=str(index_path), sha256=sha256_file(index_path)),
            jobs=jobs, gaps=gaps, identities=identities,
            input_policy='legacy_wiki_and_raw_chats_only; no curated reviews',
            summary=dict(original_gaps=len(original['gaps']), jobs=len(jobs),
                kinds=dict(Counter(j['kind'] for j in jobs)), bindings=dict(Counter(j['binding'] for j in jobs)),
                identity_results=dict(Counter(p['binding'] for p in identities)),
                gaps=dict(Counter(g['reason'] for g in gaps))))
        write_json(output / 'manifest.json', manifest)
        return manifest['summary']


def run(output, *, workers=4, attempts=2, max_jobs=None):
    if workers < 1 or attempts < 1 or (max_jobs is not None and max_jobs < 1):
        raise ValueError('workers, attempts and optional max_jobs must be positive')
    output = Path(output)
    with file_lock(output / '.lock', blocking=False):
        manifest = read_json(output / 'manifest.json')
        if manifest['pipeline'] != pipeline():
            raise ValueError('supplement pipeline changed; prepare a new supplement')
        for key in ('messages', 'parent', 'index'):
            source = manifest[key]
            if sha256_file(Path(source['path'])) != source['sha256']:
                raise ValueError(f'{key} changed since supplement preparation')
        raw = manifest['messages']
        stamps = {key: wiki_batch._stamp(manifest[key]['path']) for key in ('messages', 'parent', 'index')}
        def unchanged():
            for key, stamp in stamps.items():
                if wiki_batch._stamp(manifest[key]['path']) != stamp:
                    raise ValueError(f'{key} changed during supplement')
        state = read_json(output / 'progress.json', default=dict(jobs={}))
        state.update(stage='running', pid=os.getpid(), total=len(manifest['jobs']))
        def save():
            state['counts'] = dict(Counter(j['stage'] for j in state['jobs'].values()))
            state['updated_at'] = time.time()
            write_json(output / 'progress.json', state)
        save()
        index = wiki_source_index.SourceIndex(manifest['index']['path'], raw['path'], raw['sha256'])
        started = 0
        try:
            for job in manifest['jobs']:
                unchanged()
                previous = state['jobs'].get(job['id'], {})
                if previous.get('stage') == 'complete':
                    if sha256_file(Path(previous['knowledge'])) != previous['sha256']:
                        raise ValueError('completed supplement artifact changed')
                    wiki_prompt.export(previous['knowledge'], output / 'prompt' / job['id'])
                    continue
                if max_jobs is not None and started >= max_jobs:
                    break
                started += 1
                item = state['jobs'][job['id']] = dict(stage='running', attempts=previous.get('attempts', 0))
                state['active_job'] = job['id']
                save()
                try:
                    for locator in job['sources']:
                        if sha256_file(Path(locator['path'])) != locator['sha256']:
                            raise ValueError('legacy Wiki changed since supplement preparation')
                    rows = None if job['kind'] == 'person' else index.rows(
                        chat_id=job.get('chat_id'), lines=job.get('lines'))
                    coverage = dict(raw, selection=job.get('selection'), binding=job['binding'],
                        anchor_count=job.get('anchor_count'), anchors=job.get('anchors', []),
                        unmatched_old_lines=job.get('unmatched_old_lines', []))
                    for _ in range(attempts):
                        item['attempts'] += 1
                        save()
                        if job['kind'] == 'person':
                            result = wiki_repair.repair(Path(job['wiki']['path']), Path(raw['path']),
                                Path(job['output']), job['account'], job['self_account'], manifest['config'],
                                workers=workers, max_chars=manifest['max_chars'], structured=True,
                                include_mentions=True, reconcile_attributes=True)
                        else:
                            result = wiki_scope.generate(job, rows, coverage, Path(job['output']),
                                manifest['config'], workers=workers, max_chars=manifest['max_chars'])
                        if result['stage'] == 'complete':
                            break
                    else:
                        raise RuntimeError(f"generation incomplete: {result.get('errors', [])}")
                    unchanged()
                    knowledge = Path(job['output']) / 'content/knowledge.json'
                    exported = wiki_prompt.export(knowledge, output / 'prompt' / job['id'])
                    item.update(stage='complete', knowledge=str(knowledge), sha256=sha256_file(knowledge),
                                records=exported['records'], packets=exported['packets'])
                except Exception as exc:
                    item.update(stage='incomplete', error=f'{type(exc).__name__}: {exc}'[-1600:])
                save()
        finally:
            index.close()
        state['active_job'] = None
        state['stage'] = 'complete' if state['counts'].get('complete', 0) == state['total'] else 'incomplete'
        state['coverage_gaps'] = manifest['summary']['gaps']
        save()
        return state
