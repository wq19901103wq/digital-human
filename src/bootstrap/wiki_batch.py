"""Frozen, resumable library jobs using legacy pages and raw metadata only.

Display names locate candidate histories, never confirm identities. Each job is
account-scoped; groups/topics and ambiguous names remain explicit coverage gaps.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

from ..config import sha256_file
from ..iteration.storage import file_lock, locked, read_json, write_json
from . import wiki_job_pool, wiki_prompt, wiki_repair
from .wiki_library import digest, inventory


def pipeline():
    from . import wiki_mentions, wiki_reconcile, wiki_structured
    return {Path(module.__file__).name: sha256_file(Path(module.__file__))
            for module in (wiki_repair, wiki_mentions, wiki_reconcile, wiki_structured)}


def raw_accounts(messages, self_account):
    names, counts, locations = defaultdict(set), Counter(), {}
    hasher = hashlib.sha256()
    with Path(messages).open('rb') as stream:
        for line, raw in enumerate(stream, 1):
            hasher.update(raw)
            row = json.loads(raw)
            parts = row['chat_id'].split(':', 2)
            if len(parts) != 3 or parts[1] != self_account:
                continue
            sender = str((row.get('event') or {}).get('sender_id') or '')
            if not sender or row['is_self']:
                continue
            counts[sender] += 1
            for name in (row.get('sender'), row.get('chat_name')
                         if parts[0] == 'private' and parts[2] == sender else None):
                if not name:
                    continue
                names[name].add(sender)
                locations.setdefault((name, sender), dict(line=line, message_id=row['message_id'],
                    chat_id=row['chat_id'], timestamp=row['timestamp'], sender_id=sender,
                    row_sha256=hashlib.sha256(raw).hexdigest()))
    return names, counts, locations, hasher.hexdigest()


def prepare(wiki, messages, output, self_account, config, *, max_chars=120000, reuse=()):
    """Inventory once, retain every input page and every unresolved mapping."""
    output, wiki, messages = Path(output), Path(wiki), Path(messages)
    with file_lock(output / '.lock', blocking=False):
        if (output / 'manifest.json').exists():
            raise ValueError('batch already prepared; use run to resume')
        docs = inventory(wiki)
        names, counts, locations, source_sha = raw_accounts(messages, self_account)
        previous = []
        for path in reuse:
            directory = Path(path).resolve()
            manifest = read_json(directory / 'manifest.json')
            progress = read_json(directory / 'progress.json')
            source = directory / 'content/knowledge.json'
            if (progress['stage'] != 'complete' or manifest['messages']['sha256'] != source_sha
                    or manifest['subject']['self_account'] != self_account
                    or manifest['config'] != config or not manifest.get('structured_stage')
                    or not manifest.get('reconciliation_stage') or not manifest.get('mentions_stage')
                    or manifest['max_chars'] != max_chars):
                raise ValueError(f'incompatible or incomplete reusable run: {directory}')
            previous.append((manifest, source))
        jobs, gaps = [], []
        for doc in docs:
            base = dict(id=doc['id'], title=doc['title'], kind=doc['kind'], sources=doc['sources'])
            if doc['empty'] or doc['kind'] != 'person':
                gaps.append(dict(base, reason='empty' if doc['empty'] else 'requires_group_or_topic_pipeline'))
                continue
            exact = {Path(s['path']).stem for s in doc['sources']} & set(counts)
            candidates = exact or set().union(*(names[n] for n in (doc['title'], doc['filename'])))
            if len(candidates) != 1:
                gaps.append(dict(base, reason='ambiguous_name' if candidates else 'no_raw_account',
                                 candidates=sorted(candidates)))
                continue
            account = next(iter(candidates))
            locators = [dict(locations[n, account], kind='raw_message', path=str(messages.resolve()),
                             sha256=source_sha) for n in (doc['title'], doc['filename'])
                        if (n, account) in locations]
            job = dict(base, account=account, self_account=self_account, raw_sender_rows=counts[account],
                binding='account_filename' if exact else 'unverified_name_candidate',
                binding_evidence=locators,
                wiki=doc['sources'][0], output=str((output / 'jobs' / doc['id']).resolve()))
            for manifest, source in previous:
                if (manifest['subject']['account'] == account and any(
                        all(s[k] == manifest['wiki'][k] for k in ('path', 'sha256')) for s in doc['sources'])):
                    job['reuse'] = dict(path=str(source), sha256=sha256_file(source))
                    break
            jobs.append(job)
        # Reuse first; shorter histories then deliver complete outputs promptly.
        jobs.sort(key=lambda j: ('reuse' not in j, j['raw_sender_rows'], j['id']))
        manifest = dict(schema='wiki_batch_v1', wiki_root=str(wiki.resolve()),
            messages=dict(path=str(messages.resolve()), sha256=source_sha), self_account=self_account,
            config=config, max_chars=max_chars, jobs=jobs, gaps=gaps,
            pipeline=pipeline(),
            identity_policy='raw-name matches are leads, not confirmed page/account bindings',
            input_policy='legacy_wiki_and_raw_chats_only; no curated reviews',
            summary=dict(files=sum(len(d['sources']) for d in docs), documents=len(docs),
                jobs=len(jobs), reusable=sum('reuse' in j for j in jobs),
                bindings=dict(Counter(j['binding'] for j in jobs)),
                gaps=dict(Counter(g['reason'] for g in gaps))))
        write_json(output / 'manifest.json', manifest)
        return manifest['summary']


def _stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def status(output):
    """Report saved progress without mistaking interrupted jobs for live work."""
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    progress = read_json(output / 'progress.json', default={})
    running = locked(output / '.lock') and progress.get('stage') == 'running'
    active_ids = (progress.get('active_jobs', [progress.get('active_job')]) if running else [])
    counts = dict.fromkeys(('complete', 'running', 'incomplete', 'interrupted', 'pending'), 0)
    active_jobs = []
    for job in manifest['jobs']:
        item = progress.get('jobs', {}).get(job['id'], {})
        stage = item.get('stage', 'pending')
        if stage == 'running' and job['id'] not in active_ids:
            stage = 'interrupted'
        counts[stage] += 1
        if stage == 'running' and job['id'] in active_ids:
            detail = read_json(Path(job['output']) / 'progress.json', default={})
            active_jobs.append(dict(id=job['id'], title=job['title'], account=job['account'],
                progress={k: detail[k] for k in ('stage', 'completed', 'batches', 'failed',
                    'sections_completed', 'reread_completed', 'updated_at') if k in detail}))
    stage = progress.get('stage', 'prepared')
    if stage == 'running' and not running:
        stage = 'interrupted'
    return dict(stage=stage, running=running, total=len(manifest['jobs']), counts=counts,
                active=active_jobs[0] if active_jobs else None, active_jobs=active_jobs,
                object_workers=progress.get('object_workers', 1), workers=progress.get('workers'),
                updated_at=progress.get('updated_at'),
                coverage_gaps=manifest['summary']['gaps'])


def run(output, *, workers=4, object_workers=1, attempts=2, max_jobs=None, skip_incomplete=False):
    """Continue missing jobs; successful exact-input work is never requested again."""
    if workers < 1 or object_workers < 1 or attempts < 1 or (max_jobs is not None and max_jobs < 1):
        raise ValueError('workers, object_workers, attempts and optional max_jobs must be positive')
    output = Path(output)
    with file_lock(output / '.lock', blocking=False):
        manifest = read_json(output / 'manifest.json')
        if manifest['pipeline'] != pipeline():
            raise ValueError('generation pipeline changed; prepare a new batch')
        source = manifest['messages']
        source_stamp = _stamp(source['path'])
        if sha256_file(Path(source['path'])) != source['sha256']:
            raise ValueError('raw source changed after batch preparation')
        state = read_json(output / 'progress.json', default=dict(jobs={}))
        state.update(stage='running', pid=os.getpid(), total=len(manifest['jobs']),
                     object_workers=object_workers, workers=workers, skip_incomplete=skip_incomplete)
        def save():
            state['counts'] = dict(Counter(v['stage'] for v in state['jobs'].values()))
            state['updated_at'] = time.time()
            write_json(output / 'progress.json', state)
        def pending_jobs():
            for job in manifest['jobs']:
                if _stamp(source['path']) != source_stamp:
                    raise ValueError('raw source changed during batch')
                prior = state['jobs'].get(job['id'], {})
                if prior.get('stage') == 'complete':
                    if sha256_file(Path(prior['knowledge'])) != prior['sha256']:
                        raise ValueError('completed batch artifact changed')
                    wiki_prompt.export(prior['knowledge'], output / 'prompt' / job['id'])
                    continue
                if skip_incomplete and prior.get('stage') == 'incomplete':
                    continue
                yield job

        def execute(job, attempt):
            if _stamp(source['path']) != source_stamp:
                raise ValueError('raw source changed during batch')
            for locator in job['sources']:
                if sha256_file(Path(locator['path'])) != locator['sha256']:
                    raise ValueError('legacy Wiki changed after batch preparation')
            knowledge = Path(job['output']) / 'content/knowledge.json'
            if job.get('reuse'):
                knowledge = Path(job['reuse']['path'])
                if sha256_file(knowledge) != job['reuse']['sha256']:
                    raise ValueError('reusable Wiki changed')
            else:
                for _ in range(attempts):
                    attempt()
                    result = wiki_repair.repair(Path(job['wiki']['path']), Path(source['path']),
                        Path(job['output']), job['account'], job['self_account'], manifest['config'],
                        workers=workers, max_chars=manifest['max_chars'], structured=True,
                        include_mentions=True, reconcile_attributes=True)
                    if result['stage'] == 'complete':
                        break
                else:
                    raise RuntimeError(f"repair incomplete: {result.get('errors', [])}")
            if _stamp(source['path']) != source_stamp:
                raise ValueError('raw source changed during job')
            exported = wiki_prompt.export(knowledge, output / 'prompt' / job['id'])
            return dict(stage='complete', knowledge=str(knowledge), sha256=sha256_file(knowledge),
                        records=exported['records'], packets=exported['packets'])

        wiki_job_pool.run_jobs(pending_jobs(), execute, object_workers=object_workers,
                              state=state, save=save, max_jobs=max_jobs)
        complete = state['counts'].get('complete', 0)
        state['stage'] = 'complete' if complete == state['total'] else 'incomplete'
        state['coverage_gaps'] = manifest['summary']['gaps']
        save()
        return state
