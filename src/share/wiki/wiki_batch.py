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
import re
import time

from ...config import sha256_file
from ...judge import codex_transport
from ...iteration.storage import file_lock, locked, read_json, write_json
from . import wiki_connection_retry, wiki_job_pool, wiki_prompt, wiki_repair, wiki_selection
from .wiki_library import digest, inventory
from .wiki_objects import VERIFIED_BINDINGS


FEEDBACK_PIPELINE_UPDATES = {
    'wiki_repair.py': ('113b3c3445698773eb571286f6dcba9a1be03456cc0af4e141c7a0fcc5478de8',
                       '509566dd8d22a67cbf93da00d192944f025cce581fb4d3d995fc164ce363d1d4'),
    'wiki_structured.py': ('4809954a35b934dc678be21d2a20f570a61cb4b9d0aa628584ed7c667c7c9620',
                           'c00354d8384e078a2348e3466d067a9f05818f8f1c24000e4ff34ce20f7e03c3'),
    'wiki_revision.py': ('1984dc367ed08984f8939b55d666637065bd4cbd9097f8737eb52a7ef43a96bf',
                         '9f4992d9829ab0047830912183633f23964b7dd64804d8a968607398b715aa06'),
    'wiki_revision_claims.py': ('03fa67062f8f9c4f8072802c22a7f634ccf8c6ed4d8a0752f4ebd48a3bbb11d2',
                                '7df5a6cb1b121cf2e44fc71e4b44e65460ff0039c679833272762df228b96138'),
}

REREAD_IMPLEMENTATION_UPDATES = {
    'wiki_repair.py': ('509566dd8d22a67cbf93da00d192944f025cce581fb4d3d995fc164ce363d1d4',
                       '34ff485f3fcb6f6e6c257a8082088eac2344be5e38f7e61a206cc8bcc56347a4'),
    'wiki_mentions.py': ('012dd09f76df3166ba5f754ff1027fa2c276fc163b587a4a713312f2a3b9ec3b',
                         'f3975a5ca52a736fa39e63701f3ea2dd96417a40af177b73eb8d929b888a55da'),
}


STRUCTURED_IMPLEMENTATION_UPDATES = {
    'wiki_repair.py': ('34ff485f3fcb6f6e6c257a8082088eac2344be5e38f7e61a206cc8bcc56347a4',
                       '64a6b79311739b76fcbd61cbd3a97af735db442ca12752d47c2748b7e2b24e22'),
    'wiki_structured.py': ('c00354d8384e078a2348e3466d067a9f05818f8f1c24000e4ff34ce20f7e03c3',
                           '63922879a56e8ff465907a7ff5b2520cd5ac4e7390eee15fd6deefdba12cf90e'),
}


REFERENCE_IMPLEMENTATION_UPDATES = {
    'wiki_repair.py': ('64a6b79311739b76fcbd61cbd3a97af735db442ca12752d47c2748b7e2b24e22',
                       'ab53268eaf06c574e20f4c8c9c3d76812f7d8db0c82f247100b8f8b766653072'),
    'wiki_mentions.py': ('f3975a5ca52a736fa39e63701f3ea2dd96417a40af177b73eb8d929b888a55da',
                         'e3fe3552562f9330a33120b177e287aaba204c0972918bb7746dfb40fd8b64e5'),
    'wiki_structured.py': ('63922879a56e8ff465907a7ff5b2520cd5ac4e7390eee15fd6deefdba12cf90e',
                           '1f3cc787ade5806a305c4b9493c9d82ebf8dca10c051d6ccfc89c9cfd4acf810'),
    'wiki_reconcile.py': ('46c2c5d2e5bb4f584baeeca835e4c14ba2b0817cd7ae46021f381f64eb76eccf',
                          '4663c7db102c33d8287ae4505fb77196d4dd833af46c876c659ef7ec9206aa05'),
}


TRANSPORT_IMPLEMENTATION_UPDATES = {
    'wiki_repair.py': ('ab53268eaf06c574e20f4c8c9c3d76812f7d8db0c82f247100b8f8b766653072',
                       '8073f66bb240174c99ea276c087997df898953c91359529c6811a197f5955cdc'),
}


TRANSPORT_DIAGNOSTIC_IMPLEMENTATION_UPDATES = {
    'wiki_repair.py': ('8073f66bb240174c99ea276c087997df898953c91359529c6811a197f5955cdc',
                       '409bcaec8b780d0a5705db955697659de489f5ba0688b1de65c357b66289b105'),
}


XML_EXPORT_IMPLEMENTATION_UPDATES = {
    'wiki_structured.py': ('1f3cc787ade5806a305c4b9493c9d82ebf8dca10c051d6ccfc89c9cfd4acf810',
                           '815d08073eddcd1159b04d2be8b424854eaf4f6fa4c7bd9dc57862c49246cef5'),
}


TRANSPORT_WRAPPER_IMPLEMENTATION_UPDATES = {
    'wiki_repair.py': ('409bcaec8b780d0a5705db955697659de489f5ba0688b1de65c357b66289b105',
                       '897781de635a679b36f0d6906c1b4e5fdb2af51151fa5c520859fa1bbf6cdc4d'),
}


RETRY_TRANSPORT_SIGNALS = frozenset({'rate_limit', 'stream_disconnected', 'request_timeout'})


REQUEST_LOCATION_IMPLEMENTATION_UPDATES = {
    'wiki_repair.py': ('1e37952445f8fa4b56797569066d488e9f19251ed34635db72a74b9d2a4b6a35',
                       '632041797c04bcf106bdd625cee1789de8d9d2094a9eaf3b135e46bf6d44a87a'),
    'wiki_structured.py': ('551872f9c67445118944417216870a88a5822576a172ba90ceb431252639d12b',
                           '12a395f3658420e16601de84f027a44ee601f508f0d0baf188bd6778068c0bb7'),
}


SECTION_EVIDENCE_DIAGNOSTIC_UPDATES = {
    'wiki_repair.py': ('632041797c04bcf106bdd625cee1789de8d9d2094a9eaf3b135e46bf6d44a87a',
                       'dcf9a6cb69fc7fd2b62b7b676447e603c39e953e6458154679911af4cdf95770'),
}


FAILURE_LOCATION_DIAGNOSTIC_UPDATES = {
    'wiki_repair.py': ('dcf9a6cb69fc7fd2b62b7b676447e603c39e953e6458154679911af4cdf95770',
                       '06e9f7b9b39de02d8e302b8bc05a974d6c5b3a0e8763f9e9674540c5e440c540'),
    'wiki_mentions.py': ('e3fe3552562f9330a33120b177e287aaba204c0972918bb7746dfb40fd8b64e5',
                         '1c95b8b04955395bdf24b6ae7242ca9514ff649a2550d94aaa61e55036673a06'),
    'wiki_reconcile.py': ('4663c7db102c33d8287ae4505fb77196d4dd833af46c876c659ef7ec9206aa05',
                          '8cf651176335ad04bd07d0780436aadb86175b73309e6dfb7228547cac2ca86e'),
}


def previous_failure_location_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    updates = {name: pair for name, pair in FAILURE_LOCATION_DIAGNOSTIC_UPDATES.items() if name in current}
    if (set(current) != set(required_names) or not updates
            or any(current[name] != pair[1] for name, pair in updates.items())):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_section_evidence_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    current = previous_failure_location_pipeline(current, required_names=required_names) or current
    updates = {name: pair for name, pair in SECTION_EVIDENCE_DIAGNOSTIC_UPDATES.items() if name in current}
    if (set(current) != set(required_names) or not updates
            or any(current[name] != pair[1] for name, pair in updates.items())):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_request_location_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    updates = {name: pair for name, pair in REQUEST_LOCATION_IMPLEMENTATION_UPDATES.items() if name in current}
    if (set(current) != set(required_names) or not updates
            or any(current[name] != pair[1] for name, pair in updates.items())):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_transport_wrapper_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names):
        return None
    updates = {name: pair for name, pair in TRANSPORT_WRAPPER_IMPLEMENTATION_UPDATES.items() if name in current}
    if not updates or any(current[name] != pair[1] for name, pair in updates.items()):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_xml_export_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names):
        return None
    current = previous_transport_wrapper_pipeline(current, required_names=required_names) or current
    updates = {name: pair for name, pair in XML_EXPORT_IMPLEMENTATION_UPDATES.items() if name in current}
    if not updates or any(current[name] != pair[1] for name, pair in updates.items()):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_transport_diagnostic_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names):
        return None
    current = previous_transport_wrapper_pipeline(current, required_names=required_names) or current
    current = previous_xml_export_pipeline(current, required_names=required_names) or current
    updates = {name: pair for name, pair in TRANSPORT_DIAGNOSTIC_IMPLEMENTATION_UPDATES.items() if name in current}
    if not updates or any(current[name] != pair[1] for name, pair in updates.items()):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_transport_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names):
        return None
    current = previous_transport_diagnostic_pipeline(current, required_names=required_names) or current
    updates = {name: pair for name, pair in TRANSPORT_IMPLEMENTATION_UPDATES.items() if name in current}
    if not updates or any(current[name] != pair[1] for name, pair in updates.items()):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_reference_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names):
        return None
    current = previous_transport_pipeline(current, required_names=required_names) or current
    updates = {name: pair for name, pair in REFERENCE_IMPLEMENTATION_UPDATES.items() if name in current}
    if not updates or any(current[name] != pair[1] for name, pair in updates.items()):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_structured_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names):
        return None
    current = previous_reference_pipeline(current, required_names=required_names) or current
    updates = {name: pair for name, pair in STRUCTURED_IMPLEMENTATION_UPDATES.items() if name in current}
    if not updates or any(current[name] != pair[1] for name, pair in updates.items()):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def legacy_reread_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names):
        return None
    current = previous_structured_pipeline(current, required_names=required_names) or current
    updates = {name: pair for name, pair in REREAD_IMPLEMENTATION_UPDATES.items() if name in current}
    if not updates or any(current[name] != pair[1] for name, pair in updates.items()):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def previous_feedback_pipeline(current, *, required_names=None):
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names):
        return None
    current = legacy_reread_pipeline(current, required_names=required_names) or current
    updates = {name: pair for name, pair in FEEDBACK_PIPELINE_UPDATES.items() if name in current}
    if not updates or any(current[name] != pair[1] for name, pair in updates.items()):
        return None
    return current | {name: pair[0] for name, pair in updates.items()}


def compatible_pipeline(frozen, current=None, *, required_names=None):
    current = pipeline() if current is None else current
    required_names = pipeline().keys() if required_names is None else required_names
    if set(current) != set(required_names) or set(frozen) != set(required_names):
        return False
    if frozen == current:
        return True
    failure_location = previous_failure_location_pipeline(current, required_names=required_names)
    current = failure_location or current
    evidence = previous_section_evidence_pipeline(current, required_names=required_names)
    current = evidence or current
    previous = previous_feedback_pipeline(current, required_names=required_names)
    legacy = legacy_reread_pipeline(current, required_names=required_names)
    structured = previous_structured_pipeline(current, required_names=required_names)
    reference = previous_reference_pipeline(current, required_names=required_names)
    transport = previous_transport_pipeline(current, required_names=required_names)
    diagnostic = previous_transport_diagnostic_pipeline(current, required_names=required_names)
    xml_export = previous_xml_export_pipeline(current, required_names=required_names)
    wrapper = previous_transport_wrapper_pipeline(current, required_names=required_names)
    location = previous_request_location_pipeline(current, required_names=required_names)
    return any(snapshot is not None and frozen == snapshot
               for snapshot in (failure_location, current, location, wrapper, xml_export, diagnostic, transport, reference,
                                structured, legacy, previous))


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
    incomplete_jobs = []
    for job in manifest['jobs']:
        item = progress.get('jobs', {}).get(job['id'], {})
        stage = item.get('stage', 'pending')
        if stage == 'running' and job['id'] not in active_ids:
            stage = 'interrupted'
        counts[stage] += 1
        if stage == 'incomplete':
            detail = _incomplete_failure(job, item, inspect_records=True)
            incomplete_jobs.append(dict(id=job['id'], title=job['title'],
                reason=detail['reason'],
                terminal_signals=sorted({signal for diagnostic in detail['diagnostics']
                    for signal in diagnostic['terminal_signals'] if isinstance(signal, str)}),
                retry_transport_signals=[signal for signal in sorted(RETRY_TRANSPORT_SIGNALS)
                    if _retryable_failure(detail, {signal})],
                recovered_errors=detail['recovered_errors'],
                failure_evidence=detail['failure_evidence'],
                evidence_category=_evidence_category(detail),
                errors=[dict(stage=error.get('stage'), batch=error.get('batch'),
                    error=str(error.get('error', ''))[:300]) if isinstance(error, dict)
                    else dict(error=str(error)[:300]) for error in detail['non_transport_errors']]))
        if stage == 'running' and job['id'] in active_ids:
            detail = read_json(Path(job['output']) / 'progress.json', default={})
            active_jobs.append(dict(id=job['id'], title=job['title'], account=job['account'],
                progress={k: detail[k] for k in ('stage', 'completed', 'batches', 'failed',
                    'sections_completed', 'reread_completed', 'mention_batches',
                    'mention_completed', 'mention_reread_completed', 'updated_at') if k in detail}))
    stage = progress.get('stage', 'prepared')
    if stage == 'running' and not running:
        stage = 'interrupted'
    return dict(stage=stage, running=running, total=len(manifest['jobs']), counts=counts,
                active=active_jobs[0] if active_jobs else None, active_jobs=active_jobs,
                object_workers=progress.get('object_workers', 1), workers=progress.get('workers'),
                request_workers=progress.get('request_workers'),
                stage_workers=progress.get('stage_workers', progress.get('workers')),
                request_timeout_seconds=progress.get('request_timeout_seconds'),
                effective_request_timeout_seconds=progress.get('effective_request_timeout_seconds',
                    manifest.get('config', {}).get('timeout_seconds')),
                profile_requests=progress.get('profile_requests', False),
                profile_run_id=progress.get('profile_run_id'),
                connection_retry=wiki_connection_retry.status(output,
                    progress.get('connection_retry_run_id'), running=running),
                selected_jobs=progress.get('selected_jobs'),
                updated_at=progress.get('updated_at'),
                coverage_gaps=manifest['summary']['gaps'],
                failure_summary=dict(jobs=len(incomplete_jobs),
                    reasons=dict(Counter(job['reason'] for job in incomplete_jobs)),
                    recovered_jobs=sum(bool(job['recovered_errors']) for job in incomplete_jobs),
                    recovered_errors=sum(len(job['recovered_errors']) for job in incomplete_jobs),
                    retry_transport_signals={signal: sum(signal in job['retry_transport_signals']
                        for job in incomplete_jobs) for signal in sorted(RETRY_TRANSPORT_SIGNALS)}),
                failure_evidence_summary=dict(
                    categories=dict(Counter(job['evidence_category'] for job in incomplete_jobs)),
                    error_kinds=dict(Counter(evidence['kind'] for job in incomplete_jobs
                        for evidence in job['failure_evidence'])),
                    cached_request_files=sum(evidence.get('cache_file_present', False)
                        for job in incomplete_jobs for evidence in job['failure_evidence'])),
                incomplete_jobs=incomplete_jobs)


def _failure_tail_limit(item):
    if not isinstance(item, dict) or not isinstance(item.get('error'), str):
        return None
    limit = {None: 600, 'extracting': 600, 'drafting': 600, 'writing_final': 600,
             'rereading': 600, 'consolidating': 600, 'mentions_extract': 800,
             'mentions_reread': 800, 'structuring': 1000,
             'reconciling_attributes': 1000}.get(item.get('stage'))
    return limit if len(item['error']) == limit else None


def _failure_records(directory, *, allow_legacy=False):
    failures = directory / 'failures'
    if failures.is_symlink():
        return None
    records = []
    for path in sorted(failures.glob('*.json')):
        match = re.fullmatch(r'([0-9a-f]{64})-([1-9][0-9]*)\.json', path.name)
        if match is None or path.is_symlink():
            return None
        try:
            saved = read_json(path)
        except (OSError, ValueError, TypeError):
            return None
        if (not isinstance(saved, dict)
                or (saved.get('request_sha256') != match[1]
                    and not (allow_legacy and 'request_sha256' not in saved))
                or type(saved.get('attempt')) is not int or saved['attempt'] != int(match[2])
                or not isinstance(saved.get('error'), str)):
            return None
        records.append((path, saved))
    return records


def _record_error_match(item, error):
    if not isinstance(item, dict) or not isinstance(item.get('error'), str):
        return None
    if item['error'] == error:
        return 'exact'
    limit = _failure_tail_limit(item)
    if limit is not None and len(error) > limit and error[-limit:] == item['error']:
        return 'tail'
    return None


def _singleton_check_match(item, saved):
    if (not isinstance(item, dict) or not isinstance(item.get('error'), str)
            or saved.get('result') is None):
        return None
    error = item['error']
    wrapper = (f'unchanged request exhausted {wiki_repair.RETRY_LIMIT} correction attempts; '
               'last error: ')
    exhausted = error.startswith(wrapper)
    if exhausted:
        error = error[len(wrapper):]
    prefix = 'mechanical checks failed: '
    if not error.startswith(prefix):
        return None
    try:
        checks = json.loads(error[len(prefix):])
    except ValueError:
        return None
    if (not isinstance(checks, list) or len(checks) != 1
            or not isinstance(checks[0], dict) or checks[0].keys() != {'record', 'error'}
            or not isinstance(checks[0]['record'], str) or not checks[0]['record']
            or checks[0]['error'] != saved['error']):
        return None
    return checks[0]['record'], exhausted


def _failure_evidence(directory, item, records):
    evidence = {name: item[name] for name in ('stage', 'batch', 'section')
                if isinstance(item, dict) and name in item}
    if records is None:
        return evidence | dict(kind='unknown', reason='invalid_failure_history')
    histories = defaultdict(list)
    for path, saved in records:
        histories[path.stem.rsplit('-', 1)[0]].append((path, saved))
    matches = []
    for key, history in histories.items():
        if item.get('request_sha256') is not None and item['request_sha256'] != key:
            continue
        history.sort(key=lambda record: record[1]['attempt'])
        path, saved = history[-1]
        matched = _record_error_match(item, saved['error'])
        exhausted = False
        if matched is None:
            failures = [(path, record) for path, record in history
                        if not (record.get('result') is None
                                and wiki_repair._transport_diagnostic(record['error']))]
            path, saved = failures[-1] if failures else history[-1]
            wrapper = (f'unchanged request exhausted {wiki_repair.RETRY_LIMIT} correction attempts; '
                       f'last error: {saved["error"]}')
            matched = _record_error_match(item, wrapper)
            exhausted = matched is not None
        validation_record = None
        if matched is None:
            singleton = _singleton_check_match(item, saved)
            if singleton is not None:
                validation_record, exhausted = singleton
                matched = 'singleton_mechanical_check'
        if matched is not None:
            matches.append((key, history, path, saved, matched, exhausted, validation_record))
    if len(matches) != 1:
        return evidence | dict(kind='unknown', reason=(
            'ambiguous_failure_record' if matches else 'missing_failure_record'),
            matching_requests=len(matches))
    key, history, path, saved, matched, exhausted, validation_record = matches[0]
    if 'result' not in saved:
        kind = 'unknown'
    elif saved['result'] is not None:
        kind = 'output_rejected'
    elif wiki_repair._transport_diagnostic(saved['error']) is not None:
        kind = 'recorded_transport_failure'
    elif re.fullmatch(r'Command .+ timed out after [0-9]+(?:\.[0-9]+)? seconds',
                      saved['error'], re.DOTALL):
        kind = 'legacy_client_timeout'
    elif saved['error'].startswith('Codex Judge 执行失败（exit '):
        kind = 'legacy_client_failure'
    else:
        kind = 'unknown'
    cache_path = directory / 'cache' / f'{key}.json'
    evidence.update(kind=kind, request_sha256=key,
        request_sha256_source='record' if 'request_sha256' in saved else 'filename',
        failure_file=f'failures/{path.name}', attempt=saved['attempt'],
        latest_recorded_attempt=history[-1][1]['attempt'],
        history_complete=[record['attempt'] for _, record in history]
                         == list(range(1, history[-1][1]['attempt'] + 1)),
        result_present='result' in saved and saved['result'] is not None,
        match=matched, exhausted_wrapper=exhausted,
        cache_file_present=not cache_path.parent.is_symlink()
                           and not cache_path.is_symlink() and cache_path.is_file())
    for name in ('policy_attempt', 'retry_budget', 'retry_policy'):
        if name in saved:
            evidence[name] = saved[name]
    if validation_record is not None:
        evidence['validation_record'] = validation_record
    return evidence


def _evidence_category(detail):
    if detail['reason'] == 'transport_only':
        return 'transport_only'
    kinds = {evidence['kind'] for evidence in detail['failure_evidence']}
    if 'output_rejected' in kinds:
        return 'output_rejected'
    if kinds and kinds <= {'recorded_transport_failure', 'legacy_client_timeout', 'legacy_client_failure'}:
        return 'client_failure_evidence'
    return 'unresolved_failure_evidence'


def _pure_transport_record(saved):
    diagnostic = wiki_repair._transport_diagnostic(saved.get('error'))
    errors = saved.get('errors')
    if (diagnostic is None or 'result' not in saved or saved['result'] is not None
            or not isinstance(errors, list)
            or any(not isinstance(error, dict)
                   or wiki_repair._transport_diagnostic(error.get('error')) is None for error in errors)
            or ('failure_kind' in saved and saved['failure_kind'] != 'transport')
            or ('transport' in saved and saved['transport'] != diagnostic)
            or ('retry_policy' in saved and saved['retry_policy'] not in (
                wiki_repair.RETRY_POLICY, wiki_repair.CORRECTION_POLICY))):
        return None
    if 'retry_budget' in saved:
        if (saved['retry_budget'] != 'format_only_v1'
                or type(saved.get('policy_attempt')) is not int or saved['policy_attempt'] != 0):
            return None
    elif 'policy_attempt' in saved:
        if (type(saved['policy_attempt']) is not int
                or not 0 <= saved['policy_attempt'] <= min(saved['attempt'], wiki_repair.RETRY_LIMIT)):
            return None
    return diagnostic


def _recover_transport_failure(item, records):
    limit = _failure_tail_limit(item)
    if limit is None or records is None:
        return None
    matches = [(path, saved) for path, saved in records
               if len(saved['error']) > limit and saved['error'][-limit:] == item['error']]
    if len(matches) != 1:
        return None
    path, saved = matches[0]
    history = [record for _, record in records if record['request_sha256'] == saved['request_sha256']]
    if (saved['attempt'] != len(history)
            or sorted(record['attempt'] for record in history) != list(range(1, len(history) + 1))
            or any(_pure_transport_record(record) is None for record in history)):
        return None
    diagnostic = _pure_transport_record(saved)
    evidence = {name: item[name] for name in ('stage', 'batch', 'section') if name in item}
    evidence.update(request_sha256=saved['request_sha256'], attempt=saved['attempt'],
                    file=f'failures/{path.name}')
    return diagnostic, evidence


def _incomplete_failure(job, prior, *, inspect_records=False):
    directory = Path(job['output'])
    progress = read_json(directory / 'progress.json', default={})
    errors = progress.get('errors')
    result = dict(reason='missing_failure_evidence', diagnostics=[], non_transport_errors=[],
                  recovered_errors=[], failure_evidence=[])
    if progress.get('stage') != 'incomplete' or not isinstance(errors, list) or not errors:
        return result
    if prior.get('error') != f'RuntimeError: repair incomplete: {errors}'[-1600:]:
        return result | dict(reason='failure_evidence_mismatch')
    records, records_loaded = None, False
    for item in errors:
        details = wiki_repair._transport_diagnostic(item.get('error')) if isinstance(item, dict) else None
        if details is None and _failure_tail_limit(item) is not None:
            if not records_loaded:
                records = _failure_records(directory)
                records_loaded = True
            recovered = _recover_transport_failure(item, records)
            if recovered is not None:
                details, evidence = recovered
                result['recovered_errors'].append(evidence)
        if details is None:
            result['non_transport_errors'].append(item)
        else:
            result['diagnostics'].append(details)
    result['reason'] = 'non_transport_failure' if result['non_transport_errors'] else 'transport_only'
    if inspect_records and result['non_transport_errors']:
        if records is None:
            records = _failure_records(directory, allow_legacy=True)
        result['failure_evidence'] = [_failure_evidence(directory, item, records)
            for item in result['non_transport_errors']]
    return result


def _retryable_failure(detail, retry_signals=None):
    if detail['reason'] != 'transport_only':
        return False
    if retry_signals is not None:
        for diagnostic in detail['diagnostics']:
            terminal_signals = diagnostic['terminal_signals']
            if (not terminal_signals or not all(isinstance(signal, str) for signal in terminal_signals)
                    or not set(terminal_signals) <= retry_signals):
                return False
    return True


def _transport_incomplete(job, prior, *, retry_signals=None):
    return _retryable_failure(_incomplete_failure(job, prior), retry_signals)


def run(output, *, workers=4, object_workers=1, attempts=2, max_jobs=None, skip_incomplete=False,
        retry_transport_only=False, retry_transport_signals=None,
        wait=False, request_workers=None, request_timeout_seconds=None, profile_requests=False,
        cached_only=False, job_ids=None, connection_retry_seconds=None):
    """Continue missing jobs; successful exact-input work is never requested again."""
    if workers < 1 or object_workers < 1 or attempts < 1 or (max_jobs is not None and max_jobs < 1):
        raise ValueError('workers, object_workers, attempts and optional max_jobs must be positive')
    if skip_incomplete and retry_transport_only:
        raise ValueError('skip_incomplete and retry_transport_only are mutually exclusive')
    if cached_only and (skip_incomplete or retry_transport_only or retry_transport_signals is not None
                        or request_workers is not None or request_timeout_seconds is not None or profile_requests):
        raise ValueError('cached-only recovery cannot use live request or retry options')
    if retry_transport_signals is not None:
        if (not retry_transport_only or not isinstance(retry_transport_signals, (list, tuple, set, frozenset))
                or not retry_transport_signals
                or not all(isinstance(signal, str) for signal in retry_transport_signals)
                or not set(retry_transport_signals) <= RETRY_TRANSPORT_SIGNALS):
            raise ValueError('retry_transport_signals requires transport-only retry and known terminal signals')
        retry_transport_signals = frozenset(retry_transport_signals)
    if request_timeout_seconds is not None and (type(request_timeout_seconds) is not int or
                                              request_timeout_seconds < 1):
        raise ValueError('request timeout must be a positive integer')
    selected = None if job_ids is None else frozenset(job_ids)
    if selected is not None and not selected:
        raise ValueError('job_ids must not be empty')
    if connection_retry_seconds is not None:
        if (type(connection_retry_seconds) is not int or connection_retry_seconds < 1
                or attempts != 1 or selected is None or cached_only or skip_incomplete
                or retry_transport_only or retry_transport_signals is not None):
            raise ValueError('connection retry requires a positive interval, explicit job_ids and one live attempt')
    request_budget = wiki_job_pool.RequestBudget(request_workers) if request_workers is not None else None
    stage_workers = request_workers if request_budget is not None else workers
    output = Path(output)
    with file_lock(output / '.lock', blocking=wait):
        manifest = read_json(output / 'manifest.json')
        if selected is not None:
            unknown = selected - {job['id'] for job in manifest['jobs']}
            if unknown:
                raise ValueError(f'unknown Wiki jobs: {sorted(unknown)}')
        if not compatible_pipeline(manifest['pipeline']):
            raise ValueError('generation pipeline changed; prepare a new batch')
        completion = None
        recovery, claimed = None, frozenset()
        if manifest.get('client_recovery') or (output / '.client_recovery_v1-permit.json').exists():
            from . import wiki_client_recovery
            if attempts != 1 or skip_incomplete or retry_transport_only or cached_only:
                raise ValueError('client recovery requires one live repair attempt without other retry modes')
            recovery, claimed = wiki_client_recovery.validate_batch(manifest, output)
        if manifest.get('completion'):
            from . import wiki_completion
            completion = wiki_completion.validate_batch(manifest, output)
            if (attempts != 1 or (request_timeout_seconds is not None
                    and request_timeout_seconds != manifest['config'].get('timeout_seconds'))):
                raise ValueError('page completion preserves timeout and permits one continuation attempt')
        source = manifest['messages']
        source_stamp = _stamp(source['path'])
        selection_stamp = wiki_selection.file_stamp(source['path'])
        if wiki_selection.file_digest(source['path']) != source['sha256']:
            raise ValueError('raw source changed after batch preparation')
        state = read_json(output / 'progress.json', default=dict(jobs={}))
        connection_jobs = set()
        if connection_retry_seconds is not None:
            if recovery is not None and not selected <= set(recovery['job_ids']):
                raise ValueError('connection retry cannot expand client recovery scope')
            for job in manifest['jobs']:
                if job['id'] not in selected:
                    continue
                prior = state['jobs'].get(job['id'], {})
                if prior.get('stage') == 'complete':
                    continue
                detail = _incomplete_failure(job, prior)
                if (prior.get('stage') != 'incomplete' or not _retryable_failure(detail)
                        or not all(wiki_connection_retry.retryable_diagnostic(diagnostic)
                                   for diagnostic in detail['diagnostics'])):
                    raise ValueError(f"connection retry requires proven connection-only failure: {job['id']}")
                connection_jobs.add(job['id'])
        connections = (wiki_connection_retry.ConnectionRetries(output, connection_retry_seconds)
                       if connection_retry_seconds is not None else None)
        prior_jobs = {key: dict(value) for key, value in state['jobs'].items()} if cached_only else {}
        request_config = dict(manifest['config'])
        if request_timeout_seconds is not None:
            request_config['timeout_seconds'] = request_timeout_seconds
        timings = (wiki_job_pool.RequestTimings(output, timeout_seconds=request_config.get('timeout_seconds'))
                   if profile_requests else None)
        state.update(stage='running', pid=os.getpid(), total=len(manifest['jobs']),
                     object_workers=object_workers, workers=workers, skip_incomplete=skip_incomplete,
                     retry_transport_only=retry_transport_only,
                     retry_transport_signals=sorted(retry_transport_signals) if retry_transport_signals else None,
                     request_workers=request_workers, stage_workers=stage_workers,
                     request_timeout_seconds=request_timeout_seconds,
                     effective_request_timeout_seconds=request_config.get('timeout_seconds'),
                     profile_requests=profile_requests, profile_run_id=timings.run_id if timings else None,
                     connection_retry_seconds=connection_retry_seconds,
                     connection_retry_run_id=connections.state['run_id'] if connections else None,
                     selected_jobs=None if selected is None else sorted(selected))
        state['cached_only'] = cached_only
        if not cached_only:
            state.pop('blocker', None)
        def save():
            state['counts'] = dict(Counter(v['stage'] for v in state['jobs'].values()))
            state['updated_at'] = time.time()
            write_json(output / 'progress.json', state)
        def eligible(job):
            if selected is not None and job['id'] not in selected:
                return False
            prior = state['jobs'].get(job['id'], {})
            if connections is not None:
                return job['id'] in connection_jobs
            if recovery is not None:
                return job['id'] in recovery['job_ids'] and job['id'] not in claimed
            if completion is not None and job['id'] != completion['job_id']:
                return False
            if manifest.get('correction') and not retry_transport_only and not job.get('correction'):
                return False
            return (prior.get('stage') != 'incomplete' or
                    (not skip_incomplete and (not retry_transport_only or _transport_incomplete(
                        job, prior, retry_signals=retry_transport_signals))))

        def pending_jobs():
            ordered_jobs = sorted(manifest['jobs'], key=lambda job: job['binding'] not in VERIFIED_BINDINGS)
            for job in ordered_jobs:
                if _stamp(source['path']) != source_stamp:
                    raise ValueError('raw source changed during batch')
                prior = state['jobs'].get(job['id'], {})
                if prior.get('stage') == 'complete':
                    if sha256_file(Path(prior['knowledge'])) != prior['sha256']:
                        raise ValueError('completed batch artifact changed')
                    wiki_prompt.export(prior['knowledge'], output / 'prompt' / job['id'])
            for job in ordered_jobs:
                if _stamp(source['path']) != source_stamp:
                    raise ValueError('raw source changed during batch')
                if state['jobs'].get(job['id'], {}).get('stage') == 'complete':
                    continue
                if not eligible(job):
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
                if cached_only:
                    from . import wiki_cached
                    try:
                        recovered = wiki_cached.recover(job, manifest, workers=stage_workers,
                            verified_source_stamp=selection_stamp)
                    except Exception as exc:
                        recovered = dict(stage='incomplete', model_requests=0, recovered_requests=0,
                            errors=[dict(error=f'{type(exc).__name__}: {exc}')])
                    if recovered['stage'] != 'complete':
                        return dict(prior_jobs.get(job['id'], dict(stage='incomplete', account=job['account'],
                            attempts=0)), cached_recovery=recovered)
                    exported = wiki_prompt.export(knowledge, output / 'prompt' / job['id'])
                    return dict(stage='complete', knowledge=str(knowledge), sha256=sha256_file(knowledge),
                        records=exported['records'], packets=exported['packets'], cached_recovery=recovered)
                client = None
                if (completion is not None or request_budget is not None
                        or request_timeout_seconds is not None or timings is not None or connections is not None):
                    from ...judge.corrected import CodexJudgeClient
                    client = CodexJudgeClient(dict(request_config))
                    if request_budget is not None:
                        client = request_budget.wrap(client, timings=timings, job=job)
                    elif timings is not None:
                        client = wiki_job_pool.BudgetedClient(client, timings=timings, job=job)
                    if connections is not None:
                        client = connections.wrap(client, job)
                    if completion is not None:
                        client = wiki_completion.CompletionClient(client, completion)
                for _ in range(attempts):
                    if recovery is not None and job['id'] not in claimed:
                        wiki_client_recovery.claim(recovery, job['id'])
                    attempt()
                    result = wiki_repair.repair(Path(job['wiki']['path']), Path(source['path']),
                        Path(job['output']), job['account'], job['self_account'], manifest['config'],
                        workers=stage_workers, max_chars=manifest['max_chars'], structured=True,
                        client=client,
                        include_mentions=True, reconcile_attributes=True,
                        reread_contract=manifest.get('reread_contract', 'legacy_v1'),
                        structured_contract=manifest.get('structured_contract', 'legacy_v1'),
                        reference_contract=manifest.get('reference_contract', 'legacy_v1'),
                        **({'correction': job['correction']} if job.get('correction') else {}))
                    if result['stage'] == 'complete':
                        break
                    codex_transport.check_available()
                else:
                    raise RuntimeError(f"repair incomplete: {result.get('errors', [])}")
            if _stamp(source['path']) != source_stamp:
                raise ValueError('raw source changed during job')
            exported = wiki_prompt.export(knowledge, output / 'prompt' / job['id'])
            return dict(stage='complete', knowledge=str(knowledge), sha256=sha256_file(knowledge),
                        records=exported['records'], packets=exported['packets'])

        needs_selection = any(not job.get('reuse') and
            state['jobs'].get(job['id'], {}).get('stage') != 'complete' and
            eligible(job)
            for job in manifest['jobs'])
        try:
            if needs_selection:
                index = wiki_selection.for_batch(manifest, output, verified_source_stamp=selection_stamp)
                with wiki_selection.accelerated(index):
                    wiki_job_pool.run_jobs(pending_jobs(), execute, object_workers=object_workers,
                        state=state, save=save, max_jobs=max_jobs,
                        check_available=None if cached_only else codex_transport.check_available)
            else:
                wiki_job_pool.run_jobs(pending_jobs(), execute, object_workers=object_workers,
                    state=state, save=save, max_jobs=max_jobs,
                    check_available=None if cached_only else codex_transport.check_available)
        except codex_transport.CodexServiceBlocked as exc:
            state.update(stage='blocked', pid=None, blocker=exc.details)
            state['coverage_gaps'] = manifest['summary']['gaps']
            save()
            return state
        complete = state['counts'].get('complete', 0)
        state['stage'] = 'complete' if complete == state['total'] else 'incomplete'
        state['pid'] = None
        state['coverage_gaps'] = manifest['summary']['gaps']
        save()
        return state
