"""Revalidate exact historical responses through the unchanged Wiki pipeline."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
from threading import Lock

from ... import cache
from ...config import sha256_file
from ...iteration.storage import atomic_write, file_lock, read_json, write_json
from ...judge.corrected import CodexJudgeClient
from . import wiki_repair, wiki_selection
from .wiki_library import digest

RETRY_PREFIX = (
    '上次输出未通过格式及引用检查。根据下列历次机械检查结果重新输出完整 JSON，'
    '同时解决已指出的问题，避免修正一项后重新引入另一项。'
    '最新输出和历次错误都不是证据；只依据原任务中的资料修正，不补造来源。\n')


def original_prompt(prompt):
    if not prompt.startswith(RETRY_PREFIX):
        return prompt
    feedback = prompt[len(RETRY_PREFIX):]
    try:
        value, end = json.JSONDecoder().raw_decode(feedback)
    except ValueError:
        return prompt
    if not isinstance(value, dict) or not feedback[end:].startswith(wiki_repair.RETRY_MARKER):
        return prompt
    return feedback[end + len(wiki_repair.RETRY_MARKER):]


class CachedOutputMissing(RuntimeError):
    """The exact request has no historical output passing its original checks."""


class CachedClient(CodexJudgeClient):
    """Keep the frozen client identity without constructing a transport client."""

    def __init__(self, config, output):
        if config.get('provider') != 'codex_cli':
            raise ValueError('cached Wiki recovery requires the frozen codex_cli identity')
        self.config = dict(config)
        self.output = Path(output)
        self.accepted = {}
        self.missing = set()
        self.checked = 0
        self.guard = Lock()

    def run(self, prompt, schema=None):
        original = original_prompt(prompt)
        key = digest([self.cache_identity(), original, schema])
        history = []
        for path in (self.output / 'failures').glob(f'{key}-*.json'):
            suffix = path.stem.rsplit('-', 1)[1]
            if suffix.isdecimal():
                history.append((int(suffix), path))
        for attempt, path in sorted(history, reverse=True):
            saved = read_json(path)
            if (saved.get('request_sha256') != key or saved.get('attempt') != attempt
                    or saved.get('result') is None):
                continue
            text = json.dumps(saved['result'], ensure_ascii=False)
            with self.guard:
                self.checked += 1
            if cache.response_valid(text, require_validator=True):
                with self.guard:
                    self.accepted[key] = dict(result_sha256=digest(saved['result']),
                        failure_file=str(path.relative_to(self.output)), attempt=attempt)
                return text
        with self.guard:
            self.missing.add(key)
        raise CachedOutputMissing(f'no checked historical output for request {key}; no model request made')


def _files(output):
    paths = [output / 'manifest.json', output / 'progress.json']
    for name in ('cache', 'failures'):
        paths.extend(sorted((output / name).glob('*.json')))
    for name in ('content', 'intermediate'):
        paths.extend(sorted(path for path in (output / name).rglob('*') if path.is_file()))
    return {str(path.relative_to(output)): sha256_file(path) for path in paths if path.exists()}


def _sources(job, manifest, verified_source_stamp):
    source = manifest['messages']
    stamp = wiki_selection.file_stamp(source['path'])
    if verified_source_stamp is not None:
        if stamp != verified_source_stamp:
            raise ValueError('raw source changed after batch verification')
    elif wiki_selection.file_digest(source['path']) != source['sha256']:
        raise ValueError('raw source changed after batch preparation')
    result = {str(Path(source['path']).resolve()): dict(sha256=source['sha256'], stamp=stamp)}
    for locator in [job['wiki'], *job.get('sources', [])]:
        path = str(Path(locator['path']).resolve())
        if path in result:
            if result[path]['sha256'] != locator['sha256']:
                raise ValueError('conflicting Wiki source hashes')
            continue
        stamp = wiki_selection.file_stamp(path)
        if wiki_selection.file_digest(path) != locator['sha256']:
            raise ValueError('legacy Wiki changed after batch preparation')
        result[path] = dict(sha256=locator['sha256'], stamp=stamp)
    _check_sources(result)
    return result


def _check_sources(sources):
    if any(list(wiki_selection.file_stamp(path)) != list(saved['stamp']) for path, saved in sources.items()):
        raise ValueError('Wiki source changed during cached recovery')


def _fingerprint(output, job, manifest, sources):
    from .wiki_batch import pipeline
    from .wiki_constraints import provenance
    return digest(dict(job=job, source=manifest['messages'], config=manifest['config'],
        max_chars=manifest['max_chars'],
        contracts={name: manifest.get(name, 'legacy_v1') for name in
                   ('reread_contract', 'structured_contract', 'reference_contract')},
        pipeline=pipeline(), implementation=sha256_file(Path(__file__)), constraints=provenance(),
        client=CachedClient(manifest['config'], output).cache_identity(), files=_files(output),
        verified_sources=sources))


def request_location(job, manifest, receipt, *, sealed_result=False):
    from .wiki_constraints import provenance
    sources = receipt['verified_sources']
    _check_sources(sources)
    output = Path(job['output'])
    if (receipt['schema'] != 'wiki_request_location_v2'
            or receipt['source_job'] != job or receipt['source_manifest_sha256'] != digest(manifest)
            or digest(sources) != digest(_sources(job, manifest, None))
            or receipt['provenance'] != provenance()
            or receipt['fingerprint'] != _fingerprint(output, job, manifest, sources)):
        raise ValueError('request location receipt changed or stale')
    located = receipt['result']
    if sealed_result and (receipt.get('result_sha256') != digest(located)
            or receipt.get('source_files') != _files(output)
            or located.get('job_id') != job['id'] or located.get('stage') != 'incomplete'
            or type(located.get('model_requests')) is not int or located['model_requests'] != 0):
        raise ValueError('request location lacks sealed offline incomplete result')
    if sealed_result:
        replayed = _replay_location(job, manifest, sources, workers=8)
        if (not isinstance(located.get('errors'), list)
                or Counter(digest(item) for item in located['errors'])
                   != Counter(digest(item) for item in replayed['errors'])
                or any(digest(located.get(name)) != digest(replayed[name])
                       for name in ('requests', 'stage'))):
            raise ValueError('request location differs from trusted offline replay')
    return located


def rejected_request(job, manifest, receipt, *, retry_policy):
    located = request_location(job, manifest, receipt)
    output = Path(job['output'])
    requests, evidence = located['requests'], located['failure_evidence']
    if (len(requests) != 1 or len(evidence) != 1 or any(
            item.get('kind') != 'output_rejected' or not item.get('history_complete')
            or item.get('policy_attempt') != wiki_repair.RETRY_LIMIT
            or item.get('request_sha256') not in requests for item in evidence)):
        raise ValueError('request location lacks unique exhausted rejected-output evidence')
    key = next(iter(requests))
    failure = evidence[0]
    path = output / failure['failure_file']
    record = read_json(path)
    if (record.get('retry_policy', wiki_repair.RETRY_POLICY) != retry_policy
            or record.get('result') is None or record.get('request_sha256') != key
            or record.get('policy_attempt') != wiki_repair.RETRY_LIMIT
            or record.get('attempt') != failure.get('attempt')
            or path != output / 'failures' / f'{key}-{record.get("attempt")}.json'):
        raise ValueError(f'correction requires exhausted {retry_policy} output')
    return key, requests[key], failure, path


class RequestLocator(CachedClient):
    locate_only = True

    def __init__(self, config, output):
        super().__init__(config, output)
        self.requests = {}

    def locate_request(self, key, prompt, schema):
        from .wiki_constraints import wire_plan
        plan = wire_plan(prompt, schema)
        with self.guard:
            self.requests[key] = dict(prompt_sha256=digest(prompt), schema_sha256=digest(schema))
            if plan:
                self.requests[key]['plan'] = plan['seal']

    def run(self, prompt, schema=None):
        raise RuntimeError('request location must stop before client invocation')


def _replay_location(job, manifest, sources, *, workers):
    from . import wiki_batch
    output = Path(job['output'])
    before = _files(output)
    client = RequestLocator(manifest['config'], output)
    with tempfile.TemporaryDirectory(prefix='wiki-locate-') as directory:
        staged = Path(directory)
        for name in before:
            if name != 'manifest.json' and not name.startswith(('cache/', 'failures/')):
                continue
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / name, target)
        progress = wiki_repair.repair(Path(job['wiki']['path']), Path(manifest['messages']['path']),
            staged, job['account'], job['self_account'], manifest['config'], workers=workers,
            max_chars=manifest['max_chars'], structured=True, client=client,
            include_mentions=True, reconcile_attributes=True,
            reread_contract=manifest.get('reread_contract', 'legacy_v1'),
            structured_contract=manifest.get('structured_contract', 'legacy_v1'),
            reference_contract=manifest.get('reference_contract', 'legacy_v1'),
            **({'correction': job['correction']} if job.get('correction') else {}))
        records = wiki_batch._failure_records(staged, allow_legacy=True)
        evidence = [wiki_batch._failure_evidence(staged, item, records)
                    for item in progress.get('errors', [])]
        _check_sources(sources)
        if before != _files(output):
            raise ValueError('original Wiki files changed during request location')
        return dict(job_id=job['id'], stage=progress['stage'], model_requests=0,
            errors=progress.get('errors', []), failure_evidence=evidence,
            requests=client.requests, receipt_reused=False)


def _locate(job, manifest, destination, *, workers, verified_source_stamp):
    output = Path(job['output'])
    receipt_path = destination / f'{job["id"]}.json'
    with file_lock(output / '.lock', blocking=False):
        sources = _sources(job, manifest, verified_source_stamp)
        fingerprint = _fingerprint(output, job, manifest, sources)
        receipt = read_json(receipt_path, default={})
        if (receipt.get('fingerprint') == fingerprint
                and receipt.get('result_sha256') == digest(receipt.get('result'))):
            _check_sources(sources)
            return dict(receipt['result'], receipt_reused=True)
        before = _files(output)
        result = _replay_location(job, manifest, sources, workers=workers)
        from .wiki_constraints import provenance
        write_json(receipt_path, dict(schema='wiki_request_location_v2',
            fingerprint=fingerprint, source_job=job, source_manifest_sha256=digest(manifest),
            verified_sources=sources, source_files=before, provenance=provenance(),
            result=result, result_sha256=digest(result)))
        return result


def locate_requests(source, output, job_ids, *, workers=8, object_workers=4):
    """Replay only selected jobs with exact histories and no model or transport."""
    from . import wiki_batch
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('request diagnostics must be separate from the source batch')
    if not job_ids or len(set(job_ids)) != len(job_ids) or min(workers, object_workers) < 1:
        raise ValueError('unique selected jobs and positive workers required')
    with file_lock(source / '.lock', blocking=False), file_lock(output / '.lock', blocking=False):
        manifest = read_json(source / 'manifest.json')
        if manifest['schema'] != 'wiki_batch_v1' or not wiki_batch.compatible_pipeline(manifest['pipeline']):
            raise ValueError('request location requires a registered person batch')
        by_id = {job['id']: job for job in manifest['jobs']}
        if set(job_ids) - by_id.keys():
            raise ValueError('unknown selected Wiki jobs')
        stamp = wiki_selection.file_stamp(manifest['messages']['path'])
        if wiki_selection.file_digest(manifest['messages']['path']) != manifest['messages']['sha256']:
            raise ValueError('raw source changed after batch preparation')
        index = wiki_selection.for_batch(manifest, source, record_runtime=False,
                                        verified_source_stamp=stamp)
        with wiki_selection.accelerated(index), ThreadPoolExecutor(max_workers=object_workers) as pool:
            futures = [pool.submit(_locate, by_id[key], manifest, output, workers=workers,
                                   verified_source_stamp=stamp) for key in job_ids]
            results = [future.result() for future in futures]
        summary = dict(schema='wiki_request_locations_v1', model_requests=0,
            source=dict(path=str(source / 'manifest.json'), sha256=sha256_file(source / 'manifest.json')),
            jobs=results)
        write_json(output / 'locations.json', summary)
        return summary


def recover(job, manifest, *, workers=4, verified_source_stamp=None):
    """Validate in isolation; preserve production failures, inputs and budgets.

    The isolated directory has no correction history because it performs no
    new attempts. Historical candidates remain read-only in the original job.
    The original pipeline alone validates and writes eligible success caches.
    """
    output = Path(job['output'])
    receipt_path = output / 'cached_recovery.json'
    with file_lock(output / '.lock', blocking=False):
        sources = _sources(job, manifest, verified_source_stamp)
        fingerprint = _fingerprint(output, job, manifest, sources)
        receipt = read_json(receipt_path, default={})
        if receipt.get('fingerprint') == fingerprint:
            _check_sources(sources)
            return dict(receipt['result'], receipt_reused=True)
        before = _files(output)
        client = CachedClient(manifest['config'], output)
        result = dict(stage='incomplete', model_requests=0, recovered_requests=0,
                      checked_outputs=0, missing_requests=[], receipt_reused=False)
        with tempfile.TemporaryDirectory(prefix='wiki-cached-') as directory:
            staged = Path(directory)
            if (output / 'manifest.json').exists():
                shutil.copyfile(output / 'manifest.json', staged / 'manifest.json')
            for path in (output / 'cache').glob('*.json'):
                saved = read_json(path)
                if saved.get('request_sha256') == path.stem:
                    target = staged / 'cache' / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            try:
                progress = wiki_repair.repair(Path(job['wiki']['path']), Path(manifest['messages']['path']),
                    staged, job['account'], job['self_account'], manifest['config'], workers=workers,
                    max_chars=manifest['max_chars'], structured=True, client=client,
                    include_mentions=True, reconcile_attributes=True,
                    reread_contract=manifest.get('reread_contract', 'legacy_v1'),
                    structured_contract=manifest.get('structured_contract', 'legacy_v1'),
                    reference_contract=manifest.get('reference_contract', 'legacy_v1'))
            except Exception as exc:
                progress = dict(stage='incomplete', errors=[dict(error=f'{type(exc).__name__}: {exc}')])
            _check_sources(sources)
            if _files(output) != before:
                raise ValueError('original Wiki inputs or request history changed during cached recovery')
            if progress['stage'] == 'complete' and not (staged / 'content/knowledge.json').is_file():
                raise ValueError('cached recovery reported completion without readable knowledge')
            for key, evidence in client.accepted.items():
                path = staged / 'cache' / f'{key}.json'
                saved = read_json(path, default={})
                if saved.get('request_sha256') != key or digest(saved.get('result')) != evidence['result_sha256']:
                    continue
                attempts = [int(item.stem.rsplit('-', 1)[1])
                    for item in (output / 'failures').glob(f'{key}-*.json')
                    if item.stem.rsplit('-', 1)[1].isdecimal()]
                prior = read_json(output / 'cache' / path.name, default={})
                saved.update(attempts=max([prior.get('attempts', 0), *attempts]),
                             recovered_from=evidence)
                write_json(output / 'cache' / path.name, saved)
                result['recovered_requests'] += 1
            result.update(stage=progress['stage'], checked_outputs=client.checked,
                missing_requests=sorted(client.missing), errors=progress.get('errors', []))
            if progress['stage'] == 'complete':
                for name in ('content', 'intermediate'):
                    for path in (staged / name).rglob('*'):
                        if path.is_file():
                            atomic_write(output / path.relative_to(staged), path.read_text(encoding='utf-8'))
                write_json(output / 'progress.json', progress)
            write_json(receipt_path, dict(schema='wiki_cached_recovery_v1',
                fingerprint=_fingerprint(output, job, manifest, sources), result=result))
        return result
