"""Versioned batch continuation after a registered compatible change.

Completed artifacts stay at their original paths. Exact request caches are
copied for missing/invalid work and revalidated by the normal generation entry.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil

from ...config import sha256_file
from ...iteration.storage import file_lock, read_json, write_json
from . import wiki_batch, wiki_delivery, wiki_prompt, wiki_repair, wiki_supplement


CORRECTION_BASE = {
    'wiki_repair.py': '897781de635a679b36f0d6906c1b4e5fdb2af51151fa5c520859fa1bbf6cdc4d',
    'wiki_structured.py': '815d08073eddcd1159b04d2be8b424854eaf4f6fa4c7bd9dc57862c49246cef5',
}


def correction_requests(job, saved):
    detail = wiki_batch._incomplete_failure(job, saved, inspect_records=True)
    selected = {}
    for evidence in detail['failure_evidence']:
        if (evidence['kind'] != 'output_rejected' or not evidence['history_complete']
                or evidence.get('policy_attempt') != wiki_repair.RETRY_LIMIT):
            continue
        path = Path(job['output']) / evidence['failure_file']
        record = read_json(path)
        if record.get('retry_policy', wiki_repair.RETRY_POLICY) != wiki_repair.RETRY_POLICY:
            raise ValueError('correction upgrade cannot reset an existing correction allowance')
        selected[evidence['request_sha256']] = dict(
            source=locator(path), file=evidence['failure_file'], evidence=evidence)
    return selected


# Only validation and its mechanical error feedback changed; prompts, schemas,
# selection and compilation stayed byte-identical. Other revisions fail closed.
VALIDATION_REVISIONS = {
    'wiki_repair.py': {
        ('fec0d48fbcfb4e9428101a9390681ceff7575295e0210f9d0025722f025df2d8',
         'dd1a8fdad053d40ae186245d6093b47f2244ceab3a775cf0b67831892fb69351'),
        # 2026-10-07 包迁移 src/bootstrap/ -> src/bootstrap/wiki/，仅 import 路径变化。
        ('dd1a8fdad053d40ae186245d6093b47f2244ceab3a775cf0b67831892fb69351',
         '113b3c3445698773eb571286f6dcba9a1be03456cc0af4e141c7a0fcc5478de8'),
    },
    'wiki_structured.py': {
        ('e4c66d3f9e0291baa233b86d8dbfb7f97742d25f4a7bb4a7920b737b56e3c97f',
         '3e6f9d331fc15e2b5e19eb721e2a10b80083a0269bca2ef25d665d25e7fee7ef'),
        ('3790674fc24cdcc702477a99357db1ea84bbb07003b9421c68f73280cff99fab',
         'b01ac97b31590500d4c72b7c9d3ab0e053943817be46a8f525ef3002bbacb5af'),
        ('3790674fc24cdcc702477a99357db1ea84bbb07003b9421c68f73280cff99fab',
         'e4c66d3f9e0291baa233b86d8dbfb7f97742d25f4a7bb4a7920b737b56e3c97f'),
        ('b01ac97b31590500d4c72b7c9d3ab0e053943817be46a8f525ef3002bbacb5af',
         'e4c66d3f9e0291baa233b86d8dbfb7f97742d25f4a7bb4a7920b737b56e3c97f'),
        # 2026-10-07 包迁移，仅 import 路径变化。
        ('3e6f9d331fc15e2b5e19eb721e2a10b80083a0269bca2ef25d665d25e7fee7ef',
         '4809954a35b934dc678be21d2a20f570a61cb4b9d0aa628584ed7c667c7c9620'),
    },
    'wiki_mentions.py': {
        # 2026-10-07 包迁移，仅 import 路径变化。
        ('f55ef49dea4f750bacd8bd5791cf4e07440bb28fd5c5e00f1b7c2a437ae331b3',
         '012dd09f76df3166ba5f754ff1027fa2c276fc163b587a4a713312f2a3b9ec3b'),
    },
    'wiki_prompt.py': {
        # 2026-10-07 包迁移，仅 import 路径变化。
        ('5f06f1b0782e5dbb9f5820ddcbed11053a11c72a8795c736fa70d7b641e3bb3c',
         'f1963c6621b2f24f42983ed20816b1d927b8c83b199e0388a754dc261c44591c'),
    },
    'wiki_scope.py': {
        # 2026-10-07 包迁移，仅 import 路径变化。
        ('a11ad72dc1b796bd9a5f7e7924411832f734d5e07938a5c0aac407d14c0d31d8',
         'f75cb0f6bbc042a3dc646ffd84fe32043f618a4339e0bf09c738d2d4303d3458'),
    },
    'wiki_reconcile.py': {
        ('2bae089c1bbd110667c83aff5fc17726cdbfd7450958271e066b2a267813a036',
         '46c2c5d2e5bb4f584baeeca835e4c14ba2b0817cd7ae46021f381f64eb76eccf'),
    },
}


def compatible(previous, current):
    return wiki_batch.compatible_pipeline(previous, current, required_names=current.keys()) or (
        previous.keys() == current.keys() and all(
        previous[name] == sha or (previous[name], sha) in VALIDATION_REVISIONS.get(name, ())
        for name, sha in current.items()))


def locator(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path))


def prepare(source, output, *, parent=None, reread_contract=None, structured_contract=None,
            reference_contract=None, correction_contract=None, request_locations=None, job_ids=None,
            completion_contract=None, client_recovery_contract=None, _finalize=None):
    """Keep the frozen inventory and reuse checked work without new selection."""
    if client_recovery_contract is not None:
        from . import wiki_client_recovery
        if (client_recovery_contract != wiki_client_recovery.CONTRACT or _finalize is not None
                or any((parent, reread_contract, structured_contract, reference_contract,
                        correction_contract, completion_contract))):
            raise ValueError('client recovery cannot change other contracts or allowances')
        return wiki_client_recovery.prepare(source, output, job_ids, request_locations=request_locations)
    if (((Path(source) / 'manifest.json').exists()
            and read_json(Path(source) / 'manifest.json').get('client_recovery'))
            or (Path(source) / '.client_recovery_v1-permit.json').exists()):
        raise ValueError('client recovery must resume in its original output; no new continuation')
    if completion_contract is not None:
        from . import wiki_completion
        if (completion_contract != wiki_completion.CONTRACT or _finalize is not None
                or any((parent, reread_contract, structured_contract, reference_contract,
                        correction_contract, request_locations))):
            raise ValueError('page completion cannot change other contracts or allowances')
        return wiki_completion.prepare(source, output, job_ids)
    if correction_contract == wiki_repair.CONSTRAINT_POLICY:
        if any((parent, reread_contract, structured_contract, reference_contract)):
            raise ValueError('constraint correction cannot change other batch contracts')
        from . import wiki_constraints
        return wiki_constraints.prepare(source, output, request_locations, job_ids)
    if request_locations is not None or job_ids:
        if (correction_contract != wiki_repair.CORRECTION_POLICY or not request_locations
                or not job_ids or len(set(job_ids)) != len(job_ids)):
            raise ValueError('request locations require explicit correction and unique selected jobs')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('continuation must be separate from its source batch')
    with file_lock(source / '.lock', blocking=False), file_lock(output / '.lock', blocking=False):
        if (output / 'manifest.json').exists():
            raise ValueError('continuation already prepared; use run to resume')
        original = read_json(source / 'manifest.json')
        if original['schema'] not in ('wiki_batch_v1', 'wiki_supplement_v1'):
            raise ValueError('unsupported source batch')
        supplement = original['schema'] == 'wiki_supplement_v1'
        current = wiki_supplement.pipeline() if supplement else wiki_batch.pipeline()
        grant = source / '.correction-source_feedback_v2.json'
        if correction_contract is not None:
            base = dict(current, **CORRECTION_BASE)
            if (correction_contract != wiki_repair.CORRECTION_POLICY or supplement
                    or any((reread_contract, structured_contract, reference_contract))
                    or any(original.get(key) != value for key, value in (
                        ('reread_contract', 'reread_evidence_v2'),
                        ('structured_contract', 'structured_source_v2'),
                        ('reference_contract', 'bounded_refs_v2')))
                    or original.get('correction') or any(job.get('correction') for job in original['jobs'])
                    or grant.exists()
                    or not (original['pipeline'] == current
                            or compatible(original['pipeline'], base))):
                raise ValueError('correction upgrade requires an unused registered v2 person batch')
        if reference_contract is not None:
            if (reference_contract != 'bounded_refs_v2' or supplement
                    or original.get('reference_contract', 'legacy_v1') != 'legacy_v1'):
                raise ValueError('reference upgrade requires a legacy person batch and bounded_refs_v2')
            if not wiki_batch.compatible_pipeline(original['pipeline'], current):
                raise ValueError('reference upgrade requires a registered complete legacy pipeline')
        if structured_contract is not None:
            if (structured_contract != 'structured_source_v2' or supplement
                    or original.get('structured_contract', 'legacy_v1') != 'legacy_v1'):
                raise ValueError('structured upgrade requires a legacy person batch and structured_source_v2')
            if not wiki_batch.compatible_pipeline(original['pipeline'], current):
                raise ValueError('structured upgrade requires a registered complete legacy pipeline')
        if reread_contract is not None:
            if (reread_contract != 'reread_evidence_v2' or supplement
                    or original.get('reread_contract', 'legacy_v1') != 'legacy_v1'):
                raise ValueError('reread upgrade requires a legacy person batch and reread_evidence_v2')
            if not wiki_batch.compatible_pipeline(original['pipeline'], current):
                raise ValueError('reread upgrade requires a registered complete legacy pipeline')
        elif correction_contract is None and not compatible(original['pipeline'], current):
            raise ValueError('only registered validation-only pipeline changes may reuse completed pages')
        for key in ('messages', 'parent', 'index') if supplement else ('messages',):
            if locator(original[key]['path']) != original[key]:
                raise ValueError(f'{key} changed since source batch preparation')
        manifest = deepcopy(original)
        if reread_contract is not None:
            manifest['reread_contract'] = reread_contract
        if structured_contract is not None:
            manifest['structured_contract'] = structured_contract
        if reference_contract is not None:
            manifest['reference_contract'] = reference_contract
        if supplement:
            if parent is None:
                raise ValueError('supplement continuation requires its continued parent')
            parent_path = Path(parent).resolve() / 'manifest.json'
            parent_manifest = read_json(parent_path)
            if (parent_manifest.get('continuation', {}).get('source') != original['parent']
                    or any(parent_manifest[k] != original[k]
                           for k in ('messages', 'self_account', 'config', 'max_chars'))):
                raise ValueError('continued parent does not descend from the frozen parent')
            manifest['parent'] = locator(parent_path)
        elif parent is not None:
            raise ValueError('person batch has no parent')
        previous = read_json(source / 'progress.json', default=dict(jobs={}))
        located_requests = {}
        if request_locations is not None:
            from . import wiki_cached
            original_jobs = {job['id']: job for job in original['jobs']}
            for job_id in job_ids:
                if (job_id not in original_jobs
                        or previous['jobs'].get(job_id, {}).get('stage') != 'incomplete'):
                    raise ValueError('correction selection requires incomplete source jobs')
                receipt = read_json(Path(request_locations) / f'{job_id}.json')
                key, _, evidence, path = wiki_cached.rejected_request(
                    original_jobs[job_id], original, receipt, retry_policy=wiki_repair.RETRY_POLICY)
                located_requests[job_id] = {key: dict(source=locator(path),
                    file=evidence['failure_file'], evidence=evidence)}
        state = dict(stage='prepared', total=len(original['jobs']), jobs={}, active_job=None)
        counts = dict(reused=0, source_review=0, resumed=0, pending=0, cached_requests=0,
                      failed_requests=0)
        selected_jobs, selected_requests = 0, 0
        priorities = {}
        for job in manifest['jobs']:
            old_job = deepcopy(job)
            saved = previous['jobs'].get(job['id'], {})
            for item in job['sources']:
                if sha256_file(Path(item['path'])) != item['sha256']:
                    raise ValueError('legacy Wiki changed since source batch preparation')
            job['output'] = str(output / 'jobs' / job['id'])
            job.pop('reuse', None)
            priorities[job['id']] = 2
            if saved.get('stage') == 'complete':
                checked = wiki_delivery.inspect_job(source, old_job, saved)
                if not checked['content_issues']:
                    job['reuse'] = dict(path=saved['knowledge'], sha256=saved['sha256'])
                    wiki_prompt.export(saved['knowledge'], output / 'prompt' / job['id'])
                    state['jobs'][job['id']] = deepcopy(saved)
                    counts['reused'] += 1
                    continue
                counts['source_review'] += 1
                priorities[job['id']] = 0
            elif saved:
                counts['resumed'] += 1
                priorities[job['id']] = 1
            else:
                counts['pending'] += 1
            directory = (Path(saved['knowledge']).parent.parent if saved.get('stage') == 'complete'
                         else Path(old_job['output']))
            job['continued_from'] = str(directory)
            if correction_contract is not None:
                requests = (located_requests.get(job['id'], {}) if request_locations is not None
                            else correction_requests(old_job, saved))
                if requests:
                    job['correction'] = dict(contract=correction_contract,
                        limit=wiki_repair.RETRY_LIMIT, requests=requests)
                    selected_jobs += 1
                    selected_requests += len(requests)
                    priorities[job['id']] = -1
                if saved:
                    state['jobs'][job['id']] = deepcopy(saved)
                progress = directory / 'progress.json'
                if progress.exists():
                    target = Path(job['output']) / 'progress.json'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(progress, target)
            for kind, counter in (('cache', 'cached_requests'), ('failures', 'failed_requests')):
                for cache in sorted((directory / kind).glob('*.json')):
                    target = Path(job['output']) / kind / cache.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cache, target)
                    counts[counter] += 1
        manifest['jobs'].sort(key=lambda j: priorities[j['id']])
        manifest['pipeline'] = current
        if correction_contract is not None:
            if not selected_requests:
                raise ValueError('correction upgrade requires exhausted rejected-output evidence')
            manifest['correction'] = dict(contract=correction_contract,
                limit=wiki_repair.RETRY_LIMIT, selected_jobs=selected_jobs,
                selected_requests=selected_requests, previous_pipeline=original['pipeline'])
        manifest['continuation'] = dict(source=locator(source / 'manifest.json'),
            previous_pipeline=original['pipeline'], counts=counts,
            policy=('explicit_source_feedback_v2; finite_rejected_request_allowance; '
                    'completed_artifacts_immutable; caches_and_failures_preserved' if correction_contract else
                    'explicit_bounded_refs_v2; completed_artifacts_immutable; '
                    'exact_request_caches_and_failures_preserved' if reference_contract else
                    'explicit_structured_source_v2; completed_artifacts_immutable; '
                    'exact_request_caches_and_failures_preserved' if structured_contract else
                    'explicit_reread_evidence_v2; completed_artifacts_immutable; '
                    'exact_request_caches_and_failures_preserved' if reread_contract else
                    'registered_compatible_legacy; completed_artifacts_immutable; '
                    'exact_request_caches_and_failures_preserved'))
        manifest['summary']['reusable'] = counts['reused']
        state['counts'] = dict(Counter(job['stage'] for job in state['jobs'].values()))
        if counts['reused'] == state['total']:
            state['stage'] = 'complete'
        if correction_contract is not None:
            write_json(grant, dict(contract=correction_contract, output=str(output),
                                   source=locator(source / 'manifest.json')))
        if _finalize is not None:
            _finalize(manifest)
        write_json(output / 'progress.json', state)
        write_json(output / 'manifest.json', manifest)
        result = dict(total=state['total'], **counts)
        if correction_contract is not None:
            result.update(selected_jobs=selected_jobs, selected_requests=selected_requests)
        return result
