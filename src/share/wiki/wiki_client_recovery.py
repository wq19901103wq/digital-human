"""One auditable client recovery per explicitly selected unfinished page."""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import shutil
import time

from ...config import sha256_file
from ...iteration.storage import file_lock, read_json, write_json
from . import wiki_batch, wiki_continuation, wiki_resumption
from .wiki_library import digest

CONTRACT = 'client_recovery_v1'
LIMIT = 5
PERMIT = f'.{CONTRACT}-permit.json'
GRANT = f'.{CONTRACT}-grant.json'
LEDGER = f'.{CONTRACT}-ledger.json'


def plain_manifest(manifest):
    value = deepcopy(manifest)
    value.pop('client_recovery', None)
    return value


def _sealed(binding):
    if wiki_continuation.locator(binding['path']) != binding:
        raise ValueError('client recovery sealed evidence changed')
    return read_json(binding['path'])


def _ancestry(source, locks=None):
    ancestors, seen = [], set()
    path = Path(source).resolve() / 'manifest.json'
    while True:
        if path in seen:
            raise ValueError('cyclic client recovery ancestry')
        seen.add(path)
        if locks is not None:
            locks.enter_context(file_lock(path.parent / '.lock', blocking=False))
        binding = wiki_continuation.locator(path)
        manifest = read_json(path)
        ancestors.append(binding)
        parent = manifest.get('continuation', {}).get('source')
        if not parent:
            return ancestors
        path = Path(parent['path']).resolve()
        if path.name != 'manifest.json' or wiki_continuation.locator(path) != parent:
            raise ValueError('client recovery ancestry locator changed')


def _located_errors(job, manifest, errors, receipt):
    from . import wiki_cached
    located = wiki_cached.request_location(job, manifest, receipt, sealed_result=True)
    records, requests = located.get('errors'), located.get('requests')
    if not isinstance(records, list) or not isinstance(requests, dict) or not records:
        raise ValueError('client recovery request locations missing')
    def without_key(record):
        return {key: value for key, value in record.items() if key != 'request_sha256'}
    if (any(not isinstance(record, dict) for record in [*errors, *records])
            or Counter(digest(without_key(record)) for record in errors)
               != Counter(digest(without_key(record)) for record in records)
            or any(not isinstance(record.get('request_sha256'), str)
                   or record['request_sha256'] not in requests for record in records)
            or set(requests) != {record['request_sha256'] for record in records}):
        raise ValueError('client recovery locations differ from all saved errors')
    by_location = {digest(without_key(record)): record for record in records}
    if len(by_location) != len(records):
        raise ValueError('client recovery locations are duplicated or ambiguous')
    result = [by_location[digest(without_key(record))] for record in errors]
    if any(record.get('request_sha256', located_record['request_sha256'])
           != located_record['request_sha256'] for record, located_record in zip(errors, result)):
        raise ValueError('client recovery explicit request key changed')
    return result


def _evidence(source, manifest, previous, job_ids, locations=None):
    if locations:
        from . import wiki_selection
        index = wiki_selection.for_batch(manifest, source, record_runtime=False)
        with wiki_selection.accelerated(index):
            return _evidence_records(source, manifest, previous, job_ids, locations)
    return _evidence_records(source, manifest, previous, job_ids, locations)


def _evidence_records(source, manifest, previous, job_ids, locations):
    jobs = {job['id']: job for job in manifest['jobs']}
    proofs = {}
    for job_id in job_ids:
        job, prior = jobs.get(job_id), previous['jobs'].get(job_id, {})
        if job is None or prior.get('stage') != 'incomplete':
            raise ValueError('client recovery requires explicitly selected incomplete jobs')
        directory = source / 'jobs' / job_id
        if Path(job['output']).resolve() != directory:
            raise ValueError('client recovery output differs from frozen job')
        detail = wiki_batch._incomplete_failure(job, prior, inspect_records=True)
        progress = read_json(directory / 'progress.json', default={})
        errors = progress.get('errors', [])
        binding = (locations or {}).get(job_id)
        if binding:
            errors = _located_errors(job, manifest, errors, _sealed(binding))
        elif any(not isinstance(item.get('request_sha256'), str)
                 or not item['request_sha256'] for item in errors):
            raise ValueError('client recovery missing request key requires offline location evidence')
        records = wiki_batch._failure_records(directory, allow_legacy=True)
        evidence = [wiki_batch._failure_evidence(directory, item, records)
                    for item in errors]
        allowed = {'recorded_transport_failure', 'legacy_client_timeout', 'legacy_client_failure'}
        if (detail['reason'] != 'non_transport_failure'
                or not evidence or any(item.get('kind') not in allowed
                    or not item.get('history_complete') or item.get('result_present')
                    or item.get('cache_file_present')
                    or item.get('attempt') != item.get('latest_recorded_attempt')
                    for item in evidence)):
            raise ValueError('client recovery requires complete failed-client evidence without success cache')
        proofs[job_id] = dict(progress=wiki_continuation.locator(directory / 'progress.json'),
                              failure_evidence=evidence)
        if binding:
            proofs[job_id]['request_location'] = binding
    return proofs


def prepare(source, output, job_ids, *, request_locations=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (not job_ids or len(job_ids) > LIMIT or len(set(job_ids)) != len(job_ids)
            or not all(isinstance(job_id, str) and job_id for job_id in job_ids)):
        raise ValueError('client recovery requires one to five unique explicit job IDs')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('client recovery must be separate from its source batch')
    with ExitStack() as locks:
        locks.enter_context(file_lock(output / '.lock', blocking=False))
        ancestors = _ancestry(source, locks)
        root = Path(ancestors[-1]['path']).parent
        if any((Path(binding['path']).parent / GRANT).exists() for binding in ancestors):
            raise ValueError('client recovery already granted in this ancestry; resume its output')
        if (output / 'manifest.json').exists() or any((output / name).exists()
                for name in ('jobs', 'prompt', 'progress.json', PERMIT, LEDGER)):
            raise ValueError('client recovery output already contains artifacts; use a fresh output')
        original = read_json(source / 'manifest.json')
        if original['schema'] != 'wiki_batch_v1' or original.get('completion'):
            raise ValueError('client recovery requires a person batch without page completion')
        if original.get('client_recovery') or (source / PERMIT).exists():
            raise ValueError('client recovery cannot grant another recovery')
        previous = read_json(source / 'progress.json')
        locations = ({job_id: wiki_continuation.locator(Path(request_locations) / f'{job_id}.json')
                      for job_id in job_ids} if request_locations is not None else {})
        proof = _evidence(source, original, previous, job_ids, locations)

        def finalize(manifest):
            manifest['client_recovery'] = dict(contract=CONTRACT, unpublished=True)
            manifest['continuation']['policy'] = (
                'finite_client_recovery; one_repair_per_selected_page; '
                'existing_correction_budgets_and_success_caches_preserved')

        result = wiki_continuation.prepare(source, output, _finalize=finalize)
        if result['source_review']:
            raise ValueError('client recovery cannot reuse pages requiring source review')
        manifest = read_json(output / 'manifest.json')
        wiki_resumption._same_inventory(original, manifest)
        state = read_json(output / 'progress.json')
        inherited = {}
        for job in manifest['jobs']:
            job_id = job['id']
            if job.get('reuse'):
                continue
            prior = previous['jobs'].get(job_id)
            if prior:
                state['jobs'][job_id] = deepcopy(prior)
            directory = source / 'jobs' / job_id
            target = Path(job['output'])
            progress = directory / 'progress.json'
            if progress.exists():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(progress, target / 'progress.json')
            files = {}
            for kind in ('cache', 'failures'):
                for path in sorted((directory / kind).glob('*.json')):
                    relative = f'{kind}/{path.name}'
                    checksum = sha256_file(path)
                    if sha256_file(target / relative) != checksum:
                        raise ValueError('client recovery did not preserve inherited cache or failure budget')
                    files[relative] = checksum
            inherited[job_id] = files
        state.update(stage='prepared', pid=None, active_job=None)
        state['counts'] = dict(Counter(job['stage'] for job in state['jobs'].values()))
        write_json(output / 'progress.json', state)
        permit = dict(contract=CONTRACT, output=str(output), root=str(root),
            job_ids=sorted(job_ids), limit=1, ancestors=ancestors,
            source=ancestors[0], progress=wiki_continuation.locator(source / 'progress.json'),
            proofs=proof, inherited=inherited, manifest_sha256=digest(plain_manifest(manifest)))
        write_json(output / PERMIT, permit)
        manifest['client_recovery'] = dict(contract=CONTRACT,
                                           permit=wiki_continuation.locator(output / PERMIT))
        write_json(output / 'manifest.json', manifest)
        grant = dict(contract=CONTRACT, source=ancestors[0], output=str(output),
                     permit=manifest['client_recovery']['permit'], ledger=str(root / LEDGER))
        ledger = dict(contract=CONTRACT, grant_sha256=digest(grant),
                      job_ids=permit['job_ids'], claims={})
        ledger['seal'] = digest(ledger)
        write_json(root / LEDGER, ledger)
        write_json(root / GRANT, grant)
        return dict(result, selected_jobs=len(job_ids), client_recovery_contract=CONTRACT)


def _ledger(permit, grant):
    path = Path(permit['root']) / LEDGER
    ledger = read_json(path)
    claims = ledger.get('claims')
    if (ledger.get('contract') != CONTRACT or ledger.get('grant_sha256') != digest(grant)
            or ledger.get('job_ids') != permit['job_ids'] or not isinstance(claims, dict)
            or not set(claims) <= set(permit['job_ids']) or any(
                not isinstance(claim, dict) or claim.get('output') != permit['output']
                or type(claim.get('claimed_at')) not in (int, float) for claim in claims.values())
            or ledger.get('seal') != digest({key: value for key, value in ledger.items() if key != 'seal'})):
        raise ValueError('client recovery ledger or claim budget changed')
    return ledger


def validate_batch(manifest, output):
    output = Path(output).resolve()
    marker = manifest.get('client_recovery', {})
    if marker.get('contract') != CONTRACT or marker.get('permit', {}).get('path') != str(output / PERMIT):
        raise ValueError('client recovery permit missing or unpublished')
    permit = _sealed(marker['permit'])
    if (permit.get('contract') != CONTRACT or permit.get('output') != str(output)
            or permit.get('limit') != 1 or not permit.get('job_ids')
            or len(permit['job_ids']) > LIMIT or len(set(permit['job_ids'])) != len(permit['job_ids'])
            or digest(plain_manifest(manifest)) != permit.get('manifest_sha256')):
        raise ValueError('client recovery scope or frozen recipe changed')
    ancestors = _ancestry(Path(permit['source']['path']).parent)
    if (ancestors != permit['ancestors'] or permit['source'] != ancestors[0]
            or permit['root'] != str(Path(ancestors[-1]['path']).parent)):
        raise ValueError('client recovery ancestry changed')
    original, previous = _sealed(permit['source']), _sealed(permit['progress'])
    wiki_resumption._same_inventory(original, manifest)
    locations = {job_id: proof['request_location'] for job_id, proof in permit['proofs'].items()
                 if 'request_location' in proof}
    if _evidence(Path(permit['source']['path']).parent, original, previous,
                 permit['job_ids'], locations) != permit['proofs']:
        raise ValueError('client recovery failure evidence changed')
    for proof in permit['proofs'].values():
        _sealed(proof['progress'])
    for job_id, files in permit['inherited'].items():
        for relative, checksum in files.items():
            for directory in (Path(permit['source']['path']).parent, output):
                path = directory / 'jobs' / job_id / relative
                if sha256_file(path) != checksum:
                    raise ValueError('client recovery inherited cache or failure budget changed')
    root = Path(permit['root'])
    grant = read_json(root / GRANT)
    expected = dict(contract=CONTRACT, source=permit['source'], output=str(output),
                    permit=marker['permit'], ledger=str(root / LEDGER))
    if grant != expected:
        raise ValueError('client recovery grant missing or bound to another output')
    with file_lock(root / '.client-recovery.lock'):
        ledger = _ledger(permit, grant)
    return permit, frozenset(ledger['claims'])


def claim(permit, job_id):
    root = Path(permit['root'])
    with file_lock(root / '.client-recovery.lock'):
        grant = read_json(root / GRANT)
        ledger = _ledger(permit, grant)
        if job_id not in permit['job_ids'] or job_id in ledger['claims']:
            raise ValueError('client recovery opportunity unavailable or already claimed')
        ledger['claims'][job_id] = dict(output=permit['output'], claimed_at=time.time())
        ledger['seal'] = digest({key: value for key, value in ledger.items() if key != 'seal'})
        write_json(root / LEDGER, ledger)
