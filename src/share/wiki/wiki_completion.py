"""Resume only the downstream stages of one successfully corrected page."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from ...config import sha256_file
from ...iteration.storage import file_lock, read_json, write_json
from . import wiki_constraints, wiki_continuation, wiki_repair, wiki_structured, wiki_reconcile
from .wiki_library import digest

CONTRACT = 'page_completion_v1'
COLUMNS = ['source_line', 'time', 'sender_id', 'display_name', 'kind', 'text', 'chat_id']
BLOCKED = 'cache miss outside selected constraint requests; no model request made'


def provenance():
    from . import wiki_batch
    return dict(constraints=wiki_constraints.provenance(),
                completion=sha256_file(Path(__file__)),
                batch=sha256_file(Path(wiki_batch.__file__)),
                continuation=sha256_file(Path(wiki_continuation.__file__)))


def sealed_read(locator):
    if sha256_file(Path(locator['path'])) != locator['sha256']:
        raise ValueError('page completion evidence changed')
    return read_json(locator['path'])


def plain_manifest(manifest):
    value = deepcopy(manifest)
    value.pop('completion', None)
    for job in value['jobs']:
        job.pop('completion', None)
    return value


def evidence(source, job_id):
    manifest = read_json(source / 'manifest.json')
    jobs = {job['id']: job for job in manifest['jobs']}
    previous = read_json(source / 'progress.json')
    job = jobs.get(job_id)
    if (manifest.get('completion') or job is None or job.get('kind') != 'person'
            or previous['jobs'].get(job_id, {}).get('stage') != 'incomplete'
            or manifest.get('correction', {}).get('contract') != wiki_constraints.CONTRACT
            or [item['id'] for item in manifest['jobs'] if item.get('correction')] != [job_id]
            or any(manifest.get(name) != value for name, value in (
                ('reread_contract', 'reread_evidence_v2'),
                ('structured_contract', 'structured_source_v2'),
                ('reference_contract', 'bounded_refs_v2')))):
        raise ValueError('page completion requires one incomplete corrected person page')
    directory = Path(job['output'])
    if directory.resolve() != source / 'jobs' / job_id:
        raise ValueError('page completion source output changed')
    policy = job['correction']
    grant = wiki_constraints._grant(policy, directory / 'cache')
    ledger_path = Path(grant['ledger'])
    ledger = read_json(ledger_path)
    expected = {key: next(iter(value['requests'])) for key, value in grant['jobs'].items()}
    if (len(policy['requests']) != 1 or ledger.get('requests') != expected
            or ledger.get('grant_sha256') != policy['grant']['sha256']
            or set(ledger.get('used', {})) != set(expected)
            or ledger['used'].get(job_id) != wiki_constraints.LIMIT
            or ledger.get('seal') != digest({key: value for key, value in ledger.items() if key != 'seal'})):
        raise ValueError('page completion cannot reset or bypass correction usage')
    key = next(iter(policy['requests']))
    success_path = directory / 'cache' / f'{key}.json'
    success = read_json(success_path)
    if (success.get('request_sha256') != key
            or success.get('retry_policy') != wiki_constraints.CONTRACT
            or success.get('policy_attempt') != wiki_constraints.LIMIT
            or success.get('retry_budget') != 'all_invocations_v1'
            or success.get('grant_sha256') != policy['grant']['sha256']
            or success.get('wire_plan') != policy['requests'][key]['plan']
            or not isinstance(success.get('result'), dict)):
        raise ValueError('page completion requires a sealed successful correction cache')
    progress = read_json(directory / 'progress.json')
    errors = progress.get('errors', [])
    batches = progress.get('batches', 0)
    mentions = progress.get('mention_batches', -1)
    if (progress.get('stage') != 'incomplete' or batches < 1
            or any(progress.get(name) != batches for name in ('completed', 'reread_completed'))
            or progress.get('sections_completed') != len(wiki_repair.CATEGORIES)
            or mentions < 0 or any(progress.get(name) != mentions
                for name in ('mention_completed', 'mention_reread_completed'))
            or len(errors) != 1 or errors[0].get('stage') != 'consolidating'
            or errors[0].get('error') != BLOCKED
            or errors[0].get('request_sha256') in policy['requests']
            or len(errors[0].get('request_sha256', '')) != 64):
        raise ValueError('page completion requires finished upstream stages and a scope-only block')
    repair = read_json(directory / 'manifest.json')
    subject = dict(name=job['title'], account=job['account'], self_account=job['self_account'])
    if (repair.get('subject') != subject or repair.get('config') != manifest['config']
            or repair.get('correction') != policy
            or any(repair.get(name, {}).get(field) != locator[field]
                   for name, locator in (('wiki', job['wiki']), ('messages', manifest['messages']))
                   for field in ('path', 'sha256'))
            or repair.get('generator_sha256') != manifest['pipeline']['wiki_repair.py']):
        raise ValueError('page completion source recipe changed')
    inherited = {str(path.relative_to(directory)): sha256_file(path)
                 for kind in ('cache', 'failures') for path in sorted((directory / kind).glob('*.json'))}
    return dict(job_id=job_id, subject=subject, source=wiki_continuation.locator(source / 'manifest.json'),
        progress=wiki_continuation.locator(source / 'progress.json'),
        repair=wiki_continuation.locator(directory / 'manifest.json'),
        upstream=wiki_continuation.locator(directory / 'progress.json'),
        ledger=wiki_continuation.locator(ledger_path), success=wiki_continuation.locator(success_path),
        correction=deepcopy(policy), consolidation=errors[0]['request_sha256'], inherited=inherited)


def prepare(source, output, job_ids):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not job_ids or len(job_ids) != 1:
        raise ValueError('page completion requires exactly one explicitly selected page')
    with file_lock(source / '.lock', blocking=False), file_lock(output / '.lock', blocking=False):
        proof = evidence(source, job_ids[0])
        previous = sealed_read(proof['progress'])
        def finalize(manifest):
            historical = manifest.pop('correction')
            manifest.setdefault('correction_history', []).append(historical)
            for job in manifest['jobs']:
                historical = job.pop('correction', None)
                if historical:
                    job.setdefault('correction_history', []).append(historical)
            manifest['continuation']['policy'] = ('sealed_same_page_downstream_completion; '
                'success_cache_required; correction_ledger_unchanged; original_validators_unchanged')
            permit = dict(contract=CONTRACT, output=str(output), provenance=provenance(),
                          manifest_sha256=digest(manifest), **proof)
            permit_path = output / f'.completion-{CONTRACT}.json'
            write_json(permit_path, permit)
            marker = dict(contract=CONTRACT, job_id=proof['job_id'],
                          permit=wiki_continuation.locator(permit_path))
            manifest['completion'] = marker
            for job in manifest['jobs']:
                if job['id'] == proof['job_id']:
                    job['completion'] = deepcopy(marker)
        result = wiki_continuation.prepare(source, output, _finalize=finalize)
        state = read_json(output / 'progress.json')
        for job_id, saved in previous['jobs'].items():
            if job_id in state['jobs']:
                continue
            restored = deepcopy(saved)
            if saved.get('stage') == 'complete':
                restored.update(stage='incomplete',
                    error='completed content requires source review; outside selected page completion')
            state['jobs'][job_id] = restored
        state['counts'] = dict(Counter(job['stage'] for job in state['jobs'].values()))
        state.update(stage='prepared', pid=None, active_job=None)
        write_json(output / 'progress.json', state)
        target = output / 'jobs' / proof['job_id'] / 'progress.json'
        shutil.copyfile(proof['upstream']['path'], target)
        return dict(result, selected_jobs=1, completion_contract=CONTRACT)


def validate_batch(manifest, output):
    marker = manifest['completion']
    permit = sealed_read(marker['permit'])
    if (marker.get('contract') != CONTRACT or permit.get('contract') != CONTRACT
            or marker.get('job_id') != permit['job_id']
            or Path(permit['output']) != Path(output).resolve()
            or permit['provenance'] != provenance()
            or digest(plain_manifest(manifest)) != permit['manifest_sha256']
            or [job['id'] for job in manifest['jobs'] if job.get('completion')] != [permit['job_id']]
            or any(job['completion'] != marker for job in manifest['jobs'] if job.get('completion'))):
        raise ValueError('page completion scope or recipe changed')
    source = sealed_read(permit['source'])
    for name in ('progress', 'repair', 'upstream', 'ledger', 'success'):
        sealed_read(permit[name])
    old_job = next(job for job in source['jobs'] if job['id'] == permit['job_id'])
    wiki_constraints._grant(permit['correction'], Path(old_job['output']) / 'cache')
    directory = Path(output) / 'jobs' / permit['job_id']
    for relative, checksum in permit['inherited'].items():
        if sha256_file(directory / relative) != checksum:
            raise ValueError('page completion cannot remove or change inherited caches or failure budgets')
    return permit


class CompletionClient:
    def __init__(self, client, permit):
        self.client, self.permit = client, permit

    def cache_identity(self):
        return self.client.cache_identity()

    def run(self, prompt, schema=None):
        from .wiki_cached import original_prompt
        original = original_prompt(prompt)
        key = digest([self.cache_identity(), original, schema])
        def blocked():
            raise wiki_repair.request_error(key, 'request outside selected page downstream completion')
        if key in self.permit['correction']['requests']:
            blocked()
        tasks = (
            (wiki_repair.CONSOLIDATE_PROMPT, wiki_repair.CONSOLIDATE_SCHEMA, 'consolidating'),
            (wiki_structured.source_prompt('structured_source_v2'), wiki_structured.SCHEMA, 'structuring'),
            (wiki_reconcile.PROMPT, wiki_reconcile.SCHEMA, 'reconciling'),
        )
        for template, base, stage in tasks:
            prefix = wiki_repair.REFERENCE_PROMPT + template
            if not original.startswith(prefix):
                continue
            try:
                payload = json.loads(original[len(prefix):])
                subject = deepcopy(self.permit['subject'])
                if stage == 'reconciling':
                    subject['entity_id'] = 'entity-' + digest([subject['self_account'], subject['account']])[:20]
                if payload['subject'] != subject:
                    blocked()
                if stage == 'consolidating':
                    if (key != self.permit['consolidation'] or set(payload) != {
                            'subject', 'categories', 'sections', 'facts'}
                            or payload['categories'] != wiki_repair.CATEGORIES):
                        blocked()
                    bounds = dict(fact_ids=[fact['id'] for fact in payload['facts']])
                else:
                    fields = ({'facts', 'source_contract'} if stage == 'structuring' else {'entities', 'records'})
                    if (set(payload) != {'subject', 'columns', 'messages'} | fields
                            or payload['columns'] != COLUMNS
                            or any(len(row) != len(COLUMNS) for row in payload['messages'])):
                        blocked()
                    rows = payload['messages']
                    lines = [row[0] for row in rows]
                    if stage == 'structuring':
                        sources = [dict(line=row[0], sender_id=row[2], message_kind=row[4]) for row in rows]
                        if payload['source_contract'] != wiki_structured.source_contract(subject, sources):
                            blocked()
                        ids = [fact['id'] for fact in payload['facts']]
                        bounds = dict(lines=lines, utterance_line=lines, fact_id=ids, fact_ids=ids)
                    else:
                        bounds = dict(lines=lines, record_id=[record['id'] for record in payload['records']])
                if schema != wiki_repair.bounded_reference_schema(base, bounds):
                    blocked()
            except (ValueError, TypeError, KeyError):
                blocked()
            return self.client.run(prompt, schema)
        blocked()
