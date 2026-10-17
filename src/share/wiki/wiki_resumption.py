"""Publish ordinary continuations with their original failure state intact."""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from ...config import sha256_file
from ...iteration.storage import file_lock, read_json, write_json
from . import wiki_completion, wiki_continuation, wiki_delivery


INPUTS = ('schema', 'messages', 'self_account', 'config', 'max_chars', 'gaps',
          'reread_contract', 'structured_contract', 'reference_contract')
JOB_INPUTS = ('id', 'title', 'kind', 'account', 'self_account', 'sources', 'wiki',
              'binding', 'binding_evidence', 'raw_sender_rows')


def _same_inventory(original, manifest):
    if any(original.get(name) != manifest.get(name) for name in INPUTS):
        raise ValueError('continuation history changed frozen inputs or contracts')
    expected = {job['id']: {name: job.get(name) for name in JOB_INPUTS}
                for job in original['jobs']}
    actual = {job['id']: {name: job.get(name) for name in JOB_INPUTS}
              for job in manifest['jobs']}
    if expected != actual or len(actual) != len(manifest['jobs']):
        raise ValueError('continuation history changed frozen job inventory')


def _history(source, original, locks):
    saved, progress, directories, receipts = {}, {}, {}, {}
    seen = set()
    batch, manifest = source, original
    while True:
        if batch in seen:
            raise ValueError('cyclic continuation history')
        seen.add(batch)
        _same_inventory(original, manifest)
        state_path = batch / 'progress.json'
        state = read_json(state_path, default=dict(jobs={}))
        state_receipt = wiki_continuation.locator(state_path) if state_path.exists() else None
        for job in manifest['jobs']:
            job_id = job['id']
            prior = state['jobs'].get(job_id)
            if prior and job_id not in saved:
                if batch != source and prior.get('stage') == 'complete':
                    raise ValueError('historical completed page missing from current batch state')
                saved[job_id] = deepcopy(prior)
                receipts.setdefault(job_id, {})['state'] = state_receipt
            if saved.get(job_id, {}).get('stage') == 'complete':
                continue
            directory = Path(job['output']).resolve()
            if directory != batch / 'jobs' / job_id:
                raise ValueError('unfinished history output differs from its frozen job')
            directories.setdefault(job_id, []).append(directory)
            page_progress = directory / 'progress.json'
            if job_id not in progress and page_progress.exists():
                progress[job_id] = page_progress
                receipts.setdefault(job_id, {})['progress'] = wiki_continuation.locator(page_progress)
        outstanding = [job['id'] for job in original['jobs']
                       if job['id'] not in saved or (saved[job['id']].get('stage') != 'complete'
                                                   and job['id'] not in progress)]
        binding = manifest.get('continuation', {}).get('source')
        if not outstanding or not binding:
            return saved, progress, directories, receipts
        path = Path(binding['path']).resolve()
        if path.name != 'manifest.json':
            raise ValueError('invalid continuation history locator')
        batch = path.parent
        locks.enter_context(file_lock(batch / '.lock', blocking=False))
        if wiki_continuation.locator(path) != binding:
            raise ValueError('continuation history manifest changed')
        manifest = read_json(path)


def _copy_history(directories, target):
    for directory in directories:
        for kind in ('cache', 'failures'):
            for source in sorted((directory / kind).glob('*.json')):
                destination = target / kind / source.name
                if destination.exists():
                    if kind == 'failures' and sha256_file(destination) != sha256_file(source):
                        raise ValueError('continuation history contains conflicting failure budgets')
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)


def prepare(source, output, **options):
    if any(value for value in options.values()):
        return wiki_continuation.prepare(source, output, **options)
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('continuation must be separate from its source batch')
    with ExitStack() as locks:
        locks.enter_context(file_lock(source / '.lock', blocking=False))
        locks.enter_context(file_lock(output / '.lock', blocking=False))
        if (output / 'manifest.json').exists():
            raise ValueError('continuation already prepared; use run to resume')
        if any((output / name).exists() for name in ('jobs', 'prompt', 'progress.json')):
            raise ValueError('continuation output contains unfinished artifacts; use a fresh output')
        original = read_json(source / 'manifest.json')
        if original.get('client_recovery') or (source / '.client_recovery_v1-permit.json').exists():
            raise ValueError('client recovery must resume in its original output; no new continuation')
        if original['schema'] != 'wiki_batch_v1':
            return wiki_continuation.prepare(source, output, **options)
        if original.get('correction') or any(job.get('correction') for job in original['jobs']):
            raise ValueError('ordinary continuation cannot inherit active correction allowances')
        if not original.get('completion') and any(job.get('completion') for job in original['jobs']):
            raise ValueError('page completion marker has no batch permit')
        if original.get('completion'):
            permit = wiki_completion.validate_batch(original, source)
            job = next(job for job in original['jobs'] if job['id'] == permit['job_id'])
            completed = read_json(source / 'progress.json')['jobs'].get(job['id'], {})
            if (completed.get('stage') != 'complete'
                    or wiki_delivery.inspect_job(source, job, completed)['content_issues']):
                raise ValueError('selected page must pass delivery before retiring its completion permit')
        saved, progress, directories, receipts = _history(source, original, locks)
        current = read_json(source / 'progress.json', default=dict(jobs={}))['jobs']
        restored = sum(job_id not in current for job_id in saved)
        with TemporaryDirectory(prefix='.continuation-', dir=output) as temporary:
            staging = Path(temporary)
            def finalize(manifest):
                historical = manifest.pop('completion', None)
                if historical:
                    manifest.setdefault('completion_history', []).append(historical)
                for job in manifest['jobs']:
                    historical = job.pop('completion', None)
                    if historical:
                        job.setdefault('completion_history', []).append(historical)
                    job['output'] = str(output / 'jobs' / job['id'])
                manifest['continuation']['state_sources'] = receipts
            result = wiki_continuation.prepare(source, staging, _finalize=finalize)
            manifest = read_json(staging / 'manifest.json')
            state = read_json(staging / 'progress.json')
            for job in manifest['jobs']:
                job_id = job['id']
                if job.get('reuse'):
                    continue
                target = staging / 'jobs' / job_id
                _copy_history(directories.get(job_id, []), target)
                if job_id in progress:
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(progress[job_id], target / 'progress.json')
                prior = saved.get(job_id)
                if prior and prior.get('stage') != 'complete':
                    state['jobs'][job_id] = deepcopy(prior)
                    if prior.get('stage') == 'running':
                        state['jobs'][job_id].update(stage='incomplete',
                            error=prior.get('error') or 'previous run interrupted; recovery evidence required')
                elif prior:
                    state['jobs'][job_id] = dict(stage='incomplete',
                        error='completed page requires source review')
            result['pending'] -= restored
            result['resumed'] += restored
            for kind, counter in (('cache', 'cached_requests'), ('failures', 'failed_requests')):
                result[counter] = sum(len(list((staging / 'jobs' / job['id'] / kind).glob('*.json')))
                                      for job in manifest['jobs'])
            manifest['continuation']['counts'] = {key: value for key, value in result.items() if key != 'total'}
            state['counts'] = dict(Counter(value['stage'] for value in state['jobs'].values()))
            manifest['jobs'].sort(key=lambda job: (
                job['id'] not in state['jobs'], state['jobs'].get(job['id'], {}).get('stage') == 'complete'))
            for name in ('jobs', 'prompt'):
                if (staging / name).exists():
                    shutil.copytree(staging / name, output / name)
            write_json(output / 'progress.json', state)
            write_json(output / 'manifest.json', manifest)
            return dict(result, restored_jobs=restored)
