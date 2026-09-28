"""Versioned batch continuation after a registered validation-only change.

Completed artifacts stay at their original paths. Exact request caches are
copied for missing/invalid work and revalidated by the normal generation entry.
"""
from copy import deepcopy
from pathlib import Path
import shutil

from ..config import sha256_file
from ..iteration.storage import file_lock, read_json, write_json
from . import wiki_batch, wiki_delivery, wiki_prompt, wiki_supplement


# Only validation and its mechanical error feedback changed; prompts, schemas,
# selection and compilation stayed byte-identical. Other revisions fail closed.
VALIDATION_REVISIONS = {
    'wiki_repair.py': {
        ('fec0d48fbcfb4e9428101a9390681ceff7575295e0210f9d0025722f025df2d8',
         'dd1a8fdad053d40ae186245d6093b47f2244ceab3a775cf0b67831892fb69351'),
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
    },
    'wiki_reconcile.py': {
        ('2bae089c1bbd110667c83aff5fc17726cdbfd7450958271e066b2a267813a036',
         '46c2c5d2e5bb4f584baeeca835e4c14ba2b0817cd7ae46021f381f64eb76eccf'),
    },
}


def compatible(previous, current):
    return previous.keys() == current.keys() and all(
        previous[name] == sha or (previous[name], sha) in VALIDATION_REVISIONS.get(name, ())
        for name, sha in current.items())


def locator(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path))


def prepare(source, output, *, parent=None):
    """Keep the frozen inventory and reuse checked work without new selection."""
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
        if not compatible(original['pipeline'], current):
            raise ValueError('only registered validation-only pipeline changes may reuse completed pages')
        for key in ('messages', 'parent', 'index') if supplement else ('messages',):
            if locator(original[key]['path']) != original[key]:
                raise ValueError(f'{key} changed since source batch preparation')
        manifest = deepcopy(original)
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
        state = dict(stage='prepared', total=len(original['jobs']), jobs={}, active_job=None)
        counts = dict(reused=0, source_review=0, resumed=0, pending=0, cached_requests=0)
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
            for cache in sorted((directory / 'cache').glob('*.json')):
                target = Path(job['output']) / 'cache' / cache.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache, target)
                counts['cached_requests'] += 1
        manifest['jobs'].sort(key=lambda j: priorities[j['id']])
        manifest['pipeline'] = current
        manifest['continuation'] = dict(source=locator(source / 'manifest.json'),
            previous_pipeline=original['pipeline'], counts=counts,
            policy='registered_validation_only; completed_artifacts_immutable; cached_outputs_revalidated')
        manifest['summary']['reusable'] = counts['reused']
        state['counts'] = dict(complete=counts['reused'])
        if counts['reused'] == state['total']:
            state['stage'] = 'complete'
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'progress.json', state)
        return dict(total=state['total'], **counts)
