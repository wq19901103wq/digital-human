"""Versioned semantic revisions following immutable completed batch artifacts."""
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import time

from ..config import sha256_file
from ..iteration.storage import file_lock, read_json, write_json
from . import wiki_batch, wiki_delivery, wiki_prompt, wiki_revision, wiki_structured


def locator(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(Path(path)))


LEGACY_RUNNER_SHA256 = 'd6db8d19f19e2f57091a39ca58302c28e0d42a6920bf7d73bfba51a4ac556641'
SOURCE_CHECK_RUNNER_SHA256 = '03b4d8731658131b7af3e1e33894494bc4c2af09c0c81f907529acc68ecc476a'


def engine_for(name):
    if name == 'legacy':
        return wiki_revision
    if name == 'roles_v1':
        from . import wiki_revision_roles
        return wiki_revision_roles
    raise ValueError('unknown semantic revision engine')


def pipeline(engine='legacy'):
    backend = engine_for(engine)
    return wiki_batch.pipeline() | {
        Path(m.__file__).name: sha256_file(Path(m.__file__)) for m in (wiki_revision, wiki_prompt)
    } | {Path(m.__file__).name: sha256_file(Path(m.__file__)) for m in (backend,)} | {
        Path(__file__).name: sha256_file(Path(__file__))}


def compatible_pipeline(frozen, engine):
    current = pipeline(engine)
    if frozen == current:
        return True
    # Source validation changes admission only. The engine, model requests,
    # cached decisions and exports are byte-identical for either engine.
    if frozen == (current | {Path(__file__).name: SOURCE_CHECK_RUNNER_SHA256}):
        return True
    # The legacy engine, inputs, schema, cache and export are unchanged. Only
    # the runner gained an explicit alternative engine and chained sources.
    return engine == 'legacy' and frozen == (current | {Path(__file__).name: LEGACY_RUNNER_SHA256})


def prepare(source, output, *, engine='legacy'):
    """Freeze the inventory, not the live progress, and never mutate the source."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('revision output must be separate from its source')
    with file_lock(output / '.lock', blocking=False):
        if (output / 'manifest.json').exists():
            raise ValueError('revision already prepared; use revise to resume')
        original = read_json(source / 'manifest.json')
        if original['schema'] not in ('wiki_batch_v1', 'wiki_supplement_v1'):
            raise ValueError('revision requires a generation or revision batch')
        manifest = deepcopy(original)
        manifest.pop('parent', None)
        manifest['schema'] = 'wiki_batch_v1'
        manifest['revision'] = dict(source=locator(source / 'manifest.json'), engine=engine,
            policy='source_records_immutable; keep_exact_records; raw_chat_only; no_curated_feedback')
        manifest['pipeline'] = pipeline(engine)
        for job in manifest['jobs']:
            job.pop('reuse', None)
            job['output'] = str(output / 'jobs' / job['id'])
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'progress.json', dict(stage='prepared', jobs={}, total=len(manifest['jobs'])))
        return dict(total=len(manifest['jobs']), source=str(source))


def export(output, job, data):
    content = Path(job['output']) / 'content'
    wiki_structured.export(content, data)
    if data['subject'].get('kind') in ('group', 'topic'):
        from ..iteration.storage import atomic_write
        rendered = wiki_structured.render(data).replace('\n账号：\n', '\n类型：' + data['subject']['kind'] + '\n', 1)
        rendered = rendered.replace('## 人物属性\n', '## 群与话题属性\n', 1)
        atomic_write(content / 'wiki.md', rendered)
    knowledge = content / 'knowledge.json'
    exported = wiki_prompt.export(knowledge, output / 'prompt' / job['id'])
    return dict(stage='complete', knowledge=str(knowledge), sha256=sha256_file(knowledge),
                records=exported['records'], packets=exported['packets'])


def _job_input(manifest, job, binding, *, required=False):
    path = Path(job['output']) / 'manifest.json'
    frozen = dict(source=binding, pipeline=manifest['pipeline'], config=manifest['config'])
    if (required or path.exists()) and read_json(path) != frozen:
        raise ValueError('revision job input changed')
    return path, frozen


def run(output, *, workers=4, attempts=2, max_jobs=None, follow=False, client=None):
    """Review every newly completed source job, resume exact successful requests."""
    if workers < 1 or attempts < 1 or (max_jobs is not None and max_jobs < 1):
        raise ValueError('workers, attempts and optional max_jobs must be positive')
    output = Path(output).resolve()
    with file_lock(output / '.lock', blocking=False):
        state = read_json(output / 'progress.json')
        def save():
            state['counts'] = dict(Counter(v['stage'] for v in state['jobs'].values()))
            state['updated_at'] = time.time()
            write_json(output / 'progress.json', state)
        def fail(exc):
            # Preserve successful jobs and caches so restored inputs can resume
            # with zero requests, but never report invalid inputs as complete.
            state.update(stage='incomplete', active_job=None,
                         error=f'{type(exc).__name__}: {exc}'[-1600:])
            save()
        try:
            manifest = read_json(output / 'manifest.json')
            engine = manifest['revision'].get('engine', 'legacy')
            backend = engine_for(engine)
            if not compatible_pipeline(manifest['pipeline'], engine):
                raise ValueError('semantic revision pipeline changed; prepare a new revision')
            binding = manifest['revision']['source']
            source = Path(binding['path']).parent
            if locator(binding['path']) != binding:
                raise ValueError('revision source manifest changed')
            original = read_json(binding['path'])
            source_jobs = {j['id']: j for j in original['jobs']}
            source_stamp = wiki_batch._stamp(manifest['messages']['path'])
            rows = wiki_revision.read_rows(manifest['messages'], manifest['self_account'])
            if client is None:
                from ..judge.corrected import CodexJudgeClient
                client = CodexJudgeClient(manifest['config'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            fail(exc)
            raise
        state.update(stage='running', pid=os.getpid(), active_job=None)
        state.pop('error', None)
        save()
        started, visited, cache = 0, set(), {}
        while True:
            previous = read_json(source / 'progress.json', default=dict(jobs={}))
            for job in manifest['jobs']:
                prior = state['jobs'].get(job['id'], {})
                saved = previous['jobs'].get(job['id'], {})
                if job['id'] in visited:
                    continue
                if prior.get('stage') == 'complete':
                    try:
                        _job_input(manifest, job, prior.get('revision_source'), required=True)
                        wiki_delivery._require_revision_original(
                            manifest, job['id'], prior.get('revision_source'))
                        wiki_delivery._checked(source, source_jobs[job['id']], saved, cache)
                        wiki_delivery._checked(output, job, prior, cache)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        fail(exc)
                        raise
                    visited.add(job['id'])
                    continue
                if saved.get('stage') != 'complete':
                    continue
                if max_jobs is not None and started >= max_jobs:
                    break
                started += 1
                visited.add(job['id'])
                item = state['jobs'][job['id']] = dict(stage='running', account=job['account'],
                    attempts=prior.get('attempts', 0))
                state['active_job'] = job['id']
                save()
                try:
                    binding = dict(path=saved['knowledge'], sha256=saved['sha256'])
                    wiki_delivery._require_revision_original(manifest, job['id'], binding)
                    wiki_delivery._checked(source, source_jobs[job['id']], saved, cache)
                    data = read_json(saved['knowledge'])
                    if any(data['coverage'][k] != manifest['messages'][k] for k in ('path', 'sha256')):
                        raise ValueError('knowledge and revision raw sources differ')
                    job_manifest, frozen = _job_input(manifest, job, binding)
                    write_json(job_manifest, frozen)
                    for retry in range(attempts):
                        item['attempts'] += 1
                        save()
                        try:
                            revised = backend.revise(client, Path(job['output']), data, rows, workers)
                            issues = wiki_delivery.content_issues(revised)
                            if issues:
                                raise ValueError(f'revised content needs source review: {issues}')
                            break
                        except Exception:
                            if retry == attempts - 1:
                                raise
                    if (wiki_batch._stamp(manifest['messages']['path']) != source_stamp
                            or locator(saved['knowledge']) != binding):
                        raise ValueError('revision source changed while running')
                    wiki_delivery._require_revision_original(manifest, job['id'], binding)
                    item.update(export(output, job, revised), revision_source=binding,
                                revision_counts=revised['semantic_revision']['counts'])
                    wiki_delivery._checked(output, job, item, cache)
                except Exception as exc:
                    item.update(stage='incomplete', error=f'{type(exc).__name__}: {exc}'[-1600:])
                state['active_job'] = None
                save()
            if (not follow or (max_jobs is not None and started >= max_jobs)
                    or not wiki_batch.status(source)['running']):
                break
            time.sleep(30)
        state['stage'] = 'complete' if state['counts'].get('complete', 0) == state['total'] else 'incomplete'
        save()
        return state
