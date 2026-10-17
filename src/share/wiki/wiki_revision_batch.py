"""Versioned semantic revisions following immutable completed batch artifacts."""
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import time

from ...config import sha256_file
from ...judge import codex_transport
from ...iteration.storage import file_lock, read_json, write_json
from .. import knowledge_view
from . import wiki_batch, wiki_connection_retry, wiki_delivery, wiki_job_pool, wiki_prompt, wiki_revision, wiki_revision_output, wiki_structured


def locator(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(Path(path)))


LEGACY_RUNNER_SHA256 = 'd6db8d19f19e2f57091a39ca58302c28e0d42a6920bf7d73bfba51a4ac556641'
SOURCE_CHECK_RUNNER_SHA256 = '03b4d8731658131b7af3e1e33894494bc4c2af09c0c81f907529acc68ecc476a'
CLAIMS_DISPATCH_RUNNER_SHA256 = '736fde993739e55c34eeec297381842a2a9f9c7788619983d89f4f88f47bcb95'
SERIAL_RUNNER_SHA256 = '0f465169abc187cd0eeec4e351b9946dd73952e72bbdd1b9eb89d63b3ec3e68f'
OBJECT_POOL_RUNNER_SHA256 = 'cc35955c83d2f07810e3bd97d6a2095a3d263474cb97f3c10830048f048cfce9'
UNSEALED_VIEW_RUNNER_SHA256 = 'dfa96ff82a3cdb66e279204d9ff1b65a1a32cd63395bbca96b6590ab537aea36'
PRE_SELECTION_RUNNER_SHA256 = 'f48f8ad32491dbe422fd3e5619d6f20df8a6392cae12a00661b66b2700c60c40'
PRE_FEEDBACK_RUNNER_SHA256 = '8b085de3e13d715542139ba8ab77895c22d4333ac8ee88264a8cb9af05b86b42'
PRE_DIAGNOSTIC_RUNNER_SHA256 = 'd9d2be4619c3f356a13a8c7a2f34a9b8de0d8c63fbfb07bbc377941cc803da0e'
PRE_STRUCTURED_RUNNER_SHA256 = '14d041408aefd8ea5e53b568495a96800b6628b9c10d6ddc25cae6c5be70e495'
PRE_SERVICE_RUNNER_SHA256 = '88ef382af16752572b954afaf4419a5d08c65ff80ff165b31cb86ce81b139747'
PRE_CONNECTION_RUNNER_SHA256 = '53c03f095220a29bd1148b73ab1ed55369e9103320d453085e98489eca852775'
PRE_OUTPUT_RUNNER_SHA256 = 'fbacd6d41ae62160828bfe45cf7ec69b82751f931aea4a4323f559748823790c'
# Each entry lists (old, new) transitions oldest-first; the walker maps the
# current file digest back through every registered generation.
OUTPUT_IMPLEMENTATION_UPDATES = {
    'wiki_revision.py': [('9f4992d9829ab0047830912183633f23964b7dd64804d8a968607398b715aa06',
                          'd78f4087f4818f2d151e23aa73d7b807ee160d010aa6036169764deda2012672')],
    'wiki_revision_roles.py': [('040aa3920b8841f02de4d32ddb37bb240e4914ed1b211f1fdb1cc7fd79030e79',
                                '8bb2bdf7ed6cf8d98ca6a6e74d61c77690b419887a0c0255e19dcde257ffcca5')],
    'wiki_revision_claims.py': [('edc21bed46700e81917351f3bf320a0d6462c96ee3c3c2b9fa2057dbcf0e1eb9',
                                 'ea56ad6eda8f9ab95f17a56b6676f29b68cfb72e39bb7774e773a56db97a73c1'),
                                ('ea56ad6eda8f9ab95f17a56b6676f29b68cfb72e39bb7774e773a56db97a73c1',
                                 '261277f985e2a95d50b5acf0b972e7f2d73436b04b87600b179383787be8118d'),
                                ('261277f985e2a95d50b5acf0b972e7f2d73436b04b87600b179383787be8118d',
                                 '73b2021d4c94bd53d3c75e41b7ea278cc75bf26ad73fef000ed87d02dd0369b1')],
}
DIAGNOSTIC_CLAIMS_SHA256 = (
    '7df5a6cb1b121cf2e44fc71e4b44e65460ff0039c679833272762df228b96138',
    'edc21bed46700e81917351f3bf320a0d6462c96ee3c3c2b9fa2057dbcf0e1eb9',
)

PRE_MIGRATION_SEMANTIC_PIPELINE = {
    'wiki_repair.py': 'dd1a8fdad053d40ae186245d6093b47f2244ceab3a775cf0b67831892fb69351',
    'wiki_mentions.py': 'f55ef49dea4f750bacd8bd5791cf4e07440bb28fd5c5e00f1b7c2a437ae331b3',
    'wiki_reconcile.py': '46c2c5d2e5bb4f584baeeca835e4c14ba2b0817cd7ae46021f381f64eb76eccf',
    'wiki_structured.py': '3e6f9d331fc15e2b5e19eb721e2a10b80083a0269bca2ef25d665d25e7fee7ef',
    'wiki_revision.py': '0867ea792f816341947cd7726799a711dbff77f220652b09a54a4d61b37ef466',
    'wiki_prompt.py': '5f06f1b0782e5dbb9f5820ddcbed11053a11c72a8795c736fa70d7b641e3bb3c',
}
MIGRATED_SEMANTIC_PIPELINE = {
    'wiki_repair.py': '113b3c3445698773eb571286f6dcba9a1be03456cc0af4e141c7a0fcc5478de8',
    'wiki_mentions.py': '012dd09f76df3166ba5f754ff1027fa2c276fc163b587a4a713312f2a3b9ec3b',
    'wiki_reconcile.py': '46c2c5d2e5bb4f584baeeca835e4c14ba2b0817cd7ae46021f381f64eb76eccf',
    'wiki_structured.py': '4809954a35b934dc678be21d2a20f570a61cb4b9d0aa628584ed7c667c7c9620',
    'wiki_revision.py': '1984dc367ed08984f8939b55d666637065bd4cbd9097f8737eb52a7ef43a96bf',
    'wiki_prompt.py': '7be0cb6b366819bc3e5062780abed3c0b6b83b1e208cec535e75516b5c7cc088',
    'knowledge_view.py': 'b365c63f58137c9efd7537d39c30148367af8b7994400e8d818ab6a388741e44',
}
MIGRATED_ENGINES = {
    'legacy': (LEGACY_RUNNER_SHA256, {}, {}),
    'roles_v1': (SOURCE_CHECK_RUNNER_SHA256,
        {'wiki_revision_roles.py': '6a896619d5abc21623d71a014a0b5827ee13a61a00d0c49a7a63f38f5ecfd216'},
        {'wiki_revision_roles.py': '040aa3920b8841f02de4d32ddb37bb240e4914ed1b211f1fdb1cc7fd79030e79'}),
    'roles_claims_v2': (SERIAL_RUNNER_SHA256,
        {'wiki_revision_claims.py': '058b60a4115b56cbb679a2ffbb249ff68a149fcfc3168dd196822aa0d36d209d'},
        {'wiki_revision_claims.py': '03fa67062f8f9c4f8072802c22a7f634ccf8c6ed4d8a0752f4ebd48a3bbb11d2'}),
}

REVISION_SUPPORT_PIPELINE = {
    'wiki_repair.py': '06e9f7b9b39de02d8e302b8bc05a974d6c5b3a0e8763f9e9674540c5e440c540',
    'wiki_mentions.py': '1c95b8b04955395bdf24b6ae7242ca9514ff649a2550d94aaa61e55036673a06',
    'wiki_reconcile.py': '8cf651176335ad04bd07d0780436aadb86175b73309e6dfb7228547cac2ca86e',
    'wiki_structured.py': '12a395f3658420e16601de84f027a44ee601f508f0d0baf188bd6778068c0bb7',
    'wiki_prompt.py': '932f859684a6aa9297f8643b3f44d90ca0e4b5145e2ca786229fc4a9f75ea51d',
}


def _previous_revision_support_pipeline(current, engine):
    """Recognize one complete revision-only support snapshot, never mixed updates.

    Revision engines retain their requests, schemas and correction allowances.
    The support updates add opt-in generation contracts, aggregate validation
    diagnostics and export serialization; revision keeps the legacy contracts.
    """
    previous = MIGRATED_SEMANTIC_PIPELINE | MIGRATED_ENGINES[engine][2]
    for updates in (wiki_batch.FEEDBACK_PIPELINE_UPDATES, wiki_batch.REREAD_IMPLEMENTATION_UPDATES,
                    wiki_batch.STRUCTURED_IMPLEMENTATION_UPDATES,
                    wiki_batch.REFERENCE_IMPLEMENTATION_UPDATES,
                    wiki_batch.TRANSPORT_IMPLEMENTATION_UPDATES):
        previous |= {name: pair[1] for name, pair in updates.items() if name in previous}
    if engine == 'roles_claims_v2':
        previous['wiki_revision_claims.py'] = DIAGNOSTIC_CLAIMS_SHA256[1]
    expected = previous | REVISION_SUPPORT_PIPELINE
    if {name: sha for name, sha in current.items() if name != Path(__file__).name} != expected:
        return None
    return current | previous


def engine_for(name):
    if name == 'legacy':
        return wiki_revision
    if name == 'roles_v1':
        from . import wiki_revision_roles
        return wiki_revision_roles
    if name == 'roles_claims_v2':
        from . import wiki_revision_claims
        return wiki_revision_claims
    raise ValueError('unknown semantic revision engine')


def pipeline(engine='legacy'):
    backend = engine_for(engine)
    return wiki_batch.pipeline() | {
        Path(m.__file__).name: sha256_file(Path(m.__file__))
        for m in (wiki_revision, wiki_revision_output, wiki_prompt, knowledge_view)
    } | {Path(m.__file__).name: sha256_file(Path(m.__file__)) for m in (backend,)} | {
        Path(__file__).name: sha256_file(Path(__file__))}


def compatible_pipeline(frozen, engine):
    current = pipeline(engine)
    if frozen == current:
        return True
    current = previous_output_pipeline(current)
    if current is None:
        return False
    snapshots = [current]
    support = _previous_revision_support_pipeline(current, engine)
    if support is not None:
        current = support
        snapshots.append(current)
    transport = wiki_batch.previous_transport_pipeline(current, required_names=current.keys())
    if transport is not None:
        if frozen == (transport | {Path(__file__).name: PRE_SERVICE_RUNNER_SHA256}):
            return True
        snapshots.append(transport)
    structured = wiki_batch.previous_structured_pipeline(current, required_names=current.keys())
    if structured is not None:
        if frozen == (structured | {Path(__file__).name: PRE_STRUCTURED_RUNNER_SHA256}):
            return True
        snapshots.append(structured)
    legacy = wiki_batch.legacy_reread_pipeline(current, required_names=current.keys())
    if legacy is not None:
        snapshots.append(legacy)
    diagnostic = _previous_diagnostic_pipeline(legacy or current, engine)
    if diagnostic is not None:
        snapshots.append(diagnostic)
    for snapshot in snapshots:
        if _compatible_snapshot(frozen, snapshot, engine):
            return True
        previous = wiki_batch.previous_feedback_pipeline(snapshot, required_names=snapshot.keys())
        if previous is not None and _compatible_snapshot(frozen, previous, engine):
            return True
    return False


def previous_output_pipeline(current):
    previous = dict(current)
    name = Path(wiki_revision_output.__file__).name
    if previous.pop(name, None) != sha256_file(Path(wiki_revision_output.__file__)):
        return None
    for name, transitions in OUTPUT_IMPLEMENTATION_UPDATES.items():
        if name in previous:
            matched, changed = False, True
            while changed:
                changed = False
                for old, new in transitions:
                    if previous[name] == new:
                        previous[name] = old
                        matched, changed = True, True
            if not matched:
                return None
    return previous


def _previous_diagnostic_pipeline(current, engine):
    if engine != 'roles_claims_v2':
        return None
    expected = MIGRATED_SEMANTIC_PIPELINE | MIGRATED_ENGINES[engine][2]
    expected |= {name: pair[1] for name, pair in wiki_batch.FEEDBACK_PIPELINE_UPDATES.items()
                 if name in expected}
    expected['wiki_revision_claims.py'] = DIAGNOSTIC_CLAIMS_SHA256[1]
    runner = Path(__file__).name
    if set(current) != set(expected) | {runner} or {
            name: digest for name, digest in current.items() if name != runner} != expected:
        return None
    return current | {'wiki_revision_claims.py': DIAGNOSTIC_CLAIMS_SHA256[0]}


def _compatible_snapshot(frozen, current, engine):
    if frozen == current:
        return True
    if frozen == (current | {Path(__file__).name: PRE_OUTPUT_RUNNER_SHA256}):
        return True
    if frozen == (current | {Path(__file__).name: PRE_CONNECTION_RUNNER_SHA256}):
        return True
    if _compatible_migration(frozen, current, engine):
        return True
    if frozen == (current | {Path(__file__).name: PRE_DIAGNOSTIC_RUNNER_SHA256}):
        return True
    if frozen == (current | {Path(__file__).name: PRE_FEEDBACK_RUNNER_SHA256}):
        return True
    if frozen == (current | {Path(__file__).name: PRE_SELECTION_RUNNER_SHA256}):
        return True
    if frozen == (current | {Path(__file__).name: OBJECT_POOL_RUNNER_SHA256}):
        return True
    # Object scheduling changes concurrency only. Require exact identity of
    # every semantic engine, prompt, model input and export dependency.
    if frozen == (current | {Path(__file__).name: SERIAL_RUNNER_SHA256}):
        return True
    # Adding the claims engine leaves both existing engines byte-identical.
    # The new semantic engine has its own module binding and cannot use this alias.
    if engine in ('legacy', 'roles_v1') and frozen == (
            current | {Path(__file__).name: CLAIMS_DISPATCH_RUNNER_SHA256}):
        return True
    # Source validation changes admission only. The engine, model requests,
    # cached decisions and exports are byte-identical for either engine.
    if engine in ('legacy', 'roles_v1') and frozen == (
            current | {Path(__file__).name: SOURCE_CHECK_RUNNER_SHA256}):
        return True
    # The legacy engine, inputs, schema, cache and export are unchanged. Only
    # the runner gained an explicit alternative engine and chained sources.
    return engine == 'legacy' and frozen == (current | {Path(__file__).name: LEGACY_RUNNER_SHA256})


def _compatible_migration(frozen, current, engine):
    runner, old_engine, new_engine = MIGRATED_ENGINES[engine]
    expected = MIGRATED_SEMANTIC_PIPELINE | new_engine
    if {name: sha for name, sha in current.items() if name != Path(__file__).name} != expected:
        return False
    historical = PRE_MIGRATION_SEMANTIC_PIPELINE | old_engine | {Path(__file__).name: runner}
    unsealed = {name: sha for name, sha in expected.items() if name != 'knowledge_view.py'}
    return frozen in (historical, unsealed | {Path(__file__).name: UNSEALED_VIEW_RUNNER_SHA256})


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
        from ...iteration.storage import atomic_write
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


def run(output, *, workers=4, object_workers=1, attempts=2, max_jobs=None, follow=False,
        skip_incomplete=False, job_ids=None, wait=False, client=None, connection_retry_seconds=None,
        output_contract=None):
    """Review every newly completed source job, resume exact successful requests."""
    if workers < 1 or object_workers < 1 or attempts < 1 or (max_jobs is not None and max_jobs < 1):
        raise ValueError('workers, object_workers, attempts and optional max_jobs must be positive')
    selected = None if job_ids is None else frozenset(job_ids)
    if selected is not None and not selected:
        raise ValueError('job_ids must not be empty')
    if output_contract is not None and (output_contract != wiki_revision_output.CONTRACT
            or selected is None or attempts != 1 or skip_incomplete):
        raise ValueError('revision output contract requires known contract, job_ids, attempts=1 and no skip_incomplete')
    if connection_retry_seconds is not None and (
            type(connection_retry_seconds) is not int or connection_retry_seconds < 1
            or attempts != 1 or selected is None or skip_incomplete):
        raise ValueError('connection retry requires a positive interval, job_ids, attempts=1 and no skip_incomplete')
    output = Path(output).resolve()
    with file_lock(output / '.lock', blocking=wait):
        manifest = read_json(output / 'manifest.json')
        if selected is not None:
            unknown = selected - {job['id'] for job in manifest['jobs']}
            if unknown:
                raise ValueError(f'unknown revision jobs: {sorted(unknown)}')
        state = read_json(output / 'progress.json')
        def save():
            state['counts'] = dict(Counter(v['stage'] for v in state['jobs'].values()))
            state['updated_at'] = time.time()
            write_json(output / 'progress.json', state)
        def fail(exc):
            # Preserve successful jobs and caches so restored inputs can resume
            # with zero requests, but never report invalid inputs as complete.
            state.update(stage='incomplete', active_job=None, active_jobs=[],
                         error=f'{type(exc).__name__}: {exc}'[-1600:])
            save()
        try:
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
                from ...judge.corrected import CodexJudgeClient
                client = CodexJudgeClient(manifest['config'])
            output_binding = (wiki_revision_output.activate(output, manifest, selected, state,
                              pipeline(engine), client) if output_contract is not None else None)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            fail(exc)
            raise
        connections = (wiki_connection_retry.ConnectionRetries(output, connection_retry_seconds)
                       if connection_retry_seconds is not None else None)
        state.update(stage='running', pid=os.getpid(), active_job=None, active_jobs=[],
                     object_workers=object_workers, workers=workers, skip_incomplete=skip_incomplete,
                     connection_retry_seconds=connection_retry_seconds,
                     connection_retry_run_id=connections.state['run_id'] if connections else None,
                     output_contract=output_binding,
                     selected_jobs=None if selected is None else sorted(selected))
        state.pop('error', None)
        state.pop('blocker', None)
        save()
        started, visited, cache = 0, set(), {}
        def pending(previous):
            for job in manifest['jobs']:
                prior = state['jobs'].get(job['id'], {})
                saved = previous['jobs'].get(job['id'], {})
                if job['id'] in visited:
                    continue
                if prior.get('stage') == 'complete':
                    _job_input(manifest, job, prior.get('revision_source'), required=True)
                    wiki_delivery._require_revision_original(
                        manifest, job['id'], prior.get('revision_source'))
                    wiki_delivery._checked(source, source_jobs[job['id']], saved, cache)
                    wiki_delivery._checked(output, job, prior, cache)
                    visited.add(job['id'])
                    continue
                if selected is not None and job['id'] not in selected:
                    continue
                if skip_incomplete and prior.get('stage') == 'incomplete':
                    continue
                if saved.get('stage') != 'complete':
                    continue
                visited.add(job['id'])
                yield job

        def execute(job, attempt):
            saved = previous['jobs'][job['id']]
            binding = dict(path=saved['knowledge'], sha256=saved['sha256'])
            # Each worker has its own verification memo and per-job cache;
            # the raw rows are loaded once and shared read-only.
            checked = {}
            wiki_delivery._require_revision_original(manifest, job['id'], binding)
            wiki_delivery._checked(source, source_jobs[job['id']], saved, checked)
            data = read_json(saved['knowledge'])
            if any(data['coverage'][k] != manifest['messages'][k] for k in ('path', 'sha256')):
                raise ValueError('knowledge and revision raw sources differ')
            job_manifest, frozen = _job_input(manifest, job, binding)
            write_json(job_manifest, frozen)
            job_client = connections.wrap(client, job) if connections else client
            if output_binding is not None:
                job_client = wiki_revision_output.ContractClient(job_client, output_binding)
            for retry in range(attempts):
                attempt()
                try:
                    revised = backend.revise(job_client, Path(job['output']), data, rows, workers)
                    issues = wiki_delivery.content_issues(revised)
                    if issues:
                        raise ValueError(f'revised content needs source review: {issues}')
                    break
                except Exception:
                    codex_transport.check_available()
                    if retry == attempts - 1:
                        raise
            if (wiki_batch._stamp(manifest['messages']['path']) != source_stamp
                    or locator(saved['knowledge']) != binding):
                raise ValueError('revision source changed while running')
            wiki_delivery._require_revision_original(manifest, job['id'], binding)
            result = dict(export(output, job, revised), revision_source=binding,
                          revision_counts=revised['semantic_revision']['counts'])
            if output_binding is not None:
                result['output_contract'] = output_binding
            wiki_delivery._checked(output, job, result, checked)
            return result

        while True:
            previous = read_json(source / 'progress.json', default=dict(jobs={}))
            try:
                started += wiki_job_pool.run_jobs(pending(previous), execute,
                    object_workers=object_workers, state=state, save=save,
                    max_jobs=None if max_jobs is None else max_jobs - started,
                    check_available=codex_transport.check_available)
            except codex_transport.CodexServiceBlocked as exc:
                state.update(stage='blocked', pid=None, blocker=exc.details)
                save()
                return state
            except (OSError, ValueError, KeyError, TypeError) as exc:
                fail(exc)
                raise
            if (not follow or (max_jobs is not None and started >= max_jobs)
                    or not wiki_batch.status(source)['running']):
                break
            time.sleep(30)
        state['stage'] = 'complete' if state['counts'].get('complete', 0) == state['total'] else 'incomplete'
        state['pid'] = None
        save()
        return state
