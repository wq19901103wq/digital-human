"""Offline old/new retrieval timing using the authenticated experiment executor."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType
from unittest.mock import patch


def measure_ranking(*, sizes=(512, 8192, 65536), repeats=5):
    from itertools import islice
    from random import Random
    from statistics import median
    from src.config import sha256_file
    from src.generator.fewshot import rank

    repeats = positive(repeats)
    if not sizes or any(not isinstance(size, int) or not 1 <= size <= 1000000 for size in sizes):
        raise ValueError('ranking profile sizes must be within [1, 1000000]')
    results = []
    for size in sizes:
        for order in ('mixed', 'ordered', 'tied'):
            random = Random(0)
            scored = [(-random.random() if order != 'tied' else 0.0,
                       f'id-{position % 17}' if order != 'tied' else 'same-id',
                       {'position': position}) for position in range(size)]

            def sort_key(item):
                return (0, 1, item[0], item[1])

            if order == 'ordered':
                scored.sort(key=sort_key)
            for consumption, consumed in (('prefix', min(12, size)), ('all', size)):
                expected = sorted(scored, key=sort_key)[:consumed]
                warmup = list(islice(rank._ranked_candidates(list(scored), sort_key), consumed))
                if warmup != expected:
                    raise ValueError('ranking profile changed stable selection order')
                timings = {'stable_sort': [], 'current_ranking': []}
                for repeat in range(repeats):
                    routes = ('stable_sort', 'current_ranking') if repeat % 2 == 0 else \
                        ('current_ranking', 'stable_sort')
                    for route in routes:
                        working = list(scored)
                        started = time.perf_counter()
                        if route == 'stable_sort':
                            working.sort(key=sort_key)
                            selected = working[:consumed]
                        else:
                            selected = list(islice(rank._ranked_candidates(working, sort_key), consumed))
                        timings[route].append(time.perf_counter() - started)
                        if selected != expected:
                            raise ValueError('ranking profile changed stable selection order')
                medians = {name: median(values) for name, values in timings.items()}
                results.append(dict(candidates=size, input_order=order, consumption=consumption,
                                    consumed=consumed, seconds=timings, median_seconds=medians,
                                    kernel_speedup=medians['stable_sort'] / medians['current_ranking']
                                    if medians['current_ranking'] else None))
    return dict(schema=1, scope='synthetic_ranking_kernel_only', equivalent=True,
                model_calls=0, experiment_results_written=False, repeats=repeats,
                rank_sha256=sha256_file(Path(rank.__file__)), results=results)


def measure_eligibility(*, sizes=(512, 8192, 100000), repeats=5):
    import sqlite3
    from statistics import median
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    from src.config import sha256_file
    from src.generator import disk_history
    from src.generator.fewshot import recall_engines

    repeats = positive(repeats)
    if not sizes or any(not isinstance(size, int) or not 8 <= size <= 1000000 for size in sizes):
        raise ValueError('eligibility profile sizes must be within [8, 1000000]')
    results = []
    pool = SimpleNamespace(_minimum_identity_length=24)
    with TemporaryDirectory(prefix='dh-eligibility-profile-') as temporary:
        source = Path(temporary) / 'source.jsonl'
        source.touch()
        for size in sizes:
            connection = sqlite3.connect(':memory:')
            query_count = 0
            def count_query(statement):
                nonlocal query_count
                if statement.lstrip().upper().startswith('SELECT '):
                    query_count += 1
            connection.set_trace_callback(count_query)
            connection.executescript('''
                CREATE TABLE examples(position INTEGER PRIMARY KEY, id TEXT, chat TEXT,
                                      end_timestamp REAL, end_index INTEGER, order_verified INTEGER);
                CREATE TABLE example_messages(message_id TEXT, position INTEGER,
                                              PRIMARY KEY(message_id, position));
                CREATE INDEX example_ids ON examples(id, position);
                CREATE INDEX example_times ON examples(end_timestamp);
                CREATE INDEX example_chats ON examples(chat);
            ''')
            connection.executemany('INSERT INTO examples VALUES (?,?,?,?,?,?)',
                ((position, f'{position:024x}', f'chat-{position % 4}', position, position, 1)
                 for position in range(size)))
            connection.execute('INSERT INTO examples VALUES (?,?,?,?,?,?)',
                               (size, f'{size - 2:024x}', 'chat-0', 0, 0, 1))
            connection.execute('INSERT INTO example_messages VALUES (?,?)', ('target', 0))
            reader = disk_history.BoundedReader(source, connection)
            try:
                for fraction in (0.1, 0.5, 0.9):
                    boundary = int(size * fraction)
                    case = dict(input_cutoff=dict(timestamp=boundary, index=boundary, order_verified=True),
                                source_span=dict(chat_id=f'chat-{boundary % 4}'),
                                context_message_ids=['context'], reply_message_ids=['target'],
                                history_excluded_chat_ids=['chat-3'])
                    query = SimpleNamespace(history_case=case, exclude_ids={'explicit'})
                    expected = reader.excluded_ids(case, query.exclude_ids)
                    for temperature in ('cold_pair', 'warm_pair'):
                        timings = {name: [] for name in ('uncached', 'cached')}
                        first_seconds = {name: [] for name in timings}
                        second_seconds = {name: [] for name in timings}
                        sql_query_counts = {name: [] for name in timings}
                        route_order = []
                        cached_bytes = []
                        admissions = []
                        for repeat in range(repeats):
                            routes = ('uncached', 'cached') if repeat % 2 == 0 else ('cached', 'uncached')
                            route_order.append(routes)
                            for route in routes:
                                engine = recall_engines.SqliteIndexedRecallEngine(reader,
                                    **dict(eligibility_cache_entries=0) if route == 'uncached' else {})
                                if temperature == 'warm_pair':
                                    if engine.excluded_ids(pool, query) != expected:
                                        raise ValueError('eligibility profile changed exclusion IDs')
                                query_count = 0
                                started = time.perf_counter()
                                first = engine.excluded_ids(pool, query)
                                split = time.perf_counter()
                                second = engine.excluded_ids(pool, query)
                                ended = time.perf_counter()
                                if first != expected or second != expected:
                                    raise ValueError('eligibility profile changed exclusion IDs')
                                first_seconds[route].append(split - started)
                                second_seconds[route].append(ended - split)
                                timings[route].append(ended - started)
                                sql_query_counts[route].append(query_count)
                                if route == 'cached':
                                    cached_bytes.append(engine._eligibility_cache_bytes)
                                    admissions.append(any(value[0] is not None
                                                          for value in engine._eligibility_cache.values()))
                        medians = {name: median(values) for name, values in timings.items()}
                        paired_speedups = [uncached / cached if cached else None
                                           for uncached, cached in zip(timings['uncached'], timings['cached'])]
                        results.append(dict(rows=size + 1, cutoff_fraction=fraction,
                            temperature=temperature, excluded_ids=len(expected) - 1,
                            seconds=timings, first_seconds=first_seconds, second_seconds=second_seconds,
                            sql_query_counts=sql_query_counts, route_order=route_order,
                            paired_speedups=paired_speedups,
                            median_paired_speedup=median(value for value in paired_speedups if value is not None)
                            if any(value is not None for value in paired_speedups) else None,
                            median_seconds=medians, estimated_cache_payload_bytes=cached_bytes,
                            cache_admitted=all(admissions),
                            pair_speedup=medians['uncached'] / medians['cached'] if medians['cached'] else None))
            finally:
                reader.close()
    return dict(schema=1, scope='synthetic_eligibility_kernel_only', equivalent=True,
                model_calls=0, experiment_results_written=False, repeats=repeats,
                recall_routes_per_case=2, eligibility_cache_entries=8,
                eligibility_cache_budget_bytes=4 * 1024 * 1024, synthetic_id_length=24,
                prepared_minimum_identity_length=pool._minimum_identity_length,
                recall_engines_sha256=sha256_file(Path(recall_engines.__file__)), results=results)


def positive(value):
    value = int(value)
    if not 1 <= value <= 50:
        raise argparse.ArgumentTypeError('value must be within [1, 50]')
    return value


def inspect(instance, experiment, *, cases=3, repeats=1, materials=False):
    from src.config import valid_name
    from src.iteration import runtime, versions
    from src.iteration.pack_transport import connection_env_file
    directory = versions.PRIVATE / 'experiments' / valid_name(experiment)
    snapshot = runtime.verify(directory / 'runtime.json')
    command = [sys.executable, '-B', str(Path(__file__).resolve()),
               '--snapshot', str(snapshot), '--instance', valid_name(instance),
               '--experiment', directory.name, '--cases', str(positive(cases)),
               '--repeats', str(positive(repeats))]
    if materials:
        command.append('--materials')
    result = subprocess.run(command, cwd=snapshot, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ,
                                'DH_ENV_FILE': connection_env_file(),
                                'DH_INSTANCES_ROOT': str(versions.PRIVATE.parent),
                                'DH_SETTINGS_FILE': str(snapshot / 'settings.snapshot.yaml')})
    if result.returncode:
        raise RuntimeError(f'frozen retrieval profile failed: {result.stderr[-2000:]}')
    return json.loads(result.stdout)


def measure_materials(reuse, guard, data_ref, directories, dataset, *, profile, disk):
    calls = []
    try:
        for unused in range(2):
            with reuse.verified_materials(guard, profile=profile, disk_rows=disk):
                started = time.perf_counter()
                guard.require_materials(data_ref, directories, dataset)
                calls.append(time.perf_counter() - started)
    except Exception as exc:
        return dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                    calls_seconds=calls, metrics=profile.snapshot()['metrics'])
    return dict(status='passed', calls_seconds=calls, metrics=profile.snapshot()['metrics'])


def compare(retriever, original, recall, cases, *, repeats):
    optimized = retriever.retrieve
    def old(subject, **options):
        return [dict(row) for row in original(subject, **options)]
    legacy = MethodType(old, retriever)
    timings = {name: dict(recall_seconds=0.0, render_seconds=0.0, calls=0)
               for name in ('original', 'indexed')}
    strata = {}
    try:
        for repeat in range(repeats):
            for position, case in enumerate(cases):
                outputs = {}
                names = ('original', 'indexed') if (repeat + position) % 2 == 0 else ('indexed', 'original')
                for name in names:
                    retriever.retrieve = legacy if name == 'original' else optimized
                    started = time.perf_counter()
                    rows = recall(retriever, case)
                    timings[name]['recall_seconds'] += time.perf_counter() - started
                    started = time.perf_counter()
                    rendered = retriever.render_selected(rows)
                    timings[name]['render_seconds'] += time.perf_counter() - started
                    timings[name]['calls'] += 1
                    outputs[name] = rows, rendered
                if outputs['original'] != outputs['indexed']:
                    raise ValueError('indexed recall changed selected rows, features, order or rendering')
                if repeat == 0:
                    kind = case.get('chat_type', 'unknown')
                    strata[kind] = strata.get(kind, 0) + 1
    finally:
        retriever.retrieve = optimized
    before, after = (timings[name]['recall_seconds'] for name in ('original', 'indexed'))
    return dict(timings=timings, recall_speedup=before / after if after else None,
                equivalent=True, cases=len(cases), repeats=repeats, strata=strata,
                recall_routes_per_case=2, scope='recall_and_render_only',
                model_calls=0, experiment_results_written=False)


def _load_transport():
    """独立加载 pack_transport（profiler 子进程入口在冻结快照内运行，
    必须与活树 src.iteration.pack_transport 模块对象隔离）。"""
    transport_path = Path(__file__).with_name('pack_transport.py')
    spec = importlib.util.spec_from_file_location('_retrieval_profile_transport', transport_path)
    transport = importlib.util.module_from_spec(spec)
    exec(compile(transport_path.read_bytes(), str(transport_path), 'exec'), transport.__dict__)
    return transport


def frozen(args):
    transport = _load_transport()
    with transport.authenticated_snapshot_imports(args.snapshot):
        return _frozen_authenticated(args, transport)


def _frozen_authenticated(args, transport):
    from src.iteration import runtime, versions
    versions.switch_instance(args.instance)
    directory = versions.PRIVATE / 'experiments' / args.experiment
    snapshot = runtime.verify(directory / 'runtime.json')
    if snapshot.resolve() != args.snapshot.resolve():
        raise ValueError('experiment runtime does not match profiler snapshot')
    runtime.require_current(directory)
    # 认证 manifest 分代（pack_transport.snapshot_generation 同款判定）：旧代快照
    # 走归档 few_shot profiler；新代（fewshot 包）快照没有 few_shot.py，旧 profiler
    # 的 import 会炸，改为对新包做身份核验（清单字节一致），profile 由冻结
    # 运行器内 fewshot_profile_instrument 承担（transport receipt）。
    if transport.snapshot_generation(snapshot, runtime) == 'fewshot':
        return _frozen_modern(args, snapshot, directory)
    return _frozen_legacy(args, snapshot, directory, transport)


def _frozen_modern(args, snapshot, directory):
    """新代快照：旧 profiler 跳过；核验快照内 fewshot 包字节与认证清单一致。"""
    from src.config import sha256_file
    from src.iteration.storage import read_json
    try:
        manifest = json.loads((snapshot / 'manifest.json').read_text())
        files = manifest['files']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'cannot verify modern snapshot package identity: {exc}') from exc
    package = sorted(name for name in files
                     if name == 'src/generator/fewshot' or name.startswith('src/generator/fewshot/'))
    if not package:
        raise RuntimeError('modern snapshot manifest lists no fewshot package files')
    changed = [name for name in package if not (snapshot / name).is_file()
               or sha256_file(snapshot / name) != files[name]]
    if changed:
        raise RuntimeError('modern snapshot fewshot package identity mismatch: ' + ', '.join(changed))
    result = dict(schema=1, equivalent=True, generation='fewshot_package_v1',
                  scope='snapshot_package_identity_only', model_calls=0,
                  experiment_results_written=False, package_files=package,
                  runtime_ref=read_json(directory / 'runtime.json')['ref'])
    if args.materials:
        # 新代材料复用耗时由冻结运行器 validation.verified_materials 在 transport
        # receipt 内计时；旧 profiler 的 learning_guard/disk 适配器测量不再适用。
        result['materials'] = dict(status='skipped',
            reason='modern generation measures material reuse inside the frozen runner receipt')
    return result


def _frozen_legacy(args, snapshot, directory, transport):
    from src.generator import few_shot, learned_selection
    from src.iteration import datasets, gbdt_evidence, learning_guard, runtime, versions
    from src.iteration.storage import read_json
    from src.judge.corrected import CodexJudgeClient
    from src.llm import ChatClient
    pointers = versions.POINTERS_PATH.read_bytes()
    frozen_spec = read_json(directory / 'spec.json')
    data = versions.data_version_dir(frozen_spec['data_ref'])
    with datasets.case_path(data, frozen_spec['dataset']).open() as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    cases = cases[:args.cases]
    if not cases:
        raise ValueError('frozen case pool is empty')
    disk = transport.live_disk_history()
    adapter_path = Path(disk.__file__).with_name('indexed_retrieval.py')
    spec = importlib.util.spec_from_file_location('_retrieval_profile_indexed', adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    original = few_shot.PersonaFewShotRetriever.retrieve
    if adapter.indexed_retrieve(original) is original:
        raise ValueError('frozen retrieval is not supported by the exact indexed adapter')
    profile = transport.live_transport_profile().TransportProfile(directory / 'unused-profile.json', {})
    def forbidden(*unused, **options):
        raise RuntimeError('offline profiler prohibits model requests')
    with ExitStack() as stack:
        for owner, name in ((CodexJudgeClient, 'run'), (ChatClient, 'chat'), (ChatClient, '_send')):
            stack.enter_context(patch.object(owner, name, forbidden))
        stack.enter_context(transport.archived_runtime_paths(runtime))
        stack.enter_context(transport.archived_gbdt_paths(gbdt_evidence, runtime))
        stack.enter_context(transport.history_feature_cache(few_shot))
        stack.enter_context(disk.disk_storage(few_shot))
        started = time.perf_counter()
        retriever = few_shot.PersonaFewShotRetriever(data / 'fewshot_pool.jsonl')
        if not retriever.is_approved():
            raise ValueError('frozen history pool is not approved')
        preparation = time.perf_counter() - started
        result = compare(retriever, original, learned_selection.recall, cases, repeats=args.repeats)
        result.update(preparation_seconds=preparation, storage=disk.MODE,
                      runtime_ref=read_json(directory / 'runtime.json')['ref'],
                      selection='first_existing_frozen_cases_no_new_draw',
                      preparation_includes_index_open_or_build=True)
        if args.materials:
            directories = learning_guard._directories(frozen_spec)
            reuse = transport.live_material_reuse()
            result['materials'] = measure_materials(reuse, learning_guard, frozen_spec['data_ref'],
                directories, frozen_spec['dataset'], profile=profile, disk=disk)
    if versions.POINTERS_PATH.read_bytes() != pointers:
        raise ValueError('version pointers changed during offline profiling')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--cases', type=positive, default=3)
    parser.add_argument('--repeats', type=positive, default=1)
    parser.add_argument('--materials', action='store_true')
    with redirect_stdout(sys.stderr):
        result = frozen(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False))
