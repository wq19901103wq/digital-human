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


def frozen(args):
    transport_path = Path(__file__).with_name('pack_transport.py')
    spec = importlib.util.spec_from_file_location('_retrieval_profile_transport', transport_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    sys.path.insert(0, str(args.snapshot))
    from src.generator import few_shot, learned_selection
    from src.iteration import datasets, gbdt_evidence, learning_guard, runtime, versions
    from src.iteration.storage import read_json
    from src.judge.corrected import CodexJudgeClient
    from src.llm import ChatClient
    versions.switch_instance(args.instance)
    directory = versions.PRIVATE / 'experiments' / args.experiment
    if runtime.verify(directory / 'runtime.json').resolve() != args.snapshot.resolve():
        raise ValueError('experiment runtime does not match profiler snapshot')
    runtime.require_current(directory)
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
