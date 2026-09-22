"""Recorded transport and historical-code path compatibility for frozen workers."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache, wraps
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid


@contextmanager
def history_feature_cache(module):
    """Memoize pure text analysis; return fresh mutable results to every caller.

    Keys include the complete text and all feature flags. Per-case history
    eligibility and row overrides are deliberately outside this cache.
    """
    originals = {}
    def wrap(original):
        cached = lru_cache(maxsize=262144)(original)
        @wraps(original)
        def call(*args, **kwargs):
            return cached(*args, **kwargs).copy()
        return call
    try:
        for name in ('_situation_tags', '_situation_profile'):
            originals[name] = getattr(module, name)
            setattr(module, name, wrap(originals[name]))
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def timeout_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 3600:
        raise ValueError('request timeout must be finite and within (0, 3600] seconds')
    return value


def connection_env_file():
    """Keep credentials in the live project, outside the frozen executor."""
    return str(Path(os.environ.get('DH_ENV_FILE',
        Path(__file__).resolve().parents[2] / '.env')).resolve())


@contextmanager
def timeout_floor(client_type, seconds):
    """Change only the local wait limit; preserve request and cache identities."""
    seconds = timeout_value(seconds)
    original = client_type.__init__

    def initialize(self, settings, model_cfg):
        config = {**model_cfg, 'timeout_seconds': max(
            float(model_cfg.get('timeout_seconds', 60)), seconds)}
        original(self, settings, config)

    client_type.__init__ = initialize
    try:
        yield
    finally:
        client_type.__init__ = original


@contextmanager
def archived_gbdt_paths(module, runtime):
    """Adapt the old audit's physical-path manifest to verified code archives.

    The frozen verifier still reconstructs the complete model. Only its in-memory
    evidence-key view changes; original records, hashes and executor stay intact.
    """
    original_verify = module.verify
    if getattr(original_verify, '_archived_gbdt_paths', False):
        yield
        return

    def verify(directory, record, original, study, source_spec, arm):
        descriptor = module.read(directory / 'scorer_source.json')
        sweep = module.versions.PRIVATE / 'judge_training' / module._name(descriptor['study'])
        inputs = module.read(sweep / 'spec.json')['inputs']
        inherited = module.read(original / 'learning.json')['evidence_files']
        evidence = dict(record['evidence_files'])
        for name, expected in inputs.items():
            actual = runtime.evidence_path(name, expected)
            if str(actual.resolve()) == str(Path(name).resolve()):
                continue
            module.require(evidence.get(name) == expected,
                           'archived GBDT source is missing from original evidence')
            alias = str(actual.resolve())
            module.require(alias not in evidence or evidence[alias] == expected,
                           'archived GBDT source identity conflicts')
            evidence[alias] = expected
            if name not in inherited:
                del evidence[name]
        return original_verify(directory, {**record, 'evidence_files': evidence},
                               original, study, source_spec, arm)

    verify._archived_gbdt_paths = True
    module.verify = verify
    try:
        yield
    finally:
        module.verify = original_verify


def launch(directory, *, instance, job, workers, seconds, kind='pack'):
    from . import runtime, versions
    seconds = timeout_value(seconds)
    snapshot = runtime.verify(Path(directory) / 'runtime.json')
    # Fresh interpreter imports only the verified original executor. No snapshot
    # file, model version, request body or checkpoint is rewritten.
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
               '--snapshot', str(snapshot), '--instance', instance, '--job', job,
               '--workers', str(workers), '--timeout-seconds', str(seconds), '--kind', kind]
    return subprocess.call(command, cwd=snapshot, env={**os.environ,
        'DH_ENV_FILE': connection_env_file(),
        'DH_INSTANCES_ROOT': str(versions.PRIVATE.parent),
        'DH_JOB_DIR': str(directory),
        'DH_SETTINGS_FILE': str(snapshot / 'settings.snapshot.yaml')})


def main():
    # A long-running parent may have imported an older launch function. Resolve
    # the default here too, before frozen load_settings() looks under its ROOT.
    os.environ.setdefault('DH_ENV_FILE', connection_env_file())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--kind', choices=['pack', 'experiment', 'training'], default='pack')
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=timeout_value, required=True)
    args = parser.parse_args()
    snapshot = Path(args.snapshot).resolve()
    sys.path.insert(0, str(snapshot))
    from src.config import valid_name, sha256_file
    from src.iteration import gbdt_evidence, runtime, versions
    from src.iteration.storage import read_json, write_once_json
    from src.llm import ChatClient
    from src.generator import few_shot
    versions.switch_instance(valid_name(args.instance))
    from src.iteration import jobs
    directory = jobs.directory({'kind': args.kind, 'id': valid_name(args.job)})
    if runtime.verify(directory / 'runtime.json').resolve() != snapshot:
        raise RuntimeError('timeout worker must use the recorded frozen runtime')
    runtime.require_current(directory)
    receipt = {'schema': 1, 'kind': 'transport_timeout_floor',
               'timeout_seconds': args.timeout_seconds, 'workers': args.workers,
               'historical_code_path_compatibility': 'verified_gbdt_archives_v1',
               'history_feature_cache': 'exact_text_and_flags_copy_v1',
               'runtime': read_json(directory / 'runtime.json'),
               'launcher_sha256': sha256_file(Path(__file__)),
               'created_at': time.time(), 'pid': os.getpid()}
    write_once_json(directory / 'transport_overrides' / (uuid.uuid4().hex + '.json'), receipt)
    print(f'Frozen {args.kind} resume: request timeout floor={args.timeout_seconds:g}s; '
          'original requests and saved rows retained', flush=True)
    sys.argv = [str(snapshot / 'scripts/iterate_branches.py'), '--instance', args.instance,
                'worker', '--kind', args.kind, '--job', args.job, '--workers', str(args.workers)]
    with timeout_floor(ChatClient, args.timeout_seconds), archived_gbdt_paths(gbdt_evidence, runtime), \
            history_feature_cache(few_shot):
        runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
