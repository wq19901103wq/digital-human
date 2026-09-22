"""Pre-sealed, non-reusable acceptance batches with immutable owner bindings."""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict

from ..cache import digest
from ..config import ConfigError, sha256_file
from . import datasets, versions
from .storage import file_lock, write_json


def _root(data_ref):
    versions.data_version_dir(data_ref)
    return versions.PRIVATE / 'acceptance' / data_ref


def exists(data_ref):
    return (_root(data_ref) / 'manifest.json').is_file()


def seal(data_ref, batch_size, seed=42):
    if type(batch_size) is not int or batch_size < 1:
        raise ConfigError('batch size must be positive')
    root = _root(data_ref)
    with file_lock(root / '.lock'):
        if (root / 'manifest.json').exists():
            raise ConfigError('acceptance pool is already sealed; create a new data version')
        for p in (versions.PRIVATE / 'experiments').glob('*/spec.json'):
            spec = json.loads(p.read_text())
            if spec.get('data_ref') == data_ref and spec.get('dataset') == 'fixed_test':
                raise ConfigError('cannot seal after acceptance has been exposed')
        for p in (versions.PRIVATE / 'judge_eval').glob('*/pack.json'):
            pack = json.loads(p.read_text())
            if pack.get('data_ref') == data_ref and 'validation' in p.parent.name:
                raise ConfigError('cannot repartition an existing validation pack')
        path = datasets.case_path(versions.data_version_dir(data_ref), 'fixed_test')
        rows = [json.loads(s) for s in path.read_text().splitlines() if s]
        if len({r['case_id'] for r in rows}) != len(rows) or len(rows) < batch_size:
            raise ConfigError('acceptance pool is too small or has duplicate cases')
        if len(rows) % batch_size:
            raise ConfigError('pool size must be an exact multiple of the predeclared batch size')
        # Stratified round-robin partitioning is frozen before candidate evaluation.
        groups, rng = defaultdict(list), random.Random(seed)
        for r in rows:
            groups[(r.get('chat_type'), r.get('familiarity'))].append(r)
        batches = [[] for _ in range(len(rows) // batch_size)]
        index = 0
        for group in groups.values():
            rng.shuffle(group)
            for row in group:
                batches[index % len(batches)].append(row)
                index += 1
        value = {'schema': 1, 'data_ref': data_ref, 'source_sha256': sha256_file(path),
                 'batch_size': batch_size, 'seed': seed,
                 'batches': [{'id': f'b-{i + 1:04d}', 'case_ids': [r['case_id'] for r in batch],
                              'rows_sha256': digest(batch)} for i, batch in enumerate(batches)]}
        write_json(root / 'manifest.json', value)
        write_json(root / 'ledger.json', {'manifest_sha256': sha256_file(root / 'manifest.json'), 'claims': {}})
        return {'batches': len(batches), 'cases_per_batch': batch_size}


def binding(spec):
    return {k: spec.get(k) for k in ('kind', 'data_ref', 'baseline_ref', 'candidate_ref', 'judge_ref', 'generator_ref', 'protocol')}


def claim(spec):
    root = _root(spec['data_ref'])
    with file_lock(root / '.lock'):
        manifest = json.loads((root / 'manifest.json').read_text())
        ledger = json.loads((root / 'ledger.json').read_text())
        if sha256_file(root / 'manifest.json') != ledger['manifest_sha256']:
            raise ConfigError('sealed acceptance manifest changed')
        claims = ledger['claims']
        identity = binding(spec)
        if spec['id'] in claims:
            receipt = claims[spec['id']]['receipt']
            if receipt['binding'] != identity:
                raise ConfigError('acceptance owner cannot change its baseline or candidate')
            return receipt
        used = {c['receipt']['batch_id'] for c in claims.values()}
        batch = next((b for b in manifest['batches'] if b['id'] not in used), None)
        if batch is None:
            raise ConfigError('acceptance_exhausted: new sealed data is required')
        receipt = {'schema': 1, 'experiment_id': spec['id'], 'data_ref': spec['data_ref'],
                   'batch_id': batch['id'], 'rows_sha256': batch['rows_sha256'],
                   'manifest_sha256': ledger['manifest_sha256'], 'binding': identity}
        # Allocation consumes the batch, even if cancelled or a process crashes.
        claims[spec['id']] = {'receipt': receipt, 'status': 'reserved', 'reserved_at': time.time()}
        write_json(root / 'ledger.json', ledger)
        return receipt


def rows(receipt, spec=None):
    root = _root(receipt['data_ref'])
    manifest = json.loads((root / 'manifest.json').read_text())
    ledger = json.loads((root / 'ledger.json').read_text())
    saved = ledger['claims'].get(receipt['experiment_id'], {})
    if (saved.get('receipt') != receipt or sha256_file(root / 'manifest.json') != receipt['manifest_sha256']
            or (spec is not None and (binding(spec) != receipt['binding'] or spec['id'] != receipt['experiment_id']))):
        raise ConfigError('acceptance ownership or sealed contents changed')
    path = datasets.case_path(versions.data_version_dir(receipt['data_ref']), 'fixed_test')
    if sha256_file(path) != manifest['source_sha256']:
        raise ConfigError('acceptance source changed')
    all_rows = {r['case_id']: r for r in (json.loads(s) for s in path.read_text().splitlines() if s)}
    batch = next(b for b in manifest['batches'] if b['id'] == receipt['batch_id'])
    selected = [all_rows[cid] for cid in batch['case_ids']]
    if digest(selected) != receipt['rows_sha256']:
        raise ConfigError('acceptance batch changed')
    return selected


def transition(spec, status):
    receipt = spec.get('acceptance')
    if not receipt:
        return
    rows(receipt, spec)
    root = _root(spec['data_ref'])
    with file_lock(root / '.lock'):
        ledger = json.loads((root / 'ledger.json').read_text())
        claim = ledger['claims'][spec['id']]
        if claim['status'] == 'finished' and status != 'finished':
            raise ConfigError('finished acceptance cannot be reopened')
        claim.update(status=status, updated_at=time.time())
        write_json(root / 'ledger.json', ledger)


def status(data_ref):
    root = _root(data_ref)
    manifest = json.loads((root / 'manifest.json').read_text())
    ledger = json.loads((root / 'ledger.json').read_text())
    return {'total': len(manifest['batches']), 'consumed': len(ledger['claims']),
            'remaining': len(manifest['batches']) - len(ledger['claims']),
            'jobs': {k: {'batch': v['receipt']['batch_id'], 'status': v['status']} for k, v in ledger['claims'].items()}}
