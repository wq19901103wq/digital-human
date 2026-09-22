"""Explicit, audited initialization/migration of a compatible baseline tuple.

A migration establishes a starting point. Performance promotion still requires
an experiment and its ordinary development/fixed gates.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..cache import digest
from ..config import ConfigError
from . import learning_guard as guard, versions
from .storage import write_json


def _selection(data, generator, judge):
    return {'data': data, 'production_gen': generator, 'iteration_gen': generator,
            'production_judge': judge, 'iteration_judge': judge}


def _proof(target):
    from . import gbdt_evidence, pack_transport, runtime
    dirs = [versions.generator_dir(target['production_gen']),
            versions.judge_dir(target['production_judge'])['dir']]
    seal = guard.RunSeal([versions.data_version_dir(target['data']), *dirs])
    with pack_transport.archived_gbdt_paths(gbdt_evidence, runtime):
        proof = guard.snapshot(target['data'], dirs, 'development')
    seal.check()
    return proof, seal


@versions.transaction
def prepare(data, generator, judge, *, reason):
    if not reason or not reason.strip():
        raise ConfigError('基线初始化/迁移必须说明原因')
    before = versions.load_pointers() if versions.POINTERS_PATH.exists() else {}
    target = _selection(data, generator, judge)
    proof, seal = _proof(target)
    receipt = {'schema': 1, 'kind': 'baseline_migration', 'id': uuid.uuid4().hex,
               'before': before, 'after': target, 'source_proof': proof,
               'reason': reason, 'created_at': time.time(), 'performance_claim': False}
    receipt['sha256'] = digest(receipt)
    path = versions.PRIVATE / 'baseline_migrations' / (receipt['id'] + '.json')
    seal.check()
    write_json(path, receipt)
    return path


def _data_proof(data, report):
    from ..bootstrap.data_quality import accepted_report
    directory = versions.data_version_dir(data)
    seal = guard.RunSeal([directory], [Path(report)])
    proof = accepted_report(directory, report)
    seal.check()
    return proof, seal


@versions.transaction
def prepare_data(data, report, *, reason):
    """Adopt an accepted dataset only; never certify retained model materials."""
    if not reason or not reason.strip():
        raise ConfigError('数据基线迁移必须说明原因')
    before = versions.load_pointers()
    proof, seal = _data_proof(data, report)
    target = {k: v for k, v in before.items() if k != 'baseline_migration'}
    target['data'] = data
    receipt = dict(schema=1, kind='data_baseline_migration', id=uuid.uuid4().hex,
        before=before, after=target, source_proof=proof, reason=reason,
        created_at=time.time(), performance_claim=False, model_compatibility_verified=False,
        evaluation_started=False)
    receipt['sha256'] = digest(receipt)
    path = versions.PRIVATE / 'baseline_migrations' / (receipt['id'] + '.json')
    seal.check()
    write_json(path, receipt)
    return path


@versions.transaction
def apply(receipt_id):
    if not isinstance(receipt_id, str) or len(receipt_id) != 32 or any(c not in '0123456789abcdef' for c in receipt_id):
        raise ConfigError('非法基线迁移凭证')
    path = versions.PRIVATE / 'baseline_migrations' / (receipt_id + '.json')
    receipt = json.loads(path.read_text())
    if receipt.get('sha256') != digest({k: v for k, v in receipt.items() if k != 'sha256'}):
        raise ConfigError('迁移凭证内容变化')
    before = versions.load_pointers() if versions.POINTERS_PATH.exists() else {}
    target = {**receipt['after'], 'baseline_migration': receipt_id}
    if before == target:
        return before
    if before != receipt['before']:
        raise ConfigError('当前基线已前进；重新准备迁移，禁止覆盖新基线')
    if receipt['kind'] == 'data_baseline_migration':
        proof, seal = _data_proof(receipt['after']['data'], receipt['source_proof']['report'])
    else:
        proof, seal = _proof(receipt['after'])
    if proof != receipt['source_proof']:
        raise ConfigError('迁移材料或来源规则变化；禁止采用旧凭证')
    # Receipt persists the exact previous tuple before the single atomic switch.
    seal.check()
    versions.save_pointers(target)
    return target
