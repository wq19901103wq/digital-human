"""Canonical experiment records, frozen case membership and checked completion."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

from ..config import ConfigError, valid_name, sha256_file
from . import datasets, journal, versions
from .storage import read_json, file_lock


def serialize_case(row) -> str:
    """cases.jsonl 行的唯一规范序列化：writer 写入与重放预检必须同源。"""
    return json.dumps(row, ensure_ascii=False, allow_nan=False)


def directory(exp_id: str) -> Path:
    target = versions.PRIVATE / 'experiments' / valid_name(exp_id)
    require_location(target)
    return target


def require_location(target: Path, spec=None) -> None:
    target = Path(target).absolute()
    expected = versions.PRIVATE.absolute() / 'experiments' / valid_name(target.name)
    if target != expected or target.resolve() != expected.absolute():
        raise ConfigError('评测只能写入当前实例的 experiments/<id>，禁止其他目录或符号链接')
    if spec is not None and spec.get('id') != target.name:
        raise ConfigError('实验 ID 与目录不一致')


def planned_rows(spec):
    if spec['kind'] == 'judge_eval':
        path = versions.PRIVATE / 'judge_eval' / valid_name(spec['pack_ref']) / 'pack.json'
        if spec.get('comparison') and sha256_file(path) != spec.get('pack_sha256'):
            raise ConfigError('专项比较的冻结回复包已变化')
        rows = read_json(path)['rows']
    else:
        rows = datasets.rows_for(spec)
    if spec.get('smoke'):
        rows = rows[:int(spec.get('smoke_limit') or 5)]
    return rows


def plan(rows):
    ids = [str(row['case_id']) for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(not cid for cid in ids):
        raise ConfigError('实验必须冻结非空且不重复的 case_id 清单')
    return {'schema': 1, 'case_ids': ids}


def require_registered(target: Path):
    require_location(target)
    if any((target / name).is_symlink() for name in ('spec.json', 'state.json', 'cases.jsonl')):
        raise ConfigError('实验规格、状态和逐题结果不能是符号链接')
    spec = read_json(target / 'spec.json')
    require_location(target, spec)
    if spec.get('kind') not in ('gen_ab', 'judge_eval') or spec.get('dataset') not in ('development', 'fixed_test'):
        raise ConfigError('未知实验类型或数据用途')
    expected = plan(planned_rows(spec))
    if spec.get('record_contract') is not None and spec['record_contract'] != expected:
        raise ConfigError('冻结数据题单指纹与实验输入不一致')
    return spec, set(expected['case_ids'])


def validate_case(row, spec, ids):
    if str(row.get('case_id', '')) not in ids or row.get('status') not in ('ok', 'failed'):
        raise ConfigError('逐题记录必须属于冻结题单，并明确成功或失败')
    if row['status'] == 'failed':
        if not row.get('reason'):
            raise ConfigError('失败题记录必须保存原因')
        return
    keys = ('baseline_correct', 'candidate_correct') if spec['kind'] == 'judge_eval' else ('identified_baseline', 'identified_candidate')
    if any(type(row.get(key)) is not bool for key in keys):
        raise ConfigError('成功记录缺少两侧有效判定')


@contextmanager
def writer(target: Path):
    require_registered(target)
    with file_lock(target / '.run.lock', blocking=False), _writer(target) as append:
        yield append


@contextmanager
def _writer(target: Path):
    spec, ids = require_registered(target)
    state = read_json(target / 'state.json', default={})
    if state.get('status') == 'finished':
        raise ConfigError('已完成实验不能追加结果；行为变化必须新建实验')
    path = target / 'cases.jsonl'
    if path.is_symlink():
        raise ConfigError('逐题记录不能是符号链接')
    journal.recover(path)
    existing = {str(row['case_id']): row for row in journal.records(path)}
    with path.open('a', encoding='utf-8') as stream:
        def append(row):
            validate_case(row, spec, ids)
            old = existing.get(str(row['case_id']))
            if old and old.get('status') == 'ok':
                if old != row:
                    raise ConfigError('成功断点不得覆盖；新结果必须新建实验')
                return
            stream.write(serialize_case(row) + '\n')
            stream.flush()
            existing[str(row['case_id'])] = row
        try:
            yield append
        finally:
            stream.flush()
            os.fsync(stream.fileno())


def validate_completion(target: Path, metrics: dict, *, complete=True):
    try:
        json.dumps(metrics, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ConfigError('实验指标必须是有效 JSON，不能包含 NaN 或无穷数') from exc
    spec, ids = require_registered(target)
    path = target / 'cases.jsonl'
    blob = path.read_bytes() if path.exists() else b''
    rows, tail = journal._parse(blob)
    if tail is not None:
        raise ConfigError('逐题文件有未完成尾行，不能完成实验')
    final = {}
    for row in rows:
        validate_case(row, spec, ids)
        final[str(row['case_id'])] = row
    if complete and set(final) != ids:
        raise ConfigError('逐题记录不完整，未覆盖冻结题单，不能标记完成')
    ok = [r for r in final.values() if r['status'] == 'ok']
    bkey, ckey = ('baseline_correct', 'candidate_correct') if spec['kind'] == 'judge_eval' else ('identified_baseline', 'identified_candidate')
    expected = {'attempted': len(ids), 'pairs': len(ok), 'failures': len(final)-len(ok),
                'identified_baseline': sum(r[bkey] for r in ok),
                'identified_candidate': sum(r[ckey] for r in ok)}
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ConfigError(f'逐题结果与汇总不一致: {key}')
    failure_rate = expected['failures'] / expected['attempted']
    if 'failure_rate' in metrics and abs(metrics['failure_rate'] - failure_rate) > 0.0001:
        raise ConfigError('逐题失败数与失败率不一致')
    # Every differing case must carry the frozen independent vote sequence.
    extra = int(spec.get('protocol', {}).get('flip_extra_rounds', 0))
    if complete and extra and not spec.get('smoke'):
        for row in ok:
            if row[bkey] == row[ckey]:
                continue
            votes = row if spec['kind'] == 'judge_eval' else row.get('flip_verified')
            if not isinstance(votes, dict) or not row.get('flip_verified'):
                raise ConfigError('分歧题缺少独立补验，不能标记完成')
            for side, key in [('baseline', bkey), ('candidate', ckey)]:
                values = votes.get(side + '_votes', [])
                if (len(values) != extra + 1 or any(type(v) is not bool for v in values)
                        or values[0] != row[key]
                        or votes.get(side + '_identified_final') is not (sum(values) > len(values)/2)):
                    raise ConfigError('补验票数不完整，或初测、最终判定不一致')
    if complete:
        wins = losses = contested = 0
        judge = spec['kind'] == 'judge_eval'
        for row in ok:
            if row[bkey] == row[ckey]:
                continue
            verified = row if judge and row.get('flip_verified') else row.get('flip_verified')
            if verified:
                b, c = verified['baseline_identified_final'], verified['candidate_identified_final']
            elif judge:
                b, c = row[bkey], row[ckey]
            else:
                continue
            if b == c:
                contested += 1
            elif c if judge else b:
                wins += 1
            else:
                losses += 1
        for key, expected_value in {'wins_confirmed': wins, 'losses_confirmed': losses,
                                    'net_win_confirmed': wins-losses, 'contested': contested}.items():
            if key in metrics and metrics[key] != expected_value:
                raise ConfigError(f'逐题补验与汇总不一致: {key}')
    return spec
