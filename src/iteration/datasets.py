"""Versioned purpose manifests; legacy datasets remain readable without migration."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ConfigError, sha256_file

ROLES = {
    'gen_learning': ('Gen 学习材料', 'gen_learning.jsonl'),
    'gen_optimization': ('Gen 优化题', 'gen_optimization.jsonl'),
    'judge_training': ('Judge 训练题', 'judge_training.jsonl'),
    'development': ('Gen 开发验证', 'dev_pool.jsonl'),
    'judge_development': ('Judge 开发验证', 'judge_dev_pool.jsonl'),
    'fixed_test': ('固定验收', 'fixed_test.jsonl'),
}


def pack_cases(pack: dict) -> list[dict]:
    """Recover the original cases; generation fields are the only additions."""
    generated = {'ai_replies', 'generation_status', 'generation_error', 'generation_trace_ref'}
    return [{key: value for key, value in row.items() if key not in generated} for row in pack['rows']]


def manifest(directory: Path) -> dict:
    path = directory / 'purposes.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def case_path(directory: Path, role: str) -> Path:
    if role not in ROLES:
        raise ConfigError(f'未知数据用途: {role}')
    info = manifest(directory)
    filename = info.get('roles', {}).get(role, {}).get('file', ROLES[role][1])
    path = directory / filename
    if Path(filename).name != filename or path.is_symlink():
        raise ConfigError('用途清单路径非法')
    if info:
        entry = info.get('roles', {}).get(role)
        if not entry or not path.is_file() or sha256_file(path) != entry.get('sha256'):
            raise ConfigError(f'{role} 样本清单或内容已变化')
    return path


def snapshot(directory: Path) -> dict:
    info = manifest(directory)
    if not info:
        return {}
    for role in info['roles']:
        case_path(directory, role)
    return {'purposes_sha256': sha256_file(directory / 'purposes.json'),
            'history_policy': info['history_policy'],
            'protocol': info['protocol']}


def assert_pack(directory: Path, pack: dict, role: str) -> None:
    if not manifest(directory):
        return
    from . import acceptance
    if role == 'fixed_test' and acceptance.exists(directory.name) and not pack.get('acceptance'):
        raise ConfigError('sealed validation requires its batch receipt')
    if role == 'fixed_test' and pack.get('acceptance'):
        from .acceptance import rows as batch_rows
        expected = batch_rows(pack['acceptance'])
    else:
        expected = [json.loads(line) for line in case_path(directory, role).read_text().splitlines() if line]
    rows = pack.get('rows', [])
    if len(rows) != len(expected) or any(any(row.get(k) != v for k, v in case.items())
                                       for case, row in zip(expected, rows)):
        raise ConfigError(f'Judge 回复包不属于冻结的 {role} 清单')


def static_sources(directory: Path, assets: list[Path], role: str) -> dict:
    """Missing historic provenance stays unknown, never gets certified by copying."""
    info = manifest(directory)
    if not info:
        return {}
    cutoff = info['protocol']['development_start']
    findings = []
    for asset in assets:
        path = asset / 'learning.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        verified = (record.get('verified') is True and
                    isinstance(record.get('information_end'), (int, float)) and
                    record['information_end'] < cutoff and bool(record.get('evidence_files')))
        error = None
        if verified:
            from .runtime import verify_inputs
            try:
                verify_inputs(record['evidence_files'])
            except ConfigError as exc:
                verified = False
                error = str(exc)
        findings.append({'asset': asset.name, 'verified': verified,
                         'information_end': record.get('information_end'),
                         **({'error': error} if error else {})})
    return {'promotion_eligible': all(x['verified'] for x in findings), 'assets': findings,
            'reason': '静态资产需要可核验的学习范围，未知历史来源禁止运行或晋升'}


def acceptance_available(directory: Path, experiment_id: str) -> None:
    from . import acceptance
    if acceptance.exists(directory.name):
        if experiment_id not in acceptance.status(directory.name)['jobs']:
            raise ConfigError('sealed acceptance requires a bound batch allocation')
        return
    if not manifest(directory):
        return
    for path in (directory.parent.parent / 'experiments').glob('*/spec.json'):
        spec = json.loads(path.read_text())
        if (spec.get('data_ref') == directory.name and spec.get('dataset') == 'fixed_test'
                and path.parent.name != experiment_id):
            raise ConfigError('该验收批次已被实验锁定；恢复原实验，或用新批次比较当前基线与候选')


def cohort_metrics(cases: list[dict], records: dict, *, judge=False) -> dict:
    groups = {}
    for case in cases:
        if 'familiarity' not in case:
            continue
        key = case['familiarity'] + '/' + case['chat_type']
        group = groups.setdefault(key, {'attempted': 0, 'pairs': 0, 'failures': 0,
                                       'identified_baseline': 0, 'identified_candidate': 0})
        group['attempted'] += 1
        record = records.get(str(case['case_id']), {})
        if record.get('status') != 'ok':
            group['failures'] += 1
            continue
        group['pairs'] += 1
        for side in ('baseline', 'candidate'):
            group['identified_' + side] += bool(record.get(side + '_correct' if judge else 'identified_' + side))
    return groups


def rows_for(spec):
    if spec.get('dataset') == 'fixed_test' and spec.get('acceptance'):
        from .acceptance import rows
        return rows(spec['acceptance'], spec)
    from . import versions
    path = case_path(versions.data_version_dir(spec['data_ref']),
                     spec.get('purpose', spec['dataset']))
    return [json.loads(s) for s in path.read_text().splitlines() if s]
