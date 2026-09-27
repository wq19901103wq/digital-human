"""Versioned purpose manifests; legacy datasets remain readable without migration."""
from __future__ import annotations

import json
from collections import Counter
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


def fixed_refresh_compatible(parent_ref: str, child_ref: str, *, required=False) -> bool:
    """Authenticate a fixed-only child before carrying original development evidence."""
    from . import versions
    from ..bootstrap.acceptance_refresh import preserved_files
    if parent_ref == child_ref:
        return True
    parent, child = (versions.data_version_dir(ref) for ref in (parent_ref, child_ref))
    old, new = manifest(parent), manifest(child)
    proof = new.get('acceptance_refresh', {})
    if proof.get('parent_ref') != parent_ref:
        if required:
            raise ConfigError('数据变更不是原开发集的固定批次更新')
        return False
    def require(ok):
        if not ok:
            raise ConfigError('固定批次更新的来源或保留材料已变化')
    require(proof.get('schema') == 1 and new.get('data_ref') == child_ref)
    require(sha256_file(parent / 'manifest.json') == proof['parent_manifest_sha256'] and
            sha256_file(parent / 'purposes.json') == proof['parent_purposes_sha256'])
    require({k: v for k, v in old.items() if k not in {'data_ref', 'roles', 'acceptance_refresh'}} ==
            {k: v for k, v in new.items() if k not in {'data_ref', 'roles', 'acceptance_refresh'}})
    require({k: v for k, v in old['roles'].items() if k != 'fixed_test'} ==
            {k: v for k, v in new['roles'].items() if k != 'fixed_test'})
    require(set(proof['preserved_files']) == preserved_files(old))
    for name, digest in proof['preserved_files'].items():
        require(Path(name).name == name and not (child / name).is_symlink())
        require(sha256_file(parent / name) == digest == sha256_file(child / name))
    old_meta = json.loads((parent / 'manifest.json').read_text())
    new_meta = json.loads((child / 'manifest.json').read_text())
    require(new_meta['purposes_sha256'] == sha256_file(child / 'purposes.json'))
    for meta in (old_meta, new_meta):
        for key in ('id', 'created', 'purposes_sha256'):
            meta.pop(key, None)
        meta.get('testsets', {}).pop('fixed_test', None)
    require(old_meta == new_meta)
    require(parent_ref in proof['excluded_batches'])
    excluded = set()
    for ref, digest in proof['excluded_batches'].items():
        path = case_path(versions.data_version_dir(ref), 'fixed_test')
        require(sha256_file(path) == digest)
        with path.open() as stream:
            excluded.update(str(json.loads(line)['case_id']) for line in stream if line.strip())
    fixed = case_path(child, 'fixed_test')
    with fixed.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [str(row['case_id']) for row in rows]
    require(len(ids) == len(set(ids)) == old['protocol']['acceptance_total'] and not set(ids) & excluded)
    sampling = proof.get('fixed_sampling')
    if sampling is not None:
        cap = sampling.get('chat_cap')
        require(sampling.get('previous_chat_cap') == old['roles']['fixed_test'].get(
            'chat_cap', old['protocol']['evaluation_chat_cap']))
        require(type(cap) is int and cap > 0 and new['roles']['fixed_test'].get('chat_cap') == cap)
        require(max(Counter(row['source_span']['chat_id'] for row in rows).values()) <= cap)
    return True


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
