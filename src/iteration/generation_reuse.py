"""Reuse frozen development generations independently of their Judge outcome.

The source executor, source data/materials and each selected generation are
bound before pack creation. Missing outputs use the ordinary guarded builder.
"""
from __future__ import annotations

from ..config import ConfigError, sha256_file, valid_name
from . import runtime, versions
from .storage import read_json


def require(condition, message):
    if not condition:
        raise ConfigError('生成复用：' + message)


def source(recipe, experiment_id):
    directory = versions.PRIVATE / 'experiments' / valid_name(experiment_id)
    spec = read_json(directory / 'spec.json')
    require(spec.get('kind') == 'gen_ab' and spec.get('dataset') == 'development'
            and recipe['dataset'] == 'development', '仅复用开发生成，不跨固定验收')
    require(spec.get('data_ref') == recipe['data_ref']
            and spec.get('protocol', {}).get('force_reply') is True, '数据或强制回复条件不符')
    branches = [side for side in ('baseline', 'candidate')
                if spec.get(side + '_ref') == recipe['generator_ref']]
    require(len(branches) == 1, '来源生成器分支不唯一或不匹配')
    old, current = spec.get('learning_snapshot', {}), recipe['learning_snapshot']
    from .learning_guard import same_materials
    require(same_materials(old, current), '原始数据、用途清单或冻结生成器材料变化')
    runtime.verify_historical_implementation(directory, old.get('guard_code'))
    return directory, spec, branches[0]


def operation(trace, spec, branch, case):
    require(trace.get('case') == case and trace.get('case_id') == str(case['case_id']),
            '上文、答案或样本来源与冻结题目不符')
    require(all(trace.get('versions', {}).get(key) == spec.get(key) for key in
                ('kind', 'dataset', 'data_ref', 'baseline_ref', 'candidate_ref', 'judge_ref')),
            '生成实录版本不符')
    ops = [op for op in trace.get('operations', []) if op.get('kind') == 'generation'
           and op.get('branch') == branch and op.get('round') == 0 and op.get('status') == 'ok']
    require(len(ops) == 1, '缺少唯一的首轮成功生成')
    op = ops[0]
    require(op.get('input') == {'forced_reply': True}, '生成没有强制回复')
    replies = op.get('result', {}).get('replies')
    require(isinstance(replies, list) and bool(replies)
            and all(isinstance(s, str) and s.strip() for s in replies), '生成回复无效')
    return op['result']


def prepare(recipe, cases, experiment_id):
    directory, spec, branch = source(recipe, experiment_id)
    expected = {str(c['case_id']): c for c in cases}
    selected = {}
    # Earliest successful generation, regardless of Judge success or verdict.
    for path in sorted((directory / 'traces').glob('*.json')):
        trace = read_json(path)
        cid = trace.get('case_id')
        if cid not in expected or not any(op.get('kind') == 'generation'
                and op.get('branch') == branch and op.get('round') == 0
                and op.get('status') == 'ok' for op in trace.get('operations', [])):
            continue
        operation(trace, spec, branch, expected[cid])
        require(path.stem == trace.get('trace_ref'), '实录 ID 不符')
        rank = (trace['started_at'], path.name)
        if cid not in selected or rank < selected[cid][0]:
            selected[cid] = (rank, path)
    return {'schema': 1, 'experiment': experiment_id, 'branch': branch,
            'source_files': {name: sha256_file(directory / name)
                             for name in ('spec.json', 'runtime.json')},
            'traces': {cid: {'ref': p.stem, 'sha256': sha256_file(p)}
                       for cid, (_, p) in selected.items()}}


def load(recipe, cases):
    proof = recipe.get('generation_reuse')
    if not proof:
        return {}, []
    require(proof.get('schema') == 1, '未知复用证明格式')
    directory, spec, branch = source(recipe, proof['experiment'])
    require(proof['branch'] == branch, '来源分支变化')
    require(set(proof['source_files']) == {'spec.json', 'runtime.json'}, '来源绑定不完整')
    paths = []
    for name, digest in proof['source_files'].items():
        path = directory / name
        require(sha256_file(path) == digest, '来源配方或运行环境变化')
        paths.append(path)
    expected = {str(c['case_id']): c for c in cases}
    results = {}
    for cid, binding in proof['traces'].items():
        require(cid in expected, '复用题目不属于目标清单')
        ref = valid_name(binding['ref'])
        path = directory / 'traces' / (ref + '.json')
        require(sha256_file(path) == binding['sha256'], '来源实录变化')
        trace = read_json(path)
        require(trace.get('trace_ref') == ref, '来源实录 ID 不符')
        results[cid] = {'result': operation(trace, spec, branch, expected[cid]),
                        'source': {'experiment': proof['experiment'], **binding}}
        paths.append(path)
    return results, paths


def verify_rows(recipe, cases, rows):
    reused, _ = load(recipe, cases)
    for row in rows:
        if str(row['case_id']) in reused:
            require(row.get('ai_replies') == reused[str(row['case_id'])]['result']['replies']
                    and row.get('generation_status') != 'failed', '复用回复被替换')
