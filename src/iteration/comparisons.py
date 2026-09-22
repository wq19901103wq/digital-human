"""Register diagnostic comparisons before work, and import verifiable legacy results."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from ..config import ConfigError, sha256_file, valid_name
from . import experiment, gates, record_contract, versions
from .storage import file_lock, read_json, write_json


def register(exp_id, *, data_ref, pack_ref, comparison, change, protocol, created=None, provenance=None):
    """The only output directory a comparison producer receives is experiments/<id>."""
    target = record_contract.directory(exp_id)
    if not change.strip() or set(comparison) != {'baseline', 'candidate'}:
        raise ConfigError('专项比较必须明确改动以及对照、候选配置')
    pack_path = versions.PRIVATE / 'judge_eval' / valid_name(pack_ref) / 'pack.json'
    pack = read_json(pack_path)
    if pack.get('data_ref') != data_ref or pack_ref.startswith('pack-validation') or pack.get('acceptance'):
        raise ConfigError('专项比较仅接受同数据版本的开发回复包')
    from . import datasets
    datasets.assert_pack(versions.data_version_dir(data_ref), pack, 'judge_development')
    spec = dict(id=exp_id, kind='judge_eval', dataset='development', data_ref=data_ref,
        pack_ref=pack_ref, pack_sha256=sha256_file(pack_path), generator_ref=pack['c0_gen_version'],
        comparison=comparison, single_change=change, protocol=protocol,
        created=created or time.strftime('%Y-%m-%d %H:%M:%S'), adoption_allowed=False,
        record_contract=record_contract.plan(pack['rows']),
        provenance=provenance or {'mode': 'registered_comparison'})
    with file_lock(target / '.run.lock', blocking=False):
        if (target / 'spec.json').exists():
            old = read_json(target / 'spec.json')
            if {k:v for k,v in old.items() if k != 'created'} != {k:v for k,v in spec.items() if k != 'created'}:
                raise ConfigError('同一实验 ID 的冻结比较配置已变化')
        else:
            experiment.write_spec(target, spec)
            write_json(target / 'state.json', {'status': 'queued', 'reason': '实验已登记，等待逐题结果',
                'metrics': {'attempted': len(pack['rows']), 'pairs': 0, 'failures': 0}})
    return target


def complete(target):
    """Recompute from the case journal; producers cannot supply their own totals."""
    spec, _ = record_contract.require_registered(target)
    if not spec.get('comparison'):
        raise ConfigError('需要专项比较实验')
    rows = record_contract.planned_rows(spec)
    metrics = gates.confirmed_metrics(target, rows, spec)
    experiment.finish(target, metrics, {'verdict': 'diagnostic_complete',
        'reason': '专项比较完成；此记录不用于基线晋级或固定准入'})
    return metrics


def _inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ConfigError('历史证据必须是当前实例内的实际文件')
    return path


def _definition(item):
    if 'id' in item:
        return {key: item[key] for key in ('id', 'baseline', 'candidate')}
    return {'id': item['arm'] + '-' + item['recipe'],
            'baseline': {'arm': item['arm'], 'recipe': item['control'], 'policy': 'hybrid'},
            'candidate': {'arm': item['arm'], 'recipe': item['recipe'], 'policy': 'hybrid'}}


def _experiment_id(study, comparison):
    suffix = hashlib.sha256(comparison.encode()).hexdigest()[:10]
    slug = re.sub(r'[^A-Za-z0-9_-]', '_', comparison)[:100]
    return valid_name(f'comparison-{study}-{slug}-{suffix}')


def legacy_plan(study):
    """Read-only validation; hashes prove preservation, not a new training audit."""
    root = versions.PRIVATE
    source = root / 'judge_training' / valid_name(study)
    spec_path = _inside(source / 'spec.json', root)
    spec = read_json(spec_path)
    if spec.get('dataset') != 'development' or spec.get('adoption_allowed') is not False:
        raise ConfigError('只允许补录明确禁止晋级的历史开发比较')
    state = read_json(source / 'state.json')
    finished = state.get('status') == 'finished'
    if state.get('status') not in ('finished', 'paused', 'stopped', 'interrupted'):
        raise ConfigError('历史任务仍可能运行，不能导入变化中的结果')
    cursor, seen = source, set()
    while not (cursor / 'evaluation_source.json').is_file():
        if cursor in seen:
            raise ConfigError('历史来源链循环')
        seen.add(cursor)
        cursor = _inside(Path(read_json(cursor / 'spec.json')['source']) / 'spec.json', root).parent
    evidence_path = _inside(cursor / 'evaluation_source.json', root)
    evidence = read_json(evidence_path)
    previous = _inside(Path(evidence['source_experiment']) / 'spec.json', root)
    pack_ref = read_json(previous)['pack_ref']
    pack_path = _inside(root / 'judge_eval' / valid_name(pack_ref) / 'pack.json', root)
    if sha256_file(pack_path) != evidence['pack_sha256']:
        raise ConfigError('原始评估包哈希已变化')
    pack = read_json(pack_path)
    rows = pack['rows']
    source_files = [spec_path, source / 'state.json', evidence_path, previous, pack_path]
    if finished:
        acceptance_path = _inside(source / 'acceptance.json', root)
        accepted = read_json(acceptance_path)
        if accepted.get('status') != 'passed' or not accepted.get('evidence_hashes'):
            raise ConfigError('缺少原始独立验收及证据清单')
        for name, digest in accepted['evidence_hashes'].items():
            path = _inside(name, root.parent.parent)
            if sha256_file(path) != digest:
                raise ConfigError(f'原始验收证据已变化: {path.name}')
        results_path = _inside(source / 'results.json', root)
        results = read_json(results_path)['comparisons']
        accepted_by_id = {_definition(r)['id']: r for r in accepted['comparisons']}
        for result in results:
            audited = accepted_by_id.get(_definition(result)['id'], {})
            if result.get('metrics') != audited.get('metrics'):
                raise ConfigError('结果汇总与原始独立验收不一致')
        source_files += [acceptance_path, results_path]
    else:
        results = spec['comparisons']
    definitions = [_definition(item) for item in results]
    if not definitions or len({c['id'] for c in definitions}) != len(definitions):
        raise ConfigError('比较清单为空或重复')
    recipes = {r['id']: r for r in read_json(cursor / 'spec.json').get('recipes', [])}
    recipes.update({r['id']: r for r in spec['recipes']})
    entries = []
    for definition in definitions:
        comparison = {side: {**definition[side], 'parameters': recipes.get(definition[side]['recipe'], {})}
                      for side in ('baseline', 'candidate')}
        source_cases = source / 'comparisons' / definition['id'] / 'cases.jsonl'
        result = source_cases.with_name('result.json')
        files = list(source_files)
        metrics = None
        if finished:
            _inside(source_cases, root)
            _inside(result, root)
            if accepted['evidence_hashes'].get(str(source_cases)) != sha256_file(source_cases):
                raise ConfigError('逐题结果不属于原始验收清单')
            metrics = gates.confirmed_metrics(source_cases.parent, rows, {**spec, 'kind': 'judge_eval'})
            if metrics != read_json(result)['metrics']:
                raise ConfigError('逐题重算与原结果不一致')
            files += [source_cases, result]
        entries.append({'id': _experiment_id(study, definition['id']), 'comparison': comparison,
            'comparison_id': definition['id'], 'cases': source_cases if finished else None,
            'metrics': metrics, 'source_hashes': {str(p.relative_to(root)): sha256_file(p) for p in files}})
    return {'study': study, 'spec': spec, 'state': state, 'pack_ref': pack_ref, 'entries': entries}


def import_legacy(study, *, apply=False, comparison_id=None):
    plan = legacy_plan(study)
    if comparison_id is not None:
        plan['entries'] = [e for e in plan['entries'] if e['comparison_id'] == comparison_id]
        if not plan['entries']:
            raise ConfigError('历史比较 ID 不存在')
    if not apply:
        return {'study': study, 'comparisons': len(plan['entries']), 'source_status': plan['state']['status']}
    paths = []
    for entry in plan['entries']:
        def verify_source():
            for name, digest in entry['source_hashes'].items():
                if sha256_file(_inside(versions.PRIVATE / name, versions.PRIVATE)) != digest:
                    raise ConfigError('补录过程中原始证据发生变化')
        verify_source()
        provenance = {'mode': 'historical_import', 'study': study, 'comparison_id': entry['comparison_id'],
            'source_hashes': entry['source_hashes'], 'original_executor_certified': False}
        created = plan['spec'].get('created') or plan['state'].get('updated_at')
        if not isinstance(created, str):
            created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created)) if created else None
        target = register(entry['id'], data_ref=plan['spec']['data_ref'], pack_ref=plan['pack_ref'],
            comparison=entry['comparison'], change=f"{study} · {entry['comparison_id']}",
            protocol=plan['spec']['protocol'], created=created, provenance=provenance)
        with file_lock(target / '.run.lock', blocking=False):
            if entry['cases']:
                if experiment.state_of(target).get('status') != 'finished':
                    with record_contract.writer(target) as append:
                        for line in entry['cases'].read_text().splitlines():
                            if line.strip():
                                append(json.loads(line))
                    verify_source()
                    complete(target)
                record_contract.validate_completion(target, entry['metrics'])
                from .protocol import summarize_final_records
                if summarize_final_records(target / 'cases.jsonl')[0] != summarize_final_records(entry['cases'])[0]:
                    raise ConfigError('补录逐题记录与原始记录不一致')
            elif experiment.state_of(target).get('status') == 'queued':
                write_json(target / 'state.json', {'status': plan['state']['status'],
                    'reason': '原任务已暂停或停止；尚无此项完整评测结果',
                    'metrics': {'attempted': len(read_json(target / 'spec.json')['record_contract']['case_ids']),
                                'pairs': 0, 'failures': 0}})
        paths.append(target)
    return {'study': study, 'registered': len(paths), 'experiment_ids': [p.name for p in paths]}
