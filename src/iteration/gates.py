"""分层门槛：复核逐题证据；固定准入只采用生产→候选的直接开发对比。"""
from __future__ import annotations

import json
from math import ceil
from pathlib import Path

from ..config import ConfigError, sha256_file
from . import data_guard, datasets, protocol, versions


def diagnostic_reason(spec: dict) -> str:
    """One eligibility rule for promotion, evidence reuse and read-only reports."""
    if spec.get('comparison') or spec.get('adoption_allowed') is False:
        return '专项比较只记录候选间的开发结果，不能用于基线晋级或固定准入。'
    if spec.get('smoke'):
        return '连接检查（冒烟）不参与版本晋升。'
    if spec.get('purpose') == 'gen_optimization':
        return 'Gen 优化题用于调整策略，不参与版本晋升；须另做开发验证。'
    if spec.get('material_repair', {}).get('adoption_allowed') is False:
        return '本轮为重训控制实验，只比较两种重训方案，不参与版本晋升；实验对照不是已晋级基线。'
    if spec.get('lr_retrain', {}).get('promotion_eligible') is False:
        return '该重训控制实验已标记为不可晋升；须另与当前已晋级基线直接比较并通过来源核验。'
    if spec.get('source_audit', {}).get('promotion_eligible') is False:
        return '学习材料尚未通过边界核验，本轮只用于诊断，不能晋升。'
    return ''


def require_promotable(spec: dict) -> None:
    reason = diagnostic_reason(spec)
    if reason:
        raise ConfigError(reason)


def verify_learning(spec: dict) -> None:
    """Use the same verified historical-code paths at evaluation and adoption."""
    from . import gbdt_evidence, learning_guard, pack_transport, runtime
    with pack_transport.archived_gbdt_paths(gbdt_evidence, runtime):
        learning_guard.verify(spec)


def bind_learning(spec: dict) -> None:
    """New experiments use the same verified archive resolver as execution."""
    from . import gbdt_evidence, learning_guard, pack_transport, runtime
    with pack_transport.archived_gbdt_paths(gbdt_evidence, runtime):
        learning_guard.bind(spec)


def judge_templates(spec: dict) -> list[Path]:
    """Include the effective pairwise prompt even when it lives outside a version."""
    from ..judge.judge import DEFAULT_TEMPLATE_PATH
    from .learning_guard import require_default_judge_template
    refs = ([spec['baseline_ref'], spec['candidate_ref']] if spec['kind'] == 'judge_eval'
            else [spec['judge_ref']])
    paths = []
    for ref in refs:
        info = versions.judge_dir(ref)
        from ..judge import normalize_mode
        if normalize_mode(info['config'].get('mode', 'pairwise_llm')) != 'pairwise_llm':
            continue
        path = info['dir'] / 'prompt.md'
        if not path.is_file():
            path = DEFAULT_TEMPLATE_PATH
            require_default_judge_template(path)
        paths.append(path)
    return paths


def materials(spec: dict) -> dict:
    """冻结参与比较的模型内容和评分实现，ID 相同不代表条件相同。"""
    judge = spec['kind'] == 'judge_eval'
    dirs = [(versions.judge_dir(ref)['dir'] if judge else versions.generator_dir(ref))
            for ref in (spec['baseline_ref'], spec['candidate_ref'])]
    if judge:
        pack = json.loads((versions.PRIVATE / 'judge_eval' / spec['pack_ref'] / 'pack.json').read_text())
        dirs.append(versions.generator_dir(pack['c0_gen_version']))
    else:
        dirs.append(versions.judge_dir(spec['judge_ref'])['dir'])
    source = Path(__file__).parents[1]
    replay_inputs = None
    if spec.get('saved_draw_replay'):
        binding = spec['saved_draw_replay']
        path = Path(binding['path'])
        if spec['dataset'] != 'development' or sha256_file(path) != binding['sha256']:
            raise ConfigError('Saved development observation manifest changed')
        replay_inputs = {str(path): sha256_file(path), **{
            name: sha256_file(Path(name)) for name in json.loads(path.read_text())['inputs']}}
    result = {'versions': {d.name: {str(p.relative_to(d)): sha256_file(p)
                                 for p in sorted(d.rglob('*')) if p.is_file()
                                 and p.relative_to(d) != Path('meta.json')}
                         for d in dirs},
            'judge_templates': {str(p.resolve()): sha256_file(p) for p in judge_templates(spec)},
            'implementation': {name: sha256_file(source / name) for name in (
                'judge/judge.py', 'judge/corrected.py', 'generator/generator.py', 'generator/few_shot.py',
                'generator/history.py', 'llm.py', 'iteration/runner.py',
                'judge/gbdt.py', 'judge/fusion.py', 'judge/lr_retrain.py', 'iteration/draw_replay.py',
                'judge/embedding.py', 'judge/embedding_runtime.py',
                'iteration/training_evidence.py')},
            'protocol': {key: spec['protocol'].get(key) for key in
                         ('flip_extra_rounds', 'force_reply', 'max_failure_rate')}}
    if replay_inputs is not None:
        result['saved_draw_inputs'] = replay_inputs
    return result


def development_rows(spec: dict, *, historical_directory: Path | None = None) -> list[dict]:
    data = versions.data_version_dir(spec['data_ref'])
    if spec['kind'] == 'judge_eval':
        path = versions.PRIVATE / 'judge_eval' / spec['pack_ref'] / 'pack.json'
        if sha256_file(path) != spec.get('pack_sha256'):
            raise ConfigError('开发包内容已变化，禁止复用')
        pack = json.loads(path.read_text())
        if pack.get('data_ref', spec['data_ref']) != spec['data_ref']:
            raise ConfigError('开发包数据版本不一致')
        if pack['c0_gen_version'] != versions.load_pointers()['production_gen']:
            raise ConfigError('生产生成器已前进，开发包已过期')
        datasets.assert_pack(data, pack, 'judge_development')
        if spec.get('purpose_snapshot', {}) != datasets.snapshot(data):
            raise ConfigError('数据用途快照已变化')
        return pack['rows']
    rows = [json.loads(line) for line in datasets.case_path(data, 'development').read_text().splitlines() if line]
    configs = [versions.load_generator(spec[key])['config'] for key in ('baseline_ref', 'candidate_ref')]
    frozen = spec.get('data_snapshot')
    current = data_guard.generation_snapshot(rows, data, configs)
    if frozen != current:
        # An adopted development receipt may outlive its executor. Data still
        # must match exactly; code must match the authenticated old runtime.
        if (historical_directory is None or not isinstance(frozen, dict)
                or {k: v for k, v in frozen.items() if k != 'implementation_sha256'}
                != {k: v for k, v in current.items() if k != 'implementation_sha256'}):
            raise ConfigError('开发数据指纹已变化，禁止复用')
        from .runtime import verify_historical_implementation
        verify_historical_implementation(historical_directory, frozen.get('implementation_sha256'))
    return rows


def confirmed_metrics(directory: Path, rows: list[dict], spec: dict) -> dict:
    """原始初测与两侧独立补验齐全后，重新汇总确认净胜。"""
    records, _ = protocol.summarize_final_records(directory / 'cases.jsonl')
    ids = [str(row['case_id']) for row in rows]
    if not ids or len(set(ids)) != len(ids) or set(records) != set(ids):
        raise ConfigError('正式开发逐题记录不完整或样本不一致')
    failures = sum(records[cid].get('status') != 'ok' for cid in ids)
    failure_rate = failures / len(ids)
    if failure_rate >= float(spec['protocol'].get('max_failure_rate', 0.01)):
        raise ConfigError('正式开发失败率达到冻结上限，不能晋级或进入固定轮')
    extra = int(spec['protocol']['flip_extra_rounds'])
    if extra < 2:
        raise ConfigError('缺少既定独立补验要求（每侧至少两轮）')
    judge = spec['kind'] == 'judge_eval'
    wins = losses = contested = identified_b = identified_c = 0
    for cid in ids:
        row = records[cid]
        if row.get('status') != 'ok':
            continue
        b = row.get('baseline_correct' if judge else 'identified_baseline')
        c = row.get('candidate_correct' if judge else 'identified_candidate')
        if type(b) is not bool or type(c) is not bool:
            raise ConfigError('缺少有效初测判定')
        identified_b += b
        identified_c += c
        if b == c:
            continue
        verified = row if judge and row.get('flip_verified') is True else row.get('flip_verified')
        if not isinstance(verified, dict):
            raise ConfigError('初测分歧未完成独立补验')
        final = []
        for side, initial in (('baseline', b), ('candidate', c)):
            votes = verified.get(side + '_votes', [])
            if (len(votes) != extra + 1 or any(type(v) is not bool for v in votes)
                    or votes[0] != initial):
                raise ConfigError('独立补验票数不完整或初测不一致')
            result = sum(votes) > len(votes) / 2
            if verified.get(side + '_identified_final') is not result:
                raise ConfigError('补验多数票与最终判定不一致')
            final.append(result)
        if final[0] == final[1]:
            contested += 1
        elif (final[1] if judge else final[0]):
            wins += 1
        else:
            losses += 1
    return dict(pairs=len(ids) - failures, attempted=len(ids), failures=failures, failure_rate=failure_rate,
                identified_baseline=identified_b, identified_candidate=identified_c,
                wins_confirmed=wins, losses_confirmed=losses, contested=contested,
                net_win_confirmed=wins - losses)


def evidence(directory: Path) -> tuple[dict, dict]:
    from . import experiment
    spec, state = experiment.spec_of(directory), experiment.state_of(directory)
    require_promotable(spec)
    verify_learning(spec)
    if (spec['dataset'] != 'development' or spec.get('smoke') or
            spec.get('purpose') == 'gen_optimization' or state.get('status') != 'finished'):
        raise ConfigError('需要已完成的正式开发对比')
    frozen = spec.get('evaluation_materials')
    if frozen is None or frozen != materials(spec):
        raise ConfigError('模型或评分条件缺少冻结证据，或已发生变化')
    data = versions.data_version_dir(spec['data_ref'])
    assets = [versions.PRIVATE / ('judges' if ref.startswith('j-') else 'generators') / ref
              for ref in frozen['versions']]
    if datasets.static_sources(data, assets, 'development').get('promotion_eligible') is False:
        raise ConfigError('当前学习来源核验不通过')
    metrics = confirmed_metrics(directory, development_rows(spec), spec)
    for key, value in metrics.items():
        if key in state.get('metrics', {}) and state['metrics'][key] != value:
            raise ConfigError(f'逐题汇总与已存指标不一致: {key}')
    return spec, metrics


def require_development(directory: Path) -> None:
    spec, metrics = evidence(directory)
    if protocol.decide(metrics, 'development', spec['protocol'])['verdict'] != 'merge_to_iteration_baseline':
        raise ConfigError('完整补验后确认净胜未达到冻结门槛，不能晋级开发版')


def fixed_entry(kind: str, data_ref: str, baseline_ref: str, candidate_ref: str,
                proto: dict, *, judge_ref: str | None = None,
                development_pack: str | None = None, evidence_id: str | None = None) -> dict:
    """找同条件的直接比较；不合并不同端点的净胜，不读取固定答案。"""
    from . import experiment
    ptr = versions.load_pointers()
    production = 'production_judge' if kind == 'judge_eval' else 'production_gen'
    if data_ref != ptr['data'] or baseline_ref != ptr[production]:
        raise ConfigError('数据或生产基线已前进，固定准入证据已过期')
    if kind == 'gen_ab' and judge_ref != ptr['production_judge']:
        raise ConfigError('生产 Judge 已前进，固定准入证据已过期')
    paths = sorted((versions.PRIVATE / 'experiments').glob('*/spec.json'),
                   key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    # Judge 必须沿用最新候选的开发包，不能从历史不同样本挑选最高收益。
    candidates = []
    for path in paths:
        spec = experiment.spec_of(path.parent)
        if (spec.get('kind') == kind and spec.get('dataset') == 'development' and
                not spec.get('smoke') and spec.get('data_ref') == data_ref and
                spec.get('candidate_ref') == candidate_ref):
            candidates.append((path.parent, spec))
    if kind == 'judge_eval' and development_pack is None and candidates:
        development_pack = candidates[0][1].get('pack_ref')
    reasons = []
    for directory, spec in candidates:
        if evidence_id and directory.name != evidence_id:
            continue
        if spec['baseline_ref'] != baseline_ref:
            continue
        if kind == 'judge_eval' and spec.get('pack_ref') != development_pack:
            continue
        if kind == 'gen_ab' and spec.get('judge_ref') != judge_ref:
            continue
        if any(spec['protocol'].get(k) != proto.get(k) for k in
               ('flip_extra_rounds', 'force_reply', 'max_failure_rate')):
            continue
        try:
            _, metrics = evidence(directory)
        except ConfigError as exc:
            reasons.append(str(exc))
            continue
        need = max(1, ceil(proto['fixed_entry_min_net_win_rate'] * metrics['pairs']))
        return {'eligible': metrics['net_win_confirmed'] >= need,
                'experiment_id': directory.name, 'baseline_ref': baseline_ref,
                'candidate_ref': candidate_ref, 'data_ref': data_ref,
                'development_pack': development_pack, 'pairs': metrics['pairs'],
                'net_win_confirmed': metrics['net_win_confirmed'], 'required_net_win': need,
                'evidence_sha256': {name: sha256_file(directory / name)
                                    for name in ('spec.json', 'state.json', 'cases.jsonl')}}
    detail = '; '.join(dict.fromkeys(reasons))
    raise ConfigError('缺少同条件的生产版→最新开发版直接比较；请先在原开发集运行 --against-production'
                      + (f'（{detail}）' if detail else ''))


def require_fixed_entry(*args, **kwargs) -> dict:
    result = fixed_entry(*args, **kwargs)
    if not result['eligible']:
        raise ConfigError(f"开发版对生产版确认净胜 {result['net_win_confirmed']:+d}/{result['pairs']}，"
                          f"未达到固定轮门槛 +{result['required_net_win']}；保留开发版继续迭代")
    return result


def reusable_comparison(*args, **kwargs) -> Path | None:
    """相同条件已有完整直接比较时返回原产物，低收益也复用，不重抽结果。"""
    try:
        result = fixed_entry(*args, **kwargs)
    except ConfigError:
        return None
    return versions.PRIVATE / 'experiments' / result['experiment_id']


def verify_fixed_receipt(spec: dict) -> None:
    receipt = spec.get('fixed_entry')
    if not receipt:
        raise ConfigError('固定轮缺少直接开发对比准入凭证')
    current = require_fixed_entry(spec['kind'], spec['data_ref'], spec['baseline_ref'],
        spec['candidate_ref'], spec['protocol'], judge_ref=spec.get('judge_ref'),
        development_pack=receipt['development_pack'], evidence_id=receipt['experiment_id'])
    if current != receipt:
        raise ConfigError('固定准入证据已变化，禁止推全')
