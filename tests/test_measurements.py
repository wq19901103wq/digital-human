"""按侧测量库（P1 地基）回归测试。设计：docs/DESIGN-SIDE-MEASUREMENT-BANK.md。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.iteration import measurements, versions


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False), encoding='utf-8')


def _gen_experiment(instance: Path, name='exp-1', judge_cfg=None, base_cfg=None,
                    cand_cfg=None, data_ref='d-1', dataset='development'):
    exp = instance / 'experiments' / name
    _write(exp / 'spec.json', {
        'id': name, 'kind': 'gen_ab', 'dataset': dataset, 'purpose': dataset,
        'baseline_ref': 'g-base', 'candidate_ref': 'g-cand', 'judge_ref': 'j-1',
        'data_ref': data_ref, 'smoke': False,
        'protocol': {'flip_extra_rounds': 2, 'force_reply': True},
    })
    _write(exp / 'state.json', {'status': 'finished', 'verdict': 'merge_to_iteration_baseline',
                                'metrics': {'pairs': 2}})
    rows = [
        {'case_id': 'c1', 'status': 'ok', 'identified_baseline': True, 'identified_candidate': False,
         'baseline_replies': ['甲'], 'candidate_replies': ['乙'], 'trace_ref': 't1',
         'flip_verified': {'baseline_votes': [True, True, False], 'candidate_votes': [False, False, True]}},
        {'case_id': 'c2', 'status': 'ok', 'identified_baseline': False, 'identified_candidate': False,
         'baseline_replies': ['丙'], 'candidate_replies': ['丁'], 'trace_ref': 't2',
         'flip_verified': {'baseline_votes': [False, False, False], 'candidate_votes': [False, False, False]}},
    ]
    _write(exp / 'cases.jsonl', '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
    _write(instance / 'generators/g-base/config.json', base_cfg or {'llm': {'model': 'm0'}})
    _write(instance / 'generators/g-base/persona.md', '人格')
    (instance / 'generators/g-base/scenarios').mkdir(parents=True, exist_ok=True)
    _write(instance / 'generators/g-cand/config.json', cand_cfg or {'llm': {'model': 'm1'}})
    _write(instance / 'generators/g-cand/persona.md', '人格')
    (instance / 'generators/g-cand/scenarios').mkdir(parents=True, exist_ok=True)
    _write(instance / 'judges/j-1/config.json', judge_cfg or {'mode': 'pairwise_llm', 'llm': {'model': 'judge'}})
    _write(instance / 'judges/j-1/prompt.md', '提示词')
    _write(instance / f'data/{data_ref}/manifest.json', {'id': data_ref})
    return exp


@pytest.fixture()
def instance(tmp_path, monkeypatch):
    versions.switch_instance('demo')
    monkeypatch.setattr(versions, 'PRIVATE', tmp_path / 'instances' / 'demo')
    monkeypatch.setattr(versions, 'DATA_ROOT', versions.PRIVATE / 'data')
    monkeypatch.setattr(versions, 'GEN_ROOT', versions.PRIVATE / 'generators')
    monkeypatch.setattr(versions, 'JUDGE_ROOT', versions.PRIVATE / 'judges')
    return versions.PRIVATE


def test_parse_pair_row_yields_two_side_measurements(instance):
    exp = _gen_experiment(instance)
    rows = list(measurements.iter_side_measurements(exp))
    # 2 题 × 2 侧 × (初测 + 2 补验) = 12
    assert len(rows) == 12
    assert {r['source_exp'] for r in rows} == {'exp-1'}
    assert {r['source_line'] for r in rows} == {1, 2}
    initial = [r for r in rows if r['round_kind'] == 'initial']
    assert len(initial) == 4
    c1_base = next(r for r in initial if r['source_line'] == 1 and r['side_ref'] == 'g-base')
    assert c1_base['verdict'] == 1 and c1_base['round_no'] == 0


def test_round_fp_groups_paired_sides_and_reparse_stable(instance):
    """round_fp 标识配对轮次（同题同轮两侧共享），不是行级唯一。"""
    exp = _gen_experiment(instance)
    rows_a = list(measurements.iter_side_measurements(exp))
    rows_b = list(measurements.iter_side_measurements(exp))
    assert {r['round_fp'] for r in rows_a} == {r['round_fp'] for r in rows_b}
    # 2 题 × (初测 + 2 补验) = 6 个配对轮次；每轮两侧共享 round_fp
    assert len({r['round_fp'] for r in rows_a}) == 6
    from collections import Counter
    per_round = Counter(r['round_fp'] for r in rows_a)
    assert set(per_round.values()) == {2}  # 每轮恰好两侧两行


def test_fingerprint_sensitivity(instance):
    exp = _gen_experiment(instance)
    fp_before = {r['judge_fp'] for r in measurements.iter_side_measurements(exp)}
    _write(instance / 'judges/j-1/config.json', {'mode': 'pairwise_llm', 'llm': {'model': 'OTHER'}})
    fp_after = {r['judge_fp'] for r in measurements.iter_side_measurements(exp)}
    assert fp_before.isdisjoint(fp_after)
    side_before = {r['side_fp'] for r in measurements.iter_side_measurements(exp)}
    _write(instance / 'generators/g-cand/config.json', {'llm': {'model': 'changed'}})
    side_after = {r['side_fp'] for r in measurements.iter_side_measurements(exp)}
    assert side_before != side_after


def test_record_idempotent_and_conflict_rejected(instance):
    exp = _gen_experiment(instance)
    db = measurements.connect(instance)
    assert measurements.record_experiment(exp, db) == 12
    assert measurements.record_experiment(exp, db) == 0  # 幂等
    db.execute("UPDATE measurements SET verdict = 1 - verdict WHERE source_line = 1 AND round_no = 0")
    db.commit()
    with pytest.raises(measurements.MeasurementConflict):
        measurements.record_experiment(exp, db)
    db.close()


def test_audit_detects_tampered_bank_field(instance):
    exp = _gen_experiment(instance)
    measurements.record_experiment(exp)
    db = measurements.connect(instance)
    row = db.execute("SELECT * FROM measurements WHERE source_line = 1 LIMIT 1").fetchone()
    db.execute("UPDATE measurements SET pair_fp = 'forged' WHERE source_line = 1")
    db.commit()
    row = db.execute("SELECT * FROM measurements WHERE source_line = 1 LIMIT 1").fetchone()
    issues = measurements.audit_row(instance, row)
    assert any('pair_fp' in i for i in issues)
    db.close()


def test_audit_passes_clean_rows(instance):
    exp = _gen_experiment(instance)
    measurements.record_experiment(exp)
    db = measurements.connect(instance)
    bad = 0
    for row in db.execute('SELECT * FROM measurements'):
        bad += bool(measurements.audit_row(instance, row))
    assert bad == 0
    db.close()


def test_safely_never_raises_on_broken_experiment(instance):
    exp = _gen_experiment(instance)
    (exp / 'cases.jsonl').unlink()
    assert measurements.record_experiment_safely(exp) == 0


def test_judge_eval_rows_parse(instance):
    exp = instance / 'experiments' / 'judge-exp'
    _write(exp / 'spec.json', {
        'id': 'judge-exp', 'kind': 'judge_eval', 'dataset': 'development',
        'baseline_ref': 'j-base', 'candidate_ref': 'j-cand', 'judge_ref': 'j-base',
        'data_ref': 'd-1', 'pack_ref': 'pack-x', 'pack_sha256': 'abc',
        'protocol': {'flip_extra_rounds': 2}, 'smoke': False})
    _write(exp / 'state.json', {'status': 'finished', 'verdict': 'observe', 'metrics': {}})
    _write(exp / 'cases.jsonl', json.dumps({
        'case_id': 'k1', 'status': 'ok', 'baseline_correct': True, 'candidate_correct': False,
        'flip_verified': {'baseline_votes': [True, True, True], 'candidate_votes': [False, False, True]}}))
    _write(instance / 'judges/j-base/config.json', {'mode': 'corrected_pairwise', 'llm': {'model': 'a'}})
    _write(instance / 'judges/j-base/prompt.md', 'p')
    _write(instance / 'judges/j-cand/config.json', {'mode': 'corrected_pairwise', 'llm': {'model': 'b'}})
    _write(instance / 'judges/j-cand/prompt.md', 'p')
    rows = list(measurements.iter_side_measurements(exp))
    assert len(rows) == 6  # 1 题 × 2 侧 × 3 轮
    assert rows[0]['dataset_role'] == 'pack:pack-x'
    assert rows[0]['judge_fp'].startswith('gt') or len(rows[0]['judge_fp']) == 64


def test_draw_nonce_in_row_wins_over_derivation(instance):
    """行内 draw_nonce 优先（P2 重放逐字继承的来源）。"""
    exp = _gen_experiment(instance)
    cases = [json.loads(l) for l in (exp / 'cases.jsonl').read_text().splitlines()]
    for row in cases:
        row['draw_nonce'] = 'fixed-nonce-0'
        row['draw_nonce_r1'] = 'fixed-nonce-1'
        row['draw_nonce_r2'] = 'fixed-nonce-2'
    (exp / 'cases.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in cases))
    rows = list(measurements.iter_side_measurements(exp))
    assert {r['draw_nonce'] for r in rows if r['round_no'] == 0} == {'fixed-nonce-0'}
    assert {r['draw_nonce'] for r in rows if r['round_no'] == 2} == {'fixed-nonce-2'}


def test_flip_first_vote_mismatch_drops_verify_votes(instance):
    """flip 首票与初测矛盾：只保留初测，不采信 flip 票。"""
    exp = _gen_experiment(instance)
    cases = [json.loads(l) for l in (exp / 'cases.jsonl').read_text().splitlines()]
    cases[0]['flip_verified']['baseline_votes'] = [False, True, False]  # 首票≠初测(True)
    (exp / 'cases.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in cases))
    rows = list(measurements.iter_side_measurements(exp))
    line1_base = [r for r in rows if r['source_line'] == 1 and r['side_ref'] == 'g-base']
    assert len(line1_base) == 1 and line1_base[0]['round_kind'] == 'initial'


def test_empty_case_id_row_skipped(instance):
    exp = _gen_experiment(instance)
    cases = [json.loads(l) for l in (exp / 'cases.jsonl').read_text().splitlines()]
    cases[1]['case_id'] = ''
    (exp / 'cases.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in cases))
    rows = list(measurements.iter_side_measurements(exp))
    assert {r['source_line'] for r in rows} == {1}


def test_audit_detects_side_fp_tamper(instance):
    exp = _gen_experiment(instance)
    measurements.record_experiment(exp)
    db = measurements.connect(instance)
    row = db.execute('SELECT * FROM measurements LIMIT 1').fetchone()
    key = (row[0], row[1], row[2], row[9])
    db.execute('UPDATE measurements SET side_fp=? WHERE source_exp=? AND source_line=? AND side_fp=? AND round_fp=?',
               ('forged', *key))
    db.commit()
    tampered = db.execute('SELECT * FROM measurements WHERE source_exp=? AND source_line=? AND round_fp=? AND verdict=?',
                          (key[0], key[1], key[3], row[15])).fetchone()
    issues = measurements.audit_row(instance, tampered)
    assert any('side_fp' in i or '来源行重算无此侧' in i for i in issues)
    db.close()


def test_origin_conflict_rejected_within_batch(instance, monkeypatch):
    """同 round_fp 的归属必须一致：批内冲突与库内冲突都拒绝。"""
    exp = _gen_experiment(instance)
    rows = list(measurements.iter_side_measurements(exp))
    measurements.record_experiment(exp)  # 先正常入库
    # 批内冲突：同一 round_fp 混入不同 origin
    forged = [dict(r) for r in rows]
    forged[0] = dict(forged[0], origin_exp='other-exp')
    forged[1] = dict(forged[1], round_fp=forged[0]['round_fp'])
    monkeypatch.setattr(measurements, 'iter_side_measurements', lambda d: forged)
    with pytest.raises(measurements.MeasurementConflict):
        measurements.record_experiment(exp)
    monkeypatch.undo()
    # 库内冲突：同 round_fp 但 origin 与库内已存行不同
    rows2 = list(measurements.iter_side_measurements(exp))
    rows2[0] = dict(rows2[0], origin_exp='evil')
    monkeypatch.setattr(measurements, 'iter_side_measurements', lambda d: rows2[:1])
    with pytest.raises(measurements.MeasurementConflict):
        measurements.record_experiment(exp)


def test_unfinished_source_never_replayed(instance):
    """未完成实验的测量带 running 标记，且不参与重放规划（source_status 语义）。"""
    exp = _gen_experiment(instance)
    state = json.loads((instance / 'experiments/exp-1/state.json').read_text())
    _write(instance / 'experiments/exp-1/state.json', {**state, 'status': 'running'})
    rows = list(measurements.iter_side_measurements(instance / 'experiments/exp-1'))
    assert rows and all(r['source_status'] == 'running' for r in rows)
    spec = json.loads((instance / 'experiments/exp-1/spec.json').read_text())
    plan = measurements.plan_replay(instance, spec, ['c1', 'c2'])
    assert plan['replay'] == {} and plan['run'] == ['c1', 'c2']


def test_replay_skips_serialization_mismatch(instance, monkeypatch):
    """来源行无法被 writer 序列化逐字节还原时，该题退回执行（评审 P1-2）。"""
    exp = _gen_experiment(instance, name='exp-src')
    measurements.record_experiment(exp)
    target = _gen_experiment(instance, name='exp-dst')
    (target / 'cases.jsonl').unlink()  # 模拟新建实验（无断点）
    (target / 'state.json').unlink()
    _write(instance / 'data/d-1/dev_pool.jsonl', '\n'.join(json.dumps({'case_id': c}) for c in ('c1', 'c2')))
    spec = json.loads((target / 'spec.json').read_text())
    # 篡改来源行的序列化形态（键序不同 → dumps 结果不同）
    cases_path = instance / 'experiments/exp-src/cases.jsonl'
    row = json.loads(cases_path.read_text().splitlines()[0])
    reordered = json.dumps(row, ensure_ascii=True)  # 非 ASCII 转义，round-trip 字节不同
    lines = cases_path.read_text().splitlines()
    lines[0] = reordered
    cases_path.write_text('\n'.join(lines))
    result = measurements.apply_replay(target)
    assert result['replayed'] <= 1  # 被篡改的题回退执行
    manifest = json.loads((target / 'replay_sources.json').read_text())
    assert manifest.get('dropped_serialization_mismatch', 0) >= 1
