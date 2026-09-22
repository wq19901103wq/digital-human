"""同对复用（P2）回归测试：planner 重放 + runner 只跑缺的题，gates 不变。

设计：docs/DESIGN-SIDE-MEASUREMENT-BANK.md §6.1/§6.2/§7。重放纪律：
整题逐字节拷贝来源行；纯重放不产生新测量行；fixed_test 与 no_replay 排除。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.iteration import experiment, measurements, runner, versions
from test_iteration import _write, _use_small_settings, priv, _case  # noqa: F401
from leakage_support import isolated_legacy_provenance  # noqa: F401

pytestmark = pytest.mark.usefixtures('isolated_legacy_provenance')


class _CountingScorer:
    """按题号给确定性判定；记录调用。"""

    def __init__(self, is_candidate, calls):
        self.is_candidate = is_candidate
        self.calls = calls

    def is_ai(self, case, replies):
        self.calls.append((case['case_id'], self.is_candidate))
        wrong = {'c0', 'c2'} if not self.is_candidate else {'c1'}
        return case['case_id'] in wrong


def _pack(priv: Path, name='pack-replay'):
    rows = [{**_case(f'c{i}'), 'ai_replies': ['AI']} for i in range(4)]
    _write(priv / 'judge_eval' / name / 'pack.json', {'c0_gen_version': 'g-0001', 'rows': rows})
    return name


def _make_exp(priv, pack, calls, exp_id, candidate_ref=None):
    exp = experiment.create_judge_eval_experiment(
        'development', pack, {'llm': {'model': 'm9'}}, change='换模型',
        exp_id=exp_id, candidate_ref=candidate_ref)
    monkey = _CountingScorer
    runner.build_judge = lambda settings, info: monkey(info['id'] != 'j-0001', calls)
    return exp


@pytest.fixture
def scorer_stub(monkeypatch):
    def apply(calls):
        monkeypatch.setattr(runner, 'build_judge',
                            lambda settings, info: _CountingScorer(info['id'] != 'j-0001', calls))
    return apply


def test_same_pair_full_replay_zero_calls(priv, scorer_stub, monkeypatch):  # noqa: F811
    _use_small_settings(monkeypatch)
    pack = _pack(priv)
    calls_a = []
    exp_a = experiment.create_judge_eval_experiment(
        'development', pack, {'llm': {'model': 'm9'}}, change='换模型', exp_id='judge-a')
    scorer_stub(calls_a)
    runner.run_judge_experiment(exp_a, workers=1)
    state_a = experiment.state_of(exp_a)
    assert state_a['status'] == 'finished'
    n_calls_a = len(calls_a)

    exp_b = experiment.create_judge_eval_experiment(
        'development', pack, {}, change='换模型', exp_id='judge-b',
        candidate_ref=experiment.spec_of(exp_a)['candidate_ref'])
    calls_b = []
    scorer_stub(calls_b)
    runner.run_judge_experiment(exp_b, workers=1)
    state_b = experiment.state_of(exp_b)
    # 同对整轮重放：零评分调用；结论与来源一致
    assert calls_b == []
    assert state_b['verdict'] == state_a['verdict']
    assert state_b['metrics']['net_win_confirmed'] == state_a['metrics']['net_win_confirmed']
    # 审计线索 + 逐字节一致
    manifest = json.loads((exp_b / 'replay_sources.json').read_text())
    assert len(manifest['cases']) == 4
    src_final = runner._final_records(exp_a / 'cases.jsonl')[0]
    dst_final = runner._final_records(exp_b / 'cases.jsonl')[0]
    strip = lambda row: {k: v for k, v in row.items() if not k.startswith('draw_nonce')}
    for cid in src_final:
        assert strip(dst_final[cid]) == strip(src_final[cid])
    # 重放确实发生了（重放行带注入的 draw_nonce 溯源字段）
    assert any(k.startswith('draw_nonce') for k in dst_final['c0'])
    assert n_calls_a > 0


def test_supplement_runs_only_missing_cases(priv, scorer_stub, monkeypatch):  # noqa: F811
    """来源有争议题缺补验轮：完整题重放，缺的题才执行。"""
    _use_small_settings(monkeypatch)
    pack = _pack(priv)
    exp_a = experiment.create_judge_eval_experiment(
        'development', pack, {'llm': {'model': 'm9'}}, change='换模型', exp_id='judge-a')
    scorer_stub([])
    runner.run_judge_experiment(exp_a, workers=1)
    # 人为制造缺轮：c1 改为分歧但无 flip（重算证据后入库）
    lines = [json.loads(l) for l in (exp_a / 'cases.jsonl').read_text().splitlines() if l.strip()]
    for row in lines:
        if row['case_id'] == 'c1':
            row['candidate_correct'] = not row['baseline_correct']
            row.pop('flip_verified', None)
    (exp_a / 'cases.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in lines))
    import sqlite3
    bank = measurements.bank_path(priv)
    bank.unlink(missing_ok=True)
    measurements.record_experiment(experiment.load_experiment('judge-a'))

    exp_b = experiment.create_judge_eval_experiment(
        'development', pack, {}, change='换模型', exp_id='judge-b',
        candidate_ref=experiment.spec_of(exp_a)['candidate_ref'])
    calls_b = []
    scorer_stub(calls_b)
    runner.run_judge_experiment(exp_b, workers=1)
    state_b = experiment.state_of(exp_b)
    assert state_b['status'] == 'finished'
    assert {c for c, _ in calls_b} == {'c1'}  # 只跑了缺的题
    assert state_b['metrics']['net_win_confirmed'] == -1  # c0/c2 负 c1 正


def test_fixed_test_and_no_replay_never_replay(priv, monkeypatch):  # noqa: F811
    _use_small_settings(monkeypatch)
    pack = _pack(priv)
    exp = experiment.create_judge_eval_experiment(
        'development', pack, {'llm': {'model': 'm9'}}, change='换模型', exp_id='judge-a')
    measurements.record_experiment(exp)
    spec_fixed = {**experiment.spec_of(exp), 'dataset': 'fixed_test'}
    assert measurements.plan_replay(priv, spec_fixed, ['c0']) == {'replay': {}, 'run': ['c0']}
    spec_off = {**experiment.spec_of(exp), 'no_replay': True}
    result = measurements.apply_replay(exp)
    assert result['replayed'] == 0
    spec = experiment.spec_of(exp)
    _write(exp / 'spec.json', {**spec, 'no_replay': True})
    assert measurements.apply_replay(exp)['replayed'] == 0


def test_swapped_sides_never_replay(priv, scorer_stub, monkeypatch):  # noqa: F811
    """baseline/candidate 互换的实验 pair_fp 相同（无序对），但整行拷贝会反转判决——禁止重放。"""
    _use_small_settings(monkeypatch)
    pack = _pack(priv)
    calls_a = []
    exp_a = experiment.create_judge_eval_experiment(
        'development', pack, {'llm': {'model': 'm9'}}, change='换模型', exp_id='judge-a')
    scorer_stub(calls_a)
    runner.run_judge_experiment(exp_a, workers=1)
    spec_a = experiment.spec_of(exp_a)
    # 构造互换实验：baseline = 原 candidate，candidate = 原 baseline
    plan = measurements.plan_replay(priv, {
        **spec_a,
        'baseline_ref': spec_a['candidate_ref'],
        'candidate_ref': spec_a['baseline_ref'],
    }, [f'c{i}' for i in range(4)])
    assert plan['replay'] == {}  # 角色不一致 → 不重放


def test_replay_copies_trace_files(priv, scorer_stub, monkeypatch):  # noqa: F811
    """重放行引用的调用实录从来源实验随迁到本实验（自足审计单元）。"""
    _use_small_settings(monkeypatch)
    pack = _pack(priv)
    exp_a = experiment.create_judge_eval_experiment(
        'development', pack, {'llm': {'model': 'm9'}}, change='换模型', exp_id='judge-a')
    scorer_stub([])
    runner.run_judge_experiment(exp_a, workers=1)
    trace_refs = [json.loads(l).get('trace_ref')
                  for l in (exp_a / 'cases.jsonl').read_text().splitlines() if l.strip()]
    assert trace_refs and all((exp_a / 'traces' / f'{r}.json').is_file() for r in trace_refs)

    exp_b = experiment.create_judge_eval_experiment(
        'development', pack, {}, change='换模型', exp_id='judge-b',
        candidate_ref=experiment.spec_of(exp_a)['candidate_ref'])
    (exp_b / 'cases.jsonl').unlink(missing_ok=True)  # 强制全新实验走重放
    (exp_b / 'state.json').unlink(missing_ok=True)
    result = measurements.apply_replay(exp_b)
    assert result['replayed'] == 4
    for ref in trace_refs:
        assert (exp_b / 'traces' / f'{ref}.json').is_file(), 'trace 未随迁'
        assert (exp_b / 'traces' / f'{ref}.json').read_bytes() == \
               (exp_a / 'traces' / f'{ref}.json').read_bytes(), 'trace 字节不一致'
