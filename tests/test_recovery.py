import json

import pytest
from leakage_support import isolated_legacy_provenance  # noqa: F401

from src.config import ConfigError
from src.dashboard import report
from src.iteration import experiment


@pytest.fixture
def runs(tmp_path):
    parent = tmp_path / 'instances/demo/experiments'
    old, new = parent / 'failed', parent / 'verified'
    for directory in (old, new):
        directory.mkdir(parents=True)
        (directory / 'spec.json').write_text(json.dumps({
            'id': directory.name, 'kind': 'gen_ab', 'smoke': True, 'dataset': 'development',
            'single_change': '修复检查', 'baseline_ref': 'g1', 'candidate_ref': 'g2',
            'judge_ref': 'j1', 'created': directory.name}))
    for directory, status in ((old, 'failed'), (new, 'ok')):
        (directory / 'cases.jsonl').write_text(json.dumps({
            'case_id': 'same-case', 'status': status, 'human_reply': ['原始答案'],
            'context': [{'sender': '甲', 'text': '原始问题'}],
            **({'reason': 'HTTP 404 旧接口不支持模型'} if status == 'failed' else {})}) + '\n')
    (old / 'state.json').write_text(json.dumps({'status': 'running', 'verdict': 'experiment_incomplete',
                                              'metrics': {'attempted': 1, 'failures': 1, 'pairs': 0}}))
    (new / 'state.json').write_text(json.dumps({'status': 'finished', 'verdict': 'observe',
                                              'metrics': {'attempted': 1, 'failures': 0, 'pairs': 1}}))
    return old, new


def test_recovery_keeps_failure_evidence_and_links_verified_run(runs, tmp_path):
    old, new = runs
    evidence = (old / 'cases.jsonl').read_bytes()
    before = experiment.state_of(old)
    experiment.record_smoke_recovery(old, new, '已改用正确接口，在同题上验证通过')
    after = experiment.state_of(old)
    assert {key: after[key] for key in before} == before
    assert (old / 'cases.jsonl').read_bytes() == evidence
    assert after['resolution']['verified_by'] == new.name
    report.write_run(old)
    page = (old / 'index.html').read_text()
    assert '历史失败 · 新配置已验证' in page
    assert 'HTTP 404 旧接口不支持模型' in page
    assert '/experiments/verified/index.html' in page
    report.refresh_dashboard(old.parent, tmp_path / 'dashboard')
    assert '待处理失败' in (tmp_path / 'dashboard/index.html').read_text()


@pytest.mark.parametrize('change', ['failed_verification', 'different_case', 'different_question', 'formal_run'])
def test_recovery_requires_successful_same_case_smoke(runs, change):
    old, new = runs
    if change == 'failed_verification':
        path = new / 'state.json'
        data = json.loads(path.read_text())
        data['metrics']['failures'] = 1
    elif change == 'formal_run':
        path = old / 'spec.json'
        data = json.loads(path.read_text())
        data['smoke'] = False
    else:
        path = new / 'cases.jsonl'
        data = json.loads(path.read_text())
        if change == 'different_case':
            data['case_id'] = 'other-case'
        else:
            data['context'][0]['text'] = '换了题目'
    path.write_text(json.dumps(data))
    before = (old / 'state.json').read_bytes()
    with pytest.raises(ConfigError):
        experiment.record_smoke_recovery(old, new, '不能只改页面标签')
    assert (old / 'state.json').read_bytes() == before


@pytest.mark.parametrize('smoke,expected_calls', [(True, 2), (False, 6)])
@pytest.mark.usefixtures("isolated_legacy_provenance")
def test_smoke_checks_both_branches_without_supplemental_calls(tmp_path, monkeypatch, smoke, expected_calls):
    from src import llm
    from src.config import load_settings
    from src.iteration import runner, versions
    exp = tmp_path / 'experiments/check'
    exp.mkdir(parents=True)
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'dev_pool.jsonl').write_text(json.dumps({'case_id': 'case', 'context': [], 'human_reply': ['原回复']}) + '\n')
    settings = load_settings()
    spec = {'id': 'check', 'kind': 'gen_ab', 'dataset': 'development', 'data_ref': 'data',
            'baseline_ref': 'base', 'candidate_ref': 'candidate', 'judge_ref': 'judge',
            'smoke': smoke, 'smoke_limit': 1, 'config_diff': ['model'],
            'protocol': experiment._protocol_snapshot(settings)}
    (exp / 'spec.json').write_text(json.dumps(spec))
    calls = {'gen': 0, 'judge': 0}
    class Generator:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            calls['gen'] += 1
            return {'replies': ['模型回复'], 'latency_ms': 1}
    class Judge:
        def is_ai(self, *args):
            calls['judge'] += 1
            return calls['judge'] % 2 == 0  # 初判不同，正常评测必须触发补测
    monkeypatch.setattr(versions, 'PRIVATE', tmp_path)
    monkeypatch.setattr(versions, 'data_version_dir', lambda *_: data)
    monkeypatch.setattr(versions, 'load_generator', lambda *_: {'config': {'llm': {}}, 'dir': data})
    monkeypatch.setattr(versions, 'judge_dir', lambda *_: {})
    monkeypatch.setattr(llm, 'build_clients', lambda *_: {})
    monkeypatch.setattr(runner, 'ReplyGenerator', Generator)
    monkeypatch.setattr(runner, 'build_judge', lambda *_: Judge())
    monkeypatch.setattr(experiment, '_verify_trigger', lambda *_: None)
    for name in ('write_run', 'refresh_dashboard', 'write_instance_index'):
        monkeypatch.setattr(report, name, lambda *_: None)
    runner.run_gen_experiment(exp)
    assert calls == {'gen': expected_calls, 'judge': expected_calls}
    row = json.loads((exp / 'cases.jsonl').read_text())
    assert row['status'] == 'ok'
    assert bool(row.get('flip_verification_skipped')) == smoke
