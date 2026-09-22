"""Training configuration must agree with inference and the frozen control."""
import copy
from types import SimpleNamespace

import pytest

from scripts.legacy import train_history_judge as job
from scripts.legacy import run_history_training_pipeline as pipeline
from src.config import ConfigError


def test_feature_upgrade_preserves_initial_judge_and_existing_options():
    source = {'config': {'llm': {'model': 'gpt-5.6-luna', 'reasoning_effort': 'low',
                                'provider': 'codex_cli', 'timeout_seconds': 360},
                         'correction_threshold': .7}}
    original = copy.deepcopy(source)
    result = job.feature_config(source, SimpleNamespace(feature_model='gpt-5.6-sol', feature_effort='high'))
    assert result == {**source['config']['llm'], 'model': 'gpt-5.6-sol', 'reasoning_effort': 'high'}
    assert source == original


def test_frozen_artifact_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'frozen.json'
    job.save_once(path, {'source': 'first'})
    job.save_once(path, {'source': 'first'})
    with pytest.raises(ConfigError, match='冻结产物发生变化'):
        job.save_once(path, {'source': 'other'})
    assert job.read(path) == {'source': 'first'}


def test_evaluation_reopens_incomplete_finished_run_without_losing_successes(tmp_path, monkeypatch):
    records = {'ok': {'status': 'ok'}, 'missing': {'status': 'failed'}}
    monkeypatch.setattr(job.protocol, 'summarize_final_records', lambda p: (records, 0))
    monkeypatch.setattr(job.experiment, 'state_of', lambda p: {'status': 'finished', 'metrics': {'failures': 1}})
    calls = []

    def resume(target, workers):
        assert job.read(target / 'state.json')['status'] == 'stopped'
        assert records['ok']['status'] == 'ok'
        records['missing']['status'] = 'ok'
        calls.append(workers)

    monkeypatch.setattr(job.runner, 'run_judge_experiment', resume)
    job.evaluate_complete(tmp_path, 4, 2)
    assert calls == [4]
    assert len(list((tmp_path / 'snapshots').glob('*.json'))) == 1


def test_evaluation_cannot_report_success_with_persistent_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(job.protocol, 'summarize_final_records', lambda p: ({'case': {'status': 'failed'}}, 0))
    monkeypatch.setattr(job.experiment, 'state_of', lambda p: {'status': 'running'})
    calls = []
    monkeypatch.setattr(job.runner, 'run_judge_experiment', lambda p, workers: calls.append(workers))
    with pytest.raises(ConfigError, match='尚未全部有效'):
        job.evaluate_complete(tmp_path, 2, 1)
    assert len(calls) == 3


@pytest.mark.parametrize('codes,expected,phases', [([1, 0, 0], 'finished', 3), ([0, 1], 'stopped', 2)])
def test_pipeline_requires_successful_independent_audit(tmp_path, monkeypatch, codes, expected, phases):
    directory = tmp_path / 'instances/example-agent/judge_training/test-study'
    feature = {'model': 'gpt-5.6-sol', 'reasoning_effort': 'high'}
    job.save_once(directory / 'spec.json', {'feature_config': feature, 'data_ref': 'd-0011',
                 'generator_ref': 'g-0017', 'source_judge': 'j-0012'})
    job.save_once(directory / 'preflight_audit.json', {'status': 'passed', 'feature_config': feature,
                 'counts': {'training_features_checked': 2}})
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline', '--study', 'test-study', '--workers', '4'])
    pending = iter(codes)
    commands = []

    def start(command, cwd):
        commands.append(command)
        code = next(pending)
        return SimpleNamespace(pid=123, wait=lambda: code)

    monkeypatch.setattr(pipeline.subprocess, 'Popen', start)
    with pytest.raises(SystemExit) as ended:
        pipeline.main()
    assert (ended.value.code == 0) == (expected == 'finished')
    status = job.read(directory / 'pipeline.json')
    assert status['status'] == expected and status['adopted'] is False
    assert len(commands) == phases
    assert 'verify_history_training.py' in commands[-1][3]
    assert commands[-1][-1] == '--complete'
