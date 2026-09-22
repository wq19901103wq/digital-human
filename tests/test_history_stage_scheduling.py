"""Scheduling may overlap disjoint stages but must preserve experiment inputs."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.legacy import run_history_generation_stage as generation
from scripts.legacy import run_history_training_pipeline as pipeline
from scripts.legacy import run_history_training_stage as training
from src.config import ConfigError
from src.iteration.storage import write_json


def test_resume_compares_complete_json_value_and_rejects_content_change(tmp_path):
    path = tmp_path / 'training.json'
    rows = {'rows': [{'case_id': 'same', 'metadata': {'recent_speakers': ('a', 'b')},
                      'human_option': 'A', 'blind': {'option_A': ['hello']}}]}
    training.save_json_once(path, rows)
    original = path.read_bytes()
    training.save_json_once(path, rows)
    assert path.read_bytes() == original
    for changed in ('speaker', 'label', 'reply'):
        value = copy.deepcopy(rows)
        if changed == 'speaker':
            value['rows'][0]['metadata']['recent_speakers'] = ('a', 'c')
        elif changed == 'label':
            value['rows'][0]['human_option'] = 'B'
        else:
            value['rows'][0]['blind']['option_A'] = ['different']
        with pytest.raises(ConfigError, match='冻结产物发生变化'):
            training.save_json_once(path, value)
        assert path.read_bytes() == original


@pytest.mark.parametrize('dataset', ['training', 'development'])
def test_generation_stage_uses_original_inputs_and_checkpoint_directory(tmp_path, monkeypatch, dataset):
    study = tmp_path / 'study'
    spec = {'inputs': {'frozen': 'digest'}, 'data_ref': 'd-0011', 'dataset': 'training',
            'training_total': 2, 'evaluation_total': 2}
    rows = [{'case_id': 'a'}, {'case_id': 'b'}]
    write_json(study / 'spec.json', spec)
    write_json(study / 'sources.json', {'cases': rows})
    verified = []
    monkeypatch.setattr(generation.job.common, 'verify_files', lambda inputs: verified.append(copy.deepcopy(inputs)))
    monkeypatch.setattr(generation.versions, 'data_version_dir', lambda ref: tmp_path / ref)
    monkeypatch.setattr(generation.job, 'cases', lambda path, role: rows if role == 'judge_development' else [])
    calls = []

    def produce(target, actual_spec, actual_rows, workers):
        calls.append((target, actual_spec, actual_rows, workers))
        return {'entries': {c['case_id']: {'status': 'ok'} for c in rows}}

    monkeypatch.setattr(generation.job, 'generate', produce)
    generation.run_stage(study, dataset, 2)
    target = study if dataset == 'training' else study / 'development'
    assert calls == [(target, {**spec, 'dataset': dataset}, rows, 2)]
    assert verified == [spec['inputs'], spec['inputs']]
    assert generation.job.read(study / 'spec.json') == spec


def test_generation_stage_rejects_changed_inputs_before_requests(tmp_path, monkeypatch):
    write_json(tmp_path / 'spec.json', {'inputs': {'frozen': 'digest'}})

    def changed(inputs):
        raise ConfigError('frozen changed')

    monkeypatch.setattr(generation.job.common, 'verify_files', changed)
    monkeypatch.setattr(generation.job, 'generate', lambda *args: pytest.fail('must not request'))
    with pytest.raises(ConfigError, match='frozen changed'):
        generation.run_stage(tmp_path, 'training', 2)


def test_parallel_stages_start_both_before_join_and_retry_only_failure(tmp_path, monkeypatch):
    started = []
    codes = {'training': [None, 0], 'development': [1], 'development-retry': [0]}
    processes = []

    def start(command, cwd):
        name = command[0]
        key = 'development-retry' if name in started else name
        started.append(name)
        outcomes = iter(codes[key])

        def poll():
            assert len(started) >= 2
            return next(outcomes)

        process = SimpleNamespace(pid=len(started), poll=poll,
                                  terminate=lambda: pytest.fail('successful stage must not be stopped'))
        processes.append(process)
        return process

    monkeypatch.setattr(pipeline.subprocess, 'Popen', start)
    monkeypatch.setattr(pipeline.time, 'sleep', lambda delay: None)
    state = {'commands': []}
    assert pipeline.parallel_stages({'training': ['training'], 'development': ['development']},
                                    state, tmp_path / 'pipeline.json') == 0
    assert started == ['training', 'development', 'development']
    assert all(s['status'] == 'finished' for s in state['stages'].values())


def test_parallel_failure_stops_other_writer_and_does_not_continue(tmp_path, monkeypatch):
    stopped = []
    attempts = []

    def start(command, cwd):
        name = command[0]
        attempts.append(name)
        return SimpleNamespace(pid=len(attempts), poll=lambda: 1 if name == 'training' else None,
                               terminate=lambda: stopped.append(name), wait=lambda: -15, returncode=-15)

    monkeypatch.setattr(pipeline.subprocess, 'Popen', start)
    monkeypatch.setattr(pipeline.time, 'sleep', lambda delay: None)
    state = {'commands': []}
    assert pipeline.parallel_stages({'training': ['training'], 'development': ['development']},
                                    state, tmp_path / 'pipeline.json') == 1
    assert attempts.count('training') == 3
    assert stopped == ['development']
    assert state['stages']['development']['status'] == 'stopped'


@pytest.mark.parametrize('shared,generation_workers,evaluation_workers', [(False, 2, None), (True, 16, 16)])
def test_staged_pipeline_joins_training_and_generation_before_evaluation(
        tmp_path, monkeypatch, shared, generation_workers, evaluation_workers):
    study = tmp_path / 'instances/example-agent/judge_training/test-study'
    feature = {'model': 'gpt-5.6-sol', 'reasoning_effort': 'high'}
    write_json(study / 'spec.json', {'feature_config': feature, 'data_ref': 'd-0011',
                                   'generator_ref': 'g-0017', 'source_judge': 'j-0012'})
    write_json(study / 'preflight_audit.json', {'status': 'passed', 'feature_config': feature,
                                             'counts': {'training_features_checked': 2}})
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'sha256_file', lambda path: 'scheduler-hash')
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline', '--study', 'test-study',
                                             '--workers', '8', '--generation-workers', str(generation_workers)]
                        + (['--shared-retriever'] if shared else [])
                        + (['--evaluation-workers', str(evaluation_workers)] if evaluation_workers else []))
    events = []

    def start(command, cwd):
        if '--evaluate' in command:
            assert events[-1] == 'joined'
            if evaluation_workers:
                assert command[-2:] == ['--evaluation-workers', str(evaluation_workers)]
        events.append(command)
        return SimpleNamespace(pid=10, wait=lambda: 0)

    def parallel(commands, value, status_path):
        assert '--evaluate' not in commands['training']
        assert Path(commands['training'][3]).name == 'run_history_training_stage.py'
        assert commands['training'][-1] == '--run'
        assert '--evaluation-workers' not in commands['training']
        assert commands['training'][commands['training'].index('--workers') + 1] == '8'
        assert commands['development_generation'][-2:] == ['--dataset', 'development']
        assert commands['development_generation'][commands['development_generation'].index('--workers') + 1] == str(generation_workers)
        assert ('--shared-retriever' in commands['development_generation']) is shared
        events.append('joined')
        return 0

    monkeypatch.setattr(pipeline.subprocess, 'Popen', start)
    monkeypatch.setattr(pipeline, 'parallel_stages', parallel)
    with pytest.raises(SystemExit) as ended:
        pipeline.main()
    assert ended.value.code == 0
    assert events[0][-2:] == ['--dataset', 'training']
    assert ('--shared-retriever' in events[0]) is shared
    assert events[-1][-1] == '--complete'
    assert generation.job.read(study / 'pipeline.json')['status'] == 'finished'


@pytest.mark.parametrize('fail', [False, True])
def test_evaluation_worker_override_preserves_original_call_and_restores_hooks(monkeypatch, fail):
    argv = ['training', '--workers', '8', '--run', '--evaluate', '--evaluation-workers', '16']
    monkeypatch.setattr(training.sys, 'argv', argv)
    calls = []

    def original(target, workers, total):
        calls.append((target, workers, total))
        if fail:
            raise RuntimeError('original failure')

    monkeypatch.setattr(training.job, 'evaluate_complete', original)
    previous_save = training.job.save_once

    def run():
        assert training.sys.argv == argv[:-2]
        training.job.evaluate_complete(Path('same-study'), 8, 1000)

    monkeypatch.setattr(training.job, 'main', run)
    if fail:
        with pytest.raises(RuntimeError, match='original failure'):
            training.main()
    else:
        training.main()
    assert calls == [(Path('same-study'), 16, 1000)]
    assert training.job.evaluate_complete is original
    assert training.job.save_once is previous_save
    assert training.sys.argv is argv


@pytest.mark.parametrize('arguments', [
    ['--study', 'test', '--generation-workers', '16'],
    ['--study', 'test', '--evaluation-workers', '16'],
    ['--study', 'test', '--generation-workers', '17', '--shared-retriever'],
])
def test_pipeline_rejects_unsafe_concurrency_before_start(monkeypatch, arguments):
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline', *arguments])
    with pytest.raises(SystemExit) as ended:
        pipeline.main()
    assert ended.value.code == 2
