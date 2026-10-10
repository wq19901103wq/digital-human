"""Saved-model readiness does not fit, request models or hide native crashes."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import check
from src.infrastructure.config import ConfigError
from src.infrastructure.storage import write_json
from src.iteration import inference_check


@pytest.mark.parametrize('kind,module', [
    ('judge-gbdt', 'src.judge.gbdt'),
    ('fewshot-xgboost', 'src.generator.fewshot.boosting'),
    ('judge-embedding', 'src.judge.embedding'),
])
def test_saved_assets_use_real_online_entrypoints_without_fitting(tmp_path, monkeypatch, kind, module):
    path = tmp_path / 'model.json'
    document = {'saved': True}
    write_json(path, document)
    scorer = SimpleNamespace(dimensions=3, score=Mock(return_value=np.array([0., 0.])))
    model = SimpleNamespace(encoder={'fields': ['field']})
    runtime = SimpleNamespace(GBDTScorer=Mock(return_value=scorer),
        score_document=Mock(return_value=np.array([0., 0.])),
        torch=SimpleNamespace(set_num_threads=Mock()), restore=Mock(return_value=model),
        predict=Mock(return_value=np.array([.5, .5])))
    loader = Mock(return_value=runtime)
    monkeypatch.setattr(inference_check, 'import_module', loader)
    progress = []
    result = inference_check.verify(path, kind, progress=progress.append)
    loader.assert_called_once_with(module)
    assert result['predictions'] == 2
    assert result['training'] is False
    assert result['external_model_requests'] == 0
    assert result['source_provenance_verified'] is False
    assert result['performance_claim'] is False
    assert progress[0] == 'importing_runtime'
    if kind == 'judge-gbdt':
        runtime.GBDTScorer.assert_called_once_with(document)
        assert scorer.score.call_args.args[0].shape == (2, 3)
    elif kind == 'fewshot-xgboost':
        runtime.score_document.assert_called_once_with(document, [{}, {}])
    else:
        runtime.restore.assert_called_once_with(document)
        runtime.torch.set_num_threads.assert_called_once_with(1)
        assert runtime.predict.call_args.args[1].shape == (2, 2, 1)


@pytest.mark.parametrize('values', ([0.], [float('nan'), 0.]))
def test_nonfinite_or_wrong_shape_predictions_fail(tmp_path, monkeypatch, values):
    path = tmp_path / 'model.json'
    write_json(path, {})
    monkeypatch.setattr(inference_check, 'import_module',
                        lambda name: SimpleNamespace(score_document=lambda document, rows: values))
    with pytest.raises(ConfigError, match='形状或数值'):
        inference_check.verify(path, 'fewshot-xgboost')


def test_native_process_abort_is_a_failed_check(monkeypatch, tmp_path):
    runner = Mock(return_value=SimpleNamespace(returncode=-6, stdout='stage: importing_runtime\nabort'))
    monkeypatch.setattr(check.subprocess, 'run', runner)
    args = SimpleNamespace(kind='fewshot-xgboost', model=tmp_path / 'model.json')
    report = check.Report('inference')
    check.check_inference(args, report)
    value = report.finish()
    assert value['status'] == 'failed'
    assert 'exit=-6' in value['checks'][0]['error']
    assert 'importing_runtime' in value['checks'][0]['error']
    assert runner.call_args.kwargs['timeout'] == 60


def test_inference_command_dispatches_without_an_instance(monkeypatch, tmp_path, capsys):
    checker = Mock(side_effect=lambda args, report: report.run('prediction', lambda: True))
    monkeypatch.setattr(check, 'check_inference', checker)
    result = check.main(['inference', '--kind', 'judge-gbdt', '--model', str(tmp_path / 'model.json')])
    assert result == 0
    checker.assert_called_once()
    assert checker.call_args.args[0].kind == 'judge-gbdt'
    assert '"training_started": false' in capsys.readouterr().out


@pytest.mark.parametrize('output', ['stage: loading_saved_model\n', b'stage: loading_saved_model\n'])
def test_timeout_keeps_last_completed_stage(monkeypatch, tmp_path, output):
    runner = Mock(side_effect=check.subprocess.TimeoutExpired('python', 60, output=output))
    monkeypatch.setattr(check.subprocess, 'run', runner)
    report = check.Report('inference')
    check.check_inference(SimpleNamespace(kind='judge-gbdt', model=tmp_path / 'model.json'), report)
    value = report.finish()
    assert value['status'] == 'failed'
    assert 'timeout=60s' in value['checks'][0]['error']
    assert 'loading_saved_model' in value['checks'][0]['error']


def test_changed_model_asset_fails(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    write_json(path, {})

    def changing_prediction(document, rows):
        write_json(path, {'changed': True})
        return [0., 0.]

    monkeypatch.setattr(inference_check, 'import_module',
                        lambda name: SimpleNamespace(score_document=changing_prediction))
    with pytest.raises(ConfigError, match='模型资产变化'):
        inference_check.verify(path, 'fewshot-xgboost')
