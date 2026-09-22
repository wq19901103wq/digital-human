"""实时阶段、部分样本分母、续跑及固定集接口边界。"""
import functools
import http.server
import json
import os
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from src.dashboard import server
from src.dashboard import report
from src.iteration import experiment, versions
from src.iteration.progress import RunProgress, track_run
from test_report import _run


def test_partial_rates_exclude_failed_and_unprocessed_cases(tmp_path):
    exp, _ = _run(tmp_path, smoke=True, state={'status': 'running'})
    progress = RunProgress(exp)
    progress.stage('generating', total=1000, case_index=1, branch='baseline')
    empty = report._rate_cards(report._load_run(exp))
    assert '尚无有效判定' in empty and '0.0%' not in empty
    progress.completed({'case_id': '1', 'status': 'ok', 'identified_baseline': True, 'identified_candidate': False})
    progress.completed({'case_id': '2', 'status': 'failed'})
    progress.stage('extracting', case_index=3, branch='candidate', round=2)
    run = report._load_run(exp)
    assert run['pairs'] == 1 and run['failures'] == 1 and run['attempted'] == 1000
    page = report._latest(run)
    for expected in ['100.0%', '0.0%', '50.0%', '1 / 2 道已处理题', '大模型抽取特征', '补测第 2 轮', '第 3 / 1000 题']:
        assert expected in page
    # 续跑成功替换失败，不增加有效分母中的重复题。
    progress.completed({'case_id': '2', 'status': 'ok', 'identified_baseline': False, 'identified_candidate': True})
    counts = experiment.state_of(exp)['progress']['counts']
    assert counts == {'pairs': 2, 'failures': 0, 'identified_baseline': 1, 'identified_candidate': 1}


def test_judge_callbacks_and_process_exit_are_visible(tmp_path, monkeypatch):
    exp, _ = _run(tmp_path, kind='judge_eval', state={'status': 'running'})
    progress = RunProgress(exp)
    progress.stage('preparing', total=1000, case_index=1)
    phases = []
    class Scorer:
        def is_ai(self, *args):
            for phase in ('judging', 'extracting', 'predicting'):
                self.on_progress(phase)
                snapshot = experiment.state_of(exp)['progress']
                phases.append((snapshot['phase'], snapshot['branch'], snapshot['round']))
            return True
    scorer = Scorer()
    assert progress.judge(scorer, {}, [], 'candidate', 1)
    assert phases == [('judging', 'candidate', 1), ('extracting', 'candidate', 1), ('predicting', 'candidate', 1)]
    assert scorer.on_progress is None
    assert '运行中' in report._status({}, experiment.state_of(exp))[0]
    def gone(*_):
        raise ProcessLookupError()
    monkeypatch.setattr(os, 'kill', gone)
    page = report._stage_html(report._load_run(exp))
    assert '执行已停止' in page and 'data-stage-clock' not in page


def _select_execution_instance(exp, monkeypatch):
    instance = exp.parent.parent
    monkeypatch.setattr(versions, 'PRIVATE', instance)
    monkeypatch.setattr(versions, 'DATA_ROOT', instance / 'data')
    (instance / 'data/data/dev_pool.jsonl').write_text(json.dumps({'case_id': '1'}) + '\n')


def test_interruption_keeps_checkpoint_and_stops_progress(tmp_path, monkeypatch):
    exp, _ = _run(tmp_path, state={'status': 'running'})
    _select_execution_instance(exp, monkeypatch)
    @track_run
    def interrupted(directory, *, _progress):
        _progress.stage('generating', total=1000)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        interrupted(exp)
    state = experiment.state_of(exp)
    assert state['progress']['status'] == 'stopped'
    assert state['progress']['phase'] == 'interrupted'
    assert state['status'] == 'running'  # 仍允许续跑。


def test_finished_state_keeps_timing_but_does_not_look_running(tmp_path, monkeypatch):
    exp, _ = _run(tmp_path, state={'status': 'running'}, records=[
        {'case_id': '1', 'status': 'ok', 'identified_baseline': True, 'identified_candidate': True}])
    _select_execution_instance(exp, monkeypatch)
    progress = RunProgress(exp)
    progress.stage('generating', total=1)
    progress.stage('summarizing')
    experiment.finish(exp, {'attempted': 1, 'pairs': 1, 'failures': 0,
                           'identified_baseline': 1, 'identified_candidate': 1},
                      {'verdict': 'observe', 'reason': 'test'})
    state = experiment.state_of(exp)
    assert state['progress']['status'] == 'finished'
    assert 'generating' in state['progress']['timings']
    assert '任务已结束' in report._stage_html(report._load_run(exp))


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    handler = functools.partial(server._AuditGuardHandler, directory=str(tmp_path))
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_live_http_reads_new_stage_without_regenerating_html(tmp_path, live_server):
    exp, _ = _run(tmp_path, state={'status': 'running'})
    progress = RunProgress(exp)
    progress.stage('generating', total=1000, case_index=1, branch='baseline')
    url = live_server + '/api/live/demo/test-run'
    first = json.load(urlopen(url))
    assert '生成回复' in first['regions']['summary']
    progress.stage('extracting', branch='candidate')
    second = json.load(urlopen(url + '?cases_revision=' + first['cases_revision']))
    assert '大模型抽取特征' in second['regions']['summary']
    assert 'cases' not in second['regions']
    # 尚未生成静态 index.html，也可直接打开正在运行的任务。
    response = urlopen(live_server + '/instances/demo/experiments/test-run/index.html')
    assert response.headers['Cache-Control'] == 'no-store'
    assert '大模型抽取特征' in response.read().decode()


def test_fixed_cases_never_enter_live_payload(tmp_path, live_server):
    exp, _ = _run(tmp_path, fixed=True, records=[{'case_id': 'secret', 'status': 'failed',
        'reason': 'FIXED_SECRET', 'human_reply': ['FIXED_SECRET']}])
    # 自定义实验 ID 也按 spec.dataset 隔离，不依赖 ID 名称。
    body = urlopen(live_server + '/api/live/demo/test-run').read().decode()
    assert 'FIXED_SECRET' not in body and '固定集答案不在开发后台展开' in body


@pytest.mark.parametrize('filename', ['pack.json', 'building.json'])
def test_validation_pack_and_checkpoint_are_audit_only(tmp_path, live_server, filename):
    root = tmp_path / 'instances' / 'demo' / 'judge_eval'
    for split in ('validation', 'calibration'):
        directory = root / f'pack-{split}-rebase-test'
        directory.mkdir(parents=True)
        (directory / filename).write_text('{"human_reply": ["test answer"]}')
    path = '/instances/demo/judge_eval/pack-{}-rebase-test/' + filename
    with pytest.raises(HTTPError) as caught:
        urlopen(live_server + path.format('validation') + '?download=1')
    assert caught.value.code == 403
    assert urlopen(live_server + path.format('calibration')).status == 200


@pytest.mark.parametrize('path', ['/api/live/%2e%2e/test-run', '/api/live/demo/%2e%2e', '/api/live/demo/a/b'])
def test_live_endpoint_rejects_path_escape(tmp_path, live_server, path):
    with pytest.raises(HTTPError) as caught:
        urlopen(live_server + path)
    assert caught.value.code in (403, 404)
