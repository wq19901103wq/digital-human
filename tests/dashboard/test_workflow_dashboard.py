"""Concurrent workflow visibility, independent of training/execution mutations."""
import json
from urllib.request import urlopen

from src.dashboard import workflow_view
from src.iteration import task_state
from src.iteration.storage import write_json
from test_live_dashboard import live_server as _live_server
from test_report import _run

live_server = _live_server


def test_workflow_lists_parallel_training_and_blocked_branches_without_answers(tmp_path):
    study = tmp_path / 'judge_training/study'
    write_json(study / 'spec.json', {'data_ref': 'd-1', 'generator_ref': 'g-1', 'source_judge': 'j-1'})
    task_state.update(study, 'training_and_development', total=10, successful=10)
    task_state.update(study / 'model-a', 'extracting', total=10, successful=4, failed=0)
    task_state.update(study / 'model-b', 'extracting', total=10, successful=3, failed=1)
    write_json(tmp_path / 'judge_training/old/spec.json', {})
    write_json(tmp_path / 'judge_training/old/state.json', {'status': 'running', 'phase': 'generating', 'pid': 0})
    branch = tmp_path / 'branches/direction-a'
    write_json(branch / 'state.json', {'rounds': ['r-0001']})
    write_json(branch / 'rounds/r-0001.json', {'phase': 'blocked', 'reason': '等待新验收批次'})
    page = workflow_view.html(tmp_path)
    for value in ('model-a', 'model-b', '成功 4 / 10', '成功 3 / 10', '失败 1', '已中断', '等待新验收批次'):
        assert value in page
    task_state.update(study / 'model-a', 'fitting', total=10, successful=10)
    assert '拟合小模型' in workflow_view.html(tmp_path)
    assert not (tmp_path / 'data').exists()


def test_live_workflow_updates_training_without_regenerating_html(tmp_path, live_server):
    exp, _ = _run(tmp_path, state={'status': 'finished'})
    study = exp.parent.parent / 'judge_training/visible-study'
    write_json(study / 'spec.json', {'data_ref': 'd-0001'})
    task_state.update(study, 'extracting', total=12, successful=5, failed=0)
    endpoint = live_server + '/api/live/demo'
    first = json.load(urlopen(endpoint))
    assert '成功 5 / 12' in first['regions']['workflows']
    task_state.update(study, 'fitting', total=12, successful=12, failed=0)
    second = json.load(urlopen(endpoint))
    assert '拟合小模型' in second['regions']['workflows']
    page = urlopen(live_server + '/dashboard/demo/index.html').read().decode()
    assert 'data-live-region="workflows"' in page and 'visible-study' in page


def test_counts_are_scoped_to_the_phase(tmp_path):
    task_state.update(tmp_path, 'extracting', total=1200, successful=1200, failed=0)
    task_state.update(tmp_path, 'evaluating', total=1000)
    value = task_state.read(tmp_path)
    assert 'successful' not in value
    assert value['stage_progress']['extracting']['successful'] == 1200
    assert value['total'] == 1000


def test_legacy_mixed_counts_are_not_presented_as_valid_progress():
    value = {'status': 'finished', 'successful': 1200, 'total': 1000}
    text = workflow_view._progress(value)
    assert '1200 / 1000' not in text
    assert '未区分阶段' in text


def test_unregistered_history_is_visible_and_updates_without_handwritten_table(tmp_path, live_server):
    exp, _ = _run(tmp_path, state={'status': 'finished'})
    study = exp.parent.parent / 'judge_training/history'
    write_json(study / 'spec.json', {'adoption_allowed': False})
    cases = study / 'comparisons/pure-lr/cases.jsonl'
    cases.parent.mkdir(parents=True)
    cases.write_text('{"human_reply": "DO_NOT_RENDER"}\n')
    endpoint = live_server + '/api/live/demo'
    first = json.load(urlopen(endpoint))['regions']['workflows']
    assert '1 项历史比较尚未入表' in first
    assert 'pure-lr' in first and 'DO_NOT_RENDER' not in first
    # An imported experiment is discovered on the next request. No saved table
    # or manually curated list is consulted, and source evidence is untouched.
    spec = json.loads((exp / 'spec.json').read_text())
    spec['provenance'] = {'study': 'history', 'comparison_id': 'pure-lr'}
    write_json(exp / 'spec.json', spec)
    second = json.load(urlopen(endpoint))['regions']['workflows']
    assert '尚未入表' not in second
    assert '查看 1 项比较结果' in second
    assert cases.read_text() == '{"human_reply": "DO_NOT_RENDER"}\n'
