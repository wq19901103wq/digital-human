"""当前指针与冻结实验对照的展示口径。"""
import json

import pytest

from src.dashboard import version_view
from src.dashboard import report
from test_report import _run, _write


@pytest.mark.parametrize('fixed,branch,expected', [
    (False, False, 'iteration'), (True, False, 'production'), (False, True, 'production'),
])
def test_current_comparison_uses_correct_baseline_role(tmp_path, fixed, branch, expected):
    exp, run = _run(tmp_path, fixed=fixed)
    pointers = {'data': 'data', 'production_gen': 'prod', 'iteration_gen': 'dev', 'production_judge': 'base'}
    _write(exp.parent.parent / 'pointers.json', pointers)
    run['spec']['baseline_ref'] = pointers[expected + '_gen']
    if branch:
        run['spec']['branch_round'] = 'r-0001'
    body = report._baseline_context(run)
    label = '当前生产基线' if expected == 'production' else '当前开发基线'
    assert f'本轮对照与{label}一致' in body
    assert f'/versions/generators/{pointers[expected + "_gen"]}/index.html' in body


def test_pointer_changes_update_display_without_rewriting_experiment(tmp_path):
    exp, run = _run(tmp_path)
    before = (exp / 'spec.json').read_bytes()
    assert '本轮对照与当前开发基线一致' in report._baseline_context(run)
    ptr = report._json(exp.parent.parent / 'pointers.json')
    ptr['iteration_gen'] = 'new-dev'
    _write(exp.parent.parent / 'pointers.json', ptr)
    html = report.live_payload(exp.parent.parent)['regions']['history']
    assert '非当前基线对照' in html and '/generators/new-dev/index.html' in html
    assert '/generators/base/index.html' in html and '本轮对照' in html
    assert (exp / 'spec.json').read_bytes() == before


def test_promoted_candidate_is_not_mistaken_for_original_baseline(tmp_path):
    exp, run = _run(tmp_path)
    ptr = report._json(exp.parent.parent / 'pointers.json')
    ptr['iteration_gen'] = 'candidate'
    _write(exp.parent.parent / 'pointers.json', ptr)
    body = report._baseline_context(run)
    assert '本轮候选现为当前开发基线' in body
    assert '本轮对照与当前开发基线一致' not in body


def test_same_judge_baseline_with_old_data_is_not_current_system_evidence(tmp_path):
    exp, run = _run(tmp_path, kind='judge_eval')
    ptr = report._json(exp.parent.parent / 'pointers.json')
    ptr['data'] = 'new-data'
    _write(exp.parent.parent / 'pointers.json', ptr)
    _write(exp.parent.parent / 'judge_eval/calibration-pack/pack.json', {'c0_gen_version': 'old-generator'})
    body = report._baseline_context(run)
    assert '本轮对照与当前开发基线一致' in body
    assert '评测数据与当前不同' in body
    assert '评估包的生成器与当前生产基线不同' in body
    assert '不能直接视为对当前系统的提升' in body


def test_special_controls_are_explicit_and_current_badges_follow_pointers(tmp_path):
    exp, run = _run(tmp_path, kind='judge_eval')
    ptr = report._json(exp.parent.parent / 'pointers.json')
    ptr.update(production_judge='current-judge', iteration_judge='current-judge')
    _write(exp.parent.parent / 'pointers.json', ptr)
    run['spec']['rerun'] = {'original_experiment': 'original'}
    assert '历史方案复核' in report._baseline_context(run)
    run['spec']['lr_retrain'] = {'promotion_eligible': False}
    body = report._baseline_context(run)
    assert '重训控制实验' in body and '已标记为不可晋升' in body
    assert '本轮对照不是当前开发基线' in body
    assert version_view._current(exp.parent.parent, 'judges', 'base') == '非当前基线快照'
    current = version_view._current(exp.parent.parent, 'judges', 'current-judge')
    assert '当前生产基线' in current and '当前开发基线' in current
    assert json.loads((exp / 'spec.json').read_text())['baseline_ref'] == 'base'


def test_material_repair_is_a_control_even_when_sources_pass(tmp_path):
    exp, run = _run(tmp_path, kind='judge_eval')
    run['spec'].update(material_repair={'adoption_allowed': False},
                       source_audit={'promotion_eligible': True})
    _write(exp / 'spec.json', run['spec'])
    before = (exp / 'spec.json').read_bytes()
    assert report._phase(run['spec'])[0] == '重训控制实验'
    assert report._status(run['spec'], run['state'])[0] == '诊断对比完成'
    assert '不是已晋级基线' in report._baseline_context(run)
    assert '本轮对照' in report._history_versions(run)
    assert '本轮基线' not in report._history_versions(run)
    assert '不能替代正式晋级评测' in report._next_step(run)
    assert '不参与版本采用' in report._adoption_note(run)
    report.write_run(exp)
    assert (exp / 'spec.json').read_bytes() == before


def test_against_production_uses_production_pointer_in_display(tmp_path):
    exp, run = _run(tmp_path, kind='judge_eval')
    run['spec']['against_production'] = True
    ptr = report._json(exp.parent.parent / 'pointers.json')
    ptr['iteration_judge'] = 'new-dev'
    _write(exp.parent.parent / 'pointers.json', ptr)
    body = report._baseline_context(run)
    assert '本轮对照与当前生产基线一致' in body
    assert '生产基线直接比较' in body
    assert '本轮对照不是当前开发基线' not in body
    assert '不重复晋级开发版' in report._next_step(run)
    assert '不切换生产基线' in report._adoption_note(run)
