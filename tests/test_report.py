"""报告的真实数据边界、结果语义和页内浏览；全部使用临时实验。"""
from __future__ import annotations

import json
from pathlib import Path

from src.digital_human.iteration import report


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False), encoding='utf-8')


def _run(tmp_path, *, kind='gen_ab', fixed=False, smoke=False, records=None, state=None):
    instance = tmp_path / 'instances' / 'demo'
    exp = instance / 'experiments' / 'test-run'
    spec = {'id': exp.name, 'kind': kind, 'dataset': 'fixed_test' if fixed else 'development',
            'baseline_ref': 'base', 'candidate_ref': 'candidate', 'judge_ref': 'base',
            'data_ref': 'data', 'pack_ref': 'calibration-pack', 'smoke': smoke,
            'created': '2026-09-12 12:00:00', 'single_change': '示例任务', 'config_diff': ['模型改动']}
    _write(exp / 'spec.json', spec)
    _write(exp / 'state.json', state or {'status': 'finished', 'verdict': 'merge_to_iteration_baseline',
                                       'metrics': {'attempted': 1, 'pairs': 1, 'failures': 0,
                                                   'identified_baseline': 1, 'identified_candidate': 0,
                                                   'net_win_confirmed': 1, 'sign_p': 1.0}})
    _write(exp / 'cases.jsonl', '\n'.join(json.dumps(r, ensure_ascii=False) for r in records or []))
    for folder in ('generators', 'judges'):
        for ref in ('base', 'candidate'):
            _write(instance / folder / ref / 'config.json', {'llm': {'model': f'{folder}-{ref}'}})
            _write(instance / folder / ref / 'persona.md', '人格 <example>')
    _write(instance / 'pointers.json', {'production_gen': 'base', 'iteration_gen': 'base',
                                      'production_judge': 'base', 'iteration_judge': 'base', 'data': 'data'})
    _write(instance / 'data/data/manifest.json', {'fewshot_pool': {'total': 50}})
    return exp, report._load_run(exp)


def test_failed_smoke_is_not_running_or_a_quality_score(tmp_path):
    state = {'status': 'running', 'verdict': 'experiment_incomplete',
             'metrics': {'attempted': 5, 'pairs': 0, 'failures': 5, 'sign_p': 1.0}}
    exp, run = _run(tmp_path, smoke=True, state=state,
                    records=[{'case_id': 'x', 'status': 'failed', 'reason': 'URL 类型错误'}])
    before = (exp / 'state.json').read_bytes()
    report.write_run(exp)
    page = (exp / 'index.html').read_text()
    assert '执行失败，待重试' in page and '暂无有效评测结果' in page
    assert 'URL 类型错误' in page and '进行中' not in page
    assert '符号检验 p 值' not in report._metrics_html(run)
    assert before == (exp / 'state.json').read_bytes()
    for filename in ('cases.jsonl', 'spec.json', 'state.json', 'bill.md'):
        assert f'href="{filename}"' not in page
    assert 'href="/dashboard/demo/index.html"' in page


def test_cases_keep_human_and_baseline_separate_and_escape_text(tmp_path):
    row = {'case_id': 'x', 'status': 'failed', 'human_reply': ['真人回复'],
           'context': [{'sender': '甲', 'text': '<script>bad()</script>'}], 'reason': '模型调用失败'}
    _, run = _run(tmp_path, records=[row])
    page = report._cases_html(run)
    assert '真人实际回复' in page and '基线生成器回复' in page
    assert page.count('未记录回复') == 2  # 真人回复不能顶替缺失的两版回复。
    assert '<script>bad()</script>' not in page
    assert '&lt;script&gt;bad()&lt;/script&gt;' in page
    assert '本题失败原因' in page


def test_final_retry_is_shown_and_previous_attempt_stays_inline(tmp_path):
    rows = [{'case_id': 'x', 'status': 'failed', 'reason': '第一次失败'},
            {'case_id': 'x', 'status': 'ok', 'identified_baseline': True, 'identified_candidate': False,
             'flip_verified': {'baseline_identified_final': False, 'candidate_identified_final': True}},
            {'case_id': 'y', 'status': 'failed', 'reason': '仍失败'}]
    _, run = _run(tmp_path, records=rows)
    page = report._cases_html(run)
    assert page.count('class="case" data-item') == 2
    assert 'data-kind="loss"' in page and '候选更易被识别' in page
    assert '第一次失败' in page and '保存记录 2 条' in page


def test_judge_cases_include_frozen_pack_input_and_correct_direction(tmp_path):
    row = {'case_id': 'judge-x', 'baseline_correct': False, 'candidate_correct': True,
           'flip_verified': True, 'baseline_identified_final': False, 'candidate_identified_final': True}
    exp, run = _run(tmp_path, kind='judge_eval', records=[row])
    _write(exp.parent.parent / 'judge_eval/calibration-pack/pack.json', {'rows': [
        {'case_id': 'judge-x', 'context': [{'sender': '甲', 'text': '问题'}],
         'human_reply': ['真人回答'], 'ai_replies': ['模型回答']} ]})
    page = report._cases_html(run)
    for text in ['候选识别更准', '基线裁判', '候选裁判', '模型回答', '真人回答', '问题']:
        assert text in page
    assert '基线生成器回复' not in page


def test_fixed_answers_never_enter_html_even_in_collapsed_sections(tmp_path):
    exp, run = _run(tmp_path, fixed=True, records=[
        {'case_id': 'secret', 'context': [{'sender': '甲', 'text': 'FIXED_SECRET'}],
         'human_reply': ['FIXED_SECRET'], 'reason': 'FIXED_SECRET', 'status': 'failed'}])
    page = report._render_run_html(run)
    assert '固定集答案不在开发后台展开' in page
    assert 'FIXED_SECRET' not in page and 'data-item' not in report._cases_html(run)
    assert 'cases.jsonl' not in page


def test_dashboard_reads_its_own_instance_and_shows_no_fake_adoption(tmp_path):
    exp, run = _run(tmp_path)
    target = tmp_path / 'dashboard/demo'
    report.refresh_dashboard(exp.parent, target)
    page = (target / 'index.html').read_text()
    assert 'generators-base' in page and 'judges-base' in page
    assert '当前配置' in page and '最近任务' in page and '实验历史' in page
    assert '组内可比' not in page and '符号检验p' not in page
    assert '当前对应指针未指向本次候选' in report._adoption_note(run)


def test_case_filters_and_pagination_have_stable_result_categories(tmp_path):
    rows = [{'case_id': f'case-{i}', 'status': 'ok', 'identified_baseline': False,
             'identified_candidate': False, 'context': [{'sender': '甲', 'text': f'第{i}条内容'}]}
            for i in range(25)]
    _, run = _run(tmp_path, records=rows)
    page = report._cases_html(run)
    assert page.count('data-kind="tie"') == 25
    assert 'data-page-size="20"' in page
    assert 'data-previous' in page and 'data-next' in page


def test_judge_reason_mirrors_direction_without_changing_saved_state():
    state = {'status': 'finished', 'verdict': 'adopt', 'reason': '识别数 40→60 严格下降'}
    assert report._reason({'kind': 'judge_eval'}, state) == '识别数 40→60 严格上升'
    assert state['reason'] == '识别数 40→60 严格下降'
    assert '不形成效果改善' in report._reason({'smoke': True}, state)
