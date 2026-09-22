"""报告的真实数据边界、结果语义和页内浏览；全部使用临时实验。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard import report


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


def test_threshold_review_display_keeps_original_decision(tmp_path):
    from src.config import sha256_file
    exp, run = _run(tmp_path, fixed=True, state={
        'status': 'finished', 'verdict': 'reject',
        'metrics': {'attempted': 1000, 'pairs': 1000, 'net_win_confirmed': 9}})
    original = (exp / 'state.json').read_bytes()
    _write(exp / 'threshold_review.json', {
        'original_verdict': 'reject', 'protocol': {'formal_min_net_win_rate': .005},
        'decision': {'verdict': 'adopt', 'reason': '确认净胜 +9 ≥ +5'},
        'evidence_sha256': {name: sha256_file(exp / name) for name in ('spec.json', 'state.json')}})
    run = report._load_run(exp)
    assert '固定验收达标' in report._history_conclusion(run)
    assert '0.5%' in report._history_conclusion(run)
    assert '原协议结论：reject' in report._history_conclusion(run)
    report.write_run(exp)
    assert '按修订门槛复核' in (exp / 'index.html').read_text()
    assert (exp / 'state.json').read_bytes() == original


def test_cases_keep_human_and_baseline_separate_and_escape_text(tmp_path):
    row = {'case_id': 'x', 'status': 'failed', 'human_reply': ['真人回复'],
           'context': [{'sender': '甲', 'text': '<script>bad()</script>'}], 'reason': '模型调用失败'}
    _, run = _run(tmp_path, records=[row])
    page = report._cases_html(run)
    assert '真人实际回复' in page and '对照生成器回复' in page
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
    assert page.count('href="#case-x"') == 1  # 重试记录共用同一题的稳定链接。
    assert page.count('data-copy-case-link') == 2
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
    for text in ['候选识别更准', '对照裁判', '候选裁判', '模型回答', '真人回答', '问题']:
        assert text in page
    assert '对照生成器回复' not in page


def test_fixed_answers_never_enter_html_even_in_collapsed_sections(tmp_path):
    exp, run = _run(tmp_path, fixed=True, records=[
        {'case_id': 'secret', 'context': [{'sender': '甲', 'text': 'FIXED_SECRET'}],
         'human_reply': ['FIXED_SECRET'], 'reason': 'FIXED_SECRET', 'status': 'failed'}])
    page = report._render_run_html(run)
    assert '固定集答案不在开发后台展开' in page
    assert 'FIXED_SECRET' not in page and 'data-item' not in report._cases_html(run)
    assert 'data-case-link' not in report._cases_html(run)
    assert 'data-copy-case-link' not in report._cases_html(run)
    assert 'cases.jsonl' not in page


def test_dashboard_reads_its_own_instance_and_shows_no_fake_adoption(tmp_path):
    exp, run = _run(tmp_path)
    target = tmp_path / 'dashboard/demo'
    report.refresh_dashboard(exp.parent, target)
    page = (target / 'index.html').read_text()
    assert '当前基线' in page and '实验记录' in page
    assert 'data-live-region="task"' not in page and 'id="mechanism"' not in page
    assert '组内可比' not in page and '符号检验p' not in page
    detail = report._render_run_html(report._load_run(exp))
    assert 'generators-base' in detail and 'judges-base' in detail
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


def _history_state(net=8, verdict='diagnostic_complete'):
    return {'status': 'finished', 'verdict': verdict, 'metrics': {
        'attempted': 1000, 'pairs': 1000, 'failures': 0, 'failure_rate': 0,
        'identified_baseline': 905, 'identified_candidate': 919,
        'wins_confirmed': net + 3, 'losses_confirmed': 3,
        'net_win_confirmed': net, 'contested': 24}}


def test_historical_recipe_shows_development_gain_without_claiming_adoption(tmp_path):
    exp, run = _run(tmp_path, kind='judge_eval', state=_history_state())
    run['spec'].update(comparison={
        'baseline': {'arm': 'luna', 'recipe': 'lr_l2_c1', 'policy': 'hybrid'},
        'candidate': {'arm': 'luna', 'recipe': 'lr_l2_c1', 'policy': 'pure'}},
        protocol={'gate_schema': 2}, adoption_allowed=False, generator_ref='generator')
    run['spec'].pop('candidate_ref')
    run['spec'].pop('baseline_ref')
    _write(exp / 'spec.json', run['spec'])
    before = (exp / 'state.json').read_bytes()
    page = report._history_html([run])
    for value in ['确认净胜', '晋级结论', '实际采用（当前）', '+8', '919/1000',
                  '开发收益达标', '未绑定候选版本', 'lr_l2_c1', '纯分类器']:
        assert value in page
    assert '开发晋级达标' not in page
    versions = report._history_versions(run)
    assert '<dt>本轮对照</dt><dd>luna · lr_l2_c1 · 初判与分类器合并</dd>' in versions
    assert '<dt>本轮候选</dt><dd>luna · lr_l2_c1 · 纯分类器</dd>' in versions
    assert '初测净胜 <strong>+14</strong>' in page
    assert '补验后：胜 11 / 负 3 · 打平 24' in page
    # Both static and live history use the same conclusion, with no state mutation.
    live = report.live_payload(exp.parent.parent)
    assert live['regions']['history'] == page
    assert page in report.dashboard_html(exp.parent)
    assert (exp / 'state.json').read_bytes() == before


@pytest.mark.parametrize('kind,initial,confirmed', [
    ('judge_eval', '-2', 5),
    ('gen_ab', '+2', -5),
])
def test_history_keeps_initial_accuracy_and_confirmed_votes_separate(tmp_path, kind, initial, confirmed):
    state = _history_state(confirmed)
    state['metrics'].update(identified_baseline=932, identified_candidate=930,
                            wins_confirmed=15 if confirmed > 0 else 10,
                            losses_confirmed=10 if confirmed > 0 else 15, contested=23)
    exp, run = _run(tmp_path, kind=kind, state=state)
    before = (exp / 'state.json').read_bytes()
    rates, net = report._history_rates(run), report._history_net(run)
    assert '93.2%' in rates and '932/1000' in rates
    assert '93.0%' in rates and '930/1000' in rates
    assert f'初测净胜 <strong>{initial}</strong>' in rates
    assert f'<strong>{confirmed:+d}</strong>' in net
    assert '打平 23' in net and '未确认' not in net
    assert '初测与补验方向不同' in net
    page = report._history_html([run])
    assert '所有净胜均为候选相对本轮对照' in page
    assert '不一定是当前基线' in page
    assert report.live_payload(exp.parent.parent)['regions']['history'] == page
    assert (exp / 'state.json').read_bytes() == before


@pytest.mark.parametrize('missing', ['identified_baseline', 'identified_candidate', 'no_pairs'])
def test_history_does_not_invent_initial_net_when_counts_are_missing(tmp_path, missing):
    _, run = _run(tmp_path, kind='judge_eval', state=_history_state())
    if missing == 'no_pairs':
        run['pairs'] = 0
    else:
        run['metrics'].pop(missing)
    assert '初测净胜 <strong>—</strong>' in report._history_rates(run)
    assert '初测与补验方向不同' not in report._history_net(run)


def test_history_comparison_identities_are_escaped(tmp_path):
    _, run = _run(tmp_path, kind='judge_eval')
    run['spec']['comparison'] = {
        'baseline': {'arm': 'features', 'recipe': '<base>', 'policy': 'pure'},
        'candidate': {'arm': 'features', 'recipe': '<candidate>', 'policy': 'pure'}}
    versions = report._history_versions(run)
    assert '<dt>本轮对照</dt><dd>features · &lt;base&gt; · 纯分类器</dd>' in versions
    assert '<dt>本轮候选</dt><dd>features · &lt;candidate&gt; · 纯分类器</dd>' in versions
    assert '<base>' not in versions and '<candidate>' not in versions


@pytest.mark.parametrize('net,protocol,label', [
    (0, {'gate_schema': 2}, '开发未达标'),
    (8, {'gate_schema': 1, 'dev_min_net_win_rate': .01}, '开发未达标'),
    (8, {'gate_schema': 2}, '开发收益达标'),
])
def test_history_uses_frozen_development_gate(tmp_path, net, protocol, label):
    _, run = _run(tmp_path, kind='judge_eval', state=_history_state(net))
    run['spec'].update(adoption_allowed=False, protocol=protocol)
    assert label in report._history_conclusion(run)


def test_history_separates_pass_from_actual_pointer_and_fixed_rejection(tmp_path):
    exp, run = _run(tmp_path, kind='judge_eval', state=_history_state(13, 'merge_to_iteration_baseline'))
    assert '开发晋级达标' in report._history_conclusion(run)
    assert '候选：当前未采用' in report._history_adoption(run)
    ptr_path = exp.parent.parent / 'pointers.json'
    ptr = json.loads(ptr_path.read_text())
    ptr['iteration_judge'] = 'candidate'
    _write(ptr_path, ptr)
    assert '候选：当前开发版' in report._history_adoption(run)
    assert '对照：当前生产版' in report._history_adoption(run)
    run['spec']['dataset'] = 'fixed_test'
    run['state']['verdict'] = 'reject'
    run['metrics']['net_win_confirmed'] = 9
    assert '固定验收未达标' in report._history_conclusion(run)
    assert '开发晋级达标' not in report._history_conclusion(run)
    assert '候选：当前生产版' not in report._history_adoption(run)


def test_history_does_not_promote_incomplete_or_smoke_results(tmp_path):
    _, run = _run(tmp_path, kind='judge_eval', state=_history_state(13, 'experiment_incomplete'))
    assert '暂无结论' in report._history_conclusion(run)
    run['spec']['smoke'] = True
    assert '不参与晋级' in report._history_conclusion(run)


@pytest.mark.parametrize('accepted_state', ['current', 'missing', 'other_candidate', 'stale_basis'])
def test_history_reads_branch_development_adoption_without_changing_shared_pointers(tmp_path, accepted_state):
    exp, run = _run(tmp_path, kind='judge_eval', state=_history_state(8, 'merge_to_iteration_baseline'))
    ptr_path = exp.parent.parent / 'pointers.json'
    before = ptr_path.read_bytes()
    ptr = json.loads(before)
    basis = {key: ptr[key] for key in ('data', 'production_gen', 'production_judge')}
    run['spec'].update(branch_round='pure-lr/r-0001', production_basis=basis)
    _write(exp / 'spec.json', run['spec'])
    accepted = {'candidate_ref': 'candidate', 'experiment_id': exp.name, 'basis': dict(basis)}
    if accepted_state == 'other_candidate':
        accepted['candidate_ref'] = 'newer'
    elif accepted_state == 'stale_basis':
        accepted['basis']['data'] = 'old-data'
    if accepted_state != 'missing':
        _write(exp.parent.parent / 'branches/pure-lr/state.json', {'development': accepted})
    adoption, note = report._history_adoption(run), report._adoption_note(run)
    if accepted_state == 'current':
        assert '候选：分支 pure-lr 当前开发版' in adoption
        assert '候选已保留为分支 pure-lr 的当前开发版' in note
    else:
        assert '候选：当前未采用' in adoption
        assert '候选已保留' not in note
    assert '对照：当前生产版' in adoption
    assert '已推全' not in note
    page = report._history_html([run])
    assert report.live_payload(exp.parent.parent)['regions']['history'] == page
    assert ptr_path.read_bytes() == before


def test_history_status_filter_marks_promoted_failed_and_running(tmp_path):
    """状态筛选桶与已晋升标记：已晋升只认指针/凭证硬证据；verdict 达标不算晋升。"""
    exp, run = _run(tmp_path, state=_history_state(8, 'merge_to_iteration_baseline'))
    page = report._history_html([run])
    assert 'data-status="passed"' in page and 'data-promoted="false"' in page
    assert '>已晋升</span>' not in page
    # 指针指向候选 → 已晋升（硬证据）
    ptr_path = exp.parent.parent / 'pointers.json'
    ptr = json.loads(ptr_path.read_text())
    ptr['iteration_judge'] = 'candidate'
    _write(ptr_path, ptr)
    page = report._history_html([run])
    assert 'data-promoted="true"' in page and '>已晋升</span>' in page
    ptr['iteration_judge'] = 'base'
    _write(ptr_path, ptr)
    # 分支推全凭证 → 已晋升
    ptr['branch_promotions'] = {'test-run': 'candidate'}
    _write(ptr_path, ptr)
    assert 'data-promoted="true"' in report._history_html([run])
    ptr.pop('branch_promotions')
    _write(ptr_path, ptr)
    run['state']['verdict'] = 'reject'
    page = report._history_html([run])
    assert 'data-status="done"' in page and 'data-promoted="false"' in page
    run['state']['verdict'] = 'experiment_incomplete'
    page = report._history_html([run])
    assert 'data-status="failed"' in page and 'data-promoted="false"' in page
    run['state'] = {'status': 'running'}
    page = report._history_html([run])
    assert 'data-status="running"' in page
    # 对象与规模筛选属性
    assert 'data-object="gen"' in page
    run['spec']['kind'] = 'judge_eval'
    run['state'] = _history_state(8, 'merge_to_iteration_baseline')
    run['attempted'] = 7
    page = report._history_html([run])
    assert 'data-object="judge"' in page and 'data-size="small"' in page
    run['attempted'] = 1000
    assert 'data-size="full"' in report._history_html([run])


def test_baseline_summary_shows_branch_baselines_and_stage_limit(tmp_path):
    """分支保留的开发版显示为分支基线；stage_limit 挡住的分支给出解除命令。"""
    exp, _ = _run(tmp_path)
    instance = exp.parent.parent
    _write(instance / 'branches/gen-conversation/state.json', {
        'rounds': ['r-0003'], 'stage_limit': 'development',
        'development': {'candidate_ref': 'g-0020', 'experiment_id': 'branch-gen-conversation-r-0003-dev'}})
    _write(instance / 'branches/luna/state.json', {
        'rounds': ['r-0001'],
        'development': {'candidate_ref': 'j-0019', 'experiment_id': 'branch-luna-r-0001-dev'}})
    page = report._baseline_summary(instance)
    assert '分支基线' in page and 'gen-conversation' in page
    assert 'g-0020' in page and 'j-0019' in page
    assert 'limit --stage fixed_test gen-conversation' in page
    assert '🔒' in page
