"""机制与改动说明按版本证据展示，不混淆本轮候选和跨任务变化。"""
from pathlib import Path

from src.dashboard.changes import snapshot_delta
from src.dashboard import report
from test_report import _run, _write


def _version(instance: Path, folder, ref, config, prompt='提示词'):
    path = instance / folder / ref
    _write(path / 'config.json', config)
    _write(path / ('prompt.md' if folder == 'judges' else 'persona.md'), prompt)
    return path


def test_data_marker_change_does_not_claim_new_generation_strategy(tmp_path):
    _version(tmp_path, 'generators', 'old', {'llm': {'model': 'same'}, 'data_version': 'd1'})
    _version(tmp_path, 'generators', 'new', {'llm': {'model': 'same'}, 'data_version': 'd2'})
    result = snapshot_delta(tmp_path, 'generators', 'old', 'new')
    assert result['available'] and not result['changes']
    assert result['metadata'][0]['before'] == 'd1' and result['metadata'][0]['after'] == 'd2'


def test_same_model_detects_prompt_and_small_model_weight_changes(tmp_path):
    cfg = {'mode': 'corrected_pairwise', 'llm': {'model': 'same'}, 'assets': {'correction.json': 'declared'}}
    old = _version(tmp_path, 'judges', 'old', cfg)
    new = _version(tmp_path, 'judges', 'new', cfg, '新提示词')
    _write(old / 'correction.json', {'weights': [1]})
    _write(new / 'correction.json', {'weights': [2]})
    result = snapshot_delta(tmp_path, 'judges', 'old', 'new')
    assert {c['label'] for c in result['changes']} == {'裁判提示词', '小模型权重与特征定义'}


def test_missing_snapshot_is_not_reported_as_unchanged(tmp_path):
    _version(tmp_path, 'judges', 'old', {'llm': {'model': 'same'}})
    assert not snapshot_delta(tmp_path, 'judges', 'old', 'missing')['available']
    _version(tmp_path, 'judges', 'new', {'llm': {'model': 'same'}, 'assets': {'correction.json': 'missing'}})
    assert not snapshot_delta(tmp_path, 'judges', 'old', 'new')['available']


def test_mechanism_lives_in_run_detail_and_home_stays_baseline_plus_history(tmp_path):
    exp, _ = _run(tmp_path, smoke=True)
    instance = exp.parent.parent
    cfg = {'mode': 'corrected_pairwise', 'llm': {'model': 'luna', 'provider': 'codex_cli'}, 'correction_threshold': .7}
    path = _version(instance, 'judges', 'base', cfg)
    _write(path / 'correction.json', {'final_model': {'input_feature_count': 524}})
    home = report.dashboard_html(exp.parent)
    assert '当前基线' in home and '实验记录' in home and '这次迭代了什么' in home and '具体改动' in home
    assert 'data-live-region="task"' not in home and 'id="mechanism"' not in home
    detail = report._render_run_html(report._load_run(exp))
    for phrase in ['大模型抽特征 + 小模型预测', '① 大模型初判 + 抽特征', '② 小模型预测', '524 维', '置信度 ≥ 70%']:
        assert phrase in detail
    legacy = report._judge_pipeline({'mode': 'pairwise_llm', 'llm': {'model': 'old'}}, path)
    assert '未配置特征提取与小模型校正' in legacy and '冻结权重' not in legacy


def test_local_candidate_diff_is_separate_from_previous_task_judge_change(tmp_path):
    exp, _ = _run(tmp_path, smoke=True)
    instance = exp.parent.parent
    for ref, data in [('base', 'd1'), ('candidate', 'd2')]:
        _version(instance, 'generators', ref, {'llm': {'model': 'same'}, 'data_version': data})
    _version(instance, 'judges', 'old-judge', {'mode': 'pairwise_llm', 'llm': {'model': 'old'}})
    _version(instance, 'judges', 'base', {'mode': 'corrected_pairwise', 'llm': {'model': 'luna'}, 'correction_threshold': .7})
    old = {**report.experiment.spec_of(exp), 'id': 'previous', 'created': '2026-09-11 12:00:00', 'judge_ref': 'old-judge'}
    _write(exp.parent / 'previous/spec.json', old)
    run = report._load_run(exp)
    assert report._change_summary(run) == '两版策略相同 · 连接验证'
    page = report._iteration_html(run)
    assert '本轮对比' in page and '模型行为配置、提示词与快照资产一致' in page
    assert '生成器快照的数据标记' in page and 'd1' in page and 'd2' in page
    assert '评分裁判 · old-judge → base' in page
    assert '单一大模型配对盲测' in page and '小模型预测与校正' in page
    assert '这不是本轮候选的改动' in page


def test_history_disclosures_have_distinct_keys_and_escape_change_values(tmp_path):
    exp, _ = _run(tmp_path)
    instance = exp.parent.parent
    _version(instance, 'generators', 'candidate', {'llm': {'model': '<script>attack()</script>'}})
    old = {**report.experiment.spec_of(exp), 'id': 'older', 'created': '2026-09-11'}
    _write(exp.parent / 'older/spec.json', old)
    page = report._history_html(report._runs(exp.parent))
    assert 'data-detail-key="task-change-test-run"' in page
    assert 'data-detail-key="task-change-older"' in page
    assert '<script>attack()</script>' not in page and '&lt;script&gt;attack()&lt;/script&gt;' in page


def test_separate_feature_model_and_effort_are_visible(tmp_path):
    cfg = {'mode': 'corrected_pairwise', 'llm': {'model': 'gpt-5.6-luna', 'reasoning_effort': 'low'},
           'feature_llm': {'model': 'gpt-5.6-sol', 'reasoning_effort': 'high'}, 'correction_threshold': .7}
    page = report._judge_pipeline(cfg, tmp_path)
    for text in ['① 大模型初判', 'gpt-5.6-luna · low', '② 大模型抽特征', 'gpt-5.6-sol · high']:
        assert text in page
