"""Diagnostics explain the serving choices without new feature requests."""
import pytest

pytest.importorskip('torch')
pytest.importorskip('xgboost')
from src.generator import learned_selection, selection_analysis
from test_learned_fewshot import example
from test_fewshot_selection import Renderer


def test_diagnostics_preserve_choices_and_explain_budget_duplicates_and_count():
    first, second = example('a'), example('b', 2)
    second['reply'] = ['different']
    oversized = example('long')
    oversized['reply'] = ['x' * 5000]
    duplicate = dict(first, id='copy')
    rows, scores = [oversized, first, duplicate, second], [4, 3, 2, 1]
    plain = learned_selection.choose(rows, scores, Renderer(), count=1, budget=1000)
    decisions = []
    diagnosed = learned_selection.choose(rows, scores, Renderer(), count=1, budget=1000,
        decisions=decisions)
    assert plain == diagnosed == [first]
    assert [row['decision'] for row in decisions] == ['budget', 'selected', 'duplicate', 'count_limit']
    assert decisions[1]['combined_chars'] <= 1000


def test_missing_cached_feature_is_reported_without_scoring_or_extraction(monkeypatch):
    class Selector:
        model, cache = {}, None
        def tasks(self, case, rows):
            return {'missing': {}}, {}
    monkeypatch.setattr(selection_analysis.extraction, 'cached', lambda *args: None)
    monkeypatch.setattr(selection_analysis.extraction, 'extract_one',
        lambda *args: pytest.fail('Diagnostic must not request features'))
    result = selection_analysis.analyze_case(dict(case_id='target', chat_type='private'),
        [example('a')], Selector(), Renderer(), count=3, budgets=[2500, 5000])
    assert result['status'] == 'missing_cached_features'
    assert result['missing_feature_tasks'] == 1 and result['budgets'] == {}


def test_summary_distinguishes_changed_ids_from_selected_count():
    def choice(ids, decisions):
        return dict(selected_ids=ids, selected_count=len(ids), rendered_chars=100,
            decisions=[dict(decision=value) for value in decisions])
    records = [dict(status='complete', budgets={
        '2500': choice(['a'], ['budget', 'selected']),
        '5000': choice(['b'], ['selected', 'count_limit'])}),
        dict(status='missing_cached_features')]
    summary = selection_analysis.summarize(records, [2500, 5000])
    assert summary['complete'] == 1 and summary['missing_cached_features'] == 1
    assert summary['budgets']['2500']['cases_with_budget_rejections'] == 1
    assert summary['budgets']['5000']['selected_count_distribution'] == {1: 1}
    assert summary['budgets']['5000']['changed_from_first_budget'] == 1


def test_optional_concern_keeps_original_tasks_and_adds_only_context_requests():
    from src.generator.fewshot_ranker import concern
    case = dict(example('target', 10), case_id='target', human_reply=['FORBIDDEN_TARGET_ANSWER'])
    rows = [example('a')]
    selector = learned_selection.LearnedSelector.__new__(learned_selection.LearnedSelector)
    selector.proof = {'feature_identity': {'model': 'frozen'}}
    selector.model = {}
    original_tasks, original_refs = selector.tasks(case, rows)
    selector.model = {'feature_transform': concern.TRANSFORM}
    tasks, refs = selector.tasks(case, rows)
    assert all(tasks[key] == value for key, value in original_tasks.items())
    assert all(refs[key][:3] == value and len(refs[key]) == 5 for key, value in original_refs.items())
    assert len(tasks) == len(original_tasks) + 2
    assert all('FORBIDDEN_TARGET_ANSWER' not in request['prompt'] for request in tasks.values())
