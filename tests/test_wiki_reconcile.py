"""Cross-batch attribute reconciliation with synthetic source data."""
import json

import pytest

from src.bootstrap import wiki_reconcile as reconcile, wiki_structured as structured
from test_wiki_structured import META, fixture


def prepared(tmp_path):
    rows, coverage, value = fixture(tmp_path)
    common = {k: value['addresses'][0][k] for k in structured.COMMON}
    value['attributes'] = [dict(common, subject_ref='E0', field='occupation', value='建筑师'),
                           dict(common, subject_ref='E0', field='occupation', value='曾计划当建筑师', lines=[2]),
                           dict(common, subject_ref='E1', field='occupation', value='编辑')]
    return rows, structured.compile_records(META, [(value, {'F1': 'fact-1'})], rows, coverage)


def decision(record, kind='correct'):
    return dict(record_id=record['id'], decision=kind, reason='原记录将职业意向当作经历。',
        value='曾计划从事设计工作', lines=[2], **{k: record[k] for k in reconcile.FIELDS if k != 'lines'})


def test_grouping_never_merges_subjects_and_application_preserves_sources(tmp_path):
    rows, data = prepared(tmp_path)
    groups = reconcile.groups_for(data)
    assert len(groups) == 1 and len(groups[0]['records']) == 2
    assert groups[0]['lines'] == [1, 2]
    first, second, other = data['attributes']
    reviews = [decision(first), decision(second, 'omit')]
    reconcile.validate({'reviews': reviews}, groups[0]['records'], rows)
    result = reconcile.apply_reviews(data, reviews, rows)
    assert result['attributes'][0]['reconciled_from'] == first['id']
    assert result['attributes'][0]['id'] != first['id']
    assert result['attributes'][0]['evidence_refs'] == ['2']
    assert result['attributes'][1] == other
    assert result['attribute_reconciliation']['counts'] == {'correct': 1, 'omit': 1}
    assert result['attribute_reconciliation']['unreviewed_attributes'] == 1
    assert all('text' not in e for e in result['evidence'].values())


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unknown', 'line', 'uncertainty', 'date'])
def test_bad_review_is_rejected(tmp_path, mutation):
    rows, data = prepared(tmp_path)
    records = reconcile.groups_for(data)[0]['records']
    reviews = [decision(r) for r in records]
    if mutation == 'missing':
        reviews.pop()
    elif mutation == 'duplicate':
        reviews.append(reviews[0])
    elif mutation == 'unknown':
        reviews[0]['record_id'] = 'unknown'
    elif mutation == 'line':
        reviews[0]['lines'] = [999]
    elif mutation == 'uncertainty':
        reviews[0]['decision'] = 'unresolved'
    else:
        reviews[0].update(valid_from='2025-01-01', valid_to='2024-01-01')
    with pytest.raises(ValueError):
        reconcile.validate({'reviews': reviews}, records, rows)


def test_reconciliation_reuses_cache_and_never_sends_project_context(tmp_path):
    rows, data = prepared(tmp_path)

    class Client:
        calls = 0

        def cache_identity(self):
            return {'test': 'reconcile'}

        def run(self, prompt, schema):
            self.calls += 1
            payload = json.loads(prompt[len(reconcile.PROMPT):])
            assert set(payload) == {'subject', 'entities', 'records', 'columns', 'messages'}
            return json.dumps({'reviews': [decision(r, 'keep') for r in payload['records']]})

    client = Client()
    progress = dict(failed=0, errors=[])
    reconcile.reconcile(client, tmp_path / 'cache', data, rows, 1, progress, lambda: None)
    assert client.calls == 1
    reconcile.reconcile(client, tmp_path / 'cache', data, rows, 1, progress, lambda: None)
    assert client.calls == 1 and progress['reconciliation_completed'] == 1


@pytest.mark.parametrize('kind', ['keep', 'correct'])
def test_review_cannot_retain_or_introduce_wrong_self_report_author(tmp_path, kind):
    rows, data = prepared(tmp_path)
    record = data['attributes'][0]
    record['mode'] = 'self_report' if kind == 'keep' else 'observed'
    review = decision(record, kind)
    review.update(lines=[1], mode='self_report')
    with pytest.raises(ValueError, match='self_report_without_source_author'):
        reconcile.validate({'reviews': [review]}, [record], rows, data)
    review.update(decision='correct', lines=[2])
    reconcile.validate({'reviews': [review]}, [record], rows, data)
    review.update(lines=[1], mode='reported')
    reconcile.validate({'reviews': [review]}, [record], rows, data)
