"""Negative tests: invalid result locations and invented completion must fail."""
import json

import pytest

from src.config import ConfigError
from src.iteration import comparisons, experiment, record_contract, runner, versions
from src.iteration.storage import read_json, write_json


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    root = tmp_path / 'instance'
    monkeypatch.setattr(versions, 'PRIVATE', root)
    monkeypatch.setattr(versions, 'DATA_ROOT', root / 'data')
    data = root / 'data/d-test'
    write_json(data / 'manifest.json', {'id': 'd-test'})
    rows = [{'case_id': str(i), 'context': [], 'human_reply': ['human'], 'ai_replies': ['AI']} for i in range(2)]
    write_json(root / 'judge_eval/pack-test/pack.json', {'data_ref': 'd-test', 'c0_gen_version': 'g-test', 'rows': rows})
    return comparisons.register('comparison-test', data_ref='d-test', pack_ref='pack-test',
        comparison={'baseline': {'recipe': 'LR'}, 'candidate': {'recipe': 'GBDT'}},
        change='explicit classifier comparison', protocol={'flip_extra_rounds': 2})


def row(cid, b=True, c=True):
    value = {'case_id': str(cid), 'status': 'ok', 'baseline_correct': b, 'candidate_correct': c}
    if b != c:
        value.update(flip_verified=True, baseline_votes=[b]*3, candidate_votes=[c]*3,
                     baseline_identified_final=b, candidate_identified_final=c)
    return value


@pytest.mark.parametrize('kind', ['gen_ab', 'judge_eval'])
@pytest.mark.parametrize('failures,limit', [(0, .01), (1, .01), (9, .01), (10, .01), (11, .01), (1, .001)])
def test_development_gate_uses_frozen_failure_limit(tmp_path, kind, failures, limit):
    from src.iteration import gates, protocol
    records = []
    for cid in range(1000):
        b, c = (kind == 'gen_ab', kind == 'judge_eval') if cid < 20 else (True, True)
        record = row(cid, b, c)
        if kind == 'gen_ab':
            record['identified_baseline'] = record.pop('baseline_correct')
            record['identified_candidate'] = record.pop('candidate_correct')
            if b != c:
                record['flip_verified'] = {key: record.pop(key) for key in (
                    'baseline_votes', 'candidate_votes', 'baseline_identified_final', 'candidate_identified_final')}
        if cid >= 1000 - failures:
            record = {'case_id': str(cid), 'status': 'failed', 'reason': 'transport'}
        records.append(record)
    path = tmp_path / 'cases.jsonl'
    path.write_text('\n'.join(json.dumps(record) for record in records) + '\n')
    before = path.read_bytes()
    spec = {'kind': kind, 'protocol': {'flip_extra_rounds': 2, 'max_failure_rate': limit,
                                      'gate_schema': 3, 'dev_min_net_win_rate': 0}}
    rows = [{'case_id': str(cid)} for cid in range(1000)]
    if failures / 1000 >= limit:
        with pytest.raises(ConfigError, match='失败率达到冻结上限'):
            gates.confirmed_metrics(tmp_path, rows, spec)
    else:
        metrics = gates.confirmed_metrics(tmp_path, rows, spec)
        assert metrics['pairs'] == 1000 - failures
        assert metrics['attempted'] == 1000 and metrics['failures'] == failures
        assert metrics['failure_rate'] == failures / 1000
        assert metrics['identified_baseline'] == 1000 - failures - (20 if kind == 'judge_eval' else 0)
        assert metrics['identified_candidate'] == 1000 - failures - (20 if kind == 'gen_ab' else 0)
        assert metrics['net_win_confirmed'] == 20
        assert protocol.decide(metrics, 'development', spec['protocol'])['verdict'] == 'merge_to_iteration_baseline'
        records[0].pop('flip_verified')
        path.write_text('\n'.join(json.dumps(record) for record in records) + '\n')
        with pytest.raises(ConfigError, match='未完成独立补验'):
            gates.confirmed_metrics(tmp_path, rows, spec)
        path.write_bytes(before)
    assert path.read_bytes() == before


def test_output_cannot_escape_experiments(comparison, tmp_path):
    with pytest.raises(ConfigError, match='experiments'):
        with record_contract.writer(versions.PRIVATE / 'judge_training/wrong'):
            pytest.fail('must fail before output creation')
    assert not (versions.PRIVATE / 'judge_training').exists()
    for name in ['../escape', 'a/b', '/tmp/escape']:
        with pytest.raises(ConfigError):
            record_contract.directory(name)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (comparison.parent / 'linked').symlink_to(outside)
    with pytest.raises(ConfigError):
        record_contract.directory('linked')


def test_unregistered_and_foreign_cases_fail_before_append(comparison):
    missing = comparison.parent / 'missing'
    with pytest.raises(FileNotFoundError):
        with record_contract.writer(missing):
            pytest.fail('no spec')
    with record_contract.writer(comparison) as append:
        with pytest.raises(ConfigError):
            append(row('foreign'))
        with pytest.raises(ConfigError):
            append({'case_id': '0', 'status': 'ok'})
    assert (comparison / 'cases.jsonl').read_bytes() == b''


def test_completion_checks_all_cases_metrics_and_votes(comparison):
    with record_contract.writer(comparison) as append:
        append(row(0))
    with pytest.raises(ConfigError, match='不完整'):
        comparisons.complete(comparison)
    with record_contract.writer(comparison) as append:
        append(row(1, False, True))
    with pytest.raises(ConfigError, match='汇总不一致'):
        experiment.finish(comparison, {'pairs': 100}, {'verdict': 'diagnostic_complete', 'reason': 'invented'})
    metrics = comparisons.complete(comparison)
    assert metrics['pairs'] == 2 and metrics['net_win_confirmed'] == 1
    assert read_json(comparison / 'state.json')['status'] == 'finished'
    with pytest.raises(ConfigError, match='已完成'):
        with record_contract.writer(comparison):
            pytest.fail('cannot change results')
    with pytest.raises(ConfigError, match='不可改写'):
        experiment.finish(comparison, metrics, {'verdict': 'adopt', 'reason': 'changed'})


def test_successful_checkpoint_cannot_be_replaced(comparison):
    with record_contract.writer(comparison) as append:
        append(row(0))
        append(row(0))
        with pytest.raises(ConfigError, match='成功断点'):
            append(row(0, False, True))
    assert len((comparison / 'cases.jsonl').read_text().splitlines()) == 1


def test_comparison_blocks_ordinary_execution_and_promotion(comparison):
    from src.iteration import gates, promote
    for action in (lambda: runner.run_judge_experiment(comparison), lambda: promote.promote_judge(comparison.name),
                   lambda: gates.require_promotable(experiment.spec_of(comparison))):
        with pytest.raises(ConfigError, match='专项'):
            action()


def test_comparison_visible_in_history_with_real_parameters(comparison):
    from src.dashboard import report
    with record_contract.writer(comparison) as append:
        append(row(0))
        append(row(1, False, True))
    comparisons.complete(comparison)
    page = report._render_run_html(report._load_run(comparison))
    assert 'LR' in page and 'GBDT' in page
    assert '专项判别比较' in report.dashboard_html(comparison.parent)
    assert '2 / 2' in report.live_payload(versions.PRIVATE)['regions']['history']


def test_incomplete_votes_and_invented_net_win_are_rejected(comparison):
    with pytest.raises(ConfigError, match='NaN'):
        experiment.finish(comparison, {'failure_rate': float('nan')},
                          {'verdict': 'diagnostic_complete', 'reason': 'invalid'})
    incomplete = row(1, False, True)
    incomplete['candidate_votes'] = [True]
    with record_contract.writer(comparison) as append:
        append(row(0))
        append(incomplete)
    with pytest.raises(ConfigError, match='补验'):
        comparisons.complete(comparison)


def test_tampered_pack_or_symlink_cannot_redirect_registered_results(comparison, tmp_path):
    other = tmp_path / 'other'
    other.write_text('unchanged')
    (comparison / 'cases.jsonl').symlink_to(other)
    with pytest.raises(ConfigError, match='符号链接'):
        runner.run_judge_experiment(comparison)
    assert other.read_text() == 'unchanged'
    (comparison / 'cases.jsonl').unlink()
    pack = versions.PRIVATE / 'judge_eval/pack-test/pack.json'
    value = read_json(pack)
    value['rows'][0]['case_id'] = 'different'
    write_json(pack, value)
    with pytest.raises(ConfigError, match='回复包已变化'):
        with record_contract.writer(comparison):
            pytest.fail('must not append')


def legacy_source(comparison):
    from src.config import sha256_file
    from src.iteration import gates
    source = versions.PRIVATE / 'judge_training/legacy'
    definition = {'id': 'luna-gbdt-vs-lr', 'baseline': {'arm': 'luna', 'recipe': 'lr', 'policy': 'pure'},
                  'candidate': {'arm': 'luna', 'recipe': 'gbdt', 'policy': 'pure'}}
    write_json(source / 'spec.json', {'id': 'legacy', 'dataset': 'development', 'adoption_allowed': False,
        'data_ref': 'd-test', 'protocol': {'flip_extra_rounds': 2}, 'recipes': [
            {'id': 'lr', 'C': 1}, {'id': 'gbdt', 'depth': 2}], 'comparisons': [definition]})
    write_json(source / 'state.json', {'status': 'finished', 'updated_at': '2026-01-01 00:00:00'})
    write_json(source / 'evaluation_source.json', {'source_experiment': str(comparison),
        'pack_sha256': sha256_file(versions.PRIVATE / 'judge_eval/pack-test/pack.json')})
    directory = source / 'comparisons' / definition['id']
    directory.mkdir(parents=True)
    (directory / 'cases.jsonl').write_text('\n'.join(json.dumps(r) for r in [row(0), row(1, False, True)]) + '\n')
    metrics = gates.confirmed_metrics(directory, [{'case_id': '0'}, {'case_id': '1'}],
                                      {'kind': 'judge_eval', 'protocol': {'flip_extra_rounds': 2}})
    result = {**definition, 'metrics': metrics}
    write_json(directory / 'result.json', result)
    write_json(source / 'results.json', {'comparisons': [result]})
    write_json(source / 'acceptance.json', {'status': 'passed', 'comparisons': [result],
        'evidence_hashes': {str(p): sha256_file(p) for p in [source / 'spec.json',
            directory / 'result.json', directory / 'cases.jsonl']}})
    return source


def test_legacy_import_preserves_source_and_is_idempotent(comparison):
    source = legacy_source(comparison)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    preview = comparisons.import_legacy('legacy')
    assert preview['comparisons'] == 1
    result = comparisons.import_legacy('legacy', apply=True)
    assert comparisons.import_legacy('legacy', apply=True) == result
    target = versions.PRIVATE / 'experiments' / result['experiment_ids'][0]
    assert read_json(target / 'state.json')['metrics']['net_win_confirmed'] == 1
    assert read_json(target / 'spec.json')['provenance']['original_executor_certified'] is False
    assert all(p.read_bytes() == blob for p, blob in before.items())
    from scripts.check_rules import check_instance
    assert check_instance(versions.PRIVATE) == []


def test_legacy_tampering_fails_before_any_experiment_creation(comparison):
    source = legacy_source(comparison)
    path = source / 'comparisons/luna-gbdt-vs-lr/cases.jsonl'
    path.write_text(path.read_text().replace('false', 'true'))
    before = set(comparison.parent.iterdir())
    with pytest.raises(ConfigError, match='证据已变化'):
        comparisons.import_legacy('legacy', apply=True)
    assert set(comparison.parent.iterdir()) == before


def test_paused_legacy_comparisons_remain_paused(comparison):
    source = legacy_source(comparison)
    write_json(source / 'state.json', {'status': 'paused'})
    result = comparisons.import_legacy('legacy', apply=True)
    target = versions.PRIVATE / 'experiments' / result['experiment_ids'][0]
    assert read_json(target / 'state.json')['status'] == 'paused'
    assert not (target / 'cases.jsonl').exists()
