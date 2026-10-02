from types import SimpleNamespace

import pytest

from scripts import check
from src.config import ConfigError
from src.iteration import reply_model_report as report
from src.iteration.storage import write_json


BASELINE = 'baseline-model'
CANDIDATE = 'candidate-model'
TRACE = 'a' * 32
SOURCE_TRACE = 'b' * 32


def call(model, milliseconds, status='ok'):
    return dict(kind='llm', request=dict(body=dict(model=model)),
        status=status, elapsed_ms=milliseconds)


def cache_event(kind, layer, key='request', origin=None):
    data = dict(layer=layer, key=key)
    if origin is not None:
        data['origin'] = origin
    return dict(kind=kind, data=data)


def generation(side, events, wall=100):
    return dict(kind='generation', branch=side, round=0, status='ok',
        elapsed_ms=wall, events=events)


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    directory = tmp_path / 'experiments/current'
    spec = dict(kind='gen_ab', dataset='development', data_ref='d-0001',
        baseline_ref='g-0001', candidate_ref='g-0002', judge_ref='j-0001',
        data_snapshot=dict(case_count=2))
    write_json(directory / 'spec.json', spec)
    write_json(directory / 'state.json', dict(status='running'))
    for ref, model in [('g-0001', BASELINE), ('g-0002', CANDIDATE)]:
        write_json(tmp_path / 'generators' / ref / 'config.json',
            dict(llm={chat: dict(model=model) for chat in ('group', 'private')}))
    row = dict(case_id='case', status='ok', chat_type='private', trace_ref=TRACE,
        identified_baseline=True, identified_candidate=False,
        flip_verified=dict(baseline_identified_final=True, candidate_identified_final=False))
    rows = [row]
    monkeypatch.setattr(report.protocol, 'summarize_final_records',
        lambda path: ({item['case_id']: item for item in rows}, 0))

    def trace(name, reference, operations, case_id='case', data_ref='d-0001', dataset='development'):
        write_json(tmp_path / 'experiments' / name / 'traces' / (reference + '.json'),
            dict(case_id=case_id, trace_ref=reference, versions=dict(data_ref=data_ref, dataset=dataset),
                 operations=operations))

    trace('current', TRACE, [generation('baseline', [call(BASELINE, 1000)]),
        generation('candidate', [call(CANDIDATE, 400)])])
    return SimpleNamespace(directory=directory, rows=rows, spec=spec, trace=trace,
        origin=dict(experiment_id='previous', trace_ref=SOURCE_TRACE, operation_index=0))


def test_current_calls_empty_cohort_and_no_zero_latency(comparison):
    result = report.summarize(comparison.directory)
    quality = result['cohorts']['private']['quality']
    assert quality['net_win_confirmed'] == 1
    assert quality['identification_delta_percentage_points'] == -100
    assert quality['confirmation_evidence_validated'] is False
    speed = result['cohorts']['private']['speed']
    assert speed['paired']['mean_model_time_reduction_percentage'] == 60
    assert speed['candidate']['current_calls']['attempts'] == 1
    empty = result['cohorts']['group']
    assert empty['quality']['identification_rate']['baseline'] is None
    assert empty['speed']['paired']['mean_model_time_reduction_percentage'] is None
    assert result['complete'] is False


def test_generation_cache_follows_historical_calls_and_includes_failed_attempts(comparison):
    comparison.trace('previous', SOURCE_TRACE, [generation('candidate',
        [call(BASELINE, 400, 'failed'), call(BASELINE, 600)])])
    comparison.trace('current', TRACE, [generation('baseline',
        [cache_event('cache_hit', 'generation', origin=comparison.origin)], wall=2),
        generation('candidate', [call(CANDIDATE, 500)])])
    baseline = report.summarize(comparison.directory)['cohorts']['all']['speed']['baseline']
    assert baseline['model_time']['mean_ms'] == 1000
    assert baseline['generation_wall_time']['mean_ms'] == 2
    assert baseline['current_calls']['attempts'] == 0
    assert baseline['historical_calls']['attempts'] == 2
    assert baseline['historical_calls']['failed_attempts'] == 1
    assert baseline['historical_calls']['source_experiments'] == ['previous']


def test_request_cache_attributes_only_the_matching_key_and_deduplicates(comparison):
    comparison.trace('previous', SOURCE_TRACE, [generation('candidate', [
        cache_event('cache_lookup', 'llm_request', 'other'), call('unrelated-model', 5000),
        cache_event('cache_lookup', 'llm_request'), call(BASELINE, 1000),
        cache_event('cache_lookup', 'llm_request', 'later'), call(BASELINE, 2000)])])
    comparison.trace('current', TRACE, [generation('baseline', [
        cache_event('cache_hit', 'llm_request', origin=comparison.origin),
        cache_event('cache_hit', 'llm_request', origin=comparison.origin)]),
        generation('candidate', [call(CANDIDATE, 500)])])
    baseline = report.summarize(comparison.directory)['cohorts']['all']['speed']['baseline']
    assert baseline['model_time']['mean_ms'] == 1000
    assert baseline['historical_calls']['attempts'] == 1


def test_request_cache_can_follow_another_cache_hit(comparison):
    comparison.trace('original', 'c' * 32, [generation('candidate', [
        cache_event('cache_lookup', 'llm_request'), call(BASELINE, 1200)])])
    original = dict(experiment_id='original', trace_ref='c' * 32, operation_index=0)
    comparison.trace('previous', SOURCE_TRACE, [generation('candidate', [
        cache_event('cache_hit', 'llm_request', origin=original)])])
    comparison.trace('current', TRACE, [generation('baseline', [
        cache_event('cache_hit', 'llm_request', origin=comparison.origin)]),
        generation('candidate', [call(CANDIDATE, 300)])])
    baseline = report.summarize(comparison.directory)['cohorts']['all']['speed']['baseline']
    assert baseline['model_time']['mean_ms'] == 1200
    assert baseline['historical_calls']['source_experiments'] == ['original']


@pytest.mark.parametrize('change', ['case', 'data', 'dataset', 'model', 'key', 'index', 'trace', 'path', 'cycle', 'symlink'])
def test_invalid_provenance_is_missing_not_zero(comparison, change, tmp_path):
    events = [cache_event('cache_lookup', 'llm_request'), call(BASELINE, 1000)]
    kwargs = {}
    origin = dict(comparison.origin)
    if change == 'case':
        kwargs['case_id'] = 'other'
    elif change == 'data':
        kwargs['data_ref'] = 'd-0002'
    elif change == 'dataset':
        kwargs['dataset'] = 'fixed'
    elif change == 'model':
        events[-1] = call(CANDIDATE, 1000)
    elif change == 'key':
        events[0] = cache_event('cache_lookup', 'llm_request', 'other')
    elif change == 'index':
        origin['operation_index'] = -1
    elif change == 'trace':
        origin['trace_ref'] = '../invalid'
    elif change == 'path':
        origin['experiment_id'] = '../previous'
    elif change == 'cycle':
        events = [cache_event('cache_hit', 'llm_request', origin=origin)]
    comparison.trace('previous', SOURCE_TRACE, [generation('candidate', events)], **kwargs)
    if change == 'symlink':
        (tmp_path / 'experiments/escaped').symlink_to(tmp_path / 'generators', target_is_directory=True)
        origin['experiment_id'] = 'escaped'
    comparison.trace('current', TRACE, [generation('baseline', [
        cache_event('cache_hit', 'llm_request', origin=origin)]),
        generation('candidate', [call(CANDIDATE, 500)])])
    speed = report.summarize(comparison.directory)['cohorts']['all']['speed']
    assert speed['paired']['cases'] == 0
    assert speed['baseline']['model_time']['mean_ms'] is None
    assert sum(speed['baseline']['missing_timing_evidence'].values()) == 1


def test_groups_failures_retries_and_verified_metrics(comparison, monkeypatch):
    comparison.rows.append(dict(case_id='failed', status='failed', chat_type='group', reason='token limit'))
    monkeypatch.setattr(report.protocol, 'summarize_final_records',
        lambda path: ({row['case_id']: row for row in comparison.rows}, 1))
    write_json(comparison.directory / 'state.json', dict(status='finished'))
    metrics = dict(pairs=1, failures=1, net_win_confirmed=1, wins_confirmed=1,
                   losses_confirmed=0, contested=0)
    result = report.summarize(comparison.directory, verified_metrics=metrics)
    assert result['complete'] is True
    assert result['retries'] == 1
    assert result['cohorts']['group']['quality']['failures'] == 1
    assert result['cohorts']['group']['quality']['failure_reasons'] == {'token limit': 1}
    assert result['cohorts']['group']['quality']['failure_rate'] == 1
    assert result['cohorts']['all']['quality']['paired_success_rate'] == 0.5
    assert result['cohorts']['all']['quality']['confirmation_evidence_validated'] is True
    with pytest.raises(ConfigError, match='differs'):
        report.summarize(comparison.directory, verified_metrics={**metrics, 'net_win_confirmed': 2})


def test_extra_confirmation_draws_are_not_model_speed_samples(comparison):
    comparison.trace('current', TRACE, [generation('baseline', [call(BASELINE, 1000)]),
        generation('candidate', [call(CANDIDATE, 400)]),
        {**generation('candidate', [call(CANDIDATE, 9000)]), 'round': 1}])
    speed = report.summarize(comparison.directory)['cohorts']['all']['speed']
    assert speed['candidate']['current_calls']['attempts'] == 1
    assert speed['paired']['candidate']['mean_ms'] == 400


def test_unified_check_reports_progress_even_when_formal_gate_is_incomplete(comparison, monkeypatch):
    from src.iteration import experiment, gates
    monkeypatch.setattr(experiment, 'load_experiment', lambda name: comparison.directory)

    def incomplete(directory):
        raise ConfigError('formal evidence incomplete')

    monkeypatch.setattr(gates, 'evidence', incomplete)
    before = {path: path.read_bytes() for path in comparison.directory.rglob('*') if path.is_file()}
    result = check.Report('experiment')
    with check.offline():
        check.check_experiment(SimpleNamespace(exp='current', gate='development', reply_model_report=True), result)
    assert [item['status'] for item in result.value['checks']] == ['failed', 'passed']
    assert result.value['checks'][1]['detail']['final_records'] == 1
    assert all(path.read_bytes() == value for path, value in before.items())
