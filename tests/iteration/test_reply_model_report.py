from types import SimpleNamespace

import pytest

from scripts import check
from src.infrastructure.config import ConfigError
from src.iteration import reply_model_report as report
from src.infrastructure.storage import write_json


BASELINE = 'baseline-model'
CANDIDATE = 'candidate-model'
JUDGE = 'judge-model'
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
    write_json(tmp_path / 'judges/j-0001/config.json',
        dict(llm=dict(model=JUDGE), feature_llm=dict(model='judge-feature-model')))
    row = dict(case_id='case', status='ok', chat_type='private', trace_ref=TRACE,
        identified_baseline=True, identified_candidate=False,
        flip_verified=dict(baseline_identified_final=True, candidate_identified_final=False))
    rows = [row]
    monkeypatch.setattr(report.protocol, 'summarize_final_records',
        lambda path: ({item['case_id']: item for item in rows}, 0))

    def trace(name, reference, operations, case_id='case', data_ref='d-0001', dataset='development',
              started_at=None, ended_at=None, **versions):
        write_json(tmp_path / 'experiments' / name / 'traces' / (reference + '.json'),
            dict(case_id=case_id, case=dict(chat_type='private'), trace_ref=reference,
                 started_at=started_at, ended_at=ended_at,
                 versions={**dict(data_ref=data_ref, dataset=dataset,
                 baseline_ref='g-0001', candidate_ref='g-0002', judge_ref='j-0001'), **versions},
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


def test_judge_codex_cache_can_follow_successful_request_from_failed_stage(comparison):
    cached_call = dict(kind='codex', request=dict(config=dict(model=JUDGE)),
        status='ok', elapsed_ms=1200)
    source = dict(kind='judge', branch='candidate', round=0, status='failed',
        elapsed_ms=1600, events=[cache_event('cache_lookup', 'llm_request'),
            call(JUDGE, 100, 'failed'), cached_call,
            cache_event('cache_lookup', 'llm_request', 'unrelated'), call('other-model', 500)])
    comparison.trace('previous', SOURCE_TRACE, [source])
    comparison.trace('current', TRACE, [dict(kind='judge', branch='baseline', round=0,
        status='ok', elapsed_ms=350, events=[
            cache_event('cache_hit', 'llm_request', origin=comparison.origin),
            cache_event('cache_lookup', 'llm_request', 'feature'), call('judge-feature-model', 300)])])
    recorded = report.RecordedCalls(comparison.directory, comparison.spec)
    timing = recorded.operation(comparison.rows[0], 'judge', 'baseline', {JUDGE, 'judge-feature-model'})
    assert timing['wall_ms'] == 350
    assert timing['model_ms'] == 1600
    assert len(timing['current_calls']) == 1
    assert len(timing['historical_calls']) == 2
    assert sum(attempt['failed'] for attempt in timing['historical_calls'].values()) == 1
    _, operation = recorded.selected_operation(comparison.rows[0], 'judge', 'baseline')
    assert recorded._uncached_wall(operation, 'case') is None


@pytest.mark.parametrize('problem', ['failed_only', 'missing_key', 'wrong_model', 'comparison'])
def test_invalid_judge_cache_evidence_rejected(comparison, problem):
    source = dict(kind='judge', branch='candidate', round=0, status='failed',
        elapsed_ms=1500, events=[cache_event('cache_lookup', 'llm_request'),
            dict(kind='codex', request=dict(config=dict(model='other-model' if problem == 'wrong_model' else JUDGE)),
                status='failed' if problem == 'failed_only' else 'ok', elapsed_ms=1200)])
    comparison.trace('previous', SOURCE_TRACE, [source])
    comparison.trace('current', TRACE, [dict(kind='judge', branch='baseline', round=0,
        status='ok', elapsed_ms=2, events=[cache_event('cache_hit', 'llm_request',
            key=None if problem == 'missing_key' else 'request', origin=comparison.origin)])],
        **(dict(judge_ref='j-0002') if problem == 'comparison' else {}))
    recorded = report.RecordedCalls(comparison.directory, comparison.spec)
    with pytest.raises(ConfigError):
        recorded.operation(comparison.rows[0], 'judge', 'baseline', JUDGE)


def test_whole_stage_cache_chain_wall_gaps_do_not_change_generation_report(comparison):
    original = dict(experiment_id='original', trace_ref='c' * 32, operation_index=0)
    comparison.trace('original', 'c' * 32, [generation('candidate', [call(BASELINE, 1200)], wall=None)])
    comparison.trace('previous', SOURCE_TRACE, [generation('candidate', [
        cache_event('cache_hit', 'generation', origin=original)], wall=3)])
    comparison.trace('current', TRACE, [generation('baseline', [
        cache_event('cache_hit', 'generation', origin=comparison.origin)], wall=2),
        generation('candidate', [call(CANDIDATE, 300)])])
    recorded = report.RecordedCalls(comparison.directory, comparison.spec)
    _, operation = recorded.selected_operation(comparison.rows[0], 'generation', 'baseline')
    with pytest.raises(ConfigError):
        recorded._uncached_wall(operation, 'case')
    baseline = report.summarize(comparison.directory)['cohorts']['all']['speed']['baseline']
    assert baseline['model_time']['mean_ms'] == 1200
    assert baseline['generation_wall_time']['mean_ms'] == 2
    assert baseline['missing_timing_evidence'] == {}


def test_whole_stage_cache_wall_chain_rejects_cycle(comparison):
    comparison.trace('previous', SOURCE_TRACE, [generation('candidate', [
        cache_event('cache_hit', 'generation', origin=comparison.origin)])])
    comparison.trace('current', TRACE, [generation('baseline', [
        cache_event('cache_hit', 'generation', origin=comparison.origin)])])
    recorded = report.RecordedCalls(comparison.directory, comparison.spec)
    _, operation = recorded.selected_operation(comparison.rows[0], 'generation', 'baseline')
    with pytest.raises(ConfigError, match='cyclic'):
        recorded._uncached_wall(operation, 'case')


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


@pytest.mark.parametrize('status,completion_error', [
    ('running', None), ('finished', None), ('finished', 'frozen case coverage incomplete'),
])
def test_fixed_report_checks_completion_not_development_gate(comparison, monkeypatch,
        status, completion_error):
    from src.iteration import experiment, gates, record_contract
    comparison.spec['dataset'] = 'fixed_test'
    write_json(comparison.directory / 'spec.json', comparison.spec)
    metrics = dict(pairs=1, failures=0, net_win_confirmed=1, wins_confirmed=1,
        losses_confirmed=0, contested=0)
    write_json(comparison.directory / 'state.json', dict(status=status, metrics=metrics))
    comparison.trace('current', TRACE, [generation('baseline', [call(BASELINE, 1000)]),
        generation('candidate', [call(CANDIDATE, 400)])], dataset='fixed_test')
    monkeypatch.setattr(experiment, 'load_experiment', lambda name: comparison.directory)

    def development(directory):
        pytest.fail('fixed report must not check the development promotion gate')

    validations = []
    def validate(directory, recorded_metrics):
        validations.append((directory, recorded_metrics))
        if completion_error:
            raise ConfigError(completion_error)

    monkeypatch.setattr(gates, 'evidence', development)
    monkeypatch.setattr(record_contract, 'validate_completion', validate)
    before = {path: path.read_bytes() for path in comparison.directory.rglob('*') if path.is_file()}
    result = check.Report('experiment')
    with check.offline():
        check.check_experiment(SimpleNamespace(exp='current', gate='development', reply_model_report=True), result)
    completion, summary = result.value['checks']
    passed = status == 'finished' and completion_error is None
    assert completion['name'] == 'fixed_completion'
    assert completion['status'] == ('passed' if passed else 'failed')
    assert summary['status'] == 'passed'
    assert summary['detail']['cohorts']['all']['quality']['confirmation_evidence_validated'] is passed
    assert validations == ([(comparison.directory, metrics)] if status == 'finished' else [])
    assert all(path.read_bytes() == value for path, value in before.items())


def judge(events, *, round_number=0, status='ok', wall=100):
    return dict(kind='judge', branch='baseline', round=round_number, status=status,
                elapsed_ms=wall, events=events)


def test_judge_speed_includes_failed_cases_and_splits_confirmation(comparison):
    codex = dict(kind='codex', request=dict(config=dict(model=JUDGE)), status='ok', elapsed_ms=300)
    comparison.trace('current', TRACE, [judge([call(JUDGE, 100, 'failed'), codex], wall=500),
        judge([call(JUDGE, 200)], round_number=1, wall=250),
        judge([cache_event('cache_hit', 'judge')], wall=2),
        judge([cache_event('cache_hit', 'llm_request'),
               cache_event('cache_hit', 'blind_order')], wall=3)])
    comparison.trace('current', SOURCE_TRACE, [judge([call(JUDGE, 700, 'failed')], status='failed', wall=800),
        judge([call(JUDGE, None, 'running')], status='running', wall=None)], case_id='unfinished')
    speed = report.summarize(comparison.directory)['judge_speed']
    initial = speed['phases']['initial']
    assert speed['traces'] == 2
    assert initial['operations'] == dict(ok=3, failed=1, running=1)
    assert initial['operation_wall_time']['count'] == 4
    assert initial['cache_hits'] == dict(judge=1, llm_request=1)
    assert initial['current_calls']['completed_attempts'] == 3
    assert initial['current_calls']['failed_attempts'] == 2
    assert initial['current_calls']['recorded_running_attempts'] == 1
    assert initial['current_calls']['durations']['mean_ms'] == round(1100 / 3, 3)
    assert speed['phases']['confirmation']['current_calls']['durations']['mean_ms'] == 200
    assert speed['missing_timing_evidence'] == {}


@pytest.mark.parametrize('change', ['data_ref', 'dataset', 'judge_ref', 'baseline_ref',
                                   'candidate_ref', 'trace_ref', 'path', 'invalid_ref'])
def test_judge_speed_rejects_unbound_or_unsafe_traces(comparison, tmp_path, change):
    reference = SOURCE_TRACE if change != 'invalid_ref' else 'not-a-trace'
    versions = {change: 'different'} if change not in {'trace_ref', 'path', 'invalid_ref'} else {}
    comparison.trace('current', reference, [judge([call(JUDGE, 1234)])], **versions)
    path = comparison.directory / 'traces' / (reference + '.json')
    if change == 'trace_ref':
        import json
        value = json.loads(path.read_text())
        value['trace_ref'] = TRACE
        write_json(path, value)
    if change == 'path':
        outside = tmp_path / 'outside.json'
        path.rename(outside)
        path.symlink_to(outside)
    speed = report.summarize(comparison.directory)['judge_speed']
    assert speed['phases'] == {}
    assert sum(speed['missing_timing_evidence'].values()) == 1


@pytest.mark.parametrize('duration', [None, -1, True, float('inf'), 'unknown'])
def test_judge_speed_missing_duration_never_becomes_zero(comparison, duration):
    comparison.trace('current', TRACE, [judge([call(JUDGE, duration)], wall=None)])
    speed = report.summarize(comparison.directory)['judge_speed']
    initial = speed['phases']['initial']
    assert initial['operations'] == dict(ok=1)
    assert initial['operation_wall_time']['mean_ms'] is None
    assert initial['current_calls']['durations']['mean_ms'] is None
    assert initial['current_calls']['completed_attempts'] == 1
    assert initial['current_calls']['successful_attempts'] == 1
    assert initial['current_calls']['missing_durations_by_status'] == {'ok': 1}
    assert sum(speed['missing_timing_evidence'].values()) == 2


def test_judge_speed_keeps_valid_calls_without_operation_wall_time(comparison):
    comparison.trace('current', TRACE, [judge([call(JUDGE, 200)], wall=None),
        judge([call('unbound-model', 1000)])])
    speed = report.summarize(comparison.directory)['judge_speed']
    assert speed['phases']['initial']['current_calls']['durations']['mean_ms'] == 200
    assert sum(speed['missing_timing_evidence'].values()) == 2


def test_judge_speed_counts_failure_without_duration_before_successful_retry(comparison):
    comparison.trace('current', TRACE, [judge([call(JUDGE, None, 'failed'), call(JUDGE, 200)])])
    speed = report.summarize(comparison.directory)['judge_speed']
    calls = speed['phases']['initial']['current_calls']
    assert calls['completed_attempts'] == 2
    assert calls['successful_attempts'] == 1
    assert calls['failed_attempts'] == 1
    assert calls['missing_durations_by_status'] == {'failed': 1}
    assert calls['durations']['count'] == 1
    assert calls['durations']['mean_ms'] == 200
    assert calls['failed_durations']['mean_ms'] is None
    assert speed['phases']['initial']['models'] == {JUDGE: 2}


def timed(item, started, ended=None):
    return {**item, 'started_at': started, 'ended_at': ended}


def run_profile(comparison, pid, started, ended, status='running'):
    write_json(comparison.directory / 'transport_profiles' / f'{pid}.json',
        dict(pid=pid, started_at=started, updated_at=ended, status=status, workers=4))


@pytest.mark.parametrize('intervals,expected', [
    ([], (None, None, None, None)),
    ([(100, 200)], (100000, 100000, 0, 0)),
    ([(100, 200), (300, 400)], (300000, 200000, 100000, 0)),
    ([(100, 250), (200, 300)], (200000, 200000, 0, 50000)),
    ([(100, 400), (200, 300), (250, 450)], (350000, 350000, 0, 250000)),
    ([(100, 200), (200, 200), (200, 300)], (200000, 200000, 0, 0)),
])
def test_run_totals_merge_recorded_wall_intervals(comparison, intervals, expected):
    for pid, (started, ended) in enumerate(reversed(intervals), start=1):
        run_profile(comparison, pid, started, ended)
    wall = report.summarize(comparison.directory)['run_window_speed']['totals']['recorded_wall_time']
    assert tuple(wall[name] for name in ('elapsed_span_ms', 'recorded_active_time_ms',
        'uncovered_time_ms', 'overlapping_time_ms')) == expected
    assert wall['started_at'] == (intervals[0][0] if intervals else None)
    assert wall['ended_at'] == (max(ended for started, ended in intervals) if intervals else None)


def test_run_totals_count_unattributed_calls_once_and_exclude_cache_history(comparison):
    run_profile(comparison, 11, 100, 250)
    run_profile(comparison, 22, 200, 300)
    comparison.trace('previous', SOURCE_TRACE, [generation('baseline', [call(BASELINE, 90000)])])
    comparison.trace('current', TRACE, [
        timed(generation('baseline', [timed(call(BASELINE, 1000), 120, 121),
            timed(cache_event('cache_hit', 'generation', origin=comparison.origin), 130)]), 110, 140),
        timed(judge([timed(call(JUDGE, 2000, 'failed'), 220, 222),
            timed(call(JUDGE, None, 'failed'), 230, 231),
            timed(call(JUDGE, None, 'running'), 240)]), 210, 245),
    ], ended_at=270)
    totals = report.summarize(comparison.directory)['run_window_speed']['totals']
    assert totals['retained_final_records'] == {'ok': 1}
    assert totals['real_calls']['statuses'] == {'ok': 1, 'failed': 2, 'running': 1}
    assert totals['real_calls']['attempt_time'] == dict(completed_ms=3000, successful_ms=1000, failed_ms=2000)
    assert totals['real_calls']['missing_durations'] == {'failed': 1}
    assert totals['recorded_wall_time']['recorded_active_time_ms'] == 200000


def test_run_windows_split_actual_calls_and_initial_confirmation(comparison):
    run_profile(comparison, 22, 200, 400)
    run_profile(comparison, 11, 100, 199, 'failed')
    comparison.trace('current', TRACE, [
        timed(generation('baseline', [timed(call(BASELINE, 1000, 'failed'), 111, 112),
            timed(call(BASELINE, 3000), 113, 116)]), 110, 117),
        timed(generation('candidate', [timed(call(CANDIDATE, 500), 211, 212)]), 210, 213),
        timed(judge([timed(call(JUDGE, 100), 121, 122)]), 120, 123),
        timed(judge([timed(call(JUDGE, 200, 'failed'), 221, 222),
            timed(call(JUDGE, 400), 223, 224)]), 220, 225),
        timed(judge([timed(call(JUDGE, 600), 231, 232)], round_number=1), 230, 233),
        timed(judge([timed(cache_event('cache_hit', 'judge'), 241)]), 240, 242),
    ], started_at=105, ended_at=390)
    speed = report.summarize(comparison.directory)['run_window_speed']
    before, after = speed['windows']
    assert [before['pid'], after['pid']] == [11, 22]
    baseline = before['stages']['generation_baseline_initial']['real_calls']
    assert baseline['statuses'] == {'failed': 1, 'ok': 1}
    assert baseline['durations']['mean_ms'] == 2000
    assert before['stages']['generation_baseline_initial']['real_calls_by_model'][BASELINE]['statuses'] == {'failed': 1, 'ok': 1}
    assert before['stages']['generation_baseline_initial']['real_calls_by_model'][BASELINE]['durations']['mean_ms'] == 2000
    assert 'generation_baseline_initial' not in after['stages']
    assert after['stages']['generation_candidate_initial']['real_calls']['durations']['mean_ms'] == 500
    assert before['stages']['judge_initial']['real_calls']['durations']['mean_ms'] == 100
    assert after['stages']['judge_initial']['real_calls']['durations']['mean_ms'] == 300
    assert after['stages']['judge_confirmation']['real_calls']['durations']['mean_ms'] == 600
    assert after['stages']['judge_initial']['cache_hits'] == {'judge': 1}
    assert before['final_records'] == {}
    assert after['final_records'] == {'ok': 1}
    assert speed['unattributed']['stages'] == {}
    assert speed['missing_timing_evidence'] == {}


def test_run_window_cache_never_imports_historical_latency(comparison):
    run_profile(comparison, 11, 100, 200)
    comparison.trace('previous', SOURCE_TRACE, [generation('baseline', [call(BASELINE, 90000)])])
    comparison.trace('current', TRACE, [timed(generation('baseline', [
        timed(cache_event('cache_hit', 'generation', origin=comparison.origin), 121)], wall=2), 120, 122)],
        ended_at=130)
    summary = report.summarize(comparison.directory)
    assert summary['cohorts']['all']['speed']['baseline']['historical_calls']['durations']['mean_ms'] == 90000
    stage = summary['run_window_speed']['windows'][0]['stages']['generation_baseline_initial']
    assert stage['cache_hits'] == {'generation': 1}
    assert stage['real_calls']['durations']['count'] == 0
    assert stage['real_calls']['durations']['mean_ms'] is None


def test_run_window_includes_failed_and_running_cases_without_zero_filling(comparison):
    run_profile(comparison, 11, 100, 200)
    comparison.rows.clear()
    comparison.trace('current', TRACE, [
        timed(judge([timed(call(JUDGE, None, 'failed'), 121, 122)], status='failed', wall=None), 120, 123),
        timed(judge([timed(call(JUDGE, None, 'running'), 131)], status='running', wall=None), 130),
    ], case_id='failed-or-running')
    speed = report.summarize(comparison.directory)['run_window_speed']
    window = speed['windows'][0]
    assert window['case_ids'] == ['failed-or-running']
    stage = window['stages']['judge_initial']
    assert stage['operations'] == {'failed': 1, 'running': 1}
    assert stage['real_calls']['statuses'] == {'failed': 1, 'running': 1}
    assert stage['missing_durations'] == {'operation_failed': 1, 'call_failed': 1}
    assert stage['real_calls']['durations']['count'] == 0


@pytest.mark.parametrize('timestamp', [None, True, -1, float('inf'), 'unknown', 99, 201])
def test_run_window_missing_or_outside_timestamps_stay_unattributed(comparison, timestamp):
    run_profile(comparison, 11, 100, 200)
    comparison.trace('current', TRACE, [timed(judge([timed(call(JUDGE, 1000), timestamp, 180)]), timestamp, 190)],
        ended_at=190)
    speed = report.summarize(comparison.directory)['run_window_speed']
    assert speed['windows'][0]['stages'] == {}
    assert speed['unattributed']['stages']['judge_initial']['real_calls']['durations']['mean_ms'] == 1000
    assert sum(speed['missing_timing_evidence'].values()) == 2


def test_run_window_events_use_own_timestamp_and_crossing_calls_are_visible(comparison):
    run_profile(comparison, 11, 100, 199)
    run_profile(comparison, 22, 200, 300)
    comparison.trace('current', TRACE, [timed(judge([
        timed(call(JUDGE, 20000), 190, 210),
        timed(call(JUDGE, 3000), 220, 223),
        timed(cache_event('cache_hit', 'llm_request'), 225),
    ], wall=40000), 180, 230)], ended_at=240)
    before, after = report.summarize(comparison.directory)['run_window_speed']['windows']
    assert before['stages']['judge_initial']['operations'] == {'ok': 1}
    assert before['stages']['judge_initial']['real_calls']['crossing_window_end'] == 1
    assert after['stages']['judge_initial']['operations'] == {}
    assert after['stages']['judge_initial']['real_calls']['durations']['mean_ms'] == 3000
    assert after['stages']['judge_initial']['cache_hits'] == {'llm_request': 1}


def test_run_window_overlap_does_not_double_count(comparison):
    run_profile(comparison, 11, 100, 250)
    run_profile(comparison, 22, 200, 300)
    comparison.trace('current', TRACE, [timed(judge([timed(call(JUDGE, 1000), 220, 223)]), 210, 230)],
        ended_at=240)
    speed = report.summarize(comparison.directory)['run_window_speed']
    assert all(not window['stages'] for window in speed['windows'])
    assert speed['unattributed']['final_records'] == {'ok': 1}
    assert speed['unattributed']['stages']['judge_initial']['real_calls']['durations']['count'] == 1
    assert sum(speed['missing_timing_evidence'].values()) == 3


@pytest.mark.parametrize('change', ['data_ref', 'dataset', 'judge_ref', 'baseline_ref', 'candidate_ref'])
def test_run_window_rejects_changed_frozen_comparison(comparison, change):
    run_profile(comparison, 11, 100, 200)
    comparison.trace('current', TRACE, [timed(judge([timed(call(JUDGE, 1000), 120, 121)]), 110, 130)],
        ended_at=140, **{change: 'different'})
    speed = report.summarize(comparison.directory)['run_window_speed']
    assert speed['windows'][0]['stages'] == {}
    assert sum(speed['missing_timing_evidence'].values()) == 1


def test_run_window_keeps_unbound_model_out_of_speed_samples(comparison):
    run_profile(comparison, 11, 100, 200)
    comparison.trace('current', TRACE, [timed(judge([timed(call('not-frozen-model', 1000), 120, 121)]), 110, 130)],
        ended_at=140)
    speed = report.summarize(comparison.directory)['run_window_speed']
    assert speed['windows'][0]['stages']['judge_initial']['real_calls']['durations']['count'] == 0
    assert sum(speed['missing_timing_evidence'].values()) == 1


def test_generation_evidence_separates_failed_attempt_time_and_recorded_wall_time(comparison):
    run_profile(comparison, 11, 100, 200)
    body = dict(model=BASELINE, messages=[dict(role='user', content='问题')], system='规则', max_tokens=2500)
    failed = {**call(BASELINE, 20000, 'failed'),
        'request': dict(body=body, attempt=1, timeout_seconds=20),
        'error': dict(type='LLMError', message='The read operation timed out')}
    success = {**call(BASELINE, 10000), 'request': dict(body=body, attempt=2, timeout_seconds=20),
        'response': dict(usage=dict(input_tokens=50, output_tokens=5000, cache_read_input_tokens=10),
                         content_types=['thinking', 'text'])}
    running = {**call(BASELINE, None, 'running'), 'request': dict(body=body, attempt=3, timeout_seconds=20)}
    comparison.trace('current', TRACE, [timed(generation('baseline', [timed(failed, 110, 130),
        timed(success, 131, 141), timed(running, 142)]), 105)], ended_at=150)
    window = report.summarize(comparison.directory)['run_window_speed']['windows'][0]
    stage = window['stages']['generation_baseline_initial']
    model = stage['real_calls_by_model'][BASELINE]
    assert window['recorded_wall_time_ms'] == 100000
    assert model['attempt_time'] == dict(completed_ms=30000, successful_ms=10000, failed_ms=20000)
    assert stage['real_calls']['attempt_time'] == model['attempt_time']
    evidence = model['generation_evidence']
    assert evidence['attempt_numbers'] == {'1': 1, '2': 1, '3': 1}
    assert evidence['failure_categories'] == {'read_timeout': 1}
    observations = evidence['observations_by_status']
    assert observations['ok']['prompt_characters']['p50'] == 4
    assert observations['ok']['requested_max_tokens']['p50'] == 2500
    assert observations['ok']['output_tokens']['p50'] == 5000
    assert observations['ok']['cache_read_input_tokens']['p50'] == 10
    assert observations['ok']['content_types'] == {'thinking': 1, 'text': 1}
    assert observations['failed']['timeout_seconds']['p50'] == 20
    assert observations['failed']['output_tokens']['count'] == 0
    assert observations['running']['output_tokens']['mean'] is None


@pytest.mark.parametrize('message,category', [
    ('[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred', 'tls_eof'),
    ('tls handshake eof', 'tls_eof'),
    ('调用失败 HTTP 429（接口限流）：ServerOverloaded', 'http_429'),
    ('HTTP 429: TooManyRequests', 'http_429'),
    ('HTTP 429: The read operation timed out', 'http_429'),
    ('returned 429 output tokens', 'other_or_unknown'),
    ('HTTP 4290', 'other_or_unknown'),
    ('Connection reset by peer', 'other_or_unknown'),
    ('connection failed or timed out', 'other_or_unknown'),
])
def test_generation_failure_categories_do_not_infer_read_timeout(comparison, message, category):
    run_profile(comparison, 11, 100, 200)
    event = {**call(BASELINE, None, 'failed'), 'error': dict(type='LLMError', message=message)}
    comparison.trace('current', TRACE, [timed(generation('baseline', [timed(event, 110, 120)]), 105, 121)],
        ended_at=130)
    model = report.summarize(comparison.directory)['run_window_speed']['windows'][0][
        'stages']['generation_baseline_initial']['real_calls_by_model'][BASELINE]
    assert model['generation_evidence']['failure_categories'] == {category: 1}
    assert model['generation_evidence']['attempt_numbers'] == {'unknown': 1}
    assert model['missing_durations'] == {'failed': 1}
    assert model['durations']['count'] == 0


@pytest.mark.parametrize('value', [None, True, -1, 1.5, 'unknown'])
def test_generation_missing_usage_is_not_zero_or_a_missing_call(comparison, value):
    run_profile(comparison, 11, 100, 200)
    event = {**call(BASELINE, 1000), 'request': dict(body=dict(model=BASELINE,
        messages=[dict(content=['unsupported-block'])], system=None, max_tokens=True),
        attempt=True, timeout_seconds=value),
        'response': dict(usage=dict(output_tokens=value), content_types=None)}
    comparison.trace('current', TRACE, [timed(generation('baseline', [timed(event, 110, 111)]), 105, 112)],
        ended_at=120)
    model = report.summarize(comparison.directory)['run_window_speed']['windows'][0][
        'stages']['generation_baseline_initial']['real_calls_by_model'][BASELINE]
    observations = model['generation_evidence']['observations_by_status']['ok']
    assert observations['output_tokens']['count'] == 0
    assert observations['output_tokens']['mean'] is None
    assert observations['prompt_characters']['count'] == 0
    assert observations['requested_max_tokens']['count'] == 0
    assert model['attempt_time']['successful_ms'] == 1000
    assert model['durations']['count'] == 1
