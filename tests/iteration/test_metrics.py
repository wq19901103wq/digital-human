import json
from types import SimpleNamespace

import pytest

from scripts import check
from src.infrastructure.config import ConfigError
from src.iteration import experiment, gates, metrics, record_contract, versions
from src.infrastructure.storage import write_json


TRACE = 'a' * 32
GROUP_TRACE = 'b' * 32
SOURCE_TRACE = 'c' * 32
BASELINE = 'baseline-model'
CANDIDATE = 'candidate-model'
JUDGE = 'judge-model'
FEATURE = 'feature-model'


def call(model, elapsed, *, status='ok', backend='llm'):
    return dict(kind=backend, status=status, elapsed_ms=elapsed,
        request={('config' if backend == 'codex' else 'body'): dict(model=model)})


def operation(kind, side, events, wall=100, **changes):
    return dict(kind=kind, branch=side, round=0, status='ok', elapsed_ms=wall,
        events=events, **changes)


def cache(kind, layer='llm_request', key='request', origin=None):
    return dict(kind=kind, data=dict(layer=layer, key=key, origin=origin))


def record(case_id='private', baseline=True, candidate=False, *, judge=False):
    row = dict(case_id=case_id, status='ok', trace_ref=TRACE if case_id == 'private' else GROUP_TRACE)
    if judge:
        row.update(baseline_correct=baseline, candidate_correct=candidate)
        if baseline != candidate:
            row.update(flip_verified=True, baseline_identified_final=baseline,
                candidate_identified_final=candidate, baseline_votes=[baseline], candidate_votes=[candidate])
    else:
        row.update(chat_type=case_id, identified_baseline=baseline, identified_candidate=candidate)
        if baseline != candidate:
            row['flip_verified'] = dict(baseline_identified_final=baseline,
                candidate_identified_final=candidate, baseline_votes=[baseline], candidate_votes=[candidate])
    return row


@pytest.fixture
def saved(tmp_path, monkeypatch):
    directory = tmp_path / 'experiments/current'
    spec = dict(id='current', kind='gen_ab', dataset='development', data_ref='d-0001',
        baseline_ref='g-0001', candidate_ref='g-0002', judge_ref='j-0001',
        protocol=dict(flip_extra_rounds=0))
    planned = [dict(case_id=chat, chat_type=chat) for chat in ('group', 'private')]
    monkeypatch.setattr(versions, 'PRIVATE', tmp_path)
    monkeypatch.setattr(record_contract, 'planned_rows', lambda spec: planned)
    write_json(directory / 'state.json', dict(status='running'))
    for ref, model in (('g-0001', BASELINE), ('g-0002', CANDIDATE)):
        write_json(tmp_path / 'generators' / ref / 'config.json',
            dict(llm={chat: dict(model=model) for chat in ('group', 'private')}))
    for ref in ('j-0001', 'j-0002'):
        write_json(tmp_path / 'judges' / ref / 'config.json',
            dict(llm=dict(model=JUDGE), feature_llm=dict(model=FEATURE)))

    def configure(**changes):
        spec.update(changes)
        write_json(directory / 'spec.json', spec)

    def append(*rows):
        with record_contract.writer(directory) as writer:
            for row in rows:
                writer(row)

    def trace(operations=None, *, name='current', reference=TRACE, case_id='private', **changes):
        if operations is None:
            operations = [operation('generation', 'baseline', [call(BASELINE, 1000)], 1100),
                operation('generation', 'candidate', [call(CANDIDATE, 400)], 450),
                operation('judge', 'baseline', [call(JUDGE, 100), call(FEATURE, 300, backend='codex')], 500),
                operation('judge', 'candidate', [call(JUDGE, 200)], 250)]
        value = dict(trace_ref=reference, case_id=case_id, versions=dict(spec), operations=operations)
        value.update(changes)
        write_json(tmp_path / 'experiments' / name / 'traces' / (reference + '.json'), value)

    configure()
    trace()
    with check.offline():
        yield SimpleNamespace(directory=directory, spec=spec, planned=planned,
            configure=configure, append=append, trace=trace,
            origin=dict(experiment_id='source', trace_ref=SOURCE_TRACE, operation_index=0))


def test_per_case_stage_totals_not_mean_of_individual_calls(saved):
    saved.append(record(), record('group', baseline=False, candidate=False))
    saved.trace([operation('generation', 'baseline', [call(BASELINE, 2000, status='failed'),
        call(BASELINE, 3000)], 5200)], reference=GROUP_TRACE, case_id='group')
    result = metrics.summarize(saved.directory)
    baseline = result['cohorts']['all']['latency']['generation']['baseline']
    assert baseline['model_attempt_ms'] == dict(count=2, mean_ms=3000, p50_ms=3000,
        p95_ms=5000, missing=0, coverage_rate=1)
    assert baseline['current_calls']['attempts'] == 3
    assert baseline['current_calls']['failed_attempt_ms'] == 2000
    assert baseline['current_wall_ms']['mean_ms'] == 3150
    assert result['cohorts']['private']['latency']['judge']['baseline']['model_attempt_ms']['mean_ms'] == 400
    assert result['cohorts']['group']['latency']['judge']['baseline']['current_wall_ms']['mean_ms'] is None
    assert result['per_case'][0]['generation']['baseline']['model_attempt_ms'] == 5000
    assert result['per_case'][0]['generation']['baseline']['current_attempts'] == 2


def test_quality_cohorts_failure_and_missing_have_explicit_denominators(saved):
    saved.planned.append(dict(case_id='missing', chat_type='group'))
    saved.append(record(), dict(case_id='group', status='failed', reason='timeout'))
    result = metrics.summarize(saved.directory)
    quality = result['cohorts']['all']['quality']
    assert (quality['expected'], quality['attempted'], quality['pairs'], quality['failures'], quality['missing']) == (3, 2, 1, 1, 1)
    assert quality['coverage_rate'] == pytest.approx(2 / 3)
    assert quality['identification_rate'] == dict(baseline=1, candidate=0)
    assert quality['direction'] == 'lower_is_better'
    assert quality['net_win_confirmed'] == 1
    group = result['cohorts']['group']['quality']
    assert group['identification_rate']['baseline'] is None
    assert group['failure_rate'] == 1
    assert group['missing'] == 1
    assert result['cohorts']['group']['latency']['generation']['baseline']['current_wall_ms']['count'] == 0
    assert result['complete'] is False


def test_judge_quality_direction_and_generation_not_executed(saved):
    saved.configure(kind='judge_eval', baseline_ref='j-0001', candidate_ref='j-0002', pack_ref='frozen-pack')
    saved.append(record(baseline=False, candidate=True, judge=True),
        record('group', baseline=True, candidate=False, judge=True))
    saved.trace()
    saved.trace(reference=GROUP_TRACE, case_id='group')
    result = metrics.summarize(saved.directory)
    private = result['cohorts']['private']
    assert private['quality']['definition'] == 'judge_correct_identification'
    assert private['quality']['direction'] == 'higher_is_better'
    assert private['quality']['identification_rate'] == dict(baseline=0, candidate=1)
    assert private['quality']['identification_delta_percentage_points'] == 100
    assert private['quality']['net_win_confirmed'] == 1
    assert result['cohorts']['group']['quality']['net_win_confirmed'] == -1
    assert result['cohorts']['all']['quality']['identified'] == dict(baseline=1, candidate=1)
    latency = private['latency']['generation']['baseline']
    assert latency['status'] == 'not_executed'
    assert latency['current_wall_ms']['mean_ms'] is None
    assert result['per_case'][0]['generation'] is None
    assert private['latency']['judge']['baseline']['model_attempt_ms']['mean_ms'] == 400


def test_whole_stage_cache_tracks_uncached_wall_and_historical_attempts(saved):
    saved.append(record())
    saved.trace([operation('generation', 'candidate', [call(BASELINE, 100, status='failed'),
        call(BASELINE, 200)], 450)], name='source', reference=SOURCE_TRACE)
    saved.trace([operation('generation', 'baseline', [cache('cache_hit', 'generation', origin=saved.origin)], 2)])
    baseline = metrics.summarize(saved.directory)['cohorts']['private']['latency']['generation']['baseline']
    assert baseline['current_wall_ms']['mean_ms'] == 2
    assert baseline['pre_response_cache_wall_ms']['mean_ms'] == 450
    assert baseline['model_attempt_ms']['mean_ms'] == 300
    assert baseline['current_calls']['attempts'] == 0
    assert baseline['historical_calls']['attempts'] == 2
    assert baseline['historical_calls']['failed_attempts'] == 1


@pytest.mark.parametrize('problem', ['model', 'duration'])
def test_uncached_wall_checks_source_model_independently_of_attempt_duration(saved, problem):
    saved.append(record())
    saved.trace([operation('generation', 'candidate',
        [call(CANDIDATE if problem == 'model' else BASELINE, None if problem == 'duration' else 100)], 450)],
        name='source', reference=SOURCE_TRACE)
    saved.trace([operation('generation', 'baseline', [cache('cache_hit', 'generation', origin=saved.origin)], 2)])
    baseline = metrics.summarize(saved.directory)['cohorts']['private']['latency']['generation']['baseline']
    assert baseline['current_wall_ms']['mean_ms'] == 2
    assert baseline['pre_response_cache_wall_ms']['mean_ms'] == (None if problem == 'model' else 450)
    assert baseline['model_attempt_ms']['mean_ms'] is None


def test_partial_request_cache_can_reuse_successful_request_from_failed_stage(saved):
    saved.append(record())
    source = operation('judge', 'candidate', [cache('cache_lookup'), call(JUDGE, 100, status='failed'),
        call(JUDGE, 200)], 450)
    source['status'] = 'failed'
    saved.trace([source], name='source', reference=SOURCE_TRACE)
    saved.trace([operation('judge', 'baseline', [cache('cache_hit', origin=saved.origin),
        cache('cache_lookup', key='other'), call(FEATURE, 400)], 500)])
    baseline = metrics.summarize(saved.directory)['cohorts']['private']['latency']['judge']['baseline']
    assert baseline['current_wall_ms']['mean_ms'] == 500
    assert baseline['pre_response_cache_wall_ms']['mean_ms'] is None
    assert baseline['pre_response_cache_wall_ms']['missing'] == 1
    assert baseline['model_attempt_ms']['mean_ms'] == 700
    assert baseline['historical_calls']['attempts'] == 2
    assert baseline['current_calls']['attempts'] == 1


@pytest.mark.parametrize('problem', ['wall', 'calls', 'trace', 'comparison', 'no_model'])
def test_evidence_gaps_do_not_become_zero_or_hide_independent_durations(saved, problem):
    saved.append(record())
    stage = operation('judge', 'baseline', [call(JUDGE, 500)], 600)
    if problem == 'wall':
        stage['elapsed_ms'] = None
    elif problem == 'calls':
        stage['events'][0]['elapsed_ms'] = None
    elif problem == 'no_model':
        stage['events'] = []
    saved.trace([stage], versions={**saved.spec, 'candidate_ref': 'g-0003'} if problem == 'comparison' else dict(saved.spec))
    if problem == 'trace':
        (saved.directory / 'traces' / (TRACE + '.json')).unlink()
    baseline = metrics.summarize(saved.directory)['cohorts']['private']['latency']['judge']['baseline']
    model = baseline['model_attempt_ms']
    wall = baseline['current_wall_ms']
    assert model['mean_ms'] == (500 if problem == 'wall' else None)
    assert wall['mean_ms'] == (600 if problem in ('calls', 'no_model') else None)
    assert model['missing'] == (0 if problem == 'wall' else 1)
    assert baseline['missing_timing_evidence']


def test_confirmation_rounds_and_retried_failures_not_mixed_with_initial_latency(saved):
    saved.append(dict(case_id='private', status='failed', reason='timeout'), record())
    confirmation = operation('judge', 'baseline', [call(JUDGE, 9000)], 10000)
    confirmation['round'] = 1
    saved.trace([operation('judge', 'baseline', [call(JUDGE, 400)], 500), confirmation])
    result = metrics.summarize(saved.directory)
    assert result['retries'] == 1
    assert result['cohorts']['private']['quality']['failures'] == 0
    assert result['cohorts']['private']['latency']['judge']['baseline']['model_attempt_ms']['mean_ms'] == 400
    assert result['run_window_speed']['totals']['real_calls']['attempt_time']['completed_ms'] == 9400


@pytest.mark.parametrize('field', ['chat_type', 'case_id'])
def test_frozen_plan_mismatch_rejected(saved, field):
    row = record()
    row[field] = 'group' if field == 'chat_type' else 'unknown'
    if field == 'case_id':
        with pytest.raises(ConfigError, match='冻结题单'):
            saved.append(row)
    else:
        saved.append(row)
        with pytest.raises(ConfigError, match='chat type'):
            metrics.summarize(saved.directory)


@pytest.mark.parametrize('change', ['net_win_confirmed', 'identified_candidate'])
def test_verified_summary_mismatch_rejected(saved, change):
    saved.append(record())
    verified = dict(pairs=1, failures=0, wins_confirmed=1, losses_confirmed=0, contested=0,
        net_win_confirmed=1, identified_baseline=1, identified_candidate=0)
    verified[change] += 1
    with pytest.raises(ConfigError, match='differs'):
        metrics.summarize(saved.directory, verified_metrics=verified)


@pytest.mark.parametrize('kind,dataset', [('gen_ab', 'development'), ('judge_eval', 'development'),
    ('gen_ab', 'fixed_test'), ('judge_eval', 'fixed_test')])
def test_unified_cli_read_only_metrics_for_gen_judge_and_fixed(saved, monkeypatch, capsys, kind, dataset):
    saved.configure(kind=kind, dataset=dataset, **(dict(baseline_ref='j-0001',
        candidate_ref='j-0002', pack_ref='pack') if kind == 'judge_eval' else {}))
    saved.append(record(baseline=False, candidate=True, judge=True) if kind == 'judge_eval' else record(),
        dict(case_id='group', status='failed', reason='timeout'))
    saved.trace()
    verified = dict(attempted=2, pairs=1, failures=1, wins_confirmed=1, losses_confirmed=0,
        contested=0, net_win_confirmed=1, identified_baseline=0 if kind == 'judge_eval' else 1,
        identified_candidate=1 if kind == 'judge_eval' else 0)
    write_json(saved.directory / 'state.json', dict(status='finished', metrics=verified))
    pointers = saved.directory.parent.parent / 'pointers.json'
    write_json(pointers, dict(gen='g-0001', judge='j-0001'))
    monkeypatch.setattr(versions, 'POINTERS_PATH', pointers)
    monkeypatch.setattr(versions, 'switch_instance', lambda name: saved.directory.parent.parent)
    monkeypatch.setattr(experiment, 'load_experiment', lambda name: saved.directory)

    def evidence(directory):
        if dataset == 'fixed_test':
            pytest.fail('fixed metrics must not execute the development promotion gate')
        return saved.spec, verified

    monkeypatch.setattr(gates, 'evidence', evidence)
    monkeypatch.setattr(metrics.protocol, 'decide', lambda *args: dict(verdict='merge_to_iteration_baseline'))
    before = {path: path.read_bytes() for path in saved.directory.parent.parent.rglob('*') if path.is_file()}
    assert check.main(['experiment', '--instance', 'demo', '--exp', 'current', '--metrics-report']) == 0
    result = json.loads(capsys.readouterr().out)
    summary = next(item for item in result['checks'] if item['name'] == 'experiment_metrics')['detail']
    assert summary['complete'] is True
    assert summary['cohorts']['all']['quality']['confirmation_evidence_validated'] is True
    assert summary['kind'] == kind
    assert all(path.read_bytes() == content for path, content in before.items())


def test_incomplete_gate_does_not_hide_metrics_or_claim_validated_evidence(saved, monkeypatch):
    saved.append(record())
    monkeypatch.setattr(experiment, 'load_experiment', lambda name: saved.directory)

    def incomplete(directory):
        raise ConfigError('formal evidence incomplete')

    monkeypatch.setattr(gates, 'evidence', incomplete)
    report = check.Report('experiment')
    check.check_experiment(SimpleNamespace(exp='current', gate='development', metrics_report=True), report)
    assert report.finish()['status'] == 'failed'
    gate, summary = report.value['checks']
    assert gate['status'] == 'failed'
    assert summary['status'] == 'passed'
    assert summary['detail']['cohorts']['all']['quality']['confirmation_evidence_validated'] is False
