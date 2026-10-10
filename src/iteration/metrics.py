"""Read-only experiment metrics with per-case latency and explicit coverage."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.infrastructure.config import ConfigError
from src.infrastructure.config import valid_name
from . import protocol, record_contract
from .reply_model_report import RecordedCalls, _distribution, _milliseconds, _quality, _run_window_speed


SIDES = ('baseline', 'candidate')
TIMES = ('current_wall_ms', 'pre_response_cache_wall_ms', 'model_attempt_ms')
EVIDENCE_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, ConfigError)


def _models(directory, spec):
    root = directory.parent.parent
    generation, judge = {}, {}
    for side in SIDES:
        if spec['kind'] == 'gen_ab':
            config = json.loads((root / 'generators' / valid_name(spec[side + '_ref']) / 'config.json').read_text())
            generation[side] = {chat: config['llm'][chat]['model'] for chat in ('group', 'private')}
        ref = spec[side + '_ref'] if spec['kind'] == 'judge_eval' else spec['judge_ref']
        config = json.loads((root / 'judges' / valid_name(ref) / 'config.json').read_text())
        judge[side] = sorted({value['model'] for name in ('llm', 'feature_llm')
            if isinstance(value := config.get(name), dict) and isinstance(value.get('model'), str)})
    return generation, judge


def _quality_metrics(rows, expected, kind, validated):
    if kind == 'judge_eval':
        normalized = []
        for row in rows:
            item = dict(row)
            if row['status'] == 'ok':
                item.update(identified_baseline=not row['baseline_correct'],
                    identified_candidate=not row['candidate_correct'])
                if row.get('flip_verified'):
                    item['flip_verified'] = {side + '_identified_final': not row[side + '_identified_final']
                        for side in SIDES}
            normalized.append(item)
        result = _quality(normalized, validated)
        result['identified'] = {side: result['pairs'] - count for side, count in result['identified'].items()}
        result['identification_rate'] = {side: 1 - rate if rate is not None else None
            for side, rate in result['identification_rate'].items()}
        delta = result['identification_delta_percentage_points']
        result['identification_delta_percentage_points'] = -delta if delta is not None else None
    else:
        result = _quality(rows, validated)
    result.update(expected=expected, missing=expected - len(rows),
        coverage_rate=len(rows) / expected if expected else None,
        definition='judge_correct_identification' if kind == 'judge_eval' else 'generated_reply_identified_as_ai',
        direction='higher_is_better' if kind == 'judge_eval' else 'lower_is_better')
    return result


def _timing(recorded, row, kind, side, models):
    result = {name: None for name in TIMES}
    result.update(response_cache_hits={}, evidence_issues={}, current_calls={}, historical_calls={})
    try:
        _, operation = recorded.selected_operation(row, kind, side)
        result['response_cache_hits'] = dict(Counter(event['data']['layer']
            for event in operation.get('events', []) if event.get('kind') == 'cache_hit'
            and event.get('data', {}).get('layer') in (kind, 'llm_request')))
    except EVIDENCE_ERRORS as exc:
        result['evidence_issues']['operation'] = f'{type(exc).__name__}: {exc}'
        return result
    try:
        result['current_wall_ms'] = _milliseconds(operation.get('elapsed_ms'))
    except EVIDENCE_ERRORS as exc:
        result['evidence_issues']['current_wall_ms'] = f'{type(exc).__name__}: {exc}'
    try:
        result['pre_response_cache_wall_ms'] = recorded._uncached_wall(operation, str(row['case_id']), model=models)
        if result['pre_response_cache_wall_ms'] is None:
            result['evidence_issues']['pre_response_cache_wall_ms'] = 'partial request cache prevents end-to-end reconstruction'
    except EVIDENCE_ERRORS as exc:
        result['evidence_issues']['pre_response_cache_wall_ms'] = f'{type(exc).__name__}: {exc}'
    try:
        timing = recorded.model_attempts(row, kind, side, models)
        result.update(model_attempt_ms=timing['model_ms'], current_calls=timing['current_calls'],
            historical_calls=timing['historical_calls'])
    except EVIDENCE_ERRORS as exc:
        result['evidence_issues']['model_attempt_ms'] = f'{type(exc).__name__}: {exc}'
    return result


def _latency_summary(rows, stage, side, *, executed):
    successful = [row for row in rows if row['status'] == 'ok']
    samples = [row[stage][side] for row in successful] if executed else []
    issues, hits = Counter(), Counter()
    for sample in samples:
        issues.update(sample['evidence_issues'].values())
        hits.update(sample['response_cache_hits'])
    result = dict(status='measured' if executed else 'not_executed', round=0,
        eligible_cases=len(successful) if executed else 0, response_cache_hits=dict(hits),
        missing_timing_evidence=dict(issues))
    for name in TIMES:
        values = [sample[name] for sample in samples if sample[name] is not None]
        result[name] = {**_distribution(values), 'missing': len(samples) - len(values),
            'coverage_rate': len(values) / len(samples) if samples else None}
    for origin in ('current', 'historical'):
        calls = {}
        for sample in samples:
            calls.update(sample[origin + '_calls'])
        result[origin + '_calls'] = dict(attempts=len(calls), failed_attempts=sum(call['failed'] for call in calls.values()),
            source_experiments=sorted({identity[0] for identity in calls}),
            completed_attempt_ms=round(sum(call['elapsed_ms'] for call in calls.values()), 3),
            failed_attempt_ms=round(sum(call['elapsed_ms'] for call in calls.values() if call['failed']), 3))
    if not executed:
        result['reason'] = 'frozen reply pack reused; generation latency was not measured in this experiment'
    return result


def summarize(directory, *, verified_metrics=None):
    """Aggregate saved evidence only; no scoring, generation or pointer changes."""
    directory = Path(directory)
    spec, ids = record_contract.require_registered(directory)
    planned = {str(row['case_id']): row for row in record_contract.planned_rows(spec)}
    if any(row.get('chat_type') not in ('group', 'private') for row in planned.values()):
        raise ConfigError('frozen case plan lacks a supported chat type')
    state = json.loads((directory / 'state.json').read_text())
    final, retries = protocol.summarize_final_records(directory / 'cases.jsonl')
    for row in final.values():
        record_contract.validate_case(row, spec, ids)
        if row.get('chat_type', planned[str(row['case_id'])]['chat_type']) != planned[str(row['case_id'])]['chat_type']:
            raise ConfigError('record chat type differs from frozen case plan')
    generation_models, judge_models = _models(directory, spec)
    recorded = RecordedCalls(directory, spec)
    per_case = []
    for case_id, case in planned.items():
        row = final.get(case_id)
        item = dict(case_id=case_id, chat_type=case['chat_type'], status=row['status'] if row else 'missing')
        item.update(generation=None, judge=None)
        if row and row['status'] == 'ok':
            item['judge'] = {side: _timing(recorded, row, 'judge', side, judge_models[side]) for side in SIDES}
            if spec['kind'] == 'gen_ab':
                item['generation'] = {side: _timing(recorded, row, 'generation', side,
                    generation_models[side][case['chat_type']]) for side in SIDES}
        per_case.append(item)
    cohorts = {}
    for cohort in ('all', 'group', 'private'):
        selected = [row for row in per_case if cohort == 'all' or row['chat_type'] == cohort]
        rows = [final[row['case_id']] for row in selected if row['status'] != 'missing']
        cohorts[cohort] = dict(quality=_quality_metrics(rows, len(selected), spec['kind'], verified_metrics is not None),
            latency={stage: {side: _latency_summary(selected, stage, side,
                executed=stage == 'judge' or spec['kind'] == 'gen_ab') for side in SIDES}
                for stage in ('generation', 'judge')})
    if verified_metrics is not None:
        quality = cohorts['all']['quality']
        for name in ('pairs', 'failures', 'wins_confirmed', 'losses_confirmed', 'contested', 'net_win_confirmed'):
            if quality[name] != verified_metrics[name]:
                raise ConfigError('metric summary differs from verified experiment metrics: ' + name)
        if any(quality['identified'][side] != verified_metrics['identified_' + side] for side in SIDES):
            raise ConfigError('metric identification differs from verified experiment metrics')
    profiles = [json.loads(path.read_text()) for path in sorted((directory / 'transport_profiles').glob('*.json'))]
    run_speed = _run_window_speed(directory, spec, generation_models,
        {model for models in judge_models.values() for model in models}, profiles, list(final.values()))
    for item in per_case:
        for stage in ('generation', 'judge'):
            if item[stage] is not None:
                for timing in item[stage].values():
                    for origin in ('current', 'historical'):
                        calls = timing.pop(origin + '_calls')
                        timing[origin + '_attempts'] = len(calls)
    return dict(schema=1, experiment=directory.name, kind=spec['kind'], status=state.get('status'),
        comparison={key: spec.get(key) for key in ('dataset', 'data_ref', 'pack_ref', 'baseline_ref', 'candidate_ref', 'judge_ref')},
        expected_cases=len(planned), final_records=len(final), retries=retries,
        complete=state.get('status') == 'finished' and set(final) == ids,
        models=dict(generation=generation_models, judge=judge_models), cohorts=cohorts,
        per_case=per_case, run_window_speed=run_speed,
        notes=[
            'Quality rates use final successful pairs; failures and missing cases have separate denominators and coverage.',
            'Latency samples are final successful pairs, round zero, one complete stage per case and side; they exclude failed cases and can have survivor bias.',
            'current_wall_ms includes local preparation and response-cache lookup for that stage, not experiment startup.',
            'pre_response_cache_wall_ms follows whole-stage cache provenance to a stage without response-cache hits; partial request caching cannot reconstruct uncached end-to-end latency.',
            'Response caching means framework generation, judge and llm_request caches, not provider prompt caching.',
            'model_attempt_ms sums completed model attempts within each case before taking the across-case mean; it includes failed attempts and is not end-to-end wall time.',
            'Current and historical calls are separated; historical service conditions are not concurrent controls or proof of a causal speedup.',
            'run_window_speed includes failed/retried/confirmation traces; cumulative failed attempt time is not elapsed wall-clock loss.',
            'Missing durations remain null with coverage, never zero; generation in Judge experiments is not measured.',
            'Metrics are read-only observations, not new scoring, promotion or adoption gates.'])
