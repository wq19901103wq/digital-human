"""Read-only reply-model comparison from final records and recorded calls.

Generation wall time includes local work. Model time includes recorded API
attempts, including failures, and follows cache provenance rather than counting
a cache hit as a zero-latency model. Historical calls are not concurrent controls.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
import statistics

from src import tracing
from src.config import ConfigError, valid_name
from . import protocol


def _distribution(values):
    ordered = sorted(values)
    if not ordered:
        return dict(count=0, mean_ms=None, p50_ms=None, p95_ms=None)
    return dict(count=len(ordered), mean_ms=round(statistics.mean(ordered), 3),
        p50_ms=round(statistics.median(ordered), 3),
        p95_ms=round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 3))


def _milliseconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigError('missing or invalid recorded duration')
    return value


class RecordedCalls:
    def __init__(self, directory, spec):
        self.directory = Path(directory).resolve()
        self.root = self.directory.parent
        self.spec = spec
        self._read = lru_cache(maxsize=8)(self._read)

    def _read(self, experiment_id, trace_ref, case_id):
        directory = self.root / valid_name(experiment_id)
        if not directory.resolve().is_relative_to(self.root):
            raise ConfigError('cache source experiment escapes instance')
        trace = tracing.read(directory, trace_ref)
        versions = trace.get('versions', {})
        if (str(trace.get('case_id')) != case_id or
                versions.get('dataset') != self.spec['dataset'] or
                versions.get('data_ref') != self.spec['data_ref']):
            raise ConfigError('cache source case or dataset mismatch')
        return trace

    def _source(self, origin, case_id):
        experiment_id = origin['experiment_id']
        trace_ref = origin['trace_ref']
        index = origin['operation_index']
        trace = self._read(experiment_id, trace_ref, case_id)
        if type(index) is not int or not 0 <= index < len(trace['operations']):
            raise ConfigError('invalid cache source operation')
        operation = trace['operations'][index]
        if operation.get('kind') != 'generation' or operation.get('status') != 'ok':
            raise ConfigError('cache source is not a successful generation')
        return experiment_id, trace_ref, index, operation

    def _calls(self, experiment_id, trace_ref, index, operation, case_id, model,
               chain=(), request_key=None):
        identity = (experiment_id, trace_ref, index, request_key)
        if identity in chain or len(chain) >= 16:
            raise ConfigError('cyclic or excessive cache source chain')
        calls = {}
        active_key = None
        matched_key = False
        for event_index, event in enumerate(operation.get('events', [])):
            kind, data = event.get('kind'), event.get('data', {})
            if kind in ('cache_lookup', 'cache_hit') and data.get('layer') == 'llm_request':
                active_key = data.get('key')
                matched_key |= active_key == request_key
            include = request_key is None or active_key == request_key
            if kind == 'llm' and include:
                actual_model = event.get('request', {}).get('body', {}).get('model')
                if actual_model != model:
                    raise ConfigError('recorded reply model mismatch')
                if event.get('status') not in ('ok', 'failed'):
                    raise ConfigError('recorded model call is incomplete')
                call_id = (experiment_id, trace_ref, index, event_index)
                calls[call_id] = dict(elapsed_ms=_milliseconds(event.get('elapsed_ms')),
                    failed=event['status'] == 'failed')
            if kind == 'cache_hit' and include and data.get('layer') in ('generation', 'llm_request'):
                source = self._source(data['origin'], case_id)
                key = data.get('key') if data['layer'] == 'llm_request' else request_key
                calls.update(self._calls(*source, case_id, model, (*chain, identity), key))
        if request_key is not None and not matched_key:
            raise ConfigError('cached request key has no attributable recorded calls')
        if not calls:
            raise ConfigError('no recorded reply-model calls')
        return calls

    def generation(self, row, branch, model):
        trace_ref = row['trace_ref']
        case_id = str(row['case_id'])
        trace = self._read(self.directory.name, trace_ref, case_id)
        operations = [(index, operation) for index, operation in enumerate(trace['operations'])
            if operation.get('kind') == 'generation' and operation.get('branch') == branch
            and operation.get('round') == 0 and operation.get('status') == 'ok']
        if len(operations) != 1:
            raise ConfigError('missing or ambiguous initial generation')
        index, operation = operations[0]
        wall_ms = _milliseconds(operation.get('elapsed_ms'))
        calls = self._calls(self.directory.name, trace_ref, index, operation, case_id, model)
        current = {identity: call for identity, call in calls.items()
            if identity[:2] == (self.directory.name, trace_ref)}
        historical = {identity: call for identity, call in calls.items() if identity not in current}
        return dict(wall_ms=wall_ms, model_ms=sum(call['elapsed_ms'] for call in calls.values()),
            current_calls=current, historical_calls=historical)


def _quality(rows, evidence_validated):
    successful = [row for row in rows if row.get('status') == 'ok']
    pairs = len(successful)
    failures = [row for row in rows if row.get('status') != 'ok']
    failure_reasons = Counter(str(row.get('reason') or 'unspecified') for row in failures)
    identified = {side: sum(row['identified_' + side] is True for row in successful)
        for side in ('baseline', 'candidate')}
    confirmed = dict(wins=0, losses=0, contested=0)
    for row in successful:
        verified = row.get('flip_verified')
        if not isinstance(verified, dict):
            continue
        baseline = verified.get('baseline_identified_final')
        candidate = verified.get('candidate_identified_final')
        if type(baseline) is not bool or type(candidate) is not bool:
            raise ConfigError('missing saved confirmation outcomes')
        category = 'contested' if baseline == candidate else 'wins' if baseline else 'losses'
        confirmed[category] += 1
    rates = {side: identified[side] / pairs if pairs else None for side in identified}
    return dict(attempted=len(rows), pairs=pairs, failures=len(failures),
        paired_success_rate=(pairs / len(rows) if rows else None),
        failure_rate=(len(failures) / len(rows) if rows else None),
        failure_reasons=dict(sorted(failure_reasons.items())),
        identified=identified, identification_rate=rates,
        identification_delta_percentage_points=(round((rates['candidate'] - rates['baseline']) * 100, 3)
            if pairs else None),
        net_win_raw=identified['baseline'] - identified['candidate'],
        wins_confirmed=confirmed['wins'], losses_confirmed=confirmed['losses'],
        contested=confirmed['contested'], net_win_confirmed=confirmed['wins'] - confirmed['losses'],
        confirmation_evidence_validated=evidence_validated)


def _speed(rows, recorded, models):
    sides = {side: dict(values=[], walls=[], current={}, historical={}, errors={})
        for side in ('baseline', 'candidate')}
    paired = []
    for row in rows:
        if row.get('status') != 'ok':
            continue
        pair = {}
        for side, bucket in sides.items():
            try:
                timing = recorded.generation(row, side, models[side][row['chat_type']])
            except (OSError, ValueError, KeyError, TypeError, ConfigError) as exc:
                reason = f'{type(exc).__name__}: {exc}'
                bucket['errors'][reason] = bucket['errors'].get(reason, 0) + 1
                continue
            bucket['values'].append(timing['model_ms'])
            bucket['walls'].append(timing['wall_ms'])
            bucket['current'].update(timing['current_calls'])
            bucket['historical'].update(timing['historical_calls'])
            pair[side] = timing['model_ms']
        if len(pair) == 2:
            paired.append(pair)
    result = {}
    for side, bucket in sides.items():
        result[side] = dict(model_time=_distribution(bucket['values']),
            generation_wall_time=_distribution(bucket['walls']), missing_timing_evidence=bucket['errors'])
        for origin in ('current', 'historical'):
            calls = bucket[origin]
            result[side][origin + '_calls'] = dict(attempts=len(calls),
                failed_attempts=sum(call['failed'] for call in calls.values()),
                source_experiments=sorted({identity[0] for identity in calls}),
                durations=_distribution([call['elapsed_ms'] for call in calls.values()]))
    baseline = _distribution([pair['baseline'] for pair in paired])
    candidate = _distribution([pair['candidate'] for pair in paired])
    mean_baseline, mean_candidate = baseline['mean_ms'], candidate['mean_ms']
    result['paired'] = dict(cases=len(paired), baseline=baseline, candidate=candidate,
        mean_model_time_reduction_percentage=(round((1 - mean_candidate / mean_baseline) * 100, 3)
            if mean_baseline else None))
    return result


def summarize(directory, *, verified_metrics=None):
    directory = Path(directory)
    spec = json.loads((directory / 'spec.json').read_text())
    state = json.loads((directory / 'state.json').read_text())
    if spec.get('kind') != 'gen_ab':
        raise ConfigError('reply-model report requires a generator A/B experiment')
    models = {}
    for side in ('baseline', 'candidate'):
        ref = valid_name(spec[side + '_ref'])
        config = json.loads((directory.parent.parent / 'generators' / ref / 'config.json').read_text())
        models[side] = {chat: config['llm'][chat]['model'] for chat in ('group', 'private')}
    final, retries = protocol.summarize_final_records(directory / 'cases.jsonl')
    records = list(final.values())
    if any(row.get('chat_type') not in ('group', 'private') for row in records):
        raise ConfigError('final record lacks a supported chat type')
    recorded = RecordedCalls(directory, spec)
    cohorts = {}
    for cohort in ('all', 'group', 'private'):
        selected = records if cohort == 'all' else [row for row in records if row['chat_type'] == cohort]
        cohorts[cohort] = dict(quality=_quality(selected, verified_metrics is not None),
            speed=_speed(selected, recorded, models))
    if verified_metrics is not None:
        for name in ('pairs', 'failures', 'net_win_confirmed', 'wins_confirmed', 'losses_confirmed', 'contested'):
            if cohorts['all']['quality'][name] != verified_metrics[name]:
                raise ConfigError('reply-model summary differs from verified experiment metrics')
    profiles = []
    for path in sorted((directory / 'transport_profiles').glob('*.json')):
        profile = json.loads(path.read_text())
        profiles.append({key: profile.get(key) for key in
            ('pid', 'status', 'workers', 'started_at', 'updated_at', 'elapsed_seconds', 'metrics', 'timings_are_nested')})
    expected = spec.get('data_snapshot', {}).get('case_count')
    return dict(schema=1, experiment=directory.name, status=state.get('status'),
        comparison={key: spec.get(key) for key in ('dataset', 'data_ref', 'baseline_ref', 'candidate_ref', 'judge_ref')},
        models=models, expected_cases=expected, final_records=len(records), retries=retries,
        complete=state.get('status') == 'finished' and len(records) == expected,
        cohorts=cohorts, transport_profiles=profiles,
        notes=['Judge identification is a likeness measure, not objective factual accuracy; lower is better.',
            'Only initial generation of final successful pairs enters latency statistics; independent confirmation draws are excluded.',
            'Successful-pair latency and identification exclude failed cases and may have selection bias; failure rates and reasons must be reported alongside them.',
            'Model time sums API attempts, including failed attempts; local preparation and cache lookup time are separate.',
            'Cache provenance supplies original model durations, never fabricated zero-latency observations.',
            'Historical baseline calls are not simultaneous controls; environmental differences may affect speed.',
            'Transport profile spans are nested and concurrent; their totals must not be added as wall time.',
            'Stored confirmation outcomes are provisional unless existing formal evidence checks passed.'])
