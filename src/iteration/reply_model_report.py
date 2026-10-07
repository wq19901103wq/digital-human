"""Read-only reply-model comparison from final records and recorded calls.

Generation wall time includes local work. Model time includes recorded API
attempts, including failures, and follows cache provenance rather than counting
a cache hit as a zero-latency model. Historical calls are not concurrent controls.
"""
from __future__ import annotations

import json
import math
import re
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


def _generation_evidence(event, samples):
    status = event['status']
    request = event.get('request', {})
    body = request.get('body', {})
    attempt = request.get('attempt')
    samples['attempt_numbers'][str(attempt) if type(attempt) is int and attempt > 0 else 'unknown'] += 1
    if status == 'failed':
        error = event.get('error')
        message = str(error.get('message', '')).lower() if isinstance(error, dict) else ''
        category = ('http_429' if re.search(r"\bhttp\s+429\b", message) else
            'read_timeout' if 'read operation timed out' in message else
            'tls_eof' if 'eof' in message and ('ssl' in message or 'tls' in message) else 'other_or_unknown')
        samples['failure_categories'][category] += 1
    observations = samples['observations'].setdefault(status,
        dict(prompt_characters=[], requested_max_tokens=[], timeout_seconds=[], input_tokens=[],
             output_tokens=[], cache_read_input_tokens=[], content_types=Counter()))
    messages = body.get('messages')
    system = body.get('system', '')
    if (isinstance(messages, list) and isinstance(system, str) and
            all(isinstance(item, dict) and isinstance(item.get('content'), str) for item in messages)):
        observations['prompt_characters'].append(len(system) + sum(len(item['content']) for item in messages))
    for name, value in (('requested_max_tokens', body.get('max_tokens')),
                        ('timeout_seconds', request.get('timeout_seconds'))):
        if type(value) in (int, float) and math.isfinite(value) and value > 0:
            observations[name].append(value)
    response = event.get('response')
    if not isinstance(response, dict):
        return
    usage = response.get('usage')
    if isinstance(usage, dict):
        for name in ('input_tokens', 'output_tokens', 'cache_read_input_tokens'):
            value = usage.get(name)
            if type(value) is int and value >= 0:
                observations[name].append(value)
    content_types = response.get('content_types')
    if isinstance(content_types, list):
        observations['content_types'].update(kind for kind in ('thinking', 'text') if kind in content_types)


def _generation_evidence_summary(samples):
    return dict(attempt_numbers=dict(samples['attempt_numbers']),
        failure_categories=dict(samples['failure_categories']),
        observations_by_status={status: {name: dict(values) if name == 'content_types' else
            {key.removesuffix('_ms'): value for key, value in _distribution(values).items()}
            for name, values in observations.items()}
            for status, observations in samples['observations'].items()})


def _attempt_time(samples):
    return dict(completed_ms=round(sum(samples['durations']), 3),
        successful_ms=round(sum(samples['successful']), 3), failed_ms=round(sum(samples['failed']), 3))


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


def _judge_speed(directory, spec, models):
    buckets = {}
    errors = Counter()
    traces = 0
    for path in sorted((directory / 'traces').glob('*.json')):
        try:
            trace = tracing.read(directory, path.stem)
            if (trace.get('trace_ref') != path.stem or any(
                    trace.get('versions', {}).get(name) != spec.get(name)
                    for name in ('dataset', 'data_ref', 'judge_ref', 'baseline_ref', 'candidate_ref'))):
                raise ConfigError('judge trace frozen comparison mismatch')
            operations = trace['operations']
            if not isinstance(operations, list) or any(not isinstance(item, dict) for item in operations):
                raise ConfigError('invalid recorded operations')
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ConfigError) as exc:
            errors[f'{type(exc).__name__}: {exc}'] += 1
            continue
        traces += 1
        for operation in operations:
            if operation.get('kind') != 'judge':
                continue
            try:
                round_number = operation['round']
                if type(round_number) is not int or round_number < 0:
                    raise ConfigError('invalid judge round')
                phase = 'initial' if round_number == 0 else 'confirmation'
                bucket = buckets.setdefault(phase, dict(operations=Counter(), wall=[],
                    cache_hits=Counter(), completed=[], successful=[], failed=[], running=0,
                    attempts=Counter(), missing_durations=Counter(), observed_models=Counter()))
                status = operation['status']
                if status not in ('ok', 'failed', 'running'):
                    raise ConfigError('invalid judge operation status')
                bucket['operations'][status] += 1
                events = operation.get('events', [])
                if not isinstance(events, list):
                    raise ConfigError('invalid recorded judge events')
            except (KeyError, TypeError, ConfigError) as exc:
                errors[f'{type(exc).__name__}: {exc}'] += 1
                continue
            if status != 'running':
                try:
                    bucket['wall'].append(_milliseconds(operation.get('elapsed_ms')))
                except ConfigError as exc:
                    errors[f'{type(exc).__name__}: {exc}'] += 1
            for event in events:
                try:
                    kind = event.get('kind')
                    if kind == 'cache_hit':
                        layer = event.get('data', {}).get('layer')
                        if layer in ('judge', 'llm_request'):
                            bucket['cache_hits'][layer] += 1
                    if kind not in ('llm', 'codex'):
                        continue
                    request = event.get('request', {})
                    model = request.get('config' if kind == 'codex' else 'body', {}).get('model')
                    if model not in models:
                        raise ConfigError('recorded judge model mismatch')
                    status = event.get('status')
                    if status not in ('ok', 'failed', 'running'):
                        raise ConfigError('invalid judge model status')
                    bucket['attempts'][status] += 1
                    bucket['observed_models'][model] += 1
                    if status == 'running':
                        bucket['running'] += 1
                    else:
                        try:
                            duration = _milliseconds(event.get('elapsed_ms'))
                        except ConfigError:
                            bucket['missing_durations'][status] += 1
                            raise
                        bucket['completed'].append(duration)
                        bucket['successful' if status == 'ok' else 'failed'].append(duration)
                except (KeyError, TypeError, AttributeError, ConfigError) as exc:
                    errors[f'{type(exc).__name__}: {exc}'] += 1
    phases = {}
    for phase, bucket in buckets.items():
        phases[phase] = dict(operations=dict(bucket['operations']),
            operation_wall_time=_distribution(bucket['wall']),
            cache_hits=dict(bucket['cache_hits']), models=dict(bucket['observed_models']),
            current_calls=dict(completed_attempts=bucket['attempts']['ok'] + bucket['attempts']['failed'],
                successful_attempts=bucket['attempts']['ok'], failed_attempts=bucket['attempts']['failed'],
                missing_durations_by_status=dict(bucket['missing_durations']),
                recorded_running_attempts=bucket['running'], durations=_distribution(bucket['completed']),
                successful_durations=_distribution(bucket['successful']),
                failed_durations=_distribution(bucket['failed'])))
    return dict(traces=traces, expected_models=sorted(models), phases=phases,
        missing_timing_evidence=dict(sorted(errors.items())))


def _recorded_wall_time(windows):
    intervals = []
    for window in windows:
        started, ended = window['started_at'], window['ended_at']
        if intervals and started <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], ended)
        else:
            intervals.append([started, ended])
    if not intervals:
        return dict(started_at=None, ended_at=None, elapsed_span_ms=None,
            recorded_active_time_ms=None, uncovered_time_ms=None, overlapping_time_ms=None)
    active_seconds = sum(ended - started for started, ended in intervals)
    elapsed_seconds = intervals[-1][1] - intervals[0][0]
    profile_seconds = sum(window['ended_at'] - window['started_at'] for window in windows)
    return dict(started_at=intervals[0][0], ended_at=intervals[-1][1],
        elapsed_span_ms=round(elapsed_seconds * 1000, 3),
        recorded_active_time_ms=round(active_seconds * 1000, 3),
        uncovered_time_ms=round((elapsed_seconds - active_seconds) * 1000, 3),
        overlapping_time_ms=round((profile_seconds - active_seconds) * 1000, 3))


def _run_totals(windows, unattributed):
    statuses = Counter()
    missing = Counter()
    attempt_time = Counter()
    final_records = Counter()
    for window in [*windows, unattributed]:
        final_records.update(window['final_records'])
        for stage in window['stages'].values():
            statuses.update(stage['real_calls']['statuses'])
            attempt_time.update(stage['real_calls']['attempt_time'])
            missing.update({name.removeprefix('call_'): count for name, count in
                stage['missing_durations'].items() if name.startswith('call_')})
    return dict(recorded_wall_time=_recorded_wall_time(windows),
        retained_final_records=dict(final_records), real_calls=dict(statuses=dict(statuses),
            attempt_time={name: round(attempt_time[name], 3) for name in
                ('completed_ms', 'successful_ms', 'failed_ms')}, missing_durations=dict(missing)))


def _run_window_speed(directory, spec, models, judge_models, profiles, records):
    windows = []
    errors = Counter()
    for profile in profiles:
        try:
            started = _milliseconds(profile.get('started_at'))
            ended = _milliseconds(profile.get('updated_at'))
            if ended < started or type(profile.get('pid')) is not int:
                raise ConfigError('invalid recorded run window')
            windows.append(dict(pid=profile['pid'], started_at=started, ended_at=ended,
                recorded_wall_time_ms=round((ended - started) * 1000, 3),
                recorded_status=profile.get('status'), workers=profile.get('workers'),
                final_records=Counter(), case_ids=set(), stages={}))
        except ConfigError as exc:
            errors[str(exc)] += 1
    windows.sort(key=lambda item: item['started_at'])
    unattributed = dict(final_records=Counter(), case_ids=set(), stages={})

    def locate(timestamp):
        try:
            timestamp = _milliseconds(timestamp)
        except ConfigError:
            errors['missing or invalid recorded timestamp'] += 1
            return unattributed
        matches = [window for window in windows
            if window['started_at'] <= timestamp <= window['ended_at']]
        if len(matches) != 1:
            errors['timestamp outside or overlapping recorded run windows'] += 1
            return unattributed
        return matches[0]

    def bucket(window, stage, case_id):
        window['case_ids'].add(case_id)
        return window['stages'].setdefault(stage, dict(operations=Counter(), wall=[],
            cache_hits=Counter(), attempts=Counter(), durations=[], successful=[], failed=[],
            missing_durations=Counter(), models=Counter(), by_model={}, crossing_calls=0))

    final_traces = {row.get('trace_ref'): row for row in records}
    for path in sorted((directory / 'traces').glob('*.json')):
        try:
            trace = tracing.read(directory, path.stem)
            if (trace.get('trace_ref') != path.stem or any(
                    trace.get('versions', {}).get(name) != spec.get(name)
                    for name in ('dataset', 'data_ref', 'judge_ref', 'baseline_ref', 'candidate_ref'))):
                raise ConfigError('run-window trace frozen comparison mismatch')
            case_id = str(trace['case_id'])
            operations = trace['operations']
            if not isinstance(operations, list) or any(not isinstance(item, dict) for item in operations):
                raise ConfigError('invalid recorded operations')
            if row := final_traces.get(path.stem):
                if str(row['case_id']) != case_id:
                    raise ConfigError('final record trace case mismatch')
                window = locate(trace.get('ended_at'))
                window['case_ids'].add(case_id)
                window['final_records'][row['status']] += 1
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ConfigError) as exc:
            errors[f'{type(exc).__name__}: {exc}'] += 1
            continue
        for operation in operations:
            kind = operation.get('kind')
            if kind not in ('generation', 'judge'):
                continue
            try:
                round_number = operation['round']
                if type(round_number) is not int or round_number < 0:
                    raise ConfigError('invalid recorded round')
                phase = 'initial' if round_number == 0 else 'confirmation'
                if kind == 'generation':
                    side = operation['branch']
                    if side not in models:
                        raise ConfigError('invalid generation side')
                    chat_type = trace.get('case', {}).get('chat_type')
                    if chat_type not in models[side]:
                        raise ConfigError('run-window trace lacks a supported chat type')
                    expected_models = {models[side][chat_type]}
                    stage = f'generation_{side}_{phase}'
                else:
                    expected_models = judge_models
                    stage = f'judge_{phase}'
                status = operation['status']
                if status not in ('ok', 'failed', 'running'):
                    raise ConfigError('invalid recorded operation status')
                operation_bucket = bucket(locate(operation.get('started_at')), stage, case_id)
                operation_bucket['operations'][status] += 1
                if status != 'running':
                    try:
                        operation_bucket['wall'].append(_milliseconds(operation.get('elapsed_ms')))
                    except ConfigError:
                        operation_bucket['missing_durations']['operation_' + status] += 1
                events = operation.get('events', [])
                if not isinstance(events, list):
                    raise ConfigError('invalid recorded events')
                for event in events:
                    event_kind = event.get('kind')
                    layer = event.get('data', {}).get('layer')
                    if event_kind == 'cache_hit' and layer in ('generation', 'judge', 'llm_request'):
                        bucket(locate(event.get('started_at')), stage, case_id)['cache_hits'][layer] += 1
                    if event_kind not in ('llm', 'codex'):
                        continue
                    request = event.get('request', {})
                    model = request.get('config' if event_kind == 'codex' else 'body', {}).get('model')
                    if model not in expected_models:
                        raise ConfigError('recorded run-window model mismatch')
                    status = event.get('status')
                    if status not in ('ok', 'failed', 'running'):
                        raise ConfigError('invalid recorded model status')
                    window = locate(event.get('started_at'))
                    event_bucket = bucket(window, stage, case_id)
                    event_bucket['attempts'][status] += 1
                    event_bucket['models'][model] += 1
                    model_bucket = event_bucket['by_model'].setdefault(model,
                        dict(statuses=Counter(), durations=[], successful=[], failed=[], missing_durations=Counter(),
                             generation_evidence=dict(attempt_numbers=Counter(), failure_categories=Counter(), observations={})))
                    model_bucket['statuses'][status] += 1
                    if kind == 'generation' and event_kind == 'llm':
                        _generation_evidence(event, model_bucket['generation_evidence'])
                    if status == 'running':
                        continue
                    if window is not unattributed:
                        try:
                            event_bucket['crossing_calls'] += _milliseconds(event.get('ended_at')) > window['ended_at']
                        except ConfigError:
                            errors['missing or invalid recorded call end timestamp'] += 1
                    try:
                        duration = _milliseconds(event.get('elapsed_ms'))
                    except ConfigError:
                        event_bucket['missing_durations']['call_' + status] += 1
                        model_bucket['missing_durations'][status] += 1
                        continue
                    event_bucket['durations'].append(duration)
                    event_bucket['successful' if status == 'ok' else 'failed'].append(duration)
                    model_bucket['durations'].append(duration)
                    model_bucket['successful' if status == 'ok' else 'failed'].append(duration)
            except (KeyError, TypeError, AttributeError, ConfigError) as exc:
                errors[f'{type(exc).__name__}: {exc}'] += 1

    for window in [*windows, unattributed]:
        window['final_records'] = dict(window['final_records'])
        window['case_ids'] = sorted(window['case_ids'])
        for stage, samples in window['stages'].items():
            window['stages'][stage] = dict(operations=dict(samples['operations']),
                operation_wall_time=_distribution(samples['wall']), cache_hits=dict(samples['cache_hits']),
                models=dict(samples['models']), missing_durations=dict(samples['missing_durations']),
                real_calls_by_model={model: dict(statuses=dict(values['statuses']),
                    attempt_time=_attempt_time(values),
                    durations=_distribution(values['durations']),
                    successful_durations=_distribution(values['successful']),
                    failed_durations=_distribution(values['failed']),
                    missing_durations=dict(values['missing_durations']),
                    **(dict(generation_evidence=_generation_evidence_summary(values['generation_evidence']))
                       if values['generation_evidence']['observations'] else {}))
                    for model, values in samples['by_model'].items()},
                real_calls=dict(statuses=dict(samples['attempts']),
                    attempt_time=_attempt_time(samples),
                    durations=_distribution(samples['durations']),
                    successful_durations=_distribution(samples['successful']),
                    failed_durations=_distribution(samples['failed']),
                    crossing_window_end=samples['crossing_calls']))
    return dict(windows=windows, unattributed=unattributed, totals=_run_totals(windows, unattributed),
        missing_timing_evidence=dict(sorted(errors.items())),
        notes=['Windows use recorded process profile start/update timestamps, not cumulative current_calls.',
            'Total elapsed span runs from the first recorded start to the last recorded update; active wall time is the interval union, never the sum of overlapping profiles.',
            'Uncovered wall time has no recorded process profile; it is not evidence of idle time or a diagnosis of why execution paused.',
            'Total real-call attempt time includes unattributed calls once, excludes cache history and incomplete or missing durations, and is not wall-clock loss.',
            'Retained final records reflect current surviving results, not the number of results ever written in each window; later retries may replace earlier failures.',
            'Recorded running status is not a liveness check; a crashed process may leave it unchanged.',
            'Operations and direct model/cache events are assigned independently by their own start timestamps.',
            'Calls crossing a recorded window end are counted explicitly; missing timestamps remain unattributed.',
            'Cache hits do not import historical durations into real-call samples or create zero-latency calls.',
            'Attempt-time totals sum recorded completed API attempts, not process wall time; parallel and nested times cannot be added as elapsed experiment time.',
            'Generation request sizes and provider-reported usage are observations, not token estimates or a diagnosis of server internals; missing observations remain missing.',
            'Different case sets, retries, cache warmth and service conditions prevent a causal before/after speedup claim.'])


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
    judge_config = json.loads((directory.parent.parent / 'judges' /
        valid_name(spec['judge_ref']) / 'config.json').read_text())
    judge_models = {value['model'] for name in ('llm', 'feature_llm')
        if isinstance(value := judge_config.get(name), dict) and isinstance(value.get('model'), str)}
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
        cohorts=cohorts, judge_speed=_judge_speed(directory, spec, judge_models), transport_profiles=profiles,
        run_window_speed=_run_window_speed(directory, spec, models, judge_models, profiles, records),
        notes=['Judge identification is a likeness measure, not objective factual accuracy; lower is better.',
            'Only initial generation of final successful pairs enters latency statistics; independent confirmation draws are excluded.',
            'Successful-pair latency and identification exclude failed cases and may have selection bias; failure rates and reasons must be reported alongside them.',
            'Model time sums API attempts, including failed attempts; local preparation and cache lookup time are separate.',
            'Cache provenance supplies original model durations, never fabricated zero-latency observations.',
            'Judge latency includes all current trace attempts, including failed and unfinished cases, split by initial and confirmation rounds.',
            'Judge cache hits and recorded-running attempts have no fabricated duration and are excluded from completed-call distributions.',
            'Historical baseline calls are not simultaneous controls; environmental differences may affect speed.',
            'Transport profile spans are nested and concurrent; their totals must not be added as wall time.',
            'Stored confirmation outcomes are provisional unless existing formal evidence checks passed.'])
