"""Summarize saved ownership decisions, without model calls or raw chat exports."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .. import tracing
from ..iteration.protocol import summarize_final_records


def _observation(op, event_kind='ownership_features'):
    events = op.get('events', [])
    mappings = [e['data'] for e in events if e.get('kind') == 'blind_mapping']
    decisions = [e['data']['decision'] for e in events
                 if e.get('kind') == event_kind]
    if len(mappings) != 1 or len(decisions) != 1:
        raise ValueError('ownership or blind-mapping evidence missing/ambiguous')
    human = mappings[0].get('human_option')
    if human not in ('A', 'B'):
        raise ValueError('invalid human mapping')
    decision = decisions[0]
    base, final = (decision[k] for k in ('base_probability_a', 'probability_a'))
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) and 0 <= v <= 1 for v in (base, final)):
        raise ValueError('invalid saved probability')
    base_correct = ('A' if base >= .5 else 'B') == human
    final_correct = ('A' if final >= .5 else 'B') == human
    if op.get('result', {}).get('identified_ai') is not final_correct:
        raise ValueError('saved result differs from ownership decision')
    flags = decision['flags']
    if any(not isinstance(flags.get('option_' + side), list) for side in ('A', 'B')):
        raise ValueError('missing ownership flags')
    generated = 'B' if human == 'A' else 'A'
    h, g = bool(flags['option_' + human]), bool(flags['option_' + generated])
    features = {side: sorted({item.get('feature', 'ownership') for item in flags['option_' + side]})
                for side in (human, generated)}
    return dict(policy=decision['policy'], human_flagged=h, generated_flagged=g,
                human_features=features[human], generated_features=features[generated],
                both_flagged=h and g, triggered=bool(decision['triggered']),
                base_correct=base_correct, corrected_correct=final_correct,
                helped=not base_correct and final_correct,
                harmed=base_correct and not final_correct)


def _summary(rows, expected):
    counts = {key: sum(row[key] for row in rows) for key in (
        'human_flagged', 'generated_flagged', 'both_flagged', 'triggered',
        'base_correct', 'corrected_correct', 'helped', 'harmed')}
    n = len(rows)
    breakdown = {}
    for feature in sorted({f for row in rows for key in ('human_features', 'generated_features')
                           for f in row[key]}):
        involved = [row for row in rows if feature in row['human_features'] + row['generated_features']]
        breakdown[feature] = dict(
            human_flagged=sum(feature in row['human_features'] for row in involved),
            generated_flagged=sum(feature in row['generated_features'] for row in involved),
            helped_when_present=sum(row['helped'] for row in involved),
            harmed_when_present=sum(row['harmed'] for row in involved))
    return dict(expected=expected, observed=n, missing=expected - n, counts=counts,
                feature_breakdown=breakdown,
                rates={key: counts[key] / n if n else None for key in (
                    'human_flagged', 'generated_flagged', 'base_correct', 'corrected_correct')},
                net_corrected=counts['helped'] - counts['harmed'],
                policies=dict(Counter(row['policy'] for row in rows)))


def build_report(directory: Path, event_kind='ownership_features') -> dict:
    """Main metrics use final successful cases, candidate round zero only.

    Supplementary rounds are reported separately; missing provenance, including
    unresolved whole-Judge cache hits, is a coverage gap and never a zero flag.
    """
    directory = Path(directory)
    if event_kind not in ('ownership_features', 'contribution_features'):
        raise ValueError('unknown correction feature family')
    spec = json.loads((directory / 'spec.json').read_text())
    if spec.get('kind') != 'judge_eval':
        raise ValueError('ownership report requires a Judge comparison')
    final, retries = summarize_final_records(directory / 'cases.jsonl')
    successful = [row for row in final.values() if row.get('status') == 'ok']
    main, supplemental, issues, changed = [], [], [], []
    extra_expected = 0
    for row in successful:
        cid = str(row['case_id'])
        try:
            trace = tracing.read(directory, row.get('trace_ref', ''))
            if trace.get('case_id') != cid or trace.get('status') != 'ok':
                raise ValueError('trace case/status differs from final record')
            operations = [op for op in trace.get('operations', [])
                          if op.get('kind') == 'judge' and op.get('branch') == 'candidate'
                          and op.get('status') == 'ok']
            first = [op for op in operations if op.get('round') == 0]
            if len(first) != 1:
                raise ValueError('candidate round zero missing/ambiguous')
            observation = _observation(first[0], event_kind)
            if observation['corrected_correct'] is not row['candidate_correct']:
                raise ValueError('final row differs from first-round decision')
            main.append(observation)
            if observation['triggered'] or observation['human_flagged']:
                changed.append(dict(case_id=cid, trace_ref=row['trace_ref'], **observation))
            for op in operations:
                if not isinstance(op.get('round'), int) or op['round'] <= 0:
                    continue
                extra_expected += 1
                try:
                    supplemental.append(_observation(op, event_kind))
                except (ValueError, KeyError, TypeError) as exc:
                    issues.append(dict(case_id=cid, round=op['round'], reason=str(exc)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(dict(case_id=cid, round=0, reason=str(exc)))
    paired = Counter()
    for row in successful:
        baseline, candidate = row['baseline_correct'], row['candidate_correct']
        paired.update(baseline_correct=int(baseline), candidate_correct=int(candidate),
                      wins=int(candidate and not baseline), losses=int(baseline and not candidate))
    return dict(schema_version=1, experiment=directory.name, feature_family=event_kind,
                conditions={k: spec.get(k) for k in (
                    'data_ref', 'pack_ref', 'baseline_ref', 'candidate_ref', 'dataset')},
                final_cases=len(final), successful_cases=len(successful), retries=retries,
                paired_first_round=dict(paired), main=_summary(main, len(successful)),
                supplemental=_summary(supplemental, extra_expected),
                evidence_gaps=issues, flagged_or_overridden_cases=changed,
                interpretation=[
                    'Human flags measure correction-rule flags on observed human replies; '
                    'they are not a standalone classifier false-positive rate.',
                    'Generated flags are not error recall without independent error labels.',
                    'Help/harm compares the candidate correction to its own base score; '
                    'paired_first_round compares the two experimental versions.',
                    'Feature breakdown counts each feature once per option. Help/harm when present '
                    'can overlap across features and is not a causal ablation.',
                    'Promotion uses the experiment confirmed-round metrics, not this first-round report.',
                    'Missing evidence is excluded from rates and shown in coverage; no model requests made.'])
