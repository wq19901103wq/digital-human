"""Shared frozen-audit evidence tools. Extracted from scripts/legacy/verify_judge_dataset_results.py."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

BASE = ROOT / 'instances/example-agent'


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SavedCache:
    def __init__(self, path=None):
        path = Path(path) if path is not None else BASE / '.cache/results.sqlite3'
        self.db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row

    def get(self, key):
        row = self.db.execute('SELECT * FROM entries WHERE key=?', (key,)).fetchone()
        if row is None:
            return None
        require(hashlib.sha256(row['value'].encode()).hexdigest() == row['value_sha'], 'cache value hash')
        return {**dict(row), **{k: json.loads(row[k]) for k in ('identity', 'value', 'origin')}}


def verify_experiment(target, *, cache_required=False):
    """Recompute from last records and original traces, without runner helpers."""
    spec, state = read(target / 'spec.json'), read(target / 'state.json')
    require(state['status'] == 'finished', f'{target.name} is not finished')
    pack_path = BASE / 'judge_eval' / spec['pack_ref'] / 'pack.json'
    require(hashlib.sha256(pack_path.read_bytes()).hexdigest() == spec['pack_sha256'], 'pack hash')
    pack = read(pack_path)
    selected = [r['case_id'] for r in pack['rows']]
    require(len(selected) == len(set(selected)) == 1000, 'formal selection must contain exactly 1000 unique cases')
    records, line_count = {}, 0
    for line in (target / 'cases.jsonl').read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            records[row['case_id']] = row
            line_count += 1
    require(set(records) == set(selected), 'missing/extra case records')
    require(spec['protocol']['flip_extra_rounds'] == 2, 'original two-extra-round protocol changed')
    totals, counts, cache_keys = defaultdict(Counter), Counter(), {}
    judge_configs = {arm: read(BASE / 'judges' / spec[arm + '_ref'] / 'config.json')
                     for arm in ('baseline', 'candidate')}
    store = SavedCache()
    try:
        for case in pack['rows']:
            cid, row = case['case_id'], records[case['case_id']]
            if row['status'] == 'failed':
                counts['failures'] += 1
                continue
            require(row['status'] == 'ok', f'{cid}: invalid record status')
            trace = read(target / 'traces' / (row['trace_ref'] + '.json'))
            require(trace['status'] == 'ok' and trace['case'] == case, f'{cid}: trace/case mismatch')
            require(trace['case_id'] == cid, f'{cid}: trace identity')
            require(all(trace['versions'][k] == spec[k] for k in ('baseline_ref', 'candidate_ref', 'data_ref', 'pack_ref')),
                    f'{cid}: trace versions')
            require(all(type(row[k]) is bool for k in ('baseline_correct', 'candidate_correct')), f'{cid}: invalid votes')
            b, c = row['baseline_correct'], row['candidate_correct']
            expected = {(arm, r) for arm in ('baseline', 'candidate') for r in range(3 if b != c else 1)}
            operations = trace['operations']
            require(len(operations) == len(expected), f'{cid}: missing/extra operations')
            require({(o['branch'], o['round']) for o in operations} == expected, f'{cid}: incomplete independent rounds')
            votes, policies = defaultdict(dict), {}
            for op in operations:
                arm, round_index = op['branch'], op['round']
                require(op['kind'] == 'judge' and op['status'] == 'ok', f'{cid}: operation status')
                require(op['input']['candidate_replies'] == case['ai_replies'], f'{cid}: different AI replies')
                identified = op['result']['identified_ai']
                require(type(identified) is bool, f'{cid}: invalid trace vote')
                votes[arm][round_index] = identified
                verdict, judge_entry = None, None
                for event in op.get('events', []):
                    if event['kind'] == 'correction':
                        verdict = event['data']
                    if event['kind'] not in ('cache_hit', 'cache_lookup'):
                        continue
                    data = event['data']
                    entry = store.get(data['key'])
                    if entry is None:
                        require(event['kind'] != 'cache_hit', f'{cid}: missing cache hit')
                        continue  # A failed intermediate request need not have a saved entry.
                    policy = op['cache_policy']
                    expected_key = digest({'schema': 1, 'layer': data['layer'], 'identity': entry['identity'],
                        'partition': 'development', 'case_id': cid, 'round': round_index,
                        'sample_epoch': policy['sample_epoch'], 'epoch': policy['epoch']})
                    require(expected_key == data['key'], f'{cid}: wrong cache round/content identity')
                    require(entry['origin']['round'] == round_index, f'{cid}: reused another round')
                    key_scope = (cid, round_index)
                    require(cache_keys.setdefault(data['key'], key_scope) == key_scope, 'cache shared across independent rounds')
                    counts['cache_entries_checked'] += 1
                    if data['layer'] == 'judge':
                        judge_entry = entry
                        if event['kind'] == 'cache_hit':
                            verdict = entry['value']['last_verdict']
                if cache_required:
                    require(judge_entry is not None, f'{cid}: missing judge cache evidence')
                    require(judge_entry['value']['identified_ai'] == identified, f'{cid}: judge cache/result mismatch')
                    identity = judge_entry['identity']
                    require(identity['config'] == judge_configs[arm], f'{cid}: cached judge config mismatch')
                    require(identity['candidate_replies'] == case['ai_replies'], f'{cid}: cached AI replies mismatch')
                    require(identity['case'] == {k: case.get(k) for k in
                            ('context', 'human_reply', 'chat_type', 'source_chat_id', 'chat_name')}, f'{cid}: cached case content mismatch')
                    require(all(verdict[k] == v for k, v in judge_entry['value']['last_verdict'].items()),
                            f'{cid}: cached correction mismatch')
                require(verdict is not None, f'{cid}: missing correction evidence')
                require(verdict['human_option'] != verdict['candidate_option'] if identified
                        else verdict['human_option'] == verdict['candidate_option'], f'{cid}: correction/result mismatch')
                if round_index == 0:
                    require(identified == row[arm + '_correct'], f'{cid}: initial record/trace mismatch')
                    probability = verdict['small_model_probability_a']
                    require(isinstance(probability, (float, int)) and 0 <= probability <= 1, f'{cid}: invalid LR probability')
                    ai = verdict['candidate_option']
                    policies[arm] = {'hybrid': identified,
                        'lr_only': ('A' if probability >= .5 else 'B') != ai,
                        'initial_only': verdict['base']['human_option'] != ai}
            counts['pairs'] += 1
            counts['identified_baseline'] += b
            counts['identified_candidate'] += c
            counts['wins' if c and not b else 'losses' if b and not c else 'ties'] += 1
            if b != c:
                require(row.get('flip_verified') is True, f'{cid}: unverified disagreement')
                final = {}
                for arm in ('baseline', 'candidate'):
                    trace_votes = [votes[arm][r] for r in range(3)]
                    require(row[arm + '_votes'] == trace_votes, f'{cid}: saved/trace votes differ')
                    final[arm] = sum(trace_votes) >= 2
                    require(row[arm + '_identified_final'] == final[arm], f'{cid}: incorrect majority')
                counts['contested' if final['baseline'] == final['candidate'] else
                       'wins_confirmed' if final['candidate'] else 'losses_confirmed'] += 1
                counts['independently_verified_disagreements'] += 1
            else:
                require(not row.get('flip_verified'), f'{cid}: unexpected tie verification')
            for group in ('all', case['chat_type'], 'human_single' if len(case['human_reply']) == 1 else 'human_multi'):
                totals[group]['pairs'] += 1
                for arm, results in policies.items():
                    for policy, hit in results.items():
                        totals[group][arm + '_' + policy] += int(hit)
    finally:
        store.db.close()
    metrics = {k: counts[k] for k in ('pairs', 'failures', 'identified_baseline', 'identified_candidate',
               'wins', 'losses', 'ties', 'wins_confirmed', 'losses_confirmed', 'contested')}
    n = counts['wins_confirmed'] + counts['losses_confirmed']
    p = min(1., 2 * sum(math.comb(n, i) for i in range(min(counts['wins_confirmed'], counts['losses_confirmed']) + 1)) / 2**n)
    metrics.update(n=len(selected), attempted=len(selected), retries=line_count-len(records),
        failure_rate=round(counts['failures']/len(selected), 4),
        net_win_confirmed=counts['wins_confirmed']-counts['losses_confirmed'], sign_p=p,
        **{'identification_rate_' + arm: round(counts['identified_' + arm]/counts['pairs'], 4) if counts['pairs'] else 0.
           for arm in ('baseline', 'candidate')})
    require(metrics == state['metrics'], f'{target.name}: independently recomputed metrics differ')
    threshold = max(1, round(spec['protocol']['dev_min_net_win_rate'] * counts['pairs']))
    verdict = ('experiment_incomplete' if metrics['failure_rate'] >= spec['protocol']['max_failure_rate'] else
               'merge_to_iteration_baseline' if metrics['net_win_confirmed'] >= threshold else
               'reject' if metrics['net_win_confirmed'] < 0 else 'observe')
    require(verdict == state['verdict'], 'original protocol verdict differs')
    descriptive = {name: {**dict(c), 'rates': {k: v/c['pairs'] for k, v in c.items() if k != 'pairs'}} for name, c in totals.items()}
    return {'experiment': target.name, 'metrics': metrics, 'descriptive': descriptive, 'verdict': verdict,
            'checks': {'all_selected_cases_recorded': True, 'shared_ai_replies': True,
                       'independently_verified_disagreements': counts['independently_verified_disagreements'],
                       'cache_entries_checked': counts['cache_entries_checked']}}
