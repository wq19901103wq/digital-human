"""Opt-in current-turn retrieval fusion and budget-aware example selection.

All inputs come from the existing time-filtered retriever. This policy only
changes which complete historical examples are shown, never their contents.
"""
from __future__ import annotations

from collections import Counter
from math import sqrt

from .few_shot import _terms

POLICY = 'context_mmr_v1'


def _similarity(left, right):
    denom = sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values()))
    return sum(v * right.get(k, 0) for k, v in left.items()) / denom if denom else 0.0


def select(context_rows, latest_rows, *, count, budget, retriever):
    """Fuse two ranked lists, then greedily select relevant, nonredundant shots.

    Weighted reciprocal ranks put 65% on the last utterance and 35% on recent
    context. MMR applies a 20% redundancy penalty after the first choice. Both
    weights and the rank constant are fixed by this versioned policy.
    """
    candidates = {}
    for route, weight, rows in (('context', 0.35, context_rows), ('latest', 0.65, latest_rows)):
        for rank, row in enumerate(rows, start=1):
            item = candidates.setdefault(str(row['id']), {'row': row, 'relevance': 0.0, 'ranks': {}})
            if route not in item['ranks']:
                item['ranks'][route] = rank
                item['relevance'] += weight * 11.0 / (10.0 + rank)
    for item in candidates.values():
        row = item['row']
        context = '\n'.join(str(x) for x in row['context'])
        reply = '\n'.join(str(x) for x in row['reply'])
        item['content'] = (context, reply)
        item['context_terms'] = Counter(_terms(context))
        item['reply_terms'] = Counter(_terms(reply))
    chosen, selected = [], []
    skipped = []
    remaining = list(candidates.values())
    while remaining and len(selected) < max(0, count):
        for item in remaining:
            redundancy = max((
                0.5 * _similarity(item['context_terms'], old['context_terms'])
                + 0.5 * _similarity(item['reply_terms'], old['reply_terms'])
                for old in chosen), default=0.0)
            item['utility'] = 0.8 * item['relevance'] - 0.2 * redundancy
        item = max(remaining, key=lambda x: (x['utility'], x['relevance'], str(x['row']['id'])))
        remaining.remove(item)
        row = item['row']
        if any(item['content'] == old['content'] for old in chosen):
            skipped.append({'id': row['id'], 'reason': 'duplicate_content'})
            continue
        proposed = [*selected, row]
        block, ids = retriever.render_selected(proposed, max_chars=budget)
        # Check the closing tag as well as every complete example. An oversized
        # row must not prevent a later, shorter candidate from using the budget.
        if ids != [x['id'] for x in proposed] or len(block) > budget:
            skipped.append({'id': row['id'], 'reason': 'char_budget'})
            continue
        chosen.append(item)
        selected.append(row)
    trace = {'policy': POLICY, 'candidate_count': len(candidates), 'skipped': skipped,
             'selected': [{'id': x['row']['id'], 'ranks': x['ranks'],
                           'relevance': x['relevance'], 'utility': x['utility']} for x in chosen]}
    return selected, trace
