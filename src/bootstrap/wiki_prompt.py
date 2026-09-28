"""Compact structured Wiki cards, shared by offline export and case projection.

Offline packets are review material. Only the generator's source/time filter can
admit records into an actual historical prompt. Provenance stays in a sidecar.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import sha256_file
from ..iteration.storage import atomic_write, read_json, write_json
from .wiki_library import digest

TABLES = {'attributes': 'attribute', 'relations': 'relation', 'events': 'event',
          'addresses': 'address'}
FIELDS = ('subject_id', 'object_id', 'speaker_id', 'target_id', 'field', 'value',
          'predicate', 'detail', 'event_type', 'description', 'status', 'participants',
          'details', 'term', 'usage', 'time_scope', 'valid_from', 'valid_to',
          'temporal_kind', 'mode', 'polarity', 'attribution', 'chat_ids',
          'first_observed_at', 'last_observed_at')


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def records(data):
    if data.get('schema') != 'wiki_structured_v1':
        raise ValueError('compact cards require wiki_structured_v1')
    return [dict(r, kind=kind) for table, kind in TABLES.items() for r in data[table]]


def entity_refs(record):
    refs = {record[k] for k in ('subject_id', 'object_id', 'speaker_id', 'target_id')
            if record.get(k)}
    refs.update(p['entity_id'] for p in record.get('participants', []))
    return refs


def card(data, selected, *, prior=None):
    """Keep explicit ownership and uncertainty, never collapse unresolved names."""
    subject = data['subject']
    wanted = {subject['entity_id']} | {e['id'] for e in data['entities']
                                      if e.get('account') == subject['self_account']}
    wanted.update(ref for record in selected for ref in entity_refs(record))
    people = []
    for entity in data['entities']:
        if entity['id'] not in wanted:
            continue
        account = entity.get('account', '')
        supported_name = prior is None or (entity.get('evidence_refs') and
                                          all(prior(r) for r in entity['evidence_refs']))
        people.append(dict(id=entity['id'], kind=entity['kind'], account=account,
            name=entity['label'] if supported_name else (account or '未确认实体'),
            role='self' if account == subject['self_account'] else 'other',
            identity_status=entity['identity_status']))
    self_id = next((p['id'] for p in people if p['role'] == 'self'), None)
    if self_id is None:
        self_id = 'entity-' + digest([subject['self_account'], subject['self_account']])[:20]
        people.append(dict(id=self_id, kind='person', account=subject['self_account'],
            name=subject['self_account'], role='self', identity_status='exact_account'))
    return dict(schema='wiki_prompt_v1', subject_id=subject['entity_id'],
        self_person_id=self_id, identities=people,
        facts=[dict(ref=r['id'], kind=r['kind'], **{k: r[k] for k in FIELDS
                    if k in r and r[k] not in ('', [], None)}) for r in selected])


def project(data, accounts, chat, prior, *, max_chars=16000):
    """Only relevant exact accounts; all supporting sources must pass prior()."""
    if data['subject']['self_account'] != chat.split(':', 2)[1]:
        return None
    # Loading a friend's page just because it mentions self would expose every
    # page. Select the page by its own account, then preserve related entities.
    if data['subject']['account'] not in accounts:
        return None
    eligible = [r for r in records(data) if not r.get('scope_case_id')
                and r.get('evidence_refs') and all(prior(ref) for ref in r['evidence_refs'])]
    eligible.sort(key=lambda r: (chat in r.get('chat_ids', []),
                                r.get('last_observed_at', 0), r['id']), reverse=True)
    selected = []
    for record in eligible:
        candidate = card(data, selected + [record], prior=prior)
        # Reserve space for the final coverage annotation.
        if len(compact(candidate)) + 200 <= max_chars:
            selected.append(record)
    result = card(data, selected, prior=prior)
    result['coverage'] = dict(eligible=len(eligible), included=len(selected),
                              omitted_for_budget=len(eligible) - len(selected))
    return result


def packets(data, max_chars=16000):
    """Split without dropping records; an oversized atomic record stands alone."""
    if max_chars < 1000:
        raise ValueError('packet size must be >= 1000 characters')
    current = []
    for record in records(data):
        if current and len(compact(card(data, current + [record]))) > max_chars:
            yield card(data, current)
            current = []
        current.append(record)
    if current:
        yield card(data, current)


def export(source, output, *, max_chars=16000):
    source, output = Path(source), Path(output)
    data = read_json(source)
    cards = list(packets(data, max_chars))
    atomic_write(output / 'cards.jsonl', ''.join(compact(c) + '\n' for c in cards))
    all_records = records(data)
    write_json(output / 'sources.json', dict(
        records={r['id']: dict(evidence_refs=r['evidence_refs'], fact_ids=r.get('fact_ids', []))
                 for r in all_records},
        entities={e['id']: e['evidence_refs'] for e in data['entities']}, evidence=data['evidence']))
    report = dict(schema='wiki_prompt_export_v1', source=str(source.resolve()),
        source_sha256=sha256_file(source), format='wiki_prompt_v1',
        runtime_usable=False, historical_input_status='requires_case_projection',
        records=len(all_records), packets=len(cards), max_chars=max_chars,
        oversized_packets=sum(len(compact(c)) > max_chars for c in cards),
        files={name: sha256_file(output / name) for name in ('cards.jsonl', 'sources.json')})
    write_json(output / 'manifest.json', report)
    return report
