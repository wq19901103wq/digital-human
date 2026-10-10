"""Shared knowledge-card representation, without maintenance or query policy."""
from __future__ import annotations

from hashlib import sha256
import json

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
    return [dict(record, kind=kind) for table, kind in TABLES.items() for record in data[table]]


def entity_refs(record):
    refs = {record[key] for key in ('subject_id', 'object_id', 'speaker_id', 'target_id')
            if record.get(key)}
    refs.update(person['entity_id'] for person in record.get('participants', []))
    return refs


def card(data, selected, *, prior=None):
    """Keep explicit ownership and uncertainty, never collapse unresolved names."""
    subject = data['subject']
    wanted = {subject['entity_id']} | {entity['id'] for entity in data['entities']
                                      if entity.get('account') == subject['self_account']}
    wanted.update(ref for record in selected for ref in entity_refs(record))
    people = []
    for entity in data['entities']:
        if entity['id'] not in wanted:
            continue
        account = entity.get('account', '')
        supported_name = prior is None or (entity.get('evidence_refs') and
                                          all(prior(ref) for ref in entity['evidence_refs']))
        people.append(dict(id=entity['id'], kind=entity['kind'], account=account,
            name=entity['label'] if supported_name else (account or '未确认实体'),
            role='self' if account == subject['self_account'] else 'other',
            identity_status=entity['identity_status']))
    self_id = next((person['id'] for person in people if person['role'] == 'self'), None)
    if self_id is None:
        raw = json.dumps([subject['self_account'], subject['self_account']],
                         ensure_ascii=False, sort_keys=True).encode()
        self_id = 'entity-' + sha256(raw).hexdigest()[:20]
        people.append(dict(id=self_id, kind='person', account=subject['self_account'],
            name=subject['self_account'], role='self', identity_status='exact_account'))
    return dict(schema='wiki_prompt_v1', subject_id=subject['entity_id'],
        self_person_id=self_id, identities=people,
        facts=[dict(ref=record['id'], kind=record['kind'], **{key: record[key] for key in FIELDS
                    if key in record and record[key] not in ('', [], None)}) for record in selected])
