"""Offline structured Wiki cards and provenance for review.

Offline packets are not historical model input. Memory owns case admission;
the shared card representation keeps provenance in a separate sidecar.
"""
from __future__ import annotations

import json
from pathlib import Path

from ...config import sha256_file
from ...iteration.storage import atomic_write, read_json
from .. import knowledge_view
from ..knowledge_view import card, compact, records


def packets(data, max_chars=16000):
    """Split without dropping records; an oversized atomic record stands alone."""
    if max_chars < 1000:
        raise ValueError('packet size must be >= 1000 characters')
    all_records = records(data)
    base = card(data, [])
    complete = card(data, all_records)
    base_ids = {entity['id'] for entity in base['identities']}
    identity_chars = {}
    for entity in complete['identities']:
        if entity['id'] not in base_ids:
            identity_chars[entity['id']] = (identity_chars.get(entity['id'], 0) +
                                            len(compact(entity)) + 1)
    base_chars = len(compact(base))
    current, included = [], set(base_ids)
    current_chars = base_chars
    for record, fact in zip(all_records, complete['facts']):
        fact_chars = len(compact(fact))
        refs = knowledge_view.entity_refs(record)
        added_chars = sum(identity_chars.get(ref, 0) for ref in refs - included)
        if current and current_chars + fact_chars + 1 + added_chars > max_chars:
            yield card(data, current)
            current, included = [], set(base_ids)
            current_chars = base_chars
            added_chars = sum(identity_chars.get(ref, 0) for ref in refs - included)
        current_chars += fact_chars + bool(current) + added_chars
        current.append(record)
        included.update(refs)
    if current:
        yield card(data, current)


def _write_unchanged(path, content):
    try:
        if path.read_text(encoding='utf-8') == content:
            return
    except FileNotFoundError:
        pass
    atomic_write(path, content)


def export(source, output, *, max_chars=16000):
    source, output = Path(source), Path(output)
    data = read_json(source)
    cards = list(packets(data, max_chars))
    _write_unchanged(output / 'cards.jsonl', ''.join(compact(c) + '\n' for c in cards))
    all_records = records(data)
    sources = dict(
        records={r['id']: dict(evidence_refs=r['evidence_refs'], fact_ids=r.get('fact_ids', []))
                 for r in all_records},
        entities={e['id']: e['evidence_refs'] for e in data['entities']}, evidence=data['evidence'])
    _write_unchanged(output / 'sources.json', json.dumps(sources, ensure_ascii=False, indent=2))
    report = dict(schema='wiki_prompt_export_v1', source=str(source.resolve()),
        source_sha256=sha256_file(source), format='wiki_prompt_v1',
        runtime_usable=False, historical_input_status='requires_case_projection',
        records=len(all_records), packets=len(cards), max_chars=max_chars,
        oversized_packets=sum(len(compact(c)) > max_chars for c in cards),
        files={name: sha256_file(output / name) for name in ('cards.jsonl', 'sources.json')})
    _write_unchanged(output / 'manifest.json', json.dumps(report, ensure_ascii=False, indent=2))
    return report
