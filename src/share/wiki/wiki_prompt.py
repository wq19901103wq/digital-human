"""Offline structured Wiki cards and provenance for review.

Offline packets are not historical model input. Memory owns case admission;
the shared card representation keeps provenance in a separate sidecar.
"""
from __future__ import annotations

from pathlib import Path

from ...config import sha256_file
from ...iteration.storage import atomic_write, read_json, write_json
from ..knowledge_view import card, compact, records


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
