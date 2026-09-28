"""Deterministic object Wikis composed from verified, immutable batch results.

No model calls or name matching: exact accounts share entities, other entities
stay local to their source artifact. Evidence identity includes the source file
hash and row locator, so local line numbers cannot collide across histories.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from ..config import sha256_file
from ..iteration.storage import atomic_write, read_json
from . import wiki_prompt, wiki_structured
from .wiki_library import digest


VERIFIED_BINDINGS = {'account_filename', 'verified_utterance_anchors',
                     'raw_member_snapshot', 'exact_group_id'}


def completed(job):
    return job['stage'] == 'complete' and job['reuse_eligible']


def readable(job):
    """A valid empty review is a completed task, not a body of knowledge."""
    return completed(job) and job.get('records', 0) > 0


def eligible(job):
    return readable(job) and job['binding'] in VERIFIED_BINDINGS


def _union(left, right):
    return sorted(set(left) | set(right))


def _identity(obj, data):
    subject = data['subject']
    if subject['self_account'] != obj['self_account']:
        raise ValueError('object owner differs from material')
    if obj['kind'] == 'person':
        valid = subject['account'] == obj['account'] and subject.get('kind', 'person') == 'person'
    else:
        valid = (subject.get('kind') == 'group' and not subject['account']
                 and subject.get('scope_id') == f"group:{obj['self_account']}:{obj['account']}")
    if not valid:
        raise ValueError('object identity differs from material')


def compose(obj, jobs):
    """Preserve all fields and roles; collapse only exact records with equal evidence."""
    selected = sorted((j for j in jobs if eligible(j)), key=lambda j: (j['batch'], j['id']))
    subject_id = 'entity-' + digest([obj['kind'], obj['self_account'], obj['account']])[:20]
    data = dict(schema='wiki_structured_v1', subject=dict(name=obj['title'], kind=obj['kind'],
        account=obj['account'] if obj['kind'] == 'person' else '', self_account=obj['self_account'],
        entity_id=subject_id, scope_id=f"{obj['kind']}:{obj['self_account']}:{obj['account']}"),
        runtime_usable=False, historical_input_status='not_admitted', reviews=[], evidence={})
    entities, tables, materials, origins, fact_sources = {}, {k: {} for k in wiki_prompt.TABLES}, {}, {}, {}
    input_records = 0
    for job in selected:
        source = job['files']['knowledge.json']
        if sha256_file(Path(source['path'])) != source['sha256']:
            raise ValueError('object material changed after inspection')
        original = read_json(source['path'])
        _identity(obj, original)
        material_id = 'material-' + source['sha256']
        material = materials.setdefault(material_id, dict(**source, jobs=[],
            coverage=deepcopy(original['coverage']), reviews=deepcopy(original.get('reviews', []))))
        material['jobs'].append(dict(batch=job['batch'], id=job['id'], binding=job['binding']))
        evidence_map = {}
        for local, evidence in original['evidence'].items():
            # Exclude path from identity: copying one source does not create new evidence.
            key = 'evidence-' + digest({k: v for k, v in evidence.items() if k != 'path'})
            saved = data['evidence'].setdefault(key, deepcopy(evidence))
            saved['paths'] = _union(saved.get('paths', []), [evidence['path']])
            evidence_map[local] = key
        entity_map = {}
        for entity in original['entities']:
            if entity['id'] == original['subject']['entity_id']:
                eid = subject_id
            elif entity.get('account'):
                eid = 'entity-' + digest(['person', obj['self_account'], entity['account']])[:20]
            else:
                eid = 'entity-' + digest([material_id, entity['id']])[:20]
            entity_map[entity['id']] = eid
            record = deepcopy(entity)
            record.update(id=eid, evidence_refs=[evidence_map[r] for r in entity['evidence_refs']])
            saved = entities.setdefault(eid, record)
            if saved['kind'] != record['kind'] or saved['account'] != record['account']:
                raise ValueError('incompatible entity types for an exact account')
            saved['labels'] = _union(saved.get('labels', [saved['label']]), record.get('labels', [record['label']]))
            saved['evidence_refs'] = _union(saved['evidence_refs'], record['evidence_refs'])
        for table, records in tables.items():
            for original_record in original[table]:
                input_records += 1
                record = deepcopy(original_record)
                local_id = record.pop('id')
                local_facts = record.pop('fact_ids', [])
                for field in ('subject_id', 'object_id', 'speaker_id', 'target_id'):
                    if record.get(field):
                        record[field] = entity_map[record[field]]
                for participant in record.get('participants', []):
                    participant['entity_id'] = entity_map[participant['entity_id']]
                record['evidence_refs'] = sorted({evidence_map[r] for r in record['evidence_refs']})
                if table == 'addresses':
                    # Keep the local row for review AND an unambiguous cross-file reference.
                    record['utterance_ref'] = evidence_map[str(record['utterance_line'])]
                key = wiki_prompt.TABLES[table] + '-' + digest(record)[:20]
                saved = records.setdefault(key, dict(id=key, **record, fact_ids=[]))
                for fact in local_facts:
                    fid = 'fact-' + digest([material_id, fact])[:20]
                    fact_sources[fid] = dict(material_id=material_id, fact_id=fact)
                    saved['fact_ids'] = _union(saved['fact_ids'], [fid])
                origin = dict(material_id=material_id, record_id=local_id)
                if origin not in origins.setdefault(key, []):
                    origins[key].append(origin)
    # An empty target entity may be absent from a source; do not invent its identity.
    data['entities'] = sorted(entities.values(), key=lambda e: (e['id'] != subject_id, e['id']))
    for table, records in tables.items():
        data[table] = sorted(records.values(), key=lambda r: (
            r.get('subject_id', '') != subject_id, r.get('subject_id', ''), r.get('field', ''),
            r.get('first_observed_at', 0), r['id']))
    data['coverage'] = dict(selection='verified_object_materials', materials=materials,
        input_records=input_records, records=sum(len(rs) for rs in tables.values()),
        unique_evidence=len(data['evidence']), gaps=[
            'only_verified_completed_materials_included', 'unresolved_entities_remain_material_local',
            'different_summaries_preserved_without_semantic_reconciliation'])
    data['provenance'] = dict(records=origins, facts=fact_sources)
    return data


def render(data):
    """Use the established sections with explicit account labels and source links."""
    view = deepcopy(data)
    for entity in view['entities']:
        if entity['account']:
            entity['label'] += '〔' + entity['account'] + '〕'
        elif entity['id'] != view['subject']['entity_id']:
            entity['label'] += '〔未确认实体 ' + entity['id'][-8:] + '〕'
    text = wiki_structured.render(view)
    if data['subject']['kind'] == 'group':
        text = text.replace('\n账号：\n', '\n群号：' + data['subject']['scope_id'].split(':', 2)[2] + '\n', 1)
        text = text.replace('## 人物属性\n', '## 群属性\n', 1)
    text += '\n## 记录来源与归属说明\n\n'
    text += '以下定位连接合并记录与原材料；同一来源重复出现不算独立佐证。\n\n'
    text += '| 记录 | 归属说明 | 性质 | 有效日期（若明确） | 来源哈希:行号 |\n| --- | --- | --- | --- | --- |\n'
    def cell(value):
        return str(value).replace('|', '／').replace('\n', ' ')
    for record in wiki_prompt.records(data):
        refs = ', '.join(data['evidence'][ref]['sha256'][:12] + ':' + str(data['evidence'][ref]['line'])
                         for ref in record['evidence_refs'])
        text += '| ' + ' | '.join(map(cell, [record['id'], record['attribution'],
            record['mode'] + ' / ' + record['polarity'],
            record['valid_from'] + ' ～ ' + record['valid_to'], refs])) + ' |\n'
    unresolved = [e for e in data['entities'] if not e['account'] and e['id'] != data['subject']['entity_id']]
    text += '\n## 待核实与覆盖限制\n\n'
    text += f'另有 {len(unresolved)} 个实体缺少跨材料身份依据，保留各自局部标识。'
    text += '同一来源的不同摘要暂时并列；材料归属明确不等于每条摘要语义均已核实。\n'
    for material_id, material in data['coverage']['materials'].items():
        for review in material['reviews']:
            if review['disposition'] == 'unresolved':
                text += '\n- ' + cell(review['reason']) + '（' + material_id[-8:] + ' / ' + review['fact_id'] + '）\n'
    return text


def artifacts(data):
    """All projections share the composed records; source text remains absent."""
    def encoded(value):
        return json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    records = wiki_prompt.records(data)
    sources = dict(records={r['id']: dict(evidence_refs=r['evidence_refs'], fact_ids=r['fact_ids'])
                           for r in records},
        entities={e['id']: e['evidence_refs'] for e in data['entities']}, evidence=data['evidence'],
        provenance=data['provenance'])
    files = {'knowledge.json': encoded(data), 'wiki.md': render(data),
        'wiki.xml': wiki_structured.to_xml(data),
        'cards.jsonl': ''.join(wiki_prompt.compact(c) + '\n' for c in wiki_prompt.packets(data)),
        'sources.json': encoded(sources)}
    manifest = dict(schema='wiki_object_export_v1', runtime_usable=False,
        historical_input_status='requires_case_projection', records=len(records),
        input_records=data['coverage']['input_records'], unique_evidence=len(data['evidence']),
        materials=len(data['coverage']['materials']),
        files={name: sha256(content.encode()).hexdigest() for name, content in files.items()})
    files['manifest.json'] = encoded(manifest)
    return files


def write(files, output):
    for name, content in files.items():
        path = Path(output) / name
        if not path.exists() or path.read_text() != content:
            atomic_write(path, content)
