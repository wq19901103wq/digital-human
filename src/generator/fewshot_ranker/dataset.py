"""Load the immutable, completed supervision without exposing answers to features."""
from collections import Counter
from pathlib import Path

from ...config import sha256_file
from ...iteration.storage import read_json
from ..history import eligible
from ..history_sources import digest, require
from ..ranker_labels import _saved


def load_dataset(inventory):
    inventory = Path(inventory).resolve()
    original = read_json(inventory / 'manifest.json')
    manifest = read_json(inventory / 'labeling/manifest.json')
    files = read_json(inventory / 'inventory.json')['files']
    require(manifest['original_manifest_sha256'] == digest(original), 'Label source inventory changed')
    groups, observations, source_hashes = [], [], {}
    seen = set()
    for selected in manifest['targets']:
        identity = selected['target_id']
        require(identity not in seen and selected['split'] in ('fit', 'validation'), 'Invalid target split')
        seen.add(identity)
        name = 'targets/' + identity + '.json'
        path = inventory / name
        require(path.resolve().is_relative_to(inventory) and sha256_file(path) == files[name],
                'Frozen target file changed')
        row = read_json(path)
        require(row['target_id'] == identity and row['manifest_sha256'] == digest(original) and
                row['payload_sha256'] == selected['payload_sha256'] ==
                digest({k: v for k, v in row.items() if k != 'payload_sha256'}), 'Target binding changed')
        require(all(not row['teacher_sources'].get(k) for k in manifest['policy']['teacher_overlap_fields']),
                'Target overlaps teacher supervision')
        indices, examples = [], set()
        for entry in row['candidates']:
            example = entry['example']
            require(example['id'] not in examples and eligible(example, row['target']), 'Invalid historical candidate')
            examples.add(example['id'])
            key = digest([identity, example['id']])
            saved = _saved(inventory / 'labeling/observations' / (key + '.json'),
                           digest([digest(manifest), row['payload_sha256'], entry]))
            require(saved and saved['status'] == 'complete' and saved['split'] == selected['split'] and
                    saved['target_id'] == identity and saved['example_id'] == example['id'],
                    'Training requires every bound label to be complete')
            source_hashes[key] = saved['payload_sha256']
            indices.append(len(observations))
            # No generated text, human answer, judge features or probabilities here.
            observations.append(dict(target_id=identity, example_id=example['id'], z=saved['z'],
                split=selected['split'], recall_rank=entry['union_rank']))
        groups.append(dict(target_id=identity, split=selected['split'], target=row['target'],
                           candidates=row['candidates'], indices=indices))
    require({g['split'] for g in groups} == {'fit', 'validation'}, 'Both frozen splits are required')
    boundary = min(g['target']['input_cutoff']['timestamp'] for g in groups if g['split'] == 'validation')
    answers = {v for g in groups if g['split'] == 'validation' for v in g['target']['reply_message_ids']}
    for group in groups:
        if group['split'] == 'fit':
            materials = [group['target'], *[e['example'] for e in group['candidates']]]
            require(all(m['source_span']['end_timestamp'] < boundary and not answers.intersection(
                m['context_message_ids'] + m['reply_message_ids']) for m in materials),
                'Fit material overlaps validation answers or crosses validation time')
    summary = dict(contexts=len(groups), observations=len(observations),
        unique_examples=len({r['example_id'] for r in observations}),
        labels=dict(Counter(str(r['z']) for r in observations)),
        splits={s: dict(contexts=sum(g['split'] == s for g in groups),
                       observations=sum(r['split'] == s for r in observations),
                       labels=dict(Counter(str(r['z']) for r in observations if r['split'] == s)))
                for s in ('fit', 'validation')}, validation_boundary=boundary)
    binding = dict(label_manifest_sha256=digest(manifest), observations_sha256=digest(source_hashes),
                   inventory_sha256=digest(files))
    return groups, observations, summary, binding
