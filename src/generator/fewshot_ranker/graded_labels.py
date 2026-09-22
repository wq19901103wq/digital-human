"""Derive ordinal supervision from existing, bound positive-repeat observations."""
from collections import Counter
from pathlib import Path

from ...config import sha256_file
from ...iteration.storage import read_json
from ..history_sources import digest, require
from ..ranker_labels import _saved


def load(inventory, stability, dataset, groups, observations):
    """No clients or draws: original negatives are explicitly the lowest grade."""
    inventory, stability = Path(inventory).resolve(), Path(stability).resolve()
    manifest = read_json(stability/'manifest.json')
    original = read_json(inventory/'labeling/manifest.json')
    require(manifest['kind'] == 'positive_label_stability' and manifest['source'] == dataset
            and Path(manifest['inventory']).resolve() == inventory
            and manifest['additional_draws'] == 2 and 'regenerate' in manifest['modes']
            and all(manifest[k] == original[k] for k in ('data', 'teacher', 'generator')),
            'Repeat labels are not bound to this dataset and three-draw policy')
    watched = {stability/'manifest.json': sha256_file(stability/'manifest.json')}
    cohort = {r['identity']: r['original_payload_sha256'] for r in manifest['positive_cohort']}
    require(len(cohort) == len(manifest['positive_cohort']), 'Duplicate repeat cohort')
    draws = []
    for draw in (1, 2):
        directory = stability/'regenerate'/f'draw-{draw}'
        value = read_json(directory/'manifest.json')
        require(value == dict(data=original['data'], teacher=original['teacher'],
            generator=original['generator'], stability_manifest_sha256=digest(manifest),
            mode='regenerate', draw=draw), 'Repeat draw binding differs')
        watched[directory/'manifest.json'] = sha256_file(directory/'manifest.json')
        draws.append((directory, value))
    result, seen = [None]*len(observations), set()
    for group in groups:
        require(len(group['indices']) == len(group['candidates']), 'Repeat candidate count differs')
        target = read_json(inventory/'targets'/f'{group["target_id"]}.json')
        for index, entry in zip(group['indices'], group['candidates']):
            require(0 <= index < len(result) and result[index] is None, 'Duplicate or invalid observation index')
            obs = observations[index]
            require(obs['target_id'] == group['target_id'] and obs['example_id'] == entry['example']['id']
                    and obs['split'] == group['split'], 'Repeat candidate alignment differs')
            identity = digest([obs['target_id'], obs['example_id']])
            labels, payloads = [], []
            if obs['z'] == 1:
                saved = _saved(inventory/'labeling/observations'/f'{identity}.json',
                    digest([digest(original), target['payload_sha256'], entry]))
                require(saved and saved['status'] == 'complete' and saved['z'] == 1
                        and cohort.get(identity) == saved['payload_sha256'], 'Original positive differs from repeat cohort')
                seen.add(identity)
                for directory, draw in draws:
                    path = directory/'observations'/f'{identity}.json'
                    value = _saved(path, digest([digest(draw), target['payload_sha256'], entry]))
                    require(value and value['status'] == 'complete', 'Missing completed repeat label')
                    labels.append(value['z'])
                    payloads.append(value['payload_sha256'])
                    watched[path] = sha256_file(path)
            result[index] = dict(obs, grade=obs['z']+sum(labels), trial_count=1+len(labels),
                                 repeat_labels=labels, repeat_payload_sha256=payloads)
    require(seen == set(cohort) and all(r is not None for r in result), 'Repeat cohort is incomplete or foreign')
    distributions = {split: dict(Counter(str(r['grade']) for r in result
        if split == 'all' or r['split'] == split)) for split in ('all', 'fit', 'validation')}
    binding = dict(stability_manifest_sha256=digest(manifest),
        repeat_evidence_sha256=digest({str(p): sha for p, sha in watched.items()}),
        policy='3 > 2 > 1 > 0; original zero is lowest without retesting; equal grades skipped',
        grade_counts=distributions, original_labels_unchanged=True)
    return result, binding, watched
