"""Install the already selected embedding head with reconstructible train-only evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import cache
from ..config import sha256_file, valid_name
from ..judge import embedding as net
from . import versions
from .training_evidence import read, require


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def selected_model(directory, source, arm):
    """Reconstruct only the selected full fit, never rerun search or use evaluation labels."""
    spec, selected = read(directory / 'spec.json'), read(directory / 'selection.json')
    require(spec['kind'] == 'embedding_dnn_tuning' and spec['dataset'] == 'training'
            and spec['embedding_dim'] == 8 and spec['numeric_bypass'] is False
            and Path(spec['source']).resolve() == (source / arm).resolve()
            and spec['arm'] == arm and spec['torch_version'] == net.torch.__version__
            and spec['numpy_version'] == np.__version__, 'Embedding training conditions differ')
    for path, expected in spec['inputs'].items():
        require(sha256_file(Path(path)) == expected, 'Embedding training input changed: ' + path)
    key = valid_name(selected['primary'])
    recipe_id, seed = key.rsplit('-s', 1)
    seed = int(seed)
    require(recipe_id == selected['configurations'][0] and seed == spec['selection']['primary_seed'],
            'Embedding candidate is not the frozen primary')
    rows, entries = read(source / arm / 'training.json')['rows'], read(source / arm / 'features.json')['entries']
    require(len(rows) == spec['training_total'] and all(r['split'] == 'train' for r in rows),
            'Embedding fit must use the complete original training partition')
    raw = net.raw_pairs(rows, entries)
    encoder = net.fit_encoder(raw)
    recipe, epochs = spec['recipes'][recipe_id], selected['epochs'][recipe_id]
    result_path = directory / 'refit' / (key + '.json')
    result = read(result_path)
    require(result['model_sha256'] == cache.digest(result['model'])
            and result['job_sha256'] == cache.digest(dict(key=key, encoder=encoder, recipe=recipe,
                                                        seed=seed, epochs=epochs)),
            'Embedding checkpoint identity differs')
    fitted, metrics = net.fit(encoder, recipe, net.encode(encoder, raw),
        np.asarray([r['human_option'] == 'A' for r in rows]), np.arange(len(rows)), [], seed, epochs)
    require(result['model'] == net.document(fitted)
            and all(result[k] == value for k, value in metrics.items()),
            'Embedding weights cannot be reconstructed from the original training features')
    paths = [directory / 'spec.json', directory / 'selection.json', result_path,
             *map(Path, spec['inputs'])]
    return result['model'], paths, key


def _assemble(original, study, model, proofs, key):
    cfg, record = read(original / 'config.json'), read(original / 'learning.json')
    assets = {n: (original / n).read_bytes() for n in ['prompt.md', *cfg['assets']]}
    assets.update({'embedding.json': _bytes(model), 'embedding_source.json': _bytes(
        dict(schema=1, study=study.name, primary=key))})
    additions = {n: hashlib.sha256(assets[n]).hexdigest()
                 for n in ('embedding.json', 'embedding_source.json')}
    learning = {**record, 'asset_files': {**record['asset_files'], **additions},
        'evidence_files': {**record['evidence_files'], **{str(p.resolve()): sha256_file(p) for p in proofs}}}
    assets['learning.json'] = _bytes(learning)
    config = {**cfg, 'decision_policy': 'embedding_only', 'embedding_file': 'embedding.json',
        'assets': {**cfg['assets'], **additions,
                   'learning.json': hashlib.sha256(assets['learning.json']).hexdigest()}}
    return config, assets


def verify(directory, record, original, source, source_spec, arm):
    descriptor = read(directory / 'embedding_source.json')
    study = versions.PRIVATE / 'judge_training' / valid_name(descriptor['study'])
    require(read(study / 'spec.json')['data_ref'] == source_spec['data_ref'],
            'Embedding data version differs from reconstructed source')
    model, paths, key = selected_model(study, source, arm)
    require(descriptor == dict(schema=1, study=study.name, primary=key), 'Embedding primary differs')
    config, assets = _assemble(original, study, model, paths, key)
    require(read(directory / 'config.json') == config and record == json.loads(assets['learning.json'])
            and all((directory / n).read_bytes() == value for n, value in assets.items()),
            'Embedding learning evidence or inherited assets differ')


def create(study_name, change):
    """Freeze a version of the existing primary; normal evaluation gates remain mandatory."""
    from .learning_guard import RunSeal, require_materials
    study = versions.PRIVATE / 'judge_training' / valid_name(study_name)
    spec = read(study / 'spec.json')
    source, arm = Path(spec['source']).parent, spec['arm']
    original = versions.judge_dir(read(source / arm / 'candidate.json')['judge_ref'])['dir']
    seal = RunSeal([study, source, original])
    require_materials(spec['data_ref'], [original])
    model, proofs, key = selected_model(study, source, arm)
    config, assets = _assemble(original, study, model, proofs, key)
    seal.check()
    return versions.create_judge_version(config, {'change': change, 'embedding_study': study.name,
                                                  'primary': key}, assets=assets)
