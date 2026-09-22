"""Bind a completed ranker to reconstructed historical supervision for deployment."""
from pathlib import Path
import shutil

from ..config import ROOT, sha256_file
from ..iteration import versions
from ..iteration.storage import read_json, write_json
from .history import eligible
from .history_sources import digest, load, require, stamp

TRANSFORM = dict(semantic='local_crosses_v2', chat_id_pair=True,
                 identity_crosses='identity_crosses_v1')
SERVING_RUNTIME = tuple('src/generator/fewshot_ranker/' + name + '.py' for name in
    ('__init__', 'features', 'schema', 'extraction', 'crosses', 'identity_crosses', 'boosting'))
_verified = {}


def serving_runtime(manifest):
    """Pin code used for inference; completed training is not executed again."""
    bound = manifest['runtime']
    require(all(name in bound and sha256_file(ROOT / name) == bound[name]
                for name in SERVING_RUNTIME), 'Ranker feature/scoring runtime changed from training')
    return {name: bound[name] for name in SERVING_RUNTIME}


def _completed(directory, dependencies):
    """Read completion bindings without requiring historical code to be current."""
    manifest = read_json(directory / 'manifest.json')
    done = read_json(directory / 'completed.json')
    require(done['manifest_sha256'] == digest(manifest), 'Ranker completion manifest changed')
    dependencies.update({directory / 'manifest.json', directory / 'completed.json'})
    for name, checksum in done['artifacts'].items():
        path = directory / name
        require(path.resolve().is_relative_to(directory) and sha256_file(path) == checksum,
                'Completed ranker artifact changed')
        dependencies.add(path)
    return manifest


def reconstruct(model_dir, data_ref):
    """Validate once per dependency stamp set; no fitting or model requests."""
    from ..iteration.learning_guard import RunSeal
    from ..iteration.runtime import evidence_path
    from .ranker_report import load_completed
    from .ranker_samples import teacher_overlap
    model_dir = Path(model_dir).resolve()
    key = (str(model_dir), data_ref)
    if key in _verified:
        saved, result = _verified[key]
        require(all(stamp(p) == s for p, s in saved.items()), 'Ranker evidence changed after loading')
        return result
    # Capture dependencies before reading; the directory list excludes mutable logs/caches.
    parent = model_dir.parent
    parent_path = parent / 'manifest.json'
    early = RunSeal(files=[parent_path])
    parent_manifest = read_json(parent_path)
    source, reference = (Path(parent_manifest[k]).resolve() for k in ('source', 'reference'))
    source_path = source / 'manifest.json'
    source_seal = RunSeal(files=[source_path])
    inventory = Path(read_json(source_path)['inventory']).resolve()
    data = versions.data_version_dir(data_ref)
    paths = {parent_path, *data.glob('*.json'), data / 'messages.jsonl', data / 'fewshot_pool.jsonl'}
    for directory in (source, reference, parent):
        paths.update(p for p in directory.rglob('*.json') if 'logs' not in p.parts)
    paths.update(inventory / name for name in ('manifest.json', 'inventory.json', 'labeling/manifest.json'))
    paths.update((inventory / 'targets').glob('*.json'))
    paths.update((inventory / 'labeling/observations').glob('*.json'))
    paths.update(ROOT / name for name in parent_manifest['runtime'])
    labeling = read_json(inventory / 'labeling/manifest.json')
    require(labeling['data'] == data_ref, 'Ranker supervision data differs from original source')
    teacher = versions.judge_dir(labeling['teacher'])['dir']
    teacher_seal = RunSeal([teacher])
    teacher_record = read_json(teacher / 'learning.json')
    paths.update(p for p in teacher.rglob('*') if p.is_file())
    paths.update(evidence_path(p, h) for p, h in teacher_record['evidence_files'].items())
    seal = RunSeal(files=paths)
    dependencies = set(paths)
    child = _completed(model_dir, dependencies)
    parent_manifest = _completed(parent, dependencies)
    source_manifest = _completed(source, dependencies)
    ref = _completed(reference, dependencies)
    require(parent_manifest['kind'] == 'cached_fewshot_architecture_comparison' and
        child['parent_sha256'] == digest(parent_manifest) and child['method'] == 'xgb_chat_pair' and
        child['feature_transform'] == TRANSFORM, 'Unsupported ranker deployment recipe')
    require(parent_manifest['source_manifest_sha256'] == digest(source_manifest) and
        parent_manifest['source_completed_sha256'] == sha256_file(source / 'completed.json') and
        parent_manifest['reference_completed_sha256'] == sha256_file(reference / 'completed.json') and
        ref['source_manifest_sha256'] == digest(source_manifest) and
        ref['source_completed_sha256'] == sha256_file(source / 'completed.json') and
        parent_manifest['dataset'] == ref['dataset'] == source_manifest['dataset'], 'Ranker lineage differs')
    serving = serving_runtime(parent_manifest)
    model = read_json(model_dir / 'model.json')
    require(model['feature_transform'] == TRANSFORM, 'Model transform differs from training')
    _, groups, _, summary, _, _ = load_completed(source)
    history = load(data)
    canonical = {c['case_id']: c for role in ('gen_learning', 'gen_optimization') for c in history.role(role)}
    end = 0
    for group in groups:
        target = group['target']
        require(canonical.get(group['target_id']) == target, 'Ranker training target differs from historical role')
        end = max(end, target['source_span']['end_timestamp'])
        for entry in group['candidates']:
            example = entry['example']
            history.validate(example, example=True)
            require(eligible(example, target), 'Ranker example includes current answer or future information')
            end = max(end, example['source_span']['end_timestamp'])
    overlap, _ = teacher_overlap(history, [g['target'] for g in groups], teacher)
    require(all(not row[k] for row in overlap['targets'] for k in
        ('same_teacher_training_case', 'answer_in_teacher_training', 'answer_in_teacher_references')),
        'Ranker targets overlap teacher supervision')
    # The user approved the existing teacher for offline nonoverlapping supervision.
    # Its information cutoff still applies to the later formal generator evaluation.
    end = max(end, overlap['summary']['teacher_information_end'])
    require(end < min(c['input_cutoff']['timestamp'] for c in history.role('development')),
            'Ranker learning includes development answers or future information')
    config = read_json(source / 'transport.json')
    dependencies.add(source / 'transport.json')
    result = dict(schema=1, recipe='completed_pairwise_xgb_v1', model_directory=str(model_dir),
        data_ref=data_ref, feature_config=config, feature_identity=source_manifest['features']['client'],
        feature_transform=TRANSFORM, serving_runtime=serving, information_end=end, samples=summary,
        model_sha256=sha256_file(model_dir / 'model.json'),
        evidence_files={str(p): sha256_file(p) for p in sorted(dependencies)})
    early.check()
    source_seal.check()
    teacher_seal.check()
    seal.check()
    _verified[key] = ({p: stamp(p) for p in dependencies}, result)
    return result


def deploy(model_dir, base_dir, data_ref, output):
    from ..iteration.learning_guard import GENERATOR_RECIPES, require_materials, MaterialSeal
    base_dir, output = Path(base_dir), Path(output)
    base_seal = MaterialSeal(base_dir)
    require_materials(data_ref, [base_dir])
    inherited = read_json(base_dir / 'learning.json')
    require(inherited['asset_files'] in GENERATOR_RECIPES.values(), 'Expected approved mechanical generator')
    proof = reconstruct(model_dir, data_ref)
    for name in inherited['asset_files']:
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base_dir / name, dest)
    (output / 'ranker').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(model_dir) / 'model.json', output / 'ranker/model.json')
    write_json(output / 'ranker/provenance.json', proof)
    assets = dict(inherited['asset_files'], **{name: sha256_file(output / name)
                  for name in ('ranker/model.json', 'ranker/provenance.json')})
    write_json(output / 'learning.json', dict(schema=2, verified=True,
        information_end=proof['information_end'], asset_files=assets,
        evidence_files={**inherited['evidence_files'], **proof['evidence_files']}))
    base_seal.check()
    return output


def source_evidence_view(expected, proof, names):
    """Match relocated executable paths by exact bytes; keep data paths pinned."""
    from ..iteration.runtime import evidence_path
    evidence = dict(expected['evidence_files'])
    original = proof['evidence_files']
    for name in names:
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.py',
                'Invalid ranker runtime evidence path')
        actual = str(ROOT / name)
        if actual in original:
            continue
        checksum = evidence.pop(actual)
        aliases = [key for key, value in original.items()
                   if key.endswith('/' + name) and value == checksum]
        require(len(aliases) == 1, 'Ranker runtime evidence differs from deployment')
        evidence_path(aliases[0], checksum)
        require(aliases[0] not in evidence or evidence[aliases[0]] == checksum,
                'Conflicting ranker runtime evidence')
        evidence[aliases[0]] = checksum
    return {**expected, 'evidence_files': evidence}


def verify(data_ref, directory, record):
    from ..iteration.learning_guard import GENERATOR_RECIPES
    proof = read_json(directory / 'ranker/provenance.json')
    expected = reconstruct(proof['model_directory'], proof['data_ref'])
    manifest = read_json(Path(proof['model_directory']).parent / 'manifest.json')
    expected = source_evidence_view(expected, proof, manifest['runtime'])
    require(proof == expected and sha256_file(directory / 'ranker/model.json') == proof['model_sha256'],
            'Deployed ranker differs from completed source')
    prompt_assets = {k: v for k, v in record['asset_files'].items() if not k.startswith('ranker/')}
    require(prompt_assets in GENERATOR_RECIPES.values() and
        record['information_end'] == proof['information_end'] and
        all(record['evidence_files'].get(k) == v for k, v in proof['evidence_files'].items()),
        'Deployed ranker learning evidence is incomplete')
    if proof['data_ref'] != data_ref:
        from ..iteration.material_compatibility import ranker_sources, require_current
        require_current(data_ref, directory, ranker_sources)
