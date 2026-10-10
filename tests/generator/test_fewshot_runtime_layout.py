"""Frozen-byte relocation, canonical scoring imports and evidence boundaries."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


RECONSTRUCTION_HELPERS = (
    'report', 'dataset', 'labels', 'samples', 'runtime_sources',
    'action_comparison', 'concern_comparison', 'architecture_comparison',
    'identity_comparison', 'comparison', 'discovery', 'neural_sweep',
    'neural_models', 'optimization', 'logistic',
)


def check_import_order(first):
    importlib.import_module(first)
    from src.generator import fewshot
    from src.generator.fewshot import boosting
    from src.generator.fewshot.training import logistic, optimization
    assert boosting.logistic is logistic
    assert boosting.training is fewshot.training
    assert boosting.training.pairs is optimization.pairs
    assert boosting.training.inner_split is optimization.inner_split
    assert 'src.generator.fewshot_ranker' not in sys.modules
    assert 'src.generator.fewshot._frozen_init' not in sys.modules
    assert hasattr(fewshot, 'open_session')


@pytest.mark.parametrize('first', (
    'src.generator.fewshot', 'src.generator.fewshot.boosting',
    'src.generator.fewshot.training.optimization',
    'src.generator.fewshot.training.logistic',
))
def test_cold_import_order_preserves_scoring_and_training_api(first):
    result = subprocess.run([sys.executable, '-m', 'tests.generator.test_fewshot_runtime_layout', first],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def scoring_runtime(tmp_path, monkeypatch):
    from src.generator.fewshot import provenance
    from src.iteration import runtime, versions
    root = tmp_path / 'repo'
    monkeypatch.setattr(provenance, 'ROOT', root)
    monkeypatch.setattr(runtime, 'ROOT', root)
    monkeypatch.setattr(versions, 'PRIVATE', tmp_path / 'private')
    return root


def runtime_manifest(root, legacy):
    from src.generator.fewshot.runtime_layout import FROZEN_NAMES, LEGACY_PREFIX, PREFIX, RELOCATED
    from src.infrastructure.config import sha256_file
    bound = {}
    for name in FROZEN_NAMES:
        logical = (LEGACY_PREFIX if legacy else PREFIX) + name + '.py'
        path = root / RELOCATED.get(logical, logical)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen ' + name)
        bound[logical] = sha256_file(path)
    return {'runtime': bound}


@pytest.mark.parametrize('legacy', (True, False))
def test_serving_accepts_one_complete_generation_and_preserves_logical_keys(scoring_runtime, legacy):
    from src.generator.fewshot import action_crosses, concern, provenance
    from src.generator.fewshot.runtime_layout import FROZEN_NAMES, LEGACY_PREFIX, PREFIX
    manifest = runtime_manifest(scoring_runtime, legacy)
    manifest['feature_transform'] = {**action_crosses.TRANSFORM, **concern.TRANSFORM}
    bound = provenance.serving_runtime(manifest)
    prefix = LEGACY_PREFIX if legacy else PREFIX
    assert bound == {prefix + name + '.py': manifest['runtime'][prefix + name + '.py']
                     for name in FROZEN_NAMES}
    logical, paths = provenance.historical_runtime(manifest)
    assert logical == {str(scoring_runtime / name): checksum
                       for name, checksum in manifest['runtime'].items()}
    assert all(path.is_file() for path in paths)
    assert not list((scoring_runtime / LEGACY_PREFIX).glob('*.py'))


@pytest.mark.parametrize('legacy', (True, False))
@pytest.mark.parametrize('change', ('missing', 'tampered', 'symlink'))
def test_serving_rejects_missing_changed_or_symlinked_core(scoring_runtime, legacy, change):
    from src.generator.fewshot import provenance
    from src.generator.fewshot.runtime_layout import PREFIX
    from src.infrastructure.config import ConfigError
    manifest = runtime_manifest(scoring_runtime, legacy)
    path = scoring_runtime / PREFIX / 'schema.py'
    original = path.read_bytes()
    path.unlink()
    if change == 'tampered':
        path.write_text('changed')
    elif change == 'symlink':
        target = scoring_runtime / 'original_schema.py'
        target.write_bytes(original)
        path.symlink_to(target)
    with pytest.raises(ConfigError, match='feature/scoring runtime changed'):
        provenance.serving_runtime(manifest)


@pytest.mark.parametrize('foreign', ('schema', 'action_crosses', 'supplemental'))
@pytest.mark.parametrize('legacy', (True, False))
def test_serving_rejects_mixed_generations_even_for_optional_features(scoring_runtime, legacy, foreign):
    from src.generator.fewshot import provenance
    from src.generator.fewshot.runtime_layout import LEGACY_PREFIX, PREFIX
    from src.infrastructure.config import ConfigError
    manifest = runtime_manifest(scoring_runtime, legacy)
    manifest['runtime'][(PREFIX if legacy else LEGACY_PREFIX) + foreign + '.py'] = 'unused'
    with pytest.raises(ConfigError, match='missing or mixed'):
        provenance.serving_runtime(manifest)


def test_modern_runtime_pins_real_initializer_not_historical_evidence(scoring_runtime):
    from src.generator.fewshot import provenance
    from src.generator.fewshot.runtime_layout import PREFIX
    from src.infrastructure.config import ConfigError
    manifest = runtime_manifest(scoring_runtime, False)
    initializer = manifest['runtime'].pop(PREFIX + '__init__.py')
    manifest['runtime'][PREFIX + '_frozen_init.py'] = initializer
    with pytest.raises(ConfigError, match='feature/scoring runtime changed'):
        provenance.serving_runtime(manifest)


def test_exact_relocation_rejects_traversal_and_does_not_map_training(scoring_runtime):
    from src.generator.fewshot.runtime_layout import LEGACY_PREFIX, PREFIX, relocated_file
    assert relocated_file(scoring_runtime, LEGACY_PREFIX + 'features.py') == scoring_runtime / PREFIX / 'features.py'
    assert relocated_file(scoring_runtime, '/archive/' + LEGACY_PREFIX + 'features.py') == scoring_runtime / PREFIX / 'features.py'
    for name in (LEGACY_PREFIX + 'training.py', LEGACY_PREFIX + 'logistic.py',
                 'other/features.py', '../' + LEGACY_PREFIX + 'features.py'):
        assert relocated_file(scoring_runtime, name) is None


def test_historical_training_uses_exact_archived_bytes_not_new_optimization(scoring_runtime):
    from src.generator.fewshot import provenance
    from src.generator.fewshot.runtime_layout import LEGACY_PREFIX
    from src.infrastructure.config import ConfigError, sha256_file
    from src.iteration import runtime, versions
    path = scoring_runtime / LEGACY_PREFIX / 'training.py'
    path.parent.mkdir(parents=True)
    path.write_text('original training')
    checksum = sha256_file(path)
    runtime.archive_inputs({str(path): checksum})
    path.unlink()
    manifest = {'runtime': {LEGACY_PREFIX + 'training.py': checksum}}
    logical, actual = provenance.historical_runtime(manifest)
    assert logical == {str(path): checksum}
    blob = versions.PRIVATE / 'evidence_blobs' / checksum
    assert actual == {blob}
    blob.write_text('changed')
    with pytest.raises(ConfigError, match='changed or missing'):
        provenance.historical_runtime(manifest)


@pytest.mark.parametrize('helper', ('../__init__', '__init__', *RECONSTRUCTION_HELPERS))
def test_frozen_bytes_and_actual_bridges_are_covered_by_execution_seal(tmp_path, helper):
    from src.generator.fewshot.runtime_layout import execution_files, PREFIX, RELOCATED
    from src.infrastructure.config import ConfigError
    from src.iteration.materials.seals import RunSeal
    from tests.frozen_boundaries import frozen_violations
    assert frozen_violations() == []
    paths = execution_files(tmp_path)
    assert {tmp_path / name for name in RELOCATED.values()} <= paths
    bridges = {tmp_path / PREFIX / name for name in (
        '__init__.py', 'runtime_layout.py', 'provenance.py', 'learned_selection.py',
        'training/__init__.py', *(f'training/{name}.py' for name in RECONSTRUCTION_HELPERS))}
    assert bridges <= paths
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('sealed')
    seal = RunSeal(files=paths)
    (tmp_path / PREFIX / 'training' / (helper + '.py')).write_text('changed')
    with pytest.raises(ConfigError):
        seal.check()


@pytest.mark.parametrize('helper', RECONSTRUCTION_HELPERS)
def test_changed_reconstruction_helper_blocks_verified_ranker_reuse(tmp_path, monkeypatch, helper):
    from src.data.history.sources import digest, stamp
    from src.generator.fewshot import provenance
    from src.generator.fewshot.runtime_layout import execution_files, PREFIX
    from src.infrastructure.config import ConfigError
    paths = execution_files(tmp_path)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('sealed')
    model = tmp_path / 'model'
    key = (str(model.resolve()), 'data', digest(None))
    result = {'verified': True}
    monkeypatch.setattr(provenance, '_verified', {key: ({path: stamp(path) for path in paths}, result)})
    assert provenance.reconstruct(model, 'data') is result
    (tmp_path / PREFIX / 'training' / (helper + '.py')).write_text('changed')
    with pytest.raises(ConfigError, match='Ranker evidence changed after loading'):
        provenance.reconstruct(model, 'data')


if __name__ == '__main__':
    check_import_order(sys.argv[1])
