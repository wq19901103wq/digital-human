"""Content-addressed executable snapshots and separate historical code evidence."""
from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ..cache import digest
from ..config import ConfigError, ROOT, settings_path, sha256_file
from . import versions
from .storage import file_lock, write_json


def environment():
    packages = ('numpy', 'scipy', 'scikit-learn', 'openai', 'python-dotenv', 'PyYAML')
    return {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
            'python_sha256': sha256_file(Path(sys.executable).resolve()),
            'packages': {p: importlib.metadata.version(p) for p in packages}}


def files(root):
    paths = []
    for folder in ('src', 'scripts', 'config', 'prompts'):
        for p in (root / folder).rglob('*'):
            if p.is_symlink():
                raise ConfigError('runtime source must not contain symlinks')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                paths.append(p)
    return paths


def freeze(directory, *, root=ROOT):
    """Called at creation, before any requests; never re-sign an old run."""
    path = Path(directory) / 'runtime.json'
    if path.exists():
        return verify(path)
    root = Path(root)
    value = {'schema': 1, 'files': {str(p.relative_to(root)): sha256_file(p) for p in files(root)},
             'environment': environment(), 'settings_sha256': sha256_file(settings_path())}
    ref = digest(value)
    store = versions.PRIVATE / 'runtimes'
    with file_lock(store / '.lock'):
        target = store / ref
        if not target.exists():
            temporary = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=store))
            try:
                for name, expected in value['files'].items():
                    dest = temporary / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(root / name, dest)
                    if sha256_file(dest) != expected:
                        raise ConfigError('source changed while freezing runtime')
                shutil.copy2(settings_path(), temporary / 'settings.snapshot.yaml')
                if sha256_file(temporary / 'settings.snapshot.yaml') != value['settings_sha256']:
                    raise ConfigError('settings changed while freezing')
                write_json(temporary / 'manifest.json', value)
                os.rename(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        descriptor = {'schema': 1, 'ref': ref, 'manifest_sha256': sha256_file(target / 'manifest.json')}
        write_json(path, descriptor)
    return verify(path)


def verify(path, *, check_environment=True):
    """Verify frozen bytes; execution additionally requires the pinned environment.

    Historical receipts may be inspected on a new host without executing their code.
    """
    record = json.loads(Path(path).read_text())
    ref = record['ref']
    if len(ref) != 64 or any(c not in '0123456789abcdef' for c in ref):
        raise ConfigError('invalid runtime identifier')
    root = versions.PRIVATE / 'runtimes' / ref
    manifest = root / 'manifest.json'
    if sha256_file(manifest) != record['manifest_sha256']:
        raise ConfigError('runtime manifest changed')
    value = json.loads(manifest.read_text())
    if digest(value) != ref:
        raise ConfigError('runtime content identifier changed')
    if check_environment and environment() != value['environment']:
        raise ConfigError('runtime interpreter or dependency versions changed; use the pinned environment')
    if sha256_file(root / 'settings.snapshot.yaml') != value['settings_sha256']:
        raise ConfigError('frozen instance settings changed')
    for name, expected in value['files'].items():
        p = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or sha256_file(p) != expected:
            raise ConfigError('runtime source changed')
    return root


def verify_historical_implementation(directory, implementation):
    """Resolve a completed receipt against its original, authenticated executor."""
    root = verify(Path(directory) / 'runtime.json', check_environment=False)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not implementation:
        raise ConfigError('historical receipt has no implementation binding')
    for name, expected in implementation.items():
        if not any(manifest['files'].get(prefix + name) == expected
                   for prefix in ('src/', 'src/digital_human/')):
            raise ConfigError('historical implementation does not match frozen runtime')


def archive_inputs(inputs):
    """Archive only exact, still-verifiable historical inputs; no new attestation."""
    root = versions.PRIVATE / 'evidence_blobs'
    with file_lock(root / '.lock'):
        for name, expected in inputs.items():
            p = Path(name)
            if not p.is_file() or sha256_file(p) != expected:
                raise ConfigError('cannot archive a missing or changed historical input')
            # Only code is portable historical evidence. Data must remain at its bound path.
            if p.suffix != '.py':
                continue
            target = root / expected
            if not target.exists():
                shutil.copy2(p, target)
            if sha256_file(target) != expected:
                raise ConfigError('historical evidence blob changed')


def evidence_path(name, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ConfigError('invalid historical evidence digest')
    p = Path(name)
    if p.is_file() and sha256_file(p) == expected:
        return p
    archived = versions.PRIVATE / 'evidence_blobs' / expected
    if p.suffix == '.py' and archived.is_file() and sha256_file(archived) == expected:
        return archived
    if p.suffix == '.py':
        # Some completed trainers predate archive_inputs. Their exact code is
        # already retained by experiment snapshots; reading it does not re-sign
        # the model or allow changed data to move to a different path.
        for manifest in sorted((versions.PRIVATE / 'runtimes').glob('*/manifest.json')):
            value = json.loads(manifest.read_text())
            if digest(value) != manifest.parent.name:
                continue
            for relative, checksum in value.get('files', {}).items():
                path = Path(relative)
                if (checksum != expected or path.is_absolute() or '..' in path.parts
                        or not str(p).endswith('/' + relative) or path.suffix != '.py'):
                    continue
                candidate = manifest.parent / path
                if (candidate.resolve().is_relative_to(manifest.parent.resolve())
                        and not candidate.is_symlink() and candidate.is_file()
                        and sha256_file(candidate) == expected):
                    return candidate
    raise ConfigError(f'frozen input changed or missing: {p.name}')


def verify_inputs(inputs):
    return [evidence_path(name, expected) for name, expected in inputs.items()]


def require_current(directory):
    path = Path(directory) / 'runtime.json'
    if not path.exists():
        raise ConfigError('legacy run has no frozen runtime; resume with its original archived executor')
    snapshot = verify(path)
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    expected = manifest['files']
    if sha256_file(settings_path()) != manifest['settings_sha256']:
        raise ConfigError('settings changed; use the frozen runtime')
    if {str(p.relative_to(ROOT)): sha256_file(p) for p in files(ROOT)} != expected:
        raise ConfigError('executor changed; use the scheduler to resume the frozen runtime')
    return files(ROOT)
