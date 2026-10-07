"""Standard-library authentication before importing executable snapshot code."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def source_only_imports():
    """Keep historical bytecode intact without using it to execute source modules."""
    previous_prefix, previous_write = sys.pycache_prefix, sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(prefix='digital-human-source-only-') as directory:
        sys.pycache_prefix, sys.dont_write_bytecode = directory, True
        try:
            yield
        finally:
            sys.pycache_prefix, sys.dont_write_bytecode = previous_prefix, previous_write


def authenticate(root):
    root = Path(root)
    if root.is_symlink() or (root / 'manifest.json').is_symlink():
        raise ValueError('runtime source must not contain symlinks')
    value = json.loads((root / 'manifest.json').read_text())
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != root.name:
        raise ValueError('runtime content identifier changed')
    expected = value['files']
    for name, checksum in expected.items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('invalid runtime source path')
        if '__pycache__' in relative.parts or relative.suffix in {'.pyc', '.pyo'}:
            raise ValueError('runtime manifest must bind source, not bytecode')
        if any(part.is_symlink() for part in (path, *path.parents)
               if part == root or root in part.parents):
            raise ValueError('runtime source must not contain symlinks')
        if file_digest(path) != checksum:
            raise ValueError('runtime source changed')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('runtime source must not contain symlinks')
        if path.is_file():
            relative = path.relative_to(root)
            if relative == Path('manifest.json'):
                continue
            if (relative == Path('settings.snapshot.yaml')
                    and value.get('settings_sha256') is not None):
                continue
            if '__pycache__' in relative.parts and path.suffix == '.pyc':
                continue
            actual.add(str(relative))
    if actual != set(expected):
        raise ValueError('runtime source inventory changed')
    settings = root / 'settings.snapshot.yaml'
    if value.get('settings_sha256') is not None and (
            settings.is_symlink() or file_digest(settings) != value['settings_sha256']):
        raise ValueError('frozen instance settings changed')
    return value
