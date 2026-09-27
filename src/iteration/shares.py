"""Immutable shared knowledge versions, independently pinned by Gen and Judge.

Registration freezes content; it does not admit full-history Wiki into a model's
inputs. Runtime consumption needs a separate time-aware knowledge policy.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from ..config import ConfigError, sha256_file
from . import versions
from .storage import read_json, write_json


def root(instance: Path | None = None) -> Path:
    return (versions.PRIVATE if instance is None else instance) / 'share'


def _path(ref: str, *, instance: Path | None = None) -> Path:
    if not isinstance(ref, str) or not re.fullmatch(r's-\d{4,}', ref):
        raise ConfigError(f'非法 Share 版本: {ref}')
    directory = root(instance)
    path = directory / ref
    if directory.is_symlink() or path.is_symlink():
        raise ConfigError('Share 目录不能是符号链接')
    return path


def _files(directory: Path) -> dict[str, str]:
    files = {}
    if directory.is_symlink() or not directory.is_dir():
        raise ConfigError(f'Share 内容目录不存在或为符号链接: {directory}')
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ConfigError(f'Share 不接受符号链接: {path}')
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = sha256_file(path)
        elif not path.is_dir():
            raise ConfigError(f'Share 不接受特殊文件: {path}')
    if not files:
        raise ConfigError('Share 内容不能为空')
    return files


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def load(ref: str, *, instance: Path | None = None) -> dict:
    directory = _path(ref, instance=instance)
    manifest_path = directory / 'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ConfigError(f'Share 版本不存在: {ref}')
    manifest = read_json(manifest_path)
    if (manifest.get('schema') != 'shared_knowledge_v1' or manifest.get('id') != ref
            or manifest.get('runtime_usable') is not False
            or manifest.get('historical_input_status') != 'not_admitted'):
        raise ConfigError(f'Share 清单格式或准入状态非法: {ref}')
    if _files(directory / 'content') != manifest.get('files'):
        raise ConfigError(f'Share 内容指纹不符: {ref}')
    payload = {k: v for k, v in manifest.items() if k not in {'id', 'sha256'}}
    if _digest(payload) != manifest.get('sha256'):
        raise ConfigError(f'Share 清单指纹不符: {ref}')
    return {'id': ref, 'dir': directory, 'manifest': manifest}


@versions.transaction
def create(source: Path) -> str:
    """Freeze one reviewed material tree; identical content reuses its version."""
    source = Path(source)
    if root().resolve().is_relative_to(source.resolve()):
        raise ConfigError('Share 源目录不能包含 Share 版本库，避免递归复制')
    files = _files(source)
    payload = {'schema': 'shared_knowledge_v1', 'files': files,
               'runtime_usable': False, 'historical_input_status': 'not_admitted'}
    digest = _digest(payload)
    if root().is_symlink():
        raise ConfigError('Share 根目录不能是符号链接')
    existing = sorted(root().glob('s-*/manifest.json'))
    for path in existing:
        info = load(path.parent.name)
        if info['manifest']['sha256'] == digest:
            return info['id']
    ref = versions._next_id([p.name for p in root().iterdir()] if root().exists() else [], 's')
    destination = _path(ref)
    root().mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.share-', dir=root()))
    try:
        # Preserve links during the copy so a concurrently introduced link is
        # rejected by _files rather than followed outside the source tree.
        shutil.copytree(source, temporary / 'content', symlinks=True)
        if _files(temporary / 'content') != files:
            raise ConfigError('Share 源材料在冻结期间发生变化')
        write_json(temporary / 'manifest.json', {**payload, 'id': ref, 'sha256': digest})
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    versions._finalize_readonly(destination)
    return ref


def validate_binding(config: dict, *, instance: Path | None = None) -> dict | None:
    """Legacy models have no binding; new bindings must pin both ID and content."""
    if not {'share_ref', 'share_sha256'} & config.keys():
        return None
    if not config.get('share_ref') or not config.get('share_sha256'):
        raise ConfigError('Share 绑定必须同时包含 share_ref 和 share_sha256')
    info = load(config['share_ref'], instance=instance)
    if info['manifest']['sha256'] != config['share_sha256']:
        raise ConfigError(f"Share 绑定指纹不符: {config['share_ref']}")
    return info


def binding(ref: str) -> dict:
    info = load(ref)
    return {'share_ref': ref, 'share_sha256': info['manifest']['sha256']}


def require_runtime(config: dict, *, consumer: str | None = None) -> dict | None:
    """Do not silently ignore a binding or inject unfiltered historical facts."""
    info = validate_binding(config)
    if consumer == 'gen' and config.get('background') is not None:
        from ..generator.background import validate_policy
        validate_policy(config)
        if info is None:
            raise ConfigError('Gen background 必须绑定 Share 版本和内容指纹')
        return info
    if info is not None:
        raise ConfigError(f"Share {info['id']} 已冻结，但尚未完成按题时间裁剪和运行时接入；"
                          '当前版本仅可审阅，不能执行实验或在线推理')


def bind_model(kind: str, model_ref: str, share_ref: str) -> str:
    """Create a new model version; never change old configs or live pointers.

    Judge 版本限定平面文件：含子目录/符号链接时拒绝绑定，避免复制时丢失
    来源材料；历史 Judge 版本均为平面结构。
    """
    pin = binding(share_ref)
    if kind == 'gen':
        info = versions.load_generator(model_ref)
        if all(info['config'].get(k) == v for k, v in pin.items()):
            return model_ref
        return versions.create_generator_version(
            {**info['config'], **pin}, info['config']['data_version'], source_dir=info['dir'])
    if kind == 'judge':
        info = versions.judge_dir(model_ref)
        if all(info['config'].get(k) == v for k, v in pin.items()):
            return model_ref
        # Preserve learning/provenance files as well as declared model assets.
        paths = list(info['dir'].iterdir())
        if any(p.is_symlink() or not p.is_file() for p in paths):
            raise ConfigError('Judge 快照包含非平面文件，不能丢失来源材料后继续绑定')
        assets = {p.name: p.read_bytes() for p in paths if p.name not in {'config.json', 'meta.json'}}
        return versions.create_judge_version({**info['config'], **pin}, info['meta'], assets=assets)
    raise ConfigError(f'不支持绑定的模型类型: {kind}')


def inventory() -> list[dict]:
    """One registry view shows content versions and each consumer's own pin."""
    rows = []
    for path in sorted(root().glob('s-*/manifest.json')):
        info = load(path.parent.name)
        consumers = {'gen': [], 'judge': []}
        for kind, directory in [('gen', versions.GEN_ROOT), ('judge', versions.JUDGE_ROOT)]:
            for config_path in sorted(directory.glob('*/config.json')):
                config = read_json(config_path)
                if config.get('share_ref') == info['id']:
                    validate_binding(config)
                    consumers[kind].append(config_path.parent.name)
        rows.append({**info['manifest'], 'consumers': consumers})
    return rows
