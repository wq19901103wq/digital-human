"""materials.validation receipt v6 语义测试（S11）。

与旧适配器 v5（tests/test_disk_history.py / test_material_reuse.py 的对应
receipt 语义）保持一致的 minimal 对应集，夹具模式复用 test_disk_history 的
“临时数据目录 + 缓存根在 .cache 下”结构：

- 暖索引/暖收据复用：内容未变时二次进入不重新全量核验（跨 context 秒级复用）；
- 收据损坏/符号链接 → fail-closed 拒绝（材料核验缓存）；
- 来源/材料/数据清单变化 → 重新全量核验（旧适配器对来源变化直接拒绝；
  v6 对静态依赖变化走重新核验语义，与 test_changed_persistent_inventory_requires_validation 一致）；
- 全量核验失败不发布收据。
"""
import json
from types import SimpleNamespace

import pytest

from src.infrastructure.config import ConfigError
from src.iteration import versions
from src.iteration.materials import validation


@pytest.fixture
def materials(tmp_path, monkeypatch):
    data_root = tmp_path / 'data'
    monkeypatch.setattr(versions, 'DATA_ROOT', data_root)
    for name in ('current', 'original'):
        directory = data_root / name
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text('{}')
    directory = tmp_path / 'generator'
    directory.mkdir()
    (directory / 'prompt.md').write_text('mechanical')
    (directory / 'learning.json').write_text(json.dumps({'asset_files': {'prompt.md': 'fixed'},
                                                         'evidence_files': {}}))
    calls = []

    def full(data_ref, directories, role='development'):
        calls.append((data_ref, tuple(directories), role))
        return {'assets': [{'verified': True}]}

    # v6 路径：patch 一律指向 materials.validation 真实模块属性（设计 §4）。
    monkeypatch.setattr(validation, 'require_materials', full)
    monkeypatch.setattr(validation, 'GENERATOR_RECIPES', {'mechanical': {'prompt.md': 'fixed'}})
    return SimpleNamespace(directory=directory, data=data_root, calls=calls)


def receipt_file(tmp_path):
    root = tmp_path / '.cache' / 'material-validation'
    files = list(root.glob('*.json'))
    assert len(files) == 1
    return files[0]


def test_receipt_warm_reuse_across_contexts(materials):
    with validation.verified_materials():
        first = validation.require_materials('current', [materials.directory])
        expected = json.loads(json.dumps(first))
        first['assets'].clear()
        # 同一 context 内复用：返回深拷贝审计。
        assert validation.require_materials('current', [materials.directory]) == expected
    with validation.verified_materials():
        # 跨 context 复用收据：不再全量核验。
        assert validation.require_materials('current', [materials.directory]) == expected
    assert materials.calls == [('current', (materials.directory.absolute(),), 'development')]


@pytest.mark.parametrize('invalid', ['corrupt', 'symlink'])
def test_invalid_receipt_fails_closed(materials, tmp_path, invalid):
    with validation.verified_materials():
        validation.require_materials('current', [materials.directory])
    path = receipt_file(tmp_path)
    if invalid == 'corrupt':
        path.write_text('{}')
    else:
        target = path.with_suffix('.saved')
        path.rename(target)
        path.symlink_to(target)
    with validation.verified_materials(), pytest.raises(ConfigError, match='材料核验缓存'):
        validation.require_materials('current', [materials.directory])
    assert len(materials.calls) == 1


@pytest.mark.parametrize('changed', ['material', 'data'])
def test_changed_dependencies_require_revalidation(materials, tmp_path, changed):
    with validation.verified_materials():
        validation.require_materials('current', [materials.directory])
    paths = {'material': materials.directory / 'prompt.md',
             'data': materials.data / 'current' / 'manifest.json'}
    paths[changed].write_text('changed')
    with validation.verified_materials():
        # 变化依赖 → 收据不匹配，重新全量核验并发布新收据（不 fail-closed）。
        validation.require_materials('current', [materials.directory])
    assert len(materials.calls) == 2
    receipt_file(tmp_path)  # 新收据已发布


def test_failed_full_validation_does_not_publish_receipt(materials, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ConfigError('invalid provenance')

    monkeypatch.setattr(validation, 'require_materials', fail)
    with validation.verified_materials(), pytest.raises(ConfigError, match='invalid provenance'):
        validation.require_materials('current', [materials.directory])
    assert not list((tmp_path / '.cache' / 'material-validation').glob('*.json'))


def test_receipt_key_is_scoped_by_data_role_and_directories(materials, tmp_path):
    with validation.verified_materials():
        validation.require_materials('current', [materials.directory])
        validation.require_materials('current', [materials.directory], 'fixed_test')
        validation.require_materials('original', [materials.directory])
    # 三个 (data_ref, role/directories) 组合各自独立收据。
    assert len(list((tmp_path / '.cache' / 'material-validation').glob('*.json'))) == 3
    assert len(materials.calls) == 3


def test_receipt_with_removed_ranker_paths_requires_full_validation(tmp_path, monkeypatch):
    import hashlib
    from src.generator.fewshot import provenance
    from src.generator.fewshot.runtime_layout import LEGACY_PREFIX
    from src.iteration.materials.seals import RunSeal
    key = {'mode': validation.MODE}
    ranker_key = ('model', 'data', 'digest')
    seal = RunSeal()
    missing = str(tmp_path / LEGACY_PREFIX / 'features.py')
    payload = dict(key=key, seal=validation.seal_record(seal), environment={},
        cache_rows=[], rankers=[dict(key=ranker_key, stamps={missing: [1, 2, 3]})], audit={})
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(dict(payload=payload,
        sha256=hashlib.sha256(validation.canonical(payload).encode()).hexdigest())))
    monkeypatch.setattr(provenance, '_verified', {})
    monkeypatch.setattr(validation, 'check_stamps', lambda stamps: pytest.fail('do not drop missing stamps and reuse'))
    assert validation.read_receipt(path, key, seal, [ranker_key], {}, lambda rows: None) is None
    assert provenance._verified == {}


@pytest.mark.parametrize('helper', (
    'report', 'dataset', 'labels', 'samples', 'runtime_sources',
    'action_comparison', 'concern_comparison', 'architecture_comparison',
    'identity_comparison', 'comparison', 'discovery', 'neural_sweep',
    'neural_models', 'optimization', 'logistic',
))
def test_changed_reconstruction_helper_blocks_persisted_receipt_reuse(tmp_path, monkeypatch, helper):
    from src.data.history.sources import stamp
    from src.generator.fewshot import provenance
    from src.generator.fewshot.runtime_layout import execution_files, PREFIX
    from src.iteration.materials.seals import RunSeal
    paths = execution_files(tmp_path)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('sealed')
    ranker_key = ('model', 'data', 'digest')
    record = ({path: stamp(path) for path in paths}, {'verified': True})
    monkeypatch.setattr(provenance, '_verified', {ranker_key: record})
    seal = RunSeal()
    key = {'mode': validation.MODE}
    receipt = tmp_path / 'receipt.json'
    validation.write_receipt(receipt, key, seal, [ranker_key], {}, {}, {})
    provenance._verified.clear()
    assert validation.read_receipt(receipt, key, seal, [ranker_key], {}, lambda rows: None) is not None
    assert provenance._verified == {ranker_key: record}
    provenance._verified.clear()
    (tmp_path / PREFIX / 'training' / (helper + '.py')).write_text('changed')
    with pytest.raises(ConfigError, match='Ranker 依赖变化'):
        validation.read_receipt(receipt, key, seal, [ranker_key], {}, lambda rows: None)
    assert provenance._verified == {}
