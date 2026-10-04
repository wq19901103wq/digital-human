"""过渡兼容：清单扩展前的旧形态指纹按 runtime 归档字节核验续跑，新形态照旧。

旧断点/旧包/旧 inventory 不得因实现清单扩展而要求重跑已完成题目：
归档字节才是旧运行的身份，活树实现演进不阻断续跑。
"""
import copy
import json
from pathlib import Path

import pytest

from src import cache
from src.config import ConfigError, sha256_file
from src.generator import ranker_samples
from src.generator.fewshot import RecallSession
from src.history.sources import digest
from src.iteration import data_guard, versions
from src.iteration.storage import read_json, write_json
from leakage_support import historical
from test_ranker_samples import teacher_at


def _fake_runtime_archive(monkeypatch, tmp_path, record_dir, files: dict[str, str]):
    """建假 runtime 归档（manifest 摘要 == 目录名）并在 record_dir 写 runtime.json。"""
    store = tmp_path / 'instance' / 'runtimes'
    staging = store / '.staging-archive'
    for name, content in files.items():
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    value = {'files': {name: sha256_file(staging / name) for name in files},
             'settings_sha256': None}
    ref = cache.digest(value)
    archive = store / ref
    staging.rename(archive)
    (archive / 'manifest.json').write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
    monkeypatch.setattr(versions, 'PRIVATE', tmp_path / 'instance')
    write_json(record_dir / 'runtime.json', {'schema': 1, 'ref': ref,
               'manifest_sha256': sha256_file(archive / 'manifest.json')})
    return archive


def _current_snapshot(tmp_path):
    data = tmp_path / 'data'
    data.mkdir(parents=True)
    case = {'case_id': 'c-1', 'context': [], 'human_reply': ['回复']}
    (data / 'dev.jsonl').write_text(json.dumps(case, ensure_ascii=False) + '\n', encoding='utf-8')
    (data / 'purposes.json').write_text(json.dumps({
        'history_policy': 'complete_before_input_v1',
        'roles': {'judge_development': {'file': 'dev.jsonl', 'sha256': sha256_file(data / 'dev.jsonl')}},
        'protocol': {'development_start': 1, 'acceptance_start': 2, 'unseen_chat_ids': []},
    }), encoding='utf-8')
    return data_guard.generation_snapshot([case], data, [{'retriever': {'enabled': False}}])


def _legacy_snapshot_from_archive(current, archive, keys=None):
    """旧形态快照：全部实现件摘要用归档真实字节计算（不再从活树取）；
    归档缺失的件用占位摘要（供“缺件拒绝”路径走到 data_guard 的缺失报错）。"""
    keys = keys or {'generator/generator.py': 'src/generator/generator.py',
                    'generator/few_shot.py': 'src/generator/few_shot.py',
                    'generator/history.py': 'src/generator/history.py',
                    'iteration/datasets.py': 'src/iteration/datasets.py'}
    legacy = copy.deepcopy(current)
    legacy['implementation_sha256'] = {
        name: sha256_file(archive / relative) if (archive / relative).is_file() else '0' * 64
        for name, relative in keys.items()}
    return legacy


_LEGACY_ARCHIVE_FILES = {'src/generator/generator.py': 'old generator\n',
                         'src/generator/few_shot.py': 'old few_shot\n',
                         'src/generator/history.py': 'old history\n',
                         'src/iteration/datasets.py': 'old datasets\n'}


def _experiment(tmp_path):
    directory = tmp_path / 'experiment'
    directory.mkdir()
    return directory


def test_legacy_snapshot_resumes_against_archived_few_shot(monkeypatch, tmp_path):
    experiment = _experiment(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment, _LEGACY_ARCHIVE_FILES)
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot_from_archive(current, archive)
    data_guard.verify(experiment, current, expected=legacy, has_checkpoint=True)
    # saved（磁盘上的旧断点快照）路径同样按归档字节通过。
    write_json(experiment / 'data_snapshot.json', legacy)
    data_guard.verify(experiment, copy.deepcopy(current), has_checkpoint=True)


def test_legacy_snapshot_ignores_evolved_live_tree(monkeypatch, tmp_path):
    """活树四件实现字节已演进：旧快照身份 = 归档字节（未变）→ 续跑放行。"""
    experiment = _experiment(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment, _LEGACY_ARCHIVE_FILES)
    original = data_guard.sha256_file
    monkeypatch.setattr(data_guard, 'sha256_file',
                        lambda path: 'e' * 64 if Path(path).suffix == '.py' else original(path))
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot_from_archive(current, archive)
    data_guard.verify(experiment, current, expected=legacy, has_checkpoint=True)


def test_legacy_session_only_snapshot_resumes_against_archive(monkeypatch, tmp_path):
    """单件 'generator/fewshot/session.py' 旧形态同样按归档字节核验续跑。"""
    experiment = _experiment(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment,
                                    {'src/generator/fewshot/session.py': 'old session\n'})
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot_from_archive(
        current, archive, keys={'generator/fewshot/session.py': 'src/generator/fewshot/session.py'})
    data_guard.verify(experiment, current, expected=legacy, has_checkpoint=True)


def test_legacy_session_four_piece_snapshot_resumes(monkeypatch, tmp_path):
    """中间代四件键集（session.py 替换 few_shot.py）：全部实现件按归档字节核验。"""
    experiment = _experiment(tmp_path)
    files = {name: content for name, content in _LEGACY_ARCHIVE_FILES.items()
             if 'few_shot' not in name}
    files['src/generator/fewshot/session.py'] = 'old session\n'
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment, files)
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot_from_archive(current, archive, keys={
        'generator/generator.py': 'src/generator/generator.py',
        'generator/history.py': 'src/generator/history.py',
        'iteration/datasets.py': 'src/iteration/datasets.py',
        'generator/fewshot/session.py': 'src/generator/fewshot/session.py'})
    data_guard.verify(experiment, current, expected=legacy, has_checkpoint=True)


def test_real_archive_legacy_snapshot_positive(monkeypatch, tmp_path):
    """真实 a5118947 归档正例：旧形态四件全按归档字节 → 续跑放行。"""
    from _reference_few_shot import ARCHIVE_SHA256, archive_root
    archive = archive_root()
    if not (archive / 'manifest.json').is_file():
        pytest.skip('本机认证归档不存在（干净 checkout / CI）')
    monkeypatch.setattr(versions, 'PRIVATE', archive_root().parents[1])
    experiment = _experiment(tmp_path)
    write_json(experiment / 'runtime.json', {'schema': 1, 'ref': ARCHIVE_SHA256,
               'manifest_sha256': sha256_file(archive / 'manifest.json')})
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot_from_archive(current, archive)
    data_guard.verify(experiment, current, expected=legacy, has_checkpoint=True)
    write_json(experiment / 'data_snapshot.json', legacy)
    data_guard.verify(experiment, copy.deepcopy(current), has_checkpoint=True)


def test_tampered_archive_few_shot_rejects_legacy_resume(monkeypatch, tmp_path):
    experiment = _experiment(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment, _LEGACY_ARCHIVE_FILES)
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot_from_archive(current, archive)
    legacy['implementation_sha256']['generator/few_shot.py'] = '0' * 64
    with pytest.raises(ConfigError, match='冻结数据指纹不一致'):
        data_guard.verify(experiment, current, expected=legacy)


def test_legacy_snapshot_without_archive_implementation_rejected(monkeypatch, tmp_path):
    """归档缺少某实现件（如 datasets.py）→ 注明缺失并禁止续跑。"""
    experiment = _experiment(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment,
                                    {'src/generator/few_shot.py': 'old few_shot\n'})
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot_from_archive(current, archive)
    with pytest.raises(ConfigError, match='未包含 src/generator/generator.py'):
        data_guard.verify(experiment, current, expected=legacy)


def test_missing_runtime_archive_rejects_legacy_resume(tmp_path):
    experiment = _experiment(tmp_path)
    current = _current_snapshot(tmp_path)
    legacy = copy.deepcopy(current)
    legacy['implementation_sha256'] = {name: '0' * 64 for name in (
        'generator/generator.py', 'generator/few_shot.py',
        'generator/history.py', 'iteration/datasets.py')}
    with pytest.raises(ConfigError, match='归档证据'):
        data_guard.verify(experiment, current, expected=legacy)


def test_rank_py_evolution_does_not_block_legacy_resume(monkeypatch, tmp_path):
    """活树 rank.py 字节已演进，但旧快照身份 = 归档字节（未变）→ 续跑放行。"""
    experiment = _experiment(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment, _LEGACY_ARCHIVE_FILES)
    rank_path = Path(ranker_samples.__file__).parent / 'fewshot' / 'rank.py'
    original = data_guard.sha256_file

    def evolved(path):
        return 'e' * 64 if Path(path).resolve() == rank_path.resolve() else original(path)

    monkeypatch.setattr(data_guard, 'sha256_file', evolved)
    current = _current_snapshot(tmp_path)
    assert current['implementation_sha256']['generator/fewshot/rank.py'] == 'e' * 64
    legacy = _legacy_snapshot_from_archive(current, archive)
    data_guard.verify(experiment, current, expected=legacy, has_checkpoint=True)
    # 新形态快照不受过渡兼容保护：rank.py 失配照旧拒绝。
    saved_new = copy.deepcopy(current)
    saved_new['implementation_sha256']['generator/fewshot/rank.py'] = '0' * 64
    with pytest.raises(ConfigError, match='冻结数据指纹不一致'):
        data_guard.verify(experiment, current, expected=saved_new)


def _inventory(tmp_path):
    fixture = historical(tmp_path / 'history')
    teacher = teacher_at(tmp_path, fixture, 1)
    generator = tmp_path / 'generator'
    write_json(generator / 'config.json', dict(max_shots_per_case=3, shots_char_budget=2500))
    output = tmp_path / 'inventory'
    ranker_samples.prepare(fixture.directory, teacher, generator, output, progress=lambda _: None)
    return fixture, teacher, generator, output


def _repo_src_archived_names(inputs):
    """旧清单输入键 → runtime 归档相对名；只映射仓库 src/ 下的实现/检查件，
    数据类输入（池/cases/teacher/generator）不在归档内。"""
    repo_src = Path(ranker_samples.__file__).resolve().parents[1]
    names = {}
    for key in inputs:
        try:
            relative = Path(key).resolve().relative_to(repo_src)
        except ValueError:
            continue
        names[key] = 'src/' + relative.as_posix()
    return names


def _legacy_archive(monkeypatch, tmp_path, manifest_inputs, legacy_bytes):
    """建假 runtime 归档：旧清单的全部 src/ 输入件真实写入旧字节（未指定的用
    通用旧字节），数据类输入不进归档。返回归档目录。"""
    files = dict(legacy_bytes)
    for archived_name in _repo_src_archived_names(manifest_inputs).values():
        files.setdefault(archived_name, f'old {Path(archived_name).name}\n')
    return _fake_runtime_archive(monkeypatch, tmp_path, tmp_path, files)


def _to_legacy_manifest(output, legacy_key, legacy_sha, archive=None):
    """把新形态 manifest 改写为旧形态清单，并把断点/报告签名一并重盖为旧清单摘要，
    模拟旧代码创建的运行（manifest/records/report 绑的都是旧清单）。
    archive 提供时：其余 src/ 输入摘要一律按归档真实字节回填（旧运行身份=归档字节）；
    archive=None 时保留活树摘要（仅用于归档缺失的拒绝路径，该路径先报缺 runtime.json）。"""
    manifest = read_json(output / 'manifest.json')
    inputs = {name: sha for name, sha in manifest['inputs'].items()
              if '/src/generator/fewshot/' not in name.replace('\\', '/')}
    legacy_resolved = str(Path(legacy_key).resolve())
    inputs[legacy_resolved] = legacy_sha
    if archive is not None:
        for key, archived_name in _repo_src_archived_names(inputs).items():
            if key != legacy_resolved:
                inputs[key] = sha256_file(archive / archived_name)
    manifest['inputs'] = inputs
    write_json(output / 'manifest.json', manifest)
    for path in sorted((output / 'targets').glob('*.json')):
        record = read_json(path)
        record['manifest_sha256'] = digest(manifest)
        record['payload_sha256'] = digest({k: v for k, v in record.items() if k != 'payload_sha256'})
        write_json(path, record)
    report = read_json(output / 'report.json')
    report['manifest_sha256'] = digest(manifest)
    write_json(output / 'report.json', report)
    files = {str(p.relative_to(output)): sha256_file(p)
             for p in sorted((output / 'targets').glob('*.json'))}
    write_json(output / 'inventory.json', {'files': files})


def _no_retrieval(*args, **kwargs):
    raise AssertionError('旧清单完成后续跑不得重新检索')


def test_legacy_monolith_inventory_resumes_against_archive(monkeypatch, tmp_path):
    fixture, teacher, generator, output = _inventory(tmp_path)
    manifest = read_json(output / 'manifest.json')
    legacy_key = Path(ranker_samples.__file__).resolve().parents[1] / 'generator/few_shot.py'
    archive = _legacy_archive(monkeypatch, tmp_path, manifest['inputs'],
                              {'src/generator/few_shot.py': 'legacy few_shot\n'})
    _to_legacy_manifest(output, str(legacy_key),
                        sha256_file(archive / 'src/generator/few_shot.py'), archive)
    monkeypatch.setattr(RecallSession, 'retrieve', _no_retrieval)
    report = ranker_samples.prepare(fixture.directory, teacher, generator, output,
                                    progress=lambda _: None)
    assert report == read_json(output / 'report.json')


def test_legacy_session_only_inventory_resumes_against_archive(monkeypatch, tmp_path):
    fixture, teacher, generator, output = _inventory(tmp_path)
    manifest = read_json(output / 'manifest.json')
    legacy_key = Path(ranker_samples.__file__).parent / 'fewshot' / 'session.py'
    archive = _legacy_archive(monkeypatch, tmp_path, manifest['inputs'],
                              {'src/generator/fewshot/session.py': 'legacy session\n'})
    _to_legacy_manifest(output, str(legacy_key),
                        sha256_file(archive / 'src/generator/fewshot/session.py'), archive)
    monkeypatch.setattr(RecallSession, 'retrieve', _no_retrieval)
    report = ranker_samples.prepare(fixture.directory, teacher, generator, output,
                                    progress=lambda _: None)
    assert report == read_json(output / 'report.json')


def test_tampered_legacy_inventory_rejected(monkeypatch, tmp_path):
    fixture, teacher, generator, output = _inventory(tmp_path)
    manifest = read_json(output / 'manifest.json')
    legacy_key = Path(ranker_samples.__file__).resolve().parents[1] / 'generator/few_shot.py'
    archive = _legacy_archive(monkeypatch, tmp_path, manifest['inputs'],
                              {'src/generator/few_shot.py': 'legacy\n'})
    _to_legacy_manifest(output, str(legacy_key), '0' * 64, archive)
    with pytest.raises(ConfigError, match='旧实现清单与运行归档字节不符'):
        ranker_samples.prepare(fixture.directory, teacher, generator, output,
                               progress=lambda _: None)


def test_legacy_inventory_without_runtime_archive_rejected(tmp_path):
    fixture, teacher, generator, output = _inventory(tmp_path)
    legacy_key = Path(ranker_samples.__file__).resolve().parents[1] / 'generator/few_shot.py'
    _to_legacy_manifest(output, str(legacy_key), '0' * 64)
    with pytest.raises(ConfigError, match='runtime.json'):
        ranker_samples.prepare(fixture.directory, teacher, generator, output,
                               progress=lambda _: None)


def _legacy_ready_inventory(monkeypatch, tmp_path):
    """可放行的旧形态清单场景（单件 monolith），返回 (fixture, teacher, generator, output)。"""
    fixture, teacher, generator, output = _inventory(tmp_path)
    manifest = read_json(output / 'manifest.json')
    legacy_key = Path(ranker_samples.__file__).resolve().parents[1] / 'generator/few_shot.py'
    archive = _legacy_archive(monkeypatch, tmp_path, manifest['inputs'],
                              {'src/generator/few_shot.py': 'legacy few_shot\n'})
    _to_legacy_manifest(output, str(legacy_key),
                        sha256_file(archive / 'src/generator/few_shot.py'), archive)
    return fixture, teacher, generator, output


def test_legacy_inventory_rejects_changed_recipe(monkeypatch, tmp_path):
    fixture, teacher, generator, output = _legacy_ready_inventory(monkeypatch, tmp_path)
    manifest = read_json(output / 'manifest.json')
    manifest['recipe'] = 'tampered_recipe_v9'
    write_json(output / 'manifest.json', manifest)
    with pytest.raises(ConfigError, match='不一致'):
        ranker_samples.prepare(fixture.directory, teacher, generator, output,
                               progress=lambda _: None)


def test_legacy_inventory_rejects_changed_data_input_set(monkeypatch, tmp_path):
    fixture, teacher, generator, output = _legacy_ready_inventory(monkeypatch, tmp_path)
    manifest = read_json(output / 'manifest.json')
    manifest['inputs'][str(tmp_path / 'extra_pool.jsonl')] = '0' * 64
    write_json(output / 'manifest.json', manifest)
    with pytest.raises(ConfigError, match='输入集合不一致'):
        ranker_samples.prepare(fixture.directory, teacher, generator, output,
                               progress=lambda _: None)


def test_legacy_inventory_rejects_frozen_only_field(monkeypatch, tmp_path):
    fixture, teacher, generator, output = _legacy_ready_inventory(monkeypatch, tmp_path)
    manifest = read_json(output / 'manifest.json')
    manifest['obsolete_field'] = {'anything': True}
    write_json(output / 'manifest.json', manifest)
    with pytest.raises(ConfigError, match='不存在的字段'):
        ranker_samples.prepare(fixture.directory, teacher, generator, output,
                               progress=lambda _: None)
