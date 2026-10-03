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


def _legacy_snapshot(current, archived_few_shot_sha):
    legacy = copy.deepcopy(current)
    legacy['implementation_sha256'] = {
        'generator/generator.py': current['implementation_sha256']['generator/generator.py'],
        'generator/few_shot.py': archived_few_shot_sha,
        'generator/history.py': current['implementation_sha256']['generator/history.py'],
        'iteration/datasets.py': current['implementation_sha256']['iteration/datasets.py'],
    }
    return legacy


def _experiment(tmp_path):
    directory = tmp_path / 'experiment'
    directory.mkdir()
    return directory


def test_legacy_snapshot_resumes_against_archived_few_shot(monkeypatch, tmp_path):
    experiment = _experiment(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment,
                                    {'src/generator/few_shot.py': 'legacy = True\n'})
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot(current, sha256_file(archive / 'src/generator/few_shot.py'))
    data_guard.verify(experiment, current, expected=legacy, has_checkpoint=True)
    # saved（磁盘上的旧断点快照）路径同样按归档字节通过。
    write_json(experiment / 'data_snapshot.json', legacy)
    data_guard.verify(experiment, copy.deepcopy(current), has_checkpoint=True)


def test_tampered_archive_few_shot_rejects_legacy_resume(monkeypatch, tmp_path):
    experiment = _experiment(tmp_path)
    _fake_runtime_archive(monkeypatch, tmp_path, experiment, {'src/generator/few_shot.py': 'legacy = True\n'})
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot(current, '0' * 64)
    with pytest.raises(ConfigError, match='冻结数据指纹不一致'):
        data_guard.verify(experiment, current, expected=legacy)


def test_missing_runtime_archive_rejects_legacy_resume(tmp_path):
    experiment = _experiment(tmp_path)
    current = _current_snapshot(tmp_path)
    legacy = _legacy_snapshot(current, '0' * 64)
    with pytest.raises(ConfigError, match='归档证据'):
        data_guard.verify(experiment, current, expected=legacy)


def test_rank_py_evolution_does_not_block_legacy_resume(monkeypatch, tmp_path):
    """活树 rank.py 字节已演进，但旧快照身份 = 归档字节（未变）→ 续跑放行。"""
    experiment = _experiment(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, experiment,
                                    {'src/generator/few_shot.py': 'legacy = True\n'})
    rank_path = Path(ranker_samples.__file__).parent / 'fewshot' / 'rank.py'
    original = data_guard.sha256_file

    def evolved(path):
        return 'e' * 64 if Path(path).resolve() == rank_path.resolve() else original(path)

    monkeypatch.setattr(data_guard, 'sha256_file', evolved)
    current = _current_snapshot(tmp_path)
    assert current['implementation_sha256']['generator/fewshot/rank.py'] == 'e' * 64
    legacy = _legacy_snapshot(current, sha256_file(archive / 'src/generator/few_shot.py'))
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


def _to_legacy_manifest(output, legacy_key: str, legacy_sha: str):
    """把新形态 manifest 改写为旧形态清单，并把断点/报告签名一并重盖为旧清单摘要，
    模拟旧代码创建的运行（manifest/records/report 绑的都是旧清单）。"""
    manifest = read_json(output / 'manifest.json')
    inputs = {name: sha for name, sha in manifest['inputs'].items()
              if '/src/generator/fewshot/' not in name.replace('\\', '/')}
    inputs[str(Path(legacy_key).resolve())] = legacy_sha
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
    archive = _fake_runtime_archive(monkeypatch, tmp_path, tmp_path,
                                    {'src/generator/few_shot.py': 'legacy few_shot\n'})
    legacy_key = Path(ranker_samples.__file__).resolve().parents[1] / 'src/generator/few_shot.py'
    _to_legacy_manifest(output, str(legacy_key), sha256_file(archive / 'src/generator/few_shot.py'))
    monkeypatch.setattr(RecallSession, 'retrieve', _no_retrieval)
    report = ranker_samples.prepare(fixture.directory, teacher, generator, output,
                                    progress=lambda _: None)
    assert report == read_json(output / 'report.json')


def test_legacy_session_only_inventory_resumes_against_archive(monkeypatch, tmp_path):
    fixture, teacher, generator, output = _inventory(tmp_path)
    archive = _fake_runtime_archive(monkeypatch, tmp_path, tmp_path,
                                    {'src/generator/fewshot/session.py': 'legacy session\n'})
    session_key = Path(ranker_samples.__file__).parent / 'fewshot' / 'session.py'
    _to_legacy_manifest(output, str(session_key), sha256_file(archive / 'src/generator/fewshot/session.py'))
    monkeypatch.setattr(RecallSession, 'retrieve', _no_retrieval)
    report = ranker_samples.prepare(fixture.directory, teacher, generator, output,
                                    progress=lambda _: None)
    assert report == read_json(output / 'report.json')


def test_tampered_legacy_inventory_rejected(monkeypatch, tmp_path):
    fixture, teacher, generator, output = _inventory(tmp_path)
    _fake_runtime_archive(monkeypatch, tmp_path, tmp_path, {'src/generator/few_shot.py': 'legacy\n'})
    legacy_key = Path(ranker_samples.__file__).resolve().parents[1] / 'src/generator/few_shot.py'
    _to_legacy_manifest(output, str(legacy_key), '0' * 64)
    with pytest.raises(ConfigError, match='旧实现清单与运行归档字节不符'):
        ranker_samples.prepare(fixture.directory, teacher, generator, output,
                               progress=lambda _: None)


def test_legacy_inventory_without_runtime_archive_rejected(tmp_path):
    fixture, teacher, generator, output = _inventory(tmp_path)
    legacy_key = Path(ranker_samples.__file__).resolve().parents[1] / 'src/generator/few_shot.py'
    _to_legacy_manifest(output, str(legacy_key), '0' * 64)
    with pytest.raises(ConfigError, match='runtime.json'):
        ranker_samples.prepare(fixture.directory, teacher, generator, output,
                               progress=lambda _: None)
