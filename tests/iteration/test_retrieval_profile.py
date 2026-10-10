"""Real-runtime diagnostics compare both routes without evaluating or drawing cases."""
import argparse
import json
from types import SimpleNamespace

import pytest

from scripts import check
from src.iteration import retrieval_profile, versions


class Retriever:
    def retrieve(self, **options):
        return [dict(id='same', feature=1)]

    def render_selected(self, rows):
        return json.dumps(rows), [row['id'] for row in rows]


def test_compare_alternates_and_restores_routes():
    retriever = Retriever()
    saved = retriever.retrieve
    calls = []
    def original(subject, **options):
        calls.append('old')
        return saved(**options)
    def recall(subject, case):
        calls.append('recall')
        return subject.retrieve(query=case['query'])
    result = retrieval_profile.compare(retriever, original, recall,
        [dict(query='one', chat_type='group'), dict(query='two', chat_type='private')], repeats=2)
    assert result['equivalent'] and result['model_calls'] == 0
    assert result['strata'] == dict(group=1, private=1)
    assert result['timings']['original']['calls'] == result['timings']['indexed']['calls'] == 4
    assert calls.count('old') == 4 and calls.count('recall') == 8
    assert retriever.retrieve == saved


def test_compare_rejects_changed_features_and_restores():
    retriever = Retriever()
    saved = retriever.retrieve
    with pytest.raises(ValueError, match='changed selected rows'):
        retrieval_profile.compare(retriever, lambda subject, **options: [dict(id='same', feature=2)],
            lambda subject, case: subject.retrieve(), [dict(chat_type='group')], repeats=1)
    assert retriever.retrieve == saved


@pytest.mark.parametrize('value', [0, -1, 51])
def test_profile_counts_bounded(value):
    with pytest.raises(argparse.ArgumentTypeError):
        retrieval_profile.positive(value)


def test_ranking_profile_reports_alternating_samples_without_model_calls():
    result = retrieval_profile.measure_ranking(sizes=(128,), repeats=2)
    assert result['scope'] == 'synthetic_ranking_kernel_only'
    assert result['equivalent'] and result['model_calls'] == 0
    assert result['experiment_results_written'] is False
    assert len(result['rank_sha256']) == 64
    assert len(result['results']) == 6
    for item in result['results']:
        assert item['candidates'] == 128
        assert item['consumed'] == (12 if item['consumption'] == 'prefix' else 128)
        assert all(len(samples) == 2 for samples in item['seconds'].values())
        assert all(value >= 0 for value in item['median_seconds'].values())


@pytest.mark.parametrize('sizes', [(), (0,), (1000001,), (1.5,)])
def test_ranking_profile_rejects_unbounded_sizes(sizes):
    with pytest.raises(ValueError, match='sizes'):
        retrieval_profile.measure_ranking(sizes=sizes)


def test_ranking_profile_rejects_changed_selection(monkeypatch):
    from src.generator.fewshot import rank

    monkeypatch.setattr(rank, '_ranked_candidates', lambda *args: iter(()))
    with pytest.raises(ValueError, match='stable selection order'):
        retrieval_profile.measure_ranking(sizes=(128,), repeats=1)


def test_check_code_ranking_profile_dispatch_stays_offline(monkeypatch, capsys):
    monkeypatch.setattr(check, 'command', lambda argv: 'PASS')
    monkeypatch.setattr(retrieval_profile, 'measure_ranking',
                        lambda: dict(scope='synthetic_ranking_kernel_only', model_calls=0))
    assert check.main(['code', '--ranking-profile']) == 0
    result = json.loads(capsys.readouterr().out)
    item = next(item for item in result['checks'] if item['name'] == 'ranking_kernel_profile')
    assert item['detail']['model_calls'] == 0


def test_eligibility_profile_reports_cold_and_warm_pairs_without_model_calls():
    from statistics import median

    result = retrieval_profile.measure_eligibility(sizes=(32,), repeats=2)
    assert result['scope'] == 'synthetic_eligibility_kernel_only'
    assert result['equivalent'] and result['model_calls'] == 0
    assert result['experiment_results_written'] is False
    assert result['recall_routes_per_case'] == 2
    assert result['prepared_minimum_identity_length'] == result['synthetic_id_length']
    assert len(result['recall_engines_sha256']) == 64
    assert len(result['results']) == 6
    for item in result['results']:
        assert item['rows'] == 33 and item['cache_admitted']
        assert item['temperature'] in ('cold_pair', 'warm_pair')
        assert item['route_order'] == [('uncached', 'cached'), ('cached', 'uncached')]
        assert item['sql_query_counts']['uncached'] == [2, 2]
        assert item['sql_query_counts']['cached'] == ([1, 1] if item['temperature'] == 'cold_pair' else [0, 0])
        assert item['paired_speedups'] == [uncached / cached
                                          for uncached, cached in zip(item['seconds']['uncached'],
                                                                     item['seconds']['cached'])]
        assert item['median_paired_speedup'] == median(item['paired_speedups'])
        for route in ('uncached', 'cached'):
            assert len(item['seconds'][route]) == 2
            for total, first, second in zip(item['seconds'][route], item['first_seconds'][route],
                                             item['second_seconds'][route]):
                assert total == first + second and first >= 0 and second >= 0
        assert all(0 < value <= result['eligibility_cache_budget_bytes']
                   for value in item['estimated_cache_payload_bytes'])


@pytest.mark.parametrize('sizes', [(), (0,), (7,), (1000001,), (8.5,)])
def test_eligibility_profile_rejects_unbounded_sizes(sizes):
    with pytest.raises(ValueError, match='sizes'):
        retrieval_profile.measure_eligibility(sizes=sizes)


def test_eligibility_profile_rejects_changed_exclusions(monkeypatch):
    from src.generator.fewshot.recall_engines import SqliteIndexedRecallEngine

    monkeypatch.setattr(SqliteIndexedRecallEngine, 'excluded_ids', lambda *args: set())
    with pytest.raises(ValueError, match='changed exclusion IDs'):
        retrieval_profile.measure_eligibility(sizes=(32,), repeats=1)


def test_check_code_eligibility_profile_dispatch_stays_offline(monkeypatch, capsys):
    monkeypatch.setattr(check, 'command', lambda argv: 'PASS')
    calls = []
    monkeypatch.setattr(retrieval_profile, 'measure_eligibility', lambda **options:
                        calls.append(options) or dict(scope='synthetic_eligibility_kernel_only', model_calls=0))
    assert check.main(['code', '--eligibility-profile', '--eligibility-repeats', '9']) == 0
    assert calls == [dict(repeats=9)]
    result = json.loads(capsys.readouterr().out)
    item = next(item for item in result['checks'] if item['name'] == 'eligibility_kernel_profile')
    assert item['detail']['model_calls'] == 0


def test_check_retrieval_dispatch_is_offline(tmp_path, monkeypatch, capsys):
    pointer = tmp_path / 'pointers.json'
    pointer.write_text('{}')
    monkeypatch.setattr(versions, 'switch_instance', lambda name: tmp_path)
    monkeypatch.setattr(versions, 'POINTERS_PATH', pointer)
    calls = []
    from src.iteration import retrieval_compare
    monkeypatch.setattr(retrieval_compare, 'compare', lambda *args, **options:
        calls.append(('compare', args, options)) or dict(equivalent=True, model_calls=0))
    monkeypatch.setattr(retrieval_profile, 'inspect', lambda *args, **options:
        calls.append(('inspect', args, options)) or dict(equivalent=True, model_calls=0))
    assert check.main(['retrieval', '--instance', 'demo', '--exp', 'saved', '--cases', '2']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['model_requests_prohibited'] and not report['evaluation_started']
    assert calls == [
        ('compare', ('demo', 'saved'), dict(cases=2, baseline_runtime=None, output=None)),
        ('inspect', ('demo', 'saved'), dict(cases=2, repeats=1, materials=False)),
    ]


def test_modern_micro_snapshot_passes_check_retrieval_entry(tmp_path, monkeypatch, capsys):
    """新代（微）快照通过 check.py retrieval 完整入口：旧 profiler 不再从快照
    import few_shot，改为按认证 manifest 分代并对新包做身份核验。"""
    import argparse
    import sys
    from pathlib import Path
    from src.infrastructure.config import sha256_file
    from src.generator import fewshot as fewshot_package
    from src.iteration import pack_transport, runtime, retrieval_compare
    from src.infrastructure.storage import write_json

    # 新代微快照：真实 fewshot 包字节 + 认证清单（含冻结 transport 条目）。
    snapshot = tmp_path / 'snapshot'
    package_dir = Path(fewshot_package.__file__).parent
    sources = {}
    for path in sorted(package_dir.glob('*.py')):
        sources[f'src/generator/fewshot/{path.name}'] = path
    sources['src/iteration/pack_transport.py'] = Path(pack_transport.__file__)
    files = {}
    for name, source in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        files[name] = sha256_file(source)
    write_json(snapshot / 'manifest.json', {'files': files})
    authenticated = snapshot.with_name(runtime.digest({'files': files}))
    snapshot.rename(authenticated)
    snapshot = authenticated

    instance = tmp_path / 'instance'
    directory = instance / 'experiments' / 'frozen'
    directory.mkdir(parents=True)
    write_json(directory / 'runtime.json', {'ref': snapshot.name})
    pointer = tmp_path / 'pointers.json'
    pointer.write_text('{}')

    monkeypatch.setattr(versions, 'switch_instance', lambda name: instance)
    monkeypatch.setattr(versions, 'PRIVATE', instance)
    monkeypatch.setattr(versions, 'POINTERS_PATH', pointer)
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    monkeypatch.setattr(runtime, 'require_current', lambda path: None)
    monkeypatch.setattr(retrieval_compare, 'compare', lambda *args, **options:
        dict(equivalent=True, model_calls=0))

    def run_frozen_in_process(command, **options):
        namespace = argparse.Namespace(
            snapshot=Path(command[command.index('--snapshot') + 1]),
            instance=command[command.index('--instance') + 1],
            experiment=command[command.index('--experiment') + 1],
            cases=int(command[command.index('--cases') + 1]),
            repeats=int(command[command.index('--repeats') + 1]),
            materials='--materials' in command)
        sys_path_before = list(sys.path)
        try:
            result = retrieval_profile.frozen(namespace)
        finally:
            sys.path[:] = sys_path_before
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr='')

    monkeypatch.setattr(retrieval_profile.subprocess, 'run', run_frozen_in_process)
    assert check.main(['retrieval', '--instance', 'demo', '--exp', 'frozen', '--materials']) == 0
    report = json.loads(capsys.readouterr().out)
    profile = next(c for c in report['checks'] if c['name'] == 'frozen_retrieval_profile')
    assert profile['status'] == 'passed'
    assert profile['detail']['generation'] == 'fewshot_package_v1'
    assert profile['detail']['equivalent'] is True
    reuse = next(c for c in report['checks'] if c['name'] == 'material_reuse')
    assert reuse['status'] == 'passed'


def test_modern_snapshot_identity_mismatch_is_rejected(tmp_path, monkeypatch):
    """新代快照 fewshot 包字节与认证清单不符 → 身份核验拒绝（fail-closed）。"""
    import argparse
    from pathlib import Path
    from src.infrastructure.config import sha256_file
    from src.generator import fewshot as fewshot_package
    from src.iteration import runtime, versions
    from src.infrastructure.storage import write_json

    snapshot = tmp_path / 'snapshot'
    package_dir = Path(fewshot_package.__file__).parent
    files = {}
    for path in sorted(package_dir.glob('*.py')):
        target = snapshot / 'src' / 'generator' / 'fewshot' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        files[f'src/generator/fewshot/{path.name}'] = sha256_file(path)
    write_json(snapshot / 'manifest.json', {'files': files})
    authenticated = snapshot.with_name(runtime.digest({'files': files}))
    snapshot.rename(authenticated)
    snapshot = authenticated
    tampered = next(snapshot.glob('src/generator/fewshot/*.py'))
    tampered.write_text('# tampered\n')

    instance = tmp_path / 'instance'
    directory = instance / 'experiments' / 'frozen'
    directory.mkdir(parents=True)
    write_json(directory / 'runtime.json', {'ref': snapshot.name})
    monkeypatch.setattr(versions, 'switch_instance', lambda name: instance)
    monkeypatch.setattr(versions, 'PRIVATE', instance)
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    monkeypatch.setattr(runtime, 'require_current', lambda path: None)
    args = argparse.Namespace(snapshot=snapshot, instance='demo', experiment='frozen',
                              cases=1, repeats=1, materials=False)
    with pytest.raises(ValueError, match='source changed'):
        retrieval_profile.frozen(args)
    from src.iteration import runtime
    monkeypatch.setattr(versions, 'PRIVATE', tmp_path / 'instances' / 'demo')
    snapshot = tmp_path / 'runtime'
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    calls = []
    def launch(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout='{"model_calls":0}', stderr='')
    monkeypatch.setattr(retrieval_profile.subprocess, 'run', launch)
    assert retrieval_profile.inspect('demo', 'saved', cases=2) == dict(model_calls=0)
    command, options = calls[0]
    assert command[1] == '-B' and '--snapshot' in command
    assert options['cwd'] == snapshot
    assert options['env']['DH_INSTANCES_ROOT'] == str(versions.PRIVATE.parent)
    assert options['env']['DH_SETTINGS_FILE'] == str(snapshot / 'settings.snapshot.yaml')
    assert options['env']['DH_ENV_FILE'].endswith('.env')


def test_material_failure_preserves_successful_retrieval(tmp_path, monkeypatch, capsys):
    pointer = tmp_path / 'pointers.json'
    pointer.write_text('{}')
    monkeypatch.setattr(versions, 'switch_instance', lambda name: tmp_path)
    monkeypatch.setattr(versions, 'POINTERS_PATH', pointer)
    from src.iteration import retrieval_compare
    monkeypatch.setattr(retrieval_compare, 'compare', lambda *args, **options:
        dict(equivalent=True, model_calls=0))
    result = dict(equivalent=True, model_calls=0, timings={'indexed': 1},
                  materials=dict(status='failed', error='cached input missing'))
    monkeypatch.setattr(retrieval_profile, 'inspect', lambda *args, **options: result)
    assert check.main(['retrieval', '--instance', 'demo', '--exp', 'saved', '--materials']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['checks'][1]['detail']['timings'] == {'indexed': 1}
    assert report['checks'][1]['status'] == 'passed'
    assert report['checks'][2]['status'] == 'failed'


def test_material_diagnostic_collects_reuse_and_partial_failure():
    from contextlib import nullcontext
    calls = []
    def require(*args):
        calls.append(args)
        if len(calls) == 2:
            raise ValueError('missing cache result')
    reuse = SimpleNamespace(verified_materials=lambda *args, **options: nullcontext())
    guard = SimpleNamespace(require_materials=require)
    profile = SimpleNamespace(snapshot=lambda: dict(metrics={'reused': 1}))
    result = retrieval_profile.measure_materials(reuse, guard, 'data', [], 'fixed',
                                                 profile=profile, disk=None)
    assert result['status'] == 'failed'
    assert len(result['calls_seconds']) == 1
    assert result['metrics'] == {'reused': 1}
    assert 'missing cache result' in result['error']


def test_material_diagnostic_success_checks_two_contexts():
    from contextlib import nullcontext
    calls = []
    reuse = SimpleNamespace(verified_materials=lambda *args, **options: nullcontext())
    guard = SimpleNamespace(require_materials=lambda *args: calls.append(args))
    profile = SimpleNamespace(snapshot=lambda: dict(metrics={}))
    result = retrieval_profile.measure_materials(reuse, guard, 'data', [], 'fixed',
                                                 profile=profile, disk=None)
    assert result['status'] == 'passed'
    assert len(result['calls_seconds']) == len(calls) == 2


def test_frozen_legacy_full_call_does_not_raise_name_error(tmp_path, monkeypatch):
    """旧代 frozen() 完整调用（边界 mock）：transport 显式传参进 _frozen_legacy，
    不抛 NameError，且分代/守卫/指针核对全部走完。"""
    import sys
    import types as types_module
    import importlib.util
    from contextlib import nullcontext
    from src.iteration import runtime, versions
    from src.iteration import datasets
    from src.infrastructure.storage import write_json

    # 假旧代快照：manifest 只含 few_shot.py → 分代判定为旧代。
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'manifest.json').write_text(
        json.dumps({'files': {'src/generator/few_shot.py': 'ab' * 32}}), encoding='utf-8')

    instance = tmp_path / 'instance'
    directory = instance / 'experiments' / 'frozen'
    directory.mkdir(parents=True)
    write_json(directory / 'runtime.json', {'ref': snapshot.name})
    write_json(directory / 'spec.json', {'data_ref': 'd-1', 'dataset': 'dev'})
    cases_file = tmp_path / 'cases.jsonl'
    cases_file.write_text(json.dumps({'case_id': 'c-1', 'context': [], 'human_reply': ['回复']},
                                     ensure_ascii=False) + '\n', encoding='utf-8')

    pointer = tmp_path / 'pointers.json'
    pointer.write_text('{}')
    monkeypatch.setattr(versions, 'switch_instance', lambda name: None)
    monkeypatch.setattr(versions, 'PRIVATE', instance)
    monkeypatch.setattr(versions, 'POINTERS_PATH', pointer)
    monkeypatch.setattr(versions, 'data_version_dir', lambda ref: tmp_path / 'data')
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    monkeypatch.setattr(runtime, 'require_current', lambda path: None)
    monkeypatch.setattr(datasets, 'case_path', lambda data, name: cases_file)

    # 假 transport：记录接收方，分代判定为旧代，归档/磁盘上下文全部直通。
    loaded = []
    transport = SimpleNamespace(
        authenticated_snapshot_imports=lambda snap: nullcontext(),
        snapshot_generation=lambda snap, rt: 'few_shot',
        archived_runtime_paths=lambda rt: nullcontext(),
        archived_gbdt_paths=lambda ge, rt: nullcontext(),
        history_feature_cache=lambda fs: nullcontext(),
        live_material_reuse=lambda: None,
    )
    monkeypatch.setattr(retrieval_profile, '_load_transport',
                        lambda: loaded.append(True) or transport)

    # 假磁盘适配器模块：__file__ 旁的真实 indexed_retrieval.py（与归档同布局）。
    disk_dir = tmp_path / 'disk_history_pkg'
    disk_dir.mkdir()
    (disk_dir / 'disk_history.py').write_text(
        "MODE = 'fake'\n"
        "def disk_storage(module):\n"
        "    from contextlib import nullcontext\n"
        "    return nullcontext()\n", encoding='utf-8')
    (disk_dir / 'indexed_retrieval.py').write_text(
        'def indexed_retrieve(original):\n'
        '    def indexed(subject, **options):\n'
        '        return []\n'
        '    return indexed\n', encoding='utf-8')
    spec = importlib.util.spec_from_file_location('disk_history_fake', disk_dir / 'disk_history.py')
    disk_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(disk_module)
    transport.live_disk_history = lambda: disk_module

    class FakeProfile:
        def __init__(self, path, options):
            pass

        def snapshot(self):
            return {'metrics': {}}

    transport.live_transport_profile = lambda: SimpleNamespace(TransportProfile=FakeProfile)

    # 假旧 few_shot 模块：检索恒空、审批通过（经 sys.modules 注入，from import 生效）。
    few_shot_module = types_module.ModuleType('src.generator.few_shot')

    class FakeRetriever:
        def __init__(self, pool):
            self.pool = pool

        def is_approved(self):
            return True

        def retrieve(self, **options):
            return []

        def render_selected(self, rows, max_chars=2500):
            return '', []

    few_shot_module.PersonaFewShotRetriever = FakeRetriever
    monkeypatch.setitem(sys.modules, 'src.generator.few_shot', few_shot_module)

    args = SimpleNamespace(snapshot=snapshot, instance='demo', experiment='frozen',
                           cases=1, repeats=1, materials=False)
    sys_path_before = list(sys.path)
    try:
        result = retrieval_profile.frozen(args)
    finally:
        sys.path[:] = sys_path_before
    assert loaded == [True]
    assert result['equivalent'] is True
    assert result['runtime_ref'] == snapshot.name
    assert result['storage'] == 'fake'
