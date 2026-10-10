"""独立三方检索对比 harness：合成小池 + 假归档目录验证三分支加载与落盘格式。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from src.infrastructure.cache import digest
from src.generator.fewshot import open_session
from src.iteration import retrieval_compare, versions


def make_pool(directory: Path) -> Path:
    """合成普通小池（非历史池）：两行可信示例 + 审批 report。"""
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(id='row-a', context=['今天天气不错'], reply=['是啊'],             context_messages=[{'sender': 'x', 'text': '今天天气不错'}],
             reply_messages=[{'sender': 'me', 'text': '是啊'}],
             relationship='private', source_provenance='explicit_human_marker'),
        dict(id='row-b', context=['晚上吃什么'], reply=['火锅吧'],
             relationship='private', source_provenance='before_automation_cutoff'),
    ]
    pool = directory / 'fewshot_pool.jsonl'
    pool.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n',
                    encoding='utf-8')
    report = dict(review_status='approved',
                  examples_sha256=hashlib.sha256(pool.read_bytes()).hexdigest())
    pool.with_name('report.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    return pool


def make_cases(path: Path, count: int = 2) -> list:
    cases = [dict(case_id=f'case-{i}', chat_name=f'chat-{i}', chat_type='private',
                  context=[{'sender': 'other', 'text': f'消息{i} 天气'},
                           {'sender': 'me', 'text': f'回复{i}'}])
             for i in range(count)]
    path.write_text('\n'.join(json.dumps(case, ensure_ascii=False) for case in cases) + '\n',
                    encoding='utf-8')
    return cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fake_archive(root: Path, *, source='legacy = True\n') -> Path:
    """假归档：manifest 摘要 == 目录名、含旧代 few_shot.py、真实文件字节哈希。"""
    staging = root / '.staging-archive'
    (staging / 'src' / 'generator').mkdir(parents=True)
    (staging / 'scripts').mkdir()
    (staging / 'src' / '__init__.py').write_text('')
    (staging / 'src' / 'generator' / '__init__.py').write_text('')
    (staging / 'src' / 'generator' / 'few_shot.py').write_text(source)
    (staging / 'scripts' / 'run.py').write_text('legacy = True\n')
    (staging / 'settings.snapshot.yaml').write_text('evaluation: {}\n')
    value = {'files': {str(path.relative_to(staging)): _sha256(path)
                       for folder in ('src', 'scripts')
                       for path in (staging / folder).rglob('*') if path.is_file()},
             'settings_sha256': _sha256(staging / 'settings.snapshot.yaml')}
    archive = root / digest(value)
    staging.rename(archive)
    (archive / 'manifest.json').write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
    return archive


def route_payload(ids=('row-a', 'row-b')):
    rows = [dict(id=identity, context=['c'], reply=['r'],
                 retrieval_features=dict(production_final_score=1.0))
            for identity in ids]
    return dict(ids=[str(row['id']) for row in rows],
                row_sha256=[retrieval_compare.row_digest(row) for row in rows],
                rows=rows,
                retrieval_features=[row['retrieval_features'] for row in rows],
                render=dict(text='<style_examples>mock</style_examples>', ids=[str(i) for i in ids]))


def case_payload(case_id='case-0'):
    return dict(case_id=case_id, routes={'recent3': route_payload(), 'recent1': route_payload()})


def test_row_digest_is_normalized_and_sensitive():
    left = {'b': 1, 'a': [2, {'y': 1, 'x': 2}]}
    right = {'a': [2, {'x': 2, 'y': 1}], 'b': 1}
    assert retrieval_compare.row_digest(left) == retrieval_compare.row_digest(right)
    assert retrieval_compare.row_digest(left) != retrieval_compare.row_digest({'a': [2], 'b': 1})


def test_drive_captures_ids_row_hashes_full_trace_and_render(tmp_path):
    pool = make_pool(tmp_path / 'data')
    session = open_session(pool, index_policy='use_if_present')
    try:
        case = dict(case_id='case-0', chat_name='chat', chat_type='private',
                    context=[{'sender': 'other', 'text': '今天天气不错'}])
        routes = retrieval_compare.drive(session, case)
    finally:
        session.close()
    assert set(routes) == {'recent3', 'recent1'}
    for route in routes.values():
        assert route['ids'] and len(route['ids']) == len(route['row_sha256']) == len(route['rows'])
        assert all(trace and 'production_final_score' in trace for trace in route['retrieval_features'])
        assert route['render']['text'].startswith('<style_examples')
        assert route['render']['ids'] == route['ids']


def test_compare_subjects_equivalent_and_mismatch_fields():
    baseline = [case_payload('case-0'), case_payload('case-1')]
    assert retrieval_compare.compare_subjects(baseline, {
        'current_in_memory': [case_payload('case-0'), case_payload('case-1')],
        'current_indexed': [case_payload('case-0'), case_payload('case-1')]}) == []
    changed = case_payload('case-0')
    route = changed['routes']['recent3']
    route['ids'] = ['row-b']
    route['row_sha256'] = route['row_sha256'][:1]
    route['retrieval_features'] = route['retrieval_features'][:1]
    changed['routes']['recent1']['render']['text'] = 'different'
    mismatches = retrieval_compare.compare_subjects(baseline, {
        'current_in_memory': [changed, case_payload('case-1')],
        'current_indexed': [case_payload('case-0'), case_payload('case-1')]})
    assert {(m['case_id'], m['route'], m['subject'], tuple(m['fields'])) for m in mismatches} == {
        ('case-0', 'recent3', 'current_in_memory', ('ids', 'row_sha256', 'retrieval_features')),
        ('case-0', 'recent1', 'current_in_memory', ('render',)),
    }
    missing = retrieval_compare.compare_subjects(baseline, {'current_in_memory': [case_payload('case-0')]})
    assert any(m['fields'] == ['case_missing'] for m in missing)


def test_verify_baseline_archive_accepts_legacy_and_rejects_others(tmp_path):
    archive = make_fake_archive(tmp_path)
    assert retrieval_compare.verify_baseline_archive(archive) == archive.resolve()
    with pytest.raises(ValueError, match='manifest'):
        retrieval_compare.verify_baseline_archive(tmp_path / 'missing')
    tampered = make_fake_archive(tmp_path / 'other')
    tampered.joinpath('manifest.json').write_text('{"files": {}}')
    with pytest.raises(ValueError, match='authenticate'):
        retrieval_compare.verify_baseline_archive(tampered)


def test_verify_baseline_archive_rejects_tampered_source_bytes(tmp_path):
    """I7：篡改归档内一个 py 字节 → 完整字节核验拒绝，即使 manifest 摘要仍相符。"""
    archive = make_fake_archive(tmp_path)
    archive.joinpath('src', 'generator', 'few_shot.py').write_text('legacy = False\n')
    with pytest.raises(ValueError, match='full byte verification'):
        retrieval_compare.verify_baseline_archive(archive)


def test_verify_baseline_archive_accepts_real_a5118947_archive():
    """真实 a5118947 归档通过完整字节核验（旧基准可信）。"""
    from _reference_few_shot import archive_root
    archive = archive_root()
    if not (archive / 'manifest.json').is_file():
        pytest.skip('本机认证归档不存在（干净 checkout / CI）')
    assert retrieval_compare.verify_baseline_archive(archive) == archive.resolve()


ARCHIVED_FEW_SHOT = '''\
from pathlib import Path

class PersonaFewShotRetriever:
    def __init__(self, path, render_path=None):
        self.path = Path(path)

    def is_approved(self):
        return True

    def retrieve(self, query, **options):
        return [dict(id="row-archived", context=["c"], reply=["r"],
                     retrieval_features=dict(production_final_score=1.0, archived_module=__file__))]

    def render_selected(self, rows, max_chars=2500):
        return "rendered-by-archived", [row["id"] for row in rows]
'''


def test_worker_baseline_loads_archived_few_shot_in_isolated_process(tmp_path):
    """真子进程：归档 cwd + sys.path 首位加载假归档自己的 few_shot.py。"""
    archive = make_fake_archive(tmp_path, source=ARCHIVED_FEW_SHOT)
    make_cases(tmp_path / 'cases.jsonl', count=1)
    command = retrieval_compare.worker_command('baseline', pool=tmp_path / 'pool.jsonl',
                                               cases_file=tmp_path / 'cases.jsonl', count=1,
                                               archive=archive)
    result = subprocess.run(command, cwd=archive, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    archived_module = payload[0]['routes']['recent3']['rows'][0]['retrieval_features']['archived_module']
    assert archived_module.startswith(str(archive.resolve()))
    assert payload[0]['routes']['recent3']['render']['text'] == 'rendered-by-archived'


@pytest.mark.parametrize('engine', ['memory'])
def test_worker_current_runs_end_to_end(tmp_path, engine):
    """真子进程：当前树 fewshot 包（内存引擎）在合成池上完整 drive。"""
    pool = make_pool(tmp_path / 'data')
    make_cases(tmp_path / 'cases.jsonl', count=2)
    root = Path(__file__).resolve().parents[2]
    command = retrieval_compare.worker_command('current', pool=pool, cases_file=tmp_path / 'cases.jsonl',
                                               count=2, engine=engine)
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [item['case_id'] for item in payload] == ['case-0', 'case-1']
    assert all(item['routes']['recent3']['ids'] for item in payload)


def test_worker_current_indexed_fails_loudly_on_plain_pool(tmp_path):
    """I6：普通池（无历史策略）use_if_present 会回退内存引擎，被试必须明确失败。"""
    pool = make_pool(tmp_path / 'data')
    make_cases(tmp_path / 'cases.jsonl', count=1)
    command = retrieval_compare.worker_command('current', pool=pool,
                                               cases_file=tmp_path / 'cases.jsonl', count=1,
                                               engine='indexed')
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode != 0
    assert 'SqliteIndexedRecallEngine' in result.stderr


def test_worker_current_indexed_runs_on_frozen_history_pool(tmp_path):
    """I6 正例：历史池先 build 索引，use_if_present 必须挂 SqliteIndexed 引擎并完整 drive。"""
    from leakage_support import historical
    fixture = historical(tmp_path / 'data')
    pool = fixture.directory / 'fewshot_pool.jsonl'
    built = open_session(pool, index_policy='build')
    assert type(built._engine).__name__ == 'SqliteIndexedRecallEngine'
    built.close()
    cases = fixture.cases[:2]
    cases_file = tmp_path / 'cases.jsonl'
    cases_file.write_text('\n'.join(json.dumps(case, ensure_ascii=False) for case in cases) + '\n',
                          encoding='utf-8')
    command = retrieval_compare.worker_command('current', pool=pool, cases_file=cases_file,
                                               count=len(cases), engine='indexed')
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [item['case_id'] for item in payload] == [case['case_id'] for case in cases]
    assert all(set(item['routes']) == {'recent3', 'recent1'} for item in payload)


def test_compare_launches_three_isolated_workers_and_writes_full_output(tmp_path, monkeypatch):
    """compare() 父进程：三次独立子进程（归档 cwd + 活树两引擎）+ 落盘完整格式。"""
    instance = tmp_path / 'instances' / 'demo'
    experiment = instance / 'experiments' / 'saved'
    experiment.mkdir(parents=True)
    data = instance / 'data' / 'd0'
    data.mkdir(parents=True)
    pool = make_pool(data)
    cases = make_cases(data / 'judge_dev_pool.jsonl', count=2)
    experiment.joinpath('spec.json').write_text(json.dumps(
        dict(data_ref='d0', dataset='judge_development')))
    archive = make_fake_archive(instance / 'runtimes')
    output = tmp_path / 'out' / 'compare.json'
    root = Path(__file__).resolve().parents[2]

    monkeypatch.setattr(versions, 'PRIVATE', instance)
    monkeypatch.setattr(versions, 'data_version_dir', lambda ref: data)
    monkeypatch.setattr(retrieval_compare, 'DEFAULT_BASELINE_RUNTIME', archive.name)

    session = open_session(pool, index_policy='use_if_present')
    try:
        payload = [dict(case_id=case['case_id'], routes=retrieval_compare.drive(session, case))
                   for case in cases]
    finally:
        session.close()

    launches = []

    def fake_launch(command, *, cwd, env):
        launches.append((list(command), Path(cwd), dict(env)))
        return json.dumps(payload)

    monkeypatch.setattr(retrieval_compare, '_launch_worker', fake_launch)
    result = retrieval_compare.compare('demo', 'saved', cases=2, output=output)
    assert result['equivalent'] is True and result['mismatches'] == []
    assert result['model_calls'] == 0 and result['experiment_results_written'] is False
    assert result['baseline_runtime']['name'] == archive.name
    assert list(result['subjects']) == ['baseline_archive', 'current_in_memory', 'current_indexed']
    assert result['subjects']['current_indexed']['cases'] == payload

    # 三次独立子进程：旧基准用归档 cwd + 归档 settings，活树两引擎用仓库根 cwd。
    assert [item[1] for item in launches] == [archive.resolve(), root, root]
    assert launches[0][2]['DH_SETTINGS_FILE'] == str(archive / 'settings.snapshot.yaml')
    assert launches[0][2]['DH_INSTANCES_ROOT'] == str(instance.parent)
    assert '--archive' in launches[0][0] and '--engine' not in launches[0][0]
    assert '--engine' in launches[1][0] and '--engine' in launches[2][0]
    assert [item[0][item[0].index('--engine') + 1] for item in launches[1:]] == ['memory', 'indexed']

    # 落盘：完整数据（全部字段 + 完整行 + render 文本），可重新加载核验。
    saved = json.loads(output.read_text())
    assert saved['kind'] == 'retrieval_three_way_compare_v1'
    assert saved['compared_fields'] == list(retrieval_compare.COMPARE_FIELDS)
    route = saved['subjects']['current_in_memory']['cases'][0]['routes']['recent3']
    assert route['rows'] and 'retrieval_features' in route['rows'][0]
    assert route['render']['text'].startswith('<style_examples')
    assert saved['output'] == str(output)

    # 覆盖既有产物必须被拒绝（历史产物不可变；同值重试才允许）。
    from src.infrastructure.storage import write_once_json
    with pytest.raises(Exception):
        write_once_json(output, dict(saved, tampered=True))
    write_once_json(output, saved)  # 同值重试不报错
