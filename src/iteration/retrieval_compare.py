"""独立三方检索对比（设计 §7.2，回应评审第 10 条）。

旧基准对象完全由认证归档 runtime 自包含提供：在归档 cwd + sys.path 首位的
子进程里加载归档自己的 9月29 版 few_shot/history_sources（无 disk_history），
跑纯内存 retrieve，完全不经过活树适配器，因此独立于本次修改。被测对象为
当前树 fewshot 包的两个引擎（InMemory / SqliteIndexed，经 open_session
index_policy='use_if_present'），各在独立子进程加载。

比较内容 = 请求内容产物：完整有序行（id 序列 + 每行规范化 JSON 哈希，完整
行数据一并落盘）、include_retrieval_features trace 全字段（随行内的
retrieval_features 一起比较）、render_selected 输出。代码/索引/材料摘要
属 §6 允许变化的身份项，不纳入相等比较。完整数据落盘 analysis 目录。

cases 来源沿用现有 harness：冻结实验的 dev cases 前 N 题，不新抽题。
无模型请求、无实验写入、离线。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEFAULT_BASELINE_RUNTIME = 'a5118947eb888be5d46901f2e5334ce14fc16a932e8a503e5cc595a30d0ac1e4'

ROUTES = ('recent3', 'recent1')
# 纳入相等比较的请求内容字段；代码/索引/材料摘要等身份项不在其中。
COMPARE_FIELDS = ('ids', 'row_sha256', 'render', 'retrieval_features')

SUBJECTS = ('baseline_archive', 'current_in_memory', 'current_indexed')


def row_digest(row) -> str:
    """每行规范化 JSON 哈希（键排序 + 紧凑分隔符），与键插入顺序无关。"""
    normalized = json.dumps(row, ensure_ascii=False, sort_keys=True,
                            separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def drive(retriever, case) -> dict:
    """两路训练召回（recent3/recent1，limit=12）+ 全字段 trace + render。

    请求形状与 learned_selection.recall 的选项一致（八候选采样除外），
    include_retrieval_features=True 使返回行携带完整 retrieval_features trace。
    """
    messages = case.get('context', [])
    options = dict(chat_name=str(case.get('chat_name', '')),
        is_group=case.get('chat_type') == 'group', limit=12,
        exclude_ids={str(case.get('case_id', ''))},
        current_context_messages=[{'sender': str(m.get('sender', '')), 'text': str(m.get('text', ''))}
                                  for m in messages],
        history_case=case, include_retrieval_features=True)
    routes = {}
    for size in (3, 1):
        query = '\n'.join(str(m.get('text', '')) for m in messages[-size:])
        rows = retriever.retrieve(query=query, **options)
        text, ids = retriever.render_selected(rows)
        routes[f'recent{size}'] = dict(
            ids=[str(row['id']) for row in rows],
            row_sha256=[row_digest(row) for row in rows],
            rows=rows,
            retrieval_features=[row.get('retrieval_features') for row in rows],
            render=dict(text=text, ids=[str(i) for i in ids]),
        )
    return routes


def diff_routes(reference: dict, candidate: dict) -> list[str]:
    """单题单路的两份 drive 产物差异字段清单；空列表表示相等。"""
    return [field for field in COMPARE_FIELDS if reference[field] != candidate[field]]


def compare_subjects(baseline_cases: list, subject_cases: dict) -> list[dict]:
    """以旧基准为参照，汇总全部被试对象的差异；空列表表示三方等价。"""
    mismatches = []
    for position, reference in enumerate(baseline_cases):
        for name, cases in subject_cases.items():
            candidate = next((item for item in cases if item['case_id'] == reference['case_id']), None)
            if candidate is None:
                mismatches.append(dict(case_id=reference['case_id'], route='*', subject=name,
                                       fields=['case_missing']))
                continue
            for route in ROUTES:
                fields = diff_routes(reference['routes'][route], candidate['routes'][route])
                if fields:
                    mismatches.append(dict(case_id=reference['case_id'], route=route,
                                           subject=name, fields=fields))
    return mismatches


# ---------------------------------------------------------------------------
# 子进程 worker：__main__ 入口按 --worker 分流，复用现有 subprocess 模式。
# ---------------------------------------------------------------------------

def _load_worker_cases(path: Path, count: int) -> list[dict]:
    cases = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                cases.append(json.loads(line))
            if len(cases) >= count:
                break
    if not cases:
        raise ValueError('frozen case pool is empty')
    return cases


def _worker_baseline(args) -> list:
    """旧基准：归档 cwd + sys.path 首位加载归档自己的 few_shot（纯内存）。"""
    archive = Path(args.archive).resolve()
    sys.path.insert(0, str(archive))
    from src.generator import few_shot  # noqa: F401  归档副本，非活树
    if Path(few_shot.__file__).resolve().is_relative_to(archive) is False:
        raise RuntimeError(f'baseline worker must load archived few_shot, got {few_shot.__file__}')
    retriever = few_shot.PersonaFewShotRetriever(Path(args.pool))
    if not retriever.is_approved():
        raise ValueError('frozen history pool is not approved')
    return [dict(case_id=case['case_id'], routes=drive(retriever, case))
            for case in _load_worker_cases(Path(args.cases_file), args.count)]


def _worker_current(args) -> list:
    """新被试：当前树 fewshot 包，memory/indexed 两个引擎分别加载。"""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.generator.fewshot import prepare
    from src.generator.fewshot import session as fewshot_session
    from src.generator.fewshot.recall_engines import InMemoryRecallEngine
    if args.engine == 'indexed':
        retriever = fewshot_session.open_session(Path(args.pool), index_policy='use_if_present')
    else:
        pool = prepare.PreparedPool.load(Path(args.pool))
        retriever = fewshot_session.RecallSession(pool, InMemoryRecallEngine())
    try:
        if not retriever.is_approved():
            raise ValueError('frozen history pool is not approved')
        return [dict(case_id=case['case_id'], routes=drive(retriever, case))
                for case in _load_worker_cases(Path(args.cases_file), args.count)]
    finally:
        retriever.close()


def _worker_main(argv=None) -> list:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('baseline', 'current'), required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--engine', choices=('memory', 'indexed'))
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--cases-file', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args(argv)
    if args.worker == 'baseline':
        return _worker_baseline(args)
    return _worker_current(args)


# ---------------------------------------------------------------------------
# 父进程组装
# ---------------------------------------------------------------------------

def verify_baseline_archive(archive: Path) -> Path:
    """认证归档：manifest 摘要 == 目录名、旧代清单（含 few_shot、不含 fewshot 包）。"""
    from . import runtime
    from .pack_transport import snapshot_generation
    archive = Path(archive).resolve()
    manifest = archive / 'manifest.json'
    if not manifest.is_file():
        raise ValueError(f'baseline runtime archive is missing manifest.json: {archive}')
    value = json.loads(manifest.read_text())
    if runtime.digest(value) != archive.name:
        raise ValueError('baseline runtime archive manifest does not authenticate its directory name')
    if snapshot_generation(archive, runtime) != 'few_shot':
        raise ValueError('baseline runtime archive is not a legacy (few_shot) generation archive')
    if not (archive / 'src' / 'generator' / 'few_shot.py').is_file():
        raise ValueError('baseline runtime archive is missing src/generator/few_shot.py')
    return archive


def _launch_worker(command: list, *, cwd: Path, env: dict) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    if result.returncode:
        raise RuntimeError(f'retrieval compare worker failed: {result.stderr[-2000:]}')
    return result.stdout


def worker_command(worker: str, *, pool: Path, cases_file: Path, count: int,
                   archive: Path | None = None, engine: str | None = None) -> list:
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', worker,
               '--pool', str(pool), '--cases-file', str(cases_file), '--count', str(count)]
    if archive is not None:
        command.extend(['--archive', str(archive)])
    if engine is not None:
        command.extend(['--engine', engine])
    return command


def select_cases(instance: str, experiment: str, count: int) -> tuple[list, Path, Path]:
    """冻结实验 dev cases 前 N 题 + 冻结历史池路径 + 数据集 cases 文件路径。"""
    from src.config import valid_name
    from . import datasets, versions
    from .storage import read_json
    directory = versions.PRIVATE / 'experiments' / valid_name(experiment)
    spec = read_json(directory / 'spec.json')
    data = versions.data_version_dir(spec['data_ref'])
    cases_file = datasets.case_path(data, spec['dataset'])
    with cases_file.open(encoding='utf-8') as stream:
        cases = [json.loads(line) for line in stream if line.strip()][:count]
    if not cases:
        raise ValueError('frozen case pool is empty')
    return cases, data / 'fewshot_pool.jsonl', cases_file


def resolve_baseline(name_or_path, versions) -> Path:
    candidate = Path(str(name_or_path))
    if not candidate.is_absolute():
        candidate = versions.PRIVATE / 'runtimes' / str(name_or_path)
    return candidate.resolve()


def compare(instance: str, experiment: str, *, cases: int = 3,
            baseline_runtime: str | None = None,
            output: str | Path | None = None,
            launch=None) -> dict:
    """三方对比主入口；返回完整结果 dict（含落盘路径）。无模型请求/实验写入。"""
    launch = launch or _launch_worker
    from src.config import valid_name
    from . import versions
    from .pack_transport import connection_env_file
    from .storage import read_json, write_once_json
    started = time.time()
    case_list, pool, cases_file = select_cases(instance, experiment, cases)
    directory = versions.PRIVATE / 'experiments' / valid_name(experiment)
    spec = read_json(directory / 'spec.json')

    baseline = verify_baseline_archive(
        resolve_baseline(baseline_runtime or DEFAULT_BASELINE_RUNTIME, versions))
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
           'DH_ENV_FILE': connection_env_file(),
           'DH_INSTANCES_ROOT': str(versions.PRIVATE.parent)}
    root = Path(__file__).resolve().parents[2]

    def run(worker: str, *, cwd: Path, archive: Path | None = None,
            engine: str | None = None, settings_file: Path | None = None) -> list:
        worker_env = dict(env)
        if settings_file is not None:
            worker_env['DH_SETTINGS_FILE'] = str(settings_file)
        payload = launch(worker_command(worker, pool=pool, cases_file=cases_file,
                                        count=len(case_list), archive=archive, engine=engine),
                         cwd=cwd, env=worker_env)
        return json.loads(payload)

    baseline_payload = run('baseline', cwd=baseline, archive=baseline,
                           settings_file=baseline / 'settings.snapshot.yaml')
    memory_payload = run('current', cwd=root, engine='memory')
    indexed_payload = run('current', cwd=root, engine='indexed')

    subject_cases = {'current_in_memory': memory_payload, 'current_indexed': indexed_payload}
    mismatches = compare_subjects(baseline_payload, subject_cases)
    if output is None:
        analysis = versions.PRIVATE / 'analysis'
        analysis.mkdir(parents=True, exist_ok=True)
        output = analysis / time.strftime('retrieval-compare-%Y%m%d.json', time.localtime(started))
    output = Path(output)
    result = dict(schema=1, kind='retrieval_three_way_compare_v1',
                  equivalent=not mismatches,
                  selection='first_existing_frozen_cases_no_new_draw',
                  cases=len(case_list), routes_per_case=len(ROUTES),
                  experiment=valid_name(experiment), instance=valid_name(instance),
                  data_ref=spec['data_ref'], dataset=spec['dataset'],
                  pool=str(pool), cases_file=str(cases_file),
                  compared_fields=list(COMPARE_FIELDS),
                  identity_fields_excluded='code/index/material digests are identity, not equality (design §6)',
                  baseline_runtime=dict(name=baseline.name, path=str(baseline)),
                  subjects={
                      'baseline_archive': dict(engine='archived_few_shot_in_memory_v20260929',
                                               cases=baseline_payload),
                      'current_in_memory': dict(engine='fewshot.InMemoryRecallEngine', cases=memory_payload),
                      'current_indexed': dict(engine='fewshot.SqliteIndexedRecallEngine', cases=indexed_payload),
                  },
                  mismatches=mismatches,
                  model_calls=0, experiment_results_written=False,
                  elapsed_seconds=round(time.time() - started, 3),
                  output=str(output))
    write_once_json(output, result)
    return result


if __name__ == '__main__':
    print(json.dumps(_worker_main(), ensure_ascii=False))
