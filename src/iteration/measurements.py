"""按侧测量库（设计见 docs/DESIGN-SIDE-MEASUREMENT-BANK.md，v7 PASS）。

派生层：索引与查询用，永不作为判决权威。判决一律回读事实层 cases.jsonl
原文行重算指纹（audit_row）。实验 finish 时批量写入；失败实验整批丢弃；
同键内容一致幂等，不一致拒绝。回填与运行共用本模块。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from ..config import ConfigError, sha256_file
from . import experiment, versions
from .storage import write_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements(
  source_exp TEXT NOT NULL,
  source_line INTEGER NOT NULL,
  side_fp TEXT NOT NULL,
  data_ref TEXT NOT NULL,
  case_id TEXT NOT NULL,
  dataset_role TEXT NOT NULL,
  side_ref TEXT NOT NULL,
  judge_fp TEXT NOT NULL,
  pair_fp TEXT NOT NULL,
  round_fp TEXT NOT NULL,
  round_kind TEXT NOT NULL,
  round_no INTEGER NOT NULL,
  draw_nonce TEXT NOT NULL,
  origin_exp TEXT NOT NULL,
  origin_round_no INTEGER NOT NULL,
  verdict INTEGER NOT NULL,
  reply_sha256 TEXT,
  trace_ref TEXT,
  source_status TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY(source_exp, source_line, side_fp, round_fp)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_round_case_side ON measurements(round_fp, case_id, side_fp);
CREATE INDEX IF NOT EXISTS idx_pair ON measurements(pair_fp, round_fp);
CREATE INDEX IF NOT EXISTS idx_side ON measurements(side_fp, case_id);
CREATE INDEX IF NOT EXISTS idx_judge ON measurements(judge_fp, case_id);
"""


class MeasurementConflict(ConfigError):
    """同身份键内容不一致：数据损坏，拒绝写入。"""


# ---------- 指纹 ----------

def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def side_fingerprint(spec: dict, side: str, side_config: dict, side_dir: Path,
                     data_ref: str, dataset_role: str) -> tuple[str, str]:
    """返回 (side_ref, side_fp)。side ∈ baseline/candidate。

    gen_ab 侧 = 生成器版本（行为指纹 + 数据 + 用途）；judge_eval 侧 = Judge 版本
    （config 字节哈希 + 包哈希——包即"被测数据"）。"""
    if spec.get('kind') == 'judge_eval':
        side_ref = str(spec.get(side + '_ref', ''))
        pack = spec.get('pack_sha256') or ''
        fp = _sha({'kind': 'judge_side', 'config': side_config, 'pack': pack,
                   'data_ref': data_ref, 'dataset_role': dataset_role})
        return side_ref, fp
    side_ref = str(spec.get(side + '_ref', ''))
    from . import versions as _v
    fp = _sha({'kind': 'gen_side', 'behavior': _v.generator_behavior(side_config, side_dir),
               'data_ref': data_ref, 'dataset_role': dataset_role})
    return side_ref, fp


def judge_fingerprint(spec: dict, judge_config: dict, judge_dir: Path) -> str:
    """测量仪器指纹。gen_ab = 评分 Judge 全配置；judge_eval = 真值 + 包哈希。
    采样协议取 spec['protocol'] 全量（不止 flip 轮数/force_reply，评审 P2-4）。"""
    if spec.get('kind') == 'judge_eval':
        return _sha({'kind': 'ground_truth', 'pack': spec.get('pack_sha256'),
                     'protocol': spec.get('protocol')})
    assets = {}
    for name in ('prompt.md', 'correction.json'):
        path = judge_dir / name
        if path.is_file():
            assets[name] = sha256_file(path)
    return _sha({'kind': 'judge', 'config': judge_config, 'assets': assets,
                 'protocol': spec.get('protocol')})


def pair_fingerprint(side_fp_a: str, side_fp_b: str, judge_fp: str,
                     data_ref: str, dataset_role: str) -> str:
    ordered = sorted([side_fp_a, side_fp_b])
    return _sha({'pair': True, 'sides': ordered, 'judge': judge_fp,
                 'data_ref': data_ref, 'dataset_role': dataset_role})


def round_fingerprint(pair_fp: str, round_kind: str, draw_nonce: str) -> str:
    return _sha({'round': True, 'pair': pair_fp, 'kind': round_kind, 'nonce': draw_nonce})


# ---------- 解析 ----------

def _verdict_of(spec: dict, row: dict, side: str) -> int | None:
    if spec.get('kind') == 'judge_eval':
        value = row.get(side + '_correct')
    else:
        value = row.get('identified_' + side)
    if type(value) is not bool:
        return None
    return int(value)


def iter_side_measurements(exp_dir: Path) -> Iterable[dict]:
    """把一个实验的 cases.jsonl 解析为按侧测量行（初测 + 补验轮）。

    draw_nonce 优先取行内字段（runner 新写入，P2 重放逐字继承的来源）；
    历史行无此字段时以 (实验, 题, 轮次, 侧) 确定性派生作为过渡——该派生仅对
    原始实验解析正确（回填只解析原始实验），评审 P1-1。
    """
    spec = experiment.spec_of(exp_dir)
    data_ref = str(spec.get('data_ref', ''))
    dataset_role = ('fixed_test' if spec.get('dataset') == 'fixed_test'
                    else ('pack:' + str(spec.get('pack_ref')) if spec.get('kind') == 'judge_eval'
                          else str(spec.get('purpose') or 'development')))
    instance_dir = exp_dir.parent.parent
    kind = spec.get('kind')
    sides: dict[str, tuple[str, str]] = {}
    for side in ('baseline', 'candidate'):
        ref = str(spec.get(side + '_ref', ''))
        folder = 'judges' if kind == 'judge_eval' else 'generators'
        cfg_path = instance_dir / folder / ref / 'config.json'
        side_config = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        side_dir = instance_dir / folder / ref
        sides[side] = side_fingerprint(spec, side, side_config, side_dir, data_ref, dataset_role)
    if kind == 'judge_eval':
        judge_cfg_path = None
        judge_fp = judge_fingerprint(spec, {}, instance_dir)
    else:
        judge_ref = str(spec.get('judge_ref', ''))
        jdir = instance_dir / 'judges' / judge_ref
        jcfg = json.loads((jdir / 'config.json').read_text()) if (jdir / 'config.json').is_file() else {}
        judge_fp = judge_fingerprint(spec, jcfg, jdir)
    pair_fp = pair_fingerprint(sides['baseline'][1], sides['candidate'][1], judge_fp,
                               data_ref, dataset_role)

    cases_path = exp_dir / 'cases.jsonl'
    if not cases_path.exists():
        return
    source_status = experiment.state_of(exp_dir).get('status') or 'running'
    for line_no, line in enumerate(cases_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('status') != 'ok':
            continue
        if not row.get('case_id'):
            continue  # 空 case_id 无法锚定独立轮次，跳过（评审 P2-6）
        fv = row.get('flip_verified')
        if isinstance(fv, dict):
            flip = fv  # gen_ab：嵌套结构
        elif fv is True:
            flip = {'baseline_votes': row.get('baseline_votes'),  # judge_eval：扁平结构
                    'candidate_votes': row.get('candidate_votes')}
        else:
            flip = {}
        if sides['baseline'][1] == sides['candidate'][1]:
            continue  # 两侧无法区分（comparison 专项比较无版本绑定）：按侧归因不可能，跳过
        for side in ('baseline', 'candidate'):
            side_ref, side_fp = sides[side]
            initial = _verdict_of(spec, row, side)
            if initial is None:
                continue
            votes = flip.get(side + '_votes') or []
            if votes and votes[0] is not None and bool(votes[0]) != bool(initial):
                votes = []  # 首票与初测矛盾：flip 数据不可信，只保留初测（评审 P2-2）
            reply_value = row.get(side + '_replies')
            reply_sha = _sha(reply_value) if reply_value is not None else None
            nonce0 = row.get('draw_nonce') or str(uuid.uuid5(
                uuid.NAMESPACE_URL, f'{exp_dir.name}:{row.get("case_id")}:0'))
            yield {
                'source_exp': exp_dir.name, 'source_line': line_no, 'side_fp': side_fp,
                'data_ref': data_ref, 'case_id': str(row.get('case_id')), 'dataset_role': dataset_role,
                'side_ref': side_ref, 'judge_fp': judge_fp, 'pair_fp': pair_fp,
                'round_fp': round_fingerprint(pair_fp, 'initial', nonce0),
                'round_kind': 'initial', 'round_no': 0, 'draw_nonce': nonce0,
                'origin_exp': exp_dir.name, 'origin_round_no': 0,
                'verdict': initial, 'reply_sha256': reply_sha,
                'trace_ref': str(row.get('trace_ref') or ''), 'source_status': source_status,
                'created_at': time.time(),
            }
            for idx, vote in enumerate(votes[1:], 1):  # 首票=初测，已计（votes 已做一致性过滤）
                if type(vote) is not bool:
                    continue
                nonce = row.get(f'draw_nonce_r{idx}') or str(uuid.uuid5(
                    uuid.NAMESPACE_URL, f'{exp_dir.name}:{row.get("case_id")}:{idx}'))
                yield {
                    'source_exp': exp_dir.name, 'source_line': line_no, 'side_fp': side_fp,
                    'data_ref': data_ref, 'case_id': str(row.get('case_id')), 'dataset_role': dataset_role,
                    'side_ref': side_ref, 'judge_fp': judge_fp, 'pair_fp': pair_fp,
                    'round_fp': round_fingerprint(pair_fp, 'verify', nonce),
                    'round_kind': 'verify', 'round_no': idx, 'draw_nonce': nonce,
                    'origin_exp': exp_dir.name, 'origin_round_no': idx,
                    'verdict': int(vote), 'reply_sha256': reply_sha,
                    'trace_ref': str(row.get('trace_ref') or ''), 'source_status': source_status,
                    'created_at': time.time(),
                }


# ---------- 存储 ----------

def bank_path(instance_dir: Path) -> Path:
    return instance_dir / '.cache' / 'measurements.sqlite3'


def connect(instance_dir: Path, *, timeout: int = 10) -> sqlite3.Connection:
    path = bank_path(instance_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=timeout)  # finish 路径不得长等锁（评审 P2-5）
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript(SCHEMA)
    return db


_FIELDS = ('source_exp', 'source_line', 'side_fp', 'data_ref', 'case_id', 'dataset_role',
           'side_ref', 'judge_fp', 'pair_fp', 'round_fp', 'round_kind', 'round_no',
           'draw_nonce', 'origin_exp', 'origin_round_no', 'verdict', 'reply_sha256',
           'trace_ref', 'source_status', 'created_at')


def record_experiment(exp_dir: Path, db: sqlite3.Connection | None = None) -> int:
    """实验 finish 时批量事务写入。同键内容一致幂等忽略；不一致拒绝。
    单写者由文件锁强制（评审 P2-3）。返回新写入行数。"""
    from .storage import file_lock
    rows = list(iter_side_measurements(exp_dir))
    own = db is None
    if own:
        lock = file_lock(exp_dir.parent.parent / '.cache' / 'measurements.lock')
        lock.__enter__()
    db = db or connect(exp_dir.parent.parent)
    written = 0
    try:
        with db:
            # 批内 origin 一致性（同事务，评审 P2-1）
            origins = {}
            for row in rows:
                prev = origins.setdefault(row['round_fp'],
                                          (row['origin_exp'], row['origin_round_no']))
                if prev != (row['origin_exp'], row['origin_round_no']):
                    raise MeasurementConflict(f"round_fp 归属不一致: {row['round_fp']}")
            for row in rows:
                key = (row['source_exp'], row['source_line'], row['side_fp'], row['round_fp'])
                existing = db.execute(
                    'SELECT * FROM measurements WHERE source_exp=? AND source_line=? AND side_fp=? AND round_fp=?',
                    key).fetchone()
                if existing is None:
                    # 同一原始轮次的重放副本：round_fp+case+side 已入库且内容一致 →
                    # 跳过（纯重放不产生自有测量行，设计 §6.1）；不一致 → 冲突。
                    dup_row = db.execute(
                        'SELECT verdict, judge_fp, pair_fp, draw_nonce, origin_exp, origin_round_no '
                        'FROM measurements WHERE round_fp=? AND case_id=? AND side_fp=?',
                        (row['round_fp'], row['case_id'], row['side_fp'])).fetchone()
                    if dup_row is not None:
                        if (dup_row[0], dup_row[1], dup_row[2], dup_row[3]) != (
                                row['verdict'], row['judge_fp'], row['pair_fp'], row['draw_nonce']):
                            raise MeasurementConflict(
                                f"同轮次测量内容不一致: {row['round_fp']}")
                        continue  # 重放副本：不入库
                    db.execute(
                        'INSERT INTO measurements VALUES(' + ','.join('?' * len(_FIELDS)) + ')',
                        tuple(row[f] for f in _FIELDS))
                    written += 1
                else:
                    current = dict(zip(_FIELDS, existing))
                    for f in _FIELDS:
                        if f == 'created_at':
                            continue
                        if current[f] != row[f]:
                            raise MeasurementConflict(
                                f'测量库冲突 {key}: 字段 {f} 不一致（库 {current[f]!r} ≠ 新 {row[f]!r}）')
        return written
    finally:
        if own:
            db.close()
            lock.__exit__(None, None, None)


def record_experiment_safely(exp_dir: Path) -> int:
    """finish 路径调用：测量库是派生层，任何失败不得影响实验收尾。"""
    try:
        return record_experiment(exp_dir)
    except Exception as exc:  # noqa: BLE001 - 派生层故障降级为告警
        print(f'[measurements] 写入失败（可由 backfill 恢复）: {exp_dir.name}: {exc}')
        return 0


# ---------- 审计 ----------

def audit_row(instance_dir: Path, row) -> list[str]:
    """回读事实层来源行，重算四指纹并逐字段比对。返回问题清单（空=通过）。"""
    from . import experiment as _exp
    issues: list[str] = []
    stored = dict(zip(_FIELDS, row))
    exp_dir = instance_dir / 'experiments' / stored['source_exp']
    try:
        spec = _exp.spec_of(exp_dir)
    except ConfigError:
        return [f'来源实验不存在: {stored["source_exp"]}']
    lines = (exp_dir / 'cases.jsonl').read_text(encoding='utf-8').splitlines()
    if stored['source_line'] > len(lines):
        return [f'来源行越界: {stored["source_line"]}']
    raw = json.loads(lines[stored['source_line'] - 1])
    recomputed = {(m['round_fp'], m['side_fp']): m for m in iter_side_measurements(exp_dir)
                  if m['source_line'] == stored['source_line']}
    match = recomputed.get((stored['round_fp'], stored['side_fp']))
    if match is None:
        issues.append('来源行重算无此侧测量（case/status/verdict 不符）')
        return issues
    for f in ('side_fp', 'data_ref', 'case_id', 'dataset_role', 'side_ref', 'judge_fp',
              'pair_fp', 'round_fp', 'round_kind', 'draw_nonce', 'origin_exp',
              'origin_round_no', 'verdict'):
        if match[f] != stored[f]:
            issues.append(f'字段 {f} 不一致（库 {stored[f]!r} ≠ 重算 {match[f]!r}）')
    return issues


# ---------- 同对复用（P2：planner 重放，gates 不变） ----------

def _target_fingerprints(instance_dir: Path, spec: dict) -> tuple[str, str, str] | None:
    """由 spec + 版本目录计算 (pair_fp, judge_fp, dataset_role)；条件不齐返回 None。"""
    full = _target_fingerprints_full(instance_dir, spec)
    return None if full is None else (full[0], full[1], full[2])


def _target_fingerprints_full(instance_dir: Path, spec: dict):
    """同 _target_fingerprints，另返回 sides 映射 {side: (ref, side_fp)}。"""
    data_ref = str(spec.get('data_ref', ''))
    dataset_role = ('fixed_test' if spec.get('dataset') == 'fixed_test'
                    else ('pack:' + str(spec.get('pack_ref')) if spec.get('kind') == 'judge_eval'
                          else str(spec.get('purpose') or 'development')))
    kind = spec.get('kind')
    sides = {}
    for side in ('baseline', 'candidate'):
        ref = str(spec.get(side + '_ref', ''))
        folder = 'judges' if kind == 'judge_eval' else 'generators'
        cfg_path = instance_dir / folder / ref / 'config.json'
        if not cfg_path.is_file():
            return None
        side_config = json.loads(cfg_path.read_text())
        sides[side] = (ref, side_fingerprint(spec, side, side_config,
                                             instance_dir / folder / ref, data_ref, dataset_role)[1])
    if kind == 'judge_eval':
        judge_fp = judge_fingerprint(spec, {}, instance_dir)
    else:
        judge_ref = str(spec.get('judge_ref', ''))
        jdir = instance_dir / 'judges' / judge_ref
        jcfg_path = jdir / 'config.json'
        if not jcfg_path.is_file():
            return None
        judge_fp = judge_fingerprint(spec, json.loads(jcfg_path.read_text()), jdir)
    pair_fp = pair_fingerprint(sides['baseline'][1], sides['candidate'][1], judge_fp,
                               data_ref, dataset_role)
    return pair_fp, judge_fp, dataset_role, sides


def _migrate_trace(instance_dir: Path, source_exp: str, trace_ref, exp_dir: Path) -> None:
    """把来源实验的调用实录拷贝进本实验（独立作用域，避免与 cases.jsonl 写入规则混淆）。"""
    if not trace_ref or Path(str(trace_ref)).name != str(trace_ref):
        return
    src_trace = instance_dir / 'experiments' / source_exp / 'traces' / (str(trace_ref) + '.json')
    if not src_trace.is_file():
        return
    dst = exp_dir / 'traces' / src_trace.name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src_trace.read_bytes())


def plan_replay(instance_dir: Path, spec: dict, case_ids: list[str]) -> dict:
    """同对复用规划：按目标协议找出可从测量库整题重放的题。

    返回 {'replay': {case_id: (source_exp, source_line)}, 'run': [case_id]}。
    重放粒度为整题（缺的题由 runner 全量执行；LLM 请求缓存使生成/初判零花费，
    新花费只发生在补验轮）。固定验收与冒烟不规划（调用方保证）。
    """
    empty = {'replay': {}, 'run': list(case_ids)}
    full = _target_fingerprints_full(instance_dir, spec)
    if full is None:
        return empty
    pair_fp, judge_fp, dataset_role, target_sides = full
    need = case_ids and len(case_ids)
    db = connect(instance_dir)
    try:
        initials = {}
        for row in db.execute(
                'SELECT source_exp, case_id, side_fp, verdict FROM measurements '
                'WHERE pair_fp=? AND judge_fp=? AND dataset_role=? AND round_kind=? AND source_status=? '
                'ORDER BY created_at',
                (pair_fp, judge_fp, dataset_role, 'initial', 'finished')):
            initials.setdefault((row[0], row[1]), {})[row[2]] = row[3]
        verify_counts = {}
        for row in db.execute(
                'SELECT source_exp, case_id, side_fp, COUNT(*) FROM measurements '
                'WHERE pair_fp=? AND judge_fp=? AND dataset_role=? AND round_kind=? AND source_status=? '
                'GROUP BY source_exp, case_id, side_fp ORDER BY MIN(created_at)',
                (pair_fp, judge_fp, dataset_role, 'verify', 'finished')):
            verify_counts[(row[0], row[1], row[2])] = row[3]
    finally:
        db.close()
    extra = int((spec.get('protocol') or {}).get('flip_extra_rounds') or 0)
    # 侧角色校验（评审 P1）：pair_fp 是无序对，baseline/candidate 互换的实验
    # 对指纹相同；整行拷贝必须角色一致，否则判决反向。按来源 spec 逐个核对。
    source_specs: dict[str, dict] = {}
    role_ok: dict[str, bool] = {}
    for source_exp in {s for s, _ in initials}:
        src_spec_path = instance_dir / 'experiments' / source_exp / 'spec.json'
        if not src_spec_path.is_file():
            role_ok[source_exp] = False
            continue
        src_spec = json.loads(src_spec_path.read_text())
        src_full = _target_fingerprints_full(instance_dir, src_spec)
        source_specs[source_exp] = src_spec
        role_ok[source_exp] = bool(
            src_full and target_sides and
            src_full[3].get('baseline') == target_sides.get('baseline') and
            src_full[3].get('candidate') == target_sides.get('candidate'))
    eligible: dict[str, tuple[str, int]] = {}
    wanted = {str(c) for c in case_ids}
    for (source_exp, case_id), side_map in initials.items():
        if case_id not in wanted or len(side_map) != 2 or not role_ok.get(source_exp):
            continue
        values = list(side_map.values())
        if values[0] == values[1]:
            complete = True  # 初测一致：协议不产生补验轮
        else:
            complete = all(verify_counts.get((source_exp, case_id, side_fp)) == extra
                           for side_fp in side_map)
        if complete and case_id not in eligible:
            eligible[case_id] = source_exp  # 先到的来源优先（最早回填序）
    if not eligible:
        return empty
    # 定位每个来源实验中各题最终记录行号（最后一行 status=ok）
    final_line: dict[tuple[str, str], int] = {}
    for source_exp in {s for s in eligible.values()}:
        exp_dir = instance_dir / 'experiments' / source_exp
        cases_path = exp_dir / 'cases.jsonl'
        if not cases_path.is_file():
            continue
        last: dict[str, int] = {}
        for line_no, line in enumerate(cases_path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('status') == 'ok' and str(row.get('case_id')) in wanted:
                last[str(row['case_id'])] = line_no
        for cid, line_no in last.items():
            if eligible.get(cid) == source_exp:
                final_line[(source_exp, cid)] = line_no
    replay = {cid: (src, final_line[(src, cid)])
              for cid, src in eligible.items() if (src, cid) in final_line}
    return {'replay': replay, 'run': [c for c in case_ids if str(c) not in replay]}


def apply_replay(exp_dir: Path) -> dict:
    """实验开跑前执行重放：把同对已完成题的最终记录逐字节写进 cases.jsonl。

    纪律（设计 §6.1）：整题拷贝来源原文行（天然全字段一致）；写 replay_sources.json
    审计线索（来源实验+行号映射）；纯重放不产生新测量行（来源指向原始实验，
    银行已收录）。返回 {'replayed': n, 'run': m}。
    """
    from . import datasets as _datasets
    spec = experiment.spec_of(exp_dir)
    result = {'replayed': 0, 'run': 0}
    if (spec.get('smoke') or spec.get('dataset') == 'fixed_test'
            or spec.get('no_replay') or spec.get('kind') not in
            (experiment.KIND_GEN_AB, experiment.KIND_JUDGE_EVAL)):
        return result
    if spec.get('kind') == experiment.KIND_JUDGE_EVAL:
        pack = json.loads((exp_dir.parent.parent / 'judge_eval' / spec['pack_ref'] / 'pack.json').read_text())
        cases = pack.get('rows') or []
    else:
        cases = _datasets.rows_for(spec)
    plan = plan_replay(exp_dir.parent.parent, spec, [str(c['case_id']) for c in cases])
    replay = dict(plan['replay'])
    result['run'] = len(plan['run'])
    cases_path = exp_dir / 'cases.jsonl'
    # 目标实验已有记录的题不重放（失败重试去重，评审 P2-3）
    if cases_path.exists():
        for line in cases_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    replay.pop(str(json.loads(line).get('case_id')), None)
                except ValueError:
                    continue
    result['run'] += len(plan['replay']) - len(replay)
    if not replay:
        return result
    if experiment.state_of(exp_dir).get('status') == 'finished':
        raise MeasurementConflict('已完成实验不能重放追加')
    full = _target_fingerprints_full(exp_dir.parent.parent, spec)
    side_fps = {fp: side for side, (ref, fp) in full[3].items()} if full else {}
    db = connect(exp_dir.parent.parent)
    rows_out: list[dict] = []
    manifest: dict[str, dict] = {}
    dropped_serialization = 0
    try:
        for case_id, (source_exp, line_no) in sorted(replay.items()):
            src_lines = (exp_dir.parent.parent / 'experiments' / source_exp / 'cases.jsonl'
                         ).read_text(encoding='utf-8').splitlines()
            if line_no > len(src_lines):
                raise MeasurementConflict(f'重放来源行越界: {source_exp}:{line_no}')
            raw = src_lines[line_no - 1].strip()
            row = json.loads(raw)
            if row.get('status') != 'ok' or str(row.get('case_id')) != case_id:
                raise MeasurementConflict(f'重放来源行内容不符: {source_exp}:{line_no}')
            # 逐字节一致预检：writer 同源序列化（评审 P2），还原不了原文行的
            # 题放弃重放、交给 runner 执行（评审 P1-2）。
            from .record_contract import serialize_case
            if serialize_case(row) != raw:
                dropped_serialization += 1
                result['run'] += 1
                continue
            # 注入来源 draw_nonce：保证重放实验解析出的 round_fp 与来源一致，
            # 测量库按同轮副本跳过（不产生重复测量行，设计 §6.1）。
            for side_fp, side in side_fps.items():
                for rnd, nonce in db.execute(
                        'SELECT round_no, draw_nonce FROM measurements '
                        'WHERE source_exp=? AND case_id=? AND side_fp=? AND source_line=?',
                        (source_exp, case_id, side_fp, line_no)):
                    row['draw_nonce' if rnd == 0 else f'draw_nonce_r{rnd}'] = nonce
            # 调用实录随迁：重放行引用的 trace 属于来源实验，拷贝进本实验，
            # 保持“实验是自足审计单元”（否则 dashboard 的 trace 链接 404）。
            _migrate_trace(exp_dir.parent.parent, source_exp, row.get('trace_ref'), exp_dir)
            rows_out.append(row)
            manifest[case_id] = {'source_exp': source_exp, 'source_line': line_no}
    finally:
        db.close()
    from . import record_contract
    with record_contract.writer(exp_dir) as append:
        for row in rows_out:
            append(row)
    write_json(exp_dir / 'replay_sources.json', {
        'pair_replay': True, 'cases': manifest,
        'flip_extra_rounds': (spec.get('protocol') or {}).get('flip_extra_rounds'),
        'dropped_serialization_mismatch': dropped_serialization,
        'note': '测量字段与来源行逐字节一致（writer 同源序列化预检），附加 draw_nonce* 溯源字段（取值=来源测量库，保证同轮指纹一致、不产生重复测量行）；判决以本实验 cases.jsonl 为准并可回溯来源。',
    })
    result['replayed'] = len(rows_out)
    return result


def apply_replay_safely(exp_dir: Path) -> dict:
    """runner 入口调用：重放失败不得阻断实验执行（退回全量运行）。"""
    try:
        return apply_replay(exp_dir)
    except Exception as exc:  # noqa: BLE001 - 派生层故障降级为全量运行
        print(f'[measurements] 重放失败（已退回全量运行）: {exp_dir.name}: {exc}')
        return {'replayed': 0, 'run': -1}
