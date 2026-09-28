"""本机实验缓存：内容寻址、独立采样轮次、跨进程去重和来源索引。

仅在 CaseTrace 的作用域内启用；在线聊天和未追踪调用保持独立请求。
版本/实验 ID 用于查询，兼容性由实际输入、运行配置和实现指纹决定。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from .iteration.storage import file_lock

_scope = ContextVar("result_cache_scope", default=None)
_validator = ContextVar("result_cache_validator", default=None)

# Reviewed repository migration: rpa.py -> corrected.py changed module/class
# names and diagnostic text only. Match the complete bytes, so any subsequent
# implementation edit receives a new identity rather than reusing this alias.
# 2026-09-23 reviewed change (context_render=speaker_roster_v1): adds config-
# gated blind rendering helpers; the default legacy rendering is byte-identical
# and CodexJudgeClient is untouched, so identities pinned at g-0029..g-0032
# ranker training remain valid. Aliased by exact bytes; any further edit gets
# a fresh identity.
_CODE_DIGEST_ALIASES = {
    # Reviewed explicit source_participants_v1 rendering. Legacy messages and
    # output parsing are unchanged; new messages retain distinct cache keys.
    "5969906163561bff564bc06cd1f8a0e6a2556b1dceb4b07e1860989b3ce96783":
        "79a7643e7f65a0094bb70addf11400229e65d2b425385d3b1e0787c68f63b780",
    # Explicit Gen background only adds messages for the new policy. Legacy
    # prompts and generation semantics retain their existing cache identity.
    "e8949029199ee75ae2cacde29cdf7ab6c5debe499a3c2b2cec2847571d395292":
        "79a7643e7f65a0094bb70addf11400229e65d2b425385d3b1e0787c68f63b780",
    "ec42a97621e36437caf8e5f2124fd97482e53b4643bfd558d9dfd4fcc8a894be":
        "8b5b7a9e76d683b87d13f6d3e2c0ee79eb601643690c0c63195b1ab5df1e9071",
    "06fb0d4a72259580aff98f1c08329899ef6a584f682ca23b83b88f203b79316f":
        "8b5b7a9e76d683b87d13f6d3e2c0ee79eb601643690c0c63195b1ab5df1e9071",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_digest(*paths):
    hashes = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
    return digest([_CODE_DIGEST_ALIASES.get(value, value) for value in hashes])


def client_identity(client):
    method = getattr(client, "cache_identity", None)
    return method() if callable(method) else None  # 外部/测试客户端须明确声明可缓存


@contextmanager
def validation(check):
    """传输层只保存通过调用方语义校验的正文；格式重试不能命中坏响应。"""
    token = _validator.set(check)
    try:
        yield
    finally:
        _validator.reset(token)


def response_valid(value):
    check = _validator.get()
    try:
        return check(value) is not False if check else bool(value.strip())
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        return False


def trace_context(exp_dir, trace, operation=None, index=None):
    versions = trace["versions"]
    return {"experiment_id": Path(exp_dir).name, "experiment_path": str(Path(exp_dir).resolve()),
            "case_id": trace["case_id"], "data_ref": versions.get("data_ref"),
            "dataset": versions.get("dataset"), "versions": versions,
            "trace_ref": trace["trace_ref"], "operation_index": index,
            "branch": (operation or {}).get("branch"), "round": (operation or {}).get("round", 0),
            **(operation or {}).get("cache_policy", {})}


@contextmanager
def scope(exp_dir, trace, operation, index, policy=None):
    exp_dir = Path(exp_dir).resolve()
    # 实例路径来自 trace 的拥有者，不读可被另一线程切换的全局实例指针。
    context = trace_context(exp_dir, trace, operation, index)
    policy = policy or {}
    context.update(sample_epoch=policy.get("sample_epoch", "shared-v1"), epoch=os.getenv("DH_CACHE_EPOCH", "1"))
    operation["cache_policy"] = {"enabled": policy.get("enabled", True),
                                 "sample_epoch": context["sample_epoch"], "epoch": context["epoch"]}
    token = _scope.set({"store": get_store(exp_dir.parent.parent / ".cache"), "context": context,
                        "enabled": policy.get("enabled", True),
                        "sample_epoch": policy.get("sample_epoch", "shared-v1")})
    try:
        yield
    finally:
        _scope.reset(token)


def memo(layer, identity, produce, *, valid=None):
    from .iteration.control import check
    check()
    current = _scope.get()
    if identity is None or not current or not current["enabled"]:
        return produce()
    context = current["context"]
    key = memo_key(layer, identity, context)
    return current["store"].remember(key, layer, identity, context, produce, valid=valid)


def memo_key(layer, identity, context):
    partition = ("development" if context["dataset"] == "development"
                 else f'{context["dataset"]}:{context["experiment_id"]}')
    # 保留旧键格式及独立轮次；兼容复用不能跨固定实验或采样 epoch。
    return digest({"schema": 1, "layer": layer, "identity": identity, "partition": partition,
                   "case_id": context["case_id"], "round": context["round"],
                   "sample_epoch": context["sample_epoch"], "epoch": context["epoch"]})


def blind_input(identity):
    """仅 LR/GBDT 盲序的来源定位可变；其余输入（包括答案正文）仍精确匹配。"""
    if (not isinstance(identity, dict) or set(identity) != {"case", "candidate_replies", "client"}
            or not isinstance(identity["case"], dict)):
        return None
    return {**identity, "case": {k: v for k, v in identity["case"].items() if k != "generation_trace_ref"}}


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "results.sqlite3"
        # WAL 切换本身不遵循 busy_timeout；首次多进程建库也需要互斥。
        with file_lock(self.root / "init.lock"), self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS entries (
                    key TEXT PRIMARY KEY, layer TEXT NOT NULL, identity TEXT NOT NULL,
                    value TEXT NOT NULL, value_sha TEXT NOT NULL, origin TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS uses (
                    id INTEGER PRIMARY KEY, key TEXT NOT NULL, layer TEXT NOT NULL,
                    hit INTEGER NOT NULL, context TEXT NOT NULL, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS uses_key ON uses(key);
                CREATE INDEX IF NOT EXISTS blind_order_case
                    ON entries(json_extract(identity, '$.case.case_id')) WHERE layer='blind_order';
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY, experiment_id TEXT, case_id TEXT, data_ref TEXT,
                    dataset TEXT, layer TEXT, status TEXT, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS history_case ON history(case_id, experiment_id);
                CREATE INDEX IF NOT EXISTS history_data ON history(data_ref);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT * FROM entries WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        row = dict(row)
        if hashlib.sha256(row["value"].encode()).hexdigest() != row["value_sha"]:
            raise ValueError(f"缓存内容校验失败：{key}")
        for name in ("identity", "value", "origin"):
            row[name] = json.loads(row[name])
        return row

    def remember(self, key, layer, identity, context, produce, *, valid=None):
        # 每个键一把锁，等待者重查缓存；不把网络调用放进 SQLite 写事务。
        with file_lock(self.root / "locks" / (key + ".lock")):
            row = self.get(key)
            if row is None and layer == "blind_order":
                source = self.compatible_blind_order(identity, context)
                if source is not None:
                    self.save_blind_alias(key, identity, source)
                    row = self.get(key)
            hit = row is not None and (valid is None or valid(row["value"]))
            if hit:
                value = row["value"]
                from . import tracing
                tracing.note("cache_hit", {"layer": layer, "key": key, "origin": row["origin"]})
            else:
                from . import tracing
                tracing.note("cache_lookup", {"layer": layer, "key": key, "hit": False})
                value = produce()  # 异常和中断不写成功条目
                if valid is not None and not valid(value):
                    return value
                blob = canonical(value)
                with self.connect() as db:
                    db.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?)",
                               (key, layer, canonical(identity), blob, hashlib.sha256(blob.encode()).hexdigest(),
                                canonical(context), time.time()))
            with self.connect() as db:
                db.execute("INSERT INTO uses(key,layer,hit,context,created) VALUES (?,?,?,?,?)",
                           (key, layer, int(hit), canonical(context), time.time()))
            return value

    def blind_orders(self, case_id):
        with self.connect() as db:
            keys = db.execute("SELECT key FROM entries WHERE layer='blind_order' "
                              "AND json_extract(identity, '$.case.case_id')=? ORDER BY created,key",
                              (case_id,)).fetchall()
        return [self.get(row["key"]) for row in keys]

    def compatible_blind_order(self, identity, context):
        expected = blind_input(identity)
        if expected is None:
            return None
        for row in self.blind_orders(identity["case"].get("case_id")):
            if (type(row["value"]) is bool and blind_input(row["identity"]) == expected
                    and memo_key("blind_order", row["identity"], context) == row["key"]):
                return row  # 最早的同条件抽样；不读预测、标签或胜负来选择。
        return None

    def save_blind_alias(self, key, identity, source):
        """调用方持有目标键锁；不覆盖已经分配的盲序，冻结旧进程也可直接读取。"""
        origin = {**source["origin"], "cache_reuse": {
            "source_key": source["key"], "reason": "generation_trace_ref_only"}}
        blob = canonical(source["value"])
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO entries VALUES (?,?,?,?,?,?,?)",
                       (key, "blind_order", canonical(identity), blob,
                        hashlib.sha256(blob.encode()).hexdigest(), canonical(origin), time.time()))

    def reuse_pack_blind_orders(self, rows):
        """为重建包填充旧键别名。保持来源的分区/轮次/epoch，不创建任何新抽样。"""
        counts = {"cases": len(rows), "reused": 0, "existing": 0, "without_source": 0}
        seen = set()
        for case in rows:
            found = False
            for source in self.blind_orders(case["case_id"]):
                identity = source["identity"]
                if (blind_input(identity) is None or type(source["value"]) is not bool
                        or identity["candidate_replies"] != case["ai_replies"]):
                    continue
                target = {**identity, "case": case}
                if blind_input(target) != blind_input(identity):
                    continue
                context = source["origin"]
                if memo_key("blind_order", identity, context) != source["key"]:
                    continue
                found = True
                key = memo_key("blind_order", target, context)
                if key in seen:
                    continue
                seen.add(key)
                with file_lock(self.root / "locks" / (key + ".lock")):
                    if self.get(key) is not None:
                        counts["existing"] += 1
                    else:
                        self.save_blind_alias(key, target, source)
                        counts["reused"] += 1
            counts["without_source"] += int(not found)
        return counts

    def record(self, source_id, context, layer, status, payload):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO history VALUES (?,?,?,?,?,?,?,?)",
                       (digest(source_id), context.get("experiment_id"), context.get("case_id"),
                        context.get("data_ref"), context.get("dataset"), layer, status,
                        canonical({**context, **payload})))

    def history(self, *, experiment_id=None, case_id=None, data_ref=None, version=None, limit=50):
        clauses, params = [], []
        for name, value in (("experiment_id", experiment_id), ("case_id", case_id), ("data_ref", data_ref)):
            if value is not None:
                clauses.append(name + "=?")
                params.append(value)
        if version is not None:
            clauses.append("EXISTS (SELECT 1 FROM json_each(history.payload, '$.versions') WHERE value=?)")
            params.append(version)
        sql = "SELECT layer,status,payload FROM history"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self.connect() as db:
            rows = db.execute(sql + " ORDER BY rowid DESC LIMIT ?", (*params, max(1, limit))).fetchall()
        return [{"layer": r["layer"], "status": r["status"], **json.loads(r["payload"])} for r in rows]

    def stats(self):
        with self.connect() as db:
            entries = dict(db.execute("SELECT layer,COUNT(*) FROM entries GROUP BY layer").fetchall())
            uses = [dict(r) for r in db.execute("SELECT layer,SUM(hit) AS hits,SUM(1-hit) AS writes FROM uses GROUP BY layer")]
            history = db.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        return {"path": str(self.path), "entries": entries, "uses": uses, "history_records": history}

    def entries(self, *, layer=None, model=None, experiment_id=None, case_id=None, limit=20):
        clauses, params = [], []
        for expression, value in (("e.layer", layer), ("json_extract(e.identity,'$.client.model')", model),
                                  ("json_extract(u.context,'$.experiment_id')", experiment_id),
                                  ("json_extract(u.context,'$.case_id')", case_id)):
            if value is not None:
                clauses.append(expression + "=?")
                params.append(value)
        sql = "SELECT DISTINCT e.key,e.layer,e.origin,e.created FROM entries e JOIN uses u ON e.key=u.key"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self.connect() as db:
            rows = db.execute(sql + " ORDER BY e.created DESC LIMIT ?", (*params, max(1, limit))).fetchall()
        return [{**dict(r), "origin": json.loads(r["origin"])} for r in rows]


@lru_cache(maxsize=32)
def get_store(root):
    return Store(root)


def record_operation(exp_dir, trace, op, index):
    context = trace_context(exp_dir, trace, op, index)
    # 原始正文继续由 trace 保存，索引存输入/结果及命中来源，避免复制多份提示词。
    get_store(Path(exp_dir).resolve().parent.parent / ".cache").record(
        [str(Path(exp_dir).resolve()), trace["trace_ref"], index], context, op["kind"], op["status"],
        {"input": op.get("input"), "result": op.get("result"), "error": op.get("error"),
         "cache_hits": [e["data"] for e in op.get("events", []) if e["kind"] == "cache_hit"],
         "cache_lookups": [e["data"] for e in op.get("events", []) if e["kind"] == "cache_lookup"],
         "source_path": str(Path(exp_dir).resolve() / "traces" / (trace["trace_ref"] + ".json"))})


def import_history(instance_dir):
    """幂等索引旧实录/逐题结果；缺失完整调用身份的旧记录不伪装成可复用缓存。"""
    instance_dir = Path(instance_dir).resolve()
    store = get_store(instance_dir / ".cache")
    counts = {"cases": 0, "operations": 0, "experiments": 0, "skipped_incomplete": 0}
    for path in sorted((instance_dir / "experiments").glob("*/spec.json")):
        spec = json.loads(path.read_text())
        context = {"experiment_id": path.parent.name, "data_ref": spec.get("data_ref"),
                   "dataset": spec.get("dataset"), "versions": {k: v for k, v in spec.items() if k.endswith("_ref")}}
        state_path = path.with_name("state.json")
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        store.record(str(path), context, "experiment", state.get("status", "unknown"),
                     {"result": state.get("metrics"), "verdict": state.get("verdict"), "source_path": str(path)})
        counts["experiments"] += 1
        records = path.with_name("cases.jsonl")
        if records.exists():
            for index, line in enumerate(records.read_text().splitlines()):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    counts["skipped_incomplete"] += 1
                    continue
                store.record([str(records), index], {**context, "case_id": str(row["case_id"])}, "case",
                             row.get("status", "unknown"), {"result": row, "source_path": str(records), "line": index + 1})
                counts["cases"] += 1
    for folder in ("experiments", "judge_eval"):
        for path in sorted((instance_dir / folder).glob("*/traces/*.json")):
            try:
                trace = json.loads(path.read_text())
            except ValueError:
                counts["skipped_incomplete"] += 1
                continue
            for index, op in enumerate(trace.get("operations", [])):
                record_operation(path.parent.parent, trace, op, index)
                counts["operations"] += 1
    return counts
