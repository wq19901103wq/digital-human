"""逐题调用实录：输入先落盘，输出随后更新；不参与评测判断。"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_sink = ContextVar("case_trace_sink", default=None)


@contextmanager
def step(kind: str, request: dict):
    """传输层在真正发送之前调用；不记录认证头或模型内部思考。"""
    event = {"kind": kind, "request": request, "started_at": time.time(), "status": "running"}
    sink = _sink.get()
    if sink:
        sink(event)
    started = time.monotonic()
    try:
        yield event
    except BaseException as exc:
        event.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    else:
        event["status"] = "ok"
    finally:
        event.update(ended_at=time.time(), elapsed_ms=round((time.monotonic() - started) * 1000, 3))
        if sink:
            sink(None)


def note(kind: str, data: dict):
    sink = _sink.get()
    if sink:
        sink({"kind": kind, "data": data, "status": "ok", "started_at": time.time()})


def read(exp_dir: Path, ref: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{32}", ref):
        raise ValueError("非法调用记录 ID")
    path = exp_dir / "traces" / (ref + ".json")
    if not path.resolve().is_relative_to(exp_dir.resolve()):
        raise ValueError("调用记录路径越界")
    return json.loads(path.read_text(encoding="utf-8"))


class CaseTrace:
    def __init__(self, exp_dir: Path, case: dict, spec: dict):
        self.exp_dir = exp_dir
        self.cache_policy = spec.get("cache", {})
        self.ref = uuid.uuid4().hex
        self.path = exp_dir / "traces" / (self.ref + ".json")
        self.value = {"schema_version": 1, "trace_ref": self.ref, "case_id": str(case["case_id"]),
                      "case": case, "versions": {k: spec.get(k) for k in
                          ("kind", "dataset", "data_ref", "baseline_ref", "candidate_ref", "judge_ref", "pack_ref")},
                      "started_at": time.time(), "status": "running", "operations": []}
        self.save()

    def save(self):
        self.value["updated_at"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 原子替换让网页始终读到完整快照；临时文件默认仅本人可读写。
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as f:
            temp = Path(f.name)
            try:
                json.dump(self.value, f, ensure_ascii=False)
                f.flush()
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        temp.replace(self.path)

    @contextmanager
    def operation(self, kind: str, branch: str, round_index: int, inputs: dict):
        op = {"kind": kind, "branch": branch, "round": round_index, "input": inputs,
              "started_at": time.time(), "status": "running", "events": []}
        self.value["operations"].append(op)
        self.save()

        def capture(event):
            if event is not None:
                op["events"].append(event)
            self.save()

        token = _sink.set(capture)
        started = time.monotonic()
        try:
            from . import cache
            with cache.scope(self.exp_dir, self.value, op, len(self.value["operations"]) - 1, self.cache_policy):
                yield op
        except BaseException as exc:
            op.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        else:
            op["status"] = "ok"
        finally:
            _sink.reset(token)
            op.update(ended_at=time.time(), elapsed_ms=round((time.monotonic() - started) * 1000, 3))
            self.save()
            from . import cache
            try:
                cache.record_operation(self.exp_dir, self.value, op, len(self.value["operations"]) - 1)
            except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
                # 实录已经落盘，索引可重建；查询索引故障不能覆盖原始调用异常。
                logging.getLogger(__name__).warning("调用历史索引写入失败（可重新 index）：%s", exc)

    def finish(self, status: str):
        self.value.update(status=status, ended_at=time.time())
        self.save()
