"""串行实验的阶段与调用记录；不参与评测或晋升判断。"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext
from functools import wraps

from ..tracing import CaseTrace
from . import experiment, protocol


def is_active(progress: dict) -> bool:
    if progress.get("status") != "running" or not progress.get("pid"):
        return False
    try:
        os.kill(int(progress["pid"]), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


class RunProgress:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.spec = experiment.spec_of(exp_dir)
        self.records, _ = protocol.summarize_final_records(exp_dir / "cases.jsonl")
        now = time.time()
        self.value = {"status": "running", "pid": os.getpid(), "started_at": now,
                      "updated_at": now, "phase_started_at": now, "phase": "preparing",
                      "case_index": 0, "total": self.spec.get("smoke_limit") or 0,
                      "branch": "", "round": 0, "timings": {}}
        self._timer = time.monotonic()
        self.trace = None
        self.previous_trace = (experiment.state_of(exp_dir).get("progress") or {}).get("trace_ref")

    def _timing(self):
        phase = self.value["phase"]
        elapsed = time.monotonic() - self._timer
        self.value["timings"][phase] = self.value["timings"].get(phase, 0) + elapsed
        self._timer = time.monotonic()

    def save(self):
        judge = self.spec.get("kind") == "judge_eval"
        b_key, c_key = ("baseline_correct", "candidate_correct") if judge else ("identified_baseline", "identified_candidate")
        valid = [r for r in self.records.values() if r.get("status") != "failed" and b_key in r and c_key in r]
        self.value["counts"] = {
            "pairs": len(valid), "failures": sum(r.get("status") == "failed" for r in self.records.values()),
            "identified_baseline": sum(bool(r[b_key]) for r in valid),
            "identified_candidate": sum(bool(r[c_key]) for r in valid),
        }
        self.value["updated_at"] = time.time()
        state = experiment.state_of(self.exp_dir)
        state["progress"] = self.value
        experiment._write_json(self.exp_dir / "state.json", state)

    def stage(self, phase, **fields):
        self._timing()
        self.value.update(phase=phase, phase_started_at=time.time(), **fields)
        self.save()

    def completed(self, row):
        self.records[str(row["case_id"])] = row
        self.stage("checkpoint")

    def start_case(self, case):
        self.trace = CaseTrace(self.exp_dir, case, self.spec)
        if self.previous_trace:
            from ..tracing import read
            previous = read(self.exp_dir, self.previous_trace)
            if previous['case_id'] == str(case['case_id']):
                self.trace.value['previous_trace_ref'] = self.previous_trace
                self.trace.save()
            self.previous_trace = None
        self.value["trace_ref"] = self.trace.ref
        self.save()

    def finish_case(self, row):
        if self.trace:
            row["trace_ref"] = self.trace.ref
            self.trace.finish(row.get("status", "ok"))

    def _operation(self, kind, branch, round, inputs):
        return self.trace.operation(kind, branch, round, inputs) if self.trace else nullcontext({})

    def generate(self, generator, case, branch, forced_reply=None, round=0):
        self.stage("generating", branch=branch, round=round)
        with self._operation("generation", branch, round, {"forced_reply": True if forced_reply is None else forced_reply}) as op:
            result = generator.generate(case) if forced_reply is None else generator.generate(case, forced_reply=forced_reply)
            op["result"] = result
            return result

    def judge(self, scorer, case, replies, branch, round=0):
        self.stage("judging", branch=branch, round=round)
        previous = getattr(scorer, "on_progress", None)
        scorer.on_progress = self.stage
        try:
            with self._operation("judge", branch, round, {"candidate_replies": replies, "mode": getattr(scorer, "mode", None)}) as op:
                result = scorer.is_ai(case, replies)
                op["result"] = {"identified_ai": bool(result)}
                return result
        finally:
            scorer.on_progress = previous


def track_run(function):
    @wraps(function)
    def wrapped(exp_dir, *args, **kwargs):
        if experiment.state_of(exp_dir).get("status") == "finished":
            return function(exp_dir, *args, _progress=None, **kwargs)
        progress = RunProgress(exp_dir)
        progress.save()
        try:
            return function(exp_dir, *args, _progress=progress, **kwargs)
        except BaseException:
            if progress.trace and progress.trace.value["status"] == "running":
                progress.trace.finish("interrupted")
            progress.stage("interrupted", status="stopped", ended_at=time.time())
            raise
    return wrapped
