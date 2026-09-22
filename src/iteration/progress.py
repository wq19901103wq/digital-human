"""实验的阶段与调用记录；不参与评测或晋升判断。"""
from __future__ import annotations

import os
import time
from threading import RLock
from contextlib import nullcontext
from functools import wraps

from ..tracing import CaseTrace
from . import experiment, protocol
from .storage import write_json, file_lock


def is_active(progress: dict) -> bool:
    from .task_state import active
    return active(progress)


class RunProgress:
    def __init__(self, exp_dir):
        self.lock = RLock()
        self.exp_dir = exp_dir
        self.spec = experiment.spec_of(exp_dir)
        self.records, _ = protocol.summarize_final_records(exp_dir / "cases.jsonl")
        now = time.time()
        self.value = {"status": "running", "pid": os.getpid(), "started_at": now,
                      "updated_at": now, "phase_started_at": now, "phase": "preparing",
                      "case_index": 0, "total": self.spec.get("smoke_limit") or 0,
                      "branch": "", "round": 0, "timings": {}}
        from .task_state import process_start
        self.value["process_start"] = process_start(os.getpid())
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
        write_json(self.exp_dir / "state.json", state)

    def stage(self, phase, **fields):
        with self.lock:
            self._timing()
            self.value.update(phase=phase, phase_started_at=time.time(), **fields)
            self.save()

    def completed(self, row):
        with self.lock:
            self.records[str(row["case_id"])] = row
            if self.value.get('workers', 1) > 1:
                self.save()
            else:
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


class ParallelCaseProgress:
    """每题独立 trace 与回调；共享状态写入由父进度的锁串行化。"""
    def __init__(self, parent, index):
        self.parent, self.index = parent, index
        self.trace = None
        self.fields = {'case_index': index, 'phase': 'preparing', 'branch': '', 'round': 0}

    def stage(self, phase, **fields):
        self.fields.update(phase=phase, phase_started_at=time.time(), **fields)
        with self.parent.lock:
            self.parent.value.setdefault('active_cases', {})[str(self.index)] = dict(self.fields)
            self.parent.save()

    def start_case(self, case):
        self.trace = CaseTrace(self.parent.exp_dir, case, self.parent.spec)
        self.stage('preparing', trace_ref=self.trace.ref)

    def _operation(self, kind, branch, round, inputs):
        return self.trace.operation(kind, branch, round, inputs)

    def generate(self, generator, case, branch, forced_reply=None, round=0):
        return RunProgress.generate(self, generator, case, branch, forced_reply, round)

    def judge(self, scorer, case, replies, branch, round=0):
        self.stage('judging', branch=branch, round=round)
        previous = getattr(scorer, 'on_progress', None)
        scorer.on_progress = self.stage
        try:
            with self.trace.operation('judge', branch, round, {'candidate_replies': replies, 'mode': getattr(scorer, 'mode', None)}) as op:
                result = scorer.is_ai(case, replies)
                op['result'] = {'identified_ai': bool(result)}
                return result
        finally:
            scorer.on_progress = previous

    def finish_case(self, row):
        row['trace_ref'] = self.trace.ref
        self.trace.finish(row.get('status', 'ok'))
        with self.parent.lock:
            self.parent.value.get('active_cases', {}).pop(str(self.index), None)
            self.parent.save()


def track_run(function):
    @wraps(function)
    def wrapped(exp_dir, *args, **kwargs):
        from .control import job
        from .record_contract import require_registered
        spec, _ = require_registered(exp_dir)
        if spec.get('comparison'):
            from ..config import ConfigError
            raise ConfigError('专项比较记录不能由普通 Gen/Judge 执行器续跑')
        with file_lock(exp_dir / ".run.lock", blocking=False), job(exp_dir):
            return run_locked(exp_dir, *args, **kwargs)

    def run_locked(exp_dir, *args, **kwargs):
        if experiment.state_of(exp_dir).get("status") == "finished":
            return function(exp_dir, *args, _progress=None, **kwargs)
        from .journal import recover
        recover(exp_dir / "cases.jsonl")
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
