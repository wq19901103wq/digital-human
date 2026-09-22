"""单实例调度器，多个独立子进程；只管理显式提交的分支任务。"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from ..config import ConfigError
from . import branches, control, jobs, runtime, task_state, training, versions
from .storage import write_json, file_lock, read_json


def _load(path):
    return read_json(path, default={})


def _active(job, metadata):
    return jobs.snapshot(job, metadata)['active']


class Scheduler:
    def __init__(self, *, max_experiments=2, workers=4, max_attempts=3, retry_delay=30, timeout_seconds=None):
        if not 1 <= max_experiments <= 16 or not 1 <= workers <= 16 or max_attempts < 1 or retry_delay < 0:
            raise ConfigError("实验并发数和题目并发数须为 1–16，最大尝试次数至少 1，重试间隔非负")
        self.max_experiments = max_experiments
        self.workers = workers
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.children = {}
        from .pack_transport import timeout_value
        self.timeout_seconds = timeout_value(timeout_seconds) if timeout_seconds is not None else None

    def _reap(self):
        for key, (job, process) in list(self.children.items()):
            result = process.poll()
            if result is None:
                continue
            path = jobs.metadata_path(job)
            metadata = _load(path)
            metadata.update(exit_code=result, exited_at=time.time())
            state = _load(jobs.directory(job) / 'state.json')
            if state.get('status') == 'needs_attention':
                metadata.update(status='needs_attention', reason=state.get('reason', state.get('phase')))
            write_json(path, metadata)
            del self.children[key]

    def tick(self) -> list[dict]:
        with file_lock(versions.PRIVATE / ".scheduler.lock", blocking=False):
            return self._tick()

    def _tick(self) -> list[dict]:
        self._reap()
        pending = branches.advance() + training.pending()
        # Explicit retries also cover standalone packs and direct experiments.
        for path in (versions.PRIVATE / 'scheduling').glob('*.json'):
            metadata = _load(path)
            if metadata.get('job') and metadata.get('status') in {'queued', 'submitted', 'launch_failed'}:
                if not jobs.snapshot(metadata['job'], metadata)['finished']:
                    pending.append(metadata['job'])
        pending = list({(job['kind'], job['id']): job for job in pending}.values())
        active = set()
        # 计算本实例所有在途任务，包括旧入口启动的实验；不会接管它们的晋升。
        for kind, folder in jobs.FOLDERS.items():
            for directory in (versions.PRIVATE / folder).glob("*"):
                if not directory.is_dir() or directory.name.startswith('.'):
                    continue
                job = {"kind": kind, "id": directory.name}
                metadata = _load(jobs.metadata_path(job))
                if _active(job, metadata):
                    active.add((kind, directory.name))
                    try:
                        control.check(directory)
                    except control.StopRequested as exc:
                        # Only signal scheduler-owned process groups with a matching start identity.
                        owner = {**metadata, 'status': 'running'}
                        if metadata.get('process_start') and task_state.active(owner):
                            try:
                                os.killpg(metadata['pid'], signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        metadata.update(status='needs_attention', reason=exc.reason)
                        write_json(jobs.metadata_path(job), metadata)
        slots = max(0, self.max_experiments - len(active))
        launched = []
        for job in pending:
            key = (job["kind"], job["id"])
            metadata = _load(jobs.metadata_path(job))
            if key in active or key in self.children:
                continue
            try:
                control.check(jobs.directory(job))
                if metadata.get('status') == 'needs_attention':
                    continue
            except control.StopRequested as exc:
                metadata.update(job=job, status='needs_attention', reason=exc.reason)
                write_json(jobs.metadata_path(job), metadata)
                continue
            if metadata.get("attempts", 0) >= self.max_attempts:
                metadata["status"] = "retry_exhausted"
                write_json(jobs.metadata_path(job), metadata)
                continue
            if metadata and time.time() - metadata.get("exited_at", metadata.get("launched_at", 0)) < self.retry_delay:
                continue
            if not slots:
                break
            directory = jobs.directory(job)
            if not (directory / 'runtime.json').is_file():
                metadata.update(job=job, status='needs_attention', reason='legacy_runtime_not_frozen')
                write_json(jobs.metadata_path(job), metadata)
                continue
            try:
                snapshot = runtime.verify(directory / 'runtime.json')
            except (ConfigError, OSError, ValueError, KeyError) as exc:
                metadata.update(job=job, status='needs_attention', reason='runtime_unavailable', error=str(exc))
                write_json(jobs.metadata_path(job), metadata)
                continue
            command = [sys.executable, "-B", "-u", str(snapshot / "scripts" / "iterate_branches.py"),
                       "--instance", versions.PRIVATE.name, "worker", "--kind", job["kind"],
                       "--job", job["id"], "--workers", str(self.workers)]
            if self.timeout_seconds is not None:
                from . import pack_transport
                command = [sys.executable, '-B', '-u', str(pack_transport.__file__),
                           '--snapshot', str(snapshot), '--instance', versions.PRIVATE.name,
                           '--kind', job['kind'], '--job', job['id'], '--workers', str(self.workers),
                           '--timeout-seconds', str(self.timeout_seconds)]
            metadata.update(job=job, attempts=metadata.get("attempts", 0) + 1,
                            launched_at=time.time(), status="submitted")
            write_json(jobs.metadata_path(job), metadata)
            try:
                with (directory / "scheduler.log").open("a", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=snapshot, env={**os.environ,
                                                   "DH_INSTANCES_ROOT": str(versions.PRIVATE.parent),
                                                   "DH_JOB_DIR": str(directory),
                                                   "DH_SETTINGS_FILE": str(snapshot / "settings.snapshot.yaml")}, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               close_fds=True, start_new_session=True)
            except OSError as exc:
                metadata.update(status="launch_failed", error=str(exc), exited_at=time.time())
                write_json(jobs.metadata_path(job), metadata)
                continue
            metadata["pid"] = process.pid
            metadata["process_start"] = task_state.process_start(process.pid)
            write_json(jobs.metadata_path(job), metadata)
            self.children[key] = (job, process)
            launched.append(job)
            slots -= 1
        return launched

    def run(self, *, once=False, poll_seconds=5):
        if poll_seconds <= 0:
            raise ConfigError("轮询间隔必须为正数")
        with file_lock(versions.PRIVATE / ".scheduler.lock", blocking=False):
            while True:
                for job in self.tick():
                    print(f"已调度 {job['kind']}: {job['id']}", flush=True)
                if once:
                    return
                time.sleep(poll_seconds)


def retry(kind: str, job_id: str) -> None:
    job = {"kind": kind, "id": job_id}
    directory = jobs.directory(job)
    if not directory.is_dir():
        raise ConfigError("任务不存在，不能创建空的重试任务")
    with file_lock(versions.PRIVATE / ".scheduler.lock", blocking=False):
        job = {"kind": kind, "id": job_id}
        if _active(job, _load(jobs.metadata_path(job))):
            raise ConfigError("任务还在运行，不能重复提交")
        control.resume(jobs.directory(job))
        if kind == "training":
            task_state.update(jobs.directory(job), "queued", status="queued")
        write_json(jobs.metadata_path(job), {"job": job, "attempts": 0, "status": "queued"})


def jobs_status() -> list[dict]:
    out = []
    for path in sorted((versions.PRIVATE / "scheduling").glob("*.json")):
        metadata = _load(path)
        out.append(jobs.snapshot(metadata['job'], metadata) if metadata.get('job') else metadata)
    return out
