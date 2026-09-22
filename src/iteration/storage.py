"""原子文件写入和本机跨进程锁；锁文件保留，进程退出时由内核释放锁。"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock, local

from ..config import ConfigError

_MISSING = object()


def read_json(path: Path, *, default=_MISSING):
    """Only missing optional files use a default; corrupt JSON always fails."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        if default is _MISSING:
            raise
        return default


class LockBusy(ConfigError):
    pass


_registry_guard = Lock()
_locks: dict[str, RLock] = {}
_held = local()


@contextmanager
def file_lock(path: Path, *, blocking: bool = True):
    key = str(path.resolve())
    with _registry_guard:
        thread_lock = _locks.setdefault(key, RLock())
    if not thread_lock.acquire(blocking=blocking):
        raise LockBusy(f"任务已在运行或状态正在更新: {path}")
    held = getattr(_held, "paths", None)
    if held is None:
        _held.paths = held = set()
    try:
        if key in held:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise LockBusy(f"任务已在运行或状态正在更新: {path}") from exc
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        thread_lock.release()


def locked(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with file_lock(path, blocking=False):
            return False
    except LockBusy:
        return True


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2))


def write_once_json(path: Path, value: dict) -> None:
    """Create an immutable JSON artifact, or verify an identical retry."""
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    with file_lock(path.parent / f".{path.name}.lock"):
        if path.exists():
            if read_json(path) != value:
                raise ConfigError(f"frozen artifact changed: {path.name}")
        else:
            write_json(path, value)
