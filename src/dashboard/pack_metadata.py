"""Cache only small pack metadata, invalidating on every source file change."""
import json
from functools import lru_cache
from pathlib import Path
from threading import RLock

_lock = RLock()


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=128)
def _read(path, signature):
    pack = json.loads(path.read_text(encoding='utf-8'))
    if _signature(path) != signature:
        raise OSError('Evaluation pack changed while reading metadata')
    # Never retain rows or their private contents across dashboard requests.
    return {'c0_gen_version': pack.get('c0_gen_version')}


def summary(path: Path) -> dict:
    with _lock:
        try:
            signature = _signature(path)
        except FileNotFoundError:
            return {}
        return dict(_read(path, signature))
