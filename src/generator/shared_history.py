"""Share one pinned history index within an offline generation process."""
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from src.config import ConfigError, sha256_file
from src.generator import generator


def file_stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class PinnedRetriever:
    """Serialize original retrieval methods; per-case exclusion stays outside."""

    def __init__(self, original, path, inputs):
        self._lock = RLock()
        files = (path, path.with_name('report.json'))
        self._stamps = {p: file_stamp(p) for p in files}
        for p in files:
            if inputs.get(str(p.resolve())) != sha256_file(p):
                raise ConfigError(f'共享历史索引输入与冻结内容不符：{p}')
        self._original = original(path=path)
        if not self._original.is_approved():
            raise ConfigError('共享历史索引未通过原审批检查')
        if self._original.history_policy != 'complete_before_input_v1':
            raise ConfigError('共享索引仅用于当前完整历史回放')
        self._check()
        print('history index loaded once; original retrieval serialized', flush=True)

    def _check(self):
        if any(file_stamp(p) != stamp for p, stamp in self._stamps.items()):
            raise ConfigError('共享历史索引文件发生变化，拒绝继续生成')

    def __getattr__(self, name):
        return getattr(self._original, name)

    def is_approved(self):
        with self._lock:
            self._check()
            return True

    def retrieve(self, **kwargs):
        with self._lock:
            self._check()
            rows = self._original.retrieve(**kwargs)
            self._check()
            return rows

    def render_selected(self, rows, max_chars=2500):
        with self._lock:
            self._check()
            result = self._original.render_selected(rows, max_chars=max_chars)
            self._check()
            return result


@contextmanager
def shared_history_index(inputs):
    """Scoped factory adapter; frozen prompt, ranking and exclusion code is used."""
    original = generator.PersonaFewShotRetriever
    instances = {}
    lock = RLock()

    def factory(path, render_path=None):
        if render_path is not None:
            raise ConfigError('共享历史索引不能使用替换展示库')
        path = Path(path).resolve()
        with lock:
            if path not in instances:
                instances[path] = PinnedRetriever(original, path, inputs)
            return instances[path]

    generator.PersonaFewShotRetriever = factory
    try:
        yield instances
    finally:
        generator.PersonaFewShotRetriever = original
