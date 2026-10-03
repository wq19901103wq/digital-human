"""Cheap per-request material seals: full provenance check once, stamp watch after.

实现自 src/iteration/learning_guard.py 迁入（检索/历史来源/材料验证三线模块化设计 §3）。
"""
import json
from pathlib import Path

from ...config import ConfigError


def _require(value, message):
    # 与 validation._require 刻意重复：seals 被 validation 顶层依赖，反向引用会成环。
    if not value:
        raise ConfigError(message)


class MaterialSeal:
    """Cheap per-request check after a full provenance check at construction."""
    def __init__(self, directory):
        self.directory = Path(directory)
        from ...history.sources import stamp
        record = self.directory / 'learning.json'
        before = stamp(record) if record.exists() else None
        try:
            evidence = json.loads(record.read_text()).get('evidence_files', {}) if before else {}
        except (OSError, ValueError, AttributeError) as exc:
            raise ConfigError('学习来源记录无法读取，禁止继续') from exc
        _require(isinstance(evidence, dict), '学习来源证据格式无效，禁止继续')
        from ..runtime import evidence_path
        paths = [evidence_path(p, h) for p, h in evidence.items()]
        self.evidence = {p: stamp(p) for p in paths}
        self.stamps = self._stamps()
        _require(before == (stamp(record) if record.exists() else None), '读取期间学习来源发生变化')

    def _stamps(self):
        from ...history.sources import stamp
        return {str(p.relative_to(self.directory)): stamp(p)
                for p in self.directory.rglob('*') if p.is_file()
                and p.relative_to(self.directory) != Path('meta.json')}

    def check(self):
        from ..control import check
        check()
        from ...history.sources import stamp
        _require(self.stamps == self._stamps(), '运行中学习材料变化；禁止继续请求或复用缓存')
        _require(all(stamp(p) == value for p, value in self.evidence.items()),
                 '运行中学习证据变化；禁止继续请求或复用缓存')


class RunSeal:
    """Watch the dependency set captured *before* validation and loading.

    File stamps detect changes after complete provenance validation. Resumed
    executors may reuse its receipt only for identical inventories and dependencies.
    """
    def __init__(self, directories=(), files=()):
        from ...history.sources import stamp
        self.materials = [MaterialSeal(d) for d in directories]
        self.files = {Path(p): stamp(Path(p)) for p in files}

    def check(self):
        from ..control import check
        check()
        from ...history.sources import stamp
        for seal in self.materials:
            seal.check()
        _require(all(stamp(p) == saved for p, saved in self.files.items()),
                 '运行中冻结输入变化；禁止继续请求、写入有效结果或晋级')
