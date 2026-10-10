"""Source files used by ranker training and serving, including frozen modules."""
from src.infrastructure.config import ROOT


def runtime_files():
    folders = ('generator', 'judge', 'infrastructure', 'data', 'share')
    files = {file for folder in folders for file in (ROOT / 'src' / folder).rglob('*.py')}
    files.update(ROOT / name for name in ('src/__init__.py', 'src/history/__init__.py',
        'src/history/sources.py',
        'src/config.py', 'src/cache.py', 'src/tracing.py', 'src/llm.py',
        'src/iteration/storage.py', 'src/iteration/learning_guard.py',
        'src/iteration/parallel.py', 'src/iteration/control.py'))
    return sorted(files)
