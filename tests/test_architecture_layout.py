"""Durable domain, compatibility and runtime-source layout contracts."""
import ast
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path

import pytest

from src.generator.fewshot.training.runtime_sources import runtime_files
from src.generator.fewshot.runtime_layout import RELOCATED
from tests.frozen_boundaries import ROOT, SERVING_RUNTIME_SHA256, frozen_violations


COMPATIBILITY = (
    ('src.config', 'src.infrastructure.config'),
    ('src.cache', 'src.infrastructure.cache'),
    ('src.tracing', 'src.infrastructure.tracing'),
    ('src.iteration.storage', 'src.infrastructure.storage'),
    ('src.iteration.shares', 'src.share.versions'),
    ('src.bootstrap.acceptance_refresh', 'src.data.acceptance_refresh'),
    ('src.history.sources', 'src.data.history.sources'),
    ('src.generator.persona', 'src.generator.context.persona'),
    ('src.generator.background', 'src.generator.context.background'),
    ('src.generator.source_projection', 'src.generator.context.source_projection'),
    ('src.generator.selection', 'src.generator.fewshot.selection'),
    ('src.generator.reranker', 'src.generator.fewshot.reranker'),
    ('src.generator.shared_history', 'src.generator.fewshot.shared_history'),
    ('src.generator.recall_reuse', 'src.generator.fewshot.recall_cache'),
    ('src.generator.learned_selection', 'src.generator.fewshot.learned_selection'),
    ('src.judge.contribution', 'src.judge.scoring.contribution'),
    ('src.judge.ownership', 'src.judge.scoring.ownership'),
)


def imports(path):
    relative = path.relative_to(ROOT).with_suffix('')
    parts = relative.parts
    package = '.'.join(parts[:-1])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                module = resolve_name('.' * node.level + module, package)
            yield module
            yield from (module + '.' + alias.name for alias in node.names)


@pytest.mark.parametrize('old,new', COMPATIBILITY)
def test_compatibility_paths_share_the_same_module_and_state(old, new):
    assert import_module(old) is import_module(new)


def test_runtime_sources_cover_moved_implementations_and_frozen_serving():
    files = runtime_files()
    assert files == sorted(set(files))
    assert all(path.is_file() for path in files)
    expected = {ROOT / name for name in RELOCATED.values()}
    expected.update(ROOT / module.replace('.', '/') / '__init__.py'
                    for module in ('src.data', 'src.share', 'src.infrastructure'))
    expected.update(Path(import_module(new).__file__) for old, new in COMPATIBILITY
                    if not new.startswith('src.iteration.'))
    assert expected <= set(files)
    assert frozen_violations() == []


def test_runtime_transport_loads_the_actual_recall_cache():
    from src.generator.fewshot import recall_cache
    from src.iteration.pack_transport import live_retrieval_cache
    loaded = live_retrieval_cache()
    assert Path(loaded.__file__).resolve() == Path(recall_cache.__file__).resolve()
    assert loaded.MODE == recall_cache.MODE


def test_business_execution_does_not_import_dashboard():
    problems = []
    for path in (ROOT / 'src').rglob('*.py'):
        if 'dashboard' in path.relative_to(ROOT / 'src').parts:
            continue
        if any(name.startswith('src.dashboard') for name in imports(path)):
            problems.append(str(path.relative_to(ROOT)))
    assert problems == []


def test_shared_memory_does_not_depend_on_judge_or_generator():
    problems = []
    for path in (ROOT / 'src/share/memory').glob('*.py'):
        if any(name.startswith(('src.judge', 'src.generator')) for name in imports(path)):
            problems.append(str(path.relative_to(ROOT)))
    assert problems == []


@pytest.mark.parametrize('folder,forbidden', (
    ('src/generator/context', 'src.share.wiki'),
    ('src/share/memory', 'src.share.wiki'),
    ('src/share/wiki', 'src.share.memory'),
    ('src/share/knowledge_view.py', ('src.share.wiki', 'src.share.memory', 'src.generator', 'src.judge')),
))
def test_knowledge_maintenance_query_and_prompt_respect_dependency_direction(folder, forbidden):
    root = ROOT / folder
    paths = [root] if root.is_file() else root.glob('*.py')
    assert not [(str(path.relative_to(ROOT)), name) for path in paths
                for name in imports(path) if name.startswith(forbidden)]


def test_retired_implementations_have_no_production_imports():
    retired = {'src.bootstrap.analyze', 'src.judge.pairwise_models'}
    problems = []
    paths = [*(ROOT / 'src').rglob('*.py'), *(ROOT / 'scripts').glob('*.py')]
    for path in paths:
        if retired.intersection(imports(path)):
            problems.append(str(path.relative_to(ROOT)))
    assert problems == []


def test_domain_names_do_not_restore_redundant_or_misleading_packages():
    retired = ('src/bootstrap/wiki', 'src/generator/fewshot_ranker', 'src/generator/fewshot/ranker',
               'src/judge/models', 'src/judge/training')
    assert all(not list((ROOT / folder).rglob('*.py')) for folder in retired)
    prefixes = tuple(folder.replace('/', '.') for folder in retired)
    paths = [*(ROOT / 'src').rglob('*.py'), *(ROOT / 'scripts').glob('*.py')]
    assert not [(str(path.relative_to(ROOT)), name) for path in paths
                for name in imports(path) if name.startswith(prefixes)]


def test_training_inventory_uses_the_existing_retrieval_and_history_sources():
    from src.generator.fewshot.training import samples
    assert {path.relative_to(ROOT).as_posix() for path in samples.IMPLEMENTATION_FILES} == {
        'src/generator/fewshot/' + name
        for name in ('session.py', 'prepare.py', 'rank.py', 'recall_engines.py')}
    assert all(path.is_file() for path in samples.IMPLEMENTATION_FILES)
