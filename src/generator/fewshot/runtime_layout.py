"""Exact relocation of frozen scoring bytes, without rewriting historical identity."""
from pathlib import Path

LEGACY_PREFIX = 'src/generator/fewshot_ranker/'
PREFIX = 'src/generator/fewshot/'
FROZEN_NAMES = ('__init__', 'features', 'schema', 'extraction', 'crosses',
                'identity_crosses', 'boosting', 'action_crosses', 'concern', 'supplemental')
RELOCATED = {LEGACY_PREFIX + name + '.py':
             PREFIX + ('_frozen_init' if name == '__init__' else name) + '.py'
             for name in FROZEN_NAMES}
SERVING_NAMES = FROZEN_NAMES[:7]
SERVING_RUNTIME = tuple(PREFIX + name + '.py' for name in SERVING_NAMES)
RECONSTRUCTION_TRAINING_NAMES = (
    '__init__', 'report', 'dataset', 'labels', 'samples', 'runtime_sources',
    'action_comparison', 'concern_comparison', 'architecture_comparison',
    'identity_comparison', 'comparison', 'discovery', 'neural_sweep',
    'neural_models', 'optimization', 'logistic',
)


def relocated_file(root, name):
    path = Path(name)
    if '..' in path.parts:
        return None
    matches = [target for old, target in RELOCATED.items()
               if str(path) == old or (path.is_absolute() and str(path).endswith('/' + old))]
    return Path(root) / matches[0] if len(matches) == 1 else None


def execution_files(root):
    names = [*RELOCATED.values(), PREFIX + '__init__.py', PREFIX + 'runtime_layout.py',
             PREFIX + 'provenance.py', PREFIX + 'learned_selection.py',
             *(PREFIX + 'training/' + name + '.py' for name in RECONSTRUCTION_TRAINING_NAMES)]
    return {Path(root) / name for name in names}


def retired_runtime_path(name):
    path = Path(name)
    names = (*RELOCATED, LEGACY_PREFIX + 'logistic.py', LEGACY_PREFIX + 'training.py')
    return '..' not in path.parts and any(str(path) == old or
        (path.is_absolute() and str(path).endswith('/' + old)) for old in names)
