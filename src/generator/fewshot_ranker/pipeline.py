"""Frozen, resumable feature extraction followed automatically by local training."""
import os
from pathlib import Path
import platform
import time

from ...config import ROOT, sha256_file
from ...iteration import control, versions
from ...iteration.learning_guard import RunSeal
from ...iteration.storage import file_lock, locked, read_json, write_json, write_once_json
from ...judge.corrected import CodexJudgeClient
from ..history_sources import digest, require
from . import extraction
from .dataset import load_dataset


def status(output):
    output = Path(output)
    result = read_json(output / 'state.json', default={'status': 'not_started'})
    result['running'] = locked(output / '.run.lock')
    result['features'] = read_json(output / 'feature_progress.json', default=None)
    result['report'] = read_json(output / 'report.json', default=None)
    if not result['running'] and result['status'] in ('preparing', 'extracting', 'training'):
        result['status'] = 'interrupted'
    return result


def execute(inventory, output, feature_config, feature_cache, instance, *, workers=16, passes=3):
    from . import training
    import numpy
    import sklearn
    import torch
    inventory, output, feature_config, feature_cache = (
        Path(p).resolve() for p in (inventory, output, feature_config, feature_cache))
    require(1 <= workers <= 16 and passes >= 1, 'Invalid extraction concurrency or retry count')
    versions.switch_instance(instance)
    with file_lock(output / '.run.lock', blocking=False), control.job(output):
        def state(stage, **kw):
            value = dict(status=stage, pid=os.getpid(), updated_at=time.time(), **kw)
            write_json(output / 'state.json', value)
        try:
            state('preparing')
            source_files = [*Path(__file__).parent.glob('*.py'), ROOT / 'scripts/train_fewshot_ranker.py',
                *[ROOT / name for name in ('src/judge/embedding.py', 'src/judge/corrected.py',
                    'src/judge/corrected_v1.py', 'src/generator/history.py', 'src/generator/ranker_labels.py',
                    'src/generator/history_sources.py', 'src/iteration/storage.py', 'src/cache.py')]]
            sources = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files}
            inputs = [inventory / 'manifest.json', inventory / 'inventory.json',
                      inventory / 'labeling/manifest.json', feature_config]
            seal = RunSeal(files=[*source_files, *inputs, *inventory.glob('targets/*.json'),
                                 *inventory.glob('labeling/observations/*.json')])
            groups, observations, summary, binding = load_dataset(inventory)
            config = read_json(feature_config)
            llm = config.get('feature_llm', config.get('llm', config))
            client = CodexJudgeClient(llm)
            tasks, refs = extraction.prepare(groups, client.cache_identity())
            manifest = dict(schema=1, kind='single_example_fewshot_ranker', inventory=str(inventory),
                dataset=binding, summary=summary, features=dict(client=client.cache_identity(),
                    request_keys_sha256=digest(sorted(tasks)), count=len(tasks)), recipe=training.RECIPE,
                runtime=sources, libraries=dict(python=platform.python_version(), numpy=numpy.__version__,
                    sklearn=sklearn.__version__, torch=torch.__version__))
            write_once_json(output / 'manifest.json', manifest)
            completed = read_json(output / 'completed.json', default=None)
            if completed:
                require(completed['manifest_sha256'] == digest(manifest) and all(
                    sha256_file(output / name) == value for name, value in completed['artifacts'].items()),
                    'Completed training artifacts changed')
                state('complete', reused=True)
                return read_json(output / 'report.json')
            seal.check()
            write_json(output / 'transport.json', dict(workers=workers, passes=passes, config=llm,
                                                      feature_cache=str(feature_cache)))
            state('extracting', samples=summary, feature_tasks=len(tasks))
            values = extraction.run(tasks, feature_cache, output, client, workers=workers, passes=passes)
            seal.check()
            rows = training.assemble(groups, refs, values)
            write_once_json(output / 'feature_matrix.json', dict(manifest_sha256=digest(manifest), rows=rows))
            state('training', samples=summary, feature_tasks=len(tasks))
            report = training.train(groups, observations, rows, output)
            seal.check()
            artifacts = {name: sha256_file(output / name) for name in
                         ('model.json', 'predictions.json', 'report.json', 'REPORT.md', 'feature_matrix.json')}
            write_once_json(output / 'completed.json', dict(manifest_sha256=digest(manifest), artifacts=artifacts))
            state('complete', samples=summary, feature_tasks=len(tasks))
            return report
        except BaseException as exc:
            state('needs_attention', error_type=type(exc).__name__, error=str(exc))
            raise
