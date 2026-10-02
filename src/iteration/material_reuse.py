"""Reuse fully verified materials only within an unchanged executor lifetime."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
from functools import wraps
import hashlib
import json
from pathlib import Path
from threading import Lock, local

MODE = 'sealed_in_process_material_validation_v2'
CAPACITY = 16
POOL_CAPACITY = 131072


@contextmanager
def verified_history_pools(retrievers, sources, *, disk_rows=None, profile=None, capacity=POOL_CAPACITY):
    """Scope exact-row reuse to full material validation and pool approval."""
    if capacity <= 0:
        raise ValueError('history pool validation capacity must be positive')
    original_validate = sources.HistorySources.validate
    originals = {name: getattr(retrievers.PersonaFewShotRetriever, name) for name in ('_load', 'is_approved')}
    entries = {}
    registry = Lock()
    scoped = local()

    def source_entry(frame, source):
        entry = frame['sources'].get(source)
        if entry is None:
            source.check()
            binding = (tuple(source.files), tuple(source.stamps))
            with registry:
                saved = entries.get(source)
                known = saved[1] if saved is not None and saved[0] == binding else frozenset()
            entry = dict(binding=binding, known=known, pending=set())
            frame['sources'][source] = entry
        return entry

    @wraps(original_validate)
    def validate(source, row, *, example=False):
        frame = getattr(scoped, 'frame', None)
        if frame is None or not example:
            return original_validate(source, row, example=example)
        entry = source_entry(frame, source)
        if disk_rows is not None and isinstance(row, disk_rows.Row):
            value = dict(row.reader.row(row.offset, row.size))
            value.update({name: row.header[name] for name in value if name in row.header})
        else:
            value = dict(row)
        token = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(',', ':')).encode()).digest()
        if token in entry['known'] or token in entry['pending']:
            if profile is not None:
                profile.record('history_pool_validation_reused', 0.0)
            return row
        original_validate(source, value, example=True)
        if profile is not None:
            profile.record('history_pool_validation_full', 0.0)
        if len(entry['pending']) < capacity:
            entry['pending'].add(token)
        return row

    def pool(original):
        @wraps(original)
        def call(subject, *args, **kwargs):
            frame = getattr(scoped, 'frame', None)
            outer = frame is None
            if outer:
                frame = dict(sources={}, valid=True)
                scoped.frame = frame
            try:
                source = getattr(subject, '_history_sources', None)
                if source is not None:
                    source_entry(frame, source)
                result = original(subject, *args, **kwargs)
                if result is False:
                    frame['valid'] = False
                if outer:
                    for source, entry in frame['sources'].items():
                        source.check()
                        if entry['binding'] != (tuple(source.files), tuple(source.stamps)):
                            frame['valid'] = False
                    if frame['valid']:
                        with registry:
                            size = sum(len(saved[1]) for saved in entries.values())
                            for source, entry in frame['sources'].items():
                                if source not in entries and len(entries) >= CAPACITY:
                                    continue
                                saved = entries.get(source)
                                known = saved[1] if saved is not None and saved[0] == entry['binding'] else frozenset()
                                if saved is not None and saved[0] != entry['binding']:
                                    size -= len(saved[1])
                                pending = entry['pending'] - known
                                available = max(0, capacity - size)
                                if len(pending) > available:
                                    pending = set(sorted(pending)[:available])
                                entries[source] = entry['binding'], known.union(pending)
                                size += len(pending)
                return result
            except BaseException:
                frame['valid'] = False
                raise
            finally:
                if outer:
                    scoped.frame = None
        return call

    try:
        sources.HistorySources.validate = validate
        for name, original in originals.items():
            setattr(retrievers.PersonaFewShotRetriever, name, pool(original))
        yield pool
    finally:
        sources.HistorySources.validate = original_validate
        for name, original in originals.items():
            setattr(retrievers.PersonaFewShotRetriever, name, original)


def dependencies(module, data_ref, directories):
    from src.generator import learned_sources
    from src.iteration import versions
    from src.iteration.storage import read_json

    data = {versions.data_version_dir(data_ref)}
    rankers = []
    for directory in directories:
        record = read_json(directory / 'learning.json')
        bound = record.get('asset_files', {})
        if bound in module.GENERATOR_RECIPES.values() or bound == {'prompt.md': module.GENERIC_JUDGE_TEMPLATE}:
            continue
        if 'ranker/provenance.json' not in bound:
            return None
        proof = read_json(directory / 'ranker/provenance.json')
        key = (str(Path(proof['model_directory']).resolve()), proof['data_ref'],
               learned_sources.digest(proof['evidence_files']))
        rankers.append(key)
        data.add(versions.data_version_dir(proof['data_ref']))
    return data, rankers


def ranker_stamps(keys):
    from src.generator import learned_sources

    stamps = {}
    for key in keys:
        verified = learned_sources._verified.get(key)
        if verified is None:
            return None
        for path, value in verified[0].items():
            if path in stamps and stamps[path] != value:
                return None
            stamps[path] = value
    return stamps


def check_stamps(module, stamps):
    from src.generator.history_sources import stamp

    module._require(all(stamp(path) == saved for path, saved in stamps.items()),
                    '运行中已核验 Ranker 依赖变化；禁止复用材料核验')


@contextmanager
def verified_materials(module, *, profile=None, capacity=CAPACITY, disk_rows=None):
    """Keep first full provenance validation, seals, and all per-case guards."""
    if capacity <= 0:
        raise ValueError('material validation capacity must be positive')
    original = module.require_materials
    entries = {}
    registry = Lock()

    def timing(name):
        return profile.span(name) if profile is not None else nullcontext()

    def full(data_ref, directories, role):
        with timing('material_validation_full'):
            return original(data_ref, directories, role)

    def reuse(saved):
        seal, stamps, audit = saved
        with timing('material_validation_reuse_check'):
            seal.check()
            check_stamps(module, stamps)
        return copy.deepcopy(audit)

    @wraps(original)
    def require_materials(data_ref, directories, role='development'):
        directories = tuple(Path(directory).absolute() for directory in directories)
        key = (data_ref, directories, role)
        with registry:
            if key not in entries and len(entries) < capacity:
                entries[key] = dict(lock=Lock(), saved=None)
            entry = entries.get(key)
            saved = entry['saved'] if entry is not None else None
        if entry is None:
            return full(data_ref, directories, role)
        if saved is not None:
            return reuse(saved)
        with timing('material_validation_lock_wait'):
            entry['lock'].acquire()
        try:
            with registry:
                saved = entry['saved']
            if saved is None:
                with timing('material_validation_seal'):
                    initial = module.RunSeal(directories)
                    closure = dependencies(module, data_ref, directories)
                    initial.check()
                    if closure is not None:
                        data, rankers = closure
                        seal = module.RunSeal(data)
                        seal.materials.extend(initial.materials)
                if closure is None:
                    return full(data_ref, directories, role)
                audit = full(data_ref, directories, role)
                with timing('material_validation_seal'):
                    stamps = ranker_stamps(rankers)
                    seal.check()
                    if stamps is not None:
                        check_stamps(module, stamps)
                        verified = seal, stamps, copy.deepcopy(audit)
                        with registry:
                            entry['saved'] = verified
                return audit
        finally:
            entry['lock'].release()
        return reuse(saved)

    try:
        from src.generator import few_shot, history_sources
        history_pools = (verified_history_pools(few_shot, history_sources, disk_rows=disk_rows, profile=profile)
                         if disk_rows is not None else nullcontext())
        with history_pools as pool_validation:
            module.require_materials = (pool_validation(require_materials)
                                        if pool_validation is not None else require_materials)
            yield
    finally:
        module.require_materials = original
