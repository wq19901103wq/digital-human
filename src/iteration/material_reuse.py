"""Reuse complete verification while material inventories and dependencies stay unchanged."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
import copy
from functools import wraps
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
from threading import Lock, local

MODE = 'sealed_persistent_material_validation_v5'
CAPACITY = 16
POOL_CAPACITY = 131072


@contextmanager
def verified_history_pools(retrievers, sources, *, disk_rows=None, profile=None, capacity=POOL_CAPACITY):
    """Scope exact-row reuse to full material validation and pool approval."""
    if capacity <= 0:
        raise ValueError('history pool validation capacity must be positive')
    original_validate = sources.HistorySources.validate
    original_check = sources.HistorySources.check
    originals = {name: getattr(retrievers.PersonaFewShotRetriever, name) for name in ('_load', 'is_approved')}
    entries = {}
    registry = Lock()
    scoped = local()

    def source_entry(frame, source):
        entry = frame['sources'].get(source)
        if entry is None:
            original_check(source)
            reader = getattr(source, '_disk_reader', None)
            if reader is not None:
                frame['reads'].enter_context(reader.reading())
            binding = (tuple(source.files), tuple(source.stamps))
            with registry:
                saved = entries.get(source)
                known = saved[1] if saved is not None and saved[0] == binding else frozenset()
            entry = dict(binding=binding, known=known, pending=set())
            frame['sources'][source] = entry
        return entry

    @wraps(original_check)
    def check(source):
        frame = getattr(scoped, 'frame', None)
        if frame is None:
            return original_check(source)
        source_entry(frame, source)
        if profile is not None:
            profile.record('history_source_stamp_check_reused', 0.0)

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
                frame = dict(sources={}, valid=True, reads=ExitStack())
                scoped.frame = frame
            try:
                reader = getattr(subject, '_disk_reader', None)
                if outer and reader is not None:
                    frame['reads'].enter_context(reader.reading())
                source = getattr(subject, '_history_sources', None)
                if source is not None:
                    source_entry(frame, source)
                result = original(subject, *args, **kwargs)
                if result is False:
                    frame['valid'] = False
                if outer:
                    frame['reads'].close()
                    for source, entry in frame['sources'].items():
                        original_check(source)
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
                    try:
                        frame['reads'].close()
                    finally:
                        scoped.frame = None
        return call

    try:
        sources.HistorySources.validate = validate
        sources.HistorySources.check = check
        for name, original in originals.items():
            setattr(retrievers.PersonaFewShotRetriever, name, pool(original))
        yield pool
    finally:
        sources.HistorySources.validate = original_validate
        sources.HistorySources.check = original_check
        for name, original in originals.items():
            setattr(retrievers.PersonaFewShotRetriever, name, original)


def judge_dependencies(module, directory):
    from src.config import settings_path
    from src.iteration import runtime, versions
    from src.iteration.storage import read_json

    source = directory / 'source_judge.json'
    if not source.is_file():
        return None
    origin = read_json(source)
    names = [origin.get('study'), origin.get('arm')]
    if (origin.get('origin') not in ('history_lr_v1', 'd0011_material_repair')
            or any(not isinstance(name, str) or not name or name in ('.', '..')
                   or Path(name).name != name for name in names)
            or read_json(directory / 'config.json').get('decision_policy') != 'gbdt_only'):
        return None
    study = versions.PRIVATE / 'judge_training' / names[0]
    spec = read_json(study / 'spec.json')
    if spec.get('kind') not in ('history_lr_training', 'judge_material_repair'):
        return None
    descriptor = read_json(directory / 'scorer_source.json')
    name = descriptor.get('study')
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or Path(name).name != name):
        return None
    sweep = versions.PRIVATE / 'judge_training' / name
    sweep_spec = read_json(sweep / 'spec.json')
    original = read_json(study / names[1] / 'candidate.json')['judge_ref']
    generator = versions.generator_dir(spec['generator_ref'])
    bound = read_json(generator / 'learning.json').get('asset_files', {})
    if bound not in module.GENERATOR_RECIPES.values():
        return None
    directories = [versions.data_version_dir(spec['data_ref']), study, sweep,
                   versions.judge_dir(original)['dir'], generator]
    files = [runtime.evidence_path(path, digest) for inputs in (spec['inputs'], sweep_spec['inputs'])
             for path, digest in inputs.items()]
    files.append(settings_path())
    packages = sorted(set(sweep_spec['packages']) | {'numpy', 'scipy', 'scikit-learn'})
    return directories, files, packages


def dependencies(module, data_ref, directories):
    from src.generator import learned_sources
    from src.iteration import versions
    from src.iteration.storage import read_json

    data = {versions.data_version_dir(data_ref)}
    rankers = []
    files = set()
    packages = set()
    for directory in directories:
        record = read_json(directory / 'learning.json')
        bound = record.get('asset_files', {})
        if bound in module.GENERATOR_RECIPES.values() or bound == {'prompt.md': module.GENERIC_JUDGE_TEMPLATE}:
            continue
        if 'ranker/provenance.json' not in bound:
            judge = judge_dependencies(module, directory)
            if judge is None:
                return None
            extra, inputs, environment = judge
            data.update(extra)
            files.update(inputs)
            packages.update(environment)
            continue
        proof = read_json(directory / 'ranker/provenance.json')
        key = (str(Path(proof['model_directory']).resolve()), proof['data_ref'],
               learned_sources.digest(proof['evidence_files']))
        rankers.append(key)
        data.add(versions.data_version_dir(proof['data_ref']))
    return sorted(data, key=str), rankers, sorted(files, key=str), sorted(packages)


def environment_record(packages):
    return {name: importlib.metadata.version(name) for name in packages}


def cache_fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@contextmanager
def cached_training_reads():
    from src.iteration.training_evidence import SavedCache

    original = SavedCache.get
    scoped = local()

    @wraps(original)
    def get(store, key):
        value = original(store, key)
        records = getattr(scoped, 'records', None)
        if records is not None:
            path = store.db.execute('PRAGMA database_list').fetchone()[2]
            identity = (str(Path(path).resolve()), key)
            fingerprint = cache_fingerprint(value)
            if identity in records and records[identity] != fingerprint:
                from src.config import ConfigError
                raise ConfigError('材料核验期间训练请求缓存变化')
            records[identity] = fingerprint
        return value

    @contextmanager
    def capture():
        previous = getattr(scoped, 'records', None)
        records = {}
        scoped.records = records
        try:
            yield records
        finally:
            scoped.records = previous
            if previous is not None:
                from src.config import ConfigError
                for identity, fingerprint in records.items():
                    if identity in previous and previous[identity] != fingerprint:
                        raise ConfigError('嵌套材料核验期间训练请求缓存变化')
                    previous[identity] = fingerprint

    def check(module, records):
        stores = {}
        try:
            for (path, key), expected in records.items():
                if path not in stores:
                    stores[path] = SavedCache(Path(path))
                module._require(cache_fingerprint(original(stores[path], key)) == expected,
                                '已核验训练请求缓存变化；禁止复用材料核验')
        except sqlite3.Error as exc:
            from src.config import ConfigError
            raise ConfigError('训练请求缓存无法读取；禁止复用材料核验') from exc
        finally:
            for store in stores.values():
                store.db.close()

    SavedCache.get = get
    try:
        yield capture, check
    finally:
        SavedCache.get = original


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


def seal_record(seal):
    return dict(materials=[dict(directory=str(item.directory), stamps=item.stamps,
                               evidence={str(path): value for path, value in item.evidence.items()})
                          for item in seal.materials],
                files={str(path): value for path, value in seal.files.items()})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def receipt_path(data_ref, directories, role):
    from src.iteration import versions

    key = dict(mode=MODE, data_ref=data_ref, directories=[str(path) for path in directories], role=role)
    identity = hashlib.sha256(canonical(key).encode()).hexdigest()
    root = versions.data_version_dir(data_ref).parent.parent / '.cache' / 'material-validation'
    return root / (identity + '.json'), key


def restore_ranker_record(module, item):
    from src.generator import learned_sources
    from src.generator.history_sources import stamp
    from src.iteration.storage import read_json

    stamps = {Path(path): tuple(value) for path, value in item['stamps'].items()}
    result = copy.deepcopy(item['result'])
    if 'evidence_files' not in result:
        return stamps, result
    manifest = read_json(Path(item['key'][0]).parent / 'manifest.json')
    evidence = dict(result['evidence_files'])
    logical, actual = learned_sources.historical_runtime(manifest, evidence)
    module._require(learned_sources.serving_runtime(manifest) == result['serving_runtime'],
                    '已核验 Ranker 推理代码变化；禁止复用材料核验')
    for name in manifest['runtime']:
        aliases = [path for path in evidence if path.endswith('/' + name)]
        module._require(len(aliases) == 1, '已核验 Ranker 代码证据路径不唯一')
        evidence.pop(aliases[0])
    evidence.update(logical)
    result['evidence_files'] = evidence
    for path in actual:
        current = stamp(path)
        module._require(path not in stamps or stamps[path] == current,
                        '恢复期间已核验 Ranker 代码变化；禁止复用材料核验')
        stamps[path] = current
    return stamps, result


def read_receipt(module, path, key, seal, rankers, environment, check_cache):
    from src.generator import learned_sources

    if not path.exists():
        return None
    module._require(path.is_file() and not path.is_symlink(), '材料核验缓存不是普通文件')
    try:
        value = json.loads(path.read_text())
        payload = value['payload']
        module._require(value['sha256'] == hashlib.sha256(canonical(payload).encode()).hexdigest()
                        and payload['key'] == key, '材料核验缓存内容损坏')
        if (payload['seal'] != json.loads(canonical(seal_record(seal)))
                or payload['environment'] != environment):
            return None
        cache_rows = {(item['path'], item['key']): item['sha256'] for item in payload['cache_rows']}
        check_cache(module, cache_rows)
        records = payload['rankers']
        module._require([tuple(item['key']) for item in records] == rankers, '材料核验依赖集合不符')
        stamps = {Path(path): tuple(value) for item in records for path, value in item['stamps'].items()}
        check_stamps(module, stamps)
        restored = {tuple(item['key']): restore_ranker_record(module, item) for item in records}
        for saved, _ in restored.values():
            stamps.update(saved)
        check_stamps(module, stamps)
        seal.check()
        learned_sources._verified.update(restored)
        return seal, stamps, payload['audit'], cache_rows, environment
    except (OSError, ValueError, KeyError, TypeError) as exc:
        from src.config import ConfigError
        raise ConfigError('材料核验缓存无法读取；禁止复用') from exc


def write_receipt(path, key, seal, rankers, audit, environment, cache_rows):
    from src.generator import learned_sources
    from src.iteration.storage import write_json

    records = [dict(key=key, stamps={str(path): value for path, value in learned_sources._verified[key][0].items()},
                    result=learned_sources._verified[key][1]) for key in rankers]
    payload = dict(key=key, seal=seal_record(seal), rankers=records, audit=audit,
                   environment=environment,
                   cache_rows=[dict(path=path, key=key, sha256=value)
                               for (path, key), value in sorted(cache_rows.items())])
    write_json(path, dict(payload=payload, sha256=hashlib.sha256(canonical(payload).encode()).hexdigest()))


@contextmanager
def verified_materials(module, *, profile=None, capacity=CAPACITY, disk_rows=None):
    """Validate new material closures once; keep cheap change checks and per-case guards."""
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
        seal, stamps, audit, cache_rows, environment = saved
        with timing('material_validation_reuse_check'):
            seal.check()
            check_stamps(module, stamps)
            check_cache(module, cache_rows)
            module._require(environment_record(environment) == environment,
                            '已核验训练依赖版本变化；禁止复用材料核验')
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
                        data, rankers, files, packages = closure
                        seal = module.RunSeal(data, files)
                        seal.materials.extend(initial.materials)
                        environment = environment_record(packages)
                if closure is None:
                    return full(data_ref, directories, role)
                from src.iteration.storage import file_lock
                path, receipt_key = receipt_path(data_ref, directories, role)
                with ExitStack() as receipt_lock:
                    with timing('material_validation_receipt_wait'):
                        receipt_lock.enter_context(file_lock(path.with_suffix('.lock')))
                    with timing('material_validation_receipt_read'):
                        verified = read_receipt(module, path, receipt_key, seal, rankers,
                                                environment, check_cache)
                    if verified is None:
                        with capture_cache() as cache_rows:
                            audit = full(data_ref, directories, role)
                        with timing('material_validation_seal'):
                            stamps = ranker_stamps(rankers)
                            seal.check()
                            if stamps is not None:
                                check_stamps(module, stamps)
                                check_cache(module, cache_rows)
                                module._require(environment_record(packages) == environment,
                                                '材料核验期间训练依赖版本变化')
                                verified = seal, stamps, copy.deepcopy(audit), dict(cache_rows), environment
                                write_receipt(path, receipt_key, seal, rankers, audit,
                                              environment, cache_rows)
                    else:
                        audit = copy.deepcopy(verified[2])
                        if profile is not None:
                            profile.record('material_validation_persistent_reused', 0.0)
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
        with history_pools as pool_validation, cached_training_reads() as (capture_cache, check_cache):
            module.require_materials = (pool_validation(require_materials)
                                        if pool_validation is not None else require_materials)
            yield
    finally:
        module.require_materials = original
