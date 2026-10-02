import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from src.config import ConfigError
from src.generator import disk_history, few_shot, history_sources, learned_sources
from src.generator.history_sources import stamp
from src.iteration import control, learning_guard, material_reuse, versions
from src.iteration.transport_profile import TransportProfile
from test_leakage_guards import env as env


@pytest.fixture
def materials(tmp_path, monkeypatch):
    data_root = tmp_path / 'data'
    monkeypatch.setattr(versions, 'DATA_ROOT', data_root)
    for name in ('current', 'original'):
        directory = data_root / name
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text('{}')
    directory = tmp_path / 'generator'
    directory.mkdir()
    (directory / 'prompt.md').write_text('mechanical')
    (directory / 'learning.json').write_text(json.dumps({'asset_files': {'prompt.md': 'fixed'},
                                                       'evidence_files': {}}))
    calls = []

    def full(data_ref, directories, role='development'):
        calls.append((data_ref, tuple(directories), role))
        return {'assets': [{'verified': True}]}

    module = SimpleNamespace(require_materials=full, RunSeal=learning_guard.RunSeal,
                             _require=learning_guard._require,
                             GENERATOR_RECIPES={'mechanical': {'prompt.md': 'fixed'}},
                             GENERIC_JUDGE_TEMPLATE='generic')
    return SimpleNamespace(directory=directory, data=data_root, calls=calls, module=module)


def ranker(materials, tmp_path, monkeypatch, *, cache=True):
    hidden = tmp_path / 'hidden.json'
    hidden.write_text('{}')
    proof = {'model_directory': str(tmp_path / 'model'), 'data_ref': 'original', 'evidence_files': {}}
    path = materials.directory / 'ranker'
    path.mkdir()
    (path / 'provenance.json').write_text(json.dumps(proof))
    (materials.directory / 'learning.json').write_text(json.dumps({
        'asset_files': {'ranker/provenance.json': 'fixed'}, 'evidence_files': {}}))
    key = (proof['model_directory'], 'original', learned_sources.digest({}))
    monkeypatch.setattr(learned_sources, '_verified', {})
    original = materials.module.require_materials

    def full(*args, **kwargs):
        result = original(*args, **kwargs)
        if cache:
            learned_sources._verified[key] = ({hidden: stamp(hidden)}, {'verified': True})
        return result

    materials.module.require_materials = full
    return hidden


def test_real_full_validation_is_reused_without_changing_audit(env, monkeypatch):
    original = learning_guard.require_materials
    calls = []

    def full(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(learning_guard, 'require_materials', full)
    with material_reuse.verified_materials(learning_guard):
        first = learning_guard.require_materials('d-test', [env.gdir])
        expected = copy.deepcopy(first)
        first['assets'].clear()
        assert learning_guard.require_materials('d-test', [env.gdir]) == expected
    assert len(calls) == 1 and learning_guard.require_materials is full


def test_material_reuse_without_history_preserves_pool_validation(materials):
    originals = (history_sources.HistorySources.validate, few_shot.PersonaFewShotRetriever._load,
                 few_shot.PersonaFewShotRetriever.is_approved)
    original_materials = materials.module.require_materials
    with material_reuse.verified_materials(materials.module):
        assert materials.module.require_materials is not original_materials
        assert originals == (history_sources.HistorySources.validate, few_shot.PersonaFewShotRetriever._load,
                             few_shot.PersonaFewShotRetriever.is_approved)
        first = materials.module.require_materials('current', [materials.directory])
        assert materials.module.require_materials('current', [materials.directory]) == first
    assert len(materials.calls) == 1
    assert materials.module.require_materials is original_materials


def test_same_key_parallel_workers_validate_once(materials):
    barrier = Barrier(4)

    def work(number):
        barrier.wait(timeout=10)
        return materials.module.require_materials('current', [materials.directory])

    with material_reuse.verified_materials(materials.module), ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, range(4)))
    assert len(materials.calls) == 1
    assert all(result == results[0] and result is not results[0] for result in results[1:])


def test_verified_material_checks_and_audit_copies_are_not_serialized(materials, monkeypatch, tmp_path):
    barrier = Barrier(4)
    original_check = materials.module.RunSeal.check
    profile = TransportProfile(tmp_path / 'profile.json', {})

    def check(seal):
        barrier.wait(timeout=10)
        return original_check(seal)

    def work(number):
        return materials.module.require_materials('current', [materials.directory])

    with material_reuse.verified_materials(materials.module, profile=profile):
        expected = materials.module.require_materials('current', [materials.directory])
        monkeypatch.setattr(materials.module.RunSeal, 'check', check)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(work, range(4)))
    assert len(materials.calls) == 1
    assert all(result == expected for result in results)
    results[0]['assets'].clear()
    assert all(result == expected for result in results[1:])
    metrics = profile.metrics
    assert metrics['material_validation_lock_wait']['count'] == 1
    assert metrics['material_validation_reuse_check']['active_peak'] == 4


def test_different_keys_do_not_hold_registry_during_validation(materials):
    barrier = Barrier(2)
    original = materials.module.require_materials

    def full(*args, **kwargs):
        barrier.wait(timeout=10)
        return original(*args, **kwargs)

    materials.module.require_materials = full
    with material_reuse.verified_materials(materials.module), ThreadPoolExecutor(max_workers=2) as pool:
        assert len(list(pool.map(lambda role: materials.module.require_materials(
            'current', [materials.directory], role), ('development', 'fixed_test')))) == 2
    assert len(materials.calls) == 2


@pytest.mark.parametrize('target', ['learning', 'asset', 'new_asset', 'data', 'original_data', 'proof', 'hidden'])
def test_changed_material_or_ranker_dependency_rejects_reuse(materials, tmp_path, monkeypatch, target):
    hidden = ranker(materials, tmp_path, monkeypatch)
    paths = {'learning': materials.directory / 'learning.json', 'asset': materials.directory / 'prompt.md',
             'new_asset': materials.directory / 'new.json', 'data': materials.data / 'current/manifest.json',
             'original_data': materials.data / 'original/manifest.json',
             'proof': materials.directory / 'ranker/provenance.json', 'hidden': hidden}
    with material_reuse.verified_materials(materials.module):
        materials.module.require_materials('current', [materials.directory])
        paths[target].write_text('changed')
        with pytest.raises(ConfigError, match='变化'):
            materials.module.require_materials('current', [materials.directory])
    assert len(materials.calls) == 1


def test_evidence_change_rejects_reuse(materials, tmp_path):
    from src.config import sha256_file
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}')
    (materials.directory / 'learning.json').write_text(json.dumps({
        'asset_files': {'prompt.md': 'fixed'}, 'evidence_files': {str(evidence): sha256_file(evidence)}}))
    with material_reuse.verified_materials(materials.module):
        materials.module.require_materials('current', [materials.directory])
        evidence.write_text('changed')
        with pytest.raises(ConfigError, match='变化'):
            materials.module.require_materials('current', [materials.directory])


@pytest.mark.parametrize('bound', [{}, {'prompt.md': 'reconstructed'}, {'unknown.json': 'unknown'}])
def test_unknown_and_reconstructed_materials_keep_full_validation(materials, bound):
    (materials.directory / 'learning.json').write_text(json.dumps({'asset_files': bound}))
    with material_reuse.verified_materials(materials.module):
        for number in range(2):
            materials.module.require_materials('current', [materials.directory])
    assert len(materials.calls) == 2


def test_ranker_without_verified_dependency_closure_is_not_cached(materials, tmp_path, monkeypatch):
    ranker(materials, tmp_path, monkeypatch, cache=False)
    with material_reuse.verified_materials(materials.module):
        for number in range(2):
            materials.module.require_materials('current', [materials.directory])
    assert len(materials.calls) == 2


def test_full_validation_failure_is_not_cached(materials):
    original = materials.module.require_materials
    failures = [True]

    def full(*args, **kwargs):
        result = original(*args, **kwargs)
        if failures:
            failures.pop()
            raise ConfigError('validation failed')
        return result

    materials.module.require_materials = full
    with material_reuse.verified_materials(materials.module):
        with pytest.raises(ConfigError, match='validation failed'):
            materials.module.require_materials('current', [materials.directory])
        materials.module.require_materials('current', [materials.directory])
        materials.module.require_materials('current', [materials.directory])
    assert len(materials.calls) == 2


def test_mutation_during_full_validation_is_rejected(materials):
    original = materials.module.require_materials

    def full(*args, **kwargs):
        result = original(*args, **kwargs)
        (materials.directory / 'prompt.md').write_text('changed')
        return result

    materials.module.require_materials = full
    with material_reuse.verified_materials(materials.module):
        with pytest.raises(ConfigError, match='变化'):
            materials.module.require_materials('current', [materials.directory])


def test_capacity_falls_back_to_original_validation(materials):
    with material_reuse.verified_materials(materials.module, capacity=1):
        for role in ('development', 'fixed_test', 'development', 'fixed_test'):
            materials.module.require_materials('current', [materials.directory], role)
    assert len(materials.calls) == 3


def test_cancellation_and_context_exit_restore_original(materials, monkeypatch):
    original = materials.module.require_materials

    def cancelled():
        raise control.StopRequested('cancelled')

    with pytest.raises(control.StopRequested, match='cancelled'):
        with material_reuse.verified_materials(materials.module):
            materials.module.require_materials('current', [materials.directory])
            monkeypatch.setattr(control, 'check', cancelled)
            materials.module.require_materials('current', [materials.directory])
    assert materials.module.require_materials is original


def test_profile_records_full_validation_and_reuse(materials, tmp_path):
    profile = TransportProfile(tmp_path / 'profile.json', {})
    with material_reuse.verified_materials(materials.module, profile=profile):
        materials.module.require_materials('current', [materials.directory])
        materials.module.require_materials('current', [materials.directory])
    metrics = profile.snapshot()['metrics']
    assert metrics['material_validation_full']['count'] == 1
    assert metrics['material_validation_lock_wait']['count'] == 1
    assert metrics['material_validation_reuse_check']['count'] == 1


@pytest.mark.parametrize('capacity', [0, -1])
def test_invalid_capacity_is_rejected(materials, capacity):
    with pytest.raises(ValueError, match='capacity'):
        with material_reuse.verified_materials(materials.module, capacity=capacity):
            pytest.fail('invalid capacity accepted')


@pytest.fixture
def pool_modules(tmp_path):
    path = tmp_path / 'pool.jsonl'
    path.write_text('{}\n')

    class Source:
        def __init__(self):
            self.files = [path]
            self.stamps = [stamp(path)]
            self.calls = []

        def check(self):
            if self.stamps != [stamp(path)]:
                raise ConfigError('source changed')

        def validate(self, row, *, example=False):
            self.check()
            self.calls.append((dict(row), example))
            if row.get('invalid'):
                raise ConfigError('invalid row')
            return row

    class Retriever:
        def __init__(self, source=None):
            self._history_sources = source or Source()
            self.rows = [{'id': 'first', 'reply': ['original']}]
            self.approved = True
            self.failure = False
            self.hook = None

        def _load(self):
            for row in self.rows:
                self._history_sources.validate(row, example=True)
            if self.hook:
                self.hook()
            if self.failure:
                raise ConfigError('pool failed')
            return self.rows

        def is_approved(self):
            self._load()
            for row in self.rows:
                self._history_sources.validate(row, example=True)
            return self.approved

        def render_selected(self):
            return [self._history_sources.validate(row, example=True) for row in self.rows]

    return SimpleNamespace(retrievers=SimpleNamespace(PersonaFewShotRetriever=Retriever),
                           sources=SimpleNamespace(HistorySources=Source), path=path)


def test_pool_reuses_exact_rows_but_not_direct_case_or_render_validation(pool_modules, tmp_path):
    retriever = pool_modules.retrievers.PersonaFewShotRetriever()
    source = retriever._history_sources
    profile = TransportProfile(tmp_path / 'profile.json', {})
    with material_reuse.verified_history_pools(pool_modules.retrievers, pool_modules.sources, profile=profile):
        assert retriever.is_approved()
        assert retriever._load() is retriever.rows
        assert len(source.calls) == 1
        source.validate(retriever.rows[0], example=True)
        source.validate(retriever.rows[0], example=False)
        retriever.render_selected()
        assert len(source.calls) == 4
        retriever.rows[0]['reply'] = ['changed']
        retriever._load()
        assert len(source.calls) == 5
        assert source.calls[-1][0]['reply'] == ['changed']
    assert profile.metrics['history_pool_validation_full']['count'] == 2
    assert profile.metrics['history_pool_validation_reused']['count'] == 2


def test_material_scope_reuses_examples_across_full_checks_and_pool(
        materials, pool_modules, monkeypatch):
    monkeypatch.setattr(few_shot, 'PersonaFewShotRetriever', pool_modules.retrievers.PersonaFewShotRetriever)
    monkeypatch.setattr(history_sources, 'HistorySources', pool_modules.sources.HistorySources)
    retriever = few_shot.PersonaFewShotRetriever()
    source = retriever._history_sources
    row = retriever.rows[0]
    original = materials.module.require_materials
    guards = []

    def full(data_ref, directories, role):
        source.validate(row, example=True)
        source.validate(copy.deepcopy(row), example=True)
        source.validate(row, example=False)
        guards.append(role)
        return original(data_ref, directories, role)

    materials.module.require_materials = full
    with material_reuse.verified_materials(materials.module, disk_rows=disk_history):
        materials.module.require_materials('current', [materials.directory], 'development')
        materials.module.require_materials('current', [materials.directory], 'fixed_test')
        assert guards == ['development', 'fixed_test']
        assert len(source.calls) == 3
        assert retriever.is_approved()
        assert len(source.calls) == 3
        source.validate(row, example=True)
        retriever.render_selected()
        assert len(source.calls) == 5
        row['reply'] = ['changed']
        retriever._load()
        assert len(source.calls) == 6
    assert materials.module.require_materials is full


@pytest.mark.parametrize('failure', ['exception', 'seal'])
def test_failed_material_scope_does_not_publish_rows(materials, pool_modules, monkeypatch, failure):
    monkeypatch.setattr(few_shot, 'PersonaFewShotRetriever', pool_modules.retrievers.PersonaFewShotRetriever)
    monkeypatch.setattr(history_sources, 'HistorySources', pool_modules.sources.HistorySources)
    retriever = few_shot.PersonaFewShotRetriever()
    source = retriever._history_sources
    original = materials.module.require_materials

    def full(data_ref, directories, role):
        source.validate(retriever.rows[0], example=True)
        if failure == 'exception':
            raise ConfigError('material failed')
        (materials.directory / 'prompt.md').write_text('changed')
        return original(data_ref, directories, role)

    materials.module.require_materials = full
    with material_reuse.verified_materials(materials.module, disk_rows=disk_history):
        with pytest.raises(ConfigError):
            materials.module.require_materials('current', [materials.directory])
        retriever._load()
        assert len(source.calls) == 2


def test_material_scope_rejects_source_changes_at_exit(materials, pool_modules, monkeypatch):
    monkeypatch.setattr(few_shot, 'PersonaFewShotRetriever', pool_modules.retrievers.PersonaFewShotRetriever)
    monkeypatch.setattr(history_sources, 'HistorySources', pool_modules.sources.HistorySources)
    source = history_sources.HistorySources()
    original = materials.module.require_materials

    def full(data_ref, directories, role):
        source.validate({'id': 'first'}, example=True)
        pool_modules.path.write_text('changed\n')
        return original(data_ref, directories, role)

    materials.module.require_materials = full
    with material_reuse.verified_materials(materials.module, disk_rows=disk_history):
        with pytest.raises(ConfigError, match='source changed'):
            materials.module.require_materials('current', [materials.directory])


def test_material_scopes_do_not_reuse_another_threads_pending_rows(
        materials, pool_modules, monkeypatch):
    monkeypatch.setattr(few_shot, 'PersonaFewShotRetriever', pool_modules.retrievers.PersonaFewShotRetriever)
    monkeypatch.setattr(history_sources, 'HistorySources', pool_modules.sources.HistorySources)
    source = history_sources.HistorySources()
    original = materials.module.require_materials
    barrier = Barrier(2, timeout=10)

    def full(data_ref, directories, role):
        source.validate({'id': 'first'}, example=True)
        barrier.wait()
        source.validate({'id': 'first'}, example=True)
        return original(data_ref, directories, role)

    def work(role):
        return materials.module.require_materials('current', [materials.directory], role)

    materials.module.require_materials = full
    with material_reuse.verified_materials(materials.module, disk_rows=disk_history):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(work, ['development', 'fixed_test']))
        assert len(source.calls) == 2
        assert len(results) == 2


@pytest.mark.parametrize('failure', ['exception', 'rejected', 'invalid'])
def test_failed_pool_does_not_publish_pending_rows(pool_modules, failure):
    retriever = pool_modules.retrievers.PersonaFewShotRetriever()
    source = retriever._history_sources
    with material_reuse.verified_history_pools(pool_modules.retrievers, pool_modules.sources):
        if failure == 'rejected':
            retriever.approved = False
            assert not retriever.is_approved()
            retriever.approved = True
        else:
            retriever.failure = failure == 'exception'
            if failure == 'invalid':
                retriever.rows.append({'id': 'bad', 'invalid': True})
            with pytest.raises(ConfigError):
                retriever.is_approved()
            retriever.failure = False
            retriever.rows[:] = retriever.rows[:1]
        retriever._load()
        assert sum(row['id'] == 'first' for row, example in source.calls) == 2


def test_changed_or_distinct_source_cannot_reuse_pool_validation(pool_modules):
    retriever = pool_modules.retrievers.PersonaFewShotRetriever()
    source = retriever._history_sources
    with material_reuse.verified_history_pools(pool_modules.retrievers, pool_modules.sources):
        retriever._load()
        other = pool_modules.retrievers.PersonaFewShotRetriever()
        other._load()
        assert len(other._history_sources.calls) == 1
        pool_modules.path.write_text('{"changed":true}\n')
        with pytest.raises(ConfigError, match='source changed'):
            retriever._load()
        source.stamps = [stamp(pool_modules.path)]
        retriever._load()
        assert len(source.calls) == 2


def test_source_changed_during_pool_is_rejected_at_exit(pool_modules):
    retriever = pool_modules.retrievers.PersonaFewShotRetriever()
    retriever.hook = lambda: pool_modules.path.write_text('changed\n')
    with material_reuse.verified_history_pools(pool_modules.retrievers, pool_modules.sources):
        with pytest.raises(ConfigError, match='source changed'):
            retriever._load()


def test_pool_capacity_falls_back_to_full_validation(pool_modules):
    retriever = pool_modules.retrievers.PersonaFewShotRetriever()
    retriever.rows.append({'id': 'second'})
    with material_reuse.verified_history_pools(pool_modules.retrievers, pool_modules.sources, capacity=1):
        retriever._load()
        retriever._load()
    assert len(retriever._history_sources.calls) == 3


def test_pool_scopes_are_thread_local_and_restore_after_exception(pool_modules):
    retrievers, sources = pool_modules.retrievers, pool_modules.sources
    originals = (sources.HistorySources.validate, retrievers.PersonaFewShotRetriever._load,
                 retrievers.PersonaFewShotRetriever.is_approved)
    barrier = Barrier(4)

    def work(number):
        retriever = retrievers.PersonaFewShotRetriever()
        barrier.wait(timeout=10)
        assert retriever.is_approved()
        retriever._load()
        return len(retriever._history_sources.calls)

    with pytest.raises(ConfigError, match='exit failed'):
        with material_reuse.verified_history_pools(retrievers, sources), ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(work, range(4))) == [1] * 4
            raise ConfigError('exit failed')
    assert (sources.HistorySources.validate, retrievers.PersonaFewShotRetriever._load,
            retrievers.PersonaFewShotRetriever.is_approved) == originals


def test_disk_pool_reads_once_and_preserves_header_and_row_identity(pool_modules):
    retriever = pool_modules.retrievers.PersonaFewShotRetriever()
    calls = []

    def read(offset, size):
        calls.append((offset, size))
        return {'id': 'raw', 'reply': ['original']}

    row = disk_history.Row(SimpleNamespace(row=read), 7, 19, {'id': 'header'})
    retriever.rows = [row]
    with material_reuse.verified_history_pools(pool_modules.retrievers, pool_modules.sources, disk_rows=disk_history):
        assert retriever._load()[0] is row
        assert retriever._history_sources.calls == [({'id': 'header', 'reply': ['original']}, True)]
        assert calls == [(7, 19)]
        retriever._load()
        assert len(retriever._history_sources.calls) == 1


@pytest.mark.parametrize('disk', [False, True])
def test_real_pool_reuse_preserves_retrieval_and_rejects_changed_selected_text(env, disk):
    from contextlib import nullcontext
    from test_disk_history import queries, retriever

    requests = list(queries(env.source))
    baseline = retriever(env.source)
    expected = [baseline.retrieve(**request) for request in requests]
    storage = disk_history.disk_storage(few_shot) if disk else nullcontext()
    with storage, material_reuse.verified_history_pools(few_shot, history_sources, disk_rows=disk_history):
        candidate = retriever(env.source)
        assert candidate.is_approved()
        actual = [candidate.retrieve(**request) for request in requests]
        assert actual == expected
        assert [candidate.render_selected(rows) for rows in actual] == [
            baseline.render_selected(rows) for rows in expected]
        row = copy.deepcopy(actual[0][0])
        row['reply'] = ['unverified replacement']
        with pytest.raises(ConfigError):
            candidate.render_selected([row])
