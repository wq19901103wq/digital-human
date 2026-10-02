import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import llm
from src.config import ConfigError
from src.generator import few_shot
from src.iteration import pack_transport, runtime, versions
from test_fewshot_selection import generator
from test_leakage_guards import env as env


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf')])
def test_invalid_time_slice_is_rejected(seconds):
    with pytest.raises(ValueError, match='time slice'):
        with pack_transport.case_time_slice(seconds):
            pytest.fail('invalid time slice was accepted')


@pytest.mark.parametrize('workers', [1, 4])
def test_time_slice_drains_all_submitted_results_before_stopping(monkeypatch, workers):
    from threading import Event
    from src.iteration import control, runner
    original = runner.completed_map
    clock = [0.0]
    submitted, finished, saved = [], [], []
    ready = Event()

    def items():
        for number in range(12):
            if len(submitted) == workers:
                clock[0] = 2.0
                ready.set()
            yield number

    def work(number):
        submitted.append(number)
        if workers == 1:
            clock[0] = 2.0
        else:
            if len(submitted) == workers:
                clock[0] = 2.0
                ready.set()
            assert ready.wait(10)
        finished.append(number)
        return number

    monkeypatch.setattr(pack_transport.time, 'monotonic', lambda: clock[0])
    with pytest.raises(control.StopRequested, match='time_slice_complete'):
        with pack_transport.case_time_slice(1):
            for result in runner.completed_map(work, items(), workers):
                saved.append(result)
    assert sorted(submitted) == sorted(finished) == sorted(saved) == list(range(workers))
    assert runner.completed_map is original


@pytest.mark.parametrize('seconds', [None, 60])
def test_time_slice_allows_natural_completion_and_preserves_errors(seconds):
    from src.iteration import control, runner
    original = runner.completed_map
    with pack_transport.case_time_slice(seconds):
        assert list(runner.completed_map(lambda value: value * 2, range(4), 2))
        assert list(runner.completed_map(lambda value: value, [], 2)) == []
        def cancelled(value):
            raise control.StopRequested('cancelled')
        with pytest.raises(control.StopRequested, match='cancelled'):
            list(runner.completed_map(cancelled, [1], 2))
    assert runner.completed_map is original


def test_time_slice_profile_counts_only_persisted_results(monkeypatch):
    from src.iteration import runner, transport_profile
    profile = transport_profile.TransportProfile(Path('unused-profile.json'), {})
    with pack_transport.case_time_slice(60, profile=profile):
        results = list(runner.completed_map(lambda status: {'status': status}, ['ok', 'failed'], 1))
    assert results == [{'status': 'ok'}, {'status': 'failed'}]
    assert profile.slices[0]['success'] == 1
    assert profile.slices[0]['failed'] == 1
    assert profile.slices[0]['workers'] == 1
    assert profile.slices[0]['elapsed_seconds'] >= 0


@pytest.mark.parametrize('progress_type', ['RunProgress', 'ParallelCaseProgress'])
@pytest.mark.parametrize('phase', ['generating', 'judging'])
@pytest.mark.parametrize('preparation,expected', [(0.5, 2), (100, 2), (900, 2)])
def test_time_slice_excludes_all_lazy_preparation_once(
        monkeypatch, capsys, progress_type, phase, preparation, expected):
    from src.iteration import control, progress, runner
    clock = [0.0]
    stages = []
    kind = getattr(progress, progress_type)

    def stage(instance, phase, **fields):
        stages.append((phase, fields))

    monkeypatch.setattr(kind, 'stage', stage)
    monkeypatch.setattr(pack_transport.time, 'monotonic', lambda: clock[0])

    def work(number):
        kind.stage(None, 'preparing', case_index=number)
        if number == 0:
            clock[0] += preparation
        kind.stage(None, phase, branch='baseline')
        clock[0] += 0.6
        return {'status': 'ok', 'case_id': number}

    saved = []
    with pytest.raises(control.StopRequested, match='time_slice_complete'):
        with pack_transport.case_time_slice(1):
            saved.extend(runner.completed_map(work, range(10), 1))
    assert [row['case_id'] for row in saved] == list(range(expected))
    assert kind.stage is stage
    assert stages[::2] == [('preparing', {'case_index': number}) for number in range(expected)]
    assert stages[1::2] == [(phase, {'branch': 'baseline'})] * expected
    output = capsys.readouterr().out
    assert output.count('Time slice ready:') == 1
    assert f'credited={preparation:.3f}s' in output
    assert f'persisted success={expected}; failed=0' in output


def test_time_slice_preparation_failures_do_not_extend_budget(monkeypatch, capsys):
    from src.iteration import control, runner
    clock = [0.0]
    monkeypatch.setattr(pack_transport.time, 'monotonic', lambda: clock[0])

    def work(number):
        clock[0] += 0.6
        return {'status': 'failed', 'case_id': number}

    saved = []
    with pytest.raises(control.StopRequested, match='time_slice_complete'):
        with pack_transport.case_time_slice(1):
            saved.extend(runner.completed_map(work, range(10), 1))
    assert len(saved) == 2
    assert 'persisted success=0; failed=2' in capsys.readouterr().out


@pytest.mark.parametrize('preparation', [0.5, 900])
def test_time_slice_parallel_warmup_is_credited_once_and_drained(monkeypatch, capsys, preparation):
    from threading import Barrier
    from src.iteration import control, progress, runner
    clock = [0.0]
    barrier = Barrier(4, timeout=10)
    monkeypatch.setattr(pack_transport.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(progress.ParallelCaseProgress, 'stage', lambda *args, **kwargs: None)

    def work(number):
        if barrier.wait() == 0:
            clock[0] = preparation
        barrier.wait()
        progress.ParallelCaseProgress.stage(None, 'generating')
        barrier.wait()
        clock[0] = preparation + 1.5
        return {'status': 'ok', 'case_id': number}

    saved = []
    with pytest.raises(control.StopRequested, match='time_slice_complete'):
        with pack_transport.case_time_slice(1):
            saved.extend(runner.completed_map(work, range(12), 4))
    assert sorted(row['case_id'] for row in saved) == list(range(4))
    output = capsys.readouterr().out
    assert output.count('Time slice ready:') == 1
    assert f'credited={preparation:.3f}s' in output
    assert 'persisted success=4; failed=0' in output


def test_time_slice_restores_progress_hooks_on_cancel():
    from src.iteration import control, progress, runner
    stages = {kind: kind.stage for kind in (progress.RunProgress, progress.ParallelCaseProgress)}

    def cancelled(value):
        raise control.StopRequested('cancelled')

    with pytest.raises(control.StopRequested, match='cancelled'):
        with pack_transport.case_time_slice(1):
            list(runner.completed_map(cancelled, [1], 2))
    assert all(kind.stage is stage for kind, stage in stages.items())


def test_saved_reply_judge_does_not_wait_for_generation_archive(env):
    from src.iteration import control
    from src.iteration.storage import file_lock, write_json
    directory = env.root / 'experiments' / 'saved-judge'
    write_json(directory / 'spec.json', {'kind': 'judge_eval'})
    with file_lock(env.root.parent / '.history_memory.lock'):
        for kind in ('experiment', 'promotion'):
            required = pack_transport.needs_history_memory(directory, kind)
            assert not required
            with pack_transport.history_memory_lane(directory, required=required):
                assert not (directory / 'memory_lane.json').exists()
        control.cancel(directory)
        with pytest.raises(control.StopRequested):
            with pack_transport.history_memory_lane(directory, required=False):
                pytest.fail('cancelled Judge must not run')


@pytest.mark.parametrize('kind,spec_kind', [('pack', 'judge_eval'), ('training', 'judge_eval'),
                                         ('experiment', 'gen_ab'), ('promotion', 'gen_ab'),
                                         ('experiment', 'unknown')])
def test_history_jobs_keep_archive_capacity_limit(env, kind, spec_kind):
    from src.iteration.storage import write_json
    directory = env.root / 'job'
    write_json(directory / 'spec.json', {'kind': spec_kind})
    assert pack_transport.needs_history_memory(directory, kind)


def test_history_memory_lane_releases_on_error_and_serializes_work(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from src.iteration.storage import locked, read_json
    lock = env.root.parent / '.history_memory.lock'
    first, second = env.root / 'first', env.root / 'second'
    entered, started = Event(), Event()
    def work():
        started.set()
        with pack_transport.history_memory_lane(second):
            entered.set()
            assert read_json(second / 'memory_lane.json')['phase'] == 'active'
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='synthetic'):
            with pack_transport.history_memory_lane(first):
                assert read_json(first / 'memory_lane.json')['phase'] == 'active'
                future = pool.submit(work)
                assert started.wait(5)
                assert not entered.wait(0.05)
                raise RuntimeError('synthetic')
        future.result(timeout=5)
    assert entered.is_set() and not locked(lock)
    assert read_json(first / 'memory_lane.json')['phase'] == 'idle'
    assert read_json(second / 'memory_lane.json')['phase'] == 'idle'


@pytest.mark.parametrize('value', ['0', '3', '-1', '', 'invalid'])
def test_history_memory_capacity_rejects_unbounded_values(monkeypatch, value):
    monkeypatch.setenv('DH_HISTORY_MEMORY_LANES', value)
    with pytest.raises(ValueError, match='must be 1 or 2'):
        pack_transport.history_memory_capacity()


def test_two_history_slots_count_live_legacy_job_and_release_on_error(env, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from src.iteration.storage import LockBusy, file_lock, locked, read_json
    monkeypatch.setenv('DH_HISTORY_MEMORY_LANES', '2')
    legacy_lock = env.root.parent / '.history_memory.lock'
    extra_lock = env.root.parent / '.history_memory.1.lock'
    legacy_active, release_legacy, third_started, third_active = (Event() for _ in range(4))
    second, third = env.root / 'second', env.root / 'third'
    def legacy():
        with file_lock(legacy_lock):
            legacy_active.set()
            assert release_legacy.wait(10)
    def wait_for_slot():
        third_started.set()
        with pack_transport.history_memory_lane(third):
            third_active.set()
            assert read_json(third / 'memory_lane.json')['slot'] == 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(legacy)
        assert legacy_active.wait(5)
        try:
            # An error inside the job must propagate, even if it is LockBusy.
            with pytest.raises(LockBusy, match='job failure'):
                with pack_transport.history_memory_lane(second):
                    value = read_json(second / 'memory_lane.json')
                    assert value['capacity'] == 2 and value['slot'] == 1
                    waiting = pool.submit(wait_for_slot)
                    assert third_started.wait(5)
                    assert not third_active.wait(.2)
                    raise LockBusy('job failure')
            waiting.result(timeout=5)
        finally:
            release_legacy.set()
        old.result(timeout=5)
    assert not locked(legacy_lock) and not locked(extra_lock)
    assert read_json(second / 'memory_lane.json')['phase'] == 'idle'
    assert read_json(third / 'memory_lane.json')['phase'] == 'idle'


def test_waiting_history_job_can_be_cancelled(env, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from src.iteration import control
    from src.iteration.storage import file_lock, read_json
    monkeypatch.setenv('DH_HISTORY_MEMORY_LANES', '1')
    output = env.root / 'cancelled'
    checked = Event()
    def stop():
        checked.set()
        raise control.StopRequested('cancelled')
    def waiting():
        with pack_transport.history_memory_lane(output):
            pytest.fail('cancelled job entered archive phase')
    monkeypatch.setattr(control, 'check', stop)
    with file_lock(env.root.parent / '.history_memory.lock'):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(waiting)
            assert checked.wait(5)
            with pytest.raises(control.StopRequested, match='cancelled'):
                future.result(timeout=5)
    assert read_json(output / 'memory_lane.json')['phase'] == 'idle'


def test_history_cache_preserves_values_flags_and_mutation_isolation(monkeypatch):
    originals = {name: getattr(few_shot, name) for name in
                 ('_situation_tags', '_situation_profile')}
    expected = originals['_situation_profile']('你说什么？', enable_clarification=True)
    calls = []
    original = originals['_situation_profile']
    def profile(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(few_shot, '_situation_profile', profile)
    with pack_transport.history_feature_cache(few_shot):
        result = few_shot._situation_profile('你说什么？', enable_clarification=True)
        assert result == expected
        result['test_mutation'] = 'must not leak'
        assert few_shot._situation_profile('你说什么？', enable_clarification=True) == expected
        assert len(calls) == 1
        for text, flag in [('你说什么？', False), ('吃饭了吗', True)]:
            assert few_shot._situation_profile(text, enable_clarification=flag) == original(
                text, enable_clarification=flag)
        assert len(calls) == 3
        tags = few_shot._situation_tags('吃饭了吗')
        tags.add('must not leak')
        assert few_shot._situation_tags('吃饭了吗') == originals['_situation_tags']('吃饭了吗')
    assert few_shot._situation_profile is profile
    assert few_shot._situation_tags is originals['_situation_tags']


@pytest.mark.parametrize('name,result_type', [
    ('_situation_tags', set), ('_situation_profile', dict),
])
def test_history_cache_shares_equal_results_across_distinct_inputs(monkeypatch, name, result_type):
    from concurrent.futures import ThreadPoolExecutor
    import weakref
    class Tracked(result_type):
        pass
    original = getattr(few_shot, name)
    expected = original('吃饭了吗')
    references = []

    def analyze(text):
        result = Tracked(original(text))
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(few_shot, name, analyze)
    with pack_transport.history_feature_cache(few_shot):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(getattr(few_shot, name),
                                    ['吃饭了吗' + ' ' * number for number in range(64)]))
        assert results == [expected] * 64
        assert len({id(result) for result in results}) == 64
        assert len(references) == 64
        assert sum(reference() is not None for reference in references) == 1
    assert all(reference() is None for reference in references)


def test_history_cache_preserves_dictionary_order_when_sharing_results(monkeypatch):
    def analyze(text):
        pairs = [('first', 'same'), ('second', 'same')]
        return dict(pairs if text == 'forward' else reversed(pairs))
    monkeypatch.setattr(few_shot, '_situation_profile', analyze)
    with pack_transport.history_feature_cache(few_shot):
        for text in ('forward', 'reverse', 'forward', 'reverse'):
            assert list(few_shot._situation_profile(text).items()) == list(analyze(text).items())


def test_history_cache_bounds_full_text_keys(monkeypatch):
    calls = []

    def analyze(text):
        calls.append(text)
        return {'tag': 'same'}

    monkeypatch.setattr(few_shot, '_situation_profile', analyze)
    with pack_transport.history_feature_cache(few_shot):
        for number in range(257):
            assert few_shot._situation_profile(str(number)) == {'tag': 'same'}
        few_shot._situation_profile('256')
        assert len(calls) == 257
        few_shot._situation_profile('0')
        assert len(calls) == 258


@pytest.mark.parametrize('kind,spec_kind,required', [
    ('experiment', 'gen_ab', True), ('experiment', 'judge_eval', False),
    ('promotion', 'gen_ab', True), ('promotion', 'judge_eval', False),
    ('pack', 'unknown', True), ('training', 'unknown', True), ('features', 'gen_ab', False),
])
def test_main_records_and_scopes_disk_storage(env, tmp_path, monkeypatch, kind, spec_kind, required):
    import faulthandler
    import sys
    from src.config import sha256_file
    from src.generator import disk_history, history_sources
    from src.iteration import jobs, learning_guard, material_reuse
    from src.iteration.storage import read_json, write_json
    directory = env.root / 'experiments/synthetic'
    monkeypatch.setenv('DH_INSTANCES_ROOT', str(env.root.parent))
    write_json(directory / 'spec.json', {'kind': spec_kind})
    write_json(directory / 'runtime.json', {'frozen': 'unchanged'})
    originals = (history_sources.HistorySources.__init__, few_shot.PersonaFewShotRetriever._load,
                 few_shot.PersonaFewShotRetriever.retrieve)
    original_materials = learning_guard.require_materials
    monkeypatch.setattr(pack_transport, 'live_disk_history', lambda: disk_history)
    monkeypatch.setattr(pack_transport, 'live_material_reuse', lambda: material_reuse)
    monkeypatch.setattr(jobs, 'directory', lambda job: directory)
    monkeypatch.setattr(runtime, 'verify', lambda path: tmp_path)
    checked = []
    monkeypatch.setattr(runtime, 'require_current', lambda path: checked.append(path))
    monkeypatch.setattr(faulthandler, 'enable', lambda: None)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(sys, 'argv', ['pack_transport', '--snapshot', str(tmp_path), '--instance', env.root.name,
                                    '--job', 'synthetic', '--kind', kind, '--workers', '4',
                                    '--timeout-seconds', '360'] +
                        (['--feature-output', str(tmp_path / 'features')] if kind == 'features' else []))
    executed = []

    def run(*args, **kwargs):
        assert (history_sources.HistorySources.__init__ is not originals[0]) == required
        assert (few_shot.PersonaFewShotRetriever._load is not originals[1]) == required
        assert (few_shot.PersonaFewShotRetriever.retrieve is not originals[2]) == required
        assert (learning_guard.require_materials is not original_materials) == (kind not in {'features', 'promotion'})
        executed.append(kind)

    monkeypatch.setattr(pack_transport.runpy, 'run_path', run)
    monkeypatch.setattr(pack_transport, 'promote_saved', run)
    monkeypatch.setattr(pack_transport, 'complete_saved_features', run)
    pack_transport.main()
    assert checked == [directory] and executed == [kind]
    assert originals == (history_sources.HistorySources.__init__, few_shot.PersonaFewShotRetriever._load,
                         few_shot.PersonaFewShotRetriever.retrieve)
    assert learning_guard.require_materials is original_materials
    receipt, = [read_json(path) for path in (directory / 'transport_overrides').glob('*.json')]
    assert receipt['runtime'] == {'frozen': 'unchanged'}
    assert receipt['workers'] == 4 and receipt['history_memory_required'] == required
    assert receipt['history_storage'] == (disk_history.MODE if required else 'not_required')
    assert receipt['history_storage_cache_capacity'] == (256 if required else None)
    assert receipt['history_storage_adapter_sha256'] == (sha256_file(Path(disk_history.__file__)) if required else None)
    assert receipt['history_feature_cache_capacity'] == 256
    from src.generator import retrieval_cache
    reuse = required and kind not in {'features', 'promotion'}
    assert receipt['history_retrieval_cache'] == (retrieval_cache.MODE if reuse else 'not_required')
    assert receipt['history_retrieval_cache_capacity'] == (128 if reuse else None)
    assert receipt['history_retrieval_cache_max_bytes'] == (4 * 1024 * 1024 if reuse else None)
    from src.generator import disk_source_reads
    assert receipt['history_source_reads'] == (disk_source_reads.MODE if reuse else 'not_required')
    assert receipt['history_source_offset_capacity'] == (256 if reuse else None)
    assert receipt['history_source_record_capacity_per_worker'] == (1 if reuse else None)
    assert receipt['history_source_sqlite_cache_kib_per_worker'] == (256 if reuse else None)
    assert receipt['history_source_reads_sha256'] == (sha256_file(Path(disk_source_reads.__file__)) if reuse else None)
    material_required = kind not in {'features', 'promotion'}
    assert receipt['material_validation_reuse'] == (material_reuse.MODE if material_required else 'not_required')
    assert receipt['material_validation_reuse_sha256'] == (sha256_file(Path(material_reuse.__file__)) if material_required else None)


@pytest.mark.parametrize('text', [
    '', ' \n\t', 'Aa aA 123', '你说什么？你说什么？', 'ＡßÉ e\u0301 🙂🧑‍💻',
    '这是超过十二个字符的上下文，保留词元顺序和重复次数。',
])
def test_history_term_storage_preserves_tokens_and_fresh_lists(text):
    from collections import Counter
    original = few_shot._terms
    expected = original(text)
    with pack_transport.history_feature_cache(few_shot):
        result = few_shot._terms(text)
        assert result == expected
        assert Counter(result) == Counter(expected)
        result.append('must not leak')
        assert few_shot._terms(text=text) == expected
    assert few_shot._terms is original


def test_history_term_storage_shares_counter_keys_across_workers():
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    import sys
    texts = ['这段重复历史用于检索。今晚吃饭吗？', '另一个历史对话。今晚吃饭吗？'] * 128
    expected = [Counter(few_shot._terms(text)) for text in texts]
    with pack_transport.history_feature_cache(few_shot):
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda text: Counter(few_shot._terms(text)), texts))
    assert actual == expected
    original_keys = {id(term): term for counts in expected for term in counts}
    shared_keys = {id(term): term for counts in actual for term in counts}
    assert len(shared_keys) == len({term for counts in actual for term in counts})
    assert sum(map(sys.getsizeof, shared_keys.values())) < sum(
        map(sys.getsizeof, original_keys.values())) / 100


def test_history_cache_restores_all_functions_after_error():
    originals = {name: getattr(few_shot, name) for name in
                 ('_terms', '_situation_tags', '_situation_profile')}
    with pytest.raises(RuntimeError, match='analysis failed'):
        with pack_transport.history_feature_cache(few_shot):
            raise RuntimeError('analysis failed')
    assert all(getattr(few_shot, name) is original for name, original in originals.items())


def test_history_cache_preserves_prompts_and_per_case_time_guard(env):
    cases = env.source.roles['development']
    expected = [generator(env).build_prompt(case) for case in cases]
    with pack_transport.history_feature_cache(few_shot):
        model = generator(env)
        assert [model.build_prompt(case) for case in cases] == expected
        rendered = json.dumps(expected[0], ensure_ascii=False)
        for future in env.source.cases[3:]:
            assert all(answer not in rendered for answer in future['human_reply'])
        source = env.source.directory / 'messages.jsonl'
        source.write_text(source.read_text() + '\n')
        with pytest.raises(ConfigError):
            model.build_prompt(cases[0])


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf'), 3601])
def test_invalid_timeout_is_rejected(seconds):
    with pytest.raises(ValueError, match='timeout'):
        with pack_transport.timeout_floor(llm.ChatClient, seconds):
            pytest.fail('invalid timeout was accepted')


@pytest.mark.parametrize('protocol', ['anthropic', 'openai'])
def test_timeout_changes_transport_only(monkeypatch, protocol):
    monkeypatch.setenv('TEST_API_BASE', 'https://example.invalid/v1')
    monkeypatch.setenv('TEST_API_KEY', 'synthetic')
    settings = {'llm': {'base_url_env': 'TEST_API_BASE', 'api_key_env': 'TEST_API_KEY'}}
    config = {'model': 'synthetic', 'protocol': protocol, 'timeout_seconds': 20,
              'temperature': 0.7, 'max_tokens': 2500}
    before = copy.deepcopy(config)
    original = llm.ChatClient(settings, config)
    identity = original.cache_identity()
    bodies = []
    monkeypatch.setattr(llm.cache, 'memo', lambda kind, key, produce, **kw: produce())
    monkeypatch.setattr(llm.ChatClient, '_send', lambda self, body: bodies.append(body) or 'ok')
    original.chat([{'role': 'user', 'content': 'synthetic prompt'}], json_mode=True)
    initializer = llm.ChatClient.__init__
    with pack_transport.timeout_floor(llm.ChatClient, 180):
        updated = llm.ChatClient(settings, config)
        assert updated._timeout == 180
        if protocol == 'openai':
            assert updated._client.timeout == 180
        assert updated.cache_identity() == identity
        updated.chat([{'role': 'user', 'content': 'synthetic prompt'}], json_mode=True)
        assert llm.ChatClient(settings, {**config, 'timeout_seconds': 360})._timeout == 360
    assert bodies[0] == bodies[1]
    assert config == before
    assert llm.ChatClient.__init__ is initializer
    assert llm.ChatClient(settings, config)._timeout == 20


class CachedFeatureClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def cache_identity(self):
        return {'model': 'synthetic'}

    def run(self, prompt, schema):
        from src import cache

        def produce():
            self.calls += 1
            return next(self.responses)

        return cache.memo('llm_request', {'client': self.cache_identity(), 'prompt': prompt,
                                         'schema': schema}, produce, valid=cache.response_valid)


def concern_cache_request(client):
    from src.generator.fewshot_ranker import concern
    from src.generator.fewshot_ranker.features import context_view
    from tests.test_fewshot_ranker_concern import concerned_record
    return concern.task(context_view(concerned_record()), client.cache_identity())


def concern_cache_scope(directory):
    from src import cache
    trace = {'case_id': 'synthetic-case', 'trace_ref': 'synthetic-trace',
             'versions': {'data_ref': 'synthetic-data', 'dataset': 'fixed'}}
    return cache.scope(directory / 'experiments' / 'fixed', trace, {}, 0)


def concern_cache_response(valid):
    return json.dumps({'self_concern_state': 'concern', 'later_incoming_basis': 'reassurance_or_opinion',
                       'self_positions': [0] if valid else [1], 'incoming_positions': [1] if valid else [0]})


def test_feature_cache_retries_invalid_original_and_repair_responses(tmp_path):
    from src import cache
    from src.generator.fewshot_ranker import supplemental
    invalid, valid = concern_cache_response(False), concern_cache_response(True)
    client = CachedFeatureClient([invalid, invalid, invalid, valid])
    key, request = concern_cache_request(client)
    original = supplemental.extract_one
    with concern_cache_scope(tmp_path):
        with pytest.raises(ConfigError, match='speaker or position'):
            supplemental.extract_one(tmp_path / 'features', key, request, client)
        assert client.calls == 2
        assert supplemental.cached(tmp_path / 'features', key, request) is None
        with pack_transport.feature_response_cache_validation():
            result = supplemental.extract_one(tmp_path / 'features', key, request, client)
            assert result == json.loads(valid)
            assert client.calls == 4
            assert supplemental.extract_one(tmp_path / 'features', key, request, client) == result
            assert client.calls == 4
        assert cache.response_valid('non-feature response')
    assert supplemental.extract_one is original
    assert len(list((tmp_path / 'features' / 'failures').glob('*/*.json'))) == 3


def test_feature_cache_reuses_valid_request_response(tmp_path):
    from src.generator.fewshot_ranker import supplemental
    valid = concern_cache_response(True)
    client = CachedFeatureClient([valid])
    key, request = concern_cache_request(client)
    with concern_cache_scope(tmp_path):
        assert client.run(request['prompt'], request['schema']) == valid
        with pack_transport.feature_response_cache_validation():
            assert supplemental.extract_one(tmp_path / 'features', key, request, client) == json.loads(valid)
        assert client.calls == 1


def test_invalid_feature_responses_remain_retryable_and_restore_hooks(tmp_path):
    from src import cache
    from src.generator.fewshot_ranker import supplemental
    invalid, valid = concern_cache_response(False), concern_cache_response(True)
    client = CachedFeatureClient([invalid, invalid, invalid, invalid, valid])
    key, request = concern_cache_request(client)
    original = supplemental.extract_one
    original_failed = supplemental.failed
    with concern_cache_scope(tmp_path):
        for expected_calls in (2, 4):
            with pytest.raises(llm.LLMError, match='speaker or position') as error:
                with pack_transport.feature_response_cache_validation():
                    supplemental.extract_one(tmp_path / 'features', key, request, client)
            assert error.value.retryable
            assert not isinstance(error.value, ConfigError)
            assert client.calls == expected_calls
            assert supplemental.extract_one is original
            assert supplemental.failed is original_failed
            assert supplemental.cached(tmp_path / 'features', key, request) is None
            assert cache.response_valid('non-feature response')
        with pack_transport.feature_response_cache_validation():
            assert supplemental.extract_one(tmp_path / 'features', key, request, client) == json.loads(valid)
        assert client.calls == 5
    assert len(list((tmp_path / 'features' / 'failures').glob('*/*.json'))) == 4


@pytest.mark.parametrize('legacy', [False, True])
def test_feature_validation_preserves_client_binding_failures(tmp_path, monkeypatch, legacy):
    from src.generator.fewshot_ranker import supplemental
    from src.generator.history_sources import digest
    if legacy:
        monkeypatch.delattr(supplemental, 'failed')
    client = CachedFeatureClient([])
    _, request = concern_cache_request(client)
    request['client'] = {'model': 'different'}
    with pack_transport.feature_response_cache_validation():
        with pytest.raises(ConfigError, match='client differs'):
            supplemental.extract_one(tmp_path / 'features', digest(request), request, client)
    assert client.calls == 0
    assert not (tmp_path / 'features' / 'failures').exists()


@pytest.mark.parametrize('invalid_evidence', [False, True])
def test_feature_validation_preserves_success_cache_integrity_failures(tmp_path, invalid_evidence):
    from src.generator.fewshot_ranker import supplemental
    from src.generator.history_sources import digest
    client = CachedFeatureClient([concern_cache_response(True)])
    key, request = concern_cache_request(client)
    supplemental.extract_one(tmp_path / 'features', key, request, client)
    path = tmp_path / 'features' / (key + '.json')
    value = json.loads(path.read_text())
    if invalid_evidence:
        value['features'] = json.loads(concern_cache_response(False))
        value['payload_sha256'] = digest({name: item for name, item in value.items()
                                         if name != 'payload_sha256'})
    else:
        value['payload_sha256'] = 'tampered'
    path.write_text(json.dumps(value))
    expected = 'speaker or position' if invalid_evidence else 'cache binding changed'
    with pack_transport.feature_response_cache_validation():
        with pytest.raises(ConfigError, match=expected):
            supplemental.extract_one(tmp_path / 'features', key, request, client)
    assert client.calls == 1
    assert not (tmp_path / 'features' / 'failures').exists()


def test_feature_validation_isolates_parallel_response_and_binding_failures(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from src.generator.fewshot_ranker import supplemental
    from src.generator.history import HistoryError
    ready = Barrier(2)
    request = {'kind': supplemental.concern.KIND}

    def extract(directory, key, request, client):
        error = HistoryError('synthetic failure')
        if key == 'response':
            supplemental.failed(directory, key, request, 'invalid response', error)
        ready.wait(timeout=10)
        raise error

    monkeypatch.setattr(supplemental, 'extract_one', extract)
    original_failed = supplemental.failed
    with pack_transport.feature_response_cache_validation(), ThreadPoolExecutor(max_workers=2) as executor:
        response = executor.submit(supplemental.extract_one, tmp_path, 'response', request, None)
        binding = executor.submit(supplemental.extract_one, tmp_path, 'binding', request, None)
        with pytest.raises(llm.LLMError):
            response.result(timeout=10)
        with pytest.raises(ConfigError):
            binding.result(timeout=10)
    assert supplemental.extract_one is extract
    assert supplemental.failed is original_failed


@pytest.mark.parametrize('legacy', [False, True])
def test_feature_validation_preserves_other_extractors(monkeypatch, legacy):
    from src import cache
    from src.generator.fewshot_ranker import supplemental
    if legacy:
        monkeypatch.delattr(supplemental, 'failed')
    calls = []

    def extract(*args):
        calls.append(args)
        return cache.response_valid('non-feature response')

    monkeypatch.setattr(supplemental, 'extract_one', extract)
    arguments = ('directory', 'key', {'kind': 'unrelated'}, object())
    with pack_transport.feature_response_cache_validation():
        assert supplemental.extract_one(*arguments)
    assert calls == [arguments]
    assert supplemental.extract_one is extract


def test_legacy_feature_validation_rejects_bad_responses_without_adding_hooks(tmp_path, monkeypatch):
    from src import cache
    from src.generator.fewshot_ranker import supplemental
    invalid, valid = concern_cache_response(False), concern_cache_response(True)
    client = CachedFeatureClient([invalid, valid])
    key, request = concern_cache_request(client)

    def extract(directory, key, request, client):
        raw = client.run(request['prompt'], request['schema'])
        return supplemental.concern.validate(json.loads(raw), request)

    monkeypatch.setattr(supplemental, 'extract_one', extract)
    monkeypatch.delattr(supplemental, 'failed')
    with concern_cache_scope(tmp_path):
        with pytest.raises(ConfigError, match='speaker or position'):
            with pack_transport.feature_response_cache_validation():
                supplemental.extract_one(tmp_path, key, request, client)
        assert client.calls == 1
        assert supplemental.extract_one is extract
        assert not hasattr(supplemental, 'failed')
        assert cache.response_valid('non-feature response')
        with pack_transport.feature_response_cache_validation():
            assert supplemental.extract_one(tmp_path, key, request, client) == json.loads(valid)
            assert supplemental.extract_one(tmp_path, key, request, client) == json.loads(valid)
            assert client.calls == 2
            assert not hasattr(supplemental, 'failed')
    assert supplemental.extract_one is extract
    assert not hasattr(supplemental, 'failed')


@pytest.mark.parametrize('kind,folder', [('pack', 'judge_eval'), ('experiment', 'experiments'),
                                       ('promotion', 'experiments'), ('features', 'experiments')])
@pytest.mark.parametrize('explicit_env', [False, True])
def test_frozen_launch_keeps_checkpoint_and_rejects_tampering(env, tmp_path, monkeypatch, kind, folder, explicit_env):
    from pathlib import Path
    if explicit_env:
        expected_env = tmp_path / 'connections.env'
        monkeypatch.setenv('DH_ENV_FILE', str(expected_env))
    else:
        monkeypatch.delenv('DH_ENV_FILE', raising=False)
        expected_env = Path(pack_transport.__file__).resolve().parents[2] / '.env'
    source = tmp_path / 'source'
    (source / 'src').mkdir(parents=True)
    (source / 'src/module.py').write_text('value = 1\n')
    directory = env.root / folder / 'synthetic'
    snapshot = runtime.freeze(directory, root=source)
    checkpoint = directory / 'building.json'
    checkpoint.write_text(json.dumps({'rows': [{'case_id': 'saved'}]}))
    before = {p: p.read_bytes() for p in [checkpoint, directory / 'runtime.json',
                                        snapshot / 'manifest.json', snapshot / 'src/module.py']}
    calls = []
    monkeypatch.setattr(pack_transport.subprocess, 'call', lambda argv, **kw: calls.append((argv, kw)) or 0)
    options = {'feature_output': tmp_path / 'features'} if kind == 'features' else {}
    assert pack_transport.launch(directory, instance=env.root.name, job=directory.name,
                                 workers=4, seconds=180, kind=kind, **options) == 0
    argv, kw = calls[0]
    assert argv[argv.index('--kind') + 1] == kind
    assert argv[argv.index('--snapshot') + 1] == str(snapshot)
    if kind == 'features':
        assert argv[argv.index('--feature-output') + 1] == str(tmp_path / 'features')
    assert kw['env']['DH_SETTINGS_FILE'] == str(snapshot / 'settings.snapshot.yaml')
    assert kw['env']['DH_ENV_FILE'] == str(expected_env)
    assert all(p.read_bytes() == value for p, value in before.items())
    (snapshot / 'src/module.py').write_text('value = 2\n')
    with pytest.raises(Exception, match='source changed'):
        pack_transport.launch(directory, instance=env.root.name, job=directory.name,
                              workers=4, seconds=180, kind=kind, **options)
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['gen_ab', 'judge_eval'])
def test_saved_promotion_calls_only_original_gates_and_propagates_rejection(env, monkeypatch, kind):
    from src.iteration import experiment, promote
    directory = env.root / 'experiments/completed'
    monkeypatch.setattr(experiment, 'spec_of', lambda path: {'kind': kind})
    monkeypatch.setattr(versions, 'load_pointers', lambda: {'production_gen': 'g-old'})
    monkeypatch.setattr(pack_transport.runpy, 'run_path', lambda *a, **k: pytest.fail('must not run worker'))
    monkeypatch.setattr(llm.ChatClient, 'chat', lambda *a, **k: pytest.fail('must not request'))
    calls = []
    def accepted(name):
        calls.append(name)
        return {'production_gen': 'g-new'}
    selected = 'promote_gen' if kind == 'gen_ab' else 'promote_judge'
    other = 'promote_judge' if kind == 'gen_ab' else 'promote_gen'
    monkeypatch.setattr(promote, selected, accepted)
    monkeypatch.setattr(promote, other, lambda *a, **k: pytest.fail('wrong promotion kind'))
    assert pack_transport.promote_saved(directory) == {'production_gen': 'g-new'}
    assert calls == ['completed']
    def rejected(name):
        raise ConfigError('data, evidence or production basis changed')
    monkeypatch.setattr(promote, selected, rejected)
    with pytest.raises(ConfigError, match='production basis changed'):
        pack_transport.promote_saved(directory)
