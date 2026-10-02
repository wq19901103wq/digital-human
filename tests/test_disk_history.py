"""Disk storage must retain the authenticated frozen retriever's exact behavior."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path

import pytest

from leakage_support import historical
from test_leakage_guards import env as env
from test_fewshot_selection import generator
from src.bootstrap import history
from src.config import ConfigError, sha256_file
from src.generator import disk_history, few_shot, history_sources
from src.iteration import pack_transport
from src.iteration.storage import write_json


FLAGS = ('enable_clarification', 'enable_precise_situation', 'prefer_same_chat_after_situation',
         'prefer_same_chat_after_exact_move', 'prioritize_latest_turn',
         'avoid_concrete_examples_for_sparse_query', 'prefer_response_move_before_relevance',
         'prefer_latest_turn_form', 'prefer_group_participant_structure',
         'prefer_multi_message_reply_coverage', 'increase_semantic_topic_weight',
         'enable_information_gap_response_move')


def retriever(source, flags=()):
    result = few_shot.PersonaFewShotRetriever(source.directory / 'fewshot_pool.jsonl')
    for name in flags:
        setattr(result, name, not getattr(result, name))
    assert result.is_approved()
    return result


def queries(source):
    for case in source.roles['development'] + source.roles['fixed_test']:
        for trace in (False, True):
            for situation in (False, True):
                yield dict(query='\n'.join(row['text'] for row in case['context']),
                           chat_name=case['chat_name'], chat_id=case['source_chat_id'],
                           is_group=case['chat_type'] == 'group', limit=3,
                           exclude_ids={case['case_id']}, history_case=case,
                           current_context_messages=case['context'],
                           include_retrieval_features=trace, use_situation=situation)


@pytest.mark.parametrize('flags', [(), *[(name,) for name in FLAGS], FLAGS])
def test_selected_rows_features_render_and_order_match_memory(tmp_path, flags):
    source = historical(tmp_path / 'data')
    baseline = retriever(source, flags)
    requests = list(queries(source))
    expected = [baseline.retrieve(**request) for request in requests]
    expected_features = {name: dict(getattr(baseline, name)) for name in disk_history.FEATURES}
    with disk_history.disk_storage(few_shot):
        candidate = retriever(source, flags)
        actual = [candidate.retrieve(**request) for request in requests]
        assert actual == expected
        assert all(type(row) is dict for rows in actual for row in rows)
        assert [candidate.render_selected(rows) for rows in actual] == [
            baseline.render_selected(rows) for rows in expected]
        for name, values in expected_features.items():
            assert dict(getattr(candidate, name)) == values
            for identity, value in values.items():
                if isinstance(value, dict):
                    assert list(getattr(candidate, name)[identity].items()) == list(value.items())
        for request, rows in zip(requests, actual):
            assert all(row['id'] != request['history_case']['case_id'] and
                       row['source_span']['end_timestamp'] < request['history_case']['source_span']['start_timestamp']
                       for row in rows)


def test_prompt_and_request_identity_unchanged_and_scope_restored(env):
    baseline = generator(env)
    cases = env.source.roles['development'] + env.source.roles['fixed_test']
    expected = [baseline.build_prompt(case) for case in cases]
    original = (history_sources.HistorySources.__init__, few_shot.PersonaFewShotRetriever._load,
                few_shot.PersonaFewShotRetriever.retrieve)
    with disk_history.disk_storage(few_shot), pack_transport.history_feature_cache(few_shot):
        candidate = generator(env)
        assert [candidate.build_prompt(case) for case in cases] == expected
        assert env.calls == []
        reader = candidate._retriever._disk_reader
    assert (history_sources.HistorySources.__init__, few_shot.PersonaFewShotRetriever._load,
            few_shot.PersonaFewShotRetriever.retrieve) == original
    assert reader.stream.closed
    with pytest.raises(Exception, match='closed database'):
        reader.connection.execute('SELECT 1')


def test_chat_sequence_index_slices_and_parallel_reads_are_bounded(tmp_path):
    source = historical(tmp_path / 'data')
    expected = history_sources.HistorySources(source.directory).chats['chat-0']
    with disk_history.disk_storage(few_shot):
        stored = history_sources.load(source.directory)
        actual = stored.chats['chat-0']
        assert not isinstance(actual, list)
        assert list(actual) == expected
        for selection in (slice(None), slice(8, 25), slice(None, None, -1), slice(30, 5, -3), slice(4, 4)):
            assert actual[selection] == expected[selection]
        assert actual[-1] == expected[-1]
        with pytest.raises(IndexError):
            actual[len(actual)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(actual.__getitem__, range(len(actual)))) == expected
        assert len(stored._disk_reader.cache) <= disk_history.CAPACITY


def test_duplicate_ids_across_chunks_preserve_last_wins_features_and_buckets(tmp_path, monkeypatch):
    source = historical(tmp_path / 'data')
    rows = source.examples * 3
    history.write_rows(source.directory / 'fewshot_pool.jsonl', rows)
    write_json(source.directory / 'report.json', {'review_status': 'approved',
               'history_policy': 'complete_before_input_v1',
               'examples_sha256': sha256_file(source.directory / 'fewshot_pool.jsonl')})
    monkeypatch.setattr(disk_history, 'CAPACITY', 3)
    baseline = retriever(source)
    with disk_history.disk_storage(few_shot):
        candidate = retriever(source)
        assert [dict(row) for row in candidate._rows] == baseline._rows
        for name in disk_history.FEATURES:
            assert dict(getattr(candidate, name)) == getattr(baseline, name)
        for name in disk_history.BUCKETS:
            assert {key: [dict(row) for row in value] for key, value in getattr(candidate, name).items()} == getattr(baseline, name)
        list(candidate._terms_by_id.values())
        assert len(candidate._disk_reader.features) <= 3
        assert len(candidate._disk_reader.cache) <= 3


@pytest.mark.parametrize('changed', ['messages.jsonl', 'fewshot_pool.jsonl', 'purposes.json', 'report.json'])
def test_source_mutation_still_rejected_and_readers_closed(tmp_path, changed):
    source = historical(tmp_path / 'data')
    with pytest.raises(ConfigError, match='发生变化'):
        with disk_history.disk_storage(few_shot):
            candidate = retriever(source)
            reader = candidate._disk_reader
            path = source.directory / changed
            path.write_text(path.read_text() + '\n')
            candidate.retrieve(**next(queries(source)))
    assert reader.stream.closed


def test_wrong_source_fields_cannot_publish_index(tmp_path):
    source = historical(tmp_path / 'data')
    rows = copy.deepcopy(source.examples)
    rows[0]['context'][0] = 'unverified replacement'
    history.write_rows(source.directory / 'fewshot_pool.jsonl', rows)
    write_json(source.directory / 'report.json', {'review_status': 'approved',
               'history_policy': 'complete_before_input_v1',
               'examples_sha256': sha256_file(source.directory / 'fewshot_pool.jsonl')})
    with disk_history.disk_storage(few_shot), pytest.raises(ConfigError, match='来源'):
        retriever(source)
    assert not list((tmp_path / '.cache/disk-history').glob('*.building'))
    for path in (tmp_path / '.cache/disk-history').glob('*.sqlite3'):
        assert path.with_suffix('.json').exists()


@pytest.mark.parametrize('suffix', ['\n', 'not-json\n'])
def test_invalid_or_blank_example_line_rejected_without_partial_publication(tmp_path, suffix):
    source = historical(tmp_path / 'data')
    path = source.directory / 'fewshot_pool.jsonl'
    path.write_text(path.read_text() + suffix)
    write_json(source.directory / 'report.json', {'review_status': 'approved',
               'history_policy': 'complete_before_input_v1', 'examples_sha256': sha256_file(path)})
    with disk_history.disk_storage(few_shot), pytest.raises(ValueError):
        retriever(source)
    assert not list((tmp_path / '.cache/disk-history').glob('*.building'))


def test_warm_index_reuse_and_checksum_corruption_rejected(tmp_path):
    source = historical(tmp_path / 'data')
    with disk_history.disk_storage(few_shot):
        retriever(source)
    root = tmp_path / '.cache/disk-history'
    stamps = {path: path.stat().st_mtime_ns for path in root.glob('*.sqlite3')}
    assert len(stamps) == 2
    with disk_history.disk_storage(few_shot):
        retriever(source)
    assert {path: path.stat().st_mtime_ns for path in stamps} == stamps
    next(iter(stamps)).with_suffix('.json').write_text('{}')
    with disk_history.disk_storage(few_shot), pytest.raises(ValueError, match='checksum'):
        retriever(source)


def test_heldout_and_exclusions_match_memory(tmp_path):
    source = historical(tmp_path / 'data')
    path = source.directory / 'purposes.json'
    purpose = json.loads(path.read_text())
    purpose['protocol']['unseen_chat_ids'] = ['chat-0']
    write_json(path, purpose)
    request = next(queries(source))
    request['history_case'] = {**request['history_case'], 'history_excluded_chat_ids': ['chat-0']}
    baseline = retriever(source)
    assert baseline.retrieve(**request) == []
    with disk_history.disk_storage(few_shot):
        assert retriever(source).retrieve(**request) == []


def test_live_adapter_resolves_outside_snapshot():
    live = pack_transport.live_disk_history()
    assert Path(live.__file__).resolve() == Path(disk_history.__file__).resolve()
    assert live.MODE == disk_history.MODE
