"""Read generation provenance from its actual cache, including nested development."""
import sqlite3
from types import SimpleNamespace

import pytest

from src.iteration.training_evidence import generation_cache
from src.iteration.training_evidence import SavedCache
from src import cache, tracing


@pytest.mark.parametrize('prefix', ['/src/', '/src/digital_human/'])
def test_renamed_frozen_code_keeps_original_digest(prefix):
    from src.config import ConfigError
    from src.iteration.training_evidence import input_code_hash
    inputs = {'/old' + prefix + 'judge/rpa_v1.py': 'historical-digest'}
    assert input_code_hash({'inputs': inputs}, 'judge/corrected_v1.py') == 'historical-digest'
    inputs['/new/src/judge/corrected_v1.py'] = 'another-digest'
    with pytest.raises(ConfigError, match='ambiguous'):
        input_code_hash({'inputs': inputs}, 'judge/corrected_v1.py')


def test_unrecognized_frozen_code_is_rejected():
    from src.config import ConfigError
    from src.iteration.training_evidence import input_code_hash
    with pytest.raises(ConfigError, match='missing'):
        input_code_hash({'inputs': {'/other/rpa_v1.py': 'digest'}}, 'judge/corrected_v1.py')


def test_nested_generation_cache_is_read_and_original_reader_restored(tmp_path):
    instance = tmp_path / 'instance'
    cache.Store(instance / '.cache')
    previous = SavedCache(instance / '.cache/results.sqlite3')
    evidence = SimpleNamespace(store=previous)
    directory = instance / 'judge_training/study/development'
    trace = tracing.CaseTrace(directory, {'case_id': 'nested'}, {'dataset': 'development'})
    with trace.operation('generation', 'training', 0, {}) as op:
        cache.memo('generation', {'input': 'same'}, lambda: {'replies': ['saved']})
    key = next(e['data']['key'] for e in op['events'] if e['kind'] == 'cache_lookup')
    try:
        assert previous.get(key) is None
        with pytest.raises(ValueError, match='audit failed'):
            with generation_cache(evidence, directory):
                scoped = evidence.store
                assert scoped.get(key)['value'] == {'replies': ['saved']}
                with pytest.raises(sqlite3.OperationalError, match='readonly'):
                    scoped.db.execute('DELETE FROM entries')
                raise ValueError('audit failed')
        assert evidence.store is previous
        assert previous.get(key) is None
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            scoped.get(key)
    finally:
        previous.db.close()


def test_missing_cache_is_not_created_or_replaced_by_another_store(tmp_path):
    evidence = SimpleNamespace(store=object())
    previous = evidence.store
    directory = tmp_path / 'judge_training/study/development'
    with pytest.raises(sqlite3.OperationalError):
        with generation_cache(evidence, directory):
            pytest.fail('Missing cache must stop the audit')
    assert evidence.store is previous
    assert not (directory.parent.parent / '.cache/results.sqlite3').exists()
