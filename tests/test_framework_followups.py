"""Recovery, source-preserving derivation, migration and live workflow contracts."""
import json
import os
from collections import Counter
from datetime import datetime, timezone

import pytest

from leakage_support import historical
from test_leakage_guards import env as _env_fixture

from src.bootstrap import history
from src.config import ConfigError, sha256_file
from src.generator.history_sources import load
from src.iteration import baseline, branches, datasets, journal, learning_guard, protocol, task_state, training_evidence, versions
from src.iteration.storage import write_json

env = _env_fixture


def test_branch_derivation_keeps_verifiable_learning_assets(env):
    proposal = branches._snapshot('gen', 'g-0001')
    proposal['config']['max_shots_per_case'] = 2
    ref = branches._create_version('gen', proposal, 'd-test')
    target = versions.generator_dir(ref)
    assert (target / 'learning.json').read_bytes() == (env.gdir / 'learning.json').read_bytes()
    assert learning_guard.require_materials('d-test', [target])['promotion_eligible']
    assert target.name != env.gdir.name


@pytest.mark.parametrize('ending', [b'{"case_id":', b'{"case_id":"next', b'{"case_id":"\xe4\xb8'])
def test_recovery_preserves_completed_records_and_exact_failed_tail(tmp_path, ending):
    path = tmp_path / 'cases.jsonl'
    original = b'{"case_id":"done","status":"ok"}\n' + ending
    path.write_bytes(original)
    assert set(protocol.summarize_final_records(path)[0]) == {'done'}
    archive = journal.recover(path)
    assert archive.read_bytes() == original
    with path.open('ab') as stream:
        stream.write(b'{"case_id":"next","status":"ok"}\n')
    assert set(protocol.summarize_final_records(path)[0]) == {'done', 'next'}
    assert journal.recover(path) is None


def test_corrupt_middle_is_never_recovered(tmp_path):
    path = tmp_path / 'cases.jsonl'
    blob = b'{"case_id":\n{"case_id":"last"}\n'
    path.write_bytes(blob)
    with pytest.raises(ConfigError):
        journal.recover(path)
    assert path.read_bytes() == blob
    assert not list(tmp_path.glob('*.bak'))


def test_valid_last_row_without_newline_is_not_concatenated_on_resume(tmp_path):
    path = tmp_path / 'cases.jsonl'
    path.write_bytes(b'{"case_id":"last"}')
    archive = journal.recover(path)
    assert archive.read_bytes() == b'{"case_id":"last"}'
    assert path.read_bytes().endswith(b'\n')


def test_pid_reuse_and_exit_change_visible_status(monkeypatch):
    state = {'status': 'running', 'phase': 'extracting', 'pid': os.getpid() + 1000000, 'process_start': 'old'}
    monkeypatch.setattr(task_state.os, 'kill', lambda pid, signal: None)
    monkeypatch.setattr(task_state, 'process_start', lambda pid: 'new')
    assert task_state.resolve(state)['status'] == 'interrupted'
    assert state['status'] == 'running'  # Reading does not rewrite a producer checkpoint.


def test_resumed_task_clears_obsolete_error_and_records_identity(tmp_path):
    write_json(tmp_path / 'state.json', {'status': 'stopped', 'error': 'old attempt failed'})
    state = task_state.update(tmp_path, 'extracting')
    assert 'error' not in state and state['process_start']
    assert task_state.read(tmp_path)['producer_alive']


@pytest.mark.parametrize('updated, stamp', [
    (1789402806.0, 1789402806.0), ('1789402806.0', 1789402806.0),
    ('2026-09-14T17:40:06Z', datetime(2026, 9, 14, 17, 40, 6, tzinfo=timezone.utc).timestamp()),
    ('2026-09-15 01:40:06', datetime(2026, 9, 15, 1, 40, 6).timestamp()),
    (None, None), ('broken', None), (float('nan'), None)])
def test_live_status_accepts_legacy_timestamps_without_mutating_records(updated, stamp):
    state = {'status': 'running', 'phase': 'extracting', 'pid': os.getpid(), 'updated_at': updated}
    now = (stamp or 1789402806.0) + 100
    result = task_state.resolve(state, now=now)
    assert result['producer_alive'] and result['status'] == 'running'
    if stamp is None:
        assert result['heartbeat_age_seconds'] is None
    else:
        assert result['heartbeat_age_seconds'] == 100
    assert state['updated_at'] is updated


def test_migration_is_explicit_atomic_and_not_a_performance_promotion(env):
    previous = versions.load_pointers()
    old = {**previous, 'data': 'd-legacy'}
    versions.save_pointers(old)
    receipt = baseline.prepare('d-test', 'g-0001', 'j-0001', reason='建立有来源证明的起点')
    assert versions.load_pointers() == old
    assert json.loads(receipt.read_text())['before'] == old
    assert json.loads(receipt.read_text())['performance_claim'] is False
    adopted = baseline.apply(receipt.stem)
    assert adopted['data'] == 'd-test'
    assert baseline.apply(receipt.stem) == adopted


@pytest.mark.parametrize('change', ['pointer', 'material'])
def test_migration_rechecks_receipt_before_switch(env, change):
    receipt = baseline.prepare('d-test', 'g-0001', 'j-0001', reason='核验迁移')
    if change == 'pointer':
        versions.save_pointers({**versions.load_pointers(), 'iteration_gen': 'g-newer'})
    else:
        (env.gdir / 'persona.md').write_text('unproven future fact')
    before = versions.load_pointers()
    with pytest.raises(ConfigError):
        baseline.apply(receipt.stem)
    assert versions.load_pointers() == before


def test_declaring_future_rows_as_training_cannot_bypass_time_guard(tmp_path):
    data = historical(tmp_path / 'data')
    path = data.directory / 'judge_training.jsonl'
    history.write_rows(path, data.roles['development'])
    purpose = json.loads((data.directory / 'purposes.json').read_text())
    purpose['roles']['judge_training']['sha256'] = sha256_file(path)
    write_json(data.directory / 'purposes.json', purpose)
    with pytest.raises(ConfigError, match='未来'):
        load(data.directory).role('judge_training')


def test_fixed_evaluation_does_not_expand_static_learning_window(env):
    record = json.loads((env.gdir / 'learning.json').read_text())
    record['information_end'] = 140000
    write_json(env.gdir / 'learning.json', record)
    assert not datasets.static_sources(env.source.directory, [env.gdir], 'fixed_test')['promotion_eligible']


def test_generic_source_audit_uses_instance_roles_not_experiment_name_or_counts(env, tmp_path):
    directory = tmp_path / 'another-study'
    train, dev = env.source.roles['judge_training'], env.source.roles['judge_development']
    write_json(directory / 'sources.json', {'train': train, 'development': dev})
    write_json(directory / 'source_audit.json', {'reference_spans': []})
    write_json(directory / 'reference.json', {'facts': [], 'examples': []})
    write_json(directory / 'model_template.json', {'feature_system': {'context_schema':
        {'known_group_names': [], 'known_group_members': []}}, 'final_model': {'coefficients': [0]}})
    class Evidence:
        counts = Counter()
        def document(self, path):
            return json.loads(path.read_text())
    spec = {'inputs': {}, 'data_ref': 'd-test', 'generator_ref': 'g-0001',
            'training_total': len(train), 'evaluation_total': len(dev),
            'purpose_snapshot': datasets.snapshot(env.source.directory)}
    assert training_evidence.sources(directory, spec, Evidence())['train'] == train
    # Adding arbitrary facts cannot be certified by a source flag.
    write_json(directory / 'reference.json', {'facts': ['future answer'], 'examples': []})
    with pytest.raises(ConfigError):
        training_evidence.sources(directory, spec, Evidence())
