"""Material changes and unproven cache origins must block historical comparisons."""
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from scripts.legacy import repair_history_judge as job
from scripts.legacy import verify_repaired_history_judge as audit
from src.config import ConfigError, sha256_file
from src.iteration import gates, learning_guard as guard, runner, versions
from src.iteration.storage import write_json
from leakage_support import mechanical


def material(tmp_path, monkeypatch):
    data, asset = tmp_path / 'data', tmp_path / 'asset'
    write_json(data / 'purposes.json', {'protocol': {'development_start': 100}})
    write_json(tmp_path / 'source.json', {'verified_span_end': 99})
    record = mechanical(asset)
    record['evidence_files'] = {str(tmp_path / 'source.json'): sha256_file(tmp_path / 'source.json')}
    write_json(asset / 'learning.json', record)
    monkeypatch.setattr(versions, 'data_version_dir', lambda ref: data)
    return asset, record


@pytest.mark.parametrize('change', ['answer', 'extra', 'future', 'proof', 'missing'])
def test_learning_material_changes_block_reuse(tmp_path, monkeypatch, change):
    asset, record = material(tmp_path, monkeypatch)
    assert guard.require_materials('d', [asset])['promotion_eligible'] is True
    if change == 'answer':
        write_json(asset / 'reference.json', {'facts': ['evaluation answer']})
    elif change == 'extra':
        (asset / 'scenarios/extra.md').write_text('unverified future fact')
    elif change == 'future':
        record['information_end'] = 100
        write_json(asset / 'learning.json', record)
    elif change == 'proof':
        write_json(tmp_path / 'source.json', {'verified_span_end': 101})
    else:
        (asset / 'learning.json').unlink()
    with pytest.raises(ConfigError):
        guard.require_materials('d', [asset])


def test_resume_json_metadata_and_real_change(tmp_path):
    path = tmp_path / 'training.json'
    value = {'rows': [{'metadata': {'recent_speakers': ('a', 'b')}, 'human_option': 'A'}]}
    job.save_once(path, value)
    before = path.read_bytes()
    job.save_once(path, value)
    assert path.read_bytes() == before
    value['rows'][0]['human_option'] = 'B'
    with pytest.raises(ConfigError):
        job.save_once(path, value)


def test_generator_provenance_is_copied_and_changes_fingerprint(tmp_path):
    source = tmp_path / 'source'
    (source / 'scenarios').mkdir(parents=True)
    (source / 'persona.md').write_text('mechanical instructions')
    (source / 'scenarios/friend.md').write_text('private chat')
    cfg = {'llm': {'model': 'same'}}
    legacy = versions.generator_behavior(cfg, source)
    write_json(source / 'learning.json', {'verified': True})
    assert versions.generator_behavior(cfg, source) != legacy
    root = tmp_path / 'versions'
    versions.create_generator_version(cfg, 'd', root=root, source_dir=source, version_dir_name='g-test')
    assert (root / 'g-test/learning.json').read_bytes() == (source / 'learning.json').read_bytes()
    assert versions.generator_behavior(cfg, root / 'g-test') == versions.generator_behavior(cfg, source)


def test_finished_repaired_run_still_checks_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.experiment, 'spec_of', lambda path: {'kind': runner.experiment.KIND_JUDGE_EVAL})
    monkeypatch.setattr(runner.experiment, 'state_of', lambda path: {'status': 'finished'})
    def reject(spec):
        raise ConfigError('material changed')
    monkeypatch.setattr(guard, 'verify_repair', reject)
    with pytest.raises(ConfigError, match='material changed'):
        runner.run_judge_experiment.__wrapped__(tmp_path)


def test_clean_comparator_cannot_be_production_adoption_evidence(tmp_path):
    write_json(tmp_path / 'spec.json', {'material_repair': {'adoption_allowed': False}})
    with pytest.raises(ConfigError):
        gates.evidence(tmp_path)


def test_parsed_initial_cache_requires_actual_request_proof(tmp_path):
    cfg = {'provider': 'codex_cli', 'model': 'luna', 'reasoning_effort': 'low', 'codex_cli_version': 'version'}
    result = {'human_option': 'A', 'reason': 'text'}
    entry = {'layer': 'judge_initial', 'identity': {'prompt': 'prompt', 'client': cfg}, 'value': result,
             'origin': {'experiment_path': str(tmp_path), 'trace_ref': 'trace', 'operation_index': 0}}
    op = {'round': 0, 'events': []}
    evidence = SimpleNamespace(entries=lambda op, cid: [entry], counts=Counter(),
                               document=lambda path: {'case_id': 'case', 'operations': [op]})
    with pytest.raises(ValueError, match='no actual initial prompt evidence'):
        audit.initial_request(op, 'case', 'prompt', result, cfg, evidence)
    op['events'] = [{'kind': 'codex', 'request': {'prompt': 'prompt', 'schema': None, 'config': cfg},
                     'status': 'ok', 'response': {'text': json.dumps(result)}}]
    parsed = audit.rt._parse_api_judge_response(json.dumps(result))
    entry['value'] = parsed
    audit.initial_request(op, 'case', 'prompt', parsed, cfg, evidence)
