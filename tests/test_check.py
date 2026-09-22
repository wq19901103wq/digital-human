"""Unified checks preserve failures, provenance and request-free operation."""
import copy
import json

import pytest

from scripts import check, evaluate_judge
from src.config import ConfigError
from src.iteration import experiment, learning_guard, runner, versions
from src.iteration.storage import write_json
from src.judge.corrected import CodexJudgeClient
from src.llm import ChatClient
from test_draw_replay import saved
from test_iteration import priv, _use_small_settings
from test_leakage_guards import env
from test_result_cache import codex
from test_corrected_judge import bundle, case
from leakage_support import isolated_legacy_provenance


def test_failed_check_does_not_hide_other_failures(monkeypatch, capsys, tmp_path):
    calls = []
    def command(argv):
        calls.append(argv)
        if 'check_rules.py' in argv[0] or 'pytest' in argv:
            raise RuntimeError('injected failure')
        return 'ok'
    monkeypatch.setattr(check, 'command', command)
    path = tmp_path / 'report.json'
    assert check.main(['--output', str(path), 'code', '--tests', 'tests/test_check.py']) == 1
    report = json.loads(capsys.readouterr().out)
    assert [r['status'] for r in report['checks']] == ['failed', 'passed', 'passed', 'failed']
    assert len(calls) == 4 and json.loads(path.read_text()) == report
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        check.main(['--output', str(path), 'code'])
    assert path.read_bytes() == before and len(calls) == 4


def test_code_default_does_not_run_regression_or_demo(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(check, 'command', lambda argv: calls.append(argv))
    assert check.main(['code', '--staged']) == 0
    assert len(calls) == 3 and not any('pytest' in c for c in calls)
    assert '--staged' in calls[0] and '--staged' in calls[2]


def test_offline_check_forbids_each_model_entrypoint():
    with check.offline():
        for method in (CodexJudgeClient.run, ChatClient.chat, ChatClient._send):
            with pytest.raises(RuntimeError, match='禁止请求'):
                method(None)


def test_saved_observations_use_same_evidence_without_requests(saved):
    before = versions.POINTERS_PATH.read_bytes()
    count = len(saved.calls)
    with check.offline():
        result = check.check_replay(saved.path, saved.pack, saved.baseline, saved.candidate)
    assert result['saved_draws'] == 18
    assert versions.POINTERS_PATH.read_bytes() == before and len(saved.calls) == count


@pytest.mark.parametrize('change', ['case', 'source'])
def test_changed_observations_rejected(saved, change):
    pack = copy.deepcopy(saved.pack)
    if change == 'case':
        pack['rows'][0]['human_reply'] = ['changed']
    else:
        log = next(iter(saved.logs.values()))
        log.path.write_text(log.path.read_text() + ' ')
    with check.offline(), pytest.raises(ConfigError):
        check.check_replay(saved.path, pack, saved.baseline, saved.candidate)


def test_judge_aggregates_data_and_source_failures(saved, monkeypatch, capsys):
    write_json(saved.root / 'judge_eval/pack/pack.json', saved.pack)
    monkeypatch.setattr(versions, 'switch_instance', lambda name: saved.root)
    from src.iteration import datasets
    def changed(*args):
        raise ConfigError('dataset changed')
    def leaked(*args):
        raise ConfigError('learning answer overlap')
    monkeypatch.setattr(datasets, 'assert_pack', changed)
    monkeypatch.setattr(learning_guard, 'snapshot', leaked)
    before, count = versions.POINTERS_PATH.read_bytes(), len(saved.calls)
    assert check.main(['judge', '--instance', 'demo', '--pack', 'pack',
        '--candidate', saved.candidate['id'], '--saved-draw-manifest', str(saved.path)]) == 1
    report = json.loads(capsys.readouterr().out)
    statuses = {c['name']: c['status'] for c in report['checks']}
    assert statuses['development_pack'] == statuses['learning_sources'] == 'failed'
    assert statuses['saved_observations'] == statuses['unchanged_pointers'] == 'passed'
    assert versions.POINTERS_PATH.read_bytes() == before and len(saved.calls) == count


def test_setup_failure_is_machine_readable(priv, monkeypatch, capsys):
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    assert check.main(['judge', '--instance', 'demo', '--pack', 'missing', '--candidate', 'j-0001']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'failed'
    assert result['checks'][0]['name'] == 'validation_setup'


def test_changed_pointers_are_reported(priv, monkeypatch, capsys):
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    def concurrent_update(args, report):
        value = versions.load_pointers()
        write_json(versions.POINTERS_PATH, {**value, 'iteration_judge': 'j-0002'})
    monkeypatch.setattr(check, 'check_experiment', concurrent_update)
    assert check.main(['experiment', '--instance', 'demo']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['checks'][-1]['name'] == 'unchanged_pointers'
    assert result['checks'][-1]['status'] == 'failed'


def test_experiment_missing_independent_votes_rejected(saved, monkeypatch, capsys, isolated_legacy_provenance):
    _use_small_settings(monkeypatch)
    monkeypatch.setattr(versions, 'switch_instance', lambda name: saved.root)
    write_json(saved.root / 'judge_eval/pack/pack.json', saved.pack)
    directory = experiment.create_judge_eval_experiment('development', 'pack', {}, 'replay',
        candidate_ref=saved.candidate['id'], saved_draw_manifest=str(saved.path))
    runner.run_judge_experiment(directory)
    capsys.readouterr()
    count = len(saved.calls)
    args = ['experiment', '--instance', 'demo', '--exp', directory.name]
    assert check.main(args) == 0
    capsys.readouterr()
    path = directory / 'cases.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row['baseline_votes'] = [False]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert check.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'failed' and len(saved.calls) == count


def test_compare_cli_forwards_frozen_candidate_and_replay(monkeypatch, tmp_path):
    calls = []
    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / 'exp'
    monkeypatch.setattr(experiment, 'create_judge_eval_experiment', create)
    monkeypatch.setattr(experiment, 'state_of', lambda _: {'status': 'finished'})
    monkeypatch.setattr(evaluate_judge.sys, 'argv', ['evaluate_judge.py', 'compare',
        '--pack', 'pack', '--candidate', 'j-0002', '--saved-draw-manifest', 'draws.json'])
    evaluate_judge.main()
    assert calls[0][1]['candidate_ref'] == 'j-0002'
    assert calls[0][1]['saved_draw_manifest'] == 'draws.json'


def test_real_material_guard_rejects_forged_assets(env, monkeypatch, capsys):
    monkeypatch.setattr(versions, 'switch_instance', lambda name: env.root)
    pack = dict(data_ref='d-test', c0_gen_version='g-0001',
                rows=[{**r, 'ai_replies': ['synthetic']} for r in env.source.roles['judge_development']])
    write_json(env.root / 'judge_eval/pack/pack.json', pack)
    args = ['judge', '--instance', 'demo', '--pack', 'pack', '--candidate', 'j-0001']
    # This source fixture has real history and reviewed mechanical assets. No
    # guard is patched. The hand-made pack intentionally lacks generation proof.
    assert check.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert next(c for c in report['checks'] if c['name'] == 'learning_sources')['status'] == 'passed'
    (env.jdir / 'persona.md').write_text('unproven future information')
    assert check.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert next(c for c in report['checks'] if c['name'] == 'learning_sources')['status'] == 'failed'
    assert env.calls == []


def test_fixed_entry_requires_experiment():
    with pytest.raises(SystemExit):
        check.main(['experiment', '--instance', 'demo', '--gate', 'fixed-entry'])


def test_material_diagnostic_separates_chat_exposure_from_answers_and_time():
    from src.iteration.material_compatibility import summarize
    cases = [dict(case_id=str(i), source_span=dict(chat_id=chat),
                  input_cutoff=dict(timestamp=100), reply_message_ids=['answer-' + str(i)])
             for i, chat in enumerate(['unseen-a', 'unseen-a', 'unseen-b', 'familiar'])]
    old = dict(chat_id='unseen-a', information_end=20, message_ids=['old-message'])
    result = summarize(cases, ['unseen-a', 'unseen-b'], dict(judge=[old], ranker=[old]))
    assert result['unseen_cases'] == 3 and result['unseen_cases_exposed'] == 2
    assert result['unseen_cases_unexposed'] == 1
    assert result['answer_id_overlap_cases'] == result['time_conflict_cases'] == 0
    assert result['source_conflicts'] is False
    # Answer/time conflicts also matter for familiar cases, independently of exposure.
    bad = dict(chat_id='familiar', information_end=100, message_ids=['answer-3'])
    result = summarize(cases, ['unseen-a', 'unseen-b'], dict(judge=[old], ranker=[bad]))
    assert result['answer_id_overlap_cases'] == 1 and result['time_conflict_cases'] == 4
    assert result['unseen_cases_exposed'] == 2
    assert result['source_conflicts'] is True


def test_material_sources_are_bound_and_use_full_source_chat_id(tmp_path):
    from src.config import sha256_file
    from src.iteration import material_compatibility as materials
    directory = tmp_path / 'generator'
    inventory = tmp_path / 'inventory'
    model = tmp_path / 'architecture' / 'model'
    row = dict(source_span=dict(chat_id='actual-chat', end_timestamp=20),
               chat_id='wrong-chat', context_message_ids=['context'], reply_message_ids=['reply'])
    paths = {
        model.parent / 'manifest.json': dict(source=str(tmp_path / 'training')),
        tmp_path / 'training/manifest.json': dict(inventory=str(inventory)),
        inventory / 'labeling/manifest.json': dict(data='old', targets=[dict(target_id='target')]),
        inventory / 'targets/target.json': dict(target_id='target', target=row,
            candidates=[dict(example=dict(row, id='example'))]),
    }
    for path, value in paths.items():
        write_json(path, value)
    evidence = {str(p): sha256_file(p) for p in paths}
    proof = dict(evidence_files=evidence, model_directory=str(model), data_ref='old',
                 samples=dict(contexts=1, unique_examples=1), information_end=20)
    proof_path = directory / 'ranker/provenance.json'
    write_json(proof_path, proof)
    write_json(directory / 'learning.json', dict(verified=True, evidence_files=evidence,
        information_end=20, asset_files={'ranker/provenance.json': sha256_file(proof_path)}))
    with check.offline():
        rows, data = materials.ranker_sources(directory, {})
        assert data == 'old'
        assert rows['ranker_targets'][0]['chat_id'] == rows['ranker_examples'][0]['chat_id'] == 'actual-chat'
        target = inventory / 'targets/target.json'
        target.write_text(target.read_text() + ' ')
        with pytest.raises(ConfigError, match='来源内容变化'):
            materials.ranker_sources(directory, {})


@pytest.mark.parametrize('answer_overlap', [0, 1])
def test_material_conflicts_are_reported_without_changing_pointers(priv, monkeypatch, capsys, answer_overlap):
    from src.iteration import material_compatibility
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    monkeypatch.setattr(versions, 'generator_dir', lambda ref: priv / ref)
    monkeypatch.setattr(versions, 'judge_dir', lambda ref: dict(dir=priv / ref))
    monkeypatch.setattr(versions, 'data_version_dir', lambda ref: priv / ref)
    monkeypatch.setattr(material_compatibility, 'inspect', lambda *args:
        dict(source_conflicts=bool(answer_overlap), evaluation_authorized=False,
             summary=dict(unseen_cases_exposed=2, answer_id_overlap_cases=answer_overlap)))
    before = versions.POINTERS_PATH.read_bytes()
    assert check.main(['materials', '--instance', 'demo', '--data', 'new',
                       '--judge', 'judge', '--generator', 'gen']) == int(bool(answer_overlap))
    report = json.loads(capsys.readouterr().out)
    assert report['material_compatibility']['summary']['unseen_cases_exposed'] == 2
    assert report['status'] == ('failed' if answer_overlap else 'passed')
    assert report['model_requests_prohibited']
    assert report['checks'][-1]['status'] == 'passed'
    assert versions.POINTERS_PATH.read_bytes() == before


@pytest.mark.parametrize('conflict', ['none', 'answer', 'future', 'changed_source'])
def test_changed_data_allows_familiar_history_but_rejects_leaks(tmp_path, monkeypatch, conflict):
    from src.config import sha256_file
    from src.iteration import material_compatibility as materials
    case = dict(case_id='target', source_span=dict(chat_id='familiar'),
                input_cutoff=dict(timestamp=100), reply_message_ids=['answer'])
    path = tmp_path / 'development.jsonl'
    path.write_text(json.dumps(case) + '\n')
    source = tmp_path / 'source.json'
    write_json(source, dict(chat_id='familiar', information_end=100 if conflict == 'future' else 20,
                           message_ids=['answer' if conflict == 'answer' else 'earlier-message']))
    monkeypatch.setattr(versions, 'data_version_dir', lambda ref: tmp_path)
    monkeypatch.setattr(materials.datasets, 'case_path', lambda data, role: path)
    def load(directory, bindings):
        bindings[str(source)] = sha256_file(source)
        row = json.loads(source.read_text())
        if conflict == 'changed_source':
            source.write_text(source.read_text() + ' ')
        return dict(training=[row]), 'original-data'
    with check.offline():
        if conflict == 'none':
            assert materials.require_current('new-data', tmp_path, load)['source_conflicts'] is False
        else:
            with pytest.raises(ConfigError, match='来源发生变化' if conflict == 'changed_source' else '答案或未来'):
                materials.require_current('new-data', tmp_path, load)


@pytest.mark.parametrize('missing', [False, True])
def test_material_reconstruction_uses_formal_adapter_without_requests(priv, monkeypatch, capsys, missing):
    from contextlib import contextmanager
    from src.iteration import gbdt_evidence, material_compatibility, pack_transport, runtime
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    monkeypatch.setattr(versions, 'generator_dir', lambda ref: priv / ref)
    monkeypatch.setattr(versions, 'judge_dir', lambda ref: dict(dir=priv / ref))
    monkeypatch.setattr(versions, 'data_version_dir', lambda ref: priv / ref)
    monkeypatch.setattr(material_compatibility, 'inspect', lambda *args: dict(source_conflicts=False))
    calls = []
    @contextmanager
    def adapter(module, resolver):
        assert module is gbdt_evidence and resolver is runtime
        calls.append('enter')
        yield
        calls.append('exit')
    def reconstruct(data, directories):
        assert calls == ['enter'] and data == 'data'
        assert directories == [priv / 'judge', priv / 'gen']
        calls.append('guard')
        if missing:
            raise ConfigError('frozen input missing')
        return {'verified': True}
    monkeypatch.setattr(pack_transport, 'archived_gbdt_paths', adapter)
    monkeypatch.setattr(learning_guard, 'require_materials', reconstruct)
    before = versions.POINTERS_PATH.read_bytes()
    assert check.main(['materials', '--instance', 'demo', '--data', 'data',
        '--judge', 'judge', '--generator', 'gen', '--reconstruct']) == int(missing)
    report = json.loads(capsys.readouterr().out)
    result = next(row for row in report['checks'] if row['name'] == 'material_reconstruction')
    assert result['status'] == ('failed' if missing else 'passed')
    assert calls == ['enter', 'guard'] + ([] if missing else ['exit'])
    assert report['model_requests_prohibited'] and versions.POINTERS_PATH.read_bytes() == before
