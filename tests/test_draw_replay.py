"""Saved observations preserve request provenance and all independent votes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import cache
from src.config import ConfigError, sha256_file
from src.iteration import branches, draw_replay as replay, experiment, gates, runner, versions
from src.iteration.storage import write_json
from src.iteration.training_evidence import Evidence
from src.judge import corrected, corrected_v1 as rt
from test_corrected_judge import bundle, case
from test_result_cache import codex, judge, corrected_scorer
from test_iteration import priv, _use_small_settings
from test_gbdt_judge import document
from leakage_support import isolated_legacy_provenance


class Collector(Evidence):
    def __init__(self):
        super().__init__()
        self.entries_seen = {}

    def cache_entry(self, *args):
        entry = super().cache_entry(*args)
        if entry is not None:
            self.entries_seen[entry['key']] = replay.cache_entry_digest(entry)
        return entry


@pytest.fixture
def saved(priv, bundle, case, codex, request):
    cfg, directory = bundle
    cfg = corrected_scorer(bundle).config
    assets = {n: (directory / n).read_bytes() for n in ['prompt.md', *cfg['assets']]}
    base_id = versions.create_judge_version(cfg, {}, assets=assets)
    baseline = versions.judge_dir(base_id)
    model = rt.load_formal_judge(directory / 'correction.json')
    assets['scorer.json'] = json.dumps(document(len(model.feature_names))).encode()
    import hashlib
    cfg = {**cfg, 'decision_policy': 'gbdt_only', 'scorer_file': 'scorer.json',
           'assets': {**cfg['assets'], 'scorer.json': hashlib.sha256(assets['scorer.json']).hexdigest()}}
    cand_id = versions.create_judge_version(cfg, {}, assets=assets)
    candidate = versions.judge_dir(cand_id)
    pointers = versions.load_pointers()
    write_json(versions.POINTERS_PATH, {**pointers, 'iteration_judge': base_id, 'production_judge': base_id})
    rows = [{**case, 'case_id': f'case-{i}', 'ai_replies': ['九点']} for i in range(6)]
    pack = dict(data_ref='d-0001', c0_gen_version='g-0001', rows=rows)
    study_path = priv / 'study.json'
    write_json(study_path, dict(data_ref='d-0001', generator_ref='g-0001'))
    scorer = corrected.CorrectedJudge(baseline['config'], baseline['dir'])
    cache.get_store(priv / '.cache')
    evidence = Collector()
    draws, logs = {}, {}
    try:
        evidence.document(study_path)
        for row in rows:
            cid = row['case_id']
            draws[cid] = {}
            for rnd in range(3):
                if getattr(request, 'param', False) and row == rows[0] and rnd:
                    continue
                _, log = judge(priv, scorer, row, 'source', rnd)
                logs[cid, rnd] = log
                reference = dict(path=str(log.path), sha256=sha256_file(log.path), operation=0, round=rnd)
                path = priv / 'draws' / f'{cid}-r{rnd}.json'
                write_json(path, dict(identity=cache.digest(dict(case=row, config=baseline['config'],
                    round=rnd, study=sha256_file(study_path))), reference=reference))
                evidence.document(path)
                replay.verify_draw(row, baseline, reference, rnd, evidence)
                draws[cid][str(rnd)] = dict(path=str(path), study=str(study_path))
        value = dict(schema=1, dataset='development', data_ref='d-0001', generator_ref='g-0001',
            cases_sha256=cache.digest(rows), source_judge=baseline['id'], source_config=baseline['config'],
            draws=draws, inputs=evidence.hashes, cache_entries=evidence.entries_seen)
    finally:
        evidence.store.db.close()
    path = priv / 'saved-draws.json'
    write_json(path, value)
    binding, _ = replay.manifest(path, pack, baseline, candidate)
    return SimpleNamespace(root=priv, path=path, binding=binding, manifest=value, pack=pack,
                           baseline=baseline, candidate=candidate, logs=logs, calls=codex[0])


def open_replay(saved):
    return replay.SavedDrawReplay(saved.binding, saved.pack, saved.baseline, saved.candidate)


@pytest.mark.parametrize('saved', [True], indirect=True)
def test_supplement_missing_rounds_is_paired_independent_and_opt_in(saved):
    row = saved.pack['rows'][0]
    value = copy.deepcopy(saved.manifest)
    write_json(saved.path, value)
    info = saved.candidate
    binding, _ = replay.manifest(saved.path, saved.pack, info, info)
    offline = replay.SavedDrawReplay(binding, saved.pack, info, info)
    with pytest.raises(ConfigError, match='live fallback prohibited'):
        offline.get(row, 1, None)
    with pytest.raises(ConfigError, match='feature-only'):
        replay.SavedDrawReplay(binding, saved.pack, saved.baseline, info, supplement_missing=True)
    count = len(saved.calls)
    a, b = replay.scorers(binding, saved.pack, info, info, supplement_missing=True)
    for rnd in range(3):
        for branch, scorer in [('baseline', a), ('candidate', b)]:
            judge(saved.root, scorer, row, 'supplement', rnd, branch)
    assert len(saved.calls) == count + 2
    # New replay workers still reuse completed supplemental requests across experiments.
    a, b = replay.scorers(binding, saved.pack, info, info, supplement_missing=True)
    for rnd in (1, 2):
        judge(saved.root, a, row, 'resume-supplement', rnd)
    assert len(saved.calls) == count + 2


def test_supplement_cannot_replace_corrupt_evidence_or_initial(saved):
    row, info = saved.pack['rows'][0], saved.candidate
    player = replay.SavedDrawReplay(saved.binding, saved.pack, info, info, supplement_missing=True)
    path = Path(saved.manifest['draws'][row['case_id']]['1']['path'])
    path.write_text(path.read_text() + '\n')
    count = len(saved.calls)
    with pytest.raises(ConfigError):
        player.get(row, 1, None)
    value = copy.deepcopy(saved.manifest)
    del value['draws'][row['case_id']]['0']
    write_json(saved.path, value)
    with pytest.raises(ConfigError, match='Initial draw missing'):
        replay.manifest(saved.path, saved.pack, info, info)
    assert len(saved.calls) == count


def test_registered_replay_completes_all_votes_without_requests(saved, monkeypatch, isolated_legacy_provenance):
    _use_small_settings(monkeypatch)
    write_json(saved.root / 'judge_eval/pack/pack.json', saved.pack)
    before = versions.POINTERS_PATH.read_bytes()
    count = len(saved.calls)
    exp = experiment.create_judge_eval_experiment('development', 'pack', {}, 'saved feature GBDT',
        candidate_ref=saved.candidate['id'], saved_draw_manifest=str(saved.path))
    runner.run_judge_experiment(exp)
    state = experiment.state_of(exp)
    assert state['status'] == 'finished'
    metrics = gates.confirmed_metrics(exp, saved.pack['rows'], experiment.spec_of(exp))
    assert metrics['net_win_confirmed'] == 6
    rows = [json.loads(line) for line in (exp / 'cases.jsonl').read_text().splitlines()]
    assert all(r['baseline_votes'] == [False] * 3 and r['candidate_votes'] == [True] * 3 for r in rows)
    assert len(saved.calls) == count and versions.POINTERS_PATH.read_bytes() == before
    runner.run_judge_experiment(exp)
    assert len(saved.calls) == count


def submit_saved_branch(saved, monkeypatch):
    _use_small_settings(monkeypatch)
    monkeypatch.setattr(branches, 'load_settings', experiment.load_settings)
    write_json(saved.root / 'judge_eval/pack-calibration-test/pack.json', saved.pack)
    write_json(saved.root / 'judge_eval/pack-validation-test/pack.json', saved.pack)
    return branches.submit('saved', 'judge', 'reuse saved observations',
        candidate_ref=saved.candidate['id'], development_pack='pack-calibration-test',
        validation_pack='pack-validation-test', saved_draw_manifest=str(saved.path))


def test_branch_replay_promotes_development_without_requests_or_shared_pointer_change(
        saved, monkeypatch, isolated_legacy_provenance):
    count, before = len(saved.calls), versions.POINTERS_PATH.read_bytes()
    proposal = submit_saved_branch(saved, monkeypatch)
    assert proposal['saved_draw_replay'] == saved.binding
    for stage in ('smoke', 'development'):
        [job] = branches.advance()
        exp = experiment.load_experiment(job['id'])
        spec = experiment.spec_of(exp)
        assert spec['branch_stage'] == stage and spec['saved_draw_replay'] == saved.binding
        runner.run_judge_experiment(exp)
        assert experiment.state_of(exp)['status'] == 'finished'
        rows = (exp / 'cases.jsonl').read_text().splitlines()
        assert len(rows) == (2 if stage == 'smoke' else 6)
    branches.promote_experiment(exp.name)
    state = json.loads((saved.root / 'branches/saved/state.json').read_text())
    assert state['development']['candidate_ref'] == saved.candidate['id']
    assert state['development']['experiment_id'] == exp.name
    assert versions.POINTERS_PATH.read_bytes() == before and len(saved.calls) == count


@pytest.mark.parametrize('mutation', ['manifest', 'production'])
def test_branch_replay_blocks_changed_binding_without_live_fallback(
        saved, monkeypatch, isolated_legacy_provenance, mutation):
    count = len(saved.calls)
    submit_saved_branch(saved, monkeypatch)
    if mutation == 'manifest':
        saved.path.write_text(saved.path.read_text() + '\n')
    else:
        pointers = versions.load_pointers()
        write_json(versions.POINTERS_PATH, {**pointers, 'production_judge': saved.candidate['id']})
    assert branches.advance() == []
    record = branches.round_of('saved/r-0001')
    assert record['phase'] == 'blocked' and 'Saved development' in record['reason']
    assert len(saved.calls) == count


def test_branch_cli_forwards_saved_draw_manifest(monkeypatch):
    from scripts import iterate_branches
    calls = []
    monkeypatch.setattr(branches, 'submit', lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(iterate_branches.sys, 'argv', ['iterate_branches.py', 'submit',
        '--name', 'saved', '--kind', 'judge', '--change', 'reuse', '--candidate', 'j-0002',
        '--development-pack', 'pack-calibration-test', '--saved-draw-manifest', 'draws.json'])
    iterate_branches.main()
    assert calls[0]['saved_draw_manifest'] == 'draws.json'


def test_successful_operation_in_failed_case_is_reusable(saved):
    row = saved.pack['rows'][0]
    producer = saved.logs[row['case_id'], 0]
    value = json.loads(producer.path.read_text())
    value['status'] = 'failed'
    value['operations'].append(dict(kind='judge', branch='candidate', round=1, status='failed', events=[]))
    write_json(producer.path, value)
    scorer = corrected.CorrectedJudge(saved.baseline['config'], saved.baseline['dir'])
    count = len(saved.calls)
    _, resumed = judge(saved.root, scorer, row, 'resumed')
    evidence = Evidence()
    reference = dict(path=str(resumed.path), sha256=sha256_file(resumed.path), operation=0, round=0)
    try:
        observation = replay.verify_draw(row, saved.baseline, reference, 0, evidence)
        assert observation['initial']['human_option'] == 'B' and len(saved.calls) == count
        value['operations'][0]['status'] = 'failed'
        write_json(producer.path, value)
        with pytest.raises(ConfigError, match='operation differs'):
            replay.verify_draw(row, saved.baseline, reference, 0, evidence)
    finally:
        evidence.store.db.close()


@pytest.mark.parametrize('mutation', ['round', 'config', 'mapping', 'result', 'case'])
def test_forged_observation_rejected_even_with_updated_trace_hash(saved, mutation):
    row = saved.pack['rows'][0]
    log = saved.logs[row['case_id'], 0]
    value = json.loads(log.path.read_text())
    op = value['operations'][0]
    if mutation == 'round':
        op['round'] = 1
    elif mutation == 'case':
        value['case']['human_reply'] = ['another answer']
    elif mutation == 'result':
        op['result']['identified_ai'] = True
    elif mutation == 'mapping':
        next(e for e in op['events'] if e['kind'] == 'blind_mapping')['data']['human_option'] = 'B'
    else:
        next(e for e in op['events'] if e['kind'] == 'codex')['request']['config']['model'] = 'another-model'
    write_json(log.path, value)
    evidence = Evidence()
    try:
        with pytest.raises(ConfigError):
            replay.verify_draw(row, saved.baseline, dict(path=str(log.path),
                sha256=sha256_file(log.path), operation=0, round=0), 0, evidence)
    finally:
        evidence.store.db.close()


def test_changed_inputs_and_cache_entries_are_rejected(saved):
    row = saved.pack['rows'][0]
    player = open_replay(saved)
    player.manifest['cache_entries'] = {}
    with pytest.raises(ConfigError, match='saved cache evidence'):
        player.get(row, 0, None)
    player = open_replay(saved)
    player.manifest['inputs'] = {}
    with pytest.raises(ConfigError, match='saved request evidence'):
        player.get(row, 0, None)
    player = open_replay(saved)
    with pytest.raises(ConfigError, match='sample content'):
        player.get({**row, 'ai_replies': ['changed']}, 0, None)
    path = Path(saved.manifest['draws'][row['case_id']]['0']['path'])
    path.write_text('{}')
    with pytest.raises(ConfigError, match='input changed'):
        open_replay(saved)


def test_missing_independent_round_has_no_live_fallback(saved):
    row = saved.pack['rows'][0]
    player = open_replay(saved)
    del player.manifest['draws'][row['case_id']]['1']
    count = len(saved.calls)
    with pytest.raises(ConfigError, match='independent saved round missing'):
        player.get(row, 1, None)
    assert len(saved.calls) == count
    with pytest.raises(ConfigError, match='Invalid independent round'):
        player.get(row, True, None)


@pytest.mark.parametrize('mutation', ['fixed', 'generator', 'baseline', 'features'])
def test_cross_condition_replay_rejected(saved, mutation):
    value, base, cand = copy.deepcopy(saved.manifest), copy.deepcopy(saved.baseline), copy.deepcopy(saved.candidate)
    if mutation == 'fixed':
        value['dataset'] = 'fixed_test'
    elif mutation == 'generator':
        value['generator_ref'] = 'different'
    elif mutation == 'baseline':
        value['source_judge'] = 'different'
    else:
        cand['config']['llm']['model'] = 'different'
    write_json(saved.path, value)
    with pytest.raises(ConfigError):
        replay.manifest(saved.path, saved.pack, base, cand)
    with pytest.raises(ConfigError, match='manifest changed' if mutation != 'features' else None):
        replay.SavedDrawReplay(saved.binding, saved.pack, base, cand)


@pytest.mark.parametrize('reverse', [False, True])
def test_rebased_pure_scorers_reuse_source_rounds_without_requests(
        saved, monkeypatch, isolated_legacy_provenance, reverse):
    _use_small_settings(monkeypatch)
    config = {**saved.baseline['config'], 'decision_policy': 'lr_only'}
    lr = versions.judge_dir(versions.create_judge_version(
        config, {}, source_dir=saved.baseline['dir']))
    base, cand = (saved.candidate, lr) if reverse else (lr, saved.candidate)
    pointers = versions.load_pointers()
    write_json(versions.POINTERS_PATH, {**pointers, 'iteration_judge': base['id'],
                                      'production_judge': base['id']})
    write_json(saved.root / 'judge_eval/pack/pack.json', saved.pack)
    before, count = versions.POINTERS_PATH.read_bytes(), len(saved.calls)
    exp = experiment.create_judge_eval_experiment('development', 'pack', {}, 'rebase saved pure scorers',
        candidate_ref=cand['id'], saved_draw_manifest=str(saved.path))
    runner.run_judge_experiment(exp)
    assert experiment.state_of(exp)['status'] == 'finished'
    assert experiment.spec_of(exp)['baseline_ref'] == base['id'] != saved.baseline['id']
    assert gates.confirmed_metrics(exp, saved.pack['rows'], experiment.spec_of(exp))['net_win_confirmed'] == 0
    left, right = replay.scorers(saved.binding, saved.pack, base, cand)
    for rnd in range(3):
        for arm, scorer in [('baseline', left), ('candidate', right)]:
            hit, log = judge(saved.root, scorer, saved.pack['rows'][0], 'rebased-rounds', rnd, arm)
            assert hit is True
            reuse = next(e['data'] for e in log.value['operations'][0]['events'] if e['kind'] == 'feature_reuse')
            assert reuse['reference']['round'] == rnd
    del left.replay.manifest['draws'][saved.pack['rows'][1]['case_id']]['1']
    with pytest.raises(ConfigError, match='independent saved round missing'):
        left.replay.get(saved.pack['rows'][1], 1, None)
    assert len(saved.calls) == count and versions.POINTERS_PATH.read_bytes() == before


@pytest.mark.parametrize('mutation', ['source_config', 'source_asset', 'baseline_features', 'hybrid_baseline'])
def test_rebased_replay_rejects_changed_source_or_baseline(saved, mutation):
    base = copy.deepcopy(saved.candidate)
    value = copy.deepcopy(saved.manifest)
    if mutation == 'source_config':
        value['source_config']['llm']['model'] = 'changed'
    elif mutation == 'source_asset':
        path = saved.baseline['dir'] / 'prompt.md'
        path.chmod(0o644)
        path.write_text('changed')
    elif mutation == 'baseline_features':
        base['config']['feature_llm'] = {'model': 'changed'}
    else:
        base = copy.deepcopy(saved.baseline)
        base['config']['correction_threshold'] = .8
    write_json(saved.path, value)
    with pytest.raises(ConfigError):
        replay.manifest(saved.path, saved.pack, base, saved.candidate)


def test_rebased_replay_rejects_source_change_after_loading(saved):
    player = replay.SavedDrawReplay(saved.binding, saved.pack, saved.candidate, saved.candidate)
    path = saved.baseline['dir'] / 'prompt.md'
    path.chmod(0o644)
    path.write_text('changed')
    with pytest.raises(ConfigError, match='材料变化'):
        player.get(saved.pack['rows'][0], 0, None)


def test_fusion_reuses_registered_draws_without_requests(saved, monkeypatch, isolated_legacy_provenance):
    import hashlib
    _use_small_settings(monkeypatch)
    info = saved.candidate
    payload = json.dumps({'schema': 1, 'kind': 'gbdt_lr_score_fusion_v1',
        'weights': {'gbdt': .8, 'lr': .2}, 'normalization': {
            'method': 'training_pair_margin_rms', 'scales': {'gbdt': 2., 'lr': 3.}}}).encode()
    assets = {n: (info['dir'] / n).read_bytes() for n in ['prompt.md', *info['config']['assets']]}
    assets['fusion.json'] = payload
    cfg = {**info['config'], 'decision_policy': 'score_fusion', 'fusion_file': 'fusion.json',
        'assets': {**info['config']['assets'], 'fusion.json': hashlib.sha256(payload).hexdigest()}}
    candidate = versions.judge_dir(versions.create_judge_version(cfg, {}, assets=assets))
    pointers = versions.load_pointers()
    write_json(versions.POINTERS_PATH, {**pointers, 'iteration_judge': info['id'], 'production_judge': info['id']})
    write_json(saved.root / 'judge_eval/pack/pack.json', saved.pack)
    before, count = versions.POINTERS_PATH.read_bytes(), len(saved.calls)
    exp = experiment.create_judge_eval_experiment('development', 'pack', {}, 'fused saved features',
        candidate_ref=candidate['id'], saved_draw_manifest=str(saved.path))
    runner.run_judge_experiment(exp)
    assert experiment.state_of(exp)['status'] == 'finished'
    left, right = replay.scorers(saved.binding, saved.pack, info, candidate)
    for rnd in range(3):
        for arm, scorer in [('baseline', left), ('candidate', right)]:
            _, log = judge(saved.root, scorer, saved.pack['rows'][0], 'fusion-rounds', rnd, arm)
            reuse = next(e['data'] for e in log.value['operations'][0]['events'] if e['kind'] == 'feature_reuse')
            assert reuse['reference']['round'] == rnd
    assert len(saved.calls) == count and versions.POINTERS_PATH.read_bytes() == before
