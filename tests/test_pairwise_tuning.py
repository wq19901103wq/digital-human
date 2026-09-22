"""Recovery, draw provenance and independent scoring for the bounded sweep."""
import copy
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts.legacy import tune_guarded_pairwise_judge as job
from scripts.legacy import verify_pairwise_tuning as audit
from scripts.legacy import verify_judge_dataset_results as saved_cache
from src import cache, tracing
from src.config import ConfigError, sha256_file
from src.iteration import gates
from src.judge import pairwise_models as pm, corrected
from test_corrected_judge import bundle, case, features


@pytest.fixture
def fitted_pairs():
    rng = np.random.default_rng(21)
    a, b = rng.normal(size=(2, 40, 3))
    return a, b, (a[:, 0] > b[:, 0]).astype(float)


def test_fitting_reuses_receipt_and_recovers_orphan_weights(tmp_path, monkeypatch, fitted_pairs):
    a, b, y = fitted_pairs
    recipe, manifest = pm.recipes()[0], {'training_matrix_sha256': 'frozen'}
    original = job.fit_recipe(tmp_path, recipe, a, b, y, manifest, lambda: None)
    calls, fit = [], pm.fit
    def counted(*args):
        calls.append(True)
        return fit(*args)
    monkeypatch.setattr(pm, 'fit', counted)
    repeated = job.fit_recipe(tmp_path, recipe, a, b, y, manifest, lambda: None)
    assert repeated.document() == original.document() and not calls
    receipt = tmp_path / 'receipts' / (recipe['id'] + '.json')
    receipt.unlink()
    recovered = job.fit_recipe(tmp_path, recipe, a, b, y, manifest, lambda: None)
    assert recovered.document() == original.document() and len(calls) == 1
    with pytest.raises(ConfigError, match='来源变化'):
        job.fit_recipe(tmp_path, recipe, a, b, y, {'training_matrix_sha256': 'changed'}, lambda: None)
    receipt.unlink()
    model_path = tmp_path / 'models' / (recipe['id'] + '.json')
    changed = job.read(model_path)
    changed['parameters']['coefficients'][0] += 1
    model_path.write_text(json.dumps(changed))
    with pytest.raises(ConfigError):
        job.fit_recipe(tmp_path, recipe, a, b, y, manifest, lambda: None)
    assert not receipt.exists()  # Never certify edited weights after an interrupted save.


@pytest.mark.parametrize('kind', ['lr', 'gbdt', 'dnn'])
def test_independent_auditor_scores_actual_serialized_models(fitted_pairs, kind):
    a, b, y = fitted_pairs
    recipe = next(r for r in pm.recipes() if r['kind'] == kind)
    with threadpool_limits(limits=1):
        scorer = pm.fit(a, b, y, recipe)
        independent = [audit.probability(scorer.document(), left, right) for left, right in zip(a, b)]
        np.testing.assert_allclose(scorer.probability_a(a, b), independent, atol=1e-12, rtol=0)
        assert pm.fit(a, b, y, recipe).document() == scorer.document()


def test_independent_vote_counts_equal_formal_gate_and_reject_incomplete(tmp_path):
    records = []
    for index, (bv, cv) in enumerate([
        ([False, False, True], [True, True, False]),
        ([True, True, False], [False, False, True]),
        ([False, True, True], [True, False, True]),
        ([True], [True]),
    ]):
        row = job.record_for({'case_id': str(index)},
            [{'identified_ai': v} for v in bv], [{'identified_ai': v} for v in cv])
        row['draws'] = [{'round': i} for i in range(len(bv))]
        records.append(row)
    (tmp_path / 'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    assert audit.count(records) == gates.confirmed_metrics(tmp_path, records,
        {'kind': 'judge_eval', 'protocol': {'flip_extra_rounds': 2}})
    changed = copy.deepcopy(records)
    changed[0]['candidate_votes'] = [True]
    with pytest.raises(ValueError, match='incomplete'):
        audit.count(changed)
    with pytest.raises(ConfigError, match='独立补验'):
        job.record_for({'case_id': 'bad'}, [{'identified_ai': False}], [{'identified_ai': True}])


@pytest.fixture
def draw_environment(tmp_path, bundle, case, monkeypatch):
    cfg, version_dir = bundle
    cfg = {**cfg, 'llm': {**cfg['llm'], 'reasoning_effort': 'low', 'codex_cli_version': 'test'}}
    instance = tmp_path / 'instance'
    cache.Store(instance / '.cache')
    monkeypatch.setattr(saved_cache, 'BASE', instance)
    calls = []
    class Client(corrected.CodexJudgeClient):
        def __init__(self, config):
            self.config = config
        def _run(self, prompt, schema=None):
            calls.append((cache._scope.get()['context']['round'], bool(schema)))
            with tracing.step('codex', {'prompt': prompt, 'schema': schema, 'config': self.config}) as event:
                value = ({'option_A': features(), 'option_B': features()} if schema else
                         {'human_option': 'A', 'confidence': .9, 'reason': 'test'})
                raw = json.dumps(value)
                event['response'] = {'text': raw}
                return raw
    monkeypatch.setattr(corrected, 'CodexJudgeClient', Client)
    row = {**case, 'ai_replies': ['九点']}
    info = {'config': cfg, 'dir': version_dir}
    source, target = instance / 'experiments/source', instance / 'judge_training/study'
    spec = {'dataset': 'development'}
    trace = tracing.CaseTrace(source, row, spec)
    with trace.operation('judge', 'baseline', 0, {'candidate_replies': row['ai_replies']}) as op:
        op['result'] = {'identified_ai': corrected.CorrectedJudge(cfg, version_dir).is_ai(row, row['ai_replies'])}
    trace.finish('ok')
    job.save_once(target / 'spec.json', spec)
    reference = dict(path=str(trace.path), sha256=sha256_file(trace.path), operation=0, round=0)
    refs = {('luna', row['case_id'], 0): reference}
    return target, spec, row, info, refs, calls, trace


def test_trace_reuse_supplements_and_resume_keep_rounds_independent(draw_environment):
    directory, spec, row, info, refs, calls, trace = draw_environment
    draws = [job.get_draw(directory, spec, row, 'luna', rnd, info, refs, lambda: None) for rnd in range(3)]
    assert calls == [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True)]
    repeated = [job.get_draw(directory, spec, row, 'luna', rnd, info, refs, lambda: None) for rnd in range(3)]
    assert repeated == draws and len(calls) == 6
    # A crash before the pointer save can still reuse the successful whole-Judge cache.
    job.draw_path(directory, 'luna', row['case_id'], 2).unlink()
    cached = job.get_draw(directory, spec, row, 'luna', 2, info, refs, lambda: None)
    assert cached['initial'] == draws[2]['initial'] and len(calls) == 6
    assert cached['reference']['path'] != draws[2]['reference']['path']


@pytest.mark.parametrize('mutation', ['case', 'config', 'round', 'trace', 'spec'])
def test_saved_draw_rejects_changed_identity(draw_environment, mutation):
    directory, spec, row, info, refs, calls, trace = draw_environment
    job.get_draw(directory, spec, row, 'luna', 0, info, refs, lambda: None)
    if mutation == 'case':
        row = {**row, 'ai_replies': ['changed']}
    elif mutation == 'config':
        info = {**info, 'config': {**info['config'], 'correction_threshold': .8}}
    elif mutation == 'spec':
        (directory / 'spec.json').write_text('{"changed":true}')
    elif mutation == 'trace':
        trace.value['case']['human_reply'] = ['changed']
        trace.save()
    else:
        path = job.draw_path(directory, 'luna', row['case_id'], 0)
        saved = job.read(path)
        saved['reference']['round'] = 1
        path.write_text(json.dumps(saved))
    with pytest.raises(ConfigError):
        job.get_draw(directory, spec, row, 'luna', 0, info, refs, lambda: None)
    assert len(calls) == 2


def test_missing_initial_draw_never_issues_new_request(draw_environment):
    directory, spec, row, info, refs, calls, trace = draw_environment
    with pytest.raises(ConfigError, match='初测必须复用'):
        job.get_draw(directory, spec, row, 'luna', 0, info, {}, lambda: None)
    assert len(calls) == 2


def test_finished_status_is_terminal(tmp_path):
    job.status(tmp_path, 'finished', successful=26, total=26)
    assert job.read(tmp_path / 'state.json')['status'] == 'finished'


def test_sweep_shares_supplementary_draws_across_recipes_and_retries_only_failures(tmp_path, monkeypatch):
    """Multiple differing recipes incur one draw per arm/case/round, including retry."""
    from collections import Counter
    from threading import Lock
    rows = [{'case_id': 'first'}, {'case_id': 'second'}]
    spec = {'source': str(tmp_path / 'source'), 'recipes': pm.recipes(),
            'protocol': {'flip_extra_rounds': 2}}
    class Seal:
        def check(self):
            pass
    seal = Seal()
    frozen = {'baseline_ref': 'luna', 'candidate_ref': 'sol'}
    monkeypatch.setattr(job, 'evaluation_inputs', lambda directory: (spec, rows, tmp_path, frozen, seal, seal))
    monkeypatch.setattr(job, 'source_references', lambda *args: {})
    monkeypatch.setattr(job.versions, 'judge_dir', lambda ref: ref)
    monkeypatch.setattr(job.rt, 'load_formal_judge', lambda path: None)
    read = job.read
    monkeypatch.setattr(job, 'read', lambda path: path.stem if path.parent.name == 'models' else read(path))
    monkeypatch.setattr(job.pm.Scorer, 'from_document', lambda value: value)
    calls, lock = Counter(), Lock()
    def draw(directory, spec, case, arm, rnd, info, refs, check):
        key = (arm, case['case_id'], rnd)
        with lock:
            calls[key] += 1
            fail = key == ('sol', 'second', 1) and calls[key] == 1
        if fail:
            raise RuntimeError('transient request failure')
        return {'key': key, 'control_hit': False, 'reference': {'round': rnd, 'key': list(key)}}
    monkeypatch.setattr(job, 'get_draw', draw)
    def predict(name, model, draw):
        arm, cid, rnd = draw['key']
        changed = name in ('lr_l2_c0.1', 'lr_l2_c0.3') and (arm, cid) in {
            ('luna', 'first'), ('sol', 'second')}
        return {'identified_ai': changed}
    monkeypatch.setattr(job, 'predict', predict)
    monkeypatch.setattr(audit, 'audit', lambda directory: {'status': 'passed', 'scope': 'complete'})
    job.evaluate(tmp_path, workers=4)
    initial = {(a, c['case_id'], 0) for a in job.ARMS for c in rows}
    supplement = {(a, cid, r) for a, cid in (('luna', 'first'), ('sol', 'second')) for r in (1, 2)}
    assert set(calls) == initial | supplement
    assert calls[('sol', 'second', 1)] == 2
    assert sum(calls.values()) == 9
    results = read(tmp_path / 'results.json')['comparisons']
    assert len(results) == 26
    assert all(r['metrics']['pairs'] == 2 for r in results)
    assert sum(r['positive_development_net'] for r in results) == 4
    assert read(tmp_path / 'state.json')['status'] == 'finished'
