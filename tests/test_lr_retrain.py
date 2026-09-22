"""Frozen retraining inputs, paired feature replay and independent supplements."""
import copy
import json
from pathlib import Path

import pytest

from scripts.legacy import retrain_judge_lr as job
from src import cache, tracing
from src.config import ConfigError, sha256_file
from src.iteration import experiment, promote, runner, versions
from src.judge import lr_retrain as lr, corrected, corrected_v1 as rt
from test_iteration import priv, _write
from test_corrected_judge import bundle, case, features
from leakage_support import isolated_legacy_provenance


class FeatureClient:
    calls = []

    def __init__(self, config):
        self.config = config

    def cache_identity(self):
        return self.config

    def run(self, prompt, schema=None):
        assert schema == rt.feature_response_json_schema(), 'initial LLM judgment must not run'
        scope = cache._scope.get()
        self.calls.append(scope['context']['round'] if scope else None)
        return json.dumps({'option_A': features(), 'option_B': features()})


def source_trace(directory, spec, row, config):
    trace = tracing.CaseTrace(directory, row, spec)
    blind, metadata = corrected.blind_case(row, row['human_reply'], row['ai_replies'])
    value = {'option_A': features(), 'option_B': features()}
    with trace.operation('judge', 'candidate', 0, {}):
        tracing.note('blind_mapping', {'human_option': 'A', 'candidate_option': 'B',
            'blind_case': blind, 'context_metadata': metadata})
        with tracing.step('codex', {'prompt': rt.feature_extractor_prompt(blind),
                'schema': rt.feature_response_json_schema(), 'config': config}) as event:
            event['response'] = {'text': json.dumps(value)}
        tracing.note('features', {'valid': True, 'parsed': value})
    trace.finish('ok')
    _write(directory / 'cases.jsonl', json.dumps({'case_id': row['case_id'], 'status': 'ok', 'trace_ref': trace.ref}) + '\n')
    return trace


@pytest.mark.parametrize('mutation', ['reply', 'context', 'config', 'output', 'label'])
def test_replay_rejects_changed_inputs(tmp_path, case, mutation):
    row = {**case, 'ai_replies': ['九点']}
    cfg = {'llm': {'model': 'sol', 'reasoning_effort': 'high'}}
    spec = {'candidate_ref': 'j-source', 'dataset': 'development'}
    trace = source_trace(tmp_path / 'source', spec, row, cfg['llm'])
    replay = lr.FeatureReplay(trace.exp_dir, spec, cfg, [row])
    assert replay.saved(row, 0) is not None
    edited = copy.deepcopy(row)
    if mutation == 'reply':
        edited['ai_replies'] = ['different']
    elif mutation == 'context':
        edited['context'][0]['text'] = 'different'
    else:
        events = trace.value['operations'][0]['events']
        if mutation == 'config':
            events[1]['request']['config']['reasoning_effort'] = 'low'
        elif mutation == 'output':
            events[2]['data']['parsed']['option_A'] = features(1)
        else:
            events[0]['data']['human_option'] = 'B'
        trace.save()
    with pytest.raises(ConfigError):
        replay.get(edited, 0, FeatureClient(cfg['llm']))


def test_full_runner_pairs_features_and_preserves_extra_rounds(priv, tmp_path, case, bundle, monkeypatch, isolated_legacy_provenance):
    cfg, original = bundle
    row = {**case, 'ai_replies': ['九点']}
    source_judge = versions.create_judge_version(cfg, {}, source_dir=original)
    source = priv / 'experiments/source'
    pack = priv / 'judge_eval/pack/pack.json'
    _write(pack, {'rows': [row], 'c0_gen_version': 'g-0001'})
    source_spec = {'id': 'source', 'candidate_ref': source_judge, 'dataset': 'development',
        'kind': experiment.KIND_JUDGE_EVAL, 'pack_ref': 'pack', 'pack_sha256': sha256_file(pack)}
    _write(source / 'spec.json', source_spec)
    source_trace(source, source_spec, row, cfg['llm'])
    directory = priv / 'judge_training/job'
    directory.mkdir(parents=True)
    correction = json.loads((original / 'correction.json').read_text())
    correction['final_model']['coefficients'][0] = 1.
    artifact = directory / 'correction.json'
    _write(artifact, correction)
    spec = {'evaluation_experiment': 'paired', 'source_experiment': 'source', 'source_judge': source_judge,
        'data_ref': 'd-0001', 'training_sha256': 'training', 'protocol': experiment._protocol_snapshot(runner.load_settings()),
        'separation_audit': {'promotion_eligible': False}}
    target = job.make_evaluation(directory, spec, artifact)
    pointers = versions.load_pointers()
    FeatureClient.calls = []
    monkeypatch.setattr(corrected, 'CodexJudgeClient', FeatureClient)
    monkeypatch.setattr(rt, 'score_formal_judge_pair', lambda model, *args: .8 if model.coefficients[0] else .2)
    runner.run_judge_experiment(target, workers=1)
    assert FeatureClient.calls == [1, 2]  # round 0 is replayed, each new round drawn once for both weights
    record = json.loads((target / 'cases.jsonl').read_text())
    assert record['flip_verified'] and len(record['baseline_votes']) == len(record['candidate_votes']) == 3
    replayed = tracing.read(target, record['trace_ref'])
    for round_index in range(3):
        operations = [op for op in replayed['operations'] if op['round'] == round_index]
        a, b = [[e['data']['parsed'] for e in op['events'] if e['kind'] == 'features'][-1] for op in operations]
        assert a == b
    replay = lr.FeatureReplay(source, source_spec, cfg, [row])
    # A fresh process reuses the same supplemental draw from the request cache.
    check = tracing.CaseTrace(target, row, experiment.spec_of(target))
    for round_index in (1, 2):
        with check.operation('judge', 'baseline', round_index, {}):
            replay.get(row, round_index, FeatureClient(cfg['llm']))
    assert FeatureClient.calls == [1, 2]
    runner.run_judge_experiment(target)
    assert FeatureClient.calls == [1, 2]
    with pytest.raises(ConfigError, match='该重训控制实验已标记为不可晋升'):
        promote.promote_judge(target.name)
    assert versions.load_pointers() == pointers


def test_training_checkpoint_reuses_success_and_retries_only_failures(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.iteration import learning_guard
    # This unit exercises checkpoint/retry mechanics; source rejection is tested
    # without mocks in test_leakage_guards.
    monkeypatch.setattr(learning_guard, 'verify_training', lambda *a: SimpleNamespace(check=lambda: None))
    directory = tmp_path / 'judge_training/job'
    directory.mkdir(parents=True)
    rows = [{'case_id': str(i), 'blind': {'text': str(i)}, 'human_option': 'A'} for i in range(2)]
    spec = {'training_sha256': cache.digest(rows), 'feature_config': {'model': 'sol'},
        'inputs': {str(corrected.RUNTIME_PATH.resolve()): sha256_file(corrected.RUNTIME_PATH)}}
    calls = []
    def extract(client, blind, source_check=None):
        calls.append(blind['text'])
        if blind['text'] == '1' and calls.count('1') == 1:
            raise RuntimeError('transient failure')
        return {'option_A': features(), 'option_B': features()}
    monkeypatch.setattr(job, 'CodexJudgeClient', FeatureClient)
    monkeypatch.setattr(lr, 'extract', extract)
    monkeypatch.setattr(rt, 'feature_extractor_prompt', lambda blind: json.dumps(blind))
    assert not job.train_features(directory, spec, rows, 1, 1)
    assert job.train_features(directory, spec, rows, 1, 0)
    assert calls == ['0', '1', '1']
    assert job.train_features(directory, spec, rows, 1, 0)
    assert calls == ['0', '1', '1']
    changed = copy.deepcopy(rows)
    changed[0]['human_option'] = 'B'
    with pytest.raises(ConfigError, match='标签'):
        job.checkpoint(directory, spec, changed)
    value = job.read(directory / 'features.json')
    value['entries']['0']['features']['option_A'] = features(1)
    _write(directory / 'features.json', value)
    with pytest.raises(ConfigError, match='产出被修改'):
        job.checkpoint(directory, spec, rows)


def test_fit_rejects_evaluation_rows_before_training(bundle):
    _, directory = bundle
    rows = [{'split': 'train'} for _ in range(1200)]
    rows[-1]['split'] = 'test'
    with pytest.raises(ConfigError, match='拟合缺少冻结训练规格'):
        lr.fit(rows, {}, directory / 'correction.json')
