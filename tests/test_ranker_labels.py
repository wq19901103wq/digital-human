from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.config import ConfigError, load_settings
from src.generator import ranker_labels as labels
from src.generator.history_sources import digest
from src.iteration import versions
from src.iteration.storage import read_json, write_json
from tests.leakage_support import historical, mechanical


def row(i, *, at=None, candidates=2):
    at = i * 100 if at is None else at
    return dict(target_id=str(i), payload_sha256=f'row-{i}', split='fit',
        target=dict(case_id=str(i), chat_type='private', input_cutoff={'timestamp': at},
            source_span={'end_timestamp': at+1}, context_message_ids=[f'c{i}'],
            reply_message_ids=[f'a{i}']),
        teacher_sources=dict(same_teacher_training_case=False, answer_in_teacher_training=[],
            answer_in_teacher_references=[], time_blocked=True),
        candidates=[dict(example=dict(id=f'e{k}', source_span={'end_timestamp': 1},
                    context_message_ids=[f'ec{k}'], reply_message_ids=[f'ea{k}']))
                    for k in range(candidates)])


def test_offline_teacher_allows_later_cutoff_but_excludes_case_and_answer_overlap():
    rows = [row(i) for i in range(10)]
    for r, key in zip(rows[:3], labels.POLICY['teacher_overlap_fields']):
        r['teacher_sources'][key] = True if key.startswith('same') else ['answer-id']
    selected, audit = labels.select_targets(rows)
    assert [r['target_id'] for r in selected] == [str(i) for i in range(3, 10)]
    assert len(audit['excluded']) == 3
    assert [r['split'] for r in selected] == ['fit'] * 5 + ['validation'] * 2
    assert all(r['teacher_sources']['time_blocked'] for r in selected)


def test_chronological_ties_and_fit_material_purge():
    rows = [row(i) for i in range(10)]
    rows[7]['target']['input_cutoff']['timestamp'] = 800
    rows[0]['target']['source_span']['end_timestamp'] = 800
    rows[1]['candidates'][0]['example']['context_message_ids'] = ['a9']
    selected, audit = labels.select_targets(rows)
    assert audit['validation_boundary'] == 800
    assert [r['target_id'] for r in selected if r['split'] == 'validation'] == ['7', '8', '9']
    assert [r['target_id'] for r in audit['purged']] == ['0', '1']
    assert {r['split'] for r in selected} == {'fit', 'validation'}
    with pytest.raises(ConfigError, match='训练或验证上文为空'):
        labels.select_targets([row(0, at=100), row(1, at=100)])


def test_single_example_full_context_and_bubbles_without_target_answer(tmp_path, monkeypatch):
    fixture = historical(tmp_path / 'data' / 'd-test')
    monkeypatch.setattr(versions, 'DATA_ROOT', fixture.directory.parent)
    prompt = tmp_path / 'generator'
    mechanical(prompt)
    calls = []

    class Client:
        def chat(self, messages, **kwargs):
            calls.append(messages)
            return '{"replies": ["生成"]}'

    gen = labels.SingleExampleGenerator(load_settings(), {'shots_char_budget': 2500},
                                       Client(), prompt, fixture.directory / 'fewshot_pool.jsonl')
    case, example = fixture.cases[1], fixture.examples[0]
    assert gen.generate_one(case, example)['replies'] == ['生成']
    rendered = json.dumps(calls[0], ensure_ascii=False)
    assert all(text in rendered for text in example['reply'])
    assert all(m['text'] in rendered for m in example['context_messages'])
    assert all(text not in rendered for text in case['human_reply'])
    assert gen._example is None
    for invalid in (fixture.examples[1], fixture.examples[2]):
        with pytest.raises(ConfigError):
            gen.generate_one(case, invalid)
    assert len(calls) == 1


def test_judge_retry_reuses_generation_and_completed_label(tmp_path):
    generated, judged = [], []
    sample = row(1)
    class Generator:
        def generate_one(self, case, example):
            generated.append(example['id'])
            return {'replies': ['生成'], 'latency_ms': 1}
    class Judge:
        def is_ai(self, case, replies):
            judged.append(replies)
            if len(judged) == 1:
                raise TimeoutError('retry judge only')
            return False
    manifest = dict(data='d-test', teacher='j-test', generator='g-test')
    args = (tmp_path / 'labeling', manifest, sample, sample['candidates'][0], Generator(), Judge(), lambda: None)
    identity, failed = labels.label_one(*args)
    assert failed['status'] == 'failed' and failed['failed_stage'] == 'judge' and 'z' not in failed
    _, value = labels.label_one(*args)
    assert value['status'] == 'complete' and value['z'] == 1 and value['identified_ai'] is False
    assert len(generated) == 1 and len(judged) == 2
    assert labels.label_one(*args)[1] == read_json(args[0] / 'observations' / (identity + '.json'))
    assert len(generated) == 1 and len(judged) == 2
    with pytest.raises(ConfigError, match='条件变化'):
        labels.label_one(args[0], {**manifest, 'teacher': 'changed'}, *args[2:])


def test_generation_failure_unlabeled_and_detected_reply_is_zero(tmp_path):
    sample, calls = row(1), []
    class Generator:
        def generate_one(self, *args):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError('generation failed')
            return {'replies': ['生成']}
    judge = SimpleNamespace(is_ai=lambda *args: True)
    args = (tmp_path / 'labels', dict(data='d', teacher='j', generator='g'), sample,
            sample['candidates'][0], Generator(), judge, lambda: None)
    _, value = labels.label_one(*args)
    assert value['status'] == 'failed' and 'z' not in value and 'generation' not in value
    _, value = labels.label_one(*args)
    assert value['z'] == 0 and value['identified_ai'] is True


def test_checkpoint_rejects_corruption_and_invalid_generation(tmp_path):
    path = tmp_path / 'checkpoint.json'
    value = dict(binding='bound', status='complete', identified_ai=False, z=1,
                 generation={'replies': ['valid']})
    labels._save(path, value)
    corrupted = read_json(path)
    corrupted['z'] = 0
    write_json(path, corrupted)
    with pytest.raises(ConfigError, match='损坏'):
        labels._saved(path, 'bound')
    labels._save(path, {**value, 'z': 0})
    with pytest.raises(ConfigError, match='标签方向'):
        labels._saved(path, 'bound')
    labels._save(path, {**value, 'generation': {'replies': []}})
    with pytest.raises(ConfigError, match='生成格式'):
        labels._saved(path, 'bound')


def test_pairs_only_from_complete_mixed_target_groups():
    rows = [row(i, candidates=4) for i in range(4)]
    states = {}
    for r, values in zip(rows, ([0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1], [0, 1])):
        for entry, z in zip(r['candidates'], values):
            key = (r['target_id'], entry['example']['id'])
            states[key] = dict(status='complete', z=z, target_id=key[0], example_id=key[1], split='fit')
    summary, pairs = labels.label_summary(rows, states)
    assert summary['target_outcomes'] == dict(all_zero=1, all_one=1, mixed=1, incomplete=1)
    assert summary['labeled'] == 14 and summary['unlabeled'] == 2
    assert summary['z0'] == summary['z1'] == 7
    assert len(pairs) == 4 and all(p['target_id'] == '2' for p in pairs)
    assert {(p['winner'], p['loser']) for p in pairs} == {('e2', 'e0'), ('e2', 'e1'), ('e3', 'e0'), ('e3', 'e1')}
