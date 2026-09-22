"""抽特征实验只使用能还原实际回复边界的开发样本。"""
from copy import deepcopy

import pytest

from scripts.iterate_judge_features import clean_cases


def sample():
    messages = [
        {'chat_id': 'group:one', 'sender': '本人', 'text': '明天见', 'is_self': True, 'timestamp': 1000},
        {'chat_id': 'group:one', 'sender': '对方', 'text': '几点', 'is_self': False, 'timestamp': 1100},
        {'chat_id': 'group:one', 'sender': '本人', 'text': '八点', 'is_self': True, 'timestamp': 1120},
        {'chat_id': 'group:one', 'sender': '对方', 'text': '好', 'is_self': False, 'timestamp': 1130},
    ]
    case = {'case_id': 'case', 'source_message_id': 'group:one:2',
            'context': deepcopy(messages[:2]), 'human_reply': ['八点']}
    return case, messages


def test_verified_source_and_single_reply_are_retained():
    case, messages = sample()
    assert clean_cases([case], messages) == ([case], {})


@pytest.mark.parametrize('defect,reason', [
    ('self_boundary', 'self_boundary'),
    ('source_text', 'source_mismatch'),
    ('source_role', 'source_mismatch'),
    ('source_missing', 'source_missing'),
    ('late_reply', 'late_reply'),
    ('self_continuation', 'consecutive_self_reply'),
    ('multiple_incoming', 'multiple_incoming'),
])
def test_invalid_reply_boundaries_are_excluded(defect, reason):
    case, messages = sample()
    if defect == 'self_boundary':
        case['context'][-1]['is_self'] = True
    elif defect == 'source_text':
        messages[1]['text'] = '来源已经变化'
    elif defect == 'source_role':
        messages[2]['is_self'] = False
    elif defect == 'source_missing':
        case['source_message_id'] = 'missing:2'
    elif defect == 'late_reply':
        messages[2]['timestamp'] = 2000
    elif defect == 'self_continuation':
        messages[3]['is_self'] = True
    elif defect == 'multiple_incoming':
        messages[0]['is_self'] = False
        case['context'][0]['is_self'] = False
    assert clean_cases([case], messages) == ([], {reason: 1})


def test_pack_refusal_preserves_checkpoint_and_remaining_samples(tmp_path, monkeypatch):
    import json
    import random
    from types import SimpleNamespace
    from scripts import iterate_judge_features as study

    cases = [{'case_id': str(i), 'chat_type': 'group' if i < 3 else 'private',
              'context': [], 'human_reply': ['真人']} for i in range(4)]
    selected = cases[:3]
    rng = random.Random(42)
    rng.shuffle(selected)
    selected += cases[3:]
    rng.shuffle(selected)
    saved = {**selected[0], 'ai_replies': ['已保存回复'], 'generation_trace_ref': 'old-trace'}
    refused_id = selected[1]['case_id']
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'dev_pool.jsonl').write_text('\n'.join(json.dumps(c) for c in cases))
    directory = tmp_path / 'judge_eval' / 'pack-calibration-features-test'
    directory.mkdir(parents=True)
    checkpoint = {'selected_ids': [c['case_id'] for c in selected], 'rows': [saved],
                  'source': {'data_ref': 'd-test', 'generator_ref': 'g-test',
                             'dev_sha256': study.sha256_file(data / 'dev_pool.jsonl')}}
    (directory / 'building.json').write_text(json.dumps(checkpoint))
    monkeypatch.setattr(study.versions, 'PRIVATE', tmp_path)
    monkeypatch.setattr(study.versions, 'load_pointers', lambda: {'data': 'd-test', 'production_gen': 'g-test'})
    monkeypatch.setattr(study.versions, 'load_generator', lambda _: {'id': 'g-test', 'config': {'llm': {}}, 'dir': data})
    monkeypatch.setattr(study.versions, 'data_version_dir', lambda _: data)
    monkeypatch.setattr(study, 'load_settings', lambda: {})
    monkeypatch.setattr(study, 'load_weflow_dir', lambda _: [])
    monkeypatch.setattr(study, 'clean_cases', lambda pool, _: (pool, {}))
    monkeypatch.setattr(study, 'build_clients', lambda *_: {})
    called = []
    class Generator:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, case):
            called.append(case['case_id'])
            if case['case_id'] == refused_id:
                raise study.LLMRefusal('test stop_reason=refusal')
            return {'replies': ['新回复']}
    monkeypatch.setattr(study, 'ReplyGenerator', Generator)
    args = SimpleNamespace(source_exports=data, seed=42, sample=4, workers=4,
                           models=['same-model'], effort='high', build_limit=0)
    assert study.build_pack(args, 'test') == directory.name
    pack = json.loads((directory / 'pack.json').read_text())
    assert [r['case_id'] for r in pack['rows']] == checkpoint['selected_ids']
    assert pack['rows'][0] == saved
    assert set(called) == set(checkpoint['selected_ids'][1:])
    failed = pack['rows'][1]
    assert failed['generation_status'] == 'failed' and 'ai_replies' not in failed
    trace = json.loads((directory / 'traces' / (failed['generation_trace_ref'] + '.json')).read_text())
    assert trace['status'] == 'failed'
    assert trace['operations'][0]['error']['type'] == 'LLMRefusal'
    progress = json.loads((directory / 'progress.json').read_text())
    assert (progress['completed'], progress['successful'], progress['failed']) == (4, 3, 1)
    assert pack['generation_failures'] == 1
    called.clear()
    study.build_pack(args, 'test')
    assert called == []
