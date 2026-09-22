from copy import deepcopy

import pytest

from scripts.legacy.rerun_judge_dataset import ConfigError, saved_verdict, validate_checkpoint, validate_source


def burst():
    messages = [{'chat_id': 'group:a', 'chat_type': 'group', 'sender': str(i),
                 'text': str(i), 'is_self': i >= 2, 'timestamp': 1000 + i * 10000} for i in range(4)]
    case = {'case_id': 'one', 'source_message_id': 'group:a:2', 'chat_type': 'group',
            'context': deepcopy(messages[:2]), 'human_reply': ['2', '3']}
    return case, messages


def test_complete_burst_keeps_multiple_incoming_and_long_interval():
    case, messages = burst()
    audit = validate_source([case], messages)
    assert audit['source_verified'] == 1 and audit['excluded'] == 0
    assert audit['human_bubbles'] == {2: 1}


@pytest.mark.parametrize('defect', ['truncated', 'context', 'mid_burst', 'duplicate'])
def test_bad_source_cannot_silently_filter_or_replace(defect):
    case, messages = burst()
    if defect == 'truncated':
        case['human_reply'] = ['2']
    elif defect == 'context':
        case['context'][0]['text'] = 'changed'
    elif defect == 'mid_burst':
        case['source_message_id'] = 'group:a:3'
        case['human_reply'] = ['3']
    with pytest.raises(ConfigError):
        validate_source([case, case] if defect == 'duplicate' else [case], messages)


def test_resume_checks_content_not_just_case_id():
    case, _ = burst()
    row = {**deepcopy(case), 'ai_replies': ['ok']}
    validate_checkpoint([row], [case])
    row['human_reply'] = ['changed']
    with pytest.raises(ConfigError):
        validate_checkpoint([row], [case])


def test_lr_stats_recover_cache_without_model_call():
    verdict = {'small_model_probability_a': .8, 'candidate_option': 'B'}
    class Store:
        def get(self, key):
            assert key == 'saved'
            return {'value': {'last_verdict': verdict}}
    op = {'events': [{'kind': 'cache_hit', 'data': {'layer': 'judge', 'key': 'saved'}}]}
    assert saved_verdict(op, Store()) == verdict


@pytest.mark.parametrize('changed', ['data', 'generator', 'judge', 'implementation'])
def test_existing_results_reject_changed_provenance(tmp_path, monkeypatch, changed):
    from scripts.legacy import rerun_judge_dataset as study
    original = dict.fromkeys(['data', 'generator', 'judge', 'implementation'], 'old')
    study.write_json(tmp_path / 'provenance.json', {'inputs': original})
    study.write_json(tmp_path / 'building.json', {'rows': [{'case_id': 'saved'}]})
    monkeypatch.setattr(study, 'fingerprint', lambda _: {**original, changed: 'new'})
    with pytest.raises(ConfigError, match='拒绝混用检查点'):
        study.freeze(None, tmp_path)
    assert study.read(tmp_path / 'provenance.json')['inputs'] == original
    assert study.read(tmp_path / 'building.json')['rows'] == [{'case_id': 'saved'}]
