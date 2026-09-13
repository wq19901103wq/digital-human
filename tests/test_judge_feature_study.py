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
