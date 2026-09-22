"""New training partition, answer isolation, and content-aware resume checks."""
import copy

import pytest

from scripts.legacy import retrain_judge_lr_dataset as job
from src import cache
from src.config import ConfigError


def messages():
    rows = []
    for kind in ('group', 'private'):
        for chat in range(3):
            cid = f'{kind}:account:{chat}'
            for i in range(4):
                for offset, own in enumerate((False, True, True)):
                    rows.append({'chat_id': cid, 'chat_type': kind, 'source_chat_id': f'{kind}-{chat}',
                        'chat_name': cid, 'is_self': own, 'sender': 'self' if own else 'other',
                        'timestamp': 1700000000+i*10+offset, 'text': f'{cid}/{i}/{offset}'})
    return rows


def test_training_selection_preserves_bubbles_and_chat_cap():
    source = messages()
    rows = job.select_training(source, set(), per_type=3, cap=1)
    assert len(rows) == 6
    assert [r['chat_type'] for r in rows] == ['group', 'private'] * 3
    assert all(len(r['human_reply']) == 2 and not r['context'][-1]['is_self'] for r in rows)
    assert len({r['source_message_id'].rpartition(':')[0] for r in rows}) == 6
    assert rows == job.select_training(source, set(), per_type=3, cap=1)
    with pytest.raises(ConfigError, match='混入'):
        job.select_training(source, {source[0]['chat_id']}, per_type=3, cap=1)
    with pytest.raises(ConfigError, match='配额'):
        job.select_training(source, set(), per_type=4, cap=1)


def test_retrieval_exclusion_uses_full_source_chat_and_fails_on_unknown():
    source = [{'chat_id': 'train', 'timestamp': 1}, {'chat_id': 'dev', 'timestamp': 2}]
    pool = [{'id': 'p1', 'timestamp': 1}]
    assert job.exclusions(pool, source, {'train'}, {'dev'}) == {'train': ['p1']}
    for timestamp in (2, 3):
        with pytest.raises(ConfigError, match='来源不明或属于评测'):
            job.exclusions([{'id': 'bad', 'timestamp': timestamp}], source, {'train'}, {'dev'})


def test_retriever_must_honor_exclusions():
    class Retriever:
        def retrieve(self, **kwargs):
            assert kwargs['exclude_ids'] == {'current', 'answer', 'neighbor'}
            return [{'id': 'answer'}]
    wrapped = job.ExcludingRetriever(Retriever())
    wrapped.blocked = {'answer', 'neighbor'}
    with pytest.raises(ConfigError, match='返回被排除'):
        wrapped.retrieve(exclude_ids={'current'})
    assert wrapped.failure is not None


@pytest.mark.parametrize('mutation', ['input', 'output', 'identity'])
def test_generation_checkpoint_rejects_changed_data(mutation):
    cases = [{'case_id': 'one', 'human_reply': ['hello']}]
    value = {'identity': 'frozen', 'entries': {'one': {'status': 'ok',
        'input_sha256': cache.digest(cases[0]), 'replies': ['hi'], 'output_sha256': cache.digest(['hi'])}}}
    job.check_generations(value, cases, 'frozen')
    if mutation == 'input':
        cases[0]['human_reply'] = ['changed']
    elif mutation == 'output':
        value['entries']['one']['replies'] = ['changed']
    else:
        value['identity'] = 'another dataset'
    with pytest.raises(ConfigError):
        job.check_generations(value, cases, 'frozen')


def test_blind_mapping_stays_identical_after_partial_resume():
    cases = job.select_training(messages(), set(), per_type=3, cap=1)
    generated = {'entries': {c['case_id']: {'status': 'ok', 'replies': ['AI']} for c in cases}}
    full = job.training_rows(cases, generated)
    partial = copy.deepcopy(generated)
    partial['entries'] = dict(list(partial['entries'].items())[:2])
    assert job.training_rows(cases, partial) == full[:2]
    assert all(r['blind']['option_'+r['human_option']] == c['human_reply'] for r, c in zip(full, cases))
    assert sum(r['human_option'] == 'A' for r in full) == 3
