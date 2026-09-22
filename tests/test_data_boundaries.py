"""Observable boundaries: future data, role overlap, version identity and adoption."""
import copy
import json

import pytest

from src.bootstrap import history
from src.config import ConfigError
from src.generator.history import HistoryError, eligible
from src.iteration import datasets, experiment, versions
from test_iteration import priv as _priv_fixture
from leakage_support import historical, isolated_legacy_provenance as _legacy_fixture

isolated_legacy_provenance = _legacy_fixture

priv = _priv_fixture


def timeline(chats=1, sessions=8):
    rows = []
    for chat in range(chats):
        for session in range(sessions):
            for offset, (self_, text) in enumerate([(False, '问题'), (True, '回复'), (True, '接着说')]):
                rows.append({'chat_id': f'chat-{chat}', 'chat_type': 'group' if chat % 2 else 'private',
                    'source_chat_id': f'source-{chat}', 'chat_name': '例子', 'sender': '本人' if self_ else '朋友',
                    'is_self': self_, 'text': f'{text}-{session}',
                    'timestamp': 100000 + session * 10000 + chat * 10 + offset})
    return rows


def case(row):
    return history.as_case(row)


def test_full_burst_uses_last_reply_time_and_input_not_answer():
    rows = history.examples(timeline())
    assert rows[0]['reply'] == ['回复-0', '接着说-0']
    assert rows[0]['source_span']['end_timestamp'] == 100002
    assert rows[0]['input_cutoff']['timestamp'] == 100000
    assert not eligible(rows[0], case(rows[0]))
    assert eligible(rows[0], case(rows[1]))


def test_earlier_test_answer_is_available_later_without_train_membership():
    rows = history.examples(timeline())
    rows[0]['temporal_partition'] = 'fixed_test'
    assert eligible(rows[0], case(rows[2]))
    assert not eligible(rows[2], case(rows[0]))


def test_other_chat_future_unknown_source_and_same_second():
    rows = history.examples(timeline())
    ref = copy.deepcopy(rows[0])
    ref['source_span']['chat_id'] = 'other'
    ref['source_span']['end_timestamp'] = rows[1]['input_cutoff']['timestamp']
    assert not eligible(ref, case(rows[1]))
    ref.pop('context_message_ids')
    with pytest.raises(HistoryError):
        eligible(ref, case(rows[1]))


def test_same_answer_with_different_example_id_is_excluded():
    rows = history.examples(timeline())
    ref = copy.deepcopy(rows[0])
    ref['id'] = 'new-index-for-the-same-answer'
    target = case(rows[1])
    target['reply_message_ids'] = ref['reply_message_ids']
    assert not eligible(ref, target)


def test_holdout_is_explicit_but_familiar_history_remains_available():
    rows = history.examples(timeline())
    target = case(rows[2])
    assert eligible(rows[0], target)
    target['history_excluded_chat_ids'] = ['chat-0']
    assert not eligible(rows[0], target)


def test_global_purposes_separate_complete_fragments():
    rows = history.examples(timeline(chats=100, sessions=20))
    roles, protocol = history.plan(rows, total=8, train_total=8)
    assert max(r['source_span']['end_timestamp'] for r in roles['judge_training']) < protocol['development_start']
    sets = []
    for role in ('development', 'judge_development', 'fixed_test'):
        sets.append({mid for row in roles[role] for mid in row['context_message_ids'] + row['reply_message_ids']})
    assert sets[0] == sets[1] and protocol['development_roles_share_messages']
    assert not sets[1] & sets[2] and not sets[0] & sets[2]
    assert all(r['source_span']['start_timestamp'] >= protocol['acceptance_start'] for r in roles['fixed_test'])


def test_generator_same_behavior_reuses_id_but_records_both_builds(priv):
    cfg = {'llm': {'model': 'new'}, 'retriever': {'enabled': False}}
    first = versions.create_generator_version(cfg, 'd-0001')
    second = versions.create_generator_version(cfg, 'another-data', source_dir=versions.generator_dir(first))
    assert first == second
    records = [json.loads(p.read_text()) for p in (priv / 'generator_builds').glob('*.json')]
    assert {r['data_ref'] for r in records if r['generator_ref'] == first} == {'d-0001', 'another-data'}


def test_unchanged_smoke_does_not_create_generator_iteration(priv, isolated_legacy_provenance):
    before = {p.name for p in (priv / 'generators').iterdir()}
    created = experiment.create_gen_experiment('development', '仅验证连接', limit=5)
    spec = experiment.spec_of(created)
    assert spec['baseline_ref'] == spec['candidate_ref']
    assert before == {p.name for p in (priv / 'generators').iterdir()}


def test_role_manifest_rejects_content_change_and_paths(tmp_path):
    (tmp_path / 'dev_pool.jsonl').write_text('{}\n')
    purpose = {'roles': {'development': {'file': 'dev_pool.jsonl', 'sha256': 'wrong'}}}
    (tmp_path / 'purposes.json').write_text(json.dumps(purpose))
    with pytest.raises(ConfigError):
        datasets.case_path(tmp_path, 'development')


def test_publish_does_not_create_models_or_move_pointers(priv):
    messages = timeline(chats=100, sessions=20)
    rows = history.examples(messages)
    roles, protocol = history.plan(rows, total=8, train_total=8)
    pointers = versions.load_pointers()
    gens = list((priv / 'generators').iterdir())
    vid = history.publish(messages, rows, roles, protocol, {})
    directory = versions.data_version_dir(vid)
    assert versions.load_pointers() == pointers
    assert list((priv / 'generators').iterdir()) == gens
    assert not (directory / 'persona.md').exists()
    assert len((directory / 'fewshot_pool.jsonl').read_text().splitlines()) == len(rows)
    assert datasets.snapshot(directory)['history_policy'] == 'complete_before_input_v1'


def test_full_history_retriever_requires_boundary_and_cannot_downgrade(tmp_path):
    from src.config import sha256_file
    from src.generator.few_shot import PersonaFewShotRetriever
    from src.generator.generator import ReplyGenerator
    source = historical(tmp_path)
    rows = source.examples
    pool = tmp_path / 'fewshot_pool.jsonl'
    history.write_rows(pool, rows)
    (tmp_path / 'report.json').write_text(json.dumps({'review_status': 'approved',
        'examples_sha256': sha256_file(pool), 'history_policy': 'complete_before_input_v1'}))
    retriever = PersonaFewShotRetriever(pool)
    assert retriever.is_approved()
    with pytest.raises(HistoryError):
        retriever.retrieve(query='问题', chat_name='例子', is_group=False)
    selected = retriever.retrieve(query='问题', chat_name='例子', is_group=False, history_case=case(rows[2]))
    assert selected and all(eligible(r, case(rows[2])) for r in selected)
    gen = object.__new__(ReplyGenerator)
    gen._retriever, gen._max_shots, gen._budget = retriever, 3, 2500
    gen._learned, gen._reranker, gen._selection = None, None, None
    with pytest.raises(HistoryError):
        gen._style_block({'case_id': 'missing-times', 'context': []})


def test_new_data_can_be_checked_without_moving_baseline(priv, isolated_legacy_provenance):
    messages = timeline(chats=100, sessions=20)
    rows = history.examples(messages)
    roles, protocol = history.plan(rows, total=8, train_total=8)
    vid = history.publish(messages, rows, roles, protocol, {})
    before = versions.load_pointers()
    target = experiment.create_gen_experiment('development', '新数据接线检查', limit=2, data_ref=vid)
    spec = experiment.spec_of(target)
    assert spec['data_ref'] == vid and versions.load_pointers() == before
    assert spec['baseline_ref'] == spec['candidate_ref']
    assert spec['source_audit']['promotion_eligible'] is False


def test_model_provenance_cannot_be_certified_by_a_flag(tmp_path):
    (tmp_path / 'purposes.json').write_text(json.dumps({'protocol': {'development_start': 100}}))
    asset = tmp_path / 'model'
    asset.mkdir()
    (asset / 'learning.json').write_text(json.dumps({'verified': True, 'information_end': 0}))
    assert not datasets.static_sources(tmp_path, [asset], 'development')['promotion_eligible']


def test_gen_optimization_uses_its_own_purpose_and_cannot_promote(priv, isolated_legacy_provenance):
    messages = timeline(chats=100, sessions=20)
    rows = history.examples(messages)
    roles, protocol = history.plan(rows, total=8, train_total=8)
    vid = history.publish(messages, rows, roles, protocol, {})
    target = experiment.create_gen_experiment('development', '优化材料接线', limit=2,
                                              data_ref=vid, purpose='gen_optimization')
    spec = experiment.spec_of(target)
    assert spec['purpose'] == 'gen_optimization'
    assert not spec['source_audit']['promotion_eligible']
    assert spec['data_snapshot']['case_count'] == 2


def test_diagnostic_report_does_not_offer_adoption():
    from src.dashboard import report
    spec = {'source_audit': {'promotion_eligible': False, 'reason': '学习来源未验证'}}
    state = {'status': 'finished', 'verdict': 'merge_to_iteration_baseline'}
    run = {'spec': spec, 'state': state}
    assert report._status(spec, state)[0] == '诊断对比完成'
    assert '不能晋升' in report._next_step(run)
    assert '不参与' in report._adoption_note(run)
