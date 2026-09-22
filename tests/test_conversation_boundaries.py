"""Source-backed regressions for stale replies and time-bounded self bursts."""
import copy
import json
import random

import pytest

from scripts import check
from src.bootstrap import conversations, data_quality, history
from src.bootstrap.build_fewshot_pool import extract_examples
from src.config import ConfigError, sha256_file
from src.generator.history import HistoryError
from src.generator.history_sources import HistorySources
from src.iteration import baseline, datasets, versions
from src.iteration.storage import write_json
from test_data_boundaries import timeline
from test_iteration import priv


def messages(*sequence):
    return [dict(chat_id='chat', chat_type='private', source_chat_id='source', chat_name='合成聊天',
                 sender='本人' if self_ else '朋友', is_self=self_, text=f'合成消息{i}', timestamp=t)
            for i, (t, self_) in enumerate(sequence)]


def source_fixture(directory, source, *, rule=None, frozen=True, max_context=8):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'manifest.json', dict(id=directory.name))
    rule = rule or conversations.policy()
    rows = history.examples(source, max_context=max_context, segmentation=rule)
    history.write_rows(directory / 'messages.jsonl', source)
    history.write_rows(directory / 'fewshot_pool.jsonl', rows)
    path = directory / datasets.ROLES['gen_learning'][1]
    history.write_rows(path, [history.as_case(r) for r in rows])
    protocol = dict(development_start=10000000, acceptance_start=20000000, unseen_chat_ids=[])
    if frozen:
        protocol.update(reply_segmentation=rule, max_context=max_context)
    write_json(directory / 'purposes.json', dict(history_policy='complete_before_input_v1',
        protocol=protocol, roles={'gen_learning': dict(file=path.name, sha256=sha256_file(path), total=len(rows))}))
    write_json(directory / 'report.json', dict(review_status='approved', history_policy='complete_before_input_v1',
        examples_sha256=sha256_file(directory / 'fewshot_pool.jsonl')))
    return rows


@pytest.mark.parametrize('seconds,retained', [(600, True), (601, False), (86400, False)])
def test_first_reply_must_follow_recent_context(seconds, retained):
    stats = {}
    rows = extract_examples(messages((100, False), (100 + seconds, True)), statistics=stats)
    assert bool(rows) is retained
    assert stats.get('stale_response', 0) == int(not retained)


def test_keep_complete_short_burst_but_never_pair_later_self_talk():
    stats = {}
    source = messages((100, False), (101, True), (221, True), (342, True), (343, True),
                      (344, False), (345, True))
    rows = history.examples(source, statistics=stats)
    assert [r['reply'] for r in rows] == [['合成消息1', '合成消息2'], ['合成消息6']]
    assert rows[0]['source_span']['end_timestamp'] == 221
    assert stats['split_self_runs'] == 1 and stats['unattached_self_messages'] == 2
    assert rows[1]['source_span']['reply_start'] == 6


def test_context_uses_last_internal_gap_not_gap_outside_window():
    source = messages((100, False), (10000, False), (20000, False), (30000, False), (30001, True))
    # Window starts after the first gap; another two gaps are inside it.
    row = history.examples(source, max_context=3)[0]
    assert row['context_message_ids'] == [source[3]['message_id']]
    assert row['source_span']['start'] == 3


def test_protected_replies_remain_excluded():
    source = messages((100, False), (101, True), (102, True))
    assert extract_examples(source, protected_indices={'chat': {2}}, protect_margin=0) == []


@pytest.mark.parametrize('bad', [None, '101', float('nan'), float('inf'), -1, True])
def test_invalid_source_timestamp_fails_closed(bad):
    source = messages((100, False), (101, True))
    source[1]['timestamp'] = bad
    with pytest.raises(ConfigError):
        extract_examples(source)


def test_reversed_source_time_is_rejected():
    with pytest.raises(ConfigError, match='倒序'):
        extract_examples(messages((102, False), (101, True)))


def test_source_guard_accepts_only_complete_frozen_burst(tmp_path):
    source = messages((100, False), (101, True), (200, True), (500, True))
    rows = source_fixture(tmp_path, source)
    guard = HistorySources(tmp_path)
    assert guard.validate(rows[0], example=True) == rows[0]
    shortened = copy.deepcopy(rows[0])
    shortened['source_span']['end'] = 2
    with pytest.raises(HistoryError, match='气泡不完整'):
        guard.validate(shortened, example=True)
    extended = copy.deepcopy(rows[0])
    extended['source_span']['end'] = 4
    with pytest.raises(HistoryError, match='时间边界'):
        guard.validate(extended, example=True)


def test_source_guard_rejects_stale_answer_and_shortened_context(tmp_path):
    rows = source_fixture(tmp_path, messages((100, False), (101, False), (102, True)))
    short = copy.deepcopy(rows[0])
    short['source_span']['start'] = 1
    with pytest.raises(HistoryError, match='完整片段'):
        HistorySources(tmp_path).validate(short, example=True)
    late = tmp_path / 'late'
    rows = source_fixture(late, messages((100, False), (1000, True)),
                          rule=conversations.policy(response_gap_seconds=1800))
    purpose = json.loads((late / 'purposes.json').read_text())
    purpose['protocol']['reply_segmentation'] = conversations.policy()
    write_json(late / 'purposes.json', purpose)
    with pytest.raises(HistoryError, match='时间边界'):
        HistorySources(late).validate(rows[0], example=True)


def test_audit_respects_frozen_rule_and_counts_legacy_defects(tmp_path):
    source = messages((100, False), (1000, True), (1300, True))
    rule = conversations.policy(response_gap_seconds=1800, reply_gap_seconds=300)
    source_fixture(tmp_path / 'new', source, rule=rule)
    assert data_quality.audit(tmp_path / 'new')['passed']
    source_fixture(tmp_path / 'legacy', source, rule=rule, frozen=False)
    result = data_quality.audit(tmp_path / 'legacy')
    assert not result['passed']
    group = result['groups']['fewshot_pool']
    assert group['affected'] == 1
    assert group['reasons'] == {'stale_response': 1, 'reply_crosses_burst': 1}
    assert group['max_gaps_seconds']['response'] == 900
    assert result['isolation'] == dict(learning_development_overlap=0, learning_fixed_overlap=0,
                                       development_fixed_overlap=0)


def test_unified_data_check_keeps_failed_counts_and_leaves_pointers(priv, monkeypatch, capsys):
    source_fixture(versions.DATA_ROOT / 'd-9000', messages((100, False), (1000, True)),
                   rule=conversations.policy(response_gap_seconds=1800), frozen=False)
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    before = versions.POINTERS_PATH.read_bytes()
    assert check.main(['data', '--instance', 'demo', '--data', 'd-9000']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['data_quality']['d-9000']['groups']['fewshot_pool']['affected'] == 1
    assert report['model_requests_prohibited'] and not report['evaluation_started']
    assert versions.POINTERS_PATH.read_bytes() == before


def test_publication_checks_all_roles_and_never_finalizes_invalid_source(priv, monkeypatch):
    source = timeline(chats=100, sessions=20)
    rows = history.examples(source)
    roles, protocol = history.plan(rows, total=8, train_total=8)
    protocol['max_context'] = 8
    before = versions.POINTERS_PATH.read_bytes()
    vid = history.publish(source, rows, roles, protocol, {})
    proof = json.loads((versions.data_version_dir(vid) / 'data_quality.json').read_text())
    assert proof['passed'] and all(g['affected'] == 0 for g in proof['groups'].values())
    assert proof['groups']['development']['total'] == proof['groups']['fixed_test']['total'] == 8
    assert versions.POINTERS_PATH.read_bytes() == before
    finalized = []
    monkeypatch.setattr(versions, 'finalize_data_version', finalized.append)
    with pytest.raises(HistoryError):
        history.publish([], rows, roles, protocol, {})
    assert not finalized and versions.POINTERS_PATH.read_bytes() == before


def test_rebuild_uses_frozen_source_archive(priv, monkeypatch, capsys):
    from scripts import prepare_history
    source = versions.DATA_ROOT / 'd-9000'
    messages = timeline(chats=100, sessions=20)
    messages[0]['text'] = '段一\u2028段二\u2029段三\u0085段四'
    source_fixture(source, messages)
    write_json(priv / 'data_policy.json', dict(total=8, train_total=8))
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    monkeypatch.setattr('sys.argv', ['prepare_history.py', '--instance', 'demo',
        '--source-data', 'd-9000', '--max-context', '30', '--build'])
    before = versions.POINTERS_PATH.read_bytes()
    prepare_history.main()
    ref = capsys.readouterr().out.split('Created ')[-1].strip()
    directory = versions.data_version_dir(ref)
    manifest = json.loads((directory / 'manifest.json').read_text())
    protocol = json.loads((directory / 'purposes.json').read_text())['protocol']
    assert manifest['source_files'] == {str(source / 'messages.jsonl'): sha256_file(source / 'messages.jsonl')}
    assert protocol['source_data_ref'] == 'd-9000' and protocol['max_context'] == 30
    assert (directory / 'messages.jsonl').read_bytes() == (source / 'messages.jsonl').read_bytes()
    assert versions.POINTERS_PATH.read_bytes() == before


def test_holdout_selection_preserves_learning_capacity_without_changing_quotas():
    rows = history.examples(timeline(chats=10, sessions=20))
    groups = sorted({r['source_span']['chat_id'] for r in rows if r['relationship'] == 'group'})
    only_learning_group = random.Random(42).sample(groups, 1)[0]
    rows = [r for r in rows if r['source_span']['end_timestamp'] >= 150000
            or r['relationship'] == 'private' or r['source_span']['chat_id'] == only_learning_group]
    roles, protocol = history.plan(rows, total=4, train_total=4,
                                   development_start=150000, acceptance_start=230000)
    assert protocol['holdout_selection_attempt'] > 0
    assert only_learning_group not in protocol['unseen_chat_ids']
    assert all(len(roles[role]) == 4 for role in
               ('gen_optimization', 'judge_training', 'development', 'fixed_test'))
    assert sum(r['relationship'] == 'group' for r in roles['gen_optimization']) == 3
    assert protocol['development_start'] == 150000 and protocol['acceptance_start'] == 230000


def adoption_report(priv, monkeypatch, capsys, *options):
    directory = versions.DATA_ROOT / 'd-9000'
    source_fixture(directory, messages((100, False), (102, True)))
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    report = priv / 'acceptance.json'
    code = check.main(['--output', str(report), 'data', '--instance', 'demo',
                       '--data', directory.name, *options])
    capsys.readouterr()
    return directory, report, code


def test_data_adoption_keeps_models_and_distinct_iteration_pointers(priv, monkeypatch, capsys):
    directory, report, code = adoption_report(priv, monkeypatch, capsys)
    assert code == 0
    before = {**versions.load_pointers(), 'iteration_gen': 'g-0002',
              'iteration_judge': 'j-0003', 'baseline_migration': 'previous'}
    versions.save_pointers(before)
    receipt = baseline.prepare_data(directory.name, report, reason='完整数据验收')
    assert versions.load_pointers() == before
    proof = json.loads(receipt.read_text())
    assert proof['before'] == before and proof['kind'] == 'data_baseline_migration'
    assert all(proof[key] is False for key in
               ('performance_claim', 'model_compatibility_verified', 'evaluation_started'))
    adopted = baseline.apply(receipt.stem)
    assert adopted == {**before, 'data': directory.name, 'baseline_migration': receipt.stem}
    assert baseline.apply(receipt.stem) == adopted


@pytest.mark.parametrize('change', ['messages.jsonl', 'manifest.json', 'gen_learning.jsonl',
                                    'report', 'pointer'])
def test_data_adoption_rejects_changes_since_acceptance(priv, monkeypatch, capsys, change):
    directory, report, code = adoption_report(priv, monkeypatch, capsys)
    assert code == 0
    receipt = baseline.prepare_data(directory.name, report, reason='完整数据验收')
    if change == 'pointer':
        versions.save_pointers({**versions.load_pointers(), 'iteration_gen': 'g-newer'})
    else:
        path = report if change == 'report' else directory / change
        path.write_bytes(path.read_bytes() + b'\n')
    before = versions.POINTERS_PATH.read_bytes()
    with pytest.raises(ConfigError):
        baseline.apply(receipt.stem)
    assert versions.POINTERS_PATH.read_bytes() == before


@pytest.mark.parametrize('options,expected_code', [
    (['--response-gap-seconds', '1'], 1),
    (['--response-gap-seconds', '1800'], 0),
    (['--import-coverage-only'], 1),
])
def test_data_adoption_requires_complete_frozen_policy_acceptance(
        priv, monkeypatch, capsys, options, expected_code):
    directory, report, code = adoption_report(priv, monkeypatch, capsys, *options)
    assert code == expected_code
    before = versions.POINTERS_PATH.read_bytes()
    with pytest.raises(ConfigError, match='完整 check.py data 报告'):
        baseline.prepare_data(directory.name, report, reason='不能采用非完整验收')
    assert versions.POINTERS_PATH.read_bytes() == before
