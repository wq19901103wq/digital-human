"""Export event coverage and source-backed media boundaries; no model calls."""
import copy
import json

import pytest

from scripts import check, prepare_history
from src.bootstrap import history, import_coverage, ingest
from src.bootstrap.build_fewshot_pool import extract_examples
from src.config import ConfigError, sha256_file
from src.generator.history import HistoryError
from src.generator.history_sources import HistorySources
from src.iteration import versions
from src.iteration.storage import write_json
from test_conversation_boundaries import source_fixture
from test_data_boundaries import timeline
from test_iteration import priv


def event(n=1, **changes):
    return dict(dict(localType=1, type='文本消息', createTime=100+n, isSend=0,
        senderUsername='peer', senderDisplayName='朋友', content='合成文本',
        platformMessageId=str(n)), **changes)


def export(path, rows, *, session=None):
    write_json(path, dict(session=session or dict(type='私聊', wxid='peer', displayName='合成聊天'),
                          messages=rows))
    return path


@pytest.mark.parametrize('local,name,kind', [
    (3, '图片消息', 'image'), (34, '语音消息', 'voice'), (43, '视频消息', 'video'),
    (47, '动画表情', 'sticker'), (42, '名片消息', 'contact'), (48, '位置消息', 'location'),
    (50, '通话消息', 'call'), (10000, '系统消息', 'system'), (999, '其他消息', 'unknown'),
    ((2001 << 32) | 49, '其他消息', 'red_packet'),
    ((2000 << 32) | 49, '其他消息', 'transfer'),
    ((6 << 32) | 49, '其他消息', 'file'), ((19 << 32) | 49, '其他消息', 'forward'),
    ((5 << 32) | 49, '其他消息', 'link'), ((33 << 32) | 49, '其他消息', 'miniapp'),
])
def test_null_nontext_is_retained_as_event_not_text_answer(tmp_path, local, name, kind):
    source = export(tmp_path / 'chat.json', [event(localType=local, type=name, content=None)])
    row, = ingest.load_weflow_files([source])
    assert row['event']['kind'] == kind and row['text']
    assert row['event']['original_content'] == ''
    assert not row['event']['reply_eligible']
    assert row['event']['context_boundary'] == (kind in {'unknown', 'system'})


@pytest.mark.parametrize('local,name,extra', [
    (244813135921, '引用消息', {}), (47, '动画表情', {'appMsgType': '57'}),
    ((8 << 32) | 49, '其他消息', {'appMsgKind': 'quote'}),
    (244813135921, '引用消息', {'appMsgType': '5'}),
])
def test_quote_aliases_keep_rendered_text_and_structured_reference(local, name, extra):
    raw = event(localType=local, type=name, content='同意[引用 朋友：提议]',
        replyToMessageId='123', quotedContent='提议', quotedSender='朋友', quotedType='1', **extra)
    text, meta = ingest.normalize_event(raw)
    assert text == raw['content'] and meta['kind'] == 'quote' and meta['reply_eligible']
    assert meta['quote'] == {k: raw[k] for k in ('replyToMessageId', 'quotedContent', 'quotedSender', 'quotedType')}


def test_message_ids_prevent_same_second_loss_and_detect_conflicting_exports(tmp_path):
    raw = [event(1, createTime=100), event(2, createTime=100),
           event(3, createTime=101, isSend=1, senderUsername='self')]
    one = export(tmp_path / 'one.json', raw)
    # Source filename participates in the existing compatibility chat identity.
    two = tmp_path / 'copy' / 'one.json'
    two.parent.mkdir()
    export(two, raw)
    stats = {}
    rows = ingest.load_weflow_files([one, two], statistics=stats)
    assert len(rows) == 3 and rows[0]['event']['id'] != rows[1]['event']['id']
    assert stats['duplicates_by_type'] == {'text': 3}
    raw[0]['content'] = '另一内容'
    export(two, raw)
    with pytest.raises(ConfigError, match='ID 内容冲突'):
        ingest.load_weflow_files([one, two])


def test_fallback_identity_does_not_collapse_identical_occurrences(tmp_path):
    raw = event(platformMessageId=None)
    rows = ingest.load_weflow_files([export(tmp_path / 'one.json', [raw, raw])])
    assert len(rows) == 2 and rows[0]['event']['id'] != rows[1]['event']['id']


def test_attachment_metadata_is_preserved_and_binds_identity(tmp_path):
    rows = ingest.load_weflow_files([export(tmp_path / 'one.json', [
        event(localType=47, type='动画表情', content='[表情包]', emojiMd5='synthetic')])])
    row = rows[0]
    identity = ingest.message_identity(row)
    assert row['event']['metadata']['emojiMd5'] == 'synthetic'
    row['event']['metadata']['emojiMd5'] = 'changed'
    assert ingest.message_identity(row) != identity


@pytest.mark.parametrize('payload', ['{', '{}', '{"messages": [null]}'])
def test_malformed_export_is_never_silently_skipped(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_text(payload)
    with pytest.raises(ConfigError):
        ingest.load_weflow_files([path])


def imported(tmp_path, raw):
    return ingest.load_weflow_files([export(tmp_path / 'chat.json', raw)])


def test_context_media_remains_and_mixed_self_burst_is_not_partial_answer(tmp_path):
    rows = imported(tmp_path, [event(1), event(2, localType=3, type='图片消息', content=None),
        event(3, isSend=1), event(4, isSend=1, localType=3, type='图片消息', content=None),
        event(5, isSend=1)])
    stats = {}
    assert extract_examples(rows, statistics=stats) == []
    assert stats['non_text_reply'] == 1
    rows[3]['is_self'] = False
    examples = extract_examples(rows)
    assert len(examples) == 2 and all(len(r['reply']) == 1 for r in examples)
    assert '图片' in examples[0]['context'][-1] and '图片' in examples[1]['context'][-1]


@pytest.mark.parametrize('local,name', [(10000, '系统消息'), (999, '其他消息')])
def test_unknown_and_system_events_cut_context_and_runtime_uses_same_rule(tmp_path, local, name):
    rows = imported(tmp_path, [event(1), event(2, localType=local, type=name, content=None),
        event(3), event(4, isSend=1)])
    directory = tmp_path / 'data'
    examples = source_fixture(directory, rows)
    assert examples[0]['source_span']['start'] == 2
    assert HistorySources(directory).validate(examples[0], example=True)
    changed = copy.deepcopy(rows)
    changed[2]['event']['kind'] = 'image'
    history.write_rows(directory / 'messages.jsonl', changed)
    with pytest.raises(HistoryError, match='ID 与内容'):
        HistorySources(directory)


def test_coverage_finds_missing_events_metadata_and_legacy_collisions(tmp_path):
    raw = [event(1, createTime=100), event(2, createTime=100),
           event(3, localType=3, type='图片消息', content=None)]
    path = export(tmp_path / 'chat.json', raw)
    directory = tmp_path / 'data'
    directory.mkdir()
    write_json(directory / 'manifest.json', dict(source_files={str(path): sha256_file(path)}))
    rows = ingest.load_weflow_files([path])
    legacy = {k: v for k, v in rows[0].items() if k != 'event'}
    history.write_rows(directory / 'messages.jsonl', [legacy])
    result = import_coverage.audit(directory)
    assert not result['passed'] and result['missing'] == 2
    assert result['by_type']['text']['missing_with_same_legacy_key'] == 1
    assert result['by_type']['image']['content_unavailable'] == 1
    assert result['missing_event_metadata'] == 1
    history.write_rows(directory / 'messages.jsonl', rows)
    result = import_coverage.audit(directory)
    assert result['passed'] and result['missing'] == result['unmatched_archive_events'] == 0
    path.write_text('{}')
    with pytest.raises(ConfigError, match='冻结清单不符'):
        import_coverage.audit(directory)


def test_reimport_cli_uses_exact_frozen_exports_and_keeps_pointers(priv, monkeypatch, capsys):
    paths = []
    source_rows = timeline(chats=100, sessions=20)
    chats = {}
    for row in source_rows:
        chats.setdefault(row['chat_id'], []).append(row)
    for idx, (chat_id, rows) in enumerate(chats.items()):
        path = priv / f'export{idx}.json'
        raw = [event(n+1, createTime=r['timestamp'], isSend=int(r['is_self']), content=r['text'],
            senderUsername='self' if r['is_self'] else 'peer') for n, r in enumerate(rows)]
        paths.append(export(path, raw, session=dict(type='群聊' if rows[0]['chat_type'] == 'group'
            else '私聊', wxid=chat_id, displayName=chat_id)))
    old = versions.DATA_ROOT / 'd-9000'
    old.mkdir()
    frozen = {str(p): sha256_file(p) for p in paths}
    write_json(old / 'manifest.json', dict(source_files=frozen))
    # An unrelated broken export in the same folder must not be discovered.
    (priv / 'extra.json').write_text('{')
    write_json(priv / 'data_policy.json', dict(total=8, train_total=8))
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    monkeypatch.setattr('sys.argv', ['prepare_history.py', '--instance', 'demo',
        '--reimport-data', 'd-9000', '--max-context', '30', '--build'])
    before = versions.POINTERS_PATH.read_bytes()
    prepare_history.main()
    ref = capsys.readouterr().out.split('Created ')[-1].strip()
    directory = versions.data_version_dir(ref)
    proof = json.loads((directory / 'data_quality.json').read_text())
    assert proof['passed'] and proof['import_coverage']['passed']
    assert proof['import_coverage']['source_events'] == len(source_rows)
    assert json.loads((directory / 'manifest.json').read_text())['source_files'] == frozen
    assert versions.POINTERS_PATH.read_bytes() == before


def test_fixed_check_records_missing_events_without_mutation(priv, monkeypatch, capsys):
    old = versions.DATA_ROOT / 'd-9000'
    old.mkdir()
    path = export(priv / 'chat.json', [event(localType=3, type='图片消息', content=None)])
    write_json(old / 'manifest.json', dict(source_files={str(path): sha256_file(path)}))
    (old / 'messages.jsonl').write_text('')
    monkeypatch.setattr(versions, 'switch_instance', lambda name: priv)
    before = versions.POINTERS_PATH.read_bytes()
    assert check.main(['data', '--instance', 'demo', '--data', 'd-9000', '--import-coverage-only']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['data_quality']['d-9000']['missing'] == 1
    assert versions.POINTERS_PATH.read_bytes() == before
