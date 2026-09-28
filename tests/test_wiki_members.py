"""Raw member snapshots extend coverage without repairing legacy identities by name."""
from copy import deepcopy
import json

import pytest

from src.bootstrap import wiki_batch, wiki_delivery, wiki_members as members
from src.iteration.storage import read_json, write_json


def message(sender='peer1', stamp=1001, local=2, **kwargs):
    return dict(senderUsername=sender, createTime=stamp, isSend=int(sender == 'self'),
        localType=1, localId=local, serverId=str(local + 100), sortSeq=stamp * 1000,
        content='一条原始发言', **kwargs)


def page(tmp_path, messages=None, name='page.json', **kwargs):
    path = tmp_path / name
    write_json(path, dict(success=True, talker='room@chatroom',
        messages=messages if messages is not None else [message('self', 1000, 1), message()], **kwargs))
    return path


def roster(tmp_path):
    path = tmp_path / 'members.json'
    write_json(path, dict(success=True, chatroomId='room@chatroom', updatedAt=2000000,
        members=[dict(wxid='self', displayName='自己', isFriend=False),
            dict(wxid='peer1', displayName='同名', isFriend=False),
            dict(wxid='peer2', displayName='同名', isFriend=False),
            dict(wxid='peer3', displayName='缺消息', isFriend=False),
            dict(wxid='friend', displayName='好友', isFriend=True)]))
    return path


def test_prepare_exact_nonfriends_and_keep_missing_sources(tmp_path):
    snapshot = roster(tmp_path)
    raw = page(tmp_path, [message('peer2', 1002, 3), message('peer1', 1001, 2),
                          message('self', 1000, 1), message('friend', 1003, 4)])
    output = tmp_path / 'batch'
    summary = members.prepare(snapshot, [raw], output, 'self', {}, non_friends_only=True,
                              priority_accounts=['peer2'])
    manifest = read_json(output / 'manifest.json')
    assert summary['jobs'] == 2 and summary['gaps'] == {'no_raw_sender_messages': 1}
    jobs = manifest['jobs']
    assert [j['account'] for j in jobs] == ['peer2', 'peer1']
    assert len({j['id'] for j in jobs}) == 2
    assert {j['title'] for j in jobs} == {'同名'}
    for job in jobs:
        assert job['binding'] == 'raw_member_snapshot'
        assert job['binding_evidence'][0]['observed_at'] == 2000000
        assert job['binding_evidence'][0]['account'] == job['account']
        assert job['binding_evidence'][0]['json_pointer'].startswith('/members/')
        assert open(job['wiki']['path']).read() == '# 同名\n'
    rows = [json.loads(line) for line in (output / 'source/messages.jsonl').read_text().splitlines()]
    assert rows[1]['sender'] == 'peer1'  # Later snapshot name must not rewrite history.
    assert rows[1]['chat_name'] == 'room@chatroom'
    assert rows[1]['source']['json_pointer'] == '/messages/1'
    assert rows[1]['source']['path'] == str(raw)
    coverage = read_json(output / 'source/import.json')
    assert coverage['retained_rows'] == 4 and coverage['earliest_timestamp'] == 1000
    inspected = wiki_delivery.inspect([output])
    assert inspected['summary']['counts'] == {'pending': 2}
    assert inspected['summary']['coverage_gaps'] == {'no_raw_sender_messages': 1}
    assert inspected['gaps'][0]['account'] == 'peer3'
    with pytest.raises(ValueError, match='already prepared'):
        members.prepare(snapshot, [raw], output, 'self', {})


@pytest.mark.parametrize('quoted_type,quoted_content,expected', [
    ('1', '这是引用者所引的原话', '这是引用者所引的原话'),
    ('3', '&lt;img aeskey="private-key"/&gt;', '[引用非文本消息，类型 3]'),
])
def test_quote_preserves_outer_inner_accounts_without_media_payload(tmp_path, quoted_type, quoted_content, expected):
    quote = message()
    quote.update(localType=244813135921, senderAvatarKey='private-avatar',
        rawContent='private-transport', parsedContent='not trusted',
        content=f'<msg><appmsg><title>外层回复</title><refermsg><type>{quoted_type}</type>'
        f'<fromusr>inner-account</fromusr><displayname>原说话人</displayname>'
        f'<content>{quoted_content}</content><svrid>42</svrid></refermsg></appmsg></msg>')
    raw = page(tmp_path, [message('self', 1000, 1), quote])
    rows, _ = members.load_pages([raw], 'room@chatroom', 'self')
    row = rows[1]
    assert row['event']['sender_id'] == 'peer1'
    assert row['event']['kind'] == 'quote' and row['event']['quote']['replyToMessageId'] == '42'
    assert f'[引用 原说话人（账号 inner-account）：{expected}]\n外层回复' == row['text']
    serialized = json.dumps(row, ensure_ascii=False)
    assert all(term not in serialized for term in ('private-key', 'private-avatar', 'private-transport', '<img'))
    assert 'original_content' not in row['event']


def test_raw_media_stays_placeholder(tmp_path):
    media = message()
    media.update(localType=3, content='<msg><img aeskey="private-key"/></msg>',
                 senderAvatarKey='private-avatar')
    rows, _ = members.load_pages([page(tmp_path, [message('self', 1000, 1), media])],
                                 'room@chatroom', 'self')
    assert rows[1]['text'] == '[图片：导出未提供可读内容]'
    assert 'private-' not in json.dumps(rows)


def test_unowned_system_and_prefixed_media_are_retained_without_guessed_authors(tmp_path):
    media, system = message(local=2), message(local=3)
    media.update(senderUsername=None, localType=43, content='sender-prefix:\n<video aeskey="private-key"/>')
    system.update(senderUsername=None, localType=10000, content='一条系统通知')
    rows, coverage = members.load_pages([page(tmp_path, [message('self', 1000, 1), media, system])],
                                         'room@chatroom', 'self')
    assert [row['event']['sender_id'] for row in rows] == ['self', '', '']
    assert rows[1]['text'] == '[视频：导出未提供可读内容]'
    assert rows[2]['text'] == '[系统事件] 一条系统通知'
    assert coverage['unattributed_event_counts'] == {'video': 1, 'system': 1}
    assert 'private-key' not in json.dumps(rows) and 'sender-prefix' not in json.dumps(rows)


@pytest.mark.parametrize('content', ['<!DOCTYPE msg><msg/>', '<msg>', '<msg><appmsg/></msg>'])
def test_unparseable_quote_is_not_silently_treated_as_persons_words(tmp_path, content):
    quote = message()
    quote.update(localType=244813135921, content=content)
    with pytest.raises(ValueError, match='XML'):
        members.load_pages([page(tmp_path, [message('self', 1000, 1), quote])], 'room@chatroom', 'self')


def test_overlap_local_identity_and_source_order(tmp_path):
    first, second = message(local=3), message(local=2)
    first['serverId'] = second['serverId'] = '0'
    raw = page(tmp_path, [first, second, message('self', 1000, 1)])
    overlap = page(tmp_path, [second], name='overlap.json')
    rows, coverage = members.load_pages([raw, overlap], 'room@chatroom', 'self')
    assert coverage['duplicates'] == 1 and len(rows) == 3
    assert [r['event']['id'] for r in rows] == ['platform:101', 'local:2', 'local:3']
    assert len({r['message_id'] for r in rows}) == 3
    changed = deepcopy(second)
    changed['rawContent'] = 'different raw message despite same normalized text'
    write_json(overlap, dict(success=True, talker='room@chatroom', messages=[changed]))
    with pytest.raises(ValueError, match='conflicting'):
        members.load_pages([raw, overlap], 'room@chatroom', 'self')


@pytest.mark.parametrize('mutation', ['group', 'self', 'missing_self', 'missing_sender', 'time', 'sent', 'id'])
def test_invalid_identity_and_source_are_rejected(tmp_path, mutation):
    data = dict(success=True, talker='room@chatroom', messages=[message('self', 1000, 1), message()])
    row = data['messages'][1]
    if mutation == 'group':
        data['talker'] = 'other@chatroom'
    elif mutation == 'self':
        row['isSend'] = 1
    elif mutation == 'missing_self':
        data['messages'] = [row]
    elif mutation == 'missing_sender':
        row['senderUsername'] = '_sender_同名'
    elif mutation == 'time':
        row['createTime'] = 0
    elif mutation == 'sent':
        row['isSend'] = 0.0
    else:
        row.update(serverId='0', localId=None)
    raw = tmp_path / 'page.json'
    write_json(raw, data)
    with pytest.raises(ValueError):
        members.load_pages([raw], 'room@chatroom', 'self')


def test_existing_runner_refuses_changed_member_snapshot(tmp_path, monkeypatch):
    snapshot, raw, output = roster(tmp_path), page(tmp_path), tmp_path / 'batch'
    members.prepare(snapshot, [raw], output, 'self', {}, non_friends_only=True)
    snapshot.write_text(snapshot.read_text() + '\n')
    def forbidden(*args, **kwargs):
        raise AssertionError('changed identity input must not call generation')
    monkeypatch.setattr(wiki_batch.wiki_repair, 'repair', forbidden)
    state = wiki_batch.run(output)
    assert state['stage'] == 'incomplete'
    assert 'changed after batch preparation' in next(iter(state['jobs'].values()))['error']
