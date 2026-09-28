"""Prepare ordinary Wiki jobs from raw group-member snapshots and API pages.

The snapshot labels accounts at its own observation time. It does not identify
legacy pages, rename historical senders, or contribute biographical claims.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from ..config import sha256_file
from ..iteration.storage import atomic_write, file_lock, write_json
from . import wiki_batch
from .ingest import message_identity, message_kind, normalize_event
from .wiki_library import digest


def _plain(value):
    """Media XML and nested app payloads are not readable chat text."""
    value = str(value or '').strip()
    return value if not value.startswith('<') else '[引用内容未解析]'


def _event(message):
    # A whitelist prevents raw XML, media credentials and API transport fields
    # from entering normalized history or model context.
    adapted = {k: message[k] for k in ('createTime', 'senderUsername', 'isSend',
        'localType', 'content', 'senderDisplayName') if k in message}
    adapted['platformMessageId'] = str(message.get('serverId') or '')
    kind = message_kind(adapted)
    if kind not in ('text', 'quote', 'system'):
        adapted['content'] = ''
    elif kind == 'system' and '<' in str(adapted.get('content') or ''):
        adapted['content'] = ''
    if kind == 'quote':
        content = str(message.get('content') or '')
        if '<' in content:
            if re.search(r'<!DOCTYPE|<!ENTITY', content, re.I):
                raise ValueError('XML declarations are not supported in quote payloads')
            try:
                root = ET.fromstring(content[content.index('<'):])
            except ET.ParseError as exc:
                raise ValueError('unparseable quote XML; preserve source and repair import') from exc
            app = root if root.tag == 'appmsg' else root.find('.//appmsg')
            reference = None if app is None else app.find('refermsg')
            if app is None or reference is None:
                raise ValueError('quote XML lacks appmsg/refermsg')
            get = lambda key: reference.findtext(key) or ''
            quote_type = get('type')
            quoted = _plain(get('content')) if quote_type == '1' else f'[引用非文本消息，类型 {quote_type or "未知"}]'
            author, label = get('fromusr'), get('displayname')
            adapted.update(replyToMessageId=get('svrid'), quotedContent=quoted,
                quotedSender=label, quotedType=quote_type)
            adapted['content'] = ('[引用 ' + (label or author or '未知说话人')
                + (f'（账号 {author}）' if author else '') + '：' + quoted + ']\n'
                + _plain(app.findtext('title')))
    text, event = normalize_event(adapted)
    event.pop('original_content', None)
    return text, event


def load_pages(paths, chatroom, self_account):
    """Normalize API pages without changing any frozen experimental dataset."""
    rows, seen, inputs = [], {}, []
    duplicates = 0
    observed_self = set()
    for path in sorted({Path(p).resolve() for p in paths}):
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if data.get('success') is not True or data.get('talker') != chatroom:
            raise ValueError(f'API page failed or belongs to a different group: {path}')
        messages = data.get('messages')
        if not isinstance(messages, list):
            raise ValueError(f'API page lacks messages: {path}')
        inputs.append(dict(path=str(path), sha256=sha, messages=len(messages)))
        for index, message in enumerate(messages):
            sender = message.get('senderUsername')
            stamp, sent = message.get('createTime'), message.get('isSend')
            # System/media events may have no sender in the exporter. Retain
            # them as unowned context, never as an account or a member anchor.
            unowned = sender in (None, '') and message_kind(message) not in ('text', 'quote')
            if not unowned and (not isinstance(sender, str) or not sender or sender.startswith('_sender_')):
                raise ValueError(f'missing real sender account: {path} /messages/{index}')
            if (type(stamp) is not int or stamp <= 0
                    or type(sent) not in (int, bool) or sent not in (0, 1)):
                raise ValueError(f'invalid message time or isSend: {path} /messages/{index}')
            if sent:
                observed_self.add(sender)
            if bool(sent) != (sender == self_account):
                raise ValueError('source sender and isSend disagree with self account')
            text, event = _event(message)
            # API local IDs are scoped to a chat. Refuse absent IDs rather than
            # deduplicating coincidentally identical utterances.
            if event['id'].startswith('fallback:'):
                local_id = message.get('localId')
                if type(local_id) is not int or local_id <= 0:
                    raise ValueError('API message lacks both server and local ID')
                event['id'] = 'local:' + str(local_id)
            row = dict(chat_id=f'group:{self_account}:{chatroom}', chat_type='group',
                chat_name=chatroom, sender=message.get('senderDisplayName') or sender or '未知发送者',
                is_self=bool(sent), timestamp=stamp, text=text, event=event)
            # Compare complete raw messages too: sanitization must not hide
            # conflicting XML/metadata behind the same source ID.
            fingerprint = digest(message)
            if event['id'] in seen:
                if seen[event['id']] != fingerprint:
                    raise ValueError('conflicting API messages with the same ID')
                duplicates += 1
                continue
            seen[event['id']] = fingerprint
            row['message_id'] = message_identity(row)
            row['source'] = dict(kind='raw_api_message', path=str(path), sha256=sha,
                json_pointer=f'/messages/{index}', message_sha256=fingerprint,
                server_id=str(message.get('serverId') or ''), local_id=message.get('localId'),
                sort_sequence=message.get('sortSeq'))
            rows.append(row)
    if observed_self != {self_account}:
        raise ValueError('pages must independently establish the exporting self account')
    # Preserve exporter order within a second; content hashes are not temporal
    # evidence and would scramble the context used for pronoun resolution.
    def order(row):
        source = row['source']
        sequence, local_id = source['sort_sequence'], source['local_id']
        return (row['timestamp'], sequence if type(sequence) is int else row['timestamp'] * 1000,
                local_id if type(local_id) is int else 0, row['message_id'])
    rows.sort(key=order)
    return rows, dict(inputs=inputs, retained_rows=len(rows), duplicates=duplicates,
        earliest_timestamp=min(r['timestamp'] for r in rows),
        latest_timestamp=max(r['timestamp'] for r in rows),
        event_counts=dict(Counter(r['event']['kind'] for r in rows)),
        unattributed_event_counts=dict(Counter(r['event']['kind'] for r in rows
                                               if not r['event']['sender_id'])))


def prepare(members, pages, output, self_account, config, *, non_friends_only=False,
            priority_accounts=(), max_chars=120000):
    """Compile raw metadata into frozen jobs consumed by the existing runner."""
    members, output = Path(members).resolve(), Path(output).resolve()
    with file_lock(output / '.lock', blocking=False):
        if (output / 'manifest.json').exists():
            raise ValueError('batch already prepared; use run to resume')
        raw = members.read_bytes()
        roster = json.loads(raw)
        chatroom = roster.get('chatroomId')
        if (roster.get('success') is not True or not isinstance(chatroom, str)
                or not chatroom.endswith('@chatroom') or type(roster.get('updatedAt')) is not int
                or roster['updatedAt'] <= 0 or not isinstance(roster.get('members'), list)):
            raise ValueError('member snapshot lacks group identity or observation time')
        snapshot = dict(kind='raw_group_members', path=str(members),
            sha256=hashlib.sha256(raw).hexdigest(), chatroom_id=chatroom,
            observed_at=roster['updatedAt'])
        rows, coverage = load_pages(pages, chatroom, self_account)
        counts = Counter(r['event']['sender_id'] for r in rows)
        messages = output / 'source/messages.jsonl'
        atomic_write(messages, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        write_json(output / 'source/import.json', dict(schema='wiki_group_api_import_v1',
            adapter_sha256=sha256_file(Path(__file__)), members=snapshot, **coverage))
        jobs, gaps, seen = [], [], set()
        for index, member in enumerate(roster['members']):
            account = member.get('wxid')
            if not isinstance(account, str) or not account or account.startswith('_sender_') or account in seen:
                raise ValueError('member snapshot has a missing, placeholder or repeated account')
            seen.add(account)
            if account == self_account or (non_friends_only and member.get('isFriend') is not False):
                continue
            title = member.get('displayName') or member.get('nickname') or account
            title = re.sub(r'[\r\n]', ' ', str(title))
            job_id = 'member-' + digest([self_account, chatroom, account])[:20]
            binding = dict(snapshot, json_pointer=f'/members/{index}', account=account,
                           display_name=title, is_friend=member.get('isFriend'))
            base = dict(id=job_id, title=title, kind='person', account=account,
                self_account=self_account, chat_id=f'group:{self_account}:{chatroom}',
                binding='raw_member_snapshot', binding_evidence=[binding],
                sources=[snapshot], raw_sender_rows=counts[account])
            if not counts[account]:
                gaps.append(dict(base, reason='no_raw_sender_messages'))
                continue
            # An empty page seed contributes only the raw snapshot display label.
            # Never copy a same-name legacy biography or any reviewed aliases.
            filename = re.sub(r'[/\\\x00-\x1f]', '_', title)[:80].strip(' .') or account
            seed = output / 'seeds' / job_id / (filename + '.md')
            atomic_write(seed, '# ' + title + '\n')
            locator = dict(kind='raw_member_page_seed', path=str(seed), sha256=sha256_file(seed))
            jobs.append(dict(base, sources=[snapshot, locator], wiki=locator,
                             output=str(output / 'jobs' / job_id)))
        priorities = set(priority_accounts)
        if priorities - {j['account'] for j in jobs}:
            raise ValueError('priority account is absent or has no raw messages')
        jobs.sort(key=lambda j: (j['account'] not in priorities, j['raw_sender_rows'], j['id']))
        manifest = dict(schema='wiki_batch_v1', wiki_root=str(output / 'seeds'),
            messages=dict(path=str(messages), sha256=sha256_file(messages)),
            self_account=self_account, config=config, max_chars=max_chars,
            jobs=jobs, gaps=gaps, pipeline=wiki_batch.pipeline(),
            member_source=snapshot, source_coverage=coverage,
            identity_policy='exact raw member accounts; display labels valid only at snapshot time; '
                            'no legacy identity/alias merge',
            input_policy='raw_chats_and_raw_member_labels_only; no curated reviews',
            summary=dict(files=len(jobs), documents=len(jobs)+len(gaps), jobs=len(jobs), reusable=0,
                bindings=dict(Counter(j['binding'] for j in jobs)),
                gaps=dict(Counter(g['reason'] for g in gaps))))
        write_json(output / 'manifest.json', manifest)
        return manifest['summary']
