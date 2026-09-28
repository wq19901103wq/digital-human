"""Locator-only raw index and conservative legacy Wiki anchor recovery."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

from .wiki_repair import _stamp


def build(messages, target, self_account, expected_sha):
    """Index only the requested exporter; retain offsets, never message excerpts."""
    target, messages = Path(target), Path(messages)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix('.building')
    pending.unlink(missing_ok=True)
    hasher = hashlib.sha256()
    with sqlite3.connect(pending) as db, messages.open('rb') as stream:
        db.execute('CREATE TABLE messages (line INTEGER PRIMARY KEY, offset INTEGER, size INTEGER, '
                   'chat_id TEXT, chat_type TEXT, chat_name TEXT, sender_id TEXT, sender TEXT, '
                   'stamp REAL, day TEXT)')
        batch, offset = [], 0
        for number, raw in enumerate(stream, 1):
            hasher.update(raw)
            row = json.loads(raw)
            parts = row['chat_id'].split(':', 2)
            if len(parts) == 3 and parts[1] == self_account:
                stamp = _stamp(row['timestamp'])
                batch.append((number, offset, len(raw), row['chat_id'], row['chat_type'],
                    row.get('chat_name', ''), str((row.get('event') or {}).get('sender_id') or ''),
                    row.get('sender', ''), stamp,
                    datetime.fromtimestamp(stamp, ZoneInfo('Asia/Shanghai')).date().isoformat()))
            offset += len(raw)
            if len(batch) >= 10000:
                db.executemany('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)', batch)
                batch = []
        db.executemany('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)', batch)
        if hasher.hexdigest() != expected_sha:
            raise ValueError('raw source changed before index creation')
        db.execute('CREATE INDEX by_day ON messages(day)')
        db.execute('CREATE INDEX by_chat ON messages(chat_id, stamp, line)')
        db.execute('CREATE INDEX by_name ON messages(chat_name)')
    pending.replace(target)


class SourceIndex:
    def __init__(self, path, messages, sha256):
        self.db = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.stream = Path(messages).open('rb')
        self.path, self.sha256 = str(Path(messages).resolve()), sha256

    def close(self):
        self.stream.close()
        self.db.close()

    def query(self, where='1', args=()):
        return self.db.execute('SELECT * FROM messages WHERE ' + where + ' ORDER BY chat_id, stamp, line', args)

    def read(self, entry):
        self.stream.seek(entry['offset'])
        raw = self.stream.read(entry['size'])
        r = json.loads(raw)
        event = r.get('event') or {}
        return dict(line=entry['line'], row_sha256=hashlib.sha256(raw).hexdigest(),
            message_id=r['message_id'], timestamp=r['timestamp'], chat_id=r['chat_id'],
            chat_type=r['chat_type'], sender_id=str(event.get('sender_id') or ''),
            sender=r.get('sender', ''), is_self=r['is_self'],
            message_kind=event.get('kind', 'unknown'), text=r['text'])

    def locator(self, entry):
        return {k: v for k, v in self.read(entry).items() if k not in ('text', 'sender', 'chat_type')} | dict(
            path=self.path, sha256=self.sha256, kind='raw_message')

    def rows(self, *, chat_id=None, lines=None):
        if chat_id is not None:
            return [self.read(r) for r in self.query('chat_id=?', (chat_id,))]
        rows = []
        ordered = sorted(set(lines or []))
        for start in range(0, len(ordered), 500):
            part = ordered[start:start+500]
            rows.extend(self.read(r) for r in self.query('line IN (' + ','.join('?' for _ in part) + ')', part))
        return sorted(rows, key=lambda r: (r['chat_id'], _stamp(r['timestamp']), r['line']))

    def anchors(self, anchor, candidates=(), *, prefix=False):
        matched = []
        text = normalized(anchor['text'])
        if len(text) < (8 if prefix else 4):
            return matched
        for entry in self.query('day=?', (anchor['day'],)):
            if candidates and entry['sender_id'] not in candidates:
                continue
            if anchor.get('sender') and entry['sender'] != anchor['sender']:
                continue
            chat = anchor['chat']
            if chat in ('私聊', 'private'):
                if entry['chat_type'] != 'private':
                    continue
            elif chat not in (entry['chat_name'], entry['chat_id'].split(':', 2)[-1]):
                continue
            # Quoted/replied-to text does not prove the current sender's identity.
            primary = normalized(self.read(entry)['text'].split('[引用', 1)[0])
            if primary == text or (prefix and len(text) >= 16 and primary.startswith(text)):
                matched.append(dict(entry))
        return matched

    def windows(self, entries, radius=8, gap=600):
        keep, chats = set(), defaultdict(list)
        for entry in entries:
            chats[entry['chat_id']].append(entry)
        for chat, anchors in chats.items():
            rows = list(self.query('chat_id=?', (chat,)))
            positions = {r['line']: i for i, r in enumerate(rows)}
            for anchor in anchors:
                i = positions[anchor['line']]
                keep.update(r['line'] for r in rows[max(0, i-radius):i+radius+1]
                            if abs(r['stamp'] - anchor['stamp']) <= gap)
        return sorted(keep)


def normalized(text):
    return re.sub(r'\s+', '', text).rstrip('…').removesuffix('...')


def topic_anchors(lines):
    result = []
    for number, line in enumerate(lines, 1):
        match = re.match(r'^\s*-\s*\[(\d{4}-\d{2}-\d{2})\]\s+(.+?) @ (.+?):\s*(.*)', line)
        if match:
            day, sender, chat, text = match.groups()
            result.append(dict(line=number, day=day, sender=sender, chat=chat, text=text))
        elif result and not line.startswith('#'):
            result[-1]['text'] += '\n' + line
    for anchor in result:
        anchor['text'] = anchor['text'].split('[引用', 1)[0].strip()
    return result


def person_anchors(lines):
    """Only explicit dated utterance lists can prove a page's speaker identity."""
    result, heading = [], ''
    for number, line in enumerate(lines, 1):
        if line.startswith('#'):
            heading = line
        if '说过的话' not in heading:
            continue
        source = re.search(r'来源[：:]\s*(.+?)/(\d{4}-\d{2}-\d{2})', line)
        if source:
            quoted = re.findall(r'“([^”]+)”|"([^"]+)"', line[:source.start()])
            result.extend(dict(line=number, chat=source[1], day=source[2], text=a or b)
                          for a, b in quoted)
    return result


def resolve_person(index, lines, candidates):
    evidence, accounts, utterances = {}, set(), set()
    anchors = person_anchors(lines)
    unmatched, ambiguous = [], []
    for anchor in anchors:
        matches = index.anchors(anchor, candidates)
        senders = {r['sender_id'] for r in matches}
        if len(senders) != 1:
            unmatched.append(anchor['line'])
            if senders:
                ambiguous.append(anchor['line'])
                for entry in matches:
                    evidence[entry['line']] = index.locator(entry)
            continue
        accounts.update(senders)
        utterances.add(normalized(anchor['text']))
        for entry in matches:
            evidence[entry['line']] = index.locator(entry)
    reason = ('mixed_raw_accounts' if len(accounts) > 1 else
              'ambiguous_utterance_authorship' if ambiguous else
              'verified_utterance_anchors' if len(utterances) >= 2 and max(map(len, utterances), default=0) >= 8 else
              'insufficient_identity_evidence')
    return dict(binding=reason, account=next(iter(accounts)) if reason == 'verified_utterance_anchors' else '',
                candidates=sorted(candidates), supported_accounts=sorted(accounts),
                evidence=list(evidence.values()), unmatched_old_lines=sorted(set(unmatched)),
                ambiguous_old_lines=sorted(set(ambiguous)))


def select_topic(index, lines):
    entries, unmatched, matched = {}, [], []
    anchors = topic_anchors(lines)
    for anchor in anchors:
        matches = index.anchors(anchor, prefix=True)
        if len(matches) != 1:
            unmatched.append(anchor['line'])
            continue
        entry = matches[0]
        entries[entry['line']] = entry
        matched.append(dict(old_line=anchor['line'], source=index.locator(entry)))
    return dict(lines=index.windows(entries.values()), binding='dated_raw_topic_anchors',
                anchors=matched, anchor_count=len(anchors), unmatched_old_lines=unmatched,
                selection='old_topic_anchors_plus_same_chat_radius8_gap600')


def select_group(index, lines, title, filenames):
    ids = set(re.findall(r'(?<!\w)(\d+@chatroom)(?!\w)', '\n'.join(lines[:12])))
    ids.update(name.replace('_chatroom', '@chatroom') for name in filenames
               if re.fullmatch(r'\d+(?:@|_)chatroom', name))
    candidates = set()
    if ids:
        for group in ids:
            candidates.update(r['chat_id'] for r in index.query('chat_type=? AND chat_id LIKE ?',
                              ('group', '%:' + group)))
    else:
        for name in {title, *filenames}:
            candidates.update(r['chat_id'] for r in index.query('chat_type=? AND chat_name=?', ('group', name)))
    if len(candidates) != 1:
        return dict(binding='ambiguous_group' if candidates else 'no_raw_group', candidates=sorted(candidates))
    chat = next(iter(candidates))
    count = index.db.execute('SELECT COUNT(*) FROM messages WHERE chat_id=?', (chat,)).fetchone()[0]
    return dict(binding='exact_group_id' if ids else 'unique_group_name_candidate',
                chat_id=chat, selected_rows=count, selection='complete_exact_group_history')
