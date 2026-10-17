"""Bounded, source-bound storage for the unchanged Wiki selection algorithms."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
import time

from ...config import sha256_file
from ...iteration.storage import file_lock, read_json, write_json
from . import wiki_mentions, wiki_repair
from .wiki_library import digest

MODE = 'wiki_source_offsets_and_mention_hits_v1'
_installation_lock = RLock()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def vocabulary(jobs):
    terms = set()
    for job in jobs:
        wiki = Path(job['wiki']['path'])
        if file_digest(wiki) != job['wiki']['sha256']:
            raise ValueError('legacy Wiki changed after batch preparation')
        terms.update(wiki_mentions.legacy_names(wiki.read_text(encoding='utf-8'), wiki.stem))
        terms.add(job['account'])
    return sorted({wiki_mentions.normalize(term) for term in terms
                   if len(wiki_mentions.normalize(term)) >= 2})


class MentionMatcher:
    """Find overlapping names, including several names at the same position."""

    def __init__(self, terms):
        trie = {}
        for term in sorted(terms):
            branch = trie
            for char in term:
                branch = branch.setdefault(char, {})
            branch[None] = term

        def expression(branch):
            choices = [re.escape(char) + expression(child)
                       for char, child in branch.items() if char is not None]
            if None in branch:
                choices.append('')
            return '(?:' + '|'.join(choices) + ')' if choices else ''

        self.trie = trie
        self.pattern = re.compile('(?=(' + expression(trie) + '))') if trie else None

    def find(self, text):
        found = set()
        if self.pattern is None:
            return found
        boundary = 'abcdefghijklmnopqrstuvwxyz0123456789_'
        for match in self.pattern.finditer(text):
            branch = self.trie
            for offset, char in enumerate(match[1], 1):
                branch = branch[char]
                term = branch.get(None)
                if term is None:
                    continue
                start, end = match.start(), match.start() + offset
                if term[0].isascii() and term[0].isalnum() and start and text[start - 1] in boundary:
                    continue
                if term[-1].isascii() and term[-1].isalnum() and end < len(text) and text[end] in boundary:
                    continue
                found.add(term)
        return found


class SelectionIndex:
    """One shared locator/hit index; each reader owns its SQLite and raw stream."""

    def __init__(self, source, self_account, terms, directory, *, receipt=None,
                 verified_source_stamp=None):
        self.source = Path(source['path']).resolve()
        self.source_sha256 = source['sha256']
        self.self_account = self_account
        self.source_stamp = file_stamp(self.source)
        self.source_verified = verified_source_stamp is not None
        if self.source_verified and self.source_stamp != verified_source_stamp:
            raise ValueError('raw source changed after verification')
        self.identity = dict(mode=MODE, source_path=str(self.source),
            source_sha256=self.source_sha256, self_account=self_account, terms=terms,
            implementation_sha256=sha256_file(Path(__file__)),
            selectors={Path(module.__file__).name: sha256_file(Path(module.__file__))
                       for module in (wiki_repair, wiki_mentions)})
        self.path = Path(directory) / (digest(self.identity) + '.sqlite3')
        self.receipt = Path(receipt) if receipt else None
        self.stats = {}

    def unchanged(self):
        if file_stamp(self.source) != self.source_stamp:
            raise ValueError('raw source changed during indexed Wiki selection')
        if hasattr(self, 'index_stamp') and file_stamp(self.path) != self.index_stamp:
            raise ValueError('Wiki selection index changed during run')

    def progress(self, stage, **fields):
        if self.receipt:
            write_json(self.receipt, dict(mode=MODE, stage=stage, updated_at=time.time(),
                source_sha256=self.source_sha256, index=str(self.path),
                implementation_sha256=self.identity['implementation_sha256'],
                selectors=self.identity['selectors'], sqlite_cache_kib=2048,
                raw_text_stored=False, **fields))

    def prepare(self):
        started = time.monotonic()
        certificate = self.path.with_suffix('.json')
        with file_lock(self.path.with_suffix('.lock')):
            self.unchanged()
            if not self.source_verified:
                if file_digest(self.source) != self.source_sha256:
                    raise ValueError('raw source changed before Wiki index preparation')
                self.unchanged()
                self.source_verified = True
            if certificate.exists():
                self.index_stamp = file_stamp(self.path)
                saved = read_json(certificate)
                if saved['identity'] != self.identity or file_digest(self.path) != saved['sha256']:
                    raise ValueError('Wiki selection cache binding or bytes changed')
                self.unchanged()
                self.stats = saved['stats'] | dict(reused=True)
            else:
                self.path.unlink(missing_ok=True)
                self.progress('indexing_metadata', rows=0)
                self.stats = self.build() | dict(reused=False)
                self.index_stamp = file_stamp(self.path)
                index_digest = file_digest(self.path)
                self.unchanged()
                write_json(certificate, dict(identity=self.identity, stats=self.stats,
                                            sha256=index_digest))
            with sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True) as db:
                self.unchanged()
                self.bound_terms = frozenset(self.identity['terms']) | frozenset(
                    term for term, in db.execute('SELECT DISTINCT term FROM names WHERE conversation=1'))
            self.unchanged()
        self.stats['prepare_seconds'] = round(time.monotonic() - started, 3)
        self.progress('ready', **self.stats)
        return self

    def build(self):
        pending = self.path.with_suffix('.building')
        pending.unlink(missing_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(pending) as db:
                db.execute('PRAGMA cache_size=-2048')
                db.execute('PRAGMA temp_store=FILE')
                db.execute('CREATE TABLE raw_rows (line INTEGER PRIMARY KEY, offset INTEGER, size INTEGER, '
                    'chat_id TEXT, chat_type TEXT, peer TEXT, sender_id TEXT, stamp REAL)')
                db.execute('CREATE TABLE names (name TEXT, term TEXT, sender_id TEXT, conversation INTEGER, first_line INTEGER, '
                           'PRIMARY KEY(name, sender_id, conversation)) WITHOUT ROWID')
                rows, names, offset, total = [], [], 0, 0
                hasher = hashlib.sha256()
                with self.source.open('rb') as stream:
                    for total, raw in enumerate(stream, 1):
                        hasher.update(raw)
                        row = json.loads(raw)
                        parts = row['chat_id'].split(':')
                        if len(parts) >= 3 and parts[1] == self.self_account:
                            sender = str(row.get('event', {}).get('sender_id') or '')
                            name = str(row.get('sender') or '')
                            conversation = row['chat_type'] in ('private', 'group')
                            if sender and name:
                                names.append((name, wiki_mentions.normalize(name), sender, conversation, total))
                            if conversation:
                                rows.append((total, offset, len(raw), row['chat_id'], row['chat_type'],
                                             parts[2], sender, wiki_repair._stamp(row['timestamp'])))
                        offset += len(raw)
                        if total % 10000 == 0:
                            db.executemany('INSERT INTO raw_rows VALUES (?,?,?,?,?,?,?,?)', rows)
                            db.executemany('INSERT OR IGNORE INTO names VALUES (?,?,?,?,?)', names)
                            rows, names = [], []
                            if total % 50000 == 0:
                                self.progress('indexing_metadata', rows=total)
                db.executemany('INSERT INTO raw_rows VALUES (?,?,?,?,?,?,?,?)', rows)
                db.executemany('INSERT OR IGNORE INTO names VALUES (?,?,?,?,?)', names)
                if hasher.hexdigest() != self.source_sha256:
                    raise ValueError('raw source changed before Wiki index creation')
                db.execute('CREATE TABLE messages AS SELECT *, '
                    'ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY stamp, line) AS seq_all, '
                    "SUM(chat_type='group') OVER (PARTITION BY chat_id ORDER BY stamp, line) AS seq_group "
                    'FROM raw_rows')
                db.execute('DROP TABLE raw_rows')
                db.execute('CREATE UNIQUE INDEX by_line ON messages(line)')
                db.execute('CREATE INDEX by_chat ON messages(chat_id, seq_all)')
                db.execute('CREATE INDEX by_group ON messages(chat_id, chat_type, seq_group)')
                db.execute('CREATE INDEX by_sender ON messages(sender_id, chat_type, chat_id)')
                db.execute('CREATE INDEX by_private ON messages(peer, chat_type, chat_id)')
                db.execute('CREATE INDEX by_name_owner ON names(sender_id, conversation)')
                db.execute('CREATE INDEX by_name_term ON names(term, sender_id)')
                terms = set(self.identity['terms'])
                terms.update(term for term, in db.execute('SELECT DISTINCT term FROM names '
                             'WHERE conversation=1') if len(term) >= 2)
                matcher = MentionMatcher(terms)
                db.execute('CREATE TABLE hits (term TEXT, line INTEGER, PRIMARY KEY(term,line)) WITHOUT ROWID')
                self.progress('indexing_mentions', rows=0, terms=len(terms))
                hits = []
                hasher = hashlib.sha256()
                with self.source.open('rb') as stream:
                    for number, raw in enumerate(stream, 1):
                        hasher.update(raw)
                        row = json.loads(raw)
                        parts = row['chat_id'].split(':')
                        if len(parts) >= 3 and parts[1] == self.self_account and row['chat_type'] in ('private', 'group'):
                            hits.extend((term, number) for term in matcher.find(wiki_mentions.normalize(row['text'])))
                        if number % 10000 == 0:
                            db.executemany('INSERT INTO hits VALUES (?,?)', hits)
                            hits = []
                            if number % 50000 == 0:
                                self.progress('indexing_mentions', rows=number, terms=len(terms))
                db.executemany('INSERT INTO hits VALUES (?,?)', hits)
                if hasher.hexdigest() != self.source_sha256:
                    raise ValueError('raw source changed during Wiki mention indexing')
                stats = dict(total_rows=total, indexed_rows=db.execute('SELECT COUNT(*) FROM messages').fetchone()[0],
                             terms=len(terms), mention_hits=db.execute('SELECT COUNT(*) FROM hits').fetchone()[0])
                self.unchanged()
            pending.replace(self.path)
            return stats
        finally:
            pending.unlink(missing_ok=True)

    @contextmanager
    def reader(self):
        self.unchanged()
        db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            self.unchanged()
            db.execute('PRAGMA cache_size=-2048')
            with self.source.open('rb') as stream:
                self.unchanged()
                yield db, stream
            self.unchanged()
        finally:
            db.close()

    def rows(self, db, stream, lines):
        selected = []
        ordered = sorted(set(lines))
        for start in range(0, len(ordered), 500):
            part = ordered[start:start + 500]
            for entry in db.execute('SELECT * FROM messages WHERE line IN (' + ','.join('?' for _ in part) + ')', part):
                stream.seek(entry['offset'])
                raw = stream.read(entry['size'])
                row = json.loads(raw)
                event = row.get('event', {})
                selected.append(dict(line=entry['line'], row_sha256=hashlib.sha256(raw).hexdigest(),
                    message_id=row['message_id'], timestamp=row['timestamp'], chat_id=row['chat_id'],
                    chat_type=row['chat_type'], sender_id=str(event.get('sender_id') or ''),
                    sender=str(row.get('sender') or ''), is_self=row['is_self'],
                    message_kind=str(event.get('kind') or 'unknown'), text=row['text']))
        return sorted(selected, key=lambda row: (row['chat_id'], wiki_repair._stamp(row['timestamp']), row['line']))

    def window(self, db, anchor, radius, gap, *, groups_only=False):
        sequence = 'seq_group' if groups_only else 'seq_all'
        clause = " AND chat_type='group'" if groups_only else ''
        return db.execute('SELECT * FROM messages WHERE chat_id=?' + clause + ' AND ' + sequence +
            ' BETWEEN ? AND ? AND abs(stamp-?)<=? ORDER BY ' + sequence,
            (anchor['chat_id'], anchor[sequence] - radius, anchor[sequence] + radius, anchor['stamp'], gap))

    def select_history(self, path, account, self_account, radius=8, gap=600):
        selected, private_count, anchors = set(), 0, 0
        with self.reader() as (db, stream):
            chats = db.execute("SELECT DISTINCT chat_id,peer FROM messages WHERE (chat_type='private' AND peer=?) "
                               "OR (chat_type='group' AND sender_id=?) ORDER BY chat_id", (account, account))
            for chat in chats:
                groups_only = chat['peer'] != account
                clause = " AND chat_type='group'" if groups_only else ''
                first = db.execute('SELECT * FROM messages WHERE chat_id=?' + clause + ' ORDER BY seq_all LIMIT 1',
                                   (chat['chat_id'],)).fetchone()
                if first['chat_type'] == 'private':
                    lines = [entry[0] for entry in db.execute('SELECT line FROM messages WHERE chat_id=?', (chat['chat_id'],))]
                    selected.update(lines)
                    private_count += len(lines)
                else:
                    for anchor in db.execute('SELECT * FROM messages WHERE chat_id=? AND sender_id=?' + clause +
                                             ' ORDER BY seq_all', (chat['chat_id'], account)):
                        anchors += 1
                        selected.update(entry['line'] for entry in self.window(db, anchor, radius, gap, groups_only=groups_only))
            rows = self.rows(db, stream, selected)
        if not rows or not any(row['sender_id'] == account for row in rows):
            raise ValueError('no raw history for the exact subject account')
        return rows, dict(path=str(self.source), sha256=self.source_sha256,
            total_rows=self.stats['total_rows'], selected_rows=len(rows), private_rows=private_count,
            group_rows=len(rows) - private_count, group_anchor_rows=anchors,
            selection='all_exact_private_plus_group_account_windows', group_radius=radius,
            group_gap_seconds=gap, nickname_only_group_mentions_included=False,
            earliest_timestamp=min(row['timestamp'] for row in rows),
            latest_timestamp=max(row['timestamp'] for row in rows))

    def select_mentions(self, path, base_rows, account, self_account, wiki, title, radius=8, gap=600):
        terms = wiki_mentions.legacy_names(wiki, title)
        for row in base_rows:
            if row['sender_id'] == account and row['sender'].strip():
                terms.setdefault(row['sender'].strip(), set()).add('account_display_name')
        with self.reader() as (db, stream):
            for entry in db.execute('SELECT name FROM names WHERE sender_id=? AND conversation=1 ORDER BY first_line', (account,)):
                if entry['name'].strip():
                    terms.setdefault(entry['name'].strip(), set()).add('account_display_name')
            terms.setdefault(account, set()).add('exact_account_token')
            normalized = defaultdict(set)
            for term, origins in terms.items():
                if len(wiki_mentions.normalize(term)) >= 2:
                    normalized[wiki_mentions.normalize(term)].update(origins)
            missing = set(normalized) - self.bound_terms
            if missing:
                raise ValueError('Wiki mention terms are not bound to the prepared index')
            owners, matches = defaultdict(set), defaultdict(set)
            hits = Counter()
            for term in normalized:
                owners[term].update(entry[0] for entry in db.execute('SELECT sender_id FROM names WHERE term=?', (term,)))
                for entry in db.execute('SELECT line FROM hits WHERE term=?', (term,)):
                    matches[entry['line']].add(term)
                    hits[term] += 1
            base = {row['line'] for row in base_rows}
            selected, anchors = set(), []
            ordered = sorted(matches)
            candidates = []
            for start in range(0, len(ordered), 500):
                part = ordered[start:start + 500]
                candidates.extend(db.execute('SELECT * FROM messages WHERE line IN (' + ','.join('?' for _ in part) + ')', part))
            candidates.sort(key=lambda entry: (entry['chat_id'], entry['stamp'], entry['line']))
            for entry in candidates:
                context = [item['line'] for item in self.window(db, entry, radius, gap)]
                if not any(line not in base for line in context):
                    continue
                matched = [term for term in normalized if term in matches[entry['line']]]
                anchors.append(dict(line=entry['line'], chat_type=entry['chat_type'], terms=matched,
                    identity_status='unverified_mention', ambiguous_terms=[term for term in matched if owners[term] - {account}]))
                selected.update(context)
            rows = self.rows(db, stream, selected)
        return rows, dict(policy='legacy_and_display_name_mentions_are_leads_not_identity',
            anchors=anchors, terms=[dict(term=term, origins=sorted(origins), observed_accounts=sorted(owners[term]),
                matched_messages=hits[term]) for term, origins in sorted(normalized.items())],
            context_rows=len(rows), added_rows=len(selected - base), added_anchor_rows=len(anchors),
            added_rows_by_chat_type=dict(Counter(row['chat_type'] for row in rows if row['line'] not in base)),
            radius=radius, gap_seconds=gap)


@contextmanager
def accelerated(index):
    """Scope storage overrides to this source without editing frozen generators."""
    with _installation_lock:
        history, mentions = wiki_repair.select_history, wiki_mentions.select_mentions

        def select_history(path, account, self_account, *args, **kwargs):
            function = index.select_history if Path(path).resolve() == index.source and self_account == index.self_account else history
            return function(path, account, self_account, *args, **kwargs)

        def select_mentions(path, base_rows, account, self_account, *args, **kwargs):
            function = index.select_mentions if Path(path).resolve() == index.source and self_account == index.self_account else mentions
            return function(path, base_rows, account, self_account, *args, **kwargs)

        wiki_repair.select_history, wiki_mentions.select_mentions = select_history, select_mentions
        try:
            yield
        finally:
            wiki_repair.select_history, wiki_mentions.select_mentions = history, mentions


def for_batch(manifest, output, *, record_runtime=True, verified_source_stamp=None):
    source = manifest['messages']
    output = Path(output)
    return SelectionIndex(source, manifest['self_account'], vocabulary(manifest['jobs']),
        output.parent / '.cache/wiki-selection',
        receipt=output / 'source_selection_runtime.json' if record_runtime else None,
        verified_source_stamp=verified_source_stamp).prepare()


def check_batch(output, jobs=1):
    """Compare original and indexed selections and timings without model calls."""
    if jobs < 1:
        raise ValueError('jobs must be positive')
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    from . import wiki_batch
    if manifest['pipeline'] != wiki_batch.pipeline():
        raise ValueError('generation pipeline changed')
    source = manifest['messages']
    source_stamp = file_stamp(source['path'])
    if file_digest(source['path']) != source['sha256']:
        raise ValueError('raw source changed after batch preparation')
    index = for_batch(manifest, output, record_runtime=False, verified_source_stamp=source_stamp)
    compared = []
    eligible = [job for job in manifest['jobs'] if not job.get('reuse')]
    for job in eligible[:jobs]:
        wiki = Path(job['wiki']['path'])
        args = (source['path'], job['account'], job['self_account'])
        started = time.monotonic()
        reference, coverage = wiki_repair.select_history(*args)
        mention_args = (source['path'], reference, job['account'], job['self_account'], wiki.read_text(encoding='utf-8'), wiki.stem)
        mention_reference = wiki_mentions.select_mentions(*mention_args)
        reference_seconds = time.monotonic() - started
        started = time.monotonic()
        selected = index.select_history(*args)
        mention_selected = index.select_mentions(*mention_args)
        indexed_seconds = time.monotonic() - started
        if selected != (reference, coverage) or mention_selected != mention_reference:
            raise ValueError('indexed Wiki selection differs from frozen reference')
        compared.append(dict(job=job['id'], selected_rows=len(reference), mention_rows=len(mention_reference[0]),
            equivalent=True, reference_seconds=round(reference_seconds, 3), indexed_seconds=round(indexed_seconds, 3)))
    index.unchanged()
    return dict(mode=MODE, preparation=index.stats, jobs=compared, equivalent=True,
                model_requests=0, frozen_inputs_changed=False)
