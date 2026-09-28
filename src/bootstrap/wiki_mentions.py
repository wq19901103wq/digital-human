"""Recall mentions as untrusted leads, with bounded raw conversation windows.

Names come only from legacy Wiki fields and raw sender metadata. Matching a name
never establishes identity; the repair model must resolve it against the messages.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
import re
import unicodedata

from ..iteration.storage import write_json


def normalize(text):
    return unicodedata.normalize('NFKC', str(text)).casefold().strip()


def legacy_names(wiki, title):
    """Use only explicit name/alias fields, not names scattered through biography."""
    terms = {title: {'wiki_title'}}
    heading = ''
    for line in wiki.splitlines():
        if line.startswith('#'):
            heading = line.lstrip('# ').strip()
            continue
        value = line.strip().lstrip('-* ').strip()
        match = re.match(r'(?:姓名|名字|别名|昵称|曾用名|name|aliases)\s*[:：]\s*(.+)', value, re.I)
        if match:
            value = match[1]
        elif heading not in ('别名', '昵称', '曾用名'):
            continue
        for term in re.split(r'[、,，;；/|]', value):
            term = term.strip(' \t`\"“”「」')
            if 2 <= len(term) <= 40 and not re.search(r'[：:。！？!?（）()\[\]]', term):
                terms.setdefault(term, set()).add('unverified_legacy_name')
    return terms


def select_mentions(path, base_rows, account, self_account, wiki, title, radius=8, gap=600):
    from .wiki_repair import _stamp
    terms = legacy_names(wiki, title)
    owners, chats = defaultdict(set), defaultdict(list)
    for row in base_rows:
        if row['sender_id'] == account and row['sender'].strip():
            terms.setdefault(row['sender'].strip(), set()).add('account_display_name')
    with Path(path).open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            row = json.loads(raw)
            parts = row['chat_id'].split(':')
            if len(parts) < 3 or parts[1] != self_account:
                continue
            event = row.get('event', {})
            sender = str(event.get('sender_id') or '')
            name = str(row.get('sender') or '')
            if sender and name:
                owners[normalize(name)].add(sender)
            if row['chat_type'] not in ('private', 'group'):
                continue
            if sender == account and name.strip():
                terms.setdefault(name.strip(), set()).add('account_display_name')
            chats[row['chat_id']].append(dict(line=number, row_sha256=hashlib.sha256(raw).hexdigest(),
                message_id=row['message_id'], timestamp=row['timestamp'], chat_id=row['chat_id'],
                chat_type=row['chat_type'], sender_id=sender, sender=name, is_self=row['is_self'],
                message_kind=str(event.get('kind') or 'unknown'), text=row['text']))
    # Account strings are useful in @ mentions, but also only a retrieval lead.
    terms.setdefault(account, set()).add('exact_account_token')
    normalized = defaultdict(set)
    for term, origins in terms.items():
        if len(normalize(term)) >= 2:
            normalized[normalize(term)].update(origins)
    patterns = {term: re.compile((r'(?<![a-z0-9_])' if term[0].isascii() and term[0].isalnum() else '')
        + re.escape(term) + (r'(?![a-z0-9_])' if term[-1].isascii() and term[-1].isalnum() else ''))
        for term in normalized}
    base = {r['line'] for r in base_rows}
    selected, anchors, hits = {}, [], Counter()
    for chat in sorted(chats):
        rows = sorted(chats[chat], key=lambda r: (_stamp(r['timestamp']), r['line']))
        for i, row in enumerate(rows):
            message = normalize(row['text'])
            matches = [term for term, pattern in patterns.items() if pattern.search(message)]
            if not matches:
                continue
            hits.update(matches)
            context = [r for r in rows[max(0, i-radius):i+radius+1]
                       if abs(_stamp(r['timestamp']) - _stamp(row['timestamp'])) <= gap]
            if not any(r['line'] not in base for r in context):
                continue
            anchors.append(dict(line=row['line'], chat_type=row['chat_type'], terms=matches, identity_status='unverified_mention',
                ambiguous_terms=[term for term in matches if owners[term] - {account}]))
            selected.update((r['line'], r) for r in context)
    rows = sorted(selected.values(), key=lambda r: (r['chat_id'], _stamp(r['timestamp']), r['line']))
    return rows, dict(policy='legacy_and_display_name_mentions_are_leads_not_identity',
        anchors=anchors, terms=[dict(term=term, origins=sorted(origins),
            observed_accounts=sorted(owners[term]), matched_messages=hits[term])
            for term, origins in sorted(normalized.items())],
        context_rows=len(rows), added_rows=len(set(selected) - base),
        added_anchor_rows=len(anchors),
        added_rows_by_chat_type=dict(Counter(r['chat_type'] for r in rows if r['line'] not in base)),
        radius=radius, gap_seconds=gap)


def extend_facts(client, directory, meta, wiki_lines, rows, selection, draft, workers, max_chars,
                 progress, status):
    """Two passes only on newly recalled windows; successful base work is untouched."""
    from .wiki_repair import make_batches, extract_batch, reread_batch, identified
    batches = make_batches(rows, max_chars)
    facts, reviews = {}, {}
    progress.update(mention_batches=len(batches), mention_completed=0, mention_reread_completed=0)
    def recall(batch):
        lines = {r['line'] for r in batch}
        return dict(policy=selection['policy'], anchors=[a for a in selection['anchors'] if a['line'] in lines],
                    terms=selection['terms'])
    for stage in ('extract', 'reread'):
        progress['stage'] = 'mentions_' + stage
        status()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            if stage == 'extract':
                pending = {pool.submit(extract_batch, client, directory, meta, wiki_lines, batch,
                           recall=recall(batch)): i for i, batch in enumerate(batches)}
            else:
                pending = {pool.submit(reread_batch, client, directory, meta, wiki_lines, batch,
                           facts[i], draft, recall=recall(batch)): i for i, batch in enumerate(batches)}
            for future in as_completed(pending):
                i = pending[future]
                try:
                    value = future.result()
                    if stage == 'extract':
                        facts[i] = identified(value['facts'])
                        progress['mention_completed'] += 1
                    else:
                        reviews[i] = value
                        progress['mention_reread_completed'] += 1
                except Exception as exc:
                    progress['failed'] += 1
                    progress['errors'].append(dict(stage=progress['stage'], batch=i, error=str(exc)[-800:]))
                status()
        if progress['failed']:
            return None
    final = []
    for i in sorted(reviews):
        by_id = {f['id']: f for f in facts[i]}
        for revision in reviews[i]['revisions']:
            final.extend([by_id[revision['fact_id']]] if revision['action'] == 'retain' else revision['replacements'])
        final.extend(reviews[i]['additions'])
    write_json(directory.parent / 'intermediate/mention_reread.json', reviews)
    return identified(final)
