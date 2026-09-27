"""Exact account / confirmed alias lookup of frozen, structured Wiki pages.

No vector index, query planner or historical text retrieval is involved.
This projection is diagnostic; it does not admit a review Share for evaluation.
"""
from __future__ import annotations

from .background_features import FACT_FIELDS

VERSION = 'direct_wiki_v1'
IDENTITY_STATES = {'user_confirmed', 'source_account_bound'}


def context_accounts(directory, case):
    """Recover speaker accounts by exact saved context IDs, never by display name."""
    from ..generator import history_sources
    sources = history_sources.load(directory)
    sources.validate(case)
    chat = case['source_span']['chat_id']
    wanted = set(case['context_message_ids'])
    return [{'message_id': m.message_id, 'sender_id': (m.event or {}).get('sender_id'),
             'sender': m.sender, 'is_self': m.is_self}
            for m in sources.chats[chat] if m.message_id in wanted]


def _aliases(person, chat):
    if (person.get('status') == 'user_confirmed' and
            person.get('decision', {}).get('scope', {}).get('chat_id') == chat):
        return list(dict.fromkeys([person['name'], *(person.get('aliases') or [])]))
    return []


def _page_facts(knowledge, selected, case):
    facts, refs, corrections = [], {}, []
    for fact in knowledge.get('fact_claims', []):
        page_ids = sorted(selected.intersection(fact.get('page_ids', [])))
        if not page_ids:
            continue
        scope_case = fact.get('scope_case_id')
        if scope_case and scope_case != case['case_id']:
            continue
        status = fact.get('status', '')
        if not (status.startswith(('有据', '有依据', '行为有据')) or status in (
                '案例范围内确认', '纠正旧摘要', '限定旧摘要范围', '梦境，不作现实关系')):
            continue
        observed = fact.get('observed_at', [])
        if observed and any(t > case['input_cutoff']['timestamp'] for t in observed):
            continue
        if not observed and not scope_case:
            continue
        ref = f'fact:{len(facts) + 1}'
        facts.append({'ref': ref, 'page_ids': page_ids,
                      **{key: fact[key] for key in FACT_FIELDS if key in fact},
                      **{key: fact[key] for key in ('scope_chat_id', 'scope_case_id') if key in fact}})
        refs[ref] = {'claim_id': fact['id'], 'evidence_refs': fact.get('evidence_refs', [])}
        if scope_case:
            corrections.append(ref)
    return facts, refs, corrections


def diagnostic_projection(knowledge, case, blind, accounts):
    """Read matched pages, retaining claim scope; no similarity-ranked snippets.

Both blind options contribute mention lookups symmetrically. Their text is never
used as a Wiki fact. Labels and the reference answer are not read by this code.
"""
    chat = case.get('source_span', {}).get('chat_id', '')
    parts = chat.split(':', 2)
    if len(parts) != 3 or parts[0] not in ('private', 'group'):
        raise ValueError('直接 Wiki 读取需要来源账号和聊天 ID')
    if not isinstance(case.get('input_cutoff', {}).get('timestamp'), (int, float)):
        raise ValueError('直接 Wiki 读取需要输入截止时间')
    identities = {p['id']: p for p in knowledge.get('identities', [])
                  if p.get('status') in IDENTITY_STATES}
    pages = {p['id']: p for p in knowledge.get('pages', [])}
    self_id = 'person:' + parts[1]
    requests, unresolved, bindings = {}, [], []

    def request(page_id, reason):
        requests.setdefault(page_id, set()).add(reason)

    request(self_id, 'self_account')
    if parts[0] == 'group':
        request(chat, 'current_group')
    else:
        request('person:' + parts[2], 'private_peer_account')
    for row in accounts:
        account = parts[1] if row['is_self'] else row.get('sender_id')
        person_id = 'person:' + account if account else None
        binding = {'sender': row['sender'], 'role': 'self' if row['is_self'] else 'other',
                   'person_id': person_id, 'message_id': row.get('message_id')}
        if binding not in bindings:
            bindings.append(binding)
        if person_id:
            request(person_id, 'context_speaker_account')
        else:
            unresolved.append({'sender': row['sender'], 'reason': 'missing_sender_account'})

    mentions = '\n'.join([*blind['context_original'], *blind['option_A'], *blind['option_B']])
    aliases = {}
    for person in identities.values():
        for alias in _aliases(person, chat):
            if alias:
                aliases.setdefault(alias, set()).add(person['id'])
    for alias, people in sorted(aliases.items()):
        if alias not in mentions:
            continue
        if len(people) == 1:
            request(next(iter(people)), 'confirmed_alias:' + alias)
        else:
            unresolved.append({'alias': alias, 'reason': 'ambiguous_confirmed_alias',
                               'person_ids': sorted(people)})

    selected, lookups = set(), []
    for page_id, reasons in sorted(requests.items()):
        state = ('identity_unconfirmed' if page_id.startswith('person:') and page_id not in identities
                 else 'found' if page_id in pages else 'missing_page')
        lookups.append({'id': page_id, 'reasons': sorted(reasons), 'status': state})
        if state == 'found':
            selected.add(page_id)
    facts, refs, corrections = _page_facts(knowledge, selected, case)
    people = set(selected)
    people.update(f.get(key) for f in facts for key in ('subject_id', 'object_id', 'perspective_person_id'))
    named = []
    for person_id in sorted(p for p in people if p in identities):
        person = identities[person_id]
        named.append({'id': person_id, 'name': person['name'], 'aliases': _aliases(person, chat)})
    return {'payload': {'lookup_mode': VERSION, 'self_person_id': self_id,
                       'identities': named, 'speaker_bindings': bindings,
                       'wiki_pages': [{'id': page_id, 'title': pages[page_id]['title'],
                                       'fact_refs': [f['ref'] for f in facts if page_id in f['page_ids']]}
                                      for page_id in sorted(selected)],
                       'facts': facts, 'lookups': lookups, 'unresolved': unresolved},
            'source_refs': refs, 'case_correction_refs': corrections,
            'diagnostic_only': True, 'historical_admission': False}
