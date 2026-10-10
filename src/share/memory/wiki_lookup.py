"""Source-bound Wiki queries and explicitly separate diagnostic page lookup.

WikiLookup admits only source-backed prior knowledge. Diagnostic projection may
include case corrections and must never be used as historical admission.
"""
from __future__ import annotations

import json

from src.data.history.sources import stamp
from src.infrastructure.config import ConfigError
from src.share.knowledge_view import TABLES, card, compact, entity_refs, records
from . import FACT_FIELDS

VERSION = 'direct_wiki_v1'
IDENTITY_STATES = {'user_confirmed', 'source_account_bound'}


def project_structured(data, accounts, chat, prior, *, max_chars=16000):
    """Only relevant exact accounts; all supporting sources must pass prior()."""
    if data['subject']['self_account'] != chat.split(':', 2)[1]:
        return None
    if data['subject']['account'] not in accounts:
        return None
    eligible = [record for record in records(data) if not record.get('scope_case_id')
                and record.get('evidence_refs')
                and all(prior(ref) for ref in record['evidence_refs'])]
    eligible.sort(key=lambda record: (chat in record.get('chat_ids', []),
                                      record.get('last_observed_at', 0), record['id']), reverse=True)
    selected = []
    for record in eligible:
        candidate = card(data, selected + [record], prior=prior)
        if len(compact(candidate)) + 200 <= max_chars:
            selected.append(record)
    result = card(data, selected, prior=prior)
    result['coverage'] = dict(eligible=len(eligible), included=len(selected),
                              omitted_for_budget=len(eligible) - len(selected))
    return result


class WikiLookup:
    """Read a frozen Share against trusted history, without model or prompt logic."""

    def __init__(self, path, sources):
        self.sources = sources
        self.path = path
        self.signature = stamp(self.path)
        self.knowledge = json.loads(self.path.read_text())
        self.structured = self.knowledge.get('schema') == 'wiki_structured_v1'
        expected = (self.knowledge.get('coverage', {}).get('sha256') if self.structured else
                    self.knowledge.get('source_manifest', {}).get('messages', {}).get('sha256'))
        if expected != sources.hashes['messages.jsonl']:
            raise ConfigError('Share 的原始消息来源与本次数据不一致')
        wanted = {entry.get('message_id') for entry in self.knowledge.get('evidence', {}).values()
                  if entry.get('kind') == 'raw_message'}
        self.records = {message.message_id: (chat, message)
                        for chat, rows in sources.chats.items() for message in rows
                        if message.message_id in wanted}
        self.check()

    def check(self):
        self.sources.check()
        if stamp(self.path) != self.signature:
            raise ConfigError('运行中 Share 内容发生变化')

    def query(self, case):
        self.check()
        result = self._projection(case)
        self.check()
        return result

    def _projection(self, case):
        chat = case['source_span']['chat_id']
        account = chat.split(':', 2)[1]
        cutoff = case['input_cutoff']['timestamp']
        excluded = set(case.get('history_excluded_chat_ids', []))
        blocked = set(case['reply_message_ids'])
        evidence = self.knowledge.get('evidence', {})

        def prior(ref):
            entry = evidence.get(ref, {})
            record = self.records.get(entry.get('message_id'))
            if entry.get('kind') != 'raw_message' or record is None:
                return False
            source_chat, message = record
            return (source_chat == entry.get('chat_id') and source_chat not in excluded
                    and source_chat.split(':', 2)[1] == account
                    and message.message_id not in blocked
                    and message.timestamp == entry.get('timestamp' if self.structured else 'observed_at')
                    and (not self.structured or (
                        entry.get('sha256') == self.sources.hashes['messages.jsonl']
                        and entry.get('sender_id') == (message.event or {}).get('sender_id')
                        and entry.get('is_self') == message.is_self))
                    and message.timestamp < cutoff)

        span = case['source_span']
        context = self.sources.chats[chat][span['start']:span['reply_start']]
        if self.structured:
            accounts = {account}
            if chat.startswith('private:'):
                accounts.add(chat.split(':', 2)[2])
            accounts.update((message.event or {}).get('sender_id') for message in context)
            entities = [entity for entity in self.knowledge['entities']
                        if entity.get('evidence_refs') and all(prior(ref) for ref in entity['evidence_refs'])]
            known = {entity['id'] for entity in entities}
            knowledge = dict(self.knowledge, entities=entities)
            for table in TABLES:
                knowledge[table] = [record for record in self.knowledge[table]
                                    if entity_refs(record) <= known]
            view = (project_structured(knowledge, accounts, chat, prior)
                    if knowledge['subject']['entity_id'] in known else None)
            return view or dict(
                schema='wiki_prompt_v1', self_person_id='person:' + account,
                identities=[], facts=[], coverage={'page': 'no_matching_account'})

        facts = [fact for fact in self.knowledge.get('fact_claims', [])
                 if not fact.get('scope_case_id') and fact.get('evidence_refs')
                 and all(prior(ref) for ref in fact['evidence_refs'])
                 and fact.get('observed_at') and all(time < cutoff for time in fact['observed_at'])]
        identities = []
        allowed_ids = {evidence[ref]['message_id'] for fact in facts for ref in fact['evidence_refs']}
        for person in self.knowledge.get('identities', []):
            copy = dict(person)
            supports = person.get('decision', {}).get('historical_support', [])
            if not supports or not all(support.get('message_id') in allowed_ids for support in supports):
                copy['aliases'] = []
                copy['decision'] = {}
            identities.append(copy)
        accounts = [{'message_id': message.message_id, 'sender_id': (message.event or {}).get('sender_id'),
                     'sender': message.sender, 'is_self': message.is_self} for message in context]
        view = diagnostic_projection({**self.knowledge, 'fact_claims': facts, 'identities': identities},
            case, {'context_original': [message['text'] for message in case['context']],
                   'option_A': [], 'option_B': []}, accounts)
        by_id = {fact['id']: fact for fact in facts}
        for fact in view['payload']['facts']:
            original = by_id[view['source_refs'][fact['ref']]['claim_id']]
            fact.update({key: original[key] for key in ('description', 'validity_note') if key in original})
        return view['payload']


def context_accounts(directory, case):
    """Recover speaker accounts by exact saved context IDs, never by display name."""
    from src.data.history import sources as history_sources
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
