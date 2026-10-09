"""Conservative nickname discovery from authored chat text, for human review only."""
from collections import Counter, defaultdict, deque
import hashlib
import json
import re

from .wiki_aliases import clean, excerpt, identity_of, key
from .wiki_addresses import AddressCollector


SELF_CALL = re.compile(r'(?:^|[，,。！!（(\n])\s*(?:你们?(?:可以)?|可以|以后(?:可以)?)?叫我(?!们)\s*'
                       r'|(?:同事|大家)(?:都)?叫我(?!们)\s*')
DISLIKED = re.compile(r'(?:我不喜欢别人叫我|别叫我|不要叫我)\s*')
VOCATIVE = re.compile(r'^([\u4e00-\u9fffA-Za-z]{1,3}(?:哥哥|姐姐|老师|老板|哥|姐|总)|'
                      r'[小老阿][\u4e00-\u9fff]{1,2})(?:[，,！!~～]|$)')
NON_NAMES = {'老了', '小红书', '小米', '小公司', '小厂', '小样', '小气', '小赚', '小心', '小心点',
             '小度', '老地方', '老多了', '老早', '老是', '老冷了', '阿里啊', '干嘛',
             '小区', '小区躺', '小波动', '小组长', '小白菜', '老房子', '阿姨费',
             '小学老师', '高中老师'}
COMMAND = re.compile(r'^(?:来|去|买|吃|早|起|睡|回|做|找|上|下|帮|打|送|拿|看|别|停|报|开|'
                     r'不要|没事|走|自己|提交|拍|办|对|继续|谢谢|直接|是)|[了的吧吗呢]|update')
GENERIC_NAMES = {'老哥', '老板', '帅哥', '小哥哥', '老弟', '表哥', '太奶奶'}
ADDRESS_PREFIX = re.compile(r'^(?:(?:谢谢|感谢|羡慕|哎呀|好吧|好嘞|好的|哈哈|恭喜|辛苦)[，,\s]*)+')
PREDICATE_PREFIX = re.compile(r'^(?:都是|不是|不想|是我|就去|去当|说|当)')


def vocative_name(text):
    """Find bounded address candidates; standalone name echoes are not addresses."""
    text = ADDRESS_PREFIX.sub('', text)
    match = VOCATIVE.match(text)
    if match and not text[match.end():].strip(' ，,。.!！?？~～…'):
        match = None
    if not match:
        # A trailing address also occurs in '叫我小蓝就行了，绿总'.
        parts = re.split(r'[，,]', text)
        if len(parts) > 1 and parts[0].strip():
            tail = ADDRESS_PREFIX.sub('', parts[-1].strip()).rstrip('。.!！?？~～')
            match = VOCATIVE.fullmatch(tail)
    if not match:
        return None
    name = match[1]
    if (name in NON_NAMES or PREDICATE_PREFIX.match(name)
            or name.startswith(('我', '你', '他', '她', '大', '各', '一', '这', '那'))):
        return None
    return name


def authored_text(row):
    """Do not attribute quoted or forwarded speech to the current author."""
    if row.get('event', {}).get('kind') not in {'text', 'quote'}:
        return ''
    text = str(row.get('text', '')).split('[引用 ', 1)[0].strip()
    # Inline quotation can also contain somebody else's first person statement.
    return re.sub(r'“[^”]*”|「[^」]*」|"[^"]*"', '', text)


def self_names(text):
    """Extract bounded naming statements, not commands such as '叫我早睡'."""
    for pattern, preference in ((SELF_CALL, 'self_stated'), (DISLIKED, 'disliked')):
        for match in pattern.finditer(text):
            remainder = text[match.end():]
            tail = re.split(r'[，,。！!？?\n\[（）()]|就行|就好', remainder, 1)[0].strip()
            for name in re.split(r'\s+or\s+|或者|或是|或者说|[、/]', tail):
                name = clean(name).rstrip('啦~～')
                if not re.fullmatch(r'[\u4e00-\u9fffA-Za-z· ]{2,12}', name):
                    continue
                if name in NON_NAMES or COMMAND.search(name):
                    continue
                # Long unstructured clauses are not accepted as naming declarations.
                if (len(name) > 4 and not re.fullmatch(r'[A-Za-z ]+', name)
                        and not (len(name) <= 6 and re.match(re.escape(tail) + r'就[行好]', remainder))):
                    continue
                statement = 'reported' if preference != 'disliked' and re.search(r'(?:同事|大家)(?:都)?叫我', match[0]) else preference
                yield name, statement


def discover_context(review, messages, decisions=None, self_account=None):
    """One streaming pass; reuse the existing identity index, never merge accounts."""
    identities = {item['id']: item for item in review['identities']}
    private_members = defaultdict(set)
    for ident, item in identities.items():
        for chat in item['chats']:
            if chat.startswith('private:'):
                private_members[chat].add(ident)
    claims, previous, counts = {}, defaultdict(lambda: deque(maxlen=3)), Counter()
    addresses = AddressCollector(identities, decisions or {}, messages, self_account)
    digest = hashlib.sha256()

    def add(row, line, owner, alias, target, evidence_kind, preference=''):
        if alias == owner:
            return
        ident = key([owner, alias, target, row['chat_id'], evidence_kind, preference])
        claim = claims.setdefault(ident, dict(id=ident, kind='person', owner=owner, alias=alias,
            scope=row.get('chat_name', ''), scope_chat_id=row['chat_id'], status='pending_review',
            flags=['self_stated_review_required' if evidence_kind == 'context_self_name'
                   else 'address_target_unconfirmed'], evidence=[], evidence_count=0,
            observed_identity_candidates=[target]))
        if evidence_kind == 'context_self_name':
            claim['bound_identity'] = target
        else:
            claim['suggested_identity'] = target
        if preference:
            claim['preference'] = preference
        if alias in GENERIC_NAMES:
            claim['flags'] = sorted(set(claim['flags'] + ['generic_address']))
        claim['evidence_count'] += 1
        if len(claim['evidence']) < 4:
            claim['evidence'].append(dict(kind=evidence_kind, path=str(messages),
                **excerpt(row, line), context_before=list(previous[row['chat_id']]),
                target_identity=target, preference=preference))

    with messages.open('rb') as handle:
        for line, data in enumerate(handle, 1):
            digest.update(data)
            row = json.loads(data)
            counts['messages_scanned'] += 1
            text, ident = authored_text(row), identity_of(row)
            if not text or not ident:
                continue
            if '智能助手' in text:
                counts['assistant_statements_skipped'] += 1
                previous[row['chat_id']].append(excerpt(row, line))
                continue
            for alias, preference in self_names(text):
                add(row, line, row.get('sender', ''), alias, ident, 'context_self_name', preference)
                addresses.naming_statement(row, line, alias, preference, previous[row['chat_id']])
            # A private vocative is a hypothesis, not proof of its addressee's identity.
            members = private_members.get(row['chat_id'], set())
            alias = vocative_name(text)
            if alias and len(members) == 2 and ident in members:
                target = next(iter(members - {ident}))
                names = identities[target]['names']
                owner = max(names, key=lambda n: names[n]['count'])
                add(row, line, owner, alias, target, 'private_vocative')
                addresses.add(row, line, alias, target, owner, 'direct_address_candidate',
                              'private_addressee_unconfirmed', previous[row['chat_id']])
            addresses.confirmed_mentions(row, line, text, alias, previous[row['chat_id']])
            previous[row['chat_id']].append(excerpt(row, line))
    expected = next((x['sha256'] for x in review['inputs'] if x['path'] == str(messages.resolve())), None)
    if not expected or expected != digest.hexdigest():
        raise ValueError('历史文件与原清单不一致；请先重建原始清单，不能混用身份索引')
    # Single occurrences remain discoverable evidence but are not put in the confirmation list.
    selected, weak = [], []
    for claim in claims.values():
        (weak if claim.get('suggested_identity') and claim['evidence_count'] < 2 else selected).append(claim)
    return dict(claims=selected, weak_candidates=weak, addresses=list(addresses.records.values()),
                perspective_account=addresses.self_account,
                counts=dict(counts))
