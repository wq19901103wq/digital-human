"""Independent, instance-neutral reconstruction of history LR training evidence.

The recipe owns the algorithm; instance manifests own examples, cutoffs, models,
arm names and hyperparameters. No experiment driver is imported by the guard.
"""
from __future__ import annotations

import hashlib
import html
import json
import random
import re
import sqlite3
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .. import cache
from ..cache import digest
from ..config import ConfigError, load_settings, sha256_file
from ..generator.history import eligible
from ..generator.few_shot import PersonaFewShotRetriever
from ..generator.history_sources import load as load_history
from ..generator.persona import PersonaPromptBuilder
from ..judge import corrected_v1 as rt
from ..judge.corrected import blind_case
from ..llm import build_clients
from . import datasets, versions


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ConfigError(message)


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

class SavedCache:
    def __init__(self, path=None):
        path = Path(path) if path is not None else versions.PRIVATE / '.cache/results.sqlite3'
        self.db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row

    def get(self, key):
        row = self.db.execute('SELECT * FROM entries WHERE key=?', (key,)).fetchone()
        if row is None:
            return None
        require(hashlib.sha256(row['value'].encode()).hexdigest() == row['value_sha'], 'cache value hash')
        return {**dict(row), **{k: json.loads(row[k]) for k in ('identity', 'value', 'origin')}}
class Evidence:
    def __init__(self):
        self.store = SavedCache()
        self.counts = Counter()
        self.hashes = {}

    def document(self, path):
        blob = path.read_bytes()
        self.hashes[str(path.resolve())] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    def cache_entry(self, event, cid, round_index):
        entry = self.store.get(event['data']['key'])
        # A failed request need not have committed a cache entry.
        if entry is None and event['kind'] == 'cache_lookup':
            return None
        require(entry is not None, f'{cid}: missing successful cache entry')
        origin = entry['origin']
        require((str(origin['case_id']), origin['round']) == (str(cid), round_index),
                f'{cid}: cache reused across cases or independent rounds')
        require(entry['layer'] == event['data']['layer'], f'{cid}: cache layer mismatch')
        partition_key = ('development' if origin['dataset'] == 'development'
                         else f"{origin['dataset']}:{origin['experiment_id']}")
        expected = digest({'schema': 1, 'layer': entry['layer'], 'identity': entry['identity'],
                           'partition': partition_key, 'case_id': origin['case_id'], 'round': round_index,
                           'sample_epoch': origin['sample_epoch'], 'epoch': origin['epoch']})
        require(expected == entry['key'], f'{cid}: cache content identity mismatch')
        self.counts['cache_entries_checked'] += 1
        return entry

    def entries(self, op, cid):
        return [entry for e in op.get('events', []) if e['kind'] in ('cache_hit', 'cache_lookup')
                if (entry := self.cache_entry(e, cid, op['round'])) is not None]

    def feature_request(self, op, cid, blind, features, config):
        prompt, schema = rt.feature_extractor_prompt(blind), rt.feature_response_json_schema()
        verified = 0
        for event in op.get('events', []):
            if event['kind'] != 'codex':
                continue
            request = event['request']
            require(request == {'prompt': prompt, 'schema': schema, 'config': config},
                    f'{cid}: actual feature prompt/config/schema differs')
            if event['status'] == 'ok':
                try:
                    parsed = rt.parse_feature_response(event['response']['text'])
                except ValueError:
                    continue
                if parsed == features:
                    verified += 1
        for entry in self.entries(op, cid):
            identity = entry['identity']
            if entry['layer'] != 'llm_request' or identity.get('prompt') != prompt:
                continue
            require(identity['schema'] == schema, f'{cid}: cached feature schema differs')
            require(all(identity['client'][k] == config[k] for k in
                        ('provider', 'model', 'reasoning_effort', 'codex_cli_version')),
                    f'{cid}: cached feature configuration differs')
            require(rt.parse_feature_response(entry['value']) == features, f'{cid}: cached feature value differs')
            verified += 1
        require(verified > 0, f'{cid}: no actual or cached feature request evidence')
        self.counts['feature_requests_verified'] += 1
@contextmanager
def generation_cache(evidence, directory):
    # Nested development generation has its own producer cache location.
    previous = evidence.store
    current = SavedCache(directory.parent.parent / '.cache/results.sqlite3')
    evidence.store = current
    try:
        yield
    finally:
        evidence.store = previous
        current.db.close()

def generations(directory, spec, cases, pool, evidence, complete):
    saved = evidence.document(directory / 'generations.json')
    require(saved['identity'] == digest(spec), 'generation specification changed')
    by_id = {c['case_id']: c for c in cases}
    require(saved['entries'].keys() <= by_id.keys(), 'unselected generation case')
    gen = versions.load_generator(spec['generator_ref'])
    settings = load_settings()
    builder = PersonaPromptBuilder(settings, gen['dir'],
                                   context_identity=gen['config'].get('context_identity'))
    clients = build_clients(settings, gen['config']['llm'])
    if not isinstance(clients, dict):
        clients = {'group': clients, 'private': clients}
    valid = {}
    for cid, entry in saved['entries'].items():
        case = by_id[cid]
        require(entry['input_sha256'] == digest(case), 'generation source changed')
        if entry['status'] != 'ok':
            continue
        require(entry['output_sha256'] == digest(entry['replies']), 'generation response changed')
        trace = evidence.document(directory / 'traces' / (entry['trace_ref'] + '.json'))
        require(trace['status'] == 'ok' and trace['case'] == case and len(trace['operations']) == 1,
                'generation trace mismatch')
        op = trace['operations'][0]
        require((op['kind'], op['round'], op['status']) == ('generation', 0, 'ok'), 'generation operation mismatch')
        refs = [e['data'] for e in op['events'] if e['kind'] == 'retrieval']
        require(len(refs) == 1 and 'error' not in refs[0], 'retrieval was bypassed')
        for key in refs[0]['selected_ids']:
            require(key in pool and eligible(pool[key], case), 'future, target or held-out history was retrieved')
            evidence.counts['visible_references_checked'] += 1
        selected = [pool[key] for key in refs[0]['selected_ids']]
        history = load_history(versions.data_version_dir(spec['data_ref']))
        for example in selected:
            history.validate(example, example=True)
        budget = gen['config'].get('shots_char_budget', settings['evaluation']['few_shots_char_budget'])
        rendered, ids = PersonaFewShotRetriever.render(selected, max_chars=budget)
        require(rendered == refs[0]['rendered'] and ids == refs[0]['selected_ids'],
                'rendered few-shot text does not match its source examples')
        prompt = builder.build_messages(case, rendered, True)
        without_answer = {k: v for k, v in case.items() if k != 'human_reply'}
        require(prompt == builder.build_messages(without_answer, refs[0]['rendered'], True),
                'target answer affects generation prompt')
        cached = [e for e in evidence.entries(op, cid) if e['layer'] == 'generation']
        require(len(cached) == 1, 'generation request provenance missing')
        request = cached[0]['identity']
        expected_client = clients[case['chat_type']].cache_identity()
        if 'adapter' in expected_client:
            expected_client = {**expected_client, 'adapter': digest([input_code_hash(spec, 'llm.py')])}
        require(request['messages'] == prompt and request['client'] == expected_client
                and request['forced_reply'] is True, 'actual generation request differs from frozen prompt or client')
        require(cached[0]['value']['replies'] == entry['replies'] == op['result']['replies'], 'generation trace/cache differs')
        valid[cid] = entry
        evidence.counts[spec['dataset'] + '_generations_checked'] += 1
    if complete:
        require(len(valid) == len(cases), 'generation is incomplete')
    return valid
def norm(text):
    return re.sub(r'\s+', '', html.unescape(str(text or '')))

def source_text_matches(expected, actual):
    if norm(expected) == norm(actual):
        return True
    # The inherited pool replaces these PII fields; preserve that documented
    # transformation when checking source text. Timestamp still must match.
    for pattern, replacement in [
        (r'(?<!\d)1[3-9]\d{9}(?!\d)', '[手机号]'),
        (r'(?i)https?://\S+|www\.\S+', '[链接]'),
        (r'(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', '[邮箱]'),
        (r'(?i)wxid_[a-z0-9_]+', '[微信账号]'),
    ]:
        actual = re.sub(pattern, replacement, actual)
    pattern = '.{2,30}?'.join(re.escape(norm(s)) for s in expected.split('[联系人]'))
    return re.fullmatch(pattern, norm(actual)) is not None
def sources(directory, spec, evidence):
    from .runtime import verify_inputs
    verify_inputs(spec['inputs'])
    data = versions.data_version_dir(spec['data_ref'])
    require(datasets.snapshot(data) == spec['purpose_snapshot'], 'purpose manifest changed')
    selected = evidence.document(directory / 'sources.json')
    train, dev = selected['train'], selected['development']
    require(len(train) == spec['training_total'] and len(dev) == spec['evaluation_total'], 'formal sample counts changed')
    require({r['case_id']: r for r in train} == {r['case_id']: r for r in lines(data / 'judge_training.jsonl')},
            'training cases differ from the original purpose')
    require(dev == lines(data / 'judge_dev_pool.jsonl'), 'development cases/order changed')
    purpose = evidence.document(data / 'purposes.json')['protocol']
    cutoff, heldout = purpose['development_start'], set(purpose['unseen_chat_ids'])
    source = load_history(data)
    require({c['case_id']: c for c in source.role('judge_training')} == {c['case_id']: c for c in train}
            and len({c['case_id'] for c in train}) == len(train), 'training source mismatch')
    require(source.role('judge_development') == dev, 'development source/order mismatch')
    history = {chat: [m._asdict() for m in rows] for chat, rows in source.chats.items()}
    learned, tested = set(), set()
    for role, cases, ids in [('training', train, learned), ('development', dev, tested)]:
        for case in cases:
            span = case['source_span']
            start, middle, end = [span[k] for k in ('start', 'reply_start', 'end')]
            rows = history[span['chat_id']]
            require(0 <= start < middle < end <= len(rows), 'invalid source offsets')
            require(case['context_message_ids'] == [r['message_id'] for r in rows[start:middle]]
                    and case['reply_message_ids'] == [r['message_id'] for r in rows[middle:end]], 'message binding differs')
            require(len(case['human_reply']) == end-middle and all(r['is_self'] for r in rows[middle:end]),
                    'human bubble/source mismatch')
            require(all(source_text_matches(t, r['text']) for t, r in zip(case['human_reply'], rows[middle:end])),
                    'target text differs from source')
            require(len(case['context']) == middle-start and all(
                source_text_matches(c['text'], r['text']) and all(c[k] == r[k] for k in ('sender', 'is_self', 'timestamp'))
                for c, r in zip(case['context'], rows[start:middle])), 'context provenance differs')
            if role == 'training':
                require(rows[end-1]['timestamp'] < cutoff and span['chat_id'] not in heldout, 'future/heldout training')
            else:
                require(rows[start]['timestamp'] >= cutoff, 'development before learning cutoff')
            ids.update(r['message_id'] for r in rows[start:end])
            evidence.counts[role + '_source_spans'] += 1
    require(not learned & tested, 'learning/evaluation full-message overlap')
    audit = evidence.document(directory / 'source_audit.json')
    reference = evidence.document(directory / 'reference.json')
    require(set(reference) == {'facts', 'examples'} and not reference['facts']
            and len(reference['examples']) == len(audit['reference_spans']),
            'unverified static material retained')
    for example, span in zip(reference['examples'], audit['reference_spans']):
        rows = history[span['chat_id']][span['start']:span['end']]
        require(set(example) == {'relationship', 'context', 'human_reply'}
                and rows and example['relationship'] == rows[0]['chat_type'],
                'unverified reference fields or relationship')
        texts = example['context'] + example['human_reply']
        require(len(rows) == len(texts) and all(source_text_matches(t, r['text']) for t, r in zip(texts, rows)),
                'reference source mismatch')
        require(span['message_ids'] == [r['message_id'] for r in rows] and
                rows[-1]['timestamp'] < cutoff and span['chat_id'] not in heldout and
                not {r['message_id'] for r in rows} & tested, 'reference future/heldout/answer overlap')
        evidence.counts['static_reference_spans'] += 1
    names, members = Counter(), Counter()
    for case in train:
        _, meta = blind_case(case, [], [])
        if meta['relationship'] == 'group':
            names[str(meta['group_name'])] += 1
            members.update((str(meta['source_chat_id']), str(s)) for s in meta['recent_speakers'][:3] if s != '__self__')
    template = evidence.document(directory / 'model_template.json')
    expected = {'known_group_names': sorted(n for n, c in names.items() if c >= 5),
                'known_group_members': [list(m) for m, c in sorted(members.items()) if c >= 8]}
    require(template['feature_system']['context_schema'] == expected, 'category vocabulary leaks outside training')
    require(not np.any(template['final_model']['coefficients']), 'old LR weights retained')
    from .learning_guard import require_materials
    require_materials(spec['data_ref'], [versions.generator_dir(spec['generator_ref'])])
    return selected

def vector(model, features, blind, metadata):
    a, b, names = rt.vectorize_boolean_options(features)
    options = []
    for option, base in [('option_A', a), ('option_B', b)]:
        observed, observed_names = rt.vectorize_observable_option_booleans(blind[option], metadata)
        group, group_names = rt.vectorize_group_pattern_option_boolean(blind[option], metadata)
        ctx, ctx_names = rt.vectorize_context_booleans(metadata, model.context_schema)
        value, expanded_names = rt.expand_boolean_option_with_refined_context(
            np.concatenate((base, observed, group)), [*names, *observed_names, *group_names], ctx, ctx_names)
        require(tuple(expanded_names) == model.feature_names, 'feature schema/order differs')
        options.append(value)
    result = options[0] - options[1]
    require(result.shape == (len(model.feature_names),) and np.isfinite(result).all(), 'invalid feature vector')
    return result

def training(directory, spec, train, generated, arm, evidence, complete):
    target = directory / arm
    labels = ['A'] * (len(train)//2) + ['B'] * (len(train)-len(train)//2)
    random.Random(42).shuffle(labels)
    rows = []
    for case, human in zip(train, labels):
        cid = case['case_id']
        if cid not in generated:
            continue
        a, b = ((case['human_reply'], generated[cid]['replies']) if human == 'A' else
                (generated[cid]['replies'], case['human_reply']))
        blind, meta = blind_case(case, a, b)
        rows.append({'case_id': cid, 'split': 'train', 'human_option': human, 'source_case_id': cid,
            'source_chat_id': case['source_chat_id'], 'source_message_id': case['source_message_id'],
            'metadata': meta, 'blind': blind})
    by_id = {r['case_id']: r for r in rows}
    checkpoint = evidence.document(target / ('features.json' if complete else 'preflight.json'))
    if complete:
        require(digest(evidence.document(target / 'training.json')['rows']) == digest(rows) and len(rows) == spec['training_total'],
                'training labels/order differ')
        fspec = evidence.document(target / 'spec.json')
        require(fspec == {**spec, 'id': directory.name + '-' + arm, 'feature_config': spec['feature_configs'][arm],
                          'training_sha256': digest(rows)}, 'feature run spec differs')
        require(checkpoint['identity'] == digest({'training': digest(rows), 'config': spec['feature_configs'][arm],
            'runtime': input_code_hash(spec, 'judge/corrected_v1.py'),
            'extractor': input_code_hash(spec, 'judge/lr_retrain.py')}), 'feature checkpoint identity differs')
        require(set(checkpoint['entries']) == set(by_id), 'missing or extra training features')
    else:
        require(checkpoint['identity'] == digest(spec) and set(checkpoint['entries']) == {r['case_id'] for r in train[:2]},
                'preflight conditions changed')
    model = rt.load_formal_judge(directory / 'model_template.json')
    vectors = {}
    with generation_cache(evidence, target):
        for cid, entry in checkpoint['entries'].items():
            row = by_id[cid]
            require(entry['status'] == 'ok' and entry['input_sha256'] == digest(row)
                    and entry['features_sha256'] == digest(entry['features']), 'feature content changed')
            trace = evidence.document(target / 'traces' / (entry['trace_ref'] + '.json'))
            require(trace['status'] == 'ok' and trace['case']['blind'] == row['blind'] and len(trace['operations']) == 1,
                    'feature trace mismatch')
            operation = trace['operations'][0]
            require((operation['kind'], operation['branch'], operation['round'], operation['status']) ==
                    ('feature_extraction', 'training', 0, 'ok'), 'feature request role/round mismatch')
            evidence.feature_request(operation, cid, row['blind'], entry['features'], spec['feature_configs'][arm])
            vectors[cid] = vector(model, entry['features'], row['blind'], row['metadata'])
            evidence.counts[arm + '_training_features'] += 1
    if not complete:
        return
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        fitted = LogisticRegression(**spec['recipe']).fit(csr_matrix(np.stack([vectors[r['case_id']] for r in rows])),
                                                         np.array([r['human_option'] == 'A' for r in rows]))
    artifact = evidence.document(target / 'correction.json')
    delta = float(np.max(np.abs(fitted.coef_[0] - artifact['final_model']['coefficients'])))
    require(delta < 1e-8 and artifact['training']['features_sha256'] == sha256_file(target / 'features.json')
            and artifact['training']['evaluation_used_for_fit'] is False, 'refit cannot reproduce saved weights')
    ref = evidence.document(target / 'candidate.json')['judge_ref']
    info = versions.judge_dir(ref)
    for name in ('prompt.md', 'profile.json', 'reference.json'):
        require(sha256_file(info['dir'] / name) == sha256_file(directory / name), 'candidate static assets changed')
    require(sha256_file(info['dir'] / 'correction.json') == sha256_file(target / 'correction.json'), 'candidate weights changed')
    require(info['config']['llm'] == spec['scoring']['initial'] and info['config']['feature_llm'] == spec['feature_configs'][arm]
            and info['config']['correction_threshold'] == spec['scoring']['correction_threshold']
            and info['config'].get('decision_policy', 'hybrid') == spec['scoring']['decision_policy'],
            'hybrid configuration differs')
    return {'judge_ref': ref, 'training_cases': len(rows), 'dimensions': len(model.feature_names), 'coefficient_max_difference': delta}
def training_rows(cases, generated):
    rows = []
    labels = ['A'] * (len(cases)//2) + ['B'] * (len(cases)-len(cases)//2)
    random.Random(42).shuffle(labels)
    for case, human in zip(cases, labels):
        entry = generated['entries'].get(case['case_id'], {})
        if entry.get('status') != 'ok':
            continue
        a, b = (case['human_reply'], entry['replies']) if human == 'A' else (entry['replies'], case['human_reply'])
        blind, metadata = blind_case(case, a, b)
        rows.append({'case_id': case['case_id'], 'split': 'train', 'human_option': human,
            'source_case_id': case['case_id'], 'source_chat_id': case['source_chat_id'],
            'source_message_id': case['source_message_id'], 'metadata': metadata, 'blind': blind})
    return rows
def checkpoint(directory, spec, rows):
    path = directory / 'features.json'
    expected = cache.digest({'training': spec['training_sha256'], 'config': spec['feature_config'],
                             'runtime': input_code_hash(spec, 'judge/corrected_v1.py'),
                             'extractor': input_code_hash(spec, 'judge/lr_retrain.py')})
    value = read(path) if path.exists() else {'identity': expected, 'entries': {}}
    if value['identity'] != expected:
        raise ConfigError('特征检查点配置指纹不一致')
    by_id = {r['case_id']: r for r in rows}
    for key, entry in value['entries'].items():
        if key not in by_id or entry['input_sha256'] != cache.digest(by_id[key]):
            raise ConfigError('特征检查点样本内容或标签已变化')
        if entry['status'] == 'ok':
            rt.validate_option_features(entry['features']['option_A'])
            rt.validate_option_features(entry['features']['option_B'])
            if entry['features_sha256'] != cache.digest(entry['features']):
                raise ConfigError('特征检查点产出被修改')
    return value


def input_code_hash(spec, relative):
    # 合并前的冻结任务记录旧框架路径（…/src/digital_human/…），新旧后缀都要认。
    aliases = {relative}
    if relative == 'judge/corrected_v1.py':
        aliases.add('judge/rpa_v1.py')
    matches = [value for name, value in spec['inputs'].items()
               if any(name.endswith(prefix + alias) for prefix in ('/src/', '/src/digital_human/')
                      for alias in aliases)]
    if len(matches) != 1:
        raise ConfigError('missing or ambiguous frozen code provenance')
    return matches[0]


def verified_input_paths(spec):
    from .runtime import verify_inputs
    return verify_inputs(spec['inputs'])
