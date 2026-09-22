"""History-training audit tools. Extracted from scripts/legacy/verify_history_training.py."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from .evidence import SavedCache, digest, require
from ...config import load_settings
from ...generator.history import eligible
from ...generator.persona import PersonaPromptBuilder
from ...iteration import versions
from ...judge import corrected_v1 as rt
from ...judge.corrected import blind_case
from ...llm import build_clients


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


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
    builder = PersonaPromptBuilder(settings, gen['dir'])
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
        prompt = builder.build_messages(case, refs[0]['rendered'], True)
        without_answer = {k: v for k, v in case.items() if k != 'human_reply'}
        require(prompt == builder.build_messages(without_answer, refs[0]['rendered'], True),
                'target answer affects generation prompt')
        cached = [e for e in evidence.entries(op, cid) if e['layer'] == 'generation']
        require(len(cached) == 1, 'generation request provenance missing')
        request = cached[0]['identity']
        require(request['messages'] == prompt and request['client'] == clients[case['chat_type']].cache_identity()
                and request['forced_reply'] is True, 'actual generation request differs from frozen prompt or client')
        require(cached[0]['value']['replies'] == entry['replies'] == op['result']['replies'], 'generation trace/cache differs')
        valid[cid] = entry
        evidence.counts[spec['dataset'] + '_generations_checked'] += 1
    if complete:
        require(len(valid) == len(cases), 'generation is incomplete')
    return valid


def judge_operation(op, case, info, evidence):
    cid = case['case_id']
    require(op['status'] == 'ok' and op['kind'] == 'judge' and op['input']['candidate_replies'] == case['ai_replies'],
            'judge input or operation mismatch')
    cached = [e for e in evidence.entries(op, cid) if e['layer'] == 'judge']
    require(len(cached) == 1 and cached[0]['identity']['config'] == info['config'], 'judge configuration evidence differs')
    entry = cached[0]
    require(entry['identity']['candidate_replies'] == case['ai_replies'] and entry['identity']['case'] ==
            {k: case.get(k) for k in ('context', 'human_reply', 'chat_type', 'source_chat_id', 'chat_name')},
            'judge cache contains another case')
    # Follow the saved producer when a resumed operation is a whole-Judge cache hit.
    if not any(e['kind'] == 'blind_mapping' for e in op['events']):
        origin = entry['origin']
        source = Path(origin['experiment_path']) / 'traces' / (origin['trace_ref'] + '.json')
        trace = evidence.document(source)
        op = trace['operations'][origin['operation_index']]
        require(trace['case'] == case and op['round'] == origin['round'], 'cached Judge provenance changed')
    mapping = [e['data'] for e in op['events'] if e['kind'] == 'blind_mapping']
    features = [e['data']['parsed'] for e in op['events'] if e['kind'] == 'features' and e['data'].get('valid')]
    base = [e['data'] for e in op['events'] if e['kind'] == 'base_verdict']
    require(len(mapping) == len(base) == 1 and features, 'Judge evidence missing')
    mapping, features, base = mapping[0], features[-1], base[0]
    a, b = ((case['human_reply'], case['ai_replies']) if mapping['human_option'] == 'A'
            else (case['ai_replies'], case['human_reply']))
    blind, meta = blind_case(case, a, b)
    require(mapping['blind_case'] == blind and digest(mapping['context_metadata']) == digest(meta), 'blind mapping differs')
    prompt = rt.feature_extractor_prompt(blind)
    feature_op = {**op, 'events': [e for e in op['events'] if e['kind'] != 'codex'
                                  or e['request']['prompt'] == prompt]}
    config = {**info['config']['llm'], **info['config'].get('feature_llm', {})}
    evidence.feature_request(feature_op, cid, blind, features, config)
    model = rt.load_formal_judge(info['dir'] / 'correction.json')
    prob = rt.score_formal_judge_pair(model, features, a, b, meta)
    small = 'A' if prob >= .5 else 'B'
    corrected = small != base['human_option'] and max(prob, 1-prob) >= info['config']['correction_threshold']
    final = small if corrected else base['human_option']
    verdict = entry['value']['last_verdict']
    require(abs(prob - verdict['small_model_probability_a']) <= 1e-12 and
            verdict['human_option'] == final and verdict['correction_applied'] == corrected,
            'saved Judge verdict differs from frozen LR and threshold')
    hit = final == mapping['human_option']
    require(entry['value']['identified_ai'] == hit and op['result']['identified_ai'] == hit, 'Judge vote differs')
    return hit
