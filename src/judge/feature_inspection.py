"""Reusable saved-case feature diagnosis; never creates evaluation measurements."""
from __future__ import annotations

import json
from pathlib import Path

from .. import cache, tracing
from ..iteration import shares, versions
from ..iteration.storage import read_json, write_json
from . import background_features as features
from .corrected import CodexJudgeClient


def saved_case(experiment: Path, case_id: str, branch: str, round_index: int) -> dict:
    successful = None
    with (experiment / 'cases.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('case_id') == case_id and row.get('status') == 'ok':
                successful = row
    if successful is None:
        raise ValueError(f'没有成功的已有结果: {case_id}')
    trace = tracing.read(experiment, successful['trace_ref'])
    operations = [op for op in trace['operations'] if op['kind'] == 'judge' and
                  op['branch'] == branch and op['round'] == round_index and op['status'] == 'ok']
    if not operations:
        raise ValueError('已有结果没有所选 Judge 轮次')
    events = operations[-1]['events']
    mapping = next(e['data'] for e in reversed(events) if e['kind'] == 'blind_mapping')
    old = next((e['data']['parsed'] for e in reversed(events)
                if e['kind'] == 'features' and e['data'].get('parsed')), None)
    return {'case': trace['case'], 'mapping': mapping, 'old_features': old,
            'saved_decisions': {e['kind']: e['data']['decision'] for e in events
                                if isinstance(e.get('data'), dict) and 'decision' in e['data']},
            'versions': trace['versions'], 'source_trace': trace['trace_ref']}


def inspect_contribution(experiment: Path, case_id: str, judge_ref: str, output: Path,
                         *, branch='candidate', round_index=0, client=None):
    """Diagnose response needs on an existing blinded pair, without regeneration."""
    from . import contribution
    saved = saved_case(experiment, case_id, branch, round_index)
    config = versions.judge_dir(judge_ref)['config']
    client = client or CodexJudgeClient({**config['llm'], **config.get('feature_llm', {}),
                                        'reasoning_effort': 'medium'})
    payload = contribution.build_input(saved['mapping']['blind_case'])
    spec = {**saved['versions'], 'kind': 'feature_diagnostic', 'dataset': 'diagnostic',
            'judge_ref': judge_ref}
    trace = tracing.CaseTrace(output, saved['case'], spec)
    report = dict(schema=contribution.VERSION, diagnostic_only=True,
        formal_accuracy_evidence=False, case_id=case_id, source_experiment=str(experiment),
        source_trace=saved['source_trace'], judge_ref=judge_ref, client=client.cache_identity(),
        trace_ref=trace.ref, input=payload,
        option_origins={k: saved['mapping'][k] for k in ('human_option', 'candidate_option')})
    try:
        with trace.operation('feature_diagnostic', 'contribution', round_index, payload) as op:
            parsed = contribution.extract(client, payload)
            report['parsed'] = parsed
            prior = saved['saved_decisions'].get('ownership_features')
            if prior:
                _, report['decision'] = contribution.decide(prior['probability_a'], parsed, payload)
            op['result'] = parsed
        trace.finish('ok')
        report['status'] = 'ok'
    except BaseException:
        trace.finish('failed')
        report['status'] = 'incomplete'
        write_json(output / 'report.json', report)
        raise
    write_json(output / 'report.json', report)
    return report


def inspect(experiment: Path, case_id: str, share_ref: str, judge_ref: str,
            output: Path, *, branch: str = 'candidate', round_index: int = 0,
            client=None, conditions=None, memory_index=None, memory_model=None,
            reasoning_effort=None, extraction_mode='joint') -> dict:
    """Cached diagnostics with context or directly selected Wiki pages.

Neither these calls nor this report may be used as a formal adoption result.
"""
    if reasoning_effort not in (None, 'low', 'medium', 'high'):
        raise ValueError('不支持的诊断思考深度')
    if extraction_mode not in ('joint', 'separate_context'):
        raise ValueError('未知特征抽取方式')
    if reasoning_effort is not None and client is not None:
        raise ValueError('自定义客户端与思考深度覆盖不能同时使用')
    saved = saved_case(experiment, case_id, branch, round_index)
    conditions = conditions or ['context_only', 'with_wiki']
    if set(conditions) - {'context_only', 'with_background', 'with_wiki', 'with_memory'}:
        raise ValueError('未知诊断条件')
    share = shares.load(share_ref)
    knowledge = read_json(share['dir'] / 'content' / 'knowledge.json')
    projection = features.diagnostic_projection(knowledge, saved['case'])
    wiki = None
    if 'with_wiki' in conditions:
        from .wiki_lookup import context_accounts, diagnostic_projection
        accounts = context_accounts(versions.data_version_dir(saved['versions']['data_ref']), saved['case'])
        wiki = diagnostic_projection(knowledge, saved['case'], saved['mapping']['blind_case'], accounts)
    config = versions.judge_dir(judge_ref)['config']
    execution = {**config['llm'], **config.get('feature_llm', {})}
    if reasoning_effort is not None:
        execution['reasoning_effort'] = reasoning_effort
    client = client or CodexJudgeClient(execution)
    spec = {**saved['versions'], 'kind': 'feature_diagnostic',
            'dataset': 'diagnostic', 'judge_ref': judge_ref}
    trace = tracing.CaseTrace(output, saved['case'], spec)
    report = {'schema': features.VERSION, 'diagnostic_only': True,
              'extraction_mode': extraction_mode,
              'formal_accuracy_evidence': False, 'case_id': case_id,
              'source_experiment': str(experiment), 'source_trace': saved['source_trace'],
              'judge_ref': judge_ref, 'client': client.cache_identity(),
              'diagnostic_overrides': ({'reasoning_effort': reasoning_effort}
                                       if reasoning_effort is not None else {}),
              'share_ref': share_ref, 'share_sha256': share['manifest']['sha256'],
              'projection': projection, 'old_features': saved['old_features'],
              'option_origins': {key: saved['mapping'][key]
                                 for key in ('human_option', 'candidate_option')},
              'trace_ref': trace.ref, 'conditions': {}}
    if wiki is not None:
        report['wiki_projection'] = wiki
    try:
        for condition in dict.fromkeys(conditions):
            background = ({'identities': [], 'facts': []} if condition == 'context_only'
                          else wiki['payload'] if condition == 'with_wiki' else projection['payload'])
            payload = features.build_input(saved['mapping']['blind_case'], background)
            if condition == 'with_memory':
                from .memory_search import SearchMemory, enrich
                from .memory_dense import configured
                memory = SearchMemory(versions.data_version_dir(saved['versions']['data_ref']),
                                      saved['case'], background,
                                      backend=configured(memory_index, memory_model))
                with trace.operation('search_memory', condition, 0, memory.identity()) as operation:
                    payload = enrich(client, payload, memory)
                    operation['result'] = payload['search_memory']
                report['memory'] = {'binding': memory.identity(), 'calls': payload['search_memory']}
            attribution = None
            if extraction_mode == 'separate_context':
                from . import context_attribution
                report['extraction_protocol'] = context_attribution.VERSION
                isolated = context_attribution.build_input(payload)
                with trace.operation('context_attribution', condition, 0, isolated) as operation:
                    attribution = context_attribution.extract(client, payload)
                    operation['result'] = attribution
                report.setdefault('context_extractions', {})[condition] = {
                    'input_sha256': cache.digest(isolated), **attribution}
                write_json(output / 'report.json', report)
            feature_input = ({**payload, **attribution} if attribution is not None else payload)
            with trace.operation('background_features', condition, 0, feature_input) as operation:
                result = (context_attribution.extract_replies(client, payload, attribution)
                          if attribution is not None else features.extract(client, payload))
                operation['result'] = result
                report['conditions'][condition] = {
                    'input_sha256': cache.digest(payload), 'features': result,
                    'categorical': features.categorical_features(result)}
                tracing.note('background_features', {'parsed': result})
            # Progress is durable if a later request fails; successes are cached.
            write_json(output / 'report.json', report)
        trace.finish('ok')
        report['status'] = 'ok'
    except BaseException:
        trace.finish('failed')
        report['status'] = 'incomplete'
        write_json(output / 'report.json', report)
        raise
    write_json(output / 'report.json', report)
    return report
