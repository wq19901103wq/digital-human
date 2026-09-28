"""Cached-only diagnosis of learned recall, ranking and full-example budgets.

This report measures selection mechanics, never generation quality. It uses the
same approved historical recall and feature requests as inference and refuses
to call a model or inspect held-out acceptance outcomes.
"""
from collections import Counter
from pathlib import Path
import time

from ..config import sha256_file
from ..iteration import datasets, learned_gen, pack_transport, versions
from ..iteration.storage import file_lock, read_json, write_json
from . import few_shot, history_sources, learned_selection
from .fewshot_ranker import extraction
from .history_sources import digest, require


def analyze_case(case, rows, selector, retriever, *, count, budgets):
    tasks, refs = selector.tasks(case, rows)
    reader = extraction
    if selector.model.get('feature_transform', {}).get('self_concern') is not None:
        from .fewshot_ranker import supplemental
        reader = supplemental
    values = {key: reader.cached(selector.cache, key, request) for key, request in tasks.items()}
    missing = [key for key, value in values.items() if value is None]
    result = dict(case_id=case['case_id'], chat_type=case['chat_type'],
        recalled=len(rows), missing_feature_tasks=len(missing), budgets={})
    if missing:
        result['status'] = 'missing_cached_features'
        return result
    from .fewshot_ranker.boosting import score_document
    features = learned_selection.feature_rows(case, rows, refs, values,
        transform=selector.model.get('feature_transform'))
    scores = score_document(selector.model, features) if rows else []
    result.update(status='complete', candidates=[])
    for index, (row, score, feature) in enumerate(zip(rows, scores, features), 1):
        block, ids = retriever.render_selected([row], max_chars=2**31-1)
        require(ids == [row['id']], 'Complete historical example could not be rendered')
        result['candidates'].append(dict(example_id=row['id'], recall_rank=index,
            score=float(score), full_chars=len(block),
            reply_features={key: value for key, value in feature.items()
                if key.startswith('example_reply.')},
            target_features={key: value for key, value in feature.items()
                if key.startswith('target.') and any(term in key for term in
                    ('intent', 'needs_', 'last_self_action'))}))
    for budget in budgets:
        decisions = []
        selected = learned_selection.choose(rows, scores, retriever,
            count=count, budget=budget, decisions=decisions)
        block, ids = retriever.render_selected(selected, max_chars=budget)
        result['budgets'][str(budget)] = dict(selected_ids=ids, selected_count=len(ids),
            rendered_chars=len(block), decisions=decisions)
    return result


def summarize(records, budgets):
    complete = [row for row in records if row['status'] == 'complete']
    result = dict(total=len(records), complete=len(complete),
        missing_cached_features=len(records)-len(complete), budgets={})
    baseline = str(budgets[0])
    for budget in budgets:
        key = str(budget)
        selections = [row['budgets'][key] for row in complete]
        result['budgets'][key] = dict(
            selected_count_distribution=dict(Counter(row['selected_count'] for row in selections)),
            cases_with_budget_rejections=sum(any(d['decision'] == 'budget' for d in row['decisions'])
                for row in selections),
            changed_from_first_budget=sum(row['budgets'][key]['selected_ids'] !=
                row['budgets'][baseline]['selected_ids'] for row in complete),
            total_rendered_chars=sum(row['rendered_chars'] for row in selections))
    return result


def report(output, *, spec_path, generator, budgets, case_ids=()):
    """Reconstruct development recall under a shared memory lane; cache only."""
    output, spec_path = Path(output).resolve(), Path(spec_path).resolve()
    require(len(budgets) >= 2 and len(set(budgets)) == len(budgets) and
            all(type(value) is int and value > 0 for value in budgets),
            'Provide at least two distinct positive character budgets')
    spec = read_json(spec_path)
    require(spec['dataset'] == 'development' and not spec.get('acceptance'),
            'Selection diagnosis is restricted to development cases')
    all_cases = datasets.rows_for(spec)
    requested = set(case_ids)
    cases = [case for case in all_cases if not requested or case['case_id'] in requested]
    require(not requested or requested == {case['case_id'] for case in cases},
            'Requested case does not belong to this development population')
    cfg = versions.load_generator(generator)
    selector = learned_selection.LearnedSelector(cfg['dir'] / 'ranker',
        versions.PRIVATE / '.cache/fewshot_ranker_features')
    switches = cfg['config'].get('retriever', {})
    require(switches.get('learned') == learned_selection.POLICY, 'Generator has no learned selector')
    count = int(cfg['config']['max_shots_per_case'])
    data_dir = versions.data_version_dir(spec['data_ref'])
    source_paths = [data_dir / name for name in
        ('messages.jsonl', 'purposes.json', 'fewshot_pool.jsonl', 'report.json')]
    stamps = [history_sources.stamp(path) for path in source_paths]
    binding = dict(generator=generator, model_sha256=sha256_file(cfg['dir'] / 'ranker/model.json'),
        spec_sha256=sha256_file(spec_path), case_ids=[case['case_id'] for case in cases],
        sources={str(path): sha256_file(path) for path in source_paths}, budgets=budgets, count=count)
    with file_lock(output / '.selection_analysis.lock', blocking=False):
        records = []
        with learned_gen.history_phase(output), pack_transport.history_feature_cache(few_shot):
            retriever = few_shot.PersonaFewShotRetriever(path=data_dir / 'fewshot_pool.jsonl')
            require(retriever.is_approved(), 'Historical recall pool is not approved')
            last_report = time.time()
            for case in cases:
                rows = learned_selection.recall(retriever, case,
                    source_overlap_policy=switches.get('source_overlap_policy'))
                records.append(analyze_case(case, rows, selector, retriever, count=count, budgets=budgets))
                if time.time()-last_report >= 15 or len(records) == len(cases):
                    require(stamps == [history_sources.stamp(path) for path in source_paths],
                            'Historical recall sources changed during diagnosis')
                    write_json(output / 'progress.json', dict(completed=len(records), total=len(cases)))
                    print(f'Selection diagnosis: {len(records)}/{len(cases)}', flush=True)
                    last_report = time.time()
            retriever = None
        value = dict(schema=1, kind='cached_development_selection_diagnosis', binding=binding,
            binding_sha256=digest(binding), summary=summarize(records, budgets), cases=records,
            scope='Selection mechanics only; no model requests, answer labels or generation-quality claims')
        write_json(output / 'selection_analysis.json', value)
        return value['summary']
