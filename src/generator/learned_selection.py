"""Frozen learned few-shot selection using independent, shared feature caches."""
from pathlib import Path

from ..iteration.storage import read_json
from ..judge.corrected import CodexJudgeClient
from .history import eligible
from .history_sources import digest, require
from .fewshot_ranker import extraction, crosses, identity_crosses
from .fewshot_ranker.features import combine, context_local, reply_local

POLICY = 'pairwise_xgb_v1'
SOURCE_OVERLAP_POLICY = 'exclude_reply_in_context_v1'


def validate_source_overlap_policy(policy):
    require(policy is None or policy == SOURCE_OVERLAP_POLICY,
            f'Unsupported learned source overlap policy: {policy!r}')


def recall(retriever, case, *, source_overlap_policy=None):
    """The training recall routes, without its eight-candidate sampling step."""
    validate_source_overlap_policy(source_overlap_policy)
    messages = case.get('context', [])
    options = dict(chat_name=str(case.get('chat_name', '')),
        is_group=case.get('chat_type') == 'group', limit=12,
        exclude_ids={str(case.get('case_id', ''))},
        current_context_messages=[{'sender': str(m.get('sender', '')), 'text': str(m.get('text', ''))}
                                  for m in messages], history_case=case)
    candidates = {}
    for size in (3, 1):
        query = '\n'.join(str(m.get('text', '')) for m in messages[-size:])
        for rank, row in enumerate(retriever.retrieve(query=query, **options), 1):
            require(eligible(row, case), 'Learned selector received an ineligible example')
            if row['source_span']['end_timestamp'] >= case['input_cutoff']['timestamp']:
                continue  # Frozen training features require strictly positive history age.
            key = str(row['id'])
            if key in candidates:
                require(candidates[key][1] == row, 'Conflicting recalled example identity')
                candidates[key] = (min(rank, candidates[key][0]), row)
            else:
                candidates[key] = (rank, row)
    rows = [item[1] for _, item in sorted(candidates.items(), key=lambda p: (p[1][0], p[0]))]
    if source_overlap_policy == SOURCE_OVERLAP_POLICY:
        # Remove the whole example only after both original routes are merged.
        # Its reply is already visible in the target context; text similarity is
        # irrelevant, and extra recall would change more than this policy.
        visible = set(case['context_message_ids'])
        rows = [row for row in rows if not visible.intersection(row['reply_message_ids'])]
    return rows


def feature_rows(case, rows, refs, values, *, transform=None):
    action_expander = None
    action_policy = (transform or {}).get('reply_action_crosses')
    if action_policy is not None:
        from .fewshot_ranker import action_crosses
        require(action_policy == action_crosses.VERSION, 'Unsupported reply action feature transform')
        action_expander = action_crosses.expand
    result = []
    for example in rows:
        keys = refs[(case['case_id'], example['id'])]
        tk, ck, rk = keys[:3]
        row = combine(case, example,
            {**values[tk], **context_local(case)},
            {**values[ck], **context_local(example, example=True)},
            {**values[rk], **reply_local(example)})
        row = crosses.expand(row)
        row['id_cross.chat_id'] = identity_crosses.pair(row, 'target.chat_id', 'example_context.chat_id')
        if action_expander is not None:
            row = action_expander(row)
        if (transform or {}).get('self_concern') is not None:
            from .fewshot_ranker import concern
            require(transform['self_concern'] == concern.VERSION,
                    'Unsupported self concern feature transform')
            require(len(keys) == 5, 'Self concern feature references are missing')
            row = concern.expand(row, values[keys[3]], values[keys[4]])
        result.append(row)
    return result


def choose(rows, scores, retriever, *, count, budget, decisions=None):
    """Select complete examples; optionally record the exact rejection reasons."""
    require(len(rows) == len(scores), 'Ranker score count differs from recall')
    selected, content_seen, source_seen = [], set(), set()
    for row, score in sorted(zip(rows, scores), key=lambda p: (-float(p[1]), str(p[0]['id']))):
        require(float('-inf') < float(score) < float('inf'), 'Nonfinite ranker score')
        content = digest(dict(context=[{'sender': m['sender'], 'text': m['text']}
            for m in row['context_messages']], reply=row['reply']))
        source = digest([row['context_message_ids'], row['reply_message_ids']])
        detail = dict(example_id=row['id'], score=float(score))
        if decisions is not None:
            decisions.append(detail)
        if content in content_seen or source in source_seen:
            detail['decision'] = 'duplicate'
            continue
        if len(selected) >= count:
            detail['decision'] = 'count_limit'
            continue
        proposed = [*selected, row]
        block, ids = retriever.render_selected(proposed, max_chars=budget)
        if ids != [x['id'] for x in proposed] or len(block) > budget:
            detail['decision'] = 'budget'
            continue
        detail.update(decision='selected', combined_chars=len(block))
        selected.append(row)
        content_seen.add(content)
        source_seen.add(source)
    return selected


class LearnedSelector:
    def __init__(self, directory, feature_cache):
        self.directory = Path(directory)
        self.proof = read_json(self.directory / 'provenance.json')
        self.model = read_json(self.directory / 'model.json')
        transport = self.proof['feature_config']
        self.client = CodexJudgeClient(transport.get('config', transport))
        require(self.client.cache_identity() == self.proof['feature_identity'],
                'Learned selector feature client differs from training')
        self.cache = Path(feature_cache)

    def tasks(self, case, rows):
        groups = [dict(target_id=case['case_id'], target=case,
            candidates=[dict(example=row) for row in rows])]
        tasks, refs = extraction.prepare(groups, self.proof['feature_identity'])
        policy = self.model.get('feature_transform', {}).get('self_concern')
        if policy is not None:
            from .fewshot_ranker import concern
            require(policy == concern.VERSION, 'Unsupported self concern feature transform')
            extra, extra_refs = concern.prepare(groups, self.proof['feature_identity'])
            require(not tasks.keys() & extra.keys(), 'Feature request key collision')
            tasks.update(extra)
            refs = {key: (*value, *extra_refs[key]) for key, value in refs.items()}
        return tasks, refs

    def select(self, case, rows, *, count, budget, retriever, check):
        if not rows or count <= 0:
            return []
        from .fewshot_ranker.boosting import score_document
        check()
        tasks, refs = self.tasks(case, rows)
        extractor = extraction
        if self.model.get('feature_transform', {}).get('self_concern') is not None:
            from .fewshot_ranker import supplemental
            extractor = supplemental
        values = {key: extractor.extract_one(self.cache, key, task, self.client)
                  for key, task in tasks.items()}
        scores = score_document(self.model, feature_rows(case, rows, refs, values,
            transform=self.model.get('feature_transform')))
        check()
        return choose(rows, scores, retriever, count=count, budget=budget)
