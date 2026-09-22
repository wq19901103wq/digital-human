"""Frozen learned few-shot selection using independent, shared feature caches."""
from pathlib import Path

from ..iteration.storage import read_json
from ..judge.corrected import CodexJudgeClient
from .history import eligible
from .history_sources import digest, require
from .fewshot_ranker import extraction, crosses, identity_crosses
from .fewshot_ranker.features import combine, context_local, reply_local

POLICY = 'pairwise_xgb_v1'


def recall(retriever, case):
    """The training recall routes, without its eight-candidate sampling step."""
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
    return [item[1] for _, item in sorted(candidates.items(), key=lambda p: (p[1][0], p[0]))]


def feature_rows(case, rows, refs, values):
    result = []
    for example in rows:
        tk, ck, rk = refs[(case['case_id'], example['id'])]
        row = combine(case, example,
            {**values[tk], **context_local(case)},
            {**values[ck], **context_local(example, example=True)},
            {**values[rk], **reply_local(example)})
        row = crosses.expand(row)
        row['id_cross.chat_id'] = identity_crosses.pair(row, 'target.chat_id', 'example_context.chat_id')
        result.append(row)
    return result


def choose(rows, scores, retriever, *, count, budget):
    require(len(rows) == len(scores), 'Ranker score count differs from recall')
    selected, content_seen, source_seen = [], set(), set()
    for row, score in sorted(zip(rows, scores), key=lambda p: (-float(p[1]), str(p[0]['id']))):
        require(float('-inf') < float(score) < float('inf'), 'Nonfinite ranker score')
        content = digest(dict(context=[{'sender': m['sender'], 'text': m['text']}
            for m in row['context_messages']], reply=row['reply']))
        source = digest([row['context_message_ids'], row['reply_message_ids']])
        if content in content_seen or source in source_seen or len(selected) >= count:
            continue
        proposed = [*selected, row]
        block, ids = retriever.render_selected(proposed, max_chars=budget)
        if ids != [x['id'] for x in proposed] or len(block) > budget:
            continue
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
        return extraction.prepare([dict(target_id=case['case_id'], target=case,
            candidates=[dict(example=row) for row in rows])], self.proof['feature_identity'])

    def select(self, case, rows, *, count, budget, retriever, check):
        if not rows or count <= 0:
            return []
        from .fewshot_ranker.boosting import score_document
        check()
        tasks, refs = self.tasks(case, rows)
        values = {key: extraction.extract_one(self.cache, key, task, self.client)
                  for key, task in tasks.items()}
        scores = score_document(self.model, feature_rows(case, rows, refs, values))
        check()
        return choose(rows, scores, retriever, count=count, budget=budget)
