"""Explicitly authorized LR retraining; frozen features and paired trace replay."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .. import cache, tracing
from ..config import ConfigError
from . import corrected_v1 as rt
from .corrected import CorrectedJudge, blind_case


def difference_vector(model, features, blind, metadata):
    a, b, names = rt.vectorize_boolean_options(features)
    observed_a, observed_names = rt.vectorize_observable_option_booleans(blind['option_A'], metadata)
    observed_b, _ = rt.vectorize_observable_option_booleans(blind['option_B'], metadata)
    group_a, group_names = rt.vectorize_group_pattern_option_boolean(blind['option_A'], metadata)
    group_b, _ = rt.vectorize_group_pattern_option_boolean(blind['option_B'], metadata)
    names = [*names, *observed_names, *group_names]
    context, context_names = rt.vectorize_context_booleans(metadata, model.context_schema)
    left, left_names = rt.expand_boolean_option_with_refined_context(
        np.concatenate((a, observed_a, group_a)), names, context, context_names)
    right, right_names = rt.expand_boolean_option_with_refined_context(
        np.concatenate((b, observed_b, group_b)), names, context, context_names)
    if left_names != right_names or tuple(left_names) != model.feature_names:
        raise ConfigError('训练向量与冻结 LR 特征定义不一致')
    vector = left - right
    expected = rt.score_formal_judge_pair(model, features, blind['option_A'], blind['option_B'], metadata)
    if not np.isclose(rt.probability_from_vectors(vector, model.coefficients), expected, atol=1e-12):
        raise ConfigError('训练与推理向量不一致')
    return vector


def fit(rows, entries, artifact):
    spec_path = Path(artifact).parent / 'spec.json'
    if not spec_path.is_file():
        raise ConfigError('拟合缺少冻结训练规格，禁止直接使用旧模型模板')
    spec = json.loads(spec_path.read_text())
    if len(rows) != spec['training_total'] or any(r.get('split') != 'train' for r in rows):
        raise ConfigError('必须使用完整冻结训练清单，禁止混入评测样本或缩小训练集')
    from ..iteration.learning_guard import verify_fit
    seal = verify_fit(rows, entries, artifact)
    # Same sparse input, solver and defaults as the original train1200 recipe.
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    model = rt.load_formal_judge(artifact)
    matrix = csr_matrix(np.stack([difference_vector(model, entries[r['case_id']]['features'],
        r['blind'], r['metadata']) for r in rows]))
    labels = np.asarray([int(r['human_option'] == 'A') for r in rows])
    classifier = LogisticRegression(**spec['recipe'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        classifier.fit(matrix, labels)
    seal.check()
    return classifier


def extract(client, blind, source_check=None):
    prompt = rt.feature_extractor_prompt(blind)
    for attempt in range(2):
        if source_check:
            source_check()
        with cache.validation(rt.parse_feature_response):
            raw = client.run(prompt, rt.feature_response_json_schema())
        if source_check:
            source_check()
        try:
            features = rt.parse_feature_response(raw)
            tracing.note('features', {'valid': True, 'parsed': features, 'attempt': attempt + 1})
            return features
        except ValueError:
            if attempt:
                raise


def training_rows(source):
    blind = json.loads((source / 'development_blind.json').read_text())
    answers = json.loads((source / 'development_answers.json').read_text())
    replay = json.loads((source / 'development_replay.json').read_text())
    if answers['blind_sha256'] != rt._json_sha256(blind):
        raise ConfigError('历史训练答案与盲包指纹不一致')
    answer_map = {r['case_id']: r for r in answers['cases']}
    replay_map = {r['case_id']: r for r in replay['cases']}
    rows = []
    for case in blind['cases']:
        answer = answer_map[case['case_id']]
        if case['split'] != answer['split']:
            raise ConfigError('历史训练划分不一致')
        if case['split'] != 'train':
            continue
        source_row = replay_map[answer['source_case_id']]
        if case['relationship'] != source_row['relationship'] or answer['human_option'] not in ('A', 'B'):
            raise ConfigError('历史训练来源或标签不一致')
        if case['option_' + answer['human_option']] != source_row['human_reply_original']:
            raise ConfigError('历史训练真人回复映射不一致')
        metadata = rt.context_metadata({'relationship': case['relationship'], 'chat_id': source_row['source_chat_id']},
            source_row['c0_trace']['input_metadata'], {}, case['context_original'])
        rows.append({'case_id': case['case_id'], 'split': 'train', 'human_option': answer['human_option'],
            'source_case_id': answer['source_case_id'], 'source_chat_id': source_row['source_chat_id'],
            'source_timestamp': source_row['timestamp'], 'metadata': metadata,
            'blind': {k: case[k] for k in ('relationship', 'context_original', 'option_A', 'option_B')}})
    if len(rows) != 1200 or len({r['case_id'] for r in rows}) != 1200:
        raise ConfigError('原训练集不是 1200 个唯一样本')
    return rows


class FeatureReplay:
    """Reuse only the exact source branch/round; newly needed rounds remain independent."""
    def __init__(self, source_dir, source_spec, source_config, rows):
        self.source_dir = Path(source_dir) if source_dir is not None else None
        self.source_spec = source_spec
        self.config = {**source_config['llm'], **source_config.get('feature_llm', {})}
        self.by_id = {str(r['case_id']): r for r in rows}
        self.records = {}
        saved_lines = (self.source_dir / 'cases.jsonl').read_text().splitlines() if self.source_dir else []
        for line in saved_lines:
            record = json.loads(line)
            self.records[str(record['case_id'])] = record
        self.memo = {}

    def saved(self, case, round_index):
        record = self.records.get(str(case['case_id']), {})
        if record.get('status') != 'ok':
            return None
        trace = tracing.read(self.source_dir, record['trace_ref'])
        if trace['case'] != self.by_id[str(case['case_id'])] or trace['versions']['candidate_ref'] != self.source_spec['candidate_ref']:
            raise ConfigError('来源 trace 的数据或版本与冻结包不一致')
        for index, op in enumerate(trace['operations']):
            if op['kind'] != 'judge' or op['branch'] != 'candidate' or op['round'] != round_index or op['status'] != 'ok':
                continue
            mappings = [e['data'] for e in op['events'] if e['kind'] == 'blind_mapping']
            features = [e['data']['parsed'] for e in op['events'] if e['kind'] == 'features' and e['data'].get('valid')]
            if len(mappings) != 1 or not features:
                continue  # Whole-judge cache hits can have no materialized feature event.
            mapping = mappings[0]
            human = mapping['human_option']
            if human not in ('A', 'B') or mapping['candidate_option'] != ('B' if human == 'A' else 'A'):
                raise ConfigError('来源 trace 的 A/B 身份映射不一致')
            a, b = (case['human_reply'], case['ai_replies']) if human == 'A' else (case['ai_replies'], case['human_reply'])
            blind, metadata = blind_case(case, a, b)
            if mapping['blind_case'] != blind or cache.digest(mapping['context_metadata']) != cache.digest(metadata):
                raise ConfigError('来源 trace 的提示词上下文或选项已变化')
            prompt = rt.feature_extractor_prompt(blind)
            calls = [e for e in op['events'] if e['kind'] == 'codex' and e.get('request', {}).get('prompt') == prompt]
            if calls and any(e['request'].get('config') != self.config or
                             e['request'].get('schema') != rt.feature_response_json_schema() for e in calls):
                raise ConfigError('来源特征调用模型或推理配置不匹配')
            if not calls:
                continue  # No verifiable request provenance; normal request cache may still resolve it.
            valid_calls = [e for e in calls if e.get('status') == 'ok']
            if not valid_calls or rt.parse_feature_response(valid_calls[-1]['response']['text']) != features[-1]:
                raise ConfigError('来源特征与保存的模型响应不一致')
            rt.validate_option_features(features[-1]['option_A'])
            rt.validate_option_features(features[-1]['option_B'])
            return {'mapping': mapping, 'features': features[-1], 'provenance': {
                'experiment': self.source_dir.name, 'trace_ref': record['trace_ref'],
                'operation': index, 'round': round_index, 'operation_sha256': cache.digest(op)}}
        return None

    def get(self, case, round_index, client, source_check=None):
        if source_check:
            source_check()
        if case != self.by_id[str(case['case_id'])]:
            raise ConfigError('复用特征的样本内容与冻结评测包不一致')
        key = (str(case['case_id']), round_index)
        if key not in self.memo:
            saved = self.saved(case, round_index)
            if saved is None:
                # Both weights receive this same draw, but rounds 0/1/2 have separate keys.
                identity = {'case': case, 'client': client.cache_identity(), 'policy': 'paired-lr-v1'}
                def produce():
                    swap = cache.memo('blind_order', identity, lambda: random.random() < .5)
                    a, b = (case['ai_replies'], case['human_reply']) if swap else (case['human_reply'], case['ai_replies'])
                    blind, metadata = blind_case(case, a, b)
                    return {'mapping': {'human_option': 'B' if swap else 'A', 'candidate_option': 'A' if swap else 'B',
                        'blind_case': blind, 'context_metadata': metadata}, 'features': extract(client, blind, source_check),
                        'provenance': {'source': 'independent_supplement', 'round': round_index}}
                saved = cache.memo('paired_lr_features', identity, produce)
            self.memo[key] = saved
        if source_check:
            source_check()
        return self.memo[key]


class LRJudge(CorrectedJudge):
    """Feature-only LR inference. An experiment may supply a paired replay provider."""
    decision_policy = 'lr_only'

    def __init__(self, *args, replay=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay = replay

    def probability_a(self, features, blind, metadata):
        return rt.score_formal_judge_pair(self.model, features, blind['option_A'], blind['option_B'], metadata)

    def is_ai(self, case, candidate_replies):
        self._stage('extracting')
        if self.replay is not None:
            if candidate_replies != case['ai_replies']:
                raise ConfigError('配对复用的候选回复已变化')
            scope = cache._scope.get()
            if scope is None:
                raise ConfigError('配对复用必须在独立轮次 trace 作用域内运行')
            result = self.replay.get(case, scope['context']['round'], self.feature_client, self._check_sources)
        else:
            identity = {'case': case, 'candidate_replies': candidate_replies, 'client': self.feature_client.cache_identity()}
            swap = cache.memo('blind_order', identity, lambda: random.random() < .5)
            a, b = (candidate_replies, case['human_reply']) if swap else (case['human_reply'], candidate_replies)
            blind, metadata = blind_case(case, a, b)
            result = {'mapping': {'human_option': 'B' if swap else 'A', 'candidate_option': 'A' if swap else 'B',
                'blind_case': blind, 'context_metadata': metadata}, 'features': extract(self.feature_client, blind, self._check_sources)}
        mapping, features = result['mapping'], result['features']
        tracing.note('blind_mapping', mapping)
        tracing.note('features', {'valid': True, 'parsed': features})
        tracing.note('feature_reuse', result.get('provenance', {'source': 'live'}))
        self._stage('predicting')
        blind = mapping['blind_case']
        probability = self.probability_a(features, blind, mapping['context_metadata'])
        human = 'A' if probability >= .5 else 'B'
        self.last_verdict = {'human_option': human, 'candidate_option': mapping['candidate_option'],
            'small_model_probability_a': probability, 'decision_policy': self.decision_policy,
            'identified_ai': human == mapping['human_option']}
        tracing.note('correction', self.last_verdict)
        self._check_sources()
        return self.last_verdict['identified_ai']
