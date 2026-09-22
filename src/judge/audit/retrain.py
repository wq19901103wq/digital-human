"""Frozen LR retraining checkpoint and training-row tools. Extracted from scripts/legacy/retrain_judge_lr.py and scripts/legacy/retrain_judge_lr_dataset.py."""
from __future__ import annotations

import json
import random
from pathlib import Path

from ... import cache
from ...config import ConfigError, sha256_file
from .. import lr_retrain as lr, corrected_v1 as rt
from ..corrected import RUNTIME_PATH, blind_case


def read(path):
    return json.loads(path.read_text())


def checkpoint(directory, spec, rows):
    path = directory / 'features.json'
    expected = cache.digest({'training': spec['training_sha256'], 'config': spec['feature_config'],
                             'runtime': spec['inputs'][str(RUNTIME_PATH.resolve())],
                             'extractor': sha256_file(Path(lr.__file__))})
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


class ExcludingRetriever:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.blocked = set()
        self.failure = None

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def retrieve(self, **kwargs):
        try:
            kwargs['exclude_ids'] = set(kwargs.get('exclude_ids') or ()) | self.blocked
            rows = self.wrapped.retrieve(**kwargs)
            if any(str(r['id']) in self.blocked for r in rows):
                raise ConfigError('召回器返回被排除的训练目标会话')
            return rows
        except Exception as exc:
            self.failure = exc
            raise


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
