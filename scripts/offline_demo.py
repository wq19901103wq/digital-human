#!/usr/bin/env python3
"""Run real source guards, feature fitting and evaluation on synthetic conversations."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from src import llm, tracing  # noqa: E402
from src.bootstrap import history  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.dashboard import report  # noqa: E402
from src.iteration import acceptance, baseline, experiment, runner, training, versions  # noqa: E402
from src.iteration import training_evidence, training_operations  # noqa: E402
from src.iteration.storage import write_json  # noqa: E402
from src.judge import corrected, corrected_v1 as rt  # noqa: E402


def synthetic_messages():
    rows = []
    for kind in ('group', 'private'):
        for session in range(8):
            messages = [(False, f'合成问题 {j}') for j in range(9)] + [(True, '合成回复'), (True, '继续聊')]
            for offset, (self_, text) in enumerate(messages):
                rows.append({'chat_id': 'synthetic-' + kind, 'source_chat_id': 'synthetic-' + kind,
                    'chat_type': kind, 'chat_name': '合成聊天', 'sender': '本人' if self_ else '合成朋友',
                    'is_self': self_, 'text': f'{text}-{session}', 'timestamp': 100000 + session * 10000 + offset})
    return rows


class SyntheticChat:
    def __init__(self, config):
        self.config = config

    def cache_identity(self):
        return {'provider': 'offline_synthetic', 'model': self.config['model']}

    def chat(self, messages, **kwargs):
        return json.dumps({'replies': ['合成机器回复']})


class SyntheticJudge:
    calls = 0

    def __init__(self, config):
        self.config = config

    def cache_identity(self):
        return self.config

    def run(self, prompt, schema=None):
        type(self).calls += 1
        with tracing.step('codex', {'prompt': prompt, 'schema': schema, 'config': self.config}) as event:
            if schema:
                option = {**dict.fromkeys(rt.INTEGER_FIELDS, 1), **dict.fromkeys(rt.BOOLEAN_FIELDS, False),
                          **{key: values[0] for key, values in rt.ENUM_FIELDS.items()}}
                result = {'option_A': option, 'option_B': option}
            else:
                result = {'human_option': 'A', 'confidence': .9, 'reason': 'synthetic response'}
            raw = json.dumps(result)
            event['response'] = {'text': raw}
            return raw


def run(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('choose a new empty output directory; prior evidence is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    for dataset in ('development', 'fixed_test'):
        settings['evaluation'][dataset].update(total=4, group_ratio=.5)
    settings['evaluation']['pool_min_total'] = 0
    config = output / 'settings.yaml'
    config.write_text(yaml.safe_dump(settings, allow_unicode=True))
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {'DH_INSTANCES_ROOT': str(output / 'instances'),
            'DH_INSTANCE': 'demo', 'DH_SETTINGS_FILE': str(config)}))
        previous = versions.PRIVATE
        versions.switch_instance('demo')
        # Restore globals for callers that run this demonstration in a test process.
        for key in ('PRIVATE', 'DATA_ROOT', 'GEN_ROOT', 'JUDGE_ROOT', 'POINTERS_PATH'):
            value = {'PRIVATE': previous, 'DATA_ROOT': previous / 'data', 'GEN_ROOT': previous / 'generators',
                     'JUDGE_ROOT': previous / 'judges', 'POINTERS_PATH': previous / 'pointers.json'}[key]
            stack.callback(setattr, versions, key, value)
        def build(settings, cfg):
            return SyntheticChat(cfg)
        for module in (llm, training_operations, training_evidence):
            stack.enter_context(patch.object(module, 'build_clients', build))
        for module in (training, training_operations, corrected):
            stack.enter_context(patch.object(module, 'CodexJudgeClient', SyntheticJudge))
        messages = synthetic_messages()
        examples = history.examples(messages)
        roles, policy = history.plan(examples, total=4, train_total=4, development_start=140000,
            acceptance_start=160000, familiar_weight=1, group_weight=.5, training_group_weight=.5,
            heldout_chat_fraction=0, evaluation_chat_cap=10, learning_chat_cap=10)
        data_ref = history.publish(messages, examples, roles, policy, {})
        gen_ref = training.mechanical_generator(data_ref, {'llm': {'model': 'synthetic-generator'},
                                                          'retriever': {'enabled': True}})
        feature = {'provider': 'codex_cli', 'model': 'synthetic-features', 'reasoning_effort': 'high',
                   'codex_cli_version': 'synthetic', 'timeout_seconds': 1}
        plan = {'data_ref': data_ref, 'generator_ref': gen_ref, 'feature_configs': {'candidate': feature},
            'scoring': {'initial': {**feature, 'model': 'synthetic-judge'},
                        'correction_threshold': .7, 'decision_policy': 'hybrid'},
            'change': 'synthetic end-to-end wiring check'}
        directory = training.submit('synthetic-training', plan)
        training.run(directory.name, workers=2)
        result = json.loads((directory / 'audit.json').read_text())
        judge_ref = result['candidates']['candidate']
        receipt = baseline.prepare(data_ref, gen_ref, judge_ref, reason='initialize synthetic demonstration only')
        baseline.apply(receipt.stem)
        acceptance.seal(data_ref, 4)
        evaluation = experiment.create_gen_experiment('development', 'synthetic model configuration change',
            {'llm': {'model': 'synthetic-candidate'}}, exp_id='synthetic-evaluation')
        runner.run_gen_experiment(evaluation)
        report.refresh_dashboard(versions.PRIVATE / 'experiments', output / 'dashboard/demo')
        report.write_instance_index(versions.PRIVATE.parent, output / 'dashboard')
        summary = {'training': json.loads((directory / 'state.json').read_text())['status'],
            'source_audit': result['status'], 'training_cases': 4,
            'evaluation': experiment.state_of(evaluation)['status'],
            'evaluation_pairs': experiment.state_of(evaluation)['metrics']['pairs'],
            'data_ref': data_ref, 'generator_ref': gen_ref, 'judge_ref': judge_ref,
            'external_model_requests': 0, 'synthetic_transport_calls': SyntheticJudge.calls,
            'performance_claim': False}
        write_json(output / 'verification.json', summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False, indent=2))
