import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.offline_demo import run


def test_clean_checkout_trains_audits_and_evaluates_synthetic_data(tmp_path):
    result = run(tmp_path / 'demo')
    assert result['training'] == result['evaluation'] == 'finished'
    assert result['source_audit'] == 'passed'
    assert result['evaluation_pairs'] == 4
    assert result['external_model_requests'] == 0
    root = tmp_path / 'demo/instances/demo'
    spec = json.loads((root / 'judge_training/synthetic-training/spec.json').read_text())
    assert spec['training_total'] == 4
    assert spec['kind'] == 'history_lr_training'
    candidate = root / 'judges' / result['judge_ref']
    artifact = json.loads((candidate / 'correction.json').read_text())
    assert artifact['training']['evaluation_used_for_fit'] is False


def test_fresh_instance_cli_uses_explicit_policy_and_never_overwrites_baseline(tmp_path):
    from scripts.offline_demo import synthetic_messages
    source = tmp_path / 'synthetic.jsonl'
    source.write_text(''.join(json.dumps(m) + '\n' for m in synthetic_messages()))
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'total': 2, 'train_total': 4, 'acceptance_total': 4,
        'development_start': 140000, 'acceptance_start': 160000, 'familiar_weight': 1,
        'group_weight': .5, 'training_group_weight': .5, 'heldout_chat_fraction': 0,
        'evaluation_chat_cap': 10, 'learning_chat_cap': 10}))
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'DH_INSTANCES_ROOT': str(tmp_path / 'instances')}
    command = [sys.executable, str(root / 'scripts/bootstrap.py'), '--instance', 'demo',
        '--data', str(source), '--policy', str(policy), '--model', 'synthetic',
        '--judge-model', 'synthetic', '--initialize']
    result = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    instance = tmp_path / 'instances/demo'
    purpose = json.loads((instance / 'data' / data['data_ref'] / 'purposes.json').read_text())
    assert purpose['roles']['development']['total'] == 2
    assert purpose['roles']['fixed_test']['total'] == 4
    before = (instance / 'pointers.json').read_bytes()
    repeated = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert (instance / 'pointers.json').read_bytes() == before
