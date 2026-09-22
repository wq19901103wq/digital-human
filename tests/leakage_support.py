"""Offline source fixtures. Integration tests do not replace any guard function."""
from types import SimpleNamespace

import pytest

from src.iteration import training as recipe
from src.bootstrap import history
from src.config import sha256_file
from src.iteration import datasets, learning_guard, versions
from src.iteration.storage import write_json


def mechanical(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'persona.md').write_text(recipe.PERSONA)
    for name, content in recipe.SCENARIOS.items():
        path = directory / 'scenarios' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
    evidence = directory.parent / (directory.name + '-source.json')
    write_json(evidence, {'recipe': 'reviewed mechanical instructions'})
    record = {'verified': True, 'information_end': 0,
              'evidence_files': {str(evidence): sha256_file(evidence)},
              'asset_files': dict(learning_guard.MECHANICAL_ASSETS)}
    write_json(directory / 'learning.json', record)
    return record


def historical(directory):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'manifest.json', {'id': directory.name})
    messages = []
    for i in range(8):
        # Fill the context window within each session so every role's complete
        # span stays inside its own temporal window.
        session = [(False, f'问题{j}') for j in range(9)] + [(True, '回复'), (True, '接着说')]
        for offset, (self_, text) in enumerate(session):
            messages.append({'chat_id': 'chat-0', 'chat_type': 'private', 'source_chat_id': 'source-0',
                'chat_name': '例子', 'sender': '本人' if self_ else '朋友', 'is_self': self_,
                'text': f'{text}-{i}', 'timestamp': 100000 + i * 10000 + offset})
    examples = history.examples(messages)
    cases = [history.as_case(e) for e in examples]
    history.write_rows(directory / 'messages.jsonl', messages)
    history.write_rows(directory / 'fewshot_pool.jsonl', examples)
    roles = {'judge_training': cases[:1], 'gen_learning': cases[:1], 'gen_optimization': cases[1:2],
             'development': cases[3:5], 'judge_development': cases[3:5], 'fixed_test': cases[6:]}
    purpose = {'history_policy': 'complete_before_input_v1', 'roles': {},
               'protocol': {'development_start': 130000, 'acceptance_start': 160000, 'unseen_chat_ids': []}}
    for role, rows in roles.items():
        path = directory / datasets.ROLES[role][1]
        history.write_rows(path, rows)
        purpose['roles'][role] = {'file': path.name, 'sha256': sha256_file(path)}
    write_json(directory / 'purposes.json', purpose)
    write_json(directory / 'report.json', {'review_status': 'approved',
        'history_policy': purpose['history_policy'], 'examples_sha256': sha256_file(directory / 'fewshot_pool.jsonl')})
    return SimpleNamespace(directory=directory, messages=messages, examples=examples, cases=cases, roles=roles)


@pytest.fixture
def isolated_legacy_provenance(monkeypatch):
    """Only for old synthetic protocol/cache unit fixtures with no source archive.

    These tests exercise their original units (data_guard, votes, pointers), not
    provenance. test_leakage_guards uses real raw messages and unmodified guards.
    Never enable this fixture globally or use it in leakage integration tests.
    """
    seal = SimpleNamespace(check=lambda: None)
    monkeypatch.setattr(learning_guard, 'require_materials', lambda *a, **kw: {})
    monkeypatch.setattr(learning_guard, 'bind', lambda *a: None)
    monkeypatch.setattr(learning_guard, 'verify', lambda *a: None)
    monkeypatch.setattr(learning_guard, 'execution_seal', lambda *a: seal)
    monkeypatch.setattr(learning_guard, 'verify_pack', lambda *a: None)
    monkeypatch.setattr(learning_guard, 'snapshot', lambda *a: {})
    monkeypatch.setattr(learning_guard, 'verify_generation', lambda *a: seal)
