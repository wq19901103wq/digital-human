"""Real-runtime diagnostics compare both routes without evaluating or drawing cases."""
import argparse
import json
from types import SimpleNamespace

import pytest

from scripts import check
from src.iteration import retrieval_profile, versions


class Retriever:
    def retrieve(self, **options):
        return [dict(id='same', feature=1)]

    def render_selected(self, rows):
        return json.dumps(rows), [row['id'] for row in rows]


def test_compare_alternates_and_restores_routes():
    retriever = Retriever()
    saved = retriever.retrieve
    calls = []
    def original(subject, **options):
        calls.append('old')
        return saved(**options)
    def recall(subject, case):
        calls.append('recall')
        return subject.retrieve(query=case['query'])
    result = retrieval_profile.compare(retriever, original, recall,
        [dict(query='one', chat_type='group'), dict(query='two', chat_type='private')], repeats=2)
    assert result['equivalent'] and result['model_calls'] == 0
    assert result['strata'] == dict(group=1, private=1)
    assert result['timings']['original']['calls'] == result['timings']['indexed']['calls'] == 4
    assert calls.count('old') == 4 and calls.count('recall') == 8
    assert retriever.retrieve == saved


def test_compare_rejects_changed_features_and_restores():
    retriever = Retriever()
    saved = retriever.retrieve
    with pytest.raises(ValueError, match='changed selected rows'):
        retrieval_profile.compare(retriever, lambda subject, **options: [dict(id='same', feature=2)],
            lambda subject, case: subject.retrieve(), [dict(chat_type='group')], repeats=1)
    assert retriever.retrieve == saved


@pytest.mark.parametrize('value', [0, -1, 51])
def test_profile_counts_bounded(value):
    with pytest.raises(argparse.ArgumentTypeError):
        retrieval_profile.positive(value)


def test_check_retrieval_dispatch_is_offline(tmp_path, monkeypatch, capsys):
    pointer = tmp_path / 'pointers.json'
    pointer.write_text('{}')
    monkeypatch.setattr(versions, 'switch_instance', lambda name: tmp_path)
    monkeypatch.setattr(versions, 'POINTERS_PATH', pointer)
    calls = []
    monkeypatch.setattr(retrieval_profile, 'inspect', lambda *args, **options:
        calls.append((args, options)) or dict(equivalent=True, model_calls=0))
    assert check.main(['retrieval', '--instance', 'demo', '--exp', 'saved', '--cases', '2']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['model_requests_prohibited'] and not report['evaluation_started']
    assert calls == [(('demo', 'saved'), dict(cases=2, repeats=1, materials=False))]


def test_profile_launches_verified_isolated_executor(tmp_path, monkeypatch):
    from src.iteration import runtime
    monkeypatch.setattr(versions, 'PRIVATE', tmp_path / 'instances' / 'demo')
    snapshot = tmp_path / 'runtime'
    monkeypatch.setattr(runtime, 'verify', lambda path: snapshot)
    calls = []
    def launch(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout='{"model_calls":0}', stderr='')
    monkeypatch.setattr(retrieval_profile.subprocess, 'run', launch)
    assert retrieval_profile.inspect('demo', 'saved', cases=2) == dict(model_calls=0)
    command, options = calls[0]
    assert command[1] == '-B' and '--snapshot' in command
    assert options['cwd'] == snapshot
    assert options['env']['DH_INSTANCES_ROOT'] == str(versions.PRIVATE.parent)
    assert options['env']['DH_SETTINGS_FILE'] == str(snapshot / 'settings.snapshot.yaml')
    assert options['env']['DH_ENV_FILE'].endswith('.env')


def test_material_failure_preserves_successful_retrieval(tmp_path, monkeypatch, capsys):
    pointer = tmp_path / 'pointers.json'
    pointer.write_text('{}')
    monkeypatch.setattr(versions, 'switch_instance', lambda name: tmp_path)
    monkeypatch.setattr(versions, 'POINTERS_PATH', pointer)
    result = dict(equivalent=True, model_calls=0, timings={'indexed': 1},
                  materials=dict(status='failed', error='cached input missing'))
    monkeypatch.setattr(retrieval_profile, 'inspect', lambda *args, **options: result)
    assert check.main(['retrieval', '--instance', 'demo', '--exp', 'saved', '--materials']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['checks'][0]['detail']['timings'] == {'indexed': 1}
    assert report['checks'][0]['status'] == 'passed'
    assert report['checks'][1]['status'] == 'failed'


def test_material_diagnostic_collects_reuse_and_partial_failure():
    from contextlib import nullcontext
    calls = []
    def require(*args):
        calls.append(args)
        if len(calls) == 2:
            raise ValueError('missing cache result')
    reuse = SimpleNamespace(verified_materials=lambda *args, **options: nullcontext())
    guard = SimpleNamespace(require_materials=require)
    profile = SimpleNamespace(snapshot=lambda: dict(metrics={'reused': 1}))
    result = retrieval_profile.measure_materials(reuse, guard, 'data', [], 'fixed',
                                                 profile=profile, disk=None)
    assert result['status'] == 'failed'
    assert len(result['calls_seconds']) == 1
    assert result['metrics'] == {'reused': 1}
    assert 'missing cache result' in result['error']


def test_material_diagnostic_success_checks_two_contexts():
    from contextlib import nullcontext
    calls = []
    reuse = SimpleNamespace(verified_materials=lambda *args, **options: nullcontext())
    guard = SimpleNamespace(require_materials=lambda *args: calls.append(args))
    profile = SimpleNamespace(snapshot=lambda: dict(metrics={}))
    result = retrieval_profile.measure_materials(reuse, guard, 'data', [], 'fixed',
                                                 profile=profile, disk=None)
    assert result['status'] == 'passed'
    assert len(result['calls_seconds']) == len(calls) == 2
