import json
from pathlib import Path

import pytest

from src.infrastructure.storage import read_json, write_json
from src.judge import corrected
from src.judge.codex_transport import CodexServiceBlocked, CodexTransportError
from src.share.wiki import wiki_batch, wiki_connection_retry as connection, wiki_job_pool
from src.share.wiki import wiki_client_recovery as recovery, wiki_repair, wiki_structured
from test_wiki_client_recovery import prepared_recovery, snapshot
from test_wiki_repair import Client, feedback_fixture


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('connection retry regression must not make live model requests')
    monkeypatch.setattr(corrected.CodexJudgeClient, '__init__', forbidden)
    monkeypatch.setattr(corrected.CodexJudgeClient, '_run', forbidden)
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', lambda: None)


def diagnostic(signal='stream_disconnected'):
    return dict(exit_code=1, terminal_events=['turn.failed'], signals=[signal],
                terminal_signals=[signal], stderr_signals=[])


def connection_gap(output, job_id='unknown', details=None):
    errors = [dict(stage='structuring', batch=0,
                   error=str(CodexTransportError(details or diagnostic())))]
    write_json(output / 'jobs' / job_id / 'progress.json', dict(stage='incomplete', errors=errors))
    state = read_json(output / 'progress.json')
    state['jobs'][job_id].update(stage='incomplete',
        error=f'RuntimeError: repair incomplete: {errors}'[-1600:])
    write_json(output / 'progress.json', state)


def test_retries_release_slots_preserve_requests_and_report_waiting(tmp_path, monkeypatch):
    budget = wiki_job_pool.RequestBudget(1)
    tracker = connection.ConnectionRetries(tmp_path, 60)
    job = dict(id='selected')
    schema = dict(type='object')
    calls, sleeps = [], []
    class Disconnect(Client):
        def run(self, prompt, schema):
            calls.append((prompt, schema))
            if len(calls) <= 3:
                raise CodexTransportError(diagnostic())
            return 'accepted'
    client = Disconnect()
    wrapped = tracker.wrap(budget.wrap(client), job)
    def wait(seconds):
        assert budget.slots.acquire(blocking=False)
        budget.slots.release()
        sleeps.append(seconds)
        status = connection.status(tmp_path, tracker.state['run_id'], running=True)
        assert len(status['waiting']) == 1
        assert status['waiting'][0]['next_retry_at'] > status['waiting'][0]['updated_at']
        assert status['waiting'][0]['diagnostic'] == diagnostic()
    monkeypatch.setattr(connection.time, 'sleep', wait)
    assert wrapped.cache_identity() == client.cache_identity()
    assert wrapped.run('unchanged source', schema) == 'accepted'
    assert calls == [('unchanged source', schema)] * 4 and sleeps == [60] * 3
    status = connection.status(tmp_path, tracker.state['run_id'], running=True)
    assert status['total_retries'] == 3 and status['waiting'] == []
    events = [json.loads(line) for line in (tmp_path / 'connection-retry.jsonl').read_text().splitlines()]
    assert [event['phase'] for event in events] == ['waiting', 'retrying'] * 3 + ['recovered']
    assert 'unchanged source' not in (tmp_path / 'connection-retry.jsonl').read_text()
    assert connection.status(tmp_path, 'different-run', running=True) is None


def test_client_timeout_is_retryable_without_fabricated_terminal_signal():
    details = diagnostic('request_timeout') | dict(terminal_signals=[],
        failure_kind='subprocess_timeout', exit_code_source='client_timeout', timeout_seconds=1800)
    assert connection.retryable_diagnostic(details)
    assert not connection.retryable_diagnostic(details | dict(failure_kind='unknown'))
    assert not connection.retryable_diagnostic(diagnostic('model_catalog_error'))


@pytest.mark.parametrize('error', [ValueError('rejected output'), RuntimeError('unknown failure'),
    CodexTransportError(diagnostic('authentication_error')),
    CodexTransportError(diagnostic('rate_limit')),
    CodexTransportError(diagnostic() | dict(service_errors=[dict(code='service_unavailable')])),
    CodexTransportError(dict(exit_code=1)), CodexServiceBlocked(diagnostic())])
def test_does_not_retry_content_unknown_or_blocked_failures(tmp_path, monkeypatch, error):
    class Fail(Client):
        def run(self, prompt, schema):
            raise error
    tracker = connection.ConnectionRetries(tmp_path, 60)
    monkeypatch.setattr(connection.time, 'sleep', lambda *_: pytest.fail('not a connection retry'))
    with pytest.raises(type(error)):
        tracker.wrap(Fail(), dict(id='selected')).run('original')
    assert tracker.state['total_retries'] == 0


def test_retries_preserve_format_feedback_budget_and_success_cache(tmp_path, monkeypatch):
    directory, prompt, schema, parse = feedback_fixture(tmp_path)
    tracker = connection.ConnectionRetries(tmp_path, 60)
    failures_before = {}
    class Recover(Client):
        def run(self, request, schema):
            self.prompts.append(request)
            if len(self.prompts) <= 2:
                return '{"value": "unsupported"}'
            if not failures_before:
                failures_before.update(snapshot(tmp_path))
            if len(self.prompts) <= 5:
                raise CodexTransportError(diagnostic())
            return '{"value": "supported"}'
    client = Recover()
    wrapped = tracker.wrap(client, dict(id='selected'))
    monkeypatch.setattr(connection.time, 'sleep', lambda *_: None)
    assert wiki_repair._cached_call(wrapped, directory, prompt, schema, parse) == dict(value='supported')
    assert len(client.prompts) == 6 and client.prompts[-4:] == [client.prompts[-1]] * 4
    assert failures_before and all((tmp_path / path).read_bytes() == content
                                  for path, content in failures_before.items())
    key = wiki_repair.digest([client.cache_identity(), prompt, schema])
    assert read_json(directory / f'{key}.json')['attempts'] == 3
    assert wiki_repair._cached_call(wrapped, directory, prompt, schema, parse) == dict(value='supported')
    assert len(client.prompts) == 6 and tracker.state['total_retries'] == 3


def test_explicit_connection_resume_keeps_claim_frozen_recipe_and_completed_work(tmp_path, monkeypatch):
    source, output, knowledge, _, manifest, permit = prepared_recovery(tmp_path)
    recovery.claim(permit, 'unknown')
    connection_gap(output)
    ledger = (source / recovery.LEDGER).read_bytes()
    manifest_bytes = (output / 'manifest.json').read_bytes()
    inherited = snapshot(output / 'jobs/unknown')
    original_state = read_json(output / 'progress.json')
    configs, calls = [], []
    def client(config):
        configs.append(config)
        return Client()
    monkeypatch.setattr(corrected, 'CodexJudgeClient', client)
    def finish(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(directory.name)
        assert config == manifest['config'] and kwargs['workers'] == 8
        assert isinstance(kwargs['client'], connection.ConnectionRetryClient)
        assert recovery.validate_batch(manifest, output)[1] == {'unknown'}
        wiki_structured.export(directory / 'content', read_json(knowledge))
        return dict(stage='complete')
    monkeypatch.setattr(wiki_repair, 'repair', finish)
    options = dict(job_ids=['unknown'], connection_retry_seconds=60, attempts=1,
                   object_workers=4, request_workers=8, request_timeout_seconds=1800)
    state = wiki_batch.run(output, **options)
    assert calls == ['unknown'] and state['jobs']['unknown']['stage'] == 'complete'
    assert configs == [manifest['config'] | dict(timeout_seconds=1800)]
    for job_id in original_state['jobs']:
        if job_id != 'unknown':
            assert state['jobs'][job_id] == original_state['jobs'][job_id]
    assert (source / recovery.LEDGER).read_bytes() == ledger
    assert (output / 'manifest.json').read_bytes() == manifest_bytes
    assert snapshot(output / 'jobs/unknown') == inherited
    wiki_batch.run(output, **options)
    assert calls == ['unknown']


@pytest.mark.parametrize('case', ['content', 'scope', 'unknown', 'mismatch', 'rate_limit'])
def test_connection_resume_cannot_expand_scope_or_accept_unproven_failure(tmp_path, case):
    _, output, _, _, _, permit = prepared_recovery(tmp_path)
    recovery.claim(permit, 'unknown')
    job_ids = ['unknown']
    if case != 'content':
        connection_gap(output, details=diagnostic('rate_limit') if case == 'rate_limit' else None)
    if case == 'scope':
        job_ids.append('rejected')
    elif case == 'unknown':
        job_ids.append('not-in-manifest')
    elif case == 'mismatch':
        state = read_json(output / 'progress.json')
        state['jobs']['unknown']['error'] = 'inconsistent parent error'
        write_json(output / 'progress.json', state)
    with pytest.raises(ValueError):
        wiki_batch.run(output, attempts=1, job_ids=job_ids, connection_retry_seconds=60)


@pytest.mark.parametrize('options', [dict(connection_retry_seconds=0),
    dict(connection_retry_seconds=True), dict(attempts=2), dict(job_ids=None),
    dict(cached_only=True), dict(skip_incomplete=True), dict(retry_transport_only=True)])
def test_invalid_connection_modes_fail_before_batch_reads(tmp_path, options):
    with pytest.raises(ValueError, match='connection retry requires'):
        wiki_batch.run(tmp_path / 'missing',
            **(dict(attempts=1, job_ids=['selected'], connection_retry_seconds=60) | options))


@pytest.mark.parametrize('valid', [True, False])
def test_cli_connection_retry_is_explicit_and_passes_to_person_batch(tmp_path, monkeypatch, valid):
    from scripts import batch_wiki as entrypoint
    output = tmp_path / 'instances/batch'
    output.mkdir(parents=True)
    write_json(output / 'manifest.json', dict(schema='wiki_batch_v1'))
    calls = []
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda *_: calls.append('install'))
    monkeypatch.setattr(wiki_batch, 'run', lambda _, **kwargs: calls.append(kwargs) or dict(stage='complete'))
    arguments = ['batch_wiki', 'run', '--output', str(output), '--connection-retry-seconds', '60']
    if valid:
        arguments += ['--job-id', 'selected', '--attempts', '1']
    monkeypatch.setattr(entrypoint.sys, 'argv', arguments)
    if valid:
        assert entrypoint.main() == 0
        assert calls[0] == 'install' and calls[1]['connection_retry_seconds'] == 60
        assert calls[1]['job_ids'] == ['selected']
    else:
        with pytest.raises(SystemExit):
            entrypoint.main()
        assert calls == []


@pytest.mark.parametrize('valid', [True, False])
def test_cli_revision_connection_retry_preserves_selected_scope(tmp_path, monkeypatch, valid):
    from scripts import batch_wiki as entrypoint
    from src.share.wiki import wiki_revision_batch
    output = tmp_path / 'instances/revision'
    calls = []
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda *_: calls.append('install'))
    monkeypatch.setattr(wiki_revision_batch, 'run', lambda _, **kwargs: calls.append(kwargs) or dict(stage='complete'))
    arguments = ['batch_wiki', 'revise', '--output', str(output), '--connection-retry-seconds', '60']
    if valid:
        arguments += ['--job-id', 'selected', '--attempts', '1']
    monkeypatch.setattr(entrypoint.sys, 'argv', arguments)
    if valid:
        assert entrypoint.main() == 0
        assert calls[0] == 'install' and calls[1]['connection_retry_seconds'] == 60
        assert calls[1]['job_ids'] == ['selected']
    else:
        with pytest.raises(SystemExit):
            entrypoint.main()
        assert calls == []
