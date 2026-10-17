"""Zero-request recovery keeps exact identities, original checks and budgets."""
import json
import os
from pathlib import Path

import pytest

from src import cache
from src.config import sha256_file
from src.infrastructure.storage import read_json, write_json
from src.judge import corrected
from src.share.wiki import wiki_batch as batch
from src.share.wiki import wiki_cached as cached
from src.share.wiki import wiki_repair as repair
from src.share.wiki import wiki_selection
from src.share.wiki import wiki_structured as structured
from src.share.wiki.wiki_library import digest
from test_wiki_batch import inputs
from test_wiki_repair import Client, row, source


CONFIG = dict(provider='codex_cli', model='synthetic-model', reasoning_effort='low',
              codex_cli_version='synthetic-version', timeout_seconds=360)
SCHEMA = dict(type='object', properties=dict(value=dict(type='string')),
              required=['value'], additionalProperties=False)


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('cached recovery must not construct or call a live client')

    monkeypatch.setattr(corrected.subprocess, 'run', forbidden)
    monkeypatch.setattr(corrected.CodexJudgeClient, '__init__', forbidden)
    monkeypatch.setattr(corrected.CodexJudgeClient, '_run', forbidden)
    monkeypatch.setattr(batch.codex_transport, 'check_available', forbidden)


def historical(output, prompt='task', result=None, attempt=1, config=None, schema=None):
    client = cached.CachedClient(config or CONFIG, output)
    key = digest([client.cache_identity(), prompt, schema or SCHEMA])
    path = output / 'failures' / f'{key}-{attempt}.json'
    write_json(path, dict(request_sha256=key, attempt=attempt, error='old rejection',
                          result=result, errors=[], policy_attempt=attempt,
                          retry_budget='format_only_v1'))
    return key, path


def checked(value):
    repair.validate_result(value, SCHEMA)
    if value['value'] != 'accepted':
        raise ValueError('synthetic source attribution rejected')


def validate_text(text):
    return checked(json.loads(text))


def prepared(tmp_path):
    _, _, output = inputs(tmp_path / 'instances')
    manifest = read_json(output / 'manifest.json')
    manifest['config'] = CONFIG
    write_json(output / 'manifest.json', manifest)
    return manifest['jobs'][0], manifest, output


def test_cached_client_keeps_original_identity_without_parent_constructor(tmp_path):
    original = object.__new__(corrected.CodexJudgeClient)
    original.config = CONFIG
    client = cached.CachedClient(CONFIG, tmp_path)
    assert client.cache_identity() == original.cache_identity()
    assert cached.CachedClient(dict(CONFIG, timeout_seconds=900), tmp_path).cache_identity() == original.cache_identity()
    with pytest.raises(ValueError, match='frozen codex_cli'):
        cached.CachedClient(dict(provider='test'), tmp_path)


def test_locator_preserves_exact_exhausted_history_and_never_calls_client(tmp_path):
    key, _ = historical(tmp_path, result=dict(value='rejected'), attempt=3)
    before = cached._files(tmp_path)
    client = cached.RequestLocator(CONFIG, tmp_path)
    assert repair._cached_call(client, tmp_path / 'cache', 'task', SCHEMA,
                               checked, cache_only=True) is None
    assert client.requests == {}
    with pytest.raises(ValueError, match='exhausted') as failure:
        repair._cached_call(client, tmp_path / 'cache', 'task', SCHEMA, checked)
    assert failure.value.request_sha256 == key
    assert client.requests == {key: dict(prompt_sha256=digest('task'), schema_sha256=digest(SCHEMA))}
    assert before == cached._files(tmp_path)


def test_locator_uses_latest_transport_failure_instead_of_older_timeout(tmp_path):
    key, first = historical(tmp_path, result=None)
    record = read_json(first)
    record['error'] = 'Command synthetic-client timed out after 900 seconds'
    write_json(first, record)
    error = 'Codex transport diagnostics: ' + json.dumps(dict(
        schema='codex_transport_diagnostic_v1', exit_code=1,
        terminal_events=[], signals=['stream_disconnected'],
        terminal_signals=['stream_disconnected'], stderr_signals=[]))
    write_json(tmp_path / 'failures' / f'{key}-2.json', record | dict(attempt=2, error=error))
    before = cached._files(tmp_path)
    client = cached.RequestLocator(CONFIG, tmp_path)
    with pytest.raises(ValueError) as failure:
        repair._cached_call(client, tmp_path / 'cache', 'task', SCHEMA, checked)
    assert str(failure.value) == error and failure.value.request_sha256 == key
    assert before == cached._files(tmp_path)


def test_request_location_replays_selected_jobs_and_reuses_sealed_receipt(tmp_path, monkeypatch):
    job, manifest, output = prepared(tmp_path)
    directory = Path(job['output'])
    key, _ = historical(directory, result=dict(value='rejected'), attempt=3)
    failure_path = directory / 'failures' / f'{key}-3.json'
    saved = read_json(failure_path)
    saved['error'] = 'synthetic source attribution rejected'
    write_json(failure_path, saved)
    before = cached._files(directory)
    calls = []

    def pipeline(wiki, messages, staged, account, self_account, config, **kwargs):
        calls.append(kwargs)
        assert not list((staged / 'content').glob('*'))
        try:
            repair._cached_call(kwargs['client'], staged / 'cache', 'task', SCHEMA, checked)
        except ValueError as exc:
            return dict(stage='incomplete', errors=[repair.failure_record(exc, stage='writing_final')])
        pytest.fail('locator must stop at the exact cache miss')

    monkeypatch.setattr(repair, 'repair', pipeline)
    result = cached.locate_requests(output, tmp_path / 'diagnostic', [job['id']], workers=8)
    located = result['jobs'][0]
    assert result['model_requests'] == located['model_requests'] == 0
    assert list(located['requests']) == [key]
    assert located['failure_evidence'][0]['request_sha256'] == key
    assert located['failure_evidence'][0]['history_complete'] is False
    assert calls[0]['workers'] == 8
    assert before == cached._files(directory)
    again = cached.locate_requests(output, tmp_path / 'diagnostic', [job['id']], workers=8)
    assert again['jobs'][0]['receipt_reused'] is True and len(calls) == 1
    manifest['messages']['sha256'] = 'changed'
    write_json(output / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='raw source changed'):
        cached.locate_requests(output, tmp_path / 'diagnostic', [job['id']])


def test_location_rejects_unknown_jobs_and_unregistered_pipeline(tmp_path):
    job, manifest, output = prepared(tmp_path)
    with pytest.raises(ValueError, match='unknown selected'):
        cached.locate_requests(output, tmp_path / 'diagnostic', ['unknown'])
    manifest['pipeline']['wiki_repair.py'] = 'unregistered'
    write_json(output / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='registered person batch'):
        cached.locate_requests(output, tmp_path / 'diagnostic', [job['id']])


@pytest.mark.parametrize('prompt', ['task', 'task' + repair.RETRY_MARKER + 'quoted source marker'])
@pytest.mark.parametrize('wrapped', [False, True])
def test_prompt_and_feedback_markers_preserve_exact_request_key(tmp_path, prompt, wrapped):
    key, _ = historical(tmp_path, prompt, dict(value='accepted'))
    request = prompt
    if wrapped:
        feedback = dict(error='source includes ' + repair.RETRY_MARKER + ' as ordinary text')
        request = cached.RETRY_PREFIX + json.dumps(feedback) + repair.RETRY_MARKER + prompt
    client = cached.CachedClient(CONFIG, tmp_path)
    with cache.validation(validate_text):
        assert json.loads(client.run(request, SCHEMA)) == dict(value='accepted')
    assert set(client.accepted) == {key}


@pytest.mark.parametrize('suffix', ['not JSON', '{} no marker', '[]' + repair.RETRY_MARKER + 'task'])
def test_only_an_exact_retry_wrapper_is_removed(suffix):
    prompt = cached.RETRY_PREFIX + suffix
    assert cached.original_prompt(prompt) == prompt


def test_latest_rejection_does_not_hide_older_valid_output(tmp_path):
    key, _ = historical(tmp_path, result=dict(value='accepted'))
    historical(tmp_path, result=dict(value='wrong attribution'), attempt=2)
    historical(tmp_path, result=dict(value=123), attempt=3)
    client = cached.CachedClient(CONFIG, tmp_path)
    with cache.validation(validate_text):
        assert json.loads(client.run('task', SCHEMA)) == dict(value='accepted')
    assert client.accepted[key]['attempt'] == 1 and client.checked == 3


@pytest.mark.parametrize('change', ['model', 'effort', 'prompt', 'schema'])
def test_request_identity_changes_cannot_reuse_history(tmp_path, change):
    historical(tmp_path, result=dict(value='accepted'))
    config, prompt, schema = dict(CONFIG), 'task', dict(SCHEMA)
    if change == 'model':
        config['model'] = 'different'
    elif change == 'effort':
        config['reasoning_effort'] = 'high'
    elif change == 'prompt':
        prompt += ' changed'
    else:
        schema['description'] = 'changed'
    client = cached.CachedClient(config, tmp_path)
    with cache.validation(validate_text), pytest.raises(cached.CachedOutputMissing):
        client.run(prompt, schema)
    assert not client.accepted


@pytest.mark.parametrize('mutation', ['hash', 'attempt', 'filename', 'empty'])
def test_mismatched_history_record_is_rejected(tmp_path, mutation):
    _, path = historical(tmp_path, result=dict(value='accepted'))
    value = read_json(path)
    if mutation == 'hash':
        value['request_sha256'] = 'wrong'
    elif mutation == 'attempt':
        value['attempt'] = 7
    elif mutation == 'filename':
        path.rename(path.with_name('wrong-1.json'))
    else:
        value['result'] = None
    if mutation != 'filename':
        write_json(path, value)
    with cache.validation(validate_text), pytest.raises(cached.CachedOutputMissing):
        cached.CachedClient(CONFIG, tmp_path).run('task', SCHEMA)


def test_validator_is_required_and_context_is_restored(tmp_path):
    historical(tmp_path, result=dict(value='accepted'))
    client = cached.CachedClient(CONFIG, tmp_path)
    assert cache.response_valid('nonempty')
    assert not cache.response_valid('nonempty', require_validator=True)
    with pytest.raises(cached.CachedOutputMissing):
        client.run('task', SCHEMA)
    with cache.validation(lambda text: False), pytest.raises(cached.CachedOutputMissing):
        client.run('task', SCHEMA)
    with cache.validation(validate_text):
        assert json.loads(client.run('task', SCHEMA)) == dict(value='accepted')
    assert not cache.response_valid('nonempty', require_validator=True)


@pytest.mark.parametrize('complete', [False, True])
def test_pipeline_alone_promotes_checked_cache_and_complete_body(tmp_path, monkeypatch, complete):
    job, manifest, _ = prepared(tmp_path)
    output = Path(job['output'])
    write_json(output / 'manifest.json', dict(frozen='original'))
    write_json(output / 'progress.json', dict(stage='incomplete', errors=['prior error']))
    key, _ = historical(output, result=dict(value='accepted'))
    historical(output, result=dict(value='wrong attribution'), attempt=repair.RETRY_LIMIT)
    frozen = cached._files(output)

    def pipeline(wiki, messages, staged, *args, client, **kwargs):
        assert read_json(staged / 'manifest.json') == dict(frozen='original')
        assert not (staged / 'failures').exists()
        assert repair._cached_call(client, staged / 'cache', 'task', SCHEMA, checked) == dict(value='accepted')
        write_json(staged / 'content/knowledge.json', dict(synthetic='checked body'))
        write_json(staged / 'intermediate/facts.json', ['checked'])
        return dict(stage='complete' if complete else 'incomplete', errors=[])

    monkeypatch.setattr(repair, 'repair', pipeline)
    result = cached.recover(job, manifest, workers=8)
    assert result['stage'] == ('complete' if complete else 'incomplete')
    assert result['model_requests'] == 0 and result['recovered_requests'] == 1
    saved = read_json(output / 'cache' / f'{key}.json')
    assert saved['attempts'] == repair.RETRY_LIMIT and saved['recovered_from']['attempt'] == 1
    for name, checksum in frozen.items():
        if name != 'progress.json':
            assert sha256_file(output / name) == checksum
    if complete:
        assert read_json(output / 'content/knowledge.json') == dict(synthetic='checked body')
        assert read_json(output / 'progress.json')['stage'] == 'complete'
    else:
        assert not (output / 'content').exists() and not (output / 'intermediate').exists()
        assert sha256_file(output / 'progress.json') == frozen['progress.json']
    monkeypatch.setattr(repair, 'repair', lambda *args, **kwargs: pytest.fail('unchanged receipt must be reused'))
    assert cached.recover(job, manifest)['receipt_reused']


def test_partial_pipeline_cannot_promote_output_not_saved_by_original_checks(tmp_path, monkeypatch):
    job, manifest, _ = prepared(tmp_path)
    output = Path(job['output'])
    key, _ = historical(output, result=dict(value='accepted'))

    def pipeline(wiki, messages, staged, *args, client, **kwargs):
        with cache.validation(validate_text):
            client.run('task', SCHEMA)
        write_json(staged / 'cache' / f'{key}.json', dict(request_sha256=key, result=dict(value='modified')))
        return dict(stage='incomplete', errors=[])

    monkeypatch.setattr(repair, 'repair', pipeline)
    assert cached.recover(job, manifest)['recovered_requests'] == 0
    assert not (output / 'cache').exists()


def test_pipeline_completion_without_knowledge_is_rejected(tmp_path, monkeypatch):
    job, manifest, _ = prepared(tmp_path)
    monkeypatch.setattr(repair, 'repair', lambda *args, **kwargs: dict(stage='complete'))
    with pytest.raises(ValueError, match='without readable knowledge'):
        cached.recover(job, manifest)
    assert not (Path(job['output']) / 'cached_recovery.json').exists()


def test_real_pipeline_cache_miss_preserves_all_original_progress_and_failures(tmp_path):
    job, manifest, _ = prepared(tmp_path)
    output = Path(job['output'])
    write_json(output / 'progress.json', dict(stage='incomplete', errors=['original diagnostic']))
    historical(output, result=dict(value='unrelated old output'), attempt=repair.RETRY_LIMIT)
    before = cached._files(output)
    result = cached.recover(job, manifest)
    assert result['stage'] == 'incomplete' and result['missing_requests']
    assert result['model_requests'] == 0 and result['recovered_requests'] == 0
    assert cached._files(output) == before
    assert not (output / 'content/knowledge.json').exists()
    assert cached.recover(job, manifest)['receipt_reused']


@pytest.mark.parametrize('contract', ['legacy_v1', 'bounded_refs_v2'])
def test_real_complete_pipeline_recovers_exact_historical_outputs(tmp_path, contract):
    library = tmp_path / 'wiki' / 'users'
    library.mkdir(parents=True)
    wiki = library / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    messages = source(tmp_path, [row('private:self:person', 'person', 1000, 'RAW_PRIVATE_TEXT')])
    output = tmp_path / 'batch'
    batch.prepare(library.parent, messages, output, 'self', CONFIG)
    manifest = read_json(output / 'manifest.json')
    manifest.update(reference_contract=contract, reread_contract='reread_evidence_v2',
                    structured_contract='structured_source_v2')
    job = manifest['jobs'][0]
    archive = tmp_path / 'archive'

    class HistoricalClient(Client):
        def cache_identity(self):
            return cached.CachedClient(CONFIG, archive).cache_identity()

        def run(self, prompt, schema):
            prompt = cached.original_prompt(prompt).removeprefix(repair.REFERENCE_PROMPT)
            prefix = structured.source_prompt('structured_source_v2')
            if not prompt.startswith(prefix):
                return super().run(prompt, schema)
            payload = json.loads(prompt[len(prefix):])
            line = payload['messages'][0][0]
            return json.dumps(dict(entities=[dict(ref='E0', kind='person', label='示例人物',
                account='person', lines=[line])], attributes=[dict(subject_ref='E0', field='occupation',
                value='工程师', time_scope='历史聊天时', valid_from='', valid_to='', temporal_kind='mutable',
                mode='self_report', polarity='affirmed', lines=[line], fact_ids=['F1'],
                attribution='该账号自述其职业。')], relations=[], events=[], addresses=[],
                reviews=[dict(fact_id='F1', disposition='represented', reason='依据原始记录拆出单项属性。')]))

    progress = repair.repair(wiki, messages, archive, 'person', 'self', CONFIG,
        client=HistoricalClient(), structured=True, include_mentions=True, reconcile_attributes=True,
        reread_contract=manifest['reread_contract'], structured_contract=manifest['structured_contract'],
        reference_contract=contract)
    assert progress['stage'] == 'complete'
    destination = Path(job['output'])
    write_json(destination / 'manifest.json', read_json(archive / 'manifest.json'))
    expected = read_json(archive / 'content/knowledge.json')
    requests = list((archive / 'cache').glob('*.json'))
    for path in requests:
        saved = read_json(path)
        write_json(destination / 'failures' / f'{path.stem}-1.json',
            dict(saved, attempt=1, error='historical interrupted attempt'))
        write_json(destination / 'failures' / f'{path.stem}-{repair.RETRY_LIMIT}.json',
            dict(saved, attempt=repair.RETRY_LIMIT, result={}, error='later rejected output'))
    before = cached._files(destination)
    result = cached.recover(job, manifest, workers=4)
    assert result['stage'] == 'complete' and result['model_requests'] == 0
    assert result['recovered_requests'] == len(requests) and not result['missing_requests']
    assert read_json(destination / 'content/knowledge.json') == expected
    assert expected['attributes'][0]['value'] == '工程师'
    for name, checksum in before.items():
        assert sha256_file(destination / name) == checksum
    for path in requests:
        assert read_json(destination / 'cache' / path.name)['attempts'] == repair.RETRY_LIMIT
    assert cached.recover(job, manifest)['receipt_reused']


@pytest.mark.parametrize('change', ['source_stamp', 'cache', 'failure'])
def test_recovery_receipt_invalidates_when_inputs_change(tmp_path, monkeypatch, change):
    job, manifest, _ = prepared(tmp_path)
    calls = []
    monkeypatch.setattr(repair, 'repair', lambda *args, **kwargs:
                        calls.append(True) or dict(stage='incomplete', errors=[]))
    cached.recover(job, manifest)
    if change == 'source_stamp':
        path = Path(manifest['messages']['path'])
        stamp = path.stat()
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    else:
        directory = 'failures' if change == 'failure' else change
        write_json(Path(job['output']) / directory / 'new.json', dict(request_sha256='new', result={}))
    result = cached.recover(job, manifest)
    assert not result['receipt_reused'] and len(calls) == 2


@pytest.mark.parametrize('which', ['messages', 'wiki'])
def test_changed_source_rejects_even_an_existing_receipt(tmp_path, monkeypatch, which):
    job, manifest, _ = prepared(tmp_path)
    monkeypatch.setattr(repair, 'repair', lambda *args, **kwargs: dict(stage='incomplete', errors=[]))
    cached.recover(job, manifest)
    locator = manifest['messages'] if which == 'messages' else job['wiki']
    Path(locator['path']).write_text('changed input')
    with pytest.raises(ValueError, match='source changed|Wiki changed'):
        cached.recover(job, manifest)


def test_batch_verified_source_is_not_rehashed_per_object(tmp_path, monkeypatch):
    job, manifest, _ = prepared(tmp_path)
    source_path = Path(manifest['messages']['path'])
    stamp = wiki_selection.file_stamp(source_path)
    original = wiki_selection.file_digest

    def digest_small_source(path):
        assert Path(path).resolve() != source_path.resolve()
        return original(path)

    monkeypatch.setattr(wiki_selection, 'file_digest', digest_small_source)
    monkeypatch.setattr(repair, 'repair', lambda *args, **kwargs: dict(stage='incomplete', errors=[]))
    assert cached.recover(job, manifest, verified_source_stamp=stamp)['model_requests'] == 0
    source_path.write_text(source_path.read_text() + '\n')
    with pytest.raises(ValueError, match='after batch verification'):
        cached.recover(job, manifest, verified_source_stamp=stamp)


def test_source_change_during_recovery_cannot_publish_partial_or_complete_output(tmp_path, monkeypatch):
    job, manifest, _ = prepared(tmp_path)

    def pipeline(wiki, messages, staged, *args, **kwargs):
        Path(messages).write_text('changed mid-run')
        write_json(staged / 'content/knowledge.json', {})
        return dict(stage='complete')

    monkeypatch.setattr(repair, 'repair', pipeline)
    with pytest.raises(ValueError, match='source changed during cached recovery'):
        cached.recover(job, manifest)
    assert not (Path(job['output']) / 'content').exists()


@pytest.mark.parametrize('exception', [False, True])
def test_batch_miss_preserves_prior_diagnostic_attempts_and_blocker(tmp_path, monkeypatch, exception):
    job, manifest, output = prepared(tmp_path)
    prior = dict(stage='incomplete', error='original failure evidence', attempts=7,
                 account=job['account'], diagnostic='unchanged')
    write_json(output / 'progress.json', dict(jobs={job['id']: prior}, blocker=dict(kind='service')))
    recovered = dict(stage='incomplete', model_requests=0, missing_requests=['key'])

    def recover(job, saved, **kwargs):
        assert saved == manifest and kwargs['verified_source_stamp'] == wiki_selection.file_stamp(manifest['messages']['path'])
        if exception:
            raise ValueError('recovery source check failed')
        return recovered

    monkeypatch.setattr(cached, 'recover', recover)
    monkeypatch.setattr(batch.wiki_prompt, 'export', lambda *args: pytest.fail('incomplete cannot export'))
    result = batch.run(output, cached_only=True, object_workers=4, workers=8, max_jobs=1)
    saved = result['jobs'][job['id']]
    assert {key: saved[key] for key in prior} == prior
    assert saved['cached_recovery']['stage'] == 'incomplete'
    assert result['blocker'] == dict(kind='service') and result['counts']['incomplete'] == 1


def test_batch_only_completed_recovery_adds_exported_body(tmp_path, monkeypatch):
    job, _, output = prepared(tmp_path)
    exported = []

    def recover(job, manifest, **kwargs):
        write_json(Path(job['output']) / 'content/knowledge.json', dict(synthetic='body'))
        return dict(stage='complete', model_requests=0)

    monkeypatch.setattr(cached, 'recover', recover)
    monkeypatch.setattr(batch.wiki_prompt, 'export', lambda knowledge, target:
                        exported.append(knowledge) or dict(records=3, packets=1))
    result = batch.run(output, cached_only=True, object_workers=4, workers=8, max_jobs=1)
    assert result['counts']['complete'] == 1 and len(exported) == 1
    assert result['jobs'][job['id']]['attempts'] == 0


@pytest.mark.parametrize('option', [[], ['--request-workers', '8'], ['--request-timeout-seconds', '900'],
    ['--profile-requests'], ['--skip-incomplete'], ['--retry-transport-only']])
def test_cli_cached_only_never_installs_transport_and_rejects_live_options(tmp_path, monkeypatch, option):
    from scripts import batch_wiki as entrypoint

    _, _, output = prepared(tmp_path)
    monkeypatch.setattr(entrypoint, 'ROOT', tmp_path)
    monkeypatch.setattr(entrypoint, 'load_settings', lambda: None)
    monkeypatch.setattr(entrypoint, 'install_transport', lambda *args: pytest.fail('must not install transport'))
    calls = []
    monkeypatch.setattr(batch, 'run', lambda output, **kwargs:
                        calls.append(kwargs) or dict(stage='incomplete'))
    monkeypatch.setattr(entrypoint.sys, 'argv', ['batch_wiki', 'run', '--output', str(output), '--cached-only', *option])
    if option:
        with pytest.raises(SystemExit) as error:
            entrypoint.main()
        assert error.value.code == 2 and not calls
    else:
        assert entrypoint.main() == 1 and calls[0]['cached_only']
