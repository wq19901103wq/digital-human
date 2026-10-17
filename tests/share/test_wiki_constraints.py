"""Finite source corrections preserve original checks and resumable budgets."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from src.infrastructure.storage import read_json, write_json
from src.judge import corrected
from src.share.wiki import wiki_batch, wiki_cached, wiki_constraints as constraints
from src.share.wiki import wiki_continuation as continuation
from src.share.wiki import wiki_repair as repair, wiki_selection, wiki_structured
from src.share.wiki.wiki_library import digest
from test_wiki_continuation import correction_batch


CONFIG = dict(provider='codex_cli', model='synthetic-model', reasoning_effort='low',
              codex_cli_version='synthetic-version', timeout_seconds=360)


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('constraint regression must not make live model requests')
    monkeypatch.setattr(corrected.CodexJudgeClient, '__init__', forbidden)
    monkeypatch.setattr(corrected.CodexJudgeClient, '_run', forbidden)
    monkeypatch.setattr(wiki_batch.codex_transport, 'check_available', forbidden)


def section_task():
    meta = dict(account='person', self_account='self')
    old = [[1, 'synthetic old claim']]
    facts = [dict(id='F1', category='events', old_lines=[1])]
    claims = {1: dict(category='events', text='synthetic old claim')}
    payload = dict(subject=meta, section=repair.CATEGORIES['events'], old_wiki=old,
                   review_lines=[1], facts=facts)
    prompt = repair.SECTION_PROMPT + json.dumps(payload, ensure_ascii=False)
    return prompt, repair.SECTION_SCHEMA, (meta, old, facts, claims, 'events')


def change(action='retain', fact_ids=None):
    return dict(old_line=1, action=action, reason='synthetic reason',
                fact_ids=['F1'] if fact_ids is None else fact_ids)


def section_result():
    return dict(items=[], supported_changes=[change()], unsupported_changes=[])


def structured_task(*, anchored=True):
    anchors = [dict(ref='E0', account='person', anchor_lines=[10]),
               dict(ref='E1', account='self', anchor_lines=[11] if anchored else [])]
    payload = dict(subject=dict(account='person', self_account='self'),
                   source_contract=dict(person_refs=anchors))
    plan = constraints.wire_plan('task\n' + json.dumps(payload), wiki_structured.SCHEMA)
    value = {name: [] for name in plan['schema']['properties']}
    value['e0_entities'] = [dict(ref='E0', kind='person', account='person', label='Person', lines=[10])]
    if anchored:
        value['e1_entities'] = [dict(ref='E1', kind='person', account='self', label='Self', lines=[11])]
    return plan, value


class Client:
    def __init__(self, result=None, error=None):
        self.result = result or section_result()
        self.error = error
        self.calls = []

    def cache_identity(self):
        return 'constraint-test'

    def run(self, prompt, schema):
        self.calls.append((prompt, schema))
        if self.error is not None:
            raise self.error
        return json.dumps(self.result)


def granted(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'continued'
    write_json(source / 'manifest.json', dict(original=True))
    failure_path = source / 'jobs/selected/failures/old.json'
    write_json(failure_path, dict(result=section_result()))
    prompt, schema, _ = section_task()
    key = digest([Client().cache_identity(), prompt, schema])
    plan = constraints.wire_plan(prompt, schema)
    selected = dict(selected=dict(verified_sources={str(failure_path): dict(
        stamp=wiki_selection.file_stamp(failure_path))}, requests={key: dict(
        plan=plan['seal'], source=continuation.locator(failure_path))}))
    manifest = dict(jobs=[dict(id='selected')], continuation={})
    constraints._finalize(manifest, source, output, selected,
        source / 'grant.json', source / 'ledger.json')
    directory = output / 'jobs/selected/cache'
    write_json(directory.parent / 'manifest.json', manifest['jobs'][0])
    return directory, key, source, output


def receipts(tmp_path):
    source, knowledge = correction_batch(tmp_path)
    manifest = read_json(source / 'manifest.json')
    manifest.update(config=CONFIG, pipeline=wiki_batch.pipeline())
    write_json(source / 'manifest.json', manifest)
    job = next(item for item in manifest['jobs'] if item['id'] == 'rejected')
    directory = Path(job['output'])
    key = 'a' * 64
    for attempt in range(1, 4):
        path = directory / 'failures' / f'{key}-{attempt}.json'
        record = read_json(path)
        record['retry_policy'] = 'source_feedback_v2'
        write_json(path, record)
    sources = wiki_cached._sources(job, manifest, None)
    prompt, schema, _ = section_task()
    evidence = dict(kind='output_rejected', history_complete=True, policy_attempt=3,
        request_sha256=key, failure_file=f'failures/{key}-3.json', attempt=3)
    receipt = dict(schema='wiki_request_location_v2', source_job=job,
        source_manifest_sha256=digest(manifest), verified_sources=sources,
        provenance=constraints.provenance(),
        fingerprint=wiki_cached._fingerprint(directory, job, manifest, sources),
        result=dict(requests={key: dict(plan=constraints.wire_plan(prompt, schema)['seal'])},
                    failure_evidence=[evidence]))
    diagnostic = tmp_path / 'diagnostic'
    write_json(diagnostic / 'rejected.json', receipt)
    return source, knowledge, diagnostic, receipt


@pytest.mark.parametrize('action', ['retain', 'correct'])
def test_supported_changes_require_nonempty_evidence(action):
    prompt, schema, _ = section_task()
    plan = constraints.wire_plan(prompt, schema)
    raw = section_result()
    raw['supported_changes'] = [change(action, [])]
    with pytest.raises(ValueError, match='empty Wiki evidence'):
        constraints.normalize(raw, plan)
    raw['supported_changes'] = [change(action)]
    assert constraints.normalize(raw, plan)['changes'] == [change(action)]
    assert schema == repair.SECTION_SCHEMA


@pytest.mark.parametrize('collection,action', [('supported_changes', 'omit'),
    ('supported_changes', 'unresolved'), ('unsupported_changes', 'retain'),
    ('unsupported_changes', 'correct')])
def test_actions_cannot_cross_evidence_partitions(collection, action):
    prompt, schema, _ = section_task()
    plan = constraints.wire_plan(prompt, schema)
    raw = dict(items=[], supported_changes=[], unsupported_changes=[])
    raw[collection] = [change(action)]
    with pytest.raises(ValueError):
        constraints.normalize(raw, plan)


def test_unsupported_claims_can_remain_unresolved_without_evidence():
    prompt, schema, _ = section_task()
    raw = dict(items=[], supported_changes=[], unsupported_changes=[change('unresolved', [])])
    assert constraints.normalize(raw, constraints.wire_plan(prompt, schema))['changes'] == [change('unresolved', [])]


@pytest.mark.parametrize('field,value', [('ref', 'E2'), ('kind', 'group'),
    ('account', 'other'), ('lines', [10])])
def test_person_fields_are_bound_to_their_source_anchor(field, value):
    plan, raw = structured_task()
    raw['e1_entities'][0][field] = value
    with pytest.raises(ValueError):
        constraints.normalize(raw, plan)


@pytest.mark.parametrize('mutation', ['duplicate', 'other', 'unanchored'])
def test_person_partitions_reject_duplicates_disguised_and_unanchored_people(mutation):
    plan, raw = structured_task(anchored=mutation != 'unanchored')
    person = dict(ref='E1', kind='person', account='self', label='Self', lines=[11])
    if mutation == 'duplicate':
        raw['e1_entities'].append(person)
    elif mutation == 'other':
        raw['other_entities'].append(person)
    else:
        assert 'enum' not in plan['schema']['properties']['e1_entities']['items']['properties']['lines']['items']
        raw['e1_entities'].append(person)
    with pytest.raises(ValueError):
        constraints.normalize(raw, plan)


@pytest.mark.parametrize('anchored', [False, True])
def test_optional_people_and_canonical_schema_are_preserved(anchored):
    plan, raw = structured_task(anchored=anchored)
    normalized = constraints.normalize(raw, plan)
    repair.validate_result(normalized, wiki_structured.SCHEMA)
    assert normalized['entities'] == raw['e0_entities'] + raw['e1_entities']
    assert 'enum' not in plan['schema']['properties']['e0_entities']['items']['properties']['label']
    assert 'enum' not in wiki_structured.SCHEMA['properties']['entities']['items']['properties']['ref']
    assert constraints.wire_plan('unknown task', repair.SECTION_SCHEMA) is None


@pytest.mark.parametrize('mutation', ['unknown_fact', 'missing_coverage'])
def test_original_section_validator_still_rejects_invalid_evidence_and_coverage(tmp_path, mutation):
    directory, _, source, _ = granted(tmp_path)
    raw = section_result()
    if mutation == 'unknown_fact':
        raw['supported_changes'][0]['fact_ids'] = ['unknown']
    else:
        raw['supported_changes'] = []
    client = Client(raw)
    with pytest.raises(ValueError, match='unknown supporting fact|every assigned old Wiki line'):
        repair.generate_section(client, directory, *section_task()[2])
    assert len(client.calls) == 3
    assert read_json(source / 'ledger.json')['used'] == {'selected': 3}
    assert not list(directory.glob('*.json'))


def test_successful_canonical_cache_is_reused_without_spending_another_request(tmp_path):
    directory, key, source, _ = granted(tmp_path)
    client = Client()
    first = repair.generate_section(client, directory, *section_task()[2])
    assert first == dict(items=[], changes=[change()])
    assert read_json(directory / f'{key}.json')['result'] == first
    assert repair.generate_section(client, directory, *section_task()[2]) == first
    assert len(client.calls) == 1
    assert read_json(source / 'ledger.json')['used'] == {'selected': 1}


def test_transport_failure_counts_and_resume_cannot_reset_the_allowance(tmp_path):
    directory, _, source, _ = granted(tmp_path)
    client = Client(error=RuntimeError('synthetic transport failure'))
    for count in range(1, 4):
        with pytest.raises(RuntimeError, match='synthetic transport failure'):
            repair.generate_section(client, directory, *section_task()[2])
        assert read_json(source / 'ledger.json')['used'] == {'selected': count}
    with pytest.raises(ValueError, match='allowance exhausted'):
        repair.generate_section(client, directory, *section_task()[2])
    assert len(client.calls) == 3


def test_unselected_downstream_cache_miss_is_blocked_before_a_model_call(tmp_path):
    directory, _, source, _ = granted(tmp_path)
    client = Client()
    with pytest.raises(ValueError, match='outside selected constraint requests'):
        repair._cached_call(client, directory, 'new downstream task', repair.SECTION_SCHEMA, lambda raw: raw)
    assert client.calls == []
    assert read_json(source / 'ledger.json')['used'] == {'selected': 0}


@pytest.mark.parametrize('mutation', ['missing', 'grant', 'tampered', 'source', 'manifest', 'provenance'])
def test_stale_or_changed_allowance_fails_closed_before_request(tmp_path, monkeypatch, mutation):
    directory, _, source, _ = granted(tmp_path)
    if mutation == 'missing':
        (source / 'ledger.json').unlink()
    elif mutation in ('grant', 'tampered'):
        ledger = read_json(source / 'ledger.json')
        ledger['grant_sha256' if mutation == 'grant' else 'used'] = 'wrong' if mutation == 'grant' else {'selected': 1}
        write_json(source / 'ledger.json', ledger)
    elif mutation == 'source':
        write_json(source / 'jobs/selected/failures/old.json', dict(changed=True))
    elif mutation == 'manifest':
        write_json(source / 'manifest.json', dict(changed=True))
    else:
        monkeypatch.setattr(constraints, 'provenance', lambda: dict(changed=True))
    client = Client()
    with pytest.raises(ValueError):
        repair.generate_section(client, directory, *section_task()[2])
    assert client.calls == []


def test_concurrent_reservations_never_exceed_three(tmp_path):
    directory, key, source, _ = granted(tmp_path)
    policy = read_json(directory.parent / 'manifest.json')['correction']
    def reserve(index):
        try:
            return constraints._reserve(policy, directory, key)
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(12)))
    assert sorted(item for item in results if item is not None) == [1, 2, 3]
    assert read_json(source / 'ledger.json')['used'] == {'selected': 3}


def test_prepare_preserves_successes_and_publishes_only_the_final_contract(tmp_path):
    source, knowledge, diagnostic, _ = receipts(tmp_path)
    before = {path: path.read_bytes() for path in source.rglob('*.json')}
    output = tmp_path / 'continued'
    result = constraints.prepare(source, output, diagnostic, ['rejected'])
    assert result['selected_jobs'] == result['selected_requests'] == 1
    manifest = read_json(output / 'manifest.json')
    assert manifest['correction']['contract'] == constraints.CONTRACT
    selected = next(job for job in manifest['jobs'] if job['id'] == 'rejected')
    assert selected['correction']['contract'] == constraints.CONTRACT
    assert read_json(Path(selected['output']) / 'cache/success.json') == {'result': {'existing': True}}
    completed = next(job for job in manifest['jobs'] if job['id'] == 'page')
    assert completed['reuse'] == continuation.locator(knowledge)
    assert all(path.read_bytes() == data for path, data in before.items())
    with pytest.raises(ValueError, match='already prepared'):
        constraints.prepare(source, tmp_path / 'reset', diagnostic, ['rejected'])


def test_finalize_failure_never_publishes_a_runnable_ordinary_manifest(tmp_path, monkeypatch):
    source, _, diagnostic, _ = receipts(tmp_path)
    output = tmp_path / 'continued'
    def fail(*args):
        raise RuntimeError('synthetic prepublication failure')
    monkeypatch.setattr(constraints, '_finalize', fail)
    with pytest.raises(RuntimeError, match='prepublication'):
        constraints.prepare(source, output, diagnostic, ['rejected'])
    assert not (output / 'manifest.json').exists()
    assert not (output / 'progress.json').exists()


@pytest.mark.parametrize('mutation', ['receipt', 'provenance', 'input', 'failure', 'evidence'])
def test_prepare_rejects_stale_receipts_or_unproven_failure_records(tmp_path, mutation):
    source, _, diagnostic, receipt = receipts(tmp_path)
    if mutation == 'receipt':
        receipt['source_manifest_sha256'] = 'wrong'
    elif mutation == 'provenance':
        receipt['provenance'] = dict(changed=True)
    elif mutation == 'input':
        write_json(Path(next(iter(receipt['verified_sources']))), dict(changed=True))
    elif mutation == 'failure':
        path = Path(receipt['source_job']['output']) / receipt['result']['failure_evidence'][0]['failure_file']
        record = read_json(path)
        record['request_sha256'] = 'wrong'
        write_json(path, record)
        manifest = read_json(source / 'manifest.json')
        receipt['fingerprint'] = wiki_cached._fingerprint(Path(receipt['source_job']['output']),
            receipt['source_job'], manifest, receipt['verified_sources'])
    else:
        receipt['result']['failure_evidence'][0]['history_complete'] = False
    write_json(diagnostic / 'rejected.json', receipt)
    output = tmp_path / 'continued'
    with pytest.raises(ValueError):
        constraints.prepare(source, output, diagnostic, ['rejected'])
    assert not (output / 'manifest.json').exists()
    assert not (source / f'.correction-{constraints.CONTRACT}.json').exists()
