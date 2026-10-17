"""Opt-in wire constraints preserve semantic checks, caches and finite budgets."""
from copy import deepcopy
import json

import pytest

from src.infrastructure.storage import read_json, write_json
from src.share.wiki import wiki_connection_retry as connection
from src.share.wiki import wiki_repair as repair
from src.share.wiki import wiki_revision as revision
from src.share.wiki import wiki_revision_batch as batch
from src.share.wiki import wiki_revision_claims as claims
from src.share.wiki import wiki_revision_output as output
from src.share.wiki import wiki_revision_roles as roles
from test_wiki_prompt import knowledge
from test_wiki_revision import KeepClient, decisions
from test_wiki_revision_batch import prepared_people
from test_wiki_revision_claims import RoleClient as ClaimsClient
from test_wiki_revision_roles import RoleClient as RolesClient


def to_wire(value):
    value = deepcopy(value)
    audits = {check['record_key']: check for check in value.pop('replacement_checks', [])}
    keyed = {}
    for decision in value['decisions']:
        record_id = decision.pop('record_id')
        for field, key in (('role_checks', 'role_key'), ('slot_checks', 'slot_key')):
            if field in decision:
                mapped = {}
                for check in decision[field]:
                    name = check.pop(key)
                    mapped[name] = check['assertions'] if field == 'slot_checks' else check
                decision[field] = mapped
        keyed[record_id] = decision
    value['decisions'] = keyed
    corrections = value['corrections']
    entities = corrections.pop('entities')
    corrections.pop('reviews')
    for field, refs in (('e0_entities', {'E0'}), ('e1_entities', {'E1'})):
        corrections[field] = [entity for entity in entities if entity['ref'] in refs]
    corrections['other_entities'] = [entity for entity in entities if entity['ref'] not in ('E0', 'E1')]
    for table in output.TABLES:
        for index, record in enumerate(corrections[table]):
            audit = audits[f'{table}/{index}']
            audit.pop('record_key')
            record['audit'] = audit
    return value


class WireClient(KeepClient):
    def __init__(self, base=None):
        super().__init__()
        self.base = base or KeepClient()
        self.schemas = []

    def run(self, prompt, schema):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if repair.RETRY_MARKER in prompt:
            prompt = prompt.split(repair.RETRY_MARKER, 1)[1]
        assert prompt.startswith(output.INSTRUCTION)
        return json.dumps(to_wire(json.loads(self.base.run(prompt[len(output.INSTRUCTION):], schema))))


@pytest.mark.parametrize('backend,client_type', [(revision, KeepClient), (roles, RolesClient),
                                               (claims, ClaimsClient)])
def test_wire_normalization_preserves_validation_and_both_caches(tmp_path, backend, client_type):
    data = knowledge(tmp_path)
    rows = revision.read_rows(data['coverage'], 'self')
    records = revision.records(data)
    expected = backend.extract(client_type(), tmp_path / 'original', data, records, rows)
    wire = WireClient(client_type())
    client = output.ContractClient(wire, {'contract': output.CONTRACT})
    actual = backend.extract(client, tmp_path / 'new', data, records, rows)
    assert actual == expected
    assert backend.extract(client, tmp_path / 'new', data, records, rows) == expected
    assert len(wire.prompts) == 1
    schema = wire.schemas[0]
    assert set(schema['properties']['decisions']['properties']) == {record['id'] for record in records}
    assert 'reviews' not in schema['properties']['corrections']['properties']
    before = {path: path.read_bytes() for path in (tmp_path / 'original').glob('*.json')}
    assert backend.extract(client, tmp_path / 'original', data, records, rows) == expected
    assert len(wire.prompts) == 1
    assert all(path.read_bytes() == content for path, content in before.items())


def wire_fixture(tmp_path, kind='person'):
    data = knowledge(tmp_path)
    rows = revision.read_rows(data['coverage'], 'self')
    payload = dict(subject=dict(data['subject'], kind=kind),
        records=[dict(record, record_id=record['id']) for record in revision.records(data)],
        messages=[repair._render(row) for row in rows])
    return data, rows, output.wire_schema(revision.SCHEMA, payload)


@pytest.mark.parametrize('mutation', ['group', 'wrong_account', 'duplicate', 'reserved_other'])
def test_anchored_entities_cannot_change_identity_or_partition(tmp_path, mutation):
    data, _, schema = wire_fixture(tmp_path, 'group' if mutation == 'group' else 'person')
    raw = to_wire(decisions(data))
    entity = dict(ref='E0', kind='person', label='人物', account='person', lines=[2])
    raw['corrections']['e0_entities'] = [entity]
    if mutation == 'wrong_account':
        entity['account'] = 'self'
    elif mutation == 'duplicate':
        raw['corrections']['e0_entities'].append(deepcopy(entity))
    elif mutation == 'reserved_other':
        raw['corrections']['e0_entities'] = []
        raw['corrections']['other_entities'] = [entity]
    with pytest.raises(ValueError):
        output.normalize(raw, schema)


def test_missing_replacement_invalid_but_unresolved_is_not_wrong(tmp_path):
    data, rows, schema = wire_fixture(tmp_path)
    record = revision.records(data)[0]
    for action in ('correct', 'unresolved'):
        value = output.normalize(to_wire(decisions(data, {record['id']: action})), schema)
        assert value['corrections']['reviews'][0]['disposition'] == 'unresolved'
        if action == 'correct':
            with pytest.raises(ValueError):
                revision.validate(value, revision.records(data), rows, data['subject'], data)
        else:
            revision.validate(value, revision.records(data), rows, data['subject'], data)


def test_new_request_preserves_old_failures_and_cannot_reset_its_budget(tmp_path):
    class Invalid(KeepClient):
        def run(self, prompt, schema):
            self.prompts.append(prompt)
            return '{}'
    data = knowledge(tmp_path)
    rows = revision.read_rows(data['coverage'], 'self')
    records = revision.records(data)
    client = Invalid()
    cache = tmp_path / 'cache'
    with pytest.raises(ValueError):
        revision.extract(client, cache, data, records, rows)
    old = {path: path.read_bytes() for path in (tmp_path / 'failures').glob('*.json')}
    assert len(client.prompts) == repair.RETRY_LIMIT
    wrapped = output.ContractClient(client, {'contract': output.CONTRACT})
    with pytest.raises(ValueError):
        revision.extract(wrapped, cache, data, records, rows)
    assert len(client.prompts) == 2 * repair.RETRY_LIMIT
    with pytest.raises(ValueError, match='unchanged request exhausted'):
        revision.extract(wrapped, cache, data, records, rows)
    assert len(client.prompts) == 2 * repair.RETRY_LIMIT
    assert all(path.read_bytes() == content for path, content in old.items())


@pytest.mark.parametrize('mutation', ['selection', 'config', 'source', 'pipeline', 'client'])
def test_activation_freezes_failed_selection_and_request_inputs(tmp_path, mutation):
    _, directory = prepared_people(tmp_path, count=2)
    manifest = read_json(directory / 'manifest.json')
    state = read_json(directory / 'progress.json')
    client = KeepClient()
    pipeline = batch.pipeline()
    with pytest.raises(ValueError, match='failed jobs only'):
        output.activate(directory, manifest, ['person-0'], state, pipeline, client)
    state['jobs']['person-0'] = dict(stage='incomplete')
    binding = output.activate(directory, manifest, ['person-0'], state, pipeline, client)
    state['jobs']['person-0']['stage'] = 'complete'
    assert output.activate(directory, manifest, ['person-0'], state, pipeline, client) == binding
    selected = ['person-0']
    if mutation == 'selection':
        selected.append('person-1')
    elif mutation == 'config':
        manifest['config'] = {'model': 'changed'}
    elif mutation == 'source':
        manifest['revision']['source']['sha256'] = 'changed'
    elif mutation == 'pipeline':
        pipeline['wiki_revision.py'] = 'changed'
    else:
        client.cache_identity = lambda: {'model': 'changed'}
    with pytest.raises(ValueError, match='frozen revision output contract changed'):
        output.activate(directory, manifest, selected, state, pipeline, client)


def test_contract_connection_retry_and_completed_reuse(tmp_path, monkeypatch):
    from src.judge.codex_transport import CodexTransportError
    _, directory = prepared_people(tmp_path, count=2)
    state = read_json(directory / 'progress.json')
    state['jobs']['person-0'] = dict(stage='incomplete')
    write_json(directory / 'progress.json', state)
    before = (directory / 'manifest.json').read_bytes()
    sleeps = []
    monkeypatch.setattr(connection.time, 'sleep', sleeps.append)

    class Disconnect(WireClient):
        disconnected = False

        def run(self, prompt, schema):
            if not self.disconnected:
                self.disconnected = True
                raise CodexTransportError(dict(signals=['stream_disconnected'],
                    terminal_signals=['stream_disconnected'], stderr_signals=[],
                    terminal_events=['turn.failed'], exit_code=1))
            return super().run(prompt, schema)

    client = Disconnect()
    options = dict(client=client, workers=1, attempts=1, job_ids=['person-0'],
                   connection_retry_seconds=60, output_contract=output.CONTRACT)
    result = batch.run(directory, **options)
    assert result['jobs']['person-0']['stage'] == 'complete'
    assert sleeps == [60]
    assert batch.run(directory, **options)['jobs']['person-0']['stage'] == 'complete'
    assert len(client.prompts) == 1
    assert (directory / 'manifest.json').read_bytes() == before
    assert result['jobs'].get('person-1', {}).get('stage') != 'complete'


@pytest.mark.parametrize('engine', ['legacy', 'roles_v1', 'roles_claims_v2'])
def test_output_migration_requires_complete_registered_snapshot(engine):
    current = batch.pipeline(engine)
    previous = batch.previous_output_pipeline(current)
    previous['wiki_revision_batch.py'] = batch.PRE_OUTPUT_RUNNER_SHA256
    assert batch.compatible_pipeline(previous, engine)
    assert not batch.compatible_pipeline(current | {
        'wiki_revision_batch.py': batch.PRE_OUTPUT_RUNNER_SHA256}, engine)
    for name in current:
        assert not batch.compatible_pipeline(previous | {name: 'changed'}, engine)
