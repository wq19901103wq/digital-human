"""Fixed-only data refresh preserves development evidence and adoption gates."""
import json
from collections import Counter

import pytest

from src.bootstrap import acceptance_refresh, history
from src.config import ConfigError
from src.iteration import branches, datasets, experiment, gates, learned_gen, versions
from src.iteration.storage import read_json, write_json
from gate_support import finish_with_cases
from leakage_support import isolated_legacy_provenance as _legacy_fixture
from test_branches import tree as _tree_fixture, pass_to_fixed, submit_gen
from test_data_boundaries import timeline

tree = _tree_fixture
isolated_legacy_provenance = _legacy_fixture
pytestmark = pytest.mark.usefixtures('isolated_legacy_provenance')


@pytest.fixture
def parent(tree, monkeypatch):
    messages = timeline(chats=100, sessions=20)
    rows = history.examples(messages)
    roles, protocol = history.plan(rows, total=10, train_total=10, group_weight=.5)
    ref = history.publish(messages, rows, roles, protocol, {})
    versions.save_pointers({**versions.load_pointers(), 'data': ref})
    settings = experiment.load_settings()
    settings['evaluation']['pool_min_total'] = 1000
    for stage in ('development', 'fixed_test'):
        settings['evaluation'][stage].update(total=10, group_ratio=.5)
    # These synthetic models have no learning archive. The real source/message
    # and fixed-refresh checks remain enabled; provenance has its own suite.
    monkeypatch.setattr(datasets, 'static_sources', lambda *a: {'promotion_eligible': True})
    return ref


def refreshed(parent):
    return acceptance_refresh.refresh(parent, seed=43, build=True)['data_ref']


def test_refresh_is_immutable_disjoint_and_preserves_other_roles(parent):
    before = versions.load_pointers()
    child = refreshed(parent)
    other = acceptance_refresh.refresh(parent, seed=44, build=True)['data_ref']
    assert versions.load_pointers() == before
    parent_dir, child_dir = map(versions.data_version_dir, (parent, child))
    assert datasets.fixed_refresh_compatible(parent, child)
    for name in acceptance_refresh.preserved_files(datasets.manifest(parent_dir)):
        assert (parent_dir / name).read_bytes() == (child_dir / name).read_bytes()
        assert not (child_dir / name).stat().st_mode & 0o222
    ids = []
    for ref in (parent, child, other):
        rows = [json.loads(line) for line in datasets.case_path(
            versions.data_version_dir(ref), 'fixed_test').read_text().splitlines()]
        assert len(rows) == 10
        assert sum(row['chat_type'] == 'group' for row in rows) == 5
        assert sum(row['familiarity'] == 'familiar' for row in rows) == 8
        ids.append({row['case_id'] for row in rows})
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]


def test_capacity_report_distinguishes_parent_from_other_reservations():
    rows = [dict(id=str(i), relationship='group', familiarity='familiar',
                 source_span={'chat_id': 'chat'}) for i in range(5)]
    protocol = dict(unseen_chat_ids=[], acceptance_total=3, familiar_weight=1,
                    group_weight=1, evaluation_chat_cap=2)
    report = acceptance_refresh.capacity_report(rows, protocol, {
        'before_exclusions': set(), 'after_parent': {'0'},
        'after_all_reserved': {'0', '1', '2', '3'}})
    group = report['familiar/group']
    assert group['required'] == 3
    assert group['before_exclusions'] == {'cases': 5, 'chats': 1, 'capacity': 2}
    assert group['after_parent'] == {'cases': 4, 'chats': 1, 'capacity': 2}
    assert group['after_all_reserved'] == {'cases': 1, 'chats': 1, 'capacity': 1}
    assert group['minimum_cap_before_exclusions'] == 3
    assert group['minimum_cap_after_parent'] == 3
    assert group['minimum_cap_after_all_reserved'] is None


def test_fixed_cap_override_preserves_original_protocol(parent):
    result = acceptance_refresh.refresh(parent, seed=43, fixed_chat_cap=2, build=True)
    old = datasets.manifest(versions.data_version_dir(parent))
    new = datasets.manifest(versions.data_version_dir(result['data_ref']))
    assert new['protocol'] == old['protocol']
    assert new['acceptance_refresh']['fixed_sampling'] == dict(previous_chat_cap=25, chat_cap=2)
    assert new['roles']['fixed_test']['chat_cap'] == 2
    assert result['max_cases_per_chat'] <= 2
    assert datasets.fixed_refresh_compatible(parent, result['data_ref'])


def test_fixed_cap_override_can_resolve_capacity_shortage(parent, monkeypatch):
    purpose = datasets.manifest(versions.data_version_dir(parent))
    heldout = set(purpose['protocol']['unseen_chat_ids'])
    rows = [json.loads(line) for line in (versions.data_version_dir(parent) /
            'fewshot_pool.jsonl').read_text().splitlines()]
    excluded, bindings = acceptance_refresh.reserved_batches()
    kept_chats = {}
    for row in rows:
        chat = row['source_span']['chat_id']
        key = (chat in heldout, row['relationship'])
        kept_chats.setdefault(key, chat)
        if chat != kept_chats[key]:
            excluded.add(str(row['id']))
    monkeypatch.setattr(acceptance_refresh, 'reserved_batches', lambda: (excluded, bindings))
    with pytest.raises(ConfigError, match='固定批次容量不足'):
        acceptance_refresh.plan(parent, seed=43, fixed_chat_cap=1)
    cases, proof = acceptance_refresh.plan(parent, seed=43, fixed_chat_cap=4)
    assert len(cases) == 10 and not {c['case_id'] for c in cases} & excluded
    assert max(Counter(c['source_span']['chat_id'] for c in cases).values()) == 4
    assert proof['fixed_sampling']['chat_cap'] == 4


def test_exhausted_refresh_reports_capacity_without_creating_version(parent, monkeypatch):
    directory = versions.data_version_dir(parent)
    rows = [json.loads(line) for line in (directory / 'fewshot_pool.jsonl').read_text().splitlines()]
    monkeypatch.setattr(acceptance_refresh, 'reserved_batches',
                        lambda: ({str(row['id']) for row in rows}, {}))
    before = set(versions.DATA_ROOT.iterdir()), versions.load_pointers()
    with pytest.raises(ConfigError, match='固定批次容量不足.*after_parent'):
        refreshed(parent)
    assert before == (set(versions.DATA_ROOT.iterdir()), versions.load_pointers())


@pytest.mark.parametrize('tamper', ['development', 'protocol', 'parent', 'fixed'])
def test_changed_material_cannot_carry_development(parent, tamper):
    child = refreshed(parent)
    directory = versions.data_version_dir(child)
    purpose = datasets.manifest(directory)
    if tamper in ('development', 'fixed'):
        path = datasets.case_path(directory, 'development' if tamper == 'development' else 'fixed_test')
        path.chmod(0o644)
        path.write_text(path.read_text() + '\n')
    else:
        if tamper == 'protocol':
            purpose['protocol']['development_start'] += 1
        else:
            purpose['acceptance_refresh']['parent_ref'] = 'd-other'
        path = directory / 'purposes.json'
        path.chmod(0o644)
        path.write_text(json.dumps(purpose))
    with pytest.raises(ConfigError):
        datasets.fixed_refresh_compatible(parent, child, required=True)


def accepted_branch(parent):
    proposal = submit_gen('refresh', {'llm': {'model': 'mA'}})
    branches.set_stage_limit('refresh', 'development')
    assert pass_to_fixed() == []
    return proposal, experiment.load_experiment('branch-refresh-r-0001-dev')


def test_accepted_development_goes_directly_to_fixed_and_adoption(parent):
    proposal, dev = accepted_branch(parent)
    original = {p.name: p.read_bytes() for p in dev.iterdir() if p.is_file()}
    child = refreshed(parent)
    versions.save_pointers({**versions.load_pointers(), 'data': child})
    branches.set_stage_limit('refresh', 'fixed_test')
    jobs = branches.advance('refresh')
    assert len(jobs) == 1 and jobs[0]['id'] == 'branch-refresh-r-0002-fixed'
    assert branches.advance('refresh') == jobs
    fixed = experiment.load_experiment(jobs[0]['id'])
    spec = experiment.spec_of(fixed)
    assert spec['candidate_ref'] == proposal['candidate_ref']
    assert spec['fixed_entry']['experiment_id'] == dev.name
    assert spec['fixed_entry']['development_data_ref'] == parent
    assert spec['data_ref'] == child
    gates.verify_fixed_receipt(spec)
    assert not (versions.PRIVATE / 'experiments/branch-refresh-r-0002-dev').exists()
    assert not (versions.PRIVATE / 'experiments/branch-refresh-r-0002-smoke').exists()
    finish_with_cases(fixed, 'adopt')
    assert branches.advance('refresh') == []
    assert versions.load_pointers()['production_gen'] == proposal['candidate_ref']
    assert {p.name: p.read_bytes() for p in dev.iterdir() if p.is_file()} == original


def test_production_change_prevents_fixed_shortcut(parent):
    proposal, _ = accepted_branch(parent)
    child = refreshed(parent)
    versions.save_pointers({**versions.load_pointers(), 'data': child,
                            'production_gen': proposal['candidate_ref']})
    branches.set_stage_limit('refresh', 'fixed_test')
    assert branches.advance('refresh') == []
    assert branches.round_of('refresh/r-0002')['phase'] == 'blocked'


def test_adoption_rechecks_original_development_evidence(parent):
    _, dev = accepted_branch(parent)
    child = refreshed(parent)
    versions.save_pointers({**versions.load_pointers(), 'data': child})
    branches.set_stage_limit('refresh', 'fixed_test')
    job = branches.advance('refresh')[0]
    spec = experiment.spec_of(experiment.load_experiment(job['id']))
    state = read_json(dev / 'state.json')
    state['reason'] = 'changed after fixed entry'
    write_json(dev / 'state.json', state)
    with pytest.raises(ConfigError, match='固定准入证据已变化'):
        gates.verify_fixed_receipt(spec)


def test_learned_resume_allows_fixed_only_child(parent, tmp_path):
    base = versions.load_pointers()['production_gen']
    cfg = versions.load_generator(base)['config']
    cfg['retriever'] = {'enabled': True, 'learned': learned_gen.POLICY}
    candidate = versions.create_generator_version(cfg, parent, source_dir=versions.generator_dir(base))
    args = dict(base=base, candidate=candidate, name='learned-refresh', data=parent)
    output = tmp_path / 'development'
    learned_gen.prepare(output, **args)
    assert pass_to_fixed() == []
    original = (output / 'manifest.json').read_bytes()
    child = refreshed(parent)
    versions.save_pointers({**versions.load_pointers(), 'data': child})
    assert learned_gen.prepare(tmp_path / 'fixed', **{**args, 'data': child},
                               stage='fixed_test') == candidate
    assert (output / 'manifest.json').read_bytes() == original
    with pytest.raises(ConfigError, match='Existing branch differs'):
        learned_gen.prepare(tmp_path / 'invalid', **{**args, 'data': child})
