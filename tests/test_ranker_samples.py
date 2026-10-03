from copy import deepcopy
from pathlib import Path

import pytest

from src.config import ConfigError, sha256_file
from src.generator import ranker_samples
from src.generator.fewshot import RecallSession, open_session
from src.history.sources import load
from src.generator.ranker_samples import object_match, prepare, select_candidates, teacher_overlap
from src.iteration.storage import read_json, write_json
from tests.leakage_support import historical


def teacher_at(tmp_path, fixture, index):
    directory = tmp_path / 'teacher'
    directory.mkdir()
    row = fixture.cases[index]
    end = row['source_span']['end_timestamp']
    sources = directory / 'sources.json'
    audit = directory / 'source_audit.json'
    write_json(sources, {'train': [row]})
    write_json(audit, {'reference_spans': [], 'training_information_end': end})
    write_json(directory / 'learning.json', dict(verified=True, information_end=end,
        evidence_files={str(p): sha256_file(p) for p in (sources, audit)}))
    write_json(directory / 'config.json', {})
    return directory


def test_teacher_time_failure_is_not_claimed_answer_exposure(tmp_path):
    fixture = historical(tmp_path / 'history')
    teacher = teacher_at(tmp_path, fixture, 2)
    report, _ = teacher_overlap(load(fixture.directory), [fixture.cases[1]], teacher)
    assert report['summary']['time_blocked_targets'] == 1
    assert report['summary']['answer_in_teacher_training_targets'] == 0
    assert report['summary']['source_eligible_targets'] == 0


def test_teacher_exact_answer_overlap_and_changed_evidence(tmp_path):
    fixture = historical(tmp_path / 'history')
    teacher = teacher_at(tmp_path, fixture, 1)
    report, _ = teacher_overlap(load(fixture.directory), [fixture.cases[1]], teacher)
    assert report['summary']['same_teacher_training_case_targets'] == 1
    assert report['summary']['answer_in_teacher_training_targets'] == 1
    write_json(teacher / 'sources.json', {'train': []})
    with pytest.raises(ConfigError, match='标签器来源变化'):
        teacher_overlap(load(fixture.directory), [fixture.cases[1]], teacher)


def test_group_identity_and_whole_reply_budget(tmp_path):
    fixture = historical(tmp_path / 'history')
    case, row = fixture.cases[1], fixture.examples[0]
    assert object_match(case, row) == 'same'
    assert object_match({**case, 'chat_type': 'group'}, row) == 'unknown'
    retriever = open_session(fixture.directory / 'fewshot_pool.jsonl', index_policy='use_if_present')
    assert retriever.is_approved()
    record = select_candidates(case, {'recent3': [row], 'latest1': [row]}, retriever)
    assert len(record['candidates']) == 1
    assert record['candidates'][0]['example']['reply'] == row['reply']
    assert len(row['reply']) == 2
    assert record['split'] is None
    # A future row is rejected even if returned by a faulty retrieval caller.
    with pytest.raises(ConfigError, match='不合格'):
        select_candidates(case, {'recent3': [], 'latest1': [fixture.examples[2]]}, retriever)


def test_unbudgeted_example_cannot_be_truncated_into_candidate(tmp_path):
    fixture = historical(tmp_path / 'history')
    # Exercise the pure selector budget boundary separately from source checks:
    # 独立普通池（无 report/purposes）上打开的会话不做来源校验。
    row = deepcopy(fixture.examples[0])
    row['reply'] = ['x' * 3000, 'last bubble']
    row['context_messages'] = deepcopy(row['context_messages'])
    plain = tmp_path / 'plain'
    plain.mkdir()
    from src.bootstrap.history import write_rows
    write_rows(plain / 'fewshot_pool.jsonl', [row])
    retriever = open_session(plain / 'fewshot_pool.jsonl', index_policy='use_if_present')
    result = select_candidates(fixture.cases[1], {'recent3': [row], 'latest1': []}, retriever)
    assert not result['candidates']
    assert result['rejected'][0]['reason'] == 'complete_example_over_budget'


def test_blocked_inventory_is_resumable_and_unlabeled(tmp_path, monkeypatch):
    fixture = historical(tmp_path / 'history')
    teacher = teacher_at(tmp_path, fixture, 1)
    generator = tmp_path / 'generator'
    write_json(generator / 'config.json', dict(max_shots_per_case=3, shots_char_budget=2500))
    output = tmp_path / 'inventory'
    report = prepare(fixture.directory, teacher, generator, output, progress=lambda _: None)
    assert report['status'] == 'inventory_complete_labels_blocked'
    assert report['candidate_observations'] == 1
    assert report['labels'] == dict(z0=0, z1=0, unlabeled=1, distribution_available=False,
                                   mixed_targets=None, all_zero_targets=None, all_one_targets=None, directed_pairs=0)
    def no_retrieval(*a, **kw):
        raise AssertionError('completed inventory must resume without retrieval')
    monkeypatch.setattr(RecallSession, 'retrieve', no_retrieval)
    assert prepare(fixture.directory, teacher, generator, output, progress=lambda _: None) == read_json(output / 'report.json')
    path = next((output / 'targets').glob('*.json'))
    value = read_json(path)
    value['candidates'][0]['example']['reply'] = ['changed']
    write_json(path, value)
    with pytest.raises(ConfigError, match='已存候选清单变化'):
        prepare(fixture.directory, teacher, generator, output, progress=lambda _: None)


def test_changed_recipe_cannot_reuse_inventory(tmp_path):
    fixture = historical(tmp_path / 'history')
    teacher = teacher_at(tmp_path, fixture, 0)
    generator = tmp_path / 'generator'
    write_json(generator / 'config.json', dict(max_shots_per_case=3, shots_char_budget=2500))
    output = tmp_path / 'inventory'
    prepare(fixture.directory, teacher, generator, output, progress=lambda _: None)
    write_json(generator / 'config.json', dict(max_shots_per_case=3, shots_char_budget=2500, llm='different'))
    with pytest.raises(ConfigError, match='frozen artifact changed'):
        prepare(fixture.directory, teacher, generator, output, progress=lambda _: None)


def test_changed_rank_implementation_rejects_inventory_resume(tmp_path, monkeypatch):
    """排序实现（fewshot/rank.py）字节变化 → 实现清单失配 → 断点清单拒绝续跑。"""
    fixture = historical(tmp_path / 'history')
    teacher = teacher_at(tmp_path, fixture, 0)
    generator = tmp_path / 'generator'
    write_json(generator / 'config.json', dict(max_shots_per_case=3, shots_char_budget=2500))
    output = tmp_path / 'inventory'
    prepare(fixture.directory, teacher, generator, output, progress=lambda _: None)
    rank_path = Path(ranker_samples.__file__).parent / 'fewshot' / 'rank.py'
    original_sha256 = ranker_samples.sha256_file

    def changed(path):
        if Path(path).resolve() == rank_path.resolve():
            return '0' * 64
        return original_sha256(path)

    monkeypatch.setattr(ranker_samples, 'sha256_file', changed)
    with pytest.raises(ConfigError, match='frozen artifact changed'):
        prepare(fixture.directory, teacher, generator, output, progress=lambda _: None)
