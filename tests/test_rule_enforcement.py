import json

from scripts.check_rules import check, check_instance, inspect


def test_rule_catalog_and_write_boundaries():
    assert check() == []


def test_new_scripts_cannot_bypass_journal_or_pointer_writers():
    assert inspect('scripts/adhoc.py', "from x import save_pointers\nsave_pointers({})")
    assert inspect('scripts/adhoc.py', "p = root / 'cases.jsonl'\np.write_text('result')")
    assert inspect('scripts/adhoc.py', "p = root / 'cases.jsonl'\nwith p.open('a') as out:\n out.write('result')")
    assert not inspect('scripts/reader.py', "p = root / 'cases.jsonl'\nrows = p.read_text()")
    assert not inspect('scripts/producer.py', "with record_contract.writer(exp_dir) as append:\n append(row)")


def test_missing_rule_implementation_or_test_fails(tmp_path):
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/rules.json').write_text(json.dumps({'rules': [
        {'id': 'unimplemented', 'enforcement': 'code', 'code': ['missing.py::guard'], 'tests': []},
        {'id': 'vague-process', 'enforcement': 'sop'}]}))
    findings = check(tmp_path)
    assert any('regression' in s for s in findings)
    assert any('missing.py' in s for s in findings)
    assert any('SOP' in s for s in findings)


def test_unregistered_external_results_are_detected(tmp_path):
    path = tmp_path / 'judge_training/study/comparisons/a/cases.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('{"case_id":"a"}\n')
    assert len(check_instance(tmp_path)) == 1
    from src.config import sha256_file
    spec = tmp_path / 'experiments/imported/spec.json'
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps({'provenance': {'source_hashes': {
        str(path.relative_to(tmp_path)): sha256_file(path)}}}))
    assert check_instance(tmp_path) == []
    path.write_text('{"case_id":"changed"}\n')
    assert len(check_instance(tmp_path)) == 1
