"""分支端到端状态流和真实进程竞争；模型返回用可控结果，不调用外部服务。"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.config import ConfigError, load_settings
from src.dashboard import report
from src.iteration import branch_packs, branches, experiment, promote, runner, scheduler, versions
from src.iteration.progress import track_run
from src.iteration.storage import LockBusy, file_lock, write_json
from gate_support import finish_with_cases
from leakage_support import isolated_legacy_provenance as _legacy_fixture

isolated_legacy_provenance = _legacy_fixture


pytestmark = pytest.mark.usefixtures('isolated_legacy_provenance')


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "instances" / "demo"
    for attr, value in {"PRIVATE": root, "DATA_ROOT": root / "data", "GEN_ROOT": root / "generators",
                        "JUDGE_ROOT": root / "judges", "POINTERS_PATH": root / "pointers.json"}.items():
        monkeypatch.setattr(versions, attr, value)
    data = root / "data" / "d-0001"
    data.mkdir(parents=True)
    write_json(data / "manifest.json", {"id": "d-0001"})
    (data / "persona.md").write_text("测试人格")
    (data / "scenarios").mkdir()
    (data / "scenarios" / "friend.md").write_text("测试场景")
    cases = [{"case_id": str(i), "chat_type": "group" if i % 2 else "private",
              "context": [{"sender": "甲", "text": "在吗"}], "human_reply": ["在"]} for i in range(6)]
    for file in ("dev_pool.jsonl", "fixed_test.jsonl"):
        (data / file).write_text("\n".join(json.dumps(case) for case in cases))
    write_json(data / "report.json", {"total": 9000})
    (data / "fewshot_pool.jsonl").write_text("")
    gid = versions.create_generator_version({"llm": {"model": "m0", "temperature": 0.5},
             "retriever": {"enabled": False}, "max_shots_per_case": 3, "shots_char_budget": 100}, "d-0001")
    jid = versions.create_judge_version({"mode": "pairwise_llm", "llm": {"model": "j0"}}, {},
                                        assets={"prompt.md": b"judge prompt"})
    versions.save_pointers({"data": "d-0001", "production_gen": gid, "iteration_gen": gid,
                            "production_judge": jid, "iteration_judge": jid})
    for kind in ("calibration", "validation"):
        ref = f"pack-{kind}-test"
        write_json(root / "judge_eval" / ref / "pack.json", {"pack_id": ref, "data_ref": "d-0001",
                   "c0_gen_version": gid, "rows": [{**case, "ai_replies": ["AI"]} for case in cases]})
    settings = load_settings()
    for ds in ("development", "fixed_test"):
        settings["evaluation"][ds].update(total=6, group_ratio=0.5)
    monkeypatch.setattr(experiment, "load_settings", lambda: settings)
    monkeypatch.setattr(branches, "load_settings", lambda: settings)
    return root


def submit_gen(name, overrides):
    return branches.submit(name, "gen", "单一改动", overrides=overrides)


def submit_judge(name, overrides):
    return branches.submit(name, "judge", "单一改动", overrides=overrides,
                           development_pack="pack-calibration-test", validation_pack="pack-validation-test")


@pytest.mark.parametrize('kind', [experiment.KIND_GEN_AB, experiment.KIND_JUDGE_EVAL])
def test_worker_forwards_case_concurrency_to_both_runners(monkeypatch, kind):
    from types import SimpleNamespace
    from scripts import iterate_branches
    directory = Path('/unused/experiments/worker-test')
    calls = []
    monkeypatch.setattr(experiment, 'load_experiment', lambda _: directory)
    monkeypatch.setattr(experiment, 'spec_of', lambda _: {'kind': kind})
    name = 'run_gen_experiment' if kind == experiment.KIND_GEN_AB else 'run_judge_experiment'
    monkeypatch.setattr(runner, name, lambda path, **kwargs: calls.append((path, kwargs)))
    iterate_branches.work(SimpleNamespace(kind='experiment', job='worker-test', workers=4))
    assert calls == [(directory, {'workers': 4})]


def test_parallel_branches_receive_disjoint_sealed_batches(tree, monkeypatch):
    from src.iteration import acceptance
    (tree / 'judge_eval/pack-validation-test/pack.json').unlink()
    settings = experiment.load_settings()
    settings['evaluation']['fixed_test'].update(total=2, group_ratio=.5)
    monkeypatch.setattr(experiment, 'load_settings', lambda: settings)
    monkeypatch.setattr(branches, 'load_settings', lambda: settings)
    acceptance.seal('d-0001', 2)
    submit_gen('sealed-a', {'llm': {'model': 'mA'}})
    submit_gen('sealed-b', {'llm': {'model': 'mB'}})
    jobs = pass_to_fixed()
    assert len(jobs) == 2
    specs = [experiment.spec_of(experiment.load_experiment(job['id'])) for job in jobs]
    assert len({spec['acceptance']['batch_id'] for spec in specs}) == 2
    from src.iteration import datasets
    left, right = [datasets.rows_for(spec) for spec in specs]
    assert len(left) == len(right) == 2
    assert {r['case_id'] for r in left}.isdisjoint(r['case_id'] for r in right)
    assert acceptance.status('d-0001')['remaining'] == 1


def test_judge_branch_builds_validation_only_after_development_entry(tree, monkeypatch):
    from src.iteration import acceptance
    (tree / 'judge_eval/pack-validation-test/pack.json').unlink()
    acceptance.seal('d-0001', 2)
    branches.submit('sealed-judge', 'judge', 'human-selected model change',
                    overrides={'llm': {'model': 'j1'}}, development_pack='pack-calibration-test')
    jobs = branches.advance()
    assert acceptance.status('d-0001')['consumed'] == 0
    for job in jobs:
        finish(job, 'observe')
    for job in branches.advance():
        finish(job)
    jobs = branches.advance()
    assert jobs and jobs[0]['kind'] == 'pack'
    recipe = json.loads((tree / 'judge_eval' / jobs[0]['id'] / 'recipe.json').read_text())
    assert recipe['dataset'] == 'fixed_test'
    assert len(acceptance.rows(recipe['acceptance'])) == 2
    assert acceptance.status('d-0001')['consumed'] == 1
    monkeypatch.setattr(branches, '_entry', lambda *a, **kw: pytest.fail('repeated pack admission'))
    assert branches.advance() == jobs
    assert branches.advance() == jobs
    assert acceptance.status('d-0001')['consumed'] == 1


def finish(job, verdict="merge_to_iteration_baseline", failures=0):
    directory = experiment.load_experiment(job["id"])
    finish_with_cases(directory, verdict, failures=failures)


def pass_to_fixed():
    smoke = branches.advance()
    for job in smoke:
        assert experiment.spec_of(experiment.load_experiment(job["id"]))["smoke"]
        finish(job, "observe")
    dev = branches.advance()
    for job in dev:
        finish(job)
    return branches.advance()


def test_two_branches_roll_out_then_rebase_and_revalidate(tree):
    original = versions.load_pointers()
    submit_gen("a", {"llm": {"model": "mA"}})
    submit_gen("b", {"shots_char_budget": 200})
    fixed = pass_to_fixed()
    assert len(fixed) == 2
    assert versions.load_pointers() == original  # 开发通过不争抢共享指针
    old_b = experiment.load_experiment(next(j["id"] for j in fixed if "-b-" in j["id"]))
    old_spec = (old_b / "spec.json").read_bytes()
    for job in fixed:
        finish(job, "adopt")
    next_jobs = branches.advance()
    a = branches.round_of("a/r-0001")
    assert a["phase"] == "promoted"
    assert branches.round_of("b/r-0001")["phase"] == "superseded"
    b = branches.round_of("b/r-0002")
    assert b["baseline_ref"] == a["candidate_ref"]
    candidate = versions.load_generator(b["candidate_ref"])["config"]
    assert candidate["llm"]["model"] == "mA"
    assert candidate["shots_char_budget"] == 200
    assert (old_b / "spec.json").read_bytes() == old_spec
    assert len(next_jobs) == 1 and "smoke" in next_jobs[0]["id"]
    finish(next_jobs[0], "observe")
    for job in branches.advance():
        finish(job)
    for job in branches.advance():
        finish(job, "adopt")
    assert branches.advance() == []
    assert versions.load_pointers()["production_gen"] == b["candidate_ref"]
    assert branches.round_of("b/r-0002")["phase"] == "promoted"


@pytest.mark.parametrize('kind', ['gen', 'judge'])
def test_fixed_polling_does_not_repeat_source_or_entry_audit(tree, monkeypatch, kind):
    from src.iteration import learning_guard
    submit = submit_gen if kind == 'gen' else submit_judge
    submit('polling', {'llm': {'model': 'candidate'}})
    jobs = pass_to_fixed()
    assert len(jobs) == 1 and jobs[0]['id'].endswith('-fixed')
    directory = experiment.load_experiment(jobs[0]['id'])
    before = (directory / 'spec.json').read_bytes()
    pointers = versions.load_pointers()
    monkeypatch.setattr(learning_guard, 'verify', lambda *a: pytest.fail('poll reconstructed sources'))
    monkeypatch.setattr(branches, '_entry', lambda *a, **kw: pytest.fail('poll repeated admission'))
    for _ in range(3):
        assert branches.advance() == jobs
    assert (directory / 'spec.json').read_bytes() == before
    assert versions.load_pointers() == pointers


def test_same_field_conflict_does_not_overwrite_winner(tree):
    submit_gen("a", {"llm": {"model": "mA"}})
    submit_gen("b", {"llm": {"model": "mB"}})
    fixed = pass_to_fixed()
    for job in fixed:
        finish(job, "adopt")
    branches.advance()
    b = branches.round_of("b/r-0002")
    assert b["phase"] == "conflict" and "config.llm.model" in b["reason"]
    assert versions.current_generator("production")["config"]["llm"]["model"] == "mA"


def test_stale_direct_promotion_and_smoke_are_rejected(tree):
    submit_gen("a", {"llm": {"model": "mA"}})
    submit_gen("b", {"shots_char_budget": 200})
    smoke = branches.advance()
    finish(smoke[0])
    with pytest.raises(ConfigError, match="冒烟"):
        promote.promote_gen(smoke[0]["id"])
    finish(smoke[1])
    for job in branches.advance():
        finish(job)
    fixed = branches.advance()
    for job in fixed:
        finish(job, "adopt")
    promote.promote_gen(fixed[0]["id"])
    with pytest.raises(ConfigError, match="过期"):
        promote.promote_gen(fixed[1]["id"])


def test_failed_development_never_gets_fixed_or_promoted(tree):
    submit_gen("a", {"llm": {"model": "mA"}})
    smoke = branches.advance()[0]
    finish(smoke, "observe")
    dev = branches.advance()[0]
    finish(dev, "observe")
    assert branches.advance() == []
    assert branches.round_of("a/r-0001")["phase"] == "rejected"
    assert versions.load_pointers()["production_gen"] == "g-0001"


def test_changed_revision_stays_isolated_and_pause_blocks_promotion(tree):
    submit_gen("a", {"llm": {"model": "mA"}})
    fixed = pass_to_fixed()[0]
    finish(fixed, "adopt")
    branches.set_paused("a", True)
    assert branches.advance() == []
    with pytest.raises(ConfigError, match="暂停"):
        promote.promote_gen(fixed["id"])
    submit_gen("a", {"llm": {"model": "mA2"}})
    with pytest.raises(ConfigError, match="更新"):
        promote.promote_gen(fixed["id"])
    jobs = branches.advance()
    assert len(jobs) == 1
    assert branches.round_of("a/r-0002")["revision"] == "v-0002"


@pytest.mark.parametrize('legacy', [False, True])
def test_resume_blocked_creation_reuses_round_and_rechecks_sources(tree, monkeypatch, legacy):
    submit_gen('recover', {'llm': {'model': 'mA'}})
    original = branches._ensure_trial
    def unavailable(*args):
        raise ConfigError('historical provenance unavailable')
    monkeypatch.setattr(branches, '_ensure_trial', unavailable)
    assert branches.advance() == []
    record = branches.round_of('recover/r-0001')
    assert record['phase'] == 'blocked' and record['blocked_phase'] == 'smoke'
    if legacy:
        record.pop('blocked_phase')
        branches._save_round(record)
    branches.set_paused('recover', False)
    assert branches.advance() == []  # Resume never bypasses the original protection.
    monkeypatch.setattr(branches, '_ensure_trial', original)
    branches.set_paused('recover', False)
    jobs = branches.advance()
    assert jobs == [{'kind': 'experiment', 'id': 'branch-recover-r-0001-smoke'}]
    state = json.loads((tree / 'branches/recover/state.json').read_text())
    assert state['rounds'] == ['r-0001']


def test_active_old_round_is_not_rewritten_on_new_production(tree, monkeypatch):
    submit_gen("a", {"shots_char_budget": 200})
    smoke = branches.advance()[0]
    original = branches.round_of("a/r-0001")
    new = versions.create_generator_version({**versions.current_generator("production")["config"],
                                             "llm": {"model": "mA"}}, "d-0001")
    ptr = versions.load_pointers()
    ptr["production_gen"] = new
    versions.save_pointers(ptr)
    monkeypatch.setattr("src.iteration.storage.locked", lambda path: True)
    assert branches.advance() == []
    assert branches.round_of("a/r-0001") == original
    monkeypatch.setattr("src.iteration.storage.locked", lambda path: False)
    finish(smoke)
    assert branches.advance()[0]["id"].endswith("r-0002-smoke")


def test_judge_branches_keep_calibration_and_validation_separate(tree):
    submit_judge("judge-a", {"llm": {"model": "jA"}})
    fixed = pass_to_fixed()[0]
    spec = experiment.spec_of(experiment.load_experiment(fixed["id"]))
    assert spec["pack_ref"] == "pack-validation-test"
    assert spec["pack_sha256"]
    finish(fixed, "adopt")
    branches.advance()
    assert versions.load_pointers()["production_judge"] != "j-0001"


def test_validation_pack_stays_unparsed_until_fixed_entry(tree, monkeypatch):
    read = branches._read
    opened = []

    def guard(path):
        if path.name == 'pack.json' and 'validation' in path.parent.name:
            opened.append(path)
            assert allowed, 'validation answers opened before fixed entry'
        return read(path)

    monkeypatch.setattr(branches, '_read', guard)
    allowed = False
    submit_judge('sealed-judge', {'llm': {'model': 'jA'}})
    smoke = branches.advance()[0]
    finish(smoke, 'observe')
    dev = branches.advance()[0]
    assert opened == []
    finish(dev)
    allowed = True
    fixed = branches.advance()[0]
    assert opened and fixed['id'].endswith('-fixed')


def test_one_shot_survives_new_branch_name(tree):
    proposal = submit_gen("a", {"llm": {"model": "mA"}})
    fixed = pass_to_fixed()[0]
    finish(fixed, "reject")
    branches.advance()
    branches.submit("b", "gen", "同一候选换分支", candidate_ref=proposal["candidate_ref"])
    assert pass_to_fixed() == []
    blocked = branches.round_of("b/r-0001")
    assert blocked["phase"] == "blocked" and "one-shot" in blocked["reason"]


def test_promotion_receipt_recovers_crash_between_pointer_and_round(tree, monkeypatch):
    submit_gen("a", {"llm": {"model": "mA"}})
    fixed = pass_to_fixed()[0]
    finish(fixed, "adopt")
    save = branches._save_round
    monkeypatch.setattr(branches, "_save_round", lambda record: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError, match="crash"):
        promote.promote_gen(fixed["id"])
    adopted = versions.load_pointers()["production_gen"]
    monkeypatch.setattr(branches, "_save_round", save)
    assert branches.advance() == []
    assert branches.round_of("a/r-0001")["phase"] == "promoted"
    assert promote.promote_gen(fixed["id"])["production_gen"] == adopted


def test_rebase_handles_deletion_and_retains_unrelated_changes():
    assert branches.rebase({"a": 1, "b": 2}, {"b": 2}, {"a": 1, "b": 3}) == {"b": 3}
    assert branches.rebase({"a": 1}, {"a": 2}, {"a": 2}) == {"a": 2}
    with pytest.raises(ConfigError, match="冲突"):
        branches.rebase({"a": {"x": 1}}, {}, {"a": {"x": 2}})


def test_duplicate_runner_cannot_overwrite_progress(tree):
    submit_gen("a", {"llm": {"model": "mA"}})
    job = branches.advance()[0]
    directory = experiment.load_experiment(job["id"])
    write_json(directory / "state.json", {"status": "running", "sentinel": "unchanged"})
    before = (directory / "state.json").read_bytes()

    @track_run
    def work(exp_dir, _progress=None):
        raise AssertionError("second worker ran")

    with file_lock(directory / ".run.lock"):
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(LockBusy):
                pool.submit(work, directory).result()
    assert (directory / "state.json").read_bytes() == before


def test_cross_process_version_allocation_and_lock_release(tree):
    code = """
import sys
from pathlib import Path
from src.iteration import versions
root = Path(sys.argv[1])
versions.PRIVATE = root
versions.JUDGE_ROOT = root / 'judges'
print(versions.create_judge_version({'mode': 'pairwise_llm', 'llm': {'model': 'new'}}, {},
                                  assets={'prompt.md': b'new'}), flush=True)
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, str(tree)], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True) for _ in range(4)]
    ids = []
    for process in processes:
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, err
        ids.append(out.strip())
    assert len(set(ids)) == 4
    for ref in ids:
        assert versions.judge_dir(ref)["config"]["llm"]["model"] == "new"


def test_scheduler_ignores_cache_directories(tree):
    for folder in scheduler.jobs.FOLDERS.values():
        (tree / folder / '.cache').mkdir(parents=True)
    assert scheduler.Scheduler().tick() == []


@pytest.mark.parametrize('timeout', [None, 180])
def test_scheduler_launches_two_independent_jobs_and_bounds_retries(tree, monkeypatch, timeout):
    submit_gen("a", {"llm": {"model": "mA"}})
    submit_gen("b", {"shots_char_budget": 200})
    commands = []

    class Child:
        pid = 99999999
        result = None

        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return self.result

    monkeypatch.setattr(scheduler.subprocess, "Popen", Child)
    monkeypatch.setattr(scheduler.task_state, "process_start", lambda pid: "synthetic-start")
    clock = [100.0]
    monkeypatch.setattr(scheduler.time, "time", lambda: clock[0])
    sched = scheduler.Scheduler(max_experiments=2, workers=3, max_attempts=1, timeout_seconds=timeout)
    assert len(sched.tick()) == 2
    assert len(commands) == 2 and commands[0] != commands[1]
    assert all(command[command.index('--workers') + 1] == '3' for command in commands)
    if timeout is not None:
        assert all(command[3].endswith('pack_transport.py') for command in commands)
        assert all(command[command.index('--timeout-seconds') + 1] == '180.0' for command in commands)
        assert all(command[command.index('--kind') + 1] == 'experiment' for command in commands)
        assert all(Path(command[command.index('--snapshot') + 1]).is_dir() for command in commands)
    else:
        assert all('--timeout-seconds' not in command for command in commands)
    assert sched.tick() == []
    Child.result = 1
    clock[0] += 60
    assert sched.tick() == []
    assert all(job["status"] == "retry_exhausted" for job in scheduler.jobs_status())


def test_pack_rebuild_preserves_original_sample_and_resumes(tree, monkeypatch):
    proposal = submit_judge("judge-a", {"llm": {"model": "jA"}})
    ptr = versions.load_pointers()
    cfg = versions.current_generator("production")["config"]
    ptr["production_gen"] = versions.create_generator_version({**cfg, "llm": {"model": "mA"}}, "d-0001")
    versions.save_pointers(ptr)
    jobs = branches.advance()
    assert len(jobs) == 1 and jobs[0]["kind"] == "pack"
    assert "validation" not in jobs[0]["id"]  # 固定包在开发准入通过后才重建
    ref = next(job["id"] for job in jobs if "calibration" in job["id"])
    source_path = tree / "judge_eval" / proposal["packs"]["development"]["ref"] / "pack.json"
    before = source_path.read_bytes()
    calls = []

    class Generator:
        fail = True

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, case, forced_reply=True):
            calls.append(case["case_id"])
            if self.fail and len(calls) == 3:
                raise RuntimeError("transport")
            return {"replies": ["new"]}

    monkeypatch.setattr(branch_packs, "ReplyGenerator", Generator)
    monkeypatch.setattr(branch_packs, "build_clients", lambda *args: {})
    with pytest.raises(RuntimeError, match="transport"):
        branch_packs.build(ref)
    assert len(json.loads((tree / "judge_eval" / ref / "building.json").read_text())["rows"]) == 2
    Generator.fail = False
    branch_packs.build(ref)
    result = json.loads((tree / "judge_eval" / ref / "pack.json").read_text())
    assert [row["case_id"] for row in result["rows"]] == [str(i) for i in range(6)]
    assert calls.count("0") == 1 and calls.count("1") == 1
    assert all(row["human_reply"] == ["在"] for row in result["rows"])
    assert source_path.read_bytes() == before


def test_judge_pipeline_runs_real_scoring_and_reports_before_promotion(tree, monkeypatch):
    submit_judge("judge-a", {"llm": {"model": "jA"}})
    calls = []

    class Scorer:
        def __init__(self, ref):
            self.ref = ref

        def is_ai(self, case, replies):
            calls.append((case["case_id"], self.ref))
            return self.ref != "j-0001"

    monkeypatch.setattr(runner, "build_judge", lambda settings, info: Scorer(info["id"]))
    for stage, total in (("smoke", 2), ("dev", 6), ("fixed", 6)):
        job = branches.advance()[0]
        assert job["id"].endswith(stage)
        directory = experiment.load_experiment(job["id"])
        runner.run_judge_experiment(directory, workers=2)
        spec, state = experiment.spec_of(directory), experiment.state_of(directory)
        assert spec["created"]
        assert state["metrics"]["pairs"] == total
        assert state["metrics"]["identified_baseline"] == 0
        assert state["metrics"]["identified_candidate"] == total
        assert (directory / "index.html").exists()
        assert versions.load_pointers()["production_judge"] == "j-0001"
        if stage == "smoke":
            assert len(calls) == 4  # 两版各两题，不做补测
        elif stage == "dev":
            assert "相对生产版达到固定准入门槛后再进入固定验收" in report._adoption_note(report._load_run(directory))
    assert branches.advance() == []
    assert versions.load_pointers()["production_judge"] != "j-0001"
    assert "本轮已推全" in report._adoption_note(report._load_run(directory))


@pytest.mark.parametrize('stage', ['smoke', 'fixed'])
def test_changed_pack_is_rejected_before_any_scoring(tree, monkeypatch, stage):
    submit_judge("judge-a", {"llm": {"model": "jA"}})
    job = (branches.advance() if stage == 'smoke' else pass_to_fixed())[0]
    directory = experiment.load_experiment(job['id'])
    path = tree / 'judge_eval' / experiment.spec_of(directory)['pack_ref'] / 'pack.json'
    pack = json.loads(path.read_text())
    pack["rows"][0]["ai_replies"] = ["changed"]
    write_json(path, pack)
    monkeypatch.setattr(runner, "build_judge", lambda *args: pytest.fail("called model"))
    with pytest.raises(ConfigError, match="指纹"):
        runner.run_judge_experiment(experiment.load_experiment(job["id"]))


def test_data_drift_blocks_without_replacing_sample(tree):
    submit_gen("a", {"llm": {"model": "mA"}})
    ptr = versions.load_pointers()
    ptr["data"] = "d-0002"
    versions.save_pointers(ptr)
    assert branches.advance() == []
    record = branches.round_of("a/r-0001")
    assert record["phase"] == "blocked" and "禁止自动换样本" in record["reason"]


def test_judge_one_shot_ignores_metadata_and_recognizes_legacy_fingerprints(tree):
    import hashlib

    proposal = submit_judge("a", {"llm": {"model": "jA"}})
    fixed = pass_to_fixed()[0]
    directory = experiment.load_experiment(fixed["id"])
    spec = experiment.spec_of(directory)
    info = versions.judge_dir(proposal["candidate_ref"])
    # 模拟改造前规格：其指纹包含来源 meta。
    payload = versions.version_payload(info["dir"])
    spec["candidate_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    write_json(directory / "spec.json", spec)
    finish(fixed, "reject")
    branches.advance()
    duplicate = versions.create_judge_version(info["config"], {"different": "origin"}, source_dir=info["dir"])
    branches.submit("b", "judge", "同模型换来源", candidate_ref=duplicate,
                    development_pack="pack-calibration-test", validation_pack="pack-validation-test")
    assert pass_to_fixed() == []
    assert "one-shot" in branches.round_of("b/r-0001")["reason"]


def test_two_processes_cannot_promote_over_each_other(tree):
    submit_gen("a", {"llm": {"model": "mA"}})
    submit_gen("b", {"shots_char_budget": 200})
    fixed = pass_to_fixed()
    for job in fixed:
        finish(job, "adopt")
    code = """
import sys, time
from pathlib import Path
from src.config import ConfigError
from src.iteration import promote, versions, learning_guard
# Same isolated legacy fixture as the parent test: this child tests the
# cross-process promotion lock, not source reconstruction.
learning_guard.verify = lambda *a: None
root = Path(sys.argv[1])
versions.PRIVATE = root
versions.DATA_ROOT = root / 'data'
versions.GEN_ROOT = root / 'generators'
versions.JUDGE_ROOT = root / 'judges'
versions.POINTERS_PATH = root / 'pointers.json'
while not (root / 'start').exists():
    time.sleep(.01)
try:
    promote.promote_gen(sys.argv[2])
    print('promoted')
except ConfigError as exc:
    print(str(exc))
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, str(tree), job["id"]],
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for job in fixed]
    (tree / "start").touch()
    results = []
    for process in processes:
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, err
        results.append(out.strip())
    assert results.count("promoted") == 1
    assert sum("过期" in result for result in results) == 1
    assert len(versions.load_pointers()["branch_promotions"]) == 1


def test_small_gains_continue_from_accepted_branch_and_use_direct_endpoint(tree, monkeypatch):
    settings = branches.load_settings()
    settings["evaluation"]["adoption"]["fixed_entry_min_net_win_rate"] = .5
    monkeypatch.setattr(branches, "load_settings", lambda: settings)
    submit_gen("a", {"llm": {"model": "mA"}})
    assert pass_to_fixed() == []  # +2/6 < +3；但开发版被保留
    first = branches.round_of("a/r-0001")
    assert first["phase"] == "development_accepted"
    assert versions.load_pointers()["production_gen"] == "g-0001"
    second = submit_gen("a", {"shots_char_budget": 200})
    assert second["development_ref"] == first["candidate_ref"]
    assert versions.load_generator(second["candidate_ref"])["config"]["llm"]["model"] == "mA"
    finish(branches.advance()[0], "observe")
    dev_job = branches.advance()[0]
    assert experiment.spec_of(experiment.load_experiment(dev_job["id"]))["baseline_ref"] == first["candidate_ref"]
    finish(dev_job)  # B→C 又 +2，不能将两次 +2 相加。
    endpoint = branches.advance()[0]
    endpoint_dir = experiment.load_experiment(endpoint["id"])
    spec = experiment.spec_of(endpoint_dir)
    assert spec["against_production"] and spec["baseline_ref"] == "g-0001"
    finish_with_cases(endpoint_dir, wins=1)  # A→C 实测仅 +1
    assert branches.advance() == []
    assert branches.round_of("a/r-0002")["phase"] == "development_accepted"
    assert not any(p.name.endswith("-fixed") for p in (tree / "experiments").iterdir())


def test_scheduler_restart_recognizes_existing_child_without_duplicate(tree, monkeypatch):
    submit_gen("a", {"llm": {"model": "mA"}})
    job = branches.advance()[0]
    directory = experiment.load_experiment(job["id"])
    write_json(directory / "state.json", {"progress": {"status": "running", "pid": os.getpid()}})
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *a, **kw: pytest.fail("duplicate process"))
    assert scheduler.Scheduler().tick() == []


def test_long_branch_name_produces_valid_worker_job_id(tree):
    submit_gen("a" * 80, {"llm": {"model": "mA"}})
    assert branches._name(branches.advance()[0]["id"])


def test_development_stage_limit_retains_winner_without_fixed_or_production(tree):
    before = versions.load_pointers()
    proposal = submit_gen('limited', {'llm': {'model': 'mA'}})
    branches.set_stage_limit('limited', 'development')
    assert pass_to_fixed() == []
    assert branches.advance() == []
    state = branches.status()[0]
    assert state['current_round']['phase'] == 'development_accepted'
    assert state['development']['candidate_ref'] == proposal['candidate_ref']
    assert versions.load_pointers() == before
    assert not list((tree / 'experiments').glob('*-fixed'))
    next_proposal = submit_gen('limited', {'shots_char_budget': 200})
    assert next_proposal['development_ref'] == proposal['candidate_ref']
    assert branches.status()[0]['stage_limit'] == 'development'
    with pytest.raises(ConfigError, match='stage limit'):
        branches.set_stage_limit('limited', 'unknown')


def test_fork_accepted_development_uses_common_control(tree):
    pointers = versions.load_pointers()
    parent = submit_gen('parent', {'llm': {'model': 'mA'}})
    branches.set_stage_limit('parent', 'development')
    assert pass_to_fixed() == []
    for name, count in [('one', 1), ('five', 5)]:
        proposal = branches.submit(name, 'gen', 'few-shot count',
            overrides={'max_shots_per_case': count}, from_branch='parent')
        branches.set_stage_limit(name, 'development')
        assert proposal['development_ref'] == parent['candidate_ref']
        assert proposal['development_source']['experiment_id'] == 'branch-parent-r-0001-dev'
        config = versions.load_generator(proposal['candidate_ref'])['config']
        assert config['llm']['model'] == 'mA'
        assert config['max_shots_per_case'] == count
    jobs = branches.advance()
    assert len(jobs) == 2
    assert all(experiment.spec_of(experiment.load_experiment(job['id']))['baseline_ref']
               == parent['candidate_ref'] for job in jobs)
    assert versions.load_pointers() == pointers


@pytest.mark.parametrize('invalid', [None, 'missing', 'implementation', 'source',
                                    'data', 'data_implementation', 'missing_data_implementation'])
def test_fork_historical_receipt_after_executor_migration(tree, monkeypatch, invalid):
    from src.config import sha256_file
    from src.iteration import data_guard, gates, runtime
    original_snapshot = data_guard.generation_snapshot
    executor = {'generator/generator.py': sha256_file(Path(__file__).parents[1] / 'src/generator/generator.py')}

    def snapshot_with_executor(*args):
        return {**original_snapshot(*args), 'implementation_sha256': dict(executor)}

    monkeypatch.setattr(data_guard, 'generation_snapshot', snapshot_with_executor)
    submit_gen('parent', {'llm': {'model': 'mA'}})
    branches.set_stage_limit('parent', 'development')
    assert pass_to_fixed() == []
    directory = experiment.load_experiment('branch-parent-r-0001-dev')
    snapshot = runtime.verify(directory / 'runtime.json')
    material_bytes = {name: (directory / name).read_bytes()
                      for name in ('spec.json', 'state.json', 'cases.jsonl')}
    original = gates.materials

    def migrated(spec):
        value = original(spec)
        value['implementation']['judge/judge.py'] = '0' * 64
        return value

    monkeypatch.setattr(gates, 'materials', migrated)
    monkeypatch.setattr(runtime, 'environment', lambda: {'different': 'host'})
    executor['generator/generator.py'] = '0' * 64
    if invalid == 'missing':
        (directory / 'runtime.json').unlink()
    elif invalid == 'implementation':
        spec = experiment.spec_of(directory)
        spec['evaluation_materials']['implementation']['judge/judge.py'] = '1' * 64
        write_json(directory / 'spec.json', spec)
    elif invalid == 'source':
        (snapshot / 'src/judge/judge.py').write_text('changed')
    elif invalid == 'data':
        path = tree / 'data/d-0001/dev_pool.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['context'][0]['text'] = '变更后的输入'
        path.write_text('\n'.join(json.dumps(row) for row in rows))
    elif invalid in {'data_implementation', 'missing_data_implementation'}:
        spec = experiment.spec_of(directory)
        if invalid == 'data_implementation':
            spec['data_snapshot']['implementation_sha256']['generator/generator.py'] = '1' * 64
        else:
            del spec['data_snapshot']['implementation_sha256']
        write_json(directory / 'spec.json', spec)
    if invalid:
        with pytest.raises((ConfigError, OSError)):
            branches.submit('fork', 'gen', 'count', overrides={'max_shots_per_case': 1},
                            from_branch='parent')
        assert not (tree / 'branches/fork/state.json').exists()
    else:
        with pytest.raises(ConfigError, match='environment'):
            runtime.verify(directory / 'runtime.json')
        proposal = branches.submit('fork', 'gen', 'count',
            overrides={'max_shots_per_case': 1}, from_branch='parent')
        assert proposal['development_source']['experiment_id'] == directory.name
        assert all((directory / name).read_bytes() == blob for name, blob in material_bytes.items())


@pytest.mark.parametrize('invalid', ['unaccepted', 'stale', 'kind', 'candidate', 'evidence', 'materials'])
def test_fork_rejects_invalid_development_receipt(tree, invalid):
    submit_gen('parent', {'llm': {'model': 'mA'}})
    branches.set_stage_limit('parent', 'development')
    if invalid != 'unaccepted':
        assert pass_to_fixed() == []
        path = tree / 'branches/parent/state.json'
        state = json.loads(path.read_text())
        if invalid == 'stale':
            state['development']['basis']['production_gen'] = 'g-stale'
        elif invalid == 'kind':
            state['kind'] = 'judge'
        elif invalid == 'candidate':
            state['development']['candidate_ref'] = 'g-0001'
        elif invalid == 'evidence':
            directory = experiment.load_experiment(state['development']['experiment_id'])
            (directory / 'cases.jsonl').write_text('')
        elif invalid == 'materials':
            asset = versions.generator_dir(state['development']['candidate_ref']) / 'persona.md'
            asset.chmod(0o644)
            asset.write_text('changed material')
        write_json(path, state)
    with pytest.raises(ConfigError):
        branches.submit('fork', 'gen', 'few-shot count', overrides={'max_shots_per_case': 1},
                        from_branch='parent')
    assert not (tree / 'branches/fork/state.json').exists()


def test_direct_development_promotion_recovers_blocked_round(tree):
    before = versions.load_pointers()
    proposal = submit_gen('limited', {'llm': {'model': 'mA'}})
    branches.set_stage_limit('limited', 'development')
    finish(branches.advance()[0], 'observe')
    job = branches.advance()[0]
    finish(job)
    directory = experiment.load_experiment(job['id'])
    evidence = {name: (directory / name).read_bytes() for name in ('spec.json', 'state.json', 'cases.jsonl')}
    record = branches.round_of('limited/r-0001')
    record.update(phase='blocked', reason='historical source path unavailable')
    branches._save_round(record)
    assert branches.advance() == []
    for _ in range(2):
        assert promote.promote_gen(job['id']) == before
        assert branches.round_of('limited/r-0001')['phase'] == 'development_accepted'
        assert branches.status()[0]['development']['candidate_ref'] == proposal['candidate_ref']
        assert branches.advance() == []
    assert versions.load_pointers() == before
    assert not list((tree / 'experiments').glob('*-fixed'))
    assert all((directory / name).read_bytes() == blob for name, blob in evidence.items())
