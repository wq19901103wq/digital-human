"""跨实验复用和独立补测必须同时成立；测试不调用真实模型。"""
import copy
import hashlib
import io
import json
import multiprocessing
import sqlite3
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from src import cache, llm, tracing
from src.generator.generator import ReplyGenerator
from src.judge import corrected
from src.judge.judge import Judge
from src.dashboard import trace_view
import test_live_dashboard
import test_corrected_judge

case = test_corrected_judge.case
bundle = test_corrected_judge.bundle
live_server = test_live_dashboard.live_server


def test_reviewed_judge_rename_reuses_original_identity_but_edits_invalidate(tmp_path):
    original_sha = "8b5b7a9e76d683b87d13f6d3e2c0ee79eb601643690c0c63195b1ab5df1e9071"
    client = object.__new__(corrected.CodexJudgeClient)
    client.config = {"provider": "codex_cli", "model": "test", "reasoning_effort": "low",
                     "codex_cli_version": "test"}
    assert client.cache_identity()["adapter"] == cache.digest([original_sha])
    changed = tmp_path / "corrected.py"
    changed.write_bytes(Path(corrected.__file__).read_bytes() + b"\n# subsequent edit\n")
    changed_sha = hashlib.sha256(changed.read_bytes()).hexdigest()
    assert cache.code_digest(changed) == cache.digest([changed_sha])
    assert cache.code_digest(changed) != client.cache_identity()["adapter"]


def trace(root, case, exp="a", **spec):
    return tracing.CaseTrace(root / "experiments" / exp, case,
                             {"dataset": "development", "data_ref": "d1", "baseline_ref": "j1",
                              "candidate_ref": "j2", **spec})


def generate(root, generator, case, exp="a", round=0, branch="candidate", **spec):
    log = trace(root, case, exp, **spec)
    with log.operation("generation", branch, round, {"forced_reply": True}) as op:
        value = generator.generate(case)
        op["result"] = value
    log.finish("ok")
    return value, log


def judge(root, scorer, case, exp="a", round=0, branch="baseline", **spec):
    log = trace(root, case, exp, **spec)
    with log.operation("judge", branch, round, {"candidate_replies": ["九点"]}) as op:
        value = scorer.is_ai(case, ["九点"])
        op["result"] = {"identified_ai": value}
    log.finish("ok")
    return value, log


@pytest.fixture
def generation(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_URL", "https://example.test/api/coding")
    monkeypatch.setenv("CACHE_KEY", "cache-secret")
    settings = {"llm": {"base_url_env": "CACHE_URL", "api_key_env": "CACHE_KEY", "max_retries": 0},
                "evaluation": {"few_shots_per_case": 2, "few_shots_char_budget": 1000}}
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "persona.md").write_text("给出回复")
    calls, answers = [], []
    def request(req, **_):
        calls.append(json.loads(req.data))
        answer = answers.pop(0) if answers else '{"replies":["八点"]}'
        if isinstance(answer, BaseException):
            raise answer
        return io.BytesIO(json.dumps({"content": [{"type": "text", "text": answer}], "stop_reason": "end_turn"}).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", request)
    def build(**cfg):
        return ReplyGenerator(settings, {"retriever": {"enabled": False}},
                              llm.ChatClient(settings, {"model": "test", **cfg}), prompt_root=prompts)
    return build, calls, answers, prompts


def test_generation_reused_after_promotion_and_data_version_alias(tmp_path, generation, case):
    build, calls, _, _ = generation
    gen = build()
    a, original = generate(tmp_path, gen, case, branch="candidate")
    b, reused = generate(tmp_path, gen, case, "b", branch="baseline", data_ref="d2")
    assert a["replies"] == b["replies"] and len(calls) == 1
    store = cache.get_store(tmp_path / ".cache")
    records = store.history(case_id=case["case_id"], experiment_id="b", data_ref="d2")
    hit = records[0]["cache_hits"][0]
    assert hit["layer"] == "generation"
    assert hit["origin"]["experiment_id"] == "a" and hit["origin"]["branch"] == "candidate"
    assert hit["origin"]["trace_ref"] == original.ref
    assert "缓存复用" in trace_view.payload(tmp_path / "experiments/b", reused.ref)["html"]
    keys = store.entries(layer="llm_request", model="test", experiment_id="a", case_id=case["case_id"])
    saved = store.get(keys[0]["key"])
    assert saved["identity"]["body"] == calls[0]
    assert "cache-secret" not in json.dumps(saved)


def test_changed_effective_inputs_and_config_invalidate(tmp_path, generation, case, monkeypatch):
    build, calls, _, prompts = generation
    generate(tmp_path, build(), case)
    changed = {**case, "context": [{**case["context"][0], "text": "新的问题"}]}
    generate(tmp_path, build(), changed, "changed-input")
    generate(tmp_path, build(model="new-model"), case, "changed-model")
    generate(tmp_path, build(temperature=.1), case, "changed-temperature")
    generate(tmp_path, build(max_tokens=100), case, "changed-tokens")
    generate(tmp_path, build(model_revision="revision-2"), case, "changed-revision")
    (prompts / "persona.md").write_text("不同人格")
    generate(tmp_path, build(), case, "changed-prompt")
    monkeypatch.setenv("CACHE_URL", "https://other.test/api/coding")
    generate(tmp_path, build(), case, "changed-endpoint")
    assert len(calls) == 8


def test_identical_text_in_distinct_cases_and_cache_epochs_are_independent(tmp_path, generation, case, monkeypatch):
    build, calls, _, _ = generation
    gen = build()
    generate(tmp_path, gen, case)
    generate(tmp_path, gen, {**case, "case_id": "another-observation"}, "another-case")
    monkeypatch.setenv("DH_CACHE_EPOCH", "new-remote-model-revision")
    generate(tmp_path, gen, case, "new-epoch")
    assert len(calls) == 3


def test_rounds_are_independent_but_reused_across_experiments(tmp_path, generation, case):
    build, calls, _, _ = generation
    gen = build()
    for exp in ("a", "b"):
        for round in (0, 1, 2):
            generate(tmp_path, gen, case, exp, round)
    assert len(calls) == 3
    generate(tmp_path, gen, case, "fresh", cache={"sample_epoch": "independent-study"})
    assert len(calls) == 4
    generate(tmp_path, gen, case, "disabled", cache={"enabled": False})
    assert len(calls) == 5
    assert cache._scope.get() is None
    row = cache.get_store(tmp_path / ".cache").history(experiment_id="fresh")[0]
    assert row["sample_epoch"] == "independent-study"


def test_history_index_failure_preserves_original_result(tmp_path, generation, case, monkeypatch, caplog):
    build, _, answers, _ = generation
    def unavailable(*_):
        raise sqlite3.OperationalError("index unavailable")
    monkeypatch.setattr(cache, "record_operation", unavailable)
    result, log = generate(tmp_path, build(), case)
    assert result["replies"] == ["八点"] and log.path.exists()
    answers.append(TimeoutError("actual request failed"))
    with pytest.raises(llm.LLMError, match="actual request failed"):
        generate(tmp_path, build(), case, "failed", round=1)
    assert "索引写入失败" in caplog.text


def test_fixed_partitions_and_instances_do_not_share(tmp_path, generation, case):
    build, calls, _, _ = generation
    gen = build()
    generate(tmp_path, gen, case)
    generate(tmp_path, gen, case, "fixed-a", dataset="fixed_test")
    generate(tmp_path, gen, case, "fixed-b", dataset="fixed_test")
    generate(tmp_path, gen, case, "fixed-b", dataset="fixed_test")
    generate(tmp_path / "another-instance", gen, case)
    assert len(calls) == 4


def test_failed_and_malformed_responses_never_poison_retries(tmp_path, generation, case):
    build, calls, answers, _ = generation
    gen = build()
    answers.append(TimeoutError("network timeout"))
    with pytest.raises(llm.LLMError):
        generate(tmp_path, gen, case)
    answers.extend(['{"replies": []}', '{"replies": ["格式修复成功"]}'])
    result, _ = generate(tmp_path, gen, case, "resume")
    assert result["replies"] == ["格式修复成功"] and len(calls) == 3
    generate(tmp_path, gen, case, "reuse")
    assert len(calls) == 3
    history = cache.get_store(tmp_path / ".cache").history(experiment_id="a")
    assert history[0]["status"] == "failed"


def test_threads_share_one_actual_request(tmp_path, generation, case):
    build, calls, _, _ = generation
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda i: generate(tmp_path, build(), case, f"thread-{i}"), range(4)))
    assert len(results) == 4 and len(calls) == 1


def _process(root, start):
    start.wait(10)
    log = trace(Path(root), {"case_id": "x"}, f"process-{multiprocessing.current_process().pid}")
    def produce():
        with (Path(root) / "actual-calls.txt").open("a") as output:
            output.write("called\n")
        time.sleep(.1)
        return "result"
    with log.operation("test", "baseline", 0, {}):
        assert cache.memo("test", {"prompt": "identical"}, produce) == "result"


def test_processes_share_one_actual_request(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    processes = [ctx.Process(target=_process, args=(str(tmp_path), start)) for _ in range(3)]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join()
        assert process.exitcode == 0
    assert (tmp_path / "actual-calls.txt").read_text().splitlines() == ["called"]


@pytest.fixture
def codex(monkeypatch):
    calls, failures = [], []
    def run(command, **kwargs):
        if command == ["codex", "--version"]:
            return SimpleNamespace(stdout="codex-cli 0.144.1")
        schema = "--output-schema" in command
        calls.append((schema, kwargs["input"]))
        if schema and failures:
            raise failures.pop(0)
        result = {"option_A": test_corrected_judge.features(), "option_B": test_corrected_judge.features()} if schema else {
            "human_option": "B", "confidence": .9, "reason": "测试"}
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed","usage":{"output_tokens":12}}\n', stderr="")
    monkeypatch.setattr(corrected.subprocess, "run", run)
    monkeypatch.setattr(corrected.random, "random", lambda: .9)
    return calls, failures


def corrected_scorer(bundle, **changes):
    cfg, directory = bundle
    cfg = copy.deepcopy(cfg)
    cfg["llm"] = {"provider": "codex_cli", "model": "luna", "reasoning_effort": "low",
                  "timeout_seconds": 360, "codex_cli_version": "0.144.1"}
    cfg.update(changes)
    return corrected.CorrectedJudge(cfg, directory)


def test_judge_promoted_candidate_reused_and_three_independent_votes(tmp_path, bundle, case, codex):
    calls, _ = codex
    scorer = corrected_scorer(bundle)
    first = []
    for round in (0, 1, 2):
        result, _ = judge(tmp_path, scorer, case, "a", round, "candidate")
        first.append(result)
    assert len(calls) == 6
    for round in (0, 1, 2):
        result, log = judge(tmp_path, corrected_scorer(bundle), case, "b", round, "baseline")
        assert result == first[round]
        assert log.value["operations"][0]["events"][0]["data"]["layer"] == "judge"
    assert len(calls) == 6


def test_threshold_change_reuses_initial_and_features(tmp_path, bundle, case, codex, monkeypatch):
    calls, _ = codex
    monkeypatch.setattr(corrected.runtime, "score_formal_judge_pair", lambda *_: .8)
    result, _ = judge(tmp_path, corrected_scorer(bundle, correction_threshold=.7), case)
    changed, log = judge(tmp_path, corrected_scorer(bundle, correction_threshold=.9), case, "new-threshold")
    assert result is True and changed is False and len(calls) == 2
    hits = [e["data"]["layer"] for e in log.value["operations"][0]["events"] if e["kind"] == "cache_hit"]
    assert {"judge_initial", "judge_features", "llm_request"} <= set(hits)
    judge(tmp_path, corrected_scorer(bundle, feature_llm={"model": "sol", "reasoning_effort": "high"}), case, "new-extractor")
    assert len(calls) == 3 and calls[-1][0] is True


def test_partial_judge_resume_keeps_blinding_and_initial(tmp_path, bundle, case, codex, monkeypatch):
    calls, failures = codex
    failures.append(TimeoutError("feature request timed out"))
    with pytest.raises(TimeoutError):
        judge(tmp_path, corrected_scorer(bundle), case)
    monkeypatch.setattr(corrected.random, "random", lambda: .1)  # 续跑仍用第一次已分配的盲序
    judge(tmp_path, corrected_scorer(bundle), case, "resume")
    assert [schema for schema, _ in calls] == [False, True, True]


def test_pairwise_judge_invalid_result_is_not_cached(tmp_path, generation, case):
    build, calls, answers, _ = generation
    scorer = Judge({}, build()._llm["private"])
    answers.append('{"ai_is":"unknown"}')
    with pytest.raises(RuntimeError, match="非法"):
        judge(tmp_path, scorer, case)
    answers.append('{"ai_is":"both_human"}')
    result, _ = judge(tmp_path, scorer, case, "resume")
    assert result is False
    judge(tmp_path, scorer, case, "reuse", branch="candidate")
    assert len(calls) == 2


def test_legacy_history_index_is_idempotent_and_never_claims_reuse(tmp_path):
    exp = tmp_path / "experiments/legacy"
    exp.mkdir(parents=True)
    (exp / "spec.json").write_text(json.dumps({"data_ref": "d1", "baseline_ref": "g1", "dataset": "development"}))
    (exp / "cases.jsonl").write_text('{"case_id":"x","status":"ok","identified_baseline":true}\n{"partial":')
    assert cache.import_history(tmp_path)["skipped_incomplete"] == 1
    cache.import_history(tmp_path)
    store = cache.get_store(tmp_path / ".cache")
    rows = store.history(case_id="x", version="g1")
    assert len(rows) == 1 and rows[0]["result"]["identified_baseline"] is True
    assert store.stats()["history_records"] == 2 and store.stats()["entries"] == {}


def test_cache_database_and_symlink_blocked(live_server, tmp_path, monkeypatch):
    from src.dashboard import server
    root = server.ROOT
    store = cache.get_store(root / "instances/demo/.cache")
    alias = root / "alias-cache.sqlite3"
    alias.symlink_to(store.path)
    for path in ("/instances/demo/.cache/results.sqlite3", "/alias-cache.sqlite3?download=1"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urlopen(live_server + path)
        assert caught.value.code == 403
