"""协议 v7 单次判定（flip_extra_rounds=0）：每题判官每侧只判定一次，无补验轮。

分歧题的确认口径 = 初测票本身（单票 flip_verified 结构，下游公式零改动）；
flip_extra_rounds>0 的旧两次协议行为保持不变（本文件内有回归用例）。
"""
from __future__ import annotations

import json

import pytest

from src.infrastructure.config import ConfigError
from src.infrastructure.config import load_settings
from src.iteration import comparisons, experiment, record_contract, runner, versions
from src.infrastructure.storage import write_json
from leakage_support import isolated_legacy_provenance

pytestmark = pytest.mark.usefixtures('isolated_legacy_provenance')


def _write(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _case(cid: str, chat_type: str = "group") -> dict:
    return {"case_id": cid, "chat_type": chat_type,
            "context": [{"sender": "甲", "text": "在吗"}],
            "human_reply": ["在"], "source_message_id": f"c:{cid}"}


@pytest.fixture()
def priv(tmp_path, monkeypatch):
    """最小可用私有树：数据版本 d-0001 + g-0001 + j-0001 + 指针。"""
    priv_root = tmp_path / "private"
    data = priv_root / "data" / "d-0001"
    _write(data / "manifest.json", {"id": "d-0001", "seed": 42})
    _write(data / "dev_pool.jsonl",
           "\n".join(json.dumps(_case(f"d{i}", "group" if i % 2 else "private")) for i in range(6)))
    _write(data / "fixed_test.jsonl",
           "\n".join(json.dumps(_case(f"f{i}", "group" if i % 2 else "private")) for i in range(6)))
    _write(data / "persona.md", "<instructions>测试人格</instructions>")
    (data / "scenarios").mkdir(parents=True)
    _write(data / "scenarios" / "friend.md", "# friend")
    _write(data / "report.json", {"total": 9000, "review_status": "approved", "examples_sha256": "x"})
    _write(data / "fewshot_pool.jsonl", "")

    gen_cfg = {"llm": {"model": "m1"}, "retriever": {"enabled": False},
               "max_shots_per_case": 3, "shots_char_budget": 100}
    gdir = priv_root / "generators" / "g-0001"
    _write(gdir / "config.json", {**gen_cfg, "data_version": "d-0001", "persona_sha256": "x"})
    _write(gdir / "persona.md", "<instructions>测试人格</instructions>")
    (gdir / "scenarios").mkdir(parents=True)
    _write(gdir / "scenarios" / "friend.md", "# friend")

    jdir = priv_root / "judges" / "j-0001"
    _write(jdir / "config.json", {"mode": "pairwise_llm", "llm": {"model": "m1"}})
    _write(jdir / "meta.json", {"note": "bootstrap"})

    _write(priv_root / "pointers.json",
           {"data": "d-0001", "production_gen": "g-0001", "iteration_gen": "g-0001",
            "production_judge": "j-0001", "iteration_judge": "j-0001"})

    monkeypatch.setattr(versions, "PRIVATE", priv_root)
    monkeypatch.setattr(versions, "DATA_ROOT", priv_root / "data")
    monkeypatch.setattr(versions, "GEN_ROOT", priv_root / "generators")
    monkeypatch.setattr(versions, "JUDGE_ROOT", priv_root / "judges")
    monkeypatch.setattr(versions, "POINTERS_PATH", priv_root / "pointers.json")
    monkeypatch.setattr(experiment, "EXP_ROOT", priv_root / "experiments")
    return priv_root


def _small_settings(monkeypatch, extra_rounds: int):
    """非冒烟实验的构成准入调到 fixture 体量，并显式选定判定协议轮数。"""
    settings = load_settings()
    for ds in ("development", "fixed_test"):
        settings["evaluation"][ds].update(total=6, group_ratio=0.5)
    settings["evaluation"]["flip_verification"]["extra_rounds_per_version"] = extra_rounds
    import src.iteration.experiment as exp_mod
    monkeypatch.setattr(exp_mod, "load_settings", lambda: settings)
    return settings


def _mock_gen_components(monkeypatch):
    """生成器/判官打桩并计数：判官只认模型 m1 的回复（候选 m2 必被识别）。"""
    from src import llm
    calls = {"gen": 0, "judge": 0}

    class Generator:
        def __init__(self, settings, config, clients, **kwargs):
            self.model = config["llm"]["model"]

        def generate(self, case, **kwargs):
            calls["gen"] += 1
            return {"replies": [self.model], "latency_ms": 1}

    class Scorer:
        def is_ai(self, case, replies):
            calls["judge"] += 1
            return replies[0] == "m1"

    monkeypatch.setattr(runner, "ReplyGenerator", Generator)
    monkeypatch.setattr(runner, "build_judge", lambda *args, **kwargs: Scorer())
    monkeypatch.setattr(llm, "build_clients", lambda *args, **kwargs: {})
    return calls


def test_single_draw_gen_experiment(priv, monkeypatch):
    _small_settings(monkeypatch, extra_rounds=0)
    exp = experiment.create_gen_experiment("development", "v7 单次判定协议", {"llm": {"model": "m2"}},
                                           exp_id="single-draw")
    spec = experiment.spec_of(exp)
    assert spec["protocol"]["flip_extra_rounds"] == 0
    calls = _mock_gen_components(monkeypatch)

    runner.run_gen_experiment(exp)

    # 6 题 × 2 侧：生成与判定各一次，无任何补验轮调用
    assert calls == {"gen": 12, "judge": 12}
    state = experiment.state_of(exp)
    assert state["status"] == "finished"
    assert state["verdict"] == "merge_to_iteration_baseline"
    metrics = state["metrics"]
    assert metrics["wins_confirmed"] == 6 and metrics["losses_confirmed"] == 0
    assert metrics["contested"] == 0 and metrics["net_win_confirmed"] == 6
    assert metrics["wins"] == 6 and metrics["net_win_raw"] == 6
    rows = [json.loads(line) for line in (exp / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    for row in rows:
        assert row["identified_baseline"] is True and row["identified_candidate"] is False
        # 单票结构：final=该票，record 结构与下游公式零改动
        assert row["flip_verified"] == {
            "baseline_votes": [True], "candidate_votes": [False],
            "baseline_identified_final": True, "candidate_identified_final": False,
        }
    # 逐题记录通过完成校验（finish 内部已调用，这里显式复核）
    record_contract.validate_completion(exp, metrics)


def test_flip_two_protocol_still_regenerates_and_verifies(priv, monkeypatch):
    """旧两次协议（flip_extra_rounds=2）回归：补验循环与多数票结构不变。"""
    _small_settings(monkeypatch, extra_rounds=2)
    exp = experiment.create_gen_experiment("development", "旧协议补验", {"llm": {"model": "m2"}},
                                           exp_id="flip-two")
    assert experiment.spec_of(exp)["protocol"]["flip_extra_rounds"] == 2
    calls = _mock_gen_components(monkeypatch)

    runner.run_gen_experiment(exp)

    # 每题全部分歧：2 初测生成 + 4 补验生成 = 6 gen；2 初测判定 + 4 补验判定 = 6 judge
    assert calls == {"gen": 36, "judge": 36}
    state = experiment.state_of(exp)
    assert state["status"] == "finished"
    assert state["metrics"]["net_win_confirmed"] == 6
    rows = [json.loads(line) for line in (exp / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        verified = row["flip_verified"]
        assert verified["baseline_votes"] == [True] * 3
        assert verified["candidate_votes"] == [False] * 3
        assert verified["baseline_identified_final"] is True
        assert verified["candidate_identified_final"] is False


def test_single_draw_judge_eval_experiment(priv, monkeypatch):
    """Judge 校准实验同样按 protocol 分流：单次判定只调用两侧各一次。"""
    _small_settings(monkeypatch, extra_rounds=0)
    _write(priv / "judge_eval" / "pack-single" / "pack.json", {
        "c0_gen_version": "g-0001", "rows": [{**_case(f"c{i}"), "ai_replies": ["AI"]} for i in range(6)]})
    exp = experiment.create_judge_eval_experiment(
        "development", "pack-single", {"llm": {"model": "m9"}}, change="v7 单次判定 judge",
        exp_id="judge-single")
    assert experiment.spec_of(exp)["protocol"]["flip_extra_rounds"] == 0
    calls = {"baseline": 0, "candidate": 0}

    class Scorer:
        def __init__(self, side):
            self.side = side

        def is_ai(self, case, replies):
            calls[self.side] += 1
            return self.side == "candidate"

    from src import llm
    makers = iter([lambda: Scorer("baseline"), lambda: Scorer("candidate")])
    monkeypatch.setattr(runner, "build_judge", lambda *a, **k: next(makers)())
    monkeypatch.setattr(llm, "build_clients", lambda *a, **k: {})

    runner.run_judge_experiment(exp)

    assert calls == {"baseline": 6, "candidate": 6}
    state = experiment.state_of(exp)
    assert state["status"] == "finished"
    metrics = state["metrics"]
    assert metrics["wins_confirmed"] == 6 and metrics["net_win_confirmed"] == 6
    row = json.loads((exp / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["flip_verified"] is True
    assert row["baseline_votes"] == [False] and row["candidate_votes"] == [True]
    assert row["baseline_identified_final"] is False and row["candidate_identified_final"] is True
    record_contract.validate_completion(exp, metrics)


@pytest.fixture
def single_draw_comparison(tmp_path, monkeypatch):
    root = tmp_path / "instance"
    monkeypatch.setattr(versions, "PRIVATE", root)
    monkeypatch.setattr(versions, "DATA_ROOT", root / "data")
    data = root / "data" / "d-test"
    write_json(data / "manifest.json", {"id": "d-test"})
    rows = [{"case_id": str(i), "context": [], "human_reply": ["h"], "ai_replies": ["AI"]} for i in range(2)]
    write_json(root / "judge_eval" / "pack-test" / "pack.json",
               {"data_ref": "d-test", "c0_gen_version": "g-test", "rows": rows})
    return rows


def _single_draw_row(cid, b, c):
    value = {"case_id": str(cid), "status": "ok", "baseline_correct": b, "candidate_correct": c}
    if b != c:
        value.update(flip_verified=True, baseline_votes=[b], candidate_votes=[c],
                     baseline_identified_final=b, candidate_identified_final=c)
    return value


def test_single_draw_votes_pass_completion(single_draw_comparison, tmp_path):
    target = comparisons.register("comparison-single", data_ref="d-test", pack_ref="pack-test",
                                  comparison={"baseline": {"recipe": "LR"}, "candidate": {"recipe": "GBDT"}},
                                  change="v7 单次判定比较", protocol={"flip_extra_rounds": 0})
    with record_contract.writer(target) as append:
        append(_single_draw_row(0, True, True))
        append(_single_draw_row(1, False, True))
    metrics = comparisons.complete(target)
    assert metrics["pairs"] == 2
    assert metrics["wins_confirmed"] == 1 and metrics["net_win_confirmed"] == 1
    assert metrics["contested"] == 0


def test_single_draw_rejects_multi_vote_structure(single_draw_comparison):
    target = comparisons.register("comparison-single-bad", data_ref="d-test", pack_ref="pack-test",
                                  comparison={"baseline": {"recipe": "LR"}, "candidate": {"recipe": "GBDT"}},
                                  change="v7 下单票长度不符", protocol={"flip_extra_rounds": 0})
    bad = _single_draw_row(0, False, True)
    bad.update(baseline_votes=[False] * 3, candidate_votes=[True] * 3)
    with record_contract.writer(target) as append:
        append(bad)
        append(_single_draw_row(1, True, True))
    with pytest.raises(ConfigError, match="票数不完整"):
        comparisons.complete(target)


def test_flip_two_still_requires_full_votes(single_draw_comparison):
    """旧协议专项比较回归：分歧题缺补验票仍拒绝完成。"""
    target = comparisons.register("comparison-flip-two", data_ref="d-test", pack_ref="pack-test",
                                  comparison={"baseline": {"recipe": "LR"}, "candidate": {"recipe": "GBDT"}},
                                  change="旧协议缺票", protocol={"flip_extra_rounds": 2})
    with record_contract.writer(target) as append:
        append({"case_id": "0", "status": "ok", "baseline_correct": False, "candidate_correct": True})
        append(_single_draw_row(1, True, True))
    with pytest.raises(ConfigError, match="独立补验"):
        comparisons.complete(target)
