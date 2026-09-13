"""四对象架构测试：版本、实验、晋升。全部用 monkeypatched 私有根，不碰真实 private/。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.digital_human.config import ConfigError, load_settings
from src.digital_human.iteration import experiment, promote, runner, versions


# ---------- fixtures：最小私有树 ----------

def _write(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    elif isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _case(cid: str, chat_type: str = "group") -> dict:
    return {
        "case_id": cid, "chat_type": chat_type,
        "context": [{"sender": "甲", "text": "在吗"}],
        "human_reply": ["在"], "source_message_id": f"c:{cid}",
    }


@pytest.fixture()
def priv(tmp_path, monkeypatch):
    """最小可用私有树：数据版本 d-0001 + g-0001 + j-0001 + 指针。"""
    priv_root = tmp_path / "private"
    data = priv_root / "data" / "d-0001"
    _write(data / "manifest.json", {"id": "d-0001", "seed": 42})
    _write(data / "dev_pool.jsonl", "\n".join(json.dumps(_case(f"d{i}", "group" if i % 2 else "private")) for i in range(6)))
    _write(data / "fixed_test.jsonl", "\n".join(json.dumps(_case(f"f{i}", "group" if i % 2 else "private")) for i in range(6)))
    _write(data / "persona.md", "<instructions>测试人格</instructions>")
    (data / "scenarios").mkdir(parents=True)
    _write(data / "scenarios" / "friend.md", "# friend")
    _write(data / "report.json", {"total": 9000, "review_status": "approved",
                                  "examples_sha256": "x"})  # 规模准入用
    _write(data / "fewshot_pool.jsonl", "")

    gen_cfg = {"llm": {"model": "m1"}, "retriever": {"enabled": False},
               "max_shots_per_case": 3, "shots_char_budget": 100}
    gdir = priv_root / "generators" / "g-0001"
    _write(gdir / "config.json", {**gen_cfg, "data_version": "d-0001", "persona_sha256": "x"})
    _write(gdir / "persona.md", "<instructions>测试人格</instructions>")  # 与工作区一致
    (gdir / "scenarios").mkdir(parents=True)
    _write(gdir / "scenarios" / "friend.md", "# friend")  # 与工作区一致

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


# ---------- 版本 ----------

def test_version_write_once(priv):
    with pytest.raises(Exception, match="不可变|已存在"):
        versions.create_data_version("d-0001", {})
    with pytest.raises(Exception, match="已存在"):
        versions.create_generator_version({"llm": {}}, "d-0001", version_dir_name="g-0001")
    # 版本簿记字段不得进入行为配置
    with pytest.raises(Exception, match="簿记"):
        versions.create_generator_version({"baseline_id": "x"}, "d-0001")


def test_generator_version_self_contained(priv):
    gid = versions.create_generator_version(
        {"llm": {"model": "m2"}, "retriever": {"enabled": False}}, "d-0001")
    g = versions.load_generator(gid)
    assert g["config"]["llm"]["model"] == "m2"
    assert (g["dir"] / "persona.md").exists() and (g["dir"] / "scenarios" / "friend.md").exists()
    # 快照与数据版本工作区独立：改工作区不影响版本
    (priv / "data" / "d-0001" / "persona.md").write_text("被改", encoding="utf-8")
    assert (g["dir"] / "persona.md").read_text(encoding="utf-8") != "被改"


# ---------- 实验 ----------

def test_create_dev_experiment(priv, monkeypatch):
    # 冒烟跳过池规模/构成准入；候选创建即得普通版本号
    exp_dir = experiment.create_gen_experiment(
        "development", "测试改动：llm.model m1→m2", {"llm": {"model": "m2"}}, limit=5)
    spec = experiment.spec_of(exp_dir)
    assert spec["baseline_ref"] == "g-0001"
    assert spec["candidate_ref"].startswith("g-") and spec["candidate_ref"] != "g-0001"
    assert (priv / "generators" / spec["candidate_ref"] / "config.json").exists()
    assert spec["config_diff"] == ["llm.model: 'm1' -> 'm2'"]
    assert spec["smoke"] is True


def test_empty_diff_rule(priv):
    """SOP §2.3 统一规则：冒烟可无改动（接线验证），完整实验必须拒绝空 diff。"""
    experiment.create_gen_experiment("development", "冒烟接线", {}, limit=5)  # 冒烟允许
    import src.digital_human.iteration.experiment as exp_mod
    settings = load_settings()
    settings["evaluation"]["development"]["total"] = 6
    settings["evaluation"]["development"]["group_ratio"] = 0.5
    monkey = None  # 直接临时替换
    orig = exp_mod.load_settings
    exp_mod.load_settings = lambda: settings
    try:
        with pytest.raises(Exception, match="没有可迭代的改动"):
            experiment.create_gen_experiment("development", "无改动", {})
    finally:
        exp_mod.load_settings = orig


def test_prompt_change_is_a_valid_single_change(priv, monkeypatch):
    """提示词迭代：配置不变、工作区 persona 相对基线快照有更新 → 合法唯一改动。"""
    (priv / "data" / "d-0001" / "persona.md").write_text("<instructions>新人格</instructions>",
                                                         encoding="utf-8")
    exp = experiment.create_gen_experiment("development", "人格语气迭代", {}, limit=5)
    spec = experiment.spec_of(exp)
    assert any(d.startswith("asset:persona.md") for d in spec["config_diff"])


def test_formal_experiment_rejects_override_and_smoke(priv):
    with pytest.raises(Exception, match="禁止 override"):
        experiment.create_gen_experiment("fixed_test", "x", {"llm": {"model": "m2"}})
    with pytest.raises(Exception, match="禁止冒烟"):
        experiment.create_gen_experiment("fixed_test", "x", {}, limit=5)


def test_one_shot(priv, monkeypatch):
    # 先推进开发基线，fixed 轮才有可比较的候选（iteration ≠ production）
    e0 = _mk_dev_exp(priv, monkeypatch, model="m0")
    experiment.finish(e0, {}, {"verdict": "merge_to_iteration_baseline", "reason": "x"})
    promote.promote_gen(e0.name)
    e1 = experiment.create_gen_experiment("fixed_test", "改动A", exp_id="fixed-A")
    experiment.finish(e1, {}, {"verdict": "reject", "reason": "x"})
    with pytest.raises(Exception, match="one-shot"):
        experiment.create_gen_experiment("fixed_test", "改动A（重试）")
    # 开发基线再前进 → 新候选可测；可恢复失败必须恢复原实验
    e_dev = _mk_dev_exp(priv, monkeypatch, model="mB")
    experiment.finish(e_dev, {}, {"verdict": "merge_to_iteration_baseline", "reason": "z"})
    promote.promote_gen(e_dev.name)
    e2 = experiment.create_gen_experiment("fixed_test", "改动B", exp_id="fixed-B")
    experiment.finish(e2, {}, {"verdict": "experiment_incomplete", "reason": "y"})
    with pytest.raises(Exception, match="恢复原实验"):
        experiment.create_gen_experiment("fixed_test", "改动B（重试）")


def test_full_run_gates(priv):
    """非冒烟：构成与池规模按 settings 强制（小 fixture 必然拒绝）。"""
    with pytest.raises(Exception, match="未达准入"):
        experiment.create_gen_experiment("development", "改动", {"llm": {"model": "m2"}})
    # 池规模单独验证
    (priv / "data" / "d-0001" / "report.json").write_text(
        json.dumps({"total": 10}), encoding="utf-8")
    settings = load_settings()
    settings["evaluation"]["development"]["total"] = 6
    settings["evaluation"]["development"]["group_ratio"] = 0.5
    import src.digital_human.iteration.experiment as exp_mod
    orig = exp_mod.load_settings
    exp_mod.load_settings = lambda: settings
    try:
        with pytest.raises(Exception, match="5000"):
            experiment.create_gen_experiment("development", "改动", {"llm": {"model": "m2"}})
    finally:
        exp_mod.load_settings = orig


# ---------- 晋升 ----------


def _use_small_settings(monkeypatch):
    """非冒烟实验的构成准入：把 settings 调到 fixture 的体量（6 题、3 群聊）。"""
    settings = load_settings()
    for ds in ("development", "fixed_test"):
        settings["evaluation"][ds]["total"] = 6
        settings["evaluation"][ds]["group_ratio"] = 0.5
    import src.digital_human.iteration.experiment as exp_mod
    monkeypatch.setattr(exp_mod, "load_settings", lambda: settings)


def _mk_dev_exp(priv, monkeypatch, model="m2"):
    _use_small_settings(monkeypatch)
    return experiment.create_gen_experiment("development", f"改动→{model}", {"llm": {"model": model}},
                                            exp_id=f"dev-{model}")


def _finish_exp(exp_dir, verdict):
    experiment.finish(exp_dir, {"pairs": 1000, "net_win_confirmed": 10,
                                "identified_baseline": 700, "identified_candidate": 680},
                      {"verdict": verdict, "reason": "test"})


def test_promote_dev_advances_iteration_only(priv, monkeypatch):
    exp_dir = _mk_dev_exp(priv, monkeypatch)
    _finish_exp(exp_dir, "merge_to_iteration_baseline")
    pointers = promote.promote_gen(exp_dir.name)
    assert pointers["iteration_gen"] == experiment.spec_of(exp_dir)["candidate_ref"]
    assert pointers["production_gen"] == "g-0001"
    # 候选版本本来就是 generators/ 下的普通版本，晋升无拷贝
    assert (priv / "generators" / pointers["iteration_gen"] / "config.json").exists()


def test_promote_rejects_wrong_verdict_and_stale(priv, monkeypatch):
    exp_dir = _mk_dev_exp(priv, monkeypatch)
    _finish_exp(exp_dir, "observe")
    with pytest.raises(Exception, match="禁止后门晋升"):
        promote.promote_gen(exp_dir.name)
    # 同代基线上的两个实验都完赛；晋升其一后，另一个因基线过期被拒
    exp_ok = _mk_dev_exp(priv, monkeypatch, model="m3")
    exp_stale = _mk_dev_exp(priv, monkeypatch, model="m4")
    _finish_exp(exp_ok, "merge_to_iteration_baseline")
    _finish_exp(exp_stale, "merge_to_iteration_baseline")
    promote.promote_gen(exp_ok.name)  # iteration → g-0003
    with pytest.raises(Exception, match="已过期"):
        promote.promote_gen(exp_stale.name)


def test_promote_formal_production(priv, monkeypatch):
    exp_dev = _mk_dev_exp(priv, monkeypatch)
    _finish_exp(exp_dev, "merge_to_iteration_baseline")
    promote.promote_gen(exp_dev.name)  # iteration = g-0002
    exp_formal = experiment.create_gen_experiment("fixed_test", "正式验证 g-0002")
    assert experiment.spec_of(exp_formal)["candidate_source_ref"] == "g-0002"
    _finish_exp(exp_formal, "adopt")
    pointers = promote.promote_gen(exp_formal.name)
    assert pointers["production_gen"] == "g-0002"


def test_promote_judge_dev_then_formal(priv):
    # 开发轮：候选 = 新 Judge 版本
    (priv / "judge_eval" / "pack-cal1").mkdir(parents=True, exist_ok=True)
    _write(priv / "judge_eval" / "pack-cal1" / "pack.json", {"c0_gen_version": "g-0001", "rows": [
        {"case_id": f"c{i}", "chat_type": "group", "context": [], "human_reply": ["在"],
         "ai_replies": ["AI回复样本"]} for i in range(6)]})
    exp_dev = experiment.create_judge_eval_experiment(
        "development", "pack-cal1", {"llm": {"model": "m9"}}, change="换 judge 模型 m1→m9",
        exp_id="judge-dev-1")
    spec = experiment.spec_of(exp_dev)
    assert spec["candidate_ref"].startswith("j-")
    experiment.finish(exp_dev, {"pairs": 6, "identified_baseline": 3, "identified_candidate": 5,
                                "wins": 2, "losses": 0, "net_win_confirmed": 2},
                      {"verdict": "merge_to_iteration_baseline", "reason": "净胜2/6"})
    pointers = promote.promote_judge(exp_dev.name)
    assert pointers["iteration_judge"] == spec["candidate_ref"]
    assert pointers["production_judge"] == "j-0001"   # 生产不动
    # 固定轮：禁止 override；候选 = 开发 Judge 基线
    (priv / "judge_eval" / "pack-validation-1").mkdir(parents=True, exist_ok=True)
    _write(priv / "judge_eval" / "pack-validation-1" / "pack.json", {"c0_gen_version": "g-0001", "rows": []})
    with pytest.raises(Exception, match="禁止 override"):
        experiment.create_judge_eval_experiment("fixed_test", "pack-validation-1", {"llm": {"model": "mX"}}, "x")
    exp_f = experiment.create_judge_eval_experiment("fixed_test", "pack-validation-1", {}, "正式验证",
                                                    exp_id="judge-fixed-1")
    assert experiment.spec_of(exp_f)["candidate_ref"] == spec["candidate_ref"]
    experiment.finish(exp_f, {"pairs": 6, "identified_baseline": 3, "identified_candidate": 5,
                              "wins": 2, "losses": 0, "net_win_confirmed": 2},
                      {"verdict": "adopt", "reason": "双条件"})
    pointers = promote.promote_judge(exp_f.name)
    assert pointers["production_judge"] == spec["candidate_ref"]


# ---------- 统计（runner 纯逻辑） ----------

def test_final_records_last_state():
    import src.digital_human.iteration.runner as r
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cases.jsonl"
        rows = ([dict(case_id=f"c{i}", status="ok", chat_type="group",
                      identified_baseline=True, identified_candidate=True,
                      baseline_latency_ms=1.0, candidate_latency_ms=1.0) for i in range(9)]
                + [{"case_id": f"c{i}", "status": "failed", "reason": "x"} for i in range(9, 10)]
                + [dict(case_id="c9", status="ok", chat_type="group",
                        identified_baseline=True, identified_candidate=True,
                        baseline_latency_ms=1.0, candidate_latency_ms=1.0)])
        p.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
        final, retries = r._final_records(p)
        assert retries == 1
        assert sum(1 for x in final.values() if x["status"] == "failed") == 0


# ---------- 失败恢复（#1 回归） ----------

def test_incomplete_keeps_running_and_failed_cases_retryable(priv, monkeypatch):
    exp = _mk_dev_exp(priv, monkeypatch, model="mX")
    # 模拟跑过一轮：3 题成功、1 题失败
    cases_path = exp / "cases.jsonl"
    ok_row = {"case_id": "c0", "status": "ok", "chat_type": "group",
              "identified_baseline": True, "identified_candidate": True,
              "baseline_latency_ms": 1.0, "candidate_latency_ms": 1.0}
    rows = [dict(ok_row, case_id=f"c{i}") for i in range(3)]
    rows.append({"case_id": "c3", "status": "failed", "reason": "boom"})
    cases_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    experiment.finish(exp, {"attempted": 4, "failures": 1, "failure_rate": 0.25},
                      {"verdict": "experiment_incomplete", "reason": "超上限"})
    state = experiment.state_of(exp)
    assert state["status"] == "running", "incomplete 不算完赛，必须可恢复"

    # 恢复时的跳过集合只含成功题：失败题 c3 必须重试
    import src.digital_human.iteration.runner as r
    final, _ = r._final_records(cases_path)
    done = {cid: rec for cid, rec in final.items() if rec.get("status") == "ok"}
    assert "c3" not in done and len(done) == 3


def test_judge_direction_and_oneshot_kind(priv):
    from src.digital_human.iteration import protocol
    proto = {"max_failure_rate": 0.01, "dev_min_net_win_rate": 0.01, "formal_min_net_win_rate": 0.01}
    m = {"pairs": 100, "attempted": 100, "failures": 0, "failure_rate": 0.0,
         "identified_baseline": 40, "identified_candidate": 60,
         "wins": 20, "losses": 0, "wins_confirmed": 20, "losses_confirmed": 0,
         "net_win_confirmed": 20}
    # Judge 方向：识别数增多 + 净胜 → 通过
    assert protocol.decide(m, "fixed_test", proto, higher_is_better=True)["verdict"] == "adopt"
    # 生成器方向：同一数字必须拒绝
    assert protocol.decide(m, "fixed_test", proto)["verdict"] == "reject"
