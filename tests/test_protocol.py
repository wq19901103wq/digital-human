"""协议层测试：符号检验、数据集构成、采用门槛。纯逻辑，无需网络。"""
from __future__ import annotations

import pytest

from src.config import dataset_plan, load_settings
from src.iteration import protocol


def test_sign_test_perfect_separation():
    assert protocol.sign_test_two_sided(10, 0) == pytest.approx(0.001953, abs=1e-5)


def test_sign_test_all_ties():
    assert protocol.sign_test_two_sided(0, 0) == 1.0


def test_sign_test_symmetric():
    assert protocol.sign_test_two_sided(8, 2) == protocol.sign_test_two_sided(2, 8)


def test_sign_test_known_value():
    # n=20, k=5: 2*(C(20,0)+...+C(20,5))/2^20
    assert protocol.sign_test_two_sided(15, 5) == pytest.approx(0.0414, abs=1e-3)


def test_dataset_plan_default_75_percent_group():
    settings = load_settings()
    plan = dataset_plan(settings, "development")
    assert plan["total"] == 1000
    assert plan["group"] == 750
    assert plan["private"] == 250
    plan_f = dataset_plan(settings, "fixed_test")
    assert plan_f["group"] == 750




def _protocol() -> dict:
    s = load_settings()["evaluation"]
    return {"max_failure_rate": s.get("max_failure_rate", 0.01),
            "dev_min_net_win_rate": s["adoption"]["dev_min_net_win_rate"],
            "formal_min_net_win_rate": s["adoption"]["formal_min_net_win_rate"]}


def _metrics(net: int) -> dict:
    wins = max(net, 0)
    losses = max(-net, 0)
    return {
        "pairs": 1000,
        "identified_baseline": 700,
        "identified_candidate": 700 - net,
        "wins": wins,
        "losses": losses,
        "wins_confirmed": wins,
        "losses_confirmed": losses,
        "net_win_confirmed": net,
        "failure_rate": 0.0,
        "failures": 0,
        "attempted": 1000,
    }


def test_decide_dev_merge_at_threshold():
    settings = load_settings()
    v = protocol.decide(_metrics(10), "development", _protocol())
    assert v["verdict"] == "merge_to_iteration_baseline"


def test_decide_dev_below_threshold_observes():
    # 历史实验使用冻结的旧门槛，不随新设置改写结论。
    v = protocol.decide(_metrics(9), "development", {**_protocol(), "dev_min_net_win_rate": .01})
    assert v["verdict"] == "observe"


@pytest.mark.parametrize("net, expected", [(9, "merge_to_iteration_baseline"),
    (1, "merge_to_iteration_baseline"), (0, "observe"), (-1, "reject")])
def test_layered_development_requires_positive_net(net, expected):
    proto = {**_protocol(), "gate_schema": 2}
    assert protocol.decide(_metrics(net), "development", proto)["verdict"] == expected


def test_decide_dev_net_negative_rejects():
    settings = load_settings()
    v = protocol.decide(_metrics(-1), "development", _protocol())
    assert v["verdict"] == "reject"


@pytest.mark.parametrize('net,expected', [(1, 'merge_to_iteration_baseline'),
    (0, 'observe'), (-1, 'reject')])
def test_zero_development_threshold_still_requires_positive_net(net, expected):
    proto = {**_protocol(), 'gate_schema': 3, 'dev_min_net_win_rate': 0.,
             'fixed_entry_min_net_win_rate': .005, 'formal_min_net_win_rate': .005}
    assert protocol.decide(_metrics(net), 'development', proto)['verdict'] == expected
    assert protocol.decide(_metrics(1), 'fixed_test', proto)['verdict'] == 'reject'


@pytest.mark.parametrize('dataset,pass_verdict', [
    ('development', 'merge_to_iteration_baseline'), ('fixed_test', 'adopt')])
@pytest.mark.parametrize('pairs', [999, 1000, 1001])
def test_configurable_five_per_thousand_boundary(dataset, pass_verdict, pairs):
    from math import ceil
    proto = {**_protocol(), 'gate_schema': 3, 'dev_min_net_win_rate': .005,
             'formal_min_net_win_rate': .005}
    needed = ceil(pairs * .005)
    for net in (needed - 1, needed):
        metrics = {**_metrics(net), 'pairs': pairs, 'attempted': pairs}
        verdict = protocol.decide(metrics, dataset, proto)
        assert (verdict['verdict'] == pass_verdict) is (net == needed)
        if dataset == 'fixed_test' and net == needed:
            assert '0.5%' in verdict['reason']


def test_new_protocol_freezes_configured_thresholds():
    from src.iteration.experiment import _protocol_snapshot
    settings = load_settings()
    snapshot = _protocol_snapshot(settings)
    assert snapshot['gate_schema'] == 3
    # settings-v6 现行口径：开发晋级为净胜严格大于零（0.0），固定准入/正式采用 ≥ +5/1000
    assert snapshot['dev_min_net_win_rate'] == 0.0
    for key in ('fixed_entry_min_net_win_rate', 'formal_min_net_win_rate'):
        assert snapshot[key] == .005
    settings['evaluation']['adoption']['dev_min_net_win_rate'] = .1
    assert snapshot['dev_min_net_win_rate'] == 0.0


def test_decide_formal_requires_both_conditions():
    settings = load_settings()
    # 双条件：识别数严格下降 且 确认净胜 ≥ formal_min_net_win（默认10）
    assert protocol.decide(_metrics(10), "fixed_test", _protocol())["verdict"] == "adopt"
    # 只少识别 1 题、净胜 1 → 拒绝（波动不算改善，SOP §8.2）
    m = _metrics(1)
    v = protocol.decide(m, "fixed_test", _protocol())
    assert v["verdict"] == "reject"
    assert "净胜" in v["reason"]
    # 净胜够但识别数未下降 → 拒绝
    m = _metrics(10)
    m["identified_candidate"] = m["identified_baseline"]
    assert protocol.decide(m, "fixed_test", _protocol())["verdict"] == "reject"
