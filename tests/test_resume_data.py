"""真实 runner 的中断续跑验收；模型使用可计数替身，不请求外部服务。"""
import json

import pytest

from src.config import ConfigError
from src.iteration import branch_packs, experiment, runner, versions
from test_iteration import priv, _case, _mk_dev_exp, _write
from test_branches import tree
from leakage_support import isolated_legacy_provenance

pytestmark = pytest.mark.usefixtures('isolated_legacy_provenance')


def _gen_run(priv, monkeypatch, *, retrieval=False):
    if retrieval:
        path = priv / "generators/g-0001/config.json"
        config = json.loads(path.read_text())
        config["retriever"]["enabled"] = True
        _write(path, config)
    exp = _mk_dev_exp(priv, monkeypatch)
    calls = []
    control = {"interrupt": True}

    class Generator:
        def __init__(self, settings, config, *args, **kwargs):
            self.model = config["llm"]["model"]

        def generate(self, case, forced_reply=True):
            calls.append(("generation", case["case_id"], self.model))
            if control["interrupt"] and case["case_id"] == "d1":
                raise KeyboardInterrupt("simulated interruption")
            return {"replies": [self.model], "latency_ms": 1}

    class Scorer:
        def is_ai(self, case, replies):
            calls.append(("judge", case["case_id"], replies[0]))
            return replies == ["m1"]  # 每题都需原有的两轮独立补测

    monkeypatch.setattr(runner, "ReplyGenerator", Generator)
    monkeypatch.setattr(runner, "build_judge", lambda *args: Scorer())
    monkeypatch.setattr(experiment, "_verify_trigger", lambda *args: None)
    from src import llm
    monkeypatch.setattr(llm, "build_clients", lambda *args: {})
    return exp, calls, control


def _mutate(rows, mutation):
    if mutation == "context":
        rows[0]["context"][0]["text"] = "同一个 ID，新的上下文"
    elif mutation == "pending":
        rows[1]["context"][0]["text"] = "尚未完成的题也不能换"
    elif mutation == "human_reply":
        rows[0]["human_reply"] = ["新的真人答案"]
    elif mutation == "metadata":
        rows[0]["source_chat_id"] = "changed-source"
    elif mutation == "chat_type":
        rows[0]["chat_type"] = "group"
    elif mutation == "add":
        rows.append({**_case("new"), "ai_replies": ["AI"]})
    elif mutation == "remove":
        rows.pop(0)
    elif mutation == "order":
        rows.reverse()
    elif mutation == "ai_replies":
        rows[0]["ai_replies"] = ["新的模型答案"]
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize("mutation", ["context", "pending", "human_reply", "metadata", "chat_type",
                                      "add", "remove", "order"])
def test_changed_case_cannot_be_skipped_on_resume(priv, monkeypatch, mutation, workers):
    exp, calls, control = _gen_run(priv, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.run_gen_experiment(exp)
    path = priv / "data/d-0001/dev_pool.jsonl"
    original = path.read_bytes()
    checkpoint = (exp / "cases.jsonl").read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    _mutate(rows, mutation)
    _write(path, "\n".join(json.dumps(row) for row in rows))
    control["interrupt"] = False
    before = len(calls)
    with pytest.raises(ConfigError, match="数据|指纹"):
        runner.run_gen_experiment(exp, workers=workers)
    assert len(calls) == before
    assert (exp / "cases.jsonl").read_bytes() == checkpoint
    path.write_bytes(original)
    runner.run_gen_experiment(exp, workers=workers)
    assert (exp / "cases.jsonl").read_bytes().startswith(checkpoint)
    assert len([c for c in calls if c[1] == "d0"]) == 12  # 2 侧 × 3 轮 × 生成/评分
    assert experiment.state_of(exp)["metrics"]["attempted"] == 6
    traces = [json.loads(path.read_text()) for path in (exp / "traces").glob("*.json")]
    first = next(trace for trace in traces if trace["case_id"] == "d0")
    assert sorted(op["round"] for op in first["operations"]) == [0] * 4 + [1] * 4 + [2] * 4


@pytest.mark.parametrize('workers', [0, 17, True, 1.5, '4'])
def test_generator_rejects_invalid_workers_before_requests(priv, monkeypatch, workers):
    exp, calls, _ = _gen_run(priv, monkeypatch)
    with pytest.raises(ConfigError, match='workers'):
        runner.run_gen_experiment(exp, workers=workers)
    assert calls == []


def test_generator_parallelism_is_bounded_and_rounds_remain_isolated(priv, monkeypatch):
    from threading import Barrier, Lock, get_ident
    from src import cache, tracing
    exp, calls, control = _gen_run(priv, monkeypatch)
    control['interrupt'] = False
    original = runner.ReplyGenerator
    barrier, lock = Barrier(2), Lock()
    observed = set()

    class Generator(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.owner = get_ident()

        def generate(self, case, forced_reply=True):
            assert self.owner == get_ident()
            context = cache._scope.get()['context']
            assert context['case_id'] == case['case_id']
            tracing.note('test_generation', dict(case_id=case['case_id'], round=context['round']))
            if self.model == 'm1' and context['round'] == 0:
                with lock:
                    observed.add(get_ident())
                barrier.wait(timeout=10)  # 串行执行或错误转发 workers 会失败。
            return super().generate(case, forced_reply)

    monkeypatch.setattr(runner, 'ReplyGenerator', Generator)
    runner.run_gen_experiment(exp, workers=2)
    state = experiment.state_of(exp)
    assert len(observed) == 2
    assert state['progress']['active_cases'] == {}
    assert state['metrics']['pairs'] == state['metrics']['net_win_confirmed'] == 6
    assert len(calls) == 6 * 12
    records, _ = runner._final_records(exp / 'cases.jsonl')
    for cid, record in records.items():
        trace = tracing.read(exp, record['trace_ref'])
        assert trace['case_id'] == cid and trace['status'] == 'ok'
        assert sorted((op['round'], op['kind'], op['branch']) for op in trace['operations']) == sorted(
            (rnd, kind, branch) for rnd in range(3)
            for kind in ('generation', 'judge') for branch in ('baseline', 'candidate'))
        for op in trace['operations']:
            for event in op['events']:
                if event['kind'] == 'test_generation':
                    assert event['data'] == dict(case_id=cid, round=op['round'])


def test_generator_workers_share_one_pinned_history_index(priv, monkeypatch):
    from src.generator import generator
    exp, _, control = _gen_run(priv, monkeypatch, retrieval=True)
    control['interrupt'] = False
    pool = priv / 'data/d-0001/fewshot_pool.jsonl'
    pool.with_name('purposes.json').write_text('{}')
    loaded, retrievers = [], []

    class Retriever:
        history_policy = 'complete_before_input_v1'

        def __init__(self, path):
            loaded.append(path)

        def is_approved(self):
            return True

    original = runner.ReplyGenerator

    class Generator(original):
        def __init__(self, *args, pool_path, **kwargs):
            super().__init__(*args, **kwargs)
            retrievers.append(generator.PersonaFewShotRetriever(path=pool_path))

    monkeypatch.setattr(generator, 'PersonaFewShotRetriever', Retriever)
    monkeypatch.setattr(runner, 'ReplyGenerator', Generator)
    runner.run_gen_experiment(exp, workers=4)
    assert loaded == [pool.resolve()]
    assert len(retrievers) >= 4  # Preflight and multiple thread-local generator pairs.
    assert len({id(value) for value in retrievers}) == 1
    assert generator.PersonaFewShotRetriever is Retriever
    assert experiment.state_of(exp)['metrics']['pairs'] == 6


@pytest.mark.parametrize("filename", ["fewshot_pool.jsonl", "report.json"])
def test_pool_dependencies_block_resume_and_restore_keeps_successes(priv, monkeypatch, filename):
    exp, calls, control = _gen_run(priv, monkeypatch, retrieval=True)
    with pytest.raises(KeyboardInterrupt):
        runner.run_gen_experiment(exp)
    checkpoint = (exp / "cases.jsonl").read_bytes()
    path = priv / "data/d-0001" / filename
    original = path.read_bytes()
    _write(path, {"review_status": "pending"} if filename == "report.json" else '{"id":"new-shot"}\n')
    before = len(calls)
    with pytest.raises(ConfigError, match="fewshot"):
        runner.run_gen_experiment(exp)
    assert len(calls) == before and (exp / "cases.jsonl").read_bytes() == checkpoint
    path.write_bytes(original)
    control["interrupt"] = False
    runner.run_gen_experiment(exp)
    assert len([c for c in calls if c[1] == "d0"]) == 12


def test_unchanged_effective_inputs_reuse_checkpoint(priv, monkeypatch):
    exp, calls, control = _gen_run(priv, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.run_gen_experiment(exp)
    # 行格式/JSON 键顺序不改变题目；未使用的固定集和已关闭的检索池不影响本轮。
    data = priv / "data/d-0001"
    rows = [json.loads(line) for line in (data / "dev_pool.jsonl").read_text().splitlines()]
    _write(data / "dev_pool.jsonl", "\n\n".join(json.dumps(row, sort_keys=True) for row in rows))
    _write(data / "fixed_test.jsonl", "")
    _write(data / "fewshot_pool.jsonl", '{"id":"unused"}')
    control["interrupt"] = False
    runner.run_gen_experiment(exp)
    assert len([c for c in calls if c[1] == "d0"]) == 12
    assert experiment.state_of(exp)["metrics"]["pairs"] == 6


def _judge_run(priv, monkeypatch):
    path = priv / "judge_eval/pack-resume/pack.json"
    _write(path, {"c0_gen_version": "g-0001", "rows": [
        {**_case(f"d{i}", "private"), "ai_replies": ["AI"]} for i in range(4)]})
    exp = experiment.create_judge_eval_experiment("development", "pack-resume", {"llm": {"model": "m2"}},
                                                change="test", exp_id="judge-resume")
    calls, control = [], {"interrupt": True}

    class Scorer:
        def __init__(self, info):
            self.model = info["config"]["llm"]["model"]

        def is_ai(self, case, replies):
            calls.append((case["case_id"], self.model))
            if control["interrupt"] and case["case_id"] == "d1":
                raise KeyboardInterrupt("simulated interruption")
            return self.model == "m1"

    monkeypatch.setattr(runner, "build_judge", lambda settings, info: Scorer(info))
    return exp, path, calls, control


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("mutation", ["context", "human_reply", "ai_replies", "add", "remove", "order"])
def test_judge_resume_validates_frozen_pack(priv, monkeypatch, mutation, workers):
    exp, path, calls, control = _judge_run(priv, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.run_judge_experiment(exp)
    original = path.read_bytes()
    checkpoint = (exp / "cases.jsonl").read_bytes()
    pack = json.loads(original)
    _mutate(pack["rows"], mutation)
    _write(path, pack)
    before = len(calls)
    with pytest.raises(ConfigError, match="指纹"):
        runner.run_judge_experiment(exp, workers=workers)
    assert len(calls) == before and (exp / "cases.jsonl").read_bytes() == checkpoint
    path.write_bytes(original)
    control["interrupt"] = False
    runner.run_judge_experiment(exp, workers=workers)
    assert (exp / "cases.jsonl").read_bytes().startswith(checkpoint)
    assert calls.count(("d0", "m1")) == calls.count(("d0", "m2")) == 3
    assert experiment.state_of(exp)["metrics"]["pairs"] == 4


@pytest.mark.parametrize("kind", ["gen", "judge"])
def test_creation_freezes_data_before_first_call(priv, monkeypatch, kind):
    if kind == "gen":
        exp, calls, _ = _gen_run(priv, monkeypatch)
        path = priv / "data/d-0001/dev_pool.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        _mutate(rows, "context")
        _write(path, "\n".join(json.dumps(row) for row in rows))
        run = runner.run_gen_experiment
    else:
        exp, path, calls, _ = _judge_run(priv, monkeypatch)
        pack = json.loads(path.read_text())
        _mutate(pack["rows"], "context")
        _write(path, pack)
        run = runner.run_judge_experiment
    with pytest.raises(ConfigError, match="指纹"):
        run(exp)
    assert not calls and not (exp / "cases.jsonl").exists()


@pytest.mark.parametrize("kind", ["gen", "judge"])
@pytest.mark.parametrize("existing_results", [False, True])
def test_legacy_run_needs_evidence_before_reusing_results(priv, monkeypatch, kind, existing_results):
    if kind == "gen":
        exp, calls, control = _gen_run(priv, monkeypatch)
        run, field = runner.run_gen_experiment, "data_snapshot"
    else:
        exp, _, calls, control = _judge_run(priv, monkeypatch)
        run, field = runner.run_judge_experiment, "pack_sha256"
    spec = experiment.spec_of(exp)
    if existing_results:
        with pytest.raises(KeyboardInterrupt):
            run(exp)
        (exp / "data_snapshot.json").unlink()  # 模拟上线保护前遗留的断点
    spec.pop(field)
    _write(exp / "spec.json", spec)
    if existing_results:
        before = len(calls)
        checkpoint = (exp / "cases.jsonl").read_bytes()
        with pytest.raises(ConfigError, match="缺少可验证的数据指纹"):
            run(exp)
        assert len(calls) == before and (exp / "cases.jsonl").read_bytes() == checkpoint
        assert not (exp / "data_snapshot.json").exists()  # 不能事后补造来源
    else:
        with pytest.raises(KeyboardInterrupt):
            run(exp)
        snapshot = (exp / "data_snapshot.json").read_bytes()
        control["interrupt"] = False
        run(exp)
        assert (exp / "data_snapshot.json").read_bytes() == snapshot


def test_legacy_judge_with_frozen_hash_keeps_valid_checkpoint(priv, monkeypatch):
    exp, _, calls, control = _judge_run(priv, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.run_judge_experiment(exp)
    (exp / "data_snapshot.json").unlink()
    control["interrupt"] = False
    runner.run_judge_experiment(exp)
    assert calls.count(("d0", "m1")) == calls.count(("d0", "m2")) == 3


def test_pack_rebuild_checks_pool_before_skipping_saved_rows(tree, monkeypatch):
    from src.config import sha256_file
    ptr = versions.load_pointers()
    cfg = versions.current_generator("production")["config"]
    ptr["production_gen"] = versions.create_generator_version(
        {**cfg, "retriever": {"enabled": True}, "llm": {"model": "new"}}, "d-0001")
    source = tree / "judge_eval/pack-calibration-test/pack.json"
    ref = branch_packs.prepare({"ref": "pack-calibration-test", "sha256": sha256_file(source)},
                               ptr, "development")
    directory = tree / "judge_eval" / ref
    calls, control = [], {"interrupt": True}

    class Generator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, case, forced_reply=True):
            calls.append(case["case_id"])
            if control["interrupt"] and case["case_id"] == "2":
                raise KeyboardInterrupt("simulated interruption")
            return {"replies": ["new"]}

    monkeypatch.setattr(branch_packs, "ReplyGenerator", Generator)
    monkeypatch.setattr(branch_packs, "build_clients", lambda *args: {})
    with pytest.raises(KeyboardInterrupt):
        branch_packs.build(ref)
    checkpoint = (directory / "building.json").read_bytes()
    pool = tree / "data/d-0001/fewshot_pool.jsonl"
    original = pool.read_bytes()
    pool.write_text('{"id":"changed"}')
    with pytest.raises(ConfigError, match="fewshot"):
        branch_packs.build(ref)
    assert len(calls) == 3 and (directory / "building.json").read_bytes() == checkpoint
    assert not (directory / "pack.json").exists()
    pool.write_bytes(original)
    control["interrupt"] = False
    branch_packs.build(ref)
    pack = json.loads((directory / "pack.json").read_text())
    assert calls.count("0") == calls.count("1") == 1 and len(pack["rows"]) == 6
