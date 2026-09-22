"""并行实验分支：不可变提案/轮次，生产晋升后在新基线上重放改动。"""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path

from ..config import ConfigError, load_settings, sha256_file, valid_name
from . import datasets, experiment, gates, protocol, versions
from .storage import write_json

_MISSING = object()
_TERMINAL = {"rejected", "superseded", "promoted", "integrated", "conflict", "blocked", "development_accepted"}


_name = valid_name


def _root(name: str) -> Path:
    return versions.PRIVATE / "branches" / _name(name, 80)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def basis() -> dict:
    ptr = versions.load_pointers()
    return {key: ptr[key] for key in ("data", "production_gen", "production_judge")}


def _version(kind: str, ref: str) -> dict:
    return versions.load_generator(ref) if kind == "gen" else versions.judge_dir(ref)


def _snapshot(kind: str, ref: str) -> dict:
    info = _version(kind, ref)
    cfg = copy.deepcopy(info["config"])
    if kind == "gen":
        cfg.pop("data_version", None)
        cfg.pop("persona_sha256", None)
        names = ["persona.md", *[str(p.relative_to(info["dir"]))
                                 for p in sorted((info["dir"] / "scenarios").rglob("*")) if p.is_file()]]
        if (info["dir"] / "learning.json").is_file():
            names.append("learning.json")
        names.extend(str(p.relative_to(info['dir']))
                     for p in sorted((info['dir'] / 'ranker').rglob('*')) if p.is_file())
    else:
        names = ["prompt.md", *cfg.pop("assets", {})]
        cfg.pop("prompt_sha256", None)
    return {"config": cfg, "assets": {name: (info["dir"] / name).read_bytes() for name in names}}


def _copy(value):
    return _MISSING if value is _MISSING else copy.deepcopy(value)


def rebase(base, proposed, current, path=""):
    """三方合并；只重放分支改动，同字段不同改动拒绝自动选择。"""
    if proposed == base:
        return _copy(current)
    if current == base or current == proposed:
        return _copy(proposed)
    if all(isinstance(value, dict) for value in (base, proposed, current)):
        out = {}
        for key in sorted(base.keys() | proposed.keys() | current.keys()):
            merged = rebase(base.get(key, _MISSING), proposed.get(key, _MISSING),
                            current.get(key, _MISSING), f"{path}.{key}".strip("."))
            if merged is not _MISSING:
                out[key] = merged
        return out
    raise ConfigError(f"重放冲突: {path}（分支和新全量修改了同一字段/资产）")


def _create_version(kind: str, snapshot: dict, data_ref: str) -> str:
    if kind == "judge":
        cfg = copy.deepcopy(snapshot["config"])
        assets = snapshot["assets"]
        cfg["assets"] = {name: hashlib.sha256(blob).hexdigest()
                         for name, blob in assets.items() if name != "prompt.md"}
        return versions.create_judge_version(cfg, {"origin": "branch"}, assets=assets)
    with tempfile.TemporaryDirectory(prefix="branch-", dir=versions.PRIVATE) as temp:
        source = Path(temp)
        (source / "scenarios").mkdir()
        for name, blob in snapshot["assets"].items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        return versions.create_generator_version(snapshot["config"], data_ref, source_dir=source)


def _pack_info(pack_ref: str, current: dict, dataset: str, *, sealed: bool = False) -> dict:
    _name(pack_ref)
    path = versions.PRIVATE / "judge_eval" / pack_ref / "pack.json"
    if not path.exists():
        raise ConfigError(f"评估包不存在: {pack_ref}")
    if ("validation" in pack_ref) != (dataset == "fixed_test"):
        raise ConfigError("开发轮使用 calibration 包；固定轮必须使用 validation 包")
    if sealed:
        # 提交只冻结已有验证包的引用和内容指纹；达标后才解析题目。
        return {"ref": pack_ref, "sha256": sha256_file(path), "sealed_basis": current}
    pack = _read(path)
    if pack.get("c0_gen_version") != current["production_gen"]:
        raise ConfigError(f"评估包 {pack_ref} 不是当前生产生成器的产物")
    if pack.get("data_ref", current["data"]) != current["data"]:
        raise ConfigError(f"评估包 {pack_ref} 数据版本不一致")
    rows = pack.get("rows", [])
    if not rows or len({str(row["case_id"]) for row in rows}) != len(rows):
        raise ConfigError("评估包为空或题目 ID 重复")
    return {"ref": pack_ref, "sha256": sha256_file(path)}


def _accepted_source(name: str, kind: str, current: dict) -> dict:
    """Reuse a completed adoption receipt as a development starting point.

    This neither re-adopts a version nor re-signs its historical learning guard.
    New trials still bind and verify both models under the current source rules.
    """
    path = _root(name) / 'state.json'
    state = _read(path) if path.is_file() else {}
    accepted = state.get('development', {})
    if state.get('kind') != kind or accepted.get('basis') != current or not accepted.get('experiment_id'):
        raise ConfigError('来源分支没有同条件、已通过的开发版')
    directory = experiment.load_experiment(accepted['experiment_id'])
    spec, result = experiment.spec_of(directory), experiment.state_of(directory)
    record = round_of(spec['branch_round'])
    if (record['name'] != name or record['basis'] != current or spec.get('production_basis') != current
            or record.get('candidate_ref') != accepted.get('candidate_ref')
            or spec.get('candidate_ref') != accepted.get('candidate_ref')
            or spec.get('kind') != (experiment.KIND_GEN_AB if kind == 'gen' else experiment.KIND_JUDGE_EVAL)
            or accepted['experiment_id'] != _trial_id(record, 'development')
            or spec.get('dataset') != 'development' or result.get('status') != 'finished'
            or result.get('verdict') != 'merge_to_iteration_baseline'):
        raise ConfigError('来源分支的开发采用记录与正式实验不一致')
    gates.require_promotable(spec)
    frozen, current_materials = spec.get('evaluation_materials', {}), gates.materials(spec)
    if ({k: v for k, v in frozen.items() if k != 'implementation'}
            != {k: v for k, v in current_materials.items() if k != 'implementation'}):
        raise ConfigError('来源开发版的模型或评分条件已变化')
    if frozen.get('implementation') != current_materials['implementation']:
        from .runtime import verify_historical_implementation
        verify_historical_implementation(directory, frozen.get('implementation'))
    metrics = gates.confirmed_metrics(directory,
        gates.development_rows(spec, historical_directory=directory), spec)
    if protocol.decide(metrics, 'development', spec['protocol'])['verdict'] != 'merge_to_iteration_baseline':
        raise ConfigError('来源开发版缺少完整确认净胜')
    return {**accepted, 'branch': name, 'spec_sha256': sha256_file(directory / 'spec.json')}


@versions.transaction
def submit(name: str, kind: str, change: str, *, overrides: dict | None = None,
           candidate_ref: str | None = None, development_pack: str | None = None,
           validation_pack: str | None = None, saved_draw_manifest: str | None = None,
           prompt_recipe: str | None = None, from_branch: str | None = None) -> dict:
    """创建/更新一个方向。每次提交冻结新的提案；已有在途实验保留。"""
    root = _root(name)
    if kind not in {"gen", "judge"} or not change.strip():
        raise ConfigError("需要 gen/judge 类型和非空改动说明")
    if saved_draw_manifest is not None and kind != 'judge':
        raise ConfigError('Saved development draws require a Judge branch')
    if sum(map(bool, (overrides, candidate_ref, prompt_recipe))) != 1:
        raise ConfigError("提供非空 overrides、candidate_ref 或 prompt_recipe，三选一")
    if prompt_recipe and kind != 'gen':
        raise ConfigError('prompt_recipe requires a generator branch')
    state_path = root / "state.json"
    state = _read(state_path) if state_path.exists() else {"name": name, "kind": kind, "rounds": []}
    if state["kind"] != kind:
        raise ConfigError("已有分支不能改变 gen/judge 类型")
    current = basis()
    packs = {}
    if kind == "judge":
        from . import acceptance
        if not development_pack or (not validation_pack and not acceptance.exists(current['data'])):
            raise ConfigError("Judge 分支必须同时提供开发校准包和固定验证包")
        packs = {ds: _pack_info(ref, current, ds, sealed=ds == 'fixed_test') for ds, ref in
                 (("development", development_pack), ("fixed_test", validation_pack)) if ref}
    origin = current[f"production_{kind}"]
    accepted = state.get('development', {})
    if from_branch is not None:
        accepted = _accepted_source(from_branch, kind, current)
    development_ref = accepted.get('candidate_ref') if accepted.get('basis') == current else origin
    before = _snapshot(kind, development_ref)
    if prompt_recipe:
        from .training import mechanical_generator
        candidate_ref = mechanical_generator(current['data'], before['config'], recipe=prompt_recipe)
    if candidate_ref:
        _name(candidate_ref)
        after = _snapshot(kind, candidate_ref)
        if kind == "gen":
            from .learning_guard import require_materials
            require_materials(current["data"], [_version(kind, candidate_ref)["dir"]])
    else:
        if {"data_version", "persona_sha256", "prompt_sha256", "assets"} & overrides.keys():
            raise ConfigError("资产/指纹变更请提交完整候选版本，不能只覆盖指纹")
        after = {**before, "config": experiment._merge(before["config"], overrides)}
    if before == after:
        raise ConfigError("提案与当前全量相同，没有可迭代的改动")
    candidate_ref = candidate_ref or _create_version(kind, after, current["data"])
    replay_binding = None
    if saved_draw_manifest is not None:
        from .draw_replay import manifest
        pack = _read(versions.PRIVATE / 'judge_eval' / development_pack / 'pack.json')
        replay_binding, _ = manifest(saved_draw_manifest, pack, _version(kind, development_ref),
                                     _version(kind, candidate_ref))
    rev_dir = root / "revisions"
    revision = versions._next_id([p.stem for p in rev_dir.glob("*.json")], "v")
    proposal = {"name": name, "kind": kind, "revision": revision, "change": change.strip(),
                "basis": current, "origin_ref": origin, "candidate_ref": candidate_ref,
                "development_ref": development_ref,
                "packs": packs, "protocol": experiment._protocol_snapshot(load_settings()),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if replay_binding is not None:
        proposal['saved_draw_replay'] = replay_binding
    if from_branch is not None:
        proposal['development_source'] = accepted
    write_json(rev_dir / f"{revision}.json", proposal)
    state.update(revision=revision, paused=False)
    write_json(state_path, state)
    return proposal


def round_of(ref: str) -> dict:
    parts = ref.split("/")
    if len(parts) != 2:
        raise ConfigError(f"非法分支轮次: {ref}")
    name, rid = map(_name, parts)
    return _read(_root(name) / "rounds" / f"{rid}.json")


def _save_round(record: dict) -> None:
    write_json(_root(record["name"]) / "rounds" / f"{record['id']}.json", record)


def _trial_id(record: dict, dataset: str) -> str:
    stage = {"smoke": "smoke", "development": "dev", "entry": "entry", "fixed_test": "fixed"}[dataset]
    return f"branch-{record['name']}-{record['id']}-{stage}"


def _new_round(state: dict, proposal: dict, current: dict) -> dict:
    rounds_dir = _root(state["name"]) / "rounds"
    rid = versions._next_id([p.stem for p in rounds_dir.glob("*.json")], "r")
    record = {"id": rid, "name": state["name"], "kind": state["kind"],
              "revision": proposal["revision"], "basis": current, "phase": "preparing",
              "origin_ref": proposal["origin_ref"], "protocol": proposal["protocol"],
              "change": proposal["change"], "packs": {},
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        if proposal.get('saved_draw_replay'):
            if current != proposal['basis']:
                raise ConfigError('Saved development draws require a new proposal after production changes')
            record['saved_draw_replay'] = proposal['saved_draw_replay']
        if current["data"] != proposal["basis"]["data"]:
            raise ConfigError("数据版本已变化：需提交基于新数据的提案，禁止自动换样本")
        old = _snapshot(state["kind"], proposal["origin_ref"])
        proposed = _snapshot(state["kind"], proposal["candidate_ref"])
        base_ref = current[f"production_{state['kind']}"]
        latest = _snapshot(state["kind"], base_ref)
        merged = rebase(old, proposed, latest)
        if merged == latest:
            record.update(phase="integrated", reason="分支改动已包含在当前全量中")
        else:
            candidate = (proposal["candidate_ref"] if merged == proposed else
                         _create_version(state["kind"], merged, current["data"]))
            development_ref = proposal.get('development_ref', base_ref) if current == proposal['basis'] else base_ref
            record.update(baseline_ref=development_ref, candidate_ref=candidate)
            if state["kind"] == "judge":
                from .branch_packs import prepare
                record["packs"] = {ds: prepare(pack, current, ds)
                                   for ds, pack in proposal["packs"].items()
                                   if ds == 'development' or proposal['protocol'].get('gate_schema', 1) < 2}
                record['fixed_pack_source'] = proposal['packs'].get('fixed_test')
    except ConfigError as exc:
        record.update(phase="conflict" if "重放冲突" in str(exc) else "blocked", reason=str(exc))
    _save_round(record)
    state["rounds"].append(rid)
    write_json(_root(state["name"]) / "state.json", state)
    return record


def _ensure_trial(record: dict, dataset: str) -> Path:
    exp_id = _trial_id(record, dataset)
    path = versions.PRIVATE / "experiments" / exp_id
    if (path / "spec.json").exists():
        # Polling is not execution. The worker checks sources on every start or
        # resume, and promotion checks them again before changing adopted state.
        return path
    if basis() != record["basis"]:
        raise ConfigError("全量已前进，不能在过期轮次上创建新评测")
    smoke = dataset == "smoke"
    actual_dataset = "development" if smoke or dataset == "entry" else dataset
    if dataset == "development":
        check = experiment.load_experiment(_trial_id(record, "smoke"))
        result = experiment.state_of(check)
        if result.get("status") != "finished" or result.get("metrics", {}).get("failures") != 0:
            raise ConfigError("冒烟未完成或仍有失败，禁止启动开发全量")
    entry = None
    if dataset == "fixed_test":
        dev = experiment.load_experiment(_trial_id(record, "development"))
        if experiment.state_of(dev).get("verdict") != "merge_to_iteration_baseline" or \
                experiment.state_of(dev).get("status") != "finished":
            raise ConfigError("开发实验尚未通过，不能进入固定验收")
        if record['protocol'].get('gate_schema', 1) >= 2:
            entry = _entry(record, required=True)
    kind = record["kind"]
    baseline_ref = record['basis'][f'production_{kind}'] if dataset in ('entry', 'fixed_test') else record['baseline_ref']
    baseline = _version(kind, baseline_ref)
    candidate = _version(kind, record["candidate_ref"])
    data_dir = versions.data_version_dir(record["basis"]["data"])
    diff = protocol.config_diff(baseline["config"], candidate["config"])
    if kind == "gen":
        diff += experiment._asset_diff(baseline["dir"], candidate["dir"])
        data_dir = versions.data_version_dir(record["basis"]["data"])
        from . import acceptance
        receipt = None
        if dataset == 'fixed_test' and acceptance.exists(record['basis']['data']):
            receipt = acceptance.claim(_acceptance_spec(record))
            cases = acceptance.rows(receipt)
        else:
            cases = [json.loads(line) for line in datasets.case_path(data_dir, actual_dataset).read_text().splitlines() if line.strip()]
        if not smoke:
            experiment._verify_composition(cases, actual_dataset)
            experiment._verify_pool_minimum(data_dir)
    else:
        pack_ref = record["packs"][actual_dataset]
        _pack_info(pack_ref, record["basis"], actual_dataset)
    fp = experiment._candidate_fingerprint(candidate["config"], candidate["dir"])
    if dataset == "fixed_test":
        experiment.check_one_shot(fp, exp_id, experiment.KIND_GEN_AB if kind == "gen" else
                                  experiment.KIND_JUDGE_EVAL)
    spec = {"id": exp_id, "kind": experiment.KIND_GEN_AB if kind == "gen" else experiment.KIND_JUDGE_EVAL,
            "dataset": actual_dataset, "single_change": record["change"], "role": "production vs branch",
            "baseline_ref": baseline_ref, "candidate_ref": record["candidate_ref"],
            "candidate_source_ref": baseline_ref, "candidate_fingerprint": fp,
            "judge_ref": record["basis"]["production_judge"], "data_ref": record["basis"]["data"],
            "config_diff": diff, "protocol": record["protocol"], "smoke": smoke,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "smoke_limit": 2 if smoke else None,
            "branch_round": f"{record['name']}/{record['id']}", "production_basis": record["basis"],
            "against_production": dataset == 'entry', "branch_stage": dataset}
    if dataset == 'fixed_test':
        if kind == 'gen' and receipt:
            spec['acceptance'] = receipt
        elif kind == 'judge':
            pack = _read(versions.PRIVATE / 'judge_eval' / record['packs'][actual_dataset] / 'pack.json')
            if pack.get('acceptance'):
                spec['acceptance'] = pack['acceptance']
                spec['generator_ref'] = pack['c0_gen_version']
    if kind == "judge":
        spec["pack_ref"] = record["packs"][actual_dataset]
        spec["pack_sha256"] = sha256_file(versions.PRIVATE / "judge_eval" / spec["pack_ref"] / "pack.json")
        if record.get('saved_draw_replay') and actual_dataset == 'development':
            from .draw_replay import manifest
            binding = record['saved_draw_replay']
            if sha256_file(Path(binding['path'])) != binding['sha256']:
                raise ConfigError('Saved development observation manifest changed')
            pack = _read(versions.PRIVATE / 'judge_eval' / spec['pack_ref'] / 'pack.json')
            checked, _ = manifest(binding['path'], pack, baseline, candidate)
            spec['saved_draw_replay'] = checked
    else:
        from .data_guard import generation_snapshot
        spec["data_snapshot"] = generation_snapshot(cases[:2] if smoke else cases, data_dir,
                                                     [baseline["config"], candidate["config"]])
    if record['protocol'].get('gate_schema', 1) >= 2:
        counterpart = (_version('gen', record['basis']['production_gen']) if kind == 'judge' else
                       _version('judge', record['basis']['production_judge']))
        spec['purpose_snapshot'] = datasets.snapshot(data_dir)
        spec['source_audit'] = datasets.static_sources(data_dir, [baseline['dir'], candidate['dir'], counterpart['dir']], actual_dataset)
        if dataset == 'fixed_test':
            datasets.acceptance_available(data_dir, exp_id)
            if spec['source_audit'].get('promotion_eligible') is False:
                raise ConfigError('静态学习来源尚未核验，禁止进入固定验收')
        if entry:
            spec['fixed_entry'] = entry
    spec['evaluation_materials'] = gates.materials(spec)
    gates.bind_learning(spec)
    spec["fingerprint"] = experiment._fingerprint(spec, diff)
    experiment.write_spec(path, spec)
    return path


@versions.transaction
def promote_experiment(exp_id: str) -> dict:
    """生产推进采用同一个实例锁，核对完整生产快照，旧结果不得覆盖新全量。"""
    from .promote import _require_finished
    exp_dir, spec, state = _require_finished(exp_id)
    record = round_of(spec["branch_round"])
    branch_state = _read(_root(record["name"]) / "state.json")
    if branch_state["revision"] != record["revision"] or branch_state.get("paused"):
        raise ConfigError("提案已更新或分支已暂停，禁止推全旧轮次")
    if spec.get("against_production"):
        raise ConfigError("生产对比只用于固定轮准入")
    if exp_id != _trial_id(record, spec["dataset"]) or spec["candidate_ref"] != record["candidate_ref"]:
        raise ConfigError("实验与分支轮次不一致")
    required = "merge_to_iteration_baseline" if spec["dataset"] == "development" else "adopt"
    if state.get("verdict") != required:
        raise ConfigError(f"实验未达到 {required}，禁止推全")
    ptr = versions.load_pointers()
    # 崩溃恢复：指针和采用凭证一起原子写入，避免已推全后重复操作。
    receipt = ptr.get("branch_promotions", {}).get(exp_id)
    if receipt == record["candidate_ref"]:
        record.update(phase="promoted", promoted_by=exp_id)
        _save_round(record)
        return ptr
    if basis() != record["basis"] or spec["production_basis"] != record["basis"]:
        raise ConfigError("生产基线/数据/Judge 已前进：旧实验已过期，必须在新全量上重测")
    if spec["dataset"] == "development":
        if spec['protocol'].get('gate_schema', 1) >= 2:
            gates.require_development(exp_dir)
            branch_state['development'] = {'candidate_ref': spec['candidate_ref'], 'basis': record['basis'],
                                           'experiment_id': exp_id}
            write_json(_root(record['name']) / 'state.json', branch_state)
            if branch_state.get('stage_limit') == 'development':
                record.update(phase='development_accepted', reason='已保留有收益的开发版；本分支仅运行开发比较')
                _save_round(record)
        return ptr  # 各分支保留自己的开发版，不争抢共享 iteration 指针。
    dev = experiment.load_experiment(_trial_id(record, "development"))
    if experiment.state_of(dev).get("status") != "finished" or \
            experiment.state_of(dev).get("verdict") != "merge_to_iteration_baseline":
        raise ConfigError("缺少同轮候选的开发通过证据")
    if spec["protocol"].get("gate_schema", 1) >= 2:
        gates.verify_fixed_receipt(spec)
    _version(record["kind"], record["candidate_ref"])
    prod_key, iteration_key = f"production_{record['kind']}", f"iteration_{record['kind']}"
    old = ptr[prod_key]
    ptr[prod_key] = record["candidate_ref"]
    if ptr[iteration_key] == old:
        ptr[iteration_key] = record["candidate_ref"]
    ptr.setdefault("branch_promotions", {})[exp_id] = record["candidate_ref"]
    versions.save_pointers(ptr)
    record.update(phase="promoted", promoted_by=exp_id)
    _save_round(record)
    return ptr


def _entry(record, *, required=False):
    check = gates.require_fixed_entry if required else gates.fixed_entry
    return check(experiment.KIND_GEN_AB if record['kind'] == 'gen' else experiment.KIND_JUDGE_EVAL,
        record['basis']['data'], record['basis'][f"production_{record['kind']}"], record['candidate_ref'],
        record['protocol'], judge_ref=record['basis']['production_judge'],
        development_pack=record['packs'].get('development'))



def _acceptance_spec(record):
    return {'id': _trial_id(record, 'fixed_test'), 'kind': experiment.KIND_GEN_AB if record['kind'] == 'gen' else experiment.KIND_JUDGE_EVAL,
            'data_ref': record['basis']['data'], 'baseline_ref': record['basis'][f"production_{record['kind']}"],
            'candidate_ref': record['candidate_ref'], 'judge_ref': record['basis']['production_judge'],
            'generator_ref': record['basis']['production_gen'] if record['kind'] == 'judge' else None, 'protocol': record['protocol']}


def _fixed_job(record):
    path = versions.PRIVATE / 'experiments' / _trial_id(record, 'fixed_test')
    if (path / 'spec.json').exists():
        return {'kind': 'experiment', 'id': path.name}
    if record['protocol'].get('gate_schema', 1) >= 2:
        if record['kind'] == 'judge' and 'fixed_test' not in record['packs']:
            # Check before consuming a new batch. Once allocated, polling the
            # same pack does not reconstruct development evidence repeatedly.
            _entry(record, required=True)
            from .branch_packs import prepare, prepare_batch
            from . import acceptance
            source = record['fixed_pack_source']
            if source is None:
                receipt = acceptance.claim(_acceptance_spec(record))
                record['packs']['fixed_test'] = prepare_batch(receipt, record['basis'])
                _save_round(record)
                return {'kind': 'pack', 'id': record['packs']['fixed_test']}
            if 'sealed_basis' in source:
                checked = _pack_info(source['ref'], source['sealed_basis'], 'fixed_test')
                if checked['sha256'] != source['sha256']:
                    raise ConfigError('原评估包已变化，禁止偷偷换样本')
            record['packs']['fixed_test'] = prepare(source, record['basis'], 'fixed_test')
            _save_round(record)
        pack = record['packs'].get('fixed_test')
        if pack and not (versions.PRIVATE / 'judge_eval' / pack / 'pack.json').exists():
            return {'kind': 'pack', 'id': pack}
    return {'kind': 'experiment', 'id': _ensure_trial(record, 'fixed_test').name}


@versions.transaction
def advance(name: str | None = None) -> list[dict]:
    """推进所有分支的状态机，返回可调度任务；模型调用均在锁外的独立进程。"""
    jobs = []
    for path in sorted((versions.PRIVATE / "branches").glob("*/state.json")):
        if name is not None and path.parent.name != name:
            continue
        state = _read(path)
        if state.get("paused"):
            continue
        root = path.parent
        proposal = _read(root / "revisions" / f"{state['revision']}.json")
        record = round_of(f"{state['name']}/{state['rounds'][-1]}") if state["rounds"] else None
        current = basis()
        if record and versions.load_pointers().get("branch_promotions", {}).get(
                _trial_id(record, "fixed_test")) == record.get("candidate_ref") and record.get("candidate_ref"):
            record["phase"] = "promoted"
            _save_round(record)
        if record and record["phase"] not in _TERMINAL:
            exp_id = _trial_id(record, record["phase"] if record["phase"] != "preparing" else "smoke")
            exp_path = versions.PRIVATE / "experiments" / exp_id
            running = (exp_path / "spec.json").exists() and experiment.state_of(exp_path).get("status") != "finished"
            if record["basis"] != current or record["revision"] != state["revision"]:
                # 已开始的轮次继续保留/完成自己的快照；不把新旧基线结果混写。
                from .storage import locked
                from .progress import is_active
                if running and (locked(exp_path / ".run.lock") or
                                is_active(experiment.state_of(exp_path).get("progress", {}))):
                    continue
                record.update(phase="superseded", reason="全量或提案已更新，开启新轮次")
                _save_round(record)
        if record and record["phase"] in _TERMINAL:
            if record["revision"] == state["revision"] and (record["basis"] == current or
                                                             record["phase"] in {"promoted", "integrated"}):
                continue
            record = None
        if record is None:
            record = _new_round(state, proposal, current)
        if record["phase"] in _TERMINAL:
            continue
        try:
            if record["phase"] == "preparing":
                pending = [pack for pack in record["packs"].values()
                           if not (versions.PRIVATE / "judge_eval" / pack / "pack.json").exists()]
                if pending:
                    jobs.extend({"kind": "pack", "id": pack} for pack in pending)
                    continue
                record["phase"] = "smoke"
                _save_round(record)
            dataset = record["phase"]
            if state.get('stage_limit') == 'development' and dataset in ('entry', 'fixed_test'):
                continue
            if dataset == 'fixed_test':
                job = _fixed_job(record)
                if job['kind'] == 'pack':
                    jobs.append(job)
                    continue
                exp_path = experiment.load_experiment(job['id'])
            else:
                exp_path = _ensure_trial(record, dataset)
            result = experiment.state_of(exp_path)
            if result.get("status") != "finished":
                jobs.append({"kind": "experiment", "id": exp_path.name})
                continue
            if dataset == "smoke":
                if result.get("metrics", {}).get("failures") != 0:
                    raise ConfigError("冒烟有失败，不能启动开发全量")
                record["phase"] = "development"
                _save_round(record)
                development = _ensure_trial(record, "development")
                jobs.append({"kind": "experiment", "id": development.name})
                continue
            if dataset in ('development', 'entry') and record['protocol'].get('gate_schema', 1) >= 2:
                if dataset == 'development':
                    if result.get('verdict') != 'merge_to_iteration_baseline':
                        record.update(phase='rejected', reason=result.get('reason', '未达开发晋级门槛'))
                        _save_round(record)
                        continue
                    promote_experiment(exp_path.name)
                    # Incremental gains must be verified against production directly; never add them.
                    if state.get('stage_limit') == 'development':
                        record.update(phase='development_accepted', reason='已保留有收益的开发版；本分支仅运行开发比较')
                        _save_round(record)
                        continue
                    if record['baseline_ref'] != record['basis'][f"production_{record['kind']}"]:
                        reused = gates.reusable_comparison(
                            experiment.KIND_GEN_AB if record['kind'] == 'gen' else experiment.KIND_JUDGE_EVAL,
                            record['basis']['data'], record['basis'][f"production_{record['kind']}"],
                            record['candidate_ref'], record['protocol'],
                            judge_ref=record['basis']['production_judge'],
                            development_pack=record['packs'].get('development'))
                        if reused is None:
                            record['phase'] = 'entry'
                            _save_round(record)
                            jobs.append({'kind': 'experiment', 'id': _ensure_trial(record, 'entry').name})
                            continue
                receipt = _entry(record)
                record['development_comparison'] = receipt
                if receipt['eligible']:
                    record['phase'] = 'fixed_test'
                    _save_round(record)
                    jobs.append(_fixed_job(record))
                else:
                    record.update(phase='development_accepted', reason=
                        f"已保留开发版；对生产版确认净胜 {receipt['net_win_confirmed']:+d}/{receipt['pairs']}，"
                        f"固定准入需 +{receipt['required_net_win']}，等待下一次改动")
                    _save_round(record)
            else:
                required = 'merge_to_iteration_baseline' if dataset == 'development' else 'adopt'
                if result.get('verdict') != required:
                    record.update(phase='rejected', reason=result.get('reason', '未达晋升门槛'))
                    _save_round(record)
                elif dataset == 'development':
                    promote_experiment(exp_path.name)
                    if state.get('stage_limit') == 'development':
                        record.update(phase='development_accepted', reason='已保留有收益的开发版；本分支仅运行开发比较')
                        _save_round(record)
                        continue
                    record['phase'] = 'fixed_test'
                    _save_round(record)
                    jobs.append(_fixed_job(record))
                else:
                    promote_experiment(exp_path.name)
        except ConfigError as exc:
            record.update(blocked_phase=record['phase'], phase="blocked", reason=str(exc))
            _save_round(record)
    # 同一轮 tick 的后一个分支可能刚推全；不要再启动前面收集的旧基线任务。
    current = basis()
    def fresh(job):
        if job["kind"] == "experiment":
            return experiment.spec_of(experiment.load_experiment(job["id"]))["production_basis"] == current
        recipe = _read(versions.PRIVATE / "judge_eval" / job["id"] / "recipe.json")
        return recipe["generator_ref"] == current["production_gen"] and recipe["data_ref"] == current["data"]
    jobs = [job for job in jobs if fresh(job)]
    return list({(job["kind"], job["id"]): job for job in jobs}.values())


@versions.transaction
def set_stage_limit(name: str, stage: str) -> None:
    if stage not in {'development', 'fixed_test'}:
        raise ConfigError('stage limit must be development or fixed_test')
    path = _root(name) / 'state.json'
    state = _read(path)
    state['stage_limit'] = stage
    write_json(path, state)


@versions.transaction
def set_paused(name: str, paused: bool) -> None:
    path = _root(name) / "state.json"
    state = _read(path)
    if not paused and state['rounds']:
        record = round_of(f"{name}/{state['rounds'][-1]}")
        if record['phase'] == 'blocked':
            phase = record.get('blocked_phase')
            if phase is None and record['kind'] == 'gen' and record.get('candidate_ref') and \
                    record.get('baseline_ref') and not any(
                        (versions.PRIVATE / 'experiments' / _trial_id(record, stage) / 'spec.json').exists()
                        for stage in ('smoke', 'development', 'entry', 'fixed_test')):
                phase = 'smoke'  # Legacy block before the first experiment was created.
            if phase not in {'preparing', 'smoke', 'development', 'entry', 'fixed_test'}:
                raise ConfigError('blocked round has no recoverable stage; submit a corrected proposal')
            record.update(phase=phase)
            record.pop('reason', None)
            record.pop('blocked_phase', None)
            _save_round(record)
    state["paused"] = paused
    write_json(path, state)


def status() -> list[dict]:
    out = []
    for path in sorted((versions.PRIVATE / "branches").glob("*/state.json")):
        state = _read(path)
        record = round_of(f"{state['name']}/{state['rounds'][-1]}") if state["rounds"] else None
        out.append({**state, "current_round": record})
    return out
