"""Judge 回复包的统一构建入口：首次开发包、生成器更新后的重建、验收批次。"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from threading import local

from ..config import ConfigError, load_settings, sha256_file
from ..generator.generator import ReplyGenerator
from ..llm import LLMRefusal, build_clients
from ..tracing import CaseTrace
from . import acceptance, control, data_guard, datasets, experiment, gates, runtime, task_state, versions, learning_guard
from .parallel import completed_map
from .storage import write_json, file_lock, read_json, write_once_json


_INVALID_GENERATION = "生成结果两次都无效（JSON 结构/非空/条数校验失败）"


@versions.transaction
def prepare_initial(kind='calibration', sample=None):
    """Freeze a complete development list before any transport call.

    Fixed replies are created only by a branch after binding its acceptance
    receipt. A standalone builder cannot expose the entire sealed pool.
    """
    current = versions.load_pointers()
    if kind == 'validation':
        gates.require_fixed_entry('judge_eval', current['data'], current['production_judge'],
                                  current['iteration_judge'], experiment._protocol_snapshot(load_settings()))
        raise ConfigError('固定回复包必须由候选分支绑定验收批次后构建；请使用 iterate_branches.py')
    if kind != 'calibration':
        raise ConfigError('未知回复包用途')
    from ..generator.history_sources import load
    data = versions.data_version_dir(current['data'])
    cases = load(data).role('judge_development')
    if not cases or (sample is not None and sample != len(cases)):
        raise ConfigError(f'正式包必须使用已冻结的完整 judge_development 清单（{len(cases)} 题），禁止重新抽样')
    info = versions.load_generator(current['production_gen'])
    recipe = {'kind': 'judge_pack', 'dataset': 'development', 'purpose': 'judge_development',
              'data_ref': current['data'], 'generator_ref': info['id'],
              'data_snapshot': data_guard.generation_snapshot(cases, data, [info['config']]),
              'learning_snapshot': learning_guard.snapshot(current['data'], [info['dir']], 'judge_development')}
    learning_guard.verify_generation(recipe, cases)
    ref = time.strftime('pack-calibration-%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    directory = versions.PRIVATE / 'judge_eval' / ref
    runtime.freeze(directory)
    write_json(directory / 'recipe.json', recipe)
    return ref


def prepare(source: dict, current: dict, dataset: str) -> str:
    path = versions.PRIVATE / "judge_eval" / source["ref"] / "pack.json"
    if sha256_file(path) != source["sha256"]:
        raise ConfigError("原评估包已变化，禁止偷偷换样本")
    pack = json.loads(path.read_text())
    if pack.get('data_ref') != current['data']:
        raise ConfigError('评估包与当前数据版本不符，禁止按生成器编号直接复用')
    from . import datasets
    datasets.assert_pack(versions.data_version_dir(current['data']), pack,
                         'judge_development' if dataset == 'development' else dataset)
    learning_guard.verify_pack(pack, dataset)
    learning_guard.require_materials(current['data'], [versions.generator_dir(current['production_gen'])])
    if pack["c0_gen_version"] == current["production_gen"]:
        return source["ref"]
    recipe = {"source": source, "data_ref": current["data"],
              "generator_ref": current["production_gen"], "dataset": dataset}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    kind = "validation" if dataset == "fixed_test" else "calibration"
    ref = f"pack-{kind}-rebase-{digest}"
    target = versions.PRIVATE / "judge_eval" / ref / "recipe.json"
    if not target.exists():
        info = versions.load_generator(recipe["generator_ref"])
        data = versions.data_version_dir(recipe["data_ref"])
        recipe["data_snapshot"] = data_guard.generation_snapshot(datasets.pack_cases(pack), data, [info["config"]])
        recipe['learning_snapshot'] = learning_guard.snapshot(current['data'], [info['dir']],
            'judge_development' if dataset == 'development' else dataset)
        learning_guard.verify_generation(recipe, datasets.pack_cases(pack))
        write_json(target, recipe)
        runtime.freeze(target.parent)
    else:
        learning_guard.verify_generation(json.loads(target.read_text()), datasets.pack_cases(pack))
    return ref


def build(ref: str, workers: int = 1) -> None:
    from .branches import _name
    directory = versions.PRIVATE / "judge_eval" / _name(ref)
    with file_lock(directory / ".run.lock", blocking=False), control.job(directory):
        if (directory / 'runtime.json').exists():
            runtime.require_current(directory)
        if (directory / "pack.json").exists():
            pack = json.loads((directory / 'pack.json').read_text())
            recipe = json.loads((directory / 'recipe.json').read_text())
            learning_guard.verify_pack(pack, recipe['dataset'])
            return
        recipe = json.loads((directory / "recipe.json").read_text())
        if recipe.get('acceptance'):
            source = {'data_ref': recipe['data_ref'], 'c0_gen_version': recipe['generator_ref'],
                      'rows': acceptance.rows(recipe['acceptance']), 'acceptance': recipe['acceptance']}
            acceptance.transition({**recipe['acceptance']['binding'], 'id': recipe['acceptance']['experiment_id'],
                                   'acceptance': recipe['acceptance']}, 'opened')
        elif recipe.get('source'):
            source_path = versions.PRIVATE / "judge_eval" / recipe["source"]["ref"] / "pack.json"
            if sha256_file(source_path) != recipe["source"]["sha256"]:
                raise ConfigError("源包指纹不一致，拒绝续建")
            source = json.loads(source_path.read_text())
        else:
            from ..generator.history_sources import load
            source = {'data_ref': recipe['data_ref'], 'c0_gen_version': recipe['generator_ref'],
                      'rows': load(versions.data_version_dir(recipe['data_ref'])).role(recipe['purpose'])}
        cases = datasets.pack_cases(source)
        seal = learning_guard.verify_generation(recipe, cases)
        checkpoint = directory / "building.json"
        saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"rows": []}
        done = {str(row["case_id"]): row for row in saved["rows"]}
        info = versions.load_generator(recipe["generator_ref"])
        data = versions.data_version_dir(recipe["data_ref"])
        data_guard.verify(directory, data_guard.generation_snapshot(cases, data, [info["config"]]),
                          expected=recipe.get("data_snapshot"),
                          has_checkpoint=bool(saved["rows"]) or data_guard.has_results(directory))
        settings = load_settings()
        worker = local()
        progress = {"status": "running", "pid": os.getpid(), "total": len(cases),
                    "process_start": task_state.process_start(os.getpid()),
                    "completed": len(done), "workers": workers, "started_at": time.time(),
                    "updated_at": time.time(), "phase": "generating"}
        write_json(directory / "progress.json", progress)

        def generate(case):
            seal.check()
            if not hasattr(worker, "generator"):
                worker.generator = ReplyGenerator(settings, info["config"],
                    build_clients(settings, info["config"]["llm"]),
                    prompt_root=info["dir"], pool_path=data / "fewshot_pool.jsonl")
                worker.generator.source_check = seal.check
            trace = CaseTrace(directory, case, {"kind": "judge_pack", "dataset": recipe["dataset"],
                              "baseline_ref": info["id"], "data_ref": recipe["data_ref"]})
            try:
                with trace.operation("generation", "baseline", 0, {"forced_reply": True}) as op:
                    result = worker.generator.generate(case, forced_reply=True)
                    op["result"] = result
                trace.finish("ok")
                seal.check()
                return {**case, "ai_replies": result["replies"], "generation_trace_ref": trace.ref}
            except (LLMRefusal, RuntimeError) as exc:
                trace.finish("failed")
                if not isinstance(exc, LLMRefusal) and not (type(exc) is RuntimeError and str(exc) == _INVALID_GENERATION):
                    raise
                return {**case, "generation_status": "failed", "generation_trace_ref": trace.ref,
                        "generation_error": {"type": type(exc).__name__, "message": str(exc)}}
            except BaseException:
                trace.finish("failed")
                raise

        try:
            for row in completed_map(generate, [c for c in cases if str(c["case_id"]) not in done], workers):
                seal.check()
                done[str(row["case_id"])] = row
                write_json(checkpoint, {"rows": list(done.values())})
                progress.update(completed=len(done), updated_at=time.time())
                write_json(directory / "progress.json", progress)
            rows = [done[str(case["case_id"])] for case in cases]
            learning_guard.verify_generation(recipe, cases)
            write_json(directory / "pack.json", {
                **source, "pack_id": ref, "c0_gen_version": info["id"], "data_ref": recipe["data_ref"],
                "rebuilt_from": recipe.get("source"), "rows": rows,
                "generation_proof": learning_guard.pack_proof(recipe, rows),
                "generation_failures": sum(row.get("generation_status") == "failed" for row in rows),
                "identical_options": sum(row["human_reply"] == row.get("ai_replies") for row in rows)})
            progress["status"] = "finished"
        except BaseException:
            progress["status"] = "stopped"
            raise
        finally:
            progress["updated_at"] = time.time()
            write_json(directory / "progress.json", progress)


def prepare_batch(receipt, current):
    cases = acceptance.rows(receipt)
    ref = 'pack-validation-' + receipt['experiment_id']
    directory = versions.PRIVATE / 'judge_eval' / ref
    recipe = {'data_ref': current['data'], 'generator_ref': current['production_gen'],
              'dataset': 'fixed_test', 'acceptance': receipt}
    info = versions.load_generator(recipe['generator_ref'])
    data = versions.data_version_dir(recipe['data_ref'])
    recipe['data_snapshot'] = data_guard.generation_snapshot(cases, data, [info['config']])
    recipe['learning_snapshot'] = learning_guard.snapshot(current['data'], [info['dir']], 'fixed_test')
    learning_guard.verify_generation(recipe, cases)
    path = directory / 'recipe.json'
    if path.exists() and json.loads(path.read_text()) != recipe:
        raise ConfigError('sealed validation recipe changed')
    if not path.exists():
        write_json(path, recipe)
        runtime.freeze(directory)
    return ref


def recover_terminal_failures(ref: str) -> dict:
    """Restore documented terminal failures from old frozen pack traces.

    This repairs only the checkpoint, never the frozen executor or its identity.
    Transport/source errors remain errors, and saved results are never replaced.
    """
    from .branches import _name
    directory = versions.PRIVATE / 'judge_eval' / _name(ref)
    with file_lock(directory / '.run.lock', blocking=False), control.job(directory):
        runtime.verify(directory / 'runtime.json')
        if (directory / 'pack.json').exists():
            raise ConfigError('回复包已完成，禁止修改')
        recipe = read_json(directory / 'recipe.json')
        if recipe.get('acceptance'):
            cases = datasets.pack_cases({'rows': acceptance.rows(recipe['acceptance'])})
        elif recipe.get('source'):
            path = versions.PRIVATE / 'judge_eval' / recipe['source']['ref'] / 'pack.json'
            if sha256_file(path) != recipe['source']['sha256']:
                raise ConfigError('源包指纹不一致，拒绝恢复')
            cases = datasets.pack_cases(read_json(path))
        else:
            from ..generator.history_sources import load
            cases = load(versions.data_version_dir(recipe['data_ref'])).role(recipe['purpose'])
        seal = learning_guard.verify_generation(recipe, cases)
        info = versions.load_generator(recipe['generator_ref'])
        data_guard.verify(directory, data_guard.generation_snapshot(cases,
            versions.data_version_dir(recipe['data_ref']), [info['config']]),
            expected=recipe['data_snapshot'], has_checkpoint=True)
        checkpoint = directory / 'building.json'
        saved = read_json(checkpoint, default={'rows': []})
        done = {str(row['case_id']) for row in saved['rows']}
        expected = {str(case['case_id']): case for case in cases}
        recovered, proofs = {}, {}
        terminal = {'type': 'RuntimeError', 'message': _INVALID_GENERATION}
        for path in sorted((directory / 'traces').glob('*.json')):
            trace = read_json(path)
            case_id = str(trace.get('case_id'))
            operations = trace.get('operations', [])
            if case_id in done or trace.get('status') != 'failed' or len(operations) != 1:
                continue
            op = operations[0]
            if op.get('status') != 'failed' or op.get('error') != terminal:
                continue
            versions_match = all(trace.get('versions', {}).get(key) == value for key, value in {
                'kind': 'judge_pack', 'dataset': recipe['dataset'], 'data_ref': recipe['data_ref'],
                'baseline_ref': recipe['generator_ref']}.items())
            if (case_id not in expected or trace.get('case') != expected[case_id] or not versions_match
                    or op.get('kind') != 'generation' or op.get('branch') != 'baseline'
                    or op.get('round') != 0 or op.get('input') != {'forced_reply': True}
                    or path.stem != trace.get('trace_ref')):
                raise ConfigError('失败实录与冻结样本或生成条件不符，拒绝恢复')
            recovered[case_id] = {**expected[case_id], 'generation_status': 'failed',
                'generation_trace_ref': trace['trace_ref'], 'generation_error': terminal}
            proofs[str(path)] = sha256_file(path)
        if recovered:
            backup = directory / 'recoveries' / uuid.uuid4().hex
            write_once_json(backup / 'building.before.json', saved)
            write_once_json(backup / 'receipt.json', {'kind': 'terminal_generation_failure_recovery',
                'runtime': read_json(directory / 'runtime.json'), 'trace_files': proofs,
                'recovered': len(recovered), 'created_at': time.time()})
            seal.check()
            write_json(checkpoint, {'rows': [*saved['rows'], *recovered.values()]})
            progress = read_json(directory / 'progress.json', default={})
            progress.update(completed=len(saved['rows']) + len(recovered), updated_at=time.time())
            write_json(directory / 'progress.json', progress)
        return {'recovered_failures': len(recovered), 'completed': len(saved['rows']) + len(recovered),
                'total': len(cases)}
