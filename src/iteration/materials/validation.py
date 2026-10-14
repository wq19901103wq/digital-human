"""Mandatory source checks for creation, execution, resume and adoption.

实现自 src/iteration/learning_guard.py 迁入，并收录验证复用的正式实现（receipt v6，
设计 §4；与活适配器层 src/iteration/material_reuse.py 存在刻意重复——适配器按路径
加载、服务归档 runtime，本模块服务新代码，patch 一律指向本模块真实属性）。

Only independently reconstructible material recipes can be used. Old assets
remain on disk, but a self-declared learning.json never certifies them.
"""
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
from threading import Lock, local

from src.infrastructure.config import ConfigError
from src.infrastructure.config import sha256_file
from .. import datasets, versions
from .seals import RunSeal

# 验证复用正式实现的身份（receipt v6；活适配器层 material_reuse.MODE 仍为 v5，互不复用）。
MODE = 'sealed_persistent_material_validation_v6'
CAPACITY = 16
POOL_CAPACITY = 131072


# Reviewed mechanical instructions: no examples, personal facts or fitted values.
# A new instructional recipe requires code review and its own source validator.
MECHANICAL_ASSETS = {
    'persona.md': '9e19c78e0fde2598bf64be844b352ba890bfa06eea7730d4eaaaee51df513da0',
    'scenarios/group_chat.md': 'f92f0d797d5084930e8a6f1683f1848f1ae2b01a381fbe26cf8bce46ab61b2cc',
    'scenarios/friend.md': '486038a49bdcb864ce74766c16c256c5c77ebe978648c0fa531e492039f01d13',
}
GENERIC_PROFILE = 'c738e8701d4c5062eb9357e4e15a497b017cb006cc5430aacbace6e28c66da1e'
CONVERSATION_ASSETS = {**MECHANICAL_ASSETS,
    'persona.md': 'a81bffb7a7a8eb529fbee42f2319c6228341c9a5fd71a860071dd6cc501a81fc'}
GENERATOR_RECIPES = {
    'mechanical': MECHANICAL_ASSETS,
    'conversation': CONVERSATION_ASSETS,
    'conversation_partner': {**MECHANICAL_ASSETS,
        'persona.md': '894d05c3f901416fa2d4b87e73b1c97728444e76b4c395c81b1e865375ef798a'},
    'conversation_rhythm': {**MECHANICAL_ASSETS,
        'persona.md': '91bb955d25ec28df81cdd98744e699a1329d599aea72b6c0eee7f43cb41da1fc'},
    'conversation_rhythm_single_example': {**MECHANICAL_ASSETS,
        'persona.md': '8657ba44d0ae25b89c0931ec4148cd0bda4d3cf632f86b390fae283481fcfb5b'},
    'conversation_rhythm_role_continuity': {**MECHANICAL_ASSETS,
        'persona.md': 'c0e4dd0523823d361941ad2214243551bc9eb6c68df7ed0d762005f623894406'},
    'conversation_rhythm_partner': {**MECHANICAL_ASSETS,
        'persona.md': '5041c133e6186f9490858d703b021f5eb22f8ab2d73e8631fdb875a51149114b'},
}
GENERIC_JUDGE_TEMPLATE = '487b491f3609781dbee516c9b944a3dd0117922e21bd6af3b6f1afd70a0f6ac2'


def _require(value, message):
    if not value:
        raise ConfigError(message)


def require_default_judge_template(path):
    _require(Path(path).is_file() and sha256_file(path) == GENERIC_JUDGE_TEMPLATE,
             '默认 Judge 提示词不属于已核验的通用指令，禁止使用')


def _reconstructed_judge(data_ref, directory, record):
    """Supported clean retraining recipe; follow requests and reproduce weights.

    This executes only on a completed candidate, never while merely checking a
    generator or a paused study. It does not call an LLM or write training files.
    """
    origin_path = directory / 'source_judge.json'
    _require(origin_path.is_file(), f'模型缺少可重建的训练来源，禁止使用未知旧权重/词表：{directory}')
    origin = json.loads(origin_path.read_text())
    name, arm = origin.get('study', ''), origin.get('arm')
    _require(origin.get('origin') in ('history_lr_v1', 'd0011_material_repair')
             and name and Path(name).name == name and name not in ('.', '..')
             and isinstance(arm, str) and Path(arm).name == arm and arm not in ('.', '..'), '未支持的模型来源配方，不能靠 verified 标记放行')
    study = versions.PRIVATE / 'judge_training' / name
    spec = json.loads((study / 'spec.json').read_text())
    _require(arm in spec['feature_configs'], '训练分支未在实例配方中声明')
    # Reconstruct the actual training inputs; a new evaluation version does not
    # turn its repaired source files into the model's historical training data.
    source_data = spec['data_ref']
    from src.data.history.sources import load
    history = load(versions.data_version_dir(source_data))
    train = history.role('judge_training')
    history.role('judge_development')
    _require(sha256_file(study / 'profile.json') == GENERIC_PROFILE,
             'Judge 指令包含未核验的继承内容')
    # Independent audit checks the complete reference spans, zero starting weights,
    # train-only vocabulary, generation/feature request traces and fitted weights.
    from .. import training_evidence as audit
    from ..training_evidence import Evidence
    from ..training_evidence import generation_cache, generations
    evidence = Evidence()
    try:
        selected = audit.sources(study, spec, evidence)
        _require({c['case_id']: c for c in selected['train']} == {c['case_id']: c for c in train},
                 '实际训练题与原始训练清单不符')
        reference = json.loads((study / 'reference.json').read_text())
        from ...judge import corrected_v1 as rt
        prompt = rt.build_shared_single_case_prompt({}, reference, json.loads((study / 'profile.json').read_text()))
        prompt = prompt.replace('<blind_case>\n{}\n</blind_case>', '<blind_case>\n{{case}}\n</blind_case>')
        prompt = prompt.replace('参考资料来自与测试聊天隔离的 2026-04-01 前真人聊天，以及用户确认的本人事实。',
            '参考示例仅来自本次开发时间边界之前的可核验历史聊天，不含留出聊天或额外个人事实。')
        _require((study / 'prompt.md').read_text() == prompt, '实际 Judge 模板与核验材料不符')
        pool = {r['id']: r for r in audit.lines(versions.data_version_dir(source_data) / 'fewshot_pool.jsonl')}
        with generation_cache(evidence, study):
            generated = generations(study, spec, selected['train'], pool, evidence, True)
        result = audit.training(study, spec, selected['train'], generated, arm, evidence, True)
        original = versions.judge_dir(result['judge_ref'])['dir']
        original_record = json.loads((original / 'learning.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        if config.get('decision_policy') == 'embedding_only':
            from ..embedding_adoption import verify as verify_embedding
            verify_embedding(directory, record, original, study, spec, arm)
        elif config.get('decision_policy') == 'score_fusion':
            from ..fusion import verify as verify_fusion
            verify_fusion(directory, record, original, study, spec, arm)
        elif config.get('decision_policy') == 'gbdt_only':
            from ..gbdt_evidence import verify as verify_gbdt
            verify_gbdt(directory, record, original, study, spec, arm)
        else:
            _require(record['asset_files'] == original_record['asset_files'],
                     '派生 Judge 的学习资产与重建来源不符')
        for name, digest in original_record['asset_files'].items():
            _require(sha256_file(original / name) == digest, '原始训练资产变化')
        if source_data != data_ref:
            from ..material_compatibility import judge_sources, require_current
            require_current(data_ref, directory, judge_sources)
    except (ValueError, KeyError, OSError) as exc:
        raise ConfigError(f'模型来源重建失败：{exc}') from exc
    finally:
        evidence.store.db.close()


def require_materials(data_ref, directories, role='development'):
    directories = [Path(d) for d in directories]
    data = versions.data_version_dir(data_ref)
    _require(bool(datasets.manifest(data)), '数据缺少时间与来源清单，禁止进入正式实验')
    # Keep the existing learning cutoff even for fixed evaluation. Do not expand
    # learning eligibility merely because evaluation moved to a later window.
    audit = datasets.static_sources(data, directories, 'development')
    if audit.get('promotion_eligible') is not True:
        failures = '; '.join(row['asset'] + ': ' + row.get('error', '学习范围未验证或超出截止时间')
                             for row in audit.get('assets', []) if not row['verified'])
        raise ConfigError('学习材料来源未知或越界：禁止创建、运行、复用或晋升；' + failures)
    for directory in directories:
        record = json.loads((directory / 'learning.json').read_text())
        bound = record.get('asset_files', {})
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
                  if p.is_file() and str(p.relative_to(directory)) not in
                  ('learning.json', 'config.json', 'meta.json')}
        if not bound or set(bound) != actual:
            raise ConfigError('来源证明未覆盖全部实际学习资产')
        for name, digest in bound.items():
            path = directory / name
            if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                    or not path.is_file() or sha256_file(path) != digest):
                raise ConfigError('实际学习资产与来源证据不符')
        if bound in GENERATOR_RECIPES.values():
            _require(record.get('information_end') == 0, '机械指令不能声明额外学习来源')
        elif bound == {'prompt.md': GENERIC_JUDGE_TEMPLATE}:
            _require(record.get('information_end') == 0, '通用 Judge 指令不能声明额外学习来源')
        elif 'ranker/provenance.json' in bound:
            from src.generator.fewshot.provenance import verify as verify_ranker
            verify_ranker(data_ref, directory, record)
        else:
            _reconstructed_judge(data_ref, directory, record)
    return audit


def _directories(spec):
    if spec['kind'] == 'judge_eval':
        pack_path = versions.PRIVATE / 'judge_eval' / spec['pack_ref'] / 'pack.json'
        _require(sha256_file(pack_path) == spec.get('pack_sha256'), '回复包内容变化，禁止复用')
        pack = json.loads(pack_path.read_text())
        _require(pack.get('data_ref') == spec['data_ref'], '回复包与实验数据版本不符')
        datasets.assert_pack(versions.data_version_dir(spec['data_ref']), pack,
                             'judge_development' if spec['dataset'] == 'development' else 'fixed_test')
        verify_pack(pack, spec['dataset'])
        return [versions.judge_dir(spec[k])['dir'] for k in ('baseline_ref', 'candidate_ref')] + [
            versions.generator_dir(pack['c0_gen_version'])]
    _require(spec['kind'] == 'gen_ab', '未支持的实验来源校验入口')
    return [versions.generator_dir(spec[k]) for k in ('baseline_ref', 'candidate_ref')] + [
        versions.judge_dir(spec['judge_ref'])['dir']]


def snapshot(data_ref, directories, role='development'):
    from src.data.history.sources import load
    data = versions.data_version_dir(data_ref)
    require_materials(data_ref, directories, role)
    history = load(data)
    # Never open fixed answers as a side effect of development validation.
    history.role('judge_training')
    history.role(role)
    return {'schema': 1, 'data_ref': data_ref, 'role': role,
            'sources': history.hashes, 'purposes': datasets.snapshot(data),
            'materials': {str(d.resolve()): {str(p.relative_to(d)): sha256_file(p)
                for p in d.rglob('*') if p.is_file() and p.relative_to(d) != Path('meta.json')} for d in directories},
            # 新记录钉死真实实现路径（本文件位于 src/iteration/materials/，src 根为 parents[2]）；
            # 旧记录中的 iteration/learning_guard.py 键由归档 runtime 自包含解析，语义保留。
            'guard_code': {name: sha256_file(Path(__file__).parents[2] / name) for name in
                ('iteration/materials/validation.py', 'iteration/materials/seals.py',
                 'iteration/training_evidence.py', 'iteration/gbdt_evidence.py',
                 'iteration/fusion.py', 'iteration/embedding_adoption.py',
                 'iteration/material_compatibility.py',
                 'iteration/datasets.py', 'data/history/sources.py',
                 'generator/fewshot/provenance.py')}}


def bind(spec):
    """Only new, not-yet-run specs may receive this snapshot."""
    role = spec.get('purpose') or ('judge_development' if spec['kind'] == 'judge_eval'
                                  and spec['dataset'] == 'development' else spec['dataset'])
    spec['learning_snapshot'] = snapshot(spec['data_ref'], _directories(spec), role)


def same_materials(saved, current):
    """Validator hashes record provenance, not the identity of data/materials.

    Runtime and scoring implementations remain bound separately. Always run the
    current source checks before comparing; this does not waive those checks.
    """
    return ({k: v for k, v in saved.items() if k != 'guard_code'} ==
            {k: v for k, v in current.items() if k != 'guard_code'})


def verify(spec, *, historical_directory=None):
    _require(bool(spec.get('learning_snapshot')), '实验缺少冻结来源证明；保留旧结果，禁止补签后直接续跑')
    role = spec.get('purpose') or ('judge_development' if spec['kind'] == 'judge_eval'
                                  and spec['dataset'] == 'development' else spec['dataset'])
    current = snapshot(spec['data_ref'], _directories(spec), role)
    _require(same_materials(spec['learning_snapshot'], current), '数据、材料或来源变化；旧断点禁止直接复用')
    if spec.get('acceptance'):
        from ..acceptance import rows
        rows(spec['acceptance'], spec)
    from .. import gates
    gates.verify_materials(spec, historical_directory=historical_directory)


def verify_generation(spec, cases):
    """Training/pack generators must check sources even when all outputs exist."""
    from src.data.history.sources import load
    seal = RunSeal([versions.data_version_dir(spec['data_ref']),
                    versions.generator_dir(spec['generator_ref'])], spec.get('inputs', {}))
    history = load(versions.data_version_dir(spec['data_ref']))
    role = ('judge_training' if spec.get('dataset') == 'training' else
            'judge_development' if spec.get('dataset') == 'development' else spec.get('dataset'))
    _require(role in ('judge_training', 'judge_development', 'fixed_test'), '生成任务缺少明确的数据角色')
    from .. import acceptance
    if role == 'fixed_test' and acceptance.exists(spec['data_ref']) and not spec.get('acceptance'):
        raise ConfigError('sealed generation requires a batch receipt')
    if role == 'fixed_test' and spec.get('acceptance'):
        from ..acceptance import rows as batch_rows
        canonical = {c['case_id']: c for c in batch_rows(spec['acceptance'])}
    else:
        canonical = {c['case_id']: c for c in history.role(role)}
    _require(len({c['case_id'] for c in cases}) == len(cases), '生成任务含重复样本')
    for c in cases:
        _require(canonical.get(c['case_id']) == c, '生成样本不属于冻结的当前角色清单')
    _require(bool(spec.get('inputs') or spec.get('learning_snapshot')), '生成任务缺少冻结来源依赖')
    if spec.get('inputs'):
        # Current studies already pin source and code files. Never re-sign them.
        for p in history.files:
            _require(spec['inputs'].get(str(p.resolve())) == sha256_file(p), '生成来源未冻结或已变化')
        for name, expected in spec['inputs'].items():
            _require(Path(name).is_file() and sha256_file(Path(name)) == expected, f'冻结输入已变化：{name}')
    if spec.get('learning_snapshot'):
        current = snapshot(spec['data_ref'], [versions.generator_dir(spec['generator_ref'])],
                           spec['learning_snapshot']['role'])
        _require(same_materials(spec['learning_snapshot'], current), '生成来源依赖已变化，禁止复用')
    else:
        require_materials(spec['data_ref'], [versions.generator_dir(spec['generator_ref'])])
    seal.check()
    return seal


def execution_seal(spec):
    frozen = spec.get('learning_snapshot') or {}
    _require(bool(frozen), '实验缺少冻结来源证明；保留旧结果，禁止补签后直接续跑')
    directories = [versions.data_version_dir(spec['data_ref']),
                   *[Path(p) for p in frozen['materials']]]
    if spec['kind'] == 'judge_eval':
        directories.append(versions.PRIVATE / 'judge_eval' / spec['pack_ref'])
    from .. import gates
    source = Path(__file__).parents[2]
    from .. import runtime
    files = runtime.require_current(versions.PRIVATE / 'experiments' / spec['id'])
    files.extend(source / name for name in frozen.get('guard_code', {}))
    files.extend(source / name for name in spec.get('evaluation_materials', {}).get('implementation', {}))
    files.extend(Path(name) for name in spec.get('evaluation_materials', {}).get('saved_draw_inputs', {}))
    files.extend(gates.judge_templates(spec))
    seal = RunSeal(directories, files)
    verify(spec)
    seal.check()
    return seal


def verify_training(directory, spec, rows, entries=None, *, complete=False):
    """Reconstruct the supported clean training recipe before extraction/refit.

    Dataset labels alone cannot authorize learning. Verify raw spans, reference
    evidence, train-only vocabulary, generated negatives and actual feature
    requests. Older recipes without that evidence remain archived, not runnable.
    """
    from src.infrastructure import cache
    from src.data.history.sources import load
    from .. import training_evidence as audit
    from ..training_evidence import generation_cache, generations
    from ..training_evidence import Evidence
    from ..training_evidence import training_rows
    directory = Path(directory)
    _require(spec.get('kind') in ('history_lr_training', 'judge_material_repair'),
             '训练缺少可重建的干净来源配方；禁止继承未知旧权重、词表或参考材料')
    arm = directory.name
    _require(Path(arm).name == arm and arm not in ('.', '..'), '训练分支缺少明确来源')
    study = directory.parent
    try:
        root = json.loads((study / 'spec.json').read_text())
        _require(arm in root['feature_configs'], '训练分支未在实例配方中声明')
        from .. import runtime
        executor_files = runtime.require_current(study) if root['kind'] == 'history_lr_training' else []
        seal = RunSeal([versions.data_version_dir(root['data_ref'])],
                       [*executor_files, study / 'spec.json', *audit.verified_input_paths(root), study / 'generations.json',
                        study / 'sources.json', study / 'source_audit.json', study / 'reference.json',
                        study / 'profile.json', study / 'prompt.md', study / 'model_template.json'])
        history = load(versions.data_version_dir(root['data_ref']))
        train = history.role('judge_training')
        history.role('judge_development')
        _require(sha256_file(study / 'profile.json') == GENERIC_PROFILE,
                 'Judge 指令包含未核验的继承内容')
        _require(spec == {**root, 'id': study.name + '-' + arm,
                          'feature_config': root['feature_configs'][arm],
                          'training_sha256': cache.digest(rows)}, '训练规格或标签未绑定当前来源')
        evidence = Evidence()
        try:
            selected = audit.sources(study, root, evidence)
            from ...judge import corrected_v1 as rt
            reference = json.loads((study / 'reference.json').read_text())
            prompt = rt.build_shared_single_case_prompt({}, reference, json.loads((study / 'profile.json').read_text()))
            prompt = prompt.replace('<blind_case>\n{}\n</blind_case>', '<blind_case>\n{{case}}\n</blind_case>')
            prompt = prompt.replace('参考资料来自与测试聊天隔离的 2026-04-01 前真人聊天，以及用户确认的本人事实。',
                '参考示例仅来自本次开发时间边界之前的可核验历史聊天，不含留出聊天或额外个人事实。')
            _require((study / 'prompt.md').read_text() == prompt, '训练 Judge 模板与核验材料不符')
            _require({c['case_id']: c for c in selected['train']} == {c['case_id']: c for c in train},
                     '训练样本与原始清单不符')
            pool = {r['id']: r for r in audit.lines(history.directory / 'fewshot_pool.jsonl')}
            with generation_cache(evidence, study):
                generated = generations(study, root, selected['train'], pool, evidence, True)
            expected = training_rows(selected['train'], {'entries': generated})
            _require(cache.digest(rows) == cache.digest(expected), '训练 A/B、标签或生成回复不属于核验来源')
            if entries is not None:
                verify_features(directory, rows, entries, spec['feature_config'], evidence, complete=complete)
        finally:
            evidence.store.db.close()
        seal.check()
        return seal
    except (ValueError, KeyError, OSError) as exc:
        raise ConfigError(f'训练来源核验失败：{exc}') from exc


def verify_features(directory, rows, entries, config, evidence, *, complete=False):
    """A feature digest is insufficient without the request which produced it."""
    from src.infrastructure import cache
    from ..training_evidence import generation_cache
    by_id = {r['case_id']: r for r in rows}
    _require(len(by_id) == len(rows) and set(entries) <= by_id.keys(), '特征包含重复或额外训练样本')
    if complete:
        _require(set(entries) == set(by_id) and all(e.get('status') == 'ok' for e in entries.values()),
                 '训练特征未完整通过核验')
    with generation_cache(evidence, directory):
        for cid, entry in entries.items():
            row = by_id[cid]
            _require(entry['input_sha256'] == cache.digest(row), '训练特征输入或标签变化')
            if entry['status'] != 'ok':
                continue
            _require(entry['features_sha256'] == cache.digest(entry['features']), '训练特征输出变化')
            ref = entry.get('trace_ref', '')
            _require(bool(ref) and Path(ref).name == ref, '训练特征缺少有效请求来源')
            trace = evidence.document(Path(directory) / 'traces' / (ref + '.json'))
            _require(trace['status'] == 'ok' and trace['case']['blind'] == row['blind']
                     and len(trace['operations']) == 1, '训练特征请求与实际样本不符')
            op = trace['operations'][0]
            _require((op['kind'], op['branch'], op['round'], op['status']) ==
                     ('feature_extraction', 'training', 0, 'ok'), '训练特征请求用途或轮次不符')
            evidence.feature_request(op, cid, row['blind'], entry['features'], config)


def verify_fit(rows, entries, artifact):
    """No direct fitting from arbitrary inherited vocabulary or checkpoints."""
    from src.infrastructure import cache
    artifact = Path(artifact)
    _require(artifact.name == 'model_template.json' and (artifact.parent / 'spec.json').is_file(),
             '拟合缺少干净训练来源，禁止直接使用旧模型模板')
    root = json.loads((artifact.parent / 'spec.json').read_text())
    for arm in root.get('feature_configs', {}):
        _require(Path(arm).name == arm and arm not in ('.', '..'), '非法训练分支')
        directory = artifact.parent / arm
        if not (directory / 'spec.json').is_file() or not (directory / 'features.json').is_file():
            continue
        inputs = RunSeal([directory], [artifact, artifact.parent / 'spec.json'])
        spec = json.loads((directory / 'spec.json').read_text())
        if spec.get('training_sha256') != cache.digest(rows):
            continue
        from ..training_evidence import checkpoint
        saved = checkpoint(directory, spec, rows)
        if cache.digest(saved['entries']) == cache.digest(entries):
            seal = verify_training(directory, spec, rows, entries, complete=True)
            inputs.check()
            seal.materials.extend(inputs.materials)
            seal.files.update(inputs.files)
            return seal
    raise ConfigError('拟合特征与冻结训练分支不符')


def verify_pack(pack, dataset):
    """A newly frozen evaluation spec must not bless old unproven AI replies."""
    from src.infrastructure import cache
    proof = pack.get('generation_proof')
    _require(isinstance(proof, dict), '回复包缺少生成前冻结的来源证明；禁止补签旧回复')
    recipe = proof.get('recipe', {})
    _require(recipe.get('data_ref') == pack.get('data_ref') and
             recipe.get('generator_ref') == pack.get('c0_gen_version') and
             recipe.get('dataset') == dataset, '回复包生成来源与使用条件不符')
    _require(proof.get('rows_sha256') == cache.digest(pack['rows']), '回复包行内容与生成证明不符')
    from ..datasets import pack_cases
    verify_generation(recipe, pack_cases(pack))
    if recipe.get('generation_reuse'):
        from ..generation_reuse import verify_rows
        verify_rows(recipe, pack_cases(pack), pack['rows'])


def pack_proof(recipe, rows):
    """Called by guarded builders only after generating under a frozen recipe."""
    from src.infrastructure import cache
    return {'recipe': recipe, 'rows_sha256': cache.digest(rows)}


def verify_repair(spec):
    marker = spec.get('material_repair')
    if marker is None:
        return
    from .. import gates
    study = Path(marker['study'])
    if sha256_file(study / 'spec.json') != marker['spec_sha256']:
        raise ConfigError('修复比较的训练条件已变化')
    source = json.loads((study / 'spec.json').read_text())
    for name, digest in source['inputs'].items():
        if not Path(name).is_file() or sha256_file(Path(name)) != digest:
            raise ConfigError(f'冻结学习输入已变化：{name}')
    if versions.load_pointers() != source['baseline_pointers']:
        raise ConfigError('生产或开发基线已变化')
    if (spec['data_ref'] != source['data_ref'] or spec['protocol'] != source['protocol']
            or spec.get('lr_retrain') or spec['dataset'] != 'development'
            or marker.get('adoption_allowed') is not False):
        raise ConfigError('修复比较范围或评分口径变化')
    if spec.get('evaluation_materials') != gates.materials(spec):
        raise ConfigError('修复比较模型或评分实现已变化')
    require_materials(source['data_ref'], [versions.generator_dir(source['generator_ref']),
        *[versions.judge_dir(spec[k])['dir'] for k in ('baseline_ref', 'candidate_ref')]])


# ---------------------------------------------------------------------------
# 验证复用（设计 §4 正式化，receipt v6）。
# 与活适配器层 src/iteration/material_reuse.py 存在刻意重复：适配器按路径加载、
# 服务归档 runtime（MODE v5），本实现服务新代码（MODE v6），全部 patch/缓存均在
# 本模块自身作用域完成，不依赖任何外部模块 patch 本模块的全局名。
# ---------------------------------------------------------------------------

@contextmanager
def verified_history_pools(retrievers, sources, *, disk_rows=None, profile=None, capacity=POOL_CAPACITY):
    """Scope exact-row reuse to full material validation and pool approval."""
    if capacity <= 0:
        raise ValueError('history pool validation capacity must be positive')
    original_validate = sources.HistorySources.validate
    original_check = sources.HistorySources.check
    originals = {name: getattr(retrievers.PersonaFewShotRetriever, name) for name in ('_load', 'is_approved')}
    entries = {}
    registry = Lock()
    scoped = local()

    def source_entry(frame, source):
        entry = frame['sources'].get(source)
        if entry is None:
            original_check(source)
            reader = getattr(source, '_disk_reader', None)
            if reader is not None:
                frame['reads'].enter_context(reader.reading())
            binding = (tuple(source.files), tuple(source.stamps))
            with registry:
                saved = entries.get(source)
                known = saved[1] if saved is not None and saved[0] == binding else frozenset()
            entry = dict(binding=binding, known=known, pending=set())
            frame['sources'][source] = entry
        return entry

    @wraps(original_check)
    def check(source):
        frame = getattr(scoped, 'frame', None)
        if frame is None:
            return original_check(source)
        source_entry(frame, source)
        if profile is not None:
            profile.record('history_source_stamp_check_reused', 0.0)

    @wraps(original_validate)
    def validate(source, row, *, example=False):
        frame = getattr(scoped, 'frame', None)
        if frame is None or not example:
            return original_validate(source, row, example=example)
        entry = source_entry(frame, source)
        if disk_rows is not None and isinstance(row, disk_rows.Row):
            value = dict(row.reader.row(row.offset, row.size))
            value.update({name: row.header[name] for name in value if name in row.header})
        else:
            value = dict(row)
        token = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(',', ':')).encode()).digest()
        if token in entry['known'] or token in entry['pending']:
            if profile is not None:
                profile.record('history_pool_validation_reused', 0.0)
            return row
        original_validate(source, value, example=True)
        if profile is not None:
            profile.record('history_pool_validation_full', 0.0)
        if len(entry['pending']) < capacity:
            entry['pending'].add(token)
        return row

    def pool(original):
        @wraps(original)
        def call(subject, *args, **kwargs):
            frame = getattr(scoped, 'frame', None)
            outer = frame is None
            if outer:
                frame = dict(sources={}, valid=True, reads=ExitStack())
                scoped.frame = frame
            try:
                reader = getattr(subject, '_disk_reader', None)
                if outer and reader is not None:
                    frame['reads'].enter_context(reader.reading())
                source = getattr(subject, '_history_sources', None)
                if source is not None:
                    source_entry(frame, source)
                result = original(subject, *args, **kwargs)
                if result is False:
                    frame['valid'] = False
                if outer:
                    frame['reads'].close()
                    for source, entry in frame['sources'].items():
                        original_check(source)
                        if entry['binding'] != (tuple(source.files), tuple(source.stamps)):
                            frame['valid'] = False
                    if frame['valid']:
                        with registry:
                            size = sum(len(saved[1]) for saved in entries.values())
                            for source, entry in frame['sources'].items():
                                if source not in entries and len(entries) >= CAPACITY:
                                    continue
                                saved = entries.get(source)
                                known = saved[1] if saved is not None and saved[0] == entry['binding'] else frozenset()
                                if saved is not None and saved[0] != entry['binding']:
                                    size -= len(saved[1])
                                pending = entry['pending'] - known
                                available = max(0, capacity - size)
                                if len(pending) > available:
                                    pending = set(sorted(pending)[:available])
                                entries[source] = entry['binding'], known.union(pending)
                                size += len(pending)
                return result
            except BaseException:
                frame['valid'] = False
                raise
            finally:
                if outer:
                    try:
                        frame['reads'].close()
                    finally:
                        scoped.frame = None
        return call

    try:
        sources.HistorySources.validate = validate
        sources.HistorySources.check = check
        for name, original in originals.items():
            setattr(retrievers.PersonaFewShotRetriever, name, pool(original))
        yield pool
    finally:
        sources.HistorySources.validate = original_validate
        sources.HistorySources.check = original_check
        for name, original in originals.items():
            setattr(retrievers.PersonaFewShotRetriever, name, original)


def judge_dependencies(directory):
    from src.infrastructure.config import settings_path
    from .. import runtime
    from src.infrastructure.storage import read_json

    source = directory / 'source_judge.json'
    if not source.is_file():
        return None
    origin = read_json(source)
    names = [origin.get('study'), origin.get('arm')]
    if (origin.get('origin') not in ('history_lr_v1', 'd0011_material_repair')
            or any(not isinstance(name, str) or not name or name in ('.', '..')
                   or Path(name).name != name for name in names)
            or read_json(directory / 'config.json').get('decision_policy') != 'gbdt_only'):
        return None
    study = versions.PRIVATE / 'judge_training' / names[0]
    spec = read_json(study / 'spec.json')
    if spec.get('kind') not in ('history_lr_training', 'judge_material_repair'):
        return None
    descriptor = read_json(directory / 'scorer_source.json')
    name = descriptor.get('study')
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or Path(name).name != name):
        return None
    sweep = versions.PRIVATE / 'judge_training' / name
    sweep_spec = read_json(sweep / 'spec.json')
    original = read_json(study / names[1] / 'candidate.json')['judge_ref']
    generator = versions.generator_dir(spec['generator_ref'])
    bound = read_json(generator / 'learning.json').get('asset_files', {})
    if bound not in GENERATOR_RECIPES.values():
        return None
    directories = [versions.data_version_dir(spec['data_ref']), study, sweep,
                   versions.judge_dir(original)['dir'], generator]
    files = [runtime.evidence_path(path, digest) for inputs in (spec['inputs'], sweep_spec['inputs'])
             for path, digest in inputs.items()]
    files.append(settings_path())
    packages = sorted(set(sweep_spec['packages']) | {'numpy', 'scipy', 'scikit-learn'})
    return directories, files, packages


def dependencies(data_ref, directories):
    from src.generator.fewshot import provenance as learned_sources
    from src.infrastructure.storage import read_json

    data = {versions.data_version_dir(data_ref)}
    rankers = []
    files = set()
    packages = set()
    for directory in directories:
        record = read_json(directory / 'learning.json')
        bound = record.get('asset_files', {})
        if bound in GENERATOR_RECIPES.values() or bound == {'prompt.md': GENERIC_JUDGE_TEMPLATE}:
            continue
        if 'ranker/provenance.json' not in bound:
            judge = judge_dependencies(directory)
            if judge is None:
                return None
            extra, inputs, environment = judge
            data.update(extra)
            files.update(inputs)
            packages.update(environment)
            continue
        proof = read_json(directory / 'ranker/provenance.json')
        key = (str(Path(proof['model_directory']).resolve()), proof['data_ref'],
               learned_sources.digest(proof['evidence_files']))
        rankers.append(key)
        data.add(versions.data_version_dir(proof['data_ref']))
    return sorted(data, key=str), rankers, sorted(files, key=str), sorted(packages)


def environment_record(packages):
    return {name: importlib.metadata.version(name) for name in packages}


def cache_fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@contextmanager
def cached_training_reads():
    from ..training_evidence import SavedCache

    original = SavedCache.get
    scoped = local()

    @wraps(original)
    def get(store, key):
        value = original(store, key)
        records = getattr(scoped, 'records', None)
        if records is not None:
            path = store.db.execute('PRAGMA database_list').fetchone()[2]
            identity = (str(Path(path).resolve()), key)
            fingerprint = cache_fingerprint(value)
            if identity in records and records[identity] != fingerprint:
                raise ConfigError('材料核验期间训练请求缓存变化')
            records[identity] = fingerprint
        return value

    @contextmanager
    def capture():
        previous = getattr(scoped, 'records', None)
        records = {}
        scoped.records = records
        try:
            yield records
        finally:
            scoped.records = previous
            if previous is not None:
                for identity, fingerprint in records.items():
                    if identity in previous and previous[identity] != fingerprint:
                        raise ConfigError('嵌套材料核验期间训练请求缓存变化')
                    previous[identity] = fingerprint

    def check(records):
        stores = {}
        try:
            for (path, key), expected in records.items():
                if path not in stores:
                    stores[path] = SavedCache(Path(path))
                _require(cache_fingerprint(original(stores[path], key)) == expected,
                         '已核验训练请求缓存变化；禁止复用材料核验')
        except sqlite3.Error as exc:
            raise ConfigError('训练请求缓存无法读取；禁止复用材料核验') from exc
        finally:
            for store in stores.values():
                store.db.close()

    SavedCache.get = get
    try:
        yield capture, check
    finally:
        SavedCache.get = original


def ranker_stamps(keys):
    from src.generator.fewshot import provenance as learned_sources

    stamps = {}
    for key in keys:
        verified = learned_sources._verified.get(key)
        if verified is None:
            return None
        for path, value in verified[0].items():
            if path in stamps and stamps[path] != value:
                return None
            stamps[path] = value
    return stamps


def check_stamps(stamps):
    from src.data.history.sources import stamp

    _require(all(stamp(path) == saved for path, saved in stamps.items()),
             '运行中已核验 Ranker 依赖变化；禁止复用材料核验')


def seal_record(seal):
    return dict(materials=[dict(directory=str(item.directory), stamps=item.stamps,
                               evidence={str(path): value for path, value in item.evidence.items()})
                          for item in seal.materials],
                files={str(path): value for path, value in seal.files.items()})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def receipt_path(data_ref, directories, role):
    key = dict(mode=MODE, data_ref=data_ref, directories=[str(path) for path in directories], role=role)
    identity = hashlib.sha256(canonical(key).encode()).hexdigest()
    root = versions.data_version_dir(data_ref).parent.parent / '.cache' / 'material-validation'
    return root / (identity + '.json'), key


def restore_ranker_record(item):
    from src.generator.fewshot import provenance as learned_sources
    from src.data.history.sources import stamp
    from src.infrastructure.storage import read_json

    stamps = {Path(path): tuple(value) for path, value in item['stamps'].items()}
    result = copy.deepcopy(item['result'])
    if 'evidence_files' not in result:
        return stamps, result
    manifest = read_json(Path(item['key'][0]).parent / 'manifest.json')
    evidence = dict(result['evidence_files'])
    logical, actual = learned_sources.historical_runtime(manifest, evidence)
    _require(learned_sources.serving_runtime(manifest) == result['serving_runtime'],
             '已核验 Ranker 推理代码变化；禁止复用材料核验')
    for name in manifest['runtime']:
        aliases = [path for path in evidence if path.endswith('/' + name)]
        _require(len(aliases) == 1, '已核验 Ranker 代码证据路径不唯一')
        evidence.pop(aliases[0])
    evidence.update(logical)
    result['evidence_files'] = evidence
    for path in actual:
        current = stamp(path)
        _require(path not in stamps or stamps[path] == current,
                 '恢复期间已核验 Ranker 代码变化；禁止复用材料核验')
        stamps[path] = current
    return stamps, result


def read_receipt(path, key, seal, rankers, environment, check_cache):
    from src.generator.fewshot import provenance as learned_sources

    if not path.exists():
        return None
    _require(path.is_file() and not path.is_symlink(), '材料核验缓存不是普通文件')
    try:
        value = json.loads(path.read_text())
        payload = value['payload']
        _require(value['sha256'] == hashlib.sha256(canonical(payload).encode()).hexdigest()
                 and payload['key'] == key, '材料核验缓存内容损坏')
        if (payload['seal'] != json.loads(canonical(seal_record(seal)))
                or payload['environment'] != environment):
            return None
        cache_rows = {(item['path'], item['key']): item['sha256'] for item in payload['cache_rows']}
        check_cache(cache_rows)
        records = payload['rankers']
        _require([tuple(item['key']) for item in records] == rankers, '材料核验依赖集合不符')
        stamps = {Path(path): tuple(value) for item in records for path, value in item['stamps'].items()}
        from src.generator.fewshot.runtime_layout import retired_runtime_path
        if any(not path.is_file() and retired_runtime_path(path) for path in stamps):
            return None
        check_stamps(stamps)
        restored = {tuple(item['key']): restore_ranker_record(item) for item in records}
        for saved, _ in restored.values():
            stamps.update(saved)
        check_stamps(stamps)
        seal.check()
        learned_sources._verified.update(restored)
        return seal, stamps, payload['audit'], cache_rows, environment
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError('材料核验缓存无法读取；禁止复用') from exc


def write_receipt(path, key, seal, rankers, audit, environment, cache_rows):
    from src.generator.fewshot import provenance as learned_sources
    from src.infrastructure.storage import write_json

    records = [dict(key=key, stamps={str(path): value for path, value in learned_sources._verified[key][0].items()},
                    result=learned_sources._verified[key][1]) for key in rankers]
    payload = dict(key=key, seal=seal_record(seal), rankers=records, audit=audit,
                   environment=environment,
                   cache_rows=[dict(path=path, key=key, sha256=value)
                               for (path, key), value in sorted(cache_rows.items())])
    write_json(path, dict(payload=payload, sha256=hashlib.sha256(canonical(payload).encode()).hexdigest()))


@contextmanager
def verified_materials(*, profile=None, capacity=CAPACITY, disk_rows=None):
    """Validate new material closures once; keep cheap change checks and per-case guards.

    正式复用实现（receipt v6）。patch 只改写本模块的 require_materials 全局名，
    调用方一律经模块属性访问（validation.require_materials），保证 patch 生效点唯一。
    """
    if capacity <= 0:
        raise ValueError('material validation capacity must be positive')
    original = require_materials
    entries = {}
    registry = Lock()

    def timing(name):
        return profile.span(name) if profile is not None else nullcontext()

    def full(data_ref, directories, role):
        with timing('material_validation_full'):
            return original(data_ref, directories, role)

    def reuse(saved):
        seal, stamps, audit, cache_rows, environment = saved
        with timing('material_validation_reuse_check'):
            seal.check()
            check_stamps(stamps)
            check_cache(cache_rows)
            _require(environment_record(environment) == environment,
                     '已核验训练依赖版本变化；禁止复用材料核验')
        return copy.deepcopy(audit)

    @wraps(original)
    def require_materials_reuse(data_ref, directories, role='development'):
        directories = tuple(Path(directory).absolute() for directory in directories)
        key = (data_ref, directories, role)
        with registry:
            if key not in entries and len(entries) < capacity:
                entries[key] = dict(lock=Lock(), saved=None)
            entry = entries.get(key)
            saved = entry['saved'] if entry is not None else None
        if entry is None:
            return full(data_ref, directories, role)
        if saved is not None:
            return reuse(saved)
        with timing('material_validation_lock_wait'):
            entry['lock'].acquire()
        try:
            with registry:
                saved = entry['saved']
            if saved is None:
                with timing('material_validation_seal'):
                    initial = RunSeal(directories)
                    closure = dependencies(data_ref, directories)
                    initial.check()
                    if closure is not None:
                        data, rankers, files, packages = closure
                        seal = RunSeal(data, files)
                        seal.materials.extend(initial.materials)
                        environment = environment_record(packages)
                if closure is None:
                    return full(data_ref, directories, role)
                from src.infrastructure.storage import file_lock
                path, receipt_key = receipt_path(data_ref, directories, role)
                with ExitStack() as receipt_lock:
                    with timing('material_validation_receipt_wait'):
                        receipt_lock.enter_context(file_lock(path.with_suffix('.lock')))
                    with timing('material_validation_receipt_read'):
                        verified = read_receipt(path, receipt_key, seal, rankers,
                                                environment, check_cache)
                    if verified is None:
                        with capture_cache() as cache_rows:
                            audit = full(data_ref, directories, role)
                        with timing('material_validation_seal'):
                            stamps = ranker_stamps(rankers)
                            seal.check()
                            if stamps is not None:
                                check_stamps(stamps)
                                check_cache(cache_rows)
                                _require(environment_record(packages) == environment,
                                         '材料核验期间训练依赖版本变化')
                                verified = seal, stamps, copy.deepcopy(audit), dict(cache_rows), environment
                                write_receipt(path, receipt_key, seal, rankers, audit,
                                              environment, cache_rows)
                    else:
                        audit = copy.deepcopy(verified[2])
                        if profile is not None:
                            profile.record('material_validation_persistent_reused', 0.0)
                    with registry:
                        entry['saved'] = verified
                    return audit
        finally:
            entry['lock'].release()
        return reuse(saved)

    # disk_rows 路径仅服务旧类（适配器层场景）；新包由 PreparedPool.load 原生核验，
    # 不触碰 few_shot。条件导入保证 few_shot.py 从活树删除后本模块仍可用。
    history_pools = nullcontext()
    if disk_rows is not None:
        from ...generator import few_shot
        from src.data.history import sources as history_sources
        history_pools = verified_history_pools(few_shot, history_sources,
                                               disk_rows=disk_rows, profile=profile)
    try:
        with history_pools as pool_validation, cached_training_reads() as (capture_cache, check_cache):
            globals()['require_materials'] = (pool_validation(require_materials_reuse)
                                              if pool_validation is not None else require_materials_reuse)
            yield
    finally:
        globals()['require_materials'] = original
