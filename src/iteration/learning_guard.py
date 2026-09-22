"""Mandatory source checks for creation, execution, resume and adoption.

Only independently reconstructible material recipes can be used. Old assets
remain on disk, but a self-declared learning.json never certifies them.
"""
import json
from pathlib import Path

from ..config import ConfigError, sha256_file
from . import datasets, versions


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
    from ..generator.history_sources import load
    history = load(versions.data_version_dir(source_data))
    train = history.role('judge_training')
    history.role('judge_development')
    _require(sha256_file(study / 'profile.json') == GENERIC_PROFILE,
             'Judge 指令包含未核验的继承内容')
    # Independent audit checks the complete reference spans, zero starting weights,
    # train-only vocabulary, generation/feature request traces and fitted weights.
    from . import training_evidence as audit
    from .training_evidence import Evidence
    from .training_evidence import generation_cache, generations
    evidence = Evidence()
    try:
        selected = audit.sources(study, spec, evidence)
        _require({c['case_id']: c for c in selected['train']} == {c['case_id']: c for c in train},
                 '实际训练题与原始训练清单不符')
        reference = json.loads((study / 'reference.json').read_text())
        from ..judge import corrected_v1 as rt
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
            from .embedding_adoption import verify as verify_embedding
            verify_embedding(directory, record, original, study, spec, arm)
        elif config.get('decision_policy') == 'score_fusion':
            from .fusion import verify as verify_fusion
            verify_fusion(directory, record, original, study, spec, arm)
        elif config.get('decision_policy') == 'gbdt_only':
            from .gbdt_evidence import verify as verify_gbdt
            verify_gbdt(directory, record, original, study, spec, arm)
        else:
            _require(record['asset_files'] == original_record['asset_files'],
                     '派生 Judge 的学习资产与重建来源不符')
        for name, digest in original_record['asset_files'].items():
            _require(sha256_file(original / name) == digest, '原始训练资产变化')
        if source_data != data_ref:
            from .material_compatibility import judge_sources, require_current
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
            from ..generator.learned_sources import verify as verify_ranker
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
    from ..generator.history_sources import load
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
            'guard_code': {name: sha256_file(Path(__file__).parents[1] / name) for name in
                ('iteration/learning_guard.py', 'iteration/training_evidence.py', 'iteration/gbdt_evidence.py',
                 'iteration/fusion.py', 'iteration/embedding_adoption.py',
                 'iteration/material_compatibility.py',
                 'iteration/datasets.py', 'generator/history_sources.py', 'generator/learned_sources.py')}}


def bind(spec):
    """Only new, not-yet-run specs may receive this snapshot."""
    role = spec.get('purpose') or ('judge_development' if spec['kind'] == 'judge_eval'
                                  and spec['dataset'] == 'development' else spec['dataset'])
    spec['learning_snapshot'] = snapshot(spec['data_ref'], _directories(spec), role)


def verify(spec):
    _require(bool(spec.get('learning_snapshot')), '实验缺少冻结来源证明；保留旧结果，禁止补签后直接续跑')
    role = spec.get('purpose') or ('judge_development' if spec['kind'] == 'judge_eval'
                                  and spec['dataset'] == 'development' else spec['dataset'])
    current = snapshot(spec['data_ref'], _directories(spec), role)
    _require(spec['learning_snapshot'] == current, '数据、材料或来源规则变化；旧断点禁止直接复用')
    if spec.get('acceptance'):
        from .acceptance import rows
        rows(spec['acceptance'], spec)
    from . import gates
    _require(spec.get('evaluation_materials') == gates.materials(spec), '模型或评分条件已变化')


def verify_generation(spec, cases):
    """Training/pack generators must check sources even when all outputs exist."""
    from ..generator.history_sources import load
    seal = RunSeal([versions.data_version_dir(spec['data_ref']),
                    versions.generator_dir(spec['generator_ref'])], spec.get('inputs', {}))
    require_materials(spec['data_ref'], [versions.generator_dir(spec['generator_ref'])])
    history = load(versions.data_version_dir(spec['data_ref']))
    role = ('judge_training' if spec.get('dataset') == 'training' else
            'judge_development' if spec.get('dataset') == 'development' else spec.get('dataset'))
    _require(role in ('judge_training', 'judge_development', 'fixed_test'), '生成任务缺少明确的数据角色')
    from . import acceptance
    if role == 'fixed_test' and acceptance.exists(spec['data_ref']) and not spec.get('acceptance'):
        raise ConfigError('sealed generation requires a batch receipt')
    if role == 'fixed_test' and spec.get('acceptance'):
        from .acceptance import rows as batch_rows
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
        _require(current == spec['learning_snapshot'], '生成来源依赖已变化，禁止复用')
    seal.check()
    return seal


class MaterialSeal:
    """Cheap per-request check after a full provenance check at construction."""
    def __init__(self, directory):
        self.directory = Path(directory)
        from ..generator.history_sources import stamp
        record = self.directory / 'learning.json'
        before = stamp(record) if record.exists() else None
        try:
            evidence = json.loads(record.read_text()).get('evidence_files', {}) if before else {}
        except (OSError, ValueError, AttributeError) as exc:
            raise ConfigError('学习来源记录无法读取，禁止继续') from exc
        _require(isinstance(evidence, dict), '学习来源证据格式无效，禁止继续')
        from .runtime import evidence_path
        self.evidence = {evidence_path(p, h): stamp(evidence_path(p, h)) for p, h in evidence.items()}
        self.stamps = self._stamps()
        _require(before == (stamp(record) if record.exists() else None), '读取期间学习来源发生变化')

    def _stamps(self):
        from ..generator.history_sources import stamp
        return {str(p.relative_to(self.directory)): stamp(p)
                for p in self.directory.rglob('*') if p.is_file()
                and p.relative_to(self.directory) != Path('meta.json')}

    def check(self):
        from .control import check
        check()
        from ..generator.history_sources import stamp
        _require(self.stamps == self._stamps(), '运行中学习材料变化；禁止继续请求或复用缓存')
        _require(all(stamp(p) == value for p, value in self.evidence.items()),
                 '运行中学习证据变化；禁止继续请求或复用缓存')


class RunSeal:
    """Watch the dependency set captured *before* validation and loading.

    File stamps are an inexpensive in-process mutation check. Content hashes and
    source reconstruction remain mandatory whenever a run is opened or resumed.
    """
    def __init__(self, directories=(), files=()):
        from ..generator.history_sources import stamp
        self.materials = [MaterialSeal(d) for d in directories]
        self.files = {Path(p): stamp(Path(p)) for p in files}

    def check(self):
        from .control import check
        check()
        from ..generator.history_sources import stamp
        for seal in self.materials:
            seal.check()
        _require(all(stamp(p) == saved for p, saved in self.files.items()),
                 '运行中冻结输入变化；禁止继续请求、写入有效结果或晋级')


def execution_seal(spec):
    frozen = spec.get('learning_snapshot') or {}
    _require(bool(frozen), '实验缺少冻结来源证明；保留旧结果，禁止补签后直接续跑')
    directories = [versions.data_version_dir(spec['data_ref']),
                   *[Path(p) for p in frozen['materials']]]
    if spec['kind'] == 'judge_eval':
        directories.append(versions.PRIVATE / 'judge_eval' / spec['pack_ref'])
    from . import gates
    source = Path(__file__).parents[1]
    from . import runtime
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
    from .. import cache
    from ..generator.history_sources import load
    from . import training_evidence as audit
    from .training_evidence import generation_cache, generations
    from .training_evidence import Evidence
    from .training_evidence import training_rows
    directory = Path(directory)
    _require(spec.get('kind') in ('history_lr_training', 'judge_material_repair'),
             '训练缺少可重建的干净来源配方；禁止继承未知旧权重、词表或参考材料')
    arm = directory.name
    _require(Path(arm).name == arm and arm not in ('.', '..'), '训练分支缺少明确来源')
    study = directory.parent
    try:
        root = json.loads((study / 'spec.json').read_text())
        _require(arm in root['feature_configs'], '训练分支未在实例配方中声明')
        from . import runtime
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
            from ..judge import corrected_v1 as rt
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
    from .. import cache
    from .training_evidence import generation_cache
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
    from .. import cache
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
        from .training_evidence import checkpoint
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
    from .. import cache
    proof = pack.get('generation_proof')
    _require(isinstance(proof, dict), '回复包缺少生成前冻结的来源证明；禁止补签旧回复')
    recipe = proof.get('recipe', {})
    _require(recipe.get('data_ref') == pack.get('data_ref') and
             recipe.get('generator_ref') == pack.get('c0_gen_version') and
             recipe.get('dataset') == dataset, '回复包生成来源与使用条件不符')
    _require(proof.get('rows_sha256') == cache.digest(pack['rows']), '回复包行内容与生成证明不符')
    from .datasets import pack_cases
    verify_generation(recipe, pack_cases(pack))


def pack_proof(recipe, rows):
    """Called by guarded builders only after generating under a frozen recipe."""
    from .. import cache
    return {'recipe': recipe, 'rows_sha256': cache.digest(rows)}


def verify_repair(spec):
    marker = spec.get('material_repair')
    if marker is None:
        return
    from . import gates
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
