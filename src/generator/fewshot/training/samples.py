"""Reproducible single-example ranker inventories; no model requests or training.

Eligibility and raw-message reconstruction use the production history guards.
The inventory is useful even when a teacher is blocked, but is never described
as a labeled training set. Labels and fit/validation splits require a separately
authorized, eligible teacher and remain explicitly unset here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from src.infrastructure.config import ConfigError
from src.infrastructure.config import sha256_file
from src.infrastructure.storage import atomic_write
from src.infrastructure.storage import file_lock
from src.infrastructure.storage import read_json
from src.infrastructure.storage import write_json
from src.infrastructure.storage import write_once_json
from src.generator.fewshot import open_session
from src.generator.history import eligible
from src.data.history.sources import digest
from src.data.history.sources import load
from src.data.history.sources import require

RECIPE = dict(schema=1, role='gen_optimization', queries=[3, 1], route_limit=12,
              candidate_limit=8, baseline_limit=3, char_budget=2500,
              seed='single-example-v1', union_order='best_route_rank_then_id',
              strata_order='high_mid_low_x_same_different_unknown',
              teacher_policy='before_each_target_and_no_answer_overlap')

# 检索排序实现的完整清单（与 src/iteration/data_guard.RETRIEVAL_IMPLEMENTATION 同源）：
# 候选打分/召回语义由 session/prepare/rank/recall_engines 四件共同决定，清单变化
# 必须使已冻结清单失配，拒绝续跑。
IMPLEMENTATION_FILES = tuple(Path(__file__).parents[1] / name for name in
                             ('session.py', 'prepare.py', 'rank.py', 'recall_engines.py'))

# 过渡兼容：清单扩展前的旧 inventory 形态（只认两种，不做模糊匹配）：
# 1) 整体绑 generator/few_shot.py 单件；2) 只绑 fewshot/session.py 单件（无其余三件）。
# 旧清单按清单目录向上找到的 runtime 归档字节核验；归档缺失则失败并说明。
_LEGACY_MONOLITH = 'src/generator/few_shot.py'
_LEGACY_SESSION = 'src/generator/fewshot/session.py'
_FEWSHOT_OTHERS = ('prepare.py', 'rank.py', 'recall_engines.py')


def _legacy_implementation_inputs(inputs: dict) -> dict:
    """旧清单形态 → {清单键: 归档相对名}；非旧形态返回 {}。"""
    normalized = {str(k).replace('\\', '/'): k for k in inputs}
    monolith = [k for name, k in normalized.items() if name.endswith('/' + _LEGACY_MONOLITH)]
    if monolith:
        return {k: _LEGACY_MONOLITH for k in monolith}
    session = [k for name, k in normalized.items() if name.endswith('/' + _LEGACY_SESSION)]
    others = [k for name, k in normalized.items()
              if any(name.endswith('/src/generator/fewshot/' + n) for n in _FEWSHOT_OTHERS)]
    if len(session) == 1 and not others:
        return {session[0]: _LEGACY_SESSION}
    return {}


def _verify_legacy_manifest(output: Path, frozen: dict) -> None:
    """旧形态清单核验：仓库 src/ 下的全部输入件按 runtime 归档字节（实现件与
    本轮已改的检查件都算旧运行输入，活树现值不再是身份）；数据类输入（池、
    cases、teacher/generator 数据）仍按活树现值。"""
    runtime_json = next((parent / 'runtime.json'
                         for parent in (output, *output.parents) if (parent / 'runtime.json').is_file()), None)
    if runtime_json is None:
        raise ConfigError('旧形态候选清单缺少可核验的运行归档证据：自清单目录向上找不到 '
                          'runtime.json，无法按归档字节确认旧实现，拒绝续跑')
    from src.iteration import runtime
    try:
        archive = runtime.verify(runtime_json, check_environment=False)
        files = json.loads((archive / 'manifest.json').read_text(encoding='utf-8'))['files']
    except (OSError, KeyError, json.JSONDecodeError, ConfigError) as exc:
        raise ConfigError('旧形态候选清单的运行归档无效或已变化，无法按归档字节确认旧实现，'
                          '拒绝续跑') from exc
    repo_src = Path(__file__).resolve().parents[3]
    inputs = frozen['inputs']
    for key, expected in inputs.items():
        path = Path(key)
        try:
            relative = path.resolve().relative_to(repo_src)
        except ValueError:
            relative = None  # 数据类输入：不在仓库 src/ 下，按活树现值核验
        if relative is not None:
            archived_name = 'src/' + relative.as_posix()
            candidates = (archived_name, archived_name.replace('src/', 'src/digital_human/', 1))
            archived = next((files[name] for name in candidates if name in files), None)
            if archived is None:
                raise ConfigError(f'旧形态候选清单缺少归档证据：runtime 归档未包含 {archived_name}，'
                                  '无法确认旧输入，拒绝续跑')
            require(archived == expected,
                    f'旧实现清单与运行归档字节不符: {path.name}')
            continue
        require(path.is_file() and sha256_file(path) == expected,
                f'旧候选输入变化: {path.name}')


def _require_legacy_manifest_consistency(frozen: dict, current: dict) -> None:
    """旧清单放行仅限代码指纹的归档兼容：配方/版本/目标字段与数据类输入键集合
    必须与当前清单一致，防止换配方、换生成器资产或增删数据依赖复用旧清单。"""
    repo_src = Path(__file__).resolve().parents[3]
    frozen_data_keys = {str(key) for key in frozen.get('inputs', {})
                        if not Path(key).resolve().is_relative_to(repo_src)}
    current_data_keys = {str(key) for key in current.get('inputs', {})
                         if not Path(key).resolve().is_relative_to(repo_src)}
    require(frozen_data_keys == current_data_keys,
            '旧形态清单与当前输入集合不一致：数据类输入键集合已变化，拒绝续跑')
    for key, value in current.items():
        if key == 'inputs':
            continue
        require(frozen.get(key) == value, f'旧形态清单与当前 {key} 不一致，拒绝续跑')
    for key in frozen:
        if key != 'inputs' and key not in current:
            raise ConfigError(f'旧形态清单含当前不存在的字段 {key}，拒绝续跑')


def _resolve_manifest(output: Path, manifest: dict) -> dict:
    """manifest.json 已存在且为旧形态清单时，按归档字节核验后沿用冻结清单
    （归档字节才是旧运行的身份，实现演进不阻断续跑）；否则照旧用新清单。"""
    existing = output / 'manifest.json'
    if not existing.exists():
        return manifest
    frozen = read_json(existing)
    legacy_inputs = _legacy_implementation_inputs(frozen.get('inputs', {}))
    if legacy_inputs:
        _require_legacy_manifest_consistency(frozen, manifest)
        _verify_legacy_manifest(output, frozen)
        return frozen
    return manifest


def fingerprint(paths):
    return {str(Path(p).resolve()): sha256_file(Path(p)) for p in sorted(paths)}


def _evidence(teacher, suffix):
    matches = [(Path(p), sha) for p, sha in teacher['evidence_files'].items() if p.endswith(suffix)]
    require(len(matches) == 1, f'标签器证据不唯一或缺失: {suffix}')
    path, expected = matches[0]
    require(sha256_file(path) == expected, f'标签器来源变化: {path.name}')
    return path, read_json(path)


def teacher_overlap(source, cases, teacher_dir):
    """Separate temporal incompatibility from exact source-message overlap.

    This checks frozen learning-source evidence, not model execution; it cannot
    by itself authorize labeling. Never infer answer exposure from time alone.
    """
    teacher = read_json(teacher_dir / 'learning.json')
    require(teacher.get('verified') is True, '标签器缺少学习来源')
    end = teacher.get('information_end')
    require(type(end) in (int, float), '标签器学习截止时间缺失')
    sources_path, sources = _evidence(teacher, '/sources.json')
    audit_path, audit = _evidence(teacher, '/source_audit.json')
    train = sources['train']
    train_ids, train_cases = set(), set()
    for row in train:
        source.validate(row)
        train_cases.add(row['case_id'])
        train_ids.update(row['context_message_ids'] + row['reply_message_ids'])
    reference_ids = set()
    for span in audit['reference_spans']:
        messages = source.chats[span['chat_id']][span['start']:span['end']]
        require([m.message_id for m in messages] == span['message_ids'], '参考例子来源不符')
        require(messages and messages[-1].timestamp == span['information_end'], '参考例子时间不符')
        reference_ids.update(span['message_ids'])
    actual_end = max([r['source_span']['end_timestamp'] for r in train] +
                     [s['information_end'] for s in audit['reference_spans']])
    require(end == actual_end == audit['training_information_end'], '标签器学习截止声明与来源不符')
    rows = []
    for c in cases:
        answers = set(c['reply_message_ids'])
        overlap = answers & train_ids
        reference_overlap = answers & reference_ids
        future = end >= c['input_cutoff']['timestamp']
        rows.append(dict(target_id=c['case_id'], input_timestamp=c['input_cutoff']['timestamp'],
            teacher_time_violation=future, same_teacher_training_case=c['case_id'] in train_cases,
            answer_in_teacher_training=sorted(overlap), answer_in_teacher_references=sorted(reference_overlap),
            eligible_by_sources=not (future or overlap or reference_overlap)))
    summary = dict(targets=len(cases), teacher=teacher_dir.name, teacher_information_end=end,
        target_input_min=min(r['input_timestamp'] for r in rows),
        target_input_max=max(r['input_timestamp'] for r in rows), teacher_training_cases=len(train),
        teacher_reference_examples=len(audit['reference_spans']),
        time_blocked_targets=sum(r['teacher_time_violation'] for r in rows),
        same_teacher_training_case_targets=sum(r['same_teacher_training_case'] for r in rows),
        answer_in_teacher_training_targets=sum(bool(r['answer_in_teacher_training']) for r in rows),
        answer_in_teacher_reference_targets=sum(bool(r['answer_in_teacher_references']) for r in rows),
        source_eligible_targets=sum(r['eligible_by_sources'] for r in rows),
        label_authorized=False, scope='frozen_sources_only_not_model_execution')
    return dict(summary=summary, targets=rows), [sources_path, audit_path]


def object_match(target, example):
    # The archive has display names, not certified per-person IDs for groups.
    # A group ID or equal display name is deliberately not treated as a person.
    if target['chat_type'] == 'group' or example['relationship'] == 'group':
        return 'unknown'
    a, b = target.get('source_chat_id'), example.get('source_chat_id')
    if not a or not b:
        return 'unknown'
    return 'same' if a == b else 'different'


def select_candidates(case, routes, retriever):
    recipe = RECIPE
    entries = {}
    for name, rows in routes.items():
        for rank, row in enumerate(rows, 1):
            entry = entries.setdefault(row['id'], dict(row=row, ranks={}))
            entry['ranks'][name] = rank
    ordered = sorted(entries.values(), key=lambda e: (min(e['ranks'].values()), e['row']['id']))
    _, baseline_ids = retriever.render_selected(routes['recent3'][:recipe['baseline_limit']],
                                               max_chars=recipe['char_budget'])
    seen_source, seen_content, accepted, rejected = set(), set(), [], []
    for entry in ordered:
        row = entry['row']
        require(eligible(row, case), '召回返回了不合格的历史候选')
        source_key = digest([row['context_message_ids'], row['reply_message_ids']])
        content_key = digest(dict(context=[{'sender': m['sender'], 'text': m['text']}
                                            for m in row['context_messages']], reply=row['reply']))
        reason = None
        if source_key in seen_source or content_key in seen_content:
            reason = 'duplicate_source_or_content'
        block, ids = retriever.render_selected([row], max_chars=recipe['char_budget'])
        if ids != [row['id']] or len(block) > recipe['char_budget']:
            reason = 'complete_example_over_budget'
        if reason:
            rejected.append(dict(candidate_id=row['id'], reason=reason))
            continue
        seen_source.add(source_key)
        seen_content.add(content_key)
        accepted.append(dict(example=row, ranks=entry['ranks'], rendered_chars=len(block),
                             object_match=object_match(case, row)))
    for i, entry in enumerate(accepted):
        entry['union_rank'] = i + 1
        entry['rank_band'] = ('high', 'mid', 'low')[min(2, 3 * i // len(accepted))]
    by_id = {e['example']['id']: e for e in accepted}
    selected = []
    for cid in baseline_ids:
        if cid in by_id:
            selected.append({**by_id[cid], 'selection_reason': 'baseline'})
    selected_ids = {e['example']['id'] for e in selected}
    groups = defaultdict(list)
    seed = digest([recipe['seed'], case['case_id']])
    for e in accepted:
        if e['example']['id'] not in selected_ids:
            groups[e['rank_band'], e['object_match']].append(e)
    for group in groups.values():
        group.sort(key=lambda e: digest([seed, e['example']['id']]))
    keys = [(rank, obj) for rank in ('high', 'mid', 'low') for obj in ('same', 'different', 'unknown')]
    while len(selected) < recipe['candidate_limit'] and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < recipe['candidate_limit']:
                selected.append({**groups[key].pop(0), 'selection_reason': 'stratified'})
    return dict(target_id=case['case_id'], target=case, seed=seed,
                recalled_unique=len(entries), eligible_within_budget=len(accepted),
                baseline_ids=baseline_ids, rejected=rejected,
                recalled_routes={k: [r['id'] for r in v] for k, v in routes.items()},
                candidates=selected, split=None, label_status='not_labeled')


def summarize(records, audit, manifest):
    counts = Counter(len(r['candidates']) for r in records)
    candidates = [e for r in records for e in r['candidates']]
    unique = {e['example']['id']: e['example'] for e in candidates}
    return dict(schema=1, status='inventory_complete_labels_blocked' if not audit['summary']['source_eligible_targets']
                else 'inventory_complete_labels_pending', manifest_sha256=digest(manifest),
        targets=len(records), target_chat_types=dict(Counter(r['target']['chat_type'] for r in records)),
        target_familiarity=dict(Counter(r['target'].get('familiarity', 'unknown') for r in records)),
        target_context_messages=dict(Counter(len(r['target']['context']) for r in records)),
        candidate_count_per_target=dict(sorted(counts.items())), candidate_observations=len(candidates),
        targets_without_candidates=counts[0], targets_with_two_or_more_candidates=sum(v for k,v in counts.items() if k >= 2),
        unique_examples=len(unique), unique_example_context_messages=dict(Counter(len(e['context_messages']) for e in unique.values())),
        unique_example_reply_bubbles=dict(Counter(len(e['reply']) for e in unique.values())),
        observation_reply_bubbles=dict(Counter(len(e['example']['reply']) for e in candidates)),
        object_match=dict(Counter(e['object_match'] for e in candidates)),
        selection_reasons=dict(Counter(e['selection_reason'] for e in candidates)),
        exclusions=dict(Counter(e['reason'] for r in records for e in r['rejected'])),
        labels=dict(z0=0, z1=0, unlabeled=len(candidates), distribution_available=False,
                    mixed_targets=None, all_zero_targets=None, all_one_targets=None, directed_pairs=0),
        split=dict(status='not_frozen_pending_eligible_teacher', fit=0, validation=0),
        teacher=audit['summary'])


def prepare(data_dir, teacher_dir, generator_dir, output, progress=print):
    data_dir, teacher_dir, generator_dir, output = map(Path, (data_dir, teacher_dir, generator_dir, output))
    with file_lock(output / '.prepare.lock', blocking=False):
        source = load(data_dir)
        cases = source.role(RECIPE['role'])
        cases.sort(key=lambda c: (c['input_cutoff']['timestamp'], c['case_id']))
        audit, teacher_paths = teacher_overlap(source, cases, teacher_dir)
        cfg = read_json(generator_dir / 'config.json')
        require(cfg.get('max_shots_per_case') == 3 and cfg.get('shots_char_budget') == 2500 and
                not cfg.get('retriever', {}).get('selection'), '基线选择器不是已确认的前三条/2500预算')
        paths = [*source.files, data_dir / 'gen_optimization.jsonl', teacher_dir / 'learning.json',
                 teacher_dir / 'config.json', *teacher_paths,
                 *[p for p in generator_dir.rglob('*') if p.is_file()],
                 Path(__file__), *IMPLEMENTATION_FILES,
                 Path(__file__).parents[2] / 'history.py', Path(__file__).parents[2] / 'history_sources.py']
        hashes = fingerprint(paths)
        manifest = dict(schema=1, recipe=RECIPE, inputs=hashes, data=data_dir.name,
                        generator=generator_dir.name, teacher=teacher_dir.name,
                        target_ids=[c['case_id'] for c in cases], usage='provisional_candidate_inventory',
                        label_definition='z=1 iff frozen judge fails to identify single-shot generation')
        manifest = _resolve_manifest(output, manifest)
        write_once_json(output / 'manifest.json', manifest)
        write_once_json(output / 'teacher_sources.json', audit)
        progress(json.dumps({'teacher_sources': audit['summary']}, ensure_ascii=False))
        existing_report = output / 'report.json'
        if existing_report.exists():
            # Completed inventories have a frozen checksum list, allowing cheap
            # resume without rerunning retrieval or scanning the whole pool.
            files = read_json(output / 'inventory.json')['files']
            require(all(sha256_file(output / name) == sha for name, sha in files.items()), '已存候选清单变化')
            report = read_json(existing_report)
            require(report['manifest_sha256'] == digest(manifest), '汇总与冻结配方不符')
            return report
        retriever = open_session(data_dir / 'fewshot_pool.jsonl')
        require(retriever.is_approved(), '历史召回库未批准')
        records = []
        start = time.monotonic()
        for i, case in enumerate(cases, 1):
            path = output / 'targets' / f"{case['case_id']}.json"
            if path.exists():
                record = read_json(path)
                payload = {k: v for k, v in record.items() if k != 'payload_sha256'}
                require(record.get('payload_sha256') == digest(payload), '断点候选内容变化')
                require(record['manifest_sha256'] == digest(manifest) and record['target'] == case,
                        '断点配方或目标变化')
                for entry in record['candidates']:
                    source.validate(entry['example'], example=True)
                    require(eligible(entry['example'], case), '断点候选已不合格')
            else:
                messages = case['context']
                queries = dict(recent3='\n'.join(m['text'] for m in messages[-3:]), latest1=messages[-1]['text'])
                routes, query_cache = {}, {}
                for name, query in queries.items():
                    if query not in query_cache:
                        query_cache[query] = retriever.retrieve(query=query, chat_name=case['chat_name'],
                            is_group=case['chat_type'] == 'group', limit=RECIPE['route_limit'],
                            exclude_ids={case['case_id']}, history_case=case,
                            current_context_messages=[dict(sender=m['sender'], text=m['text']) for m in messages])
                    routes[name] = query_cache[query]
                record = select_candidates(case, routes, retriever)
                record['manifest_sha256'] = digest(manifest)
                record['teacher_sources'] = audit['targets'][i-1]
                record['payload_sha256'] = digest(record)
                source.check()
                write_once_json(path, record)
            records.append(record)
            if i == 1 or i % 25 == 0 or i == len(cases):
                state = dict(completed_targets=i, total_targets=len(cases),
                             candidate_observations=sum(len(r['candidates']) for r in records),
                             elapsed_seconds=round(time.monotonic()-start, 1), labels_completed=0)
                write_json(output / 'progress.json', state)
                progress(json.dumps(state, ensure_ascii=False))
        source.check()
        require(fingerprint(paths) == hashes, '准备期间冻结输入变化')
        report = summarize(records, audit, manifest)
        files = {str(p.relative_to(output)): sha256_file(p) for p in sorted((output / 'targets').glob('*.json'))}
        write_once_json(output / 'inventory.json', dict(files=files))
        write_once_json(existing_report, report)
        text = ['# Few-shot 候选样本准备报告', '',
                '候选已准备；标签与训练内切分尚未生成。当前标签器资格见下表。', '',
                '```json', json.dumps(report, ensure_ascii=False, indent=2), '```', '']
        atomic_write(output / 'REPORT.md', '\n'.join(text))
        return report
