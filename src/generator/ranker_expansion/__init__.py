"""Expand single-example supervision without changing a running batch's inputs."""
from collections import Counter
from pathlib import Path
import json
import time

from ...config import sha256_file
from ...iteration.datasets import case_path
from ...iteration.storage import file_lock, read_json, write_json, write_once_json, atomic_write
from ..few_shot import PersonaFewShotRetriever
from ..history import eligible
from ..history_sources import digest, load, require
from ..ranker_labels import POLICY, select_targets
from ..ranker_samples import RECIPE, fingerprint, select_candidates, summarize, teacher_overlap


def read_inventory(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    require(fingerprint(manifest['inputs']) == manifest['inputs'], '旧候选输入变化')
    audit = {r['target_id']: r for r in read_json(directory / 'teacher_sources.json')['targets']}
    rows = []
    for name, sha in read_json(directory / 'inventory.json')['files'].items():
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()), '候选路径越界')
        require(sha256_file(path) == sha, '旧候选文件变化')
        row = read_json(path)
        require(row['manifest_sha256'] == digest(manifest) and
                row['payload_sha256'] == digest({k: v for k, v in row.items() if k != 'payload_sha256'}) and
                row['teacher_sources'] == audit[row['target_id']], '旧候选来源不符')
        rows.append(row)
    require(len(rows) == len(manifest['target_ids']) and
            {r['target_id'] for r in rows} == set(manifest['target_ids']), '旧候选清单不完整')
    return manifest, rows


def isolated_rows(rows):
    """Reach a stable global time split after removing boundary-crossing rows."""
    purged = []
    while True:
        selected, selection = select_targets(rows)
        purged.extend(selection['purged'])
        if len(selected) == len(rows):
            return selected, {**selection, 'purged_during_expansion': purged}
        rows = selected


def cap_rows(rows, count):
    """Keep target priority and cap distinct observations, never synthesize rows."""
    result, remaining = [], count
    for row in rows:
        if remaining <= 0:
            break
        entries = row['candidates'][:remaining]
        if entries:
            result.append({**row, 'candidates': entries})
            remaining -= len(entries)
    return result


def prepare(data_dir, teacher_dir, generator_dir, output, *, observations, reuse_inventory, progress=print):
    data_dir, teacher_dir, generator_dir, output, reuse_inventory = map(
        Path, (data_dir, teacher_dir, generator_dir, output, reuse_inventory))
    require(type(observations) is int and observations > 0, '样本目标必须为正整数')
    require(output.resolve() != reuse_inventory.resolve(), '扩充必须写入新批次，保留旧断点')
    with file_lock(output / '.prepare.lock', blocking=False):
        source = load(data_dir)
        old_manifest, old_rows = read_inventory(reuse_inventory)
        require((old_manifest['data'], old_manifest['teacher'], old_manifest['generator']) ==
                (data_dir.name, teacher_dir.name, generator_dir.name), '扩充不能改变数据或模型条件')
        require(old_manifest['recipe'] == RECIPE, '候选选择配方变化')
        cfg = read_json(generator_dir / 'config.json')
        require(cfg.get('max_shots_per_case') == 3 and cfg.get('shots_char_budget') == 2500 and
                not cfg.get('retriever', {}).get('selection'), '生成器选择条件变化')
        roles = ('gen_learning', 'gen_optimization')
        union, counts = {}, {}
        for role in roles:
            cases = source.role(role)
            counts[role] = len(cases)
            for case in cases:
                require(case['case_id'] not in union or union[case['case_id']] == case, '角色间同 ID 内容冲突')
                union[case['case_id']] = case
        audit, teacher_paths = teacher_overlap(source, list(union.values()), teacher_dir)
        by_audit = {r['target_id']: r for r in audit['targets']}
        learning = read_json(generator_dir / 'learning.json')
        require(learning.get('verified') is True and type(learning.get('information_end')) in (int, float),
                '生成器缺少学习截止来源')
        exclusions, usable = [], {}
        for cid, case in union.items():
            reasons = [k for k in POLICY['teacher_overlap_fields'] if by_audit[cid][k]]
            if learning['information_end'] >= case['input_cutoff']['timestamp']:
                reasons.append('generator_material_not_before_target')
            if reasons:
                exclusions.append(dict(target_id=cid, reasons=reasons))
            else:
                usable[cid] = case
        old = {r['target_id']: r for r in old_rows if r['target_id'] in usable and r['candidates']}
        for cid, row in old.items():
            require(row['target'] == usable[cid] and row['teacher_sources'] == by_audit[cid],
                    '旧目标内容或 Judge 隔离条件变化')
        priority = sorted(old) + sorted(set(usable) - set(old), key=lambda cid: digest(['ranker-expansion-v1', cid]))
        paths = [*source.files, *[case_path(data_dir, role) for role in roles], *teacher_paths,
                 *[p for d in (teacher_dir, generator_dir) for p in d.rglob('*') if p.is_file()],
                 *map(Path, old_manifest['inputs']),
                 *[reuse_inventory / name for name in ('manifest.json', 'inventory.json', 'teacher_sources.json')],
                 *Path(__file__).parent.glob('*.py')]
        hashes = fingerprint(paths)
        plan = dict(schema=1, observations=observations, recipe=RECIPE, policy=POLICY, roles=roles,
                    inputs=hashes, target_ids=priority, reuse_inventory=str(reuse_inventory.resolve()))
        write_once_json(output / 'expansion_plan.json', plan)
        plan_sha = digest(plan)
        capacity = dict(requested_observations=observations, role_counts=counts, union_targets=len(union),
            eligible_targets=len(usable), excluded_targets=len(exclusions),
            exclusions=dict(Counter(k for r in exclusions for k in r['reasons'])),
            eligible_chat_types=dict(Counter(c['chat_type'] for c in usable.values())),
            observation_upper_bound=len(usable)*RECIPE['candidate_limit'],
            reusable_targets=len(old), reusable_observations=sum(len(r['candidates']) for r in old.values()),
            note='upper bound only; candidate coverage and chronological purge still apply')
        write_once_json(output / 'capacity.json', capacity)
        progress(json.dumps({'capacity': capacity}, ensure_ascii=False), flush=True)
        if (output / 'report.json').exists():
            read_inventory(output)
            return read_json(output / 'report.json')
        retriever = PersonaFewShotRetriever(data_dir / 'fewshot_pool.jsonl')
        require(retriever.is_approved(), '历史召回库未批准')
        rows, total, purged = [], 0, []
        started = time.monotonic()
        for i, cid in enumerate(priority, 1):
            case = usable[cid]
            path = output / 'preparation' / f'{cid}.json'
            if path.exists():
                record = read_json(path)
                require(record['plan_sha256'] == plan_sha and record['target'] == case and
                        record['payload_sha256'] == digest({k:v for k,v in record.items() if k != 'payload_sha256'}),
                        '扩充断点来源或内容变化')
            else:
                if cid in old:
                    record = {k:v for k,v in old[cid].items() if k not in ('payload_sha256', 'manifest_sha256')}
                else:
                    messages = case['context']
                    queries = dict(recent3='\n'.join(m['text'] for m in messages[-3:]), latest1=messages[-1]['text'])
                    routes, cache = {}, {}
                    for name, query in queries.items():
                        if query not in cache:
                            cache[query] = retriever.retrieve(query=query, chat_name=case['chat_name'],
                                is_group=case['chat_type'] == 'group', limit=RECIPE['route_limit'],
                                exclude_ids={cid}, history_case=case,
                                current_context_messages=[dict(sender=m['sender'], text=m['text']) for m in messages])
                        routes[name] = cache[query]
                    record = select_candidates(case, routes, retriever)
                record.update(plan_sha256=plan_sha, teacher_sources=by_audit[cid])
                record['payload_sha256'] = digest(record)
                write_once_json(path, record)
            for entry in record['candidates']:
                source.validate(entry['example'], example=True)
                require(eligible(entry['example'], case), '扩充候选超出目标历史边界')
            if record['candidates']:
                rows.append(record)
                total += len(record['candidates'])
            if total >= observations:
                capped = cap_rows(rows, observations)
                selected, selection = isolated_rows(capped)
                purged.extend(selection['purged_during_expansion'])
                total = sum(len(r['candidates']) for r in selected)
                if total == observations:
                    rows = selected
                    break
                retained = {r['target_id'] for r in selected}
                rows = [r for r in capped if r['target_id'] in retained]
            if i == 1 or i % 100 == 0 or i == len(priority):
                elapsed = time.monotonic()-started
                state = dict(checked_targets=i, eligible_targets=len(priority), selected_targets=len(rows),
                    candidate_observations=total, requested_observations=observations,
                    elapsed_seconds=round(elapsed, 1), targets_per_minute=round(60*i/max(1, elapsed), 2))
                write_json(output / 'progress.json', state)
                progress(json.dumps(state, ensure_ascii=False), flush=True)
        rows, selection = isolated_rows(rows)
        purged.extend(selection['purged_during_expansion'])
        total = sum(len(r['candidates']) for r in rows)
        source.check()
        require(fingerprint(paths) == hashes, '扩充期间冻结输入变化')
        manifest = dict(schema=1, recipe=RECIPE, inputs={**hashes,
            str((output / 'expansion_plan.json').resolve()): sha256_file(output / 'expansion_plan.json')},
            data=data_dir.name, generator=generator_dir.name, teacher=teacher_dir.name,
            target_ids=[r['target_id'] for r in rows], usage='offline_nonoverlapping_single_example_supervision',
            label_definition='z=1 iff frozen judge fails to identify single-shot generation')
        write_once_json(output / 'manifest.json', manifest)
        manifest_sha = digest(manifest)
        selected_audit = dict(summary={**audit['summary'], 'label_authorized': True,
            'scope': POLICY['teacher']}, targets=[by_audit[r['target_id']] for r in rows])
        write_once_json(output / 'teacher_sources.json', selected_audit)
        files = {}
        for row in rows:
            record = {k:v for k,v in row.items() if k not in ('payload_sha256', 'plan_sha256')}
            record.update(manifest_sha256=manifest_sha, split=None)
            record['payload_sha256'] = digest(record)
            path = output / 'targets' / f"{row['target_id']}.json"
            write_once_json(path, record)
            files[str(path.relative_to(output))] = sha256_file(path)
        write_once_json(output / 'inventory.json', dict(files=files))
        report = summarize(rows, selected_audit, manifest)
        report.update(status='inventory_complete_labels_pending' if total == observations else 'inventory_shortfall',
            requested_observations=observations, shortfall=max(0, observations-total), capacity=capacity,
            boundary_purged=purged, reused_candidate_observations=sum(len(r['candidates']) for r in rows if r['target_id'] in old),
            split=dict(status='global_chronological', validation_boundary=selection['validation_boundary'],
                **{s: dict(targets=sum(r['split'] == s for r in rows),
                    observations=sum(len(r['candidates']) for r in rows if r['split'] == s)) for s in ('fit', 'validation')}))
        write_once_json(output / 'report.json', report)
        write_json(output / 'progress.json', dict(status=report['status'], candidate_observations=total,
                                                  selected_targets=len(rows), requested_observations=observations))
        atomic_write(output / 'REPORT.md', '# 扩充样本清单\n\n```json\n'+json.dumps(report, ensure_ascii=False, indent=2)+'\n```\n')
        return read_json(output / 'report.json')
