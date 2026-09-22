"""Resumable single-example supervision from a frozen, nonoverlapping teacher.

The teacher supplies offline proxy labels, not an independent evaluation of
itself. Its later training cutoff is allowed; exact case/answer overlap is not.
Generation retains the normal historical input and material restrictions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from threading import local
import json
import os
import time

from .. import tracing
from ..config import ConfigError, ROOT, load_settings, settings_path, sha256_file
from ..iteration import versions
from ..iteration.learning_guard import RunSeal
from ..iteration.parallel import completed_map
from ..iteration.storage import atomic_write, file_lock, read_json, write_json, write_once_json
from ..judge.judge import build_judge
from ..llm import build_clients
from .few_shot import PersonaFewShotRetriever
from .generator import ReplyGenerator
from .history import eligible
from .history_sources import digest, load, require
from .ranker_samples import fingerprint

POLICY = dict(schema=1, teacher='offline_nonoverlapping_supervision', single_example=True,
              validation_fraction=.2, draw=0, label='int(not identified_ai)',
              teacher_overlap_fields=['same_teacher_training_case', 'answer_in_teacher_training',
                                      'answer_in_teacher_references'])


def select_targets(records):
    """Filter before chronological splitting; never split candidates of a target."""
    kept, exclusions = [], []
    for row in records:
        reasons = [key for key in POLICY['teacher_overlap_fields'] if row['teacher_sources'][key]]
        if reasons:
            exclusions.append(dict(target_id=row['target_id'], reasons=reasons))
        else:
            kept.append(row)
    kept.sort(key=lambda r: (r['target']['input_cutoff']['timestamp'], r['target_id']))
    require(len(kept) >= 2, '排除重叠后不足两个上文，不能切分')
    cut = min(len(kept)-1, max(1, len(kept)*4//5))
    boundary = kept[cut]['target']['input_cutoff']['timestamp']
    validation = [r for r in kept if r['target']['input_cutoff']['timestamp'] >= boundary]
    validation_answers = {mid for r in validation for mid in r['target']['reply_message_ids']}
    selected, purged = [], []
    for row in kept:
        split = 'validation' if row['target']['input_cutoff']['timestamp'] >= boundary else 'fit'
        materials = [row['target'], *[e['example'] for e in row['candidates']]]
        reasons = []
        if split == 'fit':
            if any(m['source_span']['end_timestamp'] >= boundary for m in materials):
                reasons.append('material_crosses_validation_boundary')
            if any(validation_answers.intersection(m['context_message_ids'] + m['reply_message_ids'])
                   for m in materials):
                reasons.append('validation_answer_in_fit_material')
        if reasons:
            purged.append(dict(target_id=row['target_id'], reasons=reasons))
        else:
            selected.append({**row, 'split': split})
    require({r['split'] for r in selected} == {'fit', 'validation'}, '时间隔离后训练或验证上文为空')
    return selected, dict(source_targets=len(records), overlap_retained=len(kept),
        excluded=exclusions, purged=purged, validation_boundary=boundary)


class SingleExampleGenerator(ReplyGenerator):
    """Inject one frozen, complete example through the standard guarded prompt."""
    def __init__(self, settings, config, clients, prompt_root, pool_path):
        super().__init__(settings, {**config, 'retriever': {'enabled': False}}, clients,
                         prompt_root=prompt_root, pool_path=pool_path)
        self._example = None

    def _style_block(self, case):
        require(self._example is not None and self._history_sources is not None, '缺少单示例或历史来源')
        self._history_sources.validate(self._example, example=True)
        require(eligible(self._example, case), '单示例不早于目标或包含目标答案')
        block, ids = PersonaFewShotRetriever.render([self._example], max_chars=self._budget)
        require(ids == [self._example['id']] and len(block) <= self._budget, '完整单示例超过篇幅预算')
        tracing.note('retrieval', dict(enabled=True, selected_ids=ids, max_shots=1,
                                     char_budget=self._budget, rendered=block, policy='frozen_single_example'))
        return block

    def generate_one(self, case, example):
        self._example = example
        try:
            return self.generate(case)
        finally:
            self._example = None


def _saved(path, binding):
    if not path.exists():
        return None
    value = read_json(path)
    payload = {k: v for k, v in value.items() if k != 'payload_sha256'}
    require(value.get('payload_sha256') == digest(payload) and value.get('binding') == binding,
            '已存观察损坏或条件变化，禁止复用')
    if 'generation' in value:
        require(ReplyGenerator._validate(value['generation'], True) is not None, '已存生成格式无效')
    if value.get('status') == 'complete':
        require(type(value.get('identified_ai')) is bool and
                value.get('z') == int(not value['identified_ai']), '已存标签方向或类型无效')
    return value


def _save(path, value):
    value = {k: v for k, v in value.items() if k != 'payload_sha256'}
    write_json(path, {**value, 'payload_sha256': digest(value)})


def label_one(output, manifest, row, entry, generator, judge, check):
    """Persist generation before judging; failed judgments never regenerate."""
    check()
    identity = digest([row['target_id'], entry['example']['id']])
    binding = digest([digest(manifest), row['payload_sha256'], entry])
    path = output / 'observations' / (identity + '.json')
    saved = _saved(path, binding)
    if saved and saved['status'] == 'complete':
        return identity, saved
    case = row['target']
    trace = tracing.CaseTrace(output, case, dict(kind='ranker_labels', dataset='ranker_labels',
        data_ref=manifest['data'], candidate_ref=manifest['generator'], judge_ref=manifest['teacher'],
        cache={'sample_epoch': digest(manifest)}))
    value = dict(binding=binding, target_id=row['target_id'], example_id=entry['example']['id'],
                 split=row['split'], status='pending', trace_ref=trace.ref,
                 attempts=(saved or {}).get('attempts', 0)+1)
    if saved and 'generation' in saved:
        value.update(generation=saved['generation'], generation_trace_ref=saved['generation_trace_ref'])
    stage = 'generation'
    try:
        if 'generation' not in value:
            with trace.operation('generation', identity, 0, {'example_id': entry['example']['id']}) as op:
                value['generation'] = generator.generate_one(case, entry['example'])
                require(ReplyGenerator._validate(value['generation'], True) is not None, '生成断点格式无效')
                op['output'] = value['generation']
            value.update(status='generated', generation_trace_ref=trace.ref)
            check()
            _save(path, value)
        stage = 'judge'
        with trace.operation('judge', identity, 0, {'candidate_replies': value['generation']['replies']}) as op:
            identified = judge.is_ai(case, value['generation']['replies'])
            require(type(identified) is bool, 'Judge 必须返回布尔识别结果')
            value.update(identified_ai=identified, z=int(not identified),
                         verdict=getattr(judge, 'last_verdict', None), status='complete')
            op['output'] = dict(identified_ai=identified, z=value['z'], verdict=value['verdict'])
        check()
        _save(path, value)
        trace.finish('ok')
    except ConfigError:
        trace.finish('blocked')
        raise
    except Exception as exc:
        check()
        value.update(status='failed', failed_stage=stage,
                     error=dict(type=type(exc).__name__, message=str(exc)))
        _save(path, value)
        trace.finish('failed')
    return identity, value


def label_summary(rows, states):
    counts = Counter(v['z'] for v in states.values() if v['status'] == 'complete')
    total = sum(len(r['candidates']) for r in rows)
    groups = defaultdict(list)
    for value in states.values():
        if value['status'] == 'complete':
            groups[value['target_id']].append(value)
    pairs, classes = [], Counter()
    for row in rows:
        values = groups[row['target_id']]
        wins = [v['example_id'] for v in values if v['z'] == 1]
        losses = [v['example_id'] for v in values if v['z'] == 0]
        category = ('no_candidates' if not row['candidates'] else
                    'incomplete' if len(values) != len(row['candidates']) else
                    'mixed' if wins and losses else 'all_one' if wins else 'all_zero')
        classes[category] += 1
        # Publish pairs only from completed targets, so retries cannot skew a group.
        if category == 'mixed':
            pairs.extend(dict(target_id=row['target_id'], split=row['split'], winner=w, loser=l)
                         for w in wins for l in losses)
    return dict(targets=len(rows), target_chat_types=dict(Counter(r['target']['chat_type'] for r in rows)),
        observations=total, generated=sum('generation' in v for v in states.values()),
        labeled=sum(counts.values()), z0=counts[0], z1=counts[1], unlabeled=total-sum(counts.values()),
        failed=sum(v['status'] == 'failed' for v in states.values()),
        failures=dict(Counter(v.get('failed_stage') for v in states.values() if v['status'] == 'failed')),
        candidate_count_per_target=dict(Counter(len(r['candidates']) for r in rows)),
        unique_examples=len({e['example']['id'] for r in rows for e in r['candidates']}),
        target_outcomes=dict(classes), directed_pairs=len(pairs),
        splits={split: dict(targets=sum(r['split'] == split for r in rows),
            observations=sum(len(r['candidates']) for r in rows if r['split'] == split),
            labels=dict(Counter(str(v['z']) for v in states.values()
                if v['status'] == 'complete' and v['split'] == split))) for split in ('fit', 'validation')}), pairs


class LabelBatch:
    def __init__(self, inventory, data, teacher, generator, instance):
        self.inventory, self.data, self.teacher, self.generator = map(Path, (inventory, data, teacher, generator))
        self.output = self.inventory / 'labeling'
        versions.switch_instance(instance)
        self.settings = load_settings()
        original = read_json(self.inventory / 'manifest.json')
        require((original['data'], original['teacher'], original['generator']) ==
                (self.data.name, self.teacher.name, self.generator.name), '目录版本与冻结候选不符')
        files = [self.inventory / n for n in ('manifest.json', 'inventory.json', 'teacher_sources.json')]
        runtime = [* (ROOT / 'src/generator').glob('*.py'), * (ROOT / 'src/judge').glob('*.py'),
            *[ROOT / 'src' / p for p in ('config.py', 'cache.py', 'tracing.py', 'llm.py',
                'iteration/learning_guard.py', 'iteration/storage.py', 'iteration/parallel.py',
                'iteration/control.py', 'iteration/quota.py')]]
        runtime = [p for p in runtime if p.exists()]
        assets = [p for directory in (self.teacher, self.generator) for p in directory.rglob('*') if p.is_file()]
        watched = [*map(Path, original['inputs']), *files, *runtime, *assets, settings_path()]
        self.seal = RunSeal(files=watched)
        require(fingerprint(original['inputs']) == original['inputs'], '冻结候选来源已变化')
        audit = {r['target_id']: r for r in read_json(self.inventory / 'teacher_sources.json')['targets']}
        records = []
        for name, sha in read_json(self.inventory / 'inventory.json')['files'].items():
            path = self.inventory / name
            require(path.resolve().is_relative_to(self.inventory.resolve()), '候选路径越界')
            require(sha256_file(path) == sha, '冻结候选内容变化')
            row = read_json(path)
            require(row['manifest_sha256'] == digest(original) and row['teacher_sources'] == audit[row['target_id']]
                    and row['payload_sha256'] == digest({k:v for k,v in row.items() if k != 'payload_sha256'}),
                    '候选来源或内容不符')
            records.append(row)
        require(set(original['target_ids']) == {r['target_id'] for r in records} and
                len(records) == len(original['target_ids']), '候选目标清单不完整')
        self.rows, selection = select_targets(records)
        source = load(self.data)
        for row in self.rows:
            source.validate(row['target'])
            for entry in row['candidates']:
                source.validate(entry['example'], example=True)
                require(eligible(entry['example'], row['target']), '冻结候选不符合目标历史边界')
        self.gen_cfg = read_json(self.generator / 'config.json')
        self.judge_cfg = read_json(self.teacher / 'config.json')
        clients = build_clients(self.settings, self.gen_cfg['llm'])
        judge = build_judge(self.settings, dict(config=self.judge_cfg, dir=self.teacher))
        self.manifest = dict(policy=POLICY, data=self.data.name, teacher=self.teacher.name,
            generator=self.generator.name, original_manifest_sha256=digest(original),
            inputs=fingerprint(watched), selection=selection,
            targets=[dict(target_id=r['target_id'], split=r['split'], payload_sha256=r['payload_sha256']) for r in self.rows],
            clients={key: client.cache_identity() for key,client in clients.items()},
            teacher_client=judge.feature_client.cache_identity(),
            label_usage='offline_ranker_supervision_not_independent_judge_evaluation')
        self.seal.check()
        write_once_json(self.output / 'manifest.json', self.manifest)
        self.local = local()
        self.states = {}
        self.items = [(r, e) for r in self.rows for e in r['candidates']]
        for row, entry in self.items:
            identity = digest([row['target_id'], entry['example']['id']])
            saved = _saved(self.output / 'observations' / (identity + '.json'),
                           digest([digest(self.manifest), row['payload_sha256'], entry]))
            if saved:
                self.states[identity] = saved

    def _one(self, item):
        self.seal.check()
        if not hasattr(self.local, 'generator'):
            self.local.generator = SingleExampleGenerator(self.settings, self.gen_cfg,
                build_clients(self.settings, self.gen_cfg['llm']), self.generator, self.data / 'fewshot_pool.jsonl')
            self.local.judge = build_judge(self.settings, dict(config=self.judge_cfg, dir=self.teacher))
            self.local.generator.source_check = self.local.judge.source_check = self.seal.check
        return label_one(self.output, self.manifest, *item, self.local.generator, self.local.judge, self.seal.check)

    def report(self, status=None):
        result, pairs = label_summary(self.rows, self.states)
        result.update(status=status or ('complete' if result['unlabeled'] == 0 else
                                       'needs_retry' if result['failed'] else 'pending'),
            teacher=self.teacher.name, generator=self.generator.name, updated_at=time.time(),
            manifest_sha256=digest(self.manifest), supervision_only=True,
            overlap_excluded=len(self.manifest['selection']['excluded']),
            boundary_purged=len(self.manifest['selection']['purged']))
        self.seal.check()
        write_json(self.output / 'report.json', result)
        write_json(self.output / 'pairs.json', dict(manifest_sha256=digest(self.manifest), pairs=pairs))
        atomic_write(self.output / 'REPORT.md', '# 单示例标签报告\n\n这是排序训练的代理标签，不是 Judge 独立评测。\n\n```json\n' +
                     json.dumps(result, ensure_ascii=False, indent=2) + '\n```\n')
        return result

    def run(self, workers=16, passes=2, progress=print):
        require(1 <= workers <= 16 and passes >= 1, '并发须为 1–16，重试轮数须为正')
        started = time.monotonic()
        initial = sum(v['status'] == 'complete' for v in self.states.values())
        write_json(self.output / 'run.json', dict(pid=os.getpid(), started_at=time.time(), workers=workers, passes=passes))
        self.report('running')
        for attempt in range(passes):
            pending = [item for item in self.items if self.states.get(
                digest([item[0]['target_id'], item[1]['example']['id']]), {}).get('status') != 'complete']
            for i, (identity, value) in enumerate(completed_map(self._one, pending, workers), 1):
                self.states[identity] = value
                if i == 1 or i % 16 == 0 or i == len(pending):
                    result = self.report('running')
                    rate = 60*(result['labeled']-initial)/max(1, time.monotonic()-started)
                    progress(json.dumps(dict(pass_number=attempt+1, labeled=result['labeled'], total=result['observations'],
                        failed=result['failed'], labels_per_minute=round(rate, 2)), ensure_ascii=False), flush=True)
        return self.report()


def execute(inventory, data, teacher, generator, instance, *, run=False, workers=16, passes=2):
    with file_lock(Path(inventory) / 'labeling' / '.run.lock', blocking=False):
        batch = LabelBatch(inventory, data, teacher, generator, instance)
        return batch.run(workers, passes) if run else batch.report()
