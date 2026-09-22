"""Repeat originally positive single-example labels without altering supervision.

Independent draw namespaces bypass earlier draws, while retries reuse successful
generation/label checkpoints within a draw. Positive-only survival is conditional
on the initial positive; it is not a full-dataset label-noise estimate.
"""
from collections import Counter
from pathlib import Path
from threading import local
from types import SimpleNamespace
import json
import os
import time

from ...config import ROOT, load_settings, settings_path, sha256_file
from ...iteration import versions
from ...iteration.learning_guard import RunSeal
from ...iteration.parallel import completed_map
from ...iteration.storage import atomic_write, file_lock, read_json, write_json, write_once_json
from ...judge.judge import build_judge
from ...llm import build_clients
from ..history_sources import digest, require
from ..ranker_labels import SingleExampleGenerator, _saved, label_one
from ..ranker_samples import fingerprint
from .dataset import load_dataset


def summarize(cohort, states, modes, repeats):
    result = {}
    for mode in modes:
        complete, histogram, rows, split_histograms = 0, Counter(), [], {}
        for item in cohort:
            draws = [states.get((mode, draw, item['identity'])) for draw in range(1, repeats+1)]
            labels = [v['z'] if v and v['status'] == 'complete' else None for v in draws]
            if all(z is not None for z in labels):
                complete += 1
                histogram[str(sum(labels))] += 1
                split_histograms.setdefault(item['row']['split'], Counter())[str(sum(labels))] += 1
            rows.append(dict(target_id=item['row']['target_id'],
                example_id=item['entry']['example']['id'], split=item['row']['split'],
                original_z=1, new_labels=labels,
                original_payload_sha256=item['original']['payload_sha256']))
        values = [v for (m, _, _), v in states.items() if m == mode]
        finished = [v for v in values if v['status'] == 'complete']
        result[mode] = dict(expected=len(cohort)*repeats, labeled=len(finished),
            generated=sum('generation' in v for v in values),
            failed=sum(v['status'] == 'failed' for v in values),
            failures=dict(Counter(v.get('failed_stage') for v in values if v['status'] == 'failed')),
            new_positives=sum(v['z'] for v in finished),
            new_positive_rate=sum(v['z'] for v in finished)/len(finished) if finished else None,
            fully_retested_samples=complete,
            new_positive_count_histogram={str(k): histogram[str(k)] for k in range(repeats+1)},
            split_histograms={s: {str(k): h[str(k)] for k in range(repeats+1)}
                              for s, h in split_histograms.items()},
            all_three_positive=histogram[str(repeats)] if repeats == 2 else None,
            rows=rows)
    return result


class StabilityBatch:
    def __init__(self, inventory, data, teacher, generator, instance, output, mode, repeats):
        require(mode in ('regenerate', 'judge', 'both') and repeats >= 1, 'Invalid repeat policy')
        self.inventory, self.data, self.teacher, self.generator, self.output = map(
            lambda p: Path(p).resolve(), (inventory, data, teacher, generator, output))
        require(self.output != self.inventory and not self.output.is_relative_to(self.inventory/'labeling'),
                'Stability output must not overwrite original labeling')
        self.modes = ['regenerate', 'judge'] if mode == 'both' else [mode]
        self.repeats, self.local = repeats, local()
        versions.switch_instance(instance)
        original_path = self.inventory/'labeling/manifest.json'
        original = read_json(original_path)
        require((original['data'], original['teacher'], original['generator']) ==
                (self.data.name, self.teacher.name, self.generator.name), 'Frozen versions differ')
        changed = [name for name, sha in original['inputs'].items()
                   if not Path(name).is_file() or sha256_file(Path(name)) != sha]
        require(not changed, f'Original labeling conditions changed: {changed}')
        require(str(settings_path().resolve()) in original['inputs'], 'Use original labeling settings')
        self.settings = load_settings()
        groups, observations, summary, source_binding = load_dataset(self.inventory)
        self.cohort = []
        watched = [*map(Path, original['inputs']), original_path,
            Path(__file__), Path(__file__).with_name('dataset.py'), ROOT/'scripts/label_fewshot_ranker.py']
        for group in groups:
            row_path = self.inventory/'targets'/f'{group["target_id"]}.json'
            row = dict(read_json(row_path), split=group['split'])
            watched.append(row_path)
            for index, entry in zip(group['indices'], group['candidates']):
                identity = digest([group['target_id'], entry['example']['id']])
                path = self.inventory/'labeling/observations'/f'{identity}.json'
                if observations[index]['z'] == 1:
                    watched.append(path)
                    saved = _saved(path, digest([digest(original), row['payload_sha256'], entry]))
                    self.cohort.append(dict(identity=identity, row=row, entry=entry, original=saved))
        require(self.cohort, 'No positive labels to repeat')
        self.gen_cfg, self.judge_cfg = (read_json(p/'config.json') for p in (self.generator, self.teacher))
        clients = build_clients(self.settings, self.gen_cfg['llm'])
        judge = build_judge(self.settings, dict(config=self.judge_cfg, dir=self.teacher))
        require({key: client.cache_identity() for key, client in clients.items()} == original['clients'] and
                judge.feature_client.cache_identity() == original['teacher_client'], 'Model client conditions changed')
        self.seal = RunSeal(files=watched)
        self.manifest = dict(schema=1, kind='positive_label_stability', data=self.data.name,
            teacher=self.teacher.name, generator=self.generator.name, source=source_binding,
            source_summary=summary, inventory=str(self.inventory), modes=self.modes,
            additional_draws=repeats, selection='all originally positive observations; no new label filtering',
            positive_cohort=[dict(identity=i['identity'], original_payload_sha256=i['original']['payload_sha256'])
                             for i in self.cohort],
            inputs=fingerprint(watched), label='int(not identified_ai)',
            cache='separate namespace per mode/draw; retry reuses same draw; original labels unchanged')
        write_once_json(self.output/'manifest.json', self.manifest)
        self.draws, self.states, self.items = {}, {}, []
        for m in self.modes:
            for draw in range(1, repeats+1):
                directory = self.output/m/f'draw-{draw}'
                manifest = dict(data=self.data.name, teacher=self.teacher.name,
                    generator=self.generator.name, stability_manifest_sha256=digest(self.manifest),
                    mode=m, draw=draw)
                write_once_json(directory/'manifest.json', manifest)
                self.draws[m, draw] = (directory, manifest)
                for item in self.cohort:
                    key = (m, draw, item['identity'])
                    self.items.append((key, item))
                    saved = _saved(directory/'observations'/f'{item["identity"]}.json',
                        digest([digest(manifest), item['row']['payload_sha256'], item['entry']]))
                    if saved:
                        self.states[key] = saved
        self.seal.check()

    def _one(self, task):
        key, item = task
        mode, draw, identity = key
        if not hasattr(self.local, 'judge'):
            self.local.judge = build_judge(self.settings, dict(config=self.judge_cfg, dir=self.teacher))
            self.local.generator = SingleExampleGenerator(self.settings, self.gen_cfg,
                build_clients(self.settings, self.gen_cfg['llm']), self.generator, self.data/'fewshot_pool.jsonl')
            self.local.judge.source_check = self.local.generator.source_check = self.seal.check
        generator = self.local.generator if mode == 'regenerate' else SimpleNamespace(
            generate_one=lambda *_: item['original']['generation'])
        directory, manifest = self.draws[mode, draw]
        _, value = label_one(directory, manifest, item['row'], item['entry'],
                             generator, self.local.judge, self.seal.check)
        return key, value

    def report(self, status=None, started=None, initial=0):
        methods = summarize(self.cohort, self.states, self.modes, self.repeats)
        labeled = sum(m['labeled'] for m in methods.values())
        expected = len(self.items)
        failed = sum(m['failed'] for m in methods.values())
        elapsed = time.monotonic()-started if started is not None else None
        rate = 60*(labeled-initial)/max(1, elapsed) if elapsed is not None else None
        value = dict(status=status or ('complete' if labeled == expected else 'needs_retry' if failed else 'pending'),
            manifest_sha256=digest(self.manifest), original_positive_samples=len(self.cohort),
            positive_contexts=len({i['row']['target_id'] for i in self.cohort}),
            cohort_splits=dict(Counter(i['row']['split'] for i in self.cohort)),
            labeled=labeled, expected=expected, failed=failed, repeats=self.repeats,
            labels_per_minute=rate, remaining_minutes=(expected-labeled)/rate if rate else None,
            seconds=elapsed, updated_at=time.time(), methods=methods,
            limitation='Positive-only repeat measures conditional survival, not full noise rate or false negatives. '
                'Regenerate combines generation and Judge variability. Judge mode holds original reply fixed. '
                'Three draws are a rough consistency check, not a precise success probability.',
            original_labels_unchanged=True)
        self.seal.check()
        write_json(self.output/'report.json', value)
        columns = [f'{k}/{self.repeats} positive' for k in range(self.repeats+1)]
        lines = ['# Positive label stability', '', f'{labeled}/{expected} repeat labels complete.', '',
            '| Mode | Complete samples | New positive repeats | ' + ' | '.join(columns) + ' |',
            '|---|---:|---:|' + '---:|' * len(columns)]
        for name, m in methods.items():
            h = m['new_positive_count_histogram']
            lines.append(f'| {name} | {m["fully_retested_samples"]} | {m["new_positives"]}/{m["labeled"]} | '
                         + ' | '.join(str(h[str(k)]) for k in range(self.repeats+1)) + ' |')
        lines += ['', value['limitation'], '', 'Original labels and training splits are unchanged.']
        atomic_write(self.output/'REPORT.md', '\n'.join(lines)+'\n')
        return value

    def run(self, workers=16, passes=3):
        require(1 <= workers <= 16 and passes >= 1, 'Invalid concurrency/retry policy')
        started, initial = time.monotonic(), sum(v['status'] == 'complete' for v in self.states.values())
        write_json(self.output/'run.json', dict(pid=os.getpid(), started_at=time.time(), workers=workers))
        self.report('running', started, initial)
        for attempt in range(passes):
            pending = [task for task in self.items if self.states.get(task[0], {}).get('status') != 'complete']
            for i, (key, value) in enumerate(completed_map(self._one, pending, workers), 1):
                self.states[key] = value
                if i == 1 or i % 16 == 0 or i == len(pending):
                    result = self.report('running', started, initial)
                    print(json.dumps({k: result[k] for k in ('labeled', 'expected', 'failed',
                        'labels_per_minute', 'remaining_minutes')}, ensure_ascii=False), flush=True)
        return self.report(started=started, initial=initial)


def execute(inventory, data, teacher, generator, instance, *, stability_output,
            stability_mode='regenerate', additional_draws=2, run=False, workers=16, passes=3,
            min_generation_timeout_seconds=None, **unused):
    with file_lock(Path(stability_output)/'.run.lock', blocking=False):
        batch = StabilityBatch(inventory, data, teacher, generator, instance, stability_output,
                               stability_mode, additional_draws)
        llm = batch.gen_cfg['llm']
        configs = {k: llm[k] for k in ('private', 'group') if k in llm} or {'default': llm}
        original = {k: cfg.get('timeout_seconds', 60) for k, cfg in configs.items()}
        if min_generation_timeout_seconds is not None:
            for cfg in configs.values():
                cfg['timeout_seconds'] = max(cfg.get('timeout_seconds', 60), min_generation_timeout_seconds)
        if run:
            write_json(batch.output/'transport.json', dict(original_generation_timeouts=original,
                effective_generation_timeouts={k: cfg.get('timeout_seconds', 60) for k, cfg in configs.items()},
                policy='transport_wait_only_request_body_and_cache_identity_unchanged'))
        return batch.run(workers, passes) if run else batch.report()
