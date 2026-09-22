"""Condition-bound label transfer; the source runner lock prevents duplicate work."""
from pathlib import Path
import json
import time

from ...iteration.storage import LockBusy, file_lock, read_json, write_json
from ..history_sources import digest, require
from ..ranker_labels import LabelBatch, _save, _saved


def reuse_source(inventory, requested=None):
    """An expanded inventory always imports its frozen source before running."""
    plan = Path(inventory) / 'expansion_plan.json'
    if not plan.exists():
        return requested
    source = Path(read_json(plan)['reuse_inventory'])
    require(requested is None or Path(requested).resolve() == source.resolve(),
            '复用来源与扩充计划不符')
    return source


def import_labels(batch, previous):
    for key in ('policy', 'data', 'teacher', 'generator', 'clients', 'teacher_client', 'label_usage'):
        require(batch.manifest[key] == previous.manifest[key], f'复用标注条件变化: {key}')
    for path, sha in previous.manifest['inputs'].items():
        if not Path(path).resolve().is_relative_to(previous.inventory.resolve()):
            require(batch.manifest['inputs'].get(path) == sha, f'复用标注输入变化: {path}')
    previous.seal.check()
    batch.seal.check()
    old = {(r['target_id'], e['example']['id']): (r, e) for r, e in previous.items}
    manifest_sha = digest(batch.manifest)
    counts = dict(complete=0, generation_only=0, already_present=0)
    for row, entry in batch.items:
        identity = digest([row['target_id'], entry['example']['id']])
        if identity in batch.states:
            counts['already_present'] += 1
            continue
        source_item = old.get((row['target_id'], entry['example']['id']))
        if source_item is None:
            continue
        old_row, old_entry = source_item
        require(row['target'] == old_row['target'] and entry['example'] == old_entry['example'],
                '同 ID 的目标或示例内容变化，禁止复用标签')
        saved = previous.states.get(identity)
        if not saved or 'generation' not in saved:
            continue
        value = {**saved, 'binding': digest([manifest_sha, row['payload_sha256'], entry]),
                 'split': row['split'], 'reused_from': dict(inventory=str(previous.inventory.resolve()),
                     manifest_sha256=digest(previous.manifest), payload_sha256=saved['payload_sha256'],
                     original_split=old_row['split'])}
        path = batch.output / 'observations' / (identity + '.json')
        _save(path, value)
        batch.states[identity] = _saved(path, value['binding'])
        counts['complete' if saved['status'] == 'complete' else 'generation_only'] += 1
    previous.seal.check()
    batch.seal.check()
    write_json(batch.output / 'reuse.json', dict(source=str(previous.inventory.resolve()), **counts))
    return counts


def import_when_idle(batch, inventory, data, teacher, generator, instance, *, wait=False, progress=print):
    """Wait for the authorized old batch, then atomically transfer its successes."""
    inventory = Path(inventory)
    require(inventory.resolve() != batch.inventory.resolve(), '不能从当前批次导入自身')
    while True:
        try:
            with file_lock(inventory / 'labeling' / '.run.lock', blocking=False):
                previous = LabelBatch(inventory, data, teacher, generator, instance)
                return import_labels(batch, previous)
        except LockBusy:
            if not wait:
                raise
            state = dict(status='waiting_for_reuse_source', source=str(inventory.resolve()), updated_at=time.time())
            write_json(batch.output / 'queue.json', state)
            progress(json.dumps(state, ensure_ascii=False), flush=True)
            time.sleep(30)
