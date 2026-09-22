"""Reuse frozen retrieval checkpoints when the requested batch size changes.

Keep the source batch immutable. The ordinary expansion runner validates and
finalizes the destination, including its time split and exact observation cap.
"""
from pathlib import Path

from ...config import sha256_file
from ...iteration.storage import file_lock, read_json, write_once_json
from ..history_sources import digest, require
from ..ranker_expansion import cap_rows, isolated_rows
from ..ranker_labels import POLICY
from ..ranker_samples import RECIPE, fingerprint


def seed_checkpoints(source, output, *, observations, reuse_inventory):
    """Copy the needed deterministic prefix, rebinding only the batch plan."""
    source, output, reuse_inventory = map(Path, (source, output, reuse_inventory))
    require(type(observations) is int and observations > 0, '样本目标必须为正整数')
    require(output.resolve() not in (source.resolve(), reuse_inventory.resolve()),
            '调整规模必须使用新目录，保留旧断点')
    with file_lock(source / '.prepare.lock', blocking=False), \
            file_lock(output / '.prepare.lock', blocking=False):
        plan_path = source / 'expansion_plan.json'
        previous = read_json(plan_path)
        require(previous['recipe'] == RECIPE and previous['policy'] == POLICY,
                '复用检索断点的采样或隔离规则变化')
        require(Path(previous['reuse_inventory']).resolve() == reuse_inventory.resolve(),
                '复用检索断点的原始样本来源变化')
        require(fingerprint(previous['inputs']) == previous['inputs'], '复用检索断点的冻结输入变化')
        plan = {**previous, 'observations': observations}
        write_once_json(output / 'expansion_plan.json', plan)
        source_sha, destination_sha = digest(previous), digest(plan)
        rows, files, total = [], {}, 0
        for cid in previous['target_ids']:
            path = source / 'preparation' / f'{cid}.json'
            require(path.resolve().is_relative_to(source.resolve()), '检索断点路径越界')
            if not path.exists():
                break
            record = read_json(path)
            payload = {k: v for k, v in record.items() if k != 'payload_sha256'}
            require(record['plan_sha256'] == source_sha and record['target_id'] == cid and
                    record['payload_sha256'] == digest(payload), '复用检索断点来源或内容变化')
            payload['plan_sha256'] = destination_sha
            copied = {**payload, 'payload_sha256': digest(payload)}
            write_once_json(output / 'preparation' / f'{cid}.json', copied)
            files[str(path.relative_to(source))] = sha256_file(path)
            if record['candidates']:
                rows.append(record)
                total += len(record['candidates'])
            if total >= observations:
                capped = cap_rows(rows, observations)
                selected, _ = isolated_rows(capped)
                total = sum(len(row['candidates']) for row in selected)
                if total == observations:
                    break
                retained = {row['target_id'] for row in selected}
                rows = [row for row in capped if row['target_id'] in retained]
        require(fingerprint(previous['inputs']) == previous['inputs'], '复用期间冻结输入变化')
        provenance = dict(source=str(source.resolve()), source_plan_sha256=source_sha,
            destination_plan_sha256=destination_sha, requested_observations=observations,
            checkpoint_files=files, copied_targets=len(files),
            source_plan_file_sha256=sha256_file(plan_path),
            operation='reuse_retrieval_only_then_finalize_with_standard_expansion')
        write_once_json(output / 'preparation_reuse.json', provenance)
        return provenance
