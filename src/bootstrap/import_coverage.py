"""Source-manifest-backed message coverage; shared by check and publication."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..config import ConfigError, sha256_file
from . import ingest


def frozen_exports(directory):
    manifest = json.loads((Path(directory) / 'manifest.json').read_text())
    files = manifest.get('source_files', {})
    if not files or any(Path(p).suffix != '.json' for p in files):
        raise ConfigError('此版本没有冻结 WeFlow 原始导出清单，不能证明消息类型覆盖')
    changed = [p for p, sha in files.items() if not Path(p).is_file() or sha256_file(Path(p)) != sha]
    if changed:
        raise ConfigError(f'原始导出与冻结清单不符（{len(changed)} 个文件），不能按旧来源重建')
    return files


def _legacy_key(row):
    content = row.get('event', {}).get('original_content', row['text']).strip()
    return (row['chat_id'], row['sender'], row['timestamp'], row['is_self'], content)


def audit(directory):
    directory = Path(directory)
    files = frozen_exports(directory)
    statistics = {}
    expected = ingest.load_weflow_files(files, statistics=statistics)
    archived, modern = Counter(), Counter()
    with (directory / 'messages.jsonl').open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if 'event' in row:
                modern[ingest.message_identity(row)] += 1
            else:
                archived[_legacy_key(row)] += 1
    legacy = bool(archived)
    legacy_keys = set(archived)
    by_type = {}
    for row in expected:
        event = row['event']
        count = by_type.setdefault(event['kind'], dict(source=0, retained=0, missing=0,
            missing_event_metadata=0, content_unavailable=0, fallback_identity=0,
            missing_with_same_legacy_key=0))
        count['source'] += 1
        count['content_unavailable'] += not bool(event.get('original_content', row['text']).strip())
        count['fallback_identity'] += event['id'].startswith('fallback:')
        key = ingest.message_identity(row)
        if modern[key]:
            modern[key] -= 1
            count['retained'] += 1
        elif legacy and archived[_legacy_key(row)]:
            archived[_legacy_key(row)] -= 1
            count['retained'] += 1
            count['missing_event_metadata'] += 1
        else:
            count['missing'] += 1
            count['missing_with_same_legacy_key'] += _legacy_key(row) in legacy_keys
    # Catch changes during the read, not only before it.
    frozen_exports(directory)
    extras = sum(archived.values()) + sum(modern.values())
    missing = sum(c['missing'] for c in by_type.values())
    metadata = sum(c['missing_event_metadata'] for c in by_type.values())
    return dict(data_ref=directory.name, passed=not (missing or extras or metadata),
        source_files=len(files), source_hashes_match=True, ingestion=statistics,
        source_events=len(expected), by_type=by_type, missing=missing,
        missing_event_metadata=metadata, unmatched_archive_events=extras,
        scope='逐条核对冻结原始导出中的消息事件及元信息；不证明导出本身已覆盖数据库所有消息或媒体内容')
