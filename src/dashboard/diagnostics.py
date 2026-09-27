"""Reusable read-only profiling of the real dashboard and live render paths."""
from __future__ import annotations

import cProfile
import json
from pathlib import Path
import pstats
import time

from . import report


def inspect(instance: Path) -> dict:
    results = {}
    for name, render in (
        ('home', lambda: report.dashboard_html(instance / 'experiments')),
        ('live', lambda: report.live_payload(instance)),
    ):
        profiler = cProfile.Profile()
        started = time.perf_counter()
        content = profiler.runcall(render)
        seconds = time.perf_counter() - started
        body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        stats = pstats.Stats(profiler).stats
        slowest = sorted(stats.items(), key=lambda item: item[1][3], reverse=True)[:15]
        results[name] = {
            'seconds': round(seconds, 4), 'bytes': len(body.encode('utf-8')),
            'profile': [dict(function=f'{Path(key[0]).name}:{key[1]}:{key[2]}',
                             calls=value[1], self_seconds=round(value[2], 4),
                             cumulative_seconds=round(value[3], 4))
                        for key, value in slowest],
        }
    return results
