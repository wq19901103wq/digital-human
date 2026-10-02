"""Bounded operational timing for live adapters of frozen executors."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import logging
import os
from pathlib import Path
from threading import Event, Lock, Thread, local
import time

MODE = 'bounded_transport_thread_local_timings_v2'


class TransportProfile:
    def __init__(self, path, metadata, *, interval_seconds=30):
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.interval_seconds = interval_seconds
        self.started_at = time.time()
        self.started = time.monotonic()
        self.buckets = []
        self.activity = {}
        self.local = local()
        self.slices = []
        self.lock = Lock()
        self.stopped = Event()
        self.thread = None
        self.disk_locks = []

    def _metric(self, name):
        bucket = getattr(self.local, 'bucket', None)
        if bucket is None:
            bucket = (Lock(), {})
            self.local.bucket = bucket
            self.local.entries = {}
            with self.lock:
                self.buckets.append(bucket)
        entry = self.local.entries.get(name)
        if entry is None:
            with self.lock:
                activity = self.activity.setdefault(name, dict(lock=Lock(), active=0, active_peak=0))
            metric = dict(count=0, failed=0, total_seconds=0.0, max_seconds=0.0)
            with bucket[0]:
                bucket[1][name] = metric
            entry = bucket[0], metric, activity
            self.local.entries[name] = entry
        return entry

    @property
    def metrics(self):
        with self.lock:
            buckets = list(self.buckets)
            activity = dict(self.activity)
        metrics = {}
        for lock, values in buckets:
            with lock:
                for name, value in values.items():
                    metric = metrics.setdefault(name, dict(count=0, failed=0, total_seconds=0.0,
                        max_seconds=0.0, active=0, active_peak=0))
                    metric['count'] += value['count']
                    metric['failed'] += value['failed']
                    metric['total_seconds'] += value['total_seconds']
                    metric['max_seconds'] = max(metric['max_seconds'], value['max_seconds'])
        for name, value in activity.items():
            if name in metrics:
                with value['lock']:
                    metrics[name]['active'] = value['active']
                    metrics[name]['active_peak'] = value['active_peak']
        return metrics

    def record(self, name, seconds, *, failed=False, count=1):
        lock, metric, activity = self._metric(name)
        with lock:
            metric['count'] += count
            metric['failed'] += bool(failed) * count
            metric['total_seconds'] += seconds
            metric['max_seconds'] = max(metric['max_seconds'], seconds)

    @contextmanager
    def span(self, name):
        lock, metric, activity = self._metric(name)
        with activity['lock']:
            activity['active'] += 1
            activity['active_peak'] = max(activity['active_peak'], activity['active'])
        started = time.monotonic()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.record(name, time.monotonic() - started, failed=failed)
            with activity['lock']:
                activity['active'] -= 1

    @contextmanager
    def admission(self, context, name='request_admission'):
        with self.span(name):
            value = context.__enter__()
        try:
            yield value
        except BaseException as exc:
            if not context.__exit__(type(exc), exc, exc.__traceback__):
                raise
        else:
            context.__exit__(None, None, None)

    def record_slice(self, **fields):
        with self.lock:
            self.slices[:] = [dict(fields)]

    def snapshot(self, status='running'):
        with self.lock:
            disk_locks = list(self.disk_locks)
        disk_wait = dict(count=0, failed=0, total_seconds=0.0, max_seconds=0.0,
                         active=0, active_peak=0)
        for lock, timing in disk_locks:
            with lock:
                disk_wait['count'] += timing['count']
                disk_wait['total_seconds'] += timing['total_seconds']
                disk_wait['max_seconds'] = max(disk_wait['max_seconds'], timing['max_seconds'])
        metrics = self.metrics
        with self.lock:
            if disk_locks:
                metrics['history_disk_lock_wait'] = disk_wait
            return dict(schema=1, mode=MODE, **self.metadata, pid=os.getpid(),
                started_at=self.started_at, updated_at=time.time(), status=status,
                elapsed_seconds=time.monotonic() - self.started,
                metrics=metrics,
                slices=[dict(value) for value in self.slices],
                timings_are_nested=True)

    def flush(self, status='running'):
        from src.iteration.storage import write_json
        try:
            write_json(self.path, self.snapshot(status))
        except (OSError, ValueError, TypeError) as exc:
            logging.getLogger(__name__).warning('Transport profile write failed: %s', exc)

    def _heartbeat(self):
        while not self.stopped.wait(self.interval_seconds):
            self.flush()

    def __enter__(self):
        self.flush()
        self.thread = Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stopped.set()
        self.thread.join()
        status = 'ok' if kind is None else 'stopped' if getattr(value, 'reason', None) == 'time_slice_complete' else 'failed'
        self.flush(status)

    @contextmanager
    def instrument(self, *, history_required=True, disk_history=None):
        from src import llm, tracing
        from src.generator import few_shot, history, history_sources, shared_history
        from src.iteration import control, datasets, learning_guard
        originals = []

        def replace(owner, name, wrapper):
            original = getattr(owner, name)
            originals.append((owner, name, original))
            setattr(owner, name, wrapper(original))

        def measured(name):
            def wrap(original):
                @wraps(original)
                def call(*args, **kwargs):
                    with self.span(name):
                        return original(*args, **kwargs)
                return call
            return wrap

        def admission(original):
            @wraps(original)
            def call(*args, **kwargs):
                return self.admission(original(*args, **kwargs))
            return call

        def model_step(original):
            @wraps(original)
            @contextmanager
            def call(kind, request):
                if kind not in {'llm', 'codex'}:
                    with original(kind, request) as value:
                        yield value
                    return
                with self.span('model_' + kind), original(kind, request) as value:
                    yield value
            return call

        def disk_cached(original):
            @wraps(original)
            def call(reader, cache, key, load):
                name = 'history_disk_row' if cache is reader.cache else 'history_disk_feature'
                missed = False

                def timed_load():
                    nonlocal missed
                    missed = True
                    self.record(name + '_miss', 0.0)
                    with self.span(name + '_read_parse'):
                        return load()

                result = original(reader, cache, key, timed_load)
                if not missed:
                    self.record(name + '_hit', 0.0)
                return result
            return call

        def disk_reader(original):
            @wraps(original)
            def call(reader, *args, **kwargs):
                original(reader, *args, **kwargs)
                reader.lock_timing = dict(count=0, total_seconds=0.0, max_seconds=0.0)
                with self.lock:
                    self.disk_locks.append((reader.lock, reader.lock_timing))
            return call

        try:
            if history_required:
                from src.generator import learned_sources, ranker_report, ranker_samples
                replace(few_shot.PersonaFewShotRetriever, 'retrieve', measured('retrieval_compute'))
                replace(few_shot.PersonaFewShotRetriever, '_load', measured('history_index_load'))
                replace(few_shot.PersonaFewShotRetriever, 'is_approved', measured('history_pool_approval'))
                replace(shared_history.PinnedRetriever, 'retrieve', measured('retrieval_total'))
                replace(history_sources.HistorySources, '__init__', measured('source_index_load'))
                replace(history_sources.HistorySources, 'validate', measured('history_source_validation'))
                replace(history_sources.HistorySources, 'role', measured('history_role_validation'))
                replace(learned_sources, 'reconstruct', measured('ranker_reconstruction'))
                replace(ranker_report, 'load_completed', measured('ranker_completed_sources'))
                replace(ranker_samples, 'teacher_overlap', measured('ranker_teacher_isolation'))
                replace(history, 'filter_rows', measured('history_eligibility_filter'))
                if disk_history is not None:
                    replace(disk_history.BoundedReader, '__init__', disk_reader)
                    replace(disk_history.BoundedReader, 'cached', disk_cached)
            replace(datasets, 'static_sources', measured('material_static_sources'))
            replace(learning_guard.RunSeal, '__init__', measured('material_seal_capture'))
            replace(learning_guard.RunSeal, 'check', measured('material_seal_check'))
            replace(control, 'request', admission)
            replace(llm, 'request', admission)
            replace(llm.ChatClient, '_send', measured('llm_with_retries'))
            replace(tracing, 'step', model_step)
            yield
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)
