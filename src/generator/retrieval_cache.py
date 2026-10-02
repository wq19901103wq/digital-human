"""Bounded exact-call reuse around a frozen, source-guarded retriever."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import os
from pathlib import Path
import pickle
import struct
from threading import Lock

MODE = 'guarded_parallel_exact_retrieval_calls_v2'
CAPACITY = 128
MAX_BYTES = 4 * 1024 * 1024


def identity(value):
    kind = type(value)
    if value is None or kind in (bool, int, str, bytes):
        return kind.__name__, value
    if kind is float:
        return 'float', struct.pack('!d', value)
    if isinstance(value, Path):
        return 'path', str(value)
    if kind is dict:
        pairs = [(identity(key), identity(item)) for key, item in value.items()]
        return 'dict', tuple(sorted(pairs, key=lambda pair: pickle.dumps(pair[0])))
    if kind in (list, tuple):
        return kind.__name__, tuple(identity(item) for item in value)
    if kind in (set, frozenset):
        return kind.__name__, tuple(sorted((identity(item) for item in value), key=pickle.dumps))
    raise TypeError('unsupported exact-retrieval input')


class CallCache:
    def __init__(self, capacity, max_bytes):
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            if key not in self.entries:
                return None
            self.entries.move_to_end(key)
            payload = self.entries[key]
        return pickle.loads(payload)

    def put(self, key, rows):
        if self.capacity <= 0 or self.max_bytes <= 0:
            return
        try:
            snapshot = [dict(row) if isinstance(row, Mapping) else row for row in rows]
            identity(snapshot)
            payload = pickle.dumps(snapshot, protocol=5)
        except (TypeError, ValueError, pickle.PickleError):
            return
        size = len(key) + len(payload) + 128
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self.entries:
                self.bytes -= len(key) + len(self.entries.pop(key)) + 128
            while self.entries and (len(self.entries) >= self.capacity or self.bytes + size > self.max_bytes):
                old_key, old_payload = self.entries.popitem(last=False)
                self.bytes -= len(old_key) + len(old_payload) + 128
            self.entries[key] = payload
            self.bytes += size


@contextmanager
def exact_retrieval_calls(shared_history, *, profile=None, capacity=CAPACITY, max_bytes=MAX_BYTES):
    originals = {name: getattr(shared_history.PinnedRetriever, name)
                 for name in ('retrieve', 'render_selected', 'is_approved')}
    caches = {}
    pending = {}
    cache_lock = Lock()

    def span(name):
        return profile.span(name) if profile is not None else nullcontext()

    def event(name):
        if profile is not None:
            profile.record(name, 0.0)

    def guarded_retrieve(retriever, **kwargs):
        retriever._check()
        rows = retriever._original.retrieve(**kwargs)
        retriever._check()
        return rows

    @wraps(originals['render_selected'])
    def render_selected(retriever, rows, max_chars=2500):
        retriever._check()
        result = retriever._original.render_selected(rows, max_chars=max_chars)
        retriever._check()
        return result

    @wraps(originals['is_approved'])
    def is_approved(retriever):
        retriever._check()
        return True

    @contextmanager
    def same_request(retriever, key):
        request_id = (retriever, key)
        with cache_lock:
            if retriever not in caches:
                caches[retriever] = CallCache(capacity, max_bytes)
            cache = caches[retriever]
            if request_id not in pending:
                pending[request_id] = [Lock(), 0]
            entry = pending[request_id]
            entry[1] += 1
        acquired = False
        try:
            with span('retrieval_lock_wait'):
                entry[0].acquire()
            acquired = True
            yield cache
        finally:
            if acquired:
                entry[0].release()
            with cache_lock:
                entry[1] -= 1
                if not entry[1]:
                    del pending[request_id]

    @wraps(originals['retrieve'])
    def retrieve(retriever, **kwargs):
        retriever._check()
        source = retriever._original._history_sources
        if source is None or retriever._original.history_policy != 'complete_before_input_v1':
            event('retrieval_cache_bypass')
            return guarded_retrieve(retriever, **kwargs)
        source.check()
        try:
            with span('retrieval_cache_key'):
                flags = {name: value for name, value in vars(retriever._original).items()
                         if not name.startswith('_')}
                request = (kwargs, flags, id(source), os.environ.get('PERSONA_FEW_SHOT_MIN_SIMILARITY'))
                key = hashlib.sha256(pickle.dumps(identity(request), protocol=5)).digest()
        except TypeError:
            event('retrieval_cache_bypass')
            return guarded_retrieve(retriever, **kwargs)
        with same_request(retriever, key) as cache:
            retriever._check()
            source.check()
            with span('retrieval_cache_lookup'):
                rows = cache.get(key)
            if rows is None:
                event('retrieval_cache_miss')
                rows = guarded_retrieve(retriever, **kwargs)
                source.check()
                retriever._check()
                with span('retrieval_cache_store'):
                    cache.put(key, rows)
            else:
                event('retrieval_cache_hit')
            source.check()
            retriever._check()
            return rows

    shared_history.PinnedRetriever.retrieve = retrieve
    shared_history.PinnedRetriever.render_selected = render_selected
    shared_history.PinnedRetriever.is_approved = is_approved
    try:
        yield caches
    finally:
        for name, original in originals.items():
            setattr(shared_history.PinnedRetriever, name, original)
        with cache_lock:
            caches.clear()
            pending.clear()
