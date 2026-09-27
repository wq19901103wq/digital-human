"""Reuse read-only report inputs within one render, never across requests.

Context-local storage keeps concurrent HTTP requests and instances independent.
Dropping the scope after rendering releases large packs and makes the next
request observe changed records, assets, and version pointers immediately.
"""
from contextvars import ContextVar
from functools import wraps


_reads = ContextVar('dashboard_reads', default=None)


def scope(function):
    @wraps(function)
    def render(*args, **kwargs):
        if _reads.get() is not None:
            return function(*args, **kwargs)
        token = _reads.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _reads.reset(token)
    return render


def once(function):
    """Memoize immutable call arguments only while a render scope is active."""
    @wraps(function)
    def read(*args, **kwargs):
        reads = _reads.get()
        if reads is None:
            return function(*args, **kwargs)
        key = (function, args, tuple(sorted(kwargs.items())))
        if key not in reads:
            reads[key] = function(*args, **kwargs)
        return reads[key]
    return read
