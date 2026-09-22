"""限制同时在途的题数；结果由调用线程逐条落盘。"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context


_worker_limit = ContextVar('parallel_worker_limit', default=16)


@contextmanager
def worker_limit(limit):
    """Scope an explicit executor ceiling without changing other callers' defaults."""
    if type(limit) is not int or limit < 1:
        raise ValueError('worker limit must be a positive integer')
    token = _worker_limit.set(limit)
    try:
        yield
    finally:
        _worker_limit.reset(token)


def completed_map(function, items, workers=1):
    limit = _worker_limit.get()
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= limit:
        raise ValueError(f'workers 必须为 1–{limit} 的整数')
    if workers == 1:
        yield from map(function, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='judge-case') as pool:
        pending = set()
        try:
            for _ in range(workers):
                item = next(iterator, None)
                if item is not None:
                    pending.add(pool.submit(copy_context().run, function, item))
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    yield future.result()
                    item = next(iterator, None)
                    if item is not None:
                        pending.add(pool.submit(copy_context().run, function, item))
        finally:
            for future in pending:
                future.cancel()
