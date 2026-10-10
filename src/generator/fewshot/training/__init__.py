"""Offline ranker samples, labels, optimization and comparisons."""


def __getattr__(name):
    if name in {'pairs', 'inner_split'}:
        from . import optimization
        return getattr(optimization, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
