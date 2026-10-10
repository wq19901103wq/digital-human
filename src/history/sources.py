"""Compatibility for frozen consumers; implementation lives in src.data.history.sources."""
import sys

from src.data.history import sources as _implementation

sys.modules[__name__] = _implementation
