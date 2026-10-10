"""Compatibility for frozen consumers; implementation lives in src.infrastructure.cache."""
import sys

from src.infrastructure import cache as _implementation

sys.modules[__name__] = _implementation
