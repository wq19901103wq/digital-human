"""Compatibility for frozen consumers; implementation lives in src.infrastructure.storage."""
import sys

from src.infrastructure import storage as _implementation

sys.modules[__name__] = _implementation
