"""Compatibility for frozen consumers; implementation lives in src.infrastructure.tracing."""
import sys

from src.infrastructure import tracing as _implementation

sys.modules[__name__] = _implementation
