"""Compatibility for frozen consumers; implementation lives in src.generator.context.background."""
import sys

from src.generator.context import background as _implementation

sys.modules[__name__] = _implementation
