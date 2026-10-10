"""Compatibility for frozen consumers; implementation lives in src.generator.context.persona."""
import sys

from src.generator.context import persona as _implementation

sys.modules[__name__] = _implementation
