"""Compatibility for frozen consumers; implementation lives in src.generator.context.source_projection."""
import sys

from src.generator.context import source_projection as _implementation

sys.modules[__name__] = _implementation
