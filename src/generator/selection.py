"""Compatibility for frozen consumers; implementation lives in src.generator.fewshot.selection."""
import sys

from src.generator.fewshot import selection as _implementation

sys.modules[__name__] = _implementation
