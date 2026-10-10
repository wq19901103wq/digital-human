"""Compatibility for frozen consumers; implementation lives in src.generator.fewshot.learned_selection."""
import sys

from src.generator.fewshot import learned_selection as _implementation

sys.modules[__name__] = _implementation
