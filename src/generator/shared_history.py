"""Compatibility for frozen consumers; implementation lives in src.generator.fewshot.shared_history."""
import sys

from src.generator.fewshot import shared_history as _implementation

sys.modules[__name__] = _implementation
