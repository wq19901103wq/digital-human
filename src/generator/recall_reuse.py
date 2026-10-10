"""Compatibility for frozen consumers; implementation lives in src.generator.fewshot.recall_cache."""
import sys

from src.generator.fewshot import recall_cache as _implementation

sys.modules[__name__] = _implementation
