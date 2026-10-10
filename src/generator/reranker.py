"""Compatibility for frozen consumers; implementation lives in src.generator.fewshot.reranker."""
import sys

from src.generator.fewshot import reranker as _implementation

sys.modules[__name__] = _implementation
