"""Compatibility for frozen consumers; implementation lives in src.infrastructure.config."""
import sys

from src.infrastructure import config as _implementation

sys.modules[__name__] = _implementation
