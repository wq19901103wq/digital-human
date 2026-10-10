"""Compatibility for frozen consumers; implementation lives in src.data.acceptance_refresh."""
import sys

from src.data import acceptance_refresh as _implementation

sys.modules[__name__] = _implementation
