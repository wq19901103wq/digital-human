"""Compatibility for frozen consumers; implementation lives in src.share.versions."""
import sys

from src.share import versions as _implementation

sys.modules[__name__] = _implementation
