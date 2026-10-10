"""Compatibility for frozen consumers; implementation lives in src.judge.scoring.ownership."""
import sys

from src.judge.scoring import ownership as _implementation

sys.modules[__name__] = _implementation
