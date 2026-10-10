"""Compatibility for frozen consumers; implementation lives in src.judge.scoring.contribution."""
import sys

from src.judge.scoring import contribution as _implementation

sys.modules[__name__] = _implementation
