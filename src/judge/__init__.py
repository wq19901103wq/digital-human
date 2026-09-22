"""Judge 机制层包。"""
from __future__ import annotations

LEGACY_CORRECTED_MODE = 'rpa_corrected_pairwise'


def normalize_mode(mode):
    """历史冻结 Judge 版本 config 中的旧 mode 名读取兼容。"""
    return 'corrected_pairwise' if mode == LEGACY_CORRECTED_MODE else mode
