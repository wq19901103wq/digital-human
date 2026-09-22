"""Source-text matching helpers. Extracted from scripts/legacy/audit_judge_dataset_rerun.py."""
from __future__ import annotations

import html
import re


def norm(text):
    return re.sub(r'\s+', '', html.unescape(str(text or '')))


def source_text_matches(expected, actual):
    if norm(expected) == norm(actual):
        return True
    # The inherited pool replaces these PII fields; preserve that documented
    # transformation when checking source text. Timestamp still must match.
    for pattern, replacement in [
        (r'(?<!\d)1[3-9]\d{9}(?!\d)', '[手机号]'),
        (r'(?i)https?://\S+|www\.\S+', '[链接]'),
        (r'(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', '[邮箱]'),
        (r'(?i)wxid_[a-z0-9_]+', '[微信账号]'),
    ]:
        actual = re.sub(pattern, replacement, actual)
    pattern = '.{2,30}?'.join(re.escape(norm(s)) for s in expected.split('[联系人]'))
    return re.fullmatch(pattern, norm(actual)) is not None
