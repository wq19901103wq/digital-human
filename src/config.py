"""实例设置加载、通用校验与命令行配置解析；默认设置仅是框架示例。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


class ConfigError(RuntimeError):
    """配置缺失/不一致：必须在任何模型调用前失败。"""


def valid_name(value: str, max_length: int = 200) -> str:
    """A single portable directory component, never a relative or absolute path."""
    if (not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value)
            or len(value) > max_length):
        raise ConfigError(f"名称只能含字母、数字、下划线和连字符（最多 {max_length} 字符）: {value}")
    return value


def parse_overrides(values: list[str]) -> dict:
    """Parse repeated dotted key=value arguments consistently for Gen and Judge.

    JSON values retain their types; unquoted model names remain strings.
    Reject ambiguous parent/child assignments instead of silently losing keys.
    """
    result = {}
    assigned = set()
    for raw in values:
        key, separator, text = raw.partition('=')
        parts = key.split('.')
        if not separator or not all(parts):
            raise ConfigError('--override 格式应为 key=value，字段名不能为空')
        if any(key.startswith(old + '.') or old.startswith(key + '.') for old in assigned):
            raise ConfigError('--override 不能同时设置父字段和子字段')
        try:
            value = json.loads(text, parse_constant=_invalid_json_constant)
        except (ValueError, TypeError):
            value = text
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
        assigned.add(key)
    return result


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings_path() -> Path:
    return Path(os.environ.get("DH_SETTINGS_FILE", str(ROOT / "config" / "settings.yaml"))).resolve()


def load_settings() -> dict[str, Any]:
    # 命令行直接运行时也读取项目连接配置；显式环境变量仍优先。
    load_dotenv(Path(os.environ.get("DH_ENV_FILE", str(ROOT / ".env"))), override=False)
    path = settings_path()
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    """settings.yaml 里的相对路径都基于项目根。"""
    return (ROOT / path_str).resolve()


def dataset_plan(settings: dict[str, Any], dataset: str) -> dict[str, int]:
    """按超参计算数据集构成：总数 + 群聊/私聊拆分。"""
    spec = settings["evaluation"][dataset]
    total = int(spec["total"])
    group = round(total * float(spec["group_ratio"]))
    return {"total": total, "group": group, "private": total - group}


def _invalid_json_constant(value):
    raise ValueError(f'non-finite JSON constant: {value}')
