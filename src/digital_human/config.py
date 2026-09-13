"""配置加载：超参(可提交) + 通用工具。

三层架构：
- 机制层：本文件 + config/settings.yaml（超参：评测规模/占比/采用门槛，与实验和数据无关）
- 版本层：private/ 下不可变版本目录 + pointers.json（见 iteration/versions.py）
- 数据层：版本目录内容（切分数据、池、快照）
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """配置缺失/不一致：必须在任何模型调用前失败。"""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_settings() -> dict[str, Any]:
    # 命令行直接运行时也读取项目连接配置；显式环境变量仍优先。
    load_dotenv(ROOT / ".env", override=False)
    path = ROOT / "config" / "settings.yaml"
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
