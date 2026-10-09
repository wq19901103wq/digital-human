"""Pytest 引导：保证仓库根与 tests/ 各级目录均可裸 import。

tests/ 无 __init__.py，靠 namespace package 解析 ``tests.<组>.test_x`` 形式的
import；同时把分组子目录加入 sys.path，使 ``from test_x import ...``、
``from leakage_support import ...`` 等裸 import 不受文件所在子目录影响。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (str(ROOT.parent), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
for sub in sorted(ROOT.iterdir()):
    if sub.is_dir() and not sub.name.startswith(("_", ".")) and sub.name != "examples":
        p = str(sub)
        if p not in sys.path:
            sys.path.insert(0, p)
