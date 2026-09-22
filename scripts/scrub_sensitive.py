#!/usr/bin/env python3
"""公开发布前的敏感信息批量清洗（幂等，可重复执行）。

只处理白名单内的文本文件；替换规则集中在 RULES，新增敏感模式时加一条即可。
用法：
  python scripts/scrub_sensitive.py [--dry-run] [--root .]
退出码：有替换未执行（dry-run 模式下存在命中）为 1，否则 0。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', 'instances', 'data',
             'runs', 'dashboard', 'node_modules', 'dist', 'build'}
TEXT_SUFFIX = {'.py', '.md', '.json', '.yaml', '.yml', '.txt', '.toml', '.cfg',
               '.html', '.js', '.css', '.sh', '.example'}
# 清洗工具自身与其配置：规则字面量是规则本体，不许清洗（否则自我损坏）。
SCRUB_EXEMPT = {'scripts/scrub_sensitive.py', 'config/public_boundary.json'}

# (名字, 正则, 替换, 说明)。顺序有意义：先长后短、先路径后单词。
RULES = [
    ('home-windows', re.compile(r'[A-Za-z]:\\\\Users\\\\yihanwang', re.I), '~', '个人 home 路径（Windows）'),
    ('home-mac', re.compile(r'/Users/yihanwang', re.I), '~', '个人 home 路径（macOS）'),
    ('home-linux', re.compile(r'/home/yihanwang', re.I), '~', '个人 home 路径（Linux）'),
    ('home-tilde', re.compile(r'~yihanwang\b', re.I), '~', '个人 home 简写（含大写）'),
    ('wxid', re.compile('wxid_' + r'[a-z0-9_]{6,}', re.I), 'masked-account-id', '聊天账号 ID'),
    ('instance-qianwang', re.compile(r'\bqianwang\b', re.I), 'example-agent', '实例名（含大小写变体）'),
    ('instance-wangyihan', re.compile(r'\bwangyihan\b', re.I), 'example-operator', '实例名（含大小写变体）'),
    ('operator-name', re.compile(r'\byihanwang\b', re.I), 'operator', '操作者名（路径已先处理）'),
]


def candidates(root: Path):
    for path in sorted(root.rglob('*')):
        if path.is_dir() or not path.is_file() or path.suffix not in TEXT_SUFFIX:
            continue
        rel = path.relative_to(root).as_posix()
        if rel in SCRUB_EXEMPT or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.startswith('.env'):  # 无后缀配置文件也纳入清洗
            yield path
            continue
        try:
            if b'\0' in path.read_bytes():
                continue
        except OSError:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--dry-run', action='store_true', help='只报告不写入')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    hits: dict[str, int] = {name: 0 for name, _, _, _ in RULES}
    files_touched = 0
    for path in candidates(root):
        text = path.read_text(encoding='utf-8', errors='replace')
        new, changed = text, False
        for name, pattern, repl, _ in RULES:
            new, n = pattern.subn(repl, new)
            if n:
                hits[name] += n
                changed = True
        if changed:
            files_touched += 1
            if not args.dry_run:
                path.write_text(new, encoding='utf-8')
    total = sum(hits.values())
    for name, n in hits.items():
        if n:
            print(f'{name}: {n} 处')
    print(f'共 {total} 处，涉及 {files_touched} 个文件' + ('（dry-run，未写入）' if args.dry_run else ''))
    return 1 if (args.dry_run and total) else 0


if __name__ == '__main__':
    sys.exit(main())
