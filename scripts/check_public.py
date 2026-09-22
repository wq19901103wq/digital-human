#!/usr/bin/env python3
"""公开仓库边界扫描：fail-closed 的"内容+路径"守门（区别于私有仓的逐文件白名单）。

公开仓的信任模型：新文件默认意图公开，所以用 ① 路径黑名单（绝不允许出现的
目录/文件）+ ② 内容模式（home 路径三平台/聊天账号/密钥/阻断词）+ ③ src 包
依赖方向（禁止 import scripts/instances/private）三层防护，外加
--history 全历史扫描。阻断词配置在 config/public_boundary.json。
不会打印命中的敏感值本身。
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTENT_PATTERNS = {
    'personal_home': re.compile(
        r'(?:/users/|/home/)[a-z][a-z0-9_.-]*[/\\]|[a-z]:\\\\users\\\\[a-z][a-z0-9_.-]*\\\\', re.I),
    'messaging_account': re.compile(r'wxid_[a-z0-9_]{6,}|\d{8,}@chatroom', re.I),
    'provider_secret': re.compile(
        r'(?<![A-Za-z0-9])(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}'
        r'|github_pat_[A-Za-z0-9_]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16})'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'internal_ip': re.compile(r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b'),
}
SKIP_SUFFIX = {'.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.woff', '.woff2', '.pdf'}


def boundary_config(root: Path) -> dict:
    path = root / 'config' / 'public_boundary.json'
    default = {'blocked_paths': ['instances', 'private', '.env', '.env.*', 'data', 'runs'],
               'blocked_words': []}
    if not path.is_file():
        return default
    cfg = json.loads(path.read_text())
    return {**default, **cfg}


def inspect(name: str, content: bytes, blocked_words: list[re.Pattern],
            blocked_path: list[re.Pattern], exempt: set) -> list[tuple[str, int, str]]:
    findings = []
    for pattern in blocked_path:
        if pattern.search(name):
            findings.append((name, 0, 'blocked_path'))
            break
    if b'\0' in content:
        return findings + [(name, 0, 'binary_requires_review')]
    text = content.decode('utf-8', errors='replace')
    for i, line in enumerate(text.splitlines(), 1):
        for kind, pattern in CONTENT_PATTERNS.items():
            if pattern.search(line):
                findings.append((name, i, kind))
        if name not in exempt:
            for word in blocked_words:
                if word.search(line):
                    findings.append((name, i, 'blocked_word'))
    if name.startswith('src/') and name.endswith('.py'):
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return findings + [(name, exc.lineno or 0, 'invalid_python')]
        for node in ast.walk(tree):
            modules = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                       [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            if any(m.split('.')[0] in {'scripts', 'instances', 'private'} for m in modules):
                findings.append((name, node.lineno, 'src_imports_cli_or_private'))
            # 动态导入同样禁止（__import__ / importlib.import_module 字面量）
            if isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Name) and target.id == '__import__' and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and str(first.value).split('.')[0] in {'scripts', 'instances', 'private'}:
                        findings.append((name, node.lineno, 'src_dynamic_import_private'))
                if (isinstance(target, ast.Attribute) and target.attr == 'import_module'
                        and node.args and isinstance(node.args[0], ast.Constant)
                        and str(node.args[0].value).split('.')[0] in {'scripts', 'instances', 'private'}):
                    findings.append((name, node.lineno, 'src_dynamic_import_private'))
    return findings


def check(root=ROOT, history=False) -> list[tuple[str, int, str]]:
    root = Path(root)
    cfg = boundary_config(root)
    blocked_words = [re.compile(rf'\b{re.escape(w)}\b', re.I) for w in cfg['blocked_words']]
    blocked_path = [re.compile(fnmatch_translate(p)) for p in cfg['blocked_paths']]
    exempt = set(cfg.get('blocked_word_exempt', []))

    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)

    issues: list[tuple[str, int, str]] = []
    targets = []
    if (root / '.git').exists():
        for row in git('ls-files', '--stage', '-z').split(b'\0'):
            if not row:
                continue
            meta, name = row.split(b'\t', 1)
            mode, oid, _ = meta.decode().split()
            name = name.decode()
            if mode not in ('100644', '100755'):
                issues.append((name, 0, 'non_regular_git_entry'))
            else:
                targets.append((name, git('cat-file', 'blob', oid)))
        if history:
            # 提交元数据：作者+提交者+message+标签（含附注）全过内容模式与阻断词。
            # 字段分隔 %x1f、记录分隔 %x1e；不能用 str.strip()（\x1f 是 Unicode 空白会被吃掉）。
            meta = subprocess.run(
                ['git', '-C', str(root), 'log', '--all',
                 '--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%s%x1f%b%x1e'],
                capture_output=True, text=True).stdout
            for record in meta.split('\x1e'):
                record = record.rstrip('\n')
                if not record:
                    continue
                fields = record.split('\x1f')
                if len(fields) < 6:
                    continue
                cid, text = fields[0], '\n'.join(f for f in fields[1:] if f)
                for kind, pattern in CONTENT_PATTERNS.items():
                    if pattern.search(text):
                        issues.append((f'commit-meta:{cid[:8]}', 0, f'git_metadata_{kind}'))
                for word in blocked_words:
                    if word.search(text):
                        issues.append((f'commit-meta:{cid[:8]}', 0, 'git_metadata_blocked_word'))
            tag_info = subprocess.run(
                ['git', '-C', str(root), 'tag', '-l',
                 '--format=%(refname:short)%x1f%(taggername)%x1f%(taggeremail)%x1f%(subject)%x1f%(body)%x1e'],
                capture_output=True, text=True).stdout
            for record in tag_info.split('\x1e'):
                record = record.rstrip('\n')
                if not record:
                    continue
                parts = record.split('\x1f')
                if not parts or not parts[0]:
                    continue
                text = '\n'.join(x for x in parts if x)
                for kind, pattern in CONTENT_PATTERNS.items():
                    if pattern.search(text):
                        issues.append((f'tag:{parts[0]}', 0, f'git_metadata_{kind}'))
                for word in blocked_words:
                    if word.search(text):
                        issues.append((f'tag:{parts[0]}', 0, 'git_metadata_blocked_word'))
            seen = set()
            for commit in git('rev-list', '--all').decode().split():
                for row in git('ls-tree', '-r', '-z', commit).split(b'\0'):
                    if not row:
                        continue
                    meta, name = row.split(b'\t', 1)
                    mode, kind, oid = meta.decode().split()
                    key = (name, oid)
                    if key in seen:
                        continue
                    seen.add(key)
                    if kind != 'blob' or mode == '120000':
                        issues.append((name.decode(), 0, 'non_regular_historical_entry'))
                    else:
                        issues.extend((f'{commit[:8]}:{p}', line, rule)
                                      for p, line, rule in
                                      inspect(name.decode(), git('cat-file', 'blob', oid),
                                              blocked_words, blocked_path, exempt))
    else:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix not in SKIP_SUFFIX and not path.is_symlink():
                rel = path.relative_to(root).as_posix()
                if any(part in {'.git', '__pycache__', '.venv'} for part in path.parts):
                    continue
                targets.append((rel, path.read_bytes()))
            elif path.is_symlink():
                issues.append((path.relative_to(root).as_posix(), 0, 'symlink_forbidden'))
    for name, content in targets:
        issues.extend(inspect(name, content, blocked_words, blocked_path, exempt))
    return sorted(set(issues))


def fnmatch_translate(pattern: str) -> str:
    # 根锚定（与 .gitignore 的 /x/ 一致）：模式匹配仓库根下的路径段序列。
    parts = [re.escape(p).replace(r'\*', '[^/]*') for p in pattern.split('/')]
    return '^' + '/'.join(parts) + '($|/)'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='扫描全部提交历史')
    args = parser.parse_args()
    issues = check(history=args.history)
    for path, line, rule in issues:
        print(f'{path}:{line}: {rule}')
    print(f'public boundary: {"PASS" if not issues else "FAIL"} ({len(issues)} findings)',
          file=sys.stderr)
    sys.exit(0 if not issues else 1)


if __name__ == '__main__':
    main()
