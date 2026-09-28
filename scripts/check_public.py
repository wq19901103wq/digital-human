#!/usr/bin/env python3
"""Fail closed on unlisted release files, instance imports and common private literals."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache', '.worktrees',
            'instances', 'private', 'data', 'runs', 'dashboard', 'build', 'dist', '.env', '.dashboard.pid'}
PATTERNS = {
    'personal_home': re.compile(r'(?:/Users/|/home/)[A-Za-z][A-Za-z0-9_.-]*/'),
    'messaging_account': re.compile(r'wxid_[A-Za-z0-9_]{8,}|\d{8,}@chatroom'),
    'provider_secret': re.compile(r'(?<![A-Za-z0-9])(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16})'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}


def inspect(name, content, allowed):
    findings = []
    if name not in allowed:
        findings.append((name, 0, 'not_in_public_manifest'))
    if b'\0' in content:
        return findings + [(name, 0, 'binary_requires_review')]
    text = content.decode('utf-8', errors='replace')
    for i, line in enumerate(text.splitlines(), 1):
        for kind, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append((name, i, kind))
    if name.startswith('src/') and name.endswith('.py'):
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return findings + [(name, exc.lineno or 0, 'invalid_python')]
        for node in ast.walk(tree):
            modules = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                       [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            if any(m.split('.')[0] in {'scripts', 'instances', 'private'} for m in modules):
                findings.append((name, node.lineno, 'framework_imports_instance_or_cli'))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'switch_instance':
                if node.args and isinstance(node.args[0], ast.Constant):
                    findings.append((name, node.lineno, 'hardcoded_instance_selection'))
    return findings


def candidates(root):
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if (relative.parts[0] in EXCLUDED or any(p == '__pycache__' or p.endswith('.egg-info') for p in relative.parts)
                or path.name.startswith('.env.')):
            continue
        if path.is_symlink():
            yield relative.as_posix(), None
        elif path.is_file() and path.suffix != '.pyc':
            yield relative.as_posix(), path.read_bytes()


def check(root=ROOT, history=False, staged=False):
    root = Path(root)
    allowed = set(json.loads((root / 'public-files.json').read_text())['files'])
    issues = []
    for name, content in candidates(root):
        issues.extend([(name, 0, 'symlink_forbidden')] if content is None else inspect(name, content, allowed))
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)
    if (root / '.git').exists():
        # Tracked ignored files are still publishable. Check the index even if the working tree hides them.
        for row in git('ls-files', '--stage', '-z').split(b'\0'):
            if not row:
                continue
            meta, name = row.split(b'\t', 1)
            mode, oid, stage = meta.decode().split()
            name = name.decode()
            if mode != '100644' and mode != '100755':
                issues.append((name, 0, 'non_regular_git_entry'))
            else:
                issues.extend(inspect(name, git('cat-file', 'blob', oid), allowed))
        if history:
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
                        issues.extend((f'{commit[:8]}:{p}', line, rule) for p, line, rule in
                                      inspect(name.decode(), git('cat-file', 'blob', oid), allowed))
    return sorted(set(issues))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    issues = check(history=args.history, staged=args.staged)
    # Never print the matched private value or file contents.
    for path, line, rule in issues:
        print(f'{path}:{line}: {rule}')
    print(f'public boundary: {"FAIL" if issues else "PASS"} ({len(issues)} findings)')
    raise SystemExit(bool(issues))


if __name__ == '__main__':
    main()
