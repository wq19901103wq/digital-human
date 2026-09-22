#!/usr/bin/env python3
"""Check executable rule coverage and reject bypasses of shared write boundaries."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POINTER_WRITERS = {'src/iteration/' + name + '.py' for name in ('baseline', 'branches', 'promote')} | {'scripts/import_wechat.py'}
JOURNAL_WRITERS = {'src/iteration/record_contract.py', 'src/iteration/journal.py'}


def inspect(name, text):
    tree = ast.parse(text, filename=name)
    scopes = [ast.Module(body=[n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])]
    scopes += [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    return sorted(set(issue for scope in scopes for issue in _inspect_scope(name, scope)))


def _inspect_scope(name, tree):
    issues = []
    tainted = set()
    def journal_path(node):
        return any((isinstance(n, ast.Constant) and n.value == 'cases.jsonl') or
                   (isinstance(n, ast.Name) and n.id in tainted) for n in ast.walk(node))
    # Track simple path/stream aliases too; checking only literal writes misses wrappers.
    changed = True
    while changed:
        before = set(tainted)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value and journal_path(node.value):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                tainted.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
            if isinstance(node, ast.withitem) and node.optional_vars and journal_path(node.context_expr):
                tainted.update(n.id for n in ast.walk(node.optional_vars) if isinstance(n, ast.Name))
        changed = tainted != before
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        method = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
        if method == 'save_pointers' and name not in POINTER_WRITERS:
            issues.append(f'{name}:{node.lineno}: pointers must change through promotion or migration')
        if name not in JOURNAL_WRITERS:
            writes = method in ('write', 'write_text', 'write_bytes', 'writelines', 'save_text', 'atomic_write')
            opening = method == 'open' and any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                and any(c in n.value for c in 'wax+') for n in [*node.args, *(k.value for k in node.keywords if k.arg == 'mode')]
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != 'cases.jsonl')
            if (writes or opening) and journal_path(node):
                issues.append(f'{name}:{node.lineno}: case journals must use record_contract.writer')
    return issues


def check(root=ROOT, *, staged=False):
    root = Path(root)
    def read(name):
        if staged:
            result = subprocess.run(['git', '-C', str(root), 'show', ':' + name], capture_output=True, text=True)
            if result.returncode:
                raise FileNotFoundError(name)
            return result.stdout
        return (root / name).read_text()
    if staged:
        names = subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()
    else:
        names = [str(p.relative_to(root)) for folder in ('src', 'scripts') for p in (root / folder).rglob('*.py')]
    findings = []
    for name in names:
        # scripts/legacy/ 是合并时归档的历史一次性脚本（冻结证据工具），新代码规则只约束现行机制层。
        if name.endswith('.py') and name.startswith(('src/', 'scripts/')) and not name.startswith('scripts/legacy/'):
            findings += inspect(name, read(name))
    try:
        rules = json.loads(read('docs/rules.json'))['rules']
    except FileNotFoundError:
        return findings + ['docs/rules.json: missing rule catalog']
    seen = set()
    for rule in rules:
        identity = rule.get('id')
        if not identity or identity in seen or rule.get('enforcement') not in ('code', 'hybrid', 'sop'):
            findings.append(f'rule catalog: invalid or duplicate rule {identity}')
        seen.add(identity)
        if rule.get('enforcement') in ('code', 'hybrid'):
            if not rule.get('code') or not rule.get('tests'):
                findings.append(f'{identity}: code rules require an implementation and regression tests')
            for reference in rule.get('code', []) + rule.get('tests', []):
                name, _, symbol = reference.partition('::')
                try:
                    tree = ast.parse(read(name))
                    if symbol and not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == symbol for n in ast.walk(tree)):
                        findings.append(f'{identity}: missing symbol {reference}')
                except (FileNotFoundError, SyntaxError):
                    findings.append(f'{identity}: missing or invalid source {name}')
        if rule.get('enforcement') in ('sop', 'hybrid') and not rule.get('human_step'):
            findings.append(f'{identity}: human judgment needs an explicit SOP step')
    return findings


def check_instance(instance):
    """Detect results produced by an external script that bypassed registration."""
    from src.config import sha256_file
    imported = {}
    for path in (instance / 'experiments').glob('*/spec.json'):
        spec = json.loads(path.read_text())
        imported.update(spec.get('provenance', {}).get('source_hashes', {}))
    findings = []
    for path in (instance / 'judge_training').glob('*/comparisons/*/cases.jsonl'):
        relative = str(path.relative_to(instance))
        if imported.get(relative) != sha256_file(path):
            findings.append(f'{relative}: result is not registered in experiments with matching content')
    return findings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--instance')
    args = parser.parse_args()
    issues = check(staged=args.staged)
    if args.instance:
        from src.iteration import versions
        issues += check_instance(versions.switch_instance(args.instance))
    print('\n'.join(issues) if issues else 'rules: PASS')
    raise SystemExit(bool(issues))
