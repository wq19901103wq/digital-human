#!/usr/bin/env python3
"""Check local documentation links and referenced command files without running examples."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    root = Path(root).resolve()
    findings = []
    for page in [root / 'README.md', *sorted((root / 'docs').glob('*.md'))]:
        for number, line in enumerate(page.read_text().splitlines(), 1):
            for link in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', line):
                url = urlsplit(link.strip('<>'))
                if url.scheme or url.netloc or not url.path:
                    continue
                target = (page.parent / unquote(url.path)).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    findings.append(f'{page.relative_to(root)}:{number}: missing local link {link}')
            for script in re.findall(r'\bscripts/[\w-]+\.py\b', line):
                if not (root / script).is_file():
                    findings.append(f'{page.relative_to(root)}:{number}: missing command {script}')
    return findings


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    issues = check()
    print('\n'.join(issues) if issues else 'documentation: PASS')
    raise SystemExit(bool(issues))
