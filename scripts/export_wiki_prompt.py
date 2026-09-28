#!/usr/bin/env python3
"""Export structured Wiki review packets; runtime still requires case filtering."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap.wiki_prompt import export  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-chars', type=int, default=16000)
    args = parser.parse_args()
    output = args.output.resolve()
    if not any(output.is_relative_to(ROOT / part) for part in ('instances', 'private')):
        parser.error('private output must stay in instances/ or private/')
    if output.is_relative_to(args.source.resolve().parent):
        parser.error('export outside the frozen source directory')
    print(json.dumps(export(args.source, output, max_chars=args.max_chars), ensure_ascii=False))


if __name__ == '__main__':
    main()
