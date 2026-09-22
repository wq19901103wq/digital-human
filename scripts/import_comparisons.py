#!/usr/bin/env python3
"""核验并补录历史专项比较；默认只检查，--apply 才创建实验记录。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.iteration import comparisons, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--comparison', help='仅补录指定比较，供小样本验证')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    versions.switch_instance(args.instance)
    print(json.dumps(comparisons.import_legacy(args.study, apply=args.apply, comparison_id=args.comparison), ensure_ascii=False))


if __name__ == '__main__':
    main()
