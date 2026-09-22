#!/usr/bin/env bash
# 公开导出：从本仓库生成公开候选树（docs/PUBLIC-RELEASE-PLAN.md §1 白名单的固化实现）。
# 用法: scripts/export_public.sh <目标目录>
# 排除项集中在此脚本的 EXCLUDE 规则中，重新导出不会漏删/多删。
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:?用法: export_public.sh <目标目录>}"

mkdir -p "$DST"
rsync -a --delete \
  --include='/src/***' \
  --include='/tests/***' \
  --include='/prompts/***' \
  --include='/config/***' \
  --include='/examples/***' \
  --include='/scripts/*.py' \
  --exclude='/scripts/legacy/***' \
  --include='/docs/SOP.md' --include='/docs/IMPLEMENTATION.md' --include='/docs/DATA_CONTRACT.md' \
  --include='/docs/RESULT-CACHE.md' --include='/docs/ASYNC-ITERATION.md' \
  --include='/docs/ARCHITECTURE.md' --include='/docs/DATA_MODEL.md' --include='/docs/DATA_BOUNDARIES.md' \
  --include='/docs/OPERATIONS.md' --include='/docs/DESIGN-SIDE-MEASUREMENT-BANK.md' \
  --include='/docs/DATA_BOUNDARIES_PROPOSAL.md' --include='/docs/DATA_BOUNDARIES_IMPLEMENTATION.md' \
  --include='/docs/FEWSHOT-RANKER-PROPOSAL.md' \
  --include='/docs/rules.json' --include='/docs/sop/***' \
  --include='/requirements.txt' --include='/.gitignore' --include='/.githooks/***' --include='/.github/***' \
  --exclude='*' \
  "$SRC/" "$DST/"

# 已知需另行处理（见 PUBLIC-RELEASE-PLAN §7）：scripts/legacy 依赖的 12 个测试、
# AGENTS.md/README/LICENSE/CONTRIBUTING/SECURITY/CHANGELOG 由发布流程在候选树内准备。
echo "导出完成: $DST（随后运行 scripts/scrub_sensitive.py 与 scripts/check_public.py --history）"
