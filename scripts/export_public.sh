#!/usr/bin/env bash
# Public export: build the public candidate tree from this repo.
# Usage: scripts/export_public.sh <target-dir>
# Exclusion rules are fixed here so re-exports are consistent.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:?usage: export_public.sh <target-dir>}"

mkdir -p "$DST"
rsync -a --delete \
  --include='/src/***' \
  --include='/tests/***' \
  --include='/prompts/***' \
  --include='/config/***' \
  --include='/examples/***' \
  --include='/scripts/' \
  --include='/scripts/*.py' \
  --exclude='/scripts/legacy/***' \
  --include='/docs/' \
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

# Downstream steps (see docs/PUBLIC-RELEASE-PLAN.md):
#   1) drop tests that import scripts.legacy (listed below)
#   2) copy public scaffolding (README/LICENSE/CONTRIBUTING/SECURITY/CHANGELOG,
#      scrub_sensitive.py, check_public.py, config/public_boundary.json)
#   3) run scrub_sensitive.py, then check_public.py (tree and --history)
#   4) run the full test suite in the candidate tree
LEGACY_TESTS="test_dnn_continuation test_guarded_audit_cohorts test_guarded_history_rerun \
test_history_stage_scheduling test_history_training test_judge_dataset_rerun test_judge_temporal \
test_learning_material_repair test_lr_dataset_training test_lr_retrain \
test_pairwise_policy_comparison test_pairwise_tuning"
echo "exported to $DST"
echo "legacy-dependent tests to drop: $LEGACY_TESTS"
