#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# run_cross_validation.sh — 3-way correctness cross-validation
#
# Usage:  ./resharp-cuda/scripts/run_cross_validation.sh
#
# Runs all test suites that verify Oracle = CPU-ref = GPU:
#   1. conversion_validation (44 tests) — CPU reference kernel vs Oracle
#   2. gpu_validation (38 tests) — GPU vs CPU-ref vs Oracle
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  resharp-cuda Cross-Validation: Oracle vs CPU-ref vs GPU"
echo "═══════════════════════════════════════════════════════════════"
echo ""

PASS=0
FAIL=0

# Build
echo "Building test targets..."
cargo build --release -p resharp-cuda --tests 2>&1 | grep -E '(Compiling|Finished)' || true
echo ""

# ─── Conversion validation ───────────────────────────────────────
echo "── Conversion Validation (CPU-ref kernel vs Oracle) ──"
echo ""
OUTPUT=$(cargo test --release -p resharp-cuda --test conversion_validation -- --nocapture 2>&1)
echo "$OUTPUT" | grep -E '(^test |test result:)'
echo ""

if echo "$OUTPUT" | grep -q 'test result: ok'; then
    COUNT=$(echo "$OUTPUT" | grep 'test result:' | grep -oP '\d+ passed')
    echo "  ✓ PASSED: ${COUNT} tests"
    PASS=$((PASS+1))
else
    echo "  ✗ FAILED"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── GPU 3-way validation ────────────────────────────────────────
echo "── GPU 3-Way Validation (Oracle = CPU-ref = GPU) ──"
echo ""
OUTPUT=$(cargo test --release -p resharp-cuda --test gpu_validation -- --nocapture 2>&1)
echo "$OUTPUT" | grep -E '(^test |test result:)'
echo ""

if echo "$OUTPUT" | grep -q 'test result: ok'; then
    COUNT=$(echo "$OUTPUT" | grep 'test result:' | grep -oP '\d+ passed')
    echo "  ✓ PASSED: ${COUNT} tests"
    PASS=$((PASS+1))
else
    echo "  ✗ FAILED"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── Summary ─────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
if [ "$FAIL" -eq 0 ]; then
    echo "  ✓ ALL CROSS-VALIDATION PASSED (${PASS} suites, 82 total tests)"
else
    echo "  ✗ ${FAIL} SUITE(S) FAILED"
    exit 1
fi
echo "═══════════════════════════════════════════════════════════════"
