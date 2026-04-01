#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# run_cuda_sanitizers.sh — GPU memory/race/init correctness checks
#
# Usage:  ./resharp-cuda/scripts/run_cuda_sanitizers.sh
#
# Runs NVIDIA compute-sanitizer with three tools:
#   1. memcheck   — out-of-bounds access, memory leaks
#   2. racecheck  — shared memory race conditions
#   3. initcheck  — uninitialized device memory reads
#
# Requires: compute-sanitizer (ships with CUDA toolkit)
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  CUDA Sanitizer Checks"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Find compute-sanitizer
SANITIZER="/usr/local/cuda/bin/compute-sanitizer"
if [ ! -x "$SANITIZER" ]; then
    SANITIZER=$(which compute-sanitizer 2>/dev/null || true)
fi
if [ -z "$SANITIZER" ] || [ ! -x "$SANITIZER" ]; then
    echo "ERROR: compute-sanitizer not found."
    echo "Install CUDA toolkit or set PATH to include /usr/local/cuda/bin"
    exit 1
fi
echo "Using: $SANITIZER"
echo "Version: $($SANITIZER --version 2>&1 | head -1)"
echo ""

# Build
echo "Building gpu_sanitizer example..."
cargo build --release --example gpu_sanitizer -p resharp-cuda 2>&1 | grep -E '(Compiling|Finished)' || true
BINARY="target/release/examples/gpu_sanitizer"
echo ""

PASS=0
FAIL=0

# ─── memcheck ────────────────────────────────────────────────────
echo "── memcheck (out-of-bounds, leaks) ──"
OUTPUT=$($SANITIZER --tool memcheck "$BINARY" 2>&1)
SUMMARY=$(echo "$OUTPUT" | grep 'ERROR SUMMARY')
echo "  $SUMMARY"
if echo "$SUMMARY" | grep -q '0 errors'; then
    echo "  ✓ PASSED"
    PASS=$((PASS+1))
else
    echo "  ✗ FAILED"
    echo "$OUTPUT" | grep -i 'error' | head -10
    FAIL=$((FAIL+1))
fi
echo ""

# ─── racecheck ───────────────────────────────────────────────────
echo "── racecheck (shared memory races) ──"
OUTPUT=$($SANITIZER --tool racecheck "$BINARY" 2>&1)
SUMMARY=$(echo "$OUTPUT" | grep 'RACECHECK SUMMARY')
echo "  $SUMMARY"
if echo "$SUMMARY" | grep -q '0 hazards'; then
    echo "  ✓ PASSED"
    PASS=$((PASS+1))
else
    echo "  ✗ FAILED"
    echo "$OUTPUT" | grep -i 'hazard' | head -10
    FAIL=$((FAIL+1))
fi
echo ""

# ─── initcheck ───────────────────────────────────────────────────
echo "── initcheck (uninitialized memory reads) ──"
OUTPUT=$($SANITIZER --tool initcheck "$BINARY" 2>&1)
SUMMARY=$(echo "$OUTPUT" | grep 'ERROR SUMMARY')
echo "  $SUMMARY"
if echo "$SUMMARY" | grep -q '0 errors'; then
    echo "  ✓ PASSED"
    PASS=$((PASS+1))
else
    echo "  ✗ FAILED"
    echo "$OUTPUT" | grep -i 'error' | head -10
    FAIL=$((FAIL+1))
fi
echo ""

# ─── Summary ─────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
if [ "$FAIL" -eq 0 ]; then
    echo "  ✓ ALL SANITIZER CHECKS PASSED (${PASS}/3)"
else
    echo "  ✗ ${FAIL}/3 SANITIZER CHECKS FAILED"
    exit 1
fi
echo "═══════════════════════════════════════════════════════════════"
