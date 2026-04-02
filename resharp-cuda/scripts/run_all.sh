#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# run_all.sh — Master script: build, validate, benchmark, report
#
# Usage:  ./resharp-cuda/scripts/run_all.sh
#
# Runs every validation and benchmark step in order:
#   1. Build all targets (release)
#   2. Run upstream resharp tests (327 tests)
#   3. Run resharp-cuda conversion tests (44 tests)
#   4. Run resharp-cuda GPU 3-way cross-validation (38 tests)
#   5. Run CUDA sanitizer checks (memcheck, racecheck, initcheck)
#   6. Run Z3 equivalence proofs
#   7. Run full benchmark suite (profile.rs)
#   8. Print summary
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0
RESULTS=()

log()   { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()    { echo -e "  ${GREEN}✓${NC} $*"; PASS=$((PASS+1)); RESULTS+=("✓ $*"); }
fail()  { echo -e "  ${RED}✗${NC} $*"; FAIL=$((FAIL+1)); RESULTS+=("✗ $*"); }
header(){ echo -e "\n${BOLD}═══ $* ═══${NC}"; }

# ─── 1. Build ────────────────────────────────────────────────────
header "Step 1: Building all targets (release)"
if cargo build --release -p resharp-cuda --examples --tests 2>&1 | tail -3; then
    ok "Build succeeded"
else
    fail "Build FAILED"
    echo "Cannot continue without a successful build."
    exit 1
fi

# ─── 2. Upstream resharp tests ───────────────────────────────────
header "Step 2: Upstream resharp tests"
UPSTREAM=$(cargo test --release -p resharp 2>&1 | grep 'test result:' | grep -oP '\d+ passed' | awk '{s+=$1} END{print s}')
if cargo test --release -p resharp 2>&1 | grep -q 'FAILED'; then
    fail "resharp upstream: FAILURES detected"
else
    ok "resharp upstream: ${UPSTREAM:-all} tests passed"
fi

# ─── 3. Conversion validation ────────────────────────────────────
header "Step 3: Conversion validation (CPU reference kernel)"
OUTPUT=$(cargo test --release -p resharp-cuda --test conversion_validation 2>&1)
COUNT=$(echo "$OUTPUT" | grep 'test result:' | grep -oP '\d+ passed')
if echo "$OUTPUT" | grep -q 'FAILED'; then
    fail "Conversion validation: FAILURES"
    echo "$OUTPUT" | grep 'FAILED'
else
    ok "Conversion validation: ${COUNT} tests passed"
fi

# ─── 4. GPU 3-way cross-validation ──────────────────────────────
header "Step 4: GPU 3-way cross-validation (oracle = cpu_ref = gpu)"
OUTPUT=$(cargo test --release -p resharp-cuda --test gpu_validation 2>&1)
COUNT=$(echo "$OUTPUT" | grep 'test result:' | grep -oP '\d+ passed')
if echo "$OUTPUT" | grep -q 'FAILED'; then
    fail "GPU validation: FAILURES"
    echo "$OUTPUT" | grep 'FAILED'
else
    ok "GPU validation: ${COUNT} tests passed"
fi

# ─── 5. CUDA sanitizers ─────────────────────────────────────────
header "Step 5: CUDA sanitizer checks"

SANITIZER="/usr/local/cuda/bin/compute-sanitizer"
if [ ! -x "$SANITIZER" ]; then
    SANITIZER=$(which compute-sanitizer 2>/dev/null || true)
fi

if [ -n "$SANITIZER" ] && [ -x "$SANITIZER" ]; then
    BINARY="target/release/examples/gpu_sanitizer"

    # memcheck
    OUTPUT=$($SANITIZER --tool memcheck "$BINARY" 2>&1)
    ERRORS=$(echo "$OUTPUT" | grep 'ERROR SUMMARY' | grep -oP '\d+ errors' | head -1)
    if echo "$ERRORS" | grep -q '^0 errors'; then
        ok "CUDA memcheck: 0 errors"
    else
        fail "CUDA memcheck: $ERRORS"
    fi

    # racecheck
    OUTPUT=$($SANITIZER --tool racecheck "$BINARY" 2>&1)
    HAZARDS=$(echo "$OUTPUT" | grep 'RACECHECK SUMMARY' | grep -oP '\d+ hazards' | head -1)
    if echo "$HAZARDS" | grep -q '^0 hazards'; then
        ok "CUDA racecheck: 0 hazards"
    else
        fail "CUDA racecheck: $HAZARDS"
    fi

    # initcheck
    OUTPUT=$($SANITIZER --tool initcheck "$BINARY" 2>&1)
    ERRORS=$(echo "$OUTPUT" | grep 'ERROR SUMMARY' | grep -oP '\d+ errors' | head -1)
    if echo "$ERRORS" | grep -q '^0 errors'; then
        ok "CUDA initcheck: 0 errors"
    else
        fail "CUDA initcheck: $ERRORS"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} compute-sanitizer not found, skipping"
    RESULTS+=("⚠ CUDA sanitizers: skipped (not found)")
fi

# ─── 6. Z3 equivalence proofs ───────────────────────────────────
header "Step 6: Z3 formal equivalence proofs"

PYTHON=""
for py in python3.11 python3 python; do
    if command -v "$py" &>/dev/null; then
        if "$py" -c "import z3" 2>/dev/null; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -n "$PYTHON" ]; then
    OUTPUT=$($PYTHON resharp-cuda/scripts/z3_equivalence_proof.py 2>&1)
    PROVED=$(echo "$OUTPUT" | grep -cE '✅|✓' || true)
    FAILED_PROOFS=$(echo "$OUTPUT" | grep -cE '❌|✗' || true)
    if [ "$FAILED_PROOFS" -eq 0 ] && [ "$PROVED" -gt 0 ]; then
        ok "Z3 proofs: ${PROVED} categories proved"
    else
        fail "Z3 proofs: ${FAILED_PROOFS} failures"
        echo "$OUTPUT" | grep '✗'
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Python with z3-solver not found, skipping"
    echo "    Install: pip install z3-solver (requires python3.11+)"
    RESULTS+=("⚠ Z3 proofs: skipped (z3-solver not found)")
fi

# ─── 7. Full benchmark ──────────────────────────────────────────
header "Step 7: Full benchmark (profile.rs)"
echo "  Running Oracle vs CPU-ref vs GPU across 1KB–10MB..."
echo ""
target/release/examples/profile
ok "Benchmark completed → resharp-cuda/results/profile_results.csv"

# ─── 8. Summary ──────────────────────────────────────────────────
header "Summary"
echo ""
for r in "${RESULTS[@]}"; do
    echo -e "  $r"
done
echo ""
echo -e "  ${GREEN}Passed: ${PASS}${NC}  ${RED}Failed: ${FAIL}${NC}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}ALL CHECKS PASSED${NC}"
else
    echo -e "  ${RED}${BOLD}${FAIL} CHECK(S) FAILED${NC}"
    exit 1
fi
