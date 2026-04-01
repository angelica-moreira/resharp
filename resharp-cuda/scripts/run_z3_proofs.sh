#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# run_z3_proofs.sh — Z3 formal equivalence proofs
#
# Usage:  ./resharp-cuda/scripts/run_z3_proofs.sh
#
# Verifies mathematical correctness of the CUDA kernels using Z3:
#   1. Transition index equivalence (bit shift vs multiply)
#   2. Nullability mask invariants (BEGIN/CENTER/END/ALWAYS)
#   3. Forward scan logic self-consistency (S=4, M=3, N=2..4)
#   4. Chunk composition equivalence (parallel prefix correctness)
#
# Requires: python3 with z3-solver package
#   Install: pip install z3-solver
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Z3 Formal Equivalence Proofs"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Find python with z3
PYTHON=""
for py in python3.11 python3 python; do
    if command -v "$py" &>/dev/null; then
        if "$py" -c "import z3; print(f'Z3 version: {z3.get_version_string()}')" 2>/dev/null; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python with z3-solver not found."
    echo ""
    echo "Install with:"
    echo "  pip install z3-solver"
    echo "  # or"
    echo "  python3.11 -m pip install z3-solver"
    exit 1
fi

echo "Using: $PYTHON ($($PYTHON --version))"
echo ""

# Run proofs
$PYTHON resharp-cuda/scripts/z3_equivalence_proof.py

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Z3 proofs complete"
echo "═══════════════════════════════════════════════════════════════"
