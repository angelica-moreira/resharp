#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# run_benchmarks.sh — Performance comparison: Oracle vs CPU-ref vs GPU
#
# Usage:  ./resharp-cuda/scripts/run_benchmarks.sh
#
# Runs the profiling benchmark and produces:
#   - Terminal output with throughput charts
#   - profile_results.csv with raw data
#   - Correctness check on every measurement (3-way validation)
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  resharp-cuda Benchmark: Oracle vs CPU-ref vs GPU"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check environment
echo "Environment:"
echo "  Rust: $(rustc --version 2>/dev/null || echo 'not found')"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'not found')"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo "not detected")
echo "  GPU:  $GPU_NAME"
echo ""

# Build
echo "Building (release)..."
cargo build --release --example profile -p resharp-cuda 2>&1 | grep -E '(Compiling|Finished)' || true
echo ""

# Run benchmark
echo "Running benchmark (1KB, 10KB, 100KB, 1MB, 10MB × 6 patterns)..."
echo "Each size runs 3–50 iterations with 3-way correctness validation."
echo ""

target/release/examples/profile

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results saved to: resharp-cuda/results/profile_results.csv"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Print CSV summary table
CSV="resharp-cuda/results/profile_results.csv"
if [ -f "$CSV" ]; then
    echo "CSV Summary (10MB results):"
    echo ""
    printf "  %-12s %-8s %-8s %-8s %-10s %-8s\n" "Pattern" "Oracle" "CPU-ref" "GPU" "GPU/CPUref" "Correct"
    printf "  %-12s %-8s %-8s %-8s %-10s %-8s\n" "───────" "──────" "──────" "──────" "────────" "──────"
    grep ',10000,' "$CSV" | while IFS=, read -r label pat kb matches oracle_us cpu_us gpu_us oracle_gbs cpu_gbs gpu_gbs correct; do
        if [ "$cpu_gbs" != "0" ] && [ "$cpu_gbs" != "0.0000" ]; then
            ratio=$(echo "scale=1; $gpu_gbs / $cpu_gbs" | bc 2>/dev/null || echo "?")
        else
            ratio="N/A"
        fi
        printf "  %-12s %-8s %-8s %-8s %-10s %-8s\n" "$label" "$oracle_gbs" "$cpu_gbs" "$gpu_gbs" "${ratio}×" "$correct"
    done
    echo ""
fi
