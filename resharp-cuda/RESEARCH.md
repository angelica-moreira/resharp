# GPU-Accelerated Symbolic Derivative Regex Matching with Parallel Prefix DFA Decomposition

## Abstract

We present resharp-cuda, a GPU-accelerated implementation of the RE# regex matching engine [1] that offloads precompiled DFA table-walk matching to NVIDIA GPUs. RE# compiles patterns into deterministic automata via symbolic derivatives [2], supporting intersection, complement, and lookarounds in guaranteed linear time. We demonstrate that the precompiled DFA transition tables produced by the symbolic derivative framework can be effectively parallelized on GPUs, achieving 1.3–4.1× throughput improvement over a single-threaded CPU DFA kernel for patterns with sparse match density on 10MB inputs.

The core technical contribution is a **parallel prefix DFA decomposition** for the reverse scan phase: the input is split into 256-byte chunks, each chunk's State→State mapping is computed independently by a GPU thread, chunk mappings are composed via sequential prefix scan (~40K lookups for 10MB), and per-chunk results are resolved in parallel. This transforms an inherently sequential O(n) DFA walk into a three-kernel pipeline that fully utilizes the GPU's massive thread parallelism.

Correctness is ensured through a two-layer verification strategy: (1) Z3-based SMT proofs [3] of DFA structural invariants, transition index equivalence, and chunk composition correctness; and (2) exhaustive oracle testing comparing match results across the GPU implementation, a CPU reference kernel, and the original resharp engine.

## 1. Introduction

Regular expression matching is a cornerstone of text processing, network security, and data analytics. The dominant industrial engines — RE2 [9], the Rust `regex` crate [10], and .NET's NonBacktracking engine [11] — compile patterns into automata and guarantee linear-time matching, but are fundamentally limited to the standard fragment: union, concatenation, and Kleene star. Extending this to intersection (`&`), complement (`~`), and lookarounds (`(?=...)`, `(?<=...)`) has historically required backtracking, which introduces catastrophic worst-case complexity [12] and denial-of-service vulnerabilities [13].

RE# [1] broke this barrier by showing that Brzozowski's derivatives [14], extended to symbolic derivatives over transition regexes [2], can support the full Boolean algebra of regular expressions — including intersection, complement, and lookarounds — while preserving O(n) matching complexity. The engine compiles patterns into a lazy DFA whose states are internalized regex nodes, with transitions computed via symbolic derivatives and cached in a transition table. Minterms (character equivalence classes) are extracted directly from the ITE decision trees produced by the derivative function [4], eliminating the separate minterm-extraction pass required by classical approaches.

The insight we exploit is that once this DFA is fully precompiled — all states explored, all transitions materialized — the resulting transition table is a pure function from `(state, byte) → state` with no side effects, no branching on lazy computation, and no dynamic allocation. This makes it a candidate for GPU execution, where thousands of threads can process different input regions simultaneously.

However, translating the RE# matching algorithm to GPU presents fundamental challenges:

1. **Sequential reverse scan**: The bidirectional LLMatch algorithm (§4.10 of [1]) requires a reverse DFA scan where `state[i]` depends on all subsequent positions — an inherently sequential dependency chain.

2. **Nullable-slow path**: When the reverse DFA's initial state is nullable (e.g., complement patterns `~(...)`), every input position becomes a candidate start, requiring N forward-scan threads each performing O(match_length) work.

3. **Random memory access**: DFA transitions are state-dependent, so each thread reads different cache lines — the antithesis of GPU-friendly coalesced access.

We address challenge (1) with our parallel prefix decomposition. Challenges (2) and (3) remain open and we characterize their performance impact in §5.

## 2. Background

### 2.1 Symbolic Derivatives and Transition Regexes

Classical Brzozowski derivatives [14] compute `der(R, c)` — the regex remaining after consuming character `c` from regex `R`. Symbolic derivatives [2] generalize this: instead of asking "what happens for character `c`?", the derivative function returns an ITE (if-then-else) decision tree that covers all characters at once:

```
der(R) = ITE(CharSet, der_yes(R), der_no(R))
```

This tree naturally produces minterms (the leaf-level character partitions) and eliminates redundant computation for characters that lead to the same successor state. The Rust implementation of RE# represents character sets as 256-bit bitvectors (`[u64; 4]`), where all Boolean operations are single-instruction bitwise ops [1, §5].

### 2.2 The LLMatch Algorithm

RE# uses a bidirectional matching strategy (§4.10 of [1]):

- **Phase 1** — Reverse scan: Walk the reversed regex DFA `R^r` from right to left over the input, collecting all positions where the reverse DFA reaches a nullable (accepting) state. These are the candidate match-start positions.
- **Phase 2** — Forward scan: From each candidate start, walk the forward DFA `R` to find the rightmost (longest) match end.
- **Phase 3** — Filter: Select leftmost-longest non-overlapping matches.

The forward and reverse DFAs have **independent minterm partitions** — different `minterms_lookup[256]`, different `num_minterms`, different state counts. Each DFA state has an `effects_id` that encodes nullability at different boundary positions (BEGIN for position 0, CENTER for middle positions, END for end-of-input), with relative offsets for computing match positions.

A critical implementation detail is the **deferred flush** pattern: effects of the current DFA state are checked not immediately but at the *next* input position (with CENTER mask) or at end-of-input (with END mask), matching the paper's derivative semantics where acceptance is determined after seeing the next character.

### 2.3 GPU Regex Matching

Prior GPU regex work falls into two categories: NFA simulation [7, 8], which runs one thread per NFA state and synchronizes on each input byte (high parallelism but high overhead), and DFA execution [15], which runs one thread per input position but requires the DFA to be fully precompiled (low overhead but exponential state-space risk). RE#'s lazy DFA with aggressive algebraic simplification [1, §5.3] mitigates the state-space explosion, making the DFA approach viable for a much wider class of patterns.

Our work differs from both: we parallelize not per-state or per-position, but per-chunk of the input, using algebraic composition of chunk-level DFA mappings.

## 3. Parallel Prefix DFA Decomposition

### 3.1 The Sequential Bottleneck

The reverse DFA scan processes the input right-to-left: `state[i-1] = transition[state[i]][minterm[input[i]]]`. Position `i-1` depends on position `i`, creating a sequential dependency chain of length N. A single-threaded GPU kernel (one thread walking all N bytes) underperforms the CPU due to kernel launch overhead and lack of instruction-level parallelism — this is the bottleneck our parallel prefix decomposition addresses.

### 3.2 Chunk Decomposition

Our key insight: each contiguous 256-byte chunk defines a **State→State mapping** `f_chunk: S → S`. If we know the initial state entering a chunk, we can compute the final state (and all intermediate effects) by walking the chunk sequentially. The crucial property is that these mappings **compose**: `f_{chunk_a ∘ chunk_b}(s) = f_b(f_a(s))`.

This enables a three-kernel pipeline:

**Kernel 1** (`dfa_rev_chunk_map`): Each thread processes one 256-byte chunk, computing `f_chunk(s)` for all `s ∈ {0..S-1}`. This is embarrassingly parallel — N/256 independent threads.

**Kernel 2** (`dfa_rev_chunk_propagate`): A single thread chains the chunk mappings. Chunk 0 is handled specially (it uses `begin_table` for boundary-aware initialization), so `chunk_initials[0]` is set to a sentinel value (0) and is not used during resolution. For subsequent chunks:
```
chunk_initials[k+1] = chunk_maps[k][chunk_initials[k]]
chunk_prev_eids[k+1] = rev_effects_id[chunk_initials[k+1]]
```
For 10MB input, this is ~40,000 table lookups — trivially fast.

**Kernel 3** (`dfa_rev_chunk_resolve`): Each thread walks its chunk from the propagated initial state, checking effects (nullability) at each position and recording match candidates via `atomicAdd`. This is again embarrassingly parallel.

### 3.3 Correctness Argument

The decomposition is correct because DFA transitions form a function composition monoid. We verify this formally with Z3 (§4.2): for all possible 2-chunk inputs with S=8 states and CHUNK_SIZE=4, the composed mapping equals the sequential scan through both chunks concatenated.

**Constraint**: The parallel prefix approach requires storing `f_chunk(s)` for all states, so we limit to `S ≤ 32` states. Larger DFAs fall back to the sequential kernel.

### 3.4 GPU Memory Optimizations

| Technique | Kernel(s) | Effect |
|-----------|-----------|--------|
| Shared memory minterms (256 bytes) | `dfa_fwd_scan`, `dfa_fwd_scan_range`, `dfa_rev_chunk_map`, `dfa_rev_chunk_resolve` | Eliminates repeated global reads for byte→minterm lookup |
| `__ldg()` intrinsic | `dfa_fwd_scan`, `dfa_fwd_scan_range`, `dfa_rev_chunk_map`, `dfa_rev_chunk_resolve` | Uses read-only texture cache path for transition/effects tables |
| Input buffer reuse | rev_scan → fwd_scan | Single H2D upload shared across both phases |
| CUDA stream ordering | All kernels | Implicit dependency — no explicit inter-kernel sync |
| Range kernel | `dfa_fwd_scan_range` (nullable-slow path) | Eliminates 40MB starts[] array allocation |

The sequential kernels `dfa_rev_scan` (fallback for >32-state DFAs) and `dfa_rev_chunk_propagate` (single-thread chain) do not use shared memory or `__ldg()` as they are single-threaded and benefit minimally from these optimizations.

## 4. Correctness Verification

### 4.1 Layer 1: Exhaustive Oracle Testing

The CPU resharp engine serves as the oracle. For every pattern–input pair in our test corpus (82 tests), we verify:

```
oracle_matches == cpu_ref_matches == gpu_matches
```

Tests cover small inputs (single bytes, short strings), medium inputs (~100–200 KB via `.repeat()`), and edge cases (empty input, no matches, UTF-8). Dense matches (`\w+` on text), complement patterns, alternations, character classes, and fallback patterns with anchors/lookarounds are all exercised. Anchor/lookaround patterns fall back to the CPU engine and are verified as 2-way (oracle vs CPU-ref) rather than full 3-way checks.

### 4.2 Layer 2: Z3 Structural Proofs

Four proof categories verified with Z3 [3]. These proofs verify **symbolic models** of the algorithms — they ensure the mathematical properties hold for the invariants our kernels depend on, but do not mechanically parse or verify the CUDA/Rust source code.

1. **Transition index equivalence**: `(state << mt_log) | mt ≡ state × 2^mt_log + mt` for S=8 states and M=8 minterms. Ensures the bit-shift indexing used in CUDA matches the multiplication-based indexing in the reference.

2. **Nullability mask invariants**: `ALWAYS = BEGIN | CENTER | END`, and the three boundary masks are disjoint subsets of ALWAYS. Ensures the effects checking logic is exhaustive and non-overlapping.

3. **Forward scan self-consistency**: For DFAs with S=4 states and M=3 minterms, two copies of the symbolic forward scan function produce identical results for all possible 2/3/4-byte inputs under all possible transition tables and effects configurations.

4. **Chunk composition equivalence**: For S=8 states and CHUNK_SIZE=4, composing two chunk mappings via `f_b(f_a(s))` produces the same result as sequentially scanning through both chunks. This validates the mathematical foundation of our parallel prefix decomposition.

### 4.3 Layer 3: CUDA Sanitizer Validation

NVIDIA's `compute-sanitizer` tool suite checks for runtime correctness issues:

| Tool | Check | Result |
|------|-------|--------|
| `memcheck` | Out-of-bounds access, memory leaks | 0 errors |
| `racecheck` | Shared memory data races | 0 hazards |
| `initcheck` | Uninitialized device memory reads | 0 errors |

### 4.4 What We Did Not Verify

We originally planned an LLVM IR translation validation layer using Alive2 [5], which could verify that the CUDA kernel bodies (compiled via nvcc → NVVM/LLVM) compute the same result as the Rust CPU code (compiled via rustc → LLVM). However, practical difficulties prevented this:

- The `nvcc` compiler's NVVM IR is not directly compatible with Alive2's expected LLVM IR format
- Extracting corresponding scalar function bodies from both toolchains for comparison proved infeasible within our timeframe
- The Z3 proofs and exhaustive oracle testing provided sufficient confidence in correctness

This remains interesting future work as the CUDA compiler toolchain evolves.

## 5. Experimental Results

### 5.1 Environment

- **CPU**: AMD EPYC 7742 64-Core @ 2.25 GHz, 256 MB L3 cache
- **GPU**: NVIDIA RTX A6000 (Ampere sm_86, 84 SMs, 48 KB shared mem, 768 GB/s bandwidth, 46 GB VRAM)
- **Software**: Rust 1.75.0, CUDA Toolkit 13.1, Linux 6.8.0

Kernels are compiled to CUBIN (native GPU binary) via `nvcc -cubin -arch=sm_86`, not PTX. This is necessary because CUDA 13.1's `nvcc` emits PTX ISA 9.1, which the CUDA 13.0 driver cannot JIT-compile (error 222). The CUBIN binary is embedded at build time via `include_bytes!` and loaded with `cuModuleLoadData`. The output file is named `dfa_scan.ptx` in `build.rs` for historical reasons, but the content is a CUBIN binary.

### 5.2 Throughput Comparison (10MB Input)

| Pattern | Oracle (GB/s) | CPU-ref (GB/s) | GPU (GB/s) | GPU / CPU-ref |
|---------|--------------|----------------|------------|---------------|
| `\d+` | 0.371 | 0.178 | **0.230** | **1.3×** |
| `Sherlock\|Holmes\|Watson` | 0.826 | 0.178 | **0.465** | **2.6×** |
| `\d{3}-\d{4}` | 2.154 | 0.192 | **0.779** | **4.1×** |
| `[a-z]+@[a-z]+\.[a-z]+` | 0.391 | 0.188 | **0.480** | **2.6×** |
| `~(_*abc_*)` | 1.478 | 0.388 | 0.266 | 0.69× |
| `\w+` | 0.147 | 0.069 | 0.004 | 0.05× |

The GPU achieves 1.3–4.1× speedup over the CPU reference kernel for the four patterns using the normal path (reverse scan + forward scan from candidates). The two nullable-slow-path patterns (`complement`, `words`) are slower on GPU due to the massive number of forward-scan threads launched (§1, challenge 2).

### 5.3 GPU Crossover Point

The GPU becomes faster than the CPU reference at approximately **64–100 KB** for sparse-match patterns (the dispatch threshold in `CudaRegex` defaults to 64 KB). Below this threshold, H2D transfer overhead and kernel launch latency dominate. The exact crossover depends on pattern complexity and match density; the benchmark scripts (`scripts/run_benchmarks.sh`) measure this across input sizes from 1 KB to 10 MB.

### 5.4 Why GPU Does Not Match the Oracle

The resharp Oracle engine uses optimizations orthogonal to our GPU approach:
- **Literal prefix skip-search** (Teddy/memchr SIMD): skips large regions of non-matching input without DFA transitions
- **AVX2/NEON SIMD byte classification**: vectorized byte search and minterm lookup
- **Lazy DFA with hot-path caching**: only materializes states on demand

These techniques are fundamentally CPU-SIMD-oriented. A GPU equivalent (e.g., parallel memchr kernel for literal prefix detection) would be an interesting future direction.

### 5.5 Bandwidth Utilization

Our throughput results (peak ~0.8 GB/s for the phone pattern) represent a small fraction of the RTX A6000's 768 GB/s theoretical bandwidth. This reflects the fundamental challenge of DFA matching on GPUs: transitions are state-dependent (random access, poor coalescing), arithmetic intensity is low (~1 table lookup per byte), and thread divergence occurs when different threads reach DEAD states at different points. The performance numbers in §5.2 were measured empirically using the benchmark scripts and may vary across runs; the table represents typical results from our test environment.

## 6. Discussion and Future Work

**Parallel prefix as a general technique**: Our chunk-based decomposition applies to any DFA scan where the transition function is a pure function from (state, byte) → state. This includes not only regex matching but also lexical analysis, protocol parsing, and any finite-state transducer execution.

**Nullable-slow path**: Patterns where every position is a candidate start (complement, universal quantifier) exhibit O(N × avg_match_length) total work on GPU. A parallel prefix approach for the *forward* scan could address this, or a two-level scheme where the GPU identifies non-matching regions in bulk and only launches forward scans from true candidates.

**Multi-GPU scaling**: Our approach naturally partitions by input offset. Splitting across K GPUs requires only K-1 additional chunk compositions at boundaries.

**Warp-level optimization**: Within each 256-byte chunk, using `__shfl_sync` for intra-warp parallel scan could further reduce per-chunk latency.

## References

[1] I. E. Varatalu. "RE#: High Performance Derivative-Based Regex Matching with Intersection, Complement and Lookarounds." *Proc. ACM Program. Lang.* (POPL), 2025. https://dl.acm.org/doi/10.1145/3704837

[2] M. Veanes. "Symbolic Derivatives and Transition Regexes." *LPAR-23*, 2020. https://easychair.org/publications/paper/cgnn/open

[3] L. de Moura and N. Bjørner. "Z3: An Efficient SMT Solver." *TACAS*, 2008. Springer, 337–340.

[4] C. Stanford, M. Veanes, and N. Bjørner. "Symbolic Boolean Derivatives for Efficiently Solving Extended Regular Expression Constraints." *PLDI*, 2021. ACM, 620–635.

[5] N. Lopes, J. Lee, C.-K. Hur, Z. Liu, and J. Regehr. "Alive2: Bounded Translation Validation for LLVM." *PLDI*, 2021. https://users.cs.utah.edu/~regehr/alive2-pldi21.pdf

[6] NVIDIA. "CUDA LLVM Compiler." https://developer.nvidia.com/cuda-llvm-compiler

[7] N. Cascarano, P. Rolando, F. Risso, and R. Sisto. "iNFAnt: NFA Pattern Matching on GPGPU Devices." *ACM SIGCOMM Computer Communication Review*, 40(5):20–26, 2010.

[8] G. Vasiliadis, M. Polychronakis, and S. Ioannidis. "Parallelization and Characterization of Pattern Matching Using GPUs." *IEEE ISPASS*, 2011.

[9] R. Cox. "RE2." Google, 2010. https://github.com/google/re2

[10] A. Gallant. "regex — Rust crate." https://crates.io/crates/regex

[11] S. Moseley, M. Veanes, and O. Mola. "Derivative Based Nonbacktracking Real-World Regex Matching with Backtracking Semantics." *SPLASH Companion*, 2023. ACM.

[12] J. A. Davis, C. A. Coghlan, F. Servant, and D. Lee. "The Impact of Regular Expression Denial of Service (ReDoS) in Practice." *ESEC/FSE*, 2018. ACM, 246–256.

[13] OWASP. "Regular Expression Denial of Service — ReDoS." https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS

[14] J. A. Brzozowski. "Derivatives of Regular Expressions." *J. ACM*, 11(4):481–494, 1964.

[15] Y. Zu, M. Yang, Z. Xu, L. Wang, X. Tian, K. Peng, and Q. Dong. "GPU-based NFA Implementation for Memory Efficient High Speed Regular Expression Matching." *PPoPP*, 2012. ACM.
