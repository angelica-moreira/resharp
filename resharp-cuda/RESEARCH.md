# GPU-Accelerated Symbolic Derivative Regex Matching with SMT-Verified DFA Translation

## Abstract

We present a methodology for translating high-performance derivative-based regex matching engines from CPU SIMD architectures to massively parallel GPU execution, with formal verification of the translation's correctness using SMT solvers. Building on RE# [1] — a state-of-the-art regex engine that compiles patterns into deterministic automata via symbolic derivatives [2], supporting intersection, complement, and lookarounds in guaranteed linear time — we demonstrate that the precompiled DFA transition tables produced by the symbolic derivative framework are naturally suited to GPU parallelism: each input position can independently walk the transition table with zero thread divergence for fully-compiled automata. The core contribution is a three-layer verification strategy that ensures semantic equivalence between the CPU and GPU matching paths: (1) Z3-based SMT checking [3] of DFA structural invariants — minterm partitioning completeness, transition table totality, nullability propagation consistency, and index arithmetic safety — extracted directly from the symbolic algebra that RE# shares with Z3's own sequence theory [4]; (2) LLVM IR refinement checking via Alive2 [5] for the scalar kernel bodies (transition lookup, delta calculation) where both Rust and CUDA compile through LLVM [6]; and (3) exhaustive functional oracle testing comparing match results across thousands of pattern–input pairs. This approach addresses a fundamental gap in GPU regex literature: existing GPU regex engines (e.g., iNFAnt [7], GPU-NFA simulators [8]) sacrifice the algebraic structure that makes derivative-based engines fast, while our translation preserves the complete DFA — including the bidirectional scan algorithm, lookaround annotations, and skip-acceleration metadata — enabling the GPU to execute the same O(n) matching algorithm at throughputs proportional to GPU memory bandwidth rather than CPU cache bandwidth.

## 1. Introduction

Regular expression matching is a cornerstone of text processing, network security, and data analytics. The dominant industrial engines — RE2 [9], the Rust `regex` crate [10], and .NET's NonBacktracking engine [11] — compile patterns into automata and guarantee linear-time matching, but are fundamentally limited to the standard fragment: union, concatenation, and Kleene star. Extending this to intersection (`&`), complement (`~`), and lookarounds (`(?=...)`, `(?<=...)`) has historically required backtracking, which introduces catastrophic worst-case complexity [12] and denial-of-service vulnerabilities [13].

RE# [1] broke this barrier by showing that Brzozowski's derivatives [14], extended to symbolic derivatives over transition regexes [2], can support the full Boolean algebra of regular expressions — including intersection, complement, and lookarounds — while preserving O(n) matching complexity. The engine compiles patterns into a lazy DFA whose states are internalized regex nodes, with transitions computed via symbolic derivatives and cached in a transition table. Minterms (character equivalence classes) are extracted directly from the ITE decision trees produced by the derivative function [4], eliminating the separate minterm-extraction pass required by classical approaches.

The insight we exploit is that once this DFA is fully precompiled — all states explored, all transitions materialized — the resulting transition table is a pure function from `(state, byte) → state` with no side effects, no branching on lazy computation, and no dynamic allocation. This makes it an ideal candidate for GPU execution, where thousands of threads can independently walk the table for different input positions simultaneously.

However, translating a CPU regex engine to GPU is not merely a matter of copying the transition table to device memory. The bidirectional matching algorithm — reverse scan to find match-start candidates, then forward scan to find the leftmost-longest match-end [1, §4.9] — must be faithfully reproduced. The nullability annotations on DFA states, which encode lookaround satisfaction and anchor semantics via the location-based derivative framework [11], must be correctly propagated. The minterm lookup, index arithmetic (`state << mt_log | minterm`), and effects checking must be bit-identical between CPU and GPU paths.

To guarantee this fidelity, we propose a three-layer verification approach:

**Layer 1: SMT-based structural verification.** We use Z3 [3] to check that the extracted DFA transition tables satisfy the algebraic invariants required by the matching algorithm. These invariants — minterm partitioning (the 256 byte values are partitioned into disjoint equivalence classes), transition totality (every `(state, minterm)` pair has a defined successor), and nullability consistency (the effects annotations correctly reflect the derivative-based nullable conditions) — are naturally expressible as SMT constraints over bitvectors. Crucially, RE# already shares its algebraic foundation with Z3's sequence theory [4], which uses symbolic derivatives for string constraint solving; our verification exploits this shared heritage.

**Layer 2: LLVM IR translation validation.** Both the Rust CPU code (via `rustc` → LLVM) and the CUDA kernel code (via `nvcc` → NVVM, which is LLVM-based [6]) produce LLVM IR. For the scalar inner-loop bodies — the transition lookup, the delta index calculation `(state << mt_log | minterm)`, and the nullability check — we can extract corresponding LLVM IR functions and verify refinement using Alive2 [5], which proves that the target IR computes the same result as the source for all inputs. This layer catches arithmetic bugs, off-by-one errors, and type-width mismatches that would be invisible to functional testing.

**Layer 3: Exhaustive oracle testing.** The CPU RE# engine serves as the oracle. For every pattern–input pair in our test corpus, we verify that the GPU path produces identical `Vec<Match>` results. This layer catches algorithmic errors in the bidirectional scan, incorrect handling of edge cases (empty input, anchors, overlapping matches), and GPU-specific issues (thread synchronization, memory coalescing artifacts).

## 2. Background

### 2.1 Symbolic Derivatives and Transition Regexes

Classical Brzozowski derivatives [14] compute `der(R, c)` — the regex remaining after consuming character `c` from regex `R`. This requires computing one derivative per character in the alphabet. Symbolic derivatives [2] generalize this: instead of asking "what happens for character `c`?", the derivative function returns an ITE (if-then-else) decision tree that covers all characters at once:

```
der(R) = ITE(CharSet, der_yes(R), der_no(R))
```

This tree naturally produces minterms (the leaf-level character partitions) and eliminates redundant computation for characters that lead to the same successor state. The Rust implementation of RE# represents character sets as 256-bit bitvectors (`[u64; 4]`), where all Boolean operations are single-instruction bitwise ops [1, §5].

### 2.2 GPU Regex Matching

Prior GPU regex work falls into two categories: NFA simulation [7, 8], which runs one thread per NFA state and synchronizes on each input byte (high parallelism but high overhead), and DFA execution [15], which runs one thread per input position but requires the DFA to be fully precompiled (low overhead but exponential state-space risk). RE#'s lazy DFA with aggressive algebraic simplification [1, §5.3] mitigates the state-space explosion, making the DFA approach viable for a much wider class of patterns.

### 2.3 Translation Validation

Alive2 [5] proves that LLVM IR transformations preserve semantics by encoding both source and target as SMT formulas and checking refinement. It has found hundreds of bugs in LLVM's optimization passes. We extend this idea beyond compiler-internal transformations to cross-architecture translation: verifying that hand-written CUDA kernels correctly implement the same computation as Rust CPU code.

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

