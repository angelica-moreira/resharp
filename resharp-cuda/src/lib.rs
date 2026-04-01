//! # resharp-cuda: CUDA-accelerated DFA matching for resharp
//!
//! This crate provides GPU acceleration for the [resharp](https://crates.io/crates/resharp) symbolic
//! derivative-based regex engine (RE#). It offloads precompiled DFA table-walk matching to NVIDIA GPUs
//! while keeping pattern compilation on the CPU.
//!
//! ## Algorithm
//!
//! Implements the two-phase LLMatch algorithm from the RE# paper (Section 4.10):
//!
//! 1. **Reverse scan** (`AllEnds(s^r[0], R^r, ∅)`): Walk the reversed DFA backward over the input
//!    to find all potential match-start positions. Uses a parallel prefix decomposition for
//!    throughput: input is split into 256-byte chunks, each processed independently, then
//!    composed via prefix scan to resolve actual DFA states.
//!
//! 2. **Forward scan** (`MaxEnd(s[i], E·(?=A))`): From each candidate start, walk the forward DFA
//!    to find the rightmost (longest) match end. Each candidate is processed by a separate GPU thread.
//!
//! 3. **Non-overlapping filter**: Select leftmost-longest non-overlapping matches on the CPU.
//!
//! ## Dispatch Strategy
//!
//! - Input ≥ `gpu_threshold` (default 64KB) + GPU available → **GPU kernels**
//! - Input < threshold or GPU unavailable → **CPU reference DFA kernel** ([`kernel`] module)
//! - Patterns with anchors/lookarounds → **CPU engine** (precompiled tables can't represent these)
//!
//! ## Example
//!
//! ```no_run
//! use resharp_cuda::CudaRegex;
//!
//! let re = CudaRegex::new(r"\d+").unwrap();
//! let input = b"abc 123 def 456";
//! let matches = re.find_all(input).unwrap();
//! assert_eq!(matches.len(), 2);
//! ```

pub mod cuda_driver;
pub mod kernel;

use resharp::{DfaTables, EngineOptions, Error, Match};

/// Minimum input size (bytes) to justify GPU transfer overhead.
/// Below this threshold, the CPU reference kernel is faster due to
/// zero transfer latency.
pub const GPU_MIN_INPUT_SIZE: usize = 64 * 1024;

/// CUDA-accelerated regex matcher.
///
/// Wraps a `resharp::Regex` with optional GPU acceleration via precompiled DFA tables.
/// Automatically dispatches between GPU kernels, CPU reference kernel, and the full
/// resharp engine based on input size and pattern complexity.
///
/// # Dispatch Priority
///
/// 1. GPU kernels (when input ≥ threshold and pattern has no anchors/lookarounds)
/// 2. CPU reference DFA kernel (for small inputs with precompiled tables)
/// 3. Full resharp engine (for patterns with anchors, lookarounds, or when extraction fails)
pub struct CudaRegex {
    cpu_regex: resharp::Regex,
    dfa: Option<DfaTables>,
    gpu_ctx: Option<cuda_driver::GpuContext>,
    /// True when the precompiled DFA tables can handle this pattern
    /// (no anchors or lookarounds that require the lazy DFA).
    can_use_precompiled: bool,
    pub gpu_threshold: usize,
}

impl CudaRegex {
    /// Compile a pattern for CUDA-accelerated matching.
    pub fn new(pattern: &str) -> Result<Self, Error> {
        Self::with_options(pattern, EngineOptions::default(), 2048)
    }

    /// Compile with custom options and precompilation threshold.
    pub fn with_options(
        pattern: &str,
        mut opts: EngineOptions,
        dfa_threshold: usize,
    ) -> Result<Self, Error> {
        opts.dfa_threshold = dfa_threshold.max(256);
        let cpu_regex = resharp::Regex::with_options(pattern, opts)?;
        let dfa = cpu_regex.extract_dfa_tables();

        // Precompiled bidirectional DFA only works for patterns without
        // anchors or lookarounds — those need the lazy DFA's context-dependent
        // derivative computation which can't be captured in static tables.
        let can_use_precompiled = dfa.as_ref().map_or(false, |d| !d.has_look && !d.has_anchors);

        // Only init GPU for patterns we can actually accelerate
        let gpu_ctx = if can_use_precompiled {
            dfa.as_ref().and_then(|d| {
                match cuda_driver::GpuContext::new(d) {
                    Ok(ctx) => Some(ctx),
                    Err(e) => {
                        eprintln!("[resharp-cuda] GPU init failed: {}", e);
                        None
                    }
                }
            })
        } else {
            None
        };

        Ok(CudaRegex {
            cpu_regex,
            dfa,
            gpu_ctx,
            can_use_precompiled,
            gpu_threshold: GPU_MIN_INPUT_SIZE,
        })
    }

    /// Set the minimum input size for GPU acceleration.
    pub fn set_gpu_threshold(&mut self, bytes: usize) {
        self.gpu_threshold = bytes;
    }

    /// Whether a precompiled GPU DFA is available.
    pub fn has_gpu_dfa(&self) -> bool { self.dfa.is_some() }

    /// Whether GPU context is initialized and ready.
    pub fn has_gpu(&self) -> bool { self.gpu_ctx.is_some() }

    /// Get the extracted DFA tables (for inspection/testing).
    pub fn dfa_tables(&self) -> Option<&DfaTables> { self.dfa.as_ref() }

    /// All non-overlapping leftmost-longest matches (GPU if available + above threshold).
    pub fn find_all(&self, input: &[u8]) -> Result<Vec<Match>, Error> {
        // Fall back to CPU engine for patterns with anchors/lookarounds
        if !self.can_use_precompiled {
            return self.cpu_regex.find_all(input);
        }
        if input.len() >= self.gpu_threshold {
            if let Some(ref ctx) = self.gpu_ctx {
                if let Some(ref dfa) = self.dfa {
                    return Ok(ctx.find_all(dfa, input));
                }
            }
        }
        if let Some(ref dfa) = self.dfa {
            return Ok(kernel::cpu_find_all(dfa, input));
        }
        self.cpu_regex.find_all(input)
    }

    /// Whether the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> Result<bool, Error> {
        if !self.can_use_precompiled {
            return self.cpu_regex.is_match(input);
        }
        if input.len() >= self.gpu_threshold {
            if let Some(ref ctx) = self.gpu_ctx {
                if let Some(ref dfa) = self.dfa {
                    return Ok(ctx.is_match(dfa, input));
                }
            }
        }
        if let Some(ref dfa) = self.dfa {
            return Ok(kernel::cpu_is_match(dfa, input));
        }
        self.cpu_regex.is_match(input)
    }

    /// Find the anchored match starting at position 0.
    pub fn find_anchored(&self, input: &[u8]) -> Result<Option<Match>, Error> {
        if !self.can_use_precompiled {
            return self.cpu_regex.find_anchored(input);
        }
        if input.len() >= self.gpu_threshold {
            if let Some(ref ctx) = self.gpu_ctx {
                if let Some(ref dfa) = self.dfa {
                    return Ok(ctx.find_anchored(dfa, input));
                }
            }
        }
        if let Some(ref dfa) = self.dfa {
            return Ok(kernel::cpu_find_anchored(dfa, input));
        }
        self.cpu_regex.find_anchored(input)
    }

    /// Force CPU-only DFA path (for benchmarking).
    pub fn find_all_cpu(&self, input: &[u8]) -> Result<Vec<Match>, Error> {
        if !self.can_use_precompiled {
            return self.cpu_regex.find_all(input);
        }
        if let Some(ref dfa) = self.dfa {
            Ok(kernel::cpu_find_all(dfa, input))
        } else {
            self.cpu_regex.find_all(input)
        }
    }

    /// Force GPU path (for benchmarking). Returns None if GPU unavailable.
    pub fn find_all_gpu(&self, input: &[u8]) -> Option<Vec<Match>> {
        let ctx = self.gpu_ctx.as_ref()?;
        let dfa = self.dfa.as_ref()?;
        Some(ctx.find_all(dfa, input))
    }

    /// Get the CPU regex (for comparison).
    pub fn cpu_regex(&self) -> &resharp::Regex { &self.cpu_regex }
}

/// Verify structural invariants on extracted DFA tables.
pub fn verify_dfa_invariants(dfa: &DfaTables) -> Vec<String> {
    let mut errors = Vec::new();
    for b in 0..256usize {
        if dfa.minterms_lookup[b] as u32 >= dfa.num_minterms {
            errors.push(format!("INV-1: byte {} → mt {} >= {}", b, dfa.minterms_lookup[b], dfa.num_minterms));
        }
    }
    if dfa.fwd_begin_table.len() < dfa.num_minterms as usize {
        errors.push(format!("INV-2: fwd_begin {} < {}", dfa.fwd_begin_table.len(), dfa.num_minterms));
    }
    let stride = 1usize << dfa.mt_log;
    for sid in 2..dfa.fwd_num_states {
        for mt in 0..dfa.num_minterms as usize {
            let idx = sid * stride + mt;
            if idx < dfa.fwd_center_table.len() {
                let next = dfa.fwd_center_table[idx] as usize;
                if next >= dfa.fwd_num_states && next > 1 {
                    errors.push(format!("INV-3: fwd {}×{} → {} OOB", sid, mt, next));
                }
            }
        }
    }
    let mut counts = vec![0u32; dfa.num_minterms as usize];
    for b in 0..256 { let mt = dfa.minterms_lookup[b] as usize; if mt < counts.len() { counts[mt] += 1; } }
    let total: u32 = counts.iter().sum();
    if total != 256 { errors.push(format!("INV-7: minterms cover {} != 256", total)); }
    errors
}
