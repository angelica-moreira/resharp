// resharp-cuda: DFA scan kernels for GPU-accelerated regex matching.
//
// Target: NVIDIA RTX A6000 (sm_86), 84 SMs, 48KB shared mem, 768 GB/s BW.
//
// Kernels (sequential):
//   dfa_rev_scan  — sequential reverse DFA walk (single thread), collects match-start candidates
//   dfa_fwd_scan  — parallel forward DFA walk (one thread per candidate), finds match-ends
//
// Kernels (parallel prefix — 3-kernel pipeline for reverse scan):
//   dfa_rev_chunk_map       — per-chunk State→State mapping computation
//   dfa_rev_chunk_propagate — sequential inter-chunk prefix propagation
//   dfa_rev_chunk_resolve   — per-chunk state resolution + effects checking
//
// The transition table lookup is: next = center_table[(state << mt_log) | mt]
// where mt = minterms[byte]. DFA_DEAD = 1, DFA_MISSING = 0.
//
// Effects system: effects_id[state] → index into effects_offsets/effects_flat.
//   effects_flat[i] = (mask << 16) | rel
//   mask bits: CENTER=0b001, BEGIN=0b010, END=0b100, ALWAYS=0b111
//   eid==0: not nullable, eid==1: always nullable (special), eid>=2: lookup effects table

#include <stdint.h>

#define DFA_DEAD   1u
#define NULL_CENTER 0x01u
#define NULL_BEGIN  0x02u
#define NULL_END    0x04u
#define NULL_ALWAYS 0x07u

// Parallel prefix constants
#define CHUNK_SIZE   256u
#define PAR_MAX_STATES 32u

// ============================================================
// Reverse DFA scan: sequential walk from input[start_pos] to input[0].
// Collects match-start candidates into hits[] via deferred flush pattern.
//
// This is a single-thread kernel — the reverse scan is inherently sequential
// because state at position i depends on all positions after i.
// (Phase 2 parallel prefix scan will parallelize this.)
// ============================================================
extern "C"
__global__ void dfa_rev_scan(
    const uint16_t* __restrict__ rev_center,       // reverse center table
    const uint16_t* __restrict__ rev_begin,        // reverse begin table
    const uint16_t* __restrict__ rev_effects_id,   // per-state effect ID
    const uint32_t* __restrict__ rev_effects_flat,  // packed (mask<<16)|rel
    const uint32_t* __restrict__ rev_effects_offsets, // offsets into effects_flat
    uint32_t rev_num_effects,                      // length of effects_offsets
    const uint8_t*  __restrict__ rev_minterms,     // reverse byte→minterm (256 entries)
    uint32_t rev_mt_log,                           // reverse log2(minterms) for shift
    const uint8_t*  __restrict__ input,
    uint32_t input_len,
    uint32_t start_pos,                            // where to start (typically input_len-1)
    uint32_t* __restrict__ hits,                   // output: match-start positions
    uint32_t* __restrict__ hit_count,              // output: number of hits
    uint32_t max_hits
) {
    // Single thread kernel
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (input_len == 0 || start_pos >= input_len) return;

    // Initialize: read byte at start_pos, use begin_table
    uint8_t byte_val = input[start_pos];
    uint32_t mt = rev_minterms[byte_val];
    uint16_t state = rev_begin[mt];

    if (state <= DFA_DEAD) return;

    // Check initial state nullability
    // In reverse scan: BEGIN mask if start_pos > 0 is actually CENTER;
    // if start_pos == 0, it's END (swapped for reverse)
    uint32_t eid = rev_effects_id[state];
    uint8_t init_mask = (start_pos == 0) ? NULL_END : NULL_CENTER;

    // Collect nulls from initial state
    if (eid == 1u) {
        if (init_mask & NULL_ALWAYS) {
            uint32_t idx = atomicAdd(hit_count, 1);
            if (idx < max_hits) hits[idx] = start_pos;  // pos + rel(0)
        }
    } else if (eid >= 2u && eid < rev_num_effects) {
        uint32_t s = rev_effects_offsets[eid];
        uint32_t e = rev_effects_offsets[eid + 1];
        for (uint32_t i = s; i < e; i++) {
            uint32_t packed = rev_effects_flat[i];
            uint8_t nmask = (uint8_t)(packed >> 16);
            uint32_t rel = packed & 0xFFFFu;
            if (nmask & init_mask) {
                uint32_t hit_pos = start_pos + rel;
                uint32_t idx = atomicAdd(hit_count, 1);
                if (idx < max_hits) hits[idx] = hit_pos;
            }
        }
    }

    // Deferred flush: track previous state's effect ID
    uint16_t prev_eid = 0;
    uint32_t pos = start_pos;

    while (pos > 0) {
        pos--;
        byte_val = input[pos];
        mt = rev_minterms[byte_val];

        // Flush previous state's effects at CENTER boundary (pos+1)
        if (prev_eid == 1u) {
            uint32_t idx = atomicAdd(hit_count, 1);
            if (idx < max_hits) hits[idx] = pos + 1;
        } else if (prev_eid >= 2u && prev_eid < rev_num_effects) {
            uint32_t s = rev_effects_offsets[prev_eid];
            uint32_t e = rev_effects_offsets[prev_eid + 1];
            for (uint32_t i = s; i < e; i++) {
                uint32_t packed = rev_effects_flat[i];
                uint8_t nmask = (uint8_t)(packed >> 16);
                uint32_t rel = packed & 0xFFFFu;
                if (nmask & NULL_CENTER) {
                    uint32_t hit_pos = (pos + 1) + rel;
                    uint32_t idx = atomicAdd(hit_count, 1);
                    if (idx < max_hits) hits[idx] = hit_pos;
                }
            }
        }

        uint32_t delta = ((uint32_t)state << rev_mt_log) | mt;
        uint16_t next = rev_center[delta];

        if (next <= DFA_DEAD) break;

        state = next;
        prev_eid = rev_effects_id[state];
    }

    // End of reverse scan (pos == 0): flush with END mask
    if (prev_eid == 1u) {
        uint32_t idx = atomicAdd(hit_count, 1);
        if (idx < max_hits) hits[idx] = 0;
    } else if (prev_eid >= 2u && prev_eid < rev_num_effects) {
        uint32_t s = rev_effects_offsets[prev_eid];
        uint32_t e = rev_effects_offsets[prev_eid + 1];
        for (uint32_t i = s; i < e; i++) {
            uint32_t packed = rev_effects_flat[i];
            uint8_t nmask = (uint8_t)(packed >> 16);
            uint32_t rel = packed & 0xFFFFu;
            if (nmask & NULL_END) {
                uint32_t hit_pos = 0 + rel;
                uint32_t idx = atomicAdd(hit_count, 1);
                if (idx < max_hits) hits[idx] = hit_pos;
            }
        }
    }
}

// ============================================================
// Forward DFA scan: each thread handles one candidate start position.
// Walks forward DFA to find the rightmost match-end.
// Uses shared memory for minterms and __ldg() for transition tables.
//
// Implements the deferred flush pattern matching the CPU reference:
// - prev_eid tracks the effect of the CURRENT state
// - Effects are flushed at the NEXT position (or at end-of-input with END mask)
// - Match-end position = pos - rel (from packed effects)
// ============================================================
extern "C"
__global__ void dfa_fwd_scan(
    const uint16_t* __restrict__ fwd_center,
    const uint16_t* __restrict__ fwd_begin,
    const uint16_t* __restrict__ fwd_effects_id,
    const uint32_t* __restrict__ fwd_effects_flat,
    const uint32_t* __restrict__ fwd_effects_offsets,
    uint32_t fwd_num_effects,
    const uint8_t*  __restrict__ fwd_minterms,     // forward byte→minterm
    uint32_t fwd_mt_log,
    const uint8_t*  __restrict__ input,
    uint32_t input_len,
    uint32_t fwd_initial,                          // forward initial state ID
    const uint32_t* __restrict__ starts,
    uint32_t* __restrict__ ends,
    uint32_t num_starts
) {
    // Load minterms into shared memory
    __shared__ uint8_t s_minterms[256];
    if (threadIdx.x < 256) {
        s_minterms[threadIdx.x] = fwd_minterms[threadIdx.x];
    }
    __syncthreads();

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_starts) return;

    uint32_t pos_begin = starts[tid];
    if (pos_begin >= input_len) { ends[tid] = 0; return; }

    // Check if initial state is nullable at this position (empty match)
    uint8_t empty_mask = (pos_begin == 0) ? NULL_BEGIN : NULL_CENTER;
    uint32_t initial_eid = __ldg(&fwd_effects_id[fwd_initial]);
    uint32_t has_empty = 0;
    if (initial_eid == 1u) {
        has_empty = (empty_mask & NULL_ALWAYS) ? 1 : 0;
    } else if (initial_eid >= 2u && initial_eid < fwd_num_effects) {
        uint32_t s = __ldg(&fwd_effects_offsets[initial_eid]);
        uint32_t e = __ldg(&fwd_effects_offsets[initial_eid + 1]);
        for (uint32_t i = s; i < e; i++) {
            uint32_t packed = __ldg(&fwd_effects_flat[i]);
            uint8_t nmask = (uint8_t)(packed >> 16);
            if (nmask & empty_mask) { has_empty = 1; break; }
        }
    }

    // First byte: use begin_table
    uint8_t byte_val = __ldg(&input[pos_begin]);
    uint32_t mt = s_minterms[byte_val];
    uint32_t curr;
    if (pos_begin == 0) {
        curr = __ldg(&fwd_begin[mt]);
    } else {
        uint32_t delta = ((uint32_t)fwd_initial << fwd_mt_log) | mt;
        curr = __ldg(&fwd_center[delta]);
    }

    if (curr <= DFA_DEAD) {
        ends[tid] = has_empty ? pos_begin : 0;
        return;
    }

    uint32_t end = input_len;
    uint32_t pos = pos_begin + 1;
    uint32_t max_end = 0;

    // Check first state's nullability
    uint8_t mask = (pos >= end) ? NULL_END : NULL_CENTER;
    uint32_t eid = __ldg(&fwd_effects_id[curr]);
    if (eid == 1u) {
        if (mask & NULL_ALWAYS) {
            max_end = pos;
        }
    } else if (eid >= 2u && eid < fwd_num_effects) {
        uint32_t s = __ldg(&fwd_effects_offsets[eid]);
        uint32_t e2 = __ldg(&fwd_effects_offsets[eid + 1]);
        for (uint32_t i = s; i < e2; i++) {
            uint32_t packed = __ldg(&fwd_effects_flat[i]);
            uint8_t nmask = (uint8_t)(packed >> 16);
            uint32_t rel = packed & 0xFFFFu;
            if ((nmask & mask) && (pos >= rel)) {
                uint32_t candidate = pos - rel;
                if (candidate > max_end) max_end = candidate;
            }
        }
    }

    if (pos >= end) {
        if (max_end > 0) { ends[tid] = max_end; }
        else { ends[tid] = has_empty ? pos_begin : 0; }
        return;
    }

    // Main scan loop with deferred flush (prev_eid)
    uint16_t prev_eid = 0;
    while (pos < end) {
        byte_val = __ldg(&input[pos]);
        mt = s_minterms[byte_val];
        uint32_t delta = (curr << fwd_mt_log) | mt;
        uint16_t next = __ldg(&fwd_center[delta]);

        if (next <= DFA_DEAD) {
            if (prev_eid == 1u) {
                if (pos > max_end) max_end = pos;
            } else if (prev_eid >= 2u && prev_eid < fwd_num_effects) {
                uint32_t s = __ldg(&fwd_effects_offsets[prev_eid]);
                uint32_t e2 = __ldg(&fwd_effects_offsets[prev_eid + 1]);
                for (uint32_t i = s; i < e2; i++) {
                    uint32_t packed = __ldg(&fwd_effects_flat[i]);
                    uint8_t nmask = (uint8_t)(packed >> 16);
                    uint32_t rel = packed & 0xFFFFu;
                    if ((nmask & NULL_CENTER) && (pos >= rel)) {
                        uint32_t candidate = pos - rel;
                        if (candidate > max_end) max_end = candidate;
                    }
                }
            }
            break;
        }

        if (prev_eid == 1u) {
            if (pos > max_end) max_end = pos;
        } else if (prev_eid >= 2u && prev_eid < fwd_num_effects) {
            uint32_t s = __ldg(&fwd_effects_offsets[prev_eid]);
            uint32_t e2 = __ldg(&fwd_effects_offsets[prev_eid + 1]);
            for (uint32_t i = s; i < e2; i++) {
                uint32_t packed = __ldg(&fwd_effects_flat[i]);
                uint8_t nmask = (uint8_t)(packed >> 16);
                uint32_t rel = packed & 0xFFFFu;
                if ((nmask & NULL_CENTER) && (pos >= rel)) {
                    uint32_t candidate = pos - rel;
                    if (candidate > max_end) max_end = candidate;
                }
            }
        }

        curr = next;
        prev_eid = __ldg(&fwd_effects_id[curr]);
        pos++;
    }

    // End-of-input: flush with END mask
    if (prev_eid != 0 && pos == end) {
        if (prev_eid == 1u) {
            if (pos > max_end) max_end = pos;
        } else if (prev_eid >= 2u && prev_eid < fwd_num_effects) {
            uint32_t s = __ldg(&fwd_effects_offsets[prev_eid]);
            uint32_t e2 = __ldg(&fwd_effects_offsets[prev_eid + 1]);
            for (uint32_t i = s; i < e2; i++) {
                uint32_t packed = __ldg(&fwd_effects_flat[i]);
                uint8_t nmask = (uint8_t)(packed >> 16);
                uint32_t rel = packed & 0xFFFFu;
                if ((nmask & NULL_END) && (pos >= rel)) {
                    uint32_t candidate = pos - rel;
                    if (candidate > max_end) max_end = candidate;
                }
            }
        }
    }

    if (max_end > 0) {
        ends[tid] = max_end;
    } else {
        ends[tid] = has_empty ? pos_begin : 0;
    }
}

// ============================================================
// Forward DFA scan for range [0, num_positions): used by nullable-slow path.
// Same as dfa_fwd_scan but starts are implicitly tid (no starts[] array needed).
// Eliminates the 40MB device allocation for the starts array.
// ============================================================
extern "C"
__global__ void dfa_fwd_scan_range(
    const uint16_t* __restrict__ fwd_center,
    const uint16_t* __restrict__ fwd_begin,
    const uint16_t* __restrict__ fwd_effects_id,
    const uint32_t* __restrict__ fwd_effects_flat,
    const uint32_t* __restrict__ fwd_effects_offsets,
    uint32_t fwd_num_effects,
    const uint8_t*  __restrict__ fwd_minterms,
    uint32_t fwd_mt_log,
    const uint8_t*  __restrict__ input,
    uint32_t input_len,
    uint32_t fwd_initial,
    uint32_t* __restrict__ ends,
    uint32_t num_positions
) {
    __shared__ uint8_t s_minterms[256];
    if (threadIdx.x < 256) {
        s_minterms[threadIdx.x] = fwd_minterms[threadIdx.x];
    }
    __syncthreads();

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_positions) return;

    uint32_t pos_begin = tid;  // implicit: start = thread index
    if (pos_begin >= input_len) { ends[tid] = 0; return; }

    uint8_t empty_mask = (pos_begin == 0) ? NULL_BEGIN : NULL_CENTER;
    uint32_t initial_eid = __ldg(&fwd_effects_id[fwd_initial]);
    uint32_t has_empty = 0;
    if (initial_eid == 1u) {
        has_empty = (empty_mask & NULL_ALWAYS) ? 1 : 0;
    } else if (initial_eid >= 2u && initial_eid < fwd_num_effects) {
        uint32_t s = __ldg(&fwd_effects_offsets[initial_eid]);
        uint32_t e = __ldg(&fwd_effects_offsets[initial_eid + 1]);
        for (uint32_t i = s; i < e; i++) {
            uint32_t packed = __ldg(&fwd_effects_flat[i]);
            uint8_t nmask = (uint8_t)(packed >> 16);
            if (nmask & empty_mask) { has_empty = 1; break; }
        }
    }

    uint8_t byte_val = __ldg(&input[pos_begin]);
    uint32_t mt = s_minterms[byte_val];
    uint32_t curr;
    if (pos_begin == 0) {
        curr = __ldg(&fwd_begin[mt]);
    } else {
        uint32_t delta = ((uint32_t)fwd_initial << fwd_mt_log) | mt;
        curr = __ldg(&fwd_center[delta]);
    }

    if (curr <= DFA_DEAD) {
        ends[tid] = has_empty ? pos_begin : 0;
        return;
    }

    uint32_t end = input_len;
    uint32_t pos = pos_begin + 1;
    uint32_t max_end = 0;

    uint8_t mask = (pos >= end) ? NULL_END : NULL_CENTER;
    uint32_t eid = __ldg(&fwd_effects_id[curr]);
    if (eid == 1u) {
        if (mask & NULL_ALWAYS) max_end = pos;
    } else if (eid >= 2u && eid < fwd_num_effects) {
        uint32_t s = __ldg(&fwd_effects_offsets[eid]);
        uint32_t e2 = __ldg(&fwd_effects_offsets[eid + 1]);
        for (uint32_t i = s; i < e2; i++) {
            uint32_t packed = __ldg(&fwd_effects_flat[i]);
            uint8_t nmask = (uint8_t)(packed >> 16);
            uint32_t rel = packed & 0xFFFFu;
            if ((nmask & mask) && (pos >= rel)) {
                uint32_t candidate = pos - rel;
                if (candidate > max_end) max_end = candidate;
            }
        }
    }

    if (pos >= end) {
        if (max_end > 0) { ends[tid] = max_end; }
        else { ends[tid] = has_empty ? pos_begin : 0; }
        return;
    }

    uint16_t prev_eid = 0;
    while (pos < end) {
        byte_val = __ldg(&input[pos]);
        mt = s_minterms[byte_val];
        uint32_t delta = (curr << fwd_mt_log) | mt;
        uint16_t next = __ldg(&fwd_center[delta]);

        if (next <= DFA_DEAD) {
            if (prev_eid == 1u) {
                if (pos > max_end) max_end = pos;
            } else if (prev_eid >= 2u && prev_eid < fwd_num_effects) {
                uint32_t s = __ldg(&fwd_effects_offsets[prev_eid]);
                uint32_t e2 = __ldg(&fwd_effects_offsets[prev_eid + 1]);
                for (uint32_t i = s; i < e2; i++) {
                    uint32_t packed = __ldg(&fwd_effects_flat[i]);
                    uint8_t nmask = (uint8_t)(packed >> 16);
                    uint32_t rel = packed & 0xFFFFu;
                    if ((nmask & NULL_CENTER) && (pos >= rel)) {
                        uint32_t candidate = pos - rel;
                        if (candidate > max_end) max_end = candidate;
                    }
                }
            }
            break;
        }

        if (prev_eid == 1u) {
            if (pos > max_end) max_end = pos;
        } else if (prev_eid >= 2u && prev_eid < fwd_num_effects) {
            uint32_t s = __ldg(&fwd_effects_offsets[prev_eid]);
            uint32_t e2 = __ldg(&fwd_effects_offsets[prev_eid + 1]);
            for (uint32_t i = s; i < e2; i++) {
                uint32_t packed = __ldg(&fwd_effects_flat[i]);
                uint8_t nmask = (uint8_t)(packed >> 16);
                uint32_t rel = packed & 0xFFFFu;
                if ((nmask & NULL_CENTER) && (pos >= rel)) {
                    uint32_t candidate = pos - rel;
                    if (candidate > max_end) max_end = candidate;
                }
            }
        }

        curr = next;
        prev_eid = __ldg(&fwd_effects_id[curr]);
        pos++;
    }

    if (prev_eid != 0 && pos == end) {
        if (prev_eid == 1u) {
            if (pos > max_end) max_end = pos;
        } else if (prev_eid >= 2u && prev_eid < fwd_num_effects) {
            uint32_t s = __ldg(&fwd_effects_offsets[prev_eid]);
            uint32_t e2 = __ldg(&fwd_effects_offsets[prev_eid + 1]);
            for (uint32_t i = s; i < e2; i++) {
                uint32_t packed = __ldg(&fwd_effects_flat[i]);
                uint8_t nmask = (uint8_t)(packed >> 16);
                uint32_t rel = packed & 0xFFFFu;
                if ((nmask & NULL_END) && (pos >= rel)) {
                    uint32_t candidate = pos - rel;
                    if (candidate > max_end) max_end = candidate;
                }
            }
        }
    }

    if (max_end > 0) {
        ends[tid] = max_end;
    } else {
        ends[tid] = has_empty ? pos_begin : 0;
    }
}

// ============================================================
// PARALLEL PREFIX DFA REVERSE SCAN (3-kernel pipeline)
//
// Parallelizes the inherently sequential reverse DFA scan by
// splitting the input into chunks. Each chunk computes a
// State→State mapping, then a sequential prefix propagation
// determines the initial state per chunk, and finally each chunk
// resolves states and checks effects in parallel.
//
// For input of N bytes with C=CHUNK_SIZE:
//   num_chunks = ceil(N / C) threads run in parallel
//   Phase 2 (propagate) is O(num_chunks) sequential = ~40K iterations
//   Overall: O(N/P * S + num_chunks) vs O(N) sequential
// ============================================================

// Kernel 1: Per-chunk transition mapping computation.
// Each thread processes CHUNK_SIZE bytes of the reversed input and
// computes the cumulative State→State mapping for its chunk.
//
// Uses shared memory for minterms lookup (256 bytes) and __ldg()
// for read-only global memory access to transition tables.
extern "C"
__global__ void dfa_rev_chunk_map(
    const uint16_t* __restrict__ rev_center,
    const uint16_t* __restrict__ rev_begin,
    const uint8_t*  __restrict__ rev_minterms,
    uint32_t rev_mt_log,
    uint32_t num_states,
    const uint8_t*  __restrict__ input,
    uint32_t input_len,
    uint32_t start_pos,
    uint16_t* __restrict__ chunk_maps,
    uint32_t num_chunks
) {
    // Load minterms into shared memory (256 bytes, one-time cost per block)
    __shared__ uint8_t s_minterms[256];
    if (threadIdx.x < 256) {
        s_minterms[threadIdx.x] = rev_minterms[threadIdx.x];
    }
    __syncthreads();

    uint32_t chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks) return;

    uint32_t r_start = chunk_id * CHUNK_SIZE;
    uint32_t r_end = r_start + CHUNK_SIZE;
    if (r_end > start_pos + 1) r_end = start_pos + 1;

    // Initialize: identity mapping (state s → state s)
    uint16_t states[PAR_MAX_STATES];
    uint32_t ns = (num_states < PAR_MAX_STATES) ? num_states : PAR_MAX_STATES;
    for (uint32_t s = 0; s < ns; s++) {
        states[s] = (uint16_t)s;
    }

    // Walk through all bytes in this chunk
    for (uint32_t r = r_start; r < r_end; r++) {
        uint32_t orig_pos = start_pos - r;
        uint8_t byte_val = __ldg(&input[orig_pos]);
        uint32_t mt = s_minterms[byte_val];

        if (r == 0) {
            uint16_t s0 = __ldg(&rev_begin[mt]);
            for (uint32_t s = 0; s < ns; s++) {
                states[s] = s0;
            }
        } else {
            for (uint32_t s = 0; s < ns; s++) {
                uint32_t idx = ((uint32_t)states[s] << rev_mt_log) | mt;
                states[s] = __ldg(&rev_center[idx]);
            }
        }
    }

    // Write output mapping
    for (uint32_t s = 0; s < ns; s++) {
        chunk_maps[chunk_id * PAR_MAX_STATES + s] = states[s];
    }
}

// Kernel 2: Sequential inter-chunk prefix propagation.
// Single-thread kernel that chains chunk mappings to determine
// the initial DFA state for each chunk.
//
// chunk_initials[0] = 0 (don't care — chunk 0 uses begin_table)
// chunk_initials[k] = chunk_maps[k-1][ chunk_initials[k-1] ]
//
// Also computes chunk_prev_eids[k] = effects_id[ chunk_initials[k] ]
// for the deferred-flush handoff at chunk boundaries.
extern "C"
__global__ void dfa_rev_chunk_propagate(
    const uint16_t* __restrict__ chunk_maps,
    const uint16_t* __restrict__ rev_effects_id,
    uint32_t num_chunks,
    uint32_t num_states,
    uint16_t* __restrict__ chunk_initials,
    uint16_t* __restrict__ chunk_prev_eids
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Chunk 0: begin_table handles initialization; these values are unused
    chunk_initials[0] = 0;
    chunk_prev_eids[0] = 0;

    uint16_t state = 0;
    for (uint32_t k = 0; k < num_chunks; k++) {
        // Look up the output state for chunk k given input state
        uint16_t idx = (state < num_states) ? state : 0;
        uint16_t out_state = chunk_maps[k * PAR_MAX_STATES + idx];
        state = out_state;

        if (k + 1 < num_chunks) {
            chunk_initials[k + 1] = out_state;
            chunk_prev_eids[k + 1] = rev_effects_id[out_state];
        }
    }
}

// Helper: flush reverse effects into hits array.
// Used by dfa_rev_chunk_resolve to avoid code duplication.
__device__ void rev_flush_effects(
    uint16_t eid,
    uint8_t flush_mask,
    uint32_t flush_pos,
    const uint32_t* __restrict__ rev_effects_flat,
    const uint32_t* __restrict__ rev_effects_offsets,
    uint32_t rev_num_effects,
    uint32_t* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t max_hits
) {
    if (eid == 0) return;
    if (eid == 1u) {
        if (flush_mask & NULL_ALWAYS) {
            uint32_t idx = atomicAdd(hit_count, 1);
            if (idx < max_hits) hits[idx] = flush_pos;
        }
        return;
    }
    if (eid >= rev_num_effects) return;
    uint32_t s = rev_effects_offsets[eid];
    uint32_t e = rev_effects_offsets[eid + 1];
    for (uint32_t i = s; i < e; i++) {
        uint32_t packed = rev_effects_flat[i];
        uint8_t nmask = (uint8_t)(packed >> 16);
        uint32_t rel = packed & 0xFFFFu;
        if (nmask & flush_mask) {
            uint32_t hit_pos = flush_pos + rel;
            uint32_t idx = atomicAdd(hit_count, 1);
            if (idx < max_hits) hits[idx] = hit_pos;
        }
    }
}

// Kernel 3: Per-chunk state resolution and effects checking.
// Each thread walks its chunk sequentially from the propagated
// initial state, applying the deferred-flush pattern and recording
// match-start candidates into the shared hits array.
// Uses shared memory for minterms and __ldg() for transition tables.
extern "C"
__global__ void dfa_rev_chunk_resolve(
    const uint16_t* __restrict__ rev_center,
    const uint16_t* __restrict__ rev_begin,
    const uint16_t* __restrict__ rev_effects_id,
    const uint32_t* __restrict__ rev_effects_flat,
    const uint32_t* __restrict__ rev_effects_offsets,
    uint32_t rev_num_effects,
    const uint8_t*  __restrict__ rev_minterms,
    uint32_t rev_mt_log,
    const uint8_t*  __restrict__ input,
    uint32_t input_len,
    uint32_t start_pos,
    const uint16_t* __restrict__ chunk_initials,
    const uint16_t* __restrict__ chunk_prev_eids,
    uint32_t num_chunks,
    uint32_t* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t max_hits
) {
    // Load minterms into shared memory
    __shared__ uint8_t s_minterms[256];
    if (threadIdx.x < 256) {
        s_minterms[threadIdx.x] = rev_minterms[threadIdx.x];
    }
    __syncthreads();

    uint32_t chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks) return;

    uint32_t r_start = chunk_id * CHUNK_SIZE;
    uint32_t r_end = r_start + CHUNK_SIZE;
    if (r_end > start_pos + 1) r_end = start_pos + 1;
    if (r_start >= r_end) return;

    uint16_t state;
    uint16_t prev_eid;

    if (chunk_id == 0) {
        // ---- First chunk: handle begin_table + immediate effects check ----
        uint8_t byte_val = __ldg(&input[start_pos]);
        uint32_t mt = s_minterms[byte_val];
        state = __ldg(&rev_begin[mt]);

        if (state <= DFA_DEAD) return;

        // Immediate effects check for initial state (not deferred)
        uint32_t eid = __ldg(&rev_effects_id[state]);
        uint8_t init_mask = (start_pos == 0) ? NULL_END : NULL_CENTER;
        rev_flush_effects(eid, init_mask, start_pos,
            rev_effects_flat, rev_effects_offsets, rev_num_effects,
            hits, hit_count, max_hits);

        prev_eid = 0;  // Deferred flush starts clean after immediate check

        // Process remaining bytes in chunk 0 (r = 1 to r_end-1)
        for (uint32_t r = r_start + 1; r < r_end; r++) {
            uint32_t orig_pos = start_pos - r;
            uint8_t bv = __ldg(&input[orig_pos]);
            uint32_t mt2 = s_minterms[bv];

            // Flush prev_eid at CENTER for position (orig_pos + 1)
            rev_flush_effects(prev_eid, NULL_CENTER, orig_pos + 1,
                rev_effects_flat, rev_effects_offsets, rev_num_effects,
                hits, hit_count, max_hits);

            uint32_t delta = ((uint32_t)state << rev_mt_log) | mt2;
            uint16_t next = __ldg(&rev_center[delta]);

            if (next <= DFA_DEAD) return;

            state = next;
            prev_eid = __ldg(&rev_effects_id[state]);

            // If this is the very last position (orig_pos == 0): END flush
            if (orig_pos == 0) {
                rev_flush_effects(prev_eid, NULL_END, 0,
                    rev_effects_flat, rev_effects_offsets, rev_num_effects,
                    hits, hit_count, max_hits);
                return;
            }
        }
        // End of chunk but NOT end of scan: don't flush (belongs to next chunk)

    } else {
        // ---- Chunks k > 0: start from propagated initial state ----
        state = __ldg(&chunk_initials[chunk_id]);
        prev_eid = __ldg(&chunk_prev_eids[chunk_id]);

        if (state <= DFA_DEAD) return;

        for (uint32_t r = r_start; r < r_end; r++) {
            uint32_t orig_pos = start_pos - r;
            uint8_t bv = __ldg(&input[orig_pos]);
            uint32_t mt2 = s_minterms[bv];

            // Flush prev_eid at CENTER for position (orig_pos + 1)
            rev_flush_effects(prev_eid, NULL_CENTER, orig_pos + 1,
                rev_effects_flat, rev_effects_offsets, rev_num_effects,
                hits, hit_count, max_hits);

            uint32_t delta = ((uint32_t)state << rev_mt_log) | mt2;
            uint16_t next = __ldg(&rev_center[delta]);

            if (next <= DFA_DEAD) return;

            state = next;
            prev_eid = __ldg(&rev_effects_id[state]);

            // If this is the very last position (orig_pos == 0): END flush
            if (orig_pos == 0) {
                rev_flush_effects(prev_eid, NULL_END, 0,
                    rev_effects_flat, rev_effects_offsets, rev_num_effects,
                    hits, hit_count, max_hits);
                return;
            }
        }
        // End of chunk but NOT end of scan: don't flush
    }
}
