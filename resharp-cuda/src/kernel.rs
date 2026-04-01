//! CPU reference implementation of the DFA scan algorithms.
//!
//! Provides a pure-Rust implementation of the two-phase RE# matching algorithm
//! that faithfully mirrors the resharp engine's `scan_fwd_slow` / `collect_rev` logic.
//! Used as:
//! - A fallback when GPU is unavailable or input is below the GPU threshold
//! - An oracle for cross-validation testing against GPU kernels
//!
//! Key implementation details:
//! - Separate forward/reverse minterm lookup tables (`minterms_lookup` vs `rev_minterms_lookup`)
//! - Nullability mask checking: BEGIN (position 0), CENTER (middle), END (end of input)
//! - Deferred flush pattern: effects of the current state are checked at the *next* position
//! - Relative offset (`rel`): match position = `pos - rel` (forward) or `pos + rel` (reverse)

use resharp::{DfaTables, Match};

const DFA_DEAD: u16 = 1;

// Nullability bit masks (from resharp-algebra/src/nulls.rs)
const NULL_CENTER: u8 = 0b001;
const NULL_BEGIN: u8  = 0b010;
const NULL_END: u8    = 0b100;
const NULL_ALWAYS: u8 = 0b111;

/// Check if any effect in the given effects_id matches the boundary mask.
/// Returns true if the state is nullable at this boundary.
#[allow(dead_code)]
fn has_any_null(dfa: &DfaTables, eid: u16, mask: u8) -> bool {
    if eid == 0 { return false; }
    if eid == 1 { return mask & NULL_ALWAYS != 0; }
    let eid = eid as usize;
    if eid >= dfa.fwd_effects_offsets.len() { return false; }
    let start = dfa.fwd_effects_offsets[eid] as usize;
    let end = dfa.fwd_effects_offsets.get(eid + 1).copied()
        .unwrap_or(dfa.fwd_effects_flat.len() as u32) as usize;
    for i in start..end {
        if let Some(&packed) = dfa.fwd_effects_flat.get(i) {
            let nmask = (packed >> 16) as u8;
            if nmask & mask != 0 { return true; }
        }
    }
    false
}

/// Forward scan: collect max_end considering nullability masks and rel offsets.
fn fwd_collect_null(
    effects_offsets: &[u32],
    effects_flat: &[u32],
    eid: u16,
    pos: usize,
    mask: u8,
    max_end: &mut usize,
) {
    if eid == 0 { return; }
    if eid == 1 {
        if mask & NULL_ALWAYS != 0 { *max_end = (*max_end).max(pos); }
        return;
    }
    let eid = eid as usize;
    if eid >= effects_offsets.len() { return; }
    let start = effects_offsets[eid] as usize;
    let end = effects_offsets.get(eid + 1).copied()
        .unwrap_or(effects_flat.len() as u32) as usize;
    for i in start..end {
        if let Some(&packed) = effects_flat.get(i) {
            let nmask = (packed >> 16) as u8;
            let rel = (packed & 0xFFFF) as usize;
            if nmask & mask != 0 {
                if pos >= rel {
                    *max_end = (*max_end).max(pos - rel);
                }
            }
        }
    }
}

/// Reverse scan: collect match-start candidates considering masks and rel offsets.
fn rev_collect_null(
    effects_offsets: &[u32],
    effects_flat: &[u32],
    eid: u16,
    pos: usize,
    mask: u8,
    nulls: &mut Vec<usize>,
) {
    if eid == 0 { return; }
    if eid == 1 {
        if mask & NULL_ALWAYS != 0 { nulls.push(pos); }
        return;
    }
    let eid = eid as usize;
    if eid >= effects_offsets.len() { return; }
    let start = effects_offsets[eid] as usize;
    let end = effects_offsets.get(eid + 1).copied()
        .unwrap_or(effects_flat.len() as u32) as usize;
    for i in start..end {
        if let Some(&packed) = effects_flat.get(i) {
            let nmask = (packed >> 16) as u8;
            let rel = (packed & 0xFFFF) as usize;
            if nmask & mask != 0 {
                nulls.push(pos + rel);
            }
        }
    }
}

/// Forward DFA scan from `pos_begin`. Returns rightmost match-end, or 0 (NO_MATCH).
fn scan_fwd(dfa: &DfaTables, data: &[u8], pos_begin: usize) -> usize {
    if pos_begin >= data.len() { return 0; }

    let empty_mask = if pos_begin == 0 { NULL_BEGIN } else { NULL_CENTER };
    let initial_eid = dfa.fwd_effects_id.get(dfa.initial_fwd as usize).copied().unwrap_or(0);
    let has_empty = {
        if initial_eid == 0 { false }
        else if initial_eid == 1 { empty_mask & NULL_ALWAYS != 0 }
        else {
            let eid = initial_eid as usize;
            if eid < dfa.fwd_effects_offsets.len() {
                let start = dfa.fwd_effects_offsets[eid] as usize;
                let end = dfa.fwd_effects_offsets.get(eid + 1).copied()
                    .unwrap_or(dfa.fwd_effects_flat.len() as u32) as usize;
                (start..end).any(|i| {
                    dfa.fwd_effects_flat.get(i).map_or(false, |&p| (p >> 16) as u8 & empty_mask != 0)
                })
            } else { false }
        }
    };

    let mt = dfa.minterms_lookup[data[pos_begin] as usize];
    let mut curr = match dfa.fwd_begin_table.get(mt as usize) {
        Some(&s) => s as u32,
        None => return if has_empty { pos_begin } else { 0 },
    };
    if curr <= DFA_DEAD as u32 {
        return if has_empty { pos_begin } else { 0 };
    }

    let end = data.len();
    let mut pos = pos_begin + 1;
    let mut max_end: usize = 0;

    // Check first state's nullability
    let mask = if pos == end { NULL_END } else { NULL_CENTER };
    let eid = dfa.fwd_effects_id.get(curr as usize).copied().unwrap_or(0);
    fwd_collect_null(&dfa.fwd_effects_offsets, &dfa.fwd_effects_flat, eid, pos, mask, &mut max_end);

    if pos == end {
        return resolve_max_end(max_end, has_empty, pos_begin);
    }

    // Main scan loop
    let mut prev_eid: u16 = 0;
    while pos < end {
        let mt = dfa.minterms_lookup[data[pos] as usize] as u32;
        let delta = (curr << dfa.mt_log | mt) as usize;
        let next = dfa.fwd_center_table.get(delta).copied().unwrap_or(DFA_DEAD);
        if next <= DFA_DEAD {
            // Flush pending eid before breaking
            if prev_eid != 0 {
                let mask = NULL_CENTER;
                fwd_collect_null(&dfa.fwd_effects_offsets, &dfa.fwd_effects_flat, prev_eid, pos, mask, &mut max_end);
            }
            break;
        }
        // Flush previous state's effects at CENTER boundary
        if prev_eid != 0 {
            fwd_collect_null(&dfa.fwd_effects_offsets, &dfa.fwd_effects_flat, prev_eid, pos, NULL_CENTER, &mut max_end);
        }
        curr = next as u32;
        prev_eid = dfa.fwd_effects_id.get(curr as usize).copied().unwrap_or(0);
        pos += 1;
    }

    // End-of-input: flush with END mask
    if prev_eid != 0 && pos == end {
        fwd_collect_null(&dfa.fwd_effects_offsets, &dfa.fwd_effects_flat, prev_eid, pos, NULL_END, &mut max_end);
    }

    resolve_max_end(max_end, has_empty, pos_begin)
}

fn resolve_max_end(max_end: usize, has_empty: bool, pos_begin: usize) -> usize {
    if max_end > 0 { max_end }
    else if has_empty { pos_begin }
    else { 0 }
}

/// Reverse DFA scan: collect match-start candidates.
fn scan_rev(dfa: &DfaTables, data: &[u8], start_pos: usize, nulls: &mut Vec<usize>) {
    if data.is_empty() || start_pos >= data.len() { return; }

    let mt = dfa.rev_minterms_lookup[data[start_pos] as usize];
    let mut curr = match dfa.rev_begin_table.get(mt as usize) {
        Some(&s) => s as u32,
        None => return,
    };
    if curr <= DFA_DEAD as u32 { return; }

    // Check initial reverse state — note: reverse BEGIN = forward END
    let begin_mask = if start_pos == 0 { NULL_END } else { NULL_CENTER };
    let eid = dfa.rev_effects_id.get(curr as usize).copied().unwrap_or(0);
    rev_collect_null(&dfa.rev_effects_offsets, &dfa.rev_effects_flat, eid, start_pos, begin_mask, nulls);

    let mut pos = start_pos;
    let mut prev_eid: u16 = 0;

    while pos > 0 {
        pos -= 1;
        let mt = dfa.rev_minterms_lookup[data[pos] as usize] as u32;

        // Flush previous state's effects at CENTER boundary
        if prev_eid != 0 {
            rev_collect_null(&dfa.rev_effects_offsets, &dfa.rev_effects_flat, prev_eid, pos + 1, NULL_CENTER, nulls);
        }

        let delta = (curr << dfa.rev_mt_log | mt) as usize;
        let next = dfa.rev_center_table.get(delta).copied().unwrap_or(DFA_DEAD);
        if next <= DFA_DEAD {
            break;
        }
        curr = next as u32;
        prev_eid = dfa.rev_effects_id.get(curr as usize).copied().unwrap_or(0);
    }

    // End of reverse scan (pos == 0): flush with END mask
    if prev_eid != 0 {
        rev_collect_null(&dfa.rev_effects_offsets, &dfa.rev_effects_flat, prev_eid, 0, NULL_END, nulls);
    }
}

/// CPU reference: find all non-overlapping leftmost-longest matches.
pub fn cpu_find_all(dfa: &DfaTables, input: &[u8]) -> Vec<Match> {
    if input.is_empty() {
        return if dfa.empty_nullable { vec![Match { start: 0, end: 0 }] } else { vec![] };
    }

    // Check if reverse initial state is nullable (special slow path)
    let rev_initial_eid = dfa.rev_effects_id.get(dfa.initial_rev as usize).copied().unwrap_or(0);
    let rev_initial_nullable = rev_initial_eid != 0;

    if rev_initial_nullable {
        // Use the "nullable slow" path: iterate positions, try forward scan from each
        return cpu_find_all_nullable_slow(dfa, input);
    }

    // Phase 1: reverse scan to collect start candidates
    let mut nulls = Vec::new();
    scan_rev(dfa, input, input.len() - 1, &mut nulls);

    // Phase 2: forward scan from each candidate
    let mut matches = Vec::new();
    if let Some(fl) = dfa.fixed_length {
        let fl = fl as usize;
        let mut last_end = 0usize;
        for &start in nulls.iter().rev() {
            if start >= last_end && start + fl <= input.len() {
                matches.push(Match { start, end: start + fl });
                last_end = start + fl;
            }
        }
    } else {
        let mut last_end = 0usize;
        for &start in nulls.iter().rev() {
            if start < last_end { continue; }
            let end = scan_fwd(dfa, input, start);
            if end > start {
                matches.push(Match { start, end });
                last_end = end;
            } else if end == start {
                // Zero-width match (anchors, etc.)
                matches.push(Match { start, end: start });
                last_end = start;
            }
        }
    }
    matches
}

/// Slow path for patterns where the reverse initial state is nullable.
/// This means every position is potentially a match start.
fn cpu_find_all_nullable_slow(dfa: &DfaTables, input: &[u8]) -> Vec<Match> {
    let mut matches = Vec::new();
    let mut pos = 0usize;
    while pos <= input.len() {
        if pos == input.len() {
            // Check empty match at end
            let initial_eid = dfa.fwd_effects_id.get(dfa.initial_fwd as usize).copied().unwrap_or(0);
            if initial_eid != 0 {
                let mask = NULL_END;
                let ok = if initial_eid == 1 { true }
                    else {
                        let eid = initial_eid as usize;
                        if eid < dfa.fwd_effects_offsets.len() {
                            let s = dfa.fwd_effects_offsets[eid] as usize;
                            let e = dfa.fwd_effects_offsets.get(eid+1).copied()
                                .unwrap_or(dfa.fwd_effects_flat.len() as u32) as usize;
                            (s..e).any(|i| dfa.fwd_effects_flat.get(i).map_or(false, |&p| (p>>16) as u8 & mask != 0))
                        } else { false }
                    };
                if ok && matches.last().map_or(true, |m: &Match| m.end <= pos) {
                    matches.push(Match { start: pos, end: pos });
                }
            }
            break;
        }
        let end = scan_fwd(dfa, input, pos);
        if end > pos {
            matches.push(Match { start: pos, end });
            pos = end;
        } else if end == pos {
            // Zero-width match
            if matches.last().map_or(true, |m: &Match| m.end <= pos) {
                matches.push(Match { start: pos, end: pos });
            }
            pos += 1;
        } else {
            pos += 1;
        }
    }
    matches
}

/// CPU reference: is_match.
pub fn cpu_is_match(dfa: &DfaTables, input: &[u8]) -> bool {
    if input.is_empty() { return dfa.empty_nullable; }

    let rev_initial_eid = dfa.rev_effects_id.get(dfa.initial_rev as usize).copied().unwrap_or(0);
    if rev_initial_eid != 0 {
        // nullable slow: try each position
        for pos in 0..input.len() {
            let end = scan_fwd(dfa, input, pos);
            if end > 0 || (end == pos && pos == 0) { return true; }
        }
        return false;
    }

    let mut nulls = Vec::new();
    scan_rev(dfa, input, input.len() - 1, &mut nulls);
    for &start in &nulls {
        let end = scan_fwd(dfa, input, start);
        if end >= start && end > 0 { return true; }
    }
    false
}

/// CPU reference: find_anchored at position 0.
pub fn cpu_find_anchored(dfa: &DfaTables, input: &[u8]) -> Option<Match> {
    if input.is_empty() {
        return if dfa.empty_nullable { Some(Match { start: 0, end: 0 }) } else { None };
    }
    let end = scan_fwd(dfa, input, 0);
    if end > 0 { Some(Match { start: 0, end }) }
    else {
        // Check zero-width match at 0
        let initial_eid = dfa.fwd_effects_id.get(dfa.initial_fwd as usize).copied().unwrap_or(0);
        if initial_eid != 0 {
            let mask = NULL_BEGIN;
            let ok = if initial_eid == 1 { true }
                else {
                    let eid = initial_eid as usize;
                    if eid < dfa.fwd_effects_offsets.len() {
                        let s = dfa.fwd_effects_offsets[eid] as usize;
                        let e = dfa.fwd_effects_offsets.get(eid+1).copied()
                            .unwrap_or(dfa.fwd_effects_flat.len() as u32) as usize;
                        (s..e).any(|i| dfa.fwd_effects_flat.get(i).map_or(false, |&p| (p>>16) as u8 & mask != 0))
                    } else { false }
                };
            if ok { return Some(Match { start: 0, end: 0 }); }
        }
        None
    }
}
