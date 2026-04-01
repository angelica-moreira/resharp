//! CUDA Driver API bindings and GPU context management.
//!
//! Uses raw FFI to `libcuda.so` — no external crate dependencies required.
//! All GPU operations (memory allocation, kernel launches, synchronization) go through
//! the CUDA Driver API (`cu*` functions).
//!
//! ## Architecture
//!
//! - **[`GpuContext`]**: Holds the CUDA context, compiled kernel functions, uploaded DFA tables,
//!   and a dedicated CUDA stream. Created once per pattern, reused across `find_all` calls.
//! - **Kernel pipeline**: 3-kernel parallel prefix reverse scan → parallel forward scan.
//!   All kernels launch on the same CUDA stream for implicit ordering without explicit sync.
//! - **CUBIN loading**: Kernels are compiled to native binary (CUBIN) at build time by `nvcc`,
//!   embedded via `include_bytes!`, and loaded with `cuModuleLoadData`.
//!
//! ## Target Hardware
//!
//! RTX A6000 (sm_86): 84 SMs, 48KB shared mem, 2100 MHz max clock,
//! 768 GB/s memory bandwidth, 46GB VRAM.

use resharp::{DfaTables, Match};
use std::ffi::CString;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;

#[allow(unused_imports)]
use std::ffi::CStr;

// CUDA Driver API types
type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;

const CUDA_SUCCESS: CUresult = 0;

extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytesize: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint,
        blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint,
        sharedMemBytes: c_uint, hStream: CUstream,
        kernelParams: *mut *mut c_void, extra: *mut *mut c_void,
    ) -> CUresult;
    #[allow(dead_code)]
    fn cuCtxSynchronize() -> CUresult;
    fn cuMemsetD8_v2(dptr: CUdeviceptr, uc: u8, n: usize) -> CUresult;
    fn cuStreamCreate(phStream: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;
    fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
    #[allow(dead_code)]
    fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, bytesize: usize, hStream: CUstream) -> CUresult;
    #[allow(dead_code)]
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, bytesize: usize, hStream: CUstream) -> CUresult;
    #[allow(dead_code)]
    fn cuMemsetD8Async(dptr: CUdeviceptr, uc: u8, n: usize, hStream: CUstream) -> CUresult;
}

// Device attribute constants
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;

fn cuda_check(result: CUresult, op: &str) -> Result<(), String> {
    if result != CUDA_SUCCESS {
        Err(format!("CUDA {} failed: error {}", op, result))
    } else {
        Ok(())
    }
}

/// GPU device info for tuning kernel launches.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub sm_count: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_sm: u32,
    pub shared_mem_per_block: u32,
}

/// All DFA tables uploaded to GPU device memory.
/// Separate fwd/rev minterms, effects_flat, effects_offsets.
struct GpuDfaBuffers {
    // Forward DFA
    d_fwd_center: CUdeviceptr,
    d_fwd_begin: CUdeviceptr,
    d_fwd_effects_id: CUdeviceptr,
    d_fwd_effects_flat: CUdeviceptr,
    d_fwd_effects_offsets: CUdeviceptr,
    d_fwd_minterms: CUdeviceptr,
    // Reverse DFA
    d_rev_center: CUdeviceptr,
    d_rev_begin: CUdeviceptr,
    d_rev_effects_id: CUdeviceptr,
    d_rev_effects_flat: CUdeviceptr,
    d_rev_effects_offsets: CUdeviceptr,
    d_rev_minterms: CUdeviceptr,
}

/// Maximum number of DFA states supported by the parallel prefix kernels.
/// For S > this, we fall back to the sequential reverse scan.
const PAR_MAX_STATES: u32 = 32;

/// Chunk size for the parallel prefix reverse scan (bytes per thread).
const CHUNK_SIZE: u32 = 256;

/// GPU execution context: CUDA device, module, and pre-uploaded DFA tables.
/// Uses a dedicated CUDA stream for all kernel launches to avoid
/// unnecessary inter-kernel synchronization.
#[allow(dead_code)]
pub struct GpuContext {
    ctx: CUcontext,
    stream: CUstream,
    _module: CUmodule,
    fwd_scan_fn: CUfunction,
    fwd_scan_range_fn: CUfunction,
    rev_scan_fn: CUfunction,
    // Parallel prefix kernels
    rev_chunk_map_fn: CUfunction,
    rev_chunk_propagate_fn: CUfunction,
    rev_chunk_resolve_fn: CUfunction,
    bufs: GpuDfaBuffers,
    pub device_info: DeviceInfo,
    // Forward DFA metadata
    fwd_initial: u32,
    fwd_mt_log: u32,
    fwd_num_effects: u32,
    // Reverse DFA metadata
    rev_mt_log: u32,
    rev_num_effects: u32,
    rev_num_states: u32,
    // Pattern metadata
    empty_nullable: bool,
    fixed_length: Option<u32>,
    rev_initial_nullable: bool,
}

// CUBIN binary embedded at build time (compiled native binary, not PTX text)
const CUBIN_DATA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dfa_scan.ptx"));

impl GpuContext {
    /// Initialize CUDA, load PTX, upload DFA tables.
    pub fn new(dfa: &DfaTables) -> Result<Self, String> {
        unsafe {
            cuda_check(cuInit(0), "cuInit")?;

            let mut device: CUdevice = 0;
            cuda_check(cuDeviceGet(&mut device, 0), "cuDeviceGet")?;

            // Query device info
            let mut sm_count: c_int = 0;
            let mut max_tpb: c_int = 0;
            let mut max_tpsm: c_int = 0;
            let mut shared_mem: c_int = 0;
            cuDeviceGetAttribute(&mut sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            cuDeviceGetAttribute(&mut max_tpb, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
            cuDeviceGetAttribute(&mut max_tpsm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
            cuDeviceGetAttribute(&mut shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);

            let device_info = DeviceInfo {
                sm_count: sm_count as u32,
                max_threads_per_block: max_tpb as u32,
                max_threads_per_sm: max_tpsm as u32,
                shared_mem_per_block: shared_mem as u32,
            };

            let mut ctx: CUcontext = ptr::null_mut();
            cuda_check(cuCtxCreate_v2(&mut ctx, 0, device), "cuCtxCreate")?;

            // Load CUBIN module (native binary, no CString needed)
            let mut module: CUmodule = ptr::null_mut();
            cuda_check(
                cuModuleLoadData(&mut module, CUBIN_DATA.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;

            // Get kernel functions
            let fwd_name = CString::new("dfa_fwd_scan").unwrap();
            let fwd_range_name = CString::new("dfa_fwd_scan_range").unwrap();
            let rev_name = CString::new("dfa_rev_scan").unwrap();
            let rev_chunk_map_name = CString::new("dfa_rev_chunk_map").unwrap();
            let rev_chunk_propagate_name = CString::new("dfa_rev_chunk_propagate").unwrap();
            let rev_chunk_resolve_name = CString::new("dfa_rev_chunk_resolve").unwrap();
            let mut fwd_scan_fn: CUfunction = ptr::null_mut();
            let mut fwd_scan_range_fn: CUfunction = ptr::null_mut();
            let mut rev_scan_fn: CUfunction = ptr::null_mut();
            let mut rev_chunk_map_fn: CUfunction = ptr::null_mut();
            let mut rev_chunk_propagate_fn: CUfunction = ptr::null_mut();
            let mut rev_chunk_resolve_fn: CUfunction = ptr::null_mut();
            cuda_check(
                cuModuleGetFunction(&mut fwd_scan_fn, module, fwd_name.as_ptr()),
                "cuModuleGetFunction(fwd)",
            )?;
            cuda_check(
                cuModuleGetFunction(&mut fwd_scan_range_fn, module, fwd_range_name.as_ptr()),
                "cuModuleGetFunction(fwd_range)",
            )?;
            cuda_check(
                cuModuleGetFunction(&mut rev_scan_fn, module, rev_name.as_ptr()),
                "cuModuleGetFunction(rev)",
            )?;
            cuda_check(
                cuModuleGetFunction(&mut rev_chunk_map_fn, module, rev_chunk_map_name.as_ptr()),
                "cuModuleGetFunction(rev_chunk_map)",
            )?;
            cuda_check(
                cuModuleGetFunction(&mut rev_chunk_propagate_fn, module, rev_chunk_propagate_name.as_ptr()),
                "cuModuleGetFunction(rev_chunk_propagate)",
            )?;
            cuda_check(
                cuModuleGetFunction(&mut rev_chunk_resolve_fn, module, rev_chunk_resolve_name.as_ptr()),
                "cuModuleGetFunction(rev_chunk_resolve)",
            )?;

            // Upload DFA tables to device
            let bufs = Self::upload_dfa(dfa)?;

            // Create a dedicated CUDA stream for all kernel launches
            let mut stream: CUstream = ptr::null_mut();
            cuda_check(cuStreamCreate(&mut stream, 0), "cuStreamCreate")?;

            // Check if reverse initial state is nullable
            let rev_initial_eid = dfa.rev_effects_id.get(dfa.initial_rev as usize).copied().unwrap_or(0);

            Ok(GpuContext {
                ctx,
                stream,
                _module: module,
                fwd_scan_fn,
                fwd_scan_range_fn,
                rev_scan_fn,
                rev_chunk_map_fn,
                rev_chunk_propagate_fn,
                rev_chunk_resolve_fn,
                bufs,
                device_info,
                fwd_initial: dfa.initial_fwd as u32,
                fwd_mt_log: dfa.mt_log,
                fwd_num_effects: dfa.fwd_effects_offsets.len() as u32,
                rev_mt_log: dfa.rev_mt_log,
                rev_num_effects: dfa.rev_effects_offsets.len() as u32,
                rev_num_states: dfa.rev_num_states as u32,
                empty_nullable: dfa.empty_nullable,
                fixed_length: dfa.fixed_length,
                rev_initial_nullable: rev_initial_eid != 0,
            })
        }
    }

    unsafe fn alloc_and_copy<T>(data: &[T]) -> Result<CUdeviceptr, String> {
        let bytes = data.len() * std::mem::size_of::<T>();
        if bytes == 0 {
            return Ok(0);
        }
        let mut dptr: CUdeviceptr = 0;
        cuda_check(cuMemAlloc_v2(&mut dptr, bytes), "cuMemAlloc")?;
        cuda_check(
            cuMemcpyHtoD_v2(dptr, data.as_ptr() as *const c_void, bytes),
            "cuMemcpyHtoD",
        )?;
        Ok(dptr)
    }

    unsafe fn upload_dfa(dfa: &DfaTables) -> Result<GpuDfaBuffers, String> {
        Ok(GpuDfaBuffers {
            // Forward DFA
            d_fwd_center: Self::alloc_and_copy(&dfa.fwd_center_table)?,
            d_fwd_begin: Self::alloc_and_copy(&dfa.fwd_begin_table)?,
            d_fwd_effects_id: Self::alloc_and_copy(&dfa.fwd_effects_id)?,
            d_fwd_effects_flat: Self::alloc_and_copy(&dfa.fwd_effects_flat)?,
            d_fwd_effects_offsets: Self::alloc_and_copy(&dfa.fwd_effects_offsets)?,
            d_fwd_minterms: Self::alloc_and_copy(&dfa.minterms_lookup)?,
            // Reverse DFA (separate minterms!)
            d_rev_center: Self::alloc_and_copy(&dfa.rev_center_table)?,
            d_rev_begin: Self::alloc_and_copy(&dfa.rev_begin_table)?,
            d_rev_effects_id: Self::alloc_and_copy(&dfa.rev_effects_id)?,
            d_rev_effects_flat: Self::alloc_and_copy(&dfa.rev_effects_flat)?,
            d_rev_effects_offsets: Self::alloc_and_copy(&dfa.rev_effects_offsets)?,
            d_rev_minterms: Self::alloc_and_copy(&dfa.rev_minterms_lookup)?,
        })
    }

    /// Find all matches using GPU kernels.
    /// Implements the RE# two-phase algorithm: reverse scan → forward scan → filter.
    /// Input is uploaded to GPU once and reused across both phases.
    /// All kernels run on a dedicated CUDA stream for minimal synchronization overhead.
    pub fn find_all(&self, dfa: &DfaTables, input: &[u8]) -> Vec<Match> {
        if input.is_empty() {
            return if self.empty_nullable { vec![Match { start: 0, end: 0 }] } else { vec![] };
        }

        // If reverse initial is nullable, every position is potentially a match start.
        if self.rev_initial_nullable {
            return self.gpu_nullable_slow(dfa, input);
        }

        // Upload input once, reuse for both phases
        let d_input = match unsafe { Self::alloc_and_copy(input) } {
            Ok(d) => d,
            Err(_) => return vec![],
        };

        // Phase 1: reverse scan on GPU to find start candidates
        let rev_hits = match self.gpu_rev_scan_with_input(input.len(), d_input) {
            Ok(h) => h,
            Err(_) => { unsafe { cuMemFree_v2(d_input); } return vec![]; },
        };
        if rev_hits.is_empty() {
            unsafe { cuMemFree_v2(d_input); }
            return vec![];
        }

        // Phase 2: forward scan on GPU from each candidate (reuses d_input)
        let fwd_ends = match self.gpu_fwd_scan_with_input(dfa, input.len(), d_input, &rev_hits) {
            Ok(e) => e,
            Err(_) => { unsafe { cuMemFree_v2(d_input); } return vec![]; },
        };

        unsafe { cuMemFree_v2(d_input); }

        // Phase 3: collect non-overlapping leftmost-longest matches
        Self::collect_matches(&rev_hits, &fwd_ends, self.fixed_length, input.len())
    }

    /// Slow path: reverse initial nullable → try forward scan from every position.
    /// Uses the range kernel to avoid allocating a huge starts[] array on device.
    fn gpu_nullable_slow(&self, dfa: &DfaTables, input: &[u8]) -> Vec<Match> {
        let fwd_ends = match self.gpu_fwd_scan_range(dfa, input) {
            Ok(e) => e,
            Err(_) => return vec![],
        };

        let mut matches = Vec::new();
        let mut pos = 0usize;
        while pos < input.len() {
            let end = fwd_ends[pos] as usize;
            if end > pos {
                matches.push(Match { start: pos, end });
                pos = end;
            } else if end == pos {
                if matches.last().map_or(true, |m: &Match| m.end <= pos) {
                    matches.push(Match { start: pos, end: pos });
                }
                pos += 1;
            } else {
                pos += 1;
            }
        }
        // Check empty match at end of input
        if self.empty_nullable && pos == input.len() {
            if matches.last().map_or(true, |m: &Match| m.end <= input.len()) {
                matches.push(Match { start: input.len(), end: input.len() });
            }
        }
        matches
    }

    fn collect_matches(starts: &[u32], ends: &[u32], fixed_length: Option<u32>, input_len: usize) -> Vec<Match> {
        let mut matches = Vec::new();
        let mut last_end = 0usize;
        for i in 0..starts.len() {
            let start = starts[i] as usize;
            let end = ends[i] as usize;
            if start < last_end { continue; }
            if let Some(fl) = fixed_length {
                let fl = fl as usize;
                if start + fl <= input_len {
                    matches.push(Match { start, end: start + fl });
                    last_end = start + fl;
                }
            } else if end > start {
                matches.push(Match { start, end });
                last_end = end;
            } else if end == start && start == 0 {
                // Zero-width match at start
                matches.push(Match { start, end: start });
                last_end = start;
            }
        }
        matches
    }

    /// GPU reverse scan: returns sorted list of potential match-start positions.
    /// Uses parallel prefix pipeline when rev_num_states <= PAR_MAX_STATES,
    /// falls back to sequential single-thread kernel otherwise.
    #[allow(dead_code)]
    fn gpu_rev_scan(&self, input: &[u8]) -> Result<Vec<u32>, String> {
        let d_input = unsafe { Self::alloc_and_copy(input)? };
        let result = self.gpu_rev_scan_with_input(input.len(), d_input);
        unsafe { cuMemFree_v2(d_input); }
        result
    }

    /// GPU reverse scan using a pre-uploaded device input buffer.
    fn gpu_rev_scan_with_input(&self, input_len: usize, d_input: CUdeviceptr) -> Result<Vec<u32>, String> {
        if self.rev_num_states <= PAR_MAX_STATES {
            self.gpu_rev_scan_parallel_impl(input_len, d_input)
        } else {
            self.gpu_rev_scan_sequential_impl(input_len, d_input)
        }
    }

    /// Parallel prefix reverse scan (3-kernel pipeline).
    fn gpu_rev_scan_parallel_impl(&self, input_len: usize, d_input: CUdeviceptr) -> Result<Vec<u32>, String> {
        let n = input_len as u32;
        let start_pos = n - 1;
        let max_hits = n;
        let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

        unsafe {
            // Allocate buffers for the 3-kernel pipeline
            let chunk_maps_bytes = (num_chunks as usize) * (PAR_MAX_STATES as usize) * 2;
            let chunk_meta_bytes = (num_chunks as usize) * 2;
            let mut d_chunk_maps: CUdeviceptr = 0;
            let mut d_chunk_initials: CUdeviceptr = 0;
            let mut d_chunk_prev_eids: CUdeviceptr = 0;
            let mut d_hits: CUdeviceptr = 0;
            let mut d_hit_count: CUdeviceptr = 0;

            cuda_check(cuMemAlloc_v2(&mut d_chunk_maps, chunk_maps_bytes), "alloc chunk_maps")?;
            cuda_check(cuMemAlloc_v2(&mut d_chunk_initials, chunk_meta_bytes), "alloc chunk_initials")?;
            cuda_check(cuMemAlloc_v2(&mut d_chunk_prev_eids, chunk_meta_bytes), "alloc chunk_prev_eids")?;
            cuda_check(cuMemAlloc_v2(&mut d_hits, input_len * 4), "alloc hits")?;
            cuda_check(cuMemAlloc_v2(&mut d_hit_count, 4), "alloc hit_count")?;
            cuda_check(cuMemsetD8_v2(d_hit_count, 0, 4), "zero hit_count")?;

            let block_size = 256u32;

            // ---- Kernel 1: chunk map ----
            let grid_map = (num_chunks + block_size - 1) / block_size;
            let mut params_map: [*mut c_void; 10] = [
                &self.bufs.d_rev_center as *const _ as *mut c_void,
                &self.bufs.d_rev_begin as *const _ as *mut c_void,
                &self.bufs.d_rev_minterms as *const _ as *mut c_void,
                &self.rev_mt_log as *const _ as *mut c_void,
                &self.rev_num_states as *const _ as *mut c_void,
                &d_input as *const _ as *mut c_void,
                &n as *const _ as *mut c_void,
                &start_pos as *const _ as *mut c_void,
                &d_chunk_maps as *const _ as *mut c_void,
                &num_chunks as *const _ as *mut c_void,
            ];

            cuda_check(cuLaunchKernel(
                self.rev_chunk_map_fn,
                grid_map, 1, 1,
                block_size, 1, 1,
                0, self.stream,
                params_map.as_mut_ptr(), ptr::null_mut(),
            ), "launch rev_chunk_map")?;
            // No sync needed — kernel 2 on same stream will wait for kernel 1

            // ---- Kernel 2: propagate (single thread) ----
            let mut params_prop: [*mut c_void; 6] = [
                &d_chunk_maps as *const _ as *mut c_void,
                &self.bufs.d_rev_effects_id as *const _ as *mut c_void,
                &num_chunks as *const _ as *mut c_void,
                &self.rev_num_states as *const _ as *mut c_void,
                &d_chunk_initials as *const _ as *mut c_void,
                &d_chunk_prev_eids as *const _ as *mut c_void,
            ];

            cuda_check(cuLaunchKernel(
                self.rev_chunk_propagate_fn,
                1, 1, 1,
                1, 1, 1,
                0, self.stream,
                params_prop.as_mut_ptr(), ptr::null_mut(),
            ), "launch rev_chunk_propagate")?;
            // No sync — kernel 3 on same stream will wait

            // ---- Kernel 3: resolve (parallel per-chunk) ----
            let grid_resolve = (num_chunks + block_size - 1) / block_size;
            let mut params_resolve: [*mut c_void; 17] = [
                &self.bufs.d_rev_center as *const _ as *mut c_void,
                &self.bufs.d_rev_begin as *const _ as *mut c_void,
                &self.bufs.d_rev_effects_id as *const _ as *mut c_void,
                &self.bufs.d_rev_effects_flat as *const _ as *mut c_void,
                &self.bufs.d_rev_effects_offsets as *const _ as *mut c_void,
                &self.rev_num_effects as *const _ as *mut c_void,
                &self.bufs.d_rev_minterms as *const _ as *mut c_void,
                &self.rev_mt_log as *const _ as *mut c_void,
                &d_input as *const _ as *mut c_void,
                &n as *const _ as *mut c_void,
                &start_pos as *const _ as *mut c_void,
                &d_chunk_initials as *const _ as *mut c_void,
                &d_chunk_prev_eids as *const _ as *mut c_void,
                &num_chunks as *const _ as *mut c_void,
                &d_hits as *const _ as *mut c_void,
                &d_hit_count as *const _ as *mut c_void,
                &max_hits as *const _ as *mut c_void,
            ];

            cuda_check(cuLaunchKernel(
                self.rev_chunk_resolve_fn,
                grid_resolve, 1, 1,
                block_size, 1, 1,
                0, self.stream,
                params_resolve.as_mut_ptr(), ptr::null_mut(),
            ), "launch rev_chunk_resolve")?;
            // Sync stream to read back results
            cuda_check(cuStreamSynchronize(self.stream), "sync rev pipeline")?;

            // Read back results
            let mut hit_count: u32 = 0;
            cuda_check(cuMemcpyDtoH_v2(
                &mut hit_count as *mut _ as *mut c_void,
                d_hit_count, 4,
            ), "read hit_count")?;

            let hit_count = hit_count.min(max_hits) as usize;
            let mut hits = vec![0u32; hit_count];
            if hit_count > 0 {
                cuda_check(cuMemcpyDtoH_v2(
                    hits.as_mut_ptr() as *mut c_void,
                    d_hits, hit_count * 4,
                ), "read hits")?;
            }

            cuMemFree_v2(d_chunk_maps);
            cuMemFree_v2(d_chunk_initials);
            cuMemFree_v2(d_chunk_prev_eids);
            cuMemFree_v2(d_hits);
            cuMemFree_v2(d_hit_count);

            hits.sort_unstable();
            Ok(hits)
        }
    }

    /// Sequential reverse scan (fallback for large state counts).
    #[allow(dead_code)]
    fn gpu_rev_scan_sequential(&self, input: &[u8]) -> Result<Vec<u32>, String> {
        let d_input = unsafe { Self::alloc_and_copy(input)? };
        let result = self.gpu_rev_scan_sequential_impl(input.len(), d_input);
        unsafe { cuMemFree_v2(d_input); }
        result
    }

    fn gpu_rev_scan_sequential_impl(&self, input_len: usize, d_input: CUdeviceptr) -> Result<Vec<u32>, String> {
        let n = input_len as u32;
        let start_pos = n - 1;
        let max_hits = n;

        unsafe {
            let mut d_hits: CUdeviceptr = 0;
            let mut d_hit_count: CUdeviceptr = 0;
            cuda_check(cuMemAlloc_v2(&mut d_hits, input_len * 4), "alloc hits")?;
            cuda_check(cuMemAlloc_v2(&mut d_hit_count, 4), "alloc hit_count")?;
            cuda_check(cuMemsetD8_v2(d_hit_count, 0, 4), "zero hit_count")?;

            // Single-thread kernel (sequential reverse scan)
            let mut params: [*mut c_void; 14] = [
                &self.bufs.d_rev_center as *const _ as *mut c_void,
                &self.bufs.d_rev_begin as *const _ as *mut c_void,
                &self.bufs.d_rev_effects_id as *const _ as *mut c_void,
                &self.bufs.d_rev_effects_flat as *const _ as *mut c_void,
                &self.bufs.d_rev_effects_offsets as *const _ as *mut c_void,
                &self.rev_num_effects as *const _ as *mut c_void,
                &self.bufs.d_rev_minterms as *const _ as *mut c_void,
                &self.rev_mt_log as *const _ as *mut c_void,
                &d_input as *const _ as *mut c_void,
                &n as *const _ as *mut c_void,
                &start_pos as *const _ as *mut c_void,
                &d_hits as *const _ as *mut c_void,
                &d_hit_count as *const _ as *mut c_void,
                &max_hits as *const _ as *mut c_void,
            ];

            cuda_check(cuLaunchKernel(
                self.rev_scan_fn,
                1, 1, 1,
                1, 1, 1,
                0, self.stream,
                params.as_mut_ptr(), ptr::null_mut(),
            ), "launch rev_scan")?;
            cuda_check(cuStreamSynchronize(self.stream), "sync rev_scan")?;

            let mut hit_count: u32 = 0;
            cuda_check(cuMemcpyDtoH_v2(
                &mut hit_count as *mut _ as *mut c_void,
                d_hit_count, 4,
            ), "read hit_count")?;

            let hit_count = hit_count.min(max_hits) as usize;
            let mut hits = vec![0u32; hit_count];
            if hit_count > 0 {
                cuda_check(cuMemcpyDtoH_v2(
                    hits.as_mut_ptr() as *mut c_void,
                    d_hits, hit_count * 4,
                ), "read hits")?;
            }

            cuMemFree_v2(d_hits);
            cuMemFree_v2(d_hit_count);

            hits.sort_unstable();
            Ok(hits)
        }
    }

    /// GPU forward scan: for each start position, find the match end.
    #[allow(dead_code)]
    fn gpu_fwd_scan(&self, dfa: &DfaTables, input: &[u8], starts: &[u32]) -> Result<Vec<u32>, String> {
        let d_input = unsafe { Self::alloc_and_copy(input)? };
        let result = self.gpu_fwd_scan_with_input(dfa, input.len(), d_input, starts);
        unsafe { cuMemFree_v2(d_input); }
        result
    }

    /// GPU forward scan with pre-uploaded input buffer.
    fn gpu_fwd_scan_with_input(&self, dfa: &DfaTables, input_len: usize, d_input: CUdeviceptr, starts: &[u32]) -> Result<Vec<u32>, String> {
        let n = input_len as u32;
        let num_starts = starts.len() as u32;
        let fwd_initial = dfa.initial_fwd as u32;

        unsafe {
            let d_starts = Self::alloc_and_copy(starts)?;
            let mut d_ends: CUdeviceptr = 0;
            cuda_check(cuMemAlloc_v2(&mut d_ends, starts.len() * 4), "alloc ends")?;
            cuda_check(cuMemsetD8_v2(d_ends, 0, starts.len() * 4), "zero ends")?;

            let block_size = 256u32;
            let grid_size = (num_starts + block_size - 1) / block_size;

            let mut params: [*mut c_void; 14] = [
                &self.bufs.d_fwd_center as *const _ as *mut c_void,
                &self.bufs.d_fwd_begin as *const _ as *mut c_void,
                &self.bufs.d_fwd_effects_id as *const _ as *mut c_void,
                &self.bufs.d_fwd_effects_flat as *const _ as *mut c_void,
                &self.bufs.d_fwd_effects_offsets as *const _ as *mut c_void,
                &self.fwd_num_effects as *const _ as *mut c_void,
                &self.bufs.d_fwd_minterms as *const _ as *mut c_void,
                &self.fwd_mt_log as *const _ as *mut c_void,
                &d_input as *const _ as *mut c_void,
                &n as *const _ as *mut c_void,
                &fwd_initial as *const _ as *mut c_void,
                &d_starts as *const _ as *mut c_void,
                &d_ends as *const _ as *mut c_void,
                &num_starts as *const _ as *mut c_void,
            ];

            cuda_check(cuLaunchKernel(
                self.fwd_scan_fn,
                grid_size, 1, 1,
                block_size, 1, 1,
                0, self.stream,
                params.as_mut_ptr(), ptr::null_mut(),
            ), "launch fwd_scan")?;
            cuda_check(cuStreamSynchronize(self.stream), "sync fwd_scan")?;

            let mut ends = vec![0u32; starts.len()];
            cuda_check(cuMemcpyDtoH_v2(
                ends.as_mut_ptr() as *mut c_void,
                d_ends, starts.len() * 4,
            ), "read ends")?;

            cuMemFree_v2(d_starts);
            cuMemFree_v2(d_ends);

            Ok(ends)
        }
    }

    /// GPU forward scan for range [0, input_len): used by nullable-slow path.
    /// Uses dfa_fwd_scan_range kernel — no starts[] array needed on device.
    fn gpu_fwd_scan_range(&self, dfa: &DfaTables, input: &[u8]) -> Result<Vec<u32>, String> {
        let n = input.len() as u32;
        let num_positions = n;
        let fwd_initial = dfa.initial_fwd as u32;

        unsafe {
            let d_input = Self::alloc_and_copy(input)?;
            let mut d_ends: CUdeviceptr = 0;
            cuda_check(cuMemAlloc_v2(&mut d_ends, input.len() * 4), "alloc ends")?;
            cuda_check(cuMemsetD8_v2(d_ends, 0, input.len() * 4), "zero ends")?;

            let block_size = 256u32;
            let grid_size = (num_positions + block_size - 1) / block_size;

            let mut params: [*mut c_void; 13] = [
                &self.bufs.d_fwd_center as *const _ as *mut c_void,
                &self.bufs.d_fwd_begin as *const _ as *mut c_void,
                &self.bufs.d_fwd_effects_id as *const _ as *mut c_void,
                &self.bufs.d_fwd_effects_flat as *const _ as *mut c_void,
                &self.bufs.d_fwd_effects_offsets as *const _ as *mut c_void,
                &self.fwd_num_effects as *const _ as *mut c_void,
                &self.bufs.d_fwd_minterms as *const _ as *mut c_void,
                &self.fwd_mt_log as *const _ as *mut c_void,
                &d_input as *const _ as *mut c_void,
                &n as *const _ as *mut c_void,
                &fwd_initial as *const _ as *mut c_void,
                &d_ends as *const _ as *mut c_void,
                &num_positions as *const _ as *mut c_void,
            ];

            cuda_check(cuLaunchKernel(
                self.fwd_scan_range_fn,
                grid_size, 1, 1,
                block_size, 1, 1,
                0, self.stream,
                params.as_mut_ptr(), ptr::null_mut(),
            ), "launch fwd_scan_range")?;
            cuda_check(cuStreamSynchronize(self.stream), "sync fwd_scan_range")?;

            let mut ends = vec![0u32; input.len()];
            cuda_check(cuMemcpyDtoH_v2(
                ends.as_mut_ptr() as *mut c_void,
                d_ends, input.len() * 4,
            ), "read ends")?;

            cuMemFree_v2(d_input);
            cuMemFree_v2(d_ends);

            Ok(ends)
        }
    }

    pub fn is_match(&self, dfa: &DfaTables, input: &[u8]) -> bool {
        !self.find_all(dfa, input).is_empty()
    }

    pub fn find_anchored(&self, dfa: &DfaTables, input: &[u8]) -> Option<Match> {
        self.find_all(dfa, input).into_iter().find(|m| m.start == 0)
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let b = &self.bufs;
            for &ptr in &[
                b.d_fwd_center, b.d_fwd_begin, b.d_fwd_effects_id,
                b.d_fwd_effects_flat, b.d_fwd_effects_offsets, b.d_fwd_minterms,
                b.d_rev_center, b.d_rev_begin, b.d_rev_effects_id,
                b.d_rev_effects_flat, b.d_rev_effects_offsets, b.d_rev_minterms,
            ] {
                if ptr != 0 { cuMemFree_v2(ptr); }
            }
            if !self.stream.is_null() { cuStreamDestroy_v2(self.stream); }
            if !self.ctx.is_null() { cuCtxDestroy_v2(self.ctx); }
        }
    }
}
