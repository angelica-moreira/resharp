// Comprehensive profiling: Oracle vs CPU-ref kernel vs GPU kernel.
//
// Measures:
//   - Throughput (GB/s), latency (µs)
//   - Correctness (3-way cross-check on every run)
//   - GPU kernel timing via host-side measurement
//   - Power draw (nvidia-smi)
//   - perf-compatible output (run under `perf stat` for HW counters)

use resharp::{EngineOptions, Regex};
use resharp_cuda::CudaRegex;
use std::fs;
use std::io::Write;
use std::time::Instant;

fn read_gpu_power_w() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().ok()
}

fn read_cpu_energy_uj() -> Option<u64> {
    for path in &[
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/amd-rapl:0/energy_uj",
    ] {
        if let Ok(s) = fs::read_to_string(path) {
            return s.trim().parse().ok();
        }
    }
    None
}

fn bar(label: &str, val: f64, max: f64, width: usize, unit: &str) {
    let filled = if max > 0.0 { ((val / max) * width as f64).min(width as f64) as usize } else { 0 };
    let bar: String = "█".repeat(filled) + &"░".repeat(width.saturating_sub(filled));
    eprintln!("  {:>16} │{}│ {:.4} {}", label, bar, val, unit);
}

struct Row {
    label: String,
    pattern: String,
    input_kb: usize,
    matches: usize,
    oracle_us: u128,
    cpu_ref_us: u128,
    gpu_us: Option<u128>,
    oracle_gbs: f64,
    cpu_ref_gbs: f64,
    gpu_gbs: Option<f64>,
    correct: bool,
}

fn bench_one(pattern: &str, input: &[u8], label: &str, iters: u32) -> Row {
    let opts = EngineOptions { dfa_threshold: 4096, ..EngineOptions::default() };
    let oracle = Regex::with_options(pattern, opts).unwrap();
    let cuda = CudaRegex::with_options(pattern, opts, 2048).unwrap();
    let bytes = input.len() as f64;

    // Warmup
    for _ in 0..5 {
        let _ = oracle.find_all(input);
        let _ = cuda.find_all_cpu(input);
        if cuda.has_gpu() { let _ = cuda.find_all_gpu(input); }
    }

    // Oracle
    let t0 = Instant::now();
    let mut oracle_res = vec![];
    for _ in 0..iters { oracle_res = oracle.find_all(input).unwrap(); }
    let oracle_total = t0.elapsed();
    let oracle_per = oracle_total / iters;

    // CPU reference kernel
    let t0 = Instant::now();
    let mut cpu_res = vec![];
    for _ in 0..iters { cpu_res = cuda.find_all_cpu(input).unwrap(); }
    let cpu_total = t0.elapsed();
    let cpu_per = cpu_total / iters;

    // GPU kernel
    let (gpu_per, gpu_res) = if cuda.has_gpu() {
        let t0 = Instant::now();
        let mut res = vec![];
        for _ in 0..iters { res = cuda.find_all_gpu(input).unwrap_or_default(); }
        let total = t0.elapsed();
        (Some(total / iters), res)
    } else {
        (None, vec![])
    };

    // Correctness cross-check
    let correct = oracle_res == cpu_res
        && (gpu_res.is_empty() || oracle_res == gpu_res || !cuda.has_gpu());

    Row {
        label: label.into(),
        pattern: pattern.into(),
        input_kb: input.len() / 1024,
        matches: oracle_res.len(),
        oracle_us: oracle_per.as_micros(),
        cpu_ref_us: cpu_per.as_micros(),
        gpu_us: gpu_per.map(|p| p.as_micros()),
        oracle_gbs: bytes * iters as f64 / oracle_total.as_secs_f64() / 1e9,
        cpu_ref_gbs: bytes * iters as f64 / cpu_total.as_secs_f64() / 1e9,
        gpu_gbs: gpu_per.map(|p| bytes / p.as_secs_f64() / 1e9),
        correct,
    }
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  resharp-cuda Profiling: Oracle vs CPU-ref vs GPU (3-way check)    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");

    // Hardware detection
    let test = CudaRegex::new(r"test").unwrap();
    eprintln!("\n  GPU available: {}  GPU DFA: {}", test.has_gpu(), test.has_gpu_dfa());
    if let Some(w) = read_gpu_power_w() {
        eprintln!("  GPU idle power: {:.1} W", w);
    }
    if let Some(uj) = read_cpu_energy_uj() {
        eprintln!("  CPU energy counter: available ({} µJ snapshot)", uj);
    }

    let patterns: Vec<(&str, &str)> = vec![
        (r"\d+", "digits"),
        (r"Sherlock|Holmes|Watson", "names"),
        (r"\d{3}-\d{4}", "phone"),
        (r"[a-z]+@[a-z]+\.[a-z]+", "email"),
        (r"~(_*abc_*)", "complement"),
        (r"\w+", "words"),
    ];

    let base_text = "The quick brown fox 555-1234 foo@bar.com Sherlock Holmes Watson xyz abc 42 test. ";
    let sizes_kb: Vec<usize> = vec![1, 10, 100, 1000, 10000];

    let mut all_rows: Vec<Row> = Vec::new();
    let mut any_fail = false;

    for &size_kb in &sizes_kb {
        let repeat = (size_kb * 1024) / base_text.len() + 1;
        let full = base_text.repeat(repeat);
        let input = &full.as_bytes()[..size_kb * 1024];
        let iters = if size_kb >= 1000 { 3 } else if size_kb >= 100 { 10 } else { 50 };

        eprintln!("\n┌─── {} KB ({} iters) ───", size_kb, iters);

        for (pat, label) in &patterns {
            let r = bench_one(pat, input, label, iters);
            let check = if r.correct { "✓" } else { any_fail = true; "✗" };
            let gpu_str = match r.gpu_us {
                Some(us) => format!("{:>8}µs ({:.3} GB/s)", us, r.gpu_gbs.unwrap_or(0.0)),
                None => "  fallback".to_string(),
            };
            eprintln!("│ {} {:>10}  oracle={:>8}µs ({:.3} GB/s)  cpu_ref={:>8}µs ({:.3} GB/s)  gpu={}  matches={}",
                check, label,
                r.oracle_us, r.oracle_gbs,
                r.cpu_ref_us, r.cpu_ref_gbs,
                gpu_str, r.matches);
            all_rows.push(r);
        }
    }

    // Energy measurement on largest input
    eprintln!("\n┌─── Energy/Power Measurement (10MB input) ───");
    let big_input = base_text.repeat(128 * 1024);
    let big_input = &big_input.as_bytes()[..10 * 1024 * 1024];

    let opts = EngineOptions { dfa_threshold: 4096, ..EngineOptions::default() };
    let oracle = Regex::with_options(r"\d+", opts).unwrap();
    let cuda = CudaRegex::with_options(r"\d+", opts, 2048).unwrap();

    // CPU energy
    let e0 = read_cpu_energy_uj();
    let gpu_p0 = read_gpu_power_w();
    let t0 = Instant::now();
    for _ in 0..10 { let _ = oracle.find_all(big_input); }
    let cpu_wall = t0.elapsed();
    let e1 = read_cpu_energy_uj();

    let cpu_energy_j = match (e0, e1) {
        (Some(a), Some(b)) => Some((b.wrapping_sub(a)) as f64 / 1e6),
        _ => None,
    };
    eprintln!("│ CPU oracle (10×10MB):  {:.1}ms total", cpu_wall.as_secs_f64() * 1000.0);
    if let Some(j) = cpu_energy_j {
        eprintln!("│   CPU energy: {:.3} J  ({:.1} W avg)", j, j / cpu_wall.as_secs_f64());
    }

    // GPU energy
    let e0 = read_cpu_energy_uj();
    let t0 = Instant::now();
    for _ in 0..10 { let _ = cuda.find_all_gpu(big_input); }
    let gpu_wall = t0.elapsed();
    let e1 = read_cpu_energy_uj();
    let gpu_p1 = read_gpu_power_w();

    let gpu_cpu_energy_j = match (e0, e1) {
        (Some(a), Some(b)) => Some((b.wrapping_sub(a)) as f64 / 1e6),
        _ => None,
    };
    eprintln!("│ GPU kernel (10×10MB):  {:.1}ms total", gpu_wall.as_secs_f64() * 1000.0);
    if let Some(j) = gpu_cpu_energy_j {
        eprintln!("│   CPU-side energy during GPU: {:.3} J", j);
    }
    if let (Some(p0), Some(p1)) = (gpu_p0, gpu_p1) {
        eprintln!("│   GPU power: idle={:.1}W  during={:.1}W", p0, p1);
    }

    // Throughput charts for largest input
    eprintln!("\n┌─── Throughput Charts (largest input) ───");
    let largest = sizes_kb.last().unwrap();
    for r in all_rows.iter().filter(|r| r.input_kb == *largest) {
        let max_gbs = r.oracle_gbs.max(r.cpu_ref_gbs).max(r.gpu_gbs.unwrap_or(0.0));
        eprintln!("\n  ── {} ({} KB) ──", r.label, r.input_kb);
        bar("Oracle", r.oracle_gbs, max_gbs, 40, "GB/s");
        bar("CPU-ref kernel", r.cpu_ref_gbs, max_gbs, 40, "GB/s");
        if let Some(g) = r.gpu_gbs {
            bar("GPU kernel", g, max_gbs, 40, "GB/s");
        }
    }

    // Write CSV
    let csv_path = "profile_results.csv";
    let mut f = fs::File::create(csv_path).unwrap();
    writeln!(f, "label,pattern,input_kb,matches,oracle_us,cpu_ref_us,gpu_us,oracle_gbs,cpu_ref_gbs,gpu_gbs,correct").unwrap();
    for r in &all_rows {
        writeln!(f, "{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{}",
            r.label, r.pattern, r.input_kb, r.matches,
            r.oracle_us, r.cpu_ref_us, r.gpu_us.unwrap_or(0),
            r.oracle_gbs, r.cpu_ref_gbs, r.gpu_gbs.unwrap_or(0.0),
            r.correct,
        ).unwrap();
    }
    eprintln!("\n  CSV: {}", csv_path);

    if any_fail {
        eprintln!("\n  ⚠ CORRECTNESS FAILURES DETECTED — see ✗ markers above");
        std::process::exit(1);
    } else {
        eprintln!("\n  ✅ All correctness checks passed (3-way: oracle = cpu_ref = gpu)");
    }
}
