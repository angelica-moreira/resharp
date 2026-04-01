// CPU vs GPU benchmark with throughput, latency, power/energy measurement.
// Generates CSV + text charts for analysis.

use resharp::{Regex, EngineOptions, Match};
use resharp_cuda::CudaRegex;
use std::time::{Instant, Duration};
use std::fs;
use std::io::Write;

struct BenchResult {
    label: String,
    pattern: String,
    input_kb: usize,
    match_count: usize,
    cpu_time: Duration,
    cuda_cpu_time: Duration,
    cuda_gpu_time: Option<Duration>,
    cpu_throughput_gbs: f64,
    cuda_cpu_throughput_gbs: f64,
    cuda_gpu_throughput_gbs: Option<f64>,
}

fn bench_pattern(pattern: &str, input: &[u8], label: &str, iters: u32) -> BenchResult {
    let opts = EngineOptions { dfa_threshold: 4096, ..EngineOptions::default() };
    let cpu = Regex::with_options(pattern, opts).unwrap();
    let cuda = CudaRegex::with_options(pattern, opts, 2048).unwrap();

    // Warmup
    for _ in 0..3 { let _ = cpu.find_all(input); let _ = cuda.find_all_cpu(input); }

    // CPU (original resharp)
    let t0 = Instant::now();
    let mut cpu_count = 0;
    for _ in 0..iters { cpu_count = cpu.find_all(input).unwrap().len(); }
    let cpu_total = t0.elapsed();
    let cpu_per = cpu_total / iters;

    // CUDA CPU-ref path (our DFA extraction + CPU kernel)
    let t0 = Instant::now();
    let mut cuda_cpu_count = 0;
    for _ in 0..iters { cuda_cpu_count = cuda.find_all_cpu(input).unwrap().len(); }
    let cuda_cpu_total = t0.elapsed();
    let cuda_cpu_per = cuda_cpu_total / iters;

    // CUDA GPU path
    let (cuda_gpu_per, cuda_gpu_count) = if cuda.has_gpu() {
        // warmup GPU
        for _ in 0..3 { let _ = cuda.find_all_gpu(input); }
        let t0 = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = cuda.find_all_gpu(input).unwrap_or_default().len(); }
        let total = t0.elapsed();
        (Some(total / iters), c)
    } else {
        (None, 0)
    };

    let bytes = input.len() as f64;
    let cpu_gbs = bytes * iters as f64 / cpu_total.as_secs_f64() / 1e9;
    let cuda_cpu_gbs = bytes * iters as f64 / cuda_cpu_total.as_secs_f64() / 1e9;
    let cuda_gpu_gbs = cuda_gpu_per.map(|p| bytes / p.as_secs_f64() / 1e9);

    BenchResult {
        label: label.to_string(),
        pattern: pattern.to_string(),
        input_kb: input.len() / 1024,
        match_count: cpu_count,
        cpu_time: cpu_per,
        cuda_cpu_time: cuda_cpu_per,
        cuda_gpu_time: cuda_gpu_per,
        cpu_throughput_gbs: cpu_gbs,
        cuda_cpu_throughput_gbs: cuda_cpu_gbs,
        cuda_gpu_throughput_gbs: cuda_gpu_gbs,
    }
}

fn read_cpu_energy() -> Option<u64> {
    // Intel RAPL or AMD energy via powercap
    for path in &[
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
    ] {
        if let Ok(s) = fs::read_to_string(path) {
            return s.trim().parse().ok();
        }
    }
    None
}

fn read_gpu_power_mw() -> Option<u32> {
    // nvidia-smi query
    let out = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"])
        .output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim().parse::<f64>().ok().map(|w| (w * 1000.0) as u32)
}

fn bar_chart(label: &str, values: &[(&str, f64)], unit: &str) {
    let max_val = values.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
    let bar_width = 40;
    eprintln!("\n  === {} ===", label);
    for (name, val) in values {
        let filled = if max_val > 0.0 { (val / max_val * bar_width as f64) as usize } else { 0 };
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
        eprintln!("  {:>14} │{}│ {:.3} {}", name, bar, val, unit);
    }
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║       resharp-cuda: CPU vs GPU Benchmark Suite             ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    // Check GPU availability
    let test_re = CudaRegex::new(r"test").unwrap();
    eprintln!("  GPU DFA available: {}", test_re.has_gpu_dfa());
    eprintln!("  GPU context ready: {}", test_re.has_gpu());

    let sizes: Vec<usize> = vec![64, 256, 1024, 4096];  // KB

    let patterns: Vec<(&str, &str)> = vec![
        (r"Sherlock", "literal"),
        (r"\d+", "digits"),
        (r"\w+", "words"),
        (r"\d{3}-\d{4}", "phone"),
        (r"[a-z]+@[a-z]+\.[a-z]+", "email"),
        (r"the|and|for|that|have", "altern"),
    ];

    let base = "The quick brown fox jumps over the lazy dog. Sherlock Holmes 555-1234 foo@bar.com and that have for ";

    let mut all_results = Vec::new();

    for &size_kb in &sizes {
        let repeat = (size_kb * 1024) / base.len() + 1;
        let input: Vec<u8> = base.repeat(repeat).into_bytes();
        let input = &input[..size_kb * 1024];

        eprintln!("\n─── Input size: {} KB ───", size_kb);

        let iters = if size_kb >= 1024 { 5 } else { 20 };
        for (pat, label) in &patterns {
            let r = bench_pattern(pat, input, label, iters);
            eprintln!("  [{:8}] cpu={:>8.3}ms  cuda_cpu={:>8.3}ms  gpu={:>8}  matches={:>6}  cpu={:.2}GB/s  cuda_cpu={:.2}GB/s  gpu={:.2}GB/s",
                r.label,
                r.cpu_time.as_secs_f64() * 1000.0,
                r.cuda_cpu_time.as_secs_f64() * 1000.0,
                r.cuda_gpu_time.map(|t| format!("{:.3}ms", t.as_secs_f64()*1000.0)).unwrap_or("N/A".into()),
                r.match_count,
                r.cpu_throughput_gbs,
                r.cuda_cpu_throughput_gbs,
                r.cuda_gpu_throughput_gbs.unwrap_or(0.0),
            );
            all_results.push(r);
        }
    }

    // Power snapshot
    eprintln!("\n─── Power/Energy Snapshot ───");
    if let Some(gpu_mw) = read_gpu_power_mw() {
        eprintln!("  GPU power draw: {:.1} W", gpu_mw as f64 / 1000.0);
    }
    if let Some(cpu_uj) = read_cpu_energy() {
        eprintln!("  CPU energy counter: {} µJ", cpu_uj);
    }

    // Throughput charts for largest input
    let largest = sizes.last().unwrap();
    let large_results: Vec<&BenchResult> = all_results.iter()
        .filter(|r| r.input_kb == *largest)
        .collect();

    for r in &large_results {
        let mut vals: Vec<(&str, f64)> = vec![
            ("CPU (resharp)", r.cpu_throughput_gbs),
            ("CUDA CPU-ref", r.cuda_cpu_throughput_gbs),
        ];
        if let Some(g) = r.cuda_gpu_throughput_gbs {
            vals.push(("CUDA GPU", g));
        }
        bar_chart(&format!("{} @ {}KB", r.label, r.input_kb), &vals, "GB/s");
    }

    // Write CSV
    let csv_path = "bench_results.csv";
    let mut f = fs::File::create(csv_path).unwrap();
    writeln!(f, "label,pattern,input_kb,matches,cpu_us,cuda_cpu_us,cuda_gpu_us,cpu_gbs,cuda_cpu_gbs,cuda_gpu_gbs").unwrap();
    for r in &all_results {
        writeln!(f, "{},{},{},{},{},{},{},{:.4},{:.4},{:.4}",
            r.label, r.pattern, r.input_kb, r.match_count,
            r.cpu_time.as_micros(), r.cuda_cpu_time.as_micros(),
            r.cuda_gpu_time.map(|t| t.as_micros()).unwrap_or(0),
            r.cpu_throughput_gbs, r.cuda_cpu_throughput_gbs,
            r.cuda_gpu_throughput_gbs.unwrap_or(0.0),
        ).unwrap();
    }
    eprintln!("\n  CSV written to {}", csv_path);
    eprintln!("  Done.");
}
