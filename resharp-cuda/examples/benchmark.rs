use resharp::{Regex, EngineOptions};
use std::time::Instant;

fn bench_pattern(label: &str, pat: &str, input: &[u8], iterations: usize) {
    let opts = EngineOptions { dfa_threshold: 4096, ..EngineOptions::default() };
    let re = Regex::with_options(pat, opts).unwrap();
    let dfa = re.extract_dfa_tables().unwrap();
    
    // Verify correctness first
    let oracle = re.find_all(input).unwrap();
    let kernel = resharp_cuda::kernel::cpu_find_all(&dfa, input);
    let correct = oracle == kernel;
    let uses_fallback = dfa.has_look || dfa.has_anchors;
    
    // Benchmark CPU oracle
    let t0 = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(re.find_all(input).unwrap());
    }
    let cpu_ns = t0.elapsed().as_nanos() as f64 / iterations as f64;
    
    // Benchmark our kernel (or fallback)
    let cuda_re = resharp_cuda::CudaRegex::with_options(pat, opts, 4096).unwrap();
    let t1 = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(cuda_re.find_all(input).unwrap());
    }
    let kernel_ns = t1.elapsed().as_nanos() as f64 / iterations as f64;
    
    let throughput_cpu = input.len() as f64 / cpu_ns * 1e3; // MB/s
    let throughput_kernel = input.len() as f64 / kernel_ns * 1e3;
    let ratio = kernel_ns / cpu_ns;
    let fallback = if uses_fallback { " [FALLBACK]" } else { "" };
    let correct_str = if correct { "✓" } else { "✗ MISMATCH" };
    
    eprintln!("{:25} {} matches={:4} cpu={:8.0}ns ({:6.0} MB/s)  kernel={:8.0}ns ({:6.0} MB/s)  ratio={:.2}x{}",
        label, correct_str, oracle.len(), cpu_ns, throughput_cpu, kernel_ns, throughput_kernel, ratio, fallback);
}

fn main() {
    eprintln!("=== CPU Oracle vs resharp-cuda Kernel Benchmark ===\n");
    
    // Generate test inputs
    let small = "The quick brown fox jumps over 42 lazy dogs. Email: test@example.com\n".repeat(100);
    let medium = small.repeat(10);
    let large = medium.repeat(10);
    
    eprintln!("Input sizes: small={}B medium={}B large={}B\n", small.len(), medium.len(), large.len());
    
    let patterns = vec![
        ("digits", r"\d+"),
        ("alpha-words", r"[a-z]+"),
        ("email-like", r"\w+@\w+\.\w+"),
        ("hex", r"[a-f0-9]+"),
        ("fixed-3digit", r"\d{3}"),
        ("no-consec-dig", r"~(_*\d\d_*)"),  // complement!
        ("intersection", r"_*fox_*&_*dog_*"),  // intersection!
        ("anchor-start", "^The"),
        ("lookbehind", r"(?<=\$)\d+"),
    ];
    
    for (label, pat) in &patterns {
        // Small input, many iterations
        bench_pattern(&format!("{} (small)", label), pat, small.as_bytes(), 5000);
    }
    
    eprintln!();
    for (label, pat) in &patterns[..7] {  // Only non-fallback for large
        bench_pattern(&format!("{} (large)", label), pat, large.as_bytes(), 100);
    }
}
