// Standalone GPU exerciser for use with CUDA validation tools:
//   compute-sanitizer --tool memcheck   (memory errors)
//   compute-sanitizer --tool racecheck  (race conditions)
//   compute-sanitizer --tool initcheck  (uninitialized memory)
//   ncu                                 (kernel profiling)
//   nsys profile                        (system-wide profiling)
//
// Usage:
//   cargo build --release --example gpu_sanitizer -p resharp-cuda
//   compute-sanitizer ./target/release/examples/gpu_sanitizer

use resharp::{EngineOptions, Regex};
use resharp_cuda::CudaRegex;

fn check(pattern: &str, input: &[u8], label: &str) {
    let opts = EngineOptions { dfa_threshold: 2048, ..EngineOptions::default() };
    let oracle = Regex::with_options(pattern, opts).unwrap();
    let cuda = CudaRegex::new(pattern).unwrap();

    let oracle_res = oracle.find_all(input).unwrap();
    let cpu_res = cuda.find_all_cpu(input).unwrap();
    let gpu_res = cuda.find_all_gpu(input);

    let ok = match &gpu_res {
        Some(g) => oracle_res == *g && oracle_res == cpu_res,
        None => oracle_res == cpu_res, // fallback pattern
    };

    let mark = if ok { "✓" } else { "✗" };
    eprintln!("{} {:>12} | oracle={:>4} cpu={:>4} gpu={:>4} | {}",
        mark, label, oracle_res.len(), cpu_res.len(),
        gpu_res.as_ref().map(|g| g.len() as i64).unwrap_or(-1),
        pattern);
    
    if !ok {
        eprintln!("  MISMATCH! oracle={:?}", &oracle_res[..oracle_res.len().min(5)]);
        eprintln!("            cpu   ={:?}", &cpu_res[..cpu_res.len().min(5)]);
        if let Some(g) = &gpu_res {
            eprintln!("            gpu   ={:?}", &g[..g.len().min(5)]);
        }
    }
}

fn main() {
    eprintln!("=== GPU Sanitizer Exercise (for compute-sanitizer / ncu / nsys) ===\n");

    let test_re = CudaRegex::new(r"test").unwrap();
    eprintln!("GPU available: {}\n", test_re.has_gpu());

    // Small inputs (catch edge cases)
    eprintln!("--- Small inputs ---");
    check(r"\d+", b"abc 123 def 456", "digits-s");
    check(r"hello", b"say hello world hello", "literal-s");
    check(r"[a-z]+@[a-z]+\.[a-z]+", b"foo@bar.com baz@qux.org", "email-s");
    check(r"\d{3}-\d{4}", b"call 555-1234 or 555-5678", "phone-s");
    check(r"~(_*abc_*)", b"xyz", "complement-s");
    check(r"cat|dog", b"the cat and the dog", "alt-s");
    check(r"a{2,4}", b"a aa aaa aaaa aaaaa", "quant-s");
    check(r".", b"x", "single-s");
    check(r"\d+", b"", "empty-s");

    // Medium inputs (stress test GPU kernels)
    eprintln!("\n--- Medium inputs (~100KB) ---");
    let med = "The quick brown fox 555-1234 foo@bar.com abc xyz ".repeat(2200);
    let med_bytes = med.as_bytes();
    let med = &med_bytes[..med_bytes.len().min(100 * 1024)];
    check(r"\d+", med, "digits-m");
    check(r"fox", med, "literal-m");
    check(r"[a-z]+@[a-z]+\.[a-z]+", med, "email-m");
    check(r"\d{3}-\d{4}", med, "phone-m");
    check(r"~(_*abc_*)", med, "complement-m");
    check(r"\w+", med, "words-m");

    // Large inputs (exercise memory allocation paths)
    eprintln!("\n--- Large inputs (~1MB) ---");
    let big = "The quick brown fox 555-1234 foo@bar.com abc xyz ".repeat(22000);
    let big_bytes = big.as_bytes();
    let big = &big_bytes[..big_bytes.len().min(1024 * 1024)];
    check(r"\d+", big, "digits-l");
    check(r"fox", big, "literal-l");
    check(r"[a-z]+@[a-z]+\.[a-z]+", big, "email-l");
    check(r"~(_*abc_*)", big, "complement-l");

    // Anchor/lookaround (GPU fallback path)
    eprintln!("\n--- Fallback patterns (anchors/lookarounds) ---");
    check(r"^hello", b"hello world", "anchor-s");
    check(r"(?<=\$)\d+", b"costs $50 or $100", "lookbehind-s");

    eprintln!("\nDone. All patterns exercised.");
}
