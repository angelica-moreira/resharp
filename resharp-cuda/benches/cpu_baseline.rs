use resharp::{Regex, EngineOptions};
use std::time::Instant;

fn main() {
    let patterns: Vec<(&str, &str)> = vec![
        (r"Sherlock", "literal"),
        (r"\w+", "word"),
        (r"\d{3}-\d{4}", "phone"),
        (r"[a-z]+@[a-z]+\.[a-z]+", "email"),
        (r"the|and|for|that|have", "alternation"),
    ];

    let base = "The quick brown fox jumps over the lazy dog. Sherlock Holmes 555-1234 foo@bar.com ";
    let input: Vec<u8> = base.repeat(13000).into_bytes();
    eprintln!("Input size: {} bytes ({:.1} MB)", input.len(), input.len() as f64 / 1048576.0);

    for (pat, label) in &patterns {
        let opts = EngineOptions { dfa_threshold: 2048, ..EngineOptions::default() };
        let re = Regex::with_options(pat, opts).unwrap();
        // warmup
        for _ in 0..5 { let _ = re.find_all(&input); }
        let start = Instant::now();
        let iters = 20u32;
        let mut count = 0usize;
        for _ in 0..iters {
            count = re.find_all(&input).unwrap().len();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters;
        let throughput = (input.len() as f64 * iters as f64) / elapsed.as_secs_f64() / 1e9;
        eprintln!("[{:12}] {:?}/iter  {:.2} GB/s  {} matches  pattern: {}", 
            label, per_iter, throughput, count, pat);
    }
}
