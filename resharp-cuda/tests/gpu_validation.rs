// GPU cross-validation tests: force GPU path and compare against
// both the CPU reference kernel AND the resharp oracle.
//
// Three-way comparison for every pattern:
//   1. resharp::Regex (oracle)
//   2. resharp_cuda::kernel::cpu_find_all (CPU reference)
//   3. GpuContext::find_all (actual GPU CUDA kernels)

use resharp::{EngineOptions, Match, Regex};
use resharp_cuda::CudaRegex;

/// Three-way cross-check: oracle vs CPU-kernel vs GPU-kernel.
fn cross_check(pattern: &str, input: &[u8], label: &str) {
    let opts = EngineOptions { dfa_threshold: 2048, ..EngineOptions::default() };
    let oracle = Regex::with_options(pattern, opts).unwrap();
    let cuda = CudaRegex::new(pattern).unwrap();

    // 1. Oracle results
    let oracle_all = oracle.find_all(input).unwrap();
    let oracle_is = oracle.is_match(input).unwrap();

    // 2. CPU reference kernel results
    let cpu_all = cuda.find_all_cpu(input).unwrap();
    assert_eq!(oracle_all, cpu_all,
        "[{}] CPU-kernel vs oracle mismatch\n  pattern: {}\n  oracle: {:?}\n  cpu:    {:?}",
        label, pattern, &oracle_all[..oracle_all.len().min(10)], &cpu_all[..cpu_all.len().min(10)]);

    // 3. GPU kernel results (force GPU path)
    if let Some(gpu_all) = cuda.find_all_gpu(input) {
        assert_eq!(oracle_all, gpu_all,
            "[{}] GPU-kernel vs oracle mismatch\n  pattern: {}\n  oracle: {:?}\n  gpu:    {:?}",
            label, pattern, &oracle_all[..oracle_all.len().min(10)], &gpu_all[..gpu_all.len().min(10)]);

        // Verify is_match consistency
        let gpu_is = !gpu_all.is_empty() || (input.is_empty() && cuda.dfa_tables().map_or(false, |d| d.empty_nullable));
        assert_eq!(oracle_is, gpu_is,
            "[{}] GPU is_match mismatch: oracle={} gpu={}", label, oracle_is, gpu_is);

        eprintln!("  [{}] ✓ oracle={} cpu={} gpu={} matches",
            label, oracle_all.len(), cpu_all.len(), gpu_all.len());
    } else {
        // GPU not available for this pattern (anchors/lookarounds → fallback)
        eprintln!("  [{}] ✓ oracle={} cpu={} (GPU fallback, pattern has anchors/lookarounds)",
            label, oracle_all.len(), cpu_all.len());
    }
}

// == GPU: Simple patterns ==
#[test] fn gpu_literal()       { cross_check(r"hello", b"say hello world hello", "literal"); }
#[test] fn gpu_no_match()      { cross_check(r"xyz", b"abcdef", "no-match"); }
#[test] fn gpu_digits()        { cross_check(r"\d+", b"abc 123 def 456", "digits"); }
#[test] fn gpu_alternation()   { cross_check(r"cat|dog|bird", b"the cat and the dog", "alt"); }
#[test] fn gpu_quantifier()    { cross_check(r"a{2,4}", b"a aa aaa aaaa aaaaa", "quant"); }
#[test] fn gpu_dot_star()      { cross_check(r"he.*lo", b"hello helo hexxxxxxlo", "dot-star"); }
#[test] fn gpu_char_range()    { cross_check(r"[A-Za-z]+", b"Hello World 123", "range"); }
#[test] fn gpu_escaped()       { cross_check(r"\.\*\+", b"match .*+ literally", "escaped"); }
#[test] fn gpu_multi_match()   { cross_check(r"\d{3}-\d{4}", b"call 555-1234 or 555-5678 today", "multi"); }

// == GPU: Extended operators (these compile into the DFA — should work on GPU) ==
#[test] fn gpu_wildcard()      { cross_check(r"a_*b", b"axyzb a  b ab", "wildcard"); }
#[test] fn gpu_intersection()  { cross_check(r"_*cat_*&_*dog_*", b"the cat and the dog", "intersect"); }
#[test] fn gpu_complement()    { cross_check(r"~(_*abc_*)", b"xyz", "complement"); }

// == GPU: Anchor/lookaround patterns (fall back to CPU engine) ==
#[test] fn gpu_anchor_start()  { cross_check(r"^hello", b"hello world", "^hello"); }
#[test] fn gpu_anchor_end()    { cross_check(r"world$", b"hello world", "world$"); }
#[test] fn gpu_pos_ahead()     { cross_check(r"\d+(?=:-)", b"price 42:- end", "pos-la"); }
#[test] fn gpu_pos_behind()    { cross_check(r"(?<=\$)\d+", b"costs $50 or $100", "pos-lb"); }
#[test] fn gpu_neg_ahead()     { cross_check(r"\d+(?!:-)", b"price 42:- 99 end", "neg-la"); }
#[test] fn gpu_neg_behind()    { cross_check(r"(?<!\$)\d+", b"costs $50 or 100", "neg-lb"); }

// == GPU: Edge cases ==
#[test] fn gpu_empty_input()   { cross_check(r"\d+", b"", "empty-in"); }
#[test] fn gpu_single_byte()   { cross_check(r".", b"x", "single"); }
#[test] fn gpu_all_bytes() {
    let input: Vec<u8> = (0..=255).collect();
    cross_check(r"_+", &input, "all-bytes");
}
#[test] fn gpu_newlines()      { cross_check(r".+", b"line1\nline2\nline3", "newlines"); }
#[test] fn gpu_utf8()          { cross_check(r"\w+", "héllo wörld".as_bytes(), "utf8"); }
#[test] fn gpu_repeated()      { cross_check(r"a+", b"aaaaaaaaaa", "repeated"); }
#[test] fn gpu_exact()         { cross_check(r"abc", b"abc", "exact"); }
#[test] fn gpu_match_start()   { cross_check(r"abc", b"abcdef", "at-start"); }
#[test] fn gpu_match_end()     { cross_check(r"abc", b"xyzabc", "at-end"); }

// == GPU: Regression patterns ==
#[test] fn gpu_phone()   { cross_check(r"\d{3}-\d{4}", b"call 555-1234 or 555-5678", "phone"); }
#[test] fn gpu_email()   { cross_check(r"[a-z]+@[a-z]+\.[a-z]+", b"send to foo@bar.com today", "email"); }
#[test] fn gpu_ip()      { cross_check(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", b"connect to 192.168.1.1 or 10.0.0.1", "ip"); }
#[test] fn gpu_hex()     { cross_check(r"#[0-9a-fA-F]{6}", b"color: #FF00AA and #123abc", "hex"); }
#[test] fn gpu_names()   { cross_check(r"Sherlock|Holmes|Watson|Irene|Adler", b"Sherlock Holmes met Irene Adler and Watson", "names"); }

// == GPU: Large inputs (above typical GPU threshold) ==
#[test] fn gpu_large_literal() {
    let input = "the quick brown fox ".repeat(5_000);
    cross_check(r"fox", input.as_bytes(), "large-literal");
}

#[test] fn gpu_large_class() {
    let input = "the quick brown fox ".repeat(5_000);
    cross_check(r"[a-z]+", input.as_bytes(), "large-class");
}

#[test] fn gpu_large_dense() {
    let input = "a1b2c3d4e5f6g7h8i9j0".repeat(5_000);
    cross_check(r"\d", input.as_bytes(), "large-dense");
}

#[test] fn gpu_large_email() {
    let input = "send to foo@bar.com or baz@qux.org today ".repeat(2_000);
    cross_check(r"[a-z]+@[a-z]+\.[a-z]+", input.as_bytes(), "large-email");
}

#[test] fn gpu_large_complement() {
    let input = "xyz hello world abc test ".repeat(2_000);
    cross_check(r"~(_*abc_*)", input.as_bytes(), "large-complement");
}

// == GPU: Batch test (multiple inputs, same pattern) ==
#[test] fn gpu_batch() {
    let pattern = r"\d{3}-\d{4}";
    let inputs: Vec<&[u8]> = vec![
        b"555-1234", b"no match", b"call 555-5678 today", b"",
        b"123-4567 and 987-6543", b"12-345", b"000-0000",
    ];
    for (i, inp) in inputs.iter().enumerate() {
        cross_check(pattern, inp, &format!("batch-{}", i));
    }
}
