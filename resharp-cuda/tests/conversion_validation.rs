// Cross-check tests: CPU resharp::Regex vs resharp-cuda (CPU ref + GPU).
// Validates that the DFA extraction + kernel reimplementation is correct.

use resharp::{Regex, EngineOptions, Match};
use resharp_cuda::{CudaRegex, verify_dfa_invariants};
use std::time::Instant;

fn assert_eq_matches(pattern: &str, input: &[u8], label: &str) {
    let cpu = Regex::with_options(pattern,
        EngineOptions { dfa_threshold: 2048, ..EngineOptions::default() }).unwrap();
    let cuda = CudaRegex::new(pattern).unwrap();

    let cpu_all = cpu.find_all(input).unwrap();
    let cuda_all = cuda.find_all(input).unwrap();
    assert_eq!(cpu_all, cuda_all,
        "[{}] find_all mismatch\n  pattern: {}\n  cpu: {:?}\n  cuda: {:?}",
        label, pattern, &cpu_all[..cpu_all.len().min(10)], &cuda_all[..cuda_all.len().min(10)]);

    let cpu_is = cpu.is_match(input).unwrap();
    let cuda_is = cuda.is_match(input).unwrap();
    assert_eq!(cpu_is, cuda_is, "[{}] is_match mismatch", label);

    let cpu_anch = cpu.find_anchored(input).unwrap();
    let cuda_anch = cuda.find_anchored(input).unwrap();
    assert_eq!(cpu_anch, cuda_anch, "[{}] find_anchored mismatch", label);
}

// == CP0: API parity ==
#[test] fn cp0_compiles() { CudaRegex::new(r"\d+").unwrap(); }
#[test] fn cp0_options() {
    let opts = EngineOptions { case_insensitive: true, ..EngineOptions::default() };
    let re = CudaRegex::with_options(r"hello", opts, 512).unwrap();
    assert!(re.is_match(b"HELLO").unwrap());
}
#[test] fn cp0_threshold() {
    let mut re = CudaRegex::new(r"test").unwrap();
    re.set_gpu_threshold(1024);
    assert_eq!(re.gpu_threshold, 1024);
}
#[test] fn cp0_has_dfa() {
    let re = CudaRegex::new(r"abc").unwrap();
    eprintln!("  has_gpu_dfa={} has_gpu={}", re.has_gpu_dfa(), re.has_gpu());
}

// == CP1: DFA extraction + invariants ==
#[test] fn cp1_extraction() {
    let re = CudaRegex::new(r"hello").unwrap();
    let dfa = re.dfa_tables().expect("DFA should be extractable");
    assert_eq!(dfa.minterms_lookup.len(), 256);
    assert!(dfa.num_minterms > 0);
    let errs = verify_dfa_invariants(dfa);
    assert!(errs.is_empty(), "Invariant violations: {:?}", errs);
}
#[test] fn cp1_char_class() {
    let re = CudaRegex::new(r"[a-z]+").unwrap();
    let dfa = re.dfa_tables().unwrap();
    assert!(dfa.num_minterms >= 2);
    let errs = verify_dfa_invariants(dfa);
    assert!(errs.is_empty(), "{:?}", errs);
}
#[test] fn cp1_minterm_coverage() {
    let re = CudaRegex::new(r"\d").unwrap();
    let dfa = re.dfa_tables().unwrap();
    for b in 0..=255u8 {
        assert!((dfa.minterms_lookup[b as usize] as u32) < dfa.num_minterms);
    }
}

// == CP2: Simple patterns ==
#[test] fn cp2_literal()       { assert_eq_matches(r"hello", b"say hello world", "literal"); }
#[test] fn cp2_no_match()      { assert_eq_matches(r"xyz", b"abcdef", "no-match"); }
#[test] fn cp2_digits()        { assert_eq_matches(r"\d+", b"abc 123 def 456", "digits"); }
#[test] fn cp2_alternation()   { assert_eq_matches(r"cat|dog|bird", b"the cat and the dog", "alt"); }
#[test] fn cp2_quantifier()    { assert_eq_matches(r"a{2,4}", b"a aa aaa aaaa aaaaa", "quant"); }
#[test] fn cp2_dot_star()      { assert_eq_matches(r"he.*lo", b"hello helo hexxxxxxlo", "dot-star"); }
#[test] fn cp2_anchor_start()  { assert_eq_matches(r"^hello", b"hello world", "^hello"); }
#[test] fn cp2_anchor_end()    { assert_eq_matches(r"world$", b"hello world", "world$"); }
#[test] fn cp2_char_range()    { assert_eq_matches(r"[A-Za-z]+", b"Hello World 123", "range"); }
#[test] fn cp2_escaped()       { assert_eq_matches(r"\.\*\+", b"match .*+ literally", "escaped"); }
#[test] fn cp2_multi_match()   { assert_eq_matches(r"\d{3}-\d{4}", b"call 555-1234 or 555-5678 today", "multi"); }

// == CP3: Extended operators ==
#[test] fn cp3_wildcard()      { assert_eq_matches(r"a_*b", b"axyzb a  b ab", "wildcard"); }
#[test] fn cp3_intersection()  { assert_eq_matches(r"_*cat_*&_*dog_*", b"the cat and the dog", "intersect"); }
#[test] fn cp3_complement()    { assert_eq_matches(r"~(_*abc_*)", b"xyz", "complement"); }

// == CP4: Lookarounds ==
#[test] fn cp4_pos_ahead()     { assert_eq_matches(r"\d+(?=:-)", b"price 42:- end", "pos-la"); }
#[test] fn cp4_pos_behind()    { assert_eq_matches(r"(?<=\$)\d+", b"costs $50 or $100", "pos-lb"); }
#[test] fn cp4_neg_ahead()     { assert_eq_matches(r"\d+(?!:-)", b"price 42:- 99 end", "neg-la"); }
#[test] fn cp4_neg_behind()    { assert_eq_matches(r"(?<!\$)\d+", b"costs $50 or 100", "neg-lb"); }

// == CP5: Edge cases ==
#[test] fn cp5_empty_input()   { assert_eq_matches(r"\d+", b"", "empty-in"); }
#[test] fn cp5_single_byte()   { assert_eq_matches(r".", b"x", "single"); }
#[test] fn cp5_all_bytes() {
    let input: Vec<u8> = (0..=255).collect();
    assert_eq_matches(r"_+", &input, "all-bytes");
}
#[test] fn cp5_newlines()      { assert_eq_matches(r".+", b"line1\nline2\nline3", "newlines"); }
#[test] fn cp5_utf8()          { assert_eq_matches(r"\w+", "héllo wörld".as_bytes(), "utf8"); }
#[test] fn cp5_repeated()      { assert_eq_matches(r"a+", b"aaaaaaaaaa", "repeated"); }
#[test] fn cp5_no_match_long() { assert_eq_matches(r"\d+", b"the quick brown fox jumps over the lazy dog", "no-match-long"); }
#[test] fn cp5_exact()         { assert_eq_matches(r"abc", b"abc", "exact"); }
#[test] fn cp5_match_start()   { assert_eq_matches(r"abc", b"abcdef", "at-start"); }
#[test] fn cp5_match_end()     { assert_eq_matches(r"abc", b"xyzabc", "at-end"); }

// == CP6: Large input cross-check ==
#[test] fn cp6_large_literal() {
    let input = "the quick brown fox ".repeat(10_000);
    let input = input.as_bytes();
    let cpu = Regex::with_options(r"fox",
        EngineOptions { dfa_threshold: 2048, ..EngineOptions::default() }).unwrap();
    let cuda = CudaRegex::new(r"fox").unwrap();

    let t0 = Instant::now();
    let cpu_m = cpu.find_all(input).unwrap();
    let cpu_t = t0.elapsed();
    let t0 = Instant::now();
    let cuda_m = cuda.find_all(input).unwrap();
    let cuda_t = t0.elapsed();

    assert_eq!(cpu_m.len(), cuda_m.len(), "CP6 match count");
    assert_eq!(cpu_m, cuda_m, "CP6 match content");
    eprintln!("  CP6 literal: {} matches, cpu={:?} cuda={:?} input={}KB",
        cpu_m.len(), cpu_t, cuda_t, input.len()/1024);
}

#[test] fn cp6_large_class() {
    let input = "the quick brown fox ".repeat(10_000);
    let input = input.as_bytes();
    let cpu = Regex::with_options(r"[a-z]+",
        EngineOptions { dfa_threshold: 2048, ..EngineOptions::default() }).unwrap();
    let cuda = CudaRegex::new(r"[a-z]+").unwrap();
    let cpu_m = cpu.find_all(input).unwrap();
    let cuda_m = cuda.find_all(input).unwrap();
    assert_eq!(cpu_m, cuda_m, "CP6 class mismatch");
}

#[test] fn cp6_large_dense() {
    let input = "a1b2c3d4e5f6g7h8i9j0".repeat(5_000);
    let input = input.as_bytes();
    let cpu = Regex::with_options(r"\d",
        EngineOptions { dfa_threshold: 2048, ..EngineOptions::default() }).unwrap();
    let cuda = CudaRegex::new(r"\d").unwrap();
    let cpu_m = cpu.find_all(input).unwrap();
    let cuda_m = cuda.find_all(input).unwrap();
    assert_eq!(cpu_m, cuda_m, "CP6 dense mismatch");
}

// == CP7: Batch ==
#[test] fn cp7_batch() {
    let pattern = r"\d{3}-\d{4}";
    let cpu = Regex::with_options(pattern,
        EngineOptions { dfa_threshold: 2048, ..EngineOptions::default() }).unwrap();
    let cuda = CudaRegex::new(pattern).unwrap();
    let inputs: Vec<&[u8]> = vec![
        b"555-1234", b"no match", b"call 555-5678 today", b"",
        b"123-4567 and 987-6543", b"12-345", b"000-0000",
    ];
    for (i, inp) in inputs.iter().enumerate() {
        assert_eq!(cpu.find_all(inp).unwrap(), cuda.find_all(inp).unwrap(),
            "CP7 batch find_all #{}", i);
        assert_eq!(cpu.is_match(inp).unwrap(), cuda.is_match(inp).unwrap(),
            "CP7 batch is_match #{}", i);
    }
}

// == Regression patterns ==
#[test] fn reg_phone()   { assert_eq_matches(r"\d{3}-\d{4}", b"call 555-1234 or 555-5678", "phone"); }
#[test] fn reg_email()   { assert_eq_matches(r"[a-z]+@[a-z]+\.[a-z]+", b"send to foo@bar.com today", "email"); }
#[test] fn reg_ip()      { assert_eq_matches(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", b"connect to 192.168.1.1 or 10.0.0.1", "ip"); }
#[test] fn reg_hex()     { assert_eq_matches(r"#[0-9a-fA-F]{6}", b"color: #FF00AA and #123abc", "hex"); }
#[test] fn reg_names()   { assert_eq_matches(r"Sherlock|Holmes|Watson|Irene|Adler", b"Sherlock Holmes met Irene Adler and Watson", "names"); }
