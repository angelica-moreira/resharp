use resharp::EngineOptions;
use resharp_cuda::CudaRegex;

fn test_pattern(pat: &str, input: &[u8], expected_label: &str) {
    let mut re = CudaRegex::with_options(pat, EngineOptions::default(), 4096).unwrap();
    // Force GPU threshold to 0 so we always try GPU
    re.gpu_threshold = 0;
    
    let cpu_result = re.cpu_regex().find_all(input).unwrap();
    let cuda_result = re.find_all(input).unwrap();
    
    let status = if cpu_result == cuda_result { "OK" } else { "FAIL" };
    eprintln!("{} [{}] {} cpu={:?} cuda={:?} has_gpu={}", 
        status, expected_label, pat, cpu_result, cuda_result, re.has_gpu());
}

fn main() {
    eprintln!("=== GPU Kernel Integration Test ===");
    eprintln!("GPU available: {}", resharp_cuda::CudaRegex::new(r"\d+").unwrap().has_gpu());
    
    let big_input = "abc 123 def 456 ghi 789 ".repeat(100);
    let big_bytes = big_input.as_bytes();
    
    test_pattern(r"\d+", big_bytes, "digits");
    test_pattern(r"[a-z]+", big_bytes, "alpha");
    test_pattern("hello", b"hello world hello again", "literal");
    test_pattern(r"\d{3}", big_bytes, "fixed-len");
    test_pattern(r"[a-f0-9]+", b"deadbeef 0xCAFE babe", "hex");
    test_pattern(r"\w+@\w+\.\w+", b"user@host.com test other@place.org", "email");
    
    eprintln!("\n=== Large input GPU test ===");
    let huge = "x".repeat(1_000_000) + "MATCH" + &"y".repeat(1_000_000);
    let huge_bytes = huge.as_bytes();
    test_pattern("MATCH", huge_bytes, "1M-needle");
    
    eprintln!("\nDone.");
}
