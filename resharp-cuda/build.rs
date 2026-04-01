// build.rs: compile CUDA kernels to PTX at build time using nvcc.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_src = "kernels/dfa_scan.cu";
    let ptx_out = out_dir.join("dfa_scan.ptx");

    // Find nvcc
    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        if std::path::Path::new("/usr/local/cuda/bin/nvcc").exists() {
            "/usr/local/cuda/bin/nvcc".to_string()
        } else {
            "nvcc".to_string()
        }
    });

    println!("cargo:rerun-if-changed={}", kernel_src);
    println!("cargo:rerun-if-env-changed=NVCC");

    let status = Command::new(&nvcc)
        .args(&[
            "-cubin",                // compile to native binary (not PTX)
            "-arch=sm_86",           // RTX A6000
            "-O3",                   // max optimization
            "--use_fast_math",       // fast math intrinsics
            "-o", ptx_out.to_str().unwrap(),
            kernel_src,
        ])
        .status()
        .expect("Failed to run nvcc. Is CUDA toolkit installed?");

    if !status.success() {
        panic!("nvcc compilation failed with status: {}", status);
    }

    // Link against CUDA driver library
    println!("cargo:rustc-link-lib=dylib=cuda");
    if std::path::Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists() {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }
    if std::path::Path::new("/usr/local/cuda/lib64/stubs/libcuda.so").exists() {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    }
}
