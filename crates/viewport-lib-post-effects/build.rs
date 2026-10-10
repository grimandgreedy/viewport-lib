//! Copy `src/shaders/*.wgsl` into `OUT_DIR`, stripped of comments and
//! indentation with the `minify-shaders` feature or on wasm32.

use std::fs;
use std::path::PathBuf;

#[path = "build/minify_wgsl.rs"]
mod minify_wgsl;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shaders_dir = PathBuf::from(&manifest_dir).join("src/shaders");

    println!("cargo:rerun-if-changed=src/shaders");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/minify_wgsl.rs");

    let minify = minify_wgsl::enabled();
    for entry in fs::read_dir(&shaders_dir)
        .unwrap_or_else(|e| panic!("build.rs: failed to read {}: {}", shaders_dir.display(), e))
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("wgsl") {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("build.rs: failed to read {}: {}", path.display(), e));
        let source = if minify {
            minify_wgsl::minify_wgsl(&source)
        } else {
            source
        };
        let out_path = PathBuf::from(&out_dir).join(path.file_name().unwrap());
        fs::write(&out_path, source)
            .unwrap_or_else(|e| panic!("build.rs: failed to write {}: {}", out_path.display(), e));
    }
}
