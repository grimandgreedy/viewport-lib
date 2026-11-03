//! Flatten every `.wgsl` under `src/` into `OUT_DIR` so `wgsl_source!` can
//! reach a shader by bare name regardless of which item type's directory it
//! lives in.
//!
//! There is no include directive to resolve. Shaders here declare only their
//! own bindings and entry points; the shared group-0 declarations and shading
//! helpers are concatenated at pipeline build time from `shared_wgsl`, which
//! is published by viewport-lib as ordinary string constants.

use std::fs;
use std::path::{Path, PathBuf};

#[path = "build/minify_wgsl.rs"]
mod minify_wgsl;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let src_dir = PathBuf::from(&manifest_dir).join("src");

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/minify_wgsl.rs");

    let mut shaders: Vec<(String, PathBuf)> = Vec::new();
    collect_wgsl(&src_dir, &mut shaders);
    shaders.sort();
    for pair in shaders.windows(2) {
        if pair[0].0 == pair[1].0 {
            panic!(
                "build.rs: duplicate shader name {} ({} and {}); names must be unique \
                 because the OUT_DIR output is flat",
                pair[0].0,
                pair[0].1.display(),
                pair[1].1.display()
            );
        }
    }

    let minify = minify_wgsl::enabled();
    for (name, path) in &shaders {
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("build.rs: failed to read {}: {}", path.display(), e));
        let source = if minify {
            minify_wgsl::minify_wgsl(&source)
        } else {
            source
        };
        let out_path = PathBuf::from(&out_dir).join(name);
        fs::write(&out_path, source)
            .unwrap_or_else(|e| panic!("build.rs: failed to write {}: {}", out_path.display(), e));
    }
}

fn collect_wgsl(dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_wgsl(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("wgsl")
            && let Some(name) = path.file_name().and_then(|s| s.to_str())
        {
            out.push((name.to_string(), path.clone()));
        }
    }
}
