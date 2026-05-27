use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shaders_dir = PathBuf::from(&manifest_dir).join("src/shaders");

    let mesh_shaders = [
        "mesh.wgsl",
        "mesh_instanced.wgsl",
        "mesh_oit.wgsl",
        "mesh_instanced_oit.wgsl",
    ];

    println!("cargo:rerun-if-changed=src/shaders/debug_vis.wgsl");
    for name in &mesh_shaders {
        println!("cargo:rerun-if-changed=src/shaders/{}", name);
    }

    for name in &mesh_shaders {
        let src_path = shaders_dir.join(name);
        let raw = fs::read_to_string(&src_path)
            .unwrap_or_else(|e| panic!("build.rs: failed to read {}: {}", src_path.display(), e));
        let preprocessed = resolve_includes(&raw, &shaders_dir, name);
        let out_path = PathBuf::from(&out_dir).join(name);
        fs::write(&out_path, preprocessed)
            .unwrap_or_else(|e| panic!("build.rs: failed to write {}: {}", out_path.display(), e));
    }
}

fn resolve_includes(source: &str, shaders_dir: &Path, shader_name: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("// #include \"") {
            if let Some(include_name) = rest.strip_suffix("\"") {
                let include_path = shaders_dir.join(include_name);
                let content = fs::read_to_string(&include_path)
                    .unwrap_or_else(|e| panic!(
                        "build.rs: {} includes \"{}\" but read failed: {}",
                        shader_name, include_name, e
                    ));
                out.push_str(&content);
                out.push('\n');
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}
