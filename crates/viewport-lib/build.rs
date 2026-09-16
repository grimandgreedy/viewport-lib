use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    // Exclusive selectors for the wgpu version legs, so the seam modules read
    // `#[cfg(wgpu27)]` ("27 and only 27") instead of the verbose
    // `all(feature = "wgpu27", not(feature = "wgpu29"))`. A new leg adds a
    // `not(feature = "...")` term to each existing alias plus its own alias, and
    // every `#[cfg(wgpuNN)]` site stays correct without edits.
    cfg_aliases::cfg_aliases! {
        wgpu27: { all(feature = "wgpu27", not(feature = "wgpu29"), not(feature = "wgpu30")) },
        wgpu29: { all(feature = "wgpu29", not(feature = "wgpu27"), not(feature = "wgpu30")) },
        wgpu30: { all(feature = "wgpu30", not(feature = "wgpu27"), not(feature = "wgpu29")) },
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shaders_dir = PathBuf::from(&manifest_dir).join("src/shaders");
    // Second scan root: item types that live as one directory each under
    // renderer/item_plugins/ keep their WGSL next to their Rust. The directory
    // may not exist until the first type moves in.
    let item_plugins_dir = PathBuf::from(&manifest_dir).join("src/renderer/item_plugins");

    println!("cargo:rerun-if-changed=src/shaders");
    println!("cargo:rerun-if-changed=src/renderer/item_plugins");
    println!("cargo:rerun-if-changed=build.rs");

    // (bare file name, full source path). Names must stay globally unique:
    // OUT_DIR output is flat and every include_str! site keys on the name.
    let mut shaders: Vec<(String, PathBuf)> = fs::read_dir(&shaders_dir)
        .unwrap_or_else(|e| panic!("build.rs: failed to read {}: {}", shaders_dir.display(), e))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("wgsl") {
                return None;
            }
            let name = path.file_name()?.to_str()?.to_string();
            Some((name, path))
        })
        .collect();
    collect_wgsl_recursive(&item_plugins_dir, &mut shaders);
    shaders.sort();
    for pair in shaders.windows(2) {
        if pair[0].0 == pair[1].0 {
            panic!(
                "build.rs: duplicate shader name {} ({} and {}); shader names must be \
                 globally unique because OUT_DIR output is flat",
                pair[0].0,
                pair[0].1.display(),
                pair[1].1.display()
            );
        }
    }

    for (_, path) in &shaders {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let is_ios = std::env::var("CARGO_CFG_TARGET_OS")
        .map(|v| v == "ios")
        .unwrap_or(false);

    for (name, src_path) in &shaders {
        let raw = fs::read_to_string(src_path)
            .unwrap_or_else(|e| panic!("build.rs: failed to read {}: {}", src_path.display(), e));
        // Includes resolve against the shader's own directory first, then
        // src/shaders/ (where the shared helpers/ directives point).
        let own_dir = src_path.parent().unwrap_or(&shaders_dir);
        let preprocessed = resolve_includes(&raw, &[own_dir, &shaders_dir], name);
        let preprocessed = if is_ios {
            patch_for_ios(&preprocessed)
        } else {
            preprocessed
        };
        let out_path = PathBuf::from(&out_dir).join(name);
        fs::write(&out_path, preprocessed)
            .unwrap_or_else(|e| panic!("build.rs: failed to write {}: {}", out_path.display(), e));
    }

    // For shaders that include deform.wgsl, produce a _noop variant where that
    // include is replaced with deform_noop.wgsl. These are loaded at runtime by
    // ViewportGpuResources::new when the device reports max_bind_groups < 3.
    // All of them live in src/shaders/.
    let deform_shaders = [
        "mesh.wgsl",
        "mesh_instanced.wgsl",
        "mesh_instanced_oit.wgsl",
        "mesh_oit.wgsl",
        "outline_mask.wgsl",
        "shadow.wgsl",
        "shadow_point.wgsl",
    ];
    for name in &deform_shaders {
        let src_path = shaders_dir.join(name);
        let raw = fs::read_to_string(&src_path)
            .unwrap_or_else(|e| panic!("build.rs: failed to read {}: {}", src_path.display(), e));
        let raw_noop = raw.replace(
            "// #include \"helpers/deform.wgsl\"",
            "// #include \"helpers/deform_noop.wgsl\"",
        );
        let preprocessed = resolve_includes(&raw_noop, &[&shaders_dir], name);
        let preprocessed = if is_ios {
            patch_for_ios(&preprocessed)
        } else {
            preprocessed
        };
        let noop_name = name.replace(".wgsl", "_noop.wgsl");
        let out_path = PathBuf::from(&out_dir).join(&noop_name);
        fs::write(&out_path, preprocessed)
            .unwrap_or_else(|e| panic!("build.rs: failed to write {}: {}", out_path.display(), e));
    }

    let mut catalog = String::from("&[\n");
    for (name, _) in &shaders {
        write!(
            &mut catalog,
            "    ShaderEntry {{\n        name: \"{name}\",\n        source: include_str!(concat!(env!(\"OUT_DIR\"), \"/{name}\")),\n    }},\n",
        )
        .unwrap();
    }
    catalog.push_str("]\n");

    let catalog_path = PathBuf::from(&out_dir).join("shader_catalog.rs");
    fs::write(&catalog_path, catalog).unwrap_or_else(|e| {
        panic!(
            "build.rs: failed to write {}: {}",
            catalog_path.display(),
            e
        )
    });
}

// Walk `dir` recursively collecting `.wgsl` files. Missing directories are
// fine (the item_plugins tree appears when the first type moves in).
fn collect_wgsl_recursive(dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_wgsl_recursive(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("wgsl") {
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                out.push((name.to_string(), path.clone()));
            }
        }
    }
}

// On iOS, Metal does not support cube array textures. Replace the binding type
// with texture_depth_2d_array (which IS supported) and stub out point shadow
// sampling to always return 1.0 (unshadowed), since cube-direction-to-face
// conversion would be needed for real sampling and point shadows are rarely
// used in mobile scenes.
fn patch_for_ios(source: &str) -> String {
    let s = source.replace("texture_depth_cube_array", "texture_depth_2d_array");
    // The textureSampleCompare call for point shadows takes a vec3 direction and
    // an array index -- neither is valid for texture_depth_2d_array. Stub it out.
    s.replace(
        "    return textureSampleCompare(\n        point_shadow_cube_tex,\n        shadow_sampler,\n        dir,\n        light.point_shadow_slot,\n        normalised - bias,\n    );",
        "    return 1.0;",
    )
}

fn resolve_includes(source: &str, search_dirs: &[&Path], shader_name: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("// #include \"") {
            if let Some(include_name) = rest.strip_suffix("\"") {
                let include_path = search_dirs
                    .iter()
                    .map(|d| d.join(include_name))
                    .find(|p| p.is_file())
                    .unwrap_or_else(|| {
                        panic!(
                            "build.rs: {} includes \"{}\" but no search directory holds it \
                             (looked in: {})",
                            shader_name,
                            include_name,
                            search_dirs
                                .iter()
                                .map(|d| d.display().to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    });
                let content = fs::read_to_string(&include_path).unwrap_or_else(|e| {
                    panic!(
                        "build.rs: {} includes \"{}\" but read failed: {}",
                        shader_name, include_name, e
                    )
                });
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
