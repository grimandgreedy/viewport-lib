//! Shader hash pinning for runtime integrity validation.
//!
//! Provides FNV-1a 64-bit hashing for every WGSL shader in `src/shaders/`.
//! The `SHADERS` catalog is generated at build time by `build.rs` from a
//! directory walk, so it stays in sync with the shader inventory automatically.
//! Use `current_shader_hashes()` to snapshot hashes at build time, then
//! `validate_shader_hashes()` at startup to detect accidental shader changes.

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

/// Compute a deterministic FNV-1a 64-bit hash of a byte slice.
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Shader catalog
// ---------------------------------------------------------------------------

/// One entry in the shader catalog.
pub struct ShaderEntry {
    /// Human-readable shader name (filename without path).
    pub name: &'static str,
    /// Full WGSL source as embedded at compile time, after preprocessor
    /// `// #include` resolution.
    pub source: &'static str,
}

/// Every shader in `src/shaders/`, embedded via `include_str!` from the
/// build-time preprocessor output in `OUT_DIR`. Order is alphabetical.
pub const SHADERS: &[ShaderEntry] = include!(concat!(env!("OUT_DIR"), "/shader_catalog.rs"));

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of a shader hash validation run.
pub struct ShaderValidation {
    /// Number of shaders whose hashes matched expected values.
    pub valid: usize,
    /// Names of shaders whose hashes did not match.
    pub mismatched: Vec<String>,
}

/// Return the current FNV-1a hash for every shader in the catalog.
///
/// Returns `(name, hash)` pairs in catalog order.
/// Use this to snapshot the expected hashes for later validation.
pub fn current_shader_hashes() -> Vec<(&'static str, u64)> {
    SHADERS
        .iter()
        .map(|s| (s.name, fnv1a_hash(s.source.as_bytes())))
        .collect()
}

/// Compare `expected` hashes against the current compiled-in shader sources.
///
/// Logs a `tracing::warn!` for each mismatch.
/// Returns `ShaderValidation` with the count of matching shaders and names of
/// mismatched ones.
///
/// Shaders not present in `expected` are skipped (not counted as mismatched).
pub fn validate_shader_hashes(expected: &[(&str, u64)]) -> ShaderValidation {
    let current: std::collections::HashMap<&str, u64> =
        current_shader_hashes().into_iter().collect();

    let mut valid = 0usize;
    let mut mismatched = Vec::new();

    for (name, exp_hash) in expected {
        match current.get(name) {
            Some(&cur_hash) if cur_hash == *exp_hash => {
                valid += 1;
            }
            Some(&cur_hash) => {
                tracing::warn!(
                    shader = %name,
                    expected = %exp_hash,
                    actual = %cur_hash,
                    "shader hash mismatch : shader may have been modified unexpectedly"
                );
                mismatched.push(name.to_string());
            }
            None => {
                tracing::warn!(
                    shader = %name,
                    "shader not found in catalog during validation"
                );
                mismatched.push(name.to_string());
            }
        }
    }

    ShaderValidation { valid, mismatched }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts `.wgsl` files across the two scan roots the build script reads
    /// (`src/shaders/` flat, `src/renderer/item_plugins/` recursive) to drive
    /// expected-count assertions without needing manual updates as shaders are
    /// added or removed.
    fn count_shader_files_on_disk() -> usize {
        fn count_recursive(dir: &std::path::Path) -> usize {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return 0;
            };
            entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let path = e.path();
                    if path.is_dir() {
                        count_recursive(&path)
                    } else if path.extension().and_then(|s| s.to_str()) == Some("wgsl") {
                        1
                    } else {
                        0
                    }
                })
                .sum()
        }
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let flat = std::fs::read_dir(manifest_dir.join("src/shaders"))
            .expect("read src/shaders/")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s == "wgsl")
                    .unwrap_or(false)
            })
            .count();
        flat + count_recursive(&manifest_dir.join("src/renderer/item_plugins"))
    }

    #[test]
    fn test_fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_hash_different_inputs_differ() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_catalog_matches_directory_contents() {
        let on_disk = count_shader_files_on_disk();
        let catalog = current_shader_hashes().len();
        assert_eq!(
            catalog, on_disk,
            "catalog has {} entries but the shader scan roots contain {} .wgsl files",
            catalog, on_disk
        );
    }

    #[test]
    fn test_current_shader_hashes_all_names_present() {
        let hashes = current_shader_hashes();
        let names: Vec<&str> = hashes.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"mesh.wgsl"));
        assert!(names.contains(&"shadow_instanced.wgsl"));
        assert!(names.contains(&"tone_map.wgsl"));
    }

    #[test]
    fn test_validate_shader_hashes_all_correct_passes() {
        let hashes = current_shader_hashes();
        let expected: Vec<(&str, u64)> = hashes.iter().map(|(n, h)| (*n, *h)).collect();
        let result = validate_shader_hashes(&expected);
        assert_eq!(result.valid, hashes.len());
        assert!(result.mismatched.is_empty());
    }

    #[test]
    fn test_validate_shader_hashes_wrong_hash_reports_mismatch() {
        let wrong_hash: Vec<(&str, u64)> = vec![("mesh.wgsl", 0xdeadbeefcafe1234)];
        let result = validate_shader_hashes(&wrong_hash);
        assert_eq!(result.valid, 0);
        assert_eq!(result.mismatched.len(), 1);
        assert_eq!(result.mismatched[0], "mesh.wgsl");
    }

    #[test]
    fn test_validate_shader_hashes_partial_expected() {
        let hashes = current_shader_hashes();
        // Only validate the first 3 shaders
        let expected: Vec<(&str, u64)> = hashes[..3].iter().map(|(n, h)| (*n, *h)).collect();
        let result = validate_shader_hashes(&expected);
        assert_eq!(result.valid, 3);
        assert!(result.mismatched.is_empty());
    }
}

#[cfg(test)]
#[path = "../../build/minify_wgsl.rs"]
mod minify_wgsl;

#[cfg(test)]
mod minify_tests {
    use super::SHADERS;
    use super::minify_wgsl::minify_wgsl;

    fn catalog(name: &str) -> &'static str {
        SHADERS
            .iter()
            .find(|e| e.name == name)
            .unwrap_or_else(|| panic!("{name} missing from the catalog"))
            .source
    }

    #[test]
    fn minify_strips_comments_and_indentation_only() {
        let src = "\
// a comment
    // <viewport-shade-slot:surface>
fn f() -> f32 {

    let a  =  1.0; // trailing
    // BEGIN_DEBUG_VIS
    // <viewport-deform-keep> out.keep = 1.0; // still code
    /* block // not a line comment */ let b = a;
    return a;
}
";
        let want = "\
// <viewport-shade-slot:surface>
fn f() -> f32 {
let a  =  1.0;
// BEGIN_DEBUG_VIS
// <viewport-deform-keep> out.keep = 1.0; // still code
/* block // not a line comment */ let b = a;
return a;
}
";
        assert_eq!(minify_wgsl(src), want);
    }

    #[test]
    fn minify_copies_have_not_drifted() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let ours = std::fs::read_to_string(root.join("build/minify_wgsl.rs")).unwrap();
        for sibling in ["viewport-lib-plugins", "viewport-lib-post-effects"] {
            let path = root.join("..").join(sibling).join("build/minify_wgsl.rs");
            // Absent when this crate is built from a published package.
            let Ok(theirs) = std::fs::read_to_string(&path) else {
                continue;
            };
            assert_eq!(
                ours,
                theirs,
                "{} differs from build/minify_wgsl.rs",
                path.display()
            );
        }
    }

    // Holds with and without `minify-shaders`: these are the comments the
    // renderer rewrites or cuts on at runtime, and a missing one fails silently.
    #[test]
    fn runtime_markers_survive_embedding() {
        let mesh = catalog("mesh.wgsl");
        for marker in [
            "// BEGIN_DEBUG_VIS",
            "// END_DEBUG_VIS",
            "// BEGIN_PBR_STRIP",
            "// END_PBR_STRIP",
            "// <viewport-shade-slot:",
            "// </viewport-shade-slot:",
            "// <viewport-deform-slots:",
            "// </viewport-deform-slots:",
            "// <viewport-deform-keep> ",
        ] {
            assert!(mesh.contains(marker), "mesh.wgsl lost {marker}");
        }
        let rt = catalog("raytrace.wgsl");
        assert!(rt.contains("// <rt-traversal>") && rt.contains("// </rt-traversal>"));

        let stripped = crate::resources::builders::strip_debug_vis(mesh, false);
        assert!(stripped.len() < mesh.len(), "debug vis block was not cut");
    }

    #[cfg(feature = "minify-shaders")]
    #[test]
    fn embedded_sources_are_minified() {
        use crate::plugin_api::shared_wgsl as s;
        let sources = SHADERS.iter().map(|e| (e.name, e.source)).chain([
            ("SHARED_SCENE_LIGHTING_WGSL", s::SHARED_SCENE_LIGHTING_WGSL),
            ("SHARED_CSM_WGSL", s::SHARED_CSM_WGSL),
            ("SHARED_CLIP_VOLUME_WGSL", s::SHARED_CLIP_VOLUME_WGSL),
            ("SHARED_BRDF_WGSL", s::SHARED_BRDF_WGSL),
            ("SHARED_OUTLINE_EDGE_WGSL", s::SHARED_OUTLINE_EDGE_WGSL),
            ("SHARED_BINDINGS_WGSL", s::SHARED_BINDINGS_WGSL),
            ("SHARED_PBR_WGSL", s::SHARED_PBR_WGSL),
            ("SHARED_OIT_WGSL", s::SHARED_OIT_WGSL),
            ("SHARED_DEPTH_READ_WGSL", s::SHARED_DEPTH_READ_WGSL),
            ("SHARED_MASK_WGSL", s::SHARED_MASK_WGSL),
            (
                "SHARED_SHADOW_BINDINGS_WGSL",
                s::SHARED_SHADOW_BINDINGS_WGSL,
            ),
            ("SHARED_PICK_WGSL", s::SHARED_PICK_WGSL),
            ("SHARED_PICK_INSTANCE_WGSL", s::SHARED_PICK_INSTANCE_WGSL),
            ("POST_EFFECT_VS_WGSL", s::POST_EFFECT_VS_WGSL),
            ("SHARED_PICK_PRIM_WGSL", s::SHARED_PICK_PRIM_WGSL),
        ]);
        for (name, src) in sources {
            assert_eq!(minify_wgsl(src), src, "{name} was embedded unminified");
        }
    }
}
