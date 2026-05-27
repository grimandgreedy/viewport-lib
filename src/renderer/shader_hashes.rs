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

    /// Counts `.wgsl` files in `src/shaders/` to drive expected-count assertions
    /// without needing manual updates as shaders are added or removed.
    fn count_shader_files_on_disk() -> usize {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let shaders_dir = std::path::Path::new(manifest_dir).join("src/shaders");
        std::fs::read_dir(&shaders_dir)
            .expect("read src/shaders/")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s == "wgsl")
                    .unwrap_or(false)
            })
            .count()
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
            "catalog has {} entries but src/shaders/ contains {} .wgsl files",
            catalog, on_disk
        );
    }

    #[test]
    fn test_current_shader_hashes_all_names_present() {
        let hashes = current_shader_hashes();
        let names: Vec<&str> = hashes.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"mesh.wgsl"));
        assert!(names.contains(&"gizmo.wgsl"));
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
