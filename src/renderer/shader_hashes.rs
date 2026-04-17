//! Shader hash pinning for runtime integrity validation.
//!
//! Provides FNV-1a 64-bit hashing for all 16 WGSL shaders in the viewport library.
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
    /// Full WGSL source as embedded at compile time.
    pub source: &'static str,
}

/// All 16 shaders embedded via `include_str!`.
///
/// Order matches the filesystem order in `src/shaders/`.
pub const SHADERS: &[ShaderEntry] = &[
    ShaderEntry {
        name: "axes_overlay.wgsl",
        source: include_str!("../shaders/axes_overlay.wgsl"),
    },
    ShaderEntry {
        name: "bloom_blur.wgsl",
        source: include_str!("../shaders/bloom_blur.wgsl"),
    },
    ShaderEntry {
        name: "bloom_threshold.wgsl",
        source: include_str!("../shaders/bloom_threshold.wgsl"),
    },
    ShaderEntry {
        name: "contact_shadow.wgsl",
        source: include_str!("../shaders/contact_shadow.wgsl"),
    },
    ShaderEntry {
        name: "fxaa.wgsl",
        source: include_str!("../shaders/fxaa.wgsl"),
    },
    ShaderEntry {
        name: "gizmo.wgsl",
        source: include_str!("../shaders/gizmo.wgsl"),
    },
    ShaderEntry {
        name: "mesh.wgsl",
        source: include_str!("../shaders/mesh.wgsl"),
    },
    ShaderEntry {
        name: "mesh_instanced.wgsl",
        source: include_str!("../shaders/mesh_instanced.wgsl"),
    },
    ShaderEntry {
        name: "outline.wgsl",
        source: include_str!("../shaders/outline.wgsl"),
    },
    ShaderEntry {
        name: "outline_composite.wgsl",
        source: include_str!("../shaders/outline_composite.wgsl"),
    },
    ShaderEntry {
        name: "overlay.wgsl",
        source: include_str!("../shaders/overlay.wgsl"),
    },
    ShaderEntry {
        name: "shadow.wgsl",
        source: include_str!("../shaders/shadow.wgsl"),
    },
    ShaderEntry {
        name: "shadow_instanced.wgsl",
        source: include_str!("../shaders/shadow_instanced.wgsl"),
    },
    ShaderEntry {
        name: "ssao.wgsl",
        source: include_str!("../shaders/ssao.wgsl"),
    },
    ShaderEntry {
        name: "ssao_blur.wgsl",
        source: include_str!("../shaders/ssao_blur.wgsl"),
    },
    ShaderEntry {
        name: "tone_map.wgsl",
        source: include_str!("../shaders/tone_map.wgsl"),
    },
];

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
                    "shader hash mismatch — shader may have been modified unexpectedly"
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
    fn test_current_shader_hashes_returns_16_entries() {
        let hashes = current_shader_hashes();
        assert_eq!(
            hashes.len(),
            16,
            "expected 16 shaders, got {}",
            hashes.len()
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
        assert_eq!(result.valid, 16);
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
