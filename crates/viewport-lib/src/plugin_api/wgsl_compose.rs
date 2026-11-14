//! Splicing a host's WGSL into an item type's own shaders.
//!
//! Material plugins and custom overlay fills splice a body the host wrote into
//! a shader the renderer owns: every top-level identifier in the body is
//! prefixed with the plugin's name so two bodies cannot collide, and the
//! composed module is run through the device's validator before anything
//! draws with it, so a bad body fails at registration with the compiler's
//! message rather than at the first draw. An item type that takes a host
//! snippet into its own shaders (the sprite type's materials, for one) does
//! the same with these.

use crate::error::{ViewportError, ViewportResult};

/// `body` with every top-level `fn`, `struct`, `alias`, `const` and `var` it
/// declares renamed to `<prefix>__<name>`, and every whole-word use of those
/// names renamed with them. Field accesses (`x.name`) are left alone.
///
/// This is the renaming material plugins and custom overlay fills get, so a
/// body written for one reads the same way in another.
pub fn prefix_identifiers(prefix: &str, body: &str) -> String {
    crate::resources::mesh_sidecar::registry::identifier_prefix(prefix, body)
}

/// Whether `source` uses `ident` as a whole identifier, by the boundary rules
/// [`prefix_identifiers`] renames with.
pub fn contains_identifier(source: &str, ident: &str) -> bool {
    crate::resources::mesh_sidecar::registry::contains_word(source, ident)
}

/// Whether `name` is a WGSL identifier a prefix can be built from: ASCII
/// letters, digits and underscores, not starting with a digit or `__`.
pub fn is_identifier(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && !name.starts_with("__")
        && name != "_"
}

/// Create a shader module from `source` under a validation error scope and
/// report what the validator said.
///
/// # Errors
///
/// [`ViewportError::ShadeShaderInvalid`] carrying `label` and the validator's
/// message when the module does not validate.
pub fn validate_module(
    device: &crate::gpu::Device,
    label: &str,
    source: &str,
) -> ViewportResult<crate::gpu::ShaderModule> {
    crate::resources::mesh_sidecar::registry::validate_with_wgpu(device, label, source).map_err(
        |e| match e {
            ViewportError::DeformShaderInvalid { reason } => {
                ViewportError::ShadeShaderInvalid { reason }
            }
            other => other,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_names_are_prefixed_and_fields_are_not() {
        let body = "fn tint(c: vec4<f32>) -> vec4<f32> { return c * k; }\nconst k: f32 = 2.0;\nfn shade(s: S) -> f32 { return tint(s.k).x; }";
        let out = prefix_identifiers("fx", body);
        assert!(out.contains("fn fx__tint("));
        assert!(out.contains("const fx__k"));
        assert!(out.contains("c * fx__k"));
        assert!(out.contains("fx__tint(s.k)"));
        assert!(contains_identifier(&out, "fx__shade"));
        assert!(!contains_identifier(&out, "shade"));
    }

    #[test]
    fn identifiers() {
        assert!(is_identifier("hit_flash2"));
        assert!(!is_identifier("2flash"));
        assert!(!is_identifier("__x"));
        assert!(!is_identifier("a-b"));
        assert!(!is_identifier(""));
    }
}
