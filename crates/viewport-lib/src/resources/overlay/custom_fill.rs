//! The registry of custom overlay fill functions and the splice that puts them
//! into the overlay shaders.
//!
//! The fill module (`helpers/overlay_fill.wgsl`) declares a stand-in
//! `overlay_custom_fill` that returns transparent. When fills are registered,
//! the overlay shader sources have that function replaced by a switch over the
//! registered functions, with their prefixed bodies appended. The stand-in is
//! found by its name and its braces rather than by a comment marker, so the
//! splice works on the minified sources too.

use crate::error::{ViewportError, ViewportResult};
use crate::plugin_api::OverlayFillDesc;
use crate::renderer::CustomFillId;

/// The overlay shaders that include the fill module.
const FILL_SHADERS: &[(&str, &str)] = &[
    (
        "overlay_text",
        crate::resources::builders::wgsl_source!("overlay_text"),
    ),
    (
        "overlay_shape",
        crate::resources::builders::wgsl_source!("overlay_shape"),
    ),
];

/// The base source of one of the fill shaders, by name.
#[cfg(test)]
pub(crate) fn fill_shader(name: &str) -> &'static str {
    FILL_SHADERS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or_default()
}

/// A registered fill: its name and its body with every top-level identifier
/// prefixed by `<name>__`.
#[derive(Clone, Debug)]
pub(crate) struct StoredOverlayFill {
    pub name: String,
    pub prefixed_body: String,
}

/// The source of the stand-in `overlay_custom_fill` in `source`, as a byte
/// range: from `fn` to its closing brace.
fn stand_in_range(source: &str) -> Option<std::ops::Range<usize>> {
    let start = source.find("fn overlay_custom_fill(")?;
    let open = start + source[start..].find('{')?;
    let mut depth = 0usize;
    for (i, c) in source[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start..open + i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// `source` with the registered `fills` spliced in, or `source` unchanged
/// when none is registered.
pub(crate) fn splice<'a>(
    source: &'a str,
    fills: &[StoredOverlayFill],
) -> std::borrow::Cow<'a, str> {
    if fills.is_empty() {
        return std::borrow::Cow::Borrowed(source);
    }
    let Some(range) = stand_in_range(source) else {
        return std::borrow::Cow::Borrowed(source);
    };
    let mut dispatch = String::from(
        "fn overlay_custom_fill(id: i32, local: vec2<f32>, params: OverlayFillParams) -> vec4<f32> {\n    var out = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n    switch (id) {\n",
    );
    for (i, f) in fills.iter().enumerate() {
        dispatch.push_str(&format!(
            "        case {i}: {{ out = {}__fill(local, params); }}\n",
            f.name
        ));
    }
    dispatch.push_str("        default: {}\n    }\n    return out;\n}\n");
    let mut out = String::with_capacity(source.len() + dispatch.len() + 256);
    out.push_str(&source[..range.start]);
    out.push_str(&dispatch);
    out.push_str(&source[range.end..]);
    for f in fills {
        out.push('\n');
        out.push_str(&f.prefixed_body);
        out.push('\n');
    }
    std::borrow::Cow::Owned(out)
}

fn invalid(reason: String) -> ViewportError {
    ViewportError::OverlayFillInvalid { reason }
}

impl crate::resources::DeviceResources {
    /// Register a fill function for [`OverlayFill::Custom`] and return the id
    /// an item's fill names it by.
    ///
    /// The body is checked by composing it into every overlay shader that
    /// draws fills and validating the result; nothing is stored when that
    /// fails. The overlay pipelines are rebuilt on the next frame that draws
    /// overlays, with the new function in them. Ids count up from zero in
    /// registration order and stay valid for the life of the renderer.
    ///
    /// See [`OverlayFillDesc`] for the function's signature and what it may
    /// do.
    ///
    /// # Errors
    ///
    /// [`ViewportError::OverlayFillInvalid`] when the name is not a WGSL
    /// identifier or is already registered, the body defines no `fill`
    /// function, or a composed overlay shader fails validation.
    ///
    /// [`OverlayFill::Custom`]: crate::renderer::types::OverlayFill::Custom
    pub fn register_overlay_fill(
        &mut self,
        device: &crate::gpu::Device,
        desc: OverlayFillDesc,
    ) -> ViewportResult<CustomFillId> {
        let name = desc.name.as_ref();
        let identifier = !name.is_empty()
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !name.starts_with(|c: char| c.is_ascii_digit())
            && !name.starts_with("__");
        if !identifier {
            return Err(invalid(format!(
                "fill name '{name}' is not a WGSL identifier"
            )));
        }
        if self.overlay_custom_fills.iter().any(|f| f.name == name) {
            return Err(invalid(format!("fill name '{name}' is already registered")));
        }
        let prefixed_body =
            crate::resources::mesh_sidecar::registry::identifier_prefix(name, &desc.wgsl_body);
        if !prefixed_body.contains(&format!("fn {name}__fill(")) {
            return Err(invalid(format!("fill '{name}' defines no `fill` function")));
        }
        let mut fills = self.overlay_custom_fills.clone();
        fills.push(StoredOverlayFill {
            name: name.to_string(),
            prefixed_body,
        });
        for (shader, base) in FILL_SHADERS {
            let composed = splice(base, &fills);
            let label = format!("overlay_fill_compose_{name}_{shader}");
            crate::resources::mesh_sidecar::registry::validate_with_wgpu(device, &label, &composed)
                .map_err(|e| match e {
                    ViewportError::DeformShaderInvalid { reason } => invalid(reason),
                    other => other,
                })?;
        }
        let id = CustomFillId(self.overlay_custom_fills.len() as u32);
        self.overlay_custom_fills = fills;
        self.drop_overlay_fill_pipelines();
        Ok(id)
    }

    /// The id of the fill function registered as `name`, if there is one.
    pub fn overlay_fill_id(&self, name: &str) -> Option<CustomFillId> {
        self.overlay_custom_fills
            .iter()
            .position(|f| f.name == name)
            .map(|i| CustomFillId(i as u32))
    }

    /// The source of overlay shader `name` with the registered fill functions
    /// spliced in.
    pub(crate) fn overlay_fill_shader_source(&self, name: &str) -> std::borrow::Cow<'static, str> {
        let base = FILL_SHADERS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| *s)
            .unwrap_or_default();
        splice(base, &self.overlay_custom_fills)
    }

    /// Drop the overlay text and shape pipelines so the next frame rebuilds
    /// them from the current fill registry. Their bind group layouts are kept,
    /// so bind groups built against them stay valid.
    fn drop_overlay_fill_pipelines(&mut self) {
        self.overlay_text.pipeline = None;
        self.overlay_text.variants = Default::default();
        self.overlay_text.layout = None;
        self.overlay_text.shader = None;
        self.overlay_shape.pipeline = None;
        self.overlay_shape.variants = Default::default();
        self.overlay_shape.layout = None;
        self.overlay_shape.shader = None;
        self.overlay_world.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stand_in_is_found_in_both_shaders() {
        for (name, src) in FILL_SHADERS {
            let r = stand_in_range(src).unwrap_or_else(|| panic!("{name} has no stand-in"));
            assert!(src[r].ends_with('}'));
        }
    }

    #[test]
    fn splice_replaces_the_stand_in_with_a_switch() {
        let fills = vec![StoredOverlayFill {
            name: "hatch".into(),
            prefixed_body: "fn hatch__fill(local: vec2<f32>, params: OverlayFillParams) -> vec4<f32> { return vec4<f32>(1.0); }".into(),
        }];
        let src = crate::resources::builders::wgsl_source!("overlay_text");
        let out = splice(src, &fills);
        assert_eq!(out.matches("fn overlay_custom_fill(").count(), 1);
        assert!(out.contains("case 0: { out = hatch__fill(local, params); }"));
        assert!(out.contains("fn hatch__fill("));
        // Nothing registered leaves the source as it is.
        assert!(matches!(splice(src, &[]), std::borrow::Cow::Borrowed(_)));
    }

    /// The spliced module is valid WGSL: naga parses and validates it.
    #[test]
    fn a_spliced_module_validates() {
        let body = "fn fill(local: vec2<f32>, params: OverlayFillParams) -> vec4<f32> {\n    let s = step(0.5, fract((local.x + local.y) / max(params.p0.x, 1.0)));\n    return mix(params.p1, params.p2, s);\n}\n";
        let fills = vec![StoredOverlayFill {
            name: "hatch".into(),
            prefixed_body: crate::resources::mesh_sidecar::registry::identifier_prefix(
                "hatch", body,
            ),
        }];
        for (name, src) in FILL_SHADERS {
            let out = splice(src, &fills);
            let module = naga::front::wgsl::parse_str(&out)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&out)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
