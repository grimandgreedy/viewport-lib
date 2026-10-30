//! A material plugin whose surface hook writes the ambient-occlusion override.

use viewport_lib::MaterialPlugin;

/// Starts from `surface_override_from(surf)`, writes `ao = 0` and, when
/// `set_flag` is true, sets `SURFACE_OVERRIDE_AO` so the zero takes effect.
///
/// Without the flag the override changes nothing, so the draw matches the
/// stock PBR path; with it the ambient term goes to black.
pub struct AoOverrideMaterialPlugin {
    name: &'static str,
    set_flag: bool,
}

impl AoOverrideMaterialPlugin {
    /// A plugin registered under `name`.
    pub fn new(name: &'static str, set_flag: bool) -> Self {
        Self { name, set_flag }
    }
}

impl MaterialPlugin for AoOverrideMaterialPlugin {
    fn name(&self) -> &str {
        self.name
    }

    fn wgsl_body(&self) -> String {
        let flags = if self.set_flag {
            "SURFACE_OVERRIDE_AO"
        } else {
            "0u"
        };
        format!(
            "\
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {{
    var ov = surface_override_from(surf);
    ov.ao = 0.0;
    ov.flags = {flags};
    return ov;
}}
"
        )
    }
}
