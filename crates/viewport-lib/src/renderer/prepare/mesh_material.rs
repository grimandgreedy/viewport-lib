//! Per-item material facts shared by the mesh draw paths.

use super::*;

/// The per-range materials to draw `item` with, when it requests them and
/// they line up with the mesh's ranges. `None` means the item draws the
/// whole mesh with its single `material`: either it never set
/// `submesh_materials`, the mesh has no ranges, or the two disagree on
/// count (the mismatch falls back rather than guessing an assignment).
pub(crate) fn active_submesh_materials<'a>(
    item: &'a MeshItem,
    mesh: &crate::resources::GpuMesh,
) -> Option<&'a [crate::resources::MaterialId]> {
    let mats = item.submesh_materials.as_deref()?;
    if mesh.submeshes.is_empty() || mats.len() != mesh.submeshes.len() {
        if !mesh.submeshes.is_empty() {
            tracing::debug!(
                mesh_index = item.mesh_id.index(),
                materials = mats.len(),
                ranges = mesh.submeshes.len(),
                "submesh_materials count does not match the mesh's ranges; \
                 drawing with the item material"
            );
        }
        return None;
    }
    Some(mats)
}

/// Whether any of the item's draws belong in the transparent pass (OIT on
/// the HDR path). Per range for a per-range item: a
/// per-range item can be in both passes at once. A wireframe item has none.
pub(crate) fn has_transparent_draws(item: &MeshItem, resources: &DeviceResources) -> bool {
    if item.settings.wireframe {
        return false;
    }
    if item.settings.opacity < 1.0 {
        return true;
    }
    match resources
        .mesh_store
        .get(item.mesh_id)
        .and_then(|m| active_submesh_materials(item, m))
    {
        Some(mats) => mats
            .iter()
            .any(|m| resources.material_or_default(*m).is_blend()),
        None => resources.material_or_default(item.material).is_blend(),
    }
}

/// The per-instance facts of an item that both packers derive the same way.
/// Material scalars are not here: they live in the material block.
pub(super) struct CommonMaterial {
    pub(super) model: [[f32; 4]; 4],
    pub(super) colour: [f32; 4],
    pub(super) selected: u32,
    pub(super) has_texture: u32,
    pub(super) has_normal_map: u32,
    pub(super) has_ao_map: u32,
    pub(super) unlit: u32,
    pub(super) receive_shadows: u32,
}

/// `resolved` carries the material's texture slots looked up against the store.
/// The has-a-texture flags come from it rather than from `is_some()`: a slot
/// holding a handle whose texture has been freed binds the fallback, so the flag
/// has to say there is no texture, or the shader samples the fallback as though
/// it were authored.
pub(super) fn common_material(
    item: &MeshItem,
    m: &crate::scene::material::Material,
    resolved: crate::resources::material_gpu::MaterialSlots,
) -> CommonMaterial {
    use crate::scene::material::TextureSlot::{Albedo, Ao, Normal};
    CommonMaterial {
        model: item.model.to_mat4().to_cols_array_2d(),
        colour: {
            // The item's own colour, when set, replaces the material's.
            let bc = item
                .settings
                .colour
                .unwrap_or(m.base_colour)
                .to_linear_rgb();
            [bc[0], bc[1], bc[2], item.settings.opacity]
        },
        selected: if item.settings.selected { 1 } else { 0 },
        has_texture: if resolved.live(Albedo) { 1 } else { 0 },
        has_normal_map: if resolved.live(Normal) { 1 } else { 0 },
        has_ao_map: if resolved.live(Ao) { 1 } else { 0 },
        unlit: if item.settings.unlit { 1 } else { 0 },
        receive_shadows: if item.settings.receive_shadows { 1 } else { 0 },
    }
}

/// Whether `m`'s normal map has a signed (snorm) format, which samples in
/// [-1, 1] and must not be remapped from [0, 1]. False when the material has no
/// live normal map.
pub(super) fn normal_map_is_signed(
    resources: &DeviceResources,
    m: &crate::scene::material::Material,
) -> bool {
    m.normal_map_id
        .and_then(|id| resources.content.textures.get(id))
        .is_some_and(|t| is_signed_format(t.view.texture().format()))
}

/// The snorm formats a normal map can be sampled from.
fn is_signed_format(format: crate::gpu::TextureFormat) -> bool {
    use crate::gpu::TextureFormat as F;
    matches!(
        format,
        F::R8Snorm
            | F::Rg8Snorm
            | F::Rgba8Snorm
            | F::R16Snorm
            | F::Rg16Snorm
            | F::Rgba16Snorm
            | F::Bc4RSnorm
            | F::Bc5RgSnorm
            | F::EacR11Snorm
            | F::EacRg11Snorm
    )
}

#[cfg(test)]
mod tests {
    use crate::scene::material::{AlphaMode, BackfacePolicy};

    /// `Identical` is a two-sided policy but not a styled one, so it is not
    /// flagged for per-item back-face handling; a two-sided `Mask` card instances
    /// on the two-sided pipeline.
    #[test]
    fn identical_is_not_a_styled_backface() {
        let mut m = crate::scene::material::Material::default();
        m.backface_policy = BackfacePolicy::Identical;
        m.alpha_mode = AlphaMode::Mask(0.45);
        assert!(m.is_two_sided());
        assert!(!m.backface_needs_per_object());
    }

    /// Snorm formats are decoded as signed; unorm and sRGB ones are remapped.
    #[test]
    fn signed_normal_map_formats() {
        use crate::gpu::TextureFormat as F;
        for f in [F::Bc5RgSnorm, F::EacRg11Snorm, F::Rg8Snorm, F::Rgba8Snorm] {
            assert!(super::is_signed_format(f), "{f:?}");
        }
        for f in [
            F::Bc5RgUnorm,
            F::EacRg11Unorm,
            F::Rg8Unorm,
            F::Rgba8Unorm,
            F::Rgba8UnormSrgb,
        ] {
            assert!(!super::is_signed_format(f), "{f:?}");
        }
    }
}
