//! Per-material UV transform buffer.
//!
//! Material UV transforms (offset, scale, rotation, and UV-set index per texture
//! slot) live once in a scene-global storage buffer rather than being duplicated
//! into every per-instance / per-object record. Each draw carries a small
//! `material_id` that indexes this buffer. Identity transforms (the common case)
//! all collapse to entry 0, so a scene with no authored transforms uploads a
//! single entry.
//!
//! The buffer is bound at group 0 (the scene-wide bind group) so every mesh
//! shader and draw variant reaches it without touching the per-object / instanced
//! group-1 layouts.

use std::collections::HashMap;

use crate::scene::material::Material;

/// Maximum distinct material transform blocks per frame. Distinct blocks are rare
/// (identity collapses to one entry), so this is a generous ceiling that is not
/// expected to bind in practice. On overflow, further materials fall back to the
/// identity entry (id 0) and [`MaterialGpuBuilder::overflowed`] is set so the
/// caller can log it.
pub(crate) const MATERIAL_GPU_CAPACITY: usize = 4096;

/// Number of texture slots carrying an independent transform, in this order:
/// 0 albedo, 1 normal, 2 ambient-occlusion, 3 metallic-roughness, 4 emissive.
pub(crate) const MATERIAL_TEX_SLOTS: usize = 5;

/// One texture slot's UV transform, vec4-packed for std430 alignment safety.
///
/// `offset_scale = (offset.x, offset.y, scale.x, scale.y)`.
/// `rot_tc = (rotation_radians, uv_set_index_as_f32, 0, 0)`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TexTransformGpu {
    pub(crate) offset_scale: [f32; 4],
    pub(crate) rot_tc: [f32; 4],
}

impl TexTransformGpu {
    pub(crate) const IDENTITY: TexTransformGpu = TexTransformGpu {
        offset_scale: [0.0, 0.0, 1.0, 1.0],
        rot_tc: [0.0, 0.0, 0.0, 0.0],
    };

    fn from_transform(t: &crate::scene::material::UvTransform) -> TexTransformGpu {
        TexTransformGpu {
            offset_scale: [t.offset[0], t.offset[1], t.scale[0], t.scale[1]],
            rot_tc: [t.rotation, t.uv_set as f32, 0.0, 0.0],
        }
    }
}

/// A material's per-slot UV transforms plus its scalar shading parameters, as the
/// instanced mesh shaders read them. Matches the WGSL `MaterialGpu` struct, 240
/// bytes. The scalars are the per-material fields that used to be duplicated into
/// every `InstanceData` record; moving them here shrinks the per-instance record
/// to O(instances) transform-free and keeps a single copy per distinct material.
///
/// Field packing (after the 160-byte transform array):
/// - `scalars0` = (ambient, diffuse, specular, shininess)
/// - `scalars1` = (metallic, roughness, normal_strength, _)
/// - `scalars2` = (emissive.r, emissive.g, emissive.b, ao_range.min)
/// - `scalars3` = (ao_range.max, param_vis_scale, _, _)
/// - `flags`    = (use_pbr, use_flat, alpha_mode, param_vis_mode)
///
/// `alpha_mode` is 0 Opaque / 1 Mask / 2 Blend / 3 BlendPremultiplied (only the
/// instanced OIT shader reads it, to skip the premultiply for mode 3).
/// `param_vis_mode` 0 means off; non-zero selects a procedural UV pattern that
/// replaces the lit colour (mirrors the per-object `uv_vis_mode`).
///
/// The `has_*` texture flags and `alpha_cutoff` / `alpha_flag` stay per-instance
/// (in `InstanceData`): the explicit `MeshInstanceItem` path bakes them at upload
/// time (it pins `material_id` to 0), and the instanced shadow-cutout pass reads
/// them without binding the material buffer (its group 0 is light-only).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialGpu {
    pub(crate) xf: [TexTransformGpu; MATERIAL_TEX_SLOTS],
    pub(crate) scalars0: [f32; 4],
    pub(crate) scalars1: [f32; 4],
    pub(crate) scalars2: [f32; 4],
    pub(crate) scalars3: [f32; 4],
    pub(crate) flags: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<MaterialGpu>() == 240);

impl MaterialGpu {
    /// Build the full GPU material block (transforms + scalars) from a material.
    /// The scalar derivations mirror `common_material` (`mesh_material.rs`).
    pub(crate) fn from_material(m: &Material) -> MaterialGpu {
        use crate::scene::material::TextureSlot::{
            Albedo, Ao, Emissive, MetallicRoughness, Normal,
        };
        let slots = [Albedo, Normal, Ao, MetallicRoughness, Emissive];
        let mut xf = [TexTransformGpu::IDENTITY; MATERIAL_TEX_SLOTS];
        for (i, slot) in slots.iter().enumerate() {
            xf[i] = TexTransformGpu::from_transform(&m.effective_texture_transform(*slot));
        }
        let e = m.emissive_nits();
        // Mirror the per-object `ObjectUniform` derivation (`per_object.rs`).
        let alpha_mode = match m.alpha_mode {
            crate::scene::material::AlphaMode::Opaque => 0u32,
            crate::scene::material::AlphaMode::Mask(_) => 1,
            crate::scene::material::AlphaMode::Blend => 2,
            crate::scene::material::AlphaMode::BlendPremultiplied => 3,
        };
        let param_vis_mode = m.param_vis.map_or(0u32, |pv| pv.mode as u32);
        let param_vis_scale = m.param_vis.map_or(8.0, |pv| pv.scale);
        MaterialGpu {
            xf,
            scalars0: [m.ambient, m.diffuse, m.specular, m.shininess],
            scalars1: [m.metallic, m.roughness, m.normal_strength, 0.0],
            scalars2: [e[0], e[1], e[2], m.ao_range[0]],
            scalars3: [m.ao_range[1], param_vis_scale, 0.0, 0.0],
            flags: [
                m.is_pbr() as u32,
                m.is_flat() as u32,
                alpha_mode,
                param_vis_mode,
            ],
        }
    }
}

/// Per-frame interner that deduplicates material transform blocks and assigns each
/// a stable `material_id` (its index in the uploaded buffer). Entry 0 is always
/// the identity block, so any material with no authored transform maps to 0.
pub(crate) struct MaterialGpuBuilder {
    entries: Vec<MaterialGpu>,
    lookup: HashMap<[u8; 240], u32>,
    /// Set when the capacity was hit and some materials were forced to entry 0.
    pub(crate) overflowed: bool,
}

impl Default for MaterialGpuBuilder {
    fn default() -> Self {
        let mut b = MaterialGpuBuilder {
            entries: Vec::new(),
            lookup: HashMap::new(),
            overflowed: false,
        };
        b.reset();
        b
    }
}

impl MaterialGpuBuilder {
    /// Clear to a single default-material entry (id 0) for a new frame. Entry 0 is
    /// the default material block, so a draw with `material_id` 0 (the normal-vis
    /// uniform, or an overflow fallback) reads sensible default shading. A default
    /// material interns straight to 0.
    pub(crate) fn reset(&mut self) {
        self.entries.clear();
        self.lookup.clear();
        self.overflowed = false;
        let default_block = MaterialGpu::from_material(&Material::default());
        self.entries.push(default_block);
        self.lookup.insert(bytemuck::cast(default_block), 0);
    }

    /// Intern a material's block, returning its `material_id` (its index in the
    /// uploaded buffer). Deduplicates by the block bytes, so instances sharing a
    /// material share an id. On overflow, returns 0.
    pub(crate) fn intern(&mut self, m: &Material) -> u32 {
        let block = MaterialGpu::from_material(m);
        let key: [u8; 240] = bytemuck::cast(block);
        if let Some(&id) = self.lookup.get(&key) {
            return id;
        }
        if self.entries.len() >= MATERIAL_GPU_CAPACITY {
            self.overflowed = true;
            return 0;
        }
        let id = self.entries.len() as u32;
        self.entries.push(block);
        self.lookup.insert(key, id);
        id
    }

    /// The blocks to upload this frame (always at least the identity entry).
    pub(crate) fn entries(&self) -> &[MaterialGpu] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::{Material, TextureSlot, UvTransform};

    #[test]
    fn identity_material_interns_to_zero() {
        let mut b = MaterialGpuBuilder::default();
        assert_eq!(b.intern(&Material::default()), 0);
        // Only the reserved identity entry exists.
        assert_eq!(b.entries().len(), 1);
    }

    #[test]
    fn rotation_makes_a_distinct_entry_and_dedups() {
        let mut b = MaterialGpuBuilder::default();
        let rotated = Material::default().with_uv_rotation(std::f32::consts::FRAC_PI_2);
        let id = b.intern(&rotated);
        assert_ne!(id, 0, "a rotated material is not identity");
        // Same transform interns to the same id (dedup), no new entry.
        assert_eq!(b.intern(&rotated), id);
        assert_eq!(b.entries().len(), 2);
    }

    #[test]
    fn scalars_pack_emissive_and_pbr() {
        let mut m = Material::default();
        m.emissive = crate::Colour::linear_rgb(1.5, 0.25, 4.0);
        m.metallic = 0.7;
        let b = MaterialGpu::from_material(&m);
        // scalars2.xyz = emissive (HDR nits), scalars1.x = metallic.
        assert_eq!(
            [b.scalars2[0], b.scalars2[1], b.scalars2[2]],
            [1.5, 0.25, 4.0]
        );
        assert!((b.scalars1[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn per_slot_override_packs_into_its_slot() {
        let m = Material::default().with_texture_transform(
            TextureSlot::Normal,
            UvTransform {
                scale: [4.0, 4.0],
                ..UvTransform::IDENTITY
            },
        );
        let block = MaterialGpu::from_material(&m);
        // Albedo (slot 0) stays identity; normal (slot 1) carries the 4x scale.
        assert_eq!(block.xf[0].offset_scale, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(block.xf[1].offset_scale, [0.0, 0.0, 4.0, 4.0]);
    }
}
