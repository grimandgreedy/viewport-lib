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
/// instanced mesh shaders read them. Matches the WGSL `MaterialGpu` struct, 304
/// bytes. The scalars are the per-material fields that used to be duplicated into
/// every `InstanceData` record; moving them here shrinks the per-instance record
/// to O(instances) transform-free and keeps a single copy per distinct material.
///
/// Field packing (after the 160-byte transform array):
/// - `scalars0` = (ambient, diffuse, specular, shininess)
/// - `scalars1` = (metallic, roughness, normal_strength, has_mr_tex)
/// - `scalars2` = (emissive.r, emissive.g, emissive.b, ao_range.min)
/// - `scalars3` = (ao_range.max, param_vis_scale, backface_policy, has_emissive_tex)
/// - `flags`    = (use_pbr, use_flat, alpha_mode, param_vis_mode)
/// - `backface_colour` = the styled-backface colour (see below)
/// - `mr_range`  = (metallic_min, metallic_max, roughness_min, roughness_max)
/// - `tex_index0` = bindless array indices (albedo, normal, ao, metallic-roughness)
/// - `tex_index1` = bindless array indices (emissive, unused, unused, unused)
///
/// `tex_index0` / `tex_index1` hold each material texture's index into the
/// bindless texture array, or `NO_TEXTURE` (`u32::MAX`) when the material has no
/// texture in that slot. They are only meaningful under the bindless texture
/// binding (Vulkan/DX12); under the per-batch binding the interner writes the
/// constant `NO_TEXTURE` into every slot so the material blocks dedup exactly as
/// before (textures are keyed by the batch, not the material buffer). The
/// bindless shader reads the index and samples the array; the per-batch shaders
/// ignore these fields.
///
/// `has_mr_tex` / `has_emissive_tex` (0 or 1) gate the instanced metallic-roughness
/// and emissive texture samples; `mr_range` remaps the raw MR sample before the
/// scalar factor, mirroring the per-object `metallic_range` / `roughness_range`.
/// These are per-material, so they ride the material buffer (the actual textures
/// are per-batch, bound on the instanced group-1 layout).
///
/// `alpha_mode` is 0 Opaque / 1 Mask / 2 Blend / 3 BlendPremultiplied (only the
/// instanced OIT shader reads it, to skip the premultiply for mode 3).
/// `param_vis_mode` 0 means off; non-zero selects a procedural UV pattern that
/// replaces the lit colour (mirrors the per-object `uv_vis_mode`).
/// `backface_policy` is 0 Cull / 1 Identical / 2 DifferentColour / 3 Tint /
/// 4..7 Pattern (stored as an exact small integer in the f32 slot). For the
/// styled policies the instanced shader flips the normal and overrides the
/// colour on back faces. `backface_colour` carries the DifferentColour rgb, the
/// Tint factor (in `.r`), or the Pattern colour (rgb); the Pattern world scale is
/// per-instance (transform-dependent) and rides `InstanceData` instead.
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
    pub(crate) backface_colour: [f32; 4],
    pub(crate) mr_range: [f32; 4],
    pub(crate) tex_index0: [u32; 4],
    pub(crate) tex_index1: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<MaterialGpu>() == 304);

/// Sentinel bindless index for "this material has no texture in this slot". The
/// bindless shader falls back to the neutral default (white albedo, flat normal,
/// and so on) rather than sampling the array.
pub(crate) const NO_TEXTURE: u32 = u32::MAX;

impl MaterialGpu {
    /// Build the full GPU material block (transforms + scalars) from a material.
    /// The scalar derivations mirror `common_material` (`mesh_material.rs`).
    ///
    /// `tex_indices` fills the bindless array indices from the material's texture
    /// ids; when false (the per-batch binding, the common case) every slot is
    /// `NO_TEXTURE`, so material blocks dedup independently of their textures.
    pub(crate) fn from_material(m: &Material, tex_indices: bool) -> MaterialGpu {
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
        // Styled back-face policy and colour (the per-item Pattern world scale
        // rides `InstanceData`, so `backface_colour.w` is left unused here).
        use crate::scene::material::BackfacePolicy;
        let backface_policy = match m.backface_policy {
            BackfacePolicy::Cull => 0u32,
            BackfacePolicy::Identical => 1,
            BackfacePolicy::DifferentColour(_) => 2,
            BackfacePolicy::Tint(_) => 3,
            BackfacePolicy::Pattern(cfg) => 4 + cfg.pattern as u32,
        };
        let backface_colour = match m.backface_policy {
            BackfacePolicy::DifferentColour(c) => {
                let c = c.to_linear_rgb();
                [c[0], c[1], c[2], 1.0]
            }
            BackfacePolicy::Tint(factor) => [factor, 0.0, 0.0, 1.0],
            BackfacePolicy::Pattern(cfg) => {
                let cc = cfg.colour.to_linear_rgb();
                [cc[0], cc[1], cc[2], 0.0]
            }
            _ => [0.0; 4],
        };
        let has_mr = m.metallic_roughness_texture_id.is_some() as u32 as f32;
        let has_emissive = m.emissive_texture_id.is_some() as u32 as f32;
        // Bindless array indices: a texture's dense slot index (a never-freed
        // handle equals its slot), or NO_TEXTURE when the slot is empty. Constant
        // NO_TEXTURE under the per-batch binding so the block dedups by scalars
        // only, exactly as before bindless.
        let idx = |id: Option<crate::resources::TextureId>| {
            if tex_indices {
                id.map_or(NO_TEXTURE, |t| t.index() as u32)
            } else {
                NO_TEXTURE
            }
        };
        MaterialGpu {
            xf,
            scalars0: [m.ambient, m.diffuse, m.specular, m.shininess],
            scalars1: [m.metallic, m.roughness, m.normal_strength, has_mr],
            scalars2: [e[0], e[1], e[2], m.ao_range[0]],
            scalars3: [
                m.ao_range[1],
                param_vis_scale,
                backface_policy as f32,
                has_emissive,
            ],
            flags: [
                m.is_pbr() as u32,
                m.is_flat() as u32,
                alpha_mode,
                param_vis_mode,
            ],
            backface_colour,
            mr_range: [
                m.metallic_range[0],
                m.metallic_range[1],
                m.roughness_range[0],
                m.roughness_range[1],
            ],
            tex_index0: [
                idx(m.texture_id),
                idx(m.normal_map_id),
                idx(m.ao_map_id),
                idx(m.metallic_roughness_texture_id),
            ],
            tex_index1: [
                idx(m.emissive_texture_id),
                NO_TEXTURE,
                NO_TEXTURE,
                NO_TEXTURE,
            ],
        }
    }
}

/// Per-frame interner that deduplicates material transform blocks and assigns each
/// a stable `material_id` (its index in the uploaded buffer). Entry 0 is always
/// the identity block, so any material with no authored transform maps to 0.
pub(crate) struct MaterialGpuBuilder {
    entries: Vec<MaterialGpu>,
    lookup: HashMap<[u8; 304], u32>,
    /// Set when the capacity was hit and some materials were forced to entry 0.
    pub(crate) overflowed: bool,
    /// When true, `from_material` fills the bindless texture array indices; when
    /// false (the per-batch binding), the index slots are constant `NO_TEXTURE`
    /// so blocks dedup independently of their textures. Fixed for the renderer's
    /// lifetime, set from the device's texture-binding mode.
    bindless: bool,
}

impl Default for MaterialGpuBuilder {
    fn default() -> Self {
        let mut b = MaterialGpuBuilder {
            entries: Vec::new(),
            lookup: HashMap::new(),
            overflowed: false,
            bindless: false,
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
        let default_block = MaterialGpu::from_material(&Material::default(), self.bindless);
        self.entries.push(default_block);
        self.lookup.insert(bytemuck::cast(default_block), 0);
    }

    /// Select whether interned blocks carry bindless texture array indices. Set
    /// once from the device's texture-binding mode; re-runs `reset` so the
    /// default entry matches.
    pub(crate) fn set_bindless(&mut self, bindless: bool) {
        self.bindless = bindless;
        self.reset();
    }

    /// Intern a material's block, returning its `material_id` (its index in the
    /// uploaded buffer). Deduplicates by the block bytes, so instances sharing a
    /// material share an id. On overflow, returns 0.
    pub(crate) fn intern(&mut self, m: &Material) -> u32 {
        let block = MaterialGpu::from_material(m, self.bindless);
        let key: [u8; 304] = bytemuck::cast(block);
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
        let b = MaterialGpu::from_material(&m, false);
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
        let block = MaterialGpu::from_material(&m, false);
        // Albedo (slot 0) stays identity; normal (slot 1) carries the 4x scale.
        assert_eq!(block.xf[0].offset_scale, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(block.xf[1].offset_scale, [0.0, 0.0, 4.0, 4.0]);
    }

    #[test]
    fn styled_backface_packs_policy_and_colour() {
        use crate::scene::material::BackfacePolicy;
        // DifferentColour: policy 2, colour in backface_colour.rgb.
        let mut m = Material::default();
        m.backface_policy =
            BackfacePolicy::DifferentColour(crate::Colour::linear_rgb(0.1, 0.2, 0.3));
        let b = MaterialGpu::from_material(&m, false);
        assert_eq!(u32::try_from(b.scalars3[2] as i64).unwrap(), 2);
        assert_eq!(
            [
                b.backface_colour[0],
                b.backface_colour[1],
                b.backface_colour[2]
            ],
            [0.1, 0.2, 0.3]
        );

        // Tint: policy 3, factor in backface_colour.r.
        let mut m = Material::default();
        m.backface_policy = BackfacePolicy::Tint(0.4);
        let b = MaterialGpu::from_material(&m, false);
        assert_eq!(b.scalars3[2] as u32, 3);
        assert!((b.backface_colour[0] - 0.4).abs() < 1e-6);

        // A default material is Cull (policy 0), no styled back-face.
        let b = MaterialGpu::from_material(&Material::default(), false);
        assert_eq!(b.scalars3[2] as u32, 0);
    }

    #[test]
    fn pbr_texture_flags_and_ranges_pack() {
        // No MR/emissive texture: both has-flags are 0.
        let b = MaterialGpu::from_material(&Material::default(), false);
        assert_eq!(b.scalars1[3], 0.0, "has_mr_tex off by default");
        assert_eq!(b.scalars3[3], 0.0, "has_emissive_tex off by default");

        let mut m = Material::default();
        m.metallic_roughness_texture_id = Some(crate::resources::TextureId::from_raw(7));
        m.emissive_texture_id = Some(crate::resources::TextureId::from_raw(8));
        m.metallic_range = [0.1, 0.9];
        m.roughness_range = [0.2, 0.8];
        let b = MaterialGpu::from_material(&m, false);
        assert_eq!(b.scalars1[3], 1.0, "has_mr_tex set");
        assert_eq!(b.scalars3[3], 1.0, "has_emissive_tex set");
        assert_eq!(b.mr_range, [0.1, 0.9, 0.2, 0.8]);
        // Per-batch binding (tex_indices = false) leaves every array index unset.
        assert_eq!(b.tex_index0, [NO_TEXTURE; 4]);
        assert_eq!(b.tex_index1, [NO_TEXTURE; 4]);
    }

    #[test]
    fn bindless_packs_texture_array_indices() {
        let mut m = Material::default();
        m.texture_id = Some(crate::resources::TextureId::from_raw(3));
        m.normal_map_id = Some(crate::resources::TextureId::from_raw(4));
        m.emissive_texture_id = Some(crate::resources::TextureId::from_raw(9));
        // Bindless on: the slot index (low 32 bits of the handle) lands in the
        // matching lane; empty slots stay NO_TEXTURE.
        let b = MaterialGpu::from_material(&m, true);
        assert_eq!(b.tex_index0[0], 3, "albedo index");
        assert_eq!(b.tex_index0[1], 4, "normal index");
        assert_eq!(b.tex_index0[2], NO_TEXTURE, "no AO texture");
        assert_eq!(b.tex_index0[3], NO_TEXTURE, "no MR texture");
        assert_eq!(b.tex_index1[0], 9, "emissive index");
        // The interner keys on the full block, so under bindless two materials
        // that differ only by texture no longer dedup together.
        let mut with_indices = MaterialGpuBuilder::default();
        with_indices.set_bindless(true);
        let a_id = with_indices.intern(&m);
        let mut m2 = m.clone();
        m2.texture_id = Some(crate::resources::TextureId::from_raw(5));
        assert_ne!(a_id, with_indices.intern(&m2), "bindless splits by texture");
        // Per-batch: the same two materials dedup (textures not in the block).
        let mut per_batch = MaterialGpuBuilder::default();
        assert_eq!(
            per_batch.intern(&m),
            per_batch.intern(&m2),
            "per-batch dedups across textures",
        );
    }
}
