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

/// A material's per-slot UV transforms. Matches the WGSL `MaterialGpu` struct
/// (an `array<TexTransform, 5>`), 160 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialGpu {
    pub(crate) xf: [TexTransformGpu; MATERIAL_TEX_SLOTS],
}

const _: () = assert!(std::mem::size_of::<MaterialGpu>() == 160);

impl MaterialGpu {
    pub(crate) const IDENTITY: MaterialGpu = MaterialGpu {
        xf: [TexTransformGpu::IDENTITY; MATERIAL_TEX_SLOTS],
    };

    /// Build the GPU transform block from a material: one entry per texture slot,
    /// each resolved to its per-texture override or the material's shared
    /// `uv_offset` / `uv_scale` / `uv_rotation`.
    pub(crate) fn from_material(m: &Material) -> MaterialGpu {
        use crate::scene::material::TextureSlot::{Albedo, Ao, Emissive, MetallicRoughness, Normal};
        let slots = [Albedo, Normal, Ao, MetallicRoughness, Emissive];
        let mut xf = [TexTransformGpu::IDENTITY; MATERIAL_TEX_SLOTS];
        for (i, slot) in slots.iter().enumerate() {
            xf[i] = TexTransformGpu::from_transform(&m.effective_texture_transform(*slot));
        }
        MaterialGpu { xf }
    }

    /// True when this block is the identity (all slots pass UVs through unchanged).
    fn is_identity(&self) -> bool {
        let id = bytemuck::bytes_of(&MaterialGpu::IDENTITY);
        bytemuck::bytes_of(self) == id
    }
}

/// Per-frame interner that deduplicates material transform blocks and assigns each
/// a stable `material_id` (its index in the uploaded buffer). Entry 0 is always
/// the identity block, so any material with no authored transform maps to 0.
pub(crate) struct MaterialGpuBuilder {
    entries: Vec<MaterialGpu>,
    lookup: HashMap<[u8; 160], u32>,
    /// Set when the capacity was hit and some materials were forced to identity.
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
    /// Clear to a single identity entry (id 0) for a new frame.
    pub(crate) fn reset(&mut self) {
        self.entries.clear();
        self.lookup.clear();
        self.overflowed = false;
        self.entries.push(MaterialGpu::IDENTITY);
        self.lookup
            .insert(bytemuck::cast(MaterialGpu::IDENTITY), 0);
    }

    /// Intern a material's transform block, returning its `material_id`. Identity
    /// blocks return 0 without touching the map. On overflow, returns 0.
    pub(crate) fn intern(&mut self, m: &Material) -> u32 {
        let block = MaterialGpu::from_material(m);
        if block.is_identity() {
            return 0;
        }
        let key: [u8; 160] = bytemuck::cast(block);
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
