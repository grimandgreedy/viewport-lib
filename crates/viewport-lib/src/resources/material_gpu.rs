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

use crate::scene::material::Material;
use viewport_lib_types::plugin::material::MATERIAL_PLUGIN_MAX_PARAMS;

/// Blocks the material buffer holds before its first growth. It doubles as
/// materials are uploaded, up to what one storage binding can hold.
pub(crate) const MATERIAL_GPU_INITIAL_CAPACITY: usize = 64;

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

    /// No offset, scale or rotation; the UV set does not count.
    fn is_identity(&self) -> bool {
        self.offset_scale == Self::IDENTITY.offset_scale && self.rot_tc[0] == 0.0
    }

    fn from_transform(t: &crate::scene::material::UvTransform) -> TexTransformGpu {
        TexTransformGpu {
            offset_scale: [t.offset[0], t.offset[1], t.scale[0], t.scale[1]],
            rot_tc: [t.rotation, t.uv_set as f32, 0.0, 0.0],
        }
    }
}

/// A material's per-slot UV transforms plus its scalar shading parameters, as the
/// instanced mesh shaders read them. Matches the WGSL `MaterialGpu` struct, 368
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
/// - `tex_index1` = (emissive bindless index, sampler code, unused, unused)
///
/// `tex_index0` / `tex_index1` hold each material texture's index into the
/// bindless texture array, or `NO_TEXTURE` (`u32::MAX`) when the material has no
/// texture in that slot. They are only meaningful under the bindless texture
/// binding; under the per-batch binding the interner writes the
/// constant `NO_TEXTURE` into every slot so the material blocks dedup exactly as
/// before (textures are keyed by the batch, not the material buffer). The
/// bindless shader reads the index and samples the array; the per-batch shaders
/// ignore these fields.
///
/// The sampler code (`sampler_code` of the material's selected sampler, or the
/// default key) is not read by any shader. It is in the block so that two
/// materials differing only in sampler intern to different ids, the same way
/// they land in different batches.
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
    /// `(scalar_min, scalar_max, 0, 0)`: the item's scalar-attribute range.
    pub(crate) item0: [f32; 4],
    /// Colour drawn where the item's scalar attribute is NaN.
    pub(crate) nan_colour: [f32; 4],
    /// `(item_flags, lightmap_mode, 0, 0)`; see [`item_flags`]. The lightmap
    /// placement rides the instance record.
    pub(crate) item_flags: [u32; 4],
    /// The material plugin's declared parameters, one word each in
    /// declaration order, defaults filled in. Zero without a plugin.
    pub(crate) params: [[f32; 4]; MATERIAL_PLUGIN_MAX_PARAMS],
}

/// Size in bytes of one [`MaterialGpu`] block.
pub(crate) const MATERIAL_GPU_BYTES: usize = 608;
const _: () = assert!(std::mem::size_of::<MaterialGpu>() == MATERIAL_GPU_BYTES);

/// Bit lanes of `MaterialGpu::item_flags[0]`. The WGSL `MAT_*` constants in the
/// mesh shaders mirror these.
pub(crate) mod item_flags {
    pub(crate) const USE_NAN_COLOUR: u32 = 1 << 0;
    pub(crate) const USE_MATCAP: u32 = 1 << 1;
    pub(crate) const MATCAP_BLENDABLE: u32 = 1 << 2;
    pub(crate) const LIGHTMAP_DIRECTIONAL: u32 = 1 << 3;
    pub(crate) const HAS_SHADOWMASK: u32 = 1 << 4;
    /// The material binds an emissive map; mirrors `scalars3.w` as a bit so a
    /// pipeline feature set can read it.
    pub(crate) const HAS_EMISSIVE_TEX: u32 = 1 << 5;
    /// Some texture slot has an offset, scale or rotation.
    pub(crate) const UV_TRANSFORM: u32 = 1 << 7;
}

/// The item-level and mesh-level shading facts that ride the material block
/// next to the `Material` fields: a scalar attribute's range and NaN colour,
/// a matcap's blend flag, and the mesh's baked lightmap mode. They are
/// not `Material` fields, but the fragment stage reads them like material
/// scalars, and interning them into the block keeps the per-instance record
/// free of them. A plain item has the default value and interns to the same
/// block as before.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct ItemShading {
    pub(crate) scalar_range: (f32, f32),
    pub(crate) nan_colour: Option<[f32; 4]>,
    /// `Some(blendable)` when the material names a matcap.
    pub(crate) matcap: Option<bool>,
    /// `(mode, directional, shadowmask)` of the mesh's lightmap registration.
    /// The placement rides the instance record, so draws at different
    /// placements share a block.
    pub(crate) lightmap: Option<(u32, bool, bool)>,
}

impl Default for ItemShading {
    fn default() -> Self {
        ItemShading {
            scalar_range: (0.0, 1.0),
            nan_colour: None,
            matcap: None,
            lightmap: None,
        }
    }
}

impl ItemShading {
    /// The shading a stored material's own block is built with: the defaults,
    /// plus the matcap blend flag, which the material decides.
    pub(crate) fn for_material(m: &Material) -> ItemShading {
        ItemShading {
            matcap: m.matcap_id().map(|id| id.blendable),
            ..ItemShading::default()
        }
    }

    /// The shading as hashable bits.
    pub(crate) fn key(&self) -> ShadingKey {
        let (mode, directional, shadowmask) = self.lightmap.unwrap_or((u32::MAX, false, false));
        let nan = self.nan_colour.unwrap_or([0.0; 4]);
        [
            self.scalar_range.0.to_bits(),
            self.scalar_range.1.to_bits(),
            self.nan_colour.is_some() as u32
                | (self.matcap.is_some() as u32) << 1
                | ((self.matcap == Some(true)) as u32) << 2
                | (directional as u32) << 3
                | (shadowmask as u32) << 4,
            nan[0].to_bits(),
            nan[1].to_bits(),
            nan[2].to_bits(),
            nan[3].to_bits(),
            mode,
        ]
    }

    /// Read the item's shading facts off the item, its material and its
    /// resident mesh.
    pub(crate) fn from_item(
        item: &crate::renderer::SceneRenderItem,
        material: &Material,
        mesh: Option<&crate::resources::GpuMesh>,
    ) -> ItemShading {
        let scalar_range = match &item.active_attribute {
            Some(attr) => item
                .scalar_range
                .or_else(|| mesh.and_then(|m| m.attribute_ranges.get(&attr.name).copied()))
                .unwrap_or((0.0, 1.0)),
            None => (0.0, 1.0),
        };
        ItemShading {
            scalar_range,
            nan_colour: item.nan_colour.map(|c| c.to_linear_rgba()),
            matcap: material.matcap_id().map(|id| id.blendable),
            lightmap: mesh.and_then(|m| m.lightmap.as_ref()).map(|lm| {
                (
                    lm.mode,
                    lm.direction_texture_id.is_some() && !lm.is_shadowmask,
                    lm.is_shadowmask,
                )
            }),
        }
    }
}

/// Sentinel bindless index for "this material has no texture in this slot". The
/// bindless shader falls back to the neutral default (white albedo, flat normal,
/// and so on) rather than sampling the array.
pub(crate) const NO_TEXTURE: u32 = u32::MAX;

/// A material's five texture slots resolved against the texture store: the array
/// index for each slot that names a live texture, `None` otherwise.
///
/// The point of resolving all five in one place is that the block's index and
/// the flag that tells the shader whether to sample it come from the same
/// lookup. Derived separately they disagree the moment a handle goes stale, and
/// a flag saying "sample" next to an index that names nothing is an out-of-bounds
/// read on the bindless array.
#[derive(Clone, Copy, Default)]
pub(crate) struct MaterialSlots([Option<u32>; MATERIAL_TEX_SLOTS]);

impl MaterialSlots {
    /// Resolve every slot of `m` against `textures`. A handle whose slot has been
    /// freed does not resolve and reads as an unset slot.
    pub(crate) fn resolve(
        m: &Material,
        textures: &crate::resources::material::texture_store::TextureStore,
    ) -> Self {
        MaterialSlots([
            textures.slot_index(m.texture_id),
            textures.slot_index(m.normal_map_id),
            textures.slot_index(m.ao_map_id),
            textures.slot_index(m.metallic_roughness_texture_id),
            textures.slot_index(m.emissive_texture_id),
        ])
    }

    /// The array index for `slot`, or `None` when it names no live texture.
    pub(crate) fn index(self, slot: crate::scene::material::TextureSlot) -> Option<u32> {
        self.0[slot as usize]
    }

    /// Whether `slot` names a live texture, which is what the shader's
    /// "has a texture here" flag must be built from.
    pub(crate) fn live(self, slot: crate::scene::material::TextureSlot) -> bool {
        self.index(slot).is_some()
    }

    /// Drop any slot whose index the bindless texture array cannot address.
    ///
    /// The array is declared with a fixed element count, so an index at or above
    /// it is out of bounds rather than merely wrong. A slot dropped here reads as
    /// unset: the material loses that texture and takes the scalar path, which is
    /// wrong but defined, and is logged once. Only applied on the bindless path,
    /// because the per-batch binding binds views directly and has no such ceiling.
    fn capped_to(mut self, capacity: u32) -> Self {
        for slot in self.0.iter_mut() {
            if slot.is_some_and(|i| i >= capacity) {
                let index = slot.take();
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        texture_index = index,
                        capacity,
                        "material texture dropped: its slot index is past the bindless texture \
                         array's capacity, so it cannot be addressed. The material renders \
                         without that map. Free textures you no longer need, or use the \
                         per-batch binding, which has no such limit."
                    );
                }
            }
        }
        self
    }
}

impl crate::resources::DeviceResources {
    /// Resolve a material's texture slots for this device's binding mode.
    ///
    /// One call feeds both the material block's array indices and the per-item
    /// flags that say whether to sample them, so the two cannot be built from
    /// different lookups. Under the bindless binding it also drops slots the
    /// texture array cannot address.
    pub(crate) fn resolve_material_slots(&self, m: &Material) -> MaterialSlots {
        let resolved = MaterialSlots::resolve(m, &self.content.textures);
        match self.instancing.material_texture_binding {
            crate::resources::mesh::instanced_bindless::MaterialTextureBinding::Bindless => {
                resolved.capped_to(
                    crate::resources::mesh::instanced_bindless::BINDLESS_TEXTURE_CAPACITY,
                )
            }
            _ => resolved,
        }
    }
}

impl MaterialGpu {
    /// Build the full GPU material block (transforms + scalars) from a material.
    /// The scalar derivations mirror `common_material` (`mesh_material.rs`).
    ///
    /// `tex_indices` fills the bindless array indices from `resolved`; when false
    /// (the per-batch binding, the common case) every index is `NO_TEXTURE`, so
    /// material blocks dedup independently of their textures. The has-a-texture
    /// flags come from `resolved` either way, because both binding paths have to
    /// take the scalar path for a slot that names no live texture.
    pub(crate) fn from_material(
        m: &Material,
        tex_indices: bool,
        resolved: MaterialSlots,
        item: &ItemShading,
    ) -> MaterialGpu {
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
            // Forward-compat: an unknown future mode renders as Opaque (the default).
            _ => 0,
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
            // Forward-compat: an unknown future policy renders as Cull (the default).
            _ => 0,
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
        let has_mr = resolved.live(MetallicRoughness) as u32 as f32;
        let has_emissive = resolved.live(Emissive) as u32 as f32;
        // Bindless array indices: the texture's slot index, or NO_TEXTURE when the
        // slot names no live texture. Constant NO_TEXTURE under the per-batch
        // binding so the block dedups by scalars only, exactly as before bindless.
        let idx = |slot| {
            if tex_indices {
                resolved.index(slot).unwrap_or(NO_TEXTURE)
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
            tex_index0: [idx(Albedo), idx(Normal), idx(Ao), idx(MetallicRoughness)],
            tex_index1: [
                idx(Emissive),
                crate::resources::material::fallbacks::sampler_code(
                    m.selected_sampler().unwrap_or_default(),
                ),
                NO_TEXTURE,
                NO_TEXTURE,
            ],
            item0: [item.scalar_range.0, item.scalar_range.1, 0.0, 0.0],
            nan_colour: item.nan_colour.unwrap_or([0.0; 4]),
            item_flags: {
                use item_flags as f;
                let on = |c: bool, bit: u32| if c { bit } else { 0 };
                let (mode, directional, shadowmask) = item.lightmap.unwrap_or((0, false, false));
                [
                    on(item.nan_colour.is_some(), f::USE_NAN_COLOUR)
                        | on(item.matcap.is_some(), f::USE_MATCAP)
                        | on(item.matcap == Some(true), f::MATCAP_BLENDABLE)
                        | on(directional, f::LIGHTMAP_DIRECTIONAL)
                        | on(shadowmask, f::HAS_SHADOWMASK)
                        | on(resolved.live(Emissive), f::HAS_EMISSIVE_TEX)
                        | on(xf.iter().any(|x| !x.is_identity()), f::UV_TRANSFORM),
                    mode,
                    0,
                    0,
                ]
            },
            params: [[0.0; 4]; MATERIAL_PLUGIN_MAX_PARAMS],
        }
    }
}

/// The GPU material table: one block per stored material at a fixed index,
/// plus variant blocks for items whose shading facts differ from their
/// material's default (`ItemShading`).
///
/// A stored material's block is written at upload and rewritten in place on
/// replace, so a handle names the same index for its lifetime. Variants are
/// keyed by material and shading, built on first use, and dropped when the
/// instanced batch cache rebuilds, since every item that names one is
/// re-packed then. Block 0 is the default material.
pub(crate) struct MaterialTable {
    blocks: Vec<MaterialGpu>,
    free: Vec<u32>,
    /// `(material, shading key) -> (block, shading)`.
    variants: crate::resources::fast_hash::FastMap<
        (crate::resources::MaterialId, ShadingKey),
        (u32, ItemShading),
    >,
    /// Lowest and one-past-highest block written since the last upload.
    dirty: Option<(usize, usize)>,
    /// When true, blocks carry bindless texture array indices; when false (the
    /// per-batch binding), the index slots are constant `NO_TEXTURE`. Fixed
    /// for the renderer's lifetime, set from the device's texture-binding mode.
    bindless: bool,
}

/// Variant blocks the table holds before the instanced batch cache is forced
/// to rebuild and drop them. Variants accumulate only from items whose shading
/// facts change between rebuilds (an animated scalar range on a static scene).
pub(crate) const MATERIAL_VARIANT_LIMIT: usize = 4096;

/// [`ItemShading`] as hashable bits.
pub(crate) type ShadingKey = [u32; 8];

impl Default for MaterialTable {
    fn default() -> Self {
        let mut t = MaterialTable {
            blocks: Vec::new(),
            free: Vec::new(),
            variants: Default::default(),
            dirty: None,
            bindless: false,
        };
        let block = t.build(
            &Material::default(),
            MaterialSlots::default(),
            &ItemShading::default(),
        );
        t.blocks.push(block);
        t.mark(0);
        t
    }
}

impl MaterialTable {
    /// Select whether blocks carry bindless texture array indices. The caller
    /// rewrites every stored block afterwards.
    pub(crate) fn set_bindless(&mut self, bindless: bool) {
        self.bindless = bindless;
    }

    pub(crate) fn build(
        &self,
        m: &Material,
        resolved: MaterialSlots,
        shading: &ItemShading,
    ) -> MaterialGpu {
        MaterialGpu::from_material(m, self.bindless, resolved, shading)
    }

    fn mark(&mut self, index: usize) {
        self.dirty = Some(match self.dirty {
            Some((lo, hi)) => (lo.min(index), hi.max(index + 1)),
            None => (index, index + 1),
        });
    }

    /// Store `block` at a free index and return it, or `None` when the table
    /// is at `capacity`.
    pub(crate) fn alloc(&mut self, block: MaterialGpu, capacity: usize) -> Option<u32> {
        let index = match self.free.pop() {
            Some(i) => {
                self.blocks[i as usize] = block;
                i
            }
            None => {
                if self.blocks.len() >= capacity {
                    return None;
                }
                self.blocks.push(block);
                (self.blocks.len() - 1) as u32
            }
        };
        self.mark(index as usize);
        Some(index)
    }

    /// Block `index`'s `item_flags`, or zeros past the end.
    pub(crate) fn item_flags(&self, index: u32) -> [u32; 4] {
        self.blocks
            .get(index as usize)
            .map_or([0; 4], |b| b.item_flags)
    }

    pub(crate) fn write(&mut self, index: u32, block: MaterialGpu) {
        self.blocks[index as usize] = block;
        self.mark(index as usize);
    }

    pub(crate) fn release(&mut self, index: u32) {
        debug_assert_ne!(index, 0, "block 0 is the default material");
        self.free.push(index);
    }

    /// The variant block for `id` drawn with `shading`, built with `build` on
    /// first use. Falls back to `base` when the table is full.
    pub(crate) fn variant(
        &mut self,
        id: crate::resources::MaterialId,
        shading: &ItemShading,
        base: u32,
        capacity: usize,
        build: impl FnOnce(&Self) -> MaterialGpu,
    ) -> u32 {
        let key = (id, shading.key());
        if let Some(&(index, _)) = self.variants.get(&key) {
            return index;
        }
        let block = build(self);
        match self.alloc(block, capacity) {
            Some(index) => {
                self.variants.insert(key, (index, *shading));
                index
            }
            None => base,
        }
    }

    /// The variants of `id`, with their shading, for a rewrite after replace.
    pub(crate) fn variants_of(&self, id: crate::resources::MaterialId) -> Vec<(u32, ItemShading)> {
        self.variants
            .iter()
            .filter(|((m, _), _)| *m == id)
            .map(|(_, v)| *v)
            .collect()
    }

    /// Drop the variants of `id`, freeing their blocks.
    pub(crate) fn release_variants_of(&mut self, id: crate::resources::MaterialId) {
        let freed: Vec<u32> = self
            .variants
            .iter()
            .filter(|((m, _), _)| *m == id)
            .map(|(_, (b, _))| *b)
            .collect();
        self.variants.retain(|(m, _), _| *m != id);
        self.free.extend(freed);
    }

    /// Drop every variant. Called when the instanced batch cache rebuilds, so
    /// no packed record still names one.
    pub(crate) fn clear_variants(&mut self) {
        let mut freed: Vec<u32> = self.variants.values().map(|(b, _)| *b).collect();
        self.variants.clear();
        // Highest first, so `alloc` pops the lowest: a rebuild of the same
        // scene hands its variants the same blocks, and its records match the
        // last upload byte for byte.
        freed.sort_unstable_by(|a, b| b.cmp(a));
        self.free.extend(freed);
    }

    pub(crate) fn variants_near_limit(&self) -> bool {
        self.variants.len() > MATERIAL_VARIANT_LIMIT
    }

    /// Every block, indexed by `material_id`. Freed indices hold stale data
    /// nothing reads.
    pub(crate) fn blocks(&self) -> &[MaterialGpu] {
        &self.blocks
    }

    /// The block range written since the last call, clearing it.
    pub(crate) fn take_dirty(&mut self) -> Option<(usize, usize)> {
        self.dirty.take()
    }

    /// Mark every block for upload, after the GPU buffer was replaced.
    pub(crate) fn mark_all(&mut self) {
        if !self.blocks.is_empty() {
            self.dirty = Some((0, self.blocks.len()));
        }
    }
}

#[cfg(test)]
mod tests {

    /// Slots for a material whose textures all resolve, keyed by the low 32 bits
    /// of each handle. The unit tests have no GPU device and so no texture store;
    /// this stands in for one that holds every texture the material names.
    fn all_live(m: &Material) -> MaterialSlots {
        MaterialSlots([
            m.texture_id.map(|t| t.index() as u32),
            m.normal_map_id.map(|t| t.index() as u32),
            m.ao_map_id.map(|t| t.index() as u32),
            m.metallic_roughness_texture_id.map(|t| t.index() as u32),
            m.emissive_texture_id.map(|t| t.index() as u32),
        ])
    }
    use super::*;
    use crate::scene::material::{Material, TextureSlot, UvTransform};

    fn bytes(m: &Material, bindless: bool) -> [u8; MATERIAL_GPU_BYTES] {
        bytemuck::cast(MaterialGpu::from_material(
            m,
            bindless,
            all_live(m),
            &ItemShading::default(),
        ))
    }

    #[test]
    fn default_table_holds_the_default_block() {
        let t = MaterialTable::default();
        assert_eq!(t.blocks().len(), 1);
        let block: [u8; MATERIAL_GPU_BYTES] = bytemuck::cast(t.blocks()[0]);
        assert_eq!(block, bytes(&Material::default(), false));
    }

    #[test]
    fn only_a_moved_slot_flags_the_uv_transform() {
        let flags = |m: &Material| {
            MaterialGpu::from_material(m, false, all_live(m), &ItemShading::default()).item_flags[0]
        };
        assert_eq!(flags(&Material::default()) & item_flags::UV_TRANSFORM, 0);
        let on_uv1 = Material::default().with_texture_transform(
            TextureSlot::Normal,
            UvTransform {
                uv_set: 1,
                ..UvTransform::default()
            },
        );
        assert_eq!(flags(&on_uv1) & item_flags::UV_TRANSFORM, 0);
        let rotated = Material::default().with_uv_rotation(0.3);
        assert_ne!(flags(&rotated) & item_flags::UV_TRANSFORM, 0);
        let tiled = Material::default().with_texture_transform(
            TextureSlot::Emissive,
            UvTransform {
                scale: [2.0, 2.0],
                ..UvTransform::default()
            },
        );
        assert_ne!(flags(&tiled) & item_flags::UV_TRANSFORM, 0);
    }

    #[test]
    fn rotation_changes_the_block() {
        let rotated = Material::default().with_uv_rotation(std::f32::consts::FRAC_PI_2);
        assert_ne!(bytes(&rotated, false), bytes(&Material::default(), false));
    }

    #[test]
    fn variants_are_keyed_by_shading_and_dropped_on_clear() {
        let mut t = MaterialTable::default();
        let id = crate::resources::MaterialId::DEFAULT;
        let ranged = ItemShading {
            scalar_range: (2.0, 5.0),
            ..ItemShading::default()
        };
        let build =
            |t: &MaterialTable| t.build(&Material::default(), MaterialSlots::default(), &ranged);
        let a = t.variant(id, &ranged, 0, 64, build);
        assert_ne!(a, 0);
        assert_eq!(
            t.variant(id, &ranged, 0, 64, build),
            a,
            "same shading, same block"
        );
        t.clear_variants();
        let b = t.variant(id, &ranged, 0, 64, build);
        assert_eq!(b, a, "the freed block is reused");
    }

    #[test]
    fn scalars_pack_emissive_and_pbr() {
        let mut m = Material::default();
        m.emissive = crate::Colour::linear_rgb(1.5, 0.25, 4.0);
        m.metallic = 0.7;
        let b = MaterialGpu::from_material(&m, false, all_live(&m), &ItemShading::default());
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
        let block = MaterialGpu::from_material(&m, false, all_live(&m), &ItemShading::default());
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
        let b = MaterialGpu::from_material(&m, false, all_live(&m), &ItemShading::default());
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
        let b = MaterialGpu::from_material(&m, false, all_live(&m), &ItemShading::default());
        assert_eq!(b.scalars3[2] as u32, 3);
        assert!((b.backface_colour[0] - 0.4).abs() < 1e-6);

        // A default material is Cull (policy 0), no styled back-face.
        let b = MaterialGpu::from_material(
            &Material::default(),
            false,
            all_live(&Material::default()),
            &ItemShading::default(),
        );
        assert_eq!(b.scalars3[2] as u32, 0);
    }

    #[test]
    fn pbr_texture_flags_and_ranges_pack() {
        // No MR/emissive texture: both has-flags are 0.
        let b = MaterialGpu::from_material(
            &Material::default(),
            false,
            all_live(&Material::default()),
            &ItemShading::default(),
        );
        assert_eq!(b.scalars1[3], 0.0, "has_mr_tex off by default");
        assert_eq!(b.scalars3[3], 0.0, "has_emissive_tex off by default");

        let mut m = Material::default();
        m.metallic_roughness_texture_id = Some(crate::resources::TextureId::from_raw(7));
        m.emissive_texture_id = Some(crate::resources::TextureId::from_raw(8));
        m.metallic_range = [0.1, 0.9];
        m.roughness_range = [0.2, 0.8];
        let b = MaterialGpu::from_material(&m, false, all_live(&m), &ItemShading::default());
        assert_eq!(b.scalars1[3], 1.0, "has_mr_tex set");
        assert_eq!(b.scalars3[3], 1.0, "has_emissive_tex set");
        assert_eq!(b.mr_range, [0.1, 0.9, 0.2, 0.8]);
        // Per-batch binding (tex_indices = false) leaves every array index unset;
        // `tex_index1.y` is the sampler code, zero for the default key.
        assert_eq!(b.tex_index0, [NO_TEXTURE; 4]);
        assert_eq!(b.tex_index1, [NO_TEXTURE, 0, NO_TEXTURE, NO_TEXTURE]);
    }

    #[test]
    fn sampler_key_splits_blocks_and_default_key_does_not() {
        use crate::scene::material::{SamplerKey, TextureSlot};
        let plain = Material::default();
        let default_key = plain.with_sampler(TextureSlot::Albedo, SamplerKey::default());
        let clamp = plain.with_sampler(TextureSlot::Albedo, SamplerKey::clamp());
        assert_eq!(bytes(&default_key, false), bytes(&plain, false));
        assert_ne!(bytes(&clamp, false), bytes(&plain, false));
    }

    #[test]
    fn bindless_packs_texture_array_indices() {
        let mut m = Material::default();
        m.texture_id = Some(crate::resources::TextureId::from_raw(3));
        m.normal_map_id = Some(crate::resources::TextureId::from_raw(4));
        m.emissive_texture_id = Some(crate::resources::TextureId::from_raw(9));
        // Bindless on: the slot index (low 32 bits of the handle) lands in the
        // matching lane; empty slots stay NO_TEXTURE.
        let b = MaterialGpu::from_material(&m, true, all_live(&m), &ItemShading::default());
        assert_eq!(b.tex_index0[0], 3, "albedo index");
        assert_eq!(b.tex_index0[1], 4, "normal index");
        assert_eq!(b.tex_index0[2], NO_TEXTURE, "no AO texture");
        assert_eq!(b.tex_index0[3], NO_TEXTURE, "no MR texture");
        assert_eq!(b.tex_index1[0], 9, "emissive index");
        // Under bindless two materials that differ only by texture build
        // different blocks; per-batch, the textures are not in the block.
        let mut m2 = m;
        m2.texture_id = Some(crate::resources::TextureId::from_raw(5));
        assert_ne!(
            bytes(&m, true),
            bytes(&m2, true),
            "bindless splits by texture"
        );
        assert_eq!(
            bytes(&m, false),
            bytes(&m2, false),
            "per-batch ignores textures"
        );
    }
}
