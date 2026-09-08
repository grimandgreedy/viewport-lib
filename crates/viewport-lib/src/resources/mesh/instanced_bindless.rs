//! Bindless material-texture variant of the instanced mesh path.
//!
//! On devices with the texture-array feature set (Vulkan/DX12), the instanced
//! colour pipelines bind one texture array once per frame and index it per
//! material, instead of binding a batch's five material textures into group 1.
//! This lets the batch key drop the texture ids, so instances of one mesh with
//! different materials collapse into a single batch (see the mode-aware batch key
//! in `renderer/prepare/instanced.rs` and ADR 0003).
//!
//! The bindless colour shader is generated from the per-batch instanced shader by
//! swapping the five group-1 texture bindings for one `binding_array` and each
//! texture sample for an array index taken from the material block. Keeping it a
//! transform of the shipped shader means the two shading paths cannot drift.
//!
//! The shadow pass is unchanged: it stays on the per-batch binding, and the batch
//! key keeps albedo for alpha-masked materials so a shadow-cutout batch still has
//! a single albedo to bind.

use crate::resources::DeviceResources;

/// How the instanced mesh path binds material textures for a draw.
///
/// `PerBatch` binds the batch's albedo/normal/AO/metallic-roughness/emissive
/// views into group 1 and includes those texture ids in the batch key, so
/// instances that differ only in material land in separate batches. `Bindless`
/// binds one texture array once per frame and lets the shader index it by a
/// per-material index, so the batch key drops the texture ids and instances of
/// one mesh with different materials collapse into a single batch.
///
/// The mode is chosen once at renderer construction from the device's enabled
/// features: `Bindless` needs the texture-array feature set (Vulkan/DX12); Metal
/// and WebGPU stay on `PerBatch`. Both paths render the same result. See ADR 0003.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MaterialTextureBinding {
    /// One group-1 texture bind per batch; texture ids are part of the batch key.
    #[default]
    PerBatch,
    /// One bindless texture array per frame; texture ids drop out of the batch key.
    Bindless,
}

/// Rewrite a per-batch instanced mesh shader (colour or OIT) into its bindless
/// form: replace the group-1 texture bindings (albedo/normal/AO/MR/emissive) with
/// a single `binding_array<texture_2d<f32>>` at binding 1, and each
/// `textureSampleGrad(<slot texture>, ...)` with a sample of the array at the
/// material's index for that slot. The sampler (binding 2) and the cull
/// visibility buffer (binding 5) are unchanged.
///
/// This asserts each substitution lands exactly once, so a change to the source
/// shader's binding names or sample sites fails the build loudly rather than
/// silently producing a wrong bindless shader.
pub(crate) fn bindlessify(src: &str) -> String {
    // The five per-slot texture declarations become one array. Keep binding 0
    // (instances), 2 (sampler) and 5 (visibility) as they are.
    let binding_block = "\
@group(1) @binding(1) var                obj_texture:        texture_2d<f32>;
@group(1) @binding(2) var                obj_sampler:        sampler;
@group(1) @binding(3) var                normal_map:         texture_2d<f32>;
@group(1) @binding(4) var                ao_map:             texture_2d<f32>;
@group(1) @binding(5) var<storage, read> visibility_indices: array<u32>;
@group(1) @binding(6) var                metallic_roughness_tex: texture_2d<f32>;
@group(1) @binding(7) var                emissive_tex:           texture_2d<f32>;";
    let bindless_block = "\
@group(1) @binding(1) var                material_textures:  binding_array<texture_2d<f32>>;
@group(1) @binding(2) var                obj_sampler:        sampler;
@group(1) @binding(5) var<storage, read> visibility_indices: array<u32>;";
    assert_eq!(
        src.matches(binding_block).count(),
        1,
        "bindless transform: the group-1 texture binding block was not found verbatim; \
         the instanced shader's binding layout changed",
    );
    let mut out = src.replace(binding_block, bindless_block);

    // Sample-site swaps. `mat` is the in-scope material block at the albedo /
    // normal / AO / MR sites; the emissive site is in a different function where
    // the material block local is `e_mat`.
    for (from, to) in [
        (
            "textureSampleGrad(obj_texture,",
            "textureSampleGrad(material_textures[mat.tex_index0.x],",
        ),
        (
            "textureSampleGrad(normal_map,",
            "textureSampleGrad(material_textures[mat.tex_index0.y],",
        ),
        (
            "textureSampleGrad(ao_map,",
            "textureSampleGrad(material_textures[mat.tex_index0.z],",
        ),
        (
            "textureSampleGrad(metallic_roughness_tex,",
            "textureSampleGrad(material_textures[mat.tex_index0.w],",
        ),
        (
            "textureSampleGrad(emissive_tex,",
            "textureSampleGrad(material_textures[e_mat.tex_index1.x],",
        ),
    ] {
        assert_eq!(
            out.matches(from).count(),
            1,
            "bindless transform: expected exactly one `{from}` sample site",
        );
        out = out.replace(from, to);
    }
    out
}

/// Fixed size of the bindless texture array binding. A material's texture index
/// is its slot in the texture store, so this caps the highest reachable slot.
/// With `PARTIALLY_BOUND_BINDING_ARRAY` the bind group may supply fewer views
/// than this; the layout just declares the ceiling. Generous for typical scenes;
/// textures uploaded into a slot at or above this index are not reachable by the
/// bindless path (they would need a larger array or a per-batch fallback).
pub(crate) const BINDLESS_TEXTURE_CAPACITY: u32 = 1024;

fn texture_array_entry(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::FRAGMENT,
        ty: crate::gpu::BindingType::Texture {
            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
            view_dimension: crate::gpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: Some(std::num::NonZeroU32::new(BINDLESS_TEXTURE_CAPACITY).unwrap()),
    }
}

fn instance_storage_entry() -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
        ty: crate::gpu::BindingType::Buffer {
            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampler_entry() -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: crate::gpu::ShaderStages::FRAGMENT,
        ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn visibility_entry() -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding: 5,
        visibility: crate::gpu::ShaderStages::VERTEX,
        ty: crate::gpu::BindingType::Buffer {
            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Group-1 layout for the bindless colour pipelines: instance storage (0), the
/// material texture array (1), the shared sampler (2). Mirrors `instance_bgl`
/// with the five per-slot textures collapsed to one array.
pub(crate) fn bindless_instance_bgl(device: &crate::gpu::Device) -> crate::gpu::BindGroupLayout {
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("bindless_instance_bgl"),
        entries: &[
            instance_storage_entry(),
            texture_array_entry(1),
            sampler_entry(),
        ],
    })
}

/// Group-1 layout for the bindless cull pipelines: the colour layout plus the
/// visibility-index buffer (5) the culled vertex path reads.
pub(crate) fn bindless_cull_bgl(device: &crate::gpu::Device) -> crate::gpu::BindGroupLayout {
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("bindless_instance_cull_bgl"),
        entries: &[
            instance_storage_entry(),
            texture_array_entry(1),
            sampler_entry(),
            visibility_entry(),
        ],
    })
}

impl DeviceResources {
    /// Build the dense list of texture views the bindless array binds, indexed by
    /// texture-store slot: slot `i` gets its live view, or the white albedo
    /// fallback when the slot is empty (a freed handle or a gap). Capped at
    /// [`BINDLESS_TEXTURE_CAPACITY`]. A material's `tex_index*` indexes straight
    /// into this list.
    fn bindless_texture_views(&self) -> Vec<&crate::gpu::TextureView> {
        let fallback = &self.material.texture.view;
        // Floor the length at 1 so the array binding is never empty (an untextured
        // scene has no slots, but the binding must still resolve). Index 0 then
        // holds the fallback white view, which no material references.
        let count = self
            .content
            .textures
            .slot_count()
            .min(BINDLESS_TEXTURE_CAPACITY as usize)
            .max(1);
        (0..count)
            .map(|i| {
                self.content
                    .textures
                    .get_by_index(i)
                    .map_or(fallback, |t| &t.view)
            })
            .collect()
    }

    /// Rebuild the frame-constant bindless colour bind group (instances + texture
    /// array + sampler) when the instance buffer or texture set has changed. A
    /// no-op unless the mode is `Bindless` and the layout and instance buffer
    /// exist. `instance_gen` is `InstancingState::instance_gen`, bumped when the
    /// shared instance storage buffer is rebuilt.
    pub(crate) fn ensure_bindless_colour_bind_group(
        &mut self,
        device: &crate::gpu::Device,
        instance_gen: u64,
    ) {
        if self.instancing.material_texture_binding != MaterialTextureBinding::Bindless {
            return;
        }
        let sig = (
            instance_gen,
            self.content.textures.slot_count(),
            self.resource_free_epoch,
        );
        if self.instancing.bindless_signature == Some(sig)
            && self.instancing.bindless_bind_group.is_some()
        {
            return;
        }
        if self.instancing.bindless_bind_group_layout.is_none()
            || self.instancing.storage_buf.is_none()
        {
            return;
        }
        let bg = {
            let layout = self.instancing.bindless_bind_group_layout.as_ref().unwrap();
            let inst_buf = self.instancing.storage_buf.as_ref().unwrap();
            let views = self.bindless_texture_views();
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("bindless_instance_bind_group"),
                layout,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: inst_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureViewArray(&views),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(&self.material.sampler),
                    },
                ],
            })
        };
        self.instancing.bindless_bind_group = Some(bg);
        self.instancing.bindless_signature = Some(sig);
    }

    /// The group-1 bind group for a direct instanced colour draw: the frame
    /// constant texture array under `Bindless`, or the batch's per-texture bind
    /// group (keyed by `mat_key`) under `PerBatch`. `None` skips the batch.
    pub(crate) fn instanced_colour_bind_group(
        &self,
        mat_key: (u64, u64, u64, u64, u64),
    ) -> Option<&crate::gpu::BindGroup> {
        if self.instancing.material_texture_binding == MaterialTextureBinding::Bindless {
            self.instancing.bindless_bind_group.as_ref()
        } else {
            self.instancing.bind_groups.get(&mat_key)
        }
    }

    /// The group-1 bind group for a culled (indirect) instanced colour draw: the
    /// per-viewport texture array under `Bindless`, or the batch's per-texture
    /// cull bind group under `PerBatch`.
    pub(crate) fn instanced_cull_colour_bind_group<'a>(
        &'a self,
        cull_state: &'a crate::resources::ViewportCullState,
        mat_key: (u64, u64, u64, u64, u64),
    ) -> Option<&'a crate::gpu::BindGroup> {
        if self.instancing.material_texture_binding == MaterialTextureBinding::Bindless {
            cull_state.bindless_cull_bind_group.as_ref()
        } else {
            cull_state.instance_cull_bind_groups.get(&mat_key)
        }
    }

    /// The per-viewport bindless cull bind group (colour array + this viewport's
    /// visibility buffer), built on demand. Cleared with the per-batch cull bind
    /// groups when the instance buffer or a texture changes.
    pub(crate) fn get_bindless_cull_bind_group<'a>(
        &self,
        cull_state: &'a mut crate::resources::ViewportCullState,
        device: &crate::gpu::Device,
    ) -> Option<&'a crate::gpu::BindGroup> {
        if cull_state.bindless_cull_bind_group.is_none() {
            let layout = self.instancing.bindless_cull_bind_group_layout.as_ref()?;
            let inst_buf = self.instancing.storage_buf.as_ref()?;
            let bg = {
                let vis_buf = cull_state.visibility_index_buf.as_ref()?;
                let views = self.bindless_texture_views();
                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("bindless_instance_cull_bind_group"),
                    layout,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: inst_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::TextureViewArray(&views),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 2,
                            resource: crate::gpu::BindingResource::Sampler(&self.material.sampler),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 5,
                            resource: vis_buf.as_entire_binding(),
                        },
                    ],
                })
            };
            cull_state.bindless_cull_bind_group = Some(bg);
        }
        cull_state.bindless_cull_bind_group.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The generated bindless colour and OIT shaders must be valid WGSL under the
    // texture-array capabilities. naga validation runs without a device, so this
    // covers the shader on the (Metal) build box where the bindless pipeline is
    // never actually created.
    fn validate_bindless(base: &str) {
        let src = bindlessify(base);
        assert!(
            src.contains("binding_array<texture_2d<f32>>"),
            "transform did not emit the texture array",
        );
        // The per-slot texture bindings and their samples must be gone (a bare
        // `emissive_tex` substring still appears in the `has_emissive_tex`
        // comment, so check the declaration and sample forms specifically).
        assert!(
            !src.contains("var                obj_texture")
                && !src.contains("var                emissive_tex")
                && !src.contains("textureSampleGrad(obj_texture")
                && !src.contains("textureSampleGrad(emissive_tex"),
            "transform left a per-slot texture reference behind",
        );
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("bindless WGSL failed to parse: {e:?}"));
        // Permit the full capability set: this is a smoke check that the bindless
        // transform yields valid WGSL (including the non-uniform array indexing),
        // not a check of any one device's limits.
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("bindless WGSL failed validation: {e:?}"));
    }

    #[test]
    fn bindless_colour_shader_validates() {
        validate_bindless(include_str!(concat!(
            env!("OUT_DIR"),
            "/mesh_instanced_noop.wgsl"
        )));
    }

    #[test]
    fn bindless_oit_shader_validates() {
        validate_bindless(include_str!(concat!(
            env!("OUT_DIR"),
            "/mesh_instanced_oit_noop.wgsl"
        )));
    }
}
