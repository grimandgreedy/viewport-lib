//! Bindless material-texture variant of the instanced mesh path.
//!
//! On devices with the texture-array feature set, the instanced
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
/// features: `Bindless` needs the texture-array feature set. Whether an adapter
/// has it is not decided by the backend, so do not infer the path from the
/// platform: Apple silicon reports the feature set through Metal argument
/// buffers and takes the bindless path, while WebGPU does not. Both paths render
/// the same result. See ADR 0003.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MaterialTextureBinding {
    /// One group-1 texture bind per batch; texture ids are part of the batch key.
    #[default]
    PerBatch,
    /// One bindless texture array per frame; texture ids drop out of the batch key.
    Bindless,
}

/// Rewrite `mesh.wgsl` or `mesh_oit.wgsl` (or a module composed from either)
/// into the bindless member's form: the five group-1 material texture
/// bindings (albedo 1, normal 3, AO 4, metallic-roughness 11, emissive 12)
/// become one `binding_array` at binding 1, and each sample of a slot texture
/// samples the array at the index the draw's material block holds for that
/// slot. Every other binding is unchanged.
///
/// Each substitution must land exactly once, so a change to the source's
/// declarations or sample sites fails loudly instead of yielding a wrong
/// shader.
pub(crate) fn bindlessify_mesh(src: &str) -> String {
    let mut out = src.to_owned();
    let mut swap = |from: &str, to: &str| {
        assert_eq!(
            out.matches(from).count(),
            1,
            "bindless transform: expected exactly one `{from}`",
        );
        out = out.replacen(from, to, 1);
    };
    swap(
        "@group(1) @binding(1) var obj_texture: texture_2d<f32>;",
        "@group(1) @binding(1) var material_textures: binding_array<texture_2d<f32>>;",
    );
    for decl in [
        "@group(1) @binding(3) var normal_map: texture_2d<f32>;",
        "@group(1) @binding(4) var ao_map: texture_2d<f32>;",
        "@group(1) @binding(11) var metallic_roughness_tex: texture_2d<f32>;",
        "@group(1) @binding(12) var emissive_tex: texture_2d<f32>;",
    ] {
        swap(decl, "");
    }
    for (texture, index) in [
        ("obj_texture", "tex_index0.x"),
        ("normal_map", "tex_index0.y"),
        ("ao_map", "tex_index0.z"),
        ("metallic_roughness_tex", "tex_index0.w"),
        ("emissive_tex", "tex_index1.x"),
    ] {
        swap(
            &format!("textureSampleGrad({texture},"),
            &format!(
                "textureSampleGrad(material_textures[material_gpu_buf[object.material_id].{index}],"
            ),
        );
    }
    format!(
        "{}{out}",
        crate::plugin_api::shared_wgsl::BINDING_ARRAY_ENABLE_WGSL
    )
}

/// The group-1 bindings of `object_bgl` the bindless member folds into the
/// texture array at binding 1.
const MESH_TEXTURE_BINDINGS: [u32; 5] = [1, 3, 4, 11, 12];

/// Fixed size of the bindless texture array binding. A material's texture index
/// is its slot in the texture store, so this caps the highest reachable slot.
/// Every entry is bound, with a neutral fallback view where the store has no
/// texture, so the array is never partially bound. Generous for typical scenes;
/// a texture uploaded into a slot at or above this index is not reachable by the
/// bindless path, and a material naming one loses that map rather than reading
/// out of bounds (it would need a larger array or the per-batch binding).
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

impl DeviceResources {
    /// The bindless member of the mesh group-1 layout, built on first use:
    /// `object_bgl` with the five material texture slots replaced by the
    /// texture array at binding 1.
    pub(crate) fn ensure_object_bindless_bgl(&mut self, device: &crate::gpu::Device) {
        if self.binds.object_bindless_bgl.is_some() {
            return;
        }
        let mut entries: Vec<_> = self
            .binds
            .object_bgl_entries
            .iter()
            .filter(|e| !MESH_TEXTURE_BINDINGS.contains(&e.binding))
            .cloned()
            .collect();
        entries.push(texture_array_entry(1));
        self.binds.object_bindless_bgl = Some(device.create_bind_group_layout(
            &crate::gpu::BindGroupLayoutDescriptor {
                label: Some("object_bindless_bgl"),
                entries: &entries,
            },
        ));
    }

    /// A mesh pipeline layout on the bindless member of group 1: the camera
    /// group and the member.
    pub(crate) fn object_bindless_layout(
        &mut self,
        device: &crate::gpu::Device,
        label: &str,
    ) -> crate::gpu::PipelineLayout {
        self.ensure_object_bindless_bgl(device);
        crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            label,
            &self.binds.camera_bgl,
            self.binds.object_bindless_bgl.as_ref().unwrap(),
        )
    }

    /// Build the list of texture views the bindless array binds, indexed by
    /// texture-store slot: slot `i` gets its live view, or the white albedo
    /// fallback when the slot is empty (a freed handle or a gap). A material's
    /// `tex_index*` indexes straight into this list.
    ///
    /// Always [`BINDLESS_TEXTURE_CAPACITY`] entries, the count the layout
    /// declares. Filling every entry rather than binding only the occupied prefix
    /// means the array is never partially bound, so the device does not need
    /// `PARTIALLY_BOUND_BINDING_ARRAY` to take this path, and an index the
    /// material block should never produce lands on a neutral view instead of on
    /// an unbound entry. The vector is rebuilt only when the bind group is, which
    /// is a change in the instance generation, the slot count, or the free epoch.
    pub(crate) fn bindless_texture_views(&self) -> Vec<&crate::gpu::TextureView> {
        let fallback = &self.material.texture.view;
        (0..BINDLESS_TEXTURE_CAPACITY as usize)
            .map(|i| {
                self.content
                    .textures
                    .get_by_index(i)
                    .map_or(fallback, |t| &t.view)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    // The naga matching this build's wgpu leg; on 27 the crate keeps its own
    // name, so only the aliases need importing.
    use super::*;
    #[cfg(wgpu29)]
    use naga29 as naga;
    #[cfg(wgpu30)]
    use naga30 as naga;

    // The generated bindless mesh and OIT shaders must be valid WGSL under the
    // texture-array capabilities. naga validation runs without a device, so this
    // covers the shader on the (Metal) build box where the bindless pipeline is
    // never actually created.
    fn validate_bindless(base: &str) {
        let src = bindlessify_mesh(base);
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
        validate_bindless(include_str!(concat!(env!("OUT_DIR"), "/mesh.wgsl")));
    }

    #[test]
    fn bindless_oit_shader_validates() {
        validate_bindless(include_str!(concat!(env!("OUT_DIR"), "/mesh_oit.wgsl")));
    }
}
