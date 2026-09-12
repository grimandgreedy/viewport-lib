//! The tone-map composite's input slots.
//!
//! `tone_map.wgsl` is a single fullscreen composite: it reads the HDR scene
//! colour plus a fixed set of effect-produced inputs (bloom, ambient
//! occlusion, contact shadows, surface LIC), the scene depth, the foreground
//! coverage mask, and the exposure state buffer. This module names that
//! binding set once, so the bind group layout, the initial and per-frame bind
//! groups, and the per-frame enable flags are all driven from the same table
//! instead of being maintained in parallel by hand.

/// Composite bind group binding indices, matching `tone_map.wgsl`.
pub(crate) mod slot {
    /// Primary colour input: the HDR scene texture, or the DoF output when
    /// depth of field is on (DoF substitutes the primary input rather than
    /// adding a slot of its own).
    pub(crate) const HDR_COLOUR: u32 = 0;
    pub(crate) const SAMPLER: u32 = 1;
    pub(crate) const PARAMS: u32 = 2;
    pub(crate) const BLOOM: u32 = 3;
    pub(crate) const AO: u32 = 4;
    pub(crate) const CONTACT_SHADOW: u32 = 5;
    pub(crate) const SCENE_DEPTH: u32 = 6;
    pub(crate) const LIC: u32 = 7;
    pub(crate) const FOREGROUND_DEPTH: u32 = 8;
    pub(crate) const EXPOSURE: u32 = 9;
    /// Colour-grading strip LUT, sampled after tone mapping. A neutral
    /// placeholder is bound when grading is off (the shader gates on the
    /// uniform's `grade_enabled`).
    pub(crate) const GRADE_LUT: u32 = 10;
}

/// GPU-side shape of one composite binding.
#[derive(Clone, Copy)]
enum BindingKind {
    /// Filterable 2D float texture.
    FilterableTexture,
    /// Depth texture (`texture_depth_2d`).
    DepthTexture,
    /// Filtering sampler.
    Sampler,
    /// Uniform buffer.
    Uniform,
    /// Read-only storage buffer.
    StorageReadOnly,
}

/// The composite's full binding set, in binding order.
const BINDINGS: &[(u32, BindingKind)] = &[
    (slot::HDR_COLOUR, BindingKind::FilterableTexture),
    (slot::SAMPLER, BindingKind::Sampler),
    (slot::PARAMS, BindingKind::Uniform),
    (slot::BLOOM, BindingKind::FilterableTexture),
    (slot::AO, BindingKind::FilterableTexture),
    (slot::CONTACT_SHADOW, BindingKind::FilterableTexture),
    (slot::SCENE_DEPTH, BindingKind::DepthTexture),
    (slot::LIC, BindingKind::FilterableTexture),
    (slot::FOREGROUND_DEPTH, BindingKind::DepthTexture),
    (slot::EXPOSURE, BindingKind::StorageReadOnly),
    (slot::GRADE_LUT, BindingKind::FilterableTexture),
];

/// Build the tone-map bind group layout from the binding table.
pub(crate) fn create_tone_map_bgl(device: &crate::gpu::Device) -> crate::gpu::BindGroupLayout {
    let entries: Vec<crate::gpu::BindGroupLayoutEntry> = BINDINGS
        .iter()
        .map(|&(binding, kind)| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: match kind {
                BindingKind::FilterableTexture => crate::gpu::BindingType::Texture {
                    sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    view_dimension: crate::gpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                BindingKind::DepthTexture => crate::gpu::BindingType::Texture {
                    sample_type: crate::gpu::TextureSampleType::Depth,
                    view_dimension: crate::gpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                BindingKind::Sampler => {
                    crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering)
                }
                BindingKind::Uniform => crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindingKind::StorageReadOnly => crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("tone_map_bgl"),
        entries: &entries,
    })
}

/// Which effect inputs feed the composite this frame.
///
/// One flag per selectable input: `true` binds the effect's live per-viewport
/// texture, `false` binds its neutral placeholder. The same flags drive the
/// `ToneMapUniform` enable lanes, so the shader's branches and the bound
/// views cannot disagree.
#[derive(Clone, Copy, Default)]
pub(crate) struct CompositeInputs {
    pub(crate) bloom: bool,
    pub(crate) ssao: bool,
    pub(crate) contact_shadows: bool,
    pub(crate) lic: bool,
    /// Depth of field substitutes the primary colour input
    /// (`slot::HDR_COLOUR`) with the DoF output rather than occupying a slot
    /// of its own.
    pub(crate) dof: bool,
    /// The foreground pass ran this frame; bind its depth as the coverage
    /// mask.
    pub(crate) foreground: bool,
    /// Colour-grading LUT to bind at `slot::GRADE_LUT`, already validated
    /// against the texture store (`None` binds the neutral placeholder).
    pub(crate) grade_lut: Option<crate::resources::TextureId>,
}
