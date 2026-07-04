//! Small constructors for the wgpu descriptor boilerplate that repeats across
//! the per-feature `ensure_*` methods.
//!
//! Each feature still owns the parts that differ (vertex layouts, shaders,
//! blend modes, topology). These cover the parts that do not: the common bind
//! group layout shapes, the three sampler archetypes, and the
//! `group 0 = camera, group 1 = per-item` pipeline layout. The per-entry
//! constructors (`uniform_entry`, `texture_entry`, `sampler_entry`) are exposed
//! so the less common multi-binding layouts can compose them instead of
//! spelling out each `BindGroupLayoutEntry` by hand.

use wgpu::ShaderStages;

/// Create a WGSL shader module. This is the one place the crate calls
/// `create_shader_module`, so a wgpu upgrade that changes shader-module
/// construction only has to be audited here.
///
/// `source` accepts a baked `&'static str` (via [`wgsl_source!`]) or an owned
/// `String` (a shader composed at runtime, e.g. by the deform registry).
pub(crate) fn wgsl_module<'a>(
    device: &wgpu::Device,
    label: &str,
    source: impl Into<std::borrow::Cow<'a, str>>,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Embed a WGSL file baked into `OUT_DIR` by `build.rs`, by base name (no
/// extension). Expands to `include_str!(...)`, so the file is compiled into the
/// binary. Pass the result to [`wgsl_module`].
///
/// `wgsl_source!("point_cloud")` -> the contents of `$OUT_DIR/point_cloud.wgsl`.
macro_rules! wgsl_source {
    ($name:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/", $name, ".wgsl"))
    };
}
pub(crate) use wgsl_source;

/// A uniform-buffer bind group layout entry (non-dynamic, no min size).
pub(crate) fn uniform_entry(binding: u32, visibility: ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A filterable float 2D texture bind group layout entry.
pub(crate) fn texture_entry(binding: u32, visibility: ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A filtering sampler bind group layout entry.
pub(crate) fn sampler_entry(binding: u32, visibility: ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Bind group layout with a single uniform buffer at binding 0.
pub(crate) fn uniform_bgl(
    device: &wgpu::Device,
    label: &str,
    visibility: ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[uniform_entry(0, visibility)],
    })
}

/// Bind group layout: filterable texture at binding 0 + filtering sampler at
/// binding 1, both visible to `visibility`. The common shape for a
/// full-screen composite / blit pass.
pub(crate) fn texture_sampler_bgl(
    device: &wgpu::Device,
    label: &str,
    visibility: ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[texture_entry(0, visibility), sampler_entry(1, visibility)],
    })
}

/// Bind group layout: uniform buffer at binding 0 (visible to `uniform_vis`),
/// a filterable texture at binding 1, and a filtering sampler at binding 2
/// (both visible to `tex_vis`). The standard scivis per-item layout: an item
/// uniform plus an optional colour-LUT texture and sampler.
pub(crate) fn uniform_texture_sampler_bgl(
    device: &wgpu::Device,
    label: &str,
    uniform_vis: ShaderStages,
    tex_vis: ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            uniform_entry(0, uniform_vis),
            texture_entry(1, tex_vis),
            sampler_entry(2, tex_vis),
        ],
    })
}

/// Linear-filtered sampler clamped to edge on all axes. The default sampler for
/// texture lookups that must not wrap (LUTs, composite targets, most content).
pub(crate) fn clamp_linear_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// Nearest-filtered sampler clamped to edge on all axes. Used where
/// interpolation would blur discrete data (index buffers, nearest blits).
pub(crate) fn clamp_nearest_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    })
}

/// Linear-filtered sampler that repeats on all axes. Used for tiling textures
/// (decals, patterned materials). `mipmap_filter` varies: most callers want
/// `Nearest`, uploaded user textures pick it from the mip chain at runtime.
pub(crate) fn repeat_linear_sampler(
    device: &wgpu::Device,
    label: &str,
    mipmap_filter: wgpu::FilterMode,
) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter,
        ..Default::default()
    })
}

/// Sampler for an equirectangular environment map: horizontal wrap (Repeat u),
/// vertical clamp (Clamp v), linear filtering including across mip levels. Used
/// by the image-based lighting passes that sample a lat-long HDR.
pub(crate) fn env_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// The parts of a scene render pipeline that vary between features. The rest of
/// the descriptor (depth format `Depth24PlusStencil8`, default stencil and
/// bias, `ColorWrites::ALL`, default front face, no multiview or cache) is held
/// constant by [`build_dual_pipeline`]. The vertex and fragment stages share
/// one shader module, which is the shape every scivis feature uses.
pub(crate) struct DualPipelineDesc<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub shader: &'a wgpu::ShaderModule,
    pub vertex_entry: &'a str,
    pub fragment_entry: &'a str,
    pub vertex_buffers: &'a [wgpu::VertexBufferLayout<'a>],
    pub blend: Option<wgpu::BlendState>,
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_write: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub sample_count: u32,
    /// LDR swapchain format; the HDR variant is always `Rgba16Float`.
    pub ldr_format: wgpu::TextureFormat,
}

/// Build the LDR + HDR pair of a scene render pipeline from the parts that vary
/// ([`DualPipelineDesc`]), holding the shared depth / stencil / target-write
/// state constant. The two variants differ only in colour target format
/// (`desc.ldr_format` vs `Rgba16Float`), which is the invariant `DualPipeline`
/// encodes.
pub(crate) fn build_dual_pipeline(
    device: &wgpu::Device,
    desc: &DualPipelineDesc,
) -> crate::resources::types::DualPipeline {
    let make = |format: wgpu::TextureFormat| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(desc.label),
            layout: Some(desc.layout),
            vertex: wgpu::VertexState {
                module: desc.shader,
                entry_point: Some(desc.vertex_entry),
                buffers: desc.vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: desc.shader,
                entry_point: Some(desc.fragment_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: desc.blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: desc.topology,
                cull_mode: desc.cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: desc.depth_write,
                depth_compare: desc.depth_compare,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: desc.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    };
    crate::resources::types::DualPipeline {
        ldr: make(desc.ldr_format),
        hdr: make(wgpu::TextureFormat::Rgba16Float),
    }
}

/// Build a full-screen pass pipeline: one triangle-list draw covering the
/// target, no depth attachment, no culling, single sample. The vertex shader
/// generates the covering triangle from `vertex_index`, so there are no vertex
/// buffers. Both stages are `vs_main` / `fs_main` in `shader`. Post-process and
/// composite passes (tone map, bloom, SSAO, FXAA, OIT composite, upscales, the
/// scatter composites) all share this shape and differ only in target format
/// and blend.
pub(crate) fn build_fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Pipeline layout with the standard scene binding convention:
/// group 0 = camera, group 1 = the feature's per-item bind group layout.
pub(crate) fn standard_scene_layout(
    device: &wgpu::Device,
    label: &str,
    camera_bgl: &wgpu::BindGroupLayout,
    per_item_bgl: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[camera_bgl, per_item_bgl],
        push_constant_ranges: &[],
    })
}
