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

/// Additive blend: `dst.rgb + src.rgb`, alpha unchanged. Used by the sprite and
/// particle draw paths for glowing / emissive accumulation.
pub(crate) const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Premultiplied-alpha blend: `src.rgb + dst.rgb * (1 - src.a)`. Used by the
/// sprite and particle draw paths when the source colour already carries its
/// alpha premultiplied.
pub(crate) const PREMULTIPLIED_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

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
        render_pipeline(
            device,
            RenderPipelineDesc {
                label: desc.label,
                layout: desc.layout,
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
                depth_stencil: Some(scene_depth_stencil(desc.depth_write, desc.depth_compare)),
                multisample: wgpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
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
    render_pipeline(
        device,
        RenderPipelineDesc {
            label,
            layout,
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
            cache: None,
        },
    )
}

/// Build an outline selection-mask pipeline: the item's geometry drawn into a
/// single-channel mask target, depth-tested against the scene so only visible
/// pixels are marked. The mask format, vertex layout, cull mode, depth write,
/// and depth compare vary per item and are passed in; the rest is fixed
/// (triangle list, `Depth24PlusStencil8`, default stencil and bias, single
/// sample, no blend, both stages `vs_main` / `fs_main`).
///
/// `depth_compare` must match the item's main opaque pipeline so the mask marks
/// exactly the pixels that survived the depth test in the colour pass. `cull` is
/// `Back` for closed solids and `None` otherwise; `depth_write` is off for
/// billboards and screen-space items that do not own scene depth.
pub(crate) fn build_outline_mask_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    mask_format: wgpu::TextureFormat,
    vertex_buffers: &[wgpu::VertexBufferLayout],
    cull: Option<wgpu::Face>,
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    render_pipeline(
        device,
        RenderPipelineDesc {
            label,
            layout,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: mask_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: cull,
                ..Default::default()
            },
            depth_stencil: Some(scene_depth_stencil(depth_write, depth_compare)),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
        },
    )
}

/// Create a compute pipeline. Every compute pipeline in the crate has the same
/// shape: a shader, its layout, and an entry point, with default compilation
/// options and no cache. This is the one place the crate calls
/// `create_compute_pipeline`, so a wgpu upgrade only has to be audited here.
pub(crate) fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Pipeline layout from a list of bind group layouts, with no push-constant
/// ranges (nothing in the crate uses push constants). This is the one place the
/// crate calls `create_pipeline_layout`, so the push-constant field that churns
/// across wgpu versions only has to be audited here.
pub(crate) fn pipeline_layout<'a>(
    device: &wgpu::Device,
    label: impl Into<wgpu::Label<'a>>,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: label.into(),
        bind_group_layouts,
        push_constant_ranges: &[],
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
    pipeline_layout(device, label, &[camera_bgl, per_item_bgl])
}

/// The parts of a render pipeline that vary between call sites. The two fields
/// that churn across wgpu versions (`multiview`, always `None`; `cache`, passed
/// through) are filled by [`render_pipeline`], so a version bump only touches
/// that one function instead of every descriptor literal.
pub(crate) struct RenderPipelineDesc<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub vertex: wgpu::VertexState<'a>,
    pub fragment: Option<wgpu::FragmentState<'a>>,
    pub primitive: wgpu::PrimitiveState,
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub multisample: wgpu::MultisampleState,
    pub cache: Option<&'a wgpu::PipelineCache>,
}

/// Create a render pipeline from the parts that vary ([`RenderPipelineDesc`]),
/// filling `multiview: None`. This is the one place the crate calls
/// `create_render_pipeline`, so the `multiview` field that changes shape across
/// wgpu versions only has to be audited here.
pub(crate) fn render_pipeline(
    device: &wgpu::Device,
    desc: RenderPipelineDesc,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(desc.label),
        layout: Some(desc.layout),
        vertex: desc.vertex,
        fragment: desc.fragment,
        primitive: desc.primitive,
        depth_stencil: desc.depth_stencil,
        multisample: desc.multisample,
        multiview: None,
        cache: desc.cache,
    })
}

/// A depth-stencil state with the given format, depth write flag, and compare
/// function, using the default stencil state and depth bias. Centralises the
/// `DepthStencilState` construction that changes shape across wgpu versions.
pub(crate) fn depth_stencil(
    format: wgpu::TextureFormat,
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format,
        depth_write_enabled,
        depth_compare,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// The scene depth-stencil state: `Depth24PlusStencil8` shared by every scene
/// render pass, parameterised by the depth write flag and compare function that
/// vary per pipeline.
pub(crate) fn scene_depth_stencil(
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::DepthStencilState {
    depth_stencil(
        wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled,
        depth_compare,
    )
}

/// Create a render-bundle encoder, filling `multiview: None`. This is the one
/// place the crate calls `create_render_bundle_encoder`, so the `multiview`
/// field that changes shape across wgpu versions is audited here alongside the
/// render-pipeline path.
pub(crate) fn render_bundle_encoder<'a>(
    device: &'a wgpu::Device,
    label: &str,
    color_formats: &[Option<wgpu::TextureFormat>],
    depth_stencil: Option<wgpu::RenderBundleDepthStencil>,
    sample_count: u32,
) -> wgpu::RenderBundleEncoder<'a> {
    device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
        label: Some(label),
        color_formats,
        depth_stencil,
        sample_count,
        multiview: None,
    })
}

/// Write `bytes` into the front of a buffer slice mapped at creation. Wraps
/// `get_mapped_range_mut` + `copy_from_slice`; the caller still owns the
/// matching `unmap()`. `bytes` may be shorter than the slice (a buffer padded
/// to a minimum size), in which case only the leading `bytes.len()` are
/// written. This is the one place the crate maps a buffer for writing, so the
/// mapped-view API change across wgpu versions is audited here.
pub(crate) fn write_mapped(slice: wgpu::BufferSlice, bytes: &[u8]) {
    slice.get_mapped_range_mut()[..bytes.len()].copy_from_slice(bytes);
}

/// Comparison sampler for shadow-map PCF: linear filtering with a depth compare
/// function, edge-clamped by default.
pub(crate) fn comparison_sampler(
    device: &wgpu::Device,
    label: &str,
    compare: wgpu::CompareFunction,
) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        compare: Some(compare),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// Linear-filtered sampler clamped to edge on all axes, with linear mip
/// filtering. Like [`clamp_linear_sampler`] but samples across the mip chain
/// (used by the volume LUT lookups).
pub(crate) fn clamp_linear_mip_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// Run `f` under a wgpu validation error scope, returning its result alongside
/// any validation error captured while it ran. This is the one place the crate
/// uses `push_error_scope` / `pop_error_scope`, so the error-scope API change
/// across wgpu versions is audited here.
pub(crate) fn capture_validation<T>(
    device: &wgpu::Device,
    f: impl FnOnce() -> T,
) -> (T, Option<wgpu::Error>) {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = f();
    let captured = block_on_simple(device.pop_error_scope());
    (value, captured)
}

/// Tiny sync executor that polls a future until it resolves. wgpu's
/// `pop_error_scope` resolves on the next driver poll, which the device itself
/// drives; spinning here is fine because validation completes without going
/// through the device's command queue.
fn block_on_simple<F: std::future::Future>(mut fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    // SAFETY: we own `fut` on the stack and never move it after this point.
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}
