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

use crate::gpu::ShaderStages;

/// Create a WGSL shader module. This is the one place the crate calls
/// `create_shader_module`, so a wgpu upgrade that changes shader-module
/// construction only has to be audited here.
///
/// `source` accepts a baked `&'static str` (via [`wgsl_source!`]) or an owned
/// `String` (a shader composed at runtime, e.g. by the deform registry).
pub(crate) fn wgsl_module<'a>(
    device: &crate::gpu::Device,
    label: &str,
    source: impl Into<std::borrow::Cow<'a, str>>,
) -> crate::gpu::ShaderModule {
    device.create_shader_module(crate::gpu::ShaderModuleDescriptor {
        label: Some(label),
        source: crate::gpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Remove the pixel-inspector debug block (bracketed by `BEGIN_DEBUG_VIS` /
/// `END_DEBUG_VIS` in debug_vis.wgsl) from a lit shader source unless `keep`
/// is set.
///
/// The block writes per-fragment values to the `debug_frag_buf` storage
/// buffer. A fragment shader with a buffer write has an observable side
/// effect, so the driver cannot reject occluded fragments with the early
/// depth test: every rasterized fragment of every hidden surface runs the
/// full lit shader. On the roman_mix reference scene that costs the scene
/// pass an order of magnitude. The lit pipelines therefore compile without
/// the block by default and are rebuilt from the full source only while
/// `DebugVis` is active (see `rebuild_mesh_pipelines`).
pub(crate) fn strip_debug_vis<'a>(
    source: impl Into<std::borrow::Cow<'a, str>>,
    keep: bool,
) -> std::borrow::Cow<'a, str> {
    let source = source.into();
    if keep {
        return source;
    }
    let Some(start) = source.find("// BEGIN_DEBUG_VIS") else {
        return source;
    };
    const END: &str = "// END_DEBUG_VIS";
    let Some(end) = source[start..].find(END) else {
        return source;
    };
    let end = start + end + END.len();
    let mut out = String::with_capacity(source.len() - (end - start));
    out.push_str(&source[..start]);
    out.push_str(&source[end..]);
    std::borrow::Cow::Owned(out)
}

/// Diagnostic knob: with `VIEWPORT_MESH_NO_DISCARD` set in the environment,
/// strip every `discard;` statement from the given mesh-shader source before
/// module creation. A fragment shader that contains `discard` forces the GPU
/// to defer depth writes, which weakens or disables early depth rejection, so
/// occluded fragments can still run the full lit shader. Compiling the mesh
/// shaders without `discard` lets a benchmark A/B that cost directly.
///
/// With the knob active the mesh shaders' discard paths (clip planes, clip
/// volumes, alpha mask) become no-ops, so only use it on scenes that render
/// none of those; it is a measurement tool, not a rendering mode.
pub(crate) fn strip_mesh_discards<'a>(
    source: impl Into<std::borrow::Cow<'a, str>>,
) -> std::borrow::Cow<'a, str> {
    let source = source.into();
    if std::env::var_os("VIEWPORT_MESH_NO_DISCARD").is_none() {
        return source;
    }
    static NOTICE: std::sync::Once = std::sync::Once::new();
    NOTICE.call_once(|| {
        eprintln!(
            "viewport-lib: VIEWPORT_MESH_NO_DISCARD active: mesh shaders compiled without \
             discard (clip planes, clip volumes, and alpha mask are no-ops)"
        );
    });
    std::borrow::Cow::Owned(strip_discards(&source))
}

/// Diagnostic knob: with `VIEWPORT_MESH_PBR_ONLY` set in the environment, remove
/// the alternate-shading-model regions bracketed by `// BEGIN_PBR_STRIP` /
/// `// END_PBR_STRIP` (Blinn-Phong, matcap, uv-vis, per-face colour) from the mesh
/// shader source before module creation, leaving a PBR-only fragment shader.
///
/// This is a benchmark A/B tool for the "compose built-in shading models as
/// separate bodies" question. It produces the smaller, specialised shader that a
/// composed PBR-only body would compile to, so its per-fragment cost can be
/// measured against the full branched shader on the same scene. Only render PBR
/// materials while it is active: the stripped shader leaves `final_rgb` undefined
/// on a non-PBR (`use_pbr == 0`) draw. It is a measurement tool, not a rendering
/// mode; the markers are inert comments when the knob is off.
pub(crate) fn strip_mesh_non_pbr<'a>(
    source: impl Into<std::borrow::Cow<'a, str>>,
) -> std::borrow::Cow<'a, str> {
    let source = source.into();
    if std::env::var_os("VIEWPORT_MESH_PBR_ONLY").is_none() {
        return source;
    }
    const BEGIN: &str = "// BEGIN_PBR_STRIP";
    const END: &str = "// END_PBR_STRIP";
    if !source.contains(BEGIN) {
        return source;
    }
    static NOTICE: std::sync::Once = std::sync::Once::new();
    NOTICE.call_once(|| {
        eprintln!(
            "viewport-lib: VIEWPORT_MESH_PBR_ONLY active: mesh shaders compiled PBR-only \
             (Blinn-Phong, matcap, uv-vis, and per-face colour paths removed)"
        );
    });
    std::borrow::Cow::Owned(strip_pbr_regions(&source))
}

/// Unconditionally remove the `BEGIN_PBR_STRIP` / `END_PBR_STRIP` regions
/// (Blinn-Phong and the alternate shading-model branches) from a mesh shader
/// source. Core of the `VIEWPORT_MESH_PBR_ONLY` knob above; also applied to
/// every shading-hook-composed module, whose materials always shade on the
/// PBR loop.
pub(crate) fn strip_pbr_regions(source: &str) -> String {
    const BEGIN: &str = "// BEGIN_PBR_STRIP";
    const END: &str = "// END_PBR_STRIP";
    let mut s = source.to_string();
    while let Some(start) = s.find(BEGIN) {
        let Some(rel_end) = s[start..].find(END) else {
            break;
        };
        let end = start + rel_end + END.len();
        s.replace_range(start..end, "");
    }
    s
}

/// Remove every `discard;` statement from a WGSL source.
///
/// A fragment shader containing `discard` restricts hardware early depth
/// rejection for every pipeline compiled from it, even when the discard is
/// behind a uniform branch that never fires: the classification is made per
/// pipeline at compile time from static shader properties. The lit instanced
/// pipelines are therefore built twice, once from the full source and once
/// from this discard-free twin; the draw loop picks the twin for opaque
/// batches whenever the frame has no active clip planes or clip volumes and
/// the batch carries no alpha-mask instances, which restores early-Z on the
/// common fully-opaque path with identical output.
pub(crate) fn strip_discards(source: &str) -> String {
    // Token-boundary check: composed sources include consumer-supplied
    // deformer bodies, and a bare substring replace would corrupt an
    // identifier like `should_discard;`. A missed strip is only a lost
    // early-Z opportunity, never a correctness problem, so err toward
    // keeping anything that is not exactly the statement `discard;`.
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(pos) = rest.find("discard;") {
        let boundary_ok = rest[..pos]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
        out.push_str(&rest[..pos]);
        if boundary_ok {
            out.push_str("/* discard stripped */");
        } else {
            out.push_str("discard;");
        }
        rest = &rest[pos + "discard;".len()..];
    }
    out.push_str(rest);
    out
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
pub(crate) fn uniform_entry(
    binding: u32,
    visibility: ShaderStages,
) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: crate::gpu::BindingType::Buffer {
            ty: crate::gpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A filterable float 2D texture bind group layout entry.
pub(crate) fn texture_entry(
    binding: u32,
    visibility: ShaderStages,
) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: crate::gpu::BindingType::Texture {
            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
            view_dimension: crate::gpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A filtering sampler bind group layout entry.
pub(crate) fn sampler_entry(
    binding: u32,
    visibility: ShaderStages,
) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Bind group layout with a single uniform buffer at binding 0.
pub(crate) fn uniform_bgl(
    device: &crate::gpu::Device,
    label: &str,
    visibility: ShaderStages,
) -> crate::gpu::BindGroupLayout {
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[uniform_entry(0, visibility)],
    })
}

/// Bind group layout: filterable texture at binding 0 + filtering sampler at
/// binding 1, both visible to `visibility`. The common shape for a
/// full-screen composite / blit pass.
pub(crate) fn texture_sampler_bgl(
    device: &crate::gpu::Device,
    label: &str,
    visibility: ShaderStages,
) -> crate::gpu::BindGroupLayout {
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[texture_entry(0, visibility), sampler_entry(1, visibility)],
    })
}

/// Bind group layout: uniform buffer at binding 0 (visible to `uniform_vis`),
/// a filterable texture at binding 1, and a filtering sampler at binding 2
/// (both visible to `tex_vis`). The standard scivis per-item layout: an item
/// uniform plus an optional colour-LUT texture and sampler.
pub(crate) fn uniform_texture_sampler_bgl(
    device: &crate::gpu::Device,
    label: &str,
    uniform_vis: ShaderStages,
    tex_vis: ShaderStages,
) -> crate::gpu::BindGroupLayout {
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
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
pub(crate) fn clamp_linear_sampler(
    device: &crate::gpu::Device,
    label: &str,
) -> crate::gpu::Sampler {
    device.create_sampler(&crate::gpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: crate::gpu::AddressMode::ClampToEdge,
        address_mode_v: crate::gpu::AddressMode::ClampToEdge,
        address_mode_w: crate::gpu::AddressMode::ClampToEdge,
        mag_filter: crate::gpu::FilterMode::Linear,
        min_filter: crate::gpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// Nearest-filtered sampler clamped to edge on all axes. Used where
/// interpolation would blur discrete data (index buffers, nearest blits).
pub(crate) fn clamp_nearest_sampler(
    device: &crate::gpu::Device,
    label: &str,
) -> crate::gpu::Sampler {
    device.create_sampler(&crate::gpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: crate::gpu::AddressMode::ClampToEdge,
        address_mode_v: crate::gpu::AddressMode::ClampToEdge,
        address_mode_w: crate::gpu::AddressMode::ClampToEdge,
        mag_filter: crate::gpu::FilterMode::Nearest,
        min_filter: crate::gpu::FilterMode::Nearest,
        ..Default::default()
    })
}

/// Linear-filtered sampler that repeats on all axes. Used for tiling textures
/// (decals, patterned materials). `mipmap_filter` varies: most callers want
/// `Nearest`, uploaded user textures pick it from the mip chain at runtime.
pub(crate) fn repeat_linear_sampler(
    device: &crate::gpu::Device,
    label: &str,
    mipmap_filter: crate::gpu::FilterMode,
) -> crate::gpu::Sampler {
    device.create_sampler(&crate::gpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: crate::gpu::AddressMode::Repeat,
        address_mode_v: crate::gpu::AddressMode::Repeat,
        address_mode_w: crate::gpu::AddressMode::Repeat,
        mag_filter: crate::gpu::FilterMode::Linear,
        min_filter: crate::gpu::FilterMode::Linear,
        mipmap_filter: dmipmap(mipmap_filter),
        ..Default::default()
    })
}

/// Wrap a mip filter for the current wgpu version's `SamplerDescriptor`. 27
/// reuses `FilterMode` for the mip filter; 28 split it into `MipmapFilterMode`.
#[cfg(feature = "wgpu27")]
pub fn dmipmap(filter: crate::gpu::FilterMode) -> crate::gpu::FilterMode {
    filter
}
#[cfg(feature = "wgpu29")]
pub fn dmipmap(filter: crate::gpu::FilterMode) -> crate::gpu::MipmapFilterMode {
    match filter {
        crate::gpu::FilterMode::Nearest => crate::gpu::MipmapFilterMode::Nearest,
        crate::gpu::FilterMode::Linear => crate::gpu::MipmapFilterMode::Linear,
    }
}

/// Sampler for an equirectangular environment map: horizontal wrap (Repeat u),
/// vertical clamp (Clamp v), linear filtering including across mip levels. Used
/// by the image-based lighting passes that sample a lat-long HDR.
pub(crate) fn env_sampler(device: &crate::gpu::Device, label: &str) -> crate::gpu::Sampler {
    device.create_sampler(&crate::gpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: crate::gpu::AddressMode::Repeat,
        address_mode_v: crate::gpu::AddressMode::ClampToEdge,
        mag_filter: crate::gpu::FilterMode::Linear,
        min_filter: crate::gpu::FilterMode::Linear,
        mipmap_filter: dmipmap(crate::gpu::FilterMode::Linear),
        ..Default::default()
    })
}

/// Additive blend: `dst.rgb + src.rgb`, alpha unchanged. Used by the sprite and
/// particle draw paths for glowing / emissive accumulation.
pub(crate) const ADDITIVE_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
    color: crate::gpu::BlendComponent {
        src_factor: crate::gpu::BlendFactor::One,
        dst_factor: crate::gpu::BlendFactor::One,
        operation: crate::gpu::BlendOperation::Add,
    },
    alpha: crate::gpu::BlendComponent {
        src_factor: crate::gpu::BlendFactor::One,
        dst_factor: crate::gpu::BlendFactor::One,
        operation: crate::gpu::BlendOperation::Add,
    },
};

/// Premultiplied-alpha blend: `src.rgb + dst.rgb * (1 - src.a)`. Used by the
/// sprite and particle draw paths when the source colour already carries its
/// alpha premultiplied.
pub(crate) const PREMULTIPLIED_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
    color: crate::gpu::BlendComponent {
        src_factor: crate::gpu::BlendFactor::One,
        dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
        operation: crate::gpu::BlendOperation::Add,
    },
    alpha: crate::gpu::BlendComponent {
        src_factor: crate::gpu::BlendFactor::One,
        dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
        operation: crate::gpu::BlendOperation::Add,
    },
};

/// The parts of a scene render pipeline that vary between features. The rest of
/// the descriptor (depth format `Depth24PlusStencil8`, default stencil and
/// bias, `ColorWrites::ALL`, default front face, no multiview or cache) is held
/// constant by [`build_dual_pipeline`]. The vertex and fragment stages share
/// one shader module, which is the shape every scivis feature uses.
pub(crate) struct DualPipelineDesc<'a> {
    pub label: &'a str,
    pub layout: &'a crate::gpu::PipelineLayout,
    pub shader: &'a crate::gpu::ShaderModule,
    pub vertex_entry: &'a str,
    pub fragment_entry: &'a str,
    pub vertex_buffers: &'a [crate::gpu::VertexBufferLayout<'a>],
    pub blend: Option<crate::gpu::BlendState>,
    pub topology: crate::gpu::PrimitiveTopology,
    pub cull_mode: Option<crate::gpu::Face>,
    pub depth_write: bool,
    pub depth_compare: crate::gpu::CompareFunction,
    pub sample_count: u32,
    /// LDR swapchain format; the HDR variant is always `Rgba16Float`.
    pub ldr_format: crate::gpu::TextureFormat,
}

/// Build the LDR + HDR pair of a scene render pipeline from the parts that vary
/// ([`DualPipelineDesc`]), holding the shared depth / stencil / target-write
/// state constant. The two variants differ only in colour target format
/// (`desc.ldr_format` vs `Rgba16Float`), which is the invariant `DualPipeline`
/// encodes.
pub(crate) fn build_dual_pipeline(
    device: &crate::gpu::Device,
    desc: &DualPipelineDesc,
) -> crate::resources::types::DualPipeline {
    let make = |format: crate::gpu::TextureFormat| {
        render_pipeline(
            device,
            RenderPipelineDesc {
                label: desc.label,
                layout: desc.layout,
                vertex: crate::gpu::VertexState {
                    module: desc.shader,
                    entry_point: Some(desc.vertex_entry),
                    buffers: desc.vertex_buffers,
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: desc.shader,
                    entry_point: Some(desc.fragment_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format,
                        blend: desc.blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: desc.topology,
                    cull_mode: desc.cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(scene_depth_stencil(desc.depth_write, desc.depth_compare)),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    };
    crate::resources::types::DualPipeline {
        ldr: make(desc.ldr_format),
        hdr: make(crate::gpu::TextureFormat::Rgba16Float),
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
    device: &crate::gpu::Device,
    label: &str,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    target_format: crate::gpu::TextureFormat,
    blend: Option<crate::gpu::BlendState>,
) -> crate::gpu::RenderPipeline {
    render_pipeline(
        device,
        RenderPipelineDesc {
            label,
            layout,
            vertex: crate::gpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(crate::gpu::ColorTargetState {
                    format: target_format,
                    blend,
                    write_mask: crate::gpu::ColorWrites::ALL,
                })],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: crate::gpu::MultisampleState::default(),
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
    device: &crate::gpu::Device,
    label: &str,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    mask_format: crate::gpu::TextureFormat,
    vertex_buffers: &[crate::gpu::VertexBufferLayout],
    cull: Option<crate::gpu::Face>,
    depth_write: bool,
    depth_compare: crate::gpu::CompareFunction,
) -> crate::gpu::RenderPipeline {
    render_pipeline(
        device,
        RenderPipelineDesc {
            label,
            layout,
            vertex: crate::gpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: vertex_buffers,
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(crate::gpu::ColorTargetState {
                    format: mask_format,
                    blend: None,
                    write_mask: crate::gpu::ColorWrites::ALL,
                })],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: cull,
                ..Default::default()
            },
            depth_stencil: Some(scene_depth_stencil(depth_write, depth_compare)),
            multisample: crate::gpu::MultisampleState::default(),
            cache: None,
        },
    )
}

/// Create a compute pipeline. Every compute pipeline in the crate has the same
/// shape: a shader, its layout, and an entry point, with default compilation
/// options and no cache. This is the one place the crate calls
/// `create_compute_pipeline`, so a wgpu upgrade only has to be audited here.
pub(crate) fn compute_pipeline(
    device: &crate::gpu::Device,
    label: &str,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    entry: &str,
) -> crate::gpu::ComputePipeline {
    device.create_compute_pipeline(&crate::gpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Pipeline layout from a list of bind group layouts, with no push-constant
/// ranges (nothing in the crate uses push constants). This is the one place the
/// crate calls `create_pipeline_layout`, so the push-constant field that churns
/// across wgpu versions only has to be audited here.
pub fn pipeline_layout<'a>(
    device: &crate::gpu::Device,
    label: impl Into<crate::gpu::Label<'a>>,
    bind_group_layouts: &[&crate::gpu::BindGroupLayout],
) -> crate::gpu::PipelineLayout {
    // 27 takes `push_constant_ranges` and a `&[&BindGroupLayout]`; 29 replaced
    // push constants with `immediate_size` and takes `&[Option<&BindGroupLayout>]`.
    #[cfg(feature = "wgpu27")]
    let layout = device.create_pipeline_layout(&crate::gpu::PipelineLayoutDescriptor {
        label: label.into(),
        bind_group_layouts,
        push_constant_ranges: &[],
    });
    #[cfg(feature = "wgpu29")]
    let layout = {
        let bgls: Vec<Option<&crate::gpu::BindGroupLayout>> =
            bind_group_layouts.iter().map(|b| Some(*b)).collect();
        device.create_pipeline_layout(&crate::gpu::PipelineLayoutDescriptor {
            label: label.into(),
            bind_group_layouts: &bgls,
            immediate_size: 0,
        })
    };
    layout
}

/// Pipeline layout with the standard scene binding convention:
/// group 0 = camera, group 1 = the feature's per-item bind group layout.
pub(crate) fn standard_scene_layout(
    device: &crate::gpu::Device,
    label: &str,
    camera_bgl: &crate::gpu::BindGroupLayout,
    per_item_bgl: &crate::gpu::BindGroupLayout,
) -> crate::gpu::PipelineLayout {
    pipeline_layout(device, label, &[camera_bgl, per_item_bgl])
}

/// The parts of a render pipeline that vary between call sites. The two fields
/// that churn across wgpu versions (`multiview`, always `None`; `cache`, passed
/// through) are filled by [`render_pipeline`], so a version bump only touches
/// that one function instead of every descriptor literal.
pub struct RenderPipelineDesc<'a> {
    /// Debug label for the pipeline.
    pub label: &'a str,
    /// Pipeline layout (bind group layouts). See [`pipeline_layout`].
    pub layout: &'a crate::gpu::PipelineLayout,
    /// Vertex stage: shader module, entry point, and vertex buffer layouts.
    pub vertex: crate::gpu::VertexState<'a>,
    /// Fragment stage and its color targets, or `None` for a depth-only pass.
    pub fragment: Option<crate::gpu::FragmentState<'a>>,
    /// Primitive topology, cull mode, and front face.
    pub primitive: crate::gpu::PrimitiveState,
    /// Depth-stencil state, or `None` for a pass without a depth attachment.
    /// Build with [`depth_stencil`] or [`scene_depth_stencil`].
    pub depth_stencil: Option<crate::gpu::DepthStencilState>,
    /// Multisample (MSAA) state.
    pub multisample: crate::gpu::MultisampleState,
    /// Optional pipeline cache to speed up creation.
    pub cache: Option<&'a crate::gpu::PipelineCache>,
}

/// Create a render pipeline from the parts that vary ([`RenderPipelineDesc`]),
/// filling `multiview: None`. This is the one place the crate calls
/// `create_render_pipeline`, so the `multiview` field that changes shape across
/// wgpu versions only has to be audited here.
pub fn render_pipeline(
    device: &crate::gpu::Device,
    desc: RenderPipelineDesc,
) -> crate::gpu::RenderPipeline {
    device.create_render_pipeline(&crate::gpu::RenderPipelineDescriptor {
        label: Some(desc.label),
        layout: Some(desc.layout),
        vertex: desc.vertex,
        fragment: desc.fragment,
        primitive: desc.primitive,
        depth_stencil: desc.depth_stencil,
        multisample: desc.multisample,
        // 29 renamed `multiview` to the `multiview_mask` bitmask form.
        #[cfg(feature = "wgpu27")]
        multiview: None,
        #[cfg(feature = "wgpu29")]
        multiview_mask: None,
        cache: desc.cache,
    })
}

/// Wrap a depth-write flag for the current wgpu version's `DepthStencilState`.
/// 27 takes a bare `bool`; 29 takes `Option<bool>`.
#[cfg(feature = "wgpu27")]
pub fn dwrite(enabled: bool) -> bool {
    enabled
}
#[cfg(feature = "wgpu29")]
pub fn dwrite(enabled: bool) -> Option<bool> {
    Some(enabled)
}

/// Wrap a depth-compare function for the current wgpu version's
/// `DepthStencilState`. 27 takes a bare `CompareFunction`; 29 takes
/// `Option<CompareFunction>`.
#[cfg(feature = "wgpu27")]
pub fn dcompare(compare: crate::gpu::CompareFunction) -> crate::gpu::CompareFunction {
    compare
}
#[cfg(feature = "wgpu29")]
pub fn dcompare(compare: crate::gpu::CompareFunction) -> Option<crate::gpu::CompareFunction> {
    Some(compare)
}

/// A depth-stencil state with the given format, depth write flag, and compare
/// function, using the default stencil state and depth bias. Centralises the
/// `DepthStencilState` construction that changes shape across wgpu versions.
pub fn depth_stencil(
    format: crate::gpu::TextureFormat,
    depth_write_enabled: bool,
    depth_compare: crate::gpu::CompareFunction,
) -> crate::gpu::DepthStencilState {
    crate::gpu::DepthStencilState {
        format,
        depth_write_enabled: dwrite(depth_write_enabled),
        depth_compare: dcompare(depth_compare),
        stencil: crate::gpu::StencilState::default(),
        bias: crate::gpu::DepthBiasState::default(),
    }
}

/// The scene depth-stencil state: `Depth24PlusStencil8` shared by every scene
/// render pass, parameterised by the depth write flag and compare function that
/// vary per pipeline.
pub fn scene_depth_stencil(
    depth_write_enabled: bool,
    depth_compare: crate::gpu::CompareFunction,
) -> crate::gpu::DepthStencilState {
    depth_stencil(
        crate::gpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled,
        depth_compare,
    )
}

/// Create a render-bundle encoder, filling `multiview: None`. This is the one
/// place the crate calls `create_render_bundle_encoder`, so the `multiview`
/// field that changes shape across wgpu versions is audited here alongside the
/// render-pipeline path.
pub(crate) fn render_bundle_encoder<'a>(
    device: &'a crate::gpu::Device,
    label: &str,
    color_formats: &[Option<crate::gpu::TextureFormat>],
    depth_stencil: Option<crate::gpu::RenderBundleDepthStencil>,
    sample_count: u32,
) -> crate::gpu::RenderBundleEncoder<'a> {
    device.create_render_bundle_encoder(&crate::gpu::RenderBundleEncoderDescriptor {
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
pub fn write_mapped(slice: crate::gpu::BufferSlice, bytes: &[u8]) {
    // 27's mapped view derefs to `[u8]` and is indexed directly; 29's
    // `BufferViewMut` is write-only and exposes a `slice(..)` -> `WriteOnly`.
    #[cfg(feature = "wgpu27")]
    slice.get_mapped_range_mut()[..bytes.len()].copy_from_slice(bytes);
    #[cfg(feature = "wgpu29")]
    slice
        .get_mapped_range_mut()
        .slice(..bytes.len())
        .copy_from_slice(bytes);
}

/// Comparison sampler for shadow-map PCF: linear filtering with a depth compare
/// function, edge-clamped by default.
pub(crate) fn comparison_sampler(
    device: &crate::gpu::Device,
    label: &str,
    compare: crate::gpu::CompareFunction,
) -> crate::gpu::Sampler {
    device.create_sampler(&crate::gpu::SamplerDescriptor {
        label: Some(label),
        compare: Some(compare),
        mag_filter: crate::gpu::FilterMode::Linear,
        min_filter: crate::gpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// Linear-filtered sampler clamped to edge on all axes, with linear mip
/// filtering. Like [`clamp_linear_sampler`] but samples across the mip chain
/// (used by the volume LUT lookups).
pub(crate) fn clamp_linear_mip_sampler(
    device: &crate::gpu::Device,
    label: &str,
) -> crate::gpu::Sampler {
    device.create_sampler(&crate::gpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: crate::gpu::AddressMode::ClampToEdge,
        address_mode_v: crate::gpu::AddressMode::ClampToEdge,
        address_mode_w: crate::gpu::AddressMode::ClampToEdge,
        mag_filter: crate::gpu::FilterMode::Linear,
        min_filter: crate::gpu::FilterMode::Linear,
        mipmap_filter: dmipmap(crate::gpu::FilterMode::Linear),
        ..Default::default()
    })
}

/// Run `f` under a wgpu validation error scope, returning its result alongside
/// any validation error captured while it ran. This is the one place the crate
/// uses `push_error_scope` / `pop_error_scope`, so the error-scope API change
/// across wgpu versions is audited here.
pub(crate) fn capture_validation<T>(
    device: &crate::gpu::Device,
    f: impl FnOnce() -> T,
) -> (T, Option<crate::gpu::Error>) {
    // 27 pops the scope through a `Device::pop_error_scope` future; 29's
    // `push_error_scope` returns a guard whose `pop()` is the future.
    #[cfg(feature = "wgpu27")]
    {
        device.push_error_scope(crate::gpu::ErrorFilter::Validation);
        let value = f();
        let captured = block_on_simple(device.pop_error_scope());
        (value, captured)
    }
    #[cfg(feature = "wgpu29")]
    {
        let guard = device.push_error_scope(crate::gpu::ErrorFilter::Validation);
        let value = f();
        let captured = block_on_simple(guard.pop());
        (value, captured)
    }
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

#[cfg(test)]
mod strip_debug_vis_tests {
    /// The lit shader families must lose the pixel-inspector storage write
    /// (which disables early depth rejection) when stripped, and keep it when
    /// the debug variant is requested.
    #[test]
    fn strips_debug_block_from_every_lit_shader() {
        let sources: [(&str, &str); 4] = [
            ("mesh", super::wgsl_source!("mesh")),
            ("mesh_instanced", super::wgsl_source!("mesh_instanced")),
            ("mesh_oit", super::wgsl_source!("mesh_oit")),
            (
                "mesh_instanced_oit",
                super::wgsl_source!("mesh_instanced_oit"),
            ),
        ];
        for (name, src) in sources {
            assert!(
                src.contains("debug_frag_buf["),
                "{name}: baked source lost the debug block; markers moved?"
            );
            let stripped = super::strip_debug_vis(src, false);
            assert!(
                !stripped.contains("debug_frag_buf["),
                "{name}: stripped module still writes debug_frag_buf"
            );
            assert!(
                !stripped.contains("BEGIN_DEBUG_VIS"),
                "{name}: stripped module kept the marker"
            );
            let kept = super::strip_debug_vis(src, true);
            assert!(
                kept.contains("debug_frag_buf["),
                "{name}: debug variant lost the write"
            );
        }
    }

    /// The discard-free twin of the lit instanced shader must lose every
    /// `discard` statement (any survivor silently forfeits the early-Z win),
    /// while identifiers that merely end in "discard" survive untouched:
    /// composed sources include consumer-supplied deformer bodies, and a
    /// substring replace would corrupt them.
    #[test]
    fn strips_discard_statements_but_not_identifiers() {
        let src = super::wgsl_source!("mesh_instanced");
        assert!(
            src.contains("discard;"),
            "baked mesh_instanced source has no discard; sites moved?"
        );
        let stripped = super::strip_discards(src);
        assert!(
            !stripped.contains("discard;"),
            "discard-free twin still contains a discard statement"
        );

        let consumer = "let should_discard;\n    if x { discard; }\n";
        let stripped = super::strip_discards(consumer);
        assert!(
            stripped.contains("should_discard;"),
            "identifier ending in discard was corrupted: {stripped}"
        );
        assert!(
            !stripped.contains("{ discard;"),
            "real discard statement survived: {stripped}"
        );
    }
}
