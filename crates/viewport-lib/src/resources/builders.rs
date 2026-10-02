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
pub fn wgsl_module<'a>(
    device: &crate::gpu::Device,
    label: &str,
    source: impl Into<std::borrow::Cow<'a, str>>,
) -> crate::gpu::ShaderModule {
    let build_start = web_time::Instant::now();
    let module = device.create_shader_module(crate::gpu::ShaderModuleDescriptor {
        label: Some(label),
        source: crate::gpu::ShaderSource::Wgsl(source.into()),
    });
    if build_log::enabled() {
        build_log::record(
            &format!("module {label}"),
            build_start.elapsed().as_secs_f32() * 1000.0,
        );
    }
    module
}

/// Prepend the module directive `@builtin(primitive_index)` needs on the
/// active wgpu leg: naga 29 requires `enable primitive_index;` before any
/// declaration, naga 27 rejects the directive. Only for sources compiled on a
/// device with [`PRIMITIVE_INDEX_FEATURE`](crate::gpu::PRIMITIVE_INDEX_FEATURE)
/// (the directive itself fails validation without the feature).
pub fn with_primitive_index_enable(src: &str) -> String {
    format!(
        "{}{}",
        crate::plugin_api::shared_wgsl::PICK_PRIM_ENABLE_WGSL,
        src
    )
}

/// Remove the pixel-inspector debug block (bracketed by `BEGIN_DEBUG_VIS` /
/// `END_DEBUG_VIS` in debug_vis.wgsl) from a lit shader source unless `keep`
/// is set.
///
/// The block declares a 24-element array of candidate quantities. It sits under
/// a uniform branch, so it does no work unless debug vis is on, but a lit
/// shader carrying the allocation pays for it in registers on every draw:
/// measured at about 5% of scene time on a fragment-bound scene, with identical
/// pixels either way. The lit pipelines therefore compile without the block by
/// default and are rebuilt from the full source only while `DebugVis` is active
/// (see `rebuild_mesh_pipelines`).
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

/// Diagnostic knob: with `VIEWPORT_MESH_BUILTIN_HOOK` set in the environment,
/// compose every lit mesh shader with the internal `builtin_pbr` shading hook
/// before module creation, so the built-in Cook-Torrance lighting runs
/// through the fragment-shading seam instead of inline.
///
/// This is the A/B tool for "what does the hook mechanism itself cost": the
/// composed module computes identical lighting via the same
/// `pbr_light_contrib`, plus the ShadingSurface fill and per-light
/// LightSample construction a real material plugin pays. Compare against the
/// plain baseline and against `VIEWPORT_MESH_PBR_ONLY` (which strips the
/// alternate branches without the hook) to separate branch-stripping gains
/// from hook overhead. Composition forces the PBR loop, so like the PBR_ONLY
/// knob it is only meaningful on scenes of PBR materials. Sources without
/// shade-slot markers (non-lit shaders) pass through untouched.
pub(crate) fn builtin_hook_env<'a>(
    source: impl Into<std::borrow::Cow<'a, str>>,
) -> std::borrow::Cow<'a, str> {
    let source = source.into();
    if std::env::var_os("VIEWPORT_MESH_BUILTIN_HOOK").is_none() {
        return source;
    }
    let Some(composed) = crate::resources::mesh_sidecar::shade::compose_builtin_pbr_hook(&source)
    else {
        return source;
    };
    static NOTICE: std::sync::Once = std::sync::Once::new();
    NOTICE.call_once(|| {
        eprintln!(
            "viewport-lib: VIEWPORT_MESH_BUILTIN_HOOK active: built-in PBR lighting runs \
             through the shading-hook seam (render PBR materials only)"
        );
    });
    std::borrow::Cow::Owned(composed)
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
pub fn uniform_entry(binding: u32, visibility: ShaderStages) -> crate::gpu::BindGroupLayoutEntry {
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
pub fn texture_entry(binding: u32, visibility: ShaderStages) -> crate::gpu::BindGroupLayoutEntry {
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
pub fn sampler_entry(binding: u32, visibility: ShaderStages) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Bind group layout with a single uniform buffer at binding 0.
pub fn uniform_bgl(
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
pub fn texture_sampler_bgl(
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
pub fn uniform_texture_sampler_bgl(
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
pub fn clamp_linear_sampler(device: &crate::gpu::Device, label: &str) -> crate::gpu::Sampler {
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
pub fn clamp_nearest_sampler(device: &crate::gpu::Device, label: &str) -> crate::gpu::Sampler {
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
pub fn repeat_linear_sampler(
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

/// Build a sampler from a material [`SamplerKey`](crate::scene::material::SamplerKey).
///
/// Maps the crate's wrap/filter enums onto the current wgpu version's
/// `SamplerDescriptor`. Anisotropy is clamped to `1..=16`, and because wgpu
/// requires linear min/mag/mip filtering whenever `anisotropy_clamp > 1`, a
/// `Nearest` key pins anisotropy back to `1` rather than silently forcing
/// linear.
pub(crate) fn sampler_from_key(
    device: &crate::gpu::Device,
    label: &str,
    key: &crate::scene::material::SamplerKey,
) -> crate::gpu::Sampler {
    use crate::scene::material::{TextureFilter, WrapMode};
    let wrap = |w: WrapMode| match w {
        WrapMode::Repeat => crate::gpu::AddressMode::Repeat,
        WrapMode::ClampToEdge => crate::gpu::AddressMode::ClampToEdge,
        WrapMode::MirrorRepeat => crate::gpu::AddressMode::MirrorRepeat,
    };
    let filter = match key.filter {
        TextureFilter::Nearest => crate::gpu::FilterMode::Nearest,
        TextureFilter::Linear => crate::gpu::FilterMode::Linear,
    };
    // Anisotropy only applies with linear filtering (wgpu constraint); keep it
    // at 1 for a nearest key so the descriptor stays valid.
    let anisotropy = match key.filter {
        TextureFilter::Linear => key.anisotropy.clamp(1, 16),
        TextureFilter::Nearest => 1,
    };
    device.create_sampler(&crate::gpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wrap(key.wrap_u),
        address_mode_v: wrap(key.wrap_v),
        address_mode_w: wrap(key.wrap_u),
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: dmipmap(filter),
        anisotropy_clamp: anisotropy,
        // `key.lod_bias` is intentionally not applied here: wgpu samplers carry
        // no LOD bias (it is a shader-side `textureSampleBias`), so the field is
        // reserved until the lit shaders take a bias. Wrap/filter/aniso are the
        // live parts.
        ..Default::default()
    })
}

/// Wrap a mip filter for the current wgpu version's `SamplerDescriptor`. 27
/// reuses `FilterMode` for the mip filter; 28 split it into `MipmapFilterMode`,
/// which 29 and 30 keep.
#[cfg(wgpu27)]
pub fn dmipmap(filter: crate::gpu::FilterMode) -> crate::gpu::FilterMode {
    filter
}
/// See the wgpu 27 sibling: 29 and 30 take a distinct `MipmapFilterMode`.
#[cfg(any(wgpu29, wgpu30))]
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
pub const ADDITIVE_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
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
pub const PREMULTIPLIED_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
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
pub struct DualPipelineDesc<'a> {
    /// Debug label; the LDR and HDR variants are suffixed from it.
    pub label: &'a str,
    /// Pipeline layout, listing the shared group 0 first.
    pub layout: &'a crate::gpu::PipelineLayout,
    /// Shader module holding both entry points.
    pub shader: &'a crate::gpu::ShaderModule,
    /// Vertex entry point name.
    pub vertex_entry: &'a str,
    /// Fragment entry point name.
    pub fragment_entry: &'a str,
    /// Vertex buffer layouts, in bind-slot order.
    pub vertex_buffers: &'a [crate::gpu::VertexBufferLayout<'a>],
    /// Colour blend state. `None` writes opaque.
    pub blend: Option<crate::gpu::BlendState>,
    /// Primitive topology.
    pub topology: crate::gpu::PrimitiveTopology,
    /// Face to cull. `None` draws both sides.
    pub cull_mode: Option<crate::gpu::Face>,
    /// Whether the pipeline writes depth.
    pub depth_write: bool,
    /// Depth comparison function.
    pub depth_compare: crate::gpu::CompareFunction,
    /// MSAA sample count; must match the target the pipeline draws into.
    pub sample_count: u32,
    /// LDR swapchain format; the HDR variant is always `Rgba16Float`.
    pub ldr_format: crate::gpu::TextureFormat,
}

/// Build the LDR + HDR pair of a scene render pipeline from the parts that vary
/// ([`DualPipelineDesc`]), holding the shared depth / stencil / target-write
/// state constant. The two variants differ only in colour target format
/// (`desc.ldr_format` vs `Rgba16Float`), which is the invariant `DualPipeline`
/// encodes.
pub fn build_dual_pipeline(
    device: &crate::gpu::Device,
    desc: &DualPipelineDesc,
) -> crate::resources::types::DualPipeline {
    let make = |format: crate::gpu::TextureFormat| {
        render_pipeline(
            device,
            RenderPipelineDesc {
                label: desc.label,
                layout: desc.layout,
                vertex_module: desc.shader,
                vertex_entry: desc.vertex_entry,
                vertex_buffers: desc.vertex_buffers,
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
pub fn build_fullscreen_pipeline(
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
            vertex_module: shader,
            vertex_entry: "vs_main",
            vertex_buffers: &[],
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
pub fn build_outline_mask_pipeline(
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
            vertex_module: shader,
            vertex_entry: "vs_main",
            vertex_buffers: vertex_buffers,
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

/// Build a surface-mask pipeline: the item's geometry drawn into the scene
/// stencil, depth-tested against the scene so only its visible pixels are
/// stamped with the pass's stencil reference. Used from
/// [`ItemTypePlugin::surface_mask`](crate::plugin_api::ItemTypePlugin::surface_mask).
///
/// The vertex layout and cull mode vary per item and are passed in; the rest is
/// fixed: triangle list, `Depth24PlusStencil8`, no colour target, no depth
/// write, single sample, both stages `vs_main` / `fs_main`. The fragment stage
/// has nothing to output: leave it empty, or have it `discard` where the item's
/// own fragment stage would, so a cut-out stamps only what it drew.
///
/// The depth test is `LessEqual` with a small bias towards the camera. This is
/// a second draw of geometry the opaque pass already drew, and without the
/// bias rounding differences between the two passes leave holes in the stamp.
pub fn build_surface_mask_pipeline(
    device: &crate::gpu::Device,
    label: &str,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    vertex_buffers: &[crate::gpu::VertexBufferLayout],
    cull: Option<crate::gpu::Face>,
) -> crate::gpu::RenderPipeline {
    render_pipeline(
        device,
        RenderPipelineDesc {
            label,
            layout,
            vertex_module: shader,
            vertex_entry: "vs_main",
            vertex_buffers,
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: cull,
                ..Default::default()
            },
            depth_stencil: Some(surface_mask_depth_stencil()),
            multisample: crate::gpu::MultisampleState::default(),
            cache: None,
        },
    )
}

/// The depth-stencil state of a surface-mask pipeline: test against the scene
/// depth without writing it, and replace the stencil with the pass's reference
/// where the test passes.
pub(crate) fn surface_mask_depth_stencil() -> crate::gpu::DepthStencilState {
    let stamp = crate::gpu::StencilFaceState {
        compare: crate::gpu::CompareFunction::Always,
        fail_op: crate::gpu::StencilOperation::Keep,
        depth_fail_op: crate::gpu::StencilOperation::Keep,
        pass_op: crate::gpu::StencilOperation::Replace,
    };
    crate::gpu::DepthStencilState {
        format: crate::gpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: dwrite(false),
        depth_compare: dcompare(crate::gpu::CompareFunction::LessEqual),
        stencil: crate::gpu::StencilState {
            front: stamp,
            back: stamp,
            read_mask: 0xff,
            write_mask: 0xff,
        },
        bias: crate::gpu::DepthBiasState {
            constant: -2,
            slope_scale: 0.0,
            clamp: 0.0,
        },
    }
}

/// Create a compute pipeline. Every compute pipeline in the crate has the same
/// shape: a shader, its layout, and an entry point, with default compilation
/// options, created against the device's pipeline cache when it has one (see
/// [`device_pipeline_cache`]). This is the one place the crate calls
/// `create_compute_pipeline`, so a wgpu upgrade only has to be audited here.
pub fn compute_pipeline(
    device: &crate::gpu::Device,
    label: &str,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    entry: &str,
) -> crate::gpu::ComputePipeline {
    let build_start = web_time::Instant::now();
    let cache = device_pipeline_cache::get(device);
    let pipeline = device.create_compute_pipeline(&crate::gpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
        cache: cache.as_ref(),
    });
    if build_log::enabled() {
        build_log::record(
            &format!("compute {label}"),
            build_start.elapsed().as_secs_f32() * 1000.0,
        );
    }
    pipeline
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
    // push constants with `immediate_size` and takes `&[Option<&BindGroupLayout>]`,
    // which 30 keeps.
    #[cfg(wgpu27)]
    let layout = device.create_pipeline_layout(&crate::gpu::PipelineLayoutDescriptor {
        label: label.into(),
        bind_group_layouts,
        push_constant_ranges: &[],
    });
    #[cfg(any(wgpu29, wgpu30))]
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
pub fn standard_scene_layout(
    device: &crate::gpu::Device,
    label: &str,
    camera_bgl: &crate::gpu::BindGroupLayout,
    per_item_bgl: &crate::gpu::BindGroupLayout,
) -> crate::gpu::PipelineLayout {
    pipeline_layout(device, label, &[camera_bgl, per_item_bgl])
}

/// The parts of a render pipeline that vary between call sites. The vertex
/// stage is given as its inputs (module, entry point, buffer layouts) rather
/// than a built `VertexState`, so [`render_pipeline`] can construct the
/// `VertexState` itself: that keeps the `buffers` field, whose shape churns
/// across wgpu versions, behind the one function. The `multiview` and `cache`
/// fields are likewise filled by [`render_pipeline`].
pub struct RenderPipelineDesc<'a> {
    /// Debug label for the pipeline.
    pub label: &'a str,
    /// Pipeline layout (bind group layouts). See [`pipeline_layout`].
    pub layout: &'a crate::gpu::PipelineLayout,
    /// Vertex shader module.
    pub vertex_module: &'a crate::gpu::ShaderModule,
    /// Vertex shader entry point.
    pub vertex_entry: &'a str,
    /// Vertex buffer layouts (empty when the shader generates its vertices).
    pub vertex_buffers: &'a [crate::gpu::VertexBufferLayout<'a>],
    /// Fragment stage and its color targets, or `None` for a depth-only pass.
    pub fragment: Option<crate::gpu::FragmentState<'a>>,
    /// Primitive topology, cull mode, and front face.
    pub primitive: crate::gpu::PrimitiveState,
    /// Depth-stencil state, or `None` for a pass without a depth attachment.
    /// Build with [`depth_stencil`] or [`scene_depth_stencil`].
    pub depth_stencil: Option<crate::gpu::DepthStencilState>,
    /// Multisample (MSAA) state.
    pub multisample: crate::gpu::MultisampleState,
    /// Pipeline cache to create against. `None` uses the cache the renderer
    /// registered for the device, when the device has one, so a site does not
    /// have to name it.
    pub cache: Option<&'a crate::gpu::PipelineCache>,
}

/// Create a render pipeline from the parts that vary ([`RenderPipelineDesc`]),
/// building the `VertexState` and filling `multiview: None`. This is the one
/// place the crate calls `create_render_pipeline` and the one place a
/// `VertexState` is constructed, so the `buffers` and `multiview` fields that
/// change shape across wgpu versions only have to be audited here.
pub fn render_pipeline(
    device: &crate::gpu::Device,
    desc: RenderPipelineDesc,
) -> crate::gpu::RenderPipeline {
    // wgpu 30 changed `VertexState::buffers` to `&[Option<VertexBufferLayout>]`;
    // 27 and 29 take `&[VertexBufferLayout]`. The wrapped Vec is a local here so
    // it outlives the `VertexState` it is borrowed into.
    #[cfg(wgpu30)]
    let vbufs: Vec<Option<crate::gpu::VertexBufferLayout>> =
        desc.vertex_buffers.iter().cloned().map(Some).collect();
    let vertex = crate::gpu::VertexState {
        module: desc.vertex_module,
        entry_point: Some(desc.vertex_entry),
        #[cfg(not(wgpu30))]
        buffers: desc.vertex_buffers,
        #[cfg(wgpu30)]
        buffers: &vbufs,
        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
    };
    let build_start = web_time::Instant::now();
    let build_label = desc.label;
    let device_cache = match desc.cache {
        Some(_) => None,
        None => device_pipeline_cache::get(device),
    };
    let cache = desc.cache.or(device_cache.as_ref());
    let pipeline = device.create_render_pipeline(&crate::gpu::RenderPipelineDescriptor {
        label: Some(desc.label),
        layout: Some(desc.layout),
        vertex,
        fragment: desc.fragment,
        primitive: desc.primitive,
        depth_stencil: desc.depth_stencil,
        multisample: desc.multisample,
        // 29 renamed `multiview` to the `multiview_mask` bitmask form, which 30 keeps.
        #[cfg(wgpu27)]
        multiview: None,
        #[cfg(any(wgpu29, wgpu30))]
        multiview_mask: None,
        cache,
    });
    build_log::record(build_label, build_start.elapsed().as_secs_f32() * 1000.0);
    pipeline
}

/// The pipeline cache each device's pipelines are created against.
///
/// wgpu takes the cache per pipeline descriptor, and pipelines are created all
/// over the crate and by item-type plugins. Naming the cache at every site is
/// how sites get missed, so the renderer registers one cache per device here
/// and [`render_pipeline`] and [`compute_pipeline`] look it up. Every renderer
/// on the same device shares the one cache, so the data any of them returns
/// covers them all.
///
/// Only a device with `Features::PIPELINE_CACHE` has an entry; on any other the
/// lookup finds nothing and pipelines are created uncached.
pub(crate) mod device_pipeline_cache {
    use std::sync::Mutex;

    struct Entry {
        device: crate::gpu::Device,
        cache: crate::gpu::PipelineCache,
        users: usize,
    }

    static CACHES: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

    /// The cache for `device`, made with `create` if this is its first user.
    /// Pair with [`release`].
    pub(crate) fn acquire(
        device: &crate::gpu::Device,
        create: impl FnOnce() -> crate::gpu::PipelineCache,
    ) -> crate::gpu::PipelineCache {
        let mut caches = CACHES.lock().unwrap();
        if let Some(entry) = caches.iter_mut().find(|e| e.device == *device) {
            entry.users += 1;
            return entry.cache.clone();
        }
        let cache = create();
        caches.push(Entry {
            device: device.clone(),
            cache: cache.clone(),
            users: 1,
        });
        cache
    }

    /// Drop one user of `device`'s cache, and the entry with the last.
    pub(crate) fn release(device: &crate::gpu::Device) {
        let mut caches = CACHES.lock().unwrap();
        if let Some(i) = caches.iter().position(|e| e.device == *device) {
            caches[i].users -= 1;
            if caches[i].users == 0 {
                caches.swap_remove(i);
            }
        }
    }

    /// The cache registered for `device`, if any.
    pub(crate) fn get(device: &crate::gpu::Device) -> Option<crate::gpu::PipelineCache> {
        let caches = CACHES.lock().unwrap();
        caches
            .iter()
            .find(|e| e.device == *device)
            .map(|e| e.cache.clone())
    }

    /// Holds one user's claim on a device's cache and gives it up when dropped.
    pub(crate) struct Lease(pub(crate) crate::gpu::Device);

    impl Drop for Lease {
        fn drop(&mut self) {
            release(&self.0);
        }
    }
}

#[cfg(test)]
mod device_pipeline_cache_tests {
    use super::device_pipeline_cache;

    fn device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
            required_features: crate::ViewportRenderer::recommended_device_features(&adapter),
            required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }))
        .ok()
    }

    /// A device with a pipeline cache registers it for as long as a renderer
    /// on the device is alive, two renderers share the one cache, and the
    /// lazily built pipelines land in it. A device without the feature
    /// registers nothing.
    #[test]
    fn a_device_has_one_cache_for_as_long_as_a_renderer_uses_it() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = crate::gpu::TextureFormat::Rgba8UnormSrgb;
        let has_cache = device
            .features()
            .contains(crate::gpu::Features::PIPELINE_CACHE);
        assert!(device_pipeline_cache::get(&device).is_none());

        let mut first = crate::DeviceResources::new(&device, format, 1);
        let second = crate::DeviceResources::new(&device, format, 1);
        assert_eq!(first.pipeline_cache.is_some(), has_cache);
        assert_eq!(device_pipeline_cache::get(&device).is_some(), has_cache);
        if !has_cache {
            eprintln!("this backend has no pipeline cache; checked that nothing is registered");
            return;
        }
        assert!(
            first.pipeline_cache == second.pipeline_cache,
            "two renderers on one device share its cache"
        );

        // The post chain is built lazily and names no cache at its call sites.
        let before = first.pipeline_cache.as_ref().unwrap().get_data();
        first.ensure_hdr_pipelines(&device, &queue, format);
        let after = first.pipeline_cache.as_ref().unwrap().get_data();
        assert!(
            after.map_or(0, |d| d.len()) > before.map_or(0, |d| d.len()),
            "the lazily built pipelines were not added to the cache"
        );

        drop(first);
        assert!(device_pipeline_cache::get(&device).is_some());
        drop(second);
        assert!(device_pipeline_cache::get(&device).is_none());
    }
}

/// `create_buffer` and `create_texture` with the allocation reported to
/// [`build_log`]. For what the renderer allocates for itself, so a startup
/// breakdown can list it.
pub(crate) trait LoggedAlloc {
    fn logged_buffer(&self, desc: &crate::gpu::BufferDescriptor) -> crate::gpu::Buffer;
    fn logged_buffer_init(
        &self,
        desc: &crate::gpu::util::BufferInitDescriptor,
    ) -> crate::gpu::Buffer;
    fn logged_texture(&self, desc: &crate::gpu::TextureDescriptor) -> crate::gpu::Texture;
}

impl LoggedAlloc for crate::gpu::Device {
    fn logged_buffer(&self, desc: &crate::gpu::BufferDescriptor) -> crate::gpu::Buffer {
        let buffer = self.create_buffer(desc);
        if build_log::enabled() {
            build_log::record_allocation(desc.label.unwrap_or("buffer"), desc.size);
        }
        buffer
    }

    fn logged_buffer_init(
        &self,
        desc: &crate::gpu::util::BufferInitDescriptor,
    ) -> crate::gpu::Buffer {
        use crate::gpu::util::DeviceExt;
        let buffer = self.create_buffer_init(desc);
        if build_log::enabled() {
            build_log::record_allocation(desc.label.unwrap_or("buffer"), buffer.size());
        }
        buffer
    }

    fn logged_texture(&self, desc: &crate::gpu::TextureDescriptor) -> crate::gpu::Texture {
        let texture = self.create_texture(desc);
        if build_log::enabled() {
            build_log::record_allocation(desc.label.unwrap_or("texture"), texture_bytes(desc));
        }
        texture
    }
}

/// Bytes a texture descriptor asks for, summed over its mip chain. Depth and
/// stencil formats have no copy size and are counted at 4 bytes per texel.
pub(crate) fn texture_bytes(desc: &crate::gpu::TextureDescriptor) -> u64 {
    let texel = desc.format.block_copy_size(None).unwrap_or(4) as u64;
    let (bw, bh) = desc.format.block_dimensions();
    let layers = match desc.dimension {
        crate::gpu::TextureDimension::D3 => 1,
        _ => desc.size.depth_or_array_layers as u64,
    };
    let mut total = 0u64;
    for mip in 0..desc.mip_level_count {
        let w = (desc.size.width >> mip).max(1) as u64;
        let h = (desc.size.height >> mip).max(1) as u64;
        let d = match desc.dimension {
            crate::gpu::TextureDimension::D3 => {
                (desc.size.depth_or_array_layers >> mip).max(1) as u64
            }
            _ => 1,
        };
        total += w.div_ceil(bw as u64) * h.div_ceil(bh as u64) * d * texel;
    }
    total * layers * desc.sample_count as u64
}

/// Optional record of what each pipeline, shader module, render target, and
/// other GPU allocation cost to create.
///
/// Off by default. Switch it on with [`enable`](build_log::enable), or by setting
/// `VPL_BUILD_LOG` in the environment on a platform that has one. Every
/// [`render_pipeline`], [`compute_pipeline`], and [`wgsl_module`] call then
/// appends its label and wall-clock cost, every per-viewport render target
/// appends its label and size, and so does every buffer and texture the renderer
/// allocates for itself (uniform and storage buffers, the geometry slab, the
/// shadow maps, the glyph atlas). Mesh and texture uploads the application asks
/// for are not recorded here; [`resident_bytes`] accounts for those.
///
/// [`resident_bytes`]: crate::ViewportRenderer::resident_bytes
///
/// Startup on a GPU backend is dominated by shader compilation, and a phase
/// breakdown cannot say which pipeline is expensive. This attributes it per
/// object. Read it back with [`drain`](build_log::drain),
/// [`drain_textures`](build_log::drain_textures) and
/// [`drain_allocations`](build_log::drain_allocations), which each return what
/// was recorded since the last call.
pub mod build_log {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, OnceLock};

    /// Seeded from the environment on first use, then settable at runtime.
    ///
    /// A web build has no environment: under `wasm32` `std::env::var` always
    /// reports the variable missing, so `VPL_BUILD_LOG` can never be set there
    /// and [`enable`] is the only way in. That is the platform where startup
    /// attribution is most wanted, so the flag is not env-only.
    static ENABLED: OnceLock<AtomicBool> = OnceLock::new();

    fn flag() -> &'static AtomicBool {
        ENABLED.get_or_init(|| AtomicBool::new(std::env::var("VPL_BUILD_LOG").is_ok()))
    }

    /// Start recording. Call before building the renderer; anything created
    /// earlier is not in the log.
    pub fn enable() {
        flag().store(true, Ordering::Relaxed);
    }

    /// Stop recording. What is already recorded stays until drained.
    pub fn disable() {
        flag().store(false, Ordering::Relaxed);
    }

    /// Whether recording is on.
    pub fn enabled() -> bool {
        flag().load(Ordering::Relaxed)
    }

    static PIPELINES: OnceLock<Mutex<Vec<(String, f32)>>> = OnceLock::new();
    static TEXTURES: OnceLock<Mutex<Vec<(String, u64)>>> = OnceLock::new();
    static ALLOCATIONS: OnceLock<Mutex<Vec<(String, u64)>>> = OnceLock::new();

    pub(super) fn record(label: &str, ms: f32) {
        if enabled() {
            PIPELINES
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push((label.to_string(), ms));
        }
    }

    /// Record a render target allocation and its size. The other half of what a
    /// consumer pays before drawing anything: per-viewport target memory.
    pub(crate) fn record_texture(label: &str, bytes: u64) {
        if enabled() {
            TEXTURES
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push((label.to_string(), bytes));
        }
    }

    /// Record a buffer or texture the renderer allocated for itself, other than
    /// a per-viewport render target.
    pub(crate) fn record_allocation(label: &str, bytes: u64) {
        if enabled() {
            ALLOCATIONS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push((label.to_string(), bytes));
        }
    }

    /// Take every pipeline and shader module recorded since the last call, in
    /// creation order, with each one's wall-clock cost in milliseconds.
    pub fn drain() -> Vec<(String, f32)> {
        match PIPELINES.get() {
            Some(l) => std::mem::take(&mut *l.lock().unwrap()),
            None => Vec::new(),
        }
    }

    /// Take every render target recorded since the last call, in allocation
    /// order, with each one's size in bytes.
    pub fn drain_textures() -> Vec<(String, u64)> {
        match TEXTURES.get() {
            Some(l) => std::mem::take(&mut *l.lock().unwrap()),
            None => Vec::new(),
        }
    }

    /// Take every buffer and non-target texture recorded since the last call,
    /// in allocation order, with each one's size in bytes.
    pub fn drain_allocations() -> Vec<(String, u64)> {
        match ALLOCATIONS.get() {
            Some(l) => std::mem::take(&mut *l.lock().unwrap()),
            None => Vec::new(),
        }
    }

    #[cfg(test)]
    mod tests {
        /// `enable` works with no environment variable set, which is the only
        /// route a web build has. Records after enabling, nothing before.
        #[test]
        fn enable_switches_recording_on_at_runtime() {
            // Not asserting the initial state: the env seeds it, and another
            // test in this binary may have enabled it already.
            super::disable();
            let _ = super::drain();
            super::record("before", 1.0);
            assert!(super::drain().is_empty(), "a disabled log records nothing");

            super::enable();
            assert!(super::enabled());
            super::record("after", 2.0);
            let got = super::drain();
            assert_eq!(got.len(), 1, "an enabled log records");
            assert_eq!(got[0].0, "after");
            assert!(super::drain().is_empty(), "drain takes what it returned");
            super::disable();
        }
    }
}

/// Wrap a depth-write flag for the current wgpu version's `DepthStencilState`.
/// 27 takes a bare `bool`; 29 and 30 take `Option<bool>`.
#[cfg(wgpu27)]
pub fn dwrite(enabled: bool) -> bool {
    enabled
}
/// See the wgpu 27 sibling: 29 and 30 take `Option<bool>`.
#[cfg(any(wgpu29, wgpu30))]
pub fn dwrite(enabled: bool) -> Option<bool> {
    Some(enabled)
}

/// Wrap a depth-compare function for the current wgpu version's
/// `DepthStencilState`. 27 takes a bare `CompareFunction`; 29 and 30 take
/// `Option<CompareFunction>`.
#[cfg(wgpu27)]
pub fn dcompare(compare: crate::gpu::CompareFunction) -> crate::gpu::CompareFunction {
    compare
}
/// See the wgpu 27 sibling: 29 and 30 take `Option<CompareFunction>`.
#[cfg(any(wgpu29, wgpu30))]
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
    // 27's mapped view derefs to `[u8]` and is indexed directly; 29 and 30's
    // `BufferViewMut` is write-only and exposes a `slice(..)` -> `WriteOnly`. 30
    // additionally returns the view as a `Result`.
    #[cfg(wgpu27)]
    slice.get_mapped_range_mut()[..bytes.len()].copy_from_slice(bytes);
    #[cfg(wgpu29)]
    slice
        .get_mapped_range_mut()
        .slice(..bytes.len())
        .copy_from_slice(bytes);
    #[cfg(wgpu30)]
    slice
        .get_mapped_range_mut()
        .expect("buffer slice was not mapped for writing")
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
pub fn clamp_linear_mip_sampler(device: &crate::gpu::Device, label: &str) -> crate::gpu::Sampler {
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
    // WebGPU resolves `pop_error_scope` through a JavaScript promise, which only
    // settles when the browser event loop turns. A synchronous spin never lets
    // it turn, so waiting here would hang the tab rather than return an error.
    // Run the work and report nothing captured; a validation failure still
    // reaches the browser console through WebGPU's own uncaptured-error event.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device;
        return (f(), None);
    }

    // 27 pops the scope through a `Device::pop_error_scope` future; 29 and 30's
    // `push_error_scope` returns a guard whose `pop()` is the future.
    #[cfg(all(wgpu27, not(target_arch = "wasm32")))]
    {
        device.push_error_scope(crate::gpu::ErrorFilter::Validation);
        let value = f();
        let captured = block_on_simple(device.pop_error_scope());
        (value, captured)
    }
    #[cfg(all(any(wgpu29, wgpu30), not(target_arch = "wasm32")))]
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
#[cfg(not(target_arch = "wasm32"))]
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
                src.contains("dbg_vals["),
                "{name}: baked source lost the debug block; markers moved?"
            );
            let stripped = super::strip_debug_vis(src, false);
            assert!(
                !stripped.contains("dbg_vals["),
                "{name}: stripped module still carries the debug block"
            );
            assert!(
                !stripped.contains("BEGIN_DEBUG_VIS"),
                "{name}: stripped module kept the marker"
            );
            let kept = super::strip_debug_vis(src, true);
            assert!(
                kept.contains("dbg_vals["),
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

/// Vertex buffer layout of the meshes in the shared arena.
///
/// A plugin that draws a [`MeshId`](crate::resources::MeshId) through
/// [`MeshGeometry`](crate::resources::MeshGeometry) binds those buffers
/// directly, so its pipeline has to declare the layout they were uploaded
/// with. Locations 0 to 4 are position, normal, colour, uv and tangent.
pub fn mesh_vertex_layout() -> crate::gpu::VertexBufferLayout<'static> {
    use crate::resources::types::VertexBufferLayoutExt as _;
    crate::resources::types::Vertex::buffer_layout()
}
