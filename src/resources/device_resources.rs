//! `DeviceResources`: the device-shared GPU resource container, its content
//! and per-viewport scope structs, and the small feature-resource clusters.

use crate::resources::types::*;

/// Per-viewport HDR/post-process GPU state.
///
/// Holds all viewport-size-dependent render targets, their associated bind
/// groups, and the per-viewport uniform buffers used by the post-process
/// pipeline.  Created lazily in `ViewportRenderer::ensure_viewport_hdr` and
/// resized automatically when the viewport dimensions change.
///
/// Shared infrastructure (pipelines, BGLs, samplers, placeholder textures,
/// SSAO noise/kernel) lives on [`DeviceResources`] and is created once
/// by `ensure_hdr_shared`.
#[allow(dead_code)]
pub(crate) struct ViewportHdrState {
    // --- HDR scene target ---
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub hdr_depth_texture: wgpu::Texture,
    pub hdr_depth_view: wgpu::TextureView,
    pub hdr_depth_only_view: wgpu::TextureView,
    pub hdr_stencil_only_view: wgpu::TextureView,

    // --- Bloom ---
    pub bloom_threshold_texture: wgpu::Texture,
    pub bloom_threshold_view: wgpu::TextureView,
    pub bloom_ping_texture: wgpu::Texture,
    pub bloom_ping_view: wgpu::TextureView,
    pub bloom_pong_texture: wgpu::Texture,
    pub bloom_pong_view: wgpu::TextureView,

    // --- SSAO ---
    pub ssao_texture: wgpu::Texture,
    pub ssao_view: wgpu::TextureView,
    pub ssao_blur_texture: wgpu::Texture,
    pub ssao_blur_view: wgpu::TextureView,

    // --- Depth of field ---
    pub dof_texture: wgpu::Texture,
    pub dof_view: wgpu::TextureView,
    pub dof_bind_group: wgpu::BindGroup,
    pub dof_uniform_buf: wgpu::Buffer,

    // --- Contact shadow ---
    pub contact_shadow_texture: wgpu::Texture,
    pub contact_shadow_view: wgpu::TextureView,

    // --- Surface LIC ---
    /// Encodes screen-space flow vector per surface pixel (Rgba8Unorm, viewport-sized).
    pub lic_vector_texture: wgpu::Texture,
    pub lic_vector_view: wgpu::TextureView,
    /// LIC intensity after advection (R8Unorm, viewport-sized). Read by tone_map.wgsl binding 7.
    pub lic_output_texture: wgpu::Texture,
    pub lic_output_view: wgpu::TextureView,
    /// Per-pixel white noise (R8Unorm, viewport-sized). One independent random value per pixel.
    /// Sampled with textureLoad (nearest) in lic_advect.wgsl to produce directional LIC contrast.
    pub lic_noise_texture: wgpu::Texture,
    pub lic_noise_view: wgpu::TextureView,
    /// Bind group for the LIC advect render pass (reads lic_vector_texture + lic_noise_texture).
    pub lic_advect_bind_group: wgpu::BindGroup,
    /// Uniform buffer for LicAdvectUniform (steps, step_size, viewport dims).
    pub lic_uniform_buf: wgpu::Buffer,

    // --- FXAA ---
    pub fxaa_texture: wgpu::Texture,
    pub fxaa_view: wgpu::TextureView,

    // --- SSAA (allocated when ssaa_factor > 1) ---
    /// Supersampled colour render target. `None` when ssaa_factor == 1.
    pub ssaa_colour_texture: Option<wgpu::Texture>,
    pub ssaa_colour_view: Option<wgpu::TextureView>,
    /// Supersampled depth render target. `None` when ssaa_factor == 1.
    pub ssaa_depth_texture: Option<wgpu::Texture>,
    pub ssaa_depth_view: Option<wgpu::TextureView>,
    /// Depth-aspect-only view of `ssaa_depth_texture`, used as the soft-particle
    /// sample source during the SSAA sprite post-pass. `None` when SSAA is off.
    pub ssaa_depth_only_view: Option<wgpu::TextureView>,
    /// Bind group for the SSAA resolve pass (reads ssaa_colour_texture). `None` when ssaa_factor == 1.
    pub ssaa_resolve_bind_group: Option<wgpu::BindGroup>,
    /// Uniform buffer holding the ssaa_factor value for the resolve shader.
    pub ssaa_uniform_buf: Option<wgpu::Buffer>,
    /// The ssaa_factor this state was created with (1 = no SSAA).
    pub ssaa_factor: u32,

    // --- OIT (lazily allocated when transparent geometry is present) ---
    pub oit_accum_texture: Option<wgpu::Texture>,
    pub oit_accum_view: Option<wgpu::TextureView>,
    pub oit_reveal_texture: Option<wgpu::Texture>,
    pub oit_reveal_view: Option<wgpu::TextureView>,
    pub oit_composite_bind_group: Option<wgpu::BindGroup>,
    pub oit_size: [u32; 2],

    // --- Outline offscreen (used by the outline prepare pass) ---
    /// R8Unorm mask: selected objects rendered as white on black.
    pub outline_mask_texture: wgpu::Texture,
    pub outline_mask_view: wgpu::TextureView,
    /// RGBA output of the edge-detection pass (composited onto the main target).
    pub outline_colour_texture: wgpu::Texture,
    pub outline_colour_view: wgpu::TextureView,
    pub outline_depth_texture: wgpu::Texture,
    pub outline_depth_view: wgpu::TextureView,
    /// Depth-aspect view of `outline_depth_texture` for sampling (the HiZ
    /// occlusion prev-depth copy on the LDR path).
    pub outline_depth_only_view: wgpu::TextureView,
    /// Bind group for the edge-detection pass (reads mask, writes to colour).
    pub outline_edge_bind_group: wgpu::BindGroup,
    /// Uniform buffer for the edge-detection pass parameters.
    pub outline_edge_uniform_buf: wgpu::Buffer,
    pub outline_composite_bind_group: wgpu::BindGroup,

    // --- Bind groups (rebuilt when viewport dimensions change) ---
    pub tone_map_bind_group: wgpu::BindGroup,
    pub bloom_threshold_bg: wgpu::BindGroup,
    /// H-blur bind group that reads from bloom_threshold (pass 0 only).
    pub bloom_blur_h_bg: wgpu::BindGroup,
    /// V-blur bind group that reads from bloom_ping.
    pub bloom_blur_v_bg: wgpu::BindGroup,
    /// H-blur bind group that reads from bloom_pong (passes 1+).
    pub bloom_blur_h_pong_bg: wgpu::BindGroup,
    pub ssao_bg: wgpu::BindGroup,
    pub ssao_blur_bg: wgpu::BindGroup,
    pub dof_bg: wgpu::BindGroup,
    pub contact_shadow_bg: wgpu::BindGroup,
    pub fxaa_bind_group: wgpu::BindGroup,

    // --- Per-viewport uniform buffers ---
    pub tone_map_uniform_buf: wgpu::Buffer,
    pub bloom_uniform_buf: wgpu::Buffer,
    /// Constant H-blur uniform buffer (horizontal=1, written once at creation).
    pub bloom_h_uniform_buf: wgpu::Buffer,
    /// Constant V-blur uniform buffer (horizontal=0, written once at creation).
    pub bloom_v_uniform_buf: wgpu::Buffer,
    pub ssao_uniform_buf: wgpu::Buffer,
    pub contact_shadow_uniform_buf: wgpu::Buffer,

    // --- Post-tone-map depth buffer (native resolution) ---
    // When scene_size == output_size (render_scale = 1.0) this is None and
    // hdr_depth_view is used directly for post-tone-map passes.
    // When scene_size != output_size the scene depth is blitted into this
    // native-resolution texture so that post-tone-map passes (grid, gizmos,
    // axes, etc.) can use it as a depth attachment alongside output_view.
    pub output_depth_texture: Option<wgpu::Texture>,
    pub output_depth_view: wgpu::TextureView,
    /// Bind group for the depth blit pass (reads hdr_depth_only_view).
    /// None when scene_size == output_size (no blit needed).
    pub depth_blit_bind_group: Option<wgpu::BindGroup>,

    // --- HDR upscale (allocated when scene_size != output_size) ---
    // When render_scale < 1.0, tone-map and FXAA run at scene resolution.
    // The result is written to upscale_texture, then upscale-blitted to output_view.
    pub upscale_texture: Option<wgpu::Texture>,
    pub upscale_view: Option<wgpu::TextureView>,
    pub upscale_bind_group: Option<wgpu::BindGroup>,

    /// Native output resolution [width, height].
    pub output_size: [u32; 2],
    /// Effective scene resolution after render scale: [output_size * render_scale].
    /// Equals output_size when render_scale = 1.0.
    pub scene_size: [u32; 2],

    // --- Decal pass depth binding (D1) ---
    /// Bind group for group 1 of the decal pass: reads hdr_depth_only_view as a depth texture.
    /// Rebuilt on viewport resize alongside the other viewport-sized bind groups.
    pub decal_depth_bg: wgpu::BindGroup,
}
/// Per-viewport scatter-pass intermediates: two RGBA16F ping-pong targets
/// driven by the temporal-accumulation logic, plus the composite bind groups
/// and previous-frame view-projection used for reprojection.
///
/// Lives on `ViewportRenderer` (not `ViewportHdrState`) so that the scatter
/// pass can allocate and mutate it without conflicting with the immutable
/// `slot_hdr` borrow held across the larger paint phase.
pub(crate) struct ScatterViewportState {
    // Textures keep the GPU allocation alive; views are sampled or rendered
    // into.
    /// Per-volume scatter draws accumulate into this target each frame.
    /// Cleared at the start of the scatter pass.
    #[allow(dead_code)]
    pub raw_current_texture: wgpu::Texture,
    pub raw_current_view: wgpu::TextureView,
    /// History ping-pong. The temporal-resolve pass reads one slot
    /// (history_prev) and writes the other (history_new). `parity` selects.
    #[allow(dead_code)]
    pub history_a_texture: wgpu::Texture,
    pub history_a_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub history_b_texture: wgpu::Texture,
    pub history_b_view: wgpu::TextureView,
    /// Composite bind group reading the raw-current texture.
    /// Used when temporal accumulation is disabled.
    pub composite_bg_raw: wgpu::BindGroup,
    /// Composite bind groups reading either history slot, used as the source
    /// after the temporal-resolve pass has written history_new.
    pub composite_bg_history_a: wgpu::BindGroup,
    pub composite_bg_history_b: wgpu::BindGroup,
    /// Temporal-resolve bind groups, keyed by which history slot is being
    /// read as the previous-frame input. Each binds raw_current + the chosen
    /// history slot.
    pub temporal_resolve_bg_read_a: wgpu::BindGroup,
    pub temporal_resolve_bg_read_b: wgpu::BindGroup,
    /// Current allocated intermediate size, [width, height].
    pub size: [u32; 2],
    /// Whether `size` reflects the downsampled (half-res) allocation.
    pub downsampled: bool,
    /// Index of the history slot the next frame writes to (0 = A, 1 = B).
    /// The other slot is read as the previous-frame history.
    pub parity: u32,
    /// True when the history slot opposite `parity` holds a usable
    /// previous-frame composite result.
    pub history_valid: bool,
    /// Previous frame's view-projection (row-major mat4).
    pub prev_view_proj: [[f32; 4]; 4],
    /// Scene colour copy sampled by the refraction pass. Allocated on demand
    /// when at least one volume has refraction enabled. Matches the HDR
    /// target's size and format.
    #[allow(dead_code)]
    pub refraction_source_texture: Option<wgpu::Texture>,
    /// View paired with `refraction_source_texture`. Bound as the source
    /// during the refraction pass and as the render target during the
    /// preceding blit-copy of the HDR scene.
    pub refraction_source_view: Option<wgpu::TextureView>,
    /// Per-viewport bind group binding `(refraction_source_view, depth)` to
    /// the refraction pass.
    pub refraction_source_bg: Option<wgpu::BindGroup>,
    /// Per-viewport bind group binding the HDR view as the source for the
    /// blit-copy that fills `refraction_source_view`.
    pub refraction_blit_bg: Option<wgpu::BindGroup>,
    /// Allocated size of the refraction source, matched to the HDR target.
    pub refraction_source_size: [u32; 2],
}
/// A render pipeline compiled for both the LDR swapchain format and the HDR
/// intermediate format (`Rgba16Float`). Used for pipelines that draw into the
/// primary scene colour attachment, which may be either format depending on
/// whether post-processing is active.
pub(crate) struct DualPipeline {
    pub ldr: wgpu::RenderPipeline,
    pub hdr: wgpu::RenderPipeline,
}

impl DualPipeline {
    /// Select the pipeline matching the current render target format.
    /// Pass `true` when drawing into the HDR scene pass (`Rgba16Float`),
    /// `false` when drawing into the LDR swapchain pass.
    pub fn for_format(&self, hdr: bool) -> &wgpu::RenderPipeline {
        if hdr { &self.hdr } else { &self.ldr }
    }
}

/// GPU object-ID picking pipeline and its bind group layouts. Lazily built.
#[derive(Default)]
pub(crate) struct PickResources {
    /// Render pipeline that outputs flat u32 object IDs to R32Uint + R32Float targets.
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    /// Group 1 layout (PickInstance storage buffer).
    pub(crate) bind_group_layout_1: Option<wgpu::BindGroupLayout>,
    /// Minimal camera-only bind group layout (group 0).
    pub(crate) camera_bgl: Option<wgpu::BindGroupLayout>,
}

/// GPU implicit-surface ray-march pipeline and layout. Lazily built.
#[derive(Default)]
pub(crate) struct ImplicitResources {
    /// Render pipeline for GPU-side implicit surface ray-marching.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Group 1 layout (ImplicitUniformRaw).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Outline mask pipeline for implicit surfaces. None until first selected item.
    pub(crate) outline_mask_pipeline: Option<wgpu::RenderPipeline>,
}

/// Screen-space image quad pipelines (plain + depth-composite) and the rect
/// outline mask pipeline. Lazily built.
#[derive(Default)]
pub(crate) struct ScreenImageResources {
    /// Render pipeline for screen-space image quads.
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    /// Group 0 layout (uniform + texture + sampler).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Depth-composite pipeline (LessEqual depth, per-pixel image depth).
    pub(crate) dc_pipeline: Option<wgpu::RenderPipeline>,
    /// Group 0 layout for the dc pipeline (uniform + colour + sampler + depth).
    pub(crate) dc_bgl: Option<wgpu::BindGroupLayout>,
    /// Outline mask pipeline for screen-space rect images. None until first selected.
    pub(crate) rect_outline_mask_pipeline: Option<wgpu::RenderPipeline>,
    /// Layout for the rect outline mask pipeline (NdcRectUniform).
    pub(crate) rect_outline_bgl: Option<wgpu::BindGroupLayout>,
}

/// Sub-object highlight pipelines (fill / edge / sprite, HDR + LDR) and layout.
/// Lazily built the first frame a sub-selection is present.
#[derive(Default)]
pub(crate) struct SubHighlightResources {
    /// Translucent face fill pipeline (HDR).
    pub(crate) fill_pipeline: Option<wgpu::RenderPipeline>,
    /// Depth-nudged billboard edge-line pipeline (HDR).
    pub(crate) edge_pipeline: Option<wgpu::RenderPipeline>,
    /// Billboard sprite pipeline for vertex/point highlights (HDR).
    pub(crate) sprite_pipeline: Option<wgpu::RenderPipeline>,
    /// Translucent face fill pipeline (LDR).
    pub(crate) fill_ldr_pipeline: Option<wgpu::RenderPipeline>,
    /// Depth-nudged billboard edge-line pipeline (LDR).
    pub(crate) edge_ldr_pipeline: Option<wgpu::RenderPipeline>,
    /// Billboard sprite pipeline for vertex/point highlights (LDR).
    pub(crate) sprite_ldr_pipeline: Option<wgpu::RenderPipeline>,
    /// Shared group 1 layout (SubHighlightUniform).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
}

/// Projected-tetrahedra transparent volume pipeline, layouts, and LUT bind
/// group cache. Lazily built.
#[derive(Default)]
pub(crate) struct ProjectedTetResources {
    /// Render pipeline for the projected tetrahedra pass.
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    /// Group 1 layout (per-volume uniform + tet storage buffer).
    pub(crate) bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Group 2 layout (per-frame colourmap LUT + sampler).
    pub(crate) lut_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Cache of LUT bind groups keyed by colourmap slot index.
    pub(crate) lut_bind_groups: std::collections::HashMap<usize, wgpu::BindGroup>,
    /// LUT bind group for the fallback colourmap.
    pub(crate) fallback_lut_bind_group: Option<wgpu::BindGroup>,
}

/// Selection-outline and x-ray pipelines, the offscreen mask/composite targets,
/// and their layouts. The mask/edge/xray/splat pipelines are built eagerly at
/// init; the offscreen textures and composite pipelines are lazily created.
pub(crate) struct OutlineResources {
    /// Group 1 layout for OutlineUniform (mask/xray pipelines).
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    /// Mask-write pipeline: selected objects as r=1.0 to an R8 mask.
    pub(crate) mask_pipeline: wgpu::RenderPipeline,
    /// Two-sided mask-write pipeline (no face culling).
    pub(crate) mask_two_sided_pipeline: wgpu::RenderPipeline,
    /// Fullscreen edge-detection pipeline: reads mask, outputs the outline ring.
    pub(crate) edge_pipeline: wgpu::RenderPipeline,
    /// Layout for the edge-detection pass (mask texture + sampler + uniform).
    pub(crate) edge_bgl: wgpu::BindGroupLayout,
    /// X-ray pipeline: draws selected objects through occluders (depth Always).
    pub(crate) xray_pipeline: wgpu::RenderPipeline,
    /// Billboard disc pipeline for the Gaussian splat outline mask pass.
    pub(crate) splat_mask_pipeline: wgpu::RenderPipeline,
    /// Offscreen RGBA texture the outline stencil pass renders into.
    pub(crate) colour_texture: Option<wgpu::Texture>,
    pub(crate) colour_view: Option<wgpu::TextureView>,
    /// Depth+stencil texture for the offscreen outline pass.
    pub(crate) depth_texture: Option<wgpu::Texture>,
    pub(crate) depth_view: Option<wgpu::TextureView>,
    /// Size of the current outline offscreen textures.
    pub(crate) target_size: [u32; 2],
    /// Fullscreen composite pipelines: single-sample LDR, MSAA, HDR.
    pub(crate) composite_pipeline_single: Option<wgpu::RenderPipeline>,
    pub(crate) composite_pipeline_msaa: Option<wgpu::RenderPipeline>,
    pub(crate) composite_pipeline_hdr: Option<wgpu::RenderPipeline>,
    pub(crate) composite_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) composite_bind_group: Option<wgpu::BindGroup>,
    pub(crate) composite_sampler: Option<wgpu::Sampler>,
}

/// Image slice render pipeline and layout. Lazily built.
#[derive(Default)]
pub(crate) struct ImageSliceResources {
    /// Image slice render pipeline. None until first slice item is submitted.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Group 1 layout for image slice uniforms.
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
}

/// Former name of [`DeviceResources`]. Renamed to reflect that this holds the
/// device-shared resources, not per-viewport state. Kept as an alias so existing
/// code keeps compiling; prefer `DeviceResources` in new code.
#[deprecated(note = "renamed to DeviceResources")]
pub type ViewportGpuResources = DeviceResources;

/// Uploaded GPU assets and their handle registries: user textures, geometry /
/// scivis stores, colourmap and matcap tables, plus the fallback LUT and the
/// zero-fill attribute buffers bound when an optional attribute is absent.
///
/// This is the content-residency set: everything addressed by a handle
/// (`TextureId`, `PolylineId`, `ColourmapId`, ...) or provided as a default when
/// a handle is missing. It carries no pipelines; the upload / registry methods
/// stay on `DeviceResources` and reach these through `self.content`. The shared
/// material and LUT samplers and the fallback material textures stay on the core
/// because the lit pass samples them on every draw.
pub struct ContentResources {
    /// Cache of material bind groups keyed by (albedo_id, normal_map_id, ao_map_id).
    /// u64::MAX sentinel = use fallback texture for that slot.
    #[allow(dead_code)]
    pub(crate) material_bind_groups: std::collections::HashMap<(u64, u64, u64), wgpu::BindGroup>,
    /// User-uploaded textures, keyed by the `texture_id` in Material. Slotted
    /// with generational ids so a freed slot cannot alias a later upload.
    pub(crate) textures: crate::resources::material::texture_store::TextureStore,
    /// Pre-uploaded polyline storage; entries are referenced from per-frame
    /// `PolylineRefItem`s.
    pub(crate) polyline_store: super::PolylineStore,
    /// Pre-uploaded streamtube storage.
    pub(crate) streamtube_store: super::StreamtubeStore,
    /// Pre-uploaded tube storage.
    pub(crate) tube_store: super::TubeStore,
    /// Pre-uploaded ribbon storage.
    pub(crate) ribbon_store: super::RibbonStore,
    /// Pre-uploaded point cloud storage.
    pub(crate) point_cloud_store: super::PointCloudStore,
    /// Pre-uploaded glyph set storage.
    pub(crate) glyph_set_store: super::GlyphSetStore,
    /// Pre-uploaded tensor glyph set storage.
    pub(crate) tensor_glyph_set_store: super::TensorGlyphSetStore,
    /// Pre-uploaded sprite set storage.
    pub(crate) sprite_set_store: super::SpriteSetStore,
    /// Pre-uploaded sprite instance set storage.
    pub(crate) sprite_instance_set_store: super::SpriteInstanceSetStore,
    /// Slotted store of all uploaded Gaussian splat sets.
    pub(crate) gaussian_splat_store: GaussianSplatStore,
    /// Uploaded 3D volume textures. Index = VolumeId value.
    pub(crate) volume_textures:
        crate::resources::handle::Registry<(wgpu::Texture, wgpu::TextureView)>,
    /// Uploaded projected-tet meshes. Index = ProjectedTetId value.
    pub(crate) projected_tet_store: crate::resources::handle::Registry<GpuProjectedTetMesh>,
    /// Glyph atlas for overlay text rendering (labels, scalar bars, rulers).
    pub(crate) glyph_atlas: crate::resources::overlay::font::GlyphAtlas,
    /// Persistent textures uploaded via `upload_overlay_texture`.
    pub(crate) overlay_textures: crate::resources::handle::Registry<OverlayShapeTextureEntry>,
    /// Matcap textures (256x256 RGBA), indexed by `MatcapId::index`.
    pub(crate) matcap_textures: Vec<wgpu::Texture>,
    /// Texture views for each uploaded matcap.
    pub(crate) matcap_views: Vec<wgpu::TextureView>,
    /// Linear-clamp sampler shared by all matcap texture lookups.
    pub(crate) matcap_sampler: Option<wgpu::Sampler>,
    /// Fallback 1x1 white view bound to binding 7 when no matcap is active.
    pub(crate) fallback_matcap_view: Option<wgpu::TextureView>,
    /// Whether built-in matcaps have been uploaded to the GPU.
    pub(crate) matcaps_initialized: bool,
    /// `MatcapId` for each built-in preset, populated by `ensure_matcaps_initialized`.
    pub(crate) builtin_matcap_ids: Option<[MatcapId; 8]>,
    /// Uploaded colourmap GPU textures. Index = ColourmapId value.
    pub(crate) colourmap_textures: Vec<wgpu::Texture>,
    /// Views into colourmap_textures. Index = ColourmapId value.
    pub(crate) colourmap_views: Vec<wgpu::TextureView>,
    /// CPU-side copy of each colourmap for egui scalar bar rendering. Index = ColourmapId value.
    pub(crate) colourmaps_cpu: Vec<[[u8; 4]; 256]>,
    /// Fallback 1x1 LUT texture (bound when has_attribute=0; content irrelevant to the shader).
    #[allow(dead_code)]
    pub(crate) fallback_lut_texture: wgpu::Texture,
    /// View of fallback_lut_texture.
    pub(crate) fallback_lut_view: wgpu::TextureView,
    /// Fallback 4-byte zero storage buffer (bound when no scalar attribute is active).
    pub(crate) fallback_scalar_buf: wgpu::Buffer,
    /// Fallback 16-byte zero storage buffer (bound to binding 8 when no face colour attribute is active).
    pub(crate) fallback_face_colour_buf: wgpu::Buffer,
    /// Fallback 12-byte zero storage buffer (bound to binding 9 when no warp attribute is active).
    pub(crate) fallback_warp_buf: wgpu::Buffer,
    /// Fallback 12-byte zero storage buffer (bound to binding 13 when no
    /// position override is active). Single `vec3<f32>(0,0,0)` entry; the
    /// shader bounds-checks `arrayLength` before reading.
    pub(crate) fallback_position_override_buf: wgpu::Buffer,
    /// Fallback 12-byte zero storage buffer (bound to binding 14 when no
    /// normal override is active).
    pub(crate) fallback_normal_override_buf: wgpu::Buffer,
    /// IDs of built-in preset colourmaps, in BuiltinColourmap discriminant order.
    /// `None` until `ensure_colourmaps_initialized()` has been called.
    pub(crate) builtin_colourmap_ids: Option<[ColourmapId; 10]>,
    /// Whether built-in colourmaps have been uploaded to the GPU.
    pub(crate) colourmaps_initialized: bool,
}

/// Device-shared GPU resources: pipelines, layouts, samplers, fallbacks, LUTs,
/// and the per-feature pipeline clusters (`decal`, `scatter`, `volume`, ...).
/// Created once at init and shared across every viewport.
///
/// Typically stored in the host framework's resource container and accessed
/// by `ViewportRenderer` during prepare() and paint().
#[allow(dead_code)]
pub struct DeviceResources {
    /// Swapchain texture format; all pipelines are compiled for this format.
    pub target_format: wgpu::TextureFormat,
    /// MSAA sample count used by all render pipelines.
    pub sample_count: u32,
    /// Optional pipeline cache shared by every pipeline built here. `Some` only
    /// when the device enables `Features::PIPELINE_CACHE`. Persist its contents
    /// across runs with `ViewportRenderer::pipeline_cache_data` to skip shader
    /// recompilation on later launches.
    pub pipeline_cache: Option<wgpu::PipelineCache>,
    /// Solid-shaded render pipeline (TriangleList topology, no blending).
    pub solid_pipeline: wgpu::RenderPipeline,
    /// Solid-shaded render pipeline with back-face culling disabled (two-sided surfaces).
    pub solid_two_sided_pipeline: wgpu::RenderPipeline,
    /// Transparent render pipeline (TriangleList topology, alpha blending).
    pub transparent_pipeline: wgpu::RenderPipeline,
    /// Wireframe render pipeline (LineList topology, same shader).
    pub wireframe_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer holding the per-frame `CameraUniform` (view-proj + eye position).
    pub camera_uniform_buf: wgpu::Buffer,
    /// Uniform buffer holding the per-frame `LightsUniform` header (count +
    /// hemisphere + IBL + debug params). The per-light array lives in
    /// `light_storage_buf` (binding 13).
    pub light_uniform_buf: wgpu::Buffer,
    /// Storage buffer of per-light `SingleLightUniform` entries (binding 13).
    ///
    /// Sized for `MAX_SCENE_LIGHTS`. The renderer truncates the consumer's
    /// light list to this cap each frame, ranking surplus lights by
    /// `LightSource::importance * proximity_weight`.
    pub light_storage_buf: wgpu::Buffer,
    /// Clustered-shading state: cluster grid, global light index list, and the
    /// per-frame cluster build pipeline. Bindings 14/15/16 of the camera bind
    /// group expose this state to every lit pipeline.
    pub clustered: crate::resources::gpu::clustered::ClusteredResources,
    /// Bind group (group 0) binding camera, light, clip-plane, and shadow uniforms.
    pub camera_bind_group: wgpu::BindGroup,
    /// Bind group layout for group 0 (shared by all scene pipelines).
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for group 1 (per-object uniform: model, material, selection).
    pub object_bind_group_layout: wgpu::BindGroupLayout,
    /// Scene meshes (slotted storage with free-list removal).
    pub(crate) mesh_store: crate::resources::mesh::mesh_store::MeshStore,
    /// Registered LOD groups. Each groups several meshes that are detail
    /// variants of one object; the renderer picks a level per frame.
    pub(crate) lod_groups: crate::resources::mesh::lod::LodGroupStore,
    /// Per-vertex deformation sidecar storage: header uniform, dummy fallback
    /// buffers, and per-mesh slot bind groups. Every mesh-family pipeline
    /// binds `@group(2)` from this state; meshes without attached deformer
    /// data fall back to the renderer-owned dummy bind group.
    pub(crate) deform: crate::resources::mesh_sidecar::deform::DeformationState,
    // --- Shadow map resources ---
    /// Shadow atlas depth texture (Depth32Float, atlas_size x atlas_size, 2x2 tile grid).
    pub shadow_map_texture: wgpu::Texture,
    /// Depth texture view for binding as a shader resource (sampling).
    pub shadow_map_view: wgpu::TextureView,
    /// Comparison sampler for PCF shadow filtering.
    pub shadow_sampler: wgpu::Sampler,
    /// Cubemap-array depth texture for point-light shadows. Layered as
    /// `MAX_POINT_SHADOW_LIGHTS * 6` faces of `POINT_SHADOW_FACE_SIZE` px.
    pub point_shadow_cube_texture: wgpu::Texture,
    /// `texture_depth_cube_array` view bound to the lit-pass bind group.
    pub point_shadow_cube_view: wgpu::TextureView,
    /// One 2D-array view per face, used as the depth attachment during the
    /// shadow render pass. `len() == MAX_POINT_SHADOW_LIGHTS * 6`, indexed
    /// as `slot * 6 + face`.
    pub point_shadow_face_views: Vec<wgpu::TextureView>,
    /// Render pipeline for the point-shadow depth pass. Same vertex layout
    /// as the cascade shadow pipeline; writes linear distance-to-light.
    pub shadow_point_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for the point-shadow per-face uniform (group 0
    /// of the point shadow pass). Kept for pipeline rebuilds.
    pub(crate) shadow_point_face_bind_group_layout: wgpu::BindGroupLayout,
    /// Per-face uniform buffer holding `view_proj`, `light_pos`, `range`
    /// for every (slot, face) of the point shadow array. Sized as
    /// `MAX_POINT_SHADOW_LIGHTS * 6 * 256` bytes (256-byte dynamic-offset
    /// stride).
    pub shadow_point_face_buf: wgpu::Buffer,
    /// Bind group for the point-shadow per-face uniform. Stride is 256;
    /// the per-face render pass sets a dynamic offset.
    pub shadow_point_face_bind_group: wgpu::BindGroup,
    /// Render pipeline for the shadow depth pass (depth-only, no fragment output).
    ///
    /// Culls front faces, so closed solids cast shadow from their back face
    /// and a solid's own front face is never compared against itself in the
    /// shadow map. Two-sided materials (`BackfacePolicy::Identical` and
    /// friends) are routed to `shadow_pipeline_two_sided` instead so both
    /// sides of cloth, foliage, and planar surfaces cast shadows.
    pub shadow_pipeline: wgpu::RenderPipeline,
    /// Shadow caster pipeline for two-sided materials. Same layout and shader
    /// as `shadow_pipeline` but with `cull_mode: None` and a larger caster-side
    /// depth bias (`CSM_SHADOW_BIAS_TWO_SIDED`) so both sides of a two-sided
    /// mesh rasterise into the shadow atlas without the surface self-shadowing
    /// where it is its own receiver.
    pub shadow_pipeline_two_sided: wgpu::RenderPipeline,
    /// Bind group layout for the shadow camera uniform (group 0 of the
    /// shadow pass). Kept on the renderer so `register_deformer` can rebuild
    /// the shadow pipeline from a freshly composed shader module.
    pub(crate) shadow_camera_bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer holding the per-cascade light-space view-projection matrix (64 bytes).
    pub shadow_uniform_buf: wgpu::Buffer,
    /// Bind group for the shadow pass (group 0: light uniform).
    pub shadow_bind_group: wgpu::BindGroup,
    /// Uniform buffer for the ShadowAtlasUniform (binding 5 of camera_bgl, 416 bytes).
    pub shadow_info_buf: wgpu::Buffer,
    /// Current shadow atlas texture size. Used to detect when atlas needs recreation.
    #[allow(dead_code)]
    pub(crate) shadow_atlas_size: u32,
    /// Non-comparison sampler for reading depth values as float (atlas viewer).
    pub shadow_atlas_depth_sampler: wgpu::Sampler,
    /// Pipeline for the shadow atlas corner overlay.
    pub shadow_atlas_viewer_pipeline: wgpu::RenderPipeline,
    /// Bind group for the atlas viewer (uniform + depth texture + sampler).
    pub shadow_atlas_viewer_bg: wgpu::BindGroup,
    /// Uniform buffer: NDC rect of the atlas viewer quad.
    pub shadow_atlas_viewer_buf: wgpu::Buffer,
    /// 16-byte sentinel bound at group 0 binding 12 when the debug fragment buffer is inactive.
    pub debug_frag_sentinel_buf: wgpu::Buffer,

    // --- Gizmo resources ---
    /// Gizmo render pipeline (TriangleList, depth_compare Always : always on top).
    pub gizmo_pipeline: wgpu::RenderPipeline,
    /// Gizmo vertex buffer (3 axis arrows, regenerated when hovered axis changes).
    pub gizmo_vertex_buffer: wgpu::Buffer,
    /// Gizmo index buffer.
    pub gizmo_index_buffer: wgpu::Buffer,
    /// Number of indices in the gizmo index buffer.
    pub gizmo_index_count: u32,
    /// Gizmo uniform buffer (model matrix: positions gizmo at selected object, scaled to screen size).
    pub gizmo_uniform_buf: wgpu::Buffer,
    /// Bind group for gizmo uniform (group 1).
    pub gizmo_bind_group: wgpu::BindGroup,
    /// Bind group layout for gizmo uniforms : stored so per-viewport gizmo bind groups can be created.
    pub(crate) gizmo_bind_group_layout: wgpu::BindGroupLayout,

    // --- Overlay resources ---
    /// Overlay render pipeline (TriangleList with alpha blending : for semi-transparent BC quads).
    pub overlay_pipeline: wgpu::RenderPipeline,
    /// Overlay wireframe pipeline (LineList, no alpha blending needed).
    pub overlay_line_pipeline: wgpu::RenderPipeline,
    /// Full-screen analytical grid pipeline (no vertex buffer : positions hardcoded in shader).
    pub grid_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer for the grid shader (GridUniform : written every frame in prepare()).
    pub grid_uniform_buf: wgpu::Buffer,
    /// Bind group for the grid uniform (group 0, single binding).
    pub grid_bind_group: wgpu::BindGroup,
    /// Bind group layout for the grid uniform (stored so per-viewport grid bind groups can be created).
    pub(crate) grid_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for overlay uniforms (group 1: model + colour uniform).
    pub overlay_bind_group_layout: wgpu::BindGroupLayout,

    // --- Constraint guide lines ---
    /// Transient constraint guide lines, rebuilt each frame in prepare().
    /// Each entry: (vertex_buffer, index_buffer, index_count, uniform_buffer, bind_group).
    pub constraint_line_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,

    // --- Axes indicator ---
    /// Screen-space axes indicator pipeline (TriangleList, no depth, alpha blending).
    pub axes_pipeline: wgpu::RenderPipeline,
    /// Vertex buffer for axes indicator geometry (rebuilt each frame).
    pub axes_vertex_buffer: wgpu::Buffer,
    /// Number of vertices in the axes indicator buffer.
    pub axes_vertex_count: u32,

    // --- Texture system ---
    /// Bind group layout for texture group (group 2: albedo + sampler + normal_map + ao_map).
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    /// Fallback 1x1 white texture used when material.texture_id is None.
    pub fallback_texture: GpuTexture,
    /// Fallback 1x1 flat normal map [128,128,255,255] (tangent-space neutral).
    pub(crate) fallback_normal_map: wgpu::Texture,
    pub(crate) fallback_normal_map_view: wgpu::TextureView,
    /// Fallback 1x1 AO map [255,255,255,255] (no occlusion).
    pub(crate) fallback_ao_map: wgpu::Texture,
    pub(crate) fallback_ao_map_view: wgpu::TextureView,
    /// Fallback 1x1 metallic-roughness texture [0, 255, 255, 255].
    /// G=1.0 and B=1.0 so scalar factors pass through unchanged when no ORM texture is set.
    pub(crate) fallback_metallic_roughness_texture: wgpu::Texture,
    pub(crate) fallback_metallic_roughness_texture_view: wgpu::TextureView,
    /// Fallback 1x1 emissive texture [0, 0, 0, 255] (no emission).
    pub(crate) fallback_emissive_texture: wgpu::Texture,
    pub(crate) fallback_emissive_texture_view: wgpu::TextureView,
    /// Shared linear-repeat sampler for material textures.
    pub(crate) material_sampler: wgpu::Sampler,
    /// Shared linear-clamp sampler for colourmap LUT lookups.
    pub(crate) lut_sampler: wgpu::Sampler,
    /// Uploaded GPU assets and their handle registries (textures, geometry /
    /// scivis stores, colourmap and matcap tables, fallback LUT and attribute buffers).
    pub(crate) content: ContentResources,
    /// Background runner used by async upload entry points. Drained once per
    /// frame during `prepare_scene` so completion is visible to the caller.
    /// Wrapped in a mutex because mpsc receivers and boxed `FnOnce`
    /// callbacks are `Send` but not `Sync`, and several host frameworks
    /// require this struct to be `Sync`.
    pub(crate) jobs: std::sync::Mutex<super::upload_jobs::JobRunner>,
    /// Typed result slots for every async upload path, keyed by job id.
    /// Grouped into one struct so the async bookkeeping is a single field
    /// rather than a score of flat ones; see `upload_jobs::JobResults`.
    pub(crate) job_results: super::upload_jobs::JobResults,

    /// Whether fallback normal map / AO map pixels have been uploaded.
    pub(crate) fallback_textures_uploaded: bool,

    // --- Shared post-processing pipelines / layouts / samplers ---
    /// FXAA/SSAA, bloom, SSAO, tone-map, DoF, contact shadows, placeholders,
    /// PP samplers, depth blit, and dyn-res upscale. Viewport-sized targets and
    /// per-frame uniforms live on `ViewportHdrState`.
    pub(crate) post: crate::resources::postprocess::PostProcessResources,

    // --- Clip planes ---
    /// Uniform buffer for clip planes (binding 4 of camera bind group).
    pub(crate) clip_planes_uniform_buf: wgpu::Buffer,
    /// Uniform buffer for the extended clip volume (binding 6 of camera bind group, 128 bytes).
    pub(crate) clip_volume_uniform_buf: wgpu::Buffer,

    // --- Outline & x-ray resources ---
    // The volume outline mask pipeline lives on `volume.outline_mask_pipeline`;
    // the glyph / tensor-glyph ones on `glyph.outline_mask_pipeline` and
    // `tensor_glyph.outline_mask_pipeline`.
    /// Outline / x-ray pipelines, offscreen mask/composite targets, and layouts.
    pub(crate) outline: OutlineResources,

    // --- Instancing and GPU-culling clusters ---
    /// Instanced-draw pipelines, shared instance storage buffer, and bind group cache.
    pub(crate) instancing: crate::resources::mesh::instancing::InstancingResources,
    /// GPU-cull inputs (per-instance AABBs, per-batch meta) and cull-variant pipelines.
    /// The cull OUTPUTS (visibility indices, indirect args, batch counters) are
    /// per-viewport and live in `ViewportCullState` on each `ViewportSlot`, not here.
    pub(crate) cull: crate::resources::mesh::instancing::CullResources,

    // --- Surface LIC shared resources ---
    /// Surface LIC pipelines and layouts (surface + advect passes).
    pub(crate) lic: crate::resources::postprocess::LicResources,

    /// HDR-format variants of core scene pipelines.
    pub(crate) hdr_solid_pipeline: Option<wgpu::RenderPipeline>,
    /// HDR two-sided variant (cull_mode: None) for analytical surfaces.
    pub(crate) hdr_solid_two_sided_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_transparent_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_wireframe_pipeline: Option<wgpu::RenderPipeline>,
    /// HDR overlay pipeline (TriangleList, Rgba16Float, alpha blending) for cap fill in HDR path.
    pub(crate) hdr_overlay_pipeline: Option<wgpu::RenderPipeline>,

    // --- Gaussian splat pipelines (lazily created) ---
    /// Gaussian splat render/sort pipelines and their bind group layouts.
    pub(crate) gaussian_splat: crate::resources::scivis::gaussian_splat::GaussianSplatResources,

    // --- Sprite billboard pipelines (lazily created) ---
    /// Sprite (emissive + lit) pipelines, layouts, refraction, and soft-particle fallbacks.
    pub(crate) sprite: crate::resources::scivis::sprite::SpriteResources,
    // The polyline outline mask pipeline lives on `polyline.outline_mask_pipeline`.

    // --- point cloud pipelines (lazily created) ---
    /// Point cloud render pipeline. None until first point cloud is submitted.
    pub(crate) point_cloud_pipeline: Option<DualPipeline>,
    /// Bind group layout for point cloud uniforms (group 1).
    pub(crate) point_cloud_bgl: Option<wgpu::BindGroupLayout>,

    // --- glyph rendering (lazily created) ---
    /// Arrow/sphere/cube glyph pipelines, layouts, and cached base meshes.
    pub(crate) glyph: crate::resources::scivis::glyph::GlyphResources,
    /// Tensor glyph pipelines and layouts.
    pub(crate) tensor_glyph: crate::resources::scivis::glyph::TensorGlyphResources,

    // --- polyline / streamtube / ribbon rendering (lazily created) ---
    /// Polyline pipelines and layouts.
    pub(crate) polyline: crate::resources::scivis::polyline::PolylineResources,
    /// Streamtube pipelines and layout.
    pub(crate) streamtube: crate::resources::scivis::tube::StreamtubeResources,
    /// Ribbon pipelines (one per blend) and layout.
    pub(crate) ribbon: crate::resources::scivis::tube::RibbonResources,

    // --- Image slice rendering (lazily created) ---
    /// Image slice render pipeline and layout.
    pub(crate) image_slice: ImageSliceResources,

    // --- volume rendering (lazily created) ---
    /// Volume render/surface-slice/outline pipelines, layouts, cube geometry, and default LUT.
    pub(crate) volume: crate::resources::volume::volumes::VolumeResources,

    // --- GPU compute filtering (lazily created) ---
    /// Compute pipeline for Clip / Threshold index compaction. None until first use.
    pub(crate) compute_filter_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for the compute filter shader (group 0). None until first use.
    pub(crate) compute_filter_bgl: Option<wgpu::BindGroupLayout>,

    // --- Order-independent transparency (OIT) : lazily created ---
    // The viewport-sized accum/reveal textures, composite bind group, and target
    // size live on ViewportHdrState; only the shared pipelines and layout sit here.
    /// Weighted-blended OIT pipelines and composite layout.
    pub(crate) oit: crate::resources::postprocess::OitResources,

    // --- Projected tetrahedra transparent volume rendering (lazily created) ---
    /// Projected-tetrahedra pipeline, layouts, and LUT bind group cache.
    pub(crate) pt: ProjectedTetResources,

    // --- Scatter-volume (participating media) rendering (lazily created) ---
    /// Scatter-volume pipelines, layouts, and per-frame upload buffers.
    pub(crate) scatter: crate::resources::volume::scatter_volume::ScatterResources,

    // --- IBL / environment map resources ---
    /// IBL irradiance equirect texture view (binding 7). None until environment uploaded.
    pub ibl_irradiance_view: Option<wgpu::TextureView>,
    /// IBL prefiltered specular equirect texture view (binding 8). None until environment uploaded.
    pub ibl_prefiltered_view: Option<wgpu::TextureView>,
    /// BRDF integration LUT texture view (binding 9). None until the first
    /// `upload_environment_map`; cached across subsequent uploads (the LUT is
    /// scene-independent: function of roughness x N.V only).
    pub ibl_brdf_lut_view: Option<wgpu::TextureView>,
    /// IBL linear-clamp sampler (binding 10).
    pub(crate) ibl_sampler: wgpu::Sampler,
    /// Skybox / full-res environment equirect texture view (binding 11). None until uploaded.
    pub ibl_skybox_view: Option<wgpu::TextureView>,
    /// Fallback 1x1 black Rgba16Float texture for IBL slots when no environment is loaded.
    #[allow(dead_code)]
    pub(crate) ibl_fallback_texture: wgpu::Texture,
    /// View of ibl_fallback_texture.
    pub(crate) ibl_fallback_view: wgpu::TextureView,
    /// Fallback 1x1 BRDF LUT placeholder; swapped for the real 128x128 LUT
    /// on the first `upload_environment_map` call. Bound to satisfy the bind
    /// group layout when no environment map has been uploaded yet.
    #[allow(dead_code)]
    pub(crate) ibl_fallback_brdf_texture: wgpu::Texture,
    pub(crate) ibl_fallback_brdf_view: wgpu::TextureView,
    /// Uploaded irradiance texture (owned, kept alive for view).
    #[allow(dead_code)]
    pub(crate) ibl_irradiance_texture: Option<wgpu::Texture>,
    /// Uploaded prefiltered specular texture (owned).
    #[allow(dead_code)]
    pub(crate) ibl_prefiltered_texture: Option<wgpu::Texture>,
    /// Uploaded BRDF LUT texture (owned).
    #[allow(dead_code)]
    pub(crate) ibl_brdf_lut_texture: Option<wgpu::Texture>,
    /// Uploaded skybox equirect texture (owned).
    #[allow(dead_code)]
    pub(crate) ibl_skybox_texture: Option<wgpu::Texture>,
    /// Skybox fullscreen render pipeline (renders equirect environment as background).
    pub(crate) skybox_pipeline: wgpu::RenderPipeline,

    // --- Ground plane ---
    /// Full-screen ground plane render pipeline (alpha blending, LessEqual depth).
    pub(crate) ground_plane_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for the ground plane (binding 0: uniform, 1: shadow depth, 2: comparison sampler).
    pub(crate) _ground_plane_bgl: wgpu::BindGroupLayout,
    /// Uniform buffer for GroundPlaneUniform (256 bytes, written each frame in prepare()).
    pub(crate) ground_plane_uniform_buf: wgpu::Buffer,
    /// Bind group for the ground plane pass (rebuilt when shadow atlas changes).
    pub(crate) ground_plane_bind_group: wgpu::BindGroup,

    // --- GPU implicit surface (lazily created) ---
    /// Implicit-surface ray-march pipeline, layout, and outline mask.
    pub(crate) implicit: ImplicitResources,

    // --- GPU marching cubes (lazily created) ---
    /// Marching-cubes compute/render pipelines, layouts, case tables, and per-item volumes.
    pub(crate) mc: crate::resources::volume::gpu_marching_cubes::McResources,

    // --- GPU particle systems ---
    /// Particle compute/draw pipelines, their layouts, and the live systems.
    pub(crate) particle: crate::resources::gpu::gpu_particles::ParticleResources,

    // --- Screen-space image overlays (lazily created) ---
    /// Screen-space image pipelines (plain + depth-composite) and rect outline mask.
    pub(crate) screen_image: ScreenImageResources,

    // --- GPU object-ID picking (lazily created) ---
    /// Object-ID pick pipeline and its bind group layouts.
    pub(crate) pick: PickResources,

    // --- Sub-object highlight (lazily created) ---
    /// Sub-object highlight pipelines (fill / edge / sprite, HDR + LDR) and layout.
    pub(crate) sub_highlight: SubHighlightResources,

    // --- Overlay text / SDF shape / backdrop-blur pipelines (lazily created) ---
    /// Overlay text pipeline, layout, and sampler.
    pub(crate) overlay_text: crate::resources::overlay::overlay_text::OverlayTextResources,
    /// SDF overlay shape pipelines (solid + textured) and sampler.
    pub(crate) overlay_shape: crate::resources::overlay::overlay_shape::OverlayShapeResources,
    /// Backdrop blur pipeline, layout, and sampler.
    pub(crate) backdrop_blur: crate::resources::overlay::overlay_shape::BackdropBlurResources,

    // --- Depth blit pipeline (lazily created, shared across all viewports) ---
    // Copies a scene-resolution depth texture to a native-resolution depth-only target.
    // Used by the HDR path when render_scale < 1.0.
    // The depth-blit and dynamic-resolution upscale pipelines live on `post`.

    // --- Runtime performance tracking ---
    /// Cumulative bytes of geometry data uploaded since the last `prepare()` reset.
    ///
    /// Incremented by `upload_mesh`, `upload_mesh_data`, and `replace_mesh_data`.
    /// Read and reset at the start of each `prepare()` call to populate
    /// `FrameStats::upload_bytes`.
    pub frame_upload_bytes: u64,

    // --- Screen-space decal pipelines (D1 + D5, lazily created) ---
    /// Decal render/exclude pipelines and their bind group layouts.
    pub(crate) decal: crate::resources::decal::DecalResources,

    // --- HiZ occlusion culling ---
    /// When true, the main-camera GPU cull runs the HiZ occlusion test on top
    /// of the frustum test. Off by default (the test is scene-dependent and a
    /// safety valve against the one-frame-stale depth source). The HiZ pyramid
    /// itself is per-viewport and lives on `ViewportCullState::hiz`.
    pub(crate) occlusion_culling_enabled: bool,
}

/// Per-viewport GPU culling outputs.
///
/// The frustum/occlusion cull runs against one camera and writes a compact
/// visibility list plus the indirect draw args for that camera. Those results
/// are viewport-specific: two viewports on different cameras must not share
/// them, or the last one to run would clobber the other. The cull INPUTS
/// (per-instance AABBs, per-batch meta) and all cull PIPELINES are
/// camera-independent and stay on `DeviceResources`.
///
/// Owned by each `ViewportSlot`. The bind-group caches here reference this
/// state's own buffers, so they are invalidated when those buffers resize.
pub(crate) struct ViewportCullState {
    /// Per-batch atomic counter buffer. Zeroed at the start of each cull dispatch.
    pub(crate) batch_counter_buf: Option<wgpu::Buffer>,
    /// Compact list of visible instance indices. Written by the compute cull pass.
    pub(crate) visibility_index_buf: Option<wgpu::Buffer>,
    pub(crate) visibility_index_capacity: usize,
    /// Indirect draw args buffer for the main pass (one DrawIndexedIndirect per batch).
    pub(crate) indirect_args_buf: Option<wgpu::Buffer>,
    /// Capacity (in batches) of the counter and indirect-args buffers.
    pub(crate) batch_output_capacity: usize,
    /// Per-texture-key bind groups for the main cull pipelines.
    /// Keyed by (albedo_id, normal_map_id, ao_map_id); invalidated when
    /// `visibility_index_buf` is resized.
    pub(crate) instance_cull_bind_groups:
        std::collections::HashMap<(u64, u64, u64), wgpu::BindGroup>,
    /// Generation of the shared instance buffers the main cull bind groups were
    /// built against. When it falls behind `InstancingState::instance_gen` the
    /// shared instance storage buffer was rebuilt, so those bind groups (which
    /// bind it at binding 0) are stale and get cleared.
    pub(crate) built_gen: u64,
    /// Hierarchical-Z max-depth pyramid for this viewport's occlusion test.
    /// Lazily created the first frame occlusion culling stores depth here, and
    /// rebuilt when the depth target changes size. Per-viewport so two viewports
    /// on different cameras reproject their own depth instead of clobbering a
    /// shared pyramid.
    pub(crate) hiz: Option<crate::resources::gpu::hiz::HizState>,
}

impl ViewportCullState {
    pub(crate) fn new() -> Self {
        Self {
            batch_counter_buf: None,
            visibility_index_buf: None,
            visibility_index_capacity: 0,
            indirect_args_buf: None,
            batch_output_capacity: 0,
            instance_cull_bind_groups: std::collections::HashMap::new(),
            built_gen: u64::MAX,
            hiz: None,
        }
    }

    /// Allocate or grow this viewport's cull output buffers to fit the current
    /// instance and batch counts. The visibility buffer grows with
    /// `instance_count`; the counter and indirect-args buffers grow with
    /// `batch_count`. Uses the same 2x growth as the shared input buffers. Bind
    /// groups referencing a reallocated buffer are cleared.
    pub(crate) fn ensure_outputs(
        &mut self,
        device: &wgpu::Device,
        instance_count: u32,
        batch_count: u32,
    ) {
        // Visibility buffer, sized like the shared AABB buffer.
        let max_instances = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<InstanceAabb>();
        let instance_count = (instance_count as usize).min(max_instances);
        if instance_count > self.visibility_index_capacity {
            let new_cap = (instance_count * 2).max(64).min(max_instances);
            let vis_size = (new_cap * std::mem::size_of::<u32>()) as u64;
            self.visibility_index_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visibility_index_buf"),
                size: vis_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.visibility_index_capacity = new_cap;
            // The cull bind groups bind the vis buffer at binding 5.
            self.instance_cull_bind_groups.clear();
        }

        // Counter and indirect-args buffers, sized like the shared batch-meta buffer.
        let max_batches = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<BatchMeta>();
        let batch_count = (batch_count as usize).min(max_batches);
        if batch_count > self.batch_output_capacity {
            let new_cap = (batch_count * 2).max(16).min(max_batches);
            let counter_size = (new_cap * std::mem::size_of::<u32>()) as u64;
            // wgpu::util::DrawIndexedIndirect is 5 x u32 = 20 bytes.
            let indirect_size = (new_cap * 20) as u64;
            self.batch_counter_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch_counter_buf"),
                size: counter_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            // iOS Metal and Android (emulator and older devices) do not
            // reliably support INDIRECT_EXECUTION. Leave these as None so the
            // renderer falls back to direct draw calls.
            if cfg!(not(any(target_os = "ios", target_os = "android"))) {
                self.indirect_args_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("indirect_args_buf"),
                    size: indirect_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));
            }
            self.batch_output_capacity = new_cap;
        }
    }
}

/// Scene-scoped GPU culling outputs for the directional shadow cascades.
///
/// Shadows are fit to the primary camera and rendered once into a shared atlas,
/// so the shadow cull is not per-viewport: it runs once per frame and its
/// outputs live here rather than on any `ViewportSlot`. Owned by
/// `InstancingState`.
pub(crate) struct ShadowCullState {
    /// Per-batch atomic counter buffer. Zeroed at the start of each cascade's
    /// cull dispatch.
    pub(crate) batch_counter_buf: Option<wgpu::Buffer>,
    /// Per-cascade visibility index buffers (grow with the instance count).
    pub(crate) shadow_vis_bufs: [Option<wgpu::Buffer>; 4],
    /// Per-cascade indirect draw args buffers (grow with the batch count).
    pub(crate) shadow_indirect_bufs: [Option<wgpu::Buffer>; 4],
    /// Per-cascade instance+visibility bind groups. Invalidated when
    /// `shadow_vis_bufs` are reallocated.
    pub(crate) shadow_cull_instance_bgs: [Option<wgpu::BindGroup>; 4],
    /// Capacity (in instances) of `shadow_vis_bufs`.
    pub(crate) vis_capacity: usize,
    /// Capacity (in batches) of the counter and indirect-args buffers.
    pub(crate) batch_output_capacity: usize,
    /// Generation of the shared instance buffers the shadow cull bind groups were
    /// built against. Mirrors `ViewportCullState::built_gen`: when it falls behind
    /// `InstancingState::instance_gen` the instance storage buffer was rebuilt, so
    /// the bind groups (which bind it at binding 0) are stale.
    pub(crate) built_gen: u64,
}

impl ShadowCullState {
    pub(crate) fn new() -> Self {
        Self {
            batch_counter_buf: None,
            shadow_vis_bufs: [None, None, None, None],
            shadow_indirect_bufs: [None, None, None, None],
            shadow_cull_instance_bgs: [None, None, None, None],
            vis_capacity: 0,
            batch_output_capacity: 0,
            built_gen: u64::MAX,
        }
    }

    /// Allocate or grow the shadow cull output buffers to fit the current
    /// instance and batch counts. Mirrors `ViewportCullState::ensure_outputs`
    /// for the shadow cascades.
    pub(crate) fn ensure_outputs(
        &mut self,
        device: &wgpu::Device,
        instance_count: u32,
        batch_count: u32,
    ) {
        let max_instances = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<InstanceAabb>();
        let instance_count = (instance_count as usize).min(max_instances);
        if instance_count > self.vis_capacity {
            let new_cap = (instance_count * 2).max(64).min(max_instances);
            let vis_size = (new_cap * std::mem::size_of::<u32>()) as u64;
            for i in 0..4 {
                self.shadow_vis_bufs[i] = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("shadow_vis_buf_{i}")),
                    size: vis_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            self.vis_capacity = new_cap;
            // The shadow cull bind groups bind these vis buffers at binding 5.
            self.shadow_cull_instance_bgs = [None, None, None, None];
        }

        let max_batches = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<BatchMeta>();
        let batch_count = (batch_count as usize).min(max_batches);
        if batch_count > self.batch_output_capacity {
            let new_cap = (batch_count * 2).max(16).min(max_batches);
            let counter_size = (new_cap * std::mem::size_of::<u32>()) as u64;
            let indirect_size = (new_cap * 20) as u64;
            self.batch_counter_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow_batch_counter_buf"),
                size: counter_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            if cfg!(not(any(target_os = "ios", target_os = "android"))) {
                for i in 0..4 {
                    self.shadow_indirect_bufs[i] =
                        Some(device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some(&format!("shadow_indirect_buf_{i}")),
                            size: indirect_size,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::INDIRECT
                                | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }));
                }
            }
            self.batch_output_capacity = new_cap;
        }
    }
}

impl DeviceResources {
    /// Create a camera bind group (group 0) for the given per-viewport buffers.
    ///
    /// Per-viewport buffers (camera, clip planes, shadow info, clip volume) are
    /// passed explicitly. Scene-global resources (lights, shadow atlas, IBL) come
    /// from shared resources on `self`.
    ///
    /// NOTE: The initial bind group in `init.rs` is constructed inline (before
    /// `Self` exists). Keep the binding layout in sync when modifying either site.
    pub(crate) fn create_camera_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        clip_planes_buf: &wgpu::Buffer,
        shadow_info_buf: &wgpu::Buffer,
        clip_volume_buf: &wgpu::Buffer,
        debug_frag_buf: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        let irr = self
            .ibl_irradiance_view
            .as_ref()
            .unwrap_or(&self.ibl_fallback_view);
        let spec = self
            .ibl_prefiltered_view
            .as_ref()
            .unwrap_or(&self.ibl_fallback_view);
        let brdf = self
            .ibl_brdf_lut_view
            .as_ref()
            .unwrap_or(&self.ibl_fallback_brdf_view);
        let skybox = self
            .ibl_skybox_view
            .as_ref()
            .unwrap_or(&self.ibl_fallback_view);

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.light_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: clip_planes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: clip_volume_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(irr),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(spec),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(brdf),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.ibl_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(skybox),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: debug_frag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.light_storage_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.clustered.grid_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.clustered.cluster_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.clustered.light_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(&self.point_shadow_cube_view),
                },
            ],
        })
    }
}

impl DeviceResources {
    /// Lazily create the GPU pick pipeline and associated bind group layouts.
    ///
    /// No-op if already created. Called from `ViewportRenderer::pick_scene_gpu`
    /// on first invocation : zero overhead when GPU picking is never used.
    pub(crate) fn ensure_pick_pipeline(&mut self, device: &wgpu::Device) {
        if self.pick.pipeline.is_some() {
            return;
        }

        // --- group 0: pick camera bind group layout ---
        // Includes binding 0 (CameraUniform) and binding 6 (ClipVolumesUniform).
        // The full camera_bind_group_layout has many more bindings; a separate
        // minimal layout is cleaner and avoids binding unused resources.
        let pick_camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick_camera_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // --- group 1: PickInstance storage buffer ---
        let pick_instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick_instance_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "pick_id_shader",
            crate::resources::builders::wgsl_source!("pick_id"),
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pick_pipeline_layout"),
            bind_group_layouts: &[&pick_camera_bgl, &pick_instance_bgl],
            push_constant_ranges: &[],
        });

        // Vertex layout: reuse the 64-byte Vertex stride but only declare position (location 0).
        let pick_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress, // 64 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[pick_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    // location 0: R32Uint object ID
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None, // replace : no blending for integer targets
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location 1: R32Float depth
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling: 3D meshes are often rendered two-sided; pick both faces.
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1, // pick pass is always 1x (no MSAA)
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.pick.camera_bgl = Some(pick_camera_bgl);
        self.pick.bind_group_layout_1 = Some(pick_instance_bgl);
        self.pick.pipeline = Some(pipeline);
    }
}
