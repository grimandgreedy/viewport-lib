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
    pub hdr_texture: crate::gpu::Texture,
    pub hdr_view: crate::gpu::TextureView,
    pub hdr_depth_texture: crate::gpu::Texture,
    pub hdr_depth_view: crate::gpu::TextureView,
    pub hdr_depth_only_view: crate::gpu::TextureView,
    pub hdr_stencil_only_view: crate::gpu::TextureView,

    // --- Bloom ---
    pub bloom_threshold_texture: crate::gpu::Texture,
    pub bloom_threshold_view: crate::gpu::TextureView,
    pub bloom_ping_texture: crate::gpu::Texture,
    pub bloom_ping_view: crate::gpu::TextureView,
    pub bloom_pong_texture: crate::gpu::Texture,
    pub bloom_pong_view: crate::gpu::TextureView,

    // --- SSAO ---
    pub ssao_texture: crate::gpu::Texture,
    pub ssao_view: crate::gpu::TextureView,
    pub ssao_blur_texture: crate::gpu::Texture,
    pub ssao_blur_view: crate::gpu::TextureView,

    // --- Depth of field ---
    pub dof_texture: crate::gpu::Texture,
    pub dof_view: crate::gpu::TextureView,
    pub dof_bind_group: crate::gpu::BindGroup,
    pub dof_uniform_buf: crate::gpu::Buffer,

    // --- Contact shadow ---
    pub contact_shadow_texture: crate::gpu::Texture,
    pub contact_shadow_view: crate::gpu::TextureView,

    // --- Surface LIC ---
    /// Encodes screen-space flow vector per surface pixel (Rgba8Unorm, viewport-sized).
    pub lic_vector_texture: crate::gpu::Texture,
    pub lic_vector_view: crate::gpu::TextureView,
    /// LIC intensity after advection (R8Unorm, viewport-sized). Read by tone_map.wgsl binding 7.
    pub lic_output_texture: crate::gpu::Texture,
    pub lic_output_view: crate::gpu::TextureView,
    /// Per-pixel white noise (R8Unorm, viewport-sized). One independent random value per pixel.
    /// Sampled with textureLoad (nearest) in lic_advect.wgsl to produce directional LIC contrast.
    pub lic_noise_texture: crate::gpu::Texture,
    pub lic_noise_view: crate::gpu::TextureView,
    /// Bind group for the LIC advect render pass (reads lic_vector_texture + lic_noise_texture).
    pub lic_advect_bind_group: crate::gpu::BindGroup,
    /// Uniform buffer for LicAdvectUniform (steps, step_size, viewport dims).
    pub lic_uniform_buf: crate::gpu::Buffer,

    // --- FXAA ---
    pub fxaa_texture: crate::gpu::Texture,
    pub fxaa_view: crate::gpu::TextureView,

    // --- SSAA (allocated when ssaa_factor > 1) ---
    /// Supersampled colour render target. `None` when ssaa_factor == 1.
    pub ssaa_colour_texture: Option<crate::gpu::Texture>,
    pub ssaa_colour_view: Option<crate::gpu::TextureView>,
    /// Supersampled depth render target. `None` when ssaa_factor == 1.
    pub ssaa_depth_texture: Option<crate::gpu::Texture>,
    pub ssaa_depth_view: Option<crate::gpu::TextureView>,
    /// Depth-aspect-only view of `ssaa_depth_texture`, used as the soft-particle
    /// sample source during the SSAA sprite post-pass. `None` when SSAA is off.
    pub ssaa_depth_only_view: Option<crate::gpu::TextureView>,
    /// Bind group for the SSAA resolve pass (reads ssaa_colour_texture). `None` when ssaa_factor == 1.
    pub ssaa_resolve_bind_group: Option<crate::gpu::BindGroup>,
    /// Uniform buffer holding the ssaa_factor value for the resolve shader.
    pub ssaa_uniform_buf: Option<crate::gpu::Buffer>,
    /// The ssaa_factor this state was created with (1 = no SSAA).
    pub ssaa_factor: u32,

    // --- OIT (lazily allocated when transparent geometry is present) ---
    pub oit_accum_texture: Option<crate::gpu::Texture>,
    pub oit_accum_view: Option<crate::gpu::TextureView>,
    pub oit_reveal_texture: Option<crate::gpu::Texture>,
    pub oit_reveal_view: Option<crate::gpu::TextureView>,
    pub oit_composite_bind_group: Option<crate::gpu::BindGroup>,
    pub oit_size: [u32; 2],

    // --- Foreground pass depth (lazily allocated when foreground items are present) ---
    /// Cleared-depth target for the foreground pass, sized to the scene target
    /// (including SSAA). Also sampled by DOF / tone map / the output depth
    /// stamp as a coverage mask (depth < 1.0 = foreground pixel).
    pub foreground_depth_texture: Option<crate::gpu::Texture>,
    pub foreground_depth_view: Option<crate::gpu::TextureView>,
    /// Depth-aspect-only view of `foreground_depth_texture` for sampling.
    pub foreground_depth_only_view: Option<crate::gpu::TextureView>,
    pub foreground_depth_size: [u32; 2],

    // --- Outline offscreen (used by the outline prepare pass) ---
    /// R8Unorm mask: selected objects rendered as white on black.
    pub outline_mask_texture: crate::gpu::Texture,
    pub outline_mask_view: crate::gpu::TextureView,
    /// RGBA output of the edge-detection pass (composited onto the main target).
    pub outline_colour_texture: crate::gpu::Texture,
    pub outline_colour_view: crate::gpu::TextureView,
    pub outline_depth_texture: crate::gpu::Texture,
    pub outline_depth_view: crate::gpu::TextureView,
    /// Depth-aspect view of `outline_depth_texture` for sampling (the HiZ
    /// occlusion prev-depth copy on the LDR path).
    pub outline_depth_only_view: crate::gpu::TextureView,
    /// Bind group for the edge-detection pass (reads mask, writes to colour).
    pub outline_edge_bind_group: crate::gpu::BindGroup,
    /// Uniform buffer for the edge-detection pass parameters.
    pub outline_edge_uniform_buf: crate::gpu::Buffer,
    pub outline_composite_bind_group: crate::gpu::BindGroup,

    // --- Bind groups (rebuilt when viewport dimensions change) ---
    pub tone_map_bind_group: crate::gpu::BindGroup,
    pub bloom_threshold_bg: crate::gpu::BindGroup,
    /// H-blur bind group that reads from bloom_threshold (pass 0 only).
    pub bloom_blur_h_bg: crate::gpu::BindGroup,
    /// V-blur bind group that reads from bloom_ping.
    pub bloom_blur_v_bg: crate::gpu::BindGroup,
    /// H-blur bind group that reads from bloom_pong (passes 1+).
    pub bloom_blur_h_pong_bg: crate::gpu::BindGroup,
    pub ssao_bg: crate::gpu::BindGroup,
    pub ssao_blur_bg: crate::gpu::BindGroup,
    pub dof_bg: crate::gpu::BindGroup,
    pub contact_shadow_bg: crate::gpu::BindGroup,
    pub fxaa_bind_group: crate::gpu::BindGroup,

    // --- Per-viewport uniform buffers ---
    pub tone_map_uniform_buf: crate::gpu::Buffer,
    pub bloom_uniform_buf: crate::gpu::Buffer,
    /// Constant H-blur uniform buffer (horizontal=1, written once at creation).
    pub bloom_h_uniform_buf: crate::gpu::Buffer,
    /// Constant V-blur uniform buffer (horizontal=0, written once at creation).
    pub bloom_v_uniform_buf: crate::gpu::Buffer,
    pub ssao_uniform_buf: crate::gpu::Buffer,
    pub contact_shadow_uniform_buf: crate::gpu::Buffer,

    // --- Post-tone-map depth buffer (native resolution) ---
    // When scene_size == output_size (render_scale = 1.0) this is None and
    // hdr_depth_view is used directly for post-tone-map passes.
    // When scene_size != output_size the scene depth is blitted into this
    // native-resolution texture so that post-tone-map passes (grid, gizmos,
    // axes, etc.) can use it as a depth attachment alongside output_view.
    pub output_depth_texture: Option<crate::gpu::Texture>,
    pub output_depth_view: crate::gpu::TextureView,
    /// Bind group for the depth blit pass (reads hdr_depth_only_view).
    /// None when scene_size == output_size (no blit needed).
    pub depth_blit_bind_group: Option<crate::gpu::BindGroup>,

    // --- HDR upscale (allocated when scene_size != output_size) ---
    // When render_scale < 1.0, tone-map and FXAA run at scene resolution.
    // The result is written to upscale_texture, then upscale-blitted to output_view.
    pub upscale_texture: Option<crate::gpu::Texture>,
    pub upscale_view: Option<crate::gpu::TextureView>,
    pub upscale_bind_group: Option<crate::gpu::BindGroup>,

    /// Native output resolution [width, height].
    pub output_size: [u32; 2],
    /// Effective scene resolution after render scale: [output_size * render_scale].
    /// Equals output_size when render_scale = 1.0.
    pub scene_size: [u32; 2],

    // --- Decal pass depth binding (D1) ---
    /// Bind group for group 1 of the decal pass: reads hdr_depth_only_view as a depth texture.
    /// Rebuilt on viewport resize alongside the other viewport-sized bind groups.
    pub decal_depth_bg: crate::gpu::BindGroup,
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
    pub raw_current_texture: crate::gpu::Texture,
    pub raw_current_view: crate::gpu::TextureView,
    /// History ping-pong. The temporal-resolve pass reads one slot
    /// (history_prev) and writes the other (history_new). `parity` selects.
    #[allow(dead_code)]
    pub history_a_texture: crate::gpu::Texture,
    pub history_a_view: crate::gpu::TextureView,
    #[allow(dead_code)]
    pub history_b_texture: crate::gpu::Texture,
    pub history_b_view: crate::gpu::TextureView,
    /// Composite bind group reading the raw-current texture.
    /// Used when temporal accumulation is disabled.
    pub composite_bg_raw: crate::gpu::BindGroup,
    /// Composite bind groups reading either history slot, used as the source
    /// after the temporal-resolve pass has written history_new.
    pub composite_bg_history_a: crate::gpu::BindGroup,
    pub composite_bg_history_b: crate::gpu::BindGroup,
    /// Temporal-resolve bind groups, keyed by which history slot is being
    /// read as the previous-frame input. Each binds raw_current + the chosen
    /// history slot.
    pub temporal_resolve_bg_read_a: crate::gpu::BindGroup,
    pub temporal_resolve_bg_read_b: crate::gpu::BindGroup,
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
    pub refraction_source_texture: Option<crate::gpu::Texture>,
    /// View paired with `refraction_source_texture`. Bound as the source
    /// during the refraction pass and as the render target during the
    /// preceding blit-copy of the HDR scene.
    pub refraction_source_view: Option<crate::gpu::TextureView>,
    /// Per-viewport bind group binding `(refraction_source_view, depth)` to
    /// the refraction pass.
    pub refraction_source_bg: Option<crate::gpu::BindGroup>,
    /// Per-viewport bind group binding the HDR view as the source for the
    /// blit-copy that fills `refraction_source_view`.
    pub refraction_blit_bg: Option<crate::gpu::BindGroup>,
    /// Allocated size of the refraction source, matched to the HDR target.
    pub refraction_source_size: [u32; 2],
}
/// A render pipeline compiled for both the LDR swapchain format and the HDR
/// intermediate format (`Rgba16Float`). Used for pipelines that draw into the
/// primary scene colour attachment, which may be either format depending on
/// whether post-processing is active.
pub(crate) struct DualPipeline {
    pub ldr: crate::gpu::RenderPipeline,
    pub hdr: crate::gpu::RenderPipeline,
}

impl DualPipeline {
    /// Select the pipeline matching the current render target format.
    /// Pass `true` when drawing into the HDR scene pass (`Rgba16Float`),
    /// `false` when drawing into the LDR swapchain pass.
    pub fn for_format(&self, hdr: bool) -> &crate::gpu::RenderPipeline {
        if hdr { &self.hdr } else { &self.ldr }
    }
}

/// GPU object-ID picking pipeline and its bind group layouts. Lazily built.
#[derive(Default)]
pub(crate) struct PickResources {
    /// Render pipeline that outputs flat u32 object IDs to R32Uint + R32Float targets.
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 1 layout (PickInstance storage buffer).
    pub(crate) bind_group_layout_1: Option<crate::gpu::BindGroupLayout>,
    /// Minimal camera-only bind group layout (group 0).
    pub(crate) camera_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Surface VERTEX pick pipeline: like `pipeline`, but the fragment writes the
    /// hit triangle's nearest corner (global vertex index) into the primitive
    /// channel. Only built when the device has SHADER_PRIMITIVE_INDEX.
    pub(crate) vertex_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for `vertex_pipeline`: the mesh vertex buffer as raw f32s
    /// (binding 0) and its triangle index buffer (binding 1), both read-only
    /// storage.
    pub(crate) vertex_mesh_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Surface EDGE pick pipeline: like `vertex_pipeline`, but the fragment writes
    /// the hit triangle's nearest edge id (`primitive_index * 3 + local_edge`) into
    /// the primitive channel. Reuses `vertex_mesh_bgl` for group 2. Only built with
    /// SHADER_PRIMITIVE_INDEX.
    pub(crate) edge_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Curve POLY_NODE pick pipeline: draws the tube/ribbon/streamtube mesh and
    /// writes the nearer of the hit triangle's two segment endpoints (global node
    /// index) into the primitive channel. Only built with SHADER_PRIMITIVE_INDEX.
    pub(crate) node_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for `node_pipeline`: the per-triangle node payload buffer
    /// (read-only storage).
    pub(crate) node_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for glyph sets. Reuses the render glyph transform and writes
    /// the set's object id.
    pub(crate) glyph_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Pick pipeline for tensor glyph sets.
    pub(crate) tensor_glyph_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 1 layout shared by the glyph and tensor glyph pick pipelines: the
    /// set's uniform (binding 0) plus the object-id uniform (binding 3).
    pub(crate) glyph_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for polylines. Reuses the polyline render vertex expansion
    /// and writes the item's object id.
    pub(crate) polyline_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for the polyline pick pipeline (per-draw object-id uniform).
    pub(crate) polyline_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for voxel volumes: rasterises the volume bounding cube and
    /// raymarches to the first in-threshold voxel, writing the item's object id
    /// and that voxel's depth. Reuses the volume render group-1 layout.
    pub(crate) volume_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for the volume pick pipeline (per-item object-id uniform).
    pub(crate) volume_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for GPU implicit SDF surfaces: raymarches the isosurface on a
    /// full-screen quad and writes the item's object id and hit depth. Reuses the
    /// implicit render group-1 uniform layout.
    pub(crate) implicit_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for the implicit pick pipeline (per-item object-id uniform).
    pub(crate) implicit_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for GPU marching-cubes surfaces: rasterises the generated MC
    /// vertex buffer and writes the job's object id and depth.
    pub(crate) mc_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 1 layout for the MC pick pipeline (per-job object-id uniform).
    pub(crate) mc_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for point clouds: reuses the render screen-space quad
    /// expansion and writes the item's object id plus the hit point's instance
    /// index.
    pub(crate) point_cloud_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for the point cloud pick pipeline (per-item object-id uniform).
    pub(crate) point_cloud_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for Gaussian splats: reuses the render covariance
    /// projection and writes the item's object id plus the hit splat's instance
    /// index.
    pub(crate) gaussian_splat_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for the Gaussian splat pick pipeline (per-item object-id uniform).
    pub(crate) gaussian_splat_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for image slices: reuses the render quad-from-vertex-index
    /// expansion and writes the item's object id. Object-level.
    pub(crate) image_slice_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for the image slice pick pipeline (per-item object-id uniform).
    pub(crate) image_slice_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pick pipeline for volume surface slices: reuses the render mesh vertex
    /// buffer and writes the item's object id. Object-level.
    pub(crate) volume_surface_slice_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for the volume surface slice pick pipeline (per-item object-id uniform).
    pub(crate) volume_surface_slice_pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
}

/// GPU implicit-surface ray-march pipeline and layout. Lazily built.
#[derive(Default)]
pub(crate) struct ImplicitResources {
    /// Render pipeline for GPU-side implicit surface ray-marching.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Group 1 layout (ImplicitUniformRaw).
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    /// Outline mask pipeline for implicit surfaces. None until first selected item.
    pub(crate) outline_mask_pipeline: Option<crate::gpu::RenderPipeline>,
}

/// Screen-space image quad pipelines (plain + depth-composite) and the rect
/// outline mask pipeline. Lazily built.
#[derive(Default)]
pub(crate) struct ScreenImageResources {
    /// Render pipeline for screen-space image quads.
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 0 layout (uniform + texture + sampler).
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    /// Depth-composite pipeline (LessEqual depth, per-pixel image depth).
    pub(crate) dc_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 0 layout for the dc pipeline (uniform + colour + sampler + depth).
    pub(crate) dc_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Outline mask pipeline for screen-space rect images. None until first selected.
    pub(crate) rect_outline_mask_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Layout for the rect outline mask pipeline (NdcRectUniform).
    pub(crate) rect_outline_bgl: Option<crate::gpu::BindGroupLayout>,
}

/// Sub-object highlight pipelines (fill / edge / sprite, HDR + LDR) and layout.
/// Lazily built the first frame a sub-selection is present.
#[derive(Default)]
pub(crate) struct SubHighlightResources {
    /// Translucent face fill pipeline (HDR).
    pub(crate) fill_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Depth-nudged billboard edge-line pipeline (HDR).
    pub(crate) edge_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Billboard sprite pipeline for vertex/point highlights (HDR).
    pub(crate) sprite_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Translucent face fill pipeline (LDR).
    pub(crate) fill_ldr_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Depth-nudged billboard edge-line pipeline (LDR).
    pub(crate) edge_ldr_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Billboard sprite pipeline for vertex/point highlights (LDR).
    pub(crate) sprite_ldr_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Shared group 1 layout (SubHighlightUniform).
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
}

/// Projected-tetrahedra transparent volume pipeline, layouts, and LUT bind
/// group cache. Lazily built.
#[derive(Default)]
pub(crate) struct ProjectedTetResources {
    /// Render pipeline for the projected tetrahedra pass.
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 1 layout (per-volume uniform + tet storage buffer).
    pub(crate) bind_group_layout: Option<crate::gpu::BindGroupLayout>,
    /// Group 2 layout (per-frame colourmap LUT + sampler).
    pub(crate) lut_bind_group_layout: Option<crate::gpu::BindGroupLayout>,
    /// Cache of LUT bind groups keyed by colourmap slot index.
    pub(crate) lut_bind_groups: std::collections::HashMap<usize, crate::gpu::BindGroup>,
    /// LUT bind group for the fallback colourmap.
    pub(crate) fallback_lut_bind_group: Option<crate::gpu::BindGroup>,
}

/// Selection-outline and x-ray pipelines, the offscreen mask/composite targets,
/// and their layouts. The mask/edge/xray/splat pipelines are built eagerly at
/// init; the offscreen textures and composite pipelines are lazily created.
pub(crate) struct OutlineResources {
    /// Group 1 layout for OutlineUniform (mask/xray pipelines).
    pub(crate) bind_group_layout: crate::gpu::BindGroupLayout,
    /// Mask-write pipeline: selected objects as r=1.0 to an R8 mask.
    pub(crate) mask_pipeline: crate::gpu::RenderPipeline,
    /// Two-sided mask-write pipeline (no face culling).
    pub(crate) mask_two_sided_pipeline: crate::gpu::RenderPipeline,
    /// Fullscreen edge-detection pipeline: reads mask, outputs the outline ring.
    pub(crate) edge_pipeline: crate::gpu::RenderPipeline,
    /// Layout for the edge-detection pass (mask texture + sampler + uniform).
    pub(crate) edge_bgl: crate::gpu::BindGroupLayout,
    /// X-ray pipeline: draws selected objects through occluders (depth Always).
    pub(crate) xray_pipeline: crate::gpu::RenderPipeline,
    /// Billboard disc pipeline for the Gaussian splat outline mask pass.
    pub(crate) splat_mask_pipeline: crate::gpu::RenderPipeline,
    /// Offscreen RGBA texture the outline stencil pass renders into.
    pub(crate) colour_texture: Option<crate::gpu::Texture>,
    pub(crate) colour_view: Option<crate::gpu::TextureView>,
    /// Depth+stencil texture for the offscreen outline pass.
    pub(crate) depth_texture: Option<crate::gpu::Texture>,
    pub(crate) depth_view: Option<crate::gpu::TextureView>,
    /// Size of the current outline offscreen textures.
    pub(crate) target_size: [u32; 2],
    /// Fullscreen composite pipelines: single-sample LDR, MSAA, HDR.
    pub(crate) composite_pipeline_single: Option<crate::gpu::RenderPipeline>,
    pub(crate) composite_pipeline_msaa: Option<crate::gpu::RenderPipeline>,
    pub(crate) composite_pipeline_hdr: Option<crate::gpu::RenderPipeline>,
    pub(crate) composite_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) composite_bind_group: Option<crate::gpu::BindGroup>,
    pub(crate) composite_sampler: Option<crate::gpu::Sampler>,
}

/// Image slice render pipeline and layout. Lazily built.
#[derive(Default)]
pub(crate) struct ImageSliceResources {
    /// Image slice render pipeline. None until first slice item is submitted.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Group 1 layout for image slice uniforms.
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
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
    pub(crate) material_bind_groups:
        std::collections::HashMap<(u64, u64, u64), crate::gpu::BindGroup>,
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
    /// Uploaded 3D volume textures, keyed by `VolumeId`. Slotted with
    /// generational ids so a freed slot cannot alias a later upload, and the
    /// per-entry byte charge feeds `ResidentBytes::volume_bytes`.
    pub(crate) volume_textures: crate::resources::handle::SlotStore<
        (crate::gpu::Texture, crate::gpu::TextureView),
        crate::resources::VolumeId,
    >,
    /// Uploaded projected-tet meshes, keyed by `ProjectedTetId`. Slotted with
    /// generational ids so a freed slot cannot alias a later upload, and the
    /// per-entry byte charge feeds `ResidentBytes::projected_tet_bytes`.
    pub(crate) projected_tet_store:
        crate::resources::handle::SlotStore<GpuProjectedTetMesh, crate::resources::ProjectedTetId>,
    /// Glyph atlas for overlay text rendering (labels, scalar bars, rulers).
    pub(crate) glyph_atlas: crate::resources::overlay::font::GlyphAtlas,
    /// Persistent textures uploaded via `upload_overlay_texture`.
    pub(crate) overlay_textures: crate::resources::handle::Registry<OverlayShapeTextureEntry>,
    /// Matcap textures (256x256 RGBA), indexed by `MatcapId::index`.
    pub(crate) matcap_textures: Vec<crate::gpu::Texture>,
    /// Texture views for each uploaded matcap.
    pub(crate) matcap_views: Vec<crate::gpu::TextureView>,
    /// Linear-clamp sampler shared by all matcap texture lookups.
    pub(crate) matcap_sampler: Option<crate::gpu::Sampler>,
    /// Fallback 1x1 white view bound to binding 7 when no matcap is active.
    pub(crate) fallback_matcap_view: Option<crate::gpu::TextureView>,
    /// Whether built-in matcaps have been uploaded to the GPU.
    pub(crate) matcaps_initialized: bool,
    /// `MatcapId` for each built-in preset, populated by `ensure_matcaps_initialized`.
    pub(crate) builtin_matcap_ids: Option<[MatcapId; 8]>,
    /// Uploaded colourmap GPU textures. Index = ColourmapId value.
    pub(crate) colourmap_textures: Vec<crate::gpu::Texture>,
    /// Views into colourmap_textures. Index = ColourmapId value.
    pub(crate) colourmap_views: Vec<crate::gpu::TextureView>,
    /// CPU-side copy of each colourmap for egui scalar bar rendering. Index = ColourmapId value.
    pub(crate) colourmaps_cpu: Vec<[[u8; 4]; 256]>,
    /// Fallback 1x1 LUT texture (bound when has_attribute=0; content irrelevant to the shader).
    #[allow(dead_code)]
    pub(crate) fallback_lut_texture: crate::gpu::Texture,
    /// View of fallback_lut_texture.
    pub(crate) fallback_lut_view: crate::gpu::TextureView,
    /// Fallback 4-byte zero storage buffer (bound when no scalar attribute is active).
    pub(crate) fallback_scalar_buf: crate::gpu::Buffer,
    /// Fallback 16-byte zero storage buffer (bound to binding 8 when no face colour attribute is active).
    pub(crate) fallback_face_colour_buf: crate::gpu::Buffer,
    /// Fallback 12-byte zero storage buffer (bound to binding 9 when no warp attribute is active).
    pub(crate) fallback_warp_buf: crate::gpu::Buffer,
    /// Fallback 12-byte zero storage buffer (bound to binding 13 when no
    /// position override is active). Single `vec3<f32>(0,0,0)` entry; the
    /// shader bounds-checks `arrayLength` before reading.
    pub(crate) fallback_position_override_buf: crate::gpu::Buffer,
    /// Fallback 12-byte zero storage buffer (bound to binding 14 when no
    /// normal override is active).
    pub(crate) fallback_normal_override_buf: crate::gpu::Buffer,
    /// Fallback 16-byte zero storage buffer (bound to binding 15 when the
    /// mesh has no extension-attribute buffer). Single `vec4<f32>(0)` entry;
    /// plugin modules clamp the vertex index so every read resolves to zero.
    pub(crate) fallback_extension_attr_buf: crate::gpu::Buffer,
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
    pub(crate) target_format: crate::gpu::TextureFormat,
    /// MSAA sample count used by all render pipelines.
    pub(crate) sample_count: u32,
    /// True while the lit pipelines are compiled with the pixel-inspector
    /// debug block. Off by default: the block's storage write disables early
    /// depth rejection (see `builders::strip_debug_vis`). Toggled per frame
    /// from the `DebugVis` state, which rebuilds the lit pipelines.
    pub(crate) debug_vis_shaders: bool,
    /// Set by `register_deformer` / `register_internal_deformer` instead of
    /// rebuilding the mesh-family pipelines inline, so a burst of
    /// registrations costs one recompose instead of one per call. Cleared by
    /// `flush_mesh_pipeline_rebuild`, which prepare runs at the start of
    /// every frame.
    pub(crate) mesh_pipelines_dirty: bool,
    /// Optional pipeline cache shared by every pipeline built here. `Some` only
    /// when the device enables `Features::PIPELINE_CACHE`. Persist its contents
    /// across runs with `ViewportRenderer::pipeline_cache_data` to skip shader
    /// recompilation on later launches.
    pub(crate) pipeline_cache: Option<crate::gpu::PipelineCache>,
    /// Core scene mesh pipelines: base LDR set (solid, two-sided, transparent,
    /// wireframe) and their lazily-built HDR variants. See
    /// `resources::scene_pipelines::SceneCorePipelines`.
    pub(crate) scene: crate::resources::scene_pipelines::SceneCorePipelines,
    /// Uniform buffer holding the per-frame `CameraUniform` (view-proj + eye position).
    pub(crate) camera_uniform_buf: crate::gpu::Buffer,
    /// Uniform buffer holding the per-frame `LightsUniform` header (count +
    /// hemisphere + IBL + debug params). The per-light array lives in
    /// `light_storage_buf` (binding 13).
    pub(crate) light_uniform_buf: crate::gpu::Buffer,
    /// Storage buffer of per-light `SingleLightUniform` entries (binding 13).
    ///
    /// Sized for `MAX_SCENE_LIGHTS`. The renderer truncates the consumer's
    /// light list to this cap each frame, ranking surplus lights by
    /// `LightSource::importance * proximity_weight`.
    pub(crate) light_storage_buf: crate::gpu::Buffer,
    /// Uploaded SH light-probe field, sampled per object at prepare time. `None`
    /// until `set_light_probes` is called.
    pub(crate) light_probes: Option<crate::resources::LightProbeSet>,
    /// Indirect-lighting storage buffer (group 0 binding 18). First region: the
    /// per-object blended SH, one 9-`vec4` block per light-probe-lit object,
    /// written each frame. Second region (from `MAX_LIGHT_PROBE_OBJECTS *
    /// SH_GPU_STRIDE_BYTES`): the environment-selection zones, `env_zone_count`
    /// live. Sharing one buffer keeps the fragment stage within the storage-buffer
    /// budget; see `load_env_zone` in `scene_lighting.wgsl`.
    pub(crate) indirect_light_buf: crate::gpu::Buffer,
    /// Uploaded adaptive probe volume (group 0 binding 20): a header plus SH per
    /// grid cell, sampled per fragment by world position. `None` until
    /// `set_light_probe_volume`; the fallback (a disabled 3-`vec4` header) is
    /// bound in its place so the binding is always valid.
    pub(crate) light_probe_volume_buf: Option<crate::gpu::Buffer>,
    /// Disabled 3-`vec4` header bound at binding 20 when no volume is set.
    pub(crate) light_probe_volume_fallback: crate::gpu::Buffer,
    /// Clustered-shading state: cluster grid, global light index list, and the
    /// per-frame cluster build pipeline. Bindings 14/15/16 of the camera bind
    /// group expose this state to every lit pipeline.
    pub(crate) clustered: crate::resources::gpu::clustered::ClusteredResources,
    /// Bind group (group 0) binding camera, light, clip-plane, and shadow uniforms.
    pub(crate) camera_bind_group: crate::gpu::BindGroup,
    /// Bind group layout for group 0 (shared by all scene pipelines).
    pub(crate) camera_bind_group_layout: crate::gpu::BindGroupLayout,
    /// Bind group layout for group 1 (per-object uniform: model, material, selection).
    pub(crate) object_bind_group_layout: crate::gpu::BindGroupLayout,
    /// Scene meshes (slotted storage with free-list removal).
    pub(crate) mesh_store: crate::resources::mesh::mesh_store::MeshStore,
    /// Shared vertex/index geometry buffers. Each mesh's `vertex_span` /
    /// `index_span` is a window into these; geometry writes recorded at upload
    /// flush in `process_uploads`.
    pub(crate) geometry: crate::resources::mesh::geometry_slab::GeometrySlab,
    /// Registered LOD groups. Each groups several meshes that are detail
    /// variants of one object; the renderer picks a level per frame.
    pub(crate) lod_groups: crate::resources::mesh::lod::LodGroupStore,
    /// Per-vertex deformation sidecar storage: header uniform, dummy fallback
    /// buffers, and per-mesh slot bind groups. Every mesh-family pipeline
    /// binds `@group(2)` from this state; meshes without attached deformer
    /// data fall back to the renderer-owned dummy bind group.
    pub(crate) deform: crate::resources::mesh_sidecar::deform::DeformationState,
    /// Registered fragment shading hooks (`register_shading_hook`). Each
    /// entry composes its own lit-shader modules; the base pipelines are
    /// untouched by registrations.
    pub(crate) shade_hooks: Vec<crate::resources::mesh_sidecar::shade::StoredShadingHook>,
    /// Per-material-plugin GPU state (`register_material_plugin`), keyed by
    /// hook index: group-3 params window + lazily built lit pipeline set.
    pub(crate) material_plugins:
        std::collections::HashMap<u32, crate::resources::mesh_sidecar::shade::MaterialPluginGpu>,
    // --- Shadow map resources ---
    /// Cascade shadow atlas, point-light shadow cube array, their depth passes,
    /// and the atlas debug viewer. See `resources::shadow::ShadowResources`.
    pub(crate) shadow: crate::resources::shadow::ShadowResources,
    /// 16-byte sentinel bound at group 0 binding 12 when the debug fragment buffer is inactive.
    pub(crate) debug_frag_sentinel_buf: crate::gpu::Buffer,

    // --- Gizmo resources ---
    /// Transform-gizmo pipeline, axis-arrow geometry, and uniform bindings.
    /// See `resources::gizmo::GizmoResources`.
    pub(crate) gizmo: crate::resources::gizmo::GizmoResources,

    // --- Overlay guide resources (grid, axes, base overlay, constraint lines) ---
    /// Floor grid, axes indicator, base overlay pipelines, and constraint guide
    /// lines. See `resources::overlay::guides::OverlayGuideResources`.
    pub(crate) guides: crate::resources::overlay::guides::OverlayGuideResources,

    // --- Texture system ---
    /// Fallback material textures (1x1 neutral maps), shared material / LUT /
    /// depth-read samplers, and the texture-group and depth-read bind group
    /// layouts. See `material::fallbacks::MaterialFallbacks`.
    pub(crate) material: crate::resources::material::fallbacks::MaterialFallbacks,
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

    // --- Shared post-processing pipelines / layouts / samplers ---
    /// FXAA/SSAA, bloom, SSAO, tone-map, DoF, contact shadows, placeholders,
    /// PP samplers, depth blit, and dyn-res upscale. Viewport-sized targets and
    /// per-frame uniforms live on `ViewportHdrState`.
    pub(crate) post: crate::resources::postprocess::PostProcessResources,

    // --- Clip planes ---
    /// Uniform buffer for clip planes (binding 4 of camera bind group).
    pub(crate) clip_planes_uniform_buf: crate::gpu::Buffer,
    /// Uniform buffer for the extended clip volume (binding 6 of camera bind group, 128 bytes).
    pub(crate) clip_volume_uniform_buf: crate::gpu::Buffer,

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
    pub(crate) point_cloud_bgl: Option<crate::gpu::BindGroupLayout>,

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
    pub(crate) compute_filter_pipeline: Option<crate::gpu::ComputePipeline>,
    /// Bind group layout for the compute filter shader (group 0). None until first use.
    pub(crate) compute_filter_bgl: Option<crate::gpu::BindGroupLayout>,

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
    /// Image-based-lighting views, fallbacks, owned array textures, environment
    /// zone count, and the skybox pipeline. See `material::environment::IblResources`.
    pub(crate) ibl: crate::resources::material::environment::IblResources,

    // --- Ground plane ---
    /// Full-screen ground plane render pipeline (alpha blending, LessEqual depth).
    pub(crate) ground_plane_pipeline: crate::gpu::RenderPipeline,
    /// Bind group layout for the ground plane (binding 0: uniform, 1: shadow depth, 2: comparison sampler).
    pub(crate) _ground_plane_bgl: crate::gpu::BindGroupLayout,
    /// Uniform buffer for GroundPlaneUniform (256 bytes, written each frame in prepare()).
    pub(crate) ground_plane_uniform_buf: crate::gpu::Buffer,
    /// Bind group for the ground plane pass (rebuilt when shadow atlas changes).
    pub(crate) ground_plane_bind_group: crate::gpu::BindGroup,

    // --- GPU implicit surface (lazily created) ---
    /// Implicit-surface ray-march pipeline, layout, and outline mask.
    pub(crate) implicit: ImplicitResources,

    // --- GPU marching cubes (lazily created) ---
    /// Marching-cubes compute/render pipelines, layouts, case tables, and per-item volumes.
    pub(crate) mc: crate::resources::volume::gpu_marching_cubes::McResources,

    // --- GPU particle systems ---
    /// Particle compute/draw pipelines, their layouts, and the live systems.
    pub(crate) particle: crate::resources::gpu::gpu_particles::ParticleResources,

    // --- External instance sets ---
    /// Consumer-buffer instanced mesh drawing (positions produced by the
    /// consumer's own GPU compute on the shared device).
    pub(crate) external_instances:
        crate::resources::gpu::external_instances::ExternalInstancesResources,

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
    pub(crate) frame_upload_bytes: u64,
    /// GPU pipelines built lazily since the last `prepare()` reset.
    ///
    /// Incremented by the `ensure_*` pipeline builders when they actually
    /// create pipelines (not on their no-op early returns). Read and reset
    /// each `prepare()` call to populate `FrameStats::pipelines_built_this_frame`.
    pub(crate) frame_pipelines_built: u32,
    /// Bumped by `free_texture` and `free_mesh`. The per-object draw cache
    /// holds bind groups that keep their referenced GPU resources alive; when
    /// this changes, the cache purges its stale entries so a freed resource's
    /// memory is actually reclaimed instead of pinned by an unused bind group.
    pub(crate) resource_free_epoch: u64,

    // --- Screen-space decal pipelines (D1 + D5, lazily created) ---
    /// Decal render/exclude pipelines and their bind group layouts.
    pub(crate) decal: crate::resources::decal::DecalResources,

    // --- HiZ occlusion culling ---
    /// When true, the main-camera GPU cull runs the HiZ occlusion test on top
    /// of the frustum test. Off by default (the test is scene-dependent and a
    /// safety valve against the one-frame-stale depth source). The HiZ pyramid
    /// itself is per-viewport and lives on `ViewportCullState::hiz`.
    pub(crate) occlusion_culling_enabled: bool,

    // --- Measurement knob ---
    /// When true, the per-object opaque scene-pass draw keeps the discarding
    /// mesh pipeline instead of selecting the discard-free early-Z twin. Off by
    /// default; this exists only so a benchmark can A/B the per-object early-Z
    /// path in a single process. It does not change rendered output, only which
    /// pipeline draws eligible opaque items.
    pub(crate) force_po_discard: bool,
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
    pub(crate) batch_counter_buf: Option<crate::gpu::Buffer>,
    /// Compact list of visible instance indices. Written by the compute cull pass.
    pub(crate) visibility_index_buf: Option<crate::gpu::Buffer>,
    pub(crate) visibility_index_capacity: usize,
    /// Indirect draw args buffer for the main pass (one DrawIndexedIndirect per batch).
    pub(crate) indirect_args_buf: Option<crate::gpu::Buffer>,
    /// Capacity (in batches) of the counter and indirect-args buffers.
    pub(crate) batch_output_capacity: usize,
    /// Per-texture-key bind groups for the main cull pipelines.
    /// Keyed by (albedo_id, normal_map_id, ao_map_id); invalidated when
    /// `visibility_index_buf` is resized.
    pub(crate) instance_cull_bind_groups:
        std::collections::HashMap<(u64, u64, u64), crate::gpu::BindGroup>,
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
        device: &crate::gpu::Device,
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
            self.visibility_index_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("visibility_index_buf"),
                size: vis_size,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
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
            self.batch_counter_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("batch_counter_buf"),
                size: counter_size,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            // iOS Metal and Android (emulator and older devices) do not
            // reliably support INDIRECT_EXECUTION. Leave these as None so the
            // renderer falls back to direct draw calls.
            if cfg!(not(any(target_os = "ios", target_os = "android"))) {
                self.indirect_args_buf =
                    Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some("indirect_args_buf"),
                        size: indirect_size,
                        usage: crate::gpu::BufferUsages::STORAGE
                            | crate::gpu::BufferUsages::INDIRECT
                            | crate::gpu::BufferUsages::COPY_DST
                            | crate::gpu::BufferUsages::COPY_SRC,
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
    pub(crate) batch_counter_buf: Option<crate::gpu::Buffer>,
    /// Per-cascade visibility index buffers (grow with the instance count).
    pub(crate) shadow_vis_bufs: [Option<crate::gpu::Buffer>; 4],
    /// Per-cascade indirect draw args buffers (grow with the batch count).
    pub(crate) shadow_indirect_bufs: [Option<crate::gpu::Buffer>; 4],
    /// Per-cascade instance+visibility bind groups. Invalidated when
    /// `shadow_vis_bufs` are reallocated.
    pub(crate) shadow_cull_instance_bgs: [Option<crate::gpu::BindGroup>; 4],
    /// Per-(cascade, albedo, normal, ao) bind groups for the alpha-cutout shadow
    /// cull pipeline: like `shadow_cull_instance_bgs` but also carries the batch's
    /// albedo texture (from the full cull BGL) so the fragment can discard leaf
    /// gaps. Invalidated with `shadow_cull_instance_bgs`.
    pub(crate) shadow_cutout_cull_bgs:
        std::collections::HashMap<(usize, u64, u64, u64), crate::gpu::BindGroup>,
    /// Capacity (in instances) of `shadow_vis_bufs`.
    pub(crate) vis_capacity: usize,
    /// Capacity (in batches) of the counter and indirect-args buffers.
    pub(crate) batch_output_capacity: usize,
    /// Generation of the shared instance buffers the shadow cull bind groups were
    /// built against. Mirrors `ViewportCullState::built_gen`: when it falls behind
    /// `InstancingState::instance_gen` the instance storage buffer was rebuilt, so
    /// the bind groups (which bind it at binding 0) are stale.
    pub(crate) built_gen: u64,
    /// Per-cascade render bundles replaying the indirect shadow draw sequence.
    /// The batch loop encodes hundreds of set/draw calls per cascade; for a
    /// stable batch list that sequence is identical every frame (per-frame
    /// variation lives in the cascade uniform and the GPU-cull-written indirect
    /// args, both referenced by the bundle, not baked into it), so it is
    /// recorded once and replayed. Viewport/scissor are render-pass state and
    /// still apply around bundle execution, so the per-cascade atlas tiles work
    /// unchanged.
    pub(crate) shadow_bundles: [Option<crate::gpu::RenderBundle>; 4],
    /// (instance_gen, batches_gen, outputs_gen, cascade_count) the bundles were
    /// recorded against; a mismatch re-records them.
    pub(crate) bundle_key: Option<(u64, u64, u64, usize)>,
    /// GPU draws recorded per cascade bundle, for FrameStats.
    pub(crate) bundle_draws: u32,
    /// Geometry buffer binds (`set_vertex_buffer` + `set_index_buffer`) recorded
    /// across the cascade bundles, for `FrameStats::shadow_buffer_binds`. Carried
    /// over on frames that replay the bundles without a rebuild.
    pub(crate) bundle_binds: u32,
    /// Bumped whenever `ensure_outputs` reallocates a buffer the bundles (or
    /// bind groups) reference.
    pub(crate) outputs_gen: u64,
}

impl ShadowCullState {
    pub(crate) fn new() -> Self {
        Self {
            batch_counter_buf: None,
            shadow_vis_bufs: [None, None, None, None],
            shadow_indirect_bufs: [None, None, None, None],
            shadow_cull_instance_bgs: [None, None, None, None],
            shadow_cutout_cull_bgs: std::collections::HashMap::new(),
            vis_capacity: 0,
            batch_output_capacity: 0,
            built_gen: u64::MAX,
            shadow_bundles: [None, None, None, None],
            bundle_key: None,
            bundle_draws: 0,
            bundle_binds: 0,
            outputs_gen: 0,
        }
    }

    /// Allocate or grow the shadow cull output buffers to fit the current
    /// instance and batch counts. Mirrors `ViewportCullState::ensure_outputs`
    /// for the shadow cascades.
    pub(crate) fn ensure_outputs(
        &mut self,
        device: &crate::gpu::Device,
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
                self.shadow_vis_bufs[i] =
                    Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some(&format!("shadow_vis_buf_{i}")),
                        size: vis_size,
                        usage: crate::gpu::BufferUsages::STORAGE
                            | crate::gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }
            self.vis_capacity = new_cap;
            // The shadow cull bind groups bind these vis buffers at binding 5.
            self.shadow_cull_instance_bgs = [None, None, None, None];
            self.shadow_cutout_cull_bgs.clear();
            self.outputs_gen = self.outputs_gen.wrapping_add(1);
        }

        let max_batches = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<BatchMeta>();
        let batch_count = (batch_count as usize).min(max_batches);
        if batch_count > self.batch_output_capacity {
            let new_cap = (batch_count * 2).max(16).min(max_batches);
            let counter_size = (new_cap * std::mem::size_of::<u32>()) as u64;
            let indirect_size = (new_cap * 20) as u64;
            self.batch_counter_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("shadow_batch_counter_buf"),
                size: counter_size,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            if cfg!(not(any(target_os = "ios", target_os = "android"))) {
                for i in 0..4 {
                    self.shadow_indirect_bufs[i] =
                        Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                            label: Some(&format!("shadow_indirect_buf_{i}")),
                            size: indirect_size,
                            usage: crate::gpu::BufferUsages::STORAGE
                                | crate::gpu::BufferUsages::INDIRECT
                                | crate::gpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }));
                }
            }
            self.batch_output_capacity = new_cap;
            self.outputs_gen = self.outputs_gen.wrapping_add(1);
        }
    }
}

impl DeviceResources {
    /// The color/swapchain texture format every pipeline here was compiled for.
    /// This is the format passed to `ViewportRenderer::new`; a viewport target
    /// must match it.
    pub fn target_format(&self) -> crate::gpu::TextureFormat {
        self.target_format
    }

    /// The MSAA sample count every render pipeline here was built with.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

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
        device: &crate::gpu::Device,
        camera_buf: &crate::gpu::Buffer,
        clip_planes_buf: &crate::gpu::Buffer,
        shadow_info_buf: &crate::gpu::Buffer,
        clip_volume_buf: &crate::gpu::Buffer,
        debug_frag_buf: &crate::gpu::Buffer,
        label: &str,
    ) -> crate::gpu::BindGroup {
        let irr = self
            .ibl
            .irradiance_view
            .as_ref()
            .unwrap_or(&self.ibl.fallback_array_view);
        let spec = self
            .ibl
            .prefiltered_view
            .as_ref()
            .unwrap_or(&self.ibl.fallback_array_view);
        let brdf = self
            .ibl
            .brdf_lut_view
            .as_ref()
            .unwrap_or(&self.ibl.fallback_brdf_view);
        let skybox = self
            .ibl
            .skybox_view
            .as_ref()
            .unwrap_or(&self.ibl.fallback_view);

        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.camera_bind_group_layout,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(&self.shadow.map_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&self.shadow.sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: self.light_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: clip_planes_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_info_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: clip_volume_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: crate::gpu::BindingResource::TextureView(irr),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: crate::gpu::BindingResource::TextureView(spec),
                },
                crate::gpu::BindGroupEntry {
                    binding: 9,
                    resource: crate::gpu::BindingResource::TextureView(brdf),
                },
                crate::gpu::BindGroupEntry {
                    binding: 10,
                    resource: crate::gpu::BindingResource::Sampler(&self.ibl.sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 11,
                    resource: crate::gpu::BindingResource::TextureView(skybox),
                },
                crate::gpu::BindGroupEntry {
                    binding: 12,
                    resource: debug_frag_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 13,
                    resource: self.light_storage_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 14,
                    resource: self.clustered.grid_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 15,
                    resource: self.clustered.cluster_grid_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 16,
                    resource: self.clustered.light_index_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 17,
                    resource: crate::gpu::BindingResource::TextureView(
                        &self.shadow.point_cube_view,
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 18,
                    resource: self.indirect_light_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 20,
                    resource: self
                        .light_probe_volume_buf
                        .as_ref()
                        .unwrap_or(&self.light_probe_volume_fallback)
                        .as_entire_binding(),
                },
            ],
        })
    }
}

impl DeviceResources {
    /// Record a lazy pipeline build for `FrameStats::pipelines_built_this_frame`.
    ///
    /// `site` is the `file!()`/`line!()` of the builder, emitted at debug level
    /// under the `viewport_lib::pipelines` target so a hitch traced to a lazy
    /// compile can be attributed to the exact builder.
    pub(crate) fn note_pipeline_built(&mut self, site: &'static str) {
        self.frame_pipelines_built += 1;
        tracing::debug!(target: "viewport_lib::pipelines", site, "lazy pipeline build");
    }
}
