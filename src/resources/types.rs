pub use crate::resources::device_resources::ContentResources;
pub use crate::resources::device_resources::DeviceResources;
pub(crate) use crate::resources::device_resources::DualPipeline;
pub(crate) use crate::resources::device_resources::ImageSliceResources;
pub(crate) use crate::resources::device_resources::ImplicitResources;
pub(crate) use crate::resources::device_resources::OutlineResources;
pub(crate) use crate::resources::device_resources::PickResources;
pub(crate) use crate::resources::device_resources::ProjectedTetResources;
pub(crate) use crate::resources::device_resources::ScatterViewportState;
pub(crate) use crate::resources::device_resources::ScreenImageResources;
pub(crate) use crate::resources::device_resources::ShadowCullState;
pub(crate) use crate::resources::device_resources::SubHighlightResources;
pub(crate) use crate::resources::device_resources::ViewportCullState;
#[allow(deprecated)]
pub use crate::resources::device_resources::ViewportGpuResources;
pub(crate) use crate::resources::device_resources::ViewportHdrState;
/// GPU resources for the 3D viewport.
///
/// Owns the render pipelines (solid + wireframe), vertex/index buffers,
/// uniform buffers, and bind groups.
// Re-exports of types relocated into their domain modules. These keep
// `crate::resources::types::*` paths resolving after the move.
pub use crate::resources::material::colourmap_data::ColourmapId;
pub use crate::resources::material::matcap_data::BuiltinMatcap;
pub use crate::resources::material::matcap_data::MatcapId;
pub use crate::resources::material::textures::GpuTexture;
pub use crate::resources::memory::ResidentBytes;
pub use crate::resources::memory::TextureMemoryStats;
pub use crate::resources::memory::VramBudget;
pub use crate::resources::mesh::gpu_mesh::GpuMesh;
pub use crate::resources::mesh::instancing::BatchMeta;
pub(crate) use crate::resources::mesh::instancing::InstanceAabb;
pub(crate) use crate::resources::mesh::instancing::InstanceData;
pub(crate) use crate::resources::mesh::instancing::ObjectUniform;
pub(crate) use crate::resources::mesh::instancing::PickInstance;
pub use crate::resources::mesh::meshes::MeshData;
pub(crate) use crate::resources::overlay::highlight::CurveMeshOutlineItem;
pub(crate) use crate::resources::overlay::highlight::OutlineEdgeUniform;
pub(crate) use crate::resources::overlay::highlight::OutlineObjectBuffers;
pub(crate) use crate::resources::overlay::highlight::OutlineUniform;
pub(crate) use crate::resources::overlay::highlight::RawGeomOutlineBuffers;
pub(crate) use crate::resources::overlay::highlight::ScreenRectOutlineBuffers;
pub(crate) use crate::resources::overlay::highlight::SplatOutlineBuffers;
pub(crate) use crate::resources::overlay::highlight::SplatOutlineMaskUniform;
pub(crate) use crate::resources::overlay::highlight::SubHighlightGpuData;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShapeGpuData;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShapeTexBatch;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShapeTexVertex;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShapeTextureEntry;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShapeVertex;
pub(crate) use crate::resources::overlay::overlay_text::LabelGpuData;
pub(crate) use crate::resources::overlay::overlay_text::OverlayTextVertex;
pub(crate) use crate::resources::overlay::overlays::BackdropBlurState;
pub(crate) use crate::resources::overlay::overlays::GridUniform;
pub(crate) use crate::resources::overlay::overlays::GroundPlaneUniform;
pub(crate) use crate::resources::overlay::overlays::OverlayUniform;
pub use crate::resources::overlay::overlays::OverlayVertex;
pub(crate) use crate::resources::postprocess::lic::LicAdvectUniform;
pub(crate) use crate::resources::postprocess::lic::LicObjectUniform;
pub use crate::resources::postprocess::lic::LicSurfaceGpuData;
pub(crate) use crate::resources::postprocess::uniforms::AtlasBlitUniform;
pub(crate) use crate::resources::postprocess::uniforms::BloomUniform;
pub(crate) use crate::resources::postprocess::uniforms::ContactShadowUniform;
pub(crate) use crate::resources::postprocess::uniforms::DofUniform;
pub(crate) use crate::resources::postprocess::uniforms::ShadowAtlasUniform;
pub(crate) use crate::resources::postprocess::uniforms::SsaoUniform;
pub(crate) use crate::resources::postprocess::uniforms::ToneMapUniform;
pub(crate) use crate::resources::scivis::gaussian_splat::GaussianSplatDrawData;
pub(crate) use crate::resources::scivis::gaussian_splat::GaussianSplatStore;
pub(crate) use crate::resources::scivis::glyph::GlyphBaseMesh;
pub use crate::resources::scivis::glyph::GlyphGpuData;
pub use crate::resources::scivis::glyph::TensorGlyphGpuData;
pub use crate::resources::scivis::image::ScreenImageGpuData;
pub use crate::resources::scivis::point_cloud::PointCloudGpuData;
pub use crate::resources::scivis::polyline::PolylineGpuData;
pub use crate::resources::scivis::sprite::SpriteGpuData;
pub use crate::resources::scivis::tube::StreamtubeGpuData;
pub(crate) use crate::resources::volume::tetmesh::GpuProjectedTetMesh;
pub(crate) use crate::resources::volume::tetmesh::ProjectedTetChunk;
pub(crate) use crate::resources::volume::tetmesh::ProjectedTetUniform;
pub(crate) use crate::resources::volume::volumes::ImageSliceGpuData;
pub use crate::resources::volume::volumes::VolumeGpuData;
pub(crate) use crate::resources::volume::volumes::VolumeSurfaceSliceGpuData;

crate::resources::handle::slot_handle! {
    /// Identifies a 3D volume texture uploaded to the GPU.
    ///
    /// A slotted, generational handle: a volume can be replaced in place
    /// ([`replace_volume`](crate::resources::DeviceResources::replace_volume)) or
    /// freed ([`free_volume`](crate::resources::DeviceResources::free_volume)) so
    /// a time-series playback reuses one slot instead of leaking a 3D texture per
    /// timestep. A handle held across a free-then-reupload resolves to nothing
    /// rather than aliasing the new occupant.
    pub struct VolumeId;
}

crate::resources::handle::slot_handle! {
    /// Identifies a projected-tetrahedra mesh uploaded to the GPU for transparent
    /// volume rendering.
    ///
    /// A slotted, generational handle: the mesh can be refreshed in place
    /// ([`replace_projected_tet`](crate::resources::DeviceResources::replace_projected_tet),
    /// [`replace_projected_tet_scalar`](crate::resources::DeviceResources::replace_projected_tet_scalar))
    /// or freed ([`free_projected_tet`](crate::resources::DeviceResources::free_projected_tet))
    /// so a transparent time-series reuses one slot instead of leaking a tet
    /// buffer per fresh upload.
    pub struct ProjectedTetId;
}

/// Scalar attribute interpolation domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeKind {
    /// One value per vertex.
    Vertex,
    /// One value per triangle (cell). Averaged to vertices at upload time.
    Cell,
    /// One value per triangle. NOT averaged : rendered flat via vertex duplication.
    /// Colourmapped through the active LUT just like `Vertex`.
    Face,
    /// One RGBA colour per triangle. NOT averaged : rendered flat via vertex duplication.
    /// Bypasses the colourmap; the per-face colour is used directly.
    FaceColour,
    /// One value per directed triangle edge. `values[3*t + k]` is the scalar on the
    /// k-th edge of triangle `t` (edge from vertex `k` to vertex `(k+1)%3`).
    /// Averaged to the two endpoint vertices for rendering.
    Edge,
    /// One value per directed triangle edge (halfedge). `values[3*t + k]` is the
    /// scalar for the k-th halfedge of triangle `t`.
    /// Rendered flat per triangle corner via vertex duplication (like `Face`).
    Halfedge,
    /// One value per triangle corner. `values[3*t + k]` is the scalar at the
    /// k-th corner of triangle `t`.
    /// Rendered flat per triangle corner via vertex duplication (like `Face`).
    Corner,
}

/// Reference to a named scalar attribute on a mesh.
#[derive(Debug, Clone)]
pub struct AttributeRef {
    /// Name of the attribute as stored in `MeshData::attributes`.
    pub name: String,
    /// Whether the attribute is per-vertex, per-cell, or per-face.
    pub kind: AttributeKind,
}

/// Scalar data for a mesh attribute.
#[derive(Debug, Clone)]
pub enum AttributeData {
    /// One `f32` per vertex.
    Vertex(Vec<f32>),
    /// One `f32` per triangle (cell). Averaged to vertices at upload.
    Cell(Vec<f32>),
    /// One `f32` per triangle. Not averaged; stored in a non-indexed expanded buffer.
    Face(Vec<f32>),
    /// One `[r, g, b, a]` per triangle. Not averaged; stored in a non-indexed expanded buffer.
    FaceColour(Vec<[f32; 4]>),
    /// One `f32` per directed triangle edge. `values[3*t + k]` = k-th edge of triangle `t`.
    /// Averaged to the two endpoint vertices for rendering.
    Edge(Vec<f32>),
    /// One `f32` per directed triangle edge (halfedge). `values[3*t + k]` = k-th halfedge of
    /// triangle `t`. Rendered flat per corner via vertex duplication (like `Face`).
    Halfedge(Vec<f32>),
    /// One `f32` per triangle corner. `values[3*t + k]` = k-th corner of triangle `t`.
    /// Rendered flat per corner via vertex duplication (like `Face`).
    Corner(Vec<f32>),
    /// One `[x, y, z]` per vertex. Uploaded as a flat `array<f32>` storage buffer (3 floats
    /// per vertex) for use in per-vertex vector field rendering (e.g. Surface LIC).
    VertexVector(Vec<[f32; 3]>),
}

/// Built-in colourmap presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinColourmap {
    /// Viridis : perceptually uniform, colourblind-friendly (purple -> teal -> yellow).
    Viridis = 0,
    /// Plasma : perceptually uniform, colourblind-friendly (blue -> pink -> yellow).
    Plasma = 1,
    /// Greyscale : linear black->white.
    Greyscale = 2,
    /// Coolwarm : diverging blue->white->red.
    Coolwarm = 3,
    /// Rainbow : HSV hue sweep 240 deg->0 deg.
    Rainbow = 4,
    /// Magma : perceptually uniform (black -> purple -> orange -> near-white).
    Magma = 5,
    /// Inferno : perceptually uniform (black -> deep-red -> orange -> light-yellow).
    Inferno = 6,
    /// Turbo : improved rainbow (Google 2019). Better perceptual uniformity than Jet.
    Turbo = 7,
    /// Jet : classic blue-cyan-green-yellow-red. Widely used in engineering.
    Jet = 8,
    /// RdBu : diverging blue->white->red (blue at t=0, red at t=1).
    RdBu = 9,
}

// ---------------------------------------------------------------------------
// Vertex and uniform structs (bytemuck::Pod for GPU buffer casting)
// ---------------------------------------------------------------------------

/// Per-vertex data: position, normal, base colour, UV coordinates, tangent.
///
/// Layout (64 bytes, 4-byte aligned):
/// - position: [f32; 3]  : offset  0, 12 bytes
/// - normal:   [f32; 3]  : offset 12, 12 bytes
/// - colour:    [f32; 4]  : offset 24, 16 bytes
/// - uv:       [f32; 2]  : offset 40,  8 bytes
/// - tangent:  [f32; 4]  : offset 48, 16 bytes  (xyz=tangent direction, w=handedness +/-1)
///
/// `tangent.w` is the bitangent handedness. Reconstruct bitangent as:
/// `bitangent = cross(normal, tangent.xyz) * tangent.w`
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Local-space vertex position (shader location 0).
    pub position: [f32; 3],
    /// Vertex normal in local space (shader location 1).
    pub normal: [f32; 3],
    /// Vertex colour RGBA in linear 0..1 (shader location 2).
    pub colour: [f32; 4],
    /// UV texture coordinates (shader location 3).
    pub uv: [f32; 2],
    /// Tangent vector [tx, ty, tz, handedness] (shader location 4). `w` is +/-1.
    pub tangent: [f32; 4],
}

impl Vertex {
    /// wgpu vertex buffer layout matching shader locations 0, 1, 2, 3, 4.
    pub fn buffer_layout() -> crate::gpu::VertexBufferLayout<'static> {
        crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position (vec3f) : offset 0
                crate::gpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: crate::gpu::VertexFormat::Float32x3,
                },
                // location 1: normal (vec3f) : offset 12
                crate::gpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: crate::gpu::VertexFormat::Float32x3,
                },
                // location 2: colour (vec4f) : offset 24
                crate::gpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 3: uv (vec2f) : offset 40
                crate::gpu::VertexAttribute {
                    offset: 40,
                    shader_location: 3,
                    format: crate::gpu::VertexFormat::Float32x2,
                },
                // location 4: tangent (vec4f) : offset 48
                crate::gpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Default shadow atlas resolution (width = height). The atlas is a 2x2 grid of tiles.
pub(crate) const SHADOW_ATLAS_SIZE: u32 = 4096;

/// Per-frame camera uniform: view-projection and eye position.
///
/// GPU uniform struct : must remain 16-byte aligned (`#[repr(C)]`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// Combined view-projection matrix (column-major).
    pub view_proj: [[f32; 4]; 4],
    /// Camera eye position in world space (used for specular lighting).
    pub eye_pos: [f32; 3],
    /// Upper bound applied to the lit (pre-emissive) mesh colour.
    ///
    /// 1.0 on the LDR path, so lit output stays normalised as before. On the
    /// HDR path this is F16_MAX, letting bright direct light and ambient stay
    /// above 1.0 into the Rgba16Float target so the tone mapper receives real
    /// HDR (emissive was already unclamped on both paths). Occupies what was
    /// alignment padding, so the struct layout is unchanged.
    pub lit_clamp: f32,
    /// Camera forward direction in world space (used for view-depth tests).
    pub forward: [f32; 3],
    /// Padding to align `forward` to 16 bytes.
    pub _pad1: f32,
    /// Inverse view-projection matrix (for reconstructing world-space rays, e.g. skybox).
    pub inv_view_proj: [[f32; 4]; 4],
    /// View matrix (world -> camera space, column-major).
    ///
    /// Used by the matcap shader to transform world-space normals to view space.
    pub view: [[f32; 4]; 4],
}

/// GPU-side per-light uniform (one entry in the `LightsUniform` array).
///
/// Layout (144 bytes, 16-byte aligned):
/// - light_view_proj:  `[[f32; 4]; 4]` = 64 bytes  (shadow matrix; only used for `lights[0]`)
/// - pos_or_dir:        `[f32; 3]`     = 12 bytes  (direction for directional, position for point/spot)
/// - light_type:        u32            =  4 bytes  (0=directional, 1=point, 2=spot)
/// - colour:             `[f32; 3]`     = 12 bytes
/// - intensity:         f32        =  4 bytes
/// - range:             f32        =  4 bytes
/// - inner_angle:       f32        =  4 bytes
/// - outer_angle:       f32        =  4 bytes
/// - _pad_align:        u32        =  4 bytes  (bridge to 16-byte boundary for spot_direction)
/// - spot_direction:    `[f32; 3]` = 12 bytes  (spot only; at offset 112 to match WGSL vec3 align)
/// - _pad:              `[f32; 5]` = 20 bytes  (tail padding to 144)
/// Total: 64+12+4+12+4+4+4+4+4+12+20 = 144 bytes
///
/// Note: WGSL `vec3<f32>` has AlignOf=16, so `spot_direction` must start at offset 112.
/// The `_pad_align` field bridges the 4-byte gap between offset 108 and 112.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SingleLightUniform {
    /// Light-space view-projection matrix for shadow map rendering.
    pub light_view_proj: [[f32; 4]; 4], // 64 bytes, offset   0
    /// World-space position (point/spot) or direction (directional) of the light.
    pub pos_or_dir: [f32; 3], // 12 bytes, offset  64
    /// Light type discriminant: 0=directional, 1=point, 2=spot.
    pub light_type: u32, //  4 bytes, offset  76
    /// Linear RGB colour of the light emission.
    pub colour: [f32; 3], // 12 bytes, offset  80
    /// Luminous intensity multiplier.
    pub intensity: f32, //  4 bytes, offset  92
    /// Maximum attenuation range (world units) for point/spot lights.
    pub range: f32, //  4 bytes, offset  96
    /// Cosine of the inner cone half-angle (spot lights).
    pub inner_angle: f32, //  4 bytes, offset 100
    /// Cosine of the outer cone half-angle (spot lights; smooth falloff edge).
    pub outer_angle: f32, //  4 bytes, offset 104
    /// Alignment padding : bridges 4-byte gap to align `spot_direction` to offset 112.
    pub _pad_align: u32, //  4 bytes, offset 108 (aligns spot_direction to 112)
    /// World-space unit direction the spot cone points toward.
    pub spot_direction: [f32; 3], // 12 bytes, offset 112
    /// Cubemap slot index for point-shadow sampling, or -1 if this light
    /// does not have a slot allocated this frame. Only meaningful when
    /// `light_type == 1` (point).
    pub point_shadow_slot: i32, //  4 bytes, offset 124
    /// Near plane for the point-shadow perspective projection (world units).
    /// Used by the lit pass to map fragment distance back into the
    /// normalised [0, 1] depth that was written into the cubemap.
    pub point_shadow_near: f32, //  4 bytes, offset 128
    /// Tail padding to reach 144-byte struct size.
    pub _pad: [f32; 3], // 12 bytes, offset 132 : total 144
}

/// GPU-side lights header uniform (binding 3 of group 0).
///
/// Per-light data lives in a separate storage buffer at binding 13
/// (`light_storage_buf`); this header carries only the global parameters
/// and the active light count. Sized so the structure cost stays well
/// inside the uniform-buffer budget regardless of how many lights are
/// active in a given frame.
///
/// Layout:
/// - count:                u32            =  4 bytes
/// - shadow_bias:          f32            =  4 bytes
/// - shadows_enabled:      u32            =  4 bytes
/// - debug_vis_mode:       u32            =  4 bytes
/// - sky_colour:            `[f32; 3]`    = 12 bytes
/// - hemisphere_intensity: f32            =  4 bytes
/// - ground_colour:         `[f32; 3]`    = 12 bytes
/// - debug_vis_scale:      f32            =  4 bytes
/// - ibl_enabled:          u32            =  4 bytes
/// - ibl_intensity:        f32            =  4 bytes
/// - ibl_rotation:         f32            =  4 bytes
/// - show_skybox:          u32            =  4 bytes
/// - debug_vis_split_x:    f32            =  4 bytes
/// - _pad_dbg:             [u32; 3]       = 12 bytes
/// Total: 16 + 16 + 16 + 16 + 16 = 80 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightsUniform {
    /// Number of active lights. Indexed against `light_storage_buf`.
    pub count: u32, //  4 bytes
    /// Shadow bias applied to depth comparisons to reduce acne.
    pub shadow_bias: f32, //  4 bytes
    /// 1 = shadow maps enabled, 0 = disabled.
    pub shadows_enabled: u32, //  4 bytes
    /// Packed debug visualization channel selectors and active flag. 0 = debug off.
    pub debug_vis_mode: u32, //  4 bytes
    /// Sky hemisphere colour for ambient contribution.
    pub sky_colour: [f32; 3], // 12 bytes
    /// Hemisphere ambient intensity multiplier.
    pub hemisphere_intensity: f32, //  4 bytes
    /// Ground hemisphere colour for ambient contribution.
    pub ground_colour: [f32; 3], // 12 bytes
    /// Scale multiplier applied to debug quantity values before display.
    pub debug_vis_scale: f32, //  4 bytes
    /// 1 = IBL environment map is active, 0 = disabled.
    pub ibl_enabled: u32, // 4 bytes
    /// IBL intensity multiplier.
    pub ibl_intensity: f32, // 4 bytes
    /// IBL Y-axis rotation in radians.
    pub ibl_rotation: f32, // 4 bytes
    /// 1 = show skybox background, 0 = use background colour.
    pub show_skybox: u32, // 4 bytes
    /// Normalized split X position (0..1) for SplitScreen debug mode.
    pub debug_vis_split_x: f32, // 4 bytes
    /// Number of active environment-selection zones (binding 19). 0 = the single
    /// default environment (layer 0); the shaders skip the per-fragment zone loop.
    pub env_zone_count: u32, // 4 bytes
    /// Reserved for future debug uniform fields.
    pub _pad_dbg: [u32; 2], // 8 bytes
}

/// Maximum number of lights packed into `light_storage_buf` per frame.
///
/// When the union of `EffectsFrame::lighting.lights` and
/// `SceneFrame::lights` exceeds this count, the renderer keeps the first
/// directional (which carries the cascaded shadow map) and ranks the rest
/// by `importance * proximity_weight`, dropping the tail. Bumping this
/// only costs `MAX_SCENE_LIGHTS * 144` bytes of GPU storage.
pub const MAX_SCENE_LIGHTS: usize = 512;

/// Alias kept for backward compatibility : existing app code imports `LightUniform`.
pub type LightUniform = LightsUniform;

/// One plane of the view frustum as uploaded to the GPU cull shader.
///
/// Matches `FrustumPlane` in `cull.wgsl` (16 bytes: vec3 + f32).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FrustumPlane {
    pub(crate) normal: [f32; 3],
    pub(crate) distance: f32,
}

/// Six-plane frustum uniform uploaded to the GPU cull pass.
///
/// Matches `FrustumUniform` in `cull.wgsl` (192 bytes).
/// Planes are in Gribb-Hartmann order: left, right, bottom, top, near, far.
/// Each plane normal points inward; `distance` is the signed offset from origin
/// along the normal (`d` in the CPU `Plane` struct).
///
/// `instance_count` and `batch_count` are the valid element counts in the
/// AABB and batch-meta storage buffers, respectively. The cull shaders use
/// these instead of `arrayLength()` to avoid processing stale data that
/// remains in oversized (2x headroom) buffers after scene shrinks.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FrustumUniform {
    pub(crate) planes: [FrustumPlane; 6],
    /// Valid instance count in the AABB buffer (not the buffer capacity).
    pub(crate) instance_count: u32,
    /// Valid batch count in the batch-meta buffer (not the buffer capacity).
    pub(crate) batch_count: u32,
    /// 1 = shadow cull dispatch (instances with `cast_shadows=0` are skipped),
    /// 0 = main cull dispatch (every visible instance participates).
    pub(crate) shadow_pass: u32,
    /// 1 = run the HiZ occlusion test after the frustum test, 0 = skip it.
    pub(crate) do_occlusion: u32,
    /// Camera view-projection for projecting AABBs to screen. Identity when
    /// occlusion is off.
    pub(crate) view_proj: [[f32; 4]; 4],
    /// HiZ mip-0 dimensions in pixels (the depth target the pyramid was built
    /// from).
    pub(crate) viewport: [f32; 2],
    pub(crate) _pad0: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<FrustumUniform>() == 192);

/// Clip planes uniform for section-view clipping (binding 4 of camera bind group).
///
/// Layout (112 bytes):
/// - planes: [[f32;4];6] = 96 bytes  (normal.xyz in .xyz, signed distance in .w)
/// - count:  u32         =  4 bytes
/// - _pad0:  u32         =  4 bytes
/// - viewport_width:  f32 = 4 bytes  (used by the outline shader for pixel expansion)
/// - viewport_height: f32 = 4 bytes
/// Total: 112 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ClipPlanesUniform {
    pub(crate) planes: [[f32; 4]; 6], //  96 bytes
    pub(crate) count: u32,            //   4 bytes
    pub(crate) _pad0: u32,            //   4 bytes
    pub(crate) viewport_width: f32,   //   4 bytes
    pub(crate) viewport_height: f32,  //   4 bytes
}

/// Maximum number of `Box` / `Sphere` clip volumes that can be active simultaneously.
///
/// Corresponds to the array size in [`ClipVolumesUniform`] and the WGSL `ClipVolumeUB` struct.
/// Planes are not counted against this limit; they have a separate cap of 6.
pub const CLIP_VOLUME_MAX: usize = 4;

/// One entry in the clip-volume uniform array : 96 bytes.
///
/// `volume_type` selects the active shape:
/// - 0 = None (slot unused)
/// - 2 = Box (oriented AABB : center, half-extents, three orientation columns)
/// - 3 = Sphere (center + radius packed into `center` and `radius`)
///
/// Layout mirrors the WGSL `ClipVolumeEntry` struct in each geometry shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClipVolumeEntry {
    /// Shape discriminant: 0=None, 2=Box, 3=Sphere.
    pub volume_type: u32,
    #[doc(hidden)]
    pub _pad: [u32; 3],
    /// Box center (world space) or sphere center. Occupies the xyz components.
    pub center: [f32; 3],
    /// Sphere radius. Unused for boxes.
    pub radius: f32,
    /// Box half-extents. Unused for spheres.
    pub half_extents: [f32; 3],
    #[doc(hidden)]
    pub _pad1: f32,
    /// Box local X axis (orientation column 0, world space). Unused for spheres.
    pub col0: [f32; 3],
    #[doc(hidden)]
    pub _pad2: f32,
    /// Box local Y axis (orientation column 1, world space). Unused for spheres.
    pub col1: [f32; 3],
    #[doc(hidden)]
    pub _pad3: f32,
    /// Box local Z axis (orientation column 2, world space). Unused for spheres.
    pub col2: [f32; 3],
    #[doc(hidden)]
    pub _pad4: f32,
}
// 16 + 16 + 16 + 16 + 16 + 16 = 96 bytes

impl ClipVolumeEntry {
    /// Build a box entry from center, half-extents, and orientation columns.
    pub fn from_box(center: [f32; 3], half_extents: [f32; 3], orientation: [[f32; 3]; 3]) -> Self {
        Self {
            volume_type: 2,
            _pad: [0; 3],
            center,
            radius: 0.0,
            half_extents,
            _pad1: 0.0,
            col0: orientation[0],
            _pad2: 0.0,
            col1: orientation[1],
            _pad3: 0.0,
            col2: orientation[2],
            _pad4: 0.0,
        }
    }

    /// Build a sphere entry from center and radius.
    pub fn from_sphere(center: [f32; 3], radius: f32) -> Self {
        Self {
            volume_type: 3,
            _pad: [0; 3],
            center,
            radius,
            ..bytemuck::Zeroable::zeroed()
        }
    }

    /// Build a cylinder entry from center, unit axis, radius, and half-length.
    ///
    /// The cylinder extends `half_length` units in each direction along `axis`
    /// from `center`. `axis` must be a unit vector.
    pub fn from_cylinder(center: [f32; 3], axis: [f32; 3], radius: f32, half_length: f32) -> Self {
        Self {
            volume_type: 4,
            _pad: [0; 3],
            center,
            radius,
            half_extents: [half_length, 0.0, 0.0],
            _pad1: 0.0,
            col0: axis,
            ..bytemuck::Zeroable::zeroed()
        }
    }
}

/// Clip volume uniform array : bound at group 0 binding 6.
///
/// Holds up to [`CLIP_VOLUME_MAX`] active `Box`, `Sphere`, or `Cylinder` clip volumes.
/// `count` is the number of valid entries; unused slots have `volume_type = 0`.
///
/// The layout mirrors the WGSL `ClipVolumeUB` struct in each geometry shader.
/// Total size: 16 (header) + 4 * 96 (entries) = 400 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClipVolumesUniform {
    /// Number of active entries in `volumes`.
    pub count: u32,
    #[doc(hidden)]
    pub _pad: [u32; 3],
    /// Clip volume entries. Only the first `count` slots are valid.
    pub volumes: [ClipVolumeEntry; CLIP_VOLUME_MAX],
}

/// Per-frame GPU data for one mesh-instance batch, created in `prepare()`.
///
/// Mesh instances reuse the scene-graph `mesh_instanced` pipeline family but
/// run through a host-built per-batch instance buffer rather than the global
/// scene instance array. Each batch is one draw call.
pub struct MeshInstanceGpuData {
    /// Mesh handle used to look up the vertex / index buffers at draw time.
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Number of instances in this batch (= draw instance count).
    pub(crate) instance_count: u32,
    /// Bind group (group 1): instance storage buf + albedo / sampler / normal / AO.
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Blend mode selected by the host. Drives pipeline selection at draw time.
    pub(crate) blend: crate::renderer::SpriteBlend,
    // Keep buffers alive for the lifetime of this struct.
    pub(crate) _instance_buf: crate::gpu::Buffer,
}
