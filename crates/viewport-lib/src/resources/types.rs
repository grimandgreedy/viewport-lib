pub use crate::resources::device_resources::ContentResources;
pub use crate::resources::device_resources::DeviceResources;
pub use crate::resources::device_resources::DualPipeline;
pub(crate) use crate::resources::device_resources::OutlineResources;
pub(crate) use crate::resources::device_resources::PickResources;
pub(crate) use crate::resources::device_resources::ProjectedTetResources;
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
pub use crate::resources::material::textures::{
    TextureData, TexturePayload, TextureRequirement, TextureRole,
};
pub use crate::resources::memory::ResidentBytes;
pub use crate::resources::memory::TextureMemoryStats;
pub use crate::resources::memory::VramBudget;
pub use crate::resources::mesh::gpu_mesh::GpuMesh;
pub use crate::resources::mesh::instancing::BatchMeta;
pub(crate) use crate::resources::mesh::instancing::InstanceAabb;
pub(crate) use crate::resources::mesh::instancing::InstanceData;
pub use crate::resources::mesh::instancing::PickInstance;
pub use crate::resources::mesh::meshes::{MeshData, SubmeshRange};
pub use crate::resources::overlay::highlight::OutlineEdgeUniform;
pub(crate) use crate::resources::overlay::highlight::OutlineObjectBuffers;
pub(crate) use crate::resources::overlay::highlight::OutlineUniform;
pub(crate) use crate::resources::overlay::highlight::SubHighlightGpuData;
pub(crate) use crate::resources::overlay::overlay_shape::ClipShapeGpu;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayBackdropVertex;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShadowLayerGpu;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShapeGpuData;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShapeInstance;
pub(crate) use crate::resources::overlay::overlay_shape::OverlayShapeTextureEntry;
pub(crate) use crate::resources::overlay::overlay_text::LabelGpuData;
pub(crate) use crate::resources::overlay::overlay_text::OverlayTextVertex;
pub(crate) use crate::resources::overlay::overlays::BackdropBlurState;
pub(crate) use crate::resources::overlay::overlays::GridUniform;
pub(crate) use crate::resources::overlay::overlays::GroundPlaneUniform;
pub use crate::resources::overlay::overlays::OverlayVertex;
pub(crate) use crate::resources::postprocess::uniforms::AtlasBlitUniform;
pub(crate) use crate::resources::postprocess::uniforms::BloomUniform;
pub(crate) use crate::resources::postprocess::uniforms::ContactShadowUniform;
pub(crate) use crate::resources::postprocess::uniforms::DofUniform;
pub(crate) use crate::resources::postprocess::uniforms::ShadowAtlasUniform;
pub(crate) use crate::resources::postprocess::uniforms::SsaoUniform;
pub(crate) use crate::resources::postprocess::uniforms::ToneMapUniform;
pub(crate) use crate::resources::volume::tetmesh::GpuProjectedTetMesh;
pub(crate) use crate::resources::volume::tetmesh::ProjectedTetChunk;
pub(crate) use crate::resources::volume::tetmesh::ProjectedTetUniform;

pub use viewport_lib_types::ids::{ProjectedTetId, VolumeId};

pub use viewport_lib_types::data::attribute::{AttributeData, AttributeKind, AttributeRef};

pub use viewport_lib_types::colourmap::{BuiltinColourmap, UnknownColourmap};

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

/// Supplies the wgpu vertex buffer layout for a vertex type.
///
/// Kept off [`Vertex`] itself as an extension trait so the vertex data type
/// carries no `wgpu` type: the layout is a property of how the GPU consumes the
/// struct, not of the struct's data. Bring it into scope to call
/// `Vertex::buffer_layout()`.
pub trait VertexBufferLayoutExt {
    /// wgpu vertex buffer layout for this vertex type.
    fn buffer_layout() -> crate::gpu::VertexBufferLayout<'static>;
}

impl VertexBufferLayoutExt for Vertex {
    /// wgpu vertex buffer layout matching shader locations 0, 1, 2, 3, 4.
    fn buffer_layout() -> crate::gpu::VertexBufferLayout<'static> {
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
    /// Upper bound applied to the lit (pre-emissive) mesh colour, which is
    /// exposed radiance.
    ///
    /// 1.0 on the LDR path, where the target cannot hold more. On the HDR path
    /// this is F16_MAX, so bright direct light and ambient stay above 1.0 in
    /// the Rgba16Float target and the tone mapper receives real HDR.
    pub lit_clamp: f32,
    /// Camera forward direction in world space (used for view-depth tests).
    pub forward: [f32; 3],
    /// The camera's exposure: the linear multiplier the shaders apply to every
    /// radiance source where they read it (lights, environment, hemisphere
    /// fill, probes, lightmaps, emissive). 1.0 under
    /// `ExposureMode::Passthrough`; under `ExposureMode::Automatic` the HDR
    /// path copies the adapted value in on the GPU before the scene draws.
    pub exposure: f32,
    /// Inverse view-projection matrix (for reconstructing world-space rays, e.g. skybox).
    pub inv_view_proj: [[f32; 4]; 4],
    /// View matrix (world -> camera space, column-major).
    ///
    /// Used by the matcap shader to transform world-space normals to view space.
    pub view: [[f32; 4]; 4],
    /// Projection matrix (view -> clip, column-major).
    pub proj: [[f32; 4]; 4],
    /// Inverse projection matrix.
    pub inv_proj: [[f32; 4]; 4],
    /// Viewport size in physical pixels: width, height, 1 / width, 1 / height.
    pub viewport: [f32; 4],
    /// Near plane, far plane, 1 for an orthographic projection (0 for
    /// perspective), and the device pixels per point of the target the scene
    /// draws into (0 for a camera uniform no viewport wrote).
    pub clip: [f32; 4],
    /// Frame time: seconds as f32, the remainder f32 drops from the f64
    /// seconds, delta seconds, and the frame index modulo 2^24.
    pub time: [f32; 4],
    /// The world origin every position above is relative to, rounded to f32.
    /// Too coarse far from zero to rebuild a position with; for procedural
    /// effects that want world coordinates.
    pub world_origin: [f32; 3],
    /// Bumped each time `world_origin` moves.
    pub origin_epoch: u32,
}

/// Byte offset of [`CameraUniform::exposure`], where the HDR path copies an
/// automatic exposure in.
pub(crate) const CAMERA_EXPOSURE_OFFSET: u64 = std::mem::offset_of!(CameraUniform, exposure) as u64;
const _: () = assert!(CAMERA_EXPOSURE_OFFSET == 92);

/// GPU-side per-light uniform (one entry in the `LightsUniform` array).
///
/// Layout (144 bytes, 16-byte aligned):
/// - light_view_proj:  `[[f32; 4]; 4]` = 64 bytes  (shadow matrix; a spot light's, when it has a layer)
/// - pos_or_dir:        `[f32; 3]`     = 12 bytes  (direction for directional, position for point/spot)
/// - light_type:        u32            =  4 bytes  (0=directional, 1=point, 2=spot)
/// - colour:             `[f32; 3]`     = 12 bytes
/// - intensity:         f32        =  4 bytes
/// - range:             f32        =  4 bytes
/// - inner_angle:       f32        =  4 bytes
/// - outer_angle:       f32        =  4 bytes
/// - shadow_kind:       u32        =  4 bytes  (0 none, 1 cascades, 2 point layers, 3 spot layer)
/// - spot_direction:    `[f32; 3]` = 12 bytes  (spot only; at offset 112 to match WGSL vec3 align)
/// - shadow_layer:      i32        =  4 bytes  (base punctual shadow layer, -1 none)
/// - point_shadow_near: f32        =  4 bytes
/// - radius:            f32        =  4 bytes
/// - channel_mask:      u32        =  4 bytes  (layer mask; AND-tested per object)
/// - _reserved:         u32        =  4 bytes  (reserved tail to 144; see the field)
/// Total: 64+12+4+12+4+4+4+4+4+12+20 = 144 bytes
///
/// Note: WGSL `vec3<f32>` has AlignOf=16, so `spot_direction` must start at offset 112.
/// `shadow_kind` fills the 4 bytes between offset 108 and 112.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SingleLightUniform {
    /// Light-space view-projection matrix of a spot light's shadow layer.
    /// Identity for every other light.
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
    /// Which shadow this light samples: 0 none, 1 the cascade atlas, 2 a point
    /// light's six layers of the punctual shadow array, 3 a spot light's one
    /// layer. Also keeps `spot_direction` at offset 112.
    pub shadow_kind: u32, //  4 bytes, offset 108
    /// World-space unit direction the spot cone points toward.
    pub spot_direction: [f32; 3], // 12 bytes, offset 112
    /// Base layer of this light in the punctual shadow array, or -1 when it
    /// has none this frame: a point light's six cube faces start here, a spot
    /// light's map is this layer.
    pub shadow_layer: i32, //  4 bytes, offset 124
    /// Near plane of the punctual shadow projection (world units).
    pub point_shadow_near: f32, //  4 bytes, offset 128
    /// Source radius (world units) for point/spot lights: clamps the
    /// inverse-square falloff near the light and sizes a finite emitter.
    pub radius: f32, //  4 bytes, offset 132
    /// Layer mask this light illuminates (offset 136). AND-tested in the lit
    /// mesh shaders against each object's mask; a light whose bits do not
    /// intersect the object's mask is skipped for that fragment. Wired from
    /// `LightSource::channel_mask`; `!0` (the default) lights everything.
    pub channel_mask: u32, //  4 bytes, offset 136
    /// Reserved tail (one lane, offset 140, to a 144-byte struct). Earmarked for
    /// one of the remaining light extensions: an area-light size word or a
    /// cookie/IES handle (only one lane is left after `channel_mask`, so the
    /// other rides a parallel per-light buffer when it lands). Unread today and
    /// uploaded as 0.
    pub _reserved: u32, //  4 bytes, offset 140 : total 144
}

const _: () = {
    assert!(std::mem::size_of::<SingleLightUniform>() == 144);
    assert!(std::mem::offset_of!(SingleLightUniform, shadow_kind) == 108);
    assert!(std::mem::offset_of!(SingleLightUniform, spot_direction) == 112);
    assert!(std::mem::offset_of!(SingleLightUniform, shadow_layer) == 124);
};

/// GPU-side lights header uniform (binding 3 of group 0).
///
/// Per-light data lives in a separate storage buffer at binding 13
/// (`light_storage_buf`); this header carries only the global parameters
/// and the active light count. Sized so the structure cost stays well
/// inside the uniform-buffer budget regardless of how many lights are
/// active in a given frame.
///
/// One header serves every viewport of a scene, so its intensities are
/// absolute (unexposed); a shader multiplies each by the viewport's
/// `CameraUniform::exposure` where it reads it.
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
/// - ibl_diffuse_scale:    f32            =  4 bytes
/// - debug_vis_split_x:    f32            =  4 bytes
/// - env_zone_count:       u32            =  4 bytes
/// - ibl_specular_scale:   f32            =  4 bytes
/// - custom_data_base:     u32            =  4 bytes
/// - ao_flags:             u32            =  4 bytes
/// - _pad_ao:              `[u32; 3]`     = 12 bytes
/// Total: 16 + 16 + 16 + 16 + 16 + 16 = 96 bytes
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
    /// IBL rotation about +Z in radians.
    pub ibl_rotation: f32, // 4 bytes
    /// Scale on the IBL diffuse term.
    pub ibl_diffuse_scale: f32, // 4 bytes
    /// Normalized split X position (0..1) for SplitScreen debug mode.
    pub debug_vis_split_x: f32, // 4 bytes
    /// Number of active environment-selection zones (binding 19). 0 = the single
    /// default environment (layer 0); the shaders skip the per-fragment zone loop.
    pub env_zone_count: u32, // 4 bytes
    /// Scale on the IBL specular term.
    pub ibl_specular_scale: f32, // 4 bytes
    /// Start of the custom-data region of the binding-18 buffer, in `vec4`
    /// elements.
    pub custom_data_base: u32, // 4 bytes
    /// Material AO options: bit 0 multi-bounce, bit 1 micro-shadowing of
    /// directional lights.
    pub ao_flags: u32, // 4 bytes
    /// Padding to a 16-byte multiple.
    pub _pad_ao: [u32; 3], // 12 bytes
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
    /// Camera layer mask. The cull kernel rejects an instance when its
    /// per-object mask (read from the instance buffer at cull binding 8) shares
    /// no bit with this. Only consulted when `do_mask_cull == 1`; wired from
    /// `CameraFrame::cull_mask`.
    pub(crate) cull_mask: u32,
    /// 1 = run the per-camera layer-mask reject (main cull, real instance buffer
    /// bound), 0 = skip it (shadow / single-mesh / plugin dispatches bind the
    /// fallback instance buffer and leave this off).
    pub(crate) do_mask_cull: u32,
    /// 1 = the order-free compaction runs after the cull kernel and will write
    /// both the visible list and the per-batch visible counts, so the cull
    /// kernel skips its own arrival-order list and counter increments. 0 = the
    /// compaction scratch this submission needs is past the device's
    /// storage-buffer binding limit, so the cull kernel writes the list itself,
    /// in arrival order.
    pub(crate) compact_enabled: u32,
    /// Size of the chunk-plan region of the compaction scratch, in u32s
    /// (`batch_count + 1`). The shader lays the rest of the buffer out after it.
    pub(crate) plan_cap: u32,
    /// Size of the per-chunk totals region of the compaction scratch, in u32s.
    pub(crate) chunk_cap: u32,
    /// Where, in u32s, the visibility buffer's words region starts: the cull
    /// copies each survivor's record words there, six to a visible slot, for
    /// the culled mesh draws to fetch per instance. 0 = no words region (the
    /// shadow and plugin culls).
    pub(crate) words_base: u32,
}

const _: () = assert!(std::mem::size_of::<FrustumUniform>() == 208);

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
/// - 2 = Box (oriented AABB : centre, half-extents, three orientation columns)
/// - 3 = Sphere (centre + radius packed into `centre` and `radius`)
///
/// Layout mirrors the WGSL `ClipVolumeEntry` struct in each geometry shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClipVolumeEntry {
    /// Shape discriminant: 0=None, 2=Box, 3=Sphere.
    pub volume_type: u32,
    #[doc(hidden)]
    pub _pad: [u32; 3],
    /// Box centre (world space) or sphere centre. Occupies the xyz components.
    pub centre: [f32; 3],
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
    /// Build a box entry from centre, half-extents, and orientation columns.
    pub fn from_box(centre: [f32; 3], half_extents: [f32; 3], orientation: [[f32; 3]; 3]) -> Self {
        Self {
            volume_type: 2,
            _pad: [0; 3],
            centre,
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

    /// Build a sphere entry from centre and radius.
    pub fn from_sphere(centre: [f32; 3], radius: f32) -> Self {
        Self {
            volume_type: 3,
            _pad: [0; 3],
            centre,
            radius,
            ..bytemuck::Zeroable::zeroed()
        }
    }

    /// Build a cylinder entry from centre, unit axis, radius, and half-length.
    ///
    /// The cylinder extends `half_length` units in each direction along `axis`
    /// from `centre`. `axis` must be a unit vector.
    pub fn from_cylinder(centre: [f32; 3], axis: [f32; 3], radius: f32, half_length: f32) -> Self {
        Self {
            volume_type: 4,
            _pad: [0; 3],
            centre,
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
/// Mesh instances draw through the HDR mesh family's blended members, from a
/// host-built per-batch instance buffer rather than the global scene
/// instance array. Each batch is one draw call.
pub struct MeshInstanceGpuData {
    /// Mesh handle used to look up the vertex / index buffers at draw time.
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Number of instances in this batch (= draw instance count).
    pub(crate) instance_count: u32,
    /// Bind group (group 1): the mesh group around `instance_buf`.
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Blend mode selected by the host. Drives pipeline selection at draw time.
    pub(crate) blend: crate::renderer::SpriteBlend,
    /// The batch's instance records, bound at group 1 binding 0 and as
    /// vertex buffer slot 1.
    pub(crate) instance_buf: crate::gpu::Buffer,
    /// The source item's painter placement: such a batch draws in the
    /// painter sequence rather than the scene pass.
    pub(crate) painter: Option<crate::PainterPlacement>,
    /// Index of the source item in `SceneFrame::mesh_instances`.
    pub(crate) item_index: usize,
}
