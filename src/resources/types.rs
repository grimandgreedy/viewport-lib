/// GPU resources for the 3D viewport.
///
/// Owns the render pipelines (solid + wireframe), vertex/index buffers,
/// uniform buffers, and bind groups.

// ---------------------------------------------------------------------------
// MeshData : CPU-side mesh representation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scalar attribute types
// ---------------------------------------------------------------------------

/// Identifies a colormap (LUT) uploaded to the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColormapId(pub usize);

/// Identifies a matcap texture uploaded to the GPU.
///
/// Obtained from [`ViewportGpuResources::upload_matcap`] or
/// [`ViewportGpuResources::builtin_matcap_id`].
/// The `blendable` flag controls whether the alpha channel tints the base
/// geometry color (`true`) or the matcap fully replaces the object color (`false`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatcapId {
    /// Index into the GPU matcap texture store.
    pub(crate) index: usize,
    /// Whether the alpha channel blends with base geometry color.
    pub blendable: bool,
}

/// Built-in matcap presets bundled with viewport-lib.
///
/// Pass to [`ViewportGpuResources::builtin_matcap_id`] after the renderer
/// has been prepared for at least one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinMatcap {
    /// Warm orange-brown with soft top-left lighting.  Blendable.
    Clay = 0,
    /// Peach tone with wide soft specular, skin-like.  Blendable.
    Wax = 1,
    /// Vivid hue-cycling sphere, colorful.  Blendable.
    Candy = 2,
    /// Neutral gray Lambertian shading.  Blendable.
    Flat = 3,
    /// Clean white with sharp specular highlight.  Static.
    Ceramic = 4,
    /// Deep translucent green stone.  Static.
    Jade = 5,
    /// Dark brownish rough surface.  Static.
    Mud = 6,
    /// View-space normal visualization (R=nx, G=ny, B=nz).  Static.
    Normal = 7,
}

/// Identifies a 3D volume texture uploaded to the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VolumeId(pub(crate) usize);

/// Scalar attribute interpolation domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeKind {
    /// One value per vertex.
    Vertex,
    /// One value per triangle (cell). Averaged to vertices at upload time.
    Cell,
    /// One value per triangle. NOT averaged : rendered flat via vertex duplication.
    /// Colormapped through the active LUT just like `Vertex`.
    Face,
    /// One RGBA color per triangle. NOT averaged : rendered flat via vertex duplication.
    /// Bypasses the colormap; the per-face color is used directly.
    FaceColor,
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
    FaceColor(Vec<[f32; 4]>),
}

/// Built-in colormap presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinColormap {
    /// Viridis : perceptually uniform.
    Viridis = 0,
    /// Plasma : perceptually uniform, brighter.
    Plasma = 1,
    /// Greyscale : linear black->white.
    Greyscale = 2,
    /// Coolwarm : diverging blue->white->red.
    Coolwarm = 3,
    /// Rainbow : HSV hue sweep 240°->0°.
    Rainbow = 4,
}

/// Raw mesh data for upload to the GPU. Framework-agnostic representation.
#[non_exhaustive]
pub struct MeshData {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex normals (must be the same length as `positions`).
    pub normals: Vec<[f32; 3]>,
    /// Triangle index list (every 3 indices form one triangle).
    pub indices: Vec<u32>,
    /// Optional per-vertex UV coordinates. `None` means zero-fill [0.0, 0.0].
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Optional per-vertex tangents [tx, ty, tz, w] where w is handedness (±1.0).
    ///
    /// `None` = auto-compute from UVs if available, or zero-fill otherwise.
    /// Tangents are required for correct normal map rendering.
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Named scalar attributes for per-vertex or per-cell scalar field visualisation.
    ///
    /// Keys are user-defined attribute names (e.g. `"pressure"`, `"velocity_mag"`).
    /// Cell attributes are averaged to vertices at upload time.
    pub attributes: std::collections::HashMap<String, AttributeData>,
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            uvs: None,
            tangents: None,
            attributes: std::collections::HashMap::new(),
        }
    }
}

impl MeshData {
    /// Compute the local-space AABB from vertex positions.
    pub fn compute_aabb(&self) -> crate::scene::aabb::Aabb {
        crate::scene::aabb::Aabb::from_positions(&self.positions)
    }
}

// ---------------------------------------------------------------------------
// Vertex and uniform structs (bytemuck::Pod for GPU buffer casting)
// ---------------------------------------------------------------------------

/// Per-vertex data: position, normal, base color, UV coordinates, tangent.
///
/// Layout (64 bytes, 4-byte aligned):
/// - position: [f32; 3]  : offset  0, 12 bytes
/// - normal:   [f32; 3]  : offset 12, 12 bytes
/// - color:    [f32; 4]  : offset 24, 16 bytes
/// - uv:       [f32; 2]  : offset 40,  8 bytes
/// - tangent:  [f32; 4]  : offset 48, 16 bytes  (xyz=tangent direction, w=handedness ±1)
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
    /// Vertex color RGBA in linear 0..1 (shader location 2).
    pub color: [f32; 4],
    /// UV texture coordinates (shader location 3).
    pub uv: [f32; 2],
    /// Tangent vector [tx, ty, tz, handedness] (shader location 4). `w` is ±1.
    pub tangent: [f32; 4],
}

impl Vertex {
    /// wgpu vertex buffer layout matching shader locations 0, 1, 2, 3, 4.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position (vec3f) : offset 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 1: normal (vec3f) : offset 12
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 2: color (vec4f) : offset 24
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 3: uv (vec2f) : offset 40
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 4: tangent (vec4f) : offset 48
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Default shadow atlas resolution (width = height). The atlas is a 2×2 grid of tiles.
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
    /// Padding to align `eye_pos` to 16 bytes.
    pub _pad: f32,
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
/// - color:             `[f32; 3]`     = 12 bytes
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
    /// Linear RGB color of the light emission.
    pub color: [f32; 3], // 12 bytes, offset  80
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
    /// Tail padding to reach 144-byte struct size.
    pub _pad: [f32; 5], // 20 bytes, offset 124 : total 144
}

/// GPU-side lights uniform (binding 3 of group 0). Supports up to 8 light sources.
///
/// Layout:
/// - count:                u32            =  4 bytes
/// - shadow_bias:          f32            =  4 bytes
/// - shadows_enabled:      u32            =  4 bytes
/// - _pad:                 u32            =  4 bytes (align header to 16)
/// - sky_color:            `[f32; 3]`     = 12 bytes
/// - hemisphere_intensity: f32            =  4 bytes
/// - ground_color:         `[f32; 3]`     = 12 bytes
/// - _pad2:                f32            =  4 bytes (align to 16)
/// - lights:               [SingleLightUniform; 8] = 8 * 144 = 1152 bytes
/// Total: 16 + 16 + 16 + 1152 = 1200 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightsUniform {
    /// Number of active lights (0–8).
    pub count: u32, //  4 bytes
    /// Shadow bias applied to depth comparisons to reduce acne.
    pub shadow_bias: f32, //  4 bytes
    /// 1 = shadow maps enabled, 0 = disabled.
    pub shadows_enabled: u32, //  4 bytes
    /// Alignment padding.
    pub _pad: u32, //  4 bytes
    /// Sky hemisphere color for ambient contribution.
    pub sky_color: [f32; 3], // 12 bytes
    /// Hemisphere ambient intensity multiplier.
    pub hemisphere_intensity: f32, //  4 bytes
    /// Ground hemisphere color for ambient contribution.
    pub ground_color: [f32; 3], // 12 bytes
    /// Alignment padding.
    pub _pad2: f32, //  4 bytes
    /// Per-light parameters (up to 8 lights).
    pub lights: [SingleLightUniform; 8], // 8 * 144 = 1152 bytes
    /// 1 = IBL environment map is active, 0 = disabled.
    pub ibl_enabled: u32, // 4 bytes
    /// IBL intensity multiplier.
    pub ibl_intensity: f32, // 4 bytes
    /// IBL Y-axis rotation in radians.
    pub ibl_rotation: f32, // 4 bytes
    /// 1 = show skybox background, 0 = use background color.
    pub show_skybox: u32, // 4 bytes
}

/// Alias kept for backward compatibility : existing app code imports `LightUniform`.
pub type LightUniform = LightsUniform;

/// Per-object uniform: world transform, material properties, selection state, and wireframe mode.
///
/// Layout (208 bytes, 16-byte aligned):
/// - model:           [[f32;4];4] = 64 bytes  offset   0
/// - color:            [f32;4]   = 16 bytes  offset  64  (base_color.xyz + opacity)
/// - selected:          u32      =  4 bytes  offset  80
/// - wireframe:         u32      =  4 bytes  offset  84
/// - ambient:           f32      =  4 bytes  offset  88
/// - diffuse:           f32      =  4 bytes  offset  92
/// - specular:          f32      =  4 bytes  offset  96
/// - shininess:         f32      =  4 bytes  offset 100
/// - has_texture:       u32      =  4 bytes  offset 104
/// - use_pbr:           u32      =  4 bytes  offset 108
/// - metallic:          f32      =  4 bytes  offset 112
/// - roughness:         f32      =  4 bytes  offset 116
/// - has_normal_map:    u32      =  4 bytes  offset 120
/// - has_ao_map:        u32      =  4 bytes  offset 124
/// - has_attribute:     u32      =  4 bytes  offset 128
/// - scalar_min:        f32      =  4 bytes  offset 132
/// - scalar_max:        f32      =  4 bytes  offset 136
/// - _pad_scalar:       u32      =  4 bytes  offset 140
/// - nan_color:        [f32;4]   = 16 bytes  offset 144
/// - use_nan_color:     u32      =  4 bytes  offset 160
/// - use_matcap:        u32      =  4 bytes  offset 164
/// - matcap_blendable:  u32      =  4 bytes  offset 168
/// - _pad2:             u32      =  4 bytes  offset 172
/// - use_face_color:    u32      =  4 bytes  offset 176
/// - uv_vis_mode:       u32      =  4 bytes  offset 180  (0=off 1=checker 2=grid 3=localcheck 4=localrad)
/// - uv_vis_scale:      f32      =  4 bytes  offset 184
/// - backface_policy:   u32      =  4 bytes  offset 188  (0=Cull 1=Identical 2=DifferentColor)
/// - backface_color:   [f32;4]   = 16 bytes  offset 192
/// Total: 208 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ObjectUniform {
    pub(crate) model: [[f32; 4]; 4],     //  64 bytes, offset   0
    pub(crate) color: [f32; 4],          //  16 bytes, offset  64
    pub(crate) selected: u32,            //   4 bytes, offset  80
    pub(crate) wireframe: u32,           //   4 bytes, offset  84
    pub(crate) ambient: f32,             //   4 bytes, offset  88
    pub(crate) diffuse: f32,             //   4 bytes, offset  92
    pub(crate) specular: f32,            //   4 bytes, offset  96
    pub(crate) shininess: f32,           //   4 bytes, offset 100
    pub(crate) has_texture: u32,         //   4 bytes, offset 104
    pub(crate) use_pbr: u32,             //   4 bytes, offset 108
    pub(crate) metallic: f32,            //   4 bytes, offset 112
    pub(crate) roughness: f32,           //   4 bytes, offset 116
    pub(crate) has_normal_map: u32,      //   4 bytes, offset 120
    pub(crate) has_ao_map: u32,          //   4 bytes, offset 124
    pub(crate) has_attribute: u32,       //   4 bytes, offset 128
    pub(crate) scalar_min: f32,          //   4 bytes, offset 132
    pub(crate) scalar_max: f32,          //   4 bytes, offset 136
    pub(crate) _pad_scalar: u32,         //   4 bytes, offset 140
    pub(crate) nan_color: [f32; 4],      //  16 bytes, offset 144
    pub(crate) use_nan_color: u32,       //   4 bytes, offset 160
    pub(crate) use_matcap: u32,          //   4 bytes, offset 164
    pub(crate) matcap_blendable: u32,    //   4 bytes, offset 168
    pub(crate) _pad2: u32,               //   4 bytes, offset 172
    pub(crate) use_face_color: u32,      //   4 bytes, offset 176
    pub(crate) uv_vis_mode: u32,         //   4 bytes, offset 180
    pub(crate) uv_vis_scale: f32,        //   4 bytes, offset 184
    pub(crate) backface_policy: u32, //   4 bytes, offset 188  (0=Cull 1=Identical 2=DifferentColor)
    pub(crate) backface_color: [f32; 4], //  16 bytes, offset 192
}

const _: () = assert!(std::mem::size_of::<ObjectUniform>() == 208);

/// Per-instance GPU data for instanced rendering. Matches the WGSL `InstanceData` struct.
///
/// Layout mirrors ObjectUniform (128 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceData {
    pub(crate) model: [[f32; 4]; 4], //  64 bytes, offset   0
    pub(crate) color: [f32; 4],      //  16 bytes, offset  64
    pub(crate) selected: u32,        //   4 bytes, offset  80
    pub(crate) wireframe: u32,       //   4 bytes, offset  84
    pub(crate) ambient: f32,         //   4 bytes, offset  88
    pub(crate) diffuse: f32,         //   4 bytes, offset  92
    pub(crate) specular: f32,        //   4 bytes, offset  96
    pub(crate) shininess: f32,       //   4 bytes, offset 100
    pub(crate) has_texture: u32,     //   4 bytes, offset 104
    pub(crate) use_pbr: u32,         //   4 bytes, offset 108
    pub(crate) metallic: f32,        //   4 bytes, offset 112
    pub(crate) roughness: f32,       //   4 bytes, offset 116
    pub(crate) has_normal_map: u32,  //   4 bytes, offset 120
    pub(crate) has_ao_map: u32,      //   4 bytes, offset 124
}

/// Per-instance GPU data for the object-ID pick pass (Phase K).
///
/// Stores only the model matrix and a sentinel object ID : none of the material
/// fields needed by the full [`InstanceData`] struct.
///
/// Layout (80 bytes):
/// - model_c0..model_c3: vec4<f32> × 4 = 64 bytes (model matrix, column-major)
/// - object_id: u32                     =  4 bytes  (sentinel: scene_items_index + 1)
/// - _pad: [u32; 3]                     = 12 bytes  (align to 16)
/// Total: 80 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PickInstance {
    pub(crate) model_c0: [f32; 4],
    pub(crate) model_c1: [f32; 4],
    pub(crate) model_c2: [f32; 4],
    pub(crate) model_c3: [f32; 4],
    pub(crate) object_id: u32,
    pub(crate) _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<PickInstance>() == 80);

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

/// Clip volume uniform : 128 bytes, bound at group 0 binding 6.
///
/// Exported for testing (size validation) and for downstream crates that may
/// need to construct the uniform directly (e.g. headless compute tools).
///
/// `volume_type` discriminant selects which clip test is applied in each shader:
/// - 0 = None (always passes, zero overhead via early return)
/// - 1 = Plane (half-space: `dot(p, normal) + dist >= 0`)
/// - 2 = Box (oriented AABB expressed via rotation columns + half-extents)
/// - 3 = Sphere (`distance(p, center) <= radius`)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClipVolumeUniform {
    /// Discriminant: 0=None, 1=Plane, 2=Box, 3=Sphere.
    pub volume_type: u32,
    /// Padding to 16-byte alignment.
    pub _pad0: [u32; 3],
    // Plane params
    /// Plane half-space normal (world space).
    pub plane_normal: [f32; 3],
    /// Plane signed distance from origin.
    pub plane_dist: f32,
    // Box params
    /// Box center in world space.
    pub box_center: [f32; 3],
    /// Padding to maintain 16-byte alignment for the next field.
    pub _pad1: f32,
    /// Box half-extents.
    pub box_half_extents: [f32; 3],
    /// Padding to maintain 16-byte alignment for the next field.
    pub _pad2: f32,
    /// Box local X axis (orientation column 0) in world space.
    pub box_col0: [f32; 3],
    /// Padding to maintain 16-byte alignment for the next field.
    pub _pad3: f32,
    /// Box local Y axis (orientation column 1) in world space.
    pub box_col1: [f32; 3],
    /// Padding to maintain 16-byte alignment for the next field.
    pub _pad4: f32,
    /// Box local Z axis (orientation column 2) in world space.
    pub box_col2: [f32; 3],
    /// Padding to maintain 16-byte alignment for the next field.
    pub _pad5: f32,
    // Sphere params
    /// Sphere center in world space.
    pub sphere_center: [f32; 3],
    /// Sphere radius.
    pub sphere_radius: f32,
}
// Total: 4 + 12 + 4*4 + 4*4 + 4*4 + 4*4 + 4*4 + 4*4 = 16 + 7*16 = 16 + 112 = 128 bytes

impl ClipVolumeUniform {
    /// Build a `ClipVolumeUniform` from a [`crate::renderer::ClipShape`] value.
    /// Returns a zeroed (None / volume_type=0) uniform for `ClipShape::Plane`.
    pub fn from_clip_shape(shape: &crate::renderer::ClipShape) -> Self {
        let mut u: Self = bytemuck::Zeroable::zeroed();
        match shape {
            crate::renderer::ClipShape::Plane {
                normal, distance, ..
            } => {
                u.volume_type = 1;
                u.plane_normal = *normal;
                u.plane_dist = *distance;
            }
            crate::renderer::ClipShape::Box {
                center,
                half_extents,
                orientation,
            } => {
                u.volume_type = 2;
                u.box_center = *center;
                u.box_half_extents = *half_extents;
                u.box_col0 = orientation[0];
                u.box_col1 = orientation[1];
                u.box_col2 = orientation[2];
            }
            crate::renderer::ClipShape::Sphere { center, radius } => {
                u.volume_type = 3;
                u.sphere_center = *center;
                u.sphere_radius = *radius;
            }
        }
        u
    }
}

/// Per-object outline uniform for the two-pass stencil outline effect.
///
/// Layout (96 bytes):
/// - model:        [[f32;4];4] = 64 bytes
/// - color:         [f32;4]   = 16 bytes  (outline RGBA)
/// - pixel_offset:  f32       =  4 bytes  (outline ring width in pixels)
/// - _pad:          [f32;3]   = 12 bytes
/// Total: 96 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OutlineUniform {
    pub(crate) model: [[f32; 4]; 4], //  64 bytes
    pub(crate) color: [f32; 4],      //  16 bytes
    pub(crate) pixel_offset: f32,    //   4 bytes
    pub(crate) _pad: [f32; 3],       //  12 bytes
}

pub(crate) struct OutlineObjectBuffers {
    pub mesh_index: usize,
    pub two_sided: bool,
    pub _mask_uniform_buf: wgpu::Buffer,
    pub mask_bind_group: wgpu::BindGroup,
}

/// Uniform for the fullscreen outline edge-detection pass (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OutlineEdgeUniform {
    pub(crate) color: [f32; 4],      // 16 bytes
    pub(crate) radius: f32,          //  4 bytes
    pub(crate) viewport_w: f32,      //  4 bytes
    pub(crate) viewport_h: f32,      //  4 bytes
    pub(crate) _pad: f32,            //  4 bytes
}

/// Tone mapping uniform.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ToneMapUniform {
    pub(crate) exposure: f32,
    pub(crate) mode: u32, // 0=Reinhard, 1=ACES, 2=KhronosNeutral
    pub(crate) bloom_enabled: u32,
    pub(crate) ssao_enabled: u32,
    pub(crate) contact_shadows_enabled: u32,
    pub(crate) _pad_tm: [u32; 3],
    pub(crate) background_color: [f32; 4],
}

/// Bloom pass uniform (16 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BloomUniform {
    pub(crate) threshold: f32,
    pub(crate) intensity: f32,
    pub(crate) horizontal: u32, // 1=horizontal pass, 0=vertical
    pub(crate) _pad: u32,
}

/// SSAO uniform (144 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SsaoUniform {
    pub(crate) inv_proj: [[f32; 4]; 4], // 64 bytes : NDC->view
    pub(crate) proj: [[f32; 4]; 4],     // 64 bytes : view->clip (for re-projection)
    pub(crate) radius: f32,
    pub(crate) bias: f32,
    pub(crate) _pad: [f32; 2],
}

/// Shadow atlas uniform (416 bytes, bound at group 0 binding 5).
///
/// Contains per-cascade view-projection matrices, split distances, atlas layout,
/// and PCSS parameters. Used by the fragment shader for CSM cascade selection
/// and shadow sampling.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ShadowAtlasUniform {
    /// 4 cascade view-projection matrices (each 64 bytes = 4×4 f32). Flattened.
    pub(crate) cascade_view_proj: [[f32; 4]; 16], // 256 bytes
    /// Distance-based split values for cascade selection (eye-to-fragment distance).
    pub(crate) cascade_splits: [f32; 4], //  16 bytes
    /// Number of active cascades (1–4).
    pub(crate) cascade_count: u32, //   4 bytes
    /// Atlas texture size in pixels (e.g. 4096.0).
    pub(crate) atlas_size: f32, //   4 bytes
    /// Shadow filter mode: 0=PCF, 1=PCSS.
    pub(crate) shadow_filter: u32, //   4 bytes
    /// PCSS light source radius in UV space.
    pub(crate) pcss_light_radius: f32, //   4 bytes
    /// Per-slot atlas UV rects: [uv_min.x, uv_min.y, uv_max.x, uv_max.y] × 8 slots.
    pub(crate) atlas_rects: [[f32; 4]; 8], // 128 bytes
}
// Total: 256 + 16 + 16 + 128 = 416 bytes

/// Contact shadow uniform (176 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ContactShadowUniform {
    pub(crate) inv_proj: [[f32; 4]; 4],  // 64 bytes
    pub(crate) proj: [[f32; 4]; 4],      // 64 bytes
    pub(crate) light_dir_view: [f32; 4], // 16 bytes
    pub(crate) world_up_view: [f32; 4],  // 16 bytes
    pub(crate) params: [f32; 4],         // 16 bytes: [max_distance, steps, thickness, pad]
}

/// Per-vertex data for overlay rendering: position only (no normal/color in vertex).
///
/// Color is provided via the OverlayUniform rather than per-vertex to keep
/// the buffer minimal : all vertices of a single overlay quad share the same color.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayVertex {
    /// World-space XYZ position of this overlay vertex.
    pub position: [f32; 3],
}

impl OverlayVertex {
    /// wgpu vertex buffer layout matching shader location 0 (position vec3f).
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

/// Per-overlay uniform: model matrix and RGBA color with alpha for transparency.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayUniform {
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) color: [f32; 4], // RGBA with alpha for transparency
}

/// Uniform buffer layout for the full-screen ground plane shader.
///
/// Matches `GroundPlaneUniform` in `ground_plane.wgsl` exactly (256 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GroundPlaneUniform {
    pub view_proj: [[f32; 4]; 4], // offset   0, 64 bytes
    pub cam_right: [f32; 4],      // offset  64, 16 bytes
    pub cam_up: [f32; 4],         // offset  80, 16 bytes
    pub cam_back: [f32; 4],       // offset  96, 16 bytes
    pub eye_pos: [f32; 3],        // offset 112, 12 bytes
    pub height: f32,              // offset 124,  4 bytes
    pub color: [f32; 4],          // offset 128, 16 bytes
    pub shadow_color: [f32; 4],   // offset 144, 16 bytes
    pub light_vp: [[f32; 4]; 4],  // offset 160, 64 bytes
    pub tan_half_fov: f32,        // offset 224,  4 bytes
    pub aspect: f32,              // offset 228,  4 bytes
    pub tile_size: f32,           // offset 232,  4 bytes
    pub shadow_bias: f32,         // offset 236,  4 bytes
    pub mode: u32,                // offset 240,  4 bytes
    pub shadow_opacity: f32,      // offset 244,  4 bytes
    pub _pad: [f32; 2],           // offset 248,  8 bytes
} // total  256 bytes

/// Uniform buffer layout for the full-screen analytical grid shader.
///
/// Contains all data needed by `grid.wgsl`: camera matrices for ray unprojection,
/// eye position, grid plane height, spacing for minor/major lines, and RGBA colors.
/// Total size: 192 bytes (fits in one 256-byte UBO slot).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GridUniform {
    /// Combined view-projection matrix for computing clip-space depth of grid hits.
    pub view_proj: [[f32; 4]; 4], // offset   0, 64 bytes
    /// Camera-to-world rotation matrix (3 columns as vec4 with w=0 padding, matching
    /// WGSL mat3x3<f32> layout). Col 0 = right, Col 1 = up, Col 2 = back (camera +Z).
    /// Used to rotate the analytical camera-space ray direction into world space,
    /// bypassing the ill-conditioned inv(view_proj) at large camera distances.
    pub cam_to_world: [[f32; 4]; 3], // offset  64, 48 bytes
    /// tan(fov_y / 2) : scales NDC x/y to camera-space ray direction.
    pub tan_half_fov: f32, // offset 112,  4 bytes
    /// Viewport aspect ratio (width / height).
    pub aspect: f32, // offset 116,  4 bytes
    /// Padding to keep snap_origin at offset 152 (8-byte aligned).
    pub _pad_ivp: [f32; 2], // offset 120,  8 bytes
    /// Eye (camera) position in world space.
    pub eye_pos: [f32; 3], // offset 128, 12 bytes
    /// Z-coordinate of the horizontal grid plane (Z-up, XY ground plane).
    pub grid_z: f32, // offset 140,  4 bytes
    /// Minor grid line spacing (world units).
    pub spacing_minor: f32, // offset 144,  4 bytes
    /// Major grid line spacing (world units, typically spacing_minor * 10).
    pub spacing_major: f32, // offset 148,  4 bytes
    /// XZ origin used to keep `hit.xz - snap_origin` small for f32 precision.
    /// Set to `floor(eye.xz / spacing_major) * spacing_major` each frame.
    pub snap_origin: [f32; 2], // offset 152,  8 bytes
    /// RGBA color for minor grid lines.
    pub color_minor: [f32; 4], // offset 160, 16 bytes
    /// RGBA color for major grid lines.
    pub color_major: [f32; 4], // offset 176, 16 bytes
                               // Total: 192 bytes
}

// ---------------------------------------------------------------------------
// GpuTexture: GPU texture with sampler and bind group
// ---------------------------------------------------------------------------

/// A GPU texture with its view, sampler, and bind group for shader binding.
pub struct GpuTexture {
    /// Underlying wgpu texture object.
    pub texture: wgpu::Texture,
    /// Full-texture view used for sampling.
    pub view: wgpu::TextureView,
    /// Sampler bound alongside the view.
    pub sampler: wgpu::Sampler,
    /// Bind group that binds `view` and `sampler` for use in shaders.
    pub bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// GpuMesh: per-object GPU buffers
// ---------------------------------------------------------------------------

/// GPU buffers and bind group for a single mesh.
pub struct GpuMesh {
    /// Interleaved position + normal vertex buffer (Vertex layout).
    pub vertex_buffer: wgpu::Buffer,
    /// Triangle index buffer (for solid rendering).
    pub index_buffer: wgpu::Buffer,
    /// Number of indices in the triangle index buffer.
    pub index_count: u32,
    /// Edge index buffer (deduplicated pairs, for wireframe LineList rendering).
    pub edge_index_buffer: wgpu::Buffer,
    /// Number of indices in the edge index buffer.
    pub edge_index_count: u32,
    /// Vertex buffer for per-vertex normal visualization lines (LineList topology).
    /// Each normal contributes two vertices: the vertex position and position + normal * 0.1.
    /// None if no normal data is available.
    pub normal_line_buffer: Option<wgpu::Buffer>,
    /// Number of vertices in the normal line buffer (2 per normal line).
    pub normal_line_count: u32,
    /// Per-object uniform buffer (model matrix, material, selection state).
    pub object_uniform_buf: wgpu::Buffer,
    /// Bind group (group 1) combining `object_uniform_buf` with texture views.
    /// Texture views are the fallback 1×1 textures by default; rebuilt when material
    /// texture assignment changes (tracked via `last_tex_key`).
    pub object_bind_group: wgpu::BindGroup,
    /// Last texture/attribute key `(albedo_id, normal_map_id, ao_map_id, lut_id, attr_name_hash, matcap_id)`
    /// used to build `object_bind_group`. `u64::MAX` = fallback / none for that slot.
    pub(crate) last_tex_key: (u64, u64, u64, u64, u64, u64),
    /// Per-named-attribute GPU storage buffers (f32 per vertex, STORAGE usage).
    pub attribute_buffers: std::collections::HashMap<String, wgpu::Buffer>,
    /// Scalar range `(min, max)` per attribute, computed at upload time.
    pub attribute_ranges: std::collections::HashMap<String, (f32, f32)>,
    /// Non-indexed vertex buffer containing 3×N expanded vertices for face-attribute rendering.
    /// `None` if no `Face` or `FaceColor` attributes exist for this mesh.
    pub face_vertex_buffer: Option<wgpu::Buffer>,
    /// Named face scalar buffers: 3N `f32` entries (value replicated for all 3 vertices of each tri).
    pub face_attribute_buffers: std::collections::HashMap<String, wgpu::Buffer>,
    /// Named face color buffers: 3N `[f32; 4]` entries (color replicated for all 3 vertices of each tri).
    pub face_color_buffers: std::collections::HashMap<String, wgpu::Buffer>,
    /// Uniform buffer for normal-line rendering: always has selected=0, wireframe=0.
    /// Updated each frame in prepare() with the object's model matrix only.
    pub normal_uniform_buf: wgpu::Buffer,
    /// Bind group referencing `normal_uniform_buf` : used when drawing normal lines.
    pub normal_bind_group: wgpu::BindGroup,
    /// Local-space axis-aligned bounding box computed from vertex positions at upload time.
    pub aabb: crate::scene::aabb::Aabb,
    /// CPU-side positions retained for cap geometry generation (clip plane cross-section fill).
    pub(crate) cpu_positions: Option<Vec<[f32; 3]>>,
    /// CPU-side triangle indices retained for cap geometry generation.
    pub(crate) cpu_indices: Option<Vec<u32>>,
}

// ---------------------------------------------------------------------------
// SciVis Phase B : GPU data types for point cloud and glyph renderers
// ---------------------------------------------------------------------------

/// Cached GPU vertex + index buffers for a glyph base mesh (arrow, sphere, cube).
pub(crate) struct GlyphBaseMesh {
    /// Vertex buffer using the full `Vertex` layout (64 bytes stride).
    pub vertex_buffer: wgpu::Buffer,
    /// Triangle index buffer.
    pub index_buffer: wgpu::Buffer,
    /// Number of indices.
    pub index_count: u32,
}

/// Per-frame GPU data for one point cloud item, created in `prepare()`.
pub struct PointCloudGpuData {
    /// Vertex buffer: one entry per point, packed as `[position: vec3, _pad: f32]` (16 bytes).
    /// The shader reads color/scalar from storage buffers indexed by `vertex_index`.
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Number of points (= draw count).
    pub(crate) point_count: u32,
    /// Bind group (group 1): uniform + LUT texture + sampler + scalar buf + color buf.
    pub(crate) bind_group: wgpu::BindGroup,
    // Keep the buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: wgpu::Buffer,
    pub(crate) _scalar_buf: wgpu::Buffer,
    pub(crate) _color_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one polyline item, created in `prepare()`.
pub struct PolylineGpuData {
    /// Instance buffer: `[xa, ya, za, xb, yb, zb, scalar_a, scalar_b]` per segment (32 bytes).
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Number of line segments (instances).  Draw call: `draw(0..6, 0..segment_count)`.
    pub(crate) segment_count: u32,
    /// Bind group (group 1): uniform + LUT texture + sampler.
    pub(crate) bind_group: wgpu::BindGroup,
    // Keep the uniform buffer alive for the lifetime of this struct.
    pub(crate) _uniform_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one screen-space image overlay, created in `prepare()` (Phase 10B).
pub struct ScreenImageGpuData {
    /// Uniform buffer: `ScreenImageUniform` (32 bytes) with NDC extents and alpha.
    pub(crate) uniform_buf: wgpu::Buffer,
    /// Uploaded RGBA8 texture for this image (recreated each frame).
    pub(crate) _texture: wgpu::Texture,
    /// Bind group (group 0): uniform + texture + sampler.
    pub(crate) bind_group: wgpu::BindGroup,
}

/// Per-frame GPU data for one glyph item, created in `prepare()`.
pub struct GlyphGpuData {
    /// Vertex buffer for the glyph base mesh (borrowed from cached `GlyphBaseMesh`).
    /// We keep a reference via raw pointer : `ViewportGpuResources` owns the mesh.
    /// Safety: the mesh lives as long as `ViewportGpuResources`.
    pub(crate) mesh_vertex_buffer: &'static wgpu::Buffer,
    /// Index buffer for the glyph base mesh.
    pub(crate) mesh_index_buffer: &'static wgpu::Buffer,
    /// Number of mesh indices.
    pub(crate) mesh_index_count: u32,
    /// Number of glyph instances.
    pub(crate) instance_count: u32,
    /// Bind group (group 1): glyph uniform + LUT texture + sampler.
    pub(crate) uniform_bind_group: wgpu::BindGroup,
    /// Bind group (group 2): instance storage buffer.
    pub(crate) instance_bind_group: wgpu::BindGroup,
    // Keep the buffers alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
    pub(crate) _instance_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one streamtube item, created in `prepare()`.
///
/// The connected tube mesh (vertices + indices) is generated CPU-side for the
/// entire item (all strips) and uploaded as a single owned buffer pair.
pub struct StreamtubeGpuData {
    /// Owned vertex buffer for the connected tube mesh (world-space positions + normals).
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Owned index buffer for the connected tube mesh.
    pub(crate) index_buffer: wgpu::Buffer,
    /// Number of indices to draw.
    pub(crate) index_count: u32,
    /// Bind group (group 1): tube uniform (color, radius).
    pub(crate) uniform_bind_group: wgpu::BindGroup,
    // Keep uniform buffer alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one volume item, created in `prepare()`.
pub struct VolumeGpuData {
    /// Bind group (group 1): volume uniform + 3D texture + sampler + color LUT + opacity LUT.
    pub(crate) bind_group: wgpu::BindGroup,
    /// Vertex buffer for the unit cube bounding box proxy.
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Index buffer for the unit cube (36 indices).
    pub(crate) index_buffer: wgpu::Buffer,
    /// Grid dimensions (stored for reference).
    pub(crate) _dims: [u32; 3],
    // Keep the uniform buffer alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
}

// ---------------------------------------------------------------------------
// ViewportHdrState: per-viewport HDR/post-process render target bundle
// ---------------------------------------------------------------------------

/// Per-viewport HDR/post-process GPU state.
///
/// Holds all viewport-size-dependent render targets, their associated bind
/// groups, and the per-viewport uniform buffers used by the post-process
/// pipeline.  Created lazily in `ViewportRenderer::ensure_viewport_hdr` and
/// resized automatically when the viewport dimensions change.
///
/// Shared infrastructure (pipelines, BGLs, samplers, placeholder textures,
/// SSAO noise/kernel) lives on [`ViewportGpuResources`] and is created once
/// by `ensure_hdr_shared`.
#[allow(dead_code)]
pub(crate) struct ViewportHdrState {
    // --- HDR scene target ---
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub hdr_depth_texture: wgpu::Texture,
    pub hdr_depth_view: wgpu::TextureView,
    pub hdr_depth_only_view: wgpu::TextureView,

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

    // --- Contact shadow ---
    pub contact_shadow_texture: wgpu::Texture,
    pub contact_shadow_view: wgpu::TextureView,

    // --- FXAA ---
    pub fxaa_texture: wgpu::Texture,
    pub fxaa_view: wgpu::TextureView,

    // --- SSAA (allocated when ssaa_factor > 1) ---
    /// Supersampled color render target. `None` when ssaa_factor == 1.
    pub ssaa_color_texture: Option<wgpu::Texture>,
    pub ssaa_color_view: Option<wgpu::TextureView>,
    /// Supersampled depth render target. `None` when ssaa_factor == 1.
    pub ssaa_depth_texture: Option<wgpu::Texture>,
    pub ssaa_depth_view: Option<wgpu::TextureView>,
    /// Bind group for the SSAA resolve pass (reads ssaa_color_texture). `None` when ssaa_factor == 1.
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
    pub outline_color_texture: wgpu::Texture,
    pub outline_color_view: wgpu::TextureView,
    pub outline_depth_texture: wgpu::Texture,
    pub outline_depth_view: wgpu::TextureView,
    /// Bind group for the edge-detection pass (reads mask, writes to color).
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

    /// Current [width, height] of all size-dependent textures.
    pub size: [u32; 2],
}

// ---------------------------------------------------------------------------
// ViewportGpuResources: top-level GPU resource container
// ---------------------------------------------------------------------------

/// All GPU resources for the 3D viewport.
///
/// Typically stored in the host framework's resource container and accessed
/// by `ViewportRenderer` during prepare() and paint().
pub struct ViewportGpuResources {
    /// Swapchain texture format; all pipelines are compiled for this format.
    pub target_format: wgpu::TextureFormat,
    /// MSAA sample count used by all render pipelines.
    pub sample_count: u32,
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
    /// Uniform buffer holding the per-frame `LightsUniform` (up to 8 lights + shadow info).
    pub light_uniform_buf: wgpu::Buffer,
    /// Bind group (group 0) binding camera, light, clip-plane, and shadow uniforms.
    pub camera_bind_group: wgpu::BindGroup,
    /// Bind group layout for group 0 (shared by all scene pipelines).
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for group 1 (per-object uniform: model, material, selection).
    pub object_bind_group_layout: wgpu::BindGroupLayout,
    /// Scene meshes (slotted storage with free-list removal).
    pub(crate) mesh_store: crate::resources::mesh_store::MeshStore,

    // --- Shadow map resources ---
    /// Shadow atlas depth texture (Depth32Float, atlas_size × atlas_size, 2×2 tile grid).
    pub shadow_map_texture: wgpu::Texture,
    /// Depth texture view for binding as a shader resource (sampling).
    pub shadow_map_view: wgpu::TextureView,
    /// Comparison sampler for PCF shadow filtering.
    pub shadow_sampler: wgpu::Sampler,
    /// Render pipeline for the shadow depth pass (depth-only, no fragment output).
    pub shadow_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer holding the per-cascade light-space view-projection matrix (64 bytes).
    pub shadow_uniform_buf: wgpu::Buffer,
    /// Bind group for the shadow pass (group 0: light uniform).
    pub shadow_bind_group: wgpu::BindGroup,
    /// Uniform buffer for the ShadowAtlasUniform (binding 5 of camera_bgl, 416 bytes).
    pub shadow_info_buf: wgpu::Buffer,
    /// Current shadow atlas texture size. Used to detect when atlas needs recreation.
    #[allow(dead_code)]
    pub(crate) shadow_atlas_size: u32,

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
    /// Bind group layout for overlay uniforms (group 1: model + color uniform).
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
    /// Fallback 1×1 white texture used when material.texture_id is None.
    pub fallback_texture: GpuTexture,
    /// Fallback 1×1 flat normal map [128,128,255,255] (tangent-space neutral).
    pub(crate) fallback_normal_map: wgpu::Texture,
    pub(crate) fallback_normal_map_view: wgpu::TextureView,
    /// Fallback 1×1 AO map [255,255,255,255] (no occlusion).
    pub(crate) fallback_ao_map: wgpu::Texture,
    pub(crate) fallback_ao_map_view: wgpu::TextureView,
    /// Shared linear-repeat sampler for material textures.
    pub(crate) material_sampler: wgpu::Sampler,
    /// Cache of material bind groups keyed by (albedo_id, normal_map_id, ao_map_id).
    /// u64::MAX sentinel = use fallback texture for that slot.
    #[allow(dead_code)]
    pub(crate) material_bind_groups: std::collections::HashMap<(u64, u64, u64), wgpu::BindGroup>,
    /// User-uploaded textures, indexed by `texture_id` in Material.
    pub textures: Vec<GpuTexture>,

    // --- Matcap texture system ---
    /// Matcap textures (256×256 RGBA), indexed by `MatcapId::index`.
    pub(crate) matcap_textures: Vec<wgpu::Texture>,
    /// Texture views for each uploaded matcap.
    pub(crate) matcap_views: Vec<wgpu::TextureView>,
    /// Linear-clamp sampler shared by all matcap texture lookups.
    pub(crate) matcap_sampler: Option<wgpu::Sampler>,
    /// Fallback 1×1 white view bound to binding 7 when no matcap is active.
    pub(crate) fallback_matcap_view: Option<wgpu::TextureView>,
    /// Whether built-in matcaps have been uploaded to the GPU.
    pub(crate) matcaps_initialized: bool,
    /// `MatcapId` for each built-in preset, populated by `ensure_matcaps_initialized`.
    pub(crate) builtin_matcap_ids: Option<[MatcapId; 8]>,

    /// Whether fallback normal map / AO map pixels have been uploaded.
    pub(crate) fallback_textures_uploaded: bool,

    // --- FXAA resources ---
    pub(crate) fxaa_texture: Option<wgpu::Texture>,
    pub(crate) fxaa_view: Option<wgpu::TextureView>,
    pub(crate) fxaa_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) fxaa_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) ssaa_resolve_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) ssaa_resolve_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) fxaa_bind_group: Option<wgpu::BindGroup>,
    /// Linear-clamp sampler shared by the FXAA pass (stored for recreating per-viewport bind groups).
    pub(crate) fxaa_sampler: Option<wgpu::Sampler>,

    // --- Clip planes ---
    /// Uniform buffer for clip planes (binding 4 of camera bind group).
    pub(crate) clip_planes_uniform_buf: wgpu::Buffer,
    /// Uniform buffer for the extended clip volume (binding 6 of camera bind group, 128 bytes).
    pub(crate) clip_volume_uniform_buf: wgpu::Buffer,

    // --- Outline & x-ray resources ---
    /// Bind group layout for OutlineUniform (group 1 for outline mask/xray pipelines).
    pub(crate) outline_bind_group_layout: wgpu::BindGroupLayout,
    /// Mask-write pipeline: renders selected objects as r=1.0 to an R8 mask (backface culled).
    pub(crate) outline_mask_pipeline: wgpu::RenderPipeline,
    /// Two-sided mask-write pipeline for selected meshes rendered without face culling.
    pub(crate) outline_mask_two_sided_pipeline: wgpu::RenderPipeline,
    /// Fullscreen edge-detection pipeline: reads mask, outputs anti-aliased outline ring.
    pub(crate) outline_edge_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for the edge-detection pass (mask texture + sampler + uniform).
    pub(crate) outline_edge_bgl: wgpu::BindGroupLayout,
    /// X-ray pipeline: draws selected objects through occluders (depth_compare Always).
    pub(crate) xray_pipeline: wgpu::RenderPipeline,
    // --- Outline offscreen resources (lazily created) ---
    /// Offscreen RGBA texture the outline stencil pass renders into.
    pub(crate) outline_color_texture: Option<wgpu::Texture>,
    pub(crate) outline_color_view: Option<wgpu::TextureView>,
    /// Depth+stencil texture for the offscreen outline pass.
    pub(crate) outline_depth_texture: Option<wgpu::Texture>,
    pub(crate) outline_depth_view: Option<wgpu::TextureView>,
    /// Size of the current outline offscreen textures.
    pub(crate) outline_target_size: [u32; 2],
    /// Fullscreen composite pipeline for single-sample LDR targets.
    pub(crate) outline_composite_pipeline_single: Option<wgpu::RenderPipeline>,
    /// Fullscreen composite pipeline for main render passes that use the renderer sample count.
    pub(crate) outline_composite_pipeline_msaa: Option<wgpu::RenderPipeline>,
    /// Fullscreen composite pipeline for HDR (Rgba16Float) targets.
    pub(crate) outline_composite_pipeline_hdr: Option<wgpu::RenderPipeline>,
    pub(crate) outline_composite_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) outline_composite_bind_group: Option<wgpu::BindGroup>,
    pub(crate) outline_composite_sampler: Option<wgpu::Sampler>,

    // --- Instancing resources (lazily created) ---
    /// Bind group layout for the instanced storage buffer + textures (group 1).
    pub(crate) instance_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Storage buffer for per-instance data.
    pub(crate) instance_storage_buf: Option<wgpu::Buffer>,
    /// Current capacity (in number of instances) of the storage buffer.
    pub(crate) instance_storage_capacity: usize,
    /// Per-texture-key bind groups for the instanced path.
    ///
    /// Each entry combines the shared instance storage buffer (binding 0) with
    /// one specific texture combination (bindings 1-4). Keyed by
    /// (albedo_id, normal_map_id, ao_map_id) using u64::MAX for fallback slots.
    /// Invalidated when the storage buffer is resized.
    pub(crate) instance_bind_groups: std::collections::HashMap<(u64, u64, u64), wgpu::BindGroup>,
    /// Instanced solid render pipeline (TriangleList, opaque).
    pub(crate) solid_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced transparent render pipeline (TriangleList, alpha blending).
    pub(crate) transparent_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced shadow render pipeline (depth-only).
    pub(crate) shadow_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// Per-cascade uniform buffers for shadow_instanced_pipeline (64 bytes each, one mat4x4).
    pub(crate) shadow_instanced_cascade_bufs: [Option<wgpu::Buffer>; 4],
    /// Per-cascade bind groups for shadow_instanced_pipeline group 0.
    pub(crate) shadow_instanced_cascade_bgs: [Option<wgpu::BindGroup>; 4],

    // --- Post-processing shared infrastructure (BGLs / pipelines / samplers / static textures) ---
    // Viewport-sized textures, bind groups, and uniform buffers are stored in
    // per-viewport ViewportHdrState (see renderer::ViewportSlot).
    // The fields below are the shared resources created once by ensure_hdr_shared().
    /// Bloom BGL: input_tex + sampler + uniform.
    pub(crate) bloom_bgl: Option<wgpu::BindGroupLayout>,
    /// SSAO BGL: depth + depth_sampler(non-filter) + noise + noise_sampler + kernel + uniform.
    pub(crate) ssao_bgl: Option<wgpu::BindGroupLayout>,
    /// SSAO blur BGL: ssao_tex + sampler.
    pub(crate) ssao_blur_bgl: Option<wgpu::BindGroupLayout>,

    // --- Post-processing (HDR, bloom, SSAO) ---
    // These are all Option<> and created lazily by ensure_hdr_target().
    /// HDR intermediate color texture (Rgba16Float, viewport-sized).
    pub(crate) hdr_texture: Option<wgpu::Texture>,
    pub(crate) hdr_view: Option<wgpu::TextureView>,
    /// HDR depth+stencil texture (Depth24PlusStencil8, viewport-sized, single-sample).
    pub(crate) hdr_depth_texture: Option<wgpu::Texture>,
    pub(crate) hdr_depth_view: Option<wgpu::TextureView>,
    /// Depth-only view of hdr_depth_texture (for SSAO binding : depth aspect only).
    pub(crate) hdr_depth_only_view: Option<wgpu::TextureView>,
    /// Last HDR target size [w, h]. Used to detect resize.
    pub(crate) hdr_size: [u32; 2],

    /// Tone mapping pipeline (renders fullscreen tri, hdr_texture -> output).
    pub(crate) tone_map_pipeline: Option<wgpu::RenderPipeline>,
    /// Tone map bind group layout.
    pub(crate) tone_map_bgl: Option<wgpu::BindGroupLayout>,
    /// Tone map bind group (rebuilt on HDR resize or placeholder change).
    pub(crate) tone_map_bind_group: Option<wgpu::BindGroup>,
    /// Tone map uniform buffer.
    pub(crate) tone_map_uniform_buf: Option<wgpu::Buffer>,

    /// Bloom threshold texture (Rgba16Float, full res).
    pub(crate) bloom_threshold_texture: Option<wgpu::Texture>,
    pub(crate) bloom_threshold_view: Option<wgpu::TextureView>,
    /// Bloom ping (Rgba16Float, half res).
    pub(crate) bloom_ping_texture: Option<wgpu::Texture>,
    pub(crate) bloom_ping_view: Option<wgpu::TextureView>,
    /// Bloom pong (Rgba16Float, half res).
    pub(crate) bloom_pong_texture: Option<wgpu::Texture>,
    pub(crate) bloom_pong_view: Option<wgpu::TextureView>,
    /// Shared bloom pipelines (threshold + blur use the same BGL, different bind groups).
    pub(crate) bloom_threshold_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) bloom_blur_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) bloom_threshold_bg: Option<wgpu::BindGroup>,
    pub(crate) bloom_blur_h_bg: Option<wgpu::BindGroup>,
    pub(crate) bloom_blur_v_bg: Option<wgpu::BindGroup>,
    /// H-blur bind group that reads from bloom_pong (used for iteration passes 2+).
    pub(crate) bloom_blur_h_pong_bg: Option<wgpu::BindGroup>,
    /// Bloom threshold uniform buffer (threshold + intensity, written each frame).
    pub(crate) bloom_uniform_buf: Option<wgpu::Buffer>,
    /// Bloom H-blur uniform buffer (horizontal=1, constant).
    pub(crate) bloom_h_uniform_buf: Option<wgpu::Buffer>,
    /// Bloom V-blur uniform buffer (horizontal=0, constant).
    pub(crate) bloom_v_uniform_buf: Option<wgpu::Buffer>,

    /// SSAO result texture (R8Unorm, full res).
    pub(crate) ssao_texture: Option<wgpu::Texture>,
    pub(crate) ssao_view: Option<wgpu::TextureView>,
    /// SSAO blur result texture (R8Unorm, full res).
    pub(crate) ssao_blur_texture: Option<wgpu::Texture>,
    pub(crate) ssao_blur_view: Option<wgpu::TextureView>,
    /// 4×4 random rotation noise texture (Rgba8Unorm, REPEAT).
    pub(crate) ssao_noise_texture: Option<wgpu::Texture>,
    pub(crate) ssao_noise_view: Option<wgpu::TextureView>,
    /// 64-sample hemisphere kernel (storage buffer, `vec4<f32>` per sample).
    pub(crate) ssao_kernel_buf: Option<wgpu::Buffer>,
    pub(crate) ssao_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) ssao_blur_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) ssao_bg: Option<wgpu::BindGroup>,
    pub(crate) ssao_blur_bg: Option<wgpu::BindGroup>,
    pub(crate) ssao_uniform_buf: Option<wgpu::Buffer>,

    // --- Contact shadow resources ---
    pub(crate) contact_shadow_texture: Option<wgpu::Texture>,
    pub(crate) contact_shadow_view: Option<wgpu::TextureView>,
    pub(crate) contact_shadow_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) contact_shadow_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) contact_shadow_bg: Option<wgpu::BindGroup>,
    pub(crate) contact_shadow_uniform_buf: Option<wgpu::Buffer>,

    /// 1×1 black Rgba16Float placeholder used when bloom is disabled.
    pub(crate) bloom_placeholder_view: Option<wgpu::TextureView>,
    /// 1×1 white R8Unorm placeholder used when SSAO is disabled.
    pub(crate) ao_placeholder_view: Option<wgpu::TextureView>,
    /// 1×1 white R8Unorm placeholder used when contact shadows are disabled.
    pub(crate) cs_placeholder_view: Option<wgpu::TextureView>,

    /// Shared post-process linear-clamp sampler.
    pub(crate) pp_linear_sampler: Option<wgpu::Sampler>,
    /// Shared post-process nearest-clamp sampler (for depth).
    pub(crate) pp_nearest_sampler: Option<wgpu::Sampler>,

    /// HDR-format variants of core scene pipelines (created lazily in ensure_hdr_target).
    pub(crate) hdr_solid_pipeline: Option<wgpu::RenderPipeline>,
    /// HDR two-sided variant (cull_mode: None) for analytical surfaces.
    pub(crate) hdr_solid_two_sided_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_transparent_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_wireframe_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_solid_instanced_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_transparent_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// HDR overlay pipeline (TriangleList, Rgba16Float, alpha blending) for cap fill in HDR path.
    pub(crate) hdr_overlay_pipeline: Option<wgpu::RenderPipeline>,

    // --- Colormap / LUT resources ---
    /// Uploaded colormap GPU textures. Index = ColormapId value.
    pub(crate) colormap_textures: Vec<wgpu::Texture>,
    /// Views into colormap_textures. Index = ColormapId value.
    pub(crate) colormap_views: Vec<wgpu::TextureView>,
    /// CPU-side copy of each colormap for egui scalar bar rendering. Index = ColormapId value.
    pub(crate) colormaps_cpu: Vec<[[u8; 4]; 256]>,
    /// Fallback 1×1 LUT texture (bound when has_attribute=0; content irrelevant to the shader).
    #[allow(dead_code)]
    pub(crate) fallback_lut_texture: wgpu::Texture,
    /// View of fallback_lut_texture.
    pub(crate) fallback_lut_view: wgpu::TextureView,
    /// Fallback 4-byte zero storage buffer (bound when no scalar attribute is active).
    pub(crate) fallback_scalar_buf: wgpu::Buffer,
    /// Fallback 16-byte zero storage buffer (bound to binding 8 when no face color attribute is active).
    pub(crate) fallback_face_color_buf: wgpu::Buffer,
    /// IDs of built-in preset colormaps, in BuiltinColormap discriminant order.
    /// `None` until `ensure_colormaps_initialized()` has been called.
    pub(crate) builtin_colormap_ids: Option<[ColormapId; 5]>,
    /// Whether built-in colormaps have been uploaded to the GPU.
    pub(crate) colormaps_initialized: bool,

    // --- SciVis Phase B: point cloud and glyph pipelines (lazily created) ---
    /// Point cloud render pipeline. None until first point cloud is submitted.
    pub(crate) point_cloud_pipeline: Option<wgpu::RenderPipeline>,
    /// Glyph render pipeline. None until first glyph set is submitted.
    pub(crate) glyph_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for point cloud uniforms (group 1).
    pub(crate) point_cloud_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for glyph uniforms (group 1).
    pub(crate) glyph_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for glyph instance storage (group 2).
    pub(crate) glyph_instance_bgl: Option<wgpu::BindGroupLayout>,
    /// Cached glyph base mesh for Arrow shape (vertex + index buffers).
    pub(crate) glyph_arrow_mesh: Option<GlyphBaseMesh>,
    /// Cached glyph base mesh for Sphere shape.
    pub(crate) glyph_sphere_mesh: Option<GlyphBaseMesh>,
    /// Cached glyph base mesh for Cube shape.
    pub(crate) glyph_cube_mesh: Option<GlyphBaseMesh>,

    // --- SciVis Phase M8: polyline rendering (lazily created) ---
    /// Polyline render pipeline. None until first polyline set is submitted.
    pub(crate) polyline_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for polyline uniforms (group 1).
    pub(crate) polyline_bgl: Option<wgpu::BindGroupLayout>,

    // --- SciVis Phase M: streamtube rendering (lazily created) ---
    /// Streamtube render pipeline. None until first streamtube item is submitted.
    pub(crate) streamtube_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for streamtube uniforms (group 1).
    pub(crate) streamtube_bgl: Option<wgpu::BindGroupLayout>,

    // --- SciVis Phase D: volume rendering (lazily created) ---
    /// Uploaded 3D volume textures. Index = VolumeId value.
    pub(crate) volume_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    /// Volume render pipeline. None until first volume is submitted.
    pub(crate) volume_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for volume uniforms (group 1).
    pub(crate) volume_bgl: Option<wgpu::BindGroupLayout>,
    /// Cached unit cube vertex+index buffers for bounding box rasterization.
    pub(crate) volume_cube_vb: Option<wgpu::Buffer>,
    pub(crate) volume_cube_ib: Option<wgpu::Buffer>,
    /// Default linear ramp opacity LUT texture (256x1, R8Unorm).
    pub(crate) volume_default_opacity_lut: Option<wgpu::Texture>,
    pub(crate) volume_default_opacity_lut_view: Option<wgpu::TextureView>,

    // --- Phase G: GPU compute filtering (lazily created) ---
    /// Compute pipeline for Clip / Threshold index compaction. None until first use.
    pub(crate) compute_filter_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for the compute filter shader (group 0). None until first use.
    pub(crate) compute_filter_bgl: Option<wgpu::BindGroupLayout>,

    // --- Phase J: Order-independent transparency (OIT) : lazily created ---
    // These fields are superseded by ViewportHdrState.oit_* but kept for ensure_oit_targets compat.
    #[allow(dead_code)]
    /// Weighted-blended accumulation texture (Rgba16Float, viewport-sized).
    pub(crate) oit_accum_texture: Option<wgpu::Texture>,
    pub(crate) oit_accum_view: Option<wgpu::TextureView>,
    /// Weighted-blended reveal (transmittance) texture (R8Unorm, viewport-sized).
    pub(crate) oit_reveal_texture: Option<wgpu::Texture>,
    pub(crate) oit_reveal_view: Option<wgpu::TextureView>,
    /// OIT mesh pipeline (non-instanced, mesh_oit.wgsl, two color targets).
    pub(crate) oit_pipeline: Option<wgpu::RenderPipeline>,
    /// OIT instanced mesh pipeline (mesh_instanced_oit.wgsl / mesh_instanced with OIT targets).
    pub(crate) oit_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// OIT composite pipeline (oit_composite.wgsl, fullscreen tri, no depth).
    pub(crate) oit_composite_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the OIT composite pass (group 0: accum + reveal + sampler).
    pub(crate) oit_composite_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group for the OIT composite pass (rebuilt on OIT target resize).
    pub(crate) oit_composite_bind_group: Option<wgpu::BindGroup>,
    /// Linear clamp sampler shared by the OIT composite pass.
    pub(crate) oit_composite_sampler: Option<wgpu::Sampler>,
    /// Last OIT target size [w, h]. Used to detect resize.
    pub(crate) oit_size: [u32; 2],

    // --- IBL / environment map resources ---
    /// IBL irradiance equirect texture view (binding 7). None until environment uploaded.
    pub ibl_irradiance_view: Option<wgpu::TextureView>,
    /// IBL prefiltered specular equirect texture view (binding 8). None until environment uploaded.
    pub ibl_prefiltered_view: Option<wgpu::TextureView>,
    /// BRDF integration LUT texture view (binding 9). None until environment uploaded.
    pub ibl_brdf_lut_view: Option<wgpu::TextureView>,
    /// IBL linear-clamp sampler (binding 10).
    pub(crate) ibl_sampler: wgpu::Sampler,
    /// Skybox / full-res environment equirect texture view (binding 11). None until uploaded.
    pub ibl_skybox_view: Option<wgpu::TextureView>,
    /// Fallback 1×1 black Rgba16Float texture for IBL slots when no environment is loaded.
    #[allow(dead_code)]
    pub(crate) ibl_fallback_texture: wgpu::Texture,
    /// View of ibl_fallback_texture.
    pub(crate) ibl_fallback_view: wgpu::TextureView,
    /// Fallback 1×1 BRDF LUT (black : placeholder, never sampled due to `ibl_enabled` guard).
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
    pub(crate) ground_plane_bgl: wgpu::BindGroupLayout,
    /// Uniform buffer for GroundPlaneUniform (256 bytes, written each frame in prepare()).
    pub(crate) ground_plane_uniform_buf: wgpu::Buffer,
    /// Bind group for the ground plane pass (rebuilt when shadow atlas changes).
    pub(crate) ground_plane_bind_group: wgpu::BindGroup,

    // --- Phase 10B: Screen-space image overlays (lazily created) ---
    /// Render pipeline for screen-space image quads. None until first screen image is submitted.
    pub(crate) screen_image_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the screen image pipeline (group 0: uniform + texture + sampler).
    pub(crate) screen_image_bgl: Option<wgpu::BindGroupLayout>,

    // --- Phase K: GPU object-ID picking (lazily created) ---
    /// Render pipeline that outputs flat u32 object IDs to R32Uint + R32Float targets.
    /// `None` until `ensure_pick_pipeline` is first called.
    pub(crate) pick_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for group 1 of the pick pipeline (PickInstance storage buffer).
    pub(crate) pick_bind_group_layout_1: Option<wgpu::BindGroupLayout>,
    /// Minimal camera-only bind group layout for the pick pipeline (group 0, one uniform binding).
    pub(crate) pick_camera_bgl: Option<wgpu::BindGroupLayout>,
}
