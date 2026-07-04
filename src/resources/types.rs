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

/// Identifies a colourmap (LUT) uploaded to the GPU.
///
/// An append-only registry handle. The inner index is public because the
/// built-in colourmaps have stable indices a consumer can name directly; the
/// registry is never freed, so this stays a plain index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColourmapId(pub usize);

/// Identifies a matcap texture uploaded to the GPU.
///
/// Obtained from [`ViewportGpuResources::upload_matcap`] or
/// [`ViewportGpuResources::builtin_matcap_id`]. An append-only registry handle.
/// The `blendable` flag controls whether the alpha channel tints the base
/// geometry colour (`true`) or the matcap fully replaces the object colour (`false`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatcapId {
    /// Index into the GPU matcap texture store.
    pub(crate) index: usize,
    /// Whether the alpha channel blends with base geometry colour.
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
    /// Vivid hue-cycling sphere, colourful.  Blendable.
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

crate::resources::handle::registry_handle! {
    /// Identifies a 3D volume texture uploaded to the GPU.
    ///
    /// An append-only registry handle: volume textures are kept for the session
    /// and not individually freed.
    pub struct VolumeId;
}

crate::resources::handle::registry_handle! {
    /// Identifies a projected-tetrahedra mesh uploaded to the GPU for transparent
    /// volume rendering.
    ///
    /// Obtained from [`ViewportGpuResources::upload_projected_tet`]. An
    /// append-only registry handle.
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

/// Raw mesh data for upload to the GPU. Framework-agnostic representation.
#[derive(Clone)]
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
    /// Optional per-vertex tangents [tx, ty, tz, w] where w is handedness (+/-1.0).
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
                // location 2: colour (vec4f) : offset 24
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
    /// Reserved for future debug uniform fields.
    pub _pad_dbg: [u32; 3], // 12 bytes
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

/// Per-object uniform: world transform, material properties, selection state, and wireframe mode.
///
/// Layout (256 bytes, 16-byte aligned):
/// - model:                    [[f32;4];4] = 64 bytes  offset   0
/// - colour:                     [f32;4]   = 16 bytes  offset  64  (base_colour.xyz + opacity)
/// - selected:                   u32      =  4 bytes  offset  80
/// - wireframe:                  u32      =  4 bytes  offset  84
/// - ambient:                    f32      =  4 bytes  offset  88
/// - diffuse:                    f32      =  4 bytes  offset  92
/// - specular:                   f32      =  4 bytes  offset  96
/// - shininess:                  f32      =  4 bytes  offset 100
/// - has_texture:                u32      =  4 bytes  offset 104
/// - use_pbr:                    u32      =  4 bytes  offset 108
/// - metallic:                   f32      =  4 bytes  offset 112
/// - roughness:                  f32      =  4 bytes  offset 116
/// - has_normal_map:             u32      =  4 bytes  offset 120
/// - has_ao_map:                 u32      =  4 bytes  offset 124
/// - has_attribute:              u32      =  4 bytes  offset 128
/// - scalar_min:                 f32      =  4 bytes  offset 132
/// - scalar_max:                 f32      =  4 bytes  offset 136
/// - _pad_scalar:                u32      =  4 bytes  offset 140
/// - nan_colour:                 [f32;4]   = 16 bytes  offset 144
/// - use_nan_colour:              u32      =  4 bytes  offset 160
/// - use_matcap:                 u32      =  4 bytes  offset 164
/// - matcap_blendable:           u32      =  4 bytes  offset 168
/// - unlit:                      u32      =  4 bytes  offset 172
/// - use_face_colour:             u32      =  4 bytes  offset 176
/// - uv_vis_mode:                u32      =  4 bytes  offset 180  (0=off 1=checker 2=grid 3=localcheck 4=localrad)
/// - uv_vis_scale:               f32      =  4 bytes  offset 184
/// - backface_policy:            u32      =  4 bytes  offset 188  (0=Cull 1=Identical 2=DifferentColour)
/// - backface_colour:            [f32;4]   = 16 bytes  offset 192
/// - has_warp:                   u32      =  4 bytes  offset 208
/// - warp_scale:                 f32      =  4 bytes  offset 212
/// - has_position_override:      u32      =  4 bytes  offset 216
/// - has_normal_override:        u32      =  4 bytes  offset 220
/// - emissive:                   [f32;3]  = 12 bytes  offset 224
/// - use_flat:                   u32      =  4 bytes  offset 236  (1=flat shading, recover N from world_pos derivatives)
/// - alpha_mode:                 u32      =  4 bytes  offset 240  (0=Opaque, 1=Mask, 2=Blend)
/// - alpha_cutoff:               f32      =  4 bytes  offset 244
/// - has_metallic_roughness_tex: u32      =  4 bytes  offset 248
/// - has_emissive_tex:           u32      =  4 bytes  offset 252
/// - uv_transform:               [f32;4]  = 16 bytes  offset 256  (offset.xy, scale.xy)
/// - deform_flags:               u32      =  4 bytes  offset 272  (bit i = deformer slot i active)
/// - _pad_after_deform:          u32      =  4 bytes  offset 276  (align next vec2 to 8)
/// - ao_range:                   [f32;2]  =  8 bytes  offset 280
/// - metallic_range:             [f32;2]  =  8 bytes  offset 288
/// - roughness_range:            [f32;2]  =  8 bytes  offset 296
/// Total: 304 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ObjectUniform {
    pub(crate) model: [[f32; 4]; 4], //  64 bytes, offset   0
    pub(crate) colour: [f32; 4],     //  16 bytes, offset  64
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
    pub(crate) has_attribute: u32,   //   4 bytes, offset 128
    pub(crate) scalar_min: f32,      //   4 bytes, offset 132
    pub(crate) scalar_max: f32,      //   4 bytes, offset 136
    /// 1 = sample the shadow atlas, 0 = treat the fragment as unshadowed.
    /// Wired from `ItemSettings.receive_shadows`.
    pub(crate) receive_shadows: u32, //   4 bytes, offset 140
    pub(crate) nan_colour: [f32; 4], //  16 bytes, offset 144
    pub(crate) use_nan_colour: u32,  //   4 bytes, offset 160
    pub(crate) use_matcap: u32,      //   4 bytes, offset 164
    pub(crate) matcap_blendable: u32, //   4 bytes, offset 168
    pub(crate) unlit: u32,           //   4 bytes, offset 172
    pub(crate) use_face_colour: u32, //   4 bytes, offset 176
    pub(crate) uv_vis_mode: u32,     //   4 bytes, offset 180
    pub(crate) uv_vis_scale: f32,    //   4 bytes, offset 184
    pub(crate) backface_policy: u32, //   4 bytes, offset 188  (0=Cull 1=Identical 2=DifferentColour)
    pub(crate) backface_colour: [f32; 4], //  16 bytes, offset 192
    pub(crate) has_warp: u32,        //   4 bytes, offset 208
    pub(crate) warp_scale: f32,      //   4 bytes, offset 212
    /// 1 when a per-vertex position storage buffer is bound at group 1 binding 13.
    /// Wired from `GpuMesh::position_override_buffer.is_some()`.
    pub(crate) has_position_override: u32, //   4 bytes, offset 216
    /// 1 when a per-vertex normal storage buffer is bound at group 1 binding 14.
    pub(crate) has_normal_override: u32, //   4 bytes, offset 220
    pub(crate) emissive: [f32; 3],   //  12 bytes, offset 224
    /// 1 = recover the shading normal from screen-space derivatives of
    /// `world_pos` (`ShadingModel::Flat`); 0 = use the interpolated vertex
    /// normal (or TBN normal map when bound).
    pub(crate) use_flat: u32, //   4 bytes, offset 236
    pub(crate) alpha_mode: u32,      //   4 bytes, offset 240  (0=Opaque, 1=Mask, 2=Blend)
    pub(crate) alpha_cutoff: f32,    //   4 bytes, offset 244
    pub(crate) has_metallic_roughness_tex: u32, //   4 bytes, offset 248
    pub(crate) has_emissive_tex: u32, //   4 bytes, offset 252
    /// Per-material UV transform applied to every texture sample.
    /// `[offset_x, offset_y, scale_x, scale_y]`. Defaults to `(0, 0, 1, 1)`
    /// (identity). Lets atlas-packed materials share one mesh instance.
    pub(crate) uv_transform: [f32; 4], //  16 bytes, offset 256
    /// Bit `i` set when deformer slot `i` is active for this draw. Zero when
    /// no deformer registry has attached data for this mesh.
    pub(crate) deform_flags: u32, //   4 bytes, offset 272
    pub(crate) _pad_after_deform: u32, //   4 bytes, offset 276 (align next vec2 to 8)
    /// Min/max remap applied to the AO map's R sample (identity `[0, 1]`).
    /// Mirrors `Material::ao_range`.
    pub(crate) ao_range: [f32; 2], //   8 bytes, offset 280
    /// Min/max remap applied to the metallic sample (B channel of the MR
    /// texture). Identity `[0, 1]`. Mirrors `Material::metallic_range`.
    pub(crate) metallic_range: [f32; 2], //   8 bytes, offset 288
    /// Min/max remap applied to the roughness sample (G channel of the MR
    /// texture). Identity `[0, 1]`. Mirrors `Material::roughness_range`.
    pub(crate) roughness_range: [f32; 2], //   8 bytes, offset 296
}

const _: () = assert!(std::mem::size_of::<ObjectUniform>() == 304);

/// Per-instance GPU data for instanced rendering. Matches the WGSL `InstanceData` struct.
///
/// Layout: 176 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceData {
    pub(crate) model: [[f32; 4]; 4], //  64 bytes, offset   0
    pub(crate) colour: [f32; 4],     //  16 bytes, offset  64
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
    pub(crate) unlit: u32,           //   4 bytes, offset 128
    /// 1 = sample the shadow atlas, 0 = treat the fragment as unshadowed.
    pub(crate) receive_shadows: u32, //   4 bytes, offset 132
    /// 1 = recover the shading normal from screen-space derivatives of
    /// `world_pos` (`ShadingModel::Flat`).
    pub(crate) use_flat: u32, //   4 bytes, offset 136
    pub(crate) _pad_inst: u32,       //   4 bytes, offset 140
    /// Per-material UV transform; mirrors `ObjectUniform::uv_transform`.
    /// `[offset_x, offset_y, scale_x, scale_y]`.
    pub(crate) uv_transform: [f32; 4], //  16 bytes, offset 144
    /// Min/max remap applied to the AO map's R sample (identity `[0, 1]`).
    /// Mirrors `Material::ao_range`. The instanced mesh shaders do not sample
    /// the MR texture today, so `metallic_range` / `roughness_range` are
    /// intentionally absent from `InstanceData`.
    pub(crate) ao_range: [f32; 2], //   8 bytes, offset 160
    pub(crate) _pad_ao_range: [f32; 2], //  8 bytes, offset 168 (struct stride to 16B)
}

const _: () = assert!(std::mem::size_of::<InstanceData>() == 176);

/// Per-instance GPU data for the object-ID pick pass.
///
/// Stores only the model matrix and a sentinel object ID : none of the material
/// fields needed by the full [`InstanceData`] struct.
///
/// Layout (80 bytes):
/// - model_c0..model_c3: vec4<f32> x 4 = 64 bytes (model matrix, column-major)
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

/// Per-instance world-space AABB, uploaded to GPU for the compute cull pass.
///
/// Layout (32 bytes):
/// - min:         [f32; 3] = 12 bytes, offset  0
/// - batch_index: u32      =  4 bytes, offset 12 (index into batch_meta_buf)
/// - max:         [f32; 3] = 12 bytes, offset 16
/// - _pad:        u32      =  4 bytes, offset 28
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceAabb {
    pub(crate) min: [f32; 3],
    pub(crate) batch_index: u32,
    pub(crate) max: [f32; 3],
    /// 1 = item participates in shadow casting, 0 = skipped during shadow cull.
    pub(crate) cast_shadows: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceAabb>() == 32);

/// Per-batch metadata read by the GPU cull pass.
///
/// One entry per batch in the `batch_meta` storage buffer attached to a
/// [`CullSubmission`](crate::plugin_api::CullSubmission). Layout (32 bytes,
/// 16-byte aligned):
///
/// - `index_count`:     `u32` - index range used by this batch's draw
/// - `first_index`:     `u32` - index buffer offset (typically 0)
/// - `instance_offset`: `u32` - first instance for this batch in the AABB buffer
/// - `instance_count`:  `u32` - number of instances belonging to this batch
/// - `vis_offset`:      `u32` - first slot in the visibility output buffer
/// - `is_transparent`:  `u32` - `1` marks a transparent batch
/// - `_pad`:            `[u32; 2]`
///
/// `vis_offset` is a prefix sum of `instance_count` across batches; for a
/// scene where instances are laid out contiguously per batch it equals
/// `instance_offset`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchMeta {
    /// Mesh index count for one instance.
    pub index_count: u32,
    /// First index offset into the bound index buffer.
    pub first_index: u32,
    /// Offset into the instance AABB buffer where this batch begins.
    pub instance_offset: u32,
    /// Number of instances in the batch.
    pub instance_count: u32,
    /// First slot in the visibility output buffer this batch writes to.
    pub vis_offset: u32,
    /// `1` if the batch is transparent, `0` for opaque.
    pub is_transparent: u32,
    /// Padding to keep the struct 16-byte aligned.
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<BatchMeta>() == 32);

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

/// Single clip-volume uniform : 128 bytes, bound at group 0 binding 6.
///
/// Deprecated in favour of [`ClipVolumesUniform`], which supports up to
/// [`CLIP_VOLUME_MAX`] simultaneous box/sphere clip volumes.
#[deprecated(
    since = "0.9.0",
    note = "use ClipVolumesUniform and ClipVolumeEntry instead"
)]
#[repr(C)]
#[derive(Copy, Clone)]
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

#[allow(deprecated)]
impl ClipVolumeUniform {
    /// Build a `ClipVolumeUniform` from a [`crate::renderer::ClipShape`] value.
    ///
    /// Deprecated: construct a [`ClipVolumeEntry`] via [`ClipVolumeEntry::from_box`]
    /// or [`ClipVolumeEntry::from_sphere`] instead.
    #[deprecated(
        since = "0.9.0",
        note = "use ClipVolumeEntry::from_box / from_sphere instead"
    )]
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
            _ => {}
        }
        u
    }
}

// SAFETY: ClipVolumeUniform is repr(C) and contains only u32/f32 fields - all Pod-compatible.
#[allow(deprecated)]
unsafe impl bytemuck::Zeroable for ClipVolumeUniform {}
#[allow(deprecated)]
unsafe impl bytemuck::Pod for ClipVolumeUniform {}

/// Per-object outline uniform for the two-pass stencil outline effect.
///
/// Layout (112 bytes):
/// - model:        [[f32;4];4] = 64 bytes
/// - colour:         [f32;4]   = 16 bytes  (outline RGBA)
/// - pixel_offset:  f32       =  4 bytes  (outline ring width in pixels)
/// - _pad:          [f32;3]   = 12 bytes
/// - deform_flags:  u32       =  4 bytes  (bit i set when deformer slot i is active for this draw)
/// - _deform_pad:   [u32;3]   = 12 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OutlineUniform {
    pub(crate) model: [[f32; 4]; 4],  //  64 bytes
    pub(crate) colour: [f32; 4],      //  16 bytes
    pub(crate) pixel_offset: f32,     //   4 bytes
    pub(crate) _pad: [f32; 3],        //  12 bytes
    pub(crate) deform_flags: u32,     //   4 bytes
    pub(crate) _deform_pad: [u32; 3], //  12 bytes
}

pub(crate) struct OutlineObjectBuffers {
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    pub two_sided: bool,
    /// Per-instance deformer id for the picked node, or `None` when the node
    /// has no per-instance deformer data. When `Some` and the renderer has
    /// per-instance data for `(mesh_id, instance_id)`, the outline mask is
    /// drawn with that bind group so the selection halo tracks the deformed
    /// silhouette.
    pub deform_instance: Option<u32>,
    pub _mask_uniform_buf: wgpu::Buffer,
    pub mask_bind_group: wgpu::BindGroup,
}

/// Per-item uniform for the Gaussian splat outline mask pass (112 bytes).
///
/// Padded to 112 bytes to match `OutlineUniform`. Both structs share the same
/// bind group layout (`outline_bgl`) and wgpu enforces the maximum required
/// size across all pipelines using that layout.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SplatOutlineMaskUniform {
    pub(crate) model: [[f32; 4]; 4], // 64 bytes
    pub(crate) viewport_w: f32,      //  4 bytes
    pub(crate) viewport_h: f32,      //  4 bytes
    pub(crate) pixel_radius: f32,    //  4 bytes
    pub(crate) _pad: [f32; 9],       // 36 bytes  (total: 112)
}

/// Per-frame GPU buffers for one selected Gaussian splat set's outline mask draw.
pub(crate) struct SplatOutlineBuffers {
    /// Object-space positions as `[f32; 3]` per splat, instance-stepped.
    pub(crate) position_buf: wgpu::Buffer,
    /// Per-instance pixel radius as `f32`, instance-stepped.
    pub(crate) size_buf: wgpu::Buffer,
    /// Number of splats (= instance count).
    pub(crate) instance_count: u32,
    /// Uniform buffer kept alive for the duration of the frame.
    pub(crate) _uniform_buf: wgpu::Buffer,
    /// Bind group for group 1 (SplatOutlineMaskUniform).
    pub(crate) bind_group: wgpu::BindGroup,
}

/// Inline geometry outline buffers for flat world-space quads (image slices).
///
/// Unlike `OutlineObjectBuffers`, the vertex/index data is owned here rather than
/// looked up via a `MeshId`.
pub(crate) struct RawGeomOutlineBuffers {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
    pub two_sided: bool,
    pub _uniform_buf: wgpu::Buffer,
    pub mask_bind_group: wgpu::BindGroup,
}

/// Per-frame outline item for a tube/streamtube/ribbon mesh.
///
/// Holds an index into the per-frame gpu_data array and a mask bind group
/// that supplies an identity model matrix to the outline_mask shader.
pub(crate) struct CurveMeshOutlineItem {
    pub index: usize,
    pub two_sided: bool,
    pub _mask_uniform_buf: wgpu::Buffer,
    pub mask_bind_group: wgpu::BindGroup,
}

/// NDC-space rect outline for screen image overlays.
pub(crate) struct ScreenRectOutlineBuffers {
    pub _uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

/// Uniform for the fullscreen outline edge-detection pass (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OutlineEdgeUniform {
    pub(crate) colour: [f32; 4], // 16 bytes
    pub(crate) radius: f32,      //  4 bytes
    pub(crate) viewport_w: f32,  //  4 bytes
    pub(crate) viewport_h: f32,  //  4 bytes
    pub(crate) _pad: f32,        //  4 bytes
}

/// Per-frame uniform for the sub-object highlight pass (48 bytes).
///
/// Shared by the fill, edge, and sprite draw calls.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SubHighlightUniform {
    pub(crate) fill_colour: [f32; 4], // 16 bytes
    pub(crate) edge_colour: [f32; 4], // 16 bytes
    pub(crate) edge_width: f32,       //  4 bytes (pixels)
    pub(crate) vertex_size: f32,      //  4 bytes (pixels)
    pub(crate) viewport_width: f32,   //  4 bytes
    pub(crate) viewport_height: f32,  //  4 bytes
                                      // total 48 bytes
}

/// GPU buffers for one frame of sub-object highlight rendering.
///
/// Rebuilt whenever [`InteractionFrame::sub_selection`] version changes.
/// All three passes (fill, edges, sprites) share a single
/// [`SubHighlightUniform`] buffer bound at group 1.
pub(crate) struct SubHighlightGpuData {
    // Face fill : flat triangle vertex list (xyz f32, 12 bytes each, non-indexed).
    pub(crate) fill_vertex_buf: wgpu::Buffer,
    pub(crate) fill_vertex_count: u32,
    // Edge lines : segment instances (pos_a xyz + pos_b xyz, 24 bytes each).
    pub(crate) edge_vertex_buf: wgpu::Buffer,
    pub(crate) edge_segment_count: u32,
    // Vertex / point sprites : positions (xyz padded to 16 bytes).
    pub(crate) sprite_vertex_buf: wgpu::Buffer,
    pub(crate) sprite_point_count: u32,
    // Shared uniform buffer.
    pub(crate) _uniform_buf: wgpu::Buffer,
    // Per-pass bind groups (group 1: SubHighlightUniform).
    pub(crate) fill_bind_group: wgpu::BindGroup,
    pub(crate) edge_bind_group: wgpu::BindGroup,
    pub(crate) sprite_bind_group: wgpu::BindGroup,
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
    pub(crate) edl_enabled: u32,
    pub(crate) edl_radius: f32,
    pub(crate) edl_strength: f32,
    pub(crate) background_colour: [f32; 4],
    pub(crate) near_plane: f32,
    pub(crate) far_plane: f32,
    pub(crate) lic_enabled: u32,
    pub(crate) lic_strength: f32,
}

const _: () = assert!(std::mem::size_of::<ToneMapUniform>() == 64);

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

/// Depth of field uniform (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DofUniform {
    pub(crate) focal_distance: f32,
    pub(crate) focal_range: f32,
    pub(crate) max_blur_radius: f32,
    pub(crate) near_plane: f32,
    pub(crate) far_plane: f32,
    pub(crate) viewport_width: f32,
    pub(crate) viewport_height: f32,
    pub(crate) _pad: f32,
}

/// Shadow atlas uniform (416 bytes, bound at group 0 binding 5).
///
/// Contains per-cascade view-projection matrices, split distances, atlas layout,
/// and PCSS parameters. Used by the fragment shader for CSM cascade selection
/// and shadow sampling.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ShadowAtlasUniform {
    /// 4 cascade view-projection matrices (each 64 bytes = 4x4 f32). Flattened.
    pub(crate) cascade_view_proj: [[f32; 4]; 16], // 256 bytes
    /// Distance-based split values for cascade selection (eye-to-fragment distance).
    pub(crate) cascade_splits: [f32; 4], //  16 bytes
    /// Number of active cascades (1-4).
    pub(crate) cascade_count: u32, //   4 bytes
    /// Backing atlas texture size in pixels (always `SHADOW_ATLAS_SIZE`).
    ///
    /// Shaders derive everything from this plus the atlas rects: one texel in
    /// UV space is `1.0 / atlas_size`, and a tile's pixel width is
    /// `atlas_size * (rect.z - rect.x)`. When the requested shadow resolution
    /// is below `SHADOW_ATLAS_SIZE`, the rects shrink but this value does not.
    pub(crate) atlas_size: f32, //   4 bytes
    /// Shadow filter mode: 0=PCF, 1=PCSS.
    pub(crate) shadow_filter: u32, //   4 bytes
    /// PCSS light source radius in UV space.
    pub(crate) pcss_light_radius: f32, //   4 bytes
    /// Per-slot atlas UV rects: [uv_min.x, uv_min.y, uv_max.x, uv_max.y] x 8 slots.
    pub(crate) atlas_rects: [[f32; 4]; 8], // 128 bytes
}
// Total: 256 + 16 + 16 + 128 = 416 bytes

/// Uniform for the shadow atlas blit overlay. 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AtlasBlitUniform {
    /// NDC rect [xmin, ymin, xmax, ymax].
    pub(crate) rect: [f32; 4],
}

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

/// Per-vertex data for overlay rendering: position only (no normal/colour in vertex).
///
/// Colour is provided via the OverlayUniform rather than per-vertex to keep
/// the buffer minimal : all vertices of a single overlay quad share the same colour.
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

/// Per-overlay uniform: model matrix and RGBA colour with alpha for transparency.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayUniform {
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) colour: [f32; 4], // RGBA with alpha for transparency
}

/// Per-vertex data for overlay text and solid screen-space quads (labels, backgrounds, leader lines).
///
/// All fields are packed into a single vertex to allow batching every overlay
/// quad into one draw call per frame.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayTextVertex {
    /// NDC position (xy, z=0 w=1 in shader).
    pub position: [f32; 2],
    /// Atlas UV coordinates.  Ignored when `use_texture` is 0.
    pub uv: [f32; 2],
    /// RGBA tint colour.
    pub colour: [f32; 4],
    /// 1.0 = sample glyph atlas alpha, 0.0 = solid colour.
    pub use_texture: f32,
    pub _pad: f32,
}

impl OverlayTextVertex {
    /// wgpu vertex buffer layout matching `overlay_text.wgsl`.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayTextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position vec2f
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 1: uv vec2f
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 2: colour vec4f
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 3: use_texture f32
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Per-frame GPU data for batched overlay label rendering.
pub(crate) struct LabelGpuData {
    /// Vertex buffer containing all label geometry for this frame.
    pub vertex_buf: wgpu::Buffer,
    /// Number of vertices to draw.
    pub vertex_count: u32,
    /// Bind group referencing the glyph atlas texture and sampler.
    pub bind_group: wgpu::BindGroup,
}

/// Per-vertex data for SDF overlay shapes.
///
/// Each shape is a bounding quad (6 vertices). The fragment shader uses
/// `local_pos` and `half_size` to evaluate a signed-distance function,
/// producing anti-aliased fill and border.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayShapeVertex {
    /// NDC position (xy).
    pub position: [f32; 2],
    /// Position relative to shape centre, in logical pixels.
    pub local_pos: [f32; 2],
    /// RGBA fill colour (pre-multiplied opacity).
    pub fill_colour: [f32; 4],
    /// RGBA border colour (pre-multiplied opacity).
    pub border_colour: [f32; 4],
    /// Half-extents of the shape bounding box in logical pixels.
    pub half_size: [f32; 2],
    /// Shape-specific radii. For RoundedRect: [top-left, top-right,
    /// bottom-right, bottom-left]. For Rect: uniform radius in [0].
    /// Unused components are zero.
    pub radii: [f32; 4],
    /// Border thickness in logical pixels.
    pub border_width: f32,
    /// Encoded shape type: 0 = Rect/RoundedRect, 1 = Circle, 2 = Ellipse, 3 = Capsule.
    pub shape_type: f32,
    /// RGBA end colour for linear gradient (equals fill_colour for solid fill).
    pub fill_colour2: [f32; 4],
    /// Gradient parameters: `[type, angle, stop_count, _pad]`.
    /// `type` selects solid/linear/radial/conical; `stop_count` is the
    /// number of active gradient stops (0 for solid, 2..4 otherwise).
    pub gradient_params: [f32; 4],
    /// RGBA shadow colour (pre-multiplied opacity).
    pub shadow_colour: [f32; 4],
    /// Shadow parameters: x = radius (pixels), y = offset_x, z = offset_y.
    pub shadow_params: [f32; 4],
    /// Clip rectangle in framebuffer pixels (x0, y0, x1, y1). All-zero means
    /// no clipping. Fragments outside the rect are discarded.
    pub clip_rect: [f32; 4],
    /// Rotation around the shape centre in radians. Applied to `local_pos`
    /// before SDF evaluation so the shape rotates inside its axis-aligned
    /// bounding box.
    pub rotation: f32,
    /// Third gradient colour stop. Unused for 2-stop and solid fills.
    pub stop_colour_c: [f32; 4],
    /// Fourth gradient colour stop. Unused for 2-stop and solid fills.
    pub stop_colour_d: [f32; 4],
    /// Stop positions in `[0, 1]` along the gradient axis, sorted ascending.
    /// For 2-stop fills only `[0]` and `[1]` are valid; remaining entries
    /// are 1.0 sentinels. The active stop count lives in
    /// `gradient_params[2]`.
    pub stop_positions: [f32; 4],
}

impl OverlayShapeVertex {
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayShapeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position vec2f
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 1: local_pos vec2f
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 2: fill_colour vec4f
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 3: border_colour vec4f
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 4: half_size vec2f
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 5: radii vec4f
                wgpu::VertexAttribute {
                    offset: 56,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 6: shape_meta vec2f (border_width, shape_type) -- combined
                wgpu::VertexAttribute {
                    offset: 72,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 7: stop_positions vec4f (gradient stop positions)
                wgpu::VertexAttribute {
                    offset: 196,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 8: fill_colour2 vec4f (end colour for gradient)
                wgpu::VertexAttribute {
                    offset: 80,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 9: gradient_params vec4f (type, angle, stop_count, pad)
                wgpu::VertexAttribute {
                    offset: 96,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 10: shadow_colour vec4f
                wgpu::VertexAttribute {
                    offset: 112,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 11: shadow_params vec4f (radius, offset_x, offset_y, border_mode)
                wgpu::VertexAttribute {
                    offset: 128,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 12: clip_rect vec4f (x0, y0, x1, y1 in framebuffer pixels)
                wgpu::VertexAttribute {
                    offset: 144,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 13: rotation f32 (radians)
                wgpu::VertexAttribute {
                    offset: 160,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Float32,
                },
                // location 14: stop_colour_c vec4f
                wgpu::VertexAttribute {
                    offset: 164,
                    shader_location: 14,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 15: stop_colour_d vec4f
                wgpu::VertexAttribute {
                    offset: 180,
                    shader_location: 15,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Per-vertex data for SDF textured overlay shapes.
///
/// Same layout as `OverlayShapeVertex` with an additional UV field at the end.
/// Used by the texture pipeline; `fill_colour` acts as a tint multiplied with
/// the sampled texel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayShapeTexVertex {
    /// NDC position (xy).
    pub position: [f32; 2],
    /// Position relative to shape centre, in logical pixels.
    pub local_pos: [f32; 2],
    /// RGBA tint colour (pre-multiplied opacity). Multiplied with texture sample.
    pub fill_colour: [f32; 4],
    /// RGBA border colour (pre-multiplied opacity).
    pub border_colour: [f32; 4],
    /// Half-extents of the shape bounding box in logical pixels.
    pub half_size: [f32; 2],
    /// Shape-specific radii (same encoding as `OverlayShapeVertex`).
    pub radii: [f32; 4],
    /// Border thickness in logical pixels.
    pub border_width: f32,
    /// Encoded shape type (same values as `OverlayShapeVertex`).
    pub shape_type: f32,
    /// Texture UV coordinates. (0,0) = top-left of image, (1,1) = bottom-right.
    /// Slightly outside [0,1] in the border/AA padding region.
    pub uv: [f32; 2],
    /// RGBA shadow colour (pre-multiplied opacity).
    pub shadow_colour: [f32; 4],
    /// Shadow parameters: x = radius (pixels), y = offset_x, z = offset_y, w = border_mode.
    pub shadow_params: [f32; 4],
    /// Per-shape flags.
    /// - `x` = is_backdrop_blur (0.0 = regular tinted sample; 1.0 = the bound
    ///   texture is the scene-blur output composited under the tint).
    /// - `y` = nine-slice centre tile mode (0 = stretch, 1 = tile).
    /// - `z` = nine-slice edge tile mode (0 = stretch, 1 = tile).
    /// - `w` = nine-slice enabled flag (0 = disabled, 1 = enabled).
    pub extras: [f32; 4],
    /// Nine-slice insets in texture UV space, `[top, right, bottom, left]`.
    /// Unused when `extras.w == 0`.
    pub nine_slice_uv: [f32; 4],
    /// Nine-slice insets as fractions of the shape's bounding box,
    /// `[top, right, bottom, left]`. Unused when `extras.w == 0`.
    pub nine_slice_frac: [f32; 4],
    /// Texture transform part A: `[offset_x, offset_y, scale_x, scale_y]`.
    /// `scale = 1.0` is 1:1. Applies when 9-slice is disabled.
    pub texture_transform_a: [f32; 4],
    /// Texture transform part B:
    /// `[rotation_radians, tile_mode (0/1/2), flip_x (0/1), flip_y (0/1)]`.
    pub texture_transform_b: [f32; 4],
}

impl OverlayShapeTexVertex {
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayShapeTexVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position vec2f
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 1: local_pos vec2f
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 2: fill_colour vec4f (tint)
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 3: border_colour vec4f
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 4: half_size vec2f
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 5: radii vec4f
                wgpu::VertexAttribute {
                    offset: 56,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 6: border_width f32
                wgpu::VertexAttribute {
                    offset: 72,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
                // location 7: shape_type f32
                wgpu::VertexAttribute {
                    offset: 76,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
                // location 8: uv vec2f
                wgpu::VertexAttribute {
                    offset: 80,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 9: shadow_colour vec4f
                wgpu::VertexAttribute {
                    offset: 88,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 10: shadow_params vec4f (radius, offset_x, offset_y, border_mode)
                wgpu::VertexAttribute {
                    offset: 104,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 11: extras vec4f (blur, centre_mode, edge_mode, nine_slice_enabled)
                wgpu::VertexAttribute {
                    offset: 120,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 12: nine_slice_uv vec4f
                wgpu::VertexAttribute {
                    offset: 136,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 13: nine_slice_frac vec4f
                wgpu::VertexAttribute {
                    offset: 152,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 14: texture_transform_a vec4f (offset.xy, scale.xy)
                wgpu::VertexAttribute {
                    offset: 168,
                    shader_location: 14,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 15: texture_transform_b vec4f (rotation, tile_mode, flip_x, flip_y)
                wgpu::VertexAttribute {
                    offset: 184,
                    shader_location: 15,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// One batch of textured SDF overlay shapes sharing a single texture.
pub(crate) struct OverlayShapeTexBatch {
    pub vertex_buf: wgpu::Buffer,
    pub vertex_count: u32,
    pub bind_group: wgpu::BindGroup,
}

/// Persistent texture entry for an overlay shape texture fill.
///
/// Stored in `ViewportGpuResources::overlay_textures`.
pub(crate) struct OverlayShapeTextureEntry {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// Per-frame GPU data for batched SDF overlay shape rendering.
pub(crate) struct OverlayShapeGpuData {
    /// Vertex buffer for solid (non-textured) shapes. `None` when all shapes are textured.
    pub vertex_buf: Option<wgpu::Buffer>,
    /// Number of solid vertices. Zero when all shapes are textured.
    pub vertex_count: u32,
    /// One batch per unique texture, drawn after solid shapes.
    pub tex_batches: Vec<OverlayShapeTexBatch>,
    /// Vertex buffer for backdrop-blur shapes (frosted glass). Uses the same
    /// vertex layout as `OverlayShapeTexVertex` with screen-space UVs.
    /// The bind group is created at render time once the blurred scene texture
    /// is available.
    pub blur_vertex_buf: Option<wgpu::Buffer>,
    /// Number of blur backdrop vertices.
    pub blur_vertex_count: u32,
    /// Maximum `backdrop_blur` value across all blur shapes this frame.
    /// Used as the spread parameter for the Gaussian blur passes.
    pub max_blur_radius: f32,
}

/// Cached GPU textures for the backdrop blur (frosted glass) effect.
///
/// Stored on `ViewportRenderer` and recreated when the viewport size changes.
/// Contains a full-resolution intermediate (for rendering the scene when the
/// output surface lacks `TEXTURE_BINDING`), two half-resolution ping-pong
/// textures for the separable blur passes, and pre-built bind groups.
pub(crate) struct BackdropBlurState {
    /// Full-resolution intermediate render target. The scene is rendered here
    /// instead of directly to the surface so the result can be sampled. Kept
    /// alive so the matching view remains valid.
    #[allow(dead_code)]
    pub intermediate_texture: wgpu::Texture,
    pub intermediate_view: wgpu::TextureView,
    /// Half-resolution blur ping-pong texture A. Kept alive for its view.
    #[allow(dead_code)]
    pub blur_a_texture: wgpu::Texture,
    pub blur_a_view: wgpu::TextureView,
    /// Half-resolution blur ping-pong texture B. Kept alive for its view.
    #[allow(dead_code)]
    pub blur_b_texture: wgpu::Texture,
    pub blur_b_view: wgpu::TextureView,
    /// Viewport physical size the textures were created for.
    pub size: [u32; 2],
    /// Format the textures were created with.
    pub format: wgpu::TextureFormat,
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
    pub colour: [f32; 4],         // offset 128, 16 bytes
    pub shadow_colour: [f32; 4],  // offset 144, 16 bytes
    pub light_vp: [[f32; 4]; 4],  // offset 160, 64 bytes
    pub tan_half_fov: f32,        // offset 224,  4 bytes
    pub aspect: f32,              // offset 228,  4 bytes
    pub tile_size: f32,           // offset 232,  4 bytes
    pub shadow_bias: f32,         // offset 236,  4 bytes
    pub mode: u32,                // offset 240,  4 bytes
    pub shadow_opacity: f32,      // offset 244,  4 bytes
    pub _pad: [f32; 2],           // offset 248,  8 bytes
    pub colour2: [f32; 4],        // offset 256, 16 bytes : second tile colour
} // total  272 bytes

/// Uniform buffer layout for the full-screen analytical grid shader.
///
/// Contains all data needed by `grid.wgsl`: camera matrices for ray unprojection,
/// eye position, grid plane height, spacing for minor/major lines, and RGBA colours.
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
    /// RGBA colour for minor grid lines.
    pub colour_minor: [f32; 4], // offset 160, 16 bytes
    /// RGBA colour for major grid lines.
    pub colour_major: [f32; 4], // offset 176, 16 bytes
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
// Async texture upload types
// ---------------------------------------------------------------------------

/// Texture memory usage reported by [`ViewportGpuResources::texture_memory_stats`].
///
/// Counts bytes and textures uploaded via both the sync and async paths.
/// Internal resources (shadow maps, colourmaps, post-process targets) are
/// not included.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureMemoryStats {
    /// Bytes currently allocated on the GPU for user-uploaded textures.
    pub used_bytes: u64,
    /// Number of live user-uploaded textures.
    pub texture_count: u32,
}

/// Resident GPU bytes for the user-uploaded working set, from
/// [`ViewportGpuResources::resident_bytes`].
///
/// These are the classes a streaming or eviction policy frees and re-uploads:
/// meshes (`upload_mesh_data` and friends), user textures (`upload_texture` and
/// friends), Gaussian splats, marching-cubes volumes, and the pre-uploaded
/// scivis curves. The query is cheap enough to poll each frame.
///
/// Built-in resources are not counted: colourmap and matcap LUTs, IBL maps, the
/// shadow atlas, and post-process render targets. They are created once (or
/// resized with the viewport) and are not part of the evictable working set.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentBytes {
    /// GPU buffer bytes across every resident mesh (geometry, attributes,
    /// overrides, per-object uniforms).
    pub mesh_bytes: u64,
    /// GPU bytes across every resident user-uploaded texture.
    pub texture_bytes: u64,
    /// GPU buffer bytes across every resident Gaussian splat set (position,
    /// scale, rotation, opacity, and SH source buffers; per-viewport sort
    /// scratch is not counted).
    pub gaussian_splat_bytes: u64,
    /// GPU buffer bytes across every resident marching-cubes volume (all slab
    /// buffers of every live volume).
    pub mc_volume_bytes: u64,
    /// GPU buffer bytes across every pre-uploaded scivis curve resource
    /// (polylines, tubes, streamtubes, ribbons, point clouds, glyph sets,
    /// tensor glyph sets, and sprite sets).
    pub scivis_bytes: u64,
}

impl ResidentBytes {
    /// Total resident bytes across every counted class.
    pub fn total(&self) -> u64 {
        self.mesh_bytes
            + self.texture_bytes
            + self.gaussian_splat_bytes
            + self.mc_volume_bytes
            + self.scivis_bytes
    }
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
    /// Texture views are the fallback 1x1 textures by default; rebuilt when material
    /// texture assignment changes (tracked via `last_tex_key`).
    pub object_bind_group: wgpu::BindGroup,
    /// Last texture/attribute key used to build `object_bind_group`. `u64::MAX` = fallback / none.
    /// Fields: `(albedo, normal_map, ao_map, lut, attr_hash, matcap, warp_hash, metallic_roughness, emissive, position_override_gen, normal_override_gen)`.
    /// The trailing two slots are bumped by `set_position_override_buffer`,
    /// `set_normal_override_buffer`, and their `clear_*` counterparts, so the
    /// next `update_mesh_texture_bind_group` call rebuilds with the real
    /// override buffer (or the fallback) at bindings 13/14.
    pub(crate) last_tex_key: (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
    /// Per-named-attribute GPU storage buffers (f32 per vertex, STORAGE usage).
    pub attribute_buffers: std::collections::HashMap<String, wgpu::Buffer>,
    /// Scalar range `(min, max)` per attribute, computed at upload time.
    pub attribute_ranges: std::collections::HashMap<String, (f32, f32)>,
    /// Non-indexed vertex buffer containing 3xN expanded vertices for face-attribute rendering.
    /// `None` if no `Face` or `FaceColour` attributes exist for this mesh.
    pub face_vertex_buffer: Option<wgpu::Buffer>,
    /// Named face scalar buffers: 3N `f32` entries (value replicated for all 3 vertices of each tri).
    pub face_attribute_buffers: std::collections::HashMap<String, wgpu::Buffer>,
    /// Named face colour buffers: 3N `[f32; 4]` entries (colour replicated for all 3 vertices of each tri).
    pub face_colour_buffers: std::collections::HashMap<String, wgpu::Buffer>,
    /// Per-vertex vector attribute buffers: flat `array<f32>` with 3 values per vertex.
    /// Uploaded from `AttributeData::VertexVector`; used by the Surface LIC surface pass.
    pub vector_attribute_buffers: std::collections::HashMap<String, wgpu::Buffer>,
    /// Optional per-vertex position override buffer.
    ///
    /// When `Some`, the standard mesh pipeline reads positions from this buffer
    /// (group 1 binding 13, an `array<vec3<f32>>`) instead of the vertex
    /// buffer's position attribute. Set via
    /// `ViewportGpuResources::set_position_override_buffer` and consumed by a
    /// `GpuPlugin`'s compute output. Mutually exclusive per frame with
    /// `write_mesh_positions_normals`; the two paths race if both are used.
    pub position_override_buffer: Option<wgpu::Buffer>,
    /// Optional per-vertex normal override buffer. Same contract as
    /// `position_override_buffer` but bound at group 1 binding 14.
    pub normal_override_buffer: Option<wgpu::Buffer>,
    /// Monotonic counter bumped each time `set_position_override_buffer` or
    /// `clear_position_override` is called. Folded into `last_tex_key` so the
    /// object bind group rebuilds on the next `prepare()` call.
    pub(crate) position_override_gen: u64,
    /// Same idea for normal override (re)bind tracking.
    pub(crate) normal_override_gen: u64,
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

impl GpuMesh {
    /// Number of vertices in the mesh, derived from the vertex buffer size.
    ///
    /// Useful for sizing or validating a position/normal override buffer
    /// against the mesh it will be bound to.
    pub fn vertex_count(&self) -> usize {
        (self.vertex_buffer.size() / std::mem::size_of::<Vertex>() as u64) as usize
    }

    /// Total GPU buffer bytes held by this mesh.
    ///
    /// Sums the geometry, attribute, override, and per-object uniform buffers.
    /// Used for the resident-bytes accounting so freeing the mesh decrements the
    /// running total by the same amount it added at upload.
    pub(crate) fn gpu_byte_size(&self) -> u64 {
        let mut bytes = self.vertex_buffer.size()
            + self.index_buffer.size()
            + self.edge_index_buffer.size()
            + self.object_uniform_buf.size()
            + self.normal_uniform_buf.size();
        for buf in [
            self.normal_line_buffer.as_ref(),
            self.face_vertex_buffer.as_ref(),
            self.position_override_buffer.as_ref(),
            self.normal_override_buffer.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            bytes += buf.size();
        }
        for map in [
            &self.attribute_buffers,
            &self.face_attribute_buffers,
            &self.face_colour_buffers,
            &self.vector_attribute_buffers,
        ] {
            bytes += map.values().map(|b| b.size()).sum::<u64>();
        }
        bytes
    }
}

// ---------------------------------------------------------------------------
// GPU data types for point cloud and glyph renderers
// ---------------------------------------------------------------------------

/// Cached GPU vertex + index buffers for a glyph base mesh (arrow, sphere, cube).
pub(crate) struct GlyphBaseMesh {
    /// Vertex buffer using the full `Vertex` layout (64 bytes stride).
    pub vertex_buffer: wgpu::Buffer,
    /// Triangle index buffer.
    pub index_buffer: wgpu::Buffer,
    /// Number of indices.
    pub index_count: u32,
    /// Edge index buffer (deduplicated pairs) for wireframe LineList rendering.
    pub edge_index_buffer: wgpu::Buffer,
    /// Number of indices in the edge buffer.
    pub edge_index_count: u32,
}

/// Per-frame GPU data for one point cloud item, created in `prepare()`.
#[derive(Clone)]
pub struct PointCloudGpuData {
    /// Vertex buffer: one entry per point, packed as `[position: vec3, _pad: f32]` (16 bytes).
    /// The shader reads colour/scalar from storage buffers indexed by `vertex_index`.
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Number of points (= draw count).
    pub(crate) point_count: u32,
    /// Bind group (group 1): uniform + LUT + sampler + scalar + colour + radius + transparency.
    pub(crate) bind_group: wgpu::BindGroup,
    // Keep the buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: wgpu::Buffer,
    pub(crate) _scalar_buf: wgpu::Buffer,
    pub(crate) _colour_buf: wgpu::Buffer,
    pub(crate) _radius_buf: wgpu::Buffer,
    pub(crate) _transparency_buf: wgpu::Buffer,
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
    pub(crate) bind_group: wgpu::BindGroup,
    /// Blend mode selected by the host. Drives pipeline selection at draw time.
    pub(crate) blend: crate::renderer::SpriteBlend,
    // Keep buffers alive for the lifetime of this struct.
    pub(crate) _instance_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one sprite batch item, created in `prepare()`.
#[derive(Clone)]
pub struct SpriteGpuData {
    /// Position vertex buffer: one `vec3` per sprite, instance-stepped.
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Number of sprites (= draw instance count).
    pub(crate) sprite_count: u32,
    /// Bind group (group 1): uniform + texture + sampler + instance storage buffer.
    pub(crate) bind_group: wgpu::BindGroup,
    /// Whether this batch was submitted with `depth_write: true`.
    pub(crate) depth_write: bool,
    /// Blend mode requested by the host for this batch.
    pub(crate) blend: crate::renderer::SpriteBlend,
    /// When true, skip the billboard draw; the wireframe overlay polyline is rendered instead.
    pub(crate) wireframe: bool,
    /// Refractive distortion strength in NDC pixels; `0.0` means a regular
    /// sprite. Routes the draw through the sprite refraction post-pass
    /// instead of the normal sprite pass.
    pub(crate) refraction_strength: f32,
    /// When true, this batch was submitted with `SpriteItem::lit = true` and
    /// is drawn through the lit sprite pipeline.
    pub(crate) lit: bool,
    /// Group 3 bind group for the lit normal-map binding. Always populated for
    /// lit batches: a fallback texture is bound when no normal map is supplied
    /// so the same pipeline layout is honoured.
    pub(crate) lit_normal_bg: Option<wgpu::BindGroup>,
    // Keep buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: wgpu::Buffer,
    pub(crate) _instance_buf: wgpu::Buffer,
}

// ---------------------------------------------------------------------------
// Gaussian splat GPU data types
// ---------------------------------------------------------------------------

/// Per-viewport sort buffers for one Gaussian splat set.
pub(crate) struct GaussianSplatViewportSort {
    /// u32 view-space depth keys (flipped for back-to-front), written by depth compute each frame.
    pub depth_buf: wgpu::Buffer,
    /// Ping/pong key buffers for radix sort.
    pub keys_ping: wgpu::Buffer,
    pub keys_pong: wgpu::Buffer,
    /// Ping/pong value (index) buffers for radix sort.
    pub vals_ping: wgpu::Buffer,
    pub vals_pong: wgpu::Buffer,
    /// 256-entry atomic histogram / prefix-sum scratch.
    pub histogram_buf: wgpu::Buffer,
    /// Render bind group (group 1). Contains sorted_indices, positions, scales, rotations,
    /// opacities, sh_coefficients, and the per-viewport SplatUniform.
    pub render_bg: wgpu::BindGroup,
    /// Eye position at last sort; skip re-sort when unchanged.
    pub last_eye: [f32; 3],
    /// Per-viewport uniform buffer holding SplatUniform (model, viewport dims, sh_degree, count).
    pub uniform_buf: wgpu::Buffer,
}

/// Persistent GPU state for one uploaded Gaussian splat set.
pub(crate) struct GaussianSplatGpuSet {
    /// Positions as vec4<f32> (w=1), one per splat.
    pub position_buf: wgpu::Buffer,
    /// Scales as vec4<f32> (w=0), one per splat.
    pub scale_buf: wgpu::Buffer,
    /// Rotations as vec4<f32> [x,y,z,w], one per splat.
    pub rotation_buf: wgpu::Buffer,
    /// Opacities as f32, one per splat.
    pub opacity_buf: wgpu::Buffer,
    /// SH coefficients as f32, count = splat_count * sh_degree.coeff_count().
    pub sh_buf: wgpu::Buffer,
    /// SH degree for this set.
    pub sh_degree: crate::renderer::ShDegree,
    /// Number of splats.
    pub count: u32,
    /// Per-viewport sort buffers; index = viewport_index. Grown lazily.
    pub viewport_sort: Vec<Option<GaussianSplatViewportSort>>,
    /// CPU positions kept for potential picking (object-space).
    #[allow(dead_code)]
    pub cpu_positions: Vec<[f32; 3]>,
    /// CPU scales kept for potential picking.
    #[allow(dead_code)]
    pub cpu_scales: Vec<[f32; 3]>,
}

impl GaussianSplatGpuSet {
    /// Resident GPU bytes for the persistent source buffers (position, scale,
    /// rotation, opacity, SH). Per-viewport sort scratch is derived and grows
    /// lazily, so it is not counted here.
    pub fn gpu_bytes(&self) -> u64 {
        self.position_buf.size()
            + self.scale_buf.size()
            + self.rotation_buf.size()
            + self.opacity_buf.size()
            + self.sh_buf.size()
    }
}

/// Per-frame draw data produced in prepare_viewport_internal.
pub(crate) struct GaussianSplatDrawData {
    /// Index into gaussian_splat_store.
    pub store_index: usize,
    /// Viewport index that prepared this data.
    pub viewport_index: usize,
    /// Model matrix for this item.
    #[allow(dead_code)]
    pub model: [[f32; 4]; 4],
    /// Number of splats.
    pub count: u32,
    /// When true, skip the splat rasterization draw; a wireframe polyline overlay is rendered instead.
    pub wireframe: bool,
}

/// One splat slot: the set (when occupied) and the slot's current generation.
pub(crate) struct GaussianSplatSlot {
    pub set: Option<GaussianSplatGpuSet>,
    pub generation: u32,
}

/// Slotted store for Gaussian splat sets with generational handles.
///
/// A removed set leaves an empty slot that a later insert reuses. Each slot
/// carries a generation bumped on removal, and a [`GaussianSplatId`] captures
/// the generation it was issued against, so a stale handle resolves to `None`
/// rather than aliasing the set now in its slot.
pub(crate) struct GaussianSplatStore {
    pub slots: Vec<GaussianSplatSlot>,
    free_list: Vec<usize>,
    allocated_bytes: u64,
}

impl GaussianSplatStore {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            allocated_bytes: 0,
        }
    }

    pub fn insert(&mut self, set: GaussianSplatGpuSet) -> crate::renderer::GaussianSplatId {
        self.allocated_bytes += set.gpu_bytes();
        if let Some(idx) = self.free_list.pop() {
            let slot = &mut self.slots[idx];
            slot.set = Some(set);
            crate::renderer::GaussianSplatId::new(idx as u32, slot.generation)
        } else {
            let idx = self.slots.len();
            self.slots.push(GaussianSplatSlot {
                set: Some(set),
                generation: 0,
            });
            crate::renderer::GaussianSplatId::new(idx as u32, 0)
        }
    }

    /// Swap the set in `id`'s slot for `set`, keeping the slot generation so the
    /// handle stays valid. Returns `true` on success, `false` for a stale handle
    /// or an empty slot.
    pub fn replace(
        &mut self,
        id: crate::renderer::GaussianSplatId,
        set: GaussianSplatGpuSet,
    ) -> bool {
        if let Some(slot) = self.slots.get_mut(id.index as usize) {
            if slot.generation == id.generation && slot.set.is_some() {
                if let Some(old) = &slot.set {
                    self.allocated_bytes = self.allocated_bytes.saturating_sub(old.gpu_bytes());
                }
                self.allocated_bytes += set.gpu_bytes();
                slot.set = Some(set);
                return true;
            }
        }
        false
    }

    /// Total resident GPU bytes across every live splat set.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Look up a set by handle, validating the generation. Returns `None` for a
    /// stale handle, an empty slot, or an out-of-range index.
    pub fn get(&self, id: crate::renderer::GaussianSplatId) -> Option<&GaussianSplatGpuSet> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.set.as_ref()
    }

    /// Look up a set by raw slot index, without a generation check. For the
    /// per-frame draw path, where the index was already validated through
    /// [`get`](Self::get) earlier in the same frame.
    pub fn get_by_index(&self, idx: usize) -> Option<&GaussianSplatGpuSet> {
        self.slots.get(idx)?.set.as_ref()
    }

    /// Mutable raw-index lookup, same contract as [`get_by_index`](Self::get_by_index).
    pub fn get_mut_by_index(&mut self, idx: usize) -> Option<&mut GaussianSplatGpuSet> {
        self.slots.get_mut(idx)?.set.as_mut()
    }

    /// Remove a set by handle, bumping the slot generation and freeing the slot.
    /// Returns `true` if a set was removed, `false` for a stale handle or an
    /// already-empty slot.
    pub fn remove(&mut self, id: crate::renderer::GaussianSplatId) -> bool {
        if let Some(slot) = self.slots.get_mut(id.index as usize) {
            if slot.generation == id.generation && slot.set.is_some() {
                if let Some(old) = &slot.set {
                    self.allocated_bytes = self.allocated_bytes.saturating_sub(old.gpu_bytes());
                }
                slot.set = None;
                slot.generation = slot.generation.wrapping_add(1);
                self.free_list.push(id.index as usize);
                return true;
            }
        }
        false
    }
}

/// Per-frame GPU data for one polyline item, created in `prepare()`.
#[derive(Clone)]
pub struct PolylineGpuData {
    /// Instance buffer: `[xa, ya, za, xb, yb, zb, scalar_a, scalar_b]` per segment (32 bytes).
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Number of line segments (instances).  Draw call: `draw(0..6, 0..segment_count)`.
    pub(crate) segment_count: u32,
    /// Bind group (group 1): uniform + LUT texture + sampler.
    pub(crate) bind_group: wgpu::BindGroup,
    // Keep the uniform buffer alive for the lifetime of this struct.
    pub(crate) _uniform_buf: wgpu::Buffer,
    /// When true, renders with the clip-exempt pipeline (no clip plane or clip volume test).
    /// Used for clip object wireframe overlays that must always be fully visible.
    pub(crate) skip_clip: bool,
    /// When true, render as thin 1px lines using the wireframe pipeline instead of thick billboards.
    pub(crate) wireframe: bool,
    /// Bind group for the wireframe pipeline (group 1: segment storage buffer).
    /// None when the wireframe pipeline has not been created yet.
    pub(crate) wireframe_bind_group: Option<wgpu::BindGroup>,
}

/// Per-frame GPU data for one screen-space image overlay, created in `prepare()`.
pub struct ScreenImageGpuData {
    /// Uniform buffer: `ScreenImageUniform` (32 bytes) with NDC extents and alpha.
    pub(crate) _uniform_buf: wgpu::Buffer,
    /// Uploaded RGBA8 texture for this image (recreated each frame).
    pub(crate) _texture: wgpu::Texture,
    /// Bind group (group 0): uniform + colour texture + sampler.
    /// Used by the regular pipeline (no depth test).
    pub(crate) bind_group: wgpu::BindGroup,
    /// Uploaded R32Float depth texture. `None` when the item has no depth data.
    pub(crate) _depth_texture: Option<wgpu::Texture>,
    /// Bind group for the depth-composite pipeline (group 0: uniform + colour + sampler + depth).
    /// `Some` only when the item carries per-pixel depth data.
    pub(crate) depth_bind_group: Option<wgpu::BindGroup>,
}

/// Per-frame GPU data for one glyph item, created in `prepare()`.
#[derive(Clone)]
pub struct GlyphGpuData {
    /// Vertex buffer for the glyph base mesh (borrowed from cached `GlyphBaseMesh`).
    /// We keep a reference via raw pointer : `ViewportGpuResources` owns the mesh.
    /// Safety: the mesh lives as long as `ViewportGpuResources`.
    pub(crate) mesh_vertex_buffer: &'static wgpu::Buffer,
    /// Triangle index buffer for the glyph base mesh.
    pub(crate) mesh_index_buffer: &'static wgpu::Buffer,
    /// Number of triangle mesh indices.
    pub(crate) mesh_index_count: u32,
    /// Edge index buffer for wireframe LineList rendering (borrowed from cached `GlyphBaseMesh`).
    pub(crate) mesh_edge_index_buffer: &'static wgpu::Buffer,
    /// Number of edge indices.
    pub(crate) mesh_edge_index_count: u32,
    /// Number of glyph instances.
    pub(crate) instance_count: u32,
    /// Whether this batch should be drawn with the wireframe pipeline.
    pub(crate) wireframe: bool,
    /// Bind group (group 1): glyph uniform + LUT texture + sampler.
    pub(crate) uniform_bind_group: wgpu::BindGroup,
    /// Bind group (group 2): instance storage buffer.
    pub(crate) instance_bind_group: wgpu::BindGroup,
    // Keep the buffers alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
    pub(crate) _instance_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one tensor glyph item, created in `prepare()`.
///
/// The sphere base mesh is borrowed from `glyph_sphere_mesh` (owned by `ViewportGpuResources`).
#[derive(Clone)]
pub struct TensorGlyphGpuData {
    /// Vertex buffer for the sphere base mesh (borrowed).
    pub(crate) mesh_vertex_buffer: &'static wgpu::Buffer,
    /// Triangle index buffer for the sphere base mesh (borrowed).
    pub(crate) mesh_index_buffer: &'static wgpu::Buffer,
    /// Number of triangle mesh indices.
    pub(crate) mesh_index_count: u32,
    /// Edge index buffer for wireframe LineList rendering (borrowed from sphere mesh).
    pub(crate) mesh_edge_index_buffer: &'static wgpu::Buffer,
    /// Number of edge indices.
    pub(crate) mesh_edge_index_count: u32,
    /// Number of tensor glyph instances.
    pub(crate) instance_count: u32,
    /// Whether this batch should be drawn with the wireframe pipeline.
    pub(crate) wireframe: bool,
    /// Bind group (group 1): uniform + LUT texture + sampler.
    pub(crate) uniform_bind_group: wgpu::BindGroup,
    /// Bind group (group 2): per-instance storage buffer.
    pub(crate) instance_bind_group: wgpu::BindGroup,
    // Keep buffers alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
    pub(crate) _instance_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one streamtube item, created in `prepare()`.
///
/// The connected tube mesh (vertices + indices) is generated CPU-side for the
/// entire item (all strips) and uploaded as a single owned buffer pair.
#[derive(Clone)]
pub struct StreamtubeGpuData {
    /// Owned vertex buffer for the connected tube mesh (world-space positions + normals).
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Owned index buffer for the connected tube mesh (triangle indices).
    pub(crate) index_buffer: wgpu::Buffer,
    /// Number of triangle indices to draw (solid mode).
    pub(crate) index_count: u32,
    /// Owned index buffer for wireframe edges (deduplicated line-list pairs).
    pub(crate) edge_index_buffer: wgpu::Buffer,
    /// Number of edge indices to draw (wireframe mode).
    pub(crate) edge_index_count: u32,
    /// Whether this item should be drawn in wireframe mode.
    pub(crate) wireframe: bool,
    /// Bind group (group 1): tube uniform (colour, radius).
    pub(crate) uniform_bind_group: wgpu::BindGroup,
    /// Blend mode for the draw. Streamtubes always set this to
    /// `SpriteBlend::AlphaBlend`; ribbons honour the value from `RibbonItem`.
    pub(crate) blend: crate::renderer::SpriteBlend,
    // Keep uniform buffer alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one volume item, created in `prepare()`.
pub struct VolumeGpuData {
    /// Bind group (group 1): volume uniform + 3D texture + sampler + colour LUT + opacity LUT.
    pub(crate) bind_group: wgpu::BindGroup,
    /// Vertex buffer for the unit cube bounding box proxy.
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Index buffer for the unit cube (36 indices).
    pub(crate) index_buffer: wgpu::Buffer,
    /// Grid dimensions (stored for reference).
    pub(crate) _dims: [u32; 3],
    // Keep the uniform buffer alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
    /// When true, skip the volume ray-march draw; an OBB wireframe polyline is rendered instead.
    pub(crate) wireframe: bool,
}

/// Per-frame GPU data for one image slice item, created in `prepare()`.
pub(crate) struct ImageSliceGpuData {
    /// Bind group (group 1): uniform + 3D texture + sampler + LUT + LUT sampler.
    pub(crate) bind_group: wgpu::BindGroup,
    // Keep buffers/samplers alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
}

/// Per-frame GPU data for one volume surface slice item, created in `prepare()`.
pub(crate) struct VolumeSurfaceSliceGpuData {
    /// Bind group (group 1): uniform + 3D texture + sampler + LUT + LUT sampler.
    pub(crate) bind_group: wgpu::BindGroup,
    // Keep uniform buffer alive.
    pub(crate) _uniform_buf: wgpu::Buffer,
    /// Mesh to draw (vertex + index buffers looked up from mesh_store at render time).
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
}

// ---------------------------------------------------------------------------
// Projected tetrahedra GPU data types
// ---------------------------------------------------------------------------

/// Uniform buffer layout for the projected tetrahedra pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ProjectedTetUniform {
    pub(crate) density: f32,
    pub(crate) scalar_min: f32,
    pub(crate) scalar_max: f32,
    pub(crate) threshold_min: f32,
    pub(crate) threshold_max: f32,
    /// 1 = skip Beer-Lambert thickness modulation and emit a flat density-scaled
    /// alpha per visible fragment. Wired from `ItemSettings.unlit`.
    pub(crate) unlit: u32,
    /// Multiplied into the final alpha. Wired from `ItemSettings.opacity`.
    pub(crate) opacity: f32,
    pub(crate) _pad: f32,
}

/// One device-limit-bounded chunk of a projected-tet mesh.
pub(crate) struct ProjectedTetChunk {
    /// Storage buffer for this chunk's tetrahedra (kept alive for the bind group).
    #[allow(dead_code)]
    pub tet_buffer: wgpu::Buffer,
    /// Number of tetrahedra in this chunk (= instanced draw count).
    pub tet_count: u32,
    /// Bind group: shared uniform + this chunk's tet buffer + colourmap + sampler.
    pub bind_group: wgpu::BindGroup,
}

/// Uploaded projected-tetrahedra mesh, stored persistently on the GPU.
///
/// Large meshes are split into multiple chunks so each storage buffer
/// stays within `max_storage_buffer_binding_size`.
/// Created by [`ViewportGpuResources::upload_projected_tet`].
pub(crate) struct GpuProjectedTetMesh {
    /// One or more device-limit-bounded chunks.
    pub chunks: Vec<ProjectedTetChunk>,
    /// Uniform buffer shared across all chunks (density, scalar_min/max). Written each frame.
    pub uniform_buffer: wgpu::Buffer,
    /// Auto-detected scalar range from the uploaded data (min, max).
    pub scalar_range: (f32, f32),
}

// ---------------------------------------------------------------------------
// Surface LIC GPU data types
// ---------------------------------------------------------------------------

/// Uniform for the LIC advect render pass (step counts and viewport dims).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LicAdvectUniform {
    pub(crate) steps: u32,
    pub(crate) step_size: f32,
    pub(crate) vp_width: f32,
    pub(crate) vp_height: f32,
}

/// Uniform for the LIC surface pass (model matrix per object, 64 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LicObjectUniform {
    pub(crate) model: [[f32; 4]; 4],
}

/// Per-frame GPU data for one Surface LIC item, created in `prepare()`.
pub struct LicSurfaceGpuData {
    /// Bind group (group 1): LicObjectUniform only. Flow vectors bound as vertex buffer 1.
    pub(crate) bind_group: wgpu::BindGroup,
    /// Owned uniform buffer for the model matrix. Kept alive by this struct.
    pub(crate) _object_uniform_buf: wgpu::Buffer,
    /// MeshId used to look up vertex + index buffers in the render pass.
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Name of the flow vector attribute for looking up the vertex buffer in the render pass.
    pub(crate) vector_attribute: String,
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

// ---------------------------------------------------------------------------
// ViewportGpuResources: top-level GPU resource container
// ---------------------------------------------------------------------------

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

/// All GPU resources for the 3D viewport.
///
/// Typically stored in the host framework's resource container and accessed
/// by `ViewportRenderer` during prepare() and paint().
#[allow(dead_code)]
pub struct ViewportGpuResources {
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
    /// Cache of material bind groups keyed by (albedo_id, normal_map_id, ao_map_id).
    /// u64::MAX sentinel = use fallback texture for that slot.
    #[allow(dead_code)]
    pub(crate) material_bind_groups: std::collections::HashMap<(u64, u64, u64), wgpu::BindGroup>,
    /// User-uploaded textures, keyed by the `texture_id` in Material. Slotted
    /// with generational ids so a freed slot cannot alias a later upload.
    pub(crate) textures: crate::resources::material::texture_store::TextureStore,
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

    // --- Matcap texture system ---
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
    /// Billboard disc pipeline for the Gaussian splat outline mask pass.
    pub(crate) splat_outline_mask_pipeline: wgpu::RenderPipeline,
    /// Mask-write pipeline for volume AABB cubes (position-only vertex layout, no face culling).
    pub(crate) volume_outline_mask_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced mask pipeline for arrow/sphere glyph outlines (reuses glyph bind groups).
    pub(crate) glyph_outline_mask_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced mask pipeline for tensor glyph outlines (reuses tensor glyph bind groups).
    pub(crate) tensor_glyph_outline_mask_pipeline: Option<wgpu::RenderPipeline>,
    // --- Outline offscreen resources (lazily created) ---
    /// Offscreen RGBA texture the outline stencil pass renders into.
    pub(crate) outline_colour_texture: Option<wgpu::Texture>,
    pub(crate) outline_colour_view: Option<wgpu::TextureView>,
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
    /// Two-sided (`cull_mode: None`) variant of `solid_instanced_pipeline` for
    /// `Identical` backface-policy meshes.
    pub(crate) solid_two_sided_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced transparent render pipeline (TriangleList, alpha blending).
    pub(crate) transparent_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced shadow render pipeline (depth-only).
    pub(crate) shadow_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None` + two-sided depth bias) variant of
    /// `shadow_instanced_pipeline` for `Identical` backface-policy batches.
    pub(crate) shadow_instanced_two_sided_pipeline: Option<wgpu::RenderPipeline>,
    /// Per-cascade uniform buffers for shadow_instanced_pipeline (64 bytes each, one mat4x4).
    pub(crate) shadow_instanced_cascade_bufs: [Option<wgpu::Buffer>; 4],
    /// Per-cascade bind groups for shadow_instanced_pipeline group 0.
    pub(crate) shadow_instanced_cascade_bgs: [Option<wgpu::BindGroup>; 4],

    // --- GPU culling inputs (scene-global, camera-independent) ---
    // The cull OUTPUTS (visibility indices, indirect args, batch counters, and
    // their bind groups) are per-viewport and live in `ViewportCullState` on
    // each `ViewportSlot`, not here.
    /// Per-instance world-space AABB buffer. Rebuilt on batch cache miss.
    pub(crate) instance_aabb_buf: Option<wgpu::Buffer>,
    pub(crate) instance_aabb_capacity: usize,
    /// Per-batch metadata buffer. Rebuilt on batch cache miss.
    pub(crate) batch_meta_buf: Option<wgpu::Buffer>,
    pub(crate) batch_meta_capacity: usize,

    // --- GPU culling pipelines ---
    /// Bind group layout for instanced cull pipelines (group 1).
    /// Extends `instance_bgl` with binding 5: visibility_indices storage buffer.
    pub(crate) instance_cull_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// HDR-pass solid instanced pipeline using `vs_main_cull` (indirect draw path).
    pub(crate) hdr_solid_instanced_cull_pipeline: Option<wgpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None`) variant of `hdr_solid_instanced_cull_pipeline`
    /// for `Identical` backface-policy meshes (indirect draw path).
    pub(crate) hdr_solid_instanced_cull_two_sided_pipeline: Option<wgpu::RenderPipeline>,
    /// OIT-pass transparent instanced pipeline using `vs_main_cull` (indirect draw path).
    pub(crate) oit_instanced_cull_pipeline: Option<wgpu::RenderPipeline>,

    // --- GPU culling : shadow cascade extension ---
    /// Shadow instanced cull pipeline (depth-only, uses `vs_shadow_cull`).
    pub(crate) shadow_instanced_cull_pipeline: Option<wgpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None` + two-sided depth bias) variant of
    /// `shadow_instanced_cull_pipeline` for `Identical` backface-policy batches.
    pub(crate) shadow_instanced_cull_two_sided_pipeline: Option<wgpu::RenderPipeline>,
    /// BGL for shadow cull instance group: binding 0 (instances) + binding 5 (visibility_indices).
    pub(crate) shadow_cull_instance_bgl: Option<wgpu::BindGroupLayout>,

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

    // --- Post-processing shared pipelines / BGLs ---
    // The viewport-sized textures, bind groups, and per-frame uniform buffers
    // for these passes live in per-viewport ViewportHdrState. Only the shared,
    // camera-independent pipelines and layouts sit here.
    /// Tone mapping pipeline (renders fullscreen tri, hdr_texture -> output).
    pub(crate) tone_map_pipeline: Option<wgpu::RenderPipeline>,
    /// Tone map bind group layout.
    pub(crate) tone_map_bgl: Option<wgpu::BindGroupLayout>,

    /// Shared bloom pipelines (threshold + blur use the same BGL, different bind groups).
    pub(crate) bloom_threshold_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) bloom_blur_pipeline: Option<wgpu::RenderPipeline>,

    /// 4x4 random rotation noise texture (Rgba8Unorm, REPEAT). Shared across viewports.
    pub(crate) ssao_noise_texture: Option<wgpu::Texture>,
    pub(crate) ssao_noise_view: Option<wgpu::TextureView>,
    /// 64-sample hemisphere kernel (storage buffer, `vec4<f32>` per sample).
    pub(crate) ssao_kernel_buf: Option<wgpu::Buffer>,
    pub(crate) ssao_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) ssao_blur_pipeline: Option<wgpu::RenderPipeline>,

    // --- Depth of field shared resources ---
    pub(crate) dof_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) dof_bgl: Option<wgpu::BindGroupLayout>,

    // --- Contact shadow shared resources ---
    pub(crate) contact_shadow_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) contact_shadow_bgl: Option<wgpu::BindGroupLayout>,

    // --- Surface LIC shared resources ---
    /// Render pipeline: renders mesh with vector storage buffer -> lic_vector_texture (Rgba8Unorm).
    pub(crate) lic_surface_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for group 1 of the LIC surface pass (object uniform + vector buffer + noise).
    pub(crate) lic_surface_bgl: Option<wgpu::BindGroupLayout>,
    /// Render pipeline: reads lic_vector_texture, writes LIC intensity to R8Unorm target.
    pub(crate) lic_advect_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the LIC advect pass.
    pub(crate) lic_advect_bgl: Option<wgpu::BindGroupLayout>,
    /// Bilinear sampler for the LIC advect pass (used to sample lic_vector_texture).
    pub(crate) lic_noise_sampler: Option<wgpu::Sampler>,
    /// 1x1 R8Unorm white placeholder bound to tone_map binding 7 when LIC is not active.
    pub(crate) lic_placeholder_view: Option<wgpu::TextureView>,

    /// 1x1 black Rgba16Float placeholder used when bloom is disabled.
    pub(crate) bloom_placeholder_view: Option<wgpu::TextureView>,
    /// 1x1 white R8Unorm placeholder used when SSAO is disabled.
    pub(crate) ao_placeholder_view: Option<wgpu::TextureView>,
    /// 1x1 white R8Unorm placeholder used when contact shadows are disabled.
    pub(crate) cs_placeholder_view: Option<wgpu::TextureView>,

    /// Shared post-process linear-clamp sampler.
    pub(crate) pp_linear_sampler: Option<wgpu::Sampler>,
    /// Shared post-process nearest-clamp sampler (for depth).
    pub(crate) pp_nearest_sampler: Option<wgpu::Sampler>,

    /// HDR-format variants of core scene pipelines.
    pub(crate) hdr_solid_pipeline: Option<wgpu::RenderPipeline>,
    /// HDR two-sided variant (cull_mode: None) for analytical surfaces.
    pub(crate) hdr_solid_two_sided_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_transparent_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_wireframe_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_solid_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None`) variant of `hdr_solid_instanced_pipeline`
    /// for `Identical` backface-policy meshes (direct draw path).
    pub(crate) hdr_solid_two_sided_instanced_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) hdr_transparent_instanced_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced HDR pipeline with additive blend, no depth write. Used by
    /// `MeshInstanceItem` batches that opt into [`SpriteBlend::Additive`].
    pub(crate) hdr_instanced_additive_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced HDR pipeline with premultiplied-alpha blend, no depth write.
    /// Used by `MeshInstanceItem` batches with [`SpriteBlend::Premultiplied`].
    pub(crate) hdr_instanced_premultiplied_pipeline: Option<wgpu::RenderPipeline>,
    /// HDR overlay pipeline (TriangleList, Rgba16Float, alpha blending) for cap fill in HDR path.
    pub(crate) hdr_overlay_pipeline: Option<wgpu::RenderPipeline>,

    // --- Colourmap / LUT resources ---
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

    // --- Gaussian splat pipelines (lazily created) ---
    /// Gaussian splat render pipeline. None until first splat set is submitted.
    pub(crate) gaussian_splat_pipeline: Option<DualPipeline>,
    /// Bind group layout for group 1 of the Gaussian splat render pipeline.
    pub(crate) gaussian_splat_bgl: Option<wgpu::BindGroupLayout>,
    /// Compute pipeline for computing view-space depth values per splat.
    pub(crate) gaussian_splat_depth_pipeline: Option<wgpu::ComputePipeline>,
    /// Compute pipeline for clearing the sort histogram.
    pub(crate) gaussian_splat_sort_clear_pipeline: Option<wgpu::ComputePipeline>,
    /// Compute pipeline for the radix sort histogram pass.
    pub(crate) gaussian_splat_sort_histogram_pipeline: Option<wgpu::ComputePipeline>,
    /// Compute pipeline for the radix sort prefix sum pass.
    pub(crate) gaussian_splat_sort_prefix_pipeline: Option<wgpu::ComputePipeline>,
    /// Compute pipeline for the radix sort scatter pass.
    pub(crate) gaussian_splat_sort_scatter_pipeline: Option<wgpu::ComputePipeline>,
    /// Compute pipeline for initializing sort index values.
    pub(crate) gaussian_splat_sort_init_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for the depth compute pass.
    pub(crate) gaussian_splat_depth_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for the sort compute passes.
    pub(crate) gaussian_splat_sort_bgl: Option<wgpu::BindGroupLayout>,
    /// Slotted store of all uploaded Gaussian splat sets.
    pub(crate) gaussian_splat_store: GaussianSplatStore,

    // --- Sprite billboard pipelines (lazily created) ---
    /// Sprite pipeline, alpha-blend, depth_write_enabled: false.
    pub(crate) sprite_pipeline: Option<DualPipeline>,
    /// Sprite pipeline, alpha-blend, depth_write_enabled: true.
    pub(crate) sprite_pipeline_depth_write: Option<DualPipeline>,
    /// Sprite pipeline, additive, depth_write_enabled: false.
    pub(crate) sprite_pipeline_additive: Option<DualPipeline>,
    /// Sprite pipeline, additive, depth_write_enabled: true.
    pub(crate) sprite_pipeline_additive_depth_write: Option<DualPipeline>,
    /// Sprite pipeline, premultiplied, depth_write_enabled: false.
    pub(crate) sprite_pipeline_premultiplied: Option<DualPipeline>,
    /// Sprite pipeline, premultiplied, depth_write_enabled: true.
    pub(crate) sprite_pipeline_premultiplied_depth_write: Option<DualPipeline>,
    /// Refractive sprite pipeline (HDR target only). Samples the scene-colour
    /// resolve at an offset driven by the sprite's texture and writes back
    /// with alpha-blend modulated by the texture's alpha channel.
    pub(crate) sprite_refraction_pipeline: Option<wgpu::RenderPipeline>,
    /// Group 2 BGL for the refraction pipeline: scene-colour texture + sampler.
    pub(crate) sprite_refraction_bgl: Option<wgpu::BindGroupLayout>,
    /// Sampler used by the refraction shader to read the scene-colour resolve.
    pub(crate) sprite_refraction_sampler: Option<wgpu::Sampler>,
    /// Bind group layout for sprite uniforms + texture + instance buffer (group 1).
    pub(crate) sprite_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for the per-pass scene-depth resolve bound at group 2.
    /// One sampleable depth texture plus a sampler. Bound during sprite draws so
    /// the fragment shader can apply soft-particle fade against opaque geometry.
    pub(crate) sprite_soft_bgl: Option<wgpu::BindGroupLayout>,
    /// Fallback bind group for the group-2 soft-particle binding, used by paths
    /// that do not have a resolved scene-depth texture available. Its contents
    /// are only sampled when the per-batch `soft_particle_distance` is set, so
    /// callers that never enable soft fade can rely on the fallback bind alone.
    pub(crate) sprite_soft_fallback_bg: Option<wgpu::BindGroup>,
    /// Sampler used for the group-2 scene-depth binding. Created alongside the
    /// fallback bind group.
    pub(crate) sprite_soft_sampler: Option<wgpu::Sampler>,
    /// 1x1 Depth32Float texture backing the fallback bind group. Held to keep
    /// the underlying texture alive for the lifetime of the bind group.
    pub(crate) sprite_soft_fallback_tex: Option<wgpu::Texture>,
    /// Lit sprite pipelines: one per (blend, depth_write) pair. Same pipeline
    /// layout as the emissive sprite, but a different shader module that
    /// samples the scene lighting and an optional tangent-space normal map.
    pub(crate) sprite_lit_pipeline: Option<DualPipeline>,
    /// Lit sprite pipeline, alpha-blend, depth_write_enabled: true.
    pub(crate) sprite_lit_pipeline_depth_write: Option<DualPipeline>,
    /// Lit sprite pipeline, additive, depth_write_enabled: false.
    pub(crate) sprite_lit_pipeline_additive: Option<DualPipeline>,
    /// Lit sprite pipeline, additive, depth_write_enabled: true.
    pub(crate) sprite_lit_pipeline_additive_depth_write: Option<DualPipeline>,
    /// Lit sprite pipeline, premultiplied, depth_write_enabled: false.
    pub(crate) sprite_lit_pipeline_premultiplied: Option<DualPipeline>,
    /// Lit sprite pipeline, premultiplied, depth_write_enabled: true.
    pub(crate) sprite_lit_pipeline_premultiplied_depth_write: Option<DualPipeline>,
    /// Bind group layout for the optional tangent-space normal map sampled by
    /// the lit sprite shader. Group 3: texture + sampler.
    pub(crate) sprite_lit_bgl: Option<wgpu::BindGroupLayout>,
    /// Fallback bind group for the lit normal map binding. Backed by a 1x1
    /// `(128, 128, 255)` texture so `NormalMap` mode falls back to a flat
    /// normal when no map is supplied.
    pub(crate) sprite_lit_fallback_bg: Option<wgpu::BindGroup>,
    /// 1x1 RGBA8Unorm texture backing the lit fallback bind group.
    pub(crate) sprite_lit_fallback_tex: Option<wgpu::Texture>,
    /// Sprite outline mask pipeline (R8Unorm, no texture sampling). None until first selected sprite.
    pub(crate) sprite_outline_mask_pipeline: Option<wgpu::RenderPipeline>,
    /// Polyline outline mask pipeline (R8Unorm, same instance layout as polyline). None until first selected polyline.
    pub(crate) polyline_outline_mask_pipeline: Option<wgpu::RenderPipeline>,

    // --- point cloud and glyph pipelines (lazily created) ---
    /// Point cloud render pipeline. None until first point cloud is submitted.
    pub(crate) point_cloud_pipeline: Option<DualPipeline>,
    /// Glyph render pipeline. None until first glyph set is submitted.
    pub(crate) glyph_pipeline: Option<DualPipeline>,
    /// Glyph wireframe pipeline (LineList, same bind groups as glyph_pipeline).
    pub(crate) glyph_wireframe_pipeline: Option<DualPipeline>,
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

    // --- Tensor glyph rendering (lazily created) ---
    /// Tensor glyph render pipeline. None until first tensor glyph set is submitted.
    pub(crate) tensor_glyph_pipeline: Option<DualPipeline>,
    /// Tensor glyph wireframe pipeline (LineList, same bind groups as tensor_glyph_pipeline).
    pub(crate) tensor_glyph_wireframe_pipeline: Option<DualPipeline>,
    /// Bind group layout for tensor glyph uniforms (group 1).
    pub(crate) tensor_glyph_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for tensor glyph instance storage (group 2).
    pub(crate) tensor_glyph_instance_bgl: Option<wgpu::BindGroupLayout>,

    // --- polyline rendering (lazily created) ---
    /// Polyline render pipeline. None until first polyline set is submitted.
    pub(crate) polyline_pipeline: Option<DualPipeline>,
    /// Clip-exempt polyline pipeline: same as polyline_pipeline but uses fs_main_no_clip
    /// so clip overlay wireframes are always fully visible.
    pub(crate) polyline_no_clip_pipeline: Option<DualPipeline>,
    /// Bind group layout for polyline uniforms (group 1).
    pub(crate) polyline_bgl: Option<wgpu::BindGroupLayout>,
    /// Wireframe polyline pipeline: thin 1px LineList, reads segment endpoints from a
    /// storage buffer. Created alongside polyline_pipeline.
    pub(crate) polyline_wireframe_pipeline: Option<DualPipeline>,
    /// Bind group layout for the wireframe polyline pipeline (group 1: segment storage buffer).
    pub(crate) polyline_wireframe_bgl: Option<wgpu::BindGroupLayout>,

    // --- streamtube rendering (lazily created) ---
    /// Streamtube render pipeline. None until first streamtube item is submitted.
    pub(crate) streamtube_pipeline: Option<DualPipeline>,
    /// Streamtube wireframe pipeline (LineList topology, cull_mode None). None until first wireframe streamtube.
    pub(crate) streamtube_wireframe_pipeline: Option<DualPipeline>,
    /// Ribbon pipeline: same layout as streamtube but cull_mode None and two-sided normals.
    /// Standard alpha blend, depth write enabled. Default for plain ribbons.
    pub(crate) ribbon_pipeline: Option<DualPipeline>,
    /// Ribbon pipeline with additive blend and depth write disabled. Picked
    /// when `RibbonItem::blend == SpriteBlend::Additive`, the right look for
    /// energy or spark trails.
    pub(crate) ribbon_pipeline_additive: Option<DualPipeline>,
    /// Ribbon pipeline with premultiplied-alpha blend and depth write disabled.
    /// Picked when `RibbonItem::blend == SpriteBlend::Premultiplied`.
    pub(crate) ribbon_pipeline_premultiplied: Option<DualPipeline>,
    /// Ribbon wireframe pipeline (LineList topology, cull_mode None).
    pub(crate) ribbon_wireframe_pipeline: Option<DualPipeline>,
    /// Bind group layout for streamtube uniforms (group 1).
    pub(crate) streamtube_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for ribbons (group 1): uniform + optional streak
    /// texture + sampler. Distinct from `streamtube_bgl` so streamtubes do not
    /// pay for a texture binding they never sample.
    pub(crate) ribbon_bgl: Option<wgpu::BindGroupLayout>,

    // --- Image slice rendering (lazily created) ---
    /// Image slice render pipeline. None until first slice item is submitted.
    pub(crate) image_slice_pipeline: Option<DualPipeline>,
    /// Bind group layout for image slice uniforms (group 1).
    pub(crate) image_slice_bgl: Option<wgpu::BindGroupLayout>,

    // --- Volume surface slice rendering (lazily created) ---
    /// Volume surface slice render pipeline. None until first item is submitted.
    pub(crate) volume_surface_slice_pipeline: Option<DualPipeline>,
    /// Bind group layout for volume surface slice uniforms (group 1).
    pub(crate) volume_surface_slice_bgl: Option<wgpu::BindGroupLayout>,

    // --- volume rendering (lazily created) ---
    /// Uploaded 3D volume textures. Index = VolumeId value.
    pub(crate) volume_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    /// Volume render pipeline. None until first volume is submitted.
    pub(crate) volume_pipeline: Option<DualPipeline>,
    /// Bind group layout for volume uniforms (group 1).
    pub(crate) volume_bgl: Option<wgpu::BindGroupLayout>,
    /// Cached unit cube vertex+index buffers for bounding box rasterization.
    pub(crate) volume_cube_vb: Option<wgpu::Buffer>,
    pub(crate) volume_cube_ib: Option<wgpu::Buffer>,
    /// Default linear ramp opacity LUT texture (256x1, R8Unorm).
    pub(crate) volume_default_opacity_lut: Option<wgpu::Texture>,
    pub(crate) volume_default_opacity_lut_view: Option<wgpu::TextureView>,

    // --- GPU compute filtering (lazily created) ---
    /// Compute pipeline for Clip / Threshold index compaction. None until first use.
    pub(crate) compute_filter_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for the compute filter shader (group 0). None until first use.
    pub(crate) compute_filter_bgl: Option<wgpu::BindGroupLayout>,

    // --- Order-independent transparency (OIT) : lazily created ---
    // These fields are superseded by ViewportHdrState.oit_* but kept for ensure_oit_targets compat.
    #[allow(dead_code)]
    /// Weighted-blended accumulation texture (Rgba16Float, viewport-sized).
    pub(crate) oit_accum_texture: Option<wgpu::Texture>,
    pub(crate) oit_accum_view: Option<wgpu::TextureView>,
    /// Weighted-blended reveal (transmittance) texture (R8Unorm, viewport-sized).
    pub(crate) oit_reveal_texture: Option<wgpu::Texture>,
    pub(crate) oit_reveal_view: Option<wgpu::TextureView>,
    /// OIT mesh pipeline (non-instanced, mesh_oit.wgsl, two colour targets).
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

    // --- Projected tetrahedra transparent volume rendering (lazily created) ---
    /// Render pipeline for the projected tetrahedra pass. None until first item submitted.
    pub(crate) pt_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for group 1 of the PT pipeline (per-volume uniform + tet storage buffer).
    pub(crate) pt_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for group 2 of the PT pipeline (per-frame colourmap LUT + sampler).
    pub(crate) pt_lut_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Cache of LUT bind groups keyed by colourmap slot index. Rebuilt lazily as
    /// new colourmaps are seen on transparent volume meshes; reset to empty when
    /// new colourmaps are uploaded.
    pub(crate) pt_lut_bind_groups: std::collections::HashMap<usize, wgpu::BindGroup>,
    /// LUT bind group for the fallback colourmap (used when an item has no
    /// `colourmap_id`). Lazily built on first use.
    pub(crate) pt_fallback_lut_bind_group: Option<wgpu::BindGroup>,
    /// Uploaded projected-tet meshes. Index = ProjectedTetId value.
    pub(crate) projected_tet_store: Vec<GpuProjectedTetMesh>,

    // --- Scatter-volume (participating media) rendering (lazily created) ---
    //
    // Pipeline layout uses four bind groups:
    //   group 0: shared camera (existing camera_bind_group_layout)
    //   group 1: per-volume `GpuScatterVolume` uniform with dynamic offset
    //   group 2: per-volume colourmap LUT + 3D density texture (cached by id)
    //   group 3: shared per-frame uniform + opaque depth + samplers
    //
    /// Render pipeline for the scatter-volume pass. None until first item submitted.
    pub(crate) scatter_pipeline: Option<wgpu::RenderPipeline>,
    /// Group 1 layout (per-volume uniform with dynamic offset).
    pub(crate) scatter_per_volume_bgl: Option<wgpu::BindGroupLayout>,
    /// Group 2 layout (per-volume LUT + density texture + samplers).
    pub(crate) scatter_per_volume_tex_bgl: Option<wgpu::BindGroupLayout>,
    /// Group 3 layout (per-frame uniform + opaque depth + samplers).
    pub(crate) scatter_frame_bgl: Option<wgpu::BindGroupLayout>,
    /// Per-volume uniform buffer holding the packed `GpuScatterVolume` array,
    /// stride-padded to `min_uniform_buffer_offset_alignment` for dynamic
    /// offsetting.
    pub(crate) scatter_per_volume_buffer: Option<wgpu::Buffer>,
    /// Bind group for the per-volume uniform (group 1). Rebuilt only when the
    /// buffer is reallocated.
    pub(crate) scatter_per_volume_bg: Option<wgpu::BindGroup>,
    /// Stride between dynamic-offset uniform slots, in bytes.
    pub(crate) scatter_per_volume_stride: u32,
    /// Capacity of `scatter_per_volume_buffer` in slots.
    pub(crate) scatter_per_volume_capacity: u32,
    /// Per-frame uniform buffer (group 3 binding 0).
    pub(crate) scatter_frame_uniform_buffer: Option<wgpu::Buffer>,
    /// Cache of group 2 bind groups, keyed by `(lut_id, density_id)`. Built
    /// on demand each frame; invalidated when the source texture vec grows.
    pub(crate) scatter_per_volume_tex_cache: Vec<((usize, usize), wgpu::BindGroup)>,
    /// Bind group for group 3, rebuilt when opaque depth view changes.
    pub(crate) scatter_frame_bg: Option<wgpu::BindGroup>,
    /// Linear sampler used to read opaque depth in the scatter pass.
    pub(crate) scatter_depth_sampler: Option<wgpu::Sampler>,
    /// Linear-clamp sampler used to read the colourmap LUT in the scatter pass.
    pub(crate) scatter_colourmap_sampler: Option<wgpu::Sampler>,
    /// 1x1x1 R32Float fallback view bound at the per-volume 3D density slot
    /// when a volume does not supply its own density texture.
    pub(crate) scatter_density_fallback_view: Option<wgpu::TextureView>,
    /// Token combining (depth view, frame uniform buffer) for `scatter_frame_bg`
    /// reuse.
    pub(crate) scatter_bound_depth: u64,
    /// Composite pipeline that samples a scatter intermediate (raw or history)
    /// and blends it onto the HDR target with premultiplied alpha-over.
    pub(crate) scatter_composite_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the composite pass (one sampled RGBA16F + sampler).
    pub(crate) scatter_composite_bgl: Option<wgpu::BindGroupLayout>,
    /// Bilinear-clamp sampler used by the composite pass.
    pub(crate) scatter_composite_sampler: Option<wgpu::Sampler>,
    /// Temporal-resolve pipeline: reads (raw_current, history_prev) and writes
    /// the mixed result to history_new. Half-res when downsample is on.
    pub(crate) scatter_temporal_resolve_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the temporal-resolve pass (uniform + raw + history
    /// + opaque depth + samplers).
    pub(crate) scatter_temporal_resolve_bgl: Option<wgpu::BindGroupLayout>,
    /// Per-frame uniform buffer for the temporal-resolve pass (mat4 + vec4
    /// temporal pack + viewport dims).
    pub(crate) scatter_temporal_resolve_uniform_buffer: Option<wgpu::Buffer>,
    /// Refraction pass: per-volume distortion using a noise-driven gradient.
    /// Writes the displaced scene colour back into the HDR target before the
    /// scatter pass runs.
    pub(crate) scatter_refraction_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the refraction pass's per-volume uniform.
    pub(crate) scatter_refraction_per_volume_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for the refraction pass's source-scene + depth bindings.
    pub(crate) scatter_refraction_source_bgl: Option<wgpu::BindGroupLayout>,
    /// Single dynamic-offset uniform buffer holding every refractive volume's
    /// packed parameters. Stride matches `min_uniform_buffer_offset_alignment`.
    pub(crate) scatter_refraction_per_volume_buffer: Option<wgpu::Buffer>,
    /// Cached stride between refractive-volume slots in
    /// `scatter_refraction_per_volume_buffer`.
    pub(crate) scatter_refraction_per_volume_stride: u32,
    /// Capacity (slot count) the per-volume buffer is currently sized for.
    pub(crate) scatter_refraction_per_volume_capacity: u32,
    /// Dynamic-offset bind group for the per-volume uniform buffer.
    pub(crate) scatter_refraction_per_volume_bg: Option<wgpu::BindGroup>,
    /// Blit pipeline that copies the HDR target into the refraction source
    /// texture before the per-volume distortion runs.
    pub(crate) scatter_refraction_blit_pipeline: Option<wgpu::RenderPipeline>,

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
    /// Render pipeline for GPU-side implicit surface ray-marching. None until first item submitted.
    pub(crate) implicit_pipeline: Option<DualPipeline>,
    /// Bind group layout for group 1 of the implicit pipeline (ImplicitUniformRaw).
    pub(crate) implicit_bgl: Option<wgpu::BindGroupLayout>,
    /// Outline mask pipeline for implicit surfaces (ray-march to R8Unorm mask). None until first selected item.
    pub(crate) implicit_outline_mask_pipeline: Option<wgpu::RenderPipeline>,

    // --- GPU marching cubes (lazily created) ---
    pub(crate) mc_classify_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) mc_prefix_sum_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) mc_generate_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) mc_surface_pipeline: Option<DualPipeline>,
    pub(crate) mc_wireframe_pipeline: Option<DualPipeline>,
    pub(crate) mc_wireframe_render_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) mc_classify_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) mc_prefix_sum_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) mc_generate_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) mc_render_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) mc_case_count_buf: Option<wgpu::Buffer>,
    pub(crate) mc_case_table_buf: Option<wgpu::Buffer>,
    pub(crate) mc_volumes: Vec<crate::resources::volume::gpu_marching_cubes::McVolumeGpuData>,
    /// Outline mask pipeline for MC surfaces (stride-24 vertex buffer, draw_indirect). None until first selected item.
    pub(crate) mc_outline_mask_pipeline: Option<wgpu::RenderPipeline>,

    // --- GPU particle systems ---
    /// Live particle systems indexed by `GpuParticleSystemId`. Slots can be
    /// reused after `drop_gpu_particle_system`.
    pub(crate) particle_systems: Vec<Option<crate::resources::gpu::gpu_particles::ParticleSystem>>,
    /// Bind group layout for the emit + sim compute pipelines (group 1:
    /// particle buffer + free-list buffer; emit/sim params are group 0).
    pub(crate) particle_sim_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for emit/sim params (group 0).
    pub(crate) particle_params_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for the particle-sprite draw pipeline (group 1).
    pub(crate) particle_draw_bgl: Option<wgpu::BindGroupLayout>,
    /// Compute pipeline that pops free-list slots and writes new particles.
    pub(crate) particle_emit_pipeline: Option<wgpu::ComputePipeline>,
    /// Compute pipeline that integrates forces and decrements lifetime.
    pub(crate) particle_sim_pipeline: Option<wgpu::ComputePipeline>,
    /// Draw pipeline variants for the particle-sprite shader, keyed by blend.
    pub(crate) particle_sprite_pipeline_alpha: Option<DualPipeline>,
    pub(crate) particle_sprite_pipeline_additive: Option<DualPipeline>,
    pub(crate) particle_sprite_pipeline_premultiplied: Option<DualPipeline>,
    /// Lit variants of the GPU particle sprite pipelines.
    pub(crate) particle_sprite_lit_pipeline_alpha: Option<DualPipeline>,
    pub(crate) particle_sprite_lit_pipeline_additive: Option<DualPipeline>,
    pub(crate) particle_sprite_lit_pipeline_premultiplied: Option<DualPipeline>,
    /// Group 2 BGL for the lit particle path: optional tangent-space normal
    /// map + filtering sampler.
    pub(crate) particle_sprite_lit_bgl: Option<wgpu::BindGroupLayout>,
    /// Fallback bind group for the lit particle normal-map binding (group 2).
    pub(crate) particle_sprite_lit_fallback_bg: Option<wgpu::BindGroup>,
    /// Bind group layout for the mesh-route particle draw pipeline (group 1).
    pub(crate) particle_mesh_draw_bgl: Option<wgpu::BindGroupLayout>,
    /// Draw pipeline variants for the particle-mesh shader, keyed by blend.
    pub(crate) particle_mesh_pipeline_alpha: Option<DualPipeline>,
    pub(crate) particle_mesh_pipeline_additive: Option<DualPipeline>,
    pub(crate) particle_mesh_pipeline_premultiplied: Option<DualPipeline>,

    // --- Screen-space image overlays (lazily created) ---
    /// Render pipeline for screen-space image quads. None until first screen image is submitted.
    pub(crate) screen_image_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the screen image pipeline (group 0: uniform + texture + sampler).
    pub(crate) screen_image_bgl: Option<wgpu::BindGroupLayout>,
    /// Depth-composite pipeline. Uses depth_compare: LessEqual and outputs
    /// frag_depth from a per-pixel image depth texture. None until first dc image is submitted.
    pub(crate) screen_image_dc_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the dc pipeline (group 0: uniform + colour tex + sampler + depth tex).
    pub(crate) screen_image_dc_bgl: Option<wgpu::BindGroupLayout>,
    /// Outline mask pipeline for screen-space rect images (NDC quad, R8Unorm target). None until first selected item.
    pub(crate) screen_rect_outline_mask_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for screen_rect_outline_mask_pipeline (binding 0: NdcRectUniform).
    pub(crate) screen_rect_outline_bgl: Option<wgpu::BindGroupLayout>,

    // --- GPU object-ID picking (lazily created) ---
    /// Render pipeline that outputs flat u32 object IDs to R32Uint + R32Float targets.
    /// `None` until `ensure_pick_pipeline` is first called.
    pub(crate) pick_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for group 1 of the pick pipeline (PickInstance storage buffer).
    pub(crate) pick_bind_group_layout_1: Option<wgpu::BindGroupLayout>,
    /// Minimal camera-only bind group layout for the pick pipeline (group 0, one uniform binding).
    pub(crate) pick_camera_bgl: Option<wgpu::BindGroupLayout>,

    // --- Sub-object highlight (lazily created) ---
    /// Translucent face fill pipeline. HDR path (Rgba16Float colour target).
    /// `None` until the first frame that has `sub_selection.is_some()`.
    pub(crate) sub_highlight_fill_pipeline: Option<wgpu::RenderPipeline>,
    /// Depth-nudged billboard edge-line pipeline. HDR path (Rgba16Float colour target).
    /// `None` until the first frame that has `sub_selection.is_some()`.
    pub(crate) sub_highlight_edge_pipeline: Option<wgpu::RenderPipeline>,
    /// Billboard sprite pipeline for vertex/point highlights. HDR path (Rgba16Float).
    pub(crate) sub_highlight_sprite_pipeline: Option<wgpu::RenderPipeline>,
    /// Translucent face fill pipeline. LDR path (swapchain `target_format`).
    pub(crate) sub_highlight_fill_ldr_pipeline: Option<wgpu::RenderPipeline>,
    /// Depth-nudged billboard edge-line pipeline. LDR path (swapchain `target_format`).
    pub(crate) sub_highlight_edge_ldr_pipeline: Option<wgpu::RenderPipeline>,
    /// Billboard sprite pipeline for vertex/point highlights. LDR path (swapchain `target_format`).
    pub(crate) sub_highlight_sprite_ldr_pipeline: Option<wgpu::RenderPipeline>,
    /// Shared bind group layout for all highlight pipelines (group 1: SubHighlightUniform).
    pub(crate) sub_highlight_bgl: Option<wgpu::BindGroupLayout>,

    // --- Font atlas (overlay text rendering) ---
    /// Glyph atlas for overlay text rendering (labels, scalar bars, rulers).
    pub(crate) glyph_atlas: crate::resources::overlay::font::GlyphAtlas,

    // --- Overlay text pipeline (lazily created) ---
    /// Render pipeline for screen-space text and solid overlay quads.
    /// `None` until the first frame with non-empty `OverlayFrame.labels`.
    pub(crate) overlay_text_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the overlay text pipeline (group 0: atlas texture + sampler).
    pub(crate) overlay_text_bgl: Option<wgpu::BindGroupLayout>,
    /// Linear sampler for the glyph atlas texture.
    pub(crate) overlay_text_sampler: Option<wgpu::Sampler>,

    // --- SDF overlay shape pipeline (lazily created) ---
    /// Render pipeline for screen-space SDF shapes (rounded rects, circles, etc.).
    /// `None` until the first frame with non-empty `OverlayFrame.shapes`.
    pub(crate) overlay_shape_pipeline: Option<wgpu::RenderPipeline>,
    /// Render pipeline for SDF shapes with texture fill.
    /// `None` until the first frame that references an `OverlayTextureId`.
    pub(crate) overlay_shape_tex_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the texture pipeline (group 0: texture + sampler).
    pub(crate) overlay_shape_tex_bgl: Option<wgpu::BindGroupLayout>,
    /// Clamp-to-edge linear sampler shared across all texture shape bind groups.
    pub(crate) overlay_shape_tex_sampler: Option<wgpu::Sampler>,
    /// Persistent textures uploaded via `upload_overlay_texture`.
    pub(crate) overlay_textures: Vec<OverlayShapeTextureEntry>,

    // --- Backdrop blur pipeline (lazily created) ---
    /// Fullscreen separable Gaussian blur pipeline used to produce the blurred
    /// scene texture for `backdrop_blur` overlay shapes.
    pub(crate) backdrop_blur_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the blur pipeline (group 0: source texture + sampler + uniforms).
    pub(crate) backdrop_blur_bgl: Option<wgpu::BindGroupLayout>,
    /// Linear clamp sampler shared by blur passes.
    pub(crate) backdrop_blur_sampler: Option<wgpu::Sampler>,

    // --- Depth blit pipeline (lazily created, shared across all viewports) ---
    // Copies a scene-resolution depth texture to a native-resolution depth-only target.
    // Used by the HDR path when render_scale < 1.0.
    pub(crate) depth_blit_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) depth_blit_bgl: Option<wgpu::BindGroupLayout>,

    // --- Dynamic resolution render target (lazily created) ---
    // Upscale pipeline: renders the scaled intermediate colour texture to the surface.
    // No depth attachment; used by render_frame_internal which controls its own encoder.
    pub(crate) dyn_res_upscale_pipeline: Option<wgpu::RenderPipeline>,
    // Depth-stencil compatible variant for use inside eframe's paint render pass,
    // which always provides a Depth24PlusStencil8 attachment. Depth writes disabled.
    pub(crate) dyn_res_upscale_ds_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the upscale pass (texture + sampler).
    pub(crate) dyn_res_upscale_bgl: Option<wgpu::BindGroupLayout>,
    /// Linear-clamp sampler for the upscale blit.
    pub(crate) dyn_res_linear_sampler: Option<wgpu::Sampler>,

    // --- Runtime performance tracking ---
    /// Cumulative bytes of geometry data uploaded since the last `prepare()` reset.
    ///
    /// Incremented by `upload_mesh`, `upload_mesh_data`, and `replace_mesh_data`.
    /// Read and reset at the start of each `prepare()` call to populate
    /// `FrameStats::upload_bytes`.
    pub frame_upload_bytes: u64,

    // --- Screen-space decal pipelines (D1, lazily created) ---
    /// Replace-blend decal pipeline (LDR + HDR). None until first decal is submitted.
    pub(crate) decal_replace_pipeline: Option<DualPipeline>,
    /// Multiply-blend decal pipeline (LDR + HDR). None until first decal is submitted.
    pub(crate) decal_multiply_pipeline: Option<DualPipeline>,
    /// Additive-blend decal pipeline (LDR + HDR). None until first decal is submitted.
    pub(crate) decal_additive_pipeline: Option<DualPipeline>,
    /// BGL for group 1 of the decal pass: depth texture + stencil texture bindings.
    pub(crate) decal_depth_bgl: Option<wgpu::BindGroupLayout>,
    /// BGL for group 2 of the decal pass: uniform buffer + albedo texture + sampler.
    pub(crate) decal_item_bgl: Option<wgpu::BindGroupLayout>,
    /// Linear-clamp sampler used by the decal fragment shader.
    pub(crate) decal_sampler: Option<wgpu::Sampler>,
    // --- D5: decal receiver masking ---
    /// Pipeline that writes stencil = 0 for non-receiver surfaces (D5).
    pub(crate) decal_exclude_pipeline: Option<wgpu::RenderPipeline>,
    /// BGL for group 1 of the decal exclude pass: one model matrix uniform buffer.
    pub(crate) decal_exclude_obj_bgl: Option<wgpu::BindGroupLayout>,

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
/// camera-independent and stay on `ViewportGpuResources`.
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
