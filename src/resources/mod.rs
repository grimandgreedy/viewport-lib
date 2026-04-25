/// Built-in colormap LUT data.
pub mod colormap_data;
/// Built-in matcap texture data (procedurally generated).
pub mod matcap_data;
/// IBL precomputation and environment map upload.
pub mod environment;
mod extra_impls;
mod init;
mod instancing;
/// Slotted GPU mesh storage with free-list removal.
pub mod mesh_store;
mod meshes;
mod overlays;
mod postprocess;
mod scivis;
mod textures;
mod types;
mod volumes;

pub use self::extra_impls::{ComputeFilterResult, lerp_attributes};
use self::extra_impls::{
    build_glyph_arrow, build_glyph_sphere, build_unit_cube,
    generate_edge_indices,
};
pub use self::types::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColormap, BuiltinMatcap, CameraUniform,
    ClipVolumeUniform, ColormapId, GpuMesh, GpuTexture, LightUniform, LightsUniform, MatcapId,
    MeshData, OverlayVertex, PointCloudGpuData, PolylineGpuData, SingleLightUniform, Vertex,
    ViewportGpuResources, VolumeGpuData, VolumeId,
};
pub(crate) use self::types::{
    BloomUniform, ClipPlanesUniform, ContactShadowUniform, GlyphBaseMesh, GlyphGpuData,
    GridUniform, GroundPlaneUniform, InstanceData, ObjectUniform, OutlineObjectBuffers, OutlineUniform, OverlayUniform,
    PickInstance, SHADOW_ATLAS_SIZE, ShadowAtlasUniform, SsaoUniform, StreamtubeGpuData,
    ToneMapUniform, ViewportHdrState,
};
