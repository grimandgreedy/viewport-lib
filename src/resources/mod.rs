/// Built-in colourmap LUT data.
pub mod colourmap_data;
/// Dynamic resolution intermediate render target.
pub(crate) mod dyn_res;
/// IBL precomputation and environment map upload.
pub mod environment;
mod extra_impls;
/// Font atlas and single-line text layout for overlay rendering.
pub(crate) mod font;
/// GPU marching cubes compute pipeline (Phase 17).
pub mod gpu_marching_cubes;
mod highlight;
/// GPU implicit surface types and pipeline (Phase 16).
pub mod implicit;
mod init;
mod instancing;
/// Built-in matcap texture data (procedurally generated).
pub mod matcap_data;
/// Slotted GPU mesh storage with free-list removal.
pub mod mesh_store;
mod meshes;
mod overlay_text;
mod overlays;
mod postprocess;
mod scivis;
/// Sparse voxel grid topology processing (boundary face extraction).
pub mod sparse_volume;
mod textures;
mod types;
/// Unstructured volume mesh topology processing (tet / hex boundary extraction).
pub mod volume_mesh;
mod volumes;

pub use self::extra_impls::{ComputeFilterResult, lerp_attributes};
use self::extra_impls::{
    build_glyph_arrow, build_glyph_sphere, build_unit_cube, generate_edge_indices,
};
pub use self::font::{FontError, FontHandle};
pub use self::gpu_marching_cubes::{GpuMarchingCubesJob, VolumeGpuId};
pub use self::implicit::{
    GpuImplicitItem, GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive,
};
pub use self::sparse_volume::SparseVolumeGridData;
#[allow(deprecated)]
pub use self::types::ClipVolumeUniform;
pub use self::types::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColourmap, BuiltinMatcap, CLIP_VOLUME_MAX,
    CameraUniform, ClipVolumeEntry, ClipVolumesUniform, ColourmapId, GpuMesh, GpuTexture,
    LightUniform, LightsUniform, MatcapId, MeshData, OverlayVertex, PointCloudGpuData,
    PolylineGpuData, ProjectedTetId, ScreenImageGpuData, SingleLightUniform, SkinWeights, Vertex,
    ViewportGpuResources, VolumeGpuData, VolumeId,
};
pub(crate) use self::types::{
    BatchMeta, BloomUniform, ClipPlanesUniform, ContactShadowUniform, CurveMeshOutlineItem,
    DofUniform, DualPipeline,
    FrustumPlane, FrustumUniform, GaussianSplatDrawData, GlyphBaseMesh, GlyphGpuData,
    GpuProjectedTetMesh, GridUniform, GroundPlaneUniform, ImageSliceGpuData, InstanceAabb,
    InstanceData, LabelGpuData, LicAdvectUniform, LicObjectUniform, LicSurfaceGpuData,
    ObjectUniform, OutlineEdgeUniform, OutlineObjectBuffers, OutlineUniform, OverlayTextVertex,
    OverlayUniform, PickInstance, ProjectedTetUniform, RawGeomOutlineBuffers, SHADOW_ATLAS_SIZE,
    ScreenRectOutlineBuffers, ShadowAtlasUniform, SplatOutlineBuffers, SplatOutlineMaskUniform,
    SpriteGpuData, SsaoUniform, StreamtubeGpuData, SubHighlightGpuData, TensorGlyphGpuData,
    ToneMapUniform, ViewportHdrState, VolumeSurfaceSliceGpuData,
};
#[allow(deprecated)]
pub use self::volume_mesh::{
    CELL_SENTINEL, TET_SENTINEL, VolumeMeshData, extract_clipped_volume_faces,
};
