/// Built-in colormap LUT data.
pub mod colormap_data;
/// Font atlas and single-line text layout for overlay rendering.
pub(crate) mod font;
/// GPU implicit surface types and pipeline (Phase 16).
pub mod implicit;
/// GPU marching cubes compute pipeline (Phase 17).
pub mod gpu_marching_cubes;
/// IBL precomputation and environment map upload.
pub mod environment;
/// Dynamic resolution intermediate render target.
pub(crate) mod dyn_res;
mod extra_impls;
mod highlight;
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
mod textures;
mod types;
/// Sparse voxel grid topology processing (boundary face extraction).
pub mod sparse_volume;
/// Unstructured volume mesh topology processing (tet / hex boundary extraction).
pub mod volume_mesh;
mod volumes;

pub use self::extra_impls::{ComputeFilterResult, lerp_attributes};
pub use self::font::{FontError, FontHandle};
use self::extra_impls::{
    build_glyph_arrow, build_glyph_sphere, build_unit_cube, generate_edge_indices,
};
#[allow(deprecated)]
pub use self::types::ClipVolumeUniform;
pub use self::types::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColormap, BuiltinMatcap, CameraUniform,
    ClipVolumeEntry, ClipVolumesUniform, CLIP_VOLUME_MAX, ColormapId, GpuMesh, GpuTexture,
    LightUniform, LightsUniform, MatcapId,
    MeshData, OverlayVertex, PointCloudGpuData, PolylineGpuData, ProjectedTetId,
    ScreenImageGpuData, SingleLightUniform, Vertex, ViewportGpuResources, VolumeGpuData, VolumeId,
};
pub(crate) use self::types::{
    BatchMeta, BloomUniform, ClipPlanesUniform, ContactShadowUniform, DofUniform, DualPipeline,
    FrustumPlane, FrustumUniform,
    GaussianSplatDrawData,
    GlyphBaseMesh, GlyphGpuData, GridUniform, GroundPlaneUniform, GpuProjectedTetMesh,
    InstanceAabb, InstanceData, LabelGpuData, LicAdvectUniform, LicObjectUniform,
    LicSurfaceGpuData, ObjectUniform, OutlineEdgeUniform, OutlineObjectBuffers, OutlineUniform,
    OverlayTextVertex, OverlayUniform, PickInstance, ProjectedTetUniform, SHADOW_ATLAS_SIZE,
    ImageSliceGpuData, VolumeSurfaceSliceGpuData, ShadowAtlasUniform, SplatOutlineBuffers,
    SplatOutlineMaskUniform, SpriteGpuData, SsaoUniform,
    StreamtubeGpuData, SubHighlightGpuData,
    TensorGlyphGpuData, ToneMapUniform, ViewportHdrState,
    RawGeomOutlineBuffers, ScreenRectOutlineBuffers,
};
pub use self::implicit::{
    GpuImplicitItem, GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive,
};
pub use self::gpu_marching_cubes::{GpuMarchingCubesJob, VolumeGpuId};
pub use self::sparse_volume::SparseVolumeGridData;
#[allow(deprecated)]
pub use self::volume_mesh::{CELL_SENTINEL, TET_SENTINEL, VolumeMeshData, extract_clipped_volume_faces};
