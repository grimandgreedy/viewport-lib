/// Screen-space decal pipeline (D1).
pub(crate) mod decal;
mod extra_impls;
/// GPU compute resources: clustered shading, hierarchical-Z, particles, and dynamic resolution.
pub mod gpu;
mod init;
/// Texture, matcap, colourmap, and environment/IBL resources.
pub mod material;
/// Mesh storage, instancing, level-of-detail, and mesh-family pipelines.
pub mod mesh;
pub(crate) mod mesh_sidecar;
pub(crate) mod overlay;
mod plugin_builders;
mod postprocess;
mod scivis;
mod types;
/// Background runner for long-running uploads.
pub mod upload_jobs;
/// Volume, implicit-surface, marching-cubes, and unstructured volume-mesh resources.
pub mod volume;

pub use self::extra_impls::{ComputeFilterResult, lerp_attributes};
use self::extra_impls::{
    build_glyph_arrow, build_glyph_sphere, build_unit_cube, generate_edge_indices,
};
pub use self::gpu::gpu_particles::{GpuParticleSystemConfig, GpuParticleSystemId, ParticleRender};
pub use self::material::textures::{CompressedTextureDesc, supports_texture_format};
pub use self::mesh::lod::{LodGroup, LodGroupId, LodLevel, LodTransition, projected_screen_size};
pub use self::mesh_sidecar::deform::{
    DEFORM_SLOT_PARAMS_BYTES, DeformSlotHandle, deform_slot_params_byte_offset,
};
pub use self::mesh_sidecar::registry::{
    DEFORM_PARAMS_PER_SLOT_PUB, DEFORM_SLOT_COUNT_PUB, DeformStage, DeformerDesc, DeformerId,
};
pub use self::overlay::font::{FontError, FontHandle};
pub use self::plugin_builders::{
    HDR_COLOR_FORMAT, MASK_COLOR_FORMAT, PICK_COLOR_FORMAT, PluginPipelineOpts, SCENE_DEPTH_FORMAT,
    SHADOW_DEPTH_FORMAT,
};
pub use self::scivis::curve_store::{
    GlyphSetId, PointCloudId, PolylineId, RibbonId, SpriteInstanceSetId, SpriteSetId, StreamtubeId,
    TensorGlyphSetId, TubeId,
};
pub(crate) use self::scivis::curve_store::{
    GlyphSetStore, PointCloudStore, PolylineStore, RibbonStore, SpriteInstanceSetStore,
    SpriteSetStore, StreamtubeStore, TensorGlyphSetStore, TubeStore,
};
pub use self::types::BatchMeta;
#[allow(deprecated)]
pub use self::types::ClipVolumeUniform;
pub(crate) use self::types::ScatterViewportState;
pub(crate) use self::types::{
    AtlasBlitUniform, BackdropBlurState, BloomUniform, ClipPlanesUniform, ContactShadowUniform,
    CurveMeshOutlineItem, DofUniform, DualPipeline, FrustumPlane, FrustumUniform,
    GaussianSplatDrawData, GlyphBaseMesh, GlyphGpuData, GpuProjectedTetMesh, GridUniform,
    GroundPlaneUniform, ImageSliceGpuData, InstanceAabb, InstanceData, LabelGpuData,
    LicAdvectUniform, LicObjectUniform, LicSurfaceGpuData, MeshInstanceGpuData, ObjectUniform,
    OutlineEdgeUniform, OutlineObjectBuffers, OutlineUniform, OverlayShapeGpuData,
    OverlayShapeTexBatch, OverlayShapeTexVertex, OverlayShapeVertex, OverlayTextVertex,
    OverlayUniform, PickInstance, ProjectedTetUniform, RawGeomOutlineBuffers, SHADOW_ATLAS_SIZE,
    ScreenRectOutlineBuffers, ShadowAtlasUniform, ShadowCullState, SplatOutlineBuffers,
    SplatOutlineMaskUniform, SpriteGpuData, SsaoUniform, StreamtubeGpuData, SubHighlightGpuData,
    TensorGlyphGpuData, ToneMapUniform, ViewportCullState, ViewportHdrState,
    VolumeSurfaceSliceGpuData,
};
pub use self::types::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColourmap, BuiltinMatcap, CLIP_VOLUME_MAX,
    CameraUniform, ClipVolumeEntry, ClipVolumesUniform, ColourmapId, GpuMesh, GpuTexture,
    LightUniform, LightsUniform, MAX_SCENE_LIGHTS, MatcapId, MeshData, OverlayVertex,
    PointCloudGpuData, PolylineGpuData, ProjectedTetId, ResidentBytes, ScreenImageGpuData,
    SingleLightUniform, TextureMemoryStats, Vertex, ViewportGpuResources, VolumeGpuData, VolumeId,
};
#[cfg(feature = "future")]
pub use self::upload_jobs::JobHandle;
pub use self::upload_jobs::{FrameBudget, JobId, Jobs, ProgressHandle, ResultSlot, UploadStatus};
pub use self::volume::gpu_marching_cubes::VolumeGpuId;
pub use self::volume::implicit::{GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive};
pub use self::volume::sparse_volume::SparseVolumeGridData;
#[allow(deprecated)]
pub use self::volume::tetmesh::{TetMesh, TetMeshAttributes};
#[allow(deprecated)] // TET_SENTINEL is a deprecated alias kept for downstream compatibility.
pub use self::volume::volume_mesh::{
    CELL_SENTINEL, TET_SENTINEL, VolumeMeshData, extract_clipped_volume_faces,
};
pub use crate::renderer::GpuImplicitItem;
pub use crate::renderer::GpuMarchingCubesJob;
