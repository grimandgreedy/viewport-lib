/// Shared constructors for common wgpu bind-group-layout, sampler, and
/// pipeline-layout descriptors, used by the per-feature `ensure_*` methods.
pub(crate) mod builders;
/// Screen-space decal pipeline (D1).
pub(crate) mod decal;
/// `DeviceResources` and its content, scope, and feature-resource structs.
pub(crate) mod device_resources;
/// GPU compute resources: clustered shading, hierarchical-Z, particles, and dynamic resolution.
pub mod gpu;
/// Shared generational handle primitive and the `ContentHandle` interface.
pub mod handle;
mod init;
/// Texture, matcap, colourmap, and environment/IBL resources.
pub mod material;
/// GPU memory accounting and the hardware VRAM budget query.
mod memory;
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

pub use self::gpu::compute_filter::ComputeFilterResult;
pub use self::gpu::external_instances::{ExternalInstanceSetConfig, ExternalInstanceSetId};
pub use self::gpu::gpu_particles::{GpuParticleSystemConfig, GpuParticleSystemId, ParticleRender};
pub use self::handle::ContentHandle;
pub use self::material::texture_store::TextureId;
pub use self::material::textures::{CompressedTextureDesc, supports_texture_format};
pub use self::memory::vram_budget;
use self::mesh::geometry::{build_glyph_arrow, build_glyph_sphere, build_unit_cube};
pub use self::mesh::lod::{LodGroup, LodGroupId, LodLevel, LodTransition, projected_screen_size};
pub use self::mesh::meshes::OverrideBufferSlice;
pub use self::mesh::meshes::lerp_attributes;
pub use self::mesh_sidecar::deform::{
    DEFORM_SLOT_PARAMS_BYTES, DeformSlotHandle, DeformSourceSlice, deform_slot_params_byte_offset,
};
pub use self::mesh_sidecar::registry::{
    DEFORM_PARAMS_PER_SLOT_PUB, DEFORM_SLOT_COUNT_PUB, DeformStage, DeformerDesc, DeformerId,
};
pub use self::mesh_sidecar::shade::{
    MATERIAL_PLUGIN_PARAM_VEC4S, MaterialPlugin, MaterialPluginParamsHandle, MaterialPluginStats,
    ShadingHookDesc, ShadingHookId,
};
pub use self::overlay::font::{FontError, FontHandle};
pub use self::plugin_builders::{
    HDR_COLOR_FORMAT, MASK_COLOR_FORMAT, PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT,
    PluginPipelineOpts, SCENE_DEPTH_FORMAT, SHADOW_DEPTH_FORMAT,
};
pub use self::scivis::curve_store::{
    GlyphSetId, PointCloudId, PolylineId, RibbonId, SpriteInstanceSetId, SpriteSetId, StreamtubeId,
    TensorGlyphSetId, TubeId,
};
pub(crate) use self::scivis::curve_store::{
    GlyphSetStore, PointCloudStore, PolylineStore, RibbonStore, SpriteInstanceSetStore,
    SpriteSetStore, StreamtubeStore, TensorGlyphSetStore, TubeStore,
};
// BatchMeta is published to plugins through `plugin_api::cull`; keep the
// `resources` path crate-internal so there is a single public home for it.
pub(crate) use self::types::BatchMeta;
pub(crate) use self::types::ScatterViewportState;
#[allow(deprecated)]
pub use self::types::ViewportGpuResources;
// GlyphBaseMesh and OverlayUniform are re-exported for crate-internal use even
// though their current consumers reference them through their domain modules.
#[allow(unused_imports)]
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
    ClipVolumeEntry, ClipVolumesUniform, ColourmapId, DeviceResources, MatcapId, MeshData,
    ProjectedTetId, ResidentBytes, TextureMemoryStats, VolumeId, VramBudget,
};
// GPU-side layout types (uniform blocks, vertex and per-item buffer structs).
// These mirror shader-side memory and have no use outside the renderer, so they
// stay crate-internal. Plugins build against the WGSL contract in
// `plugin_api::shared_wgsl` instead.
pub(crate) use self::types::{
    CameraUniform, GpuMesh, GpuTexture, LightUniform, LightsUniform, MAX_SCENE_LIGHTS,
    OverlayVertex, PointCloudGpuData, PolylineGpuData, ScreenImageGpuData, SingleLightUniform,
    Vertex, VolumeGpuData,
};
#[cfg(feature = "future")]
pub use self::upload_jobs::JobHandle;
pub use self::upload_jobs::{FrameBudget, JobId, Jobs, ProgressHandle, ResultSlot, UploadStatus};
pub use self::volume::gpu_marching_cubes::McVolumeId;
pub use self::volume::implicit::{GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive};
pub use self::volume::sparse_volume::SparseVolumeGridData;
#[allow(deprecated)]
pub use self::volume::tetmesh::{TetMesh, TetMeshAttributes};
pub use self::volume::volume_mesh::{CELL_SENTINEL, VolumeMeshData};
pub use crate::renderer::GpuImplicitItem;
pub use crate::renderer::GpuMarchingCubesJob;
