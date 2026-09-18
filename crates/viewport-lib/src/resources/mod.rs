/// Shared constructors for common wgpu bind-group-layout, sampler, and
/// pipeline-layout descriptors, used by the per-feature `ensure_*` methods.
pub(crate) mod builders;
pub(crate) mod custom_data;
/// `DeviceResources` and its content, scope, and feature-resource structs.
pub(crate) mod device_resources;
/// GPU compute resources: clustered shading, hierarchical-Z, and dynamic resolution.
pub mod gpu;
/// Ground-plane pipeline, uniform, and bind group.
pub(crate) mod ground_plane;
/// Shared generational handle primitive and the `ContentHandle` interface.
pub mod handle;
mod init;
pub mod light_probes;
/// Scene lighting buffers, light-probe field, and adaptive probe volume.
pub(crate) mod lighting;
/// Baked lightmap consumption (per-mesh UV1 sidecar + lightmap texture).
pub mod lightmap;
/// Texture, matcap, colourmap, and environment/IBL resources.
pub mod material;
pub(crate) mod material_gpu;
/// GPU memory accounting and the hardware VRAM budget query.
mod memory;
/// Mesh storage, instancing, level-of-detail, and mesh-family pipelines.
pub mod mesh;
pub(crate) mod mesh_sidecar;
pub(crate) mod overlay;
/// Lazy GPU pick-pipeline construction (`ensure_*_pick_pipeline` methods).
mod pick_pipelines;
mod plugin_builders;
mod postprocess;
pub(crate) mod resource_deps;
/// Group-0/1 camera, per-object, and clip bind plumbing.
pub(crate) mod scene_bindings;
/// Core scene mesh pipelines (base LDR set plus HDR variants).
pub(crate) mod scene_pipelines;
pub(crate) mod scivis;
/// Shadow-map GPU resources (cascade atlas, point-shadow cube array, debug viewer).
pub(crate) mod shadow;
#[cfg(test)]
pub(crate) mod test_support;
mod types;
/// Background runner for long-running uploads.
pub mod upload_jobs;
/// Volume, marching-cubes, and unstructured volume-mesh resources.
pub mod volume;

pub use self::gpu::compute_filter::ComputeFilterResult;
pub use self::handle::ContentHandle;
pub use self::light_probes::{
    LightProbe, LightProbeSet, LightProbeVolume, SHCoefficients, evaluate_sh,
    project_equirect_to_sh,
};
pub use self::lightmap::{LightmapData, LightmapMode};
pub use self::material::environment::{EnvironmentMapId, EnvironmentZone};
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
pub use self::overlay::font::{FontError, FontHandle, TextMetrics};
pub(crate) use self::overlay::geometry::{CompiledOverlay, CompiledSource, OverlayInstance};
pub use self::plugin_builders::{
    GlyphBaseMeshRef, HDR_COLOR_FORMAT, MASK_COLOR_FORMAT, MeshDraw, MeshGeometry,
    PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT, PluginPipelineOpts, SCENE_DEPTH_FORMAT,
    SHADOW_DEPTH_FORMAT,
};
pub use self::resource_deps::{ResourceGate, Revalidate};
pub use crate::renderer::item_plugins::curves::types::{RibbonId, StreamtubeId, TubeId};
pub use crate::renderer::item_plugins::external_instances::types::{
    ExternalInstanceSetConfig, ExternalInstanceSetId,
};
pub use crate::renderer::item_plugins::glyph::types::GlyphSetId;
pub use crate::renderer::item_plugins::gpu_particles::types::{
    GpuParticleSystemConfig, GpuParticleSystemId, ParticleRender,
};
pub use crate::renderer::item_plugins::point_cloud::types::PointCloudId;
pub use crate::renderer::item_plugins::polyline::types::PolylineId;
pub use crate::renderer::item_plugins::sprite::types::{SpriteInstanceSetId, SpriteSetId};
pub use crate::renderer::item_plugins::tensor_glyph::types::TensorGlyphSetId;
// Gaussian splat upload vocabulary. Owned here (not in `renderer`) so nothing in
// `resources` reaches up to `renderer` for these types.
pub(crate) use self::scivis::polyline::{PolylineKey, PolylineVariantSet};
pub use viewport_lib_types::data::point::{GaussianSplatData, ShDegree};
pub use viewport_lib_types::ids::GaussianSplatId;
// BatchMeta is published to plugins through `plugin_api::cull`; keep the
// `resources` path crate-internal so there is a single public home for it.
pub(crate) use self::types::BatchMeta;
#[allow(deprecated)]
pub use self::types::ViewportGpuResources;
// GlyphBaseMesh and OverlayUniform are re-exported for crate-internal use even
// though their current consumers reference them through their domain modules.
#[allow(unused_imports)]
pub(crate) use self::postprocess::composite::CompositeInputs;
pub(crate) use self::postprocess::lic::LIC_STRENGTH_ENCODE_MAX;
pub(crate) use self::postprocess::producer::{PostStage, ProducerFrameInputs, ProducerTiming};
pub(crate) use self::types::{
    AtlasBlitUniform, BackdropBlurState, BloomUniform, ClipPlanesUniform, ClipShapeGpu,
    ContactShadowUniform, DofUniform, DualPipeline, FrustumPlane, FrustumUniform, GlyphBaseMesh,
    GpuProjectedTetMesh, GridUniform, GroundPlaneUniform, InstanceAabb, InstanceData, LabelGpuData,
    LicAdvectUniform, LicObjectUniform, LicSurfaceGpuData, MeshInstanceGpuData, ObjectUniform,
    OutlineEdgeUniform, OutlineObjectBuffers, OutlineUniform, OverlayShadowLayerGpu,
    OverlayShapeGpuData, OverlayShapeTexBatch, OverlayShapeTexVertex, OverlayShapeVertex,
    OverlayTextVertex, OverlayUniform, PickInstance, ProjectedTetUniform, SHADOW_ATLAS_SIZE,
    ShadowAtlasUniform, ShadowCullState, SplatOutlineMaskUniform, SsaoUniform, SubHighlightGpuData,
    ToneMapUniform, ViewportCullState, ViewportHdrState,
};
pub use self::types::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColourmap, BuiltinMatcap, CLIP_VOLUME_MAX,
    ClipVolumeEntry, ClipVolumesUniform, ColourmapId, DeviceResources, MatcapId, MeshData,
    ProjectedTetId, ResidentBytes, SubmeshRange, TextureData, TextureMemoryStats, TexturePayload,
    TextureRole, TextureSlot, VolumeId, VramBudget,
};
// GPU-side layout types (uniform blocks, vertex and per-item buffer structs).
// These mirror shader-side memory and have no use outside the renderer, so they
// stay crate-internal. Plugins build against the WGSL contract in
// `plugin_api::shared_wgsl` instead.
pub(crate) use self::types::{
    CameraUniform, GpuMesh, GpuTexture, LightUniform, LightsUniform, MAX_SCENE_LIGHTS,
    OverlayVertex, PolylineGpuData, SingleLightUniform, Vertex, VertexBufferLayoutExt,
};
#[cfg(feature = "future")]
pub use self::upload_jobs::JobHandle;
pub use self::upload_jobs::{FrameBudget, JobId, Jobs, ProgressHandle, ResultSlot, UploadStatus};
pub use self::volume::sparse_volume::SparseVolumeGridData;
#[allow(deprecated)]
pub use self::volume::tetmesh::{TetMesh, TetMeshAttributes};
pub use self::volume::volume_mesh::{CELL_SENTINEL, VolumeMeshData};
pub use crate::renderer::GpuMarchingCubesItem;
pub use crate::renderer::item_plugins::gpu_marching_cubes::types::McVolumeId;
pub use crate::renderer::{
    GpuImplicitItem, GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive,
};
