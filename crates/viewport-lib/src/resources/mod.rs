/// Shared constructors for common wgpu bind-group-layout, sampler, and
/// pipeline-layout descriptors, used by the per-feature `ensure_*` methods.
pub(crate) mod builders;
/// Opt-in record of what each pipeline and shader module cost to create, for
/// attributing startup time. See [`builders::build_log`].
pub use builders::build_log;
/// A growable GPU buffer addressed in elements, with a ranged write.
pub mod content_buffer;
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
/// A pipeline built on first use, on the calling thread or a worker.
pub(crate) mod pipeline_slot;
pub use pipeline_slot::PipelineCompilation;
mod plugin_builders;
mod postprocess;
pub(crate) use postprocess::TargetGroups;
/// A GPU readback that can be polled instead of waited on.
#[cfg(any(feature = "raytrace", feature = "bake"))]
pub(crate) mod readback;
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

pub use self::content_buffer::ContentBuffer;
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
use self::mesh::geometry::build_unit_cube;
pub use self::mesh::geometry::generate_edge_indices;
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
    HDR_COLOR_FORMAT, MASK_COLOR_FORMAT, MeshDraw, MeshGeometry, PICK_COLOR_FORMAT,
    PICK_DEPTH_CHANNEL_FORMAT, PipelineBuilder, PluginPipelineOpts, SCENE_DEPTH_FORMAT,
    SHADOW_DEPTH_FORMAT,
};
pub use self::resource_deps::{ResourceGate, Revalidate};
pub(crate) use self::scivis::polyline::{PolylineKey, PolylineVariantSet};
pub use self::types::OutlineEdgeUniform;
pub use crate::renderer::item_plugins::polyline::types::PolylineId;
// BatchMeta is published to plugins through `plugin_api::cull`; keep the
// `resources` path crate-internal so there is a single public home for it.
pub(crate) use self::types::BatchMeta;
#[allow(deprecated)]
pub use self::types::ViewportGpuResources;
// GlyphBaseMesh and OverlayUniform are re-exported for crate-internal use even
// though their current consumers reference them through their domain modules.
#[allow(unused_imports)]
pub(crate) use self::postprocess::composite::CompositeInputs;
pub(crate) use self::postprocess::producer::{
    PostProducer, PostStage, ProducerFrameInputs, ProducerTiming,
};
pub(crate) use self::types::{
    AtlasBlitUniform, BackdropBlurState, BloomUniform, ClipPlanesUniform, ClipShapeGpu,
    ContactShadowUniform, DofUniform, DualPipeline, FrustumPlane, FrustumUniform,
    GpuProjectedTetMesh, GridUniform, GroundPlaneUniform, InstanceAabb, InstanceData, LabelGpuData,
    MeshInstanceGpuData, ObjectUniform, OutlineObjectBuffers, OutlineUniform,
    OverlayShadowLayerGpu, OverlayShapeGpuData, OverlayShapeTexBatch, OverlayShapeTexVertex,
    OverlayShapeVertex, OverlayTextVertex, ProjectedTetUniform, SHADOW_ATLAS_SIZE,
    ShadowAtlasUniform, ShadowCullState, SsaoUniform, SubHighlightGpuData, ToneMapUniform,
    ViewportCullState, ViewportHdrState,
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
    OverlayVertex, PolylineGpuData, SingleLightUniform, VertexBufferLayoutExt,
};
// The mesh vertex is the exception: `plugin_api::builders::mesh_vertex_layout`
// publishes its buffer layout, so an item type building geometry for a
// pipeline that declares that layout has to be able to fill the struct too.
pub use self::types::Vertex;
// Likewise the per-instance pick record: `plugin_api::shared_wgsl`'s
// `SHARED_PICK_INSTANCE_WGSL` declares its shader-side counterpart, so an item
// type drawing instanced pick geometry has to be able to fill it.
pub use self::types::PickInstance;
#[cfg(feature = "future")]
pub use self::upload_jobs::JobHandle;
pub use self::upload_jobs::{FrameBudget, JobId, Jobs, ProgressHandle, ResultSlot, UploadStatus};
#[allow(deprecated)]
pub use self::volume::tetmesh::{TetMesh, TetMeshAttributes};
pub use self::volume::volume_mesh::{CELL_SENTINEL, VolumeMeshData};
