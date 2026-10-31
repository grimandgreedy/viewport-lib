//! The basic mesh path the compatibility tier draws `SceneRenderItem`s with: the
//! per-instance record its vertex stage reads as a vertex stream, the
//! pipeline variants and the mesh records its uploads build.

use crate::resources::mesh::gpu_mesh::GpuMesh;
use crate::resources::types::Vertex;

/// One draw's per-object data, read by `mesh_basic.wgsl` as an
/// instance-stepped vertex buffer at slot 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BasicInstance {
    pub(crate) model: [[f32; 4]; 4],
    /// Linear RGB and opacity.
    pub(crate) base_colour: [f32; 4],
    /// Emissive RGB times its strength, then the alpha cutoff.
    pub(crate) emissive: [f32; 4],
    /// Metallic, roughness, then the scalar range. The metallic and roughness
    /// are carried for a lighting model that reads them; the basic one does
    /// not.
    pub(crate) material: [f32; 4],
    /// Flags, the NaN colour as RGBA8, the emissive exposure weight as `f32`
    /// bits, one unused word.
    pub(crate) params: [u32; 4],
}

/// Bits of `BasicInstance::params[0]`, matching the `BASIC_*` constants in
/// `mesh_basic.wgsl`.
pub(crate) mod basic_flags {
    pub(crate) const UNLIT: u32 = 1;
    pub(crate) const TWO_SIDED: u32 = 2;
    pub(crate) const ALPHA_MASK: u32 = 4;
    pub(crate) const HAS_SCALAR: u32 = 8;
    pub(crate) const USE_NAN_COLOUR: u32 = 16;
    pub(crate) const IGNORE_CLIP: u32 = 32;
    pub(crate) const WIREFRAME: u32 = 64;
}

const INSTANCE_ATTRIBUTES: [crate::gpu::VertexAttribute; 8] = crate::gpu::vertex_attr_array![
    5 => Float32x4,
    6 => Float32x4,
    7 => Float32x4,
    8 => Float32x4,
    9 => Float32x4,
    10 => Float32x4,
    11 => Float32x4,
    12 => Uint32x4,
];

const SCALAR_ATTRIBUTES: [crate::gpu::VertexAttribute; 1] =
    crate::gpu::vertex_attr_array![13 => Float32];

/// The stride of [`BasicInstance`] in the instance stream.
pub(crate) const BASIC_INSTANCE_STRIDE: u64 = std::mem::size_of::<BasicInstance>() as u64;

/// Which variant of the basic mesh pipeline a draw needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BasicMeshKey {
    /// Alpha blended with depth writes off, for a translucent item.
    pub(crate) blend: bool,
    /// Both faces drawn, for a two-sided material.
    pub(crate) two_sided: bool,
    /// A per-vertex scalar stream at slot 2.
    pub(crate) scalar: bool,
    /// The mesh's edges as a line list, for a wireframe item.
    pub(crate) lines: bool,
}

/// The basic mesh pipelines, built the first time a draw needs each variant.
#[derive(Default)]
pub(crate) struct BasicMeshPipelines {
    module: Option<crate::gpu::ShaderModule>,
    pipelines: std::collections::HashMap<BasicMeshKey, crate::gpu::RenderPipeline>,
}

impl BasicMeshPipelines {
    /// Build the variant for `key` if it does not exist yet.
    pub(crate) fn ensure(
        &mut self,
        device: &crate::gpu::Device,
        capabilities: &crate::capabilities::Capabilities,
        layouts: [&crate::gpu::BindGroupLayout; 2],
        target_format: crate::gpu::TextureFormat,
        key: BasicMeshKey,
    ) {
        if self.pipelines.contains_key(&key) {
            return;
        }
        let module = self.module.get_or_insert_with(|| {
            let source = crate::resources::builders::adapt_wgsl(MESH_BASIC_WGSL, capabilities);
            crate::resources::builders::wgsl_module(device, "mesh_basic", source.into_owned())
        });
        let layout = crate::resources::builders::pipeline_layout(device, "mesh_basic", &layouts);
        let vertex_buffers = [
            crate::resources::builders::mesh_vertex_layout(),
            crate::gpu::VertexBufferLayout {
                array_stride: BASIC_INSTANCE_STRIDE,
                step_mode: crate::gpu::VertexStepMode::Instance,
                attributes: &INSTANCE_ATTRIBUTES,
            },
            crate::gpu::VertexBufferLayout {
                array_stride: 4,
                step_mode: crate::gpu::VertexStepMode::Vertex,
                attributes: &SCALAR_ATTRIBUTES,
            },
        ];
        let buffers = if key.scalar {
            &vertex_buffers[..]
        } else {
            &vertex_buffers[..2]
        };
        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "mesh_basic",
                layout: &layout,
                vertex_module: module,
                vertex_entry: if key.scalar { "vs_scalar" } else { "vs_main" },
                vertex_buffers: buffers,
                fragment: Some(crate::gpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: key.blend.then_some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: if key.lines {
                        crate::gpu::PrimitiveTopology::LineList
                    } else {
                        crate::gpu::PrimitiveTopology::TriangleList
                    },
                    cull_mode: (!key.two_sided && !key.lines).then_some(crate::gpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    capabilities.scene_depth_format,
                    !key.blend,
                    crate::plugin_api::depth::DEPTH_COMPARE,
                )),
                multisample: crate::gpu::MultisampleState::default(),
                cache: None,
            },
        );
        self.pipelines.insert(key, pipeline);
    }

    pub(crate) fn get(&self, key: BasicMeshKey) -> Option<&crate::gpu::RenderPipeline> {
        self.pipelines.get(&key)
    }
}

pub(crate) const MESH_BASIC_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/mesh_basic.wgsl"));

impl crate::resources::DeviceResources {
    /// The mesh record for an upload on the compatibility tier: the geometry goes to
    /// the shared slab as on the full tier, and the object bind group is the
    /// basic mesh path's group 1 with the fallback albedo and LUT.
    pub(crate) fn create_basic_mesh(
        device: &crate::gpu::Device,
        geometry: &mut crate::resources::mesh::geometry_slab::GeometrySlab,
        compat: &crate::resources::compatibility_tier::CompatibilityTier,
        fallback_lut_view: &crate::gpu::TextureView,
        vertices: &[Vertex],
        indices: &[u32],
        uv1_bytes: Option<Vec<u8>>,
    ) -> GpuMesh {
        let vertex_bytes = std::mem::size_of_val(vertices) as u64;
        let index_bytes = std::mem::size_of_val(indices) as u64;
        let vertex_span = geometry.alloc_vertex(device, vertex_bytes);
        let index_span = geometry.alloc_index(device, index_bytes);
        geometry.enqueue_vertex(vertex_span, bytemuck::cast_slice(vertices).to_vec());
        geometry.enqueue_index(index_span, bytemuck::cast_slice(indices).to_vec());
        let has_uv1 = uv1_bytes.is_some();
        if let Some(bytes) = uv1_bytes {
            geometry.ensure_uv1_chunk(device, vertex_span.chunk);
            geometry.enqueue_uv1(vertex_span, bytes);
        }
        let aabb = crate::scene::aabb::Aabb::from_positions(
            &vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
        );
        let mut mesh = Self::basic_mesh_from_spans(
            device,
            compat,
            fallback_lut_view,
            vertex_span,
            index_span,
            indices.len() as u32,
            aabb,
            has_uv1,
        );
        mesh.cpu_indices = Some(indices.to_vec());
        mesh
    }

    /// [`create_basic_mesh`](Self::create_basic_mesh) around slab windows
    /// already filled, for the chunked upload.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn basic_mesh_from_spans(
        device: &crate::gpu::Device,
        compat: &crate::resources::compatibility_tier::CompatibilityTier,
        fallback_lut_view: &crate::gpu::TextureView,
        vertex_span: crate::resources::mesh::geometry_slab::SlabSpan,
        index_span: crate::resources::mesh::geometry_slab::SlabSpan,
        index_count: u32,
        aabb: crate::scene::aabb::Aabb,
        has_uv1: bool,
    ) -> GpuMesh {
        // The basic path reads per-object data from its instance stream, so the
        // record's object buffer is a placeholder.
        let placeholder = || {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("forward_object_placeholder"),
                size: 16,
                usage: crate::gpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            })
        };
        let bind_group = |label| {
            basic_mesh_bind_group(
                device,
                compat,
                label,
                &compat.fallback_albedo_view,
                fallback_lut_view,
            )
        };
        Self::mesh_record(
            vertex_span,
            index_span,
            index_count,
            aabb,
            has_uv1,
            (placeholder(), bind_group("forward_object_bind_group")),
            (placeholder(), bind_group("forward_normal_bind_group")),
        )
    }
}

/// The basic mesh path's group 1 for one albedo texture and LUT.
pub(crate) fn basic_mesh_bind_group(
    device: &crate::gpu::Device,
    compat: &crate::resources::compatibility_tier::CompatibilityTier,
    label: &str,
    albedo: &crate::gpu::TextureView,
    lut: &crate::gpu::TextureView,
) -> crate::gpu::BindGroup {
    device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some(label),
        layout: &compat.mesh_bgl,
        entries: &[
            crate::gpu::BindGroupEntry {
                binding: 0,
                resource: crate::gpu::BindingResource::TextureView(albedo),
            },
            crate::gpu::BindGroupEntry {
                binding: 1,
                resource: crate::gpu::BindingResource::Sampler(&compat.material_sampler),
            },
            crate::gpu::BindGroupEntry {
                binding: 2,
                resource: crate::gpu::BindingResource::TextureView(lut),
            },
            crate::gpu::BindGroupEntry {
                binding: 3,
                resource: crate::gpu::BindingResource::Sampler(&compat.lut_sampler),
            },
        ],
    })
}
