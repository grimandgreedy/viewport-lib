//! Instanced mesh rendering off a consumer-owned positions buffer.
//!
//! An external instance set draws one mesh per element of a `wgpu::Buffer`
//! the consumer produced on the renderer's own device (a GPU physics solver,
//! a compute skinning pass, any GPU-resident producer). The buffer holds
//! tightly packed `[x, y, z]` `f32` triples, 12 bytes per instance, and is
//! bound directly as a read-only storage buffer: no CPU copy, no per-frame
//! upload. Synchronisation is queue-submission order; the consumer submits
//! its compute work before the frame is rendered.
//!
//! The host calls
//! [`DeviceResources::create_external_instance_set`](crate::resources::DeviceResources::create_external_instance_set)
//! once, then submits an
//! [`ExternalInstancesItem`](crate::renderer::ExternalInstancesItem) per
//! frame. `first_instance` / `instance_count` on the item select a window of
//! the buffer through the draw call's instance range, so several items can
//! render disjoint regions of one pooled buffer.
//!
//! Draw support matches the GPU particle mesh route: HDR path only, opaque,
//! no shadows, no picking, no culling.
use crate::resources::VertexBufferLayoutExt;

use crate::gpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// External instance sets, their lazily built pipeline, and layout.
#[derive(Default)]
pub(crate) struct ExternalInstancesResources {
    /// Live sets. Slots hold `None` after `drop_external_instance_set`.
    pub(crate) sets: Vec<Option<ExternalInstanceSet>>,
    /// Group 1 layout: per-item uniform + positions storage buffer.
    pub(crate) draw_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Opaque depth-tested draw pipeline (LDR + HDR formats).
    pub(crate) pipeline: Option<crate::resources::DualPipeline>,
}

pub use viewport_lib_types::ids::ExternalInstanceSetId;

/// Persistent configuration for an external instance set.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExternalInstanceSetConfig {
    /// Mesh drawn once per instance.
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Consumer-owned same-device buffer of tightly packed `[x, y, z]` `f32`
    /// triples, 12 bytes per instance. Must have
    /// [`wgpu::BufferUsages::STORAGE`](crate::gpu::BufferUsages::STORAGE).
    /// The renderer only reads it.
    pub positions: crate::gpu::Buffer,
}

impl ExternalInstanceSetConfig {
    /// Set drawing `mesh_id` at every position in `positions`.
    pub fn new(
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        positions: crate::gpu::Buffer,
    ) -> Self {
        Self { mesh_id, positions }
    }
}

/// Persistent renderer-side state for one external instance set.
pub(crate) struct ExternalInstanceSet {
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// The consumer's buffer. The clone shares the underlying allocation, so
    /// the renderer keeps it alive even if the consumer drops its handle.
    pub(crate) positions: crate::gpu::Buffer,
}

/// Per-item uniform for the external-instances draw (96 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct ExternalInstancesUniform {
    pub(crate) model: [[f32; 4]; 4], // 64 bytes
    pub(crate) colour: [f32; 4],     // 16 bytes
    pub(crate) scale: f32,           //  4 bytes
    pub(crate) _pad: [f32; 3],       // 12 bytes
}

/// Per-frame draw data for one submitted `ExternalInstancesItem`.
pub(crate) struct ExternalInstancesGpuData {
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Kept alive for the frame; referenced by `bind_group`.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Draw instance range into the positions buffer, already clamped to the
    /// buffer's element count.
    pub(crate) first_instance: u32,
    pub(crate) instance_count: u32,
}

impl crate::resources::DeviceResources {
    /// Register a consumer-owned positions buffer for instanced mesh drawing.
    ///
    /// The buffer stays owned by the consumer and is read in place every
    /// frame the set is submitted; whatever the consumer's compute passes
    /// last wrote is what renders. If the consumer reallocates the buffer
    /// (for example a pool that grew), call
    /// [`set_external_instance_set_buffer`](Self::set_external_instance_set_buffer)
    /// with the new buffer; the old one keeps rendering until then.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::ExternalBufferUsageMissing`](crate::error::ViewportError::ExternalBufferUsageMissing)
    /// if `config.positions` was created without `STORAGE` usage, or
    /// [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `config.mesh_id` is not registered.
    pub fn create_external_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        config: &ExternalInstanceSetConfig,
    ) -> crate::error::ViewportResult<ExternalInstanceSetId> {
        if !config
            .positions
            .usage()
            .contains(crate::gpu::BufferUsages::STORAGE)
        {
            return Err(crate::error::ViewportError::ExternalBufferUsageMissing {
                missing: "STORAGE",
            });
        }
        if !self.mesh_store.contains(config.mesh_id) {
            return Err(crate::error::ViewportError::StaleHandle {
                index: config.mesh_id.index(),
                count: self.mesh_store.len(),
            });
        }
        self.ensure_external_instances_pipelines(device);
        let set = ExternalInstanceSet {
            mesh_id: config.mesh_id,
            positions: config.positions.clone(),
        };
        // Reuse a dropped slot if one exists, otherwise append.
        if let Some(idx) = self
            .external_instances
            .sets
            .iter()
            .position(|s| s.is_none())
        {
            self.external_instances.sets[idx] = Some(set);
            Ok(ExternalInstanceSetId::from_index(idx))
        } else {
            self.external_instances.sets.push(Some(set));
            Ok(ExternalInstanceSetId::from_index(
                self.external_instances.sets.len() - 1,
            ))
        }
    }

    /// Re-point an external instance set at a new positions buffer.
    ///
    /// Use when the consumer reallocated its pool: the renderer holds a
    /// clone of the old `wgpu::Buffer`, so without this call it keeps
    /// rendering the old allocation's last contents.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::ExternalBufferUsageMissing`](crate::error::ViewportError::ExternalBufferUsageMissing)
    /// if `positions` lacks `STORAGE` usage, or
    /// [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `id` does not resolve to a live set.
    pub fn set_external_instance_set_buffer(
        &mut self,
        id: ExternalInstanceSetId,
        positions: crate::gpu::Buffer,
    ) -> crate::error::ViewportResult<()> {
        if !positions
            .usage()
            .contains(crate::gpu::BufferUsages::STORAGE)
        {
            return Err(crate::error::ViewportError::ExternalBufferUsageMissing {
                missing: "STORAGE",
            });
        }
        let count = self.external_instances.sets.len();
        let set = self
            .external_instances
            .sets
            .get_mut(id.index())
            .and_then(|s| s.as_mut())
            .ok_or(crate::error::ViewportError::StaleHandle {
                index: id.index(),
                count,
            })?;
        set.positions = positions;
        Ok(())
    }

    /// Drop an external instance set. Items still submitted with its id are
    /// skipped. The consumer's buffer is released (the renderer's clone is
    /// dropped; the allocation lives while the consumer holds a handle).
    pub fn drop_external_instance_set(&mut self, id: ExternalInstanceSetId) {
        if let Some(slot) = self.external_instances.sets.get_mut(id.index()) {
            *slot = None;
        }
    }

    /// Lazily build the external-instances draw pipeline and layout.
    pub(crate) fn ensure_external_instances_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.external_instances.draw_bgl.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let draw_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("external_instances_bgl"),
            entries: &[
                crate::resources::builders::uniform_entry(
                    0,
                    crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
                ),
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "external_instances_shader",
            crate::resources::builders::wgsl_source!("external_instances"),
        );

        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "external_instances_layout",
            &self.binds.camera_bgl,
            &draw_bgl,
        );

        // Opaque, depth-tested and depth-written: the instances participate
        // in normal occlusion against the opaque scene.
        self.external_instances.pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "external_instances_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[crate::resources::types::Vertex::buffer_layout()],
                blend: None,
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: self.sample_count,
                ldr_format: self.target_format,
            },
        ));
        self.external_instances.draw_bgl = Some(draw_bgl);
    }

    /// Build per-frame draw data for this frame's submitted items.
    pub(crate) fn upload_external_instances(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        items: &[crate::renderer::ExternalInstancesItem],
    ) -> Vec<ExternalInstancesGpuData> {
        self.ensure_external_instances_pipelines(device);
        let Some(draw_bgl) = self.external_instances.draw_bgl.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for item in items {
            if item.settings.hidden || item.instance_count == 0 {
                continue;
            }
            let Some(set) = self
                .external_instances
                .sets
                .get(item.set_id.index())
                .and_then(|s| s.as_ref())
            else {
                continue;
            };
            // Clamp the requested window to the buffer's whole elements so a
            // stale count after a pool shrink cannot read past the end.
            let buffer_elements = (set.positions.size() / 12) as u32;
            let first = item.first_instance.min(buffer_elements);
            let count = item.instance_count.min(buffer_elements - first);
            if count == 0 {
                continue;
            }
            let uniform = ExternalInstancesUniform {
                model: item.model,
                colour: item.colour,
                scale: item.scale,
                _pad: [0.0; 3],
            };
            let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("external_instances_uniform_buf"),
                contents: bytemuck::bytes_of(&uniform),
                usage: crate::gpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("external_instances_bg"),
                layout: draw_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: set.positions.as_entire_binding(),
                    },
                ],
            });
            out.push(ExternalInstancesGpuData {
                mesh_id: set.mesh_id,
                _uniform_buf: uniform_buf,
                bind_group,
                first_instance: first,
                instance_count: count,
            });
        }
        let _ = queue;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceResources;
    use crate::geometry::primitives;

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn positions_buffer(
        device: &crate::gpu::Device,
        elements: usize,
        usage: crate::gpu::BufferUsages,
    ) -> crate::gpu::Buffer {
        device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("test_positions"),
            size: (elements * 12) as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn create_and_drop_roundtrip() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();

        let buf = positions_buffer(&device, 8, crate::gpu::BufferUsages::STORAGE);
        let id = resources
            .create_external_instance_set(
                &device,
                &ExternalInstanceSetConfig::new(mesh_id, buf.clone()),
            )
            .unwrap();
        assert!(resources.external_instances.sets[id.index()].is_some());

        // Re-point works.
        let bigger = positions_buffer(&device, 16, crate::gpu::BufferUsages::STORAGE);
        resources
            .set_external_instance_set_buffer(id, bigger)
            .unwrap();
        assert_eq!(
            resources.external_instances.sets[id.index()]
                .as_ref()
                .unwrap()
                .positions
                .size(),
            16 * 12
        );

        resources.drop_external_instance_set(id);
        assert!(resources.external_instances.sets[id.index()].is_none());

        // Dropped slot is reused by the next create.
        let id2 = resources
            .create_external_instance_set(&device, &ExternalInstanceSetConfig::new(mesh_id, buf))
            .unwrap();
        assert_eq!(id2.index(), id.index());

        // Re-pointing a dropped id fails.
        resources.drop_external_instance_set(id2);
        let other = positions_buffer(&device, 4, crate::gpu::BufferUsages::STORAGE);
        let err = resources.set_external_instance_set_buffer(id2, other);
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::StaleHandle { .. })
        ));
    }

    #[test]
    fn create_rejects_non_storage_buffer() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();

        let buf = positions_buffer(&device, 8, crate::gpu::BufferUsages::VERTEX);
        let err = resources
            .create_external_instance_set(&device, &ExternalInstanceSetConfig::new(mesh_id, buf));
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::ExternalBufferUsageMissing { missing: "STORAGE" })
        ));
    }
}
