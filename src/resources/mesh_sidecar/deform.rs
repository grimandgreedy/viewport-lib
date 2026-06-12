//! Per-vertex deformation sidecar storage.
//!
//! Hosts the `@group(2)` bind group used by every mesh-family pipeline.
//! The deformation registry routes registered deformers through four slots
//! plus a shared header uniform; when no slots are attached, every mesh
//! binds the renderer-owned dummy bind group so the pipeline layout stays
//! satisfied at zero cost.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::error::ViewportResult;
use crate::resources::ViewportGpuResources;
use crate::resources::mesh_sidecar::registry::{
    DeformerDesc, DeformerId, MESH_FAMILY_SHADERS, StoredDeformer, allocate_slot, compose_shader,
    lookup_source, validate_name, validate_with_wgpu,
};
use crate::resources::mesh_store::MeshId;

/// Maximum number of registered deformer slots. Limited by
/// `maxStorageBuffersPerShaderStage` minus the two storage buffers already
/// consumed by group 1 (scalar, face colour, warp, position/normal overrides
/// fit because they share the per-mesh group, but four slot storage buffers
/// at group 2 keeps the total inside the default vertex-stage budget of 8).
pub const DEFORM_SLOT_COUNT: usize = 4;

/// `vec4<f32>` count per slot inside the shared header uniform. Keep in sync
/// with `DeformHeader` and the `slot_params` field in `deform.wgsl`.
pub const DEFORM_PARAMS_PER_SLOT: usize = 4;

/// Shared header uniform; one region per slot plus a global time value.
/// Mirrors `DeformHeader` in `deform.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct DeformHeader {
    pub time_seconds: f32,
    pub _pad: [f32; 3],
    /// Flat array: slot `i` reads `slot_params[i * DEFORM_PARAMS_PER_SLOT .. (i+1) * DEFORM_PARAMS_PER_SLOT]`.
    pub slot_params: [[f32; 4]; DEFORM_SLOT_COUNT * DEFORM_PARAMS_PER_SLOT],
}

impl DeformHeader {
    pub fn zeroed() -> Self {
        Self {
            time_seconds: 0.0,
            _pad: [0.0; 3],
            slot_params: [[0.0; 4]; DEFORM_SLOT_COUNT * DEFORM_PARAMS_PER_SLOT],
        }
    }
}

/// Per-mesh deformation storage.
///
/// Holds one storage buffer per attached slot and a bind group binding the
/// shared header plus all four slot buffers (unattached slots fall back to
/// the renderer-owned dummy storage buffer).
pub(crate) struct MeshDeform {
    pub slot_buffers: [Option<wgpu::Buffer>; DEFORM_SLOT_COUNT],
    pub bind_group: wgpu::BindGroup,
    /// Bit `i` set when slot `i` has a buffer attached.
    pub flag_bits: u32,
}

/// Renderer-side deformation state.
pub(crate) struct DeformationState {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub header_buffer: wgpu::Buffer,
    /// One-element storage buffer bound in place of unattached slots.
    pub dummy_slot_buffer: wgpu::Buffer,
    /// Bind group used when a mesh has no attached deformer data. Bound by
    /// every mesh-family draw once the deformer registry has rebuilt the
    /// mesh pipelines with the deform layout in slot 2.
    #[allow(dead_code)]
    pub dummy_bind_group: wgpu::BindGroup,
    pub meshes: HashMap<MeshId, MeshDeform>,
    pub header_cpu: DeformHeader,
    /// Currently registered deformers, in registration order. The order does
    /// not drive composition (the composer sorts by stage + priority + name);
    /// it is preserved so descriptors can be replayed verbatim on a device
    /// reset.
    pub registrations: Vec<crate::resources::mesh_sidecar::registry::StoredDeformer>,
}

impl DeformationState {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deform_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let header_cpu = DeformHeader::zeroed();
        let header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_header"),
            contents: bytemuck::bytes_of(&header_cpu),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let dummy_slot_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_dummy_slot"),
            contents: bytemuck::bytes_of(&[0u32; 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let dummy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform_dummy_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dummy_slot_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            bind_group_layout,
            header_buffer,
            dummy_slot_buffer,
            dummy_bind_group,
            meshes: HashMap::new(),
            header_cpu,
            registrations: Vec::new(),
        }
    }

    /// Returns the bind group to use for a given mesh: the per-mesh group
    /// when any slot is attached, otherwise the renderer-owned dummy. Used
    /// at draw time once the registry has rebuilt the mesh pipelines to bind
    /// group 2.
    #[allow(dead_code)]
    pub fn bind_group_for(&self, mesh_id: MeshId) -> &wgpu::BindGroup {
        self.meshes
            .get(&mesh_id)
            .map(|m| &m.bind_group)
            .unwrap_or(&self.dummy_bind_group)
    }

    /// `deform_flags` value to write into the `ObjectUniform` for `mesh_id`.
    pub fn flag_bits(&self, mesh_id: MeshId) -> u32 {
        self.meshes.get(&mesh_id).map(|m| m.flag_bits).unwrap_or(0)
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device, mesh_id: MeshId) {
        let Some(m) = self.meshes.get(&mesh_id) else {
            return;
        };
        let slot_bindings: [&wgpu::Buffer; DEFORM_SLOT_COUNT] = [
            m.slot_buffers[0].as_ref().unwrap_or(&self.dummy_slot_buffer),
            m.slot_buffers[1].as_ref().unwrap_or(&self.dummy_slot_buffer),
            m.slot_buffers[2].as_ref().unwrap_or(&self.dummy_slot_buffer),
            m.slot_buffers[3].as_ref().unwrap_or(&self.dummy_slot_buffer),
        ];
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform_mesh_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slot_bindings[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: slot_bindings[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: slot_bindings[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: slot_bindings[3].as_entire_binding(),
                },
            ],
        });
        if let Some(m) = self.meshes.get_mut(&mesh_id) {
            m.bind_group = bind_group;
        }
    }

    /// Attach raw bytes to a specific slot for `mesh_id`. Creates a fresh
    /// storage buffer and rebuilds the per-mesh bind group.
    pub fn attach_slot(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        slot: usize,
        data: &[u8],
    ) {
        assert!(slot < DEFORM_SLOT_COUNT);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_slot"),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        if !self.meshes.contains_key(&mesh_id) {
            let bg = self.create_empty_bind_group(device);
            self.meshes.insert(
                mesh_id,
                MeshDeform {
                    slot_buffers: [None, None, None, None],
                    bind_group: bg,
                    flag_bits: 0,
                },
            );
        }
        let entry = self.meshes.get_mut(&mesh_id).unwrap();
        entry.slot_buffers[slot] = Some(buffer);
        entry.flag_bits |= 1u32 << slot;
        self.rebuild_bind_group(device, mesh_id);
    }

    /// Detach a slot. If no slots remain, drops the per-mesh entry so the
    /// dummy bind group is used again.
    pub fn detach_slot(&mut self, device: &wgpu::Device, mesh_id: MeshId, slot: usize) -> bool {
        assert!(slot < DEFORM_SLOT_COUNT);
        let Some(m) = self.meshes.get_mut(&mesh_id) else {
            return false;
        };
        let had = m.slot_buffers[slot].take().is_some();
        if had {
            m.flag_bits &= !(1u32 << slot);
        }
        if m.flag_bits == 0 {
            self.meshes.remove(&mesh_id);
        } else if had {
            self.rebuild_bind_group(device, mesh_id);
        }
        had
    }

    pub fn has_slot(&self, mesh_id: MeshId, slot: usize) -> bool {
        self.meshes
            .get(&mesh_id)
            .map(|m| m.slot_buffers[slot].is_some())
            .unwrap_or(false)
    }

    fn create_empty_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform_mesh_bg_empty"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.dummy_slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.dummy_slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.dummy_slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.dummy_slot_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// Public API on ViewportGpuResources
// ---------------------------------------------------------------------------

impl ViewportGpuResources {
    /// Write the shared header uniform's `time_seconds` field. Cheap; safe to
    /// call per frame.
    pub fn set_deform_time(&mut self, queue: &wgpu::Queue, time_seconds: f32) {
        self.deform.header_cpu.time_seconds = time_seconds;
        queue.write_buffer(
            &self.deform.header_buffer,
            0,
            bytemuck::bytes_of(&self.deform.header_cpu),
        );
    }

    /// Write the four `vec4<f32>` parameter words for one slot in the shared
    /// header uniform.
    pub fn set_deform_slot_params(
        &mut self,
        queue: &wgpu::Queue,
        slot: usize,
        params: [[f32; 4]; DEFORM_PARAMS_PER_SLOT],
    ) {
        assert!(slot < DEFORM_SLOT_COUNT);
        let base = slot * DEFORM_PARAMS_PER_SLOT;
        for (i, p) in params.iter().enumerate() {
            self.deform.header_cpu.slot_params[base + i] = *p;
        }
        queue.write_buffer(
            &self.deform.header_buffer,
            0,
            bytemuck::bytes_of(&self.deform.header_cpu),
        );
    }

    /// Attach raw bytes for one deformer slot on the given mesh. The composer
    /// defines the per-vertex stride; the caller is responsible for matching
    /// it. Data is uploaded and visible to the per-mesh deformation bind
    /// group, but is only read once a deformer is registered against the
    /// slot.
    pub fn attach_deform_slot(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        slot: usize,
        data: &[u8],
    ) {
        self.deform.attach_slot(device, mesh_id, slot, data);
    }

    /// Detach a slot's data. Returns `true` if any data was removed.
    pub fn detach_deform_slot(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        slot: usize,
    ) -> bool {
        self.deform.detach_slot(device, mesh_id, slot)
    }

    /// Returns `true` when the mesh has data attached at the given slot.
    pub fn has_deform_slot(&self, mesh_id: MeshId, slot: usize) -> bool {
        self.deform.has_slot(mesh_id, slot)
    }

    /// Register a deformer against the mesh shader family.
    ///
    /// Validates the descriptor's name and allocates a slot, composes every
    /// mesh-family base shader with the new deformer plus all previously
    /// registered ones, and runs each composed module through wgpu's
    /// validator. On any failure the registration is rolled back: the
    /// previously composed sources stay live and the returned error names
    /// the shader that failed.
    ///
    /// Pipeline rebuild is not yet wired through this call. Once registered,
    /// the deformer is reserved against the slot budget and its
    /// `wgsl_body` has been validated against every shader it would be
    /// spliced into, but the live pipelines continue to run the identity
    /// path until the registry-driven pipeline rebuild lands.
    ///
    /// # Errors
    ///
    /// - [`ViewportError::DeformNameTaken`] when `desc.name` is already
    ///   registered.
    /// - [`ViewportError::DeformShaderInvalid`] when `desc.name` is not a
    ///   valid WGSL identifier, or when any composed module fails
    ///   validation.
    /// - [`ViewportError::DeformSlotsExhausted`] when all
    ///   `DEFORM_SLOT_COUNT` slots are in use.
    ///
    /// [`ViewportError::DeformNameTaken`]: crate::error::ViewportError::DeformNameTaken
    /// [`ViewportError::DeformShaderInvalid`]: crate::error::ViewportError::DeformShaderInvalid
    /// [`ViewportError::DeformSlotsExhausted`]: crate::error::ViewportError::DeformSlotsExhausted
    pub fn register_deformer(
        &mut self,
        device: &wgpu::Device,
        desc: DeformerDesc,
    ) -> ViewportResult<DeformerId> {
        validate_name(&self.deform.registrations, desc.name)?;
        let slot = allocate_slot(&self.deform.registrations)?;

        let candidate = StoredDeformer { desc, slot };
        let mut trial = self.deform.registrations.clone();
        trial.push(candidate.clone());

        for shader_name in MESH_FAMILY_SHADERS {
            let Some(base) = lookup_source(shader_name) else {
                return Err(crate::error::ViewportError::DeformShaderInvalid {
                    reason: format!(
                        "internal: shader '{shader_name}' missing from shader catalog"
                    ),
                });
            };
            let composed = compose_shader(base, &trial);
            let label = format!("deform_compose_{shader_name}");
            validate_with_wgpu(device, &label, &composed)?;
        }

        self.deform.registrations.push(candidate);
        Ok(DeformerId(slot))
    }

    /// Number of currently registered deformers.
    pub fn registered_deformer_count(&self) -> usize {
        self.deform.registrations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headless() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("deform_tests"),
            ..Default::default()
        }))
        .ok()?;
        Some((device, queue))
    }

    #[test]
    fn attach_marks_flag_bit_and_swaps_bind_group() {
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let mesh = MeshId(7);
        assert_eq!(s.flag_bits(mesh), 0);
        assert!(!s.has_slot(mesh, 0));

        s.attach_slot(&device, mesh, 0, &[0u8; 16]);
        assert!(s.has_slot(mesh, 0));
        assert_eq!(s.flag_bits(mesh), 0b0001);

        s.attach_slot(&device, mesh, 2, &[0u8; 32]);
        assert!(s.has_slot(mesh, 2));
        assert_eq!(s.flag_bits(mesh), 0b0101);
    }

    #[test]
    fn detach_clears_flag_bit_and_drops_entry_when_empty() {
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let mesh = MeshId(11);
        s.attach_slot(&device, mesh, 1, &[0u8; 16]);
        assert_eq!(s.flag_bits(mesh), 0b0010);

        assert!(s.detach_slot(&device, mesh, 1));
        assert_eq!(s.flag_bits(mesh), 0);
        assert!(!s.meshes.contains_key(&mesh));
        assert!(!s.detach_slot(&device, mesh, 1));
    }

    #[test]
    fn slot_index_assert_traps_out_of_range() {
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                s.attach_slot(&device, MeshId(0), DEFORM_SLOT_COUNT, &[])
            }));
        assert!(result.is_err());
    }

    #[test]
    fn register_deformer_validates_and_assigns_slot() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
        let resources = renderer.resources_mut();
        let desc = DeformerDesc {
            name: "wind",
            stage: crate::resources::mesh_sidecar::registry::DeformStage::WorldSpace,
            priority: 0,
            wgsl_body: "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    var o = v;\n    o.position.z = o.position.z + 0.001;\n    return o;\n}\n".to_string(),
            per_vertex_stride: 4,
        };
        let id = resources.register_deformer(&device, desc).expect("register");
        assert_eq!(id.slot(), 0);
        assert_eq!(resources.registered_deformer_count(), 1);

        // Duplicate name is rejected without changing slot count.
        let dup = DeformerDesc {
            name: "wind",
            stage: crate::resources::mesh_sidecar::registry::DeformStage::ObjectSpace,
            priority: 0,
            wgsl_body: "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex { return v; }".to_string(),
            per_vertex_stride: 4,
        };
        let err = resources.register_deformer(&device, dup).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::DeformNameTaken { .. }
        ));
        assert_eq!(resources.registered_deformer_count(), 1);

        // Invalid WGSL body fails validation, leaves prior registration alone.
        let bad = DeformerDesc {
            name: "wave",
            stage: crate::resources::mesh_sidecar::registry::DeformStage::WorldSpace,
            priority: 0,
            wgsl_body: "this is not valid wgsl".to_string(),
            per_vertex_stride: 4,
        };
        let err = resources.register_deformer(&device, bad).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::DeformShaderInvalid { .. }
        ));
        assert_eq!(resources.registered_deformer_count(), 1);
    }
}
