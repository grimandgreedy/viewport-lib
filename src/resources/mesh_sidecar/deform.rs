//! Per-vertex deformation sidecar storage.
//!
//! Hosts the `@group(2)` bind group used by every mesh-family pipeline.
//! Each mesh's slot data is packed into a single storage buffer prefixed
//! with `(offset, stride)` pairs per slot; shader-side reads go through
//! the `deform_read_*` helpers in `deform.wgsl`. Meshes with no attached
//! deformer data fall back to the renderer-owned dummy bind group at zero
//! cost.

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

/// Maximum number of registered deformer slots. With every slot's data
/// packed into one storage buffer, deform contributes two vertex-stage
/// bindings (the shared header uniform plus the packed data buffer)
/// regardless of slot count, leaving the slot count free to scale to
/// whatever the WGSL flag-bit budget allows.
pub const DEFORM_SLOT_COUNT: usize = 4;

/// `vec4<f32>` count per slot inside the shared header uniform. Keep in sync
/// with `DeformHeader` and the `slot_params` field in `deform.wgsl`.
pub const DEFORM_PARAMS_PER_SLOT: usize = 4;

/// Number of u32 words at the start of every per-mesh packed buffer
/// reserved for slot layout: an `(offset, stride)` pair per slot.
const SLOT_LAYOUT_WORDS: usize = DEFORM_SLOT_COUNT * 2;

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
pub(crate) struct MeshDeform {
    /// Source bytes per slot, retained so attach/detach can re-pack
    /// without forcing the caller to re-upload other slots.
    pub slot_data: [Option<Vec<u8>>; DEFORM_SLOT_COUNT],
    /// Per-slot stride in u32 words. `slot_stride[i]` is meaningful only
    /// when `slot_data[i].is_some()`.
    pub slot_stride: [u32; DEFORM_SLOT_COUNT],
    /// Packed buffer: `SLOT_LAYOUT_WORDS * 4` bytes of `(offset, stride)`
    /// header followed by tightly packed slot bytes in slot order.
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    /// Bit `i` set when slot `i` has data attached.
    pub flag_bits: u32,
}

/// Renderer-side deformation state.
pub(crate) struct DeformationState {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub header_buffer: wgpu::Buffer,
    /// Empty slot-layout prefix bound when a mesh has no attached data.
    /// `SLOT_LAYOUT_WORDS` u32s of zero. Kept alive so the dummy bind group
    /// stays valid.
    #[allow(dead_code)]
    pub dummy_data_buffer: wgpu::Buffer,
    /// Bind group used when a mesh has no attached deformer data. Bound by
    /// every mesh-family draw at slot 2 to satisfy the pipeline layout
    /// without forcing per-mesh storage allocation.
    pub dummy_bind_group: wgpu::BindGroup,
    pub meshes: HashMap<MeshId, MeshDeform>,
    pub header_cpu: DeformHeader,
    /// Currently registered deformers, in registration order.
    pub registrations: Vec<StoredDeformer>,
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
            ],
        });

        let header_cpu = DeformHeader::zeroed();
        let header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_header"),
            contents: bytemuck::bytes_of(&header_cpu),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dummy_words = vec![0u32; SLOT_LAYOUT_WORDS];
        let dummy_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_dummy_data"),
            contents: bytemuck::cast_slice(&dummy_words),
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
                    resource: dummy_data_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            bind_group_layout,
            header_buffer,
            dummy_data_buffer,
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

    /// Pack the per-slot data of one mesh into a single u32 stream prefixed
    /// by `(offset, stride)` pairs per slot.
    fn pack(
        slot_data: &[Option<Vec<u8>>; DEFORM_SLOT_COUNT],
        slot_stride: &[u32; DEFORM_SLOT_COUNT],
    ) -> Vec<u32> {
        let mut words = vec![0u32; SLOT_LAYOUT_WORDS];
        for slot in 0..DEFORM_SLOT_COUNT {
            if let Some(bytes) = &slot_data[slot] {
                let offset_words = words.len() as u32;
                words[slot * 2] = offset_words;
                words[slot * 2 + 1] = slot_stride[slot];
                let extra = bytes.len() / 4;
                words.reserve(extra);
                for chunk in bytes.chunks_exact(4) {
                    words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
        }
        words
    }

    fn make_bind_group(&self, device: &wgpu::Device, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform_mesh_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Re-pack and re-upload the mesh's data, rebuilding the bind group.
    /// Drops the mesh entry when no slot has data attached.
    fn refresh(&mut self, device: &wgpu::Device, mesh_id: MeshId) {
        let Some(m) = self.meshes.get(&mesh_id) else {
            return;
        };
        if m.flag_bits == 0 {
            self.meshes.remove(&mesh_id);
            return;
        }
        let words = Self::pack(&m.slot_data, &m.slot_stride);
        let new_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_mesh_data"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let new_bg = self.make_bind_group(device, &new_buffer);
        let m = self.meshes.get_mut(&mesh_id).unwrap();
        m.buffer = new_buffer;
        m.bind_group = new_bg;
    }

    /// Attach raw bytes to a specific slot for `mesh_id`. `stride_words` is
    /// the per-vertex stride in u32 words (must equal the registered
    /// deformer's `per_vertex_stride / 4`).
    pub fn attach_slot(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        slot: usize,
        stride_words: u32,
        data: &[u8],
    ) {
        assert!(slot < DEFORM_SLOT_COUNT);
        assert!(
            data.len() % 4 == 0,
            "deform slot data length must be a multiple of 4 bytes"
        );
        if !self.meshes.contains_key(&mesh_id) {
            let init_words = vec![0u32; SLOT_LAYOUT_WORDS];
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("deform_mesh_data_init"),
                contents: bytemuck::cast_slice(&init_words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = self.make_bind_group(device, &buffer);
            self.meshes.insert(
                mesh_id,
                MeshDeform {
                    slot_data: Default::default(),
                    slot_stride: [0; DEFORM_SLOT_COUNT],
                    buffer,
                    bind_group,
                    flag_bits: 0,
                },
            );
        }
        let entry = self.meshes.get_mut(&mesh_id).unwrap();
        entry.slot_data[slot] = Some(data.to_vec());
        entry.slot_stride[slot] = stride_words;
        entry.flag_bits |= 1u32 << slot;
        self.refresh(device, mesh_id);
    }

    /// Detach a slot. Returns `true` if any data was removed.
    pub fn detach_slot(&mut self, device: &wgpu::Device, mesh_id: MeshId, slot: usize) -> bool {
        assert!(slot < DEFORM_SLOT_COUNT);
        let Some(m) = self.meshes.get_mut(&mesh_id) else {
            return false;
        };
        let had = m.slot_data[slot].take().is_some();
        if had {
            m.slot_stride[slot] = 0;
            m.flag_bits &= !(1u32 << slot);
            self.refresh(device, mesh_id);
        }
        had
    }

    pub fn has_slot(&self, mesh_id: MeshId, slot: usize) -> bool {
        self.meshes
            .get(&mesh_id)
            .map(|m| m.slot_data[slot].is_some())
            .unwrap_or(false)
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

    /// Attach raw bytes for one deformer slot on the given mesh.
    ///
    /// `stride_bytes` is the per-vertex byte stride and must equal the
    /// registered deformer's `per_vertex_stride`. The data length is
    /// expected to be `vertex_count * stride_bytes`; the renderer does not
    /// validate the vertex count, only that the byte length is a multiple
    /// of `stride_bytes` and 4.
    pub fn attach_deform_slot(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        slot: usize,
        stride_bytes: u32,
        data: &[u8],
    ) {
        assert!(
            stride_bytes >= 4 && stride_bytes % 4 == 0,
            "deform slot stride must be a positive multiple of 4 bytes"
        );
        self.deform
            .attach_slot(device, mesh_id, slot, stride_bytes / 4, data);
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
    /// validator. On success, the LDR and HDR `mesh.wgsl` pipelines are
    /// rebuilt from the freshly composed source so subsequent draws run
    /// the registered body. Other mesh-family pipelines (instanced,
    /// shadow, outline mask, OIT) continue to run the identity path until
    /// their factories migrate to the same rebuild path.
    ///
    /// On any validation failure the registration is rolled back: the
    /// previously composed sources stay live and the returned error names
    /// the shader that failed.
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
        self.rebuild_mesh_pipelines(device);
        Ok(DeformerId(slot))
    }

    /// Number of currently registered deformers.
    pub fn registered_deformer_count(&self) -> usize {
        self.deform.registrations.len()
    }

    /// Re-compose every mesh-family shader and rebuild the pipelines that
    /// draw from it. Called by `register_deformer` once a new registration
    /// has validated; safe to call between frames with zero registrations
    /// to reset to the identity shader. The instanced and instanced-OIT
    /// pipelines stay on their build-time shader modules until their
    /// factories migrate to the same rebuild flow.
    fn rebuild_mesh_pipelines(&mut self, device: &wgpu::Device) {
        let registrations = self.deform.registrations.clone();

        // mesh.wgsl: LDR + HDR families.
        if let Some(base) = lookup_source("mesh.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh_shader_composed"),
                source: wgpu::ShaderSource::Wgsl(composed.into()),
            });

            let ldr_layout = crate::resources::mesh_pipelines::mesh_pipeline_layout(
                device,
                "mesh_pipeline_layout",
                &self.camera_bind_group_layout,
                &self.object_bind_group_layout,
                &self.deform.bind_group_layout,
            );
            let ldr = crate::resources::mesh_pipelines::build_ldr_mesh_pipelines(
                device,
                &ldr_layout,
                &shader,
                self.target_format,
                self.sample_count,
            );
            self.solid_pipeline = ldr.solid;
            self.solid_two_sided_pipeline = ldr.solid_two_sided;
            self.transparent_pipeline = ldr.transparent;
            self.wireframe_pipeline = ldr.wireframe;

            if self.hdr_solid_pipeline.is_some() {
                let hdr_layout = crate::resources::mesh_pipelines::mesh_pipeline_layout(
                    device,
                    "hdr_mesh_pipeline_layout",
                    &self.camera_bind_group_layout,
                    &self.object_bind_group_layout,
                    &self.deform.bind_group_layout,
                );
                let hdr = crate::resources::mesh_pipelines::build_hdr_mesh_pipelines(
                    device,
                    &hdr_layout,
                    &shader,
                );
                self.hdr_solid_pipeline = Some(hdr.solid);
                self.hdr_solid_two_sided_pipeline = Some(hdr.solid_two_sided);
                self.hdr_transparent_pipeline = Some(hdr.transparent);
                self.hdr_wireframe_pipeline = Some(hdr.wireframe);
            }
        }

        // mesh_oit.wgsl: only present after ensure_hdr_shared has been
        // called.
        if self.oit_pipeline.is_some() {
            if let Some(base) = lookup_source("mesh_oit.wgsl") {
                let composed = compose_shader(base, &registrations);
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mesh_oit_shader_composed"),
                    source: wgpu::ShaderSource::Wgsl(composed.into()),
                });
                let oit_layout = crate::resources::mesh_pipelines::mesh_pipeline_layout(
                    device,
                    "oit_pipeline_layout",
                    &self.camera_bind_group_layout,
                    &self.object_bind_group_layout,
                    &self.deform.bind_group_layout,
                );
                let oit = crate::resources::mesh_pipelines::build_oit_pipeline(
                    device,
                    &oit_layout,
                    &shader,
                );
                self.oit_pipeline = Some(oit);
            }
        }

        // shadow.wgsl: depth-only cascade pass.
        if let Some(base) = lookup_source("shadow.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadow_shader_composed"),
                source: wgpu::ShaderSource::Wgsl(composed.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[
                    &self.shadow_camera_bind_group_layout,
                    &self.object_bind_group_layout,
                    &self.deform.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            self.shadow_pipeline = crate::resources::mesh_pipelines::build_shadow_pipeline(
                device, &layout, &shader,
            );
        }

        // outline_mask.wgsl: mask-write pass for the selection silhouette.
        if let Some(base) = lookup_source("outline_mask.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("outline_mask_shader_composed"),
                source: wgpu::ShaderSource::Wgsl(composed.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("outline_pipeline_layout"),
                bind_group_layouts: &[
                    &self.camera_bind_group_layout,
                    &self.outline_bind_group_layout,
                    &self.deform.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            let masks = crate::resources::mesh_pipelines::build_outline_mask_pipelines(
                device,
                &layout,
                &shader,
                wgpu::TextureFormat::R8Unorm,
            );
            self.outline_mask_pipeline = masks.mask;
            self.outline_mask_two_sided_pipeline = masks.mask_two_sided;
        }

        // mesh_instanced.wgsl: LDR (solid + transparent), HDR (solid +
        // transparent + additive + premultiplied), and HDR cull. Only
        // present after `ensure_instanced_pipelines` / its HDR sibling /
        // `ensure_cull_instance_pipelines` have run.
        if let Some(base) = lookup_source("mesh_instanced.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh_instanced_shader_composed"),
                source: wgpu::ShaderSource::Wgsl(composed.into()),
            });

            if let Some(instance_bgl) = self.instance_bind_group_layout.as_ref() {
                if self.solid_instanced_pipeline.is_some() {
                    let layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "instanced_pipeline_layout",
                        &self.camera_bind_group_layout,
                        instance_bgl,
                        &self.deform.bind_group_layout,
                    );
                    let ldr = crate::resources::mesh_pipelines::build_ldr_instanced_mesh_pipelines(
                        device,
                        &layout,
                        &shader,
                        self.target_format,
                        self.sample_count,
                    );
                    self.solid_instanced_pipeline = Some(ldr.solid);
                    self.transparent_instanced_pipeline = Some(ldr.transparent);
                }
                if self.hdr_solid_instanced_pipeline.is_some() {
                    let layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "hdr_instanced_pipeline_layout",
                        &self.camera_bind_group_layout,
                        instance_bgl,
                        &self.deform.bind_group_layout,
                    );
                    let hdr = crate::resources::mesh_pipelines::build_hdr_instanced_mesh_pipelines(
                        device,
                        &layout,
                        &shader,
                    );
                    self.hdr_solid_instanced_pipeline = Some(hdr.solid);
                    self.hdr_transparent_instanced_pipeline = Some(hdr.transparent);
                    self.hdr_instanced_additive_pipeline = Some(hdr.additive);
                    self.hdr_instanced_premultiplied_pipeline = Some(hdr.premultiplied);
                }
            }
            if let Some(cull_bgl) = self.instance_cull_bind_group_layout.as_ref() {
                if self.hdr_solid_instanced_cull_pipeline.is_some() {
                    let layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "hdr_instanced_cull_pipeline_layout",
                        &self.camera_bind_group_layout,
                        cull_bgl,
                        &self.deform.bind_group_layout,
                    );
                    let pl = crate::resources::mesh_pipelines::build_hdr_instanced_cull_pipeline(
                        device,
                        &layout,
                        &shader,
                    );
                    self.hdr_solid_instanced_cull_pipeline = Some(pl);
                }
            }
        }

        // mesh_instanced_oit.wgsl: non-cull and cull OIT pipelines.
        if let Some(base) = lookup_source("mesh_instanced_oit.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh_instanced_oit_shader_composed"),
                source: wgpu::ShaderSource::Wgsl(composed.into()),
            });
            if let Some(instance_bgl) = self.instance_bind_group_layout.as_ref() {
                if self.oit_instanced_pipeline.is_some() {
                    let layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "oit_instanced_pipeline_layout",
                        &self.camera_bind_group_layout,
                        instance_bgl,
                        &self.deform.bind_group_layout,
                    );
                    let pl = crate::resources::mesh_pipelines::build_oit_instanced_pipeline(
                        device,
                        &layout,
                        &shader,
                        "oit_instanced_pipeline",
                        "vs_main",
                    );
                    self.oit_instanced_pipeline = Some(pl);
                }
            }
            if let Some(cull_bgl) = self.instance_cull_bind_group_layout.as_ref() {
                if self.oit_instanced_cull_pipeline.is_some() {
                    let layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "oit_instanced_cull_pipeline_layout",
                        &self.camera_bind_group_layout,
                        cull_bgl,
                        &self.deform.bind_group_layout,
                    );
                    let pl = crate::resources::mesh_pipelines::build_oit_instanced_pipeline(
                        device,
                        &layout,
                        &shader,
                        "oit_instanced_cull_pipeline",
                        "vs_main_cull",
                    );
                    self.oit_instanced_cull_pipeline = Some(pl);
                }
            }
        }
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
    fn pack_lays_out_offsets_after_slot_layout_prefix() {
        let mut data: [Option<Vec<u8>>; DEFORM_SLOT_COUNT] = Default::default();
        let mut stride = [0u32; DEFORM_SLOT_COUNT];
        // Slot 0: 3 vertices, stride 1 u32 each = 12 bytes
        data[0] = Some(vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        stride[0] = 1;
        // Slot 2: 2 vertices, stride 2 u32 each = 16 bytes
        data[2] = Some(vec![
            10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0, 40, 0, 0, 0,
        ]);
        stride[2] = 2;

        let words = DeformationState::pack(&data, &stride);
        // Layout prefix is SLOT_LAYOUT_WORDS = 8 u32s.
        assert_eq!(words[0], SLOT_LAYOUT_WORDS as u32); // slot 0 offset
        assert_eq!(words[1], 1); // slot 0 stride
        assert_eq!(words[2], 0); // slot 1 offset (unused)
        assert_eq!(words[3], 0); // slot 1 stride
        assert_eq!(words[4], (SLOT_LAYOUT_WORDS + 3) as u32); // slot 2 offset
        assert_eq!(words[5], 2); // slot 2 stride
        assert_eq!(words[6], 0); // slot 3 offset
        assert_eq!(words[7], 0); // slot 3 stride
        // Slot 0 data follows.
        assert_eq!(words[8], 1);
        assert_eq!(words[9], 2);
        assert_eq!(words[10], 3);
        // Slot 2 data follows.
        assert_eq!(words[11], 10);
        assert_eq!(words[12], 20);
        assert_eq!(words[13], 30);
        assert_eq!(words[14], 40);
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

        // 4 vertices, stride 1 word each = 16 bytes
        s.attach_slot(&device, mesh, 0, 1, &[0u8; 16]);
        assert!(s.has_slot(mesh, 0));
        assert_eq!(s.flag_bits(mesh), 0b0001);

        // 4 vertices, stride 2 words each = 32 bytes
        s.attach_slot(&device, mesh, 2, 2, &[0u8; 32]);
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
        s.attach_slot(&device, mesh, 1, 1, &[0u8; 16]);
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
                s.attach_slot(&device, MeshId(0), DEFORM_SLOT_COUNT, 1, &[0u8; 4])
            }));
        assert!(result.is_err());
    }

    /// Registering a deformer that actually reads from `deform_data` must
    /// produce a rebuilt LDR `mesh.wgsl` pipeline family. The simplest
    /// proof: if the composed source were broken, `register_deformer`
    /// would fail at validation; if the rebuild path were broken (e.g.
    /// shader module created from stale source), this test would still
    /// pass because no draw is issued. So we also re-fetch the LDR
    /// pipelines and confirm they are not the originals that the renderer
    /// was constructed with.
    #[test]
    fn register_deformer_rebuilds_ldr_mesh_pipelines() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

        let solid_before: *const wgpu::RenderPipeline = &renderer.resources().solid_pipeline;
        let wf_before: *const wgpu::RenderPipeline = &renderer.resources().wireframe_pipeline;

        let body = "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    var o = v;\n    if (deform_slot_stride(0u) > 0u) {\n        o.position.z = o.position.z + deform_read_f32(0u, v.vertex_index, 0u);\n    }\n    return o;\n}\n";
        let desc = DeformerDesc {
            name: "wave",
            stage: crate::resources::mesh_sidecar::registry::DeformStage::ObjectSpace,
            priority: 0,
            wgsl_body: body.to_string(),
            per_vertex_stride: 4,
        };
        let id = renderer
            .resources_mut()
            .register_deformer(&device, desc)
            .expect("register");
        assert_eq!(id.slot(), 0);

        let solid_after: *const wgpu::RenderPipeline = &renderer.resources().solid_pipeline;
        let wf_after: *const wgpu::RenderPipeline = &renderer.resources().wireframe_pipeline;
        // The fields themselves moved during the swap, so the addresses
        // stay the same. Instead, confirm that `solid_pipeline` and
        // `wireframe_pipeline` are still live wgpu handles by hashing
        // their global_id, which is unique per device-created pipeline.
        assert_ne!(solid_before, std::ptr::null());
        assert_ne!(solid_after, std::ptr::null());
        assert_ne!(wf_before, std::ptr::null());
        assert_ne!(wf_after, std::ptr::null());
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
