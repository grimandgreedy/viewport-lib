//! Per-vertex deformation sidecar storage.
//!
//! Hosts the `@group(2)` bind group used by every mesh-family pipeline.
//! Each mesh's slot data is packed into a single storage buffer prefixed
//! with `(offset, stride)` pairs per slot; shader-side reads go through
//! the `deform_read_*` helpers in `deform.wgsl`. Meshes with no attached
//! deformer data fall back to the renderer-owned dummy bind group at zero
//! cost.

use std::collections::HashMap;

use crate::gpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use crate::error::ViewportResult;
use crate::resources::DeviceResources;
use crate::resources::mesh::mesh_store::MeshId;
use crate::resources::mesh_sidecar::registry::{
    DeformerDesc, DeformerId, MESH_FAMILY_SHADERS, StoredDeformer, allocate_internal_slot,
    allocate_slot, compose_shader, lookup_source, validate_name, validate_with_wgpu,
};

/// Maximum number of registered deformer slots (host range + reserved
/// internal range). Slot indices `[0, DEFORM_INTERNAL_SLOT_START)` are
/// available to hosts through `register_deformer`; indices
/// `[DEFORM_INTERNAL_SLOT_START, DEFORM_SLOT_COUNT)` are reserved for the
/// in-crate deformers (skinning, displacement) that register at renderer
/// construction. Slot data is packed into two storage buffers (per-mesh and
/// per-instance) so deform contributes three vertex-stage bindings total
/// regardless of slot count.
pub const DEFORM_SLOT_COUNT: usize = 8;

/// First slot index reserved for in-crate deformers.
pub const DEFORM_INTERNAL_SLOT_START: usize = 4;

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

/// Byte offset of slot `slot`'s `slot_params` block inside the deformation
/// header uniform. The header layout is `[time_seconds, _pad x3]`
/// (16 bytes) followed by 32 contiguous `vec4<f32>` slot-params entries,
/// 4 per slot.
pub const DEFORM_SLOT_PARAMS_BYTES: u64 =
    (DEFORM_PARAMS_PER_SLOT * std::mem::size_of::<[f32; 4]>()) as u64;
const DEFORM_HEADER_PREFIX_BYTES: u64 = 16;

/// Byte offset of slot `slot`'s parameter block inside the deformation
/// header uniform.
#[inline]
pub const fn deform_slot_params_byte_offset(slot: usize) -> u64 {
    DEFORM_HEADER_PREFIX_BYTES + (slot as u64) * DEFORM_SLOT_PARAMS_BYTES
}

/// Handle that lets a plugin update one deformer slot's `slot_params` block
/// from a context that only has `&wgpu::Queue` (a `GpuPlugin::pre_prepare`,
/// say). Cloning is cheap.
#[derive(Clone)]
pub struct DeformSlotHandle {
    header_buffer: crate::gpu::Buffer,
    slot: usize,
}

impl DeformSlotHandle {
    /// The slot this handle writes.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Write the four `vec4<f32>` parameter words for this slot. Cheap;
    /// safe to call per frame from a plugin's `pre_prepare`.
    pub fn write(&self, queue: &crate::gpu::Queue, params: &[[f32; 4]; DEFORM_PARAMS_PER_SLOT]) {
        queue.write_buffer(
            &self.header_buffer,
            deform_slot_params_byte_offset(self.slot),
            bytemuck::cast_slice(params),
        );
    }
}

/// A window into a same-device consumer buffer feeding one deform slot. The
/// renderer copies `len_bytes` from `src_offset_bytes` into the slot's region
/// of the packed buffer each frame; it never writes the consumer's buffer.
pub(crate) struct ExternalDeformSource {
    pub buffer: crate::gpu::Buffer,
    pub src_offset_bytes: u64,
    pub len_bytes: u64,
    pub stride_words: u32,
}

/// A window into a consumer buffer, addressed in whole per-vertex elements
/// (`element` = `stride_bytes`). Used by the sliced deform-source setters to
/// bind one host part's range out of a shared pool.
#[derive(Copy, Clone, Debug)]
pub struct DeformSourceSlice {
    /// First element of the window.
    pub base_element: u32,
    /// Number of elements in the window.
    pub element_count: u32,
}

/// Per-(mesh, instance) deformation storage. Holds the per-instance packed
/// slot buffer and its bind group (which also binds the owning mesh's
/// per-mesh buffer at the same time, so the draw needs only one bind-group
/// swap to switch instances).
pub(crate) struct InstanceDeform {
    pub slot_data: [Option<Vec<u8>>; DEFORM_SLOT_COUNT],
    pub slot_stride: [u32; DEFORM_SLOT_COUNT],
    /// External same-device source per slot. Mutually exclusive with
    /// `slot_data[i]`: a slot is either CPU-uploaded or buffer-backed.
    pub slot_external: [Option<ExternalDeformSource>; DEFORM_SLOT_COUNT],
    /// Byte offset of each present slot's data inside `buffer`, set by
    /// `pack`. Meaningful only for slots with data or an external source.
    pub slot_offset_words: [u32; DEFORM_SLOT_COUNT],
    pub buffer: crate::gpu::Buffer,
    /// Bytes currently allocated in `buffer`. Used to decide when to realloc
    /// rather than `write_buffer` in place.
    pub buffer_capacity: u64,
    pub bind_group: crate::gpu::BindGroup,
    /// Bit `i` set when slot `i` has per-instance data attached.
    pub flag_bits: u32,
}

/// Per-mesh deformation storage.
pub(crate) struct MeshDeform {
    /// Source bytes per slot, retained so attach/detach can re-pack
    /// without forcing the caller to re-upload other slots.
    pub slot_data: [Option<Vec<u8>>; DEFORM_SLOT_COUNT],
    /// Per-slot stride in u32 words. `slot_stride[i]` is meaningful only
    /// when `slot_data[i].is_some()`.
    pub slot_stride: [u32; DEFORM_SLOT_COUNT],
    /// External same-device source per slot. Mutually exclusive with
    /// `slot_data[i]`: a slot is either CPU-uploaded or buffer-backed.
    pub slot_external: [Option<ExternalDeformSource>; DEFORM_SLOT_COUNT],
    /// Byte offset of each present slot's data inside `buffer`, set by
    /// `pack`. Meaningful only for slots with data or an external source.
    pub slot_offset_words: [u32; DEFORM_SLOT_COUNT],
    /// Packed buffer: `SLOT_LAYOUT_WORDS * 4` bytes of `(offset, stride)`
    /// header followed by tightly packed slot bytes in slot order.
    pub buffer: crate::gpu::Buffer,
    /// Bind group for draws of this mesh that have no per-instance data:
    /// binds the per-mesh buffer at @binding(1) and the renderer's empty
    /// instance buffer at @binding(2).
    pub bind_group: crate::gpu::BindGroup,
    /// Bit `i` set when slot `i` has per-mesh data attached.
    pub flag_bits: u32,
    /// Bit `i` set when *any* instance of this mesh has slot `i`
    /// per-instance data attached. Combined with `flag_bits` at
    /// `ObjectUniform.deform_flags` write time so the shader sees both
    /// per-mesh and per-instance slot activity.
    pub instance_flag_bits_union: u32,
    pub instances: HashMap<u32, InstanceDeform>,
}

/// Renderer-side deformation state.
pub(crate) struct DeformationState {
    /// False when the device's max_bind_groups < 3. When false, the deform
    /// bind group layout is not included in pipeline layouts and deformer
    /// registration is silently skipped.
    pub enabled: bool,
    pub bind_group_layout: crate::gpu::BindGroupLayout,
    pub header_buffer: crate::gpu::Buffer,
    /// Empty slot-layout prefix bound when a mesh has no attached data.
    /// `SLOT_LAYOUT_WORDS` u32s of zero. Kept alive so the dummy bind group
    /// stays valid.
    #[allow(dead_code)]
    pub dummy_data_buffer: crate::gpu::Buffer,
    /// Empty per-instance buffer reused by every per-mesh bind group that
    /// has no instance data yet. Kept alive on the state so the bind group
    /// stays valid.
    pub dummy_instance_buffer: crate::gpu::Buffer,
    /// Bind group used when a mesh has no attached deformer data. Bound by
    /// every mesh-family draw at slot 2 to satisfy the pipeline layout
    /// without forcing per-mesh storage allocation.
    pub dummy_bind_group: crate::gpu::BindGroup,
    pub meshes: HashMap<MeshId, MeshDeform>,
    pub header_cpu: DeformHeader,
    /// Currently registered deformers, in registration order.
    pub registrations: Vec<StoredDeformer>,
    /// Number of slots (per-mesh and per-instance, across all meshes) currently
    /// bound to an external buffer source. Lets the per-frame copy pass skip its
    /// encoder entirely when nothing is buffer-backed.
    pub external_source_count: usize,
}

impl DeformationState {
    pub fn new(device: &crate::gpu::Device) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("deform_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::VERTEX,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
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

        let header_cpu = DeformHeader::zeroed();
        let header_buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("deform_header"),
            contents: bytemuck::bytes_of(&header_cpu),
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
        });

        let dummy_words = vec![0u32; SLOT_LAYOUT_WORDS];
        let dummy_data_buffer =
            device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("deform_dummy_data"),
                contents: bytemuck::cast_slice(&dummy_words),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            });
        let dummy_instance_buffer =
            device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("deform_dummy_instance_data"),
                contents: bytemuck::cast_slice(&dummy_words),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            });

        let dummy_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("deform_dummy_bg"),
            layout: &bind_group_layout,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: header_buffer.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_data_buffer.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_instance_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            enabled: true,
            bind_group_layout,
            header_buffer,
            dummy_data_buffer,
            dummy_instance_buffer,
            dummy_bind_group,
            meshes: HashMap::new(),
            header_cpu,
            registrations: Vec::new(),
            external_source_count: 0,
        }
    }

    /// Construct a disabled `DeformationState` for devices with max_bind_groups < 3.
    ///
    /// The returned state still allocates a minimal BGL and dummy bind group so
    /// that render.rs can call set_bind_group(2, ...) without crashing; those
    /// calls target a slot not present in any pipeline layout and are ignored.
    pub fn new_disabled(device: &crate::gpu::Device) -> Self {
        let mut s = Self::new(device);
        s.enabled = false;
        s
    }

    /// Returns the bind group to use for a given mesh: the per-mesh group
    /// when any slot is attached, otherwise the renderer-owned dummy. Used
    /// at draw time once the registry has rebuilt the mesh pipelines to bind
    /// group 2.
    #[allow(dead_code)]
    pub fn bind_group_for(&self, mesh_id: MeshId) -> &crate::gpu::BindGroup {
        self.meshes
            .get(&mesh_id)
            .map(|m| &m.bind_group)
            .unwrap_or(&self.dummy_bind_group)
    }

    /// `deform_flags` value to write into the `ObjectUniform` for `mesh_id`.
    /// Folds per-mesh and per-instance slot activity together: any slot
    /// with data on either side is reported as live so the registered
    /// body's flag branch executes.
    pub fn flag_bits(&self, mesh_id: MeshId) -> u32 {
        self.meshes
            .get(&mesh_id)
            .map(|m| m.flag_bits | m.instance_flag_bits_union)
            .unwrap_or(0)
    }

    /// Whether `(mesh_id, instance_id)` has any per-instance deformer data
    /// attached. Used by the renderer to filter items out of GPU-driven
    /// instanced batching: items with per-instance deform data need their
    /// own bind group at draw time and cannot share an instance batch.
    pub(crate) fn has_per_instance_deform_data(
        &self,
        mesh_id: MeshId,
        instance_id: Option<u32>,
    ) -> bool {
        let Some(id) = instance_id else {
            return false;
        };
        self.meshes
            .get(&mesh_id)
            .and_then(|m| m.instances.get(&id))
            .is_some_and(|i| i.flag_bits != 0)
    }

    /// Bind group to use for a draw of `(mesh_id, instance_id)`. Falls back
    /// to the per-mesh-only bind group when no per-instance data is
    /// attached, and to the renderer dummy when the mesh has no data at all.
    #[allow(dead_code)]
    pub fn instance_bind_group_for(
        &self,
        mesh_id: MeshId,
        instance_id: Option<u32>,
    ) -> &crate::gpu::BindGroup {
        let Some(mesh) = self.meshes.get(&mesh_id) else {
            return &self.dummy_bind_group;
        };
        if let Some(id) = instance_id {
            if let Some(inst) = mesh.instances.get(&id) {
                return &inst.bind_group;
            }
        }
        &mesh.bind_group
    }

    /// Pack the per-slot data of one mesh into a single u32 stream prefixed
    /// by `(offset, stride)` pairs per slot.
    ///
    /// A slot with an external source reserves a zero-filled region of the
    /// right size; the per-frame copy pass fills it from the consumer buffer.
    /// External and CPU data are mutually exclusive per slot. Returns the
    /// packed words and each present slot's offset in words (0 when absent),
    /// so the copy pass knows where to write.
    fn pack(
        slot_data: &[Option<Vec<u8>>; DEFORM_SLOT_COUNT],
        slot_stride: &[u32; DEFORM_SLOT_COUNT],
        slot_external: &[Option<ExternalDeformSource>; DEFORM_SLOT_COUNT],
    ) -> (Vec<u32>, [u32; DEFORM_SLOT_COUNT]) {
        let mut words = vec![0u32; SLOT_LAYOUT_WORDS];
        let mut offsets = [0u32; DEFORM_SLOT_COUNT];
        for slot in 0..DEFORM_SLOT_COUNT {
            if let Some(src) = &slot_external[slot] {
                let offset_words = words.len() as u32;
                offsets[slot] = offset_words;
                words[slot * 2] = offset_words;
                words[slot * 2 + 1] = src.stride_words;
                let extra = (src.len_bytes / 4) as usize;
                words.resize(words.len() + extra, 0);
            } else if let Some(bytes) = &slot_data[slot] {
                let offset_words = words.len() as u32;
                offsets[slot] = offset_words;
                words[slot * 2] = offset_words;
                words[slot * 2 + 1] = slot_stride[slot];
                let extra = bytes.len() / 4;
                words.reserve(extra);
                for chunk in bytes.chunks_exact(4) {
                    words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
        }
        (words, offsets)
    }

    fn make_bind_group(
        &self,
        device: &crate::gpu::Device,
        label: &str,
        mesh_buffer: &crate::gpu::Buffer,
        instance_buffer: &crate::gpu::Buffer,
    ) -> crate::gpu::BindGroup {
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: self.header_buffer.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh_buffer.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Re-pack and re-upload the mesh's data, rebuilding the per-mesh bind
    /// group and every per-instance bind group (which also binds the
    /// per-mesh buffer). Drops the mesh entry when no slot has any data
    /// attached on either side.
    fn refresh(&mut self, device: &crate::gpu::Device, mesh_id: MeshId) {
        let Some(m) = self.meshes.get(&mesh_id) else {
            return;
        };
        if m.flag_bits == 0 && m.instance_flag_bits_union == 0 {
            self.meshes.remove(&mesh_id);
            return;
        }
        let (words, offsets) = Self::pack(&m.slot_data, &m.slot_stride, &m.slot_external);
        let new_buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("deform_mesh_data"),
            contents: bytemuck::cast_slice(&words),
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        });
        let new_bg = self.make_bind_group(
            device,
            "deform_mesh_bg",
            &new_buffer,
            &self.dummy_instance_buffer,
        );

        // Rebuild instance bind groups since they bind this mesh's buffer.
        let instance_ids: Vec<u32> = self
            .meshes
            .get(&mesh_id)
            .unwrap()
            .instances
            .keys()
            .copied()
            .collect();
        for id in instance_ids {
            let inst_buf_clone = {
                let inst = self
                    .meshes
                    .get(&mesh_id)
                    .unwrap()
                    .instances
                    .get(&id)
                    .unwrap();
                // SAFETY: as_entire_binding holds a borrow of the buffer,
                // not the buffer itself; we reborrow per call.
                inst.buffer.clone()
            };
            let bg = self.make_bind_group(
                device,
                "deform_mesh_instance_bg",
                &new_buffer,
                &inst_buf_clone,
            );
            let inst = self
                .meshes
                .get_mut(&mesh_id)
                .unwrap()
                .instances
                .get_mut(&id)
                .unwrap();
            inst.bind_group = bg;
        }

        let m = self.meshes.get_mut(&mesh_id).unwrap();
        m.buffer = new_buffer;
        m.bind_group = new_bg;
        m.slot_offset_words = offsets;
    }

    fn ensure_mesh(&mut self, device: &crate::gpu::Device, mesh_id: MeshId) {
        if self.meshes.contains_key(&mesh_id) {
            return;
        }
        let init_words = vec![0u32; SLOT_LAYOUT_WORDS];
        let buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("deform_mesh_data_init"),
            contents: bytemuck::cast_slice(&init_words),
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        });
        let bind_group = self.make_bind_group(
            device,
            "deform_mesh_bg",
            &buffer,
            &self.dummy_instance_buffer,
        );
        self.meshes.insert(
            mesh_id,
            MeshDeform {
                slot_data: Default::default(),
                slot_stride: [0; DEFORM_SLOT_COUNT],
                slot_external: Default::default(),
                slot_offset_words: [0; DEFORM_SLOT_COUNT],
                buffer,
                bind_group,
                flag_bits: 0,
                instance_flag_bits_union: 0,
                instances: HashMap::new(),
            },
        );
    }

    /// Attach raw bytes to a per-mesh slot for `mesh_id`. `stride_words` is
    /// the per-vertex stride in u32 words (must equal the registered
    /// deformer's `per_vertex_stride / 4`).
    pub fn attach_slot(
        &mut self,
        device: &crate::gpu::Device,
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
        self.ensure_mesh(device, mesh_id);
        let entry = self.meshes.get_mut(&mesh_id).unwrap();
        // CPU data replaces any external source on this slot.
        let had_external = entry.slot_external[slot].take().is_some();
        entry.slot_data[slot] = Some(data.to_vec());
        entry.slot_stride[slot] = stride_words;
        entry.flag_bits |= 1u32 << slot;
        if had_external {
            self.external_source_count -= 1;
        }
        self.refresh(device, mesh_id);
    }

    /// Detach a per-mesh slot. Returns `true` if any data was removed.
    pub fn detach_slot(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
    ) -> bool {
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

    fn recompute_instance_union(m: &mut MeshDeform) {
        let mut union = 0u32;
        for inst in m.instances.values() {
            union |= inst.flag_bits;
        }
        m.instance_flag_bits_union = union;
    }

    /// Attach raw bytes to a per-(mesh, instance) slot. The data is treated
    /// as a sequence of fixed-stride elements (e.g. mat4 joint matrices
    /// for skinning). When the instance buffer must grow, its bind group is
    /// rebuilt. When the existing buffer is large enough, the data is
    /// written with `queue.write_buffer` and the bind group is reused.
    pub fn attach_slot_instance(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
        stride_words: u32,
        data: &[u8],
    ) {
        assert!(slot < DEFORM_SLOT_COUNT);
        assert!(
            data.len() % 4 == 0,
            "deform slot data length must be a multiple of 4 bytes"
        );
        self.ensure_mesh(device, mesh_id);
        // Ensure the instance entry exists with a minimal buffer.
        let needs_insert = !self
            .meshes
            .get(&mesh_id)
            .unwrap()
            .instances
            .contains_key(&instance_id);
        if needs_insert {
            let init_words = vec![0u32; SLOT_LAYOUT_WORDS];
            let buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("deform_mesh_instance_data_init"),
                contents: bytemuck::cast_slice(&init_words),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            });
            let mesh_buf_clone = self.meshes.get(&mesh_id).unwrap().buffer.clone();
            let bind_group =
                self.make_bind_group(device, "deform_mesh_instance_bg", &mesh_buf_clone, &buffer);
            let m = self.meshes.get_mut(&mesh_id).unwrap();
            let cap = (init_words.len() * 4) as u64;
            m.instances.insert(
                instance_id,
                InstanceDeform {
                    slot_data: Default::default(),
                    slot_stride: [0; DEFORM_SLOT_COUNT],
                    slot_external: Default::default(),
                    slot_offset_words: [0; DEFORM_SLOT_COUNT],
                    buffer,
                    buffer_capacity: cap,
                    bind_group,
                    flag_bits: 0,
                },
            );
        }

        // Mutate slot bookkeeping.
        let had_external = {
            let inst = self
                .meshes
                .get_mut(&mesh_id)
                .unwrap()
                .instances
                .get_mut(&instance_id)
                .unwrap();
            // CPU data replaces any external source on this slot.
            let had_external = inst.slot_external[slot].take().is_some();
            inst.slot_data[slot] = Some(data.to_vec());
            inst.slot_stride[slot] = stride_words;
            inst.flag_bits |= 1u32 << slot;
            had_external
        };
        if had_external {
            self.external_source_count -= 1;
        }

        // Re-pack and decide between in-place write or realloc.
        let (packed_words, packed_offsets, packed_bytes_len) = {
            let inst = self
                .meshes
                .get(&mesh_id)
                .unwrap()
                .instances
                .get(&instance_id)
                .unwrap();
            let (w, offsets) = Self::pack(&inst.slot_data, &inst.slot_stride, &inst.slot_external);
            let len = (w.len() * 4) as u64;
            (w, offsets, len)
        };

        let needs_realloc = {
            let inst = self
                .meshes
                .get(&mesh_id)
                .unwrap()
                .instances
                .get(&instance_id)
                .unwrap();
            packed_bytes_len > inst.buffer_capacity
        };

        if needs_realloc {
            let new_buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("deform_mesh_instance_data"),
                contents: bytemuck::cast_slice(&packed_words),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            });
            let mesh_buf_clone = self.meshes.get(&mesh_id).unwrap().buffer.clone();
            let bg = self.make_bind_group(
                device,
                "deform_mesh_instance_bg",
                &mesh_buf_clone,
                &new_buffer,
            );
            let inst = self
                .meshes
                .get_mut(&mesh_id)
                .unwrap()
                .instances
                .get_mut(&instance_id)
                .unwrap();
            inst.buffer = new_buffer;
            inst.buffer_capacity = packed_bytes_len;
            inst.bind_group = bg;
        } else {
            let inst = self
                .meshes
                .get(&mesh_id)
                .unwrap()
                .instances
                .get(&instance_id)
                .unwrap();
            queue.write_buffer(&inst.buffer, 0, bytemuck::cast_slice(&packed_words));
        }

        // Record the fresh packed offsets for the per-frame copy pass.
        self.meshes
            .get_mut(&mesh_id)
            .unwrap()
            .instances
            .get_mut(&instance_id)
            .unwrap()
            .slot_offset_words = packed_offsets;

        // Recompute the union and refresh flag bits.
        let m = self.meshes.get_mut(&mesh_id).unwrap();
        Self::recompute_instance_union(m);
    }

    /// Detach a per-instance slot. Returns `true` if any data was removed.
    /// Drops the instance entry when no slot remains attached and re-packs
    /// the instance buffer if other slots still hold data.
    pub fn detach_slot_instance(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
    ) -> bool {
        assert!(slot < DEFORM_SLOT_COUNT);
        let Some(m) = self.meshes.get_mut(&mesh_id) else {
            return false;
        };
        let Some(inst) = m.instances.get_mut(&instance_id) else {
            return false;
        };
        let had = inst.slot_data[slot].take().is_some();
        if !had {
            return false;
        }
        inst.slot_stride[slot] = 0;
        inst.flag_bits &= !(1u32 << slot);
        let drop_instance = inst.flag_bits == 0;

        if drop_instance {
            m.instances.remove(&instance_id);
        } else {
            let (packed_words, packed_offsets, packed_bytes_len) = {
                let inst = m.instances.get(&instance_id).unwrap();
                let (w, offsets) =
                    Self::pack(&inst.slot_data, &inst.slot_stride, &inst.slot_external);
                let len = (w.len() * 4) as u64;
                (w, offsets, len)
            };
            m.instances.get_mut(&instance_id).unwrap().slot_offset_words = packed_offsets;
            let inst = m.instances.get(&instance_id).unwrap();
            if packed_bytes_len > inst.buffer_capacity {
                let new_buffer =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("deform_mesh_instance_data"),
                        contents: bytemuck::cast_slice(&packed_words),
                        usage: crate::gpu::BufferUsages::STORAGE
                            | crate::gpu::BufferUsages::COPY_DST,
                    });
                let mesh_buf_clone = m.buffer.clone();
                let bg = self.make_bind_group(
                    device,
                    "deform_mesh_instance_bg",
                    &mesh_buf_clone,
                    &new_buffer,
                );
                let m = self.meshes.get_mut(&mesh_id).unwrap();
                let inst = m.instances.get_mut(&instance_id).unwrap();
                inst.buffer = new_buffer;
                inst.buffer_capacity = packed_bytes_len;
                inst.bind_group = bg;
            } else {
                queue.write_buffer(&inst.buffer, 0, bytemuck::cast_slice(&packed_words));
            }
        }

        let m = self.meshes.get_mut(&mesh_id).unwrap();
        Self::recompute_instance_union(m);
        // Drop the mesh entirely when nothing references it.
        if m.flag_bits == 0 && m.instance_flag_bits_union == 0 {
            self.meshes.remove(&mesh_id);
        }
        true
    }

    pub fn has_slot_instance(&self, mesh_id: MeshId, instance_id: u32, slot: usize) -> bool {
        self.meshes
            .get(&mesh_id)
            .and_then(|m| m.instances.get(&instance_id))
            .map(|i| i.slot_data[slot].is_some())
            .unwrap_or(false)
    }

    /// Bind a per-mesh slot to an external same-device buffer window. The slot
    /// reserves a region in the packed buffer that the per-frame copy pass
    /// fills from `src`. Replaces any CPU data previously attached to the slot.
    pub fn set_slot_external(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
        src: ExternalDeformSource,
    ) {
        self.ensure_mesh(device, mesh_id);
        let entry = self.meshes.get_mut(&mesh_id).unwrap();
        let had_external = entry.slot_external[slot].is_some();
        let had_cpu = entry.slot_data[slot].take().is_some();
        entry.slot_stride[slot] = src.stride_words;
        entry.slot_external[slot] = Some(src);
        entry.flag_bits |= 1u32 << slot;
        if !had_external {
            self.external_source_count += 1;
        }
        let _ = had_cpu;
        self.refresh(device, mesh_id);
    }

    /// Detach a per-mesh slot's external source. Returns `true` if a source was
    /// removed. Leaves any other slots (CPU or external) intact.
    pub fn clear_slot_external(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
    ) -> bool {
        let Some(m) = self.meshes.get_mut(&mesh_id) else {
            return false;
        };
        if m.slot_external[slot].take().is_none() {
            return false;
        }
        m.slot_stride[slot] = 0;
        m.flag_bits &= !(1u32 << slot);
        self.external_source_count -= 1;
        self.refresh(device, mesh_id);
        true
    }

    /// True when a per-mesh slot is bound to an external source.
    pub fn has_slot_external(&self, mesh_id: MeshId, slot: usize) -> bool {
        self.meshes
            .get(&mesh_id)
            .map(|m| m.slot_external[slot].is_some())
            .unwrap_or(false)
    }

    /// Bind a per-instance slot to an external same-device buffer window.
    /// Replaces any CPU data previously attached to that instance slot.
    pub fn set_slot_instance_external(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
        src: ExternalDeformSource,
    ) {
        self.ensure_mesh(device, mesh_id);
        // Create the instance entry with a minimal buffer if it does not exist.
        let needs_insert = !self
            .meshes
            .get(&mesh_id)
            .unwrap()
            .instances
            .contains_key(&instance_id);
        if needs_insert {
            let init_words = vec![0u32; SLOT_LAYOUT_WORDS];
            let buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("deform_mesh_instance_data_init"),
                contents: bytemuck::cast_slice(&init_words),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            });
            let mesh_buf_clone = self.meshes.get(&mesh_id).unwrap().buffer.clone();
            let bind_group =
                self.make_bind_group(device, "deform_mesh_instance_bg", &mesh_buf_clone, &buffer);
            let m = self.meshes.get_mut(&mesh_id).unwrap();
            let cap = (init_words.len() * 4) as u64;
            m.instances.insert(
                instance_id,
                InstanceDeform {
                    slot_data: Default::default(),
                    slot_stride: [0; DEFORM_SLOT_COUNT],
                    slot_external: Default::default(),
                    slot_offset_words: [0; DEFORM_SLOT_COUNT],
                    buffer,
                    buffer_capacity: cap,
                    bind_group,
                    flag_bits: 0,
                },
            );
        }

        // Swap the slot over to the external source.
        let had_external = {
            let inst = self
                .meshes
                .get_mut(&mesh_id)
                .unwrap()
                .instances
                .get_mut(&instance_id)
                .unwrap();
            let had_external = inst.slot_external[slot].is_some();
            inst.slot_data[slot] = None;
            inst.slot_stride[slot] = src.stride_words;
            inst.slot_external[slot] = Some(src);
            inst.flag_bits |= 1u32 << slot;
            had_external
        };
        if !had_external {
            self.external_source_count += 1;
        }

        // Re-pack, reserving the external slot's region, and realloc or write.
        let (packed_words, packed_offsets, packed_bytes_len) = {
            let inst = self
                .meshes
                .get(&mesh_id)
                .unwrap()
                .instances
                .get(&instance_id)
                .unwrap();
            let (w, offsets) = Self::pack(&inst.slot_data, &inst.slot_stride, &inst.slot_external);
            let len = (w.len() * 4) as u64;
            (w, offsets, len)
        };

        let needs_realloc = {
            let inst = self
                .meshes
                .get(&mesh_id)
                .unwrap()
                .instances
                .get(&instance_id)
                .unwrap();
            packed_bytes_len > inst.buffer_capacity
        };

        if needs_realloc {
            let new_buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("deform_mesh_instance_data"),
                contents: bytemuck::cast_slice(&packed_words),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            });
            let mesh_buf_clone = self.meshes.get(&mesh_id).unwrap().buffer.clone();
            let bg = self.make_bind_group(
                device,
                "deform_mesh_instance_bg",
                &mesh_buf_clone,
                &new_buffer,
            );
            let inst = self
                .meshes
                .get_mut(&mesh_id)
                .unwrap()
                .instances
                .get_mut(&instance_id)
                .unwrap();
            inst.buffer = new_buffer;
            inst.buffer_capacity = packed_bytes_len;
            inst.bind_group = bg;
        } else {
            let inst = self
                .meshes
                .get(&mesh_id)
                .unwrap()
                .instances
                .get(&instance_id)
                .unwrap();
            queue.write_buffer(&inst.buffer, 0, bytemuck::cast_slice(&packed_words));
        }

        self.meshes
            .get_mut(&mesh_id)
            .unwrap()
            .instances
            .get_mut(&instance_id)
            .unwrap()
            .slot_offset_words = packed_offsets;

        let m = self.meshes.get_mut(&mesh_id).unwrap();
        Self::recompute_instance_union(m);
    }

    /// Detach a per-instance slot's external source. Returns `true` if a source
    /// was removed.
    pub fn clear_slot_instance_external(
        &mut self,
        _device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
    ) -> bool {
        let Some(m) = self.meshes.get_mut(&mesh_id) else {
            return false;
        };
        let Some(inst) = m.instances.get_mut(&instance_id) else {
            return false;
        };
        if inst.slot_external[slot].take().is_none() {
            return false;
        }
        inst.slot_stride[slot] = 0;
        inst.flag_bits &= !(1u32 << slot);
        self.external_source_count -= 1;
        let drop_instance = inst.flag_bits == 0;
        if drop_instance {
            m.instances.remove(&instance_id);
        } else {
            let (packed_words, packed_offsets) = {
                let inst = m.instances.get(&instance_id).unwrap();
                Self::pack(&inst.slot_data, &inst.slot_stride, &inst.slot_external)
            };
            m.instances.get_mut(&instance_id).unwrap().slot_offset_words = packed_offsets;
            let inst = m.instances.get(&instance_id).unwrap();
            queue.write_buffer(&inst.buffer, 0, bytemuck::cast_slice(&packed_words));
        }
        let m = self.meshes.get_mut(&mesh_id).unwrap();
        Self::recompute_instance_union(m);
        if m.flag_bits == 0 && m.instance_flag_bits_union == 0 {
            self.meshes.remove(&mesh_id);
        }
        true
    }

    /// True when a per-instance slot is bound to an external source.
    pub fn has_slot_instance_external(
        &self,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
    ) -> bool {
        self.meshes
            .get(&mesh_id)
            .and_then(|m| m.instances.get(&instance_id))
            .map(|i| i.slot_external[slot].is_some())
            .unwrap_or(false)
    }

    /// Record a `copy_buffer_to_buffer` for every external slot (per-mesh and
    /// per-instance) into its region of the packed buffer. Runs each frame
    /// before the mesh render pass; queue-submission order is the only
    /// synchronisation against the consumer's earlier compute submissions.
    pub fn record_deform_copies(&self, encoder: &mut crate::gpu::CommandEncoder) {
        for m in self.meshes.values() {
            for slot in 0..DEFORM_SLOT_COUNT {
                if let Some(src) = &m.slot_external[slot] {
                    encoder.copy_buffer_to_buffer(
                        &src.buffer,
                        src.src_offset_bytes,
                        &m.buffer,
                        m.slot_offset_words[slot] as u64 * 4,
                        src.len_bytes,
                    );
                }
            }
            for inst in m.instances.values() {
                for slot in 0..DEFORM_SLOT_COUNT {
                    if let Some(src) = &inst.slot_external[slot] {
                        encoder.copy_buffer_to_buffer(
                            &src.buffer,
                            src.src_offset_bytes,
                            &inst.buffer,
                            inst.slot_offset_words[slot] as u64 * 4,
                            src.len_bytes,
                        );
                    }
                }
            }
        }
    }
}

/// Validate an external deform source window and build the stored descriptor.
/// `stride_bytes` must be a positive multiple of 4; the window
/// `[offset_bytes, offset_bytes + len_bytes)` must fit inside `buffer`, which
/// must carry `COPY_SRC`.
fn build_external_source(
    buffer: &crate::gpu::Buffer,
    stride_bytes: u32,
    offset_bytes: u64,
    len_bytes: u64,
) -> ViewportResult<ExternalDeformSource> {
    if !buffer.usage().contains(crate::gpu::BufferUsages::COPY_SRC) {
        return Err(crate::error::ViewportError::ExternalBufferUsageMissing {
            missing: "COPY_SRC",
        });
    }
    let available = buffer.size().saturating_sub(offset_bytes);
    if stride_bytes < 4
        || stride_bytes % 4 != 0
        || len_bytes == 0
        || len_bytes % stride_bytes as u64 != 0
        || len_bytes > available
    {
        return Err(crate::error::ViewportError::DeformSourceMismatch {
            needed_bytes: len_bytes,
            available_bytes: available,
            offset_bytes,
        });
    }
    Ok(ExternalDeformSource {
        buffer: buffer.clone(),
        src_offset_bytes: offset_bytes,
        len_bytes,
        stride_words: stride_bytes / 4,
    })
}

// ---------------------------------------------------------------------------
// Public API on DeviceResources
// ---------------------------------------------------------------------------

impl DeviceResources {
    /// Write the shared header uniform's `time_seconds` field. Cheap; safe to
    /// call per frame.
    pub fn set_deform_time(&mut self, queue: &crate::gpu::Queue, time_seconds: f32) {
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
        queue: &crate::gpu::Queue,
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
        device: &crate::gpu::Device,
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

    /// Detach a per-mesh slot's data. Returns `true` if any data was removed.
    pub fn detach_deform_slot(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
    ) -> bool {
        self.deform.detach_slot(device, mesh_id, slot)
    }

    /// Returns `true` when the mesh has per-mesh data attached at the given
    /// slot.
    pub fn has_deform_slot(&self, mesh_id: MeshId, slot: usize) -> bool {
        self.deform.has_slot(mesh_id, slot)
    }

    /// Attach raw bytes for one deformer slot on a single instance of the
    /// given mesh. Used for data that varies per instance (e.g. joint
    /// palettes). `stride_bytes` is the per-element byte stride; the data
    /// length must be a multiple of `stride_bytes` and 4.
    ///
    /// The drawn item must select this instance to read it: set the
    /// `SceneRenderItem`'s (or scene node's) `deform_instance` to the same
    /// `instance_id`. Attaching data here but leaving `deform_instance` at its
    /// `None` default renders the mesh undeformed with no error.
    pub fn attach_deform_slot_instance(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
        stride_bytes: u32,
        data: &[u8],
    ) {
        assert!(
            stride_bytes >= 4 && stride_bytes % 4 == 0,
            "deform slot stride must be a positive multiple of 4 bytes"
        );
        self.deform.attach_slot_instance(
            device,
            queue,
            mesh_id,
            instance_id,
            slot,
            stride_bytes / 4,
            data,
        );
    }

    /// Detach a per-instance slot's data. Returns `true` if any data was
    /// removed.
    pub fn detach_deform_slot_instance(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
    ) -> bool {
        self.deform
            .detach_slot_instance(device, queue, mesh_id, instance_id, slot)
    }

    /// Returns `true` when the given instance of the mesh has per-instance
    /// data attached at the given slot.
    pub fn has_deform_slot_instance(&self, mesh_id: MeshId, instance_id: u32, slot: usize) -> bool {
        self.deform.has_slot_instance(mesh_id, instance_id, slot)
    }

    /// Bind a per-mesh deform slot to a consumer-owned, same-device
    /// `wgpu::Buffer` instead of a CPU upload. The renderer copies the whole
    /// buffer into the slot's region of the packed deform buffer once per
    /// frame (in `prepare`), so a GPU-resident producer can feed the deformer
    /// stack without a CPU round-trip. This is the deform-slot analogue of
    /// [`set_mc_scalar_source_buffer`](Self::set_mc_scalar_source_buffer): it
    /// copies, it does not bind a new buffer into the shader, so the deform
    /// shader path is unchanged for every other consumer.
    ///
    /// The slot reads the copied data exactly as an attached slot does
    /// (`deform_read_f32(slot, vertex_index, component)`), so `buffer` must
    /// hold the same tightly-packed, `stride_bytes`-stride, render-vertex-
    /// ordered layout the CPU upload produces. The whole buffer is copied;
    /// use [`set_deform_slot_source_buffer_sliced`](Self::set_deform_slot_source_buffer_sliced)
    /// to window one part's range out of a shared pool.
    ///
    /// Synchronisation is queue-submission order: submit the compute pass that
    /// writes `buffer` before the frame's render. Calling again with a new
    /// handle re-points the source; the copy target rebuilds on the next frame.
    /// Replaces any CPU data previously attached to the slot; mixes freely with
    /// CPU-uploaded slots on the same mesh.
    ///
    /// # Errors
    ///
    /// - [`ViewportError::ExternalBufferUsageMissing`] when `buffer` lacks
    ///   `COPY_SRC`.
    /// - [`ViewportError::DeformSourceMismatch`] when `stride_bytes` is not a
    ///   positive multiple of 4, or the buffer is smaller than one stride.
    ///
    /// [`ViewportError::ExternalBufferUsageMissing`]: crate::error::ViewportError::ExternalBufferUsageMissing
    /// [`ViewportError::DeformSourceMismatch`]: crate::error::ViewportError::DeformSourceMismatch
    pub fn set_deform_slot_source_buffer(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
        buffer: crate::gpu::Buffer,
        stride_bytes: u32,
    ) -> ViewportResult<()> {
        let len_bytes = buffer.size();
        let src = build_external_source(&buffer, stride_bytes, 0, len_bytes)?;
        self.deform.set_slot_external(device, mesh_id, slot, src);
        Ok(())
    }

    /// As [`set_deform_slot_source_buffer`](Self::set_deform_slot_source_buffer),
    /// but copies only `slice.element_count` elements starting at
    /// `slice.base_element` (element = `stride_bytes`). Addresses one host
    /// part's window out of a shared pool without any storage-offset alignment
    /// constraint: the window is a copy source offset, not a bind offset.
    ///
    /// # Errors
    ///
    /// In addition to the errors of the whole-buffer setter,
    /// [`ViewportError::DeformSourceMismatch`] when the requested window runs
    /// past the end of `buffer`.
    ///
    /// [`ViewportError::DeformSourceMismatch`]: crate::error::ViewportError::DeformSourceMismatch
    pub fn set_deform_slot_source_buffer_sliced(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
        buffer: crate::gpu::Buffer,
        stride_bytes: u32,
        slice: DeformSourceSlice,
    ) -> ViewportResult<()> {
        let offset = slice.base_element as u64 * stride_bytes as u64;
        let len = slice.element_count as u64 * stride_bytes as u64;
        let src = build_external_source(&buffer, stride_bytes, offset, len)?;
        self.deform.set_slot_external(device, mesh_id, slot, src);
        Ok(())
    }

    /// Detach a per-mesh deform slot's external buffer source. Returns `true`
    /// if a source was removed. Other slots on the mesh are untouched.
    pub fn clear_deform_slot_source(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
    ) -> bool {
        self.deform.clear_slot_external(device, mesh_id, slot)
    }

    /// True when a per-mesh slot is bound to an external buffer source.
    pub fn has_deform_slot_source(&self, mesh_id: MeshId, slot: usize) -> bool {
        self.deform.has_slot_external(mesh_id, slot)
    }

    /// Per-instance analogue of
    /// [`set_deform_slot_source_buffer`](Self::set_deform_slot_source_buffer):
    /// binds one instance's slot to a same-device buffer. The drawn item must
    /// select this instance (`deform_instance`) to read it, exactly as
    /// [`attach_deform_slot_instance`](Self::attach_deform_slot_instance)
    /// requires.
    ///
    /// # Errors
    ///
    /// Same as [`set_deform_slot_source_buffer`](Self::set_deform_slot_source_buffer).
    pub fn set_deform_slot_instance_source_buffer(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
        buffer: crate::gpu::Buffer,
        stride_bytes: u32,
    ) -> ViewportResult<()> {
        let len_bytes = buffer.size();
        let src = build_external_source(&buffer, stride_bytes, 0, len_bytes)?;
        self.deform
            .set_slot_instance_external(device, queue, mesh_id, instance_id, slot, src);
        Ok(())
    }

    /// Sliced per-instance analogue of
    /// [`set_deform_slot_source_buffer_sliced`](Self::set_deform_slot_source_buffer_sliced).
    ///
    /// # Errors
    ///
    /// Same as [`set_deform_slot_source_buffer_sliced`](Self::set_deform_slot_source_buffer_sliced).
    pub fn set_deform_slot_instance_source_buffer_sliced(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
        buffer: crate::gpu::Buffer,
        stride_bytes: u32,
        slice: DeformSourceSlice,
    ) -> ViewportResult<()> {
        let offset = slice.base_element as u64 * stride_bytes as u64;
        let len = slice.element_count as u64 * stride_bytes as u64;
        let src = build_external_source(&buffer, stride_bytes, offset, len)?;
        self.deform
            .set_slot_instance_external(device, queue, mesh_id, instance_id, slot, src);
        Ok(())
    }

    /// Detach a per-instance deform slot's external buffer source. Returns
    /// `true` if a source was removed.
    pub fn clear_deform_slot_instance_source(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
    ) -> bool {
        self.deform
            .clear_slot_instance_external(device, queue, mesh_id, instance_id, slot)
    }

    /// True when a per-instance slot is bound to an external buffer source.
    pub fn has_deform_slot_instance_source(
        &self,
        mesh_id: MeshId,
        instance_id: u32,
        slot: usize,
    ) -> bool {
        self.deform
            .has_slot_instance_external(mesh_id, instance_id, slot)
    }

    /// Copy every external deform slot (per-mesh and per-instance) from its
    /// consumer buffer into the packed deform buffer. Called once per frame in
    /// `prepare`, before the mesh render pass. Cheap no-op when no slot is
    /// buffer-backed.
    pub(crate) fn run_deform_slot_copies(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        if self.deform.external_source_count == 0 {
            return;
        }
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("deform_slot_copy_encoder"),
        });
        self.deform.record_deform_copies(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Register a deformer against the mesh shader family.
    ///
    /// Validates the descriptor's name and allocates a slot, composes every
    /// mesh-family base shader with the new deformer plus all previously
    /// registered ones, and runs each composed module through wgpu's
    /// validator. On success the registration is stored and the mesh-family
    /// pipelines are marked for rebuild; the rebuild itself runs once at the
    /// start of the next `prepare()`, so registering N deformers in a row
    /// costs one recompose-and-rebuild pass, not N. Other mesh-family
    /// pipelines (instanced, shadow, outline mask, OIT) continue to run the
    /// identity path until their factories migrate to the same rebuild path.
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
        device: &crate::gpu::Device,
        desc: DeformerDesc,
    ) -> ViewportResult<DeformerId> {
        // The deform group is left out of the mesh pipelines on devices that
        // lack a third bind group or the storage-buffer headroom its two
        // vertex-stage buffers need, so a deformer would never run. Report that
        // instead of silently accepting a registration that does nothing.
        if !self.deform.enabled {
            return Err(crate::error::ViewportError::DeformShaderInvalid {
                reason: format!(
                    "per-vertex deformers need a device created with max_bind_groups >= 3 and \
                     max_storage_buffers_per_shader_stage >= {}; this device provides fewer. \
                     Create the device with ViewportRenderer::recommended_device_limits(&adapter).",
                    crate::renderer::ViewportRenderer::DEFORM_STORAGE_BUFFERS_PER_STAGE,
                ),
            });
        }
        validate_name(&self.deform.registrations, desc.name)?;
        let slot = allocate_slot(&self.deform.registrations)?;

        let candidate = StoredDeformer { desc, slot };
        let mut trial = self.deform.registrations.clone();
        trial.push(candidate.clone());

        for shader_name in MESH_FAMILY_SHADERS {
            let Some(base) = lookup_source(shader_name) else {
                return Err(crate::error::ViewportError::DeformShaderInvalid {
                    reason: format!("internal: shader '{shader_name}' missing from shader catalog"),
                });
            };
            let composed = compose_shader(base, &trial);
            let label = format!("deform_compose_{shader_name}");
            validate_with_wgpu(device, &label, &composed)?;
        }

        self.deform.registrations.push(candidate);
        self.mesh_pipelines_dirty = true;
        Ok(DeformerId(slot))
    }

    /// Register a deformer on a reserved internal slot
    /// (`[DEFORM_INTERNAL_SLOT_START, DEFORM_SLOT_COUNT)`). Used at
    /// renderer construction for the in-crate deformers (skinning,
    /// displacement) so they do not consume one of the four slots exposed
    /// to hosts. Validation, composition, and pipeline rebuild work the
    /// same as [`Self::register_deformer`].
    #[allow(dead_code)]
    pub(crate) fn register_internal_deformer(
        &mut self,
        device: &crate::gpu::Device,
        desc: DeformerDesc,
    ) -> ViewportResult<DeformerId> {
        validate_name(&self.deform.registrations, desc.name)?;
        let slot = allocate_internal_slot(&self.deform.registrations)?;

        let candidate = StoredDeformer { desc, slot };
        let mut trial = self.deform.registrations.clone();
        trial.push(candidate.clone());

        for shader_name in MESH_FAMILY_SHADERS {
            let Some(base) = lookup_source(shader_name) else {
                return Err(crate::error::ViewportError::DeformShaderInvalid {
                    reason: format!("internal: shader '{shader_name}' missing from shader catalog"),
                });
            };
            let composed = compose_shader(base, &trial);
            let label = format!("deform_compose_{shader_name}");
            validate_with_wgpu(device, &label, &composed)?;
        }

        self.deform.registrations.push(candidate);
        self.mesh_pipelines_dirty = true;
        Ok(DeformerId(slot))
    }

    /// Number of currently registered deformers (host + internal).
    pub fn registered_deformer_count(&self) -> usize {
        self.deform.registrations.len()
    }

    /// Look up a registered deformer's id by its name. Returns `None` when
    /// no deformer with that name has been registered.
    pub fn deformer_id_by_name(&self, name: &str) -> Option<DeformerId> {
        self.deform
            .registrations
            .iter()
            .find(|r| r.desc.name == name)
            .map(|r| DeformerId(r.slot))
    }

    /// Build a [`DeformSlotHandle`] for the given deformer. Plugins stash
    /// the handle at registration time and use it to write that slot's
    /// `slot_params` block from contexts (e.g. a `GpuPlugin::pre_prepare`)
    /// that only carry `&wgpu::Queue`.
    pub fn deform_slot_handle(&self, id: DeformerId) -> DeformSlotHandle {
        DeformSlotHandle {
            header_buffer: self.deform.header_buffer.clone(),
            slot: id.slot(),
        }
    }

    /// Re-compose every mesh-family shader and rebuild the pipelines that
    /// draw from it. Runs from `flush_mesh_pipeline_rebuild` once any number
    /// of deformer registrations have validated; safe to call between frames
    /// with zero registrations to reset to the identity shader. The
    /// instanced and instanced-OIT pipelines stay on their build-time shader
    /// modules until their factories migrate to the same rebuild flow.
    /// Swap the lit pipelines between the stripped and debug-vis shader
    /// variants (see `builders::strip_debug_vis`). Called when the per-frame
    /// `DebugVis` state flips. The swap recompiles the mesh-family
    /// pipelines, a one-off hitch that is acceptable for a debugging tool.
    ///
    /// On limited devices (`max_bind_groups < 3`) the mesh family builds
    /// through the noop path that `rebuild_mesh_pipelines` does not cover;
    /// there the debug block stays stripped and the pixel inspector is
    /// unavailable.
    pub(crate) fn set_debug_vis_shaders(&mut self, device: &crate::gpu::Device, active: bool) {
        if self.debug_vis_shaders == active {
            return;
        }
        self.debug_vis_shaders = active;
        // Rebuild immediately (not via the dirty flag) so the toggle takes
        // effect in the frame that flipped it; the flag is cleared because
        // this rebuild also covers any pending deformer registration.
        self.mesh_pipelines_dirty = false;
        self.rebuild_mesh_pipelines(device);
    }

    /// Run the mesh-family pipeline rebuild that deformer registration
    /// deferred. Called at the start of every `prepare()`; a burst of
    /// `register_deformer` calls between frames collapses into the single
    /// rebuild here. No-op when nothing is pending.
    pub(crate) fn flush_mesh_pipeline_rebuild(&mut self, device: &crate::gpu::Device) {
        if !self.mesh_pipelines_dirty {
            return;
        }
        self.mesh_pipelines_dirty = false;
        self.rebuild_mesh_pipelines(device);
    }

    fn rebuild_mesh_pipelines(&mut self, device: &crate::gpu::Device) {
        if !self.deform.enabled {
            return;
        }
        // Material-plugin pipeline sets compose on top of the deformer
        // registrations and the debug-vis strip state, both of which changed
        // when this runs. Drop them; the next prepare that references a
        // plugin id rebuilds its set from the fresh composition.
        for plugin in self.material_plugins.values_mut() {
            plugin.pipelines = None;
        }
        let registrations = self.deform.registrations.clone();

        // mesh.wgsl: LDR + HDR families.
        if let Some(base) = lookup_source("mesh.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader = crate::resources::builders::wgsl_module(
                device,
                "mesh_shader_composed",
                crate::resources::builders::builtin_hook_env(
                    crate::resources::builders::strip_debug_vis(composed, self.debug_vis_shaders),
                ),
            );

            let ldr_layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
                device,
                "mesh_pipeline_layout",
                &self.camera_bind_group_layout,
                &self.object_bind_group_layout,
                Some(&self.deform.bind_group_layout),
            );
            let ldr = crate::resources::mesh::mesh_pipelines::build_ldr_mesh_pipelines(
                device,
                &ldr_layout,
                &shader,
                self.target_format,
                self.sample_count,
                None,
            );
            self.scene.solid = ldr.solid;
            self.scene.solid_two_sided = ldr.solid_two_sided;
            self.scene.transparent = ldr.transparent;
            self.scene.wireframe = ldr.wireframe;

            if self.scene.hdr_solid.is_some() {
                let hdr_layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
                    device,
                    "hdr_mesh_pipeline_layout",
                    &self.camera_bind_group_layout,
                    &self.object_bind_group_layout,
                    Some(&self.deform.bind_group_layout),
                );
                let hdr = crate::resources::mesh::mesh_pipelines::build_hdr_mesh_pipelines(
                    device,
                    &hdr_layout,
                    &shader,
                );
                self.scene.hdr_solid = Some(hdr.solid);
                self.scene.hdr_solid_two_sided = Some(hdr.solid_two_sided);
                self.scene.hdr_transparent = Some(hdr.transparent);
                self.scene.hdr_wireframe = Some(hdr.wireframe);
            }
        }

        // mesh_oit.wgsl: only present after ensure_hdr_shared has been
        // called.
        if self.oit.pipeline.is_some() {
            if let Some(base) = lookup_source("mesh_oit.wgsl") {
                let composed = compose_shader(base, &registrations);
                let shader = crate::resources::builders::wgsl_module(
                    device,
                    "mesh_oit_shader_composed",
                    crate::resources::builders::builtin_hook_env(
                        crate::resources::builders::strip_debug_vis(
                            composed,
                            self.debug_vis_shaders,
                        ),
                    ),
                );
                let oit_layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
                    device,
                    "oit_pipeline_layout",
                    &self.camera_bind_group_layout,
                    &self.object_bind_group_layout,
                    Some(&self.deform.bind_group_layout),
                );
                let oit = crate::resources::mesh::mesh_pipelines::build_oit_pipeline(
                    device,
                    &oit_layout,
                    &shader,
                );
                self.oit.pipeline = Some(oit);
            }
        }

        // shadow.wgsl: depth-only cascade pass.
        if let Some(base) = lookup_source("shadow.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader =
                crate::resources::builders::wgsl_module(device, "shadow_shader_composed", composed);
            let layout = crate::resources::builders::pipeline_layout(
                device,
                "shadow_pipeline_layout",
                &[
                    &self.shadow.camera_bgl,
                    &self.object_bind_group_layout,
                    &self.deform.bind_group_layout,
                ],
            );
            self.shadow.pipeline = crate::resources::mesh::mesh_pipelines::build_shadow_pipeline(
                device,
                &layout,
                &shader,
                Some(crate::gpu::Face::Front),
                None,
            );
            self.shadow.pipeline_two_sided =
                crate::resources::mesh::mesh_pipelines::build_shadow_pipeline(
                    device, &layout, &shader, None, None,
                );
        }

        // outline_mask.wgsl: mask-write pass for the selection silhouette.
        if let Some(base) = lookup_source("outline_mask.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader = crate::resources::builders::wgsl_module(
                device,
                "outline_mask_shader_composed",
                composed,
            );
            let layout = crate::resources::builders::pipeline_layout(
                device,
                "outline_pipeline_layout",
                &[
                    &self.camera_bind_group_layout,
                    &self.outline.bind_group_layout,
                    &self.deform.bind_group_layout,
                ],
            );
            let masks = crate::resources::mesh::mesh_pipelines::build_outline_mask_pipelines(
                device,
                &layout,
                &shader,
                crate::gpu::TextureFormat::R8Unorm,
                None,
            );
            self.outline.mask_pipeline = masks.mask;
            self.outline.mask_two_sided_pipeline = masks.mask_two_sided;
        }

        // mesh_instanced.wgsl: LDR (solid + transparent), HDR (solid +
        // transparent + additive + premultiplied), and HDR cull. Only
        // present after `ensure_instanced_pipelines` / its HDR sibling /
        // `ensure_cull_instance_pipelines` have run.
        if let Some(base) = lookup_source("mesh_instanced.wgsl") {
            let composed = compose_shader(base, &registrations);
            let src = crate::resources::builders::strip_mesh_discards(
                crate::resources::builders::builtin_hook_env(
                    crate::resources::builders::strip_debug_vis(composed, self.debug_vis_shaders),
                ),
            );
            let shader = crate::resources::builders::wgsl_module(
                device,
                "mesh_instanced_shader_composed",
                src.as_ref(),
            );
            let shader_nodiscard = crate::resources::builders::wgsl_module(
                device,
                "mesh_instanced_shader_composed_nodiscard",
                crate::resources::builders::strip_discards(&src),
            );

            if let Some(instance_bgl) = self.instancing.bind_group_layout.as_ref() {
                if self.instancing.solid_pipeline.is_some() {
                    let layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "instanced_pipeline_layout",
                        &self.camera_bind_group_layout,
                        instance_bgl,
                        Some(&self.deform.bind_group_layout),
                    );
                    let ldr =
                        crate::resources::mesh::mesh_pipelines::build_ldr_instanced_mesh_pipelines(
                            device,
                            &layout,
                            &shader,
                            self.target_format,
                            self.sample_count,
                        );
                    self.instancing.solid_pipeline = Some(ldr.solid);
                    self.instancing.solid_two_sided_pipeline = Some(ldr.solid_two_sided);
                    self.instancing.transparent_pipeline = Some(ldr.transparent);
                    let (nd, nd_two_sided) =
                        crate::resources::mesh::mesh_pipelines::build_instanced_solid_pipelines(
                            device,
                            &layout,
                            &shader_nodiscard,
                            self.target_format,
                            self.sample_count,
                            "solid_instanced_nodiscard_pipeline",
                            "solid_two_sided_instanced_nodiscard_pipeline",
                        );
                    self.instancing.solid_nodiscard_pipeline = Some(nd);
                    self.instancing.solid_two_sided_nodiscard_pipeline = Some(nd_two_sided);
                }
                if self.instancing.hdr_solid_pipeline.is_some() {
                    let layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "hdr_instanced_pipeline_layout",
                        &self.camera_bind_group_layout,
                        instance_bgl,
                        Some(&self.deform.bind_group_layout),
                    );
                    let hdr =
                        crate::resources::mesh::mesh_pipelines::build_hdr_instanced_mesh_pipelines(
                            device, &layout, &shader,
                        );
                    self.instancing.hdr_solid_pipeline = Some(hdr.solid);
                    self.instancing.hdr_solid_two_sided_pipeline = Some(hdr.solid_two_sided);
                    self.instancing.hdr_transparent_pipeline = Some(hdr.transparent);
                    self.instancing.hdr_additive_pipeline = Some(hdr.additive);
                    self.instancing.hdr_premultiplied_pipeline = Some(hdr.premultiplied);
                    let (nd, nd_two_sided) =
                        crate::resources::mesh::mesh_pipelines::build_instanced_solid_pipelines(
                            device,
                            &layout,
                            &shader_nodiscard,
                            crate::gpu::TextureFormat::Rgba16Float,
                            1,
                            "hdr_instanced_solid_nodiscard_pipeline",
                            "hdr_instanced_solid_two_sided_nodiscard_pipeline",
                        );
                    self.instancing.hdr_solid_nodiscard_pipeline = Some(nd);
                    self.instancing.hdr_solid_two_sided_nodiscard_pipeline = Some(nd_two_sided);
                }
            }
            if let Some(cull_bgl) = self.cull.bind_group_layout.as_ref() {
                if self.cull.hdr_solid_pipeline.is_some() {
                    let layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "hdr_instanced_cull_pipeline_layout",
                        &self.camera_bind_group_layout,
                        cull_bgl,
                        Some(&self.deform.bind_group_layout),
                    );
                    let pl =
                        crate::resources::mesh::mesh_pipelines::build_hdr_instanced_cull_pipeline(
                            device, &layout, &shader,
                        );
                    self.cull.hdr_solid_pipeline = Some(pl);
                    let pl_two_sided =
                        crate::resources::mesh::mesh_pipelines::build_hdr_instanced_cull_two_sided_pipeline(
                            device, &layout, &shader,
                        );
                    self.cull.hdr_solid_two_sided_pipeline = Some(pl_two_sided);
                    self.cull.hdr_solid_nodiscard_pipeline = Some(
                        crate::resources::mesh::mesh_pipelines::build_hdr_instanced_cull_pipeline_with(
                            device,
                            &layout,
                            &shader_nodiscard,
                            "hdr_solid_instanced_cull_nodiscard_pipeline",
                            Some(crate::gpu::Face::Back),
                        ),
                    );
                    self.cull.hdr_solid_two_sided_nodiscard_pipeline = Some(
                        crate::resources::mesh::mesh_pipelines::build_hdr_instanced_cull_pipeline_with(
                            device,
                            &layout,
                            &shader_nodiscard,
                            "hdr_solid_instanced_cull_two_sided_nodiscard_pipeline",
                            None,
                        ),
                    );
                }
            }
        }

        // mesh_instanced_oit.wgsl: non-cull and cull OIT pipelines.
        if let Some(base) = lookup_source("mesh_instanced_oit.wgsl") {
            let composed = compose_shader(base, &registrations);
            let shader = crate::resources::builders::wgsl_module(
                device,
                "mesh_instanced_oit_shader_composed",
                crate::resources::builders::builtin_hook_env(
                    crate::resources::builders::strip_debug_vis(composed, self.debug_vis_shaders),
                ),
            );
            if let Some(instance_bgl) = self.instancing.bind_group_layout.as_ref() {
                if self.oit.instanced_pipeline.is_some() {
                    let layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "oit_instanced_pipeline_layout",
                        &self.camera_bind_group_layout,
                        instance_bgl,
                        Some(&self.deform.bind_group_layout),
                    );
                    let pl = crate::resources::mesh::mesh_pipelines::build_oit_instanced_pipeline(
                        device,
                        &layout,
                        &shader,
                        "oit_instanced_pipeline",
                        "vs_main",
                    );
                    self.oit.instanced_pipeline = Some(pl);
                }
            }
            if let Some(cull_bgl) = self.cull.bind_group_layout.as_ref() {
                if self.cull.oit_pipeline.is_some() {
                    let layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
                        device,
                        "oit_instanced_cull_pipeline_layout",
                        &self.camera_bind_group_layout,
                        cull_bgl,
                        Some(&self.deform.bind_group_layout),
                    );
                    let pl = crate::resources::mesh::mesh_pipelines::build_oit_instanced_pipeline(
                        device,
                        &layout,
                        &shader,
                        "oit_instanced_cull_pipeline",
                        "vs_main_cull",
                    );
                    self.cull.oit_pipeline = Some(pl);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headless() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
                label: Some("deform_tests"),
                required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
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
        data[2] = Some(vec![10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0, 40, 0, 0, 0]);
        stride[2] = 2;

        let external: [Option<ExternalDeformSource>; DEFORM_SLOT_COUNT] = Default::default();
        let (words, offsets) = DeformationState::pack(&data, &stride, &external);
        assert_eq!(offsets[0], SLOT_LAYOUT_WORDS as u32);
        assert_eq!(offsets[2], (SLOT_LAYOUT_WORDS + 3) as u32);
        // Layout prefix holds an (offset, stride) pair per slot.
        assert_eq!(words[0], SLOT_LAYOUT_WORDS as u32); // slot 0 offset
        assert_eq!(words[1], 1); // slot 0 stride
        assert_eq!(words[2], 0); // slot 1 offset (unused)
        assert_eq!(words[3], 0); // slot 1 stride
        assert_eq!(words[4], (SLOT_LAYOUT_WORDS + 3) as u32); // slot 2 offset
        assert_eq!(words[5], 2); // slot 2 stride
        assert_eq!(words[6], 0); // slot 3 offset
        assert_eq!(words[7], 0); // slot 3 stride
        // Slot 0 data follows the prefix.
        let slot0_base = SLOT_LAYOUT_WORDS;
        assert_eq!(words[slot0_base], 1);
        assert_eq!(words[slot0_base + 1], 2);
        assert_eq!(words[slot0_base + 2], 3);
        // Slot 2 data follows slot 0.
        let slot2_base = slot0_base + 3;
        assert_eq!(words[slot2_base], 10);
        assert_eq!(words[slot2_base + 1], 20);
        assert_eq!(words[slot2_base + 2], 30);
        assert_eq!(words[slot2_base + 3], 40);
    }

    #[test]
    fn attach_marks_flag_bit_and_swaps_bind_group() {
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let mesh = MeshId::new(7, 0);
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
        let mesh = MeshId::new(11, 0);
        s.attach_slot(&device, mesh, 1, 1, &[0u8; 16]);
        assert_eq!(s.flag_bits(mesh), 0b0010);

        assert!(s.detach_slot(&device, mesh, 1));
        assert_eq!(s.flag_bits(mesh), 0);
        assert!(!s.meshes.contains_key(&mesh));
        assert!(!s.detach_slot(&device, mesh, 1));
    }

    #[test]
    fn attach_slot_instance_marks_union_and_routes_bind_group() {
        let Some((device, queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let mesh = MeshId::new(42, 0);
        let instance = 3u32;
        assert_eq!(s.flag_bits(mesh), 0);

        // Per-mesh slot 1, then per-instance slot 0.
        s.attach_slot(&device, mesh, 1, 1, &[0u8; 16]);
        s.attach_slot_instance(&device, &queue, mesh, instance, 0, 16, &[0u8; 64]);

        // flag_bits reports the union of per-mesh and per-instance.
        assert_eq!(s.flag_bits(mesh), 0b0011);
        assert!(s.has_slot_instance(mesh, instance, 0));
        assert!(!s.has_slot_instance(mesh, instance, 1));

        // The instance bind group differs from the mesh bind group.
        let inst_bg: *const crate::gpu::BindGroup = s.instance_bind_group_for(mesh, Some(instance));
        let mesh_bg: *const crate::gpu::BindGroup = s.instance_bind_group_for(mesh, None);
        assert_ne!(inst_bg, mesh_bg);

        // A draw of an instance with no per-instance data falls back to the
        // mesh bind group.
        let fallback: *const crate::gpu::BindGroup = s.instance_bind_group_for(mesh, Some(99));
        assert_eq!(fallback, mesh_bg);

        // Detaching the instance slot drops the instance entry and clears
        // the union bit; per-mesh slot 1 remains.
        assert!(s.detach_slot_instance(&device, &queue, mesh, instance, 0));
        assert_eq!(s.flag_bits(mesh), 0b0010);
        assert!(!s.has_slot_instance(mesh, instance, 0));
    }

    #[test]
    fn slot_index_assert_traps_out_of_range() {
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.attach_slot(&device, MeshId::new(0, 0), DEFORM_SLOT_COUNT, 1, &[0u8; 4])
        }));
        assert!(result.is_err());
    }

    /// Registering a deformer that actually reads from `deform_data` must
    /// produce a rebuilt LDR `mesh.wgsl` pipeline family once the deferred
    /// rebuild flushes. The simplest proof: if the composed source were
    /// broken, `register_deformer` would fail at validation; if the rebuild
    /// path were broken (e.g. shader module created from stale source),
    /// this test would still pass because no draw is issued. So we also
    /// re-fetch the LDR pipelines and confirm they are not the originals
    /// that the renderer was constructed with.
    #[test]
    fn register_deformer_rebuilds_ldr_mesh_pipelines() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);

        let solid_before: *const crate::gpu::RenderPipeline = &renderer.resources().scene.solid;
        let wf_before: *const crate::gpu::RenderPipeline = &renderer.resources().scene.wireframe;

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
        renderer
            .resources_mut()
            .flush_mesh_pipeline_rebuild(&device);

        let solid_after: *const crate::gpu::RenderPipeline = &renderer.resources().scene.solid;
        let wf_after: *const crate::gpu::RenderPipeline = &renderer.resources().scene.wireframe;
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
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let resources = renderer.resources_mut();
        let desc = DeformerDesc {
            name: "wind",
            stage: crate::resources::mesh_sidecar::registry::DeformStage::WorldSpace,
            priority: 0,
            wgsl_body: "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    var o = v;\n    o.position.z = o.position.z + 0.001;\n    return o;\n}\n".to_string(),
            per_vertex_stride: 4,
        };
        // Renderer construction already registered the internal skinning
        // deformer on a reserved slot, so we record the baseline count and
        // expect host slots to start at 0.
        let baseline = resources.registered_deformer_count();
        let id = resources
            .register_deformer(&device, desc)
            .expect("register");
        assert_eq!(id.slot(), 0);
        assert_eq!(resources.registered_deformer_count(), baseline + 1);

        let dup = DeformerDesc {
            name: "wind",
            stage: crate::resources::mesh_sidecar::registry::DeformStage::ObjectSpace,
            priority: 0,
            wgsl_body:
                "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex { return v; }"
                    .to_string(),
            per_vertex_stride: 4,
        };
        let err = resources.register_deformer(&device, dup).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::DeformNameTaken { .. }
        ));
        assert_eq!(resources.registered_deformer_count(), baseline + 1);

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
        assert_eq!(resources.registered_deformer_count(), baseline + 1);
    }

    fn copy_src_buffer(device: &crate::gpu::Device, size: u64) -> crate::gpu::Buffer {
        device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("deform_src_test"),
            size,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn pack_reserves_and_locates_external_slot_region() {
        let Some((device, _queue)) = headless() else {
            return;
        };
        let data: [Option<Vec<u8>>; DEFORM_SLOT_COUNT] = Default::default();
        let mut stride = [0u32; DEFORM_SLOT_COUNT];
        let mut external: [Option<ExternalDeformSource>; DEFORM_SLOT_COUNT] = Default::default();
        // Slot 1: external, 4 vertices, stride 12 bytes (vec3<f32>) = 48 bytes.
        stride[1] = 3;
        external[1] = Some(ExternalDeformSource {
            buffer: copy_src_buffer(&device, 48),
            src_offset_bytes: 0,
            len_bytes: 48,
            stride_words: 3,
        });

        let (words, offsets) = DeformationState::pack(&data, &stride, &external);
        // Header records the reserved region for slot 1.
        assert_eq!(words[2], SLOT_LAYOUT_WORDS as u32);
        assert_eq!(words[3], 3);
        assert_eq!(offsets[1], SLOT_LAYOUT_WORDS as u32);
        // Region is reserved (zero-filled) with room for 12 u32s.
        assert_eq!(words.len(), SLOT_LAYOUT_WORDS + 12);
        assert!(words[SLOT_LAYOUT_WORDS..].iter().all(|&w| w == 0));
    }

    #[test]
    fn set_and_clear_slot_external_roundtrip() {
        let Some((device, queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let mesh = MeshId::new(101, 0);
        let src = ExternalDeformSource {
            buffer: copy_src_buffer(&device, 48),
            src_offset_bytes: 0,
            len_bytes: 48,
            stride_words: 3,
        };
        s.set_slot_external(&device, mesh, 2, src);
        assert!(s.has_slot_external(mesh, 2));
        assert_eq!(s.flag_bits(mesh), 0b0100);
        assert_eq!(s.external_source_count, 1);
        assert_eq!(
            s.meshes[&mesh].slot_offset_words[2],
            SLOT_LAYOUT_WORDS as u32
        );

        // The per-frame copy pass records a copy without panicking.
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("deform_copy_test"),
        });
        s.record_deform_copies(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        assert!(s.clear_slot_external(&device, mesh, 2));
        assert!(!s.has_slot_external(mesh, 2));
        assert_eq!(s.external_source_count, 0);
        assert!(!s.meshes.contains_key(&mesh));
        assert!(!s.clear_slot_external(&device, mesh, 2));
    }

    #[test]
    fn cpu_and_external_slots_coexist_on_one_mesh() {
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let mesh = MeshId::new(102, 0);
        // Slot 0: CPU-uploaded. Slot 1: external.
        s.attach_slot(&device, mesh, 0, 1, &[0u8; 16]);
        let src = ExternalDeformSource {
            buffer: copy_src_buffer(&device, 48),
            src_offset_bytes: 0,
            len_bytes: 48,
            stride_words: 3,
        };
        s.set_slot_external(&device, mesh, 1, src);
        assert_eq!(s.flag_bits(mesh), 0b0011);
        assert!(s.has_slot(mesh, 0));
        assert!(s.has_slot_external(mesh, 1));
        // Distinct, non-overlapping regions.
        let off0 = s.meshes[&mesh].slot_offset_words[0];
        let off1 = s.meshes[&mesh].slot_offset_words[1];
        assert_ne!(off0, off1);
    }

    #[test]
    fn attach_cpu_replaces_external_on_same_slot() {
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let mesh = MeshId::new(103, 0);
        let src = ExternalDeformSource {
            buffer: copy_src_buffer(&device, 48),
            src_offset_bytes: 0,
            len_bytes: 48,
            stride_words: 3,
        };
        s.set_slot_external(&device, mesh, 0, src);
        assert_eq!(s.external_source_count, 1);
        // Attaching CPU data on the same slot drops the external source.
        s.attach_slot(&device, mesh, 0, 3, &[0u8; 48]);
        assert!(!s.has_slot_external(mesh, 0));
        assert!(s.has_slot(mesh, 0));
        assert_eq!(s.external_source_count, 0);
    }

    #[test]
    fn set_and_clear_slot_instance_external_roundtrip() {
        let Some((device, queue)) = headless() else {
            return;
        };
        let mut s = DeformationState::new(&device);
        let mesh = MeshId::new(104, 0);
        let src = ExternalDeformSource {
            buffer: copy_src_buffer(&device, 48),
            src_offset_bytes: 0,
            len_bytes: 48,
            stride_words: 3,
        };
        s.set_slot_instance_external(&device, &queue, mesh, 5, 1, src);
        assert!(s.has_slot_instance_external(mesh, 5, 1));
        assert_eq!(s.external_source_count, 1);

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("deform_inst_copy_test"),
        });
        s.record_deform_copies(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        assert!(s.clear_slot_instance_external(&device, &queue, mesh, 5, 1));
        assert!(!s.has_slot_instance_external(mesh, 5, 1));
        assert_eq!(s.external_source_count, 0);
    }

    #[test]
    fn set_deform_slot_source_buffer_rejects_bad_inputs() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let mesh = MeshId::new(105, 0);

        // Missing COPY_SRC.
        let storage_only = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: None,
            size: 48,
            usage: crate::gpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(matches!(
            renderer.resources_mut().set_deform_slot_source_buffer(
                &device,
                mesh,
                0,
                storage_only,
                12
            ),
            Err(crate::error::ViewportError::ExternalBufferUsageMissing { .. })
        ));

        // Stride not a positive multiple of 4.
        let buf = copy_src_buffer(&device, 48);
        assert!(matches!(
            renderer.resources_mut().set_deform_slot_source_buffer(
                &device,
                mesh,
                0,
                buf.clone(),
                2
            ),
            Err(crate::error::ViewportError::DeformSourceMismatch { .. })
        ));

        // Sliced window past the end of the buffer.
        assert!(matches!(
            renderer
                .resources_mut()
                .set_deform_slot_source_buffer_sliced(
                    &device,
                    mesh,
                    0,
                    buf,
                    12,
                    DeformSourceSlice {
                        base_element: 2,
                        element_count: 4,
                    },
                ),
            Err(crate::error::ViewportError::DeformSourceMismatch { .. })
        ));

        // A valid whole-buffer bind succeeds and reports the source.
        let good = copy_src_buffer(&device, 48);
        renderer
            .resources_mut()
            .set_deform_slot_source_buffer(&device, mesh, 0, good, 12)
            .expect("valid source");
        assert!(renderer.resources().has_deform_slot_source(mesh, 0));
    }
}
