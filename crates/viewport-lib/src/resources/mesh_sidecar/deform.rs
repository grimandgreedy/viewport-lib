//! Per-vertex deformation sidecar storage.
//!
//! A mesh's deform slot data lives in its sidecar record, as streams beside
//! its attributes, so a draw reaches it through the record chunk it already
//! binds. Each mesh with data also carries a deform table (and each instance
//! with data its own), which locates the slots; a draw's instance record names
//! the table it reads. The shader-side reads go through the `deform_read_*`
//! helpers in `deform.wgsl`.

use std::collections::HashMap;

use crate::gpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use crate::error::ViewportResult;
use crate::resources::DeviceResources;
use crate::resources::mesh::mesh_store::MeshId;
use crate::resources::mesh::sidecar_pool::{SidecarRecord, StreamData, StreamKind};
use crate::resources::mesh_sidecar::registry::{
    DeformerDesc, DeformerId, MESH_FAMILY_SHADERS, StoredDeformer, allocate_internal_slot,
    allocate_slot, compose_shader, defines_keep, lookup_source, validate_name, validate_with_wgpu,
};

/// Maximum number of registered deformer slots (host range + reserved
/// internal range). Slot indices `[0, DEFORM_INTERNAL_SLOT_START)` are
/// available to hosts through `register_deformer`; indices
/// `[DEFORM_INTERNAL_SLOT_START, DEFORM_SLOT_COUNT)` are reserved for the
/// in-crate deformers (skinning, displacement) that register at renderer
/// construction. Slot data lives in the mesh's sidecar record, so deform adds
/// no storage binding however many slots are in use.
pub const DEFORM_SLOT_COUNT: usize = 8;

/// First slot index reserved for in-crate deformers.
pub const DEFORM_INTERNAL_SLOT_START: usize = 4;

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
/// renderer copies `len_bytes` from `src_offset_bytes` into the slot's stream
/// each frame; it never writes the consumer's buffer.
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

/// One side of a mesh's deform data: the per-mesh slots, or one instance's.
#[derive(Default)]
pub(crate) struct DeformSlots {
    /// Source bytes per slot, retained so a record move re-uploads from
    /// current bytes.
    pub slot_data: [Option<Vec<u8>>; DEFORM_SLOT_COUNT],
    /// Per-slot stride in u32 words, meaningful while the slot holds data, an
    /// external source or an alias.
    pub slot_stride: [u32; DEFORM_SLOT_COUNT],
    /// External same-device source per slot. Mutually exclusive with
    /// `slot_data[i]`: a slot is either CPU-uploaded or buffer-backed.
    pub slot_external: [Option<ExternalDeformSource>; DEFORM_SLOT_COUNT],
    /// Bit `i` set when slot `i` has data on this side.
    pub flag_bits: u32,
}

impl DeformSlots {
    /// Words of slot `slot`'s stream in the record, `None` when the slot reads
    /// no stream of its own.
    fn stream_words(&self, slot: usize) -> Option<Vec<u32>> {
        if let Some(src) = &self.slot_external[slot] {
            return Some(vec![0; (src.len_bytes / 4) as usize]);
        }
        self.slot_data[slot]
            .as_ref()
            .map(|bytes| bytemuck::pod_collect_to_vec(bytes))
    }

    /// Clear slot `slot`, returning whether it held anything and whether that
    /// was an external source.
    fn clear(&mut self, slot: usize) -> (bool, bool) {
        let had_data = self.slot_data[slot].take().is_some();
        let had_external = self.slot_external[slot].take().is_some();
        self.slot_stride[slot] = 0;
        self.flag_bits &= !(1u32 << slot);
        (had_data || had_external, had_external)
    }
}

/// Per-mesh deformation state. The slot data itself lives in the mesh's
/// sidecar record (see [`DeviceResources::sync_deform_record`]); this is the
/// host copy and the bookkeeping the record is built from.
#[derive(Default)]
pub(crate) struct MeshDeform {
    pub mesh: DeformSlots,
    /// A slot fed from one of the mesh's own scalar attributes, by name. Its
    /// table entry points at that attribute's stream, so it holds no stream of
    /// its own and follows the attribute through every record move.
    pub slot_alias: [Option<String>; DEFORM_SLOT_COUNT],
    /// Bit `i` set when *any* instance of this mesh has slot `i` per-instance
    /// data attached. Combined with the per-mesh bits at
    /// `InstanceData::deform_flags` write time so the shader sees both
    /// per-mesh and per-instance slot activity.
    pub instance_flag_bits_union: u32,
    pub instances: HashMap<u32, DeformSlots>,
}

impl MeshDeform {
    fn recompute_instance_union(&mut self) {
        self.instance_flag_bits_union = self.instances.values().fold(0, |u, i| u | i.flag_bits);
    }

    fn is_empty(&self) -> bool {
        self.mesh.flag_bits == 0 && self.instance_flag_bits_union == 0
    }

    /// The deform streams the mesh's record should hold, with their words:
    /// the per-mesh slots, each instance's slots, the mesh table and one
    /// table per instance with data. Instances in id order, so the layout
    /// does not depend on map order.
    fn streams(&self) -> Vec<(StreamKind, Vec<u32>)> {
        let mut out = Vec::new();
        for slot in 0..DEFORM_SLOT_COUNT {
            if let Some(words) = self.mesh.stream_words(slot) {
                out.push((StreamKind::DeformSlot(slot as u8), words));
            }
        }
        let mut ids: Vec<u32> = self.instances.keys().copied().collect();
        ids.sort_unstable();
        for &id in &ids {
            let inst = &self.instances[&id];
            for slot in 0..DEFORM_SLOT_COUNT {
                if let Some(words) = inst.stream_words(slot) {
                    out.push((StreamKind::DeformInstanceSlot(slot as u8, id), words));
                }
            }
        }
        out.push((StreamKind::DeformTable(None), vec![0; DEFORM_TABLE_WORDS]));
        for &id in &ids {
            if self.instances[&id].flag_bits != 0 {
                out.push((
                    StreamKind::DeformTable(Some(id)),
                    vec![0; DEFORM_TABLE_WORDS],
                ));
            }
        }
        out
    }

    /// The table for the per-mesh slots and, with `instance`, that instance's
    /// slots, as absolute words of the record's chunk.
    fn table(&self, record: &SidecarRecord, instance: Option<u32>) -> Vec<u32> {
        let mut words = vec![0u32; DEFORM_TABLE_WORDS];
        for slot in 0..DEFORM_SLOT_COUNT {
            let own = record.first_of(StreamKind::DeformSlot(slot as u8));
            let alias = self.slot_alias[slot]
                .as_deref()
                .and_then(|name| record.find(StreamKind::Scalar, name));
            if let Some(s) = own.or(alias) {
                words[2 * slot] = record.stream_word(s);
                words[2 * slot + 1] = self.mesh.slot_stride[slot];
            }
        }
        if let Some(id) = instance {
            let inst = &self.instances[&id];
            for slot in 0..DEFORM_SLOT_COUNT {
                if let Some(s) = record.first_of(StreamKind::DeformInstanceSlot(slot as u8, id)) {
                    words[DEFORM_TABLE_WORDS / 2 + 2 * slot] = record.stream_word(s);
                    words[DEFORM_TABLE_WORDS / 2 + 2 * slot + 1] = inst.slot_stride[slot];
                }
            }
        }
        words
    }
}

/// Words in a deform table: an `(offset, stride)` pair per slot for the
/// per-mesh data, then one per slot for the per-instance data. Mirrors
/// `DEFORM_INSTANCE_PAIRS` in `deform.wgsl`.
pub(crate) const DEFORM_TABLE_WORDS: usize = 4 * DEFORM_SLOT_COUNT;

/// Renderer-side deformation state.
pub(crate) struct DeformationState {
    /// Bumped whenever a mesh's or an instance's deform slot data is attached
    /// or removed, so the instanced batch cache repacks the records whose
    /// `deform_flags` changed.
    pub(crate) epoch: u64,
    /// The header uniform, bound at group 0 binding 23 of every pass that
    /// runs the deform hooks.
    pub header_buffer: crate::gpu::Buffer,
    pub meshes: HashMap<MeshId, MeshDeform>,
    pub header_cpu: DeformHeader,
    /// Currently registered deformers, in registration order.
    pub registrations: Vec<StoredDeformer>,
    /// Bit per slot whose registered body defines the optional `keep` hook.
    /// Kept beside the registrations so the per-item early-Z check does not
    /// re-scan WGSL every frame.
    pub keep_slots: u32,
    /// Number of slots (per-mesh and per-instance, across all meshes) currently
    /// bound to an external buffer source. Lets the per-frame copy pass skip its
    /// encoder entirely when nothing is buffer-backed.
    pub external_source_count: usize,
    /// Slot-write costs accumulated since the last `take_counters`. Read and
    /// reset once per frame by `prepare`.
    pub counters: DeformCounters,
}

/// What deform-slot writes did to their GPU storage since the counters were
/// last taken.
///
/// Slot data lives in the mesh's sidecar record. A write that fits the slot's
/// existing stream lands in place and touches none of these. One that changes
/// the record's shape (an attach, a detach, a new source) moves the record,
/// copying its other streams, and counts here.
///
/// Surfaced per frame as the `deform_*` fields on
/// [`FrameStats`](crate::renderer::stats::FrameStats).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeformCounters {
    /// Sidecar records moved or created by deform-slot changes.
    pub buffer_reallocations: u32,
}

impl DeformationState {
    pub fn new(device: &crate::gpu::Device) -> Self {
        let header_cpu = DeformHeader::zeroed();
        let header_buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("deform_header"),
            contents: bytemuck::bytes_of(&header_cpu),
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
        });
        Self {
            epoch: 0,
            header_buffer,
            meshes: HashMap::new(),
            header_cpu,
            registrations: Vec::new(),
            keep_slots: 0,
            external_source_count: 0,
            counters: DeformCounters::default(),
        }
    }

    /// Take the accumulated slot-write counters, resetting them. Called once
    /// per frame by `prepare` so each frame's figures cover that frame alone.
    pub fn take_counters(&mut self) -> DeformCounters {
        std::mem::take(&mut self.counters)
    }

    /// `deform_flags` value to write into the instance record for `mesh_id`.
    /// Folds per-mesh and per-instance slot activity together: any slot
    /// with data on either side is reported as live so the registered
    /// body's flag branch executes.
    pub fn flag_bits(&self, mesh_id: MeshId) -> u32 {
        self.meshes
            .get(&mesh_id)
            .map(|m| m.mesh.flag_bits | m.instance_flag_bits_union)
            .unwrap_or(0)
    }

    /// The deform table a draw of `(mesh_id, instance_id)` reads, as a word of
    /// the mesh's record chunk: the instance's table when it has data, the
    /// mesh's otherwise, 0 when the mesh has no deform data.
    pub(crate) fn table_word(
        &self,
        mesh_id: MeshId,
        instance_id: Option<u32>,
        record: Option<&SidecarRecord>,
    ) -> u32 {
        let (Some(m), Some(record)) = (self.meshes.get(&mesh_id), record) else {
            return 0;
        };
        let instance =
            instance_id.filter(|id| m.instances.get(id).is_some_and(|i| i.flag_bits != 0));
        record
            .first_of(StreamKind::DeformTable(instance))
            .map_or(0, |s| record.stream_word(s))
    }

    /// Whether a draw of `(mesh_id, instance_id)` runs a deformer that defines
    /// `keep`, and so may discard. Such a draw cannot take the discard-free
    /// early-Z pipeline twin, which has every `discard` stripped.
    pub(crate) fn may_discard(&self, mesh_id: MeshId, instance_id: Option<u32>) -> bool {
        if self.keep_slots == 0 {
            return false;
        }
        let Some(m) = self.meshes.get(&mesh_id) else {
            return false;
        };
        let instance_bits = instance_id
            .and_then(|id| m.instances.get(&id))
            .map_or(0, |i| i.flag_bits);
        (m.mesh.flag_bits | instance_bits) & self.keep_slots != 0
    }

    /// The per-mesh slots of `mesh_id` or one instance's, created on demand.
    fn slots_mut(&mut self, mesh_id: MeshId, instance_id: Option<u32>) -> &mut DeformSlots {
        let m = self.meshes.entry(mesh_id).or_default();
        match instance_id {
            Some(id) => m.instances.entry(id).or_default(),
            None => &mut m.mesh,
        }
    }

    /// Store CPU data for a slot, replacing any source on it.
    fn set_data(
        &mut self,
        mesh_id: MeshId,
        instance_id: Option<u32>,
        slot: usize,
        stride_words: u32,
        data: &[u8],
    ) {
        assert!(slot < DEFORM_SLOT_COUNT);
        assert!(
            data.len() % 4 == 0,
            "deform slot data length must be a multiple of 4 bytes"
        );
        if instance_id.is_none() {
            self.drop_alias(mesh_id, slot);
        }
        let slots = self.slots_mut(mesh_id, instance_id);
        let had_external = slots.slot_external[slot].take().is_some();
        slots.slot_data[slot] = Some(data.to_vec());
        slots.slot_stride[slot] = stride_words;
        slots.flag_bits |= 1u32 << slot;
        if had_external {
            self.external_source_count -= 1;
        }
        self.after_change(mesh_id);
    }

    /// Bind a slot to an external buffer window, replacing any data on it.
    fn set_external(
        &mut self,
        mesh_id: MeshId,
        instance_id: Option<u32>,
        slot: usize,
        src: ExternalDeformSource,
    ) {
        assert!(slot < DEFORM_SLOT_COUNT);
        if instance_id.is_none() {
            self.drop_alias(mesh_id, slot);
        }
        let slots = self.slots_mut(mesh_id, instance_id);
        let had_external = slots.slot_external[slot].is_some();
        slots.slot_data[slot] = None;
        slots.slot_stride[slot] = src.stride_words;
        slots.slot_external[slot] = Some(src);
        slots.flag_bits |= 1u32 << slot;
        if !had_external {
            self.external_source_count += 1;
        }
        self.after_change(mesh_id);
    }

    /// Point a per-mesh slot at one of the mesh's scalar attributes.
    fn set_alias(&mut self, mesh_id: MeshId, slot: usize, name: &str) {
        assert!(slot < DEFORM_SLOT_COUNT);
        let (_, had_external) = self.slots_mut(mesh_id, None).clear(slot);
        if had_external {
            self.external_source_count -= 1;
        }
        let m = self.meshes.get_mut(&mesh_id).expect("created above");
        m.slot_alias[slot] = Some(name.to_string());
        m.mesh.slot_stride[slot] = 1;
        m.mesh.flag_bits |= 1u32 << slot;
        self.after_change(mesh_id);
    }

    fn drop_alias(&mut self, mesh_id: MeshId, slot: usize) -> bool {
        self.meshes
            .get_mut(&mesh_id)
            .is_some_and(|m| m.slot_alias[slot].take().is_some())
    }

    /// Clear a slot. `external` selects what may be cleared: a source (an
    /// external buffer or an alias) or CPU data. Returns whether anything was
    /// removed.
    fn clear_slot(
        &mut self,
        mesh_id: MeshId,
        instance_id: Option<u32>,
        slot: usize,
        external: bool,
    ) -> bool {
        assert!(slot < DEFORM_SLOT_COUNT);
        let Some(m) = self.meshes.get_mut(&mesh_id) else {
            return false;
        };
        let slots = match instance_id {
            Some(id) => match m.instances.get_mut(&id) {
                Some(i) => i,
                None => return false,
            },
            None => &mut m.mesh,
        };
        let holds = if external {
            slots.slot_external[slot].is_some()
                || (instance_id.is_none() && m.slot_alias[slot].is_some())
        } else {
            slots.slot_data[slot].is_some()
        };
        if !holds {
            return false;
        }
        let (_, had_external) = slots.clear(slot);
        if instance_id.is_none() {
            m.slot_alias[slot] = None;
        }
        if let Some(id) = instance_id {
            if m.instances[&id].flag_bits == 0 {
                m.instances.remove(&id);
            }
        }
        if had_external {
            self.external_source_count -= 1;
        }
        self.after_change(mesh_id);
        true
    }

    fn after_change(&mut self, mesh_id: MeshId) {
        self.epoch = self.epoch.wrapping_add(1);
        if let Some(m) = self.meshes.get_mut(&mesh_id) {
            m.recompute_instance_union();
            if m.is_empty() {
                self.meshes.remove(&mesh_id);
            }
        }
    }

    pub fn has_slot(&self, mesh_id: MeshId, slot: usize) -> bool {
        self.meshes
            .get(&mesh_id)
            .is_some_and(|m| m.mesh.slot_data[slot].is_some())
    }

    pub fn has_slot_instance(&self, mesh_id: MeshId, instance_id: u32, slot: usize) -> bool {
        self.meshes
            .get(&mesh_id)
            .and_then(|m| m.instances.get(&instance_id))
            .is_some_and(|i| i.slot_data[slot].is_some())
    }

    /// True when a per-mesh slot is fed from an external buffer or one of the
    /// mesh's attributes.
    pub fn has_slot_external(&self, mesh_id: MeshId, slot: usize) -> bool {
        self.meshes
            .get(&mesh_id)
            .is_some_and(|m| m.mesh.slot_external[slot].is_some() || m.slot_alias[slot].is_some())
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
            .is_some_and(|i| i.slot_external[slot].is_some())
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
            .set_data(mesh_id, None, slot, stride_bytes / 4, data);
        self.sync_deform_record(device, mesh_id, &[StreamKind::DeformSlot(slot as u8)]);
    }

    /// Overwrite part of a mesh's deform-slot data without re-packing it.
    ///
    /// The call a deformer that updates every frame should be making.
    /// [`attach_deform_slot`](Self::attach_deform_slot) is structural: it
    /// moves the mesh's sidecar record, copying every other stream on it,
    /// because a slot's length may have changed. A per-frame update never
    /// changes the shape, so it needs none of that.
    ///
    /// `first_element` is in slot elements, not bytes: element `n` starts at
    /// `n * stride_bytes` within the slot, using the stride the attach
    /// established. Pass `0` and the whole slot's worth of bytes to replace it
    /// entirely, which is still far cheaper than re-attaching, or a window to
    /// write only the region that moved.
    ///
    /// Attach the slot once to establish its length and stride, then write:
    ///
    /// ```no_run
    /// # use viewport_lib::resources::DeviceResources;
    /// # fn f(resources: &mut DeviceResources, device: &viewport_lib::gpu::Device,
    /// #      queue: &viewport_lib::gpu::Queue, mesh: viewport_lib::MeshId,
    /// #      offsets: &[[f32; 3]]) -> viewport_lib::error::ViewportResult<()> {
    /// resources.attach_deform_slot(device, mesh, 0, 12, bytemuck::cast_slice(offsets));
    /// // Each frame after that, for the vertices that moved:
    /// resources.write_deform_slot_range(queue, mesh, 0, 0, bytemuck::cast_slice(offsets))?;
    /// # Ok(()) }
    /// ```
    ///
    /// Errors with `DeformSlotNotAttached` when the slot holds no CPU data (a
    /// slot fed by [`set_deform_slot_source_buffer`](Self::set_deform_slot_source_buffer)
    /// is filled by the per-frame copy pass and is not writable this way), and
    /// with `DeformSlotWriteOutOfRange` when the window does not fit what is
    /// attached or `data` is not a whole number of elements.
    pub fn write_deform_slot_range(
        &mut self,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        slot: usize,
        first_element: u32,
        data: &[u8],
    ) -> crate::error::ViewportResult<()> {
        assert!(slot < DEFORM_SLOT_COUNT);
        let not_attached = || crate::error::ViewportError::DeformSlotNotAttached {
            mesh_id: mesh_id.index(),
            slot,
        };
        let entry = self
            .deform
            .meshes
            .get_mut(&mesh_id)
            .ok_or_else(not_attached)?;
        if entry.mesh.slot_external[slot].is_some() {
            return Err(not_attached());
        }
        let stride_bytes = entry.mesh.slot_stride[slot] as usize * 4;
        let bytes = entry.mesh.slot_data[slot]
            .as_mut()
            .ok_or_else(not_attached)?;
        if stride_bytes == 0 {
            return Err(not_attached());
        }
        if data.is_empty() {
            return Ok(());
        }
        if data.len() % stride_bytes != 0 {
            return Err(crate::error::ViewportError::DeformSlotWriteOutOfRange {
                slot,
                first_element,
                element_count: 0,
                slot_elements: (bytes.len() / stride_bytes) as u32,
            });
        }
        let element_count = (data.len() / stride_bytes) as u32;
        let slot_elements = (bytes.len() / stride_bytes) as u32;
        if first_element.saturating_add(element_count) > slot_elements {
            return Err(crate::error::ViewportError::DeformSlotWriteOutOfRange {
                slot,
                first_element,
                element_count,
                slot_elements,
            });
        }
        // The host copy moves with the GPU stream, so a later record move
        // re-uploads current bytes rather than resurrecting stale ones.
        let local = first_element as usize * stride_bytes;
        bytes[local..local + data.len()].copy_from_slice(data);
        if let Some(record) = self
            .mesh_store
            .get(mesh_id)
            .and_then(|m| m.sidecar.as_ref())
        {
            if let Some(stream) = record.first_of(StreamKind::DeformSlot(slot as u8)) {
                let at = record.byte_range(stream).start + local as u64;
                self.sidecar
                    .write_bytes_with(queue, record.chunk(), at, data);
            }
        }
        Ok(())
    }

    /// The bytes currently retained for one per-mesh slot, or `None` when the
    /// slot holds no CPU data. Crate-internal: this is the host mirror of what
    /// the GPU buffer holds, and exists so a test can check the two agree after
    /// a ranged write.
    #[cfg(test)]
    pub(crate) fn deform_slot_bytes(&self, mesh_id: MeshId, slot: usize) -> Option<Vec<u8>> {
        self.deform
            .meshes
            .get(&mesh_id)
            .and_then(|m| m.mesh.slot_data[slot].clone())
    }

    /// Detach a per-mesh slot's data. Returns `true` if any data was removed.
    pub fn detach_deform_slot(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
    ) -> bool {
        let removed = self.deform.clear_slot(mesh_id, None, slot, false);
        if removed {
            self.sync_deform_record(device, mesh_id, &[]);
        }
        removed
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
        let kind = StreamKind::DeformInstanceSlot(slot as u8, instance_id);
        // Same length and stride as what is there: a per-frame update (a joint
        // palette, say), written in place.
        let same_shape = self
            .deform
            .meshes
            .get(&mesh_id)
            .and_then(|m| m.instances.get(&instance_id))
            .is_some_and(|i| {
                i.slot_stride[slot] == stride_bytes / 4
                    && i.slot_data[slot]
                        .as_ref()
                        .is_some_and(|b| b.len() == data.len())
            });
        let in_place = same_shape.then(|| {
            self.mesh_store
                .get(mesh_id)
                .and_then(|m| m.sidecar.as_ref())
                .and_then(|r| r.first_of(kind).map(|s| (r.chunk(), r.byte_range(s).start)))
        });
        if let Some(Some((chunk, at))) = in_place {
            let inst = self
                .deform
                .meshes
                .get_mut(&mesh_id)
                .and_then(|m| m.instances.get_mut(&instance_id))
                .expect("checked above");
            inst.slot_data[slot] = Some(data.to_vec());
            self.sidecar.write_bytes_with(queue, chunk, at, data);
            return;
        }
        self.deform
            .set_data(mesh_id, Some(instance_id), slot, stride_bytes / 4, data);
        self.sync_deform_record(device, mesh_id, &[kind]);
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
        let _ = queue;
        let removed = self
            .deform
            .clear_slot(mesh_id, Some(instance_id), slot, false);
        if removed {
            self.sync_deform_record(device, mesh_id, &[]);
        }
        removed
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
        self.deform.set_external(mesh_id, None, slot, src);
        self.sync_deform_record(device, mesh_id, &[StreamKind::DeformSlot(slot as u8)]);
        Ok(())
    }

    /// Feed a deformer slot's per-mesh data from one of the mesh's own per-vertex
    /// scalar attributes, one `f32` per vertex (stride 4), by name.
    ///
    /// The slot reads the attribute's own storage, so a `replace_attribute` on
    /// the mesh reaches the deformer with no further call, and nothing is
    /// copied. Use it for a deformer that reads a field the mesh already
    /// carries, rather than uploading the same values again. Returns
    /// [`ViewportError::AttributeNotFound`] when the mesh has no scalar attribute
    /// of that name.
    ///
    /// [`ViewportError::AttributeNotFound`]: crate::error::ViewportError::AttributeNotFound
    pub fn set_deform_slot_source_attribute(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        slot: usize,
        name: &str,
    ) -> ViewportResult<()> {
        // The attribute is a stream of the mesh's own record; the slot's table
        // entry points at it.
        let found = self
            .mesh_store
            .get(mesh_id)
            .and_then(|mesh| mesh.sidecar.as_ref())
            .is_some_and(|record| record.find(StreamKind::Scalar, name).is_some());
        if !found {
            return Err(crate::error::ViewportError::AttributeNotFound {
                mesh_id: mesh_id.index(),
                name: name.to_string(),
            });
        }
        self.deform.set_alias(mesh_id, slot, name);
        self.sync_deform_record(device, mesh_id, &[]);
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
        self.deform.set_external(mesh_id, None, slot, src);
        self.sync_deform_record(device, mesh_id, &[StreamKind::DeformSlot(slot as u8)]);
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
        let removed = self.deform.clear_slot(mesh_id, None, slot, true);
        if removed {
            self.sync_deform_record(device, mesh_id, &[]);
        }
        removed
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
        let _ = queue;
        let src = build_external_source(&buffer, stride_bytes, 0, len_bytes)?;
        self.deform
            .set_external(mesh_id, Some(instance_id), slot, src);
        let kind = StreamKind::DeformInstanceSlot(slot as u8, instance_id);
        self.sync_deform_record(device, mesh_id, &[kind]);
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
        let _ = queue;
        let src = build_external_source(&buffer, stride_bytes, offset, len)?;
        self.deform
            .set_external(mesh_id, Some(instance_id), slot, src);
        let kind = StreamKind::DeformInstanceSlot(slot as u8, instance_id);
        self.sync_deform_record(device, mesh_id, &[kind]);
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
        let _ = queue;
        let removed = self
            .deform
            .clear_slot(mesh_id, Some(instance_id), slot, true);
        if removed {
            self.sync_deform_record(device, mesh_id, &[]);
        }
        removed
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
    /// consumer buffer into its stream in the mesh's sidecar record. Called
    /// once per frame in `prepare`, before the mesh render pass and after the
    /// pool's queued writes. Cheap no-op when no slot is buffer-backed.
    ///
    /// The copies go into `sink` like every other prepare pass, so under
    /// `prepare_deferred` they are submitted with the frame on the driving
    /// thread, after anything the caller submits ahead of the batch.
    pub(crate) fn run_deform_slot_copies(
        &self,
        device: &crate::gpu::Device,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        if self.deform.external_source_count == 0 {
            return;
        }
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("deform_slot_copy_encoder"),
        });
        for (mesh_id, m) in &self.deform.meshes {
            let Some(record) = self
                .mesh_store
                .get(*mesh_id)
                .and_then(|g| g.sidecar.as_ref())
            else {
                continue;
            };
            let chunk = self.sidecar.chunk_buffer(record.chunk());
            let sides = std::iter::once((None, &m.mesh))
                .chain(m.instances.iter().map(|(id, i)| (Some(*id), i)));
            for (instance, slots) in sides {
                for slot in 0..DEFORM_SLOT_COUNT {
                    let Some(src) = &slots.slot_external[slot] else {
                        continue;
                    };
                    let kind = match instance {
                        Some(id) => StreamKind::DeformInstanceSlot(slot as u8, id),
                        None => StreamKind::DeformSlot(slot as u8),
                    };
                    if let Some(stream) = record.first_of(kind) {
                        let dst = record.byte_range(stream);
                        let bytes = src.len_bytes.min(dst.end - dst.start);
                        encoder.copy_buffer_to_buffer(
                            &src.buffer,
                            src.src_offset_bytes,
                            chunk,
                            dst.start,
                            bytes,
                        );
                    }
                }
            }
        }
        sink.push(encoder.finish());
    }

    /// Bring `mesh_id`'s sidecar record in step with its deform state: the
    /// per-mesh and per-instance slot streams it should hold, and the tables
    /// that locate them. Streams in `replace` are uploaded again from the host
    /// copy even when one of the same shape is already there.
    ///
    /// A record that already holds the right streams is left where it is and
    /// only its tables are rewritten. Otherwise it moves, keeping its other
    /// streams; a mesh with no record gets one, and a record left holding
    /// nothing a draw needs is freed. Every record move rewrites the tables,
    /// so call this after anything else moves a deformed mesh's record too.
    pub(crate) fn sync_deform_record(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        replace: &[StreamKind],
    ) {
        let Some(mesh) = self.mesh_store.get_mut(mesh_id) else {
            return;
        };
        let wanted = self
            .deform
            .meshes
            .get(&mesh_id)
            .map_or_else(Vec::new, MeshDeform::streams);
        let holds = |kind: StreamKind, len: u32| {
            !replace.contains(&kind)
                && wanted
                    .iter()
                    .any(|(k, w)| *k == kind && w.len() as u32 == len)
        };
        let (keeps_all, adds) = match &mesh.sidecar {
            Some(record) => {
                let keeps_all = record
                    .streams
                    .iter()
                    .filter(|s| s.kind.is_deform())
                    .all(|s| holds(s.kind, s.len));
                let adds: Vec<StreamData> = wanted
                    .iter()
                    .filter(|(k, w)| {
                        !record
                            .first_of(*k)
                            .is_some_and(|s| holds(s.kind, s.len) && s.len == w.len() as u32)
                    })
                    .map(|(k, w)| StreamData::from_words(*k, w.clone()))
                    .collect();
                (keeps_all, adds)
            }
            None => (
                true,
                wanted
                    .iter()
                    .map(|(k, w)| StreamData::from_words(*k, w.clone()))
                    .collect(),
            ),
        };
        let moved = !keeps_all || !adds.is_empty();
        if moved {
            mesh.sidecar = match mesh.sidecar.take() {
                Some(old) => Some(self.sidecar.rebuild(
                    device,
                    old,
                    |s| !s.kind.is_deform() || holds(s.kind, s.len),
                    adds,
                )),
                None => {
                    let header = Self::sidecar_header(&self.geometry, mesh);
                    Some(self.sidecar.build(device, header, adds))
                }
            };
            let needs_record = mesh.sidecar.as_ref().is_some_and(|r| !r.streams.is_empty())
                || mesh.has_uv1
                || mesh.position_override_buffer.is_some()
                || mesh.normal_override_buffer.is_some();
            if !needs_record {
                if let Some(old) = mesh.sidecar.take() {
                    self.sidecar.free(&old);
                }
            }
            self.deform.counters.buffer_reallocations += 1;
            self.mesh_sidecar_epoch = self.mesh_sidecar_epoch.wrapping_add(1);
        }
        let (Some(record), Some(m)) = (mesh.sidecar.as_ref(), self.deform.meshes.get(&mesh_id))
        else {
            return;
        };
        for stream in record.streams.iter() {
            if let StreamKind::DeformTable(instance) = stream.kind {
                self.sidecar
                    .write_stream(record, stream, m.table(record, instance));
            }
        }
    }

    /// Register a deformer against the mesh shader family.
    ///
    /// Validates the descriptor's name and allocates a slot, composes every
    /// mesh-family base shader with the new deformer plus all previously
    /// registered ones, and runs each composed module through wgpu's
    /// validator. On success the registration is stored and the mesh-family
    /// pipelines (colour, OIT, shadow, outline mask and pick) are marked for
    /// rebuild; the rebuild itself runs once at the start of the next
    /// `prepare()`, so registering N deformers in a row costs one
    /// recompose-and-rebuild pass, not N.
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

        if defines_keep(&candidate) {
            self.deform.keep_slots |= 1u32 << slot;
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

        if defines_keep(&candidate) {
            self.deform.keep_slots |= 1u32 << slot;
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

    /// Swap the lit pipelines between the stripped and debug-vis shader
    /// variants (see `builders::strip_debug_vis`). Called when the per-frame
    /// `DebugVis` state flips. The swap recompiles the mesh-family
    /// pipelines, a one-off hitch that is acceptable for a debugging tool.
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

    /// Re-compose every mesh-family shader and rebuild the pipelines that
    /// draw from it. Runs from `flush_mesh_pipeline_rebuild` once any number
    /// of deformer registrations have validated, and when the debug-vis state
    /// flips.
    fn rebuild_mesh_pipelines(&mut self, device: &crate::gpu::Device) {
        // Every composed source is about to change, so the shared modules
        // built from the old ones are no use to anyone.
        self.shader_modules.lock().unwrap().clear();
        // Material-plugin pipeline sets compose on top of the deformer
        // registrations and the debug-vis strip state, both of which changed
        // when this runs. Drop them; the next prepare that references a
        // plugin id rebuilds its set from the fresh composition.
        for plugin in self.material_plugins.values_mut() {
            plugin.pipelines = None;
            plugin.pipelines_bindless = None;
        }
        self.material_plugin_epoch = self.material_plugin_epoch.wrapping_add(1);
        // Every family is rebuilt through its own `ensure_*`, which composes
        // the registered deformers in, so there is one copy of each build. A
        // family that has not been built yet is left alone: its first use
        // builds it from the same composition.
        if self.scene.ldr.is_some() {
            self.scene.ldr = None;
            self.ensure_ldr_mesh_pipelines(device);
        }
        if self.scene.hdr.is_some() {
            self.scene.hdr = None;
            self.ensure_hdr_mesh_pipelines(device);
        }
        if self.oit.pipeline.is_some() {
            self.oit.pipeline = None;
            self.ensure_oit_mesh_pipelines(device);
        }
        // The shadow passes run the deformer too: a caster skinned on the GPU
        // keeps the bind pose in its vertex buffer, so a depth pass that does
        // not deform rasterises the bind pose and the shadow never moves a limb.
        if self.shadow.pipeline.is_some() {
            self.shadow.pipeline = None;
            self.ensure_cascade_shadow_pipelines(device);
        }
        if self.shadow.point_pipeline.is_some() {
            self.shadow.point_pipeline = None;
            self.ensure_point_shadow_pipeline(device);
        }
        if self.outline.mask_pipeline.is_some() {
            self.outline.mask_pipeline = None;
            self.ensure_outline_pipelines(device);
        }
        // The object pick deforms too, so a deformed mesh is picked where it is
        // drawn. Rebuilt on the next pick; its layouts are kept.
        self.pick.pipeline = None;

        // The GPU-culled families rebuild through their own `ensure_*` too,
        // which build the bindless twins beside them, so a bindless bind group
        // can never meet a per-batch pipeline. Nothing rebuilds unless it was
        // already built.
        if self.cull.hdr.is_some() {
            self.cull.hdr = None;
            self.ensure_hdr_cull_pipelines(device);
        }
        if self.cull.oit.is_some() {
            self.cull.oit = None;
            self.ensure_oit_cull_pipelines(device);
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
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
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

    /// A renderer with one uploaded grid mesh, for the record-level tests.
    fn with_mesh() -> Option<(
        crate::gpu::Device,
        crate::gpu::Queue,
        crate::renderer::ViewportRenderer,
        MeshId,
        usize,
    )> {
        let (device, queue) = headless()?;
        let mut renderer = crate::renderer::ViewportRenderer::new(
            &device,
            crate::gpu::TextureFormat::Bgra8UnormSrgb,
        );
        let mesh = crate::geometry::primitives::grid_plane(1.0, 1.0, 2, 2);
        let vertices = mesh.positions.len();
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &mesh)
            .unwrap();
        Some((device, queue, renderer, mesh_id, vertices))
    }

    fn record(resources: &DeviceResources, mesh_id: MeshId) -> Option<SidecarRecord> {
        resources.mesh_store.get(mesh_id)?.sidecar.clone()
    }

    /// `count` words of the record's chunk from `word`, read back from the GPU
    /// after the pool's queued work has run.
    fn read_words(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &mut DeviceResources,
        mesh_id: MeshId,
        word: u32,
        count: usize,
    ) -> Vec<u32> {
        resources.flush_sidecars(device, queue);
        let record = record(resources, mesh_id).expect("record");
        let bytes = count as u64 * 4;
        let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("deform_test_readback"),
            size: bytes,
            usage: crate::gpu::BufferUsages::MAP_READ | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            resources.sidecar.chunk_buffer(record.chunk()),
            word as u64 * 4,
            &staging,
            0,
            bytes,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(crate::gpu::MapMode::Read, |_| {});
        let _ = device.poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let words = bytemuck::cast_slice::<u8, u32>(&crate::gpu::mapped_range(slice)).to_vec();
        staging.unmap();
        words
    }

    /// The table a draw of `(mesh_id, instance)` reads, from the GPU.
    fn table_on_gpu(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &mut DeviceResources,
        mesh_id: MeshId,
        instance: Option<u32>,
    ) -> Vec<u32> {
        let rec = record(resources, mesh_id);
        let word = resources.deform.table_word(mesh_id, instance, rec.as_ref());
        assert_ne!(word, 0, "the draw names a table");
        read_words(device, queue, resources, mesh_id, word, DEFORM_TABLE_WORDS)
    }

    #[test]
    fn the_table_locates_each_slot_stream_in_the_record() {
        let Some((device, queue, mut renderer, mesh, n)) = with_mesh() else {
            return;
        };
        let res = renderer.resources_mut();
        let slot0: Vec<u32> = (0..n as u32).map(|i| 100 + i).collect();
        let slot2: Vec<u32> = (0..2 * n as u32).map(|i| 500 + i).collect();
        res.attach_deform_slot(&device, mesh, 0, 4, bytemuck::cast_slice(&slot0));
        res.attach_deform_slot(&device, mesh, 2, 8, bytemuck::cast_slice(&slot2));
        assert_eq!(res.deform.flag_bits(mesh), 0b0101);

        let table = table_on_gpu(&device, &queue, res, mesh, None);
        assert_eq!(table[1], 1, "slot 0 stride");
        assert_eq!((table[2], table[3]), (0, 0), "slot 1 is empty");
        assert_eq!(table[5], 2, "slot 2 stride");
        assert!(
            table[DEFORM_TABLE_WORDS / 2..].iter().all(|&w| w == 0),
            "the mesh table has no instance half"
        );
        assert_eq!(read_words(&device, &queue, res, mesh, table[0], n), slot0);
        assert_eq!(
            read_words(&device, &queue, res, mesh, table[4], 2 * n),
            slot2
        );
    }

    #[test]
    fn detaching_the_last_slot_frees_the_record_it_created() {
        let Some((device, _queue, mut renderer, mesh, n)) = with_mesh() else {
            return;
        };
        let res = renderer.resources_mut();
        // A plain grid has no sidecars of its own, so the deform data is what
        // gives it a record.
        assert!(record(res, mesh).is_none());
        res.attach_deform_slot(&device, mesh, 1, 4, &vec![0u8; n * 4]);
        assert!(record(res, mesh).is_some());
        assert_eq!(res.deform.flag_bits(mesh), 0b0010);

        assert!(res.detach_deform_slot(&device, mesh, 1));
        assert_eq!(res.deform.flag_bits(mesh), 0);
        assert!(!res.deform.meshes.contains_key(&mesh));
        assert!(record(res, mesh).is_none());
        assert!(!res.detach_deform_slot(&device, mesh, 1));
    }

    #[test]
    fn an_instance_reads_its_own_table_and_others_the_mesh_one() {
        let Some((device, queue, mut renderer, mesh, n)) = with_mesh() else {
            return;
        };
        let res = renderer.resources_mut();
        let palette: Vec<u32> = (0..32).collect();
        res.attach_deform_slot(&device, mesh, 1, 4, &vec![0u8; n * 4]);
        res.attach_deform_slot_instance(
            &device,
            &queue,
            mesh,
            3,
            0,
            64,
            bytemuck::cast_slice(&palette),
        );
        // The flags fold the per-mesh and per-instance slots together.
        assert_eq!(res.deform.flag_bits(mesh), 0b0011);
        assert!(res.has_deform_slot_instance(mesh, 3, 0));
        assert!(!res.has_deform_slot_instance(mesh, 3, 1));

        let rec = record(res, mesh);
        let mesh_table = res.deform.table_word(mesh, None, rec.as_ref());
        let own = res.deform.table_word(mesh, Some(3), rec.as_ref());
        assert_ne!(own, mesh_table);
        // An instance with no data of its own reads the mesh table.
        assert_eq!(
            res.deform.table_word(mesh, Some(99), rec.as_ref()),
            mesh_table
        );

        let table = table_on_gpu(&device, &queue, res, mesh, Some(3));
        let half = DEFORM_TABLE_WORDS / 2;
        assert_eq!(table[3], 1, "the instance table carries the mesh slots too");
        assert_eq!(table[half + 1], 16, "palette stride, in words");
        assert_eq!(
            read_words(&device, &queue, res, mesh, table[half], 32),
            palette
        );

        // A same-shape palette lands in place: no record move.
        let _ = res.deform.take_counters();
        let next: Vec<u32> = (100..132).collect();
        res.attach_deform_slot_instance(
            &device,
            &queue,
            mesh,
            3,
            0,
            64,
            bytemuck::cast_slice(&next),
        );
        assert_eq!(res.deform.take_counters().buffer_reallocations, 0);
        assert_eq!(
            read_words(&device, &queue, res, mesh, table[half], 32),
            next
        );

        assert!(res.detach_deform_slot_instance(&device, &queue, mesh, 3, 0));
        assert_eq!(res.deform.flag_bits(mesh), 0b0010);
        assert!(!res.has_deform_slot_instance(mesh, 3, 0));
        let rec = record(res, mesh);
        assert_eq!(
            res.deform.table_word(mesh, Some(3), rec.as_ref()),
            res.deform.table_word(mesh, None, rec.as_ref())
        );
    }

    #[test]
    fn slot_index_assert_traps_out_of_range() {
        let Some((device, _queue, mut renderer, mesh, _)) = with_mesh() else {
            return;
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            renderer.resources_mut().attach_deform_slot(
                &device,
                mesh,
                DEFORM_SLOT_COUNT,
                4,
                &[0u8; 4],
            )
        }));
        assert!(result.is_err());
    }

    /// Anything else that moves a deformed mesh's record carries the deform
    /// streams along and rewrites the tables to their new words.
    #[test]
    fn a_record_move_keeps_the_table_on_the_streams() {
        let Some((device, queue, mut renderer, mesh, n)) = with_mesh() else {
            return;
        };
        let res = renderer.resources_mut();
        let slot0: Vec<u32> = (0..n as u32).map(|i| 7 + i).collect();
        res.attach_deform_slot(&device, mesh, 0, 4, bytemuck::cast_slice(&slot0));
        let before = record(res, mesh).unwrap();
        let radiance = res
            .upload_texture(
                &device,
                &queue,
                crate::TextureData::linear(1, 1, vec![255; 4]),
            )
            .unwrap();
        res.set_lightmap(
            &device,
            mesh,
            &vec![glam::Vec2::ZERO; n],
            crate::resources::LightmapData::NonDirectional { radiance },
            crate::resources::LightmapMode::Replace,
        )
        .unwrap();
        let after = record(res, mesh).unwrap();
        assert_ne!(
            (before.chunk(), before.base_word()),
            (after.chunk(), after.base_word()),
            "the lightmap moved the record"
        );
        let table = table_on_gpu(&device, &queue, res, mesh, None);
        assert_eq!(read_words(&device, &queue, res, mesh, table[0], n), slot0);
    }

    /// A deformer that reads its slot data registers before any mesh has
    /// drawn. The LDR `mesh.wgsl` family does not exist yet, so the deferred
    /// rebuild leaves it alone, and the first use builds it from the composed
    /// source. A broken composition would fail `register_deformer` at
    /// validation; a module that failed to compile would fail the build here.
    #[test]
    fn a_registered_deformer_reaches_the_ldr_mesh_pipelines() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);

        // Nothing has been drawn, so the base family is not composed yet.
        assert!(renderer.resources().scene.ldr.is_none());

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

        // Nothing has drawn a mesh, so there is still nothing to rebuild.
        assert!(renderer.resources().scene.ldr.is_none());

        // First use composes the family with the deformer in, and under
        // `Blocking` a read builds the member asked for.
        renderer.set_pipeline_compilation(crate::PipelineCompilation::Blocking);
        renderer.resources_mut().ensure_ldr_mesh_pipelines(&device);
        assert!(renderer.resources().scene.ldr.is_some());
        use crate::resources::scene_pipelines::MeshColourFamily as _;
        assert!(
            renderer
                .resources()
                .scene
                .ldr_family()
                .opaque(crate::renderer::pipeline_key::PipelineKey::default())
                .is_some()
        );
        assert!(renderer.resources().scene.wireframe().is_some());
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
    fn an_external_source_reserves_its_stream_and_is_copied_into_it() {
        let Some((device, queue, mut renderer, mesh, n)) = with_mesh() else {
            return;
        };
        let res = renderer.resources_mut();
        // Three words a vertex.
        let values: Vec<u32> = (0..3 * n as u32).map(|i| 900 + i).collect();
        let src = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("deform_src_test"),
            contents: bytemuck::cast_slice(&values),
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_SRC,
        });
        res.set_deform_slot_source_buffer(&device, mesh, 2, src, 12)
            .unwrap();
        assert!(res.has_deform_slot_source(mesh, 2));
        assert_eq!(res.deform.flag_bits(mesh), 0b0100);
        assert_eq!(res.deform.external_source_count, 1);

        res.flush_sidecars(&device, &queue);
        res.run_deform_slot_copies(&device, &mut crate::renderer::SubmitSink::inline(&queue));
        let table = table_on_gpu(&device, &queue, res, mesh, None);
        assert_eq!(table[5], 3);
        assert_eq!(
            read_words(&device, &queue, res, mesh, table[4], 3 * n),
            values
        );

        // CPU data on another slot sits beside it.
        res.attach_deform_slot(&device, mesh, 0, 4, &vec![0u8; n * 4]);
        let table = table_on_gpu(&device, &queue, res, mesh, None);
        assert_ne!(table[0], table[4]);

        // CPU data on the same slot replaces the source.
        res.attach_deform_slot(&device, mesh, 2, 12, &vec![0u8; n * 12]);
        assert!(!res.has_deform_slot_source(mesh, 2));
        assert!(res.has_deform_slot(mesh, 2));
        assert_eq!(res.deform.external_source_count, 0);
    }

    #[test]
    fn set_and_clear_slot_sources_roundtrip() {
        let Some((device, queue, mut renderer, mesh, n)) = with_mesh() else {
            return;
        };
        let res = renderer.resources_mut();
        let buf = copy_src_buffer(&device, n as u64 * 12);
        res.set_deform_slot_source_buffer(&device, mesh, 2, buf.clone(), 12)
            .unwrap();
        res.set_deform_slot_instance_source_buffer(&device, &queue, mesh, 5, 1, buf, 12)
            .unwrap();
        assert!(res.has_deform_slot_instance_source(mesh, 5, 1));
        assert_eq!(res.deform.external_source_count, 2);
        res.flush_sidecars(&device, &queue);
        res.run_deform_slot_copies(&device, &mut crate::renderer::SubmitSink::inline(&queue));

        assert!(res.clear_deform_slot_instance_source(&device, &queue, mesh, 5, 1));
        assert!(!res.has_deform_slot_instance_source(mesh, 5, 1));
        assert!(res.clear_deform_slot_source(&device, mesh, 2));
        assert!(!res.has_deform_slot_source(mesh, 2));
        assert_eq!(res.deform.external_source_count, 0);
        assert!(!res.deform.meshes.contains_key(&mesh));
        assert!(!res.clear_deform_slot_source(&device, mesh, 2));
    }

    #[test]
    fn set_deform_slot_source_buffer_rejects_bad_inputs() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let mesh = MeshId::from_index(105);

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
