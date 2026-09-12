// GPU-driven culling compute shader.
//
// cull_instances (workgroup_size 64):
//   One thread per instance. Tests the world-space AABB against the 6 frustum
//   planes and records its verdict in the instance's own scratch slot.
//
// plan_chunks / chunk_counts / scatter_visible:
//   Pack the survivors into the visibility buffer in instance order. See the
//   compaction notes further down.
//
// write_indirect_args (workgroup_size 64):
//   One thread per batch. Reads the final visible count from the batch counter,
//   writes one DrawIndexedIndirect entry, then zeroes the counter for the next
//   frame.
//
//   write_indirect_args must run in its own compute pass. wgpu inserts a
//   storage-buffer barrier between compute passes automatically, which
//   guarantees it sees every write from the passes before it.
//
// All buffers share a single bind group (group 0).

struct FrustumPlane {
    normal:   vec3<f32>,
    distance: f32,
}

struct FrustumUniform {
    planes:         array<FrustumPlane, 6>,
    instance_count: u32,
    batch_count:    u32,
    // 1 = shadow cull dispatch (skip non-cast-shadow instances), 0 = main cull.
    shadow_pass:    u32,
    // 1 = run the HiZ occlusion test after the frustum test, 0 = skip it.
    do_occlusion:   u32,
    // Camera view-projection, used to project instance AABBs to screen for the
    // occlusion test. Identity when occlusion is off.
    view_proj:      mat4x4<f32>,
    // HiZ mip-0 dimensions in pixels (matches the depth target the pyramid was
    // built from), used to map projected boxes to texel footprints.
    viewport:       vec2<f32>,
    // Camera layer mask: an instance is culled when its per-object mask shares
    // no bit with this. Only meaningful when do_mask_cull == 1.
    cull_mask:      u32,
    // 1 = run the per-camera layer-mask reject (main cull with a real instance
    // buffer bound at binding 8), 0 = skip it (shadow / single-mesh / plugin
    // dispatches bind the fallback instance buffer).
    do_mask_cull:   u32,
    // 1 = the compaction phases below run after this dispatch and write both
    // the visible list and the per-batch visible counts. 0 = the submission
    // outgrew the chunk plan, so `cull_instances` writes the list itself in
    // arrival order and counts survivors atomically.
    compact_enabled: u32,
    // Pad to a 16-byte multiple. Three scalars, not a vec3: a vec3 aligns to
    // 16 and would push the struct to 224 bytes.
    _pad0:           u32,
    _pad1:           u32,
    _pad2:           u32,
}

struct InstanceAabb {
    min:          vec3<f32>,
    batch_index:  u32,
    max:          vec3<f32>,
    // 1 = participates in shadow casting, 0 = skipped during shadow cull.
    cast_shadows: u32,
}

// Per-instance record, matching the 144-byte `InstanceData` in
// `mesh_instanced.wgsl` and the Rust `InstanceData`. The cull reads only
// `object_mask` (offset 140); the other fields keep the array stride at 144 so
// the instance index aligns with `instance_aabbs`.
struct InstanceData {
    model:                  mat4x4<f32>,  // offset 0
    colour:                 vec4<f32>,    // offset 64
    selected:               u32,          // offset 80
    wireframe:              u32,          // offset 84
    has_texture:            u32,          // offset 88
    has_normal_map:         u32,          // offset 92
    has_ao_map:             u32,          // offset 96
    unlit:                  u32,          // offset 100
    receive_shadows:        u32,          // offset 104
    material_id:            u32,          // offset 108
    alpha_cutoff:           f32,          // offset 112
    alpha_flag:             u32,          // offset 116
    has_light_probe:        u32,          // offset 120
    light_probe_index:      u32,          // offset 124
    ignore_clip:            u32,          // offset 128
    custom_data_id:         u32,          // offset 132
    backface_pattern_scale: f32,          // offset 136
    object_mask:            u32,          // offset 140
}

struct BatchMeta {
    index_count:     u32,
    first_index:     u32,
    instance_offset: u32,
    instance_count:  u32,
    vis_offset:      u32,
    is_transparent:  u32,
    base_vertex:     i32,
    _pad1:           u32,
}

// Matches the wgpu DrawIndexedIndirect layout:
//   index_count, instance_count, first_index, base_vertex (i32), first_instance.
struct DrawIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform>             frustum:            FrustumUniform;
@group(0) @binding(1) var<storage, read>       instance_aabbs:     array<InstanceAabb>;
@group(0) @binding(2) var<storage, read>       batch_metas:        array<BatchMeta>;
@group(0) @binding(3) var<storage, read_write> batch_counters:     array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> visibility_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect_args:      array<DrawIndirect>;
// HiZ max-depth pyramid (mip 0 = full-res scene depth). Bound with a 1x1
// fallback when occlusion is off so the layout is always satisfied.
@group(0) @binding(6) var                      hiz_tex:            texture_2d<f32>;
// Cull breakdown counters: [0] = instances entering the cull (post shadow
// opt-out), [1] = instances surviving the frustum test (before occlusion).
// The drawn count comes from the indirect args. Cleared each main dispatch.
@group(0) @binding(7) var<storage, read_write> cull_stats:         array<atomic<u32>, 2>;
// Per-instance records (144-byte stride), read only for `object_mask` in the
// layer-mask reject. Bound with a 1-element fallback when do_mask_cull is off
// (shadow / single-mesh / plugin dispatches) so the layout is always satisfied.
@group(0) @binding(8) var<storage, read>       instance_data:      array<InstanceData>;
// --- Deterministic compaction -------------------------------------------
//
// The visible list has to be packed in instance order, not in the order
// threads happen to finish, or an unchanged scene submits its draws in a
// different order every frame and the rendered image is not reproducible.
//
// The pack runs as a scan over fixed-size chunks of each batch's instance
// range: `plan_chunks` lays the chunks out, `chunk_counts` counts survivors in
// each, and `scatter_visible` sums the earlier chunks' counts to find its base
// and writes each survivor at `base + its rank inside the chunk`. Every step is
// a pure function of instance index, so the result does not depend on
// scheduling.
//
// All of the compaction's scratch lives in one buffer, in three regions. It is
// one binding because the cull layout sits exactly at wgpu's default limit of 8
// storage buffers per stage with it: a second binding here would stop the
// renderer working on a device created with default limits.
//
//   PLAN  [b]        = index of batch b's first chunk, [batch_count] = total
//   TOTAL [chunk]    = survivors in that chunk
//   FLAGS [instance] = the cull's verdict: the instance's own index, or
//                      CULLED_SLOT
//
// There is no chunk-to-batch table. PLAN is non-decreasing, so a chunk finds
// its batch by bisecting PLAN once per workgroup, which is about 13 steps at
// the batch ceiling. Materialising the table instead cost a serial store per
// chunk in a single invocation, and that was the compaction's largest cost.
//
// FLAGS is separate from `visibility_indices` on purpose: chunks scatter in
// parallel, and packing in place would let one chunk's writes land in another
// chunk's not-yet-read range.
@group(0) @binding(9) var<storage, read_write> compact_scratch: array<u32>;

// Instances per compaction chunk, and the workgroup width that walks one.
const CHUNK: u32 = 256u;
// Region bases inside `compact_scratch`. Must match MAX_PLAN_* in indirect.rs.
const PLAN_BASE: u32 = 0u;
const PLAN_CAP: u32 = 8193u;
const CHUNK_CAP: u32 = 16384u;
const TOTAL_BASE: u32 = PLAN_BASE + PLAN_CAP;
const FLAGS_BASE: u32 = TOTAL_BASE + CHUNK_CAP;
// Marker left in a scratch slot whose instance the cull rejected.
const CULLED_SLOT: u32 = 0xffffffffu;

// Returns true if the AABB is entirely on the outer (negative) side of the plane.
// Uses the positive-vertex method: take the corner most aligned with the plane
// normal; if that corner is still outside, the whole AABB is outside.
fn aabb_outside_plane(aabb_min: vec3<f32>, aabb_max: vec3<f32>, plane: FrustumPlane) -> bool {
    let n = plane.normal;
    let px = select(aabb_min.x, aabb_max.x, n.x >= 0.0);
    let py = select(aabb_min.y, aabb_max.y, n.y >= 0.0);
    let pz = select(aabb_min.z, aabb_max.z, n.z >= 0.0);
    return dot(n, vec3<f32>(px, py, pz)) + plane.distance < 0.0;
}

// Returns true if the screen-space box of the AABB is entirely behind the
// nearest HiZ depth covering it, i.e. every pixel it could touch is already
// occupied by closer geometry. Conservative: any uncertainty (a corner behind
// the near plane, a zero-area box) returns false so a visible instance is
// never culled.
fn aabb_occluded(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    var ndc_min = vec3<f32>(1.0e30, 1.0e30, 1.0e30);
    var ndc_max = vec3<f32>(-1.0e30, -1.0e30, -1.0e30);
    for (var c = 0u; c < 8u; c++) {
        let corner = vec3<f32>(
            select(aabb_min.x, aabb_max.x, (c & 1u) != 0u),
            select(aabb_min.y, aabb_max.y, (c & 2u) != 0u),
            select(aabb_min.z, aabb_max.z, (c & 4u) != 0u),
        );
        let clip = frustum.view_proj * vec4<f32>(corner, 1.0);
        // Behind or on the near plane: cannot project reliably, treat as visible.
        if clip.w <= 0.0 {
            return false;
        }
        let ndc = clip.xyz / clip.w;
        ndc_min = min(ndc_min, ndc);
        ndc_max = max(ndc_max, ndc);
    }

    // Nearest depth the box can reach (wgpu NDC z is 0 at the near plane).
    let nearest_z = ndc_min.z;

    // NDC xy [-1,1] -> uv [0,1] with y flipped, then to pixels.
    let uv_min = vec2<f32>(ndc_min.x * 0.5 + 0.5, 1.0 - (ndc_max.y * 0.5 + 0.5));
    let uv_max = vec2<f32>(ndc_max.x * 0.5 + 0.5, 1.0 - (ndc_min.y * 0.5 + 0.5));
    let px_min = clamp(uv_min, vec2<f32>(0.0), vec2<f32>(1.0)) * frustum.viewport;
    let px_max = clamp(uv_max, vec2<f32>(0.0), vec2<f32>(1.0)) * frustum.viewport;
    let box = px_max - px_min;
    let max_side = max(box.x, box.y);
    if max_side <= 0.0 {
        return false;
    }

    // Pick the mip whose texels are ~the box size, so the loop below reads at
    // most a 2x2 footprint.
    let max_level = i32(textureNumLevels(hiz_tex)) - 1;
    let level = clamp(i32(ceil(log2(max_side))), 0, max_level);
    let lvl_dim = vec2<f32>(textureDimensions(hiz_tex, level));
    let scale = lvl_dim / frustum.viewport;
    let t_min = vec2<i32>(floor(px_min * scale));
    let t_max = vec2<i32>(floor(px_max * scale));
    let lvl_last = vec2<i32>(lvl_dim) - vec2<i32>(1, 1);

    var hiz_max = 0.0;
    for (var y = t_min.y; y <= t_max.y; y++) {
        for (var x = t_min.x; x <= t_max.x; x++) {
            let cx = clamp(x, 0, lvl_last.x);
            let cy = clamp(y, 0, lvl_last.y);
            hiz_max = max(hiz_max, textureLoad(hiz_tex, vec2<i32>(cx, cy), level).r);
        }
    }

    // Occluded when the closest the box can be is still farther than the
    // farthest occluder pixel in its footprint.
    return nearest_z > hiz_max;
}

@compute @workgroup_size(64)
fn cull_instances(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= frustum.instance_count {
        return;
    }

    let aabb = instance_aabbs[i];

    // Claim this instance's own scratch slot and mark it culled before any of
    // the early-outs below, so every slot carries a definite verdict each frame
    // and the compaction never reads a stale one. A survivor overwrites it with
    // its index at the end of this function.
    let slot_meta = batch_metas[aabb.batch_index];
    let slot_local = i - slot_meta.instance_offset;
    if slot_local < slot_meta.instance_count {
        compact_scratch[FLAGS_BASE + slot_meta.vis_offset + slot_local] = CULLED_SLOT;
    }

    // Per-receiver shadow opt-out: shadow cull dispatches skip instances that
    // are marked as non-shadow casters via `ItemSettings.cast_shadows = false`.
    if frustum.shadow_pass == 1u && aabb.cast_shadows == 0u {
        return;
    }

    // Per-camera layer cull: drop an instance whose object mask shares no bit
    // with the camera's cull mask. Gated, so dispatches that bind the fallback
    // instance buffer (shadow, single-mesh, plugin submissions) skip it. This
    // mirrors the CPU-side collect cull and is what makes a per-viewport
    // `cull_mask` filter the shared instanced batches in multi-viewport.
    if frustum.do_mask_cull == 1u
        && (instance_data[i].object_mask & frustum.cull_mask) == 0u {
        return;
    }

    // Total instances entering the cull this frame.
    atomicAdd(&cull_stats[0], 1u);

    // Reject if outside any of the 6 frustum planes.
    for (var p = 0u; p < 6u; p++) {
        if aabb_outside_plane(aabb.min, aabb.max, frustum.planes[p]) {
            return;
        }
    }

    // Survived the frustum test (counted before the occlusion test so the
    // breakdown can attribute the two cull stages separately).
    atomicAdd(&cull_stats[1], 1u);

    // Drop instances fully hidden behind nearer geometry.
    if frustum.do_occlusion == 1u && aabb_occluded(aabb.min, aabb.max) {
        return;
    }

    // Visible. Record the verdict in this instance's own slot; where it lands
    // in the drawn list is decided by the compaction below, from the instance
    // index, never from the order threads arrived here.
    if slot_local >= slot_meta.instance_count {
        return;
    }
    compact_scratch[FLAGS_BASE + slot_meta.vis_offset + slot_local] = i;

    // When the compaction runs it writes the list and the per-batch counts, so
    // there is nothing more to do here. Only the over-capacity fallback needs
    // the arrival-order list and the atomic tally, and it pays for both: every
    // survivor in a batch contends on that one counter.
    if frustum.compact_enabled == 1u {
        return;
    }
    let slot = atomicAdd(&batch_counters[aabb.batch_index], 1u);
    if slot < slot_meta.instance_count {
        visibility_indices[slot_meta.vis_offset + slot] = i;
    }
}

// Workgroup scratch shared by the counting and scattering phases.
var<workgroup> scan: array<u32, CHUNK>;

// Step 1 of the compaction: lay out the chunk space.
//
// Chunks are batch-aligned (a chunk never spans two batches), so a chunk's
// survivor count belongs to exactly one batch's scan. Serial in one invocation,
// but one iteration per batch: a scene drawing one instanced mesh runs this
// loop once regardless of how many instances it has.
@compute @workgroup_size(1)
fn plan_chunks() {
    var next = 0u;
    for (var b = 0u; b < frustum.batch_count; b++) {
        compact_scratch[PLAN_BASE + b] = next;
        next = next + (batch_metas[b].instance_count + CHUNK - 1u) / CHUNK;
    }
    compact_scratch[PLAN_BASE + frustum.batch_count] = next;
}

// The batch owning chunk `ch`: the last batch whose first chunk is at or before
// it. PLAN is non-decreasing, so this bisects. Empty batches share a PLAN entry
// with their successor, and the upper bisect keeps the non-empty one, which is
// the batch the chunk actually belongs to.
fn chunk_owner(ch: u32) -> u32 {
    var lo = 0u;
    var hi = frustum.batch_count;
    while lo + 1u < hi {
        let mid = lo + (hi - lo) / 2u;
        if compact_scratch[PLAN_BASE + mid] <= ch {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Step 2: survivors per chunk, one workgroup per chunk.
@compute @workgroup_size(CHUNK)
fn chunk_counts(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ch = wg.x;
    if ch >= compact_scratch[PLAN_BASE + frustum.batch_count] {
        return;
    }
    let owner = chunk_owner(ch);
    let bmeta = batch_metas[owner];
    let idx = (ch - compact_scratch[PLAN_BASE + owner]) * CHUNK + lid.x;

    var survives = 0u;
    if idx < bmeta.instance_count && compact_scratch[FLAGS_BASE + bmeta.vis_offset + idx] != CULLED_SLOT {
        survives = 1u;
    }
    scan[lid.x] = survives;
    workgroupBarrier();

    // Tree reduction to scan[0].
    for (var stride = CHUNK / 2u; stride > 0u; stride = stride >> 1u) {
        if lid.x < stride {
            scan[lid.x] = scan[lid.x] + scan[lid.x + stride];
        }
        workgroupBarrier();
    }
    if lid.x == 0u {
        compact_scratch[TOTAL_BASE + ch] = scan[0];
    }
}

// Step 3: scatter survivors into the visible list at `chunk base + rank`.
//
// A chunk's base is the number of survivors in its batch's earlier chunks, so
// the workgroup derives it by summing those counts itself: lanes read the
// predecessor counts strided and tree-reduce them. That is `chunks_before /
// CHUNK` reads per lane, against a dispatch launch for a separate scan pass,
// and a batch would need thousands of chunks before the sum came close to
// costing what the launch did.
//
// The batch's last chunk also knows the batch total (its base plus its own
// count), so it stores that as the batch's visible count for
// `write_indirect_args`. Deriving it here is what lets the cull kernel skip a
// per-instance atomic on a single address per batch.
@compute @workgroup_size(CHUNK)
fn scatter_visible(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ch = wg.x;
    if ch >= compact_scratch[PLAN_BASE + frustum.batch_count] {
        return;
    }
    let owner = chunk_owner(ch);
    let bmeta = batch_metas[owner];
    let first = compact_scratch[PLAN_BASE + owner];
    let idx = (ch - first) * CHUNK + lid.x;

    // Sum of the earlier chunks' survivor counts: this chunk's base.
    var before = 0u;
    for (var c = first + lid.x; c < ch; c = c + CHUNK) {
        before = before + compact_scratch[TOTAL_BASE + c];
    }
    scan[lid.x] = before;
    workgroupBarrier();
    for (var stride = CHUNK / 2u; stride > 0u; stride = stride >> 1u) {
        if lid.x < stride {
            scan[lid.x] = scan[lid.x] + scan[lid.x + stride];
        }
        workgroupBarrier();
    }
    let base = scan[0];

    var entry = CULLED_SLOT;
    if idx < bmeta.instance_count {
        entry = compact_scratch[FLAGS_BASE + bmeta.vis_offset + idx];
    }
    let survives = select(0u, 1u, entry != CULLED_SLOT);

    // Every lane has read `base`, so the scratch can be reused for the rank
    // scan: an inclusive scan of the survivor flags.
    workgroupBarrier();
    scan[lid.x] = survives;
    workgroupBarrier();
    for (var offset = 1u; offset < CHUNK; offset = offset << 1u) {
        var add = 0u;
        if lid.x >= offset {
            add = scan[lid.x - offset];
        }
        workgroupBarrier();
        scan[lid.x] = scan[lid.x] + add;
        workgroupBarrier();
    }

    if survives == 1u {
        let rank = scan[lid.x] - 1u;
        visibility_indices[bmeta.vis_offset + base + rank] = entry;
    }

    // Last chunk of the batch: publish the batch's visible count. Batches with
    // no chunks at all keep the zero `write_indirect_args` left behind.
    if lid.x == 0u && ch + 1u == compact_scratch[PLAN_BASE + owner + 1u] {
        atomicStore(&batch_counters[owner], base + scan[CHUNK - 1u]);
    }
}

@compute @workgroup_size(64)
fn write_indirect_args(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    if b >= frustum.batch_count {
        return;
    }

    let bmeta         = batch_metas[b];
    let visible_count = atomicLoad(&batch_counters[b]);

    // Write one DrawIndexedIndirect struct.
    // first_instance = vis_offset so the vertex shader indexes into
    // visibility_indices starting at the right offset for this batch.
    // (Requires INDIRECT_FIRST_INSTANCE device feature.)
    indirect_args[b] = DrawIndirect(
        bmeta.index_count,
        visible_count,
        bmeta.first_index,
        bmeta.base_vertex,
        bmeta.vis_offset,
    );

    // Zero the counter ready for the next frame's cull_instances dispatch.
    atomicStore(&batch_counters[b], 0u);
}
