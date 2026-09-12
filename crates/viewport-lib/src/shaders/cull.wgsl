// GPU-driven culling compute shader.
//
// Dispatch 1 - cull_instances (workgroup_size 64):
//   One thread per instance. Tests the world-space AABB against the 6 frustum
//   planes. On pass: atomically claims a visibility slot and writes the instance
//   index into the visibility buffer.
//
// Dispatch 2 - write_indirect_args (workgroup_size 64):
//   One thread per batch. Reads the final visible count from the batch counter,
//   writes one DrawIndexedIndirect entry, then zeroes the counter for the next
//   frame.
//
//   The two dispatches must run in separate compute passes. wgpu inserts a
//   storage-buffer barrier between compute passes automatically, which
//   guarantees dispatch 2 sees all writes from dispatch 1.
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
// The pack runs as a standard three-phase scan over fixed-size chunks of each
// batch's instance range: count survivors per chunk, exclusive-scan those
// counts within the batch, then scatter each survivor to `chunk base + its
// rank inside the chunk`. Every step is a pure function of instance index, so
// the result does not depend on scheduling.
//
// All of the compaction's scratch lives in one buffer, in four regions. It is
// one binding because the cull layout sits exactly at wgpu's default limit of 8
// storage buffers per stage with it: a second binding here would stop the
// renderer working on a device created with default limits.
//
//   PLAN  [b]        = index of batch b's first chunk, [batch_count] = total
//   OWNER [chunk]    = the batch that chunk belongs to
//   TOTAL [chunk]    = survivors in that chunk, rewritten by the scan as its base
//   FLAGS [instance] = the cull's verdict: the instance's own index, or
//                      CULLED_SLOT
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
const OWNER_BASE: u32 = PLAN_BASE + PLAN_CAP;
const CHUNK_CAP: u32 = 16384u;
const TOTAL_BASE: u32 = OWNER_BASE + CHUNK_CAP;
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

    // Visible. The counter still accumulates atomically (a count does not care
    // about order, and `write_indirect_args` reads it), but the instance's
    // position in the drawn list comes from the compaction below, not from the
    // order threads arrived here.
    let b    = aabb.batch_index;
    let slot = atomicAdd(&batch_counters[b], 1u);
    if slot_local < slot_meta.instance_count {
        compact_scratch[FLAGS_BASE + slot_meta.vis_offset + slot_local] = i;
        // Arrival-order list as well, so a submission too large for the chunk
        // plan (see MAX_PLAN_* in indirect.rs) still has a correct list to draw
        // when the compaction is skipped. When it runs, it overwrites this with
        // the instance-ordered one.
        if slot < slot_meta.instance_count {
            visibility_indices[slot_meta.vis_offset + slot] = i;
        }
    }
}

// Workgroup scratch shared by the counting and scattering phases.
var<workgroup> scan: array<u32, CHUNK>;

// Phase 1 of the compaction: lay out the chunk space.
//
// Chunks are batch-aligned (a chunk never spans two batches), so a chunk's
// survivor count belongs to exactly one batch's scan. Serial in one invocation:
// it walks batches, not instances, and writes one entry per chunk.
@compute @workgroup_size(1)
fn plan_chunks() {
    var next = 0u;
    for (var b = 0u; b < frustum.batch_count; b++) {
        compact_scratch[PLAN_BASE + b] = next;
        let chunks = (batch_metas[b].instance_count + CHUNK - 1u) / CHUNK;
        for (var c = 0u; c < chunks; c++) {
            compact_scratch[OWNER_BASE + next + c] = b;
        }
        next = next + chunks;
    }
    compact_scratch[PLAN_BASE + frustum.batch_count] = next;
}

// Phase 2: survivors per chunk, one workgroup per chunk.
@compute @workgroup_size(CHUNK)
fn chunk_counts(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ch = wg.x;
    if ch >= compact_scratch[PLAN_BASE + frustum.batch_count] {
        return;
    }
    let owner = compact_scratch[OWNER_BASE + ch];
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

// Phase 3: exclusive scan of the chunk counts within each batch, so each chunk
// learns where its survivors start. Serial over chunks in one invocation: the
// work is one add per chunk, and chunks are instances/256.
@compute @workgroup_size(1)
fn scan_chunks() {
    for (var b = 0u; b < frustum.batch_count; b++) {
        let first = compact_scratch[PLAN_BASE + b];
        let last = compact_scratch[PLAN_BASE + b + 1u];
        var base = 0u;
        for (var c = first; c < last; c++) {
            let count = compact_scratch[TOTAL_BASE + c];
            compact_scratch[TOTAL_BASE + c] = base;
            base = base + count;
        }
    }
}

// Phase 4: scatter survivors into the visible list at `chunk base + rank`.
@compute @workgroup_size(CHUNK)
fn scatter_visible(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ch = wg.x;
    if ch >= compact_scratch[PLAN_BASE + frustum.batch_count] {
        return;
    }
    let owner = compact_scratch[OWNER_BASE + ch];
    let bmeta = batch_metas[owner];
    let idx = (ch - compact_scratch[PLAN_BASE + owner]) * CHUNK + lid.x;

    var entry = CULLED_SLOT;
    if idx < bmeta.instance_count {
        entry = compact_scratch[FLAGS_BASE + bmeta.vis_offset + idx];
    }
    let survives = select(0u, 1u, entry != CULLED_SLOT);

    // Inclusive scan of the survivor flags, so each lane learns its rank.
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
        visibility_indices[bmeta.vis_offset + compact_scratch[TOTAL_BASE + ch] + rank] = entry;
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
