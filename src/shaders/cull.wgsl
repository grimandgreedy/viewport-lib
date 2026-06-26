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
    _pad0:          vec2<f32>,
}

struct InstanceAabb {
    min:          vec3<f32>,
    batch_index:  u32,
    max:          vec3<f32>,
    // 1 = participates in shadow casting, 0 = skipped during shadow cull.
    cast_shadows: u32,
}

struct BatchMeta {
    index_count:     u32,
    first_index:     u32,
    instance_offset: u32,
    instance_count:  u32,
    vis_offset:      u32,
    is_transparent:  u32,
    _pad0:           u32,
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

    // Per-receiver shadow opt-out: shadow cull dispatches skip instances that
    // are marked as non-shadow casters via `ItemSettings.cast_shadows = false`.
    if frustum.shadow_pass == 1u && aabb.cast_shadows == 0u {
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

    // Visible: atomically claim a slot in the visibility buffer for this batch.
    let b    = aabb.batch_index;
    let slot = atomicAdd(&batch_counters[b], 1u);
    let bmeta = batch_metas[b];
    // Guard against counter overflow if instance_count drifts from buffer size.
    if slot < bmeta.instance_count {
        visibility_indices[bmeta.vis_offset + slot] = i;
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
        0i,
        bmeta.vis_offset,
    );

    // Zero the counter ready for the next frame's cull_instances dispatch.
    atomicStore(&batch_counters[b], 0u);
}
