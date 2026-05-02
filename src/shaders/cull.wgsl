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
    planes: array<FrustumPlane, 6>,
}

struct InstanceAabb {
    min:         vec3<f32>,
    batch_index: u32,
    max:         vec3<f32>,
    _pad:        u32,
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

@compute @workgroup_size(64)
fn cull_instances(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&instance_aabbs) {
        return;
    }

    let aabb = instance_aabbs[i];

    // Reject if outside any of the 6 frustum planes.
    for (var p = 0u; p < 6u; p++) {
        if aabb_outside_plane(aabb.min, aabb.max, frustum.planes[p]) {
            return;
        }
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
    if b >= arrayLength(&batch_metas) {
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
