// Cut deformer body. Moves nothing; its `keep` removes the part of the mesh
// outside every cut the drawn item selects.
//
// Per-instance data, one 80-byte element each: element 0 holds the cut count
// in word 0, then one element per cut:
//   w0 kind (0 plane, 1 box, 2 sphere, 3 range)   w1 flip (0 or 1)
//   w4..w7   plane: normal.xyz, distance
//            box and sphere: centre.xyz (sphere: radius in w7)
//            range: min, max
//   w8..w19  box only: three unit axes, each followed by its half-extent
// Per-mesh data, when a range is used: the scalar field, one f32 per vertex.

fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    return v;
}

fn word(ctx: DeformContext, e: u32, k: u32) -> f32 {
    return deform_read_instance_f32(ctx.slot, e, k);
}

fn vec_at(ctx: DeformContext, e: u32, k: u32) -> vec3<f32> {
    return vec3<f32>(word(ctx, e, k), word(ctx, e, k + 1u), word(ctx, e, k + 2u));
}

// Signed: positive on the kept side, roughly the distance to the cut.
fn signed_keep(v: DeformVertex, ctx: DeformContext, e: u32) -> f32 {
    let kind = deform_read_instance_u32(ctx.slot, e, 0u);
    let p = v.position;
    var s = 1.0e30;
    switch kind {
        case 0u: {
            s = dot(p, vec_at(ctx, e, 4u)) - word(ctx, e, 7u);
        }
        case 1u: {
            let d = p - vec_at(ctx, e, 4u);
            let q = vec3<f32>(
                abs(dot(d, vec_at(ctx, e, 8u))) - word(ctx, e, 11u),
                abs(dot(d, vec_at(ctx, e, 12u))) - word(ctx, e, 15u),
                abs(dot(d, vec_at(ctx, e, 16u))) - word(ctx, e, 19u),
            );
            s = -max(q.x, max(q.y, q.z));
        }
        case 2u: {
            s = word(ctx, e, 7u) - distance(p, vec_at(ctx, e, 4u));
        }
        case 3u: {
            // No field attached to the mesh: nothing to threshold on.
            if deform_slot_stride(ctx.slot) == 0u {
                return 1.0e30;
            }
            let f = deform_read_f32(ctx.slot, v.vertex_index, 0u);
            s = min(f - word(ctx, e, 4u), word(ctx, e, 5u) - f);
        }
        default: {}
    }
    if deform_read_instance_u32(ctx.slot, e, 1u) != 0u {
        s = -s;
    }
    return s;
}

fn keep(v: DeformVertex, ctx: DeformContext) -> f32 {
    // An item that selects no cut, on a mesh that carries a field for one
    // that does, is kept whole.
    if deform_instance_slot_stride(ctx.slot) == 0u {
        return 1.0e30;
    }
    let count = deform_read_instance_u32(ctx.slot, 0u, 0u);
    var k = 1.0e30;
    for (var e = 1u; e <= count; e++) {
        k = min(k, signed_keep(v, ctx, e));
    }
    return k;
}
