// Projected tetrahedra shader for transparent unstructured volume rendering.
//
// Each tetrahedron is rendered as a screen-space AABB bounding quad (6 vertices,
// 2 triangles). The vertex shader reads tet data from a storage buffer and computes
// the screen-space AABB. The fragment shader performs a ray-tet half-space intersection,
// computes Beer-Lambert thickness opacity, and writes weighted-blended OIT output
// (same targets as mesh_oit.wgsl).
//
// Group 0: Camera uniform (shared camera_bind_group_layout, only bindings 0 and 4 used).
// Group 1: PT uniforms + tet storage buffer + colourmap LUT + sampler.
//
// Opaque-surface clipping: the OIT render pass loads the opaque depth buffer and
// uses LessEqual depth compare without depth writes, so the hardware depth test
// clips bounding-quad fragments behind already-rendered opaque geometry.

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
};

// Section-view clip planes (matches mesh.wgsl ClipPlanes, binding 4 of camera group).
struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:              u32,
    _pad0:              u32,
    viewport_width:     f32,
    viewport_height:    f32,
};

struct PtUniforms {
    density:       f32,
    scalar_min:    f32,
    scalar_max:    f32,
    threshold_min: f32,
    threshold_max: f32,
    unlit:         u32,
};

// One tetrahedron on the GPU: four vec4 slots (64 bytes, 16-byte aligned).
// v0.xyz = world position of vertex 0,  v0.w = per-tet scalar value.
// v1..v3: world positions (w unused).
struct GpuTet {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    v3: vec4<f32>,
};

struct OitOut {
    @location(0) accum:  vec4<f32>,  // Rgba16Float accumulation buffer
    @location(1) reveal: f32,         // R8Unorm reveal (transmittance) buffer
};

// Group 0: shared with all other passes.
@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;

// Group 1: projected-tet specific.
@group(1) @binding(0) var<uniform>        uniforms:         PtUniforms;
@group(1) @binding(1) var<storage, read>  tets:             array<GpuTet>;
@group(1) @binding(2) var                 colourmap_lut:     texture_2d<f32>;
@group(1) @binding(3) var                 colourmap_sampler: sampler;

struct VsOut {
    @builtin(position)              clip_pos: vec4<f32>,
    // Flat (non-interpolated) tet vertex world positions and scalar.
    @location(0) @interpolate(flat) v0:       vec4<f32>,   // xyz=pos, w=scalar
    @location(1) @interpolate(flat) v1:       vec3<f32>,
    @location(2) @interpolate(flat) v2:       vec3<f32>,
    @location(3) @interpolate(flat) v3:       vec3<f32>,
    // Interpolated NDC XY for ray reconstruction.
    @location(4)                    ndc_xy:   vec2<f32>,
};

// Map vertex_index in [0,5] to one of the 4 AABB corners (two-triangle quad).
// Corner order: 0=min_x/min_y, 1=max_x/min_y, 2=max_x/max_y, 3=min_x/max_y.
fn corner_select(vi: u32) -> vec2<u32> {
    // triangle 0: corners 0,1,3  triangle 1: corners 0,3,2
    // Splits along BL->TR diagonal so both triangles cover the full AABB rectangle.
    let idx = array<u32, 6>(0u, 1u, 3u, 0u, 3u, 2u);
    let c = idx[vi];
    return vec2<u32>(c & 1u, (c >> 1u) & 1u);  // (x_hi, y_hi)
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vi: u32,
    @builtin(instance_index) ti: u32,
) -> VsOut {
    let tet = tets[ti];

    // Project all 4 world-space vertices to clip space, then NDC.
    let c0 = camera.view_proj * vec4<f32>(tet.v0.xyz, 1.0);
    let c1 = camera.view_proj * vec4<f32>(tet.v1.xyz, 1.0);
    let c2 = camera.view_proj * vec4<f32>(tet.v2.xyz, 1.0);
    let c3 = camera.view_proj * vec4<f32>(tet.v3.xyz, 1.0);

    // Perspective divide to NDC. Clamp w to avoid division by zero.
    let eps = 1e-6;
    let w0 = select(c0.w, eps, abs(c0.w) < eps);
    let w1 = select(c1.w, eps, abs(c1.w) < eps);
    let w2 = select(c2.w, eps, abs(c2.w) < eps);
    let w3 = select(c3.w, eps, abs(c3.w) < eps);
    let n0 = c0.xyz / w0;
    let n1 = c1.xyz / w1;
    let n2 = c2.xyz / w2;
    let n3 = c3.xyz / w3;

    // Screen-space AABB of the projected XY positions.
    let min_xy = min(min(n0.xy, n1.xy), min(n2.xy, n3.xy));
    let max_xy = max(max(n0.xy, n1.xy), max(n2.xy, n3.xy));

    // Clamp AABB to clip space [-1, 1] to avoid emitting huge quads
    // for tets that straddle the near plane.
    let clamped_min = max(min_xy, vec2<f32>(-1.0));
    let clamped_max = min(max_xy, vec2<f32>( 1.0));

    // Use the minimum NDC z as the quad depth (closest point = passes depth test).
    let quad_z = min(min(n0.z, n1.z), min(n2.z, n3.z));

    // Select AABB corner from vertex_index.
    let corner = corner_select(vi % 6u);
    let ndc_x = select(clamped_min.x, clamped_max.x, corner.x != 0u);
    let ndc_y = select(clamped_min.y, clamped_max.y, corner.y != 0u);

    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc_x, ndc_y, clamp(quad_z, 0.0, 1.0), 1.0);
    out.v0       = tet.v0;
    out.v1       = tet.v1.xyz;
    out.v2       = tet.v2.xyz;
    out.v3       = tet.v3.xyz;
    out.ndc_xy   = vec2<f32>(ndc_x, ndc_y);
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

// Intersect a ray (eye + t*dir) with one half-space of a tetrahedron face.
// face_a, face_b, face_c: the three face vertices.
// opposite: the tet vertex NOT on this face (used to determine outward direction).
// Updates t_enter / t_exit so the ray segment stays inside the tet.
fn update_slab(
    face_a: vec3<f32>, face_b: vec3<f32>, face_c: vec3<f32>,
    opposite: vec3<f32>,
    eye: vec3<f32>, ray_dir: vec3<f32>,
    t_enter: ptr<function, f32>,
    t_exit:  ptr<function, f32>,
) {
    // Compute face normal from edge cross product.
    var n = cross(face_b - face_a, face_c - face_a);
    // Ensure the normal points OUTWARD (away from the opposite vertex).
    // If dot(n, opposite - face_a) > 0, n is pointing toward opposite (inward),
    // so flip it.
    if dot(n, opposite - face_a) > 0.0 {
        n = -n;
    }
    let denom = dot(n, ray_dir);
    if abs(denom) < 1e-10 {
        return;  // ray parallel to this face
    }
    // t where ray meets this face plane: eye + t*dir on the plane => dot(n, p-face_a)=0
    let t = dot(n, face_a - eye) / denom;
    if denom < 0.0 {
        // Ray moving toward the outside (denom<0 with outward n means ray entering).
        *t_enter = max(*t_enter, t);
    } else {
        // Ray moving outward: exiting this half-space.
        *t_exit = min(*t_exit, t);
    }
}

@fragment
fn fs_main(in: VsOut) -> OitOut {
    // Reconstruct world-space ray direction from NDC position.
    let ndc_near = vec4<f32>(in.ndc_xy, 0.0, 1.0);
    let ndc_far  = vec4<f32>(in.ndc_xy, 1.0, 1.0);
    let world_near_h = camera.inv_view_proj * ndc_near;
    let world_far_h  = camera.inv_view_proj * ndc_far;
    let world_near = world_near_h.xyz / world_near_h.w;
    let world_far  = world_far_h.xyz  / world_far_h.w;
    let ray_dir = normalize(world_far - world_near);

    let eye = camera.eye_pos;

    let p0 = in.v0.xyz;
    let p1 = in.v1;
    let p2 = in.v2;
    let p3 = in.v3;

    // Slab test: intersect ray with all 4 tet half-spaces.
    // update_slab uses the opposite vertex to guarantee an outward-pointing normal,
    // so the result is correct regardless of tet handedness.
    var t_enter: f32 = 0.0;
    var t_exit:  f32 = 1e30;

    update_slab(p1, p2, p3, p0, eye, ray_dir, &t_enter, &t_exit);
    update_slab(p0, p2, p3, p1, eye, ray_dir, &t_enter, &t_exit);
    update_slab(p0, p1, p3, p2, eye, ray_dir, &t_enter, &t_exit);
    update_slab(p0, p1, p2, p3, eye, ray_dir, &t_enter, &t_exit);

    // Section-view clip planes: constrain [t_enter, t_exit] to the kept half-spaces.
    for (var i = 0u; i < clip_planes.count; i++) {
        let plane = clip_planes.planes[i];  // (nx, ny, nz, d); kept when dot(p,n)+d >= 0
        let n_plane = plane.xyz;
        let bias = dot(n_plane, eye) + plane.w;  // signed distance of eye from plane
        let denom = dot(n_plane, ray_dir);
        if abs(denom) < 1e-10 {
            // Ray parallel to plane. Discard if eye is on the clipped side.
            if bias < 0.0 {
                discard;
            }
        } else {
            let t_plane = -bias / denom;
            if denom > 0.0 {
                // Ray moves into the kept half-space at t_plane.
                t_enter = max(t_enter, t_plane);
            } else {
                // Ray exits the kept half-space at t_plane.
                t_exit = min(t_exit, t_plane);
            }
        }
    }

    // Face bias: treat rays that miss the tet by up to FACE_BIAS world units as
    // grazing hits. This fills sub-pixel seam gaps at tet boundaries without
    // moving the bounding quad (which causes wobble when the AABB shifts
    // frame-to-frame during camera motion). The resulting alpha for a bare
    // graze is ~density * FACE_BIAS ~= 5e-4, essentially invisible but stable.
    const FACE_BIAS: f32 = 1e-3;
    if t_enter >= t_exit + FACE_BIAS {
        discard;
    }
    let thickness = max(t_exit - t_enter, 0.0);

    // Beer-Lambert opacity, or a flat per-fragment alpha when `unlit` is set.
    // The flat-alpha mode skips the per-fragment thickness modulation so the
    // mesh reads as a translucent solid LUT colour, matching the spirit of
    // `ItemSettings.unlit` on other item types (no per-fragment shading).
    let alpha = select(
        1.0 - exp(-uniforms.density * thickness),
        clamp(uniforms.density * 0.25, 0.05, 0.6),
        uniforms.unlit != 0u,
    );

    // Map scalar to [0,1] and sample colourmap LUT.
    let scalar = in.v0.w;

    // Threshold: discard tets outside [threshold_min, threshold_max].
    if scalar < uniforms.threshold_min || scalar > uniforms.threshold_max {
        discard;
    }
    let t_scalar = clamp(
        (scalar - uniforms.scalar_min) / max(uniforms.scalar_max - uniforms.scalar_min, 1e-10),
        0.0, 1.0,
    );
    let colour = textureSample(colourmap_lut, colourmap_sampler, vec2<f32>(t_scalar, 0.5));

    // Weighted-blended OIT (same formula as mesh_oit.wgsl).
    let depth = in.clip_pos.z;
    let w = alpha * max(1e-2, 1.0 - depth);
    var out: OitOut;
    out.accum  = vec4<f32>(colour.rgb * alpha * w, alpha * w);
    out.reveal = alpha;
    return out;
}
