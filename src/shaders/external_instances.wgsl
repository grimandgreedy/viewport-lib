// External-instances shader: instanced mesh draw whose per-instance
// translation comes from a consumer-owned storage buffer of tightly packed
// [x, y, z] f32 triples (12-byte stride).
//
// The mesh vertex buffer is bound on slot 0 with the standard `Vertex`
// layout; position and normal are consumed. The positions buffer is indexed
// by `@builtin(instance_index)`: the draw call's instance range selects the
// window of the buffer to render, so no per-instance offset uniform is
// needed. WGSL's array<vec3<f32>> has a 16-byte element stride, so the
// buffer is declared array<f32> and indexed 3 * i by hand.
//
// Simple fixed-direction lambert shading; the item colour is the albedo.

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct ClipPlanes {
    planes:          array<vec4<f32>, 6>,
    count:           u32,
    _pad0:           u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct DrawUniform {
    model:  mat4x4<f32>,
    colour: vec4<f32>,
    scale:  f32,
    _pad0:  f32,
    _pad1:  f32,
    _pad2:  f32,
};

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(4) var<uniform>       clip_planes: ClipPlanes;

@group(1) @binding(0) var<uniform>       draw_ub:     DrawUniform;
@group(1) @binding(1) var<storage, read> positions:   array<f32>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:   vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn, @builtin(instance_index) ii: u32) -> VertexOut {
    var out: VertexOut;

    let pi = ii * 3u;
    let plen = arrayLength(&positions);
    if pi + 2u >= plen {
        // Out of range (stale count after a pool shrink): degenerate vertex,
        // the rasteriser drops the whole instance.
        out.clip_pos     = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.world_pos    = vec3<f32>(0.0);
        out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
        return out;
    }
    let instance_pos = vec3<f32>(positions[pi], positions[pi + 1u], positions[pi + 2u]);

    let local = instance_pos + in.position * draw_ub.scale;
    let world4 = draw_ub.model * vec4<f32>(local, 1.0);
    let model3 = mat3x3<f32>(
        draw_ub.model[0].xyz,
        draw_ub.model[1].xyz,
        draw_ub.model[2].xyz,
    );

    out.clip_pos     = camera.view_proj * world4;
    out.world_pos    = world4.xyz;
    out.world_normal = normalize(model3 * in.normal);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }
    // Headlight lambert: lit from the camera with an ambient floor, so
    // instances read as solids from any orbit angle without scene lighting.
    let n = normalize(in.world_normal);
    let l = normalize(camera.eye_pos - in.world_pos);
    let ndl = max(dot(n, l), 0.0);
    let shade = 0.25 + 0.75 * ndl;
    return vec4<f32>(draw_ub.colour.rgb * shade, draw_ub.colour.a);
}
