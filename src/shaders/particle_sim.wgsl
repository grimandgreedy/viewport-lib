// Particle sim kernel.
//
// One thread per slot. Skips dead slots (lifetime <= 0). For each live
// particle, applies every active force, integrates position, and decrements
// the remaining lifetime by `dt`. A particle whose new lifetime is at or below
// zero is considered dead next frame and becomes available for emit reuse.

struct Particle {
    position:     vec3<f32>,
    lifetime:     f32,
    velocity:     vec3<f32>,
    max_lifetime: f32,
    colour:       vec4<f32>,
    size:         f32,
    _pad:         vec3<f32>,
};

// `_pad0/_pad1/_pad2` are three plain u32s rather than a `vec3<u32>` so the
// struct stays 48-byte aligned, matching the Rust `GpuForce` layout. A
// `vec3<u32>` would impose 16-byte alignment and inflate the struct to 64
// bytes.
struct GpuForce {
    kind:  u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    v0:    vec4<f32>,
    v1:    vec4<f32>,
};

struct SimParams {
    dt:          f32,
    capacity:    u32,
    force_count: u32,
    _pad:        u32,
    forces:      array<GpuForce, 8>,
};

@group(0) @binding(0) var<uniform>             params:    SimParams;
@group(1) @binding(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn sim_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if tid >= params.capacity { return; }

    var p = particles[tid];
    if p.lifetime <= 0.0 { return; }

    let dt = params.dt;
    let count = min(params.force_count, 8u);
    for (var i = 0u; i < count; i = i + 1u) {
        let f = params.forces[i];
        if f.kind == 0u {
            // Gravity: constant acceleration.
            p.velocity = p.velocity + f.v0.xyz * dt;
        } else if f.kind == 1u {
            // Drag: fraction of velocity lost per second.
            p.velocity = p.velocity * max(0.0, 1.0 - f.v0.x * dt);
        } else if f.kind == 2u {
            // Point attractor: acceleration scales as strength / (dist + falloff)^2.
            let to = f.v0.xyz - p.position;
            let dist = length(to);
            if dist > 1e-4 {
                let denom = dist + f.v1.x;
                let accel = f.v0.w / (denom * denom);
                p.velocity = p.velocity + (to / dist) * accel * dt;
            }
        }
    }

    p.position = p.position + p.velocity * dt;
    p.lifetime = p.lifetime - dt;
    particles[tid] = p;
}
