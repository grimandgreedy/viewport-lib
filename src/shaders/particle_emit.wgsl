// Particle emit kernel.
//
// One thread per slot in the particle buffer. Threads whose slot is dead
// (lifetime <= 0) compete for the limited spawn budget via an atomic counter;
// threads whose slot is still alive return immediately. Successful threads
// write a fresh particle drawn from the emitter's spawn shape and velocity
// distribution.

struct Particle {
    position:     vec3<f32>,
    lifetime:     f32,
    velocity:     vec3<f32>,
    max_lifetime: f32,
    colour:       vec4<f32>,
    size:         f32,
    spawn_seed:   f32,
    _pad:         vec2<f32>,
};

struct EmitParams {
    spawn_min:        vec3<f32>,
    spawn_kind:       u32,        // 0=Point, 1=Box, 2=Sphere
    spawn_max:        vec3<f32>,
    spawn_radius:     f32,
    vel_min:          vec3<f32>,
    vel_kind:         u32,        // 0=Fixed, 1=UniformBox, 2=UniformCone
    vel_max:          vec3<f32>,
    cone_half_angle:  f32,
    vel_axis:         vec3<f32>,
    cone_min_speed:   f32,
    colour:           vec4<f32>,
    spawn_count:      u32,
    capacity:         u32,
    rng_seed:         u32,
    size:             f32,
    lifetime_min:     f32,
    lifetime_max:     f32,
    cone_max_speed:   f32,
    _pad:             f32,
};

@group(0) @binding(0) var<uniform>             params:    EmitParams;
@group(1) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(1) @binding(1) var<storage, read_write> emit_remaining: atomic<u32>;

// PCG hash for cheap, decent-quality per-thread randomness.
fn pcg(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) / 4294967295.0;
}

@compute @workgroup_size(64)
fn emit_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if tid >= params.capacity { return; }

    let cur = particles[tid];
    if cur.lifetime > 0.0 { return; }

    // Each emitting thread claims one spawn ticket. If we under-claim
    // (counter hit 0) we restore it so the count stays correct.
    let claim = atomicSub(&emit_remaining, 1u);
    if claim == 0u || claim > params.spawn_count {
        atomicAdd(&emit_remaining, 1u);
        return;
    }

    var rng = pcg(params.rng_seed ^ tid);

    let position = spawn_position(&rng);
    let velocity = spawn_velocity(&rng);
    let life     = mix(params.lifetime_min, params.lifetime_max, rand_f(&rng));

    // Stable per-spawn seed used by the mesh draw route for `Random` align
    // rotation. Held untouched by the sim until the slot is recycled.
    let seed_bits = pcg(params.rng_seed ^ (tid * 0x68E31DA4u));
    let spawn_seed = f32(seed_bits) / 4294967295.0;

    var p: Particle;
    p.position     = position;
    p.lifetime     = life;
    p.velocity     = velocity;
    p.max_lifetime = life;
    p.colour       = params.colour;
    p.size         = params.size;
    p.spawn_seed   = spawn_seed;
    particles[tid] = p;
}

fn spawn_position(rng: ptr<function, u32>) -> vec3<f32> {
    if params.spawn_kind == 0u {
        return params.spawn_min;
    } else if params.spawn_kind == 1u {
        return vec3<f32>(
            mix(params.spawn_min.x, params.spawn_max.x, rand_f(rng)),
            mix(params.spawn_min.y, params.spawn_max.y, rand_f(rng)),
            mix(params.spawn_min.z, params.spawn_max.z, rand_f(rng)),
        );
    } else {
        // Sphere: rejection sample inside the unit ball, then scale. Capped at
        // 8 attempts so the validator can prove termination; the success rate
        // is ~52% per try so 8 misses is vanishingly rare.
        var p = vec3<f32>(0.0);
        for (var i = 0u; i < 8u; i = i + 1u) {
            p = vec3<f32>(
                rand_f(rng) * 2.0 - 1.0,
                rand_f(rng) * 2.0 - 1.0,
                rand_f(rng) * 2.0 - 1.0,
            );
            if dot(p, p) <= 1.0 {
                break;
            }
        }
        return params.spawn_min + p * params.spawn_radius;
    }
}

fn spawn_velocity(rng: ptr<function, u32>) -> vec3<f32> {
    if params.vel_kind == 0u {
        return params.vel_min;
    } else if params.vel_kind == 1u {
        return vec3<f32>(
            mix(params.vel_min.x, params.vel_max.x, rand_f(rng)),
            mix(params.vel_min.y, params.vel_max.y, rand_f(rng)),
            mix(params.vel_min.z, params.vel_max.z, rand_f(rng)),
        );
    } else {
        // Cone around `vel_axis`: pick a direction in a spherical cap, then
        // scale by a uniform speed.
        let axis = normalize(params.vel_axis);
        let cos_min = cos(params.cone_half_angle);
        let cos_a   = mix(cos_min, 1.0, rand_f(rng));
        let sin_a   = sqrt(max(0.0, 1.0 - cos_a * cos_a));
        let phi     = rand_f(rng) * 6.2831853;

        // Build an orthonormal basis around `axis`.
        let helper = select(vec3<f32>(1.0, 0.0, 0.0),
                            vec3<f32>(0.0, 1.0, 0.0),
                            abs(axis.x) > 0.9);
        let tangent   = normalize(cross(axis, helper));
        let bitangent = cross(axis, tangent);

        let dir = axis * cos_a
                + tangent   * (sin_a * cos(phi))
                + bitangent * (sin_a * sin(phi));
        let speed = mix(params.cone_min_speed, params.cone_max_speed, rand_f(rng));
        return dir * speed;
    }
}
