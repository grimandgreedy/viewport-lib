// Particle mesh shader: instanced mesh draw whose per-instance transform is
// composed in the vertex stage from a live GPU particle.
//
// The mesh vertex buffer is bound on slot 0 with the standard `Vertex` layout
// (position, normal, colour, uv, tangent); only position and uv are consumed
// here. Per-instance data comes from a storage buffer indexed by
// `@builtin(instance_index)`. Unlit; the particle colour multiplies any
// optional albedo sample.

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

struct MeshDrawUniform {
    align:       u32,        // 0=Identity, 1=Velocity, 2=Random
    has_texture: u32,
    _pad0:       u32,
    _pad1:       u32,
};

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

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(4) var<uniform>       clip_planes: ClipPlanes;

@group(1) @binding(0) var<uniform>       draw_ub:     MeshDrawUniform;
@group(1) @binding(1) var                albedo:      texture_2d<f32>;
@group(1) @binding(2) var                albedo_samp: sampler;
@group(1) @binding(3) var<storage, read> particles:   array<Particle>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:   vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       colour:    vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
    @location(2)       uv:        vec2<f32>,
};

// PCG-style hash. Same kernel as the emit shader; reused so the random
// rotation derived here is reproducible across draws of the same particle.
fn hash_u32(x: u32) -> u32 {
    var state = x * 747796405u + 2891336453u;
    let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_f(x: u32) -> f32 {
    return f32(hash_u32(x)) / 4294967295.0;
}

// Build a 3x3 rotation matrix from a unit quaternion (x, y, z, w).
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (yy + zz),       2.0 * (xy + wz),       2.0 * (xz - wy)),
        vec3<f32>(      2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz),       2.0 * (yz + wx)),
        vec3<f32>(      2.0 * (xz + wy),       2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
    );
}

// Uniform random unit quaternion via Shoemake's parameterisation.
fn random_quat(seed: f32) -> vec4<f32> {
    let bits = bitcast<u32>(seed);
    let u1 = hash_f(bits);
    let u2 = hash_f(bits ^ 0x9E3779B1u);
    let u3 = hash_f(bits ^ 0x85EBCA77u);
    let r1 = sqrt(1.0 - u1);
    let r2 = sqrt(u1);
    let t1 = 6.2831853 * u2;
    let t2 = 6.2831853 * u3;
    return vec4<f32>(r1 * sin(t1), r1 * cos(t1), r2 * sin(t2), r2 * cos(t2));
}

// Shortest-arc rotation that maps +Y to `dir`. `dir` must be unit length.
fn align_y_to(dir: vec3<f32>) -> mat3x3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let c = dot(up, dir);
    if c > 0.9999 {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    if c < -0.9999 {
        // Antiparallel: 180-degree rotation around any axis orthogonal to up.
        return mat3x3<f32>(
            vec3<f32>(1.0,  0.0,  0.0),
            vec3<f32>(0.0, -1.0,  0.0),
            vec3<f32>(0.0,  0.0, -1.0),
        );
    }
    let axis = cross(up, dir);
    let s = length(axis);
    let k = axis / s;
    // Rodrigues' formula with sin = s, cos = c.
    let K = mat3x3<f32>(
        vec3<f32>( 0.0,   k.z, -k.y),
        vec3<f32>(-k.z,  0.0,   k.x),
        vec3<f32>( k.y, -k.x,  0.0),
    );
    let I = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    return I + K * s + (K * K) * (1.0 - c);
}

@vertex
fn vs_main(in: VertexIn, @builtin(instance_index) ii: u32) -> VertexOut {
    var out: VertexOut;
    let p = particles[ii];

    if p.lifetime <= 0.0 {
        // Dead slot. Emit a clip-space vertex outside any meaningful frustum so
        // the rasteriser drops every triangle of this instance.
        out.clip_pos  = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.colour    = vec4<f32>(0.0);
        out.world_pos = vec3<f32>(0.0);
        out.uv        = vec2<f32>(0.0);
        return out;
    }

    var rot = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    if draw_ub.align == 1u {
        let speed2 = dot(p.velocity, p.velocity);
        if speed2 > 1e-8 {
            rot = align_y_to(p.velocity / sqrt(speed2));
        }
    } else if draw_ub.align == 2u {
        rot = quat_to_mat3(random_quat(p.spawn_seed));
    }

    let local = rot * (in.position * p.size);
    let world_pos = p.position + local;

    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.colour    = p.colour;
    out.world_pos = world_pos;
    out.uv        = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }

    var colour = in.colour;
    if draw_ub.has_texture != 0u {
        colour = colour * textureSample(albedo, albedo_samp, in.uv);
    }
    if colour.a <= 0.001 { discard; }
    return colour;
}
