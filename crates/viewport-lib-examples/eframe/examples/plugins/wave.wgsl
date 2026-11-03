// GPU wave compute shader for `WavePlugin`. Reads the bind-pose vertex
// positions (flat `array<f32>` with 3 floats per vertex) and writes the
// displaced positions and their analytic normals into one pooled output
// buffer: positions at elements `0..vertex_count`, normals at
// `vertex_count..2 * vertex_count`. The renderer binds each region through
// `set_position_override_buffer_sliced` / `set_normal_override_buffer_sliced`,
// the way a GPU physics engine exposes windows of one pooled allocation.

struct Uniforms {
    // Total elapsed seconds. Drives the wave phase.
    time: f32,
    // Vertical amplitude of the wave (world units).
    amplitude: f32,
    // Spatial frequency (radians per world unit) along x and y.
    frequency: f32,
    // Vertex count. Used to bounds-check the dispatch and locate the
    // normals region.
    vertex_count: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read>       rest_positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> pool:           array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= u.vertex_count {
        return;
    }
    let base = i * 3u;
    let x = rest_positions[base];
    let y = rest_positions[base + 1u];
    let z_rest = rest_positions[base + 2u];

    // Two sinusoids combined for a more interesting surface than a single wave.
    let phase_x = u.frequency * x + u.time;
    let phase_y = u.frequency * 1.7 * y + u.time * 1.3;
    let dz = u.amplitude * sin(phase_x)
           + u.amplitude * 0.5 * sin(phase_y);
    let z = z_rest + dz;

    pool[base] = x;
    pool[base + 1u] = y;
    pool[base + 2u] = z;

    // Analytic surface normal from the gradient of the height field.
    // For z = h(x, y) the upward normal is normalize((-dh/dx, -dh/dy, 1)).
    let dhdx = u.amplitude * u.frequency * cos(phase_x);
    let dhdy = u.amplitude * 0.5 * u.frequency * 1.7 * cos(phase_y);
    let n = normalize(vec3<f32>(-dhdx, -dhdy, 1.0));

    let nbase = (u.vertex_count + i) * 3u;
    pool[nbase] = n.x;
    pool[nbase + 1u] = n.y;
    pool[nbase + 2u] = n.z;
}
