// @viewport-wgsl-version: 2
// Section-view clipping: the clip-plane and clip-volume uniforms at group 0
// bindings 4 and 6, and the tests a fragment stage discards on.
//
// SHARED_BINDINGS_WGSL includes this. A body that declares its own group-0
// bindings instead, such as one lit through the scene-lighting section,
// composes it alone as SHARED_CLIP_WGSL. Never both.

struct ClipPlanes {
    planes:          array<vec4<f32>, 6>,
    count:           u32,
    _pad0:           u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct ClipVolumeEntry {
    volume_type:  u32,
    _pad_a:       u32,
    _pad_b:       u32,
    _pad_c:       u32,
    centre:       vec3<f32>,
    radius:       f32,
    half_extents: vec3<f32>,
    _pad1:        f32,
    col0:         vec3<f32>,
    _pad2:        f32,
    col1:         vec3<f32>,
    _pad3:        f32,
    col2:         vec3<f32>,
    _pad4:        f32,
};

struct ClipVolumeUB {
    count:    u32,
    _pad_a:   u32,
    _pad_b:   u32,
    _pad_c:   u32,
    volumes:  array<ClipVolumeEntry, 4>,
};

@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

// Section-view clip planes: returns false when `world_pos` is on the
// clipped side of any active plane. Plugin fragment shaders call this and
// `discard` when it returns false to match the lib's clipping behaviour.
fn viewport_pass_clip_planes(world_pos: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(world_pos, plane.xyz) + plane.w < 0.0 {
            return false;
        }
    }
    return true;
}

// Composable clip volumes (box / sphere / cylinder): returns true when
// `world_pos` is inside every active clip volume. Returns true when no
// volumes are active.
fn viewport_pass_clip_volumes(world_pos: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_volume.count; i = i + 1u) {
        let e = clip_volume.volumes[i];
        if e.volume_type == 2u {
            let d = world_pos - e.centre;
            let local = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
            if abs(local.x) > e.half_extents.x
                || abs(local.y) > e.half_extents.y
                || abs(local.z) > e.half_extents.z {
                return false;
            }
        } else if e.volume_type == 3u {
            let ds = world_pos - e.centre;
            if dot(ds, ds) > e.radius * e.radius { return false; }
        } else if e.volume_type == 4u {
            let axis = e.col0;
            let d = world_pos - e.centre;
            let along = dot(d, axis);
            if abs(along) > e.half_extents.x { return false; }
            let radial = d - axis * along;
            if dot(radial, radial) > e.radius * e.radius { return false; }
        }
    }
    return true;
}

// Combined clip test. Returns true when the fragment should be kept,
// false when it should be discarded. Plugin fragment shaders typically:
//
//   if !viewport_clip_test(in.world_pos) { discard; }
fn viewport_clip_test(world_pos: vec3<f32>) -> bool {
    return viewport_pass_clip_planes(world_pos)
        && viewport_pass_clip_volumes(world_pos);
}
