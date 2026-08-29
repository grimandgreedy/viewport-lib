/// 4x4 identity matrix used as the default `model` for items that support a
/// per-frame transform. Column-major to match wgpu and glam conventions.
pub(super) const IDENTITY_MAT4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub use viewport_lib_types::ids::PickId;
