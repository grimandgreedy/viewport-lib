//! World positions accepted in f32 or f64.

use glam::{DVec3, Vec3};

/// A world-space position that converts to the f64 [`DVec3`] the renderer
/// keeps positions in: `DVec3`, `Vec3`, `[f64; 3]` or `[f32; 3]`.
///
/// Functions that place something in the world (a light, a clip shape, an
/// overlay anchor) take `impl IntoWorldPosition`, so an f32 value still works
/// and an f64 one keeps its precision far from the world origin. A literal
/// such as `[1.0, 2.0, 3.0]` is read as `[f64; 3]`.
pub trait IntoWorldPosition {
    /// The position as `DVec3`.
    fn into_world_position(self) -> DVec3;
}

impl IntoWorldPosition for DVec3 {
    fn into_world_position(self) -> DVec3 {
        self
    }
}

impl IntoWorldPosition for Vec3 {
    fn into_world_position(self) -> DVec3 {
        self.as_dvec3()
    }
}

impl IntoWorldPosition for [f64; 3] {
    fn into_world_position(self) -> DVec3 {
        DVec3::from_array(self)
    }
}

impl IntoWorldPosition for [f32; 3] {
    fn into_world_position(self) -> DVec3 {
        Vec3::from_array(self).as_dvec3()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take(p: impl IntoWorldPosition) -> DVec3 {
        p.into_world_position()
    }

    #[test]
    fn every_form_converts() {
        let expect = DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(take(expect), expect);
        assert_eq!(take(Vec3::new(1.0, 2.0, 3.0)), expect);
        assert_eq!(take([1.0f32, 2.0, 3.0]), expect);
        assert_eq!(take([1.0, 2.0, 3.0]), expect);
        // A literal reads as f64, so it keeps a value f32 would round.
        assert_eq!(take([1_048_576.01, 0.0, 0.0]).x, 1_048_576.01);
    }
}
