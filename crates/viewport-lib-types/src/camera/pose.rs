//! Where a camera sits and which way it faces.

use glam::{DQuat, DVec3, Quat, Vec3};

/// A camera's placement in the world: its eye position in f64 and its
/// orientation.
///
/// `orientation` maps camera-space axes to world space. The camera looks down
/// its local `-Z` with `+Y` up, so the identity looks straight down the world
/// `-Z` axis (a top view in the Z-up world).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pose {
    /// Eye position in world space.
    pub position: DVec3,
    /// Camera-to-world rotation.
    pub orientation: DQuat,
}

impl Default for Pose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Pose {
    /// At the world origin, looking down `-Z`.
    pub const IDENTITY: Pose = Pose {
        position: DVec3::ZERO,
        orientation: DQuat::IDENTITY,
    };

    /// A pose from an eye position and an orientation.
    pub fn new(position: impl Into<DVec3>, orientation: DQuat) -> Self {
        Self {
            position: position.into(),
            orientation,
        }
    }

    /// The pose of an orbit camera: the eye sits `distance` from `center`
    /// along the orientation's local `+Z`, so `eye = center + orientation * (Z * distance)`.
    pub fn orbit(center: impl Into<DVec3>, distance: f32, orientation: Quat) -> Self {
        Self {
            position: center.into() + eye_offset(orientation, distance),
            orientation: orientation.as_dquat(),
        }
    }

    /// The pose at `eye` looking at `target`, rolled so `up` points as close
    /// to the screen's up as it can. When `up` is parallel to the view
    /// direction another axis is used.
    pub fn look_at(eye: impl Into<DVec3>, target: impl Into<DVec3>, up: Vec3) -> Self {
        let eye = eye.into();
        let forward = (target.into() - eye).normalize_or(DVec3::NEG_Z);
        let mut up = up.as_dvec3().normalize_or(DVec3::Z);
        if forward.cross(up).length_squared() < 1e-12 {
            up = if forward.z.abs() < 0.9 {
                DVec3::Z
            } else {
                DVec3::Y
            };
        }
        let right = forward.cross(up).normalize();
        let up = right.cross(forward);
        let basis = glam::DMat3::from_cols(right, up, -forward);
        Self {
            position: eye,
            orientation: DQuat::from_mat3(&basis).normalize(),
        }
    }

    /// The orientation in f32.
    pub fn orientation_f32(&self) -> Quat {
        self.orientation.as_quat()
    }

    /// The direction the camera looks, in world space (the local `-Z`).
    pub fn forward(&self) -> Vec3 {
        -(self.orientation_f32() * Vec3::Z)
    }

    /// The camera's right direction in world space (the local `+X`).
    pub fn right(&self) -> Vec3 {
        self.orientation_f32() * Vec3::X
    }

    /// The camera's up direction in world space (the local `+Y`).
    pub fn up(&self) -> Vec3 {
        self.orientation_f32() * Vec3::Y
    }

    /// The point `distance` in front of the eye along the view direction.
    pub fn target_at(&self, distance: f32) -> DVec3 {
        self.position - eye_offset(self.orientation_f32(), distance)
    }
}

/// The eye's offset from an orbit centre. Both [`Pose::orbit`] and
/// [`Pose::target_at`] use it, so `target_at` returns an orbit pose's centre
/// exactly.
pub(crate) fn eye_offset(orientation: Quat, distance: f32) -> DVec3 {
    (orientation * (Vec3::Z * distance)).as_dvec3()
}
