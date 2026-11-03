//! Movement-input helper shared across the character camera controllers.
//!
//! [`FirstPersonCameraController`](crate::controllers::first_person::FirstPersonCameraController) and
//! [`ThirdPersonCameraController`](crate::controllers::third_person::ThirdPersonCameraController) expose a
//! `forward_dir()` / `right_dir()` pair defining the ground-plane basis the
//! player walks in. [`wish_xy_from_actions`] resolves the standard fly-movement
//! actions against that basis and returns a normalised horizontal wish vector,
//! so hosts do not reimplement the same movement block per view.
//! [`wish_from_actions`] is the free-flight form: it adds up and down and takes a
//! full 3D basis, as the orbit controller's
//! [`NavigationMode::Fly`](crate::input::NavigationMode::Fly) uses.
//!
//! ```ignore
//! let wish = wish_xy_from_actions(&frame, fps.forward_dir(), fps.right_dir());
//! let velocity_xy = wish * walk_speed;
//! ```

use glam::{Vec2, Vec3};

use crate::input::action::Action;
use crate::input::action_frame::ActionFrame;

/// Resolve the forward / back / left / right movement actions into a horizontal
/// wish vector in the camera's ground-plane basis.
///
/// `forward_xy` and `right_xy` are the camera's horizontal forward and right
/// vectors (the controllers expose them via `forward_dir` / `right_dir`). Active
/// actions accumulate; the result is normalised so diagonal motion does not move
/// faster than orthogonal motion. Returns [`Vec2::ZERO`] when no movement action
/// is active. Callers multiply by walk speed and apply any sprint modifier.
///
/// Reads the [`Action::FlyForward`] / [`Action::FlyBackward`] /
/// [`Action::FlyLeft`] / [`Action::FlyRight`] entries from the frame, so the
/// same binding preset that drives free-fly movement also drives character
/// movement.
pub fn wish_xy_from_actions(frame: &ActionFrame, forward_xy: Vec3, right_xy: Vec3) -> Vec2 {
    let mut wish = Vec3::ZERO;
    if frame.is_active(Action::FlyForward) {
        wish += forward_xy;
    }
    if frame.is_active(Action::FlyBackward) {
        wish -= forward_xy;
    }
    if frame.is_active(Action::FlyRight) {
        wish += right_xy;
    }
    if frame.is_active(Action::FlyLeft) {
        wish -= right_xy;
    }
    Vec2::new(wish.x, wish.y)
        .try_normalize()
        .unwrap_or(Vec2::ZERO)
}

/// Resolve all six fly actions (forward / back / left / right / up / down) into a
/// unit wish vector in the given basis, or zero when nothing is held or the
/// held actions cancel.
///
/// The free-flight counterpart of [`wish_xy_from_actions`]: the basis is
/// typically the camera's own forward, right and up, so forward follows the
/// pitch of the view. Normalising keeps a diagonal at the same speed as a
/// single direction.
pub fn wish_from_actions(frame: &ActionFrame, forward: Vec3, right: Vec3, up: Vec3) -> Vec3 {
    let mut wish = Vec3::ZERO;
    if frame.is_active(Action::FlyForward) {
        wish += forward;
    }
    if frame.is_active(Action::FlyBackward) {
        wish -= forward;
    }
    if frame.is_active(Action::FlyRight) {
        wish += right;
    }
    if frame.is_active(Action::FlyLeft) {
        wish -= right;
    }
    if frame.is_active(Action::FlyUp) {
        wish += up;
    }
    if frame.is_active(Action::FlyDown) {
        wish -= up;
    }
    wish.try_normalize().unwrap_or(Vec3::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::action_frame::ResolvedActionState;

    fn frame_with(actions: &[Action]) -> ActionFrame {
        let mut f = ActionFrame::default();
        for &a in actions {
            f.actions.insert(a, ResolvedActionState::Held);
        }
        f
    }

    fn approx_eq(a: Vec2, b: Vec2) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn no_actions_returns_zero() {
        let wish = wish_xy_from_actions(&frame_with(&[]), Vec3::Y, Vec3::X);
        assert_eq!(wish, Vec2::ZERO);
    }

    #[test]
    fn forward_only_returns_forward() {
        let wish = wish_xy_from_actions(&frame_with(&[Action::FlyForward]), Vec3::Y, Vec3::X);
        assert!(approx_eq(wish, Vec2::Y));
    }

    #[test]
    fn right_only_returns_right() {
        let wish = wish_xy_from_actions(&frame_with(&[Action::FlyRight]), Vec3::Y, Vec3::X);
        assert!(approx_eq(wish, Vec2::X));
    }

    #[test]
    fn diagonal_is_normalised() {
        let wish = wish_xy_from_actions(
            &frame_with(&[Action::FlyForward, Action::FlyRight]),
            Vec3::Y,
            Vec3::X,
        );
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert!(approx_eq(wish, Vec2::new(inv_sqrt2, inv_sqrt2)));
    }

    #[test]
    fn opposing_actions_cancel() {
        let wish = wish_xy_from_actions(
            &frame_with(&[Action::FlyForward, Action::FlyBackward]),
            Vec3::Y,
            Vec3::X,
        );
        assert_eq!(wish, Vec2::ZERO);
    }

    #[test]
    fn rotated_basis_steers_correctly() {
        // yaw = PI/2: forward = +X, right = -Y.
        let wish = wish_xy_from_actions(
            &frame_with(&[Action::FlyForward]),
            Vec3::X,
            Vec3::new(0.0, -1.0, 0.0),
        );
        assert!(approx_eq(wish, Vec2::X));
    }

    #[test]
    fn wish_from_actions_normalises_three_axes() {
        let wish = wish_from_actions(
            &frame_with(&[Action::FlyForward, Action::FlyRight, Action::FlyUp]),
            Vec3::Y,
            Vec3::X,
            Vec3::Z,
        );
        assert!((wish.length() - 1.0).abs() < 1e-5);
        assert!((wish - Vec3::ONE / 3.0_f32.sqrt()).length() < 1e-5);
    }

    #[test]
    fn wish_from_actions_is_zero_with_nothing_held() {
        let wish = wish_from_actions(&frame_with(&[]), Vec3::Y, Vec3::X, Vec3::Z);
        assert_eq!(wish, Vec3::ZERO);
    }
}
