//! Keyframe camera animation with Catmull-Rom interpolation.

use crate::camera::camera::{Camera, CameraTarget};

// ---------------------------------------------------------------------------
// CameraTrack
// ---------------------------------------------------------------------------

/// A time-sorted sequence of camera keyframes for animation.
///
/// Build the track with [`push`](Self::push), then call
/// [`interpolate_camera`] each frame to get a smoothly interpolated
/// [`CameraTarget`] that you can apply to a [`Camera`].
///
/// Center and distance are interpolated with a Catmull-Rom spline.
/// Orientation is interpolated with spherical linear interpolation (slerp)
/// between adjacent keyframes.
#[derive(Clone, Debug, Default)]
pub struct CameraTrack {
    /// Time-sorted keyframes: `(time_seconds, target)`.
    keyframes: Vec<(f64, CameraTarget)>,
}

impl CameraTrack {
    /// Create an empty track.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a track from a pre-built list of `(time, target)` pairs.
    ///
    /// The list is sorted by time; duplicate times are kept (the later one wins
    /// during interpolation).
    pub fn from_keyframes(mut keyframes: Vec<(f64, CameraTarget)>) -> Self {
        keyframes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { keyframes }
    }

    /// Append a keyframe at `time` seconds, keeping the list sorted.
    pub fn push(&mut self, time: f64, target: CameraTarget) {
        let pos = self
            .keyframes
            .partition_point(|(t, _)| *t <= time);
        self.keyframes.insert(pos, (time, target));
    }

    /// Return a slice of all keyframes in time order.
    pub fn keyframes(&self) -> &[(f64, CameraTarget)] {
        &self.keyframes
    }

    /// Duration from the first to the last keyframe, or `0.0` if empty.
    pub fn duration(&self) -> f64 {
        match (self.keyframes.first(), self.keyframes.last()) {
            (Some(first), Some(last)) => (last.0 - first.0).max(0.0),
            _ => 0.0,
        }
    }

    /// Return `true` if the track has no keyframes.
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Number of keyframes.
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Interpolate a [`CameraTarget`] from a [`CameraTrack`] at time `t`.
///
/// - Center and distance use Catmull-Rom spline interpolation.
/// - Orientation uses spherical linear interpolation between the two
///   surrounding keyframes.
///
/// If the track is empty a default `CameraTarget` is returned. If `t` is
/// before the first keyframe, the first keyframe is returned; if after the
/// last, the last keyframe is returned.
pub fn interpolate_camera(track: &CameraTrack, t: f64) -> CameraTarget {
    let kfs = track.keyframes();

    match kfs.len() {
        0 => CameraTarget {
            center: glam::Vec3::ZERO,
            distance: 5.0,
            orientation: glam::Quat::IDENTITY,
        },
        1 => kfs[0].1,
        _ => {
            // Clamp to track range.
            if t <= kfs[0].0 {
                return kfs[0].1;
            }
            if t >= kfs[kfs.len() - 1].0 {
                return kfs[kfs.len() - 1].1;
            }

            // Find segment index i such that kfs[i].time <= t < kfs[i+1].time.
            let i = kfs.partition_point(|(kt, _)| *kt <= t).saturating_sub(1);
            let i = i.min(kfs.len() - 2);

            let t0 = kfs[i].0;
            let t1 = kfs[i + 1].0;
            let s = if (t1 - t0).abs() < 1e-12 {
                0.0_f32
            } else {
                ((t - t0) / (t1 - t0)) as f32
            };

            // Gather 4 control points (with phantom endpoints at boundaries).
            let p1 = kfs[i].1;
            let p2 = kfs[i + 1].1;
            let p0 = if i > 0 {
                kfs[i - 1].1
            } else {
                // Phantom: reflect p2 through p1.
                CameraTarget {
                    center: p1.center * 2.0 - p2.center,
                    distance: p1.distance * 2.0 - p2.distance,
                    orientation: p1.orientation, // kept simple for boundary
                }
            };
            let p3 = if i + 2 < kfs.len() {
                kfs[i + 2].1
            } else {
                // Phantom: reflect p1 through p2.
                CameraTarget {
                    center: p2.center * 2.0 - p1.center,
                    distance: p2.distance * 2.0 - p1.distance,
                    orientation: p2.orientation,
                }
            };

            CameraTarget {
                center: catmull_rom_vec3(p0.center, p1.center, p2.center, p3.center, s),
                distance: catmull_rom_f32(p0.distance, p1.distance, p2.distance, p3.distance, s)
                    .max(Camera::MIN_DISTANCE),
                orientation: p1.orientation.slerp(p2.orientation, s).normalize(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn catmull_rom_f32(p0: f32, p1: f32, p2: f32, p3: f32, s: f32) -> f32 {
    let s2 = s * s;
    let s3 = s2 * s;
    0.5 * (2.0 * p1
        + (-p0 + p2) * s
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * s2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * s3)
}

fn catmull_rom_vec3(
    p0: glam::Vec3,
    p1: glam::Vec3,
    p2: glam::Vec3,
    p3: glam::Vec3,
    s: f32,
) -> glam::Vec3 {
    glam::Vec3::new(
        catmull_rom_f32(p0.x, p1.x, p2.x, p3.x, s),
        catmull_rom_f32(p0.y, p1.y, p2.y, p3.y, s),
        catmull_rom_f32(p0.z, p1.z, p2.z, p3.z, s),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn target(x: f32, d: f32) -> CameraTarget {
        CameraTarget {
            center: glam::Vec3::new(x, 0.0, 0.0),
            distance: d,
            orientation: glam::Quat::IDENTITY,
        }
    }

    #[test]
    fn test_empty_track_returns_default() {
        let track = CameraTrack::new();
        let t = interpolate_camera(&track, 0.0);
        assert_eq!(t.distance, 5.0);
    }

    #[test]
    fn test_single_keyframe() {
        let mut track = CameraTrack::new();
        track.push(0.0, target(3.0, 7.0));
        let t = interpolate_camera(&track, 5.0);
        assert!((t.center.x - 3.0).abs() < 1e-5);
        assert!((t.distance - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp_before_start() {
        let mut track = CameraTrack::new();
        track.push(1.0, target(1.0, 1.0));
        track.push(2.0, target(2.0, 2.0));
        let t = interpolate_camera(&track, 0.0);
        assert!((t.center.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp_after_end() {
        let mut track = CameraTrack::new();
        track.push(1.0, target(1.0, 1.0));
        track.push(2.0, target(2.0, 2.0));
        let t = interpolate_camera(&track, 5.0);
        assert!((t.center.x - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_midpoint_two_keyframes() {
        let mut track = CameraTrack::new();
        track.push(0.0, target(0.0, 4.0));
        track.push(1.0, target(2.0, 8.0));
        // At exactly t=0.5, Catmull-Rom with phantom endpoints on a linear
        // sequence should give the midpoint.
        let t = interpolate_camera(&track, 0.5);
        assert!((t.center.x - 1.0).abs() < 0.05, "center.x={}", t.center.x);
        assert!((t.distance - 6.0).abs() < 0.1, "distance={}", t.distance);
    }

    #[test]
    fn test_keyframe_hit_exact() {
        let mut track = CameraTrack::new();
        track.push(0.0, target(0.0, 1.0));
        track.push(1.0, target(5.0, 3.0));
        track.push(2.0, target(10.0, 5.0));
        // At t=1.0 exactly we should be at the second keyframe.
        let t = interpolate_camera(&track, 1.0);
        assert!((t.center.x - 5.0).abs() < 1e-4, "center.x={}", t.center.x);
        assert!((t.distance - 3.0).abs() < 1e-4, "distance={}", t.distance);
    }

    #[test]
    fn test_from_keyframes_sorts() {
        let kfs = vec![
            (2.0_f64, target(2.0, 2.0)),
            (0.0_f64, target(0.0, 0.0)),
            (1.0_f64, target(1.0, 1.0)),
        ];
        let track = CameraTrack::from_keyframes(kfs);
        assert_eq!(track.keyframes()[0].0, 0.0);
        assert_eq!(track.keyframes()[1].0, 1.0);
        assert_eq!(track.keyframes()[2].0, 2.0);
    }

    #[test]
    fn test_duration() {
        let mut track = CameraTrack::new();
        assert_eq!(track.duration(), 0.0);
        track.push(1.0, target(0.0, 1.0));
        track.push(4.0, target(1.0, 2.0));
        assert!((track.duration() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_push_keeps_sorted() {
        let mut track = CameraTrack::new();
        track.push(3.0, target(3.0, 3.0));
        track.push(1.0, target(1.0, 1.0));
        track.push(2.0, target(2.0, 2.0));
        assert_eq!(track.keyframes()[0].0, 1.0);
        assert_eq!(track.keyframes()[1].0, 2.0);
        assert_eq!(track.keyframes()[2].0, 3.0);
    }
}
