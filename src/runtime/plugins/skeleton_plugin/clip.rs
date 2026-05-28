//! Animation clip data model and sampling.
//!
//! A [`Sampler`] holds keyframe times paired with values and produces a
//! [`TrackValue`] at any time `t`. A [`Track`] binds a sampler to one channel
//! (translation, rotation, or scale) on one joint. An [`AnimationClip`] is a
//! collection of tracks plus a duration; [`AnimationClip::sample_into`]
//! applies all tracks to a [`Pose`] at time `t`.
//!
//! # Semantics
//!
//! `sample_into` only touches joints that appear in the clip's tracks. For each
//! such joint, the local transform is decomposed into translation/rotation/
//! scale, the channels named by the clip's tracks are overwritten with sampled
//! values, untouched channels keep their existing values, and the result is
//! recomposed. Joints not mentioned by any track are left alone. This matches
//! the glTF animation model and lets a clip carry only the channels it needs
//! while a bind pose supplies the rest.
//!
//! # Out of scope
//!
//! - Cubic-spline interpolation (not yet implemented).
//! - Per-track weight blending via `BlendNode` (not yet implemented).

use super::skeleton::Pose;

/// Which component of a joint's local transform a track drives.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Channel {
    /// Drives the joint's local translation (Vec3 sampler values).
    Translation,
    /// Drives the joint's local rotation (Quat sampler values).
    Rotation,
    /// Drives the joint's local scale (Vec3 sampler values).
    Scale,
}

/// How a sampler blends between adjacent keyframes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Interpolation {
    /// Hold the value of the lower keyframe until the next one starts.
    Step,
    /// Vec3 channels use componentwise lerp; Quat channels use slerp.
    Linear,
}

/// Sampled value at a point in time. Variant must match the parent track's
/// channel: `Translation`/`Scale` produce `Vec3`, `Rotation` produces `Quat`.
#[derive(Copy, Clone, Debug)]
pub enum TrackValue {
    /// Translation or scale sample.
    Vec3(glam::Vec3),
    /// Rotation sample.
    Quat(glam::Quat),
}

/// Per-keyframe values for a track. The variant must match the channel of any
/// track using this sampler.
#[derive(Clone, Debug)]
pub enum TrackValues {
    /// Keyframe values for translation or scale tracks.
    Vec3(Vec<glam::Vec3>),
    /// Keyframe values for rotation tracks.
    Quat(Vec<glam::Quat>),
}

impl TrackValues {
    fn len(&self) -> usize {
        match self {
            TrackValues::Vec3(v) => v.len(),
            TrackValues::Quat(v) => v.len(),
        }
    }
}

/// Keyframe times paired with values plus an interpolation mode.
///
/// `times` must be non-empty, strictly increasing, and the same length as the
/// inner `values` vector.
#[derive(Clone, Debug)]
pub struct Sampler {
    /// Interpolation mode between adjacent keyframes.
    pub interpolation: Interpolation,
    /// Keyframe times in seconds. Must be non-empty and strictly increasing.
    pub times: Vec<f32>,
    /// Keyframe values, same length as `times`.
    pub values: TrackValues,
}

impl Sampler {
    /// Sample at time `t`. Times outside `[times.first(), times.last()]` clamp
    /// to the nearest endpoint.
    pub fn sample(&self, t: f32) -> TrackValue {
        debug_assert!(!self.times.is_empty(), "sampler has no keyframes");
        debug_assert_eq!(self.times.len(), self.values.len(), "times/values length mismatch");

        let n = self.times.len();
        if t <= self.times[0] {
            return self.value_at(0);
        }
        if t >= self.times[n - 1] {
            return self.value_at(n - 1);
        }

        // Segment: times[i] <= t < times[i+1].
        let i = self.times.partition_point(|&x| x <= t).saturating_sub(1);
        let j = i + 1;
        match self.interpolation {
            Interpolation::Step => self.value_at(i),
            Interpolation::Linear => {
                let t0 = self.times[i];
                let t1 = self.times[j];
                let alpha = (t - t0) / (t1 - t0);
                self.lerp(i, j, alpha)
            }
        }
    }

    fn value_at(&self, i: usize) -> TrackValue {
        match &self.values {
            TrackValues::Vec3(v) => TrackValue::Vec3(v[i]),
            TrackValues::Quat(v) => TrackValue::Quat(v[i]),
        }
    }

    fn lerp(&self, a: usize, b: usize, alpha: f32) -> TrackValue {
        match &self.values {
            TrackValues::Vec3(v) => TrackValue::Vec3(v[a].lerp(v[b], alpha)),
            TrackValues::Quat(v) => TrackValue::Quat(v[a].slerp(v[b], alpha)),
        }
    }
}

/// One animation track: a sampler driving one channel on one joint.
#[derive(Clone, Debug)]
pub struct Track {
    /// Index into the target skeleton's joint list.
    pub joint: usize,
    /// Which component of the joint's local transform this track drives.
    pub channel: Channel,
    /// Keyframe sampler producing values for this channel.
    pub sampler: Sampler,
}

/// A collection of tracks that together animate one or more joints over time.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    /// Length of the clip in seconds. Used by the player to loop the playhead.
    pub duration: f32,
    /// Per-channel tracks. Multiple tracks may target the same joint.
    pub tracks: Vec<Track>,
}

impl AnimationClip {
    /// Apply all tracks to `pose` at time `t`.
    ///
    /// For each joint mentioned by any track, the joint's existing local
    /// transform is decomposed into scale/rotation/translation; sampled values
    /// from this clip's tracks overwrite the matching channels; the untouched
    /// channels keep their existing values; and the joint is recomposed as
    /// `T * R * S`.
    ///
    /// Joints not mentioned by any track are left unchanged. Tracks naming a
    /// joint index outside `pose.local_transforms` are silently skipped.
    pub fn sample_into(&self, t: f32, pose: &mut Pose) {
        // Sample once per track, grouped by joint. Three Option slots per joint
        // hold an optional sampled value for translation, rotation, scale.
        let mut affected: Vec<(usize, [Option<TrackValue>; 3])> = Vec::new();

        for track in &self.tracks {
            if track.joint >= pose.local_transforms.len() {
                continue;
            }
            let v = track.sampler.sample(t);
            let slot = channel_slot(track.channel);

            if let Some((_, channels)) = affected.iter_mut().find(|(j, _)| *j == track.joint) {
                channels[slot] = Some(v);
            } else {
                let mut channels = [None, None, None];
                channels[slot] = Some(v);
                affected.push((track.joint, channels));
            }
        }

        for (joint_idx, channels) in affected {
            let local = &mut pose.local_transforms[joint_idx];
            let (s_old, r_old, t_old) = local.to_scale_rotation_translation();
            let t_new = match channels[0] {
                Some(TrackValue::Vec3(v)) => v,
                _ => t_old,
            };
            let r_new = match channels[1] {
                Some(TrackValue::Quat(q)) => q,
                _ => r_old,
            };
            let s_new = match channels[2] {
                Some(TrackValue::Vec3(v)) => v,
                _ => s_old,
            };
            *local = glam::Affine3A::from_scale_rotation_translation(s_new, r_new, t_new);
        }
    }
}

fn channel_slot(c: Channel) -> usize {
    match c {
        Channel::Translation => 0,
        Channel::Rotation => 1,
        Channel::Scale => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn approx_eq_vec3(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).length() < eps
    }

    fn approx_eq_quat(a: Quat, b: Quat, eps: f32) -> bool {
        // Quaternions are equivalent up to sign.
        (a.dot(b).abs() - 1.0).abs() < eps
    }

    #[test]
    fn linear_sampler_lerps_vec3_between_keyframes() {
        let s = Sampler {
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0, 2.0],
            values: TrackValues::Vec3(vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO]),
        };
        match s.sample(0.5) {
            TrackValue::Vec3(v) => assert!(approx_eq_vec3(v, Vec3::new(1.0, 0.0, 0.0), 1e-5)),
            _ => panic!("wrong variant"),
        }
        match s.sample(1.75) {
            TrackValue::Vec3(v) => assert!(approx_eq_vec3(v, Vec3::new(0.5, 0.0, 0.0), 1e-5)),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn step_sampler_holds_lower_keyframe() {
        let s = Sampler {
            interpolation: Interpolation::Step,
            times: vec![0.0, 1.0, 2.0],
            values: TrackValues::Vec3(vec![Vec3::ZERO, Vec3::X, Vec3::Y]),
        };
        // Anywhere in [0, 1) yields the value at index 0.
        match s.sample(0.999) {
            TrackValue::Vec3(v) => assert_eq!(v, Vec3::ZERO),
            _ => panic!(),
        }
        // Exactly at a keyframe yields that keyframe.
        match s.sample(1.0) {
            TrackValue::Vec3(v) => assert_eq!(v, Vec3::X),
            _ => panic!(),
        }
        match s.sample(1.5) {
            TrackValue::Vec3(v) => assert_eq!(v, Vec3::X),
            _ => panic!(),
        }
    }

    #[test]
    fn sampler_clamps_outside_range() {
        let s = Sampler {
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0],
            values: TrackValues::Vec3(vec![Vec3::ZERO, Vec3::X]),
        };
        match s.sample(-5.0) {
            TrackValue::Vec3(v) => assert_eq!(v, Vec3::ZERO),
            _ => panic!(),
        }
        match s.sample(100.0) {
            TrackValue::Vec3(v) => assert_eq!(v, Vec3::X),
            _ => panic!(),
        }
    }

    #[test]
    fn quat_sampler_slerps_between_keyframes() {
        let s = Sampler {
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0],
            values: TrackValues::Quat(vec![
                Quat::IDENTITY,
                Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            ]),
        };
        match s.sample(0.5) {
            TrackValue::Quat(q) => {
                let expected = Quat::from_rotation_x(std::f32::consts::FRAC_PI_4);
                assert!(approx_eq_quat(q, expected, 1e-4), "got {q:?}");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn sample_into_overwrites_only_animated_channels() {
        // Bind pose: joint 1 sits at (0, 0, 2) with no rotation.
        let mut pose = Pose::identity(2);
        pose.local_transforms[1] =
            glam::Affine3A::from_translation(Vec3::new(0.0, 0.0, 2.0));

        // Clip: rotate joint 1 only. Translation should be preserved from the
        // bind pose because the clip has no translation track.
        let clip = AnimationClip {
            duration: 1.0,
            tracks: vec![Track {
                joint: 1,
                channel: Channel::Rotation,
                sampler: Sampler {
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: TrackValues::Quat(vec![
                        Quat::IDENTITY,
                        Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    ]),
                },
            }],
        };

        clip.sample_into(0.5, &mut pose);

        let (scale, rot, trans) = pose.local_transforms[1].to_scale_rotation_translation();
        // Translation preserved from bind pose.
        assert!(approx_eq_vec3(trans, Vec3::new(0.0, 0.0, 2.0), 1e-4));
        // Rotation taken from the clip at t=0.5 (45 deg around X).
        let expected_rot = Quat::from_rotation_x(std::f32::consts::FRAC_PI_4);
        assert!(approx_eq_quat(rot, expected_rot, 1e-4));
        // Scale untouched -> still 1.
        assert!(approx_eq_vec3(scale, Vec3::ONE, 1e-4));
    }

    #[test]
    fn sample_into_two_tracks_compose_correctly() {
        // Joint 0 bind pose is identity. Animate both translation and scale.
        let mut pose = Pose::identity(1);
        let clip = AnimationClip {
            duration: 2.0,
            tracks: vec![
                Track {
                    joint: 0,
                    channel: Channel::Translation,
                    sampler: Sampler {
                        interpolation: Interpolation::Linear,
                        times: vec![0.0, 2.0],
                        values: TrackValues::Vec3(vec![Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0)]),
                    },
                },
                Track {
                    joint: 0,
                    channel: Channel::Scale,
                    sampler: Sampler {
                        interpolation: Interpolation::Step,
                        times: vec![0.0, 1.0],
                        values: TrackValues::Vec3(vec![Vec3::ONE, Vec3::splat(2.0)]),
                    },
                },
            ],
        };

        clip.sample_into(1.5, &mut pose);

        let (scale, _rot, trans) = pose.local_transforms[0].to_scale_rotation_translation();
        assert!(approx_eq_vec3(trans, Vec3::new(3.0, 0.0, 0.0), 1e-4));
        assert!(approx_eq_vec3(scale, Vec3::splat(2.0), 1e-4));
    }

    #[test]
    fn out_of_range_joint_index_is_skipped() {
        let mut pose = Pose::identity(2);
        let clip = AnimationClip {
            duration: 1.0,
            tracks: vec![Track {
                joint: 99,
                channel: Channel::Translation,
                sampler: Sampler {
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: TrackValues::Vec3(vec![Vec3::new(1.0, 2.0, 3.0)]),
                },
            }],
        };
        clip.sample_into(0.5, &mut pose);
        // Pose untouched.
        assert_eq!(pose.local_transforms[0], glam::Affine3A::IDENTITY);
        assert_eq!(pose.local_transforms[1], glam::Affine3A::IDENTITY);
    }
}
