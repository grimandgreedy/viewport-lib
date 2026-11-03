//! The stretch of a pick ray the frame's clip objects leave standing.
//!
//! Every clip shape keeps a convex region (a half-space, or the inside of a
//! box, a sphere or a cylinder) and the shapes combine by AND, so what they
//! keep of a ray is one interval of it, or nothing. The CPU pick casts within
//! that interval, which is how it comes to agree with the drawn frame: a
//! surface the clip cut away is not there to be hit, and the surface behind
//! it is.
//!
//! The objects counted are the ones the shaders test: enabled, clipping
//! geometry, the first six planes and the first [`CLIP_VOLUME_MAX`] volumes.

use crate::renderer::{ClipObject, ClipShape};
use crate::resources::CLIP_VOLUME_MAX;

/// The kept part of the ray `origin + direction * t`, as `[start, end]` in
/// that `t`, starting from `0`. `None` when the clip keeps none of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ClipSpan {
    pub start: f32,
    pub end: f32,
}

impl ClipSpan {
    /// The whole ray: what a frame with no clip objects keeps.
    pub const WHOLE: ClipSpan = ClipSpan {
        start: 0.0,
        end: f32::INFINITY,
    };

    /// What `objects` keep of the ray.
    pub fn of(
        objects: &[ClipObject],
        origin: glam::Vec3,
        direction: glam::Vec3,
    ) -> Option<ClipSpan> {
        let mut span = ClipSpan::WHOLE;
        let mut planes = 0;
        let mut volumes = 0;
        for obj in objects.iter().filter(|o| o.enabled && o.clip_geometry) {
            let kept = match obj.shape {
                ClipShape::Plane {
                    normal, distance, ..
                } if planes < 6 => {
                    planes += 1;
                    half_space(origin, direction, glam::Vec3::from(normal), distance)
                }
                ClipShape::Box {
                    centre,
                    half_extents,
                    orientation,
                } if volumes < CLIP_VOLUME_MAX => {
                    volumes += 1;
                    let d = (origin.as_dvec3() - centre).as_vec3();
                    let axes = orientation.map(glam::Vec3::from);
                    let mut s = ClipSpan::FREE;
                    for (axis, half) in axes.into_iter().zip(half_extents) {
                        s = s.and(slab(d, direction, axis, half)?);
                    }
                    Some(s)
                }
                ClipShape::Sphere { centre, radius } if volumes < CLIP_VOLUME_MAX => {
                    volumes += 1;
                    let d = (origin.as_dvec3() - centre).as_vec3();
                    quadratic(
                        direction.dot(direction),
                        2.0 * d.dot(direction),
                        d.dot(d) - radius * radius,
                    )
                }
                ClipShape::Cylinder {
                    centre,
                    axis,
                    radius,
                    half_length,
                } if volumes < CLIP_VOLUME_MAX => {
                    volumes += 1;
                    let a = glam::Vec3::from(axis);
                    let d = (origin.as_dvec3() - centre).as_vec3();
                    let along = slab(d, direction, a, half_length)?;
                    // The distance from the axis, squared, as a quadratic in t.
                    let dr = d - a * d.dot(a);
                    let vr = direction - a * direction.dot(a);
                    let round = if vr.length_squared() < f32::EPSILON {
                        (dr.length_squared() <= radius * radius).then_some(ClipSpan::FREE)
                    } else {
                        quadratic(vr.dot(vr), 2.0 * dr.dot(vr), dr.dot(dr) - radius * radius)
                    };
                    round.map(|r| r.and(along))
                }
                _ => continue,
            };
            span = span.and(kept?);
            if span.start > span.end {
                return None;
            }
        }
        Some(span)
    }

    /// The whole line, before it is cut to the ray.
    const FREE: ClipSpan = ClipSpan {
        start: f32::NEG_INFINITY,
        end: f32::INFINITY,
    };

    fn and(self, other: ClipSpan) -> ClipSpan {
        ClipSpan {
            start: self.start.max(other.start),
            end: self.end.min(other.end),
        }
    }

    /// Whether this is the whole ray, which is the common case and the one
    /// the pick can take without casting twice.
    pub fn is_whole(&self) -> bool {
        *self == ClipSpan::WHOLE
    }
}

/// Where `dot(p, normal) + distance >= 0` along the line.
fn half_space(
    origin: glam::Vec3,
    direction: glam::Vec3,
    normal: glam::Vec3,
    distance: f64,
) -> Option<ClipSpan> {
    let at = (origin.as_dvec3().dot(normal.as_dvec3()) + distance) as f32;
    let rate = direction.dot(normal);
    if rate.abs() < f32::EPSILON {
        return (at >= 0.0).then_some(ClipSpan::FREE);
    }
    let t = -at / rate;
    Some(if rate > 0.0 {
        ClipSpan {
            start: t,
            end: f32::INFINITY,
        }
    } else {
        ClipSpan {
            start: f32::NEG_INFINITY,
            end: t,
        }
    })
}

/// Where `|dot(p, axis)| <= half` along the line, `p` measured from the
/// shape's centre.
fn slab(
    origin: glam::Vec3,
    direction: glam::Vec3,
    axis: glam::Vec3,
    half: f32,
) -> Option<ClipSpan> {
    let o = origin.dot(axis);
    let d = direction.dot(axis);
    if d.abs() < f32::EPSILON {
        return (o.abs() <= half).then_some(ClipSpan::FREE);
    }
    let (a, b) = ((-half - o) / d, (half - o) / d);
    Some(ClipSpan {
        start: a.min(b),
        end: a.max(b),
    })
}

/// Where `a t^2 + b t + c <= 0`, for `a > 0`.
fn quadratic(a: f32, b: f32, c: f32) -> Option<ClipSpan> {
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    Some(ClipSpan {
        start: (-b - root) / (2.0 * a),
        end: (-b + root) / (2.0 * a),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// A ray from z = 10 straight down.
    fn down() -> (glam::Vec3, glam::Vec3) {
        (
            glam::Vec3::new(0.0, 0.0, 10.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
        )
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn no_objects_keep_the_whole_ray() {
        let (o, d) = down();
        assert!(ClipSpan::of(&[], o, d).unwrap().is_whole());
    }

    #[test]
    fn a_box_keeps_the_ray_between_its_walls() {
        let (o, d) = down();
        let clip = ClipObject::box_shape([0.0, 0.0, 0.0], [1.0, 1.0, 2.0], IDENTITY);
        let s = ClipSpan::of(&[clip], o, d).unwrap();
        assert!(close(s.start, 8.0) && close(s.end, 12.0), "{s:?}");
    }

    #[test]
    fn a_box_beside_the_ray_keeps_none_of_it() {
        let (o, d) = down();
        let clip = ClipObject::box_shape([5.0, 0.0, 0.0], [1.0, 1.0, 1.0], IDENTITY);
        assert_eq!(ClipSpan::of(&[clip], o, d), None);
    }

    #[test]
    fn a_plane_keeps_its_side() {
        let (o, d) = down();
        // Keep z <= 3: normal -z, distance 3.
        let clip = ClipObject {
            shape: ClipShape::Plane {
                normal: [0.0, 0.0, -1.0],
                distance: 3.0,
                cap_colour: None,
            },
            ..Default::default()
        };
        let s = ClipSpan::of(&[clip], o, d).unwrap();
        assert!(close(s.start, 7.0) && s.end.is_infinite(), "{s:?}");
    }

    #[test]
    fn a_sphere_and_a_cylinder_keep_their_insides() {
        let (o, d) = down();
        let s = ClipSpan::of(&[ClipObject::sphere([0.0, 0.0, 0.0], 2.0)], o, d).unwrap();
        assert!(close(s.start, 8.0) && close(s.end, 12.0), "{s:?}");
        let cylinder = ClipObject {
            shape: ClipShape::Cylinder {
                centre: [0.0, 0.0, 0.0].into(),
                axis: [0.0, 0.0, 1.0],
                radius: 1.0,
                half_length: 3.0,
            },
            ..Default::default()
        };
        let s = ClipSpan::of(&[cylinder], o, d).unwrap();
        assert!(close(s.start, 7.0) && close(s.end, 13.0), "{s:?}");
    }

    #[test]
    fn objects_combine_by_and_and_disabled_ones_count_for_nothing() {
        let (o, d) = down();
        let tall = ClipObject::box_shape([0.0, 0.0, 0.0], [1.0, 1.0, 5.0], IDENTITY);
        let ball = ClipObject::sphere([0.0, 0.0, 0.0], 2.0);
        let s = ClipSpan::of(&[tall, ball], o, d).unwrap();
        assert!(close(s.start, 8.0) && close(s.end, 12.0), "{s:?}");
        let off = ClipObject {
            enabled: false,
            ..ClipObject::sphere([9.0, 9.0, 9.0], 0.1)
        };
        assert!(ClipSpan::of(&[off], o, d).unwrap().is_whole());
    }

    #[test]
    fn a_shape_behind_the_eye_keeps_nothing_ahead() {
        let (o, d) = down();
        let clip = ClipObject::sphere([0.0, 0.0, 20.0], 2.0);
        assert_eq!(ClipSpan::of(&[clip], o, d), None);
    }
}
