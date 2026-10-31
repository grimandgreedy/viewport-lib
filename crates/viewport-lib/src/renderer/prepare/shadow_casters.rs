//! Which lights cast shadows this frame: the cascade caster and the point and
//! spot lights that get layers of the punctual shadow array.

use crate::renderer::punctual_shadow_pool::{SLOT_LAYERS, ShadowKey};
use crate::renderer::types::{LightKind, LightSource};

/// Score multiplier a light already holding layers gets against a newcomer,
/// so two lights of near-equal rank do not swap their shadows every frame.
const INCUMBENT_BONUS: f32 = 1.25;

/// A point or spot light chosen to cast this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PunctualCaster {
    /// Index into the frame's ranked light list.
    pub(super) light: usize,
    pub(super) key: ShadowKey,
    /// A point light, which takes six layers; a spot takes one.
    pub(super) cube: bool,
}

/// The lights that cast shadows this frame.
#[derive(Debug, Default)]
pub(super) struct ShadowCasters {
    /// Index into the ranked light list of the light that takes the cascades:
    /// the first directional light with `cast_shadows` set.
    pub(super) cascades: Option<usize>,
    /// Point and spot lights given punctual layers, in descending score.
    pub(super) punctual: Vec<PunctualCaster>,
}

/// Choose this frame's shadow casters from `lights`, the ranked list.
///
/// Each casting point light costs six layers and each spot one. They are
/// taken in descending `importance * range / (range + d)`, `d` the distance
/// from `eye`, until `budget_layers` is spent; a point that no longer fits is
/// skipped and a later spot can still take a layer. A light for which
/// `incumbent` is true scores [`INCUMBENT_BONUS`] times higher.
pub(super) fn select_shadow_casters(
    lights: &[(usize, &LightSource)],
    key_of: impl Fn(usize) -> ShadowKey,
    eye: glam::Vec3,
    budget_layers: u32,
    incumbent: impl Fn(ShadowKey) -> bool,
) -> ShadowCasters {
    let cascades = lights
        .iter()
        .position(|(_, l)| l.cast_shadows && matches!(l.kind, LightKind::Directional { .. }));

    let mut candidates: Vec<(f32, PunctualCaster)> = lights
        .iter()
        .enumerate()
        .filter(|(_, (_, l))| l.cast_shadows)
        .filter_map(|(i, &(source, l))| {
            let (position, range, cube) = match l.kind {
                LightKind::Point {
                    position, range, ..
                } => (position, range, true),
                LightKind::Spot {
                    position,
                    range,
                    direction,
                    ..
                } => {
                    // A spot with no direction has no frustum to render.
                    if glam::Vec3::from(direction).length_squared() < 1e-8 {
                        return None;
                    }
                    (position, range, false)
                }
                _ => return None,
            };
            let key = key_of(source);
            let d = (position.as_vec3() - eye).length();
            let proximity = (range / (range + d)).clamp(0.0, 1.0);
            let mut score = l.importance.max(0.0) * proximity;
            if incumbent(key) {
                score *= INCUMBENT_BONUS;
            }
            Some((
                score,
                PunctualCaster {
                    light: i,
                    key,
                    cube,
                },
            ))
        })
        .collect();
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut left = budget_layers;
    let mut punctual = Vec::new();
    for (_, c) in candidates {
        let cost = if c.cube { SLOT_LAYERS } else { 1 };
        if cost <= left {
            left -= cost;
            punctual.push(c);
        }
    }
    ShadowCasters { cascades, punctual }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::types::Candela;

    fn point(x: f32) -> LightSource {
        LightSource::point_candela([x, 0.0, 0.0], Candela(10.0), 10.0, 0.0)
    }

    fn spot(x: f32) -> LightSource {
        LightSource::spot_candela(
            [x, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            Candela(10.0),
            10.0,
            0.3,
            0.5,
            0.0,
        )
    }

    fn key(index: usize) -> ShadowKey {
        ShadowKey {
            source: 0,
            index: index as u32,
        }
    }

    fn select(lights: &[LightSource], budget: u32, held: &[usize]) -> ShadowCasters {
        let ranked: Vec<(usize, &LightSource)> = lights.iter().enumerate().collect();
        select_shadow_casters(&ranked, key, glam::Vec3::ZERO, budget, |k| {
            held.iter().any(|&h| key(h) == k)
        })
    }

    #[test]
    fn the_first_casting_directional_takes_the_cascades() {
        let mut quiet = LightSource::default();
        quiet.cast_shadows = false;
        let lights = [quiet, point(1.0), LightSource::default()];
        assert_eq!(select(&lights, 48, &[]).cascades, Some(2));
        assert_eq!(select(&[point(1.0)], 48, &[]).cascades, None);
    }

    #[test]
    fn points_cost_six_layers_and_spots_one() {
        let lights = [point(1.0), point(2.0), spot(3.0), spot(4.0)];
        let picked = select(&lights, 8, &[]).punctual;
        // The second point does not fit after the first; both spots do.
        let lit: Vec<usize> = picked.iter().map(|c| c.light).collect();
        assert_eq!(lit, vec![0, 2, 3]);
        let picked = select(&lights, 14, &[]).punctual;
        assert_eq!(picked.len(), 4);
    }

    #[test]
    fn a_light_that_does_not_cast_is_skipped() {
        let mut quiet = point(1.0);
        quiet.cast_shadows = false;
        let picked = select(&[quiet, spot(2.0)], 48, &[]).punctual;
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].light, 1);
    }

    #[test]
    fn an_incumbent_keeps_its_layers_against_a_slightly_closer_light() {
        // Light 1 is a little nearer the eye, but light 0 already holds the
        // only slot.
        let lights = [point(1.2), point(1.0)];
        let picked = select(&lights, 6, &[0]).punctual;
        assert_eq!(picked[0].light, 0);
        // Without the slot, the nearer light wins.
        let picked = select(&lights, 6, &[]).punctual;
        assert_eq!(picked[0].light, 1);
    }
}
