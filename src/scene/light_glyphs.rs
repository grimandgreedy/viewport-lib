//! Light glyph + influence-volume wireframe emission.
//!
//! [`build_light_glyphs`] walks the scene-graph lights and produces a
//! `GlyphItem` per light (sphere for point, arrow for spot or directional)
//! plus a `PolylineItem` per selected non-directional light showing its
//! influence volume. Both carry `settings.pick_id = node_id` so the
//! standard pick + selection-outline machinery applies.

use std::collections::HashMap;

use crate::interaction::selection::Selection;
use crate::renderer::PickId;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem, sphere_wireframe_polyline};
use crate::scene::material::ItemSettings;
use crate::scene::scene::Scene;
use crate::scene::LayerId;
use crate::{LightKind, LightSource};

/// World-space half-size used for the on-screen light icon.
const GLYPH_SIZE: f32 = 0.28;

/// Walk every scene-graph light and emit:
/// - one `GlyphItem` per light (the on-screen icon, always visible),
/// - one `PolylineItem` per selected non-directional light (range sphere
///   for points; cone outline for spots; directionals get no extra
///   wireframe at this phase).
///
/// All emitted items carry `settings.pick_id = node_id` so clicking the
/// glyph or the wireframe returns the light's `NodeId` through the
/// standard pick API. `settings.selected` mirrors the consumer's
/// `Selection` so the existing outline pass highlights selected lights.
///
/// Layer visibility and node visibility are honoured (matches the same
/// filter applied in [`Scene::collect_lights`]).
pub fn build_light_glyphs(
    scene: &Scene,
    selection: &Selection,
) -> (Vec<GlyphItem>, Vec<PolylineItem>) {
    let layer_visible: HashMap<LayerId, bool> = scene
        .layers()
        .iter()
        .map(|l| (l.id, l.visible))
        .collect();

    let mut glyphs: Vec<GlyphItem> = Vec::new();
    let mut polylines: Vec<PolylineItem> = Vec::new();

    for node in scene.nodes() {
        let Some(src) = node.light.as_ref() else {
            continue;
        };
        if !node.is_visible() {
            continue;
        }
        if !layer_visible.get(&node.layer()).copied().unwrap_or(true) {
            continue;
        }

        let id = node.id();
        let world = node.world_transform();
        let translation = world.col(3).truncate();
        let is_selected = selection.contains(id);

        let colour_rgba = [src.colour[0], src.colour[1], src.colour[2], 1.0];
        let mut settings = ItemSettings::default();
        settings.pick_id = PickId(id);
        settings.selected = is_selected;
        settings.unlit = true;
        settings.cast_shadows = false;
        settings.receive_shadows = false;

        let (glyph_type, vector) = match &src.kind {
            LightKind::Directional { direction } => {
                let d = glam::Vec3::from(*direction);
                let d = if d.length_squared() > 1.0e-12 {
                    d.normalize()
                } else {
                    glam::Vec3::Z
                };
                let v = world.transform_vector3(d).normalize_or_zero();
                (GlyphType::Arrow, v)
            }
            LightKind::Point { .. } => (GlyphType::Sphere, glam::Vec3::Z),
            LightKind::Spot { direction, .. } => {
                let d = glam::Vec3::from(*direction);
                let d = if d.length_squared() > 1.0e-12 {
                    d.normalize()
                } else {
                    glam::Vec3::NEG_Z
                };
                let v = world.transform_vector3(d).normalize_or_zero();
                (GlyphType::Arrow, v)
            }
        };

        let mut g = GlyphItem::default();
        g.glyph_type = glyph_type;
        g.positions.push(translation.into());
        let vec_scaled: [f32; 3] = (vector * GLYPH_SIZE).into();
        g.vectors.push(vec_scaled);
        g.scale = GLYPH_SIZE;
        g.scale_by_magnitude = matches!(glyph_type, GlyphType::Arrow);
        g.use_default_colour = true;
        g.default_colour = colour_rgba;
        g.settings = settings;
        glyphs.push(g);

        if is_selected {
            let world_src = resolve_light_for_glyph(src, world);
            let outline_colour = [
                colour_rgba[0],
                colour_rgba[1],
                colour_rgba[2],
                0.8,
            ];
            match world_src.kind {
                LightKind::Point { position, range } => {
                    let mut pl = sphere_wireframe_polyline(position, range, 48, outline_colour);
                    pl.line_width = 1.5;
                    pl.settings.pick_id = PickId(id);
                    pl.settings.selected = true;
                    pl.settings.unlit = true;
                    polylines.push(pl);
                }
                LightKind::Spot {
                    position,
                    direction,
                    range,
                    outer_angle,
                    ..
                } => {
                    let mut pl = spot_cone_polyline(
                        position,
                        direction,
                        range,
                        outer_angle,
                        24,
                        outline_colour,
                    );
                    pl.settings.pick_id = PickId(id);
                    pl.settings.selected = true;
                    pl.settings.unlit = true;
                    polylines.push(pl);
                }
                LightKind::Directional { .. } => {}
            }
        }
    }

    (glyphs, polylines)
}

/// Same world-space resolution as `Scene::collect_lights` uses for the
/// shading data, hoisted here so the wireframe matches what the shader
/// actually evaluates.
fn resolve_light_for_glyph(src: &LightSource, world: glam::Mat4) -> LightSource {
    let translation = world.col(3).truncate();
    let kind = match &src.kind {
        LightKind::Directional { direction } => {
            let rotated = world
                .transform_vector3(glam::Vec3::from(*direction))
                .normalize_or_zero();
            LightKind::Directional {
                direction: rotated.into(),
            }
        }
        LightKind::Point { range, .. } => LightKind::Point {
            position: translation.into(),
            range: *range,
        },
        LightKind::Spot {
            direction,
            range,
            inner_angle,
            outer_angle,
            ..
        } => {
            let rotated = world
                .transform_vector3(glam::Vec3::from(*direction))
                .normalize_or_zero();
            LightKind::Spot {
                position: translation.into(),
                direction: rotated.into(),
                range: *range,
                inner_angle: *inner_angle,
                outer_angle: *outer_angle,
            }
        }
    };
    LightSource {
        kind,
        colour: src.colour,
        intensity: src.intensity,
        importance: src.importance,
    }
}

/// Build a polyline outline for a spot cone: the rim circle at the cone
/// base plus four meridian lines from apex to the rim.
fn spot_cone_polyline(
    apex: [f32; 3],
    direction: [f32; 3],
    range: f32,
    outer_angle: f32,
    segments: u32,
    colour: [f32; 4],
) -> PolylineItem {
    let n = segments.max(8) as usize;
    let apex_v = glam::Vec3::from(apex);
    let dir = glam::Vec3::from(direction).normalize_or_zero();
    let dir = if dir.length_squared() > 1.0e-8 {
        dir
    } else {
        glam::Vec3::NEG_Z
    };
    let up_ref = if dir.z.abs() > 0.95 {
        glam::Vec3::Y
    } else {
        glam::Vec3::Z
    };
    let right = dir.cross(up_ref).normalize_or_zero();
    let up = right.cross(dir).normalize_or_zero();

    let rim_radius = range * outer_angle.sin();
    let rim_center = apex_v + dir * range * outer_angle.cos();

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n + 1 + 4 * 2);
    let mut strips: Vec<u32> = Vec::with_capacity(5);

    for i in 0..=n {
        let t = i as f32 / n as f32 * std::f32::consts::TAU;
        let p = rim_center + right * (rim_radius * t.cos()) + up * (rim_radius * t.sin());
        positions.push(p.into());
    }
    strips.push((n + 1) as u32);

    for k in 0..4 {
        let t = k as f32 / 4.0 * std::f32::consts::TAU;
        let rim = rim_center + right * (rim_radius * t.cos()) + up * (rim_radius * t.sin());
        positions.push(apex);
        positions.push(rim.into());
        strips.push(2);
    }

    PolylineItem {
        positions,
        strip_lengths: strips,
        default_colour: colour,
        line_width: 1.5,
        ..Default::default()
    }
}
