//! Light indicator emission: the on-screen icon for each scene-graph light,
//! plus the influence-volume wireframe for a selected one.
//!
//! [`build_light_indicators`] walks the scene-graph lights and describes an
//! indicator per light, without deciding what it is drawn as. The caller turns
//! that into whatever it likes;
//! [`LightIndicators::to_mesh_instances`] covers the usual case of a small
//! arrow or sphere per light. Everything carries `settings.pick_id = node_id`
//! so the standard pick and selection-outline machinery applies.

use crate::Colour;
use std::collections::HashMap;

use crate::interaction::select::selection::Selection;
use crate::renderer::PickId;
use crate::renderer::{MeshInstanceItem, PolylineItem, sphere_wireframe_polyline};
use crate::scene::LayerId;
use crate::scene::material::ItemSettings;
use crate::scene::scene::Scene;
use crate::{LightKind, LightSource};

/// World-space half-size used for the on-screen light icon.
const GLYPH_SIZE: f32 = 0.28;

/// One scene-graph light's on-screen indicator, as a description rather than
/// a drawn thing.
///
/// A directional or spot light points somewhere, so its indicator is oriented;
/// a point light is not. The caller decides what shape stands for each.
#[derive(Clone, Copy, Debug)]
pub struct LightMarker {
    /// World-space position of the light.
    pub position: glam::Vec3,
    /// Unit direction the light points, or `None` for a point light. When set,
    /// the indicator should be oriented along it.
    pub direction: Option<glam::Vec3>,
    /// World half-size for the indicator. A fixed world size rather than a
    /// screen-space one, so a light reads as being somewhere in the scene.
    pub size: f32,
    /// The light's own colour, so the indicator identifies which light it is.
    pub colour: crate::Colour,
    /// The light's `NodeId`, so a pick returns the light.
    pub pick_id: PickId,
    /// Whether the consumer's `Selection` holds this light.
    pub selected: bool,
}

/// What [`build_light_indicators`] found: an indicator per visible light, and
/// an influence-volume outline per selected one.
#[derive(Clone, Default)]
pub struct LightIndicators {
    /// One per visible scene-graph light.
    pub markers: Vec<LightMarker>,
    /// Range sphere for a selected point light, cone outline for a selected
    /// spot. Directional lights get none. These need no mesh, so they can be
    /// submitted as they are.
    pub outlines: Vec<PolylineItem>,
}

impl LightIndicators {
    /// Draw each marker as one of two meshes: `directional` for a light that
    /// points somewhere, oriented along its direction, and `point` for one
    /// that does not.
    ///
    /// Both meshes should be unit-sized and centred on the origin, with
    /// `directional` pointing along +Z. `primitives::arrow` and
    /// `primitives::icosphere` are what these were drawn with before.
    ///
    /// Returns up to two items, one batch per mesh, and skips a batch with no
    /// instances.
    pub fn to_mesh_instances(
        &self,
        directional: crate::resources::mesh::mesh_store::MeshId,
        point: crate::resources::mesh::mesh_store::MeshId,
    ) -> Vec<MeshInstanceItem> {
        let mut out = Vec::new();
        for (mesh, oriented) in [(directional, true), (point, false)] {
            let batch: Vec<&LightMarker> = self
                .markers
                .iter()
                .filter(|m| m.direction.is_some() == oriented)
                .collect();
            if batch.is_empty() {
                continue;
            }
            let mut item = MeshInstanceItem::default();
            item.mesh_id = mesh;
            item.transforms = batch
                .iter()
                .map(|m| {
                    let rotation = match m.direction {
                        // The mesh points along +Z, so rotate that onto the
                        // light's direction.
                        Some(d) => glam::Quat::from_rotation_arc(glam::Vec3::Z, d),
                        None => glam::Quat::IDENTITY,
                    };
                    glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(m.size),
                        rotation,
                        m.position,
                    )
                    .to_cols_array_2d()
                })
                .collect();
            item.colours = batch.iter().map(|m| m.colour).collect();
            // An indicator is an affordance: it should neither cast nor
            // receive the light it stands for.
            item.settings.unlit = true;
            item.settings.cast_shadows = false;
            item.settings.receive_shadows = false;
            // One batch, one id. A caller wanting to pick an individual light
            // submits one item per marker instead.
            if let Some(first) = batch.first() {
                item.settings.pick_id = first.pick_id;
                item.settings.selected = batch.iter().any(|m| m.selected);
            }
            out.push(item);
        }
        out
    }
}

/// Walk every scene-graph light and describe its indicator.
///
/// Everything carries `pick_id = node_id`, so picking an indicator returns the
/// light's `NodeId` through the standard pick API, and `selected` mirrors the
/// consumer's `Selection`.
///
/// Layer visibility and node visibility are honoured (matches the same
/// filter applied in [`Scene::collect_lights`]).
pub fn build_light_indicators(scene: &Scene, selection: &Selection) -> LightIndicators {
    let layer_visible: HashMap<LayerId, bool> =
        scene.layers().iter().map(|l| (l.id, l.visible)).collect();

    let mut markers: Vec<LightMarker> = Vec::new();
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

        let colour_rgba = src.colour.with_alpha(1.0).to_linear_rgba();
        let mut settings = ItemSettings::default();
        settings.pick_id = PickId(id);
        settings.selected = is_selected;
        settings.unlit = true;
        settings.cast_shadows = false;
        settings.receive_shadows = false;

        // A light that points somewhere gets an oriented indicator; a point
        // light does not, so it carries no direction.
        let direction = match &src.kind {
            LightKind::Directional { direction } => {
                let d = glam::Vec3::from(*direction);
                let d = if d.length_squared() > 1.0e-12 {
                    d.normalize()
                } else {
                    glam::Vec3::Z
                };
                Some(world.transform_vector3(d).normalize_or_zero())
            }
            LightKind::Point { .. } => None,
            LightKind::Spot { direction, .. } => {
                let d = glam::Vec3::from(*direction);
                let d = if d.length_squared() > 1.0e-12 {
                    d.normalize()
                } else {
                    glam::Vec3::NEG_Z
                };
                Some(world.transform_vector3(d).normalize_or_zero())
            }
            _ => unreachable!("unhandled LightKind variant"),
        };

        markers.push(LightMarker {
            position: translation,
            direction,
            size: GLYPH_SIZE,
            colour: Colour::from_linear_array(colour_rgba),
            pick_id: settings.pick_id,
            selected: settings.selected,
        });

        if is_selected {
            let world_src = resolve_light_for_glyph(src, world);
            let outline_colour = [colour_rgba[0], colour_rgba[1], colour_rgba[2], 0.8];
            match world_src.kind {
                LightKind::Point {
                    position, range, ..
                } => {
                    let mut pl = sphere_wireframe_polyline(
                        position,
                        range,
                        48,
                        Colour::from_linear_array(outline_colour),
                    );
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
                _ => {}
            }
        }
    }

    LightIndicators {
        markers,
        outlines: polylines,
    }
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
        LightKind::Point { range, radius, .. } => LightKind::Point {
            position: translation.into(),
            range: *range,
            radius: *radius,
        },
        LightKind::Spot {
            direction,
            range,
            inner_angle,
            outer_angle,
            radius,
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
                radius: *radius,
            }
        }
        _ => unreachable!("unhandled LightKind variant"),
    };
    let mut out = LightSource::default();
    out.kind = kind;
    out.colour = src.colour;
    out.intensity = src.intensity;
    out.importance = src.importance;
    out.cast_shadows = src.cast_shadows;
    out.channel_mask = src.channel_mask;
    out
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
        glam::Vec3::X
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
        default_colour: Colour::from_linear_array(colour),
        line_width: 1.5,
        ..Default::default()
    }
}
