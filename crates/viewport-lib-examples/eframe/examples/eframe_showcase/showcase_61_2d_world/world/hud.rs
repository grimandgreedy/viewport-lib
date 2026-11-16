//! The screen overlays: the HUD, the name tag, the damage numbers and the
//! minimap.

use super::maps::torch_flame;
use super::{
    CELL, COLS, DAMAGE_LIFE, DIRT, GRASS, PLANK, PLAYER_H, STONE, TORCHES, World, WorldStats,
};
use glam::Vec2;
use viewport_lib as vpl;
use viewport_lib_plugins::item_types::tilemap::{Tile, Tilemap};
use vpl::{
    Alignment, AnchorX, AnchorY, Colour, FontHandle, LabelItem, OverlayFill, OverlayFrame,
    OverlayMeshItem, OverlayShape, OverlayShapeItem, OverlayStroke, ShadowLayer, WorldPlane,
    WorldTextSize,
};

impl World {
    /// The screen overlays: the HUD, the name tag and the damage numbers.
    /// `hud_font` draws the HP and score labels in a bitmap face.
    pub(super) fn overlays(
        &self,
        stats: &WorldStats,
        hud_font: Option<FontHandle>,
    ) -> OverlayFrame {
        let mut ovl = OverlayFrame::default();
        ovl.time = self.time as f64;
        let p = &self.player;
        // World-sized, but never smaller than 12 or larger than 28 pixels.
        ovl.labels.push(
            LabelItem::new("Wren")
                .with_world_anchor([p.position.x, -3.0, p.position.y + PLAYER_H + 5.0])
                .with_align(Alignment::new(AnchorX::Middle, AnchorY::Baseline))
                .with_world_size(
                    5.0,
                    WorldTextSize::new(WorldPlane::Xz)
                        .with_min_px(12.0)
                        .with_max_px(28.0),
                )
                .with_colour(Colour::srgb(0.88, 0.96, 1.0, 1.0))
                .with_shadow(text_outline(0.5)),
        );
        for d in &self.damage {
            let t = d.age / DAMAGE_LIFE;
            ovl.labels.push(
                LabelItem::new(d.amount.to_string())
                    .with_world_anchor([d.at.x, -3.0, d.at.y + 4.0 + 12.0 * t])
                    .with_align(Alignment::new(AnchorX::Middle, AnchorY::Baseline))
                    .with_world_size(7.0, WorldTextSize::new(WorldPlane::Xz))
                    .with_colour(d.colour)
                    .with_shadow(text_outline(0.7))
                    .with_opacity((1.0 - t * t).clamp(0.0, 1.0)),
            );
        }
        if self.options.hud {
            self.hud(&mut ovl, stats, hud_font);
        }
        ovl
    }

    /// The HUD, laid out in the window's logical pixels: screen overlays draw
    /// over the scaled scene at the window's resolution under either
    /// presentation.
    fn hud(&self, ovl: &mut OverlayFrame, stats: &WorldStats, hud_font: Option<FontHandle>) {
        let size = self.window_size;
        // A bitmap face draws at whole multiples of its 8-pixel size.
        let face = |label: LabelItem| match hud_font {
            Some(font) => label.with_font(font),
            None => label,
        };
        let panel = Colour::srgb(0.05, 0.06, 0.1, 0.72);
        let rim = OverlayStroke::new(1.0, Colour::srgb(0.75, 0.8, 0.95, 0.6));

        // Health: a panel, a track and a bar, all shapes.
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 8.0 },
                [16.0, 16.0],
                [232.0, 36.0],
            )
            .with_fill(OverlayFill::Solid(panel))
            .with_stroke(rim.clone()),
        );
        ovl.labels.push(face(
            LabelItem::new("HP")
                .with_position([28.0, 34.0])
                .with_align_y(AnchorY::Middle)
                .with_font_size(14.0)
                .with_colour(Colour::srgb(1.0, 0.85, 0.8, 1.0))
                .with_z_order(2),
        ));
        let frac = self.player.health / 100.0;
        let (bx, by, bw, bh) = (56.0, 27.0, 180.0, 14.0);
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 4.0 },
                [bx, by],
                [bw, bh],
            )
            .with_fill(OverlayFill::Solid(Colour::srgb(0.25, 0.05, 0.06, 1.0)))
            .with_z_order(1),
        );
        if frac > 0.0 {
            ovl.shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 4.0 },
                    [bx, by],
                    [bw * frac, bh],
                )
                .with_fill(OverlayFill::linear_gradient(
                    Colour::srgb(0.85, 0.12, 0.12, 1.0),
                    Colour::srgb(1.0, 0.62, 0.2, 1.0),
                    0.0,
                ))
                .with_z_order(2),
            );
        }

        // Score, top right.
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 8.0 },
                [size.x - 16.0 - 252.0, 16.0],
                [252.0, 30.0],
            )
            .with_fill(OverlayFill::Solid(panel))
            .with_stroke(rim.clone()),
        );
        ovl.labels.push(face(
            LabelItem::new(format!("SCORE {:06}", self.score))
                .with_position([size.x - 28.0, 31.0])
                .with_align(Alignment::new(AnchorX::Right, AnchorY::Middle))
                .with_font_size(16.0)
                .with_colour(Colour::srgb(1.0, 0.9, 0.45, 1.0))
                .with_z_order(2),
        ));

        // The minimap: the map as one mesh, the view as a stroked rect, and
        // the player and torches as dots.
        let (mx, my) = (size.x - 16.0 - 252.0, 54.0);
        let (mw, mh) = (
            MINIMAP_COLS as f32 * MINIMAP_SCALE,
            MINIMAP_ROWS as f32 * MINIMAP_SCALE,
        );
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 6.0 },
                [mx, my],
                [252.0, mh + 12.0],
            )
            .with_fill(OverlayFill::Solid(panel))
            .with_stroke(rim.clone()),
        );
        let (ox, oy) = (mx + (252.0 - mw) * 0.5, my + 6.0);
        ovl.meshes
            .push(self.minimap.clone().with_position([ox, oy]).with_z_order(1));
        let to_map = |p: Vec2| {
            [
                ox + p.x / CELL * MINIMAP_SCALE,
                oy + (MINIMAP_ROWS as f32 - p.y / CELL) * MINIMAP_SCALE,
            ]
        };
        let view = self.camera.visible_rect();
        let (a, b) = (
            to_map(Vec2::new(view.min.x, view.max.y)),
            to_map(Vec2::new(view.max.x, view.min.y)),
        );
        let (a, b) = (
            [a[0].max(ox), a[1].max(oy)],
            [b[0].min(ox + mw), b[1].min(oy + mh)],
        );
        if b[0] > a[0] && b[1] > a[1] {
            ovl.shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 1.0 },
                    a,
                    [b[0] - a[0], b[1] - a[1]],
                )
                .with_fill(OverlayFill::Solid(Colour::srgb(1.0, 1.0, 1.0, 0.08)))
                .with_stroke(OverlayStroke::new(1.0, Colour::srgb(1.0, 1.0, 1.0, 0.85)))
                .with_z_order(3),
            );
        }
        for &(col, row) in &TORCHES {
            let c = to_map(torch_flame(col, row));
            ovl.shapes.push(
                OverlayShapeItem::new(OverlayShape::Circle, [c[0] - 1.5, c[1] - 1.5], [3.0, 3.0])
                    .with_fill(OverlayFill::Solid(Colour::srgb(1.0, 0.6, 0.2, 1.0)))
                    .with_z_order(4),
            );
        }
        let c = to_map(self.player.position + Vec2::new(0.0, 8.0));
        ovl.shapes.push(
            OverlayShapeItem::new(OverlayShape::Circle, [c[0] - 2.5, c[1] - 2.5], [5.0, 5.0])
                .with_fill(OverlayFill::Solid(Colour::srgb(0.4, 0.75, 1.0, 1.0)))
                .with_stroke(OverlayStroke::new(1.0, Colour::WHITE))
                .with_z_order(5),
        );

        // The previous frame's counters.
        let s = &stats.frame;
        let text = format!(
            "draw calls {} (plugin {})   overlay draws {}   layers {}   glyph rasters {}   vector rasters {}   2D lights {}  edges {}",
            s.draw_calls,
            s.plugin_draw_calls,
            s.overlay_draw_calls,
            s.overlay_layers_composited,
            s.overlay_glyph_rasters,
            s.overlay_vector_rasters,
            stats.lights.lights,
            stats.lights.edges,
        );
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 6.0 },
                [10.0, size.y - 32.0],
                [(size.x - 20.0).min(760.0), 22.0],
            )
            .with_fill(OverlayFill::Solid(panel)),
        );
        ovl.labels.push(
            LabelItem::new(text)
                .with_position([18.0, size.y - 27.0])
                .with_align_y(AnchorY::Top)
                .with_font_size(12.0)
                .with_colour(Colour::srgb(0.85, 0.95, 0.85, 1.0))
                .with_z_order(1),
        );
    }
}

/// Rows of the map the minimap shows, from the bottom.
const MINIMAP_ROWS: u32 = 16;
const MINIMAP_COLS: u32 = COLS;
/// Logical pixels per map cell on the minimap.
const MINIMAP_SCALE: f32 = 1.5;

/// The lower rows of the map as one overlay mesh: a quad per run of like
/// cells along a row, coloured by tile.
pub(super) fn minimap_mesh(map: &Tilemap) -> OverlayMeshItem {
    let colour = |t: u32| match t {
        GRASS => Some(Colour::srgb(0.3, 0.68, 0.3, 1.0)),
        DIRT => Some(Colour::srgb(0.45, 0.32, 0.2, 1.0)),
        STONE => Some(Colour::srgb(0.62, 0.62, 0.7, 1.0)),
        PLANK => Some(Colour::srgb(0.75, 0.52, 0.3, 1.0)),
        _ => None,
    };
    let mut positions = Vec::new();
    let mut colours = Vec::new();
    let mut indices = Vec::new();
    for row in 0..MINIMAP_ROWS {
        let mut col = 0;
        while col < MINIMAP_COLS {
            let tile = map
                .get(col as i32, row as i32)
                .map_or(Tile::EMPTY_INDEX, |t| t.index);
            let start = col;
            while col < MINIMAP_COLS
                && map
                    .get(col as i32, row as i32)
                    .map_or(Tile::EMPTY_INDEX, |t| t.index)
                    == tile
            {
                col += 1;
            }
            let Some(c) = colour(tile) else {
                continue;
            };
            let s = MINIMAP_SCALE;
            let (x0, x1) = (start as f32 * s, col as f32 * s);
            let (y0, y1) = (
                (MINIMAP_ROWS - row - 1) as f32 * s,
                (MINIMAP_ROWS - row) as f32 * s,
            );
            let base = positions.len() as u32;
            positions.extend_from_slice(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
            colours.extend_from_slice(&[c; 4]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    OverlayMeshItem::new(positions, indices).with_colours(colours)
}

/// A thin dark outline behind world-sized text, in its own units.
fn text_outline(spread: f32) -> ShadowLayer {
    ShadowLayer::new(Colour::srgb(0.05, 0.03, 0.02, 1.0), 0.0, [0.0, 0.0]).with_spread(spread)
}
