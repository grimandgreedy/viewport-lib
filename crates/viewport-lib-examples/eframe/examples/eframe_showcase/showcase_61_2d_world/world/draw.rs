//! Writing the world into a frame: the items, the lights and the world
//! overlay layers.

use super::art::cell_uv;
use super::maps::torch_flame;
use super::{
    CELL, GROUND_Z, GUARD_Z, ISO_LAYER, ISO_TILE, REFERENCE_SIZE, SIDE_LAYER, STONES, TORCHES,
    World, WorldAssets, WorldOptions, WorldStats,
};
use glam::{DVec3, Mat4, Vec2, Vec3};
use viewport_lib as vpl;
use viewport_lib_plugins::item_types::gpu_particles::{
    ForceField, GpuParticleSystemItem, SpawnShape, VelocityDist,
};
use viewport_lib_plugins::item_types::lighting_2d::{
    Light2d, Lighting2dItem, occluders_from_sprites,
};
use viewport_lib_plugins::item_types::sprite::{
    SpriteEffects, SpriteFlip, SpriteItem, SpriteOrder, SpriteOutline, SpriteSizeMode,
};
use viewport_lib_plugins::item_types::tilemap::{
    TilemapDraw, TilemapId, TilemapItem, TilemapPainter, TilemapPlane,
};
use vpl::{
    CameraFrame, Candela, Colour, FillRule, FrameData, LabelItem, LightSource, LineCap, MeshId,
    MeshInstanceItem, OverlayBlend, OverlayFill, OverlayFrame, OverlayPolylineItem, OverlayShape,
    OverlayShapeItem, OverlayStroke, Presentation, ReferenceFit, SpriteBlend, SubPath,
    TextureFilter, TextureId, WorldOverlayLayer, WorldPlane, painter_key,
};

impl World {
    /// The camera this frame draws through. The frame's viewport is the
    /// window's even under the reference presentation; the camera's aspect is
    /// the size the scene draws at.
    pub(crate) fn camera_frame(&self) -> CameraFrame {
        let size = self.window_size;
        CameraFrame::from_camera(&self.camera.camera(), [size.x, size.y])
            .with_pixels_per_point(self.pixels_per_point)
    }

    /// Write this frame of the world into `fd`: camera, items, lights,
    /// overlays and effects.
    pub(crate) fn fill_frame(&self, fd: &mut FrameData, assets: &WorldAssets, stats: &WorldStats) {
        let o = self.options;
        fd.camera = self.camera_frame();
        fd.viewport.show_axes_indicator = false;
        fd.viewport.show_grid = false;
        if o.reference {
            fd.viewport.presentation = Presentation::reference(
                REFERENCE_SIZE,
                ReferenceFit::IntegerBars {
                    colour: Colour::srgb(0.03, 0.03, 0.05, 1.0),
                },
            );
        }
        fd.viewport.background_colour = Some(if o.night {
            Colour::srgb(0.02, 0.03, 0.08, 1.0)
        } else {
            Colour::srgb(0.48, 0.68, 0.92, 1.0)
        });
        // The light layer runs on either path; bloom, the vignette and the
        // GPU particle torch flames on the HDR path only.
        fd.effects.display.mode = if o.hdr {
            vpl::PipelineMode::Hdr
        } else {
            vpl::PipelineMode::Direct
        };
        fd.effects.post_process.bloom.enabled = o.bloom;
        fd.effects.post_process.bloom.threshold = 1.0;
        fd.effects.post_process.bloom.intensity = 0.12;
        fd.effects.post_process.vignette.enabled = o.vignette;
        fd.effects.post_process.vignette.amount = 0.55;

        let which = if o.pixel_art { 0 } else { 1 };
        let mut far = TilemapItem::new(assets.hills[which]);
        far.plane = TilemapPlane::Xz;
        far.cell_size = [CELL * 2.0, CELL * 2.0];
        far.origin = [-150.0, 0.0];
        // Lag the camera across the screen but keep step with it vertically,
        // so the hills stay level with the ground under a pan or a zoom.
        far.parallax = [0.4, 1.0];
        far.elevation = -1.0;
        far.pixel_snap = o.pixel_art;
        let mut near = TilemapItem::new(assets.ground[which]);
        near.plane = TilemapPlane::Xz;
        near.cell_size = [CELL, CELL];
        near.time = self.time;
        near.pixel_snap = o.pixel_art;
        let (iso_ground, iso_trees) = self.iso_items(assets, o);
        let iso_figure = self.iso_figure_item(&iso_ground, assets.atlas, o);
        let tilemaps = fd.scene.items_mut::<TilemapItem>();
        tilemaps.push(far);
        tilemaps.push(near);
        if o.iso_sheet {
            tilemaps.push(iso_ground);
            tilemaps.push(iso_trees);
        }

        let mut stones = self.stones_item(assets.atlas, o);
        if o.normal_stones {
            stones.normal_texture_id = Some(assets.normals);
        }
        let mut light = Lighting2dItem::default();
        light.plane = TilemapPlane::Xz;
        light.shadows = o.shadows;
        light.normals = o.normal_stones;
        if o.night {
            light.ambient = Colour::linear_rgb(0.06, 0.07, 0.13);
            light.intensity = 1.0;
        } else {
            light.ambient = Colour::linear_rgb(0.92, 0.92, 0.95);
            light.intensity = 0.35;
        }
        light.occluders = self.wall_occluders.clone();
        light
            .occluders
            .extend(occluders_from_sprites(&stones, TilemapPlane::Xz));
        light.lights = self.plane_lights(o);
        fd.scene.items_mut::<Lighting2dItem>().push(light);

        let sprites = fd.scene.items_mut::<SpriteItem>();
        sprites.push(self.critters_item(assets.atlas, o));
        sprites.push(stones);
        sprites.push(self.player_item(assets.atlas, o));
        sprites.push(self.wisps_item(assets.atlas, o));
        if o.guard {
            sprites.push(self.guard_item(assets.atlas, o));
        }
        if o.iso_sheet {
            sprites.push(iso_figure);
        }
        if let (true, Some(shield)) = (o.guard, assets.shield) {
            fd.scene.mesh_instances.push(self.shield_item(shield));
        }

        let particles = fd.scene.items_mut::<GpuParticleSystemItem>();
        for (i, (&id, &(col, row))) in assets.torches.iter().zip(TORCHES.iter()).enumerate() {
            let flame = torch_flame(col, row);
            let mut item = GpuParticleSystemItem::new(id, self.dt);
            item.emitter.rate = 70.0;
            item.emitter.lifetime = (0.3, 0.75);
            item.emitter.initial_velocity = VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.4,
                min_speed: 6.0,
                max_speed: 16.0 + (i % 3) as f32 * 2.0,
            };
            item.emitter.spawn_shape = SpawnShape::Sphere {
                centre: [flame.x, -2.0, flame.y],
                radius: 1.2,
            };
            item.emitter.colour = Colour::linear(3.2, 1.25, 0.3, 1.0);
            item.emitter.size = 3.0;
            item.forces.push(ForceField::Gravity([0.0, 0.0, 12.0]));
            item.forces.push(ForceField::Drag(1.8));
            particles.push(item);
        }

        fd.effects.lighting.lights = self.lights(o);
        fd.scene.world_overlays.push(shop_sign(self.time));
        fd.scene.world_overlays.push(ley_line(self.time));
        let hud_font = assets.hud_font.filter(|_| o.bitmap_hud);
        fd.overlays = self.overlays(stats, hud_font);
    }

    /// The light item's own lights, in plane coordinates: the player's
    /// lantern as a cone facing the way they walk, six units in front of the
    /// plane for the stones' normals and with soft shadows, and a violet glow
    /// over the guard that lights only the actors' painter layer, so the tiles behind
    /// it stay dark.
    fn plane_lights(&self, o: WorldOptions) -> Vec<Light2d> {
        let mut lights = Vec::new();
        if o.cone_lantern {
            let p = &self.player;
            let dir = if p.facing_left { -1.0 } else { 1.0 };
            let at = p.position + Vec2::new(0.0, 9.0);
            lights.push(
                Light2d::cone([at.x, at.y], 110.0, [dir, -0.2], 0.3, 0.6)
                    .with_colour(Colour::linear_rgb(1.0, 0.92, 0.75))
                    .with_intensity(1.4)
                    .with_height(6.0)
                    .with_shadow_softness(if o.soft_shadows { 6.0 } else { 0.0 }),
            );
        }
        if o.guard && o.actor_light {
            lights.push(
                Light2d::point([self.guard.x, GUARD_Z + 9.0], 40.0)
                    .with_colour(Colour::linear_rgb(0.75, 0.3, 1.0))
                    .with_intensity(2.0)
                    .with_layers(SIDE_LAYER..=SIDE_LAYER),
            );
        }
        lights
    }

    /// The frame's point lights: the torches, the player's lantern unless it
    /// is the cone, and the wisps, each placed on the plane in front of the
    /// ground.
    fn lights(&self, o: WorldOptions) -> Vec<LightSource> {
        let t = self.time;
        let at = |p: Vec2| DVec3::new(p.x as f64, -4.0, p.y as f64);
        let mut lights = Vec::new();
        for (i, &(col, row)) in TORCHES.iter().enumerate() {
            let k = i as f32;
            let flicker = 1.0 + 0.12 * (t * 13.0 + k).sin() + 0.08 * (t * 7.3 + 2.0 * k).sin();
            let mut l = LightSource::point_candela(
                at(torch_flame(col, row)),
                Candela(1.5 * flicker),
                75.0,
                0.0,
            );
            l.colour = Colour::linear_rgb(1.0, 0.62, 0.3);
            lights.push(l);
        }
        if !o.cone_lantern {
            let mut lantern = LightSource::point_candela(
                at(self.player.position + Vec2::new(0.0, 9.0)),
                Candela(1.0),
                60.0,
                0.0,
            );
            lantern.colour = Colour::linear_rgb(1.0, 0.92, 0.75);
            lights.push(lantern);
        }
        for p in self.wisps() {
            let mut l = LightSource::point_candela(at(p), Candela(0.9), 50.0, 0.0);
            l.colour = Colour::linear_rgb(0.35, 0.85, 1.0);
            lights.push(l);
        }
        lights
    }

    fn critters_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let mut item = pixel_sprites(Some(atlas), o);
        // Behind the walls, which cover them through the depth test.
        item.order = SpriteOrder::Painter {
            layer: SIDE_LAYER,
            depth_test: true,
        };
        let mut effects = SpriteEffects::default();
        for c in &self.critters {
            item.positions.push([c.position.x, 0.5, c.position.y]);
            let frame = (c.phase as usize) % 4;
            item.uv_rects.push(cell_uv(frame, 1, 1, 1));
            // The art faces right.
            item.flips.push(if c.speed < 0.0 {
                SpriteFlip::HORIZONTAL
            } else {
                SpriteFlip::NONE
            });
            // Feet lower on screen stand nearer, so they draw later.
            item.sort_keys
                .push(painter_key(SIDE_LAYER, -c.position.y, 0).key);
            item.colours.push(c.tint);
            effects.flash.push(c.flash);
        }
        item.effects = Some(effects);
        item
    }

    fn stones_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let mut item = pixel_sprites(Some(atlas), o);
        item.order = SpriteOrder::Painter {
            layer: SIDE_LAYER,
            depth_test: false,
        };
        for &x in &STONES {
            item.positions.push([x, -0.5, GROUND_Z]);
            item.sizes.push([8.0, 16.0]);
            item.uv_rects.push(cell_uv(4, 2, 1, 2));
            item.sort_keys
                .push(painter_key(SIDE_LAYER, -GROUND_Z, 0).key);
        }
        item
    }

    fn player_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let p = &self.player;
        let mut item = pixel_sprites(Some(atlas), o);
        item.order = SpriteOrder::Painter {
            layer: SIDE_LAYER,
            depth_test: false,
        };
        let frame = if p.on_ground { p.phase as usize % 4 } else { 3 };
        item.positions.push([p.position.x, -1.0, p.position.y]);
        item.sort_keys
            .push(painter_key(SIDE_LAYER, -p.position.y, 1).key);
        item.sizes.push([8.0, 16.0]);
        item.uv_rects.push(cell_uv(frame, 2, 1, 2));
        item.flips.push(if p.facing_left {
            SpriteFlip::HORIZONTAL
        } else {
            SpriteFlip::NONE
        });
        let mut effects = SpriteEffects::default();
        effects.flash = vec![p.flash];
        if o.outline {
            effects.outline = Some(SpriteOutline::new(Colour::srgb(0.1, 0.06, 0.04, 1.0), 2.0));
        }
        item.effects = Some(effects);
        item
    }

    /// The guard's spear (member 0, behind) and body (member 1), keyed at the
    /// guard's sort value so the critters' keys fall either side of the
    /// whole guard. Behind the walls like the critters.
    fn guard_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let g = &self.guard;
        let mut item = pixel_sprites(Some(atlas), o);
        item.order = SpriteOrder::Painter {
            layer: SIDE_LAYER,
            depth_test: true,
        };
        let flip = if g.dir < 0.0 {
            SpriteFlip::HORIZONTAL
        } else {
            SpriteFlip::NONE
        };
        let frame = g.phase as usize % 4;
        let parts = [
            (g.x + g.dir * 4.0, cell_uv(5, 2, 1, 2), 0),
            (g.x, cell_uv(frame, 2, 1, 2), 1),
        ];
        for (x, uv, member) in parts {
            item.positions.push([x, 0.5, GUARD_Z]);
            item.sizes.push([8.0, 16.0]);
            item.uv_rects.push(uv);
            item.flips.push(flip);
            item.sort_keys
                .push(painter_key(SIDE_LAYER, -GUARD_Z, member).key);
        }
        // Grey armour where the player wears blue.
        item.colours = vec![Colour::WHITE, Colour::srgb(0.8, 0.7, 0.7, 1.0)];
        item
    }

    /// The guard's round shield, a flat mesh in front of the body (member 2).
    /// The disc lies on the XY plane, so it is turned to face the camera.
    fn shield_item(&self, mesh: MeshId) -> MeshInstanceItem {
        let g = &self.guard;
        let at = Vec3::new(g.x + g.dir * 2.5, 0.45, GUARD_Z + 6.0);
        let mut item = MeshInstanceItem::default();
        item.mesh_id = mesh;
        item.transforms = vec![
            (Mat4::from_translation(at) * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2))
                .to_cols_array_2d(),
        ];
        item.colours = vec![Colour::srgb(0.95, 0.72, 0.22, 1.0)];
        item.painter = Some(painter_key(SIDE_LAYER, -GUARD_Z, 2));
        item
    }

    /// The isometric sheet's ground (cutout) and trees (blended, one painter
    /// batch per row at key `-y`).
    fn iso_items(&self, assets: &WorldAssets, o: WorldOptions) -> (TilemapItem, TilemapItem) {
        let which = if o.pixel_art { 0 } else { 1 };
        let place = |id: TilemapId| {
            let mut item = TilemapItem::new(id);
            item.plane = TilemapPlane::Xz;
            item.origin = self.iso_origin.to_array();
            item.cell_size = ISO_TILE.map(|s| s as f32);
            item.elevation = -0.8;
            item.pixel_snap = o.pixel_art;
            item
        };
        let ground = place(assets.iso_ground[which]);
        let mut trees = place(assets.iso_trees[which]);
        trees.draw = TilemapDraw::Blended {
            layer: ISO_LAYER,
            depth_test: false,
            painter: TilemapPainter::PerRow { offset: 0.0 },
        };
        (ground, trees)
    }

    /// The figure walking round the sheet, keyed `-feet y` like the tree
    /// rows, so a tree whose row is lower on the sheet covers it and one
    /// higher up stands behind it.
    fn iso_figure_item(
        &self,
        ground: &TilemapItem,
        atlas: TextureId,
        o: WorldOptions,
    ) -> SpriteItem {
        let (feet, left) = self.iso_figure(ground);
        let mut item = pixel_sprites(Some(atlas), o);
        item.order = SpriteOrder::Painter {
            layer: ISO_LAYER,
            depth_test: false,
        };
        item.positions.push([feet.x, 0.7, feet.y]);
        item.sizes.push([8.0, 16.0]);
        let frame = (self.iso_walked * 0.6) as usize % 4;
        item.uv_rects.push(cell_uv(frame, 2, 1, 2));
        item.flips.push(if left {
            SpriteFlip::HORIZONTAL
        } else {
            SpriteFlip::NONE
        });
        item.colours.push(Colour::srgb(0.6, 1.0, 0.6, 1.0));
        item.sort_keys.push(-feet.y);
        item
    }

    fn wisps_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let mut item = pixel_sprites(Some(atlas), o);
        item.blend = SpriteBlend::Additive;
        item.default_pivot = [0.5, 0.5];
        item.order = SpriteOrder::Painter {
            layer: 4,
            depth_test: false,
        };
        for p in self.wisps() {
            item.positions.push([p.x, -2.0, p.y]);
            item.uv_rects.push(cell_uv(4, 1, 1, 1));
            item.colours.push(Colour::linear(1.2, 3.0, 4.0, 1.0));
        }
        item
    }
}

/// A batch drawn in world units with its feet on its position, filtered and
/// snapped for the current presentation.
fn pixel_sprites(atlas: Option<TextureId>, o: WorldOptions) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = atlas;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_size = 8.0;
    item.default_pivot = [0.5, 0.0];
    item.filter = if o.pixel_art {
        TextureFilter::Nearest
    } else {
        TextureFilter::Linear
    };
    item.pixel_snap = o.pixel_art;
    item
}

/// A shop sign of vector art on a world layer: a post, a board with a rim,
/// a stroked potion-bottle path and a label. Layer y runs down the plane
/// from the layer's origin, which is the board's top left.
fn shop_sign(time: f32) -> WorldOverlayLayer {
    let mut frame = OverlayFrame::default();
    frame.time = time as f64;
    let wood = Colour::srgb(0.36, 0.22, 0.12, 1.0);
    frame.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 1.0 },
            [26.0, 26.0],
            [5.0, 18.0],
        )
        .with_fill(OverlayFill::Solid(wood)),
    );
    frame.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 3.0 },
            [0.0, 0.0],
            [58.0, 27.0],
        )
        .with_fill(OverlayFill::linear_gradient(
            Colour::srgb(0.96, 0.84, 0.54, 1.0),
            Colour::srgb(0.66, 0.43, 0.22, 1.0),
            1.4,
        ))
        .with_stroke(OverlayStroke::new(1.2, Colour::srgb(0.22, 0.12, 0.05, 1.0))),
    );
    // A potion bottle: neck, shoulders and a round belly, stroked over a
    // tinted fill.
    let bottle = SubPath::new([5.0, 0.0])
        .line_to([9.0, 0.0])
        .line_to([9.0, 5.0])
        .cubic_to([15.0, 7.0], [15.0, 17.0], [7.0, 17.0])
        .cubic_to([-1.0, 17.0], [-1.0, 7.0], [5.0, 5.0])
        .close();
    frame.shapes.push(
        OverlayShapeItem::vector(vec![bottle], FillRule::NonZero, [5.0, 5.0], [14.0, 17.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.75, 0.2, 0.55, 0.85)))
            .with_stroke(OverlayStroke::new(1.0, Colour::srgb(0.15, 0.05, 0.12, 1.0))),
    );
    frame.labels.push(
        LabelItem::new("POTIONS")
            .with_position([22.0, 9.0])
            .with_font_size(7.0)
            .with_colour(Colour::srgb(0.2, 0.1, 0.05, 1.0)),
    );
    WorldOverlayLayer::new(WorldPlane::Xz)
        .with_origin([268.0, GROUND_Z + 44.0])
        .with_elevation(0.5)
        .with_order(0)
        .with_frame(frame)
}

/// A glowing trail along the ground, rippling with time: bright enough in
/// linear light to bloom, added over what is below.
fn ley_line(time: f32) -> WorldOverlayLayer {
    let mut frame = OverlayFrame::default();
    let points = (0..=96)
        .map(|i| {
            let x = i as f32 * 4.0;
            [x, 4.0 + 2.5 * (x * 0.08 + time * 2.0).sin()]
        })
        .collect();
    frame.polylines.push(
        OverlayPolylineItem::new(points)
            .with_thickness(1.5)
            .with_cap(LineCap::Round)
            .with_colour(Colour::linear(0.8, 2.6, 4.0, 1.0))
            .with_blend(OverlayBlend::Additive),
    );
    WorldOverlayLayer::new(WorldPlane::Xz)
        .with_origin([360.0, GROUND_Z + 8.0])
        .with_elevation(0.5)
        .with_order(0)
        .with_frame(frame)
}

/// One frame of the world as a whole `FrameData`, for drivers without a host
/// frame to fill.
#[allow(dead_code)]
pub(crate) fn frame(world: &World, assets: &WorldAssets, stats: &WorldStats) -> FrameData {
    let mut fd = FrameData::new(world.camera_frame(), vpl::SceneFrame::default());
    world.fill_frame(&mut fd, assets, stats);
    fd
}
