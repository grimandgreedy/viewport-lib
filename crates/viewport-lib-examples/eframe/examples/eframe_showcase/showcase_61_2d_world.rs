//! Showcase 61: 2D World.
//!
//! A scrolling side-view world under `Camera2d` on `Plane2d::Xz`. In a Z-up
//! library the XZ plane is the one where up on screen is world +Z, so the
//! jump, gravity and the torch flames all move along the world's own up axis;
//! the camera looks along +Y and depth is world Y. The camera, both tilemaps,
//! the 2D light layer and both world overlay layers all name the same plane;
//! the sprites sort by `-z`, their feet's height, so a critter lower on screen
//! covers the ones behind it. The table at the top of `world.rs` lists the
//! depth of each layer.
//!
//! On screen:
//! - Two tilemap layers: the ground and walls at parallax 1, and far hills
//!   that scroll at 0.4 of the camera's speed across and 1 up and down.
//! - Three hundred critters from one atlas, each picking its walk frame per
//!   frame and flipping to face its way, behind the walls through the depth
//!   test.
//! - The player: walked with the keyboard (or by the autopilot), outlined,
//!   flashing when hit, followed by the camera with a dead zone.
//! - Twelve point lights (eight torches, the player's lantern, three wisps)
//!   through `Lighting2dItem`, with hard shadows from the walls, the plank
//!   platforms and three standing-stone sprites.
//! - GPU particles with additive blend for the torch flames.
//! - Bloom and vignette.
//! - A HUD of screen overlays (health bar shapes, a score label, a minimap
//!   mesh), a shop sign of vector art and a glowing trail on world overlay
//!   layers, a world-sized name tag with a minimum size, and floating damage
//!   numbers.
//!
//! The world itself is in `world.rs` and builds its frame without egui; this
//! file maps the keys and the controls panel onto it.
//!
//! Keys (with the pointer over the viewport): A / D or the arrows walk, W or
//! Up jumps, J attacks, F takes a hit, G shakes the camera, P switches pixel
//! art, N switches night. Scroll zooms; under pixel art the zoom steps by
//! whole pixels per texel.

mod world;

use crate::App;
use crate::eframe::egui;
use viewport_lib as vpl;
use viewport_lib_plugins::item_types::lighting_2d::{Lighting2dPlugin, TYPE_NAME as LIGHTING_2D};

pub(crate) struct World2dState {
    pub world: world::World,
    pub assets: Option<world::WorldAssets>,
    /// The previous frame's counters, for the HUD's stats strip.
    pub stats: world::WorldStats,
    /// Inputs from the controls panel, applied on the next step.
    pub pending: world::WorldInput,
}

impl Default for World2dState {
    fn default() -> Self {
        Self {
            world: world::World::new(),
            assets: None,
            stats: world::WorldStats::default(),
            pending: world::WorldInput::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

fn controls(app: &mut App, ui: &mut egui::Ui) {
    let st = &mut app.world2d_state;
    let o = &mut st.world.options;
    ui.checkbox(&mut o.pixel_art, "Pixel art (P)")
        .on_hover_text("Nearest filtering, pixel snapping and integer zoom; off is smooth");
    ui.checkbox(&mut o.night, "Night (N)");
    ui.checkbox(&mut o.shadows, "Occluder shadows");
    ui.checkbox(&mut o.bloom, "Bloom");
    ui.checkbox(&mut o.vignette, "Vignette");
    ui.checkbox(&mut o.outline, "Player outline");
    ui.checkbox(&mut o.hud, "HUD");
    ui.checkbox(&mut o.autopilot, "Autopilot")
        .on_hover_text("Walks the player until a movement key is pressed");
    ui.horizontal(|ui| {
        if ui.button("Shake (G)").clicked() {
            st.pending.shake = true;
        }
        if ui.button("Hit (F)").clicked() {
            st.pending.hurt = true;
        }
        if ui.button("Attack (J)").clicked() {
            st.pending.attack = true;
        }
    });

    ui.separator();
    ui.label(egui::RichText::new("Keys over the viewport").strong());
    ui.label("A / D or arrows: walk\nW or Up: jump\nJ: attack   F: take a hit\nG: shake   P: pixel art   N: night\nScroll: zoom");

    ui.separator();
    let s = &st.stats;
    ui.label(egui::RichText::new("Last frame").strong());
    ui.label(format!(
        "draw calls {} (plugin {})\noverlay draws {}\nlayers composited {}\nglyph rasters {}  vector rasters {}\n2D lights {}  shadow edges {}  passes {}",
        s.frame.draw_calls,
        s.frame.plugin_draw_calls,
        s.frame.overlay_draw_calls,
        s.frame.overlay_layers_composited,
        s.frame.overlay_glyph_rasters,
        s.frame.overlay_vector_rasters,
        s.lights.lights,
        s.lights.edges,
        s.lights.passes,
    ));
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// This frame's keys, and the one-shot toggles they flip.
fn read_keys(app: &mut App, cx: &crate::ViewportCtx) -> world::WorldInput {
    let mut input = std::mem::take(&mut app.world2d_state.pending);
    // Keys belong to a focused text field, or to the viewport under the
    // pointer.
    if cx.egui.wants_keyboard_input() || !cx.response.hovered() {
        return input;
    }
    let o = &mut app.world2d_state.world.options;
    cx.egui.input(|i| {
        use egui::Key;
        input.left |= i.key_down(Key::A) || i.key_down(Key::ArrowLeft);
        input.right |= i.key_down(Key::D) || i.key_down(Key::ArrowRight);
        input.jump |= i.key_down(Key::W) || i.key_down(Key::ArrowUp);
        input.attack |= i.key_pressed(Key::J);
        input.hurt |= i.key_pressed(Key::F);
        input.shake |= i.key_pressed(Key::G);
        if i.key_pressed(Key::P) {
            o.pixel_art = !o.pixel_art;
        }
        if i.key_pressed(Key::N) {
            o.night = !o.night;
        }
    });
    input
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScWorld2d;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScWorld2d = ScWorld2d;

impl crate::Showcase for ScWorld2d {
    fn needs_build(&self, app: &crate::App) -> bool {
        app.world2d_state.assets.is_none()
    }

    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        app.world2d_state.assets = Some(world::build_assets(renderer, &app.device, &app.queue));
    }

    /// Step the world before the host resolves the camera, with the host's
    /// resolved pointer input for the wheel zoom.
    fn advance(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        let input = read_keys(app, cx);
        let nav = app.input.resolve();
        let dt = cx.egui.input(|i| i.stable_dt);
        let size = glam::Vec2::new(cx.rect.width(), cx.rect.height());
        let ppp = cx.egui.pixels_per_point();
        app.world2d_state
            .world
            .step(&input, Some(&nav), size, ppp, dt);
    }

    /// The camera is the world's `Camera2d`; the orbit camera stays put.
    fn drive_camera(&self, _app: &mut crate::App, _cx: &crate::ViewportCtx) -> bool {
        true
    }

    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
        if let Some(rs) = ctx.frame.wgpu_render_state() {
            let guard = rs.renderer.read();
            if let Some(renderer) = guard.callback_resources.get::<vpl::ViewportRenderer>() {
                app.world2d_state.stats.frame = renderer.last_frame_stats();
                if let Some(plugin) = renderer.item_type_plugin::<Lighting2dPlugin>(LIGHTING_2D) {
                    app.world2d_state.stats.lights = plugin.last_stats();
                }
            }
        }
        let st = &app.world2d_state;
        if let Some(assets) = &st.assets {
            st.world.fill_frame(fd, assets, &st.stats);
        }
    }

    fn tick(&self, _app: &mut crate::App, cx: &crate::ViewportCtx) {
        cx.egui.request_repaint();
    }

    /// Step 0 is the day in pixel art, 1 the same smooth, 2 the night.
    fn tour(&self, app: &mut crate::App, step: u32) -> bool {
        let o = &mut app.world2d_state.world.options;
        match step {
            0 => {}
            1 => o.pixel_art = false,
            2 => {
                o.pixel_art = true;
                o.night = true;
            }
            _ => return false,
        }
        true
    }

    fn controls(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        _frame: &crate::eframe::Frame,
    ) {
        controls(app, ui)
    }
}
