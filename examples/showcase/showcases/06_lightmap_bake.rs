//! The whole offline lightmapper, run live: unwrap -> texel G-buffer -> GI solve
//! -> denoise -> encode -> consume.
//!
//! A small room (grey floor, red and green side walls, a back wall) with a torus
//! sitting on the floor. The floor and the torus have their lighting baked by the
//! path tracer: the torus is UV-unwrapped with xatlas, both are rasterised into
//! their lightmap atlas to get a world point per texel, a GI hemisphere is shot
//! from each, the noisy atlas is denoised and dilated, and the result is encoded
//! and sampled back onto the surface by UV1.
//!
//! The top chip switches **Baked GI** against **Realtime only** : the same room
//! lit by one realtime light with flat ambient. Baked adds the soft contact
//! shadow under the torus, the red/green colour bleed onto the floor and the
//! torus, and the ambient occlusion in the torus' inner ring : indirect light no
//! single realtime light reproduces. The side panel re-bakes at different sample
//! counts, toggles the denoiser (watch the noise return), and shows the baked
//! atlas on a floating panel.

use eframe::egui;
use glam::{Mat3, Mat4, Vec2, Vec3};
use viewport_lib::bake::{TexelGeometry, rasterize_texel_gbuffer};
use viewport_lib::raytrace::{
    RtLight, RtMaterial, RtScene, RtSettings, TexelSurfaces, bake_lightmap_directional,
};
use viewport_lib::resources::{LightmapData, LightmapMode, TextureId};
use viewport_lib::{
    BackfacePolicy, ItemSettings, LightKind, LightSource, Material, MeshData, MeshId, NodeId,
    primitives,
};
use viewport_lib_bake::denoise::{DenoiseParams, denoise, dilate};
use viewport_lib_bake::encode::{Encoding, encode};
use viewport_lib_bake::stitch::{StitchGeometry, StitchParams, stitch};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

// Requested unwrap resolution. xatlas packs into its own size (often larger than
// this), and each baked piece then bakes and uploads at that actual packed size
// (see `Piece::atlas_w`): the texel G-buffer, denoise, and texture must all match
// the size the piece's uv1 is normalised to, or the charts squash and read as
// blocky steps on the mesh.
const ATLAS: u32 = 512;
/// Key-light direction (toward the light); shared by the bake and the realtime
/// mode so the two are directly comparable. Raked off vertical so the torus casts
/// a long, obvious shadow across the floor.
const LIGHT_DIR: Vec3 = Vec3::new(0.4, -0.32, 1.0);
const FLOOR_ALBEDO: [f32; 3] = [0.80, 0.80, 0.80];
const TORUS_ALBEDO: [f32; 3] = [0.82, 0.80, 0.72];
const RED: [f32; 3] = [0.85, 0.12, 0.10];
const GREEN: [f32; 3] = [0.15, 0.75, 0.20];
const BACK: [f32; 3] = [0.75, 0.75, 0.78];

/// One surface in the room. `pos`/`nrm`/`idx` are the local geometry (kept so the
/// bake can transform it to world space and rasterise its texel G-buffer); `uv1`
/// is its unique lightmap UV. `baked` surfaces get a lightmap; the walls are flat
/// context.
struct Piece {
    mesh: MeshId,
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    idx: Vec<u32>,
    uv1: Vec<Vec2>,
    /// Atlas size this piece's `uv1` is normalised to. The unwrap packs into its
    /// own size (often not the requested resolution), and the texel G-buffer and
    /// uploaded texture must match it: bake at a different size and the charts
    /// are squashed and read as blocky steps on the mesh.
    atlas_w: u32,
    atlas_h: u32,
    xf: Mat4,
    albedo: [f32; 3],
    baked: bool,
    /// Cached bake outputs, so denoise/encode can re-run without re-tracing.
    raw_irradiance: Vec<[f32; 4]>,
    raw_direction: Vec<[f32; 4]>,
    gbuf_pos: Vec<[f32; 4]>,
    gbuf_nrm: Vec<[f32; 4]>,
    tex: Option<TextureId>,
}

pub struct LightmapBakeShowcase {
    /// 0 = baked GI, 1 = realtime only.
    mode: usize,
    shown: Option<usize>,
    built: bool,
    pieces: Vec<Piece>,
    nodes: Vec<NodeId>,
    // Bake controls.
    samples: u32,
    denoise: bool,
    baked_at: Option<u32>,
    need_reencode: bool,
    applied: Option<(usize, bool)>,
    /// Realtime shadow casting (Realtime-only mode). Off by default so the
    /// realtime view is flat and the bake's contribution is unambiguous.
    realtime_shadows: bool,
    // Floating atlas panel.
    atlas_mesh: Option<MeshId>,
    atlas_uv: Vec<Vec2>,
    show_atlas: bool,
    atlas_node: Option<NodeId>,
    // Stats for the panel.
    torus_charts: u32,
    /// Actual packed atlas size of the torus (xatlas picks it; it is usually not
    /// the requested resolution).
    torus_atlas: (u32, u32),
    bake_ms: u32,
    directionality: f32,
    request_rebake: bool,
}

impl LightmapBakeShowcase {
    pub fn new() -> Self {
        Self {
            mode: 0,
            shown: None,
            built: false,
            pieces: Vec::new(),
            nodes: Vec::new(),
            samples: 64,
            denoise: true,
            baked_at: None,
            need_reencode: false,
            applied: None,
            realtime_shadows: false,
            atlas_mesh: None,
            atlas_uv: Vec::new(),
            show_atlas: false,
            atlas_node: None,
            torus_charts: 0,
            torus_atlas: (ATLAS, ATLAS),
            bake_ms: 0,
            directionality: 0.0,
            request_rebake: false,
        }
    }

    /// Trace the shared room and, for each baked piece, path-trace its lightmap
    /// and cache the raw irradiance + direction atlases (no denoise/encode yet).
    fn trace_all(&mut self, ctx: &mut ShowcaseCtx) {
        let device = ctx.device;
        let queue = ctx.queue;

        // The scene every bake traces against: all pieces in world space, plus a
        // key light and a soft sky. The coloured walls are what bleed.
        let mut scene = RtScene::new();
        // A dim sky keeps the shadow and colour bleed high-contrast; the key
        // light does the lighting.
        scene.set_sky([0.16, 0.18, 0.24], [0.03, 0.03, 0.04]);
        for p in &self.pieces {
            let (wp, wn) = world_geo(&p.pos, &p.nrm, p.xf);
            scene.add_mesh(
                &wp,
                &p.idx,
                Some(&wn),
                RtMaterial {
                    base_colour: p.albedo,
                    roughness: 0.9,
                    ..RtMaterial::default()
                },
            );
        }
        // Tuned for the HDR display path: the baked radiance now feeds the
        // renderer's tonemapper once (linear `Rgba16Float` upload), so the key is
        // dimmer than the old value that was sized for a pre-tonemapped upload.
        scene.add_light(RtLight::Directional {
            direction: LIGHT_DIR.normalize().to_array(),
            colour: [2.1, 2.05, 1.9],
        });

        let settings = RtSettings {
            samples: self.samples,
            max_bounces: 4,
            denoise: false,
        };
        for i in 0..self.pieces.len() {
            if !self.pieces[i].baked {
                continue;
            }
            let (pos, nrm, uv1, idx, xf, aw, ah) = {
                let p = &self.pieces[i];
                (
                    p.pos.clone(),
                    p.nrm.clone(),
                    p.uv1.iter().map(|u| [u.x, u.y]).collect::<Vec<_>>(),
                    p.idx.clone(),
                    p.xf,
                    p.atlas_w,
                    p.atlas_h,
                )
            };
            let gbuf = rasterize_texel_gbuffer(
                device,
                queue,
                &TexelGeometry {
                    positions: &pos,
                    normals: &nrm,
                    uv1: &uv1,
                    indices: &idx,
                    model: xf,
                },
                aw,
                ah,
            );
            let bake = bake_lightmap_directional(
                device,
                queue,
                &scene,
                &TexelSurfaces {
                    width: gbuf.width,
                    height: gbuf.height,
                    world_pos: &gbuf.world_pos,
                    world_normal: &gbuf.world_normal,
                },
                &settings,
            );
            let p = &mut self.pieces[i];
            p.raw_irradiance = to_rgba4(&bake.irradiance);
            p.raw_direction = to_rgba4(&bake.direction);
            p.gbuf_pos = gbuf.world_pos;
            p.gbuf_nrm = gbuf.world_normal;
        }
        self.baked_at = Some(self.samples);
    }

    /// Denoise (or not), encode, tonemap, and upload each baked piece's atlas.
    /// Cheap : re-runs on the cached raw bake when the denoise toggle flips.
    fn encode_all(&mut self, ctx: &mut ShowcaseCtx) {
        let device = ctx.device;
        let queue = ctx.queue;
        let mut directionality_sum = 0.0f64;
        let mut directionality_n = 0u64;
        for i in 0..self.pieces.len() {
            if !self.pieces[i].baked || self.pieces[i].raw_irradiance.is_empty() {
                continue;
            }
            let (aw, ah) = (self.pieces[i].atlas_w, self.pieces[i].atlas_h);
            // Denoise is optional (the toggle), but dilation is not: it fills the
            // empty chart gutter so bilinear sampling at a chart edge never reads
            // the black border. Always dilate, so denoise-off still has clean
            // chart edges rather than black seam lines.
            let denoised = if self.denoise {
                denoise(
                    &self.pieces[i].raw_irradiance,
                    &self.pieces[i].gbuf_pos,
                    &self.pieces[i].gbuf_nrm,
                    aw,
                    ah,
                    &DenoiseParams::default(),
                )
            } else {
                self.pieces[i].raw_irradiance.clone()
            };
            // Stitch cross-chart seams: make the two sides of every chart cut
            // agree so the boundary stops showing as a thin line, then dilate the
            // corrected charts into the gutter.
            let uv1: Vec<[f32; 2]> = self.pieces[i].uv1.iter().map(|u| [u.x, u.y]).collect();
            let stitched = stitch(
                &denoised,
                aw,
                ah,
                &StitchGeometry {
                    positions: &self.pieces[i].pos,
                    uv1: &uv1,
                    indices: &self.pieces[i].idx,
                },
                &StitchParams::default(),
            );
            let cleaned = dilate(&stitched, aw, ah, 6);
            // Encode into the neutral directional lightmap (exercises the encoder
            // and yields the directionality stat); the display samples the
            // radiance channel.
            let lm = encode(
                aw,
                ah,
                &cleaned,
                Some(&self.pieces[i].raw_direction),
                &self.pieces[i].gbuf_nrm,
                Encoding::DominantDirection,
            );
            if let Some(dir) = lm.direction() {
                for d in dir {
                    if d[3] > 0.0 {
                        directionality_sum += d[3] as f64;
                        directionality_n += 1;
                    }
                }
            }
            // Upload linear diffuse radiance (incident irradiance / pi) through
            // the HDR path. The material keeps the true albedo, so Replace mode
            // (base_colour * lm.rgb) gives albedo * E/pi. No albedo baked in, no
            // exposure applied here: the renderer's HDR pipeline tonemaps once for
            // display. (The old 8-bit sRGB upload pre-tonemapped, so the render
            // path tonemapped it a second time.)
            let inv_pi = 1.0 / std::f32::consts::PI;
            let mut radiance = vec![0.0f32; lm.radiance().len() * 4];
            for (t, px) in lm.radiance().iter().enumerate() {
                if px[3] <= 0.5 {
                    continue;
                }
                radiance[t * 4] = px[0] * inv_pi;
                radiance[t * 4 + 1] = px[1] * inv_pi;
                radiance[t * 4 + 2] = px[2] * inv_pi;
                radiance[t * 4 + 3] = 1.0;
            }
            let tex = ctx
                .session
                .resources_mut()
                .upload_texture_hdr(device, queue, aw, ah, &radiance)
                .unwrap();
            self.pieces[i].tex = Some(tex);
        }
        self.directionality = if directionality_n > 0 {
            (directionality_sum / directionality_n as f64) as f32
        } else {
            0.0
        };
        // Force the lightmaps to re-apply with the new textures.
        self.applied = None;
    }

    /// Build (or rebuild) the scene nodes for the current mode.
    fn rebuild(&mut self, session: &mut viewport_lib::ViewportSession) {
        if !self.nodes.is_empty() {
            let ids = std::mem::take(&mut self.nodes);
            session.scene_mut().remove_many(&ids);
        }
        self.atlas_node = None;

        let baked_mode = self.mode == 0;

        // Lighting: baked mode leans on the lightmaps (runtime lights off, dim
        // ambient for the walls); realtime mode lights everything with one key
        // light plus hemisphere ambient : the flat look baking improves on.
        {
            let l = &mut session.effects_mut().lighting;
            if baked_mode {
                l.lights = Vec::new();
                l.hemisphere_intensity = 0.28;
                l.sky_colour = [0.5, 0.54, 0.62];
                l.ground_colour = [0.16, 0.16, 0.18];
            } else {
                let mut key = LightSource::default();
                key.kind = LightKind::Directional {
                    direction: LIGHT_DIR.to_array(),
                };
                key.colour = [1.0, 0.98, 0.95];
                key.intensity = 1.1;
                // The light always keeps its shadow cascades; whether a shadow
                // actually appears is gated per-object below, so the toggle is
                // honoured reliably by the shadow pass.
                key.cast_shadows = true;
                l.lights = vec![key];
                // Low ambient so the realtime shadow (when the toggle is on) is
                // not washed out by fill light.
                l.hemisphere_intensity = 0.18;
                l.sky_colour = [0.6, 0.64, 0.72];
                l.ground_colour = [0.2, 0.2, 0.22];
                // Shadow rendering persists on the shared session across
                // showcases, so set it explicitly rather than assuming a prior
                // showcase left it on. Fit the shadow frustum to this room (auto
                // is 20, looser than the scene needs).
                l.shadows_enabled = true;
                l.shadow_extent_override = Some(13.0);
            }
        }

        for p in &self.pieces {
            // Every piece keeps its true albedo: the baked lightmap now stores
            // material-independent incident radiance (E/pi), and Replace mode
            // multiplies it by the material's base_colour (albedo).
            let mut mat = Material::pbr(p.albedo, 0.0, 0.9);
            mat.backface_policy = BackfacePolicy::Identical;
            let id = session.scene_mut().add(Some(p.mesh), p.xf, mat);
            // Per-object cast-shadows: the shadow pass skips items with this off,
            // so the toggle reliably shows/hides the realtime shadow. Baked mode
            // has no realtime light, so this is inert there.
            let mut ap = ItemSettings::default();
            ap.cast_shadows = self.realtime_shadows;
            session.scene_mut().set_appearance(id, ap);
            self.nodes.push(id);
        }

        // The baked atlas, mounted flat on the back wall like a poster so it
        // reads as a preview of the torus' lightmap rather than stray geometry.
        if baked_mode && self.show_atlas {
            if let Some(atlas) = self.atlas_mesh {
                let mut mat = Material::pbr([1.0, 1.0, 1.0], 0.0, 1.0);
                mat.backface_policy = BackfacePolicy::Identical;
                let xf = Mat4::from_translation(Vec3::new(5.5, 6.85, 4.0))
                    * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
                let id = session.scene_mut().add(Some(atlas), xf, mat);
                // The preview poster is not part of the lighting: no casting or
                // receiving shadows.
                let mut ap = ItemSettings::default();
                ap.cast_shadows = false;
                ap.receive_shadows = false;
                session.scene_mut().set_appearance(id, ap);
                self.atlas_node = Some(id);
                self.nodes.push(id);
            }
        }

        self.applied = None;
    }

    /// Attach or clear the baked lightmaps to match the current mode.
    fn apply_lightmaps(&mut self, ctx: &mut ShowcaseCtx) {
        let state = (self.mode, self.show_atlas);
        if self.applied == Some(state) {
            return;
        }
        let baked_mode = self.mode == 0;
        let device = ctx.device;

        // The atlas panel samples the torus' radiance texture across a full quad.
        let torus_tex = self
            .pieces
            .iter()
            .find(|p| p.baked && p.albedo == TORUS_ALBEDO)
            .and_then(|p| p.tex);

        let res = ctx.session.resources_mut();
        for p in &self.pieces {
            if !p.baked {
                continue;
            }
            match (baked_mode, p.tex) {
                (true, Some(tex)) => {
                    let _ = res.set_lightmap(
                        device,
                        p.mesh,
                        &p.uv1,
                        LightmapData::NonDirectional { radiance: tex },
                        LightmapMode::Replace,
                    );
                }
                _ => {
                    let _ = res.clear_lightmap(p.mesh);
                }
            }
        }
        if let (Some(atlas), Some(tex)) = (self.atlas_mesh, torus_tex) {
            if baked_mode && self.show_atlas {
                let _ = res.set_lightmap(
                    device,
                    atlas,
                    &self.atlas_uv,
                    LightmapData::NonDirectional { radiance: tex },
                    LightmapMode::Replace,
                );
            } else {
                let _ = res.clear_lightmap(atlas);
            }
        }
        self.applied = Some(state);
    }
}

impl Showcase for LightmapBakeShowcase {
    fn name(&self) -> &str {
        "Lightmap bake"
    }

    fn setup(&mut self, ctx: &mut SetupCtx) {
        // Fresh meshes mean a fresh bake, even if this instance is re-set up on a
        // later visit.
        self.baked_at = None;
        self.built = false;
        self.applied = None;
        self.need_reencode = false;

        let mut pieces = Vec::new();

        // Floor.
        let floor = primitives::plane(20.0, 14.0);
        pieces.push(make_piece(ctx, &floor, Mat4::IDENTITY, FLOOR_ALBEDO, true));

        // Torus hero, UV-unwrapped with xatlas so its lightmap UVs are unique.
        let torus = primitives::torus(2.2, 0.8, 64, 32);
        let unwrapped = viewport_lib_bake::unwrap(
            &viewport_lib_bake::UnwrapInput {
                positions: &torus.positions,
                normals: Some(&torus.normals),
                indices: &torus.indices,
            },
            &viewport_lib_bake::UnwrapOptions {
                resolution: ATLAS,
                // Generous inter-chart padding so dilation can fill the gutter and
                // bilinear sampling at a chart edge never reads across the seam.
                padding: 6,
                ..Default::default()
            },
        )
        .expect("unwrap torus");
        self.torus_charts = unwrapped.chart_count;
        self.torus_atlas = (unwrapped.width, unwrapped.height);
        let torus_mesh = build_mesh(
            ctx,
            &unwrapped.positions,
            &unwrapped.normals,
            &unwrapped.uv1,
            &unwrapped.indices,
        );
        let torus_xf =
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.2)) * Mat4::from_rotation_x(0.35);
        pieces.push(Piece {
            mesh: torus_mesh,
            pos: unwrapped.positions,
            nrm: unwrapped.normals,
            idx: unwrapped.indices,
            uv1: unwrapped
                .uv1
                .iter()
                .map(|u| Vec2::new(u[0], u[1]))
                .collect(),
            atlas_w: unwrapped.width,
            atlas_h: unwrapped.height,
            xf: torus_xf,
            albedo: TORUS_ALBEDO,
            baked: true,
            raw_irradiance: Vec::new(),
            raw_direction: Vec::new(),
            gbuf_pos: Vec::new(),
            gbuf_nrm: Vec::new(),
            tex: None,
        });

        // Walls: coloured context, not baked, part of the trace scene for bleed.
        let side = primitives::plane(14.0, 7.0);
        let hp = std::f32::consts::FRAC_PI_2;
        pieces.push(make_piece(
            ctx,
            &side,
            Mat4::from_translation(Vec3::new(-10.0, 0.0, 3.5)) * Mat4::from_rotation_y(hp),
            RED,
            false,
        ));
        pieces.push(make_piece(
            ctx,
            &side,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 3.5)) * Mat4::from_rotation_y(-hp),
            GREEN,
            false,
        ));
        let back = primitives::plane(20.0, 7.0);
        pieces.push(make_piece(
            ctx,
            &back,
            Mat4::from_translation(Vec3::new(0.0, 7.0, 3.5)) * Mat4::from_rotation_x(-hp),
            BACK,
            false,
        ));

        self.pieces = pieces;

        // The floating atlas quad and its full-quad UV1.
        let panel = primitives::plane(6.0, 6.0);
        self.atlas_uv = panel
            .uvs
            .as_ref()
            .map(|uvs| uvs.iter().map(|u| Vec2::new(u[0], u[1])).collect())
            .unwrap_or_default();
        self.atlas_mesh = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &panel)
                .unwrap(),
        );

        ctx.session.viewport_frame_mut().show_grid = false;
        ctx.session.camera_mut().distance = 30.0;
        ctx.session.camera_mut().orientation = glam::Quat::from_rotation_x(0.5);

        self.rebuild(ctx.session);
        self.shown = Some(self.mode);
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        if self.shown != Some(self.mode) {
            self.rebuild(ctx.session);
            self.shown = Some(self.mode);
        }

        if self.mode == 0 {
            // Re-trace on first entry, a rebake request, or a sample-count change;
            // re-encode only (cheap, no tracing) when the denoiser is toggled.
            let need_trace = self.request_rebake
                || self.baked_at.is_none()
                || self.baked_at != Some(self.samples);
            if need_trace {
                self.request_rebake = false;
                let t0 = std::time::Instant::now();
                self.trace_all(ctx);
                self.encode_all(ctx);
                self.bake_ms = t0.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;
                self.need_reencode = false;
                self.built = true;
            } else if self.need_reencode {
                self.need_reencode = false;
                self.encode_all(ctx);
            }
        }

        self.apply_lightmaps(ctx);
        ctx.drive_camera();
    }

    fn description(&self) -> &str {
        match self.mode {
            0 => {
                "Baked GI: the floor and torus lighting is path-traced offline : unwrap, \
                 texel G-buffer, GI solve, denoise, encode. Soft contact shadow, red/green \
                 colour bleed, and inner-ring occlusion are all baked in."
            }
            _ => {
                "Realtime only: the same room lit by one realtime light and flat ambient. \
                 No bounce, no colour bleed, no baked occlusion : switch to Baked GI to see \
                 what the offline solve adds."
            }
        }
    }

    fn has_controls(&self) -> bool {
        true
    }

    fn top_overlay(&mut self, ui: &mut egui::Ui) {
        if let Some(i) = crate::ui::segmented(ui, self.mode, &["Baked GI", "Realtime only"]) {
            self.mode = i;
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Lightmap bake");
        ui.add_space(4.0);
        ui.label(
            "The full offline lightmapper, run live: xatlas unwrap, texel G-buffer, \
             path-traced GI, guided denoise, encode.",
        );
        ui.add_space(8.0);

        ui.add_enabled_ui(self.mode == 0, |ui| {
            ui.label("Samples per texel:");
            ui.add(egui::Slider::new(&mut self.samples, 16..=512));
            if ui.button("Rebake").clicked() {
                self.request_rebake = true;
            }
            ui.add_space(6.0);
            if ui.checkbox(&mut self.denoise, "Denoise").changed() {
                // Re-encode from the cached raw bake next frame : no re-trace.
                self.need_reencode = true;
            }
            if ui
                .checkbox(&mut self.show_atlas, "Show baked atlas")
                .changed()
            {
                self.shown = None; // force a scene rebuild to add/remove the panel
            }
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);
        if ui
            .checkbox(&mut self.realtime_shadows, "Realtime cast shadows")
            .changed()
        {
            self.shown = None; // rebuild to re-set the key light
        }
        ui.label(
            "Affects the Realtime-only view. Off: flat, no shadow, so Baked GI shows \
             exactly what the bake adds. On: a hard realtime shadow, but still no \
             colour bleed or occlusion.",
        );

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);
        ui.label(egui::RichText::new("Bake stats").strong());
        ui.label(format!("Torus charts: {}", self.torus_charts));
        ui.label(format!(
            "Torus atlas: {} x {}",
            self.torus_atlas.0, self.torus_atlas.1
        ));
        ui.label(format!("Samples: {}", self.baked_at.unwrap_or(0)));
        ui.label(format!("Bake time: {} ms", self.bake_ms));
        ui.label(format!("Mean directionality: {:.2}", self.directionality));
        ui.add_space(8.0);
        ui.label(
            "Denoise off shows the raw Monte-Carlo noise; the atlas panel shows the \
             torus' baked lightmap in UV space.",
        );
    }
}

/// Upload a primitive as a baked-or-context [`Piece`], keeping its local geometry.
fn make_piece(
    ctx: &mut SetupCtx,
    mesh: &MeshData,
    xf: Mat4,
    albedo: [f32; 3],
    baked: bool,
) -> Piece {
    let id = ctx
        .session
        .resources_mut()
        .upload_mesh_data(ctx.device, mesh)
        .unwrap();
    let uv1 = mesh
        .uvs
        .as_ref()
        .map(|uvs| uvs.iter().map(|u| Vec2::new(u[0], u[1])).collect())
        .unwrap_or_default();
    Piece {
        mesh: id,
        pos: mesh.positions.clone(),
        nrm: mesh.normals.clone(),
        idx: mesh.indices.clone(),
        uv1,
        // Non-unwrapped pieces (the floor) use their own [0,1] plane UVs, one
        // chart, so any square atlas resolution works.
        atlas_w: ATLAS,
        atlas_h: ATLAS,
        xf,
        albedo,
        baked,
        raw_irradiance: Vec::new(),
        raw_direction: Vec::new(),
        gbuf_pos: Vec::new(),
        gbuf_nrm: Vec::new(),
        tex: None,
    }
}

/// Build a mesh from raw arrays (used for the unwrapped, re-indexed torus).
fn build_mesh(
    ctx: &mut SetupCtx,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uv0: &[[f32; 2]],
    indices: &[u32],
) -> MeshId {
    let mut m = MeshData::default();
    m.positions = positions.to_vec();
    m.normals = normals.to_vec();
    m.indices = indices.to_vec();
    m.uvs = Some(uv0.to_vec());
    ctx.session
        .resources_mut()
        .upload_mesh_data(ctx.device, &m)
        .unwrap()
}

/// Transform local positions and normals into world space for the trace scene.
fn world_geo(positions: &[[f32; 3]], normals: &[[f32; 3]], xf: Mat4) -> (Vec<Vec3>, Vec<Vec3>) {
    let nm = Mat3::from_mat4(xf);
    let wp = positions
        .iter()
        .map(|p| xf.transform_point3(Vec3::from_array(*p)))
        .collect();
    let wn = normals
        .iter()
        .map(|n| (nm * Vec3::from_array(*n)).normalize_or_zero())
        .collect();
    (wp, wn)
}

fn to_rgba4(rgba: &[f32]) -> Vec<[f32; 4]> {
    rgba.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}
