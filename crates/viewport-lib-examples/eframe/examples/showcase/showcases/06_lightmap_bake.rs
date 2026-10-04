//! The whole offline lightmapper, run live: unwrap -> texel G-buffer -> GI solve
//! -> denoise -> encode -> consume.
//!
//! A small room (grey floor, red and green side walls, a back wall) holds four
//! baked hero objects: a torus, an icosphere, a normal-mapped cuboid, and a
//! finely tessellated torus. Each is UV-unwrapped with xatlas, rasterised into
//! its lightmap atlas to get a world point per texel, has a GI hemisphere shot
//! from each texel, then the noisy atlas is denoised, seam-stitched, dilated,
//! encoded, and sampled back onto the surface by UV1. Three cases are exercised
//! side by side: the cuboid gets a directional lightmap (its bump normal map
//! catches the baked light direction); the front torus deliberately spills its
//! unwrap across several atlas pages and loads as a texture array with a
//! per-vertex page index; the rest are single-page HDR radiance.
//!
//! The top chip switches four modes. **Baked GI** lights the bake with a
//! directional key. **Emissive GI** swaps that for a glowing ceiling panel, so the
//! room is lit entirely by an area light the bake finds with area-light next-event
//! estimation (low-noise soft shading and soft contact shadows a directional light
//! cannot give). **Mixed** keeps the same baked lighting but leaves the realtime key
//! on and consumes the lightmap subtractively: a floating dynamic sphere casts a
//! realtime shadow across the baked floor without double-counting the baked direct
//! light. **Realtime only** is the flat comparison: the same room under one realtime
//! light with flat ambient, no bounce or baked occlusion. The side panel re-bakes at
//! different sample counts, toggles the denoiser (watch the noise return), shows the
//! baked atlas on a floating panel, and reports how many pages the multi-page hero
//! spilled into.

use crate::eframe::egui;
use glam::{Mat3, Mat4, Vec2, Vec3};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;
use viewport_lib as vpl;
use viewport_lib::wgpu;
use viewport_lib_lightbake::denoise::{DenoiseParams, denoise, dilate};
use viewport_lib_lightbake::encode::{Encoding, encode};
use viewport_lib_lightbake::stitch::{StitchGeometry, StitchParams, stitch};
use vpl::bake::TexelGeometry;
use vpl::raytrace::{RtLight, RtMaterial, RtScene, RtSettings, TexelSurfaces, Tracer};
use vpl::resources::{LightmapData, LightmapMode, TextureId};
use vpl::{
    BackfacePolicy, ItemSettings, LightKind, LightSource, Material, MeshData, MeshId, NodeId,
    ShadowFilter, primitives,
};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

// Requested unwrap resolution. xatlas packs into its own size (often larger than
// this), and each baked piece then bakes and uploads at that actual packed size
// (see `Piece::atlas_w`): the texel G-buffer, denoise, and texture must all match
// the size the piece's uv1 is normalised to, or the charts squash and read as
// blocky steps on the mesh.
const ATLAS: u32 = 512;

/// Texel-samples of GI the live bake submits per frame: one dispatch over
/// the whole atlas, with as many samples as fit. About 6 ms of GPU time on an
/// M4 Pro tracing in hardware, and about 5 ms with the compute traversal, so
/// a frame drawn during the bake is not held up behind a long solve; the bake
/// takes more frames instead. An atlas larger than the budget still takes one
/// sample a frame.
const HARDWARE_TEXEL_SAMPLES_PER_FRAME: u32 = 1_000_000;
const SOFTWARE_TEXEL_SAMPLES_PER_FRAME: u32 = 250_000;

/// Samples per dispatch for an atlas of `texels`, under the frame budget.
fn samples_per_dispatch(texels: u32, hardware: bool) -> u32 {
    let budget = if hardware {
        HARDWARE_TEXEL_SAMPLES_PER_FRAME
    } else {
        SOFTWARE_TEXEL_SAMPLES_PER_FRAME
    };
    (budget / texels.max(1)).clamp(1, 16)
}
/// Key-light direction (toward the light); shared by the bake and the realtime
/// mode so the two are directly comparable. Raked off vertical so the torus casts
/// a long, obvious shadow across the floor.
const LIGHT_DIR: Vec3 = Vec3::new(0.4, -0.32, 1.0);
const FLOOR_ALBEDO: [f32; 3] = [0.80, 0.80, 0.80];
const TORUS_ALBEDO: [f32; 3] = [0.82, 0.80, 0.72];
const RED: [f32; 3] = [0.85, 0.12, 0.10];
const GREEN: [f32; 3] = [0.15, 0.75, 0.20];
const BACK: [f32; 3] = [0.75, 0.75, 0.78];
const SPHERE_ALBEDO: [f32; 3] = [0.78, 0.80, 0.85];
const BOX_ALBEDO: [f32; 3] = [0.80, 0.72, 0.55];
const KNOT_ALBEDO: [f32; 3] = [0.72, 0.58, 0.82];

/// Modes. Baked GI, Emissive GI, and Mixed all path-trace a lightmap; Baked and
/// Mixed share the same bake (directional key), Emissive integrates a glowing panel
/// instead. Mixed then keeps the realtime light on and consumes the lightmap
/// subtractively. Realtime is the flat comparison.
const BAKED_MODE: usize = 0;
const EMISSIVE_MODE: usize = 1;
const MIXED_MODE: usize = 2;
const REALTIME_MODE: usize = 3;

/// Mixed mode: a floating dynamic sphere (not lightmapped) hovers over the room and
/// casts a realtime shadow onto the baked floor. Its transform.
fn dynamic_occluder_xf() -> Mat4 {
    Mat4::from_translation(Vec3::new(-1.5, -0.5, 4.0))
}

/// Emissive GI mode replaces the directional key with a glowing ceiling panel, so
/// the room is lit entirely by an area light: soft, directionless illumination and
/// soft contact shadows the directional key cannot produce. Tuned for the HDR
/// display path (linear radiance, tonemapped once).
const PANEL_RADIANCE: [f32; 3] = [6.0, 5.8, 5.2];
/// The panel geometry: a horizontal quad near the ceiling, centred over the room.
fn panel_xf() -> Mat4 {
    Mat4::from_translation(Vec3::new(0.0, 0.0, 6.4))
}

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
    /// Per-vertex atlas page from the unwrap. Empty for non-unwrapped pieces (all
    /// page 0). When `atlas_count > 1` this drives `set_lightmap_paged` so each
    /// vertex samples its own layer of the texture-array lightmap.
    pages: Vec<u32>,
    /// Number of atlas pages this piece's lightmap spans. 1 for the single-page
    /// hero objects; > 1 for the multi-page hero, whose charts spilled several
    /// pages and load as one texture array.
    atlas_count: u32,
    /// Optional tangent-space normal map. When set, the piece renders with it and
    /// its baked lightmap is directional (so the bumps respond to the baked light
    /// direction).
    normal_tex: Option<TextureId>,
    /// Cached bake outputs per atlas page, so denoise/encode can re-run without
    /// re-tracing. Single-page pieces have one entry; the multi-page hero has one
    /// per page (each page rasterises and bakes only its own charts).
    raw_irradiance: Vec<Vec<[f32; 4]>>,
    raw_direction: Vec<Vec<[f32; 4]>>,
    gbuf_pos: Vec<Vec<[f32; 4]>>,
    gbuf_nrm: Vec<Vec<[f32; 4]>>,
    /// When true, this piece does not get its own lightmap texture: its baked
    /// atlas is packed into a shared scene atlas and it is bound with
    /// `set_scene_lightmap` using `scene_layer` + `scene_scale_bias`. Used for the
    /// single-page non-directional heroes to demonstrate scene-level atlasing.
    scene_atlas: bool,
    /// Placement in the shared scene atlas (page layer + sub-rect transform), set
    /// by the packer during encode. Identity/0 until then.
    scene_scale_bias: [f32; 4],
    scene_layer: u32,
    /// Baked radiance atlas (linear HDR), sampled at binding 17. A single texture
    /// for single-page pieces, an N-layer array for the multi-page hero, or the
    /// shared scene atlas for `scene_atlas` pieces.
    tex: Option<TextureId>,
    /// Baked dominant-direction atlas (linear HDR), sampled at binding 18. Set
    /// only for normal-mapped pieces, which want the directional response.
    dir_tex: Option<TextureId>,
}

pub struct LightmapBakeShowcase {
    /// 0 = baked GI (directional), 1 = emissive GI (area light), 2 = realtime only.
    mode: usize,
    shown: Option<usize>,
    built: bool,
    pieces: Vec<Piece>,
    nodes: Vec<NodeId>,
    // Bake controls.
    samples: u32,
    denoise: bool,
    /// The samples and scene variant the shown lightmaps were baked for;
    /// `None` until the first bake lands.
    baked_target: Option<(u32, usize)>,
    /// The bake running across frames, if any.
    live: Option<LiveBake>,
    /// Smoothed frame time while no bake runs, to compare the bake's frames to.
    idle_frame_ms: f32,
    need_reencode: bool,
    applied: Option<(usize, bool, usize)>,
    /// Realtime shadow casting (Realtime-only mode). Off by default so the
    /// realtime view is flat and the bake's contribution is unambiguous.
    realtime_shadows: bool,
    // Floating atlas panel.
    atlas_mesh: Option<MeshId>,
    atlas_uv: Vec<Vec2>,
    show_atlas: bool,
    /// Which baked atlas the poster shows: an index into `atlas_sources()` (scene
    /// atlas pages, the cuboid's atlas, the multi-page torus's pages).
    atlas_view: usize,
    atlas_node: Option<NodeId>,
    // Stats for the panel.
    torus_charts: u32,
    /// Actual packed atlas size of the torus (xatlas picks it; it is usually not
    /// the requested resolution).
    torus_atlas: (u32, u32),
    /// Atlas pages the multi-page hero spilled into, and its per-page size.
    knot_pages: u32,
    knot_atlas: (u32, u32),
    /// The shared scene atlas the floor/torus/sphere pack into, and stats about
    /// it (how many objects, how many pages, page size).
    scene_atlas_tex: Option<TextureId>,
    scene_objects: u32,
    scene_pages: u32,
    scene_page_size: u32,
    bake_ms: u32,
    timings: BakeTimings,
    directionality: f32,
    request_rebake: bool,
    /// The glowing ceiling panel shown (and emitting) in Emissive GI mode.
    emissive_panel: Option<MeshId>,
    /// The floating dynamic sphere shown in Mixed mode, casting a realtime shadow
    /// onto the baked floor.
    dynamic_occluder: Option<MeshId>,
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
            baked_target: None,
            live: None,
            idle_frame_ms: 0.0,
            need_reencode: false,
            applied: None,
            realtime_shadows: false,
            atlas_mesh: None,
            atlas_uv: Vec::new(),
            show_atlas: false,
            atlas_view: 0,
            atlas_node: None,
            torus_charts: 0,
            torus_atlas: (ATLAS, ATLAS),
            knot_pages: 1,
            knot_atlas: (ATLAS, ATLAS),
            scene_atlas_tex: None,
            scene_objects: 0,
            scene_pages: 0,
            scene_page_size: 0,
            bake_ms: 0,
            timings: BakeTimings::default(),
            directionality: 0.0,
            request_rebake: false,
            emissive_panel: None,
            dynamic_occluder: None,
        }
    }

    /// True in the baked modes (Baked GI, Emissive GI, Mixed), false for Realtime.
    fn baked_mode(&self) -> bool {
        self.mode != REALTIME_MODE
    }

    /// Which scene variant the bake integrates against: Emissive uses the glowing
    /// panel (1), everything else the directional key (0). Baked and Mixed share
    /// variant 0, so switching between them reuses the cached bake with no re-trace.
    fn bake_scene_variant(&self) -> usize {
        if self.mode == EMISSIVE_MODE { 1 } else { 0 }
    }

    /// Build the ray-traced scene every bake integrates against: all pieces in
    /// world space (occluders), a key light, and a soft sky. Shared by the
    /// per-piece bake and the scene-atlas `bake_scene_prepared` call, so both see
    /// the same occluders and lighting.
    fn build_rt_scene(&self) -> RtScene {
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
                    base_colour: p.albedo.into(),
                    roughness: 0.9,
                    ..RtMaterial::default()
                },
            );
        }
        if self.mode == EMISSIVE_MODE {
            // Emissive GI: no analytic light. A glowing ceiling panel is the only
            // source, found by the bake's area-light NEE (LM-emis). Added as an
            // emissive mesh so it both lights the room and occludes.
            let panel = panel_mesh();
            let (wp, wn) = world_geo(&panel.positions, &panel.normals, panel_xf());
            scene.add_mesh(
                &wp,
                &panel.indices,
                Some(&wn),
                RtMaterial {
                    base_colour: [0.0, 0.0, 0.0].into(),
                    emissive: PANEL_RADIANCE.into(),
                    ..RtMaterial::default()
                },
            );
        } else {
            // Baked GI: a directional key. Tuned for the HDR display path: the baked
            // radiance feeds the renderer's tonemapper once (linear upload).
            scene.add_light(RtLight::Directional {
                direction: LIGHT_DIR.normalize().to_array(),
                colour: [2.1, 2.05, 1.9].into(),
            });
        }
        scene
    }

    fn rt_settings(&self) -> RtSettings {
        RtSettings {
            samples: self.samples,
            max_bounces: 4,
            denoise: false,
            seed: 0,
        }
    }

    /// Start a bake that runs across frames: the per-piece solves when `trace`
    /// (otherwise the cached raw atlases are cleaned up again), then the
    /// per-piece cleanup on a worker thread beside the scene atlas. A bake
    /// already running is dropped.
    fn start_bake(&mut self, ctx: &mut ShowcaseCtx, trace: bool) {
        // The cleanup stages spread over every core by default, which leaves
        // the thread drawing the frames and the driver's own threads waiting
        // for one. Keep two free while a bake runs beside the frames.
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        viewport_lib_lightbake::set_threads(cores.saturating_sub(2).max(1));
        self.timings = BakeTimings {
            unwrap_ms: self.timings.unwrap_ms,
            unwraps: self.timings.unwraps,
            ..Default::default()
        };
        if vpl::resources::build_log::enabled() {
            let _ = vpl::resources::build_log::drain();
        }
        let mut pending = VecDeque::new();
        if trace {
            for (i, p) in self.pieces.iter_mut().enumerate() {
                // Scene-atlas heroes are baked by the scene job, not here.
                if !p.baked || p.scene_atlas {
                    continue;
                }
                // Each atlas page bakes on its own: a page holds a disjoint set
                // of charts, so its texel G-buffer rasterises only that page's
                // triangles.
                let pages = p.atlas_count.max(1) as usize;
                p.raw_irradiance = vec![Vec::new(); pages];
                p.raw_direction = vec![Vec::new(); pages];
                p.gbuf_pos = vec![Vec::new(); pages];
                p.gbuf_nrm = vec![Vec::new(); pages];
                for page in 0..pages as u32 {
                    if !page_indices(&p.idx, &p.pages, page).is_empty() {
                        pending.push_back((i, page));
                    }
                }
            }
        }
        // The tracer and the G-buffer pipeline cost a few milliseconds to build
        // with a warm shader cache and most of a second without one, so they
        // are built on a worker thread rather than in this frame.
        let scene = self.build_rt_scene();
        let (device, queue) = (ctx.device.clone(), ctx.queue.clone());
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let t = Instant::now();
            let tracer = Tracer::new(&device, &queue, &scene);
            let gbuffer = std::sync::Arc::new(vpl::bake::TexelGBufferPass::new(&device));
            let _ = tx.send(BakeTools {
                tracer,
                gbuffer,
                ms: ms_since(t),
            });
        });
        self.live = Some(LiveBake {
            target: (self.samples, self.bake_scene_variant()),
            started: Instant::now(),
            piece_total: pending.len(),
            pending,
            tools_rx: Some(rx),
            tools: None,
            solving: None,
            uploads: VecDeque::new(),
            texture_jobs: Vec::new(),
            encode: None,
            encode_started: false,
            encoded: false,
            frame_ms: Vec::new(),
            scene: None,
            scene_started: Instant::now(),
            scene_done: false,
        });
    }

    /// Move the running bake on by one frame's worth of work. Nothing here
    /// waits on the GPU or on the cleanup thread.
    fn step_bake(&mut self, ctx: &mut ShowcaseCtx) {
        let Some(mut live) = self.live.take() else {
            return;
        };
        let frame_start = Instant::now();
        let (device, queue) = (ctx.device, ctx.queue);
        let settings = self.rt_settings();

        if let Some(rx) = &live.tools_rx
            && let Ok(tools) = rx.try_recv()
        {
            self.timings.tracer_ms = tools.ms;
            self.timings.tracers = 1;
            self.timings.hardware = tools.tracer.backend() == vpl::raytrace::RtBackend::Hardware;
            live.tools_rx = None;
            live.tools = Some(tools);
        }

        // The per-piece solves, one at a time: a texel G-buffer, then the GI
        // solve over it, a few dispatches a frame.
        if live.solving.is_none()
            && let Some(tools) = &live.tools
            && let Some((i, page)) = live.pending.pop_front()
        {
            let p = &self.pieces[i];
            let indices = page_indices(&p.idx, &p.pages, page);
            let uv1: Vec<[f32; 2]> = p.uv1.iter().map(|u| [u.x, u.y]).collect();
            let job = tools.gbuffer.begin(
                device,
                queue,
                &TexelGeometry {
                    positions: &p.pos,
                    normals: &p.nrm,
                    uv1: &uv1,
                    indices: &indices,
                    model: p.xf,
                },
                p.atlas_w,
                p.atlas_h,
            );
            live.solving = Some((i, page, PieceSolve::Gbuffer(job)));
        }
        if let Some((i, page, solve)) = live.solving.take() {
            live.solving = match solve {
                PieceSolve::Gbuffer(mut job) => match job.poll(device) {
                    Some(gbuf) => {
                        let tools = live.tools.as_ref().expect("solving needs the tools");
                        let hardware = tools.tracer.backend() == vpl::raytrace::RtBackend::Hardware;
                        let job = tools
                            .tracer
                            .begin_directional(device, &texel_surfaces(&gbuf), &settings)
                            .samples_per_dispatch(samples_per_dispatch(
                                gbuf.width * gbuf.height,
                                hardware,
                            ));
                        Some((i, page, PieceSolve::Trace(gbuf, job)))
                    }
                    None => Some((i, page, PieceSolve::Gbuffer(job))),
                },
                PieceSolve::Trace(gbuf, mut job) => {
                    job.step(device, queue, 1);
                    match job.poll(device) {
                        Some(bake) => {
                            let p = &mut self.pieces[i];
                            let k = page as usize;
                            p.raw_irradiance[k] = to_rgba4(&bake.irradiance);
                            p.raw_direction[k] = to_rgba4(&bake.direction);
                            p.gbuf_pos[k] = gbuf.world_pos;
                            p.gbuf_nrm[k] = gbuf.world_normal;
                            None
                        }
                        None => Some((i, page, PieceSolve::Trace(gbuf, job))),
                    }
                }
            };
            if live.solving.is_none() && live.pending.is_empty() {
                self.timings.trace_ms = ms_since(live.started);
            }
        }

        // Once the per-piece solves are in: their cleanup on a worker thread,
        // and the scene atlas beside it.
        let solves_done = live.solving.is_none() && live.pending.is_empty();
        if solves_done && live.tools.is_some() && !live.encode_started {
            let inputs: Vec<PieceInput> = self
                .pieces
                .iter()
                .enumerate()
                .filter(|(_, p)| p.baked && !p.raw_irradiance.is_empty())
                .map(|(i, p)| PieceInput::new(i, p))
                .collect();
            let denoise_on = self.denoise;
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let out: Vec<PieceEncoded> = inputs
                    .iter()
                    .map(|inp| encode_piece(inp, denoise_on))
                    .collect();
                // The bake may have been replaced meanwhile; then nobody listens.
                let _ = tx.send(out);
            });
            live.encode = Some(rx);
            live.encode_started = true;
            live.scene_started = Instant::now();
            let tools = live.tools.take().expect("checked above");
            live.scene = self.start_scene_job(ctx, tools, settings);
            live.scene_done = live.scene.is_none();
        }
        if let Some(rx) = &live.encode
            && let Ok(encoded) = rx.try_recv()
        {
            live.encode = None;
            live.uploads.extend(encoded);
        }
        // One piece's textures a frame, so the uploads do not all land at once.
        if let Some(piece) = live.uploads.pop_front() {
            self.upload_piece(ctx, &mut live, piece);
        }
        if !live.texture_jobs.is_empty() {
            let res = ctx.session.resources_mut();
            res.process_uploads(device, queue);
            let pieces = &mut self.pieces;
            live.texture_jobs
                .retain(|&(piece, direction, job)| match res.upload_status(job) {
                    vpl::resources::UploadStatus::Pending { .. } => true,
                    vpl::resources::UploadStatus::Ready => {
                        let tex = res.upload_result_texture(job).ok();
                        if direction {
                            pieces[piece].dir_tex = tex;
                        } else {
                            pieces[piece].tex = tex;
                        }
                        false
                    }
                    _ => false,
                });
            self.applied = None;
        }
        live.encoded = live.encode_started
            && live.encode.is_none()
            && live.uploads.is_empty()
            && live.texture_jobs.is_empty();
        if let Some(stage) = &mut live.scene
            && let Some(bake) = stage.job.step(&mut stage.passes)
        {
            self.timings.scene_ms = ms_since(live.scene_started);
            let pieces = std::mem::take(&mut stage.pieces);
            live.scene = None;
            live.scene_done = true;
            self.upload_scene_atlas(ctx, &pieces, bake);
        }

        let t = &mut self.timings;
        let work = ms_since(frame_start);
        t.frames += 1;
        t.work_ms += work;
        t.worst_work_ms = t.worst_work_ms.max(work);
        // The first frame's time is the frame before the bake started.
        if t.frames > 1 {
            t.worst_frame_ms = t.worst_frame_ms.max(ctx.dt * 1000.0);
            live.frame_ms.push(ctx.dt * 1000.0);
        }
        if live.encoded && live.scene_done {
            self.finish_bake(&live);
        } else {
            self.live = Some(live);
        }
    }

    /// The scene-atlas heroes' bake as a job, driven by the same polled GPU
    /// passes. `None` when no piece goes into the scene atlas.
    fn start_scene_job(
        &mut self,
        ctx: &mut ShowcaseCtx,
        tools: BakeTools,
        settings: RtSettings,
    ) -> Option<SceneStage> {
        let pieces: Vec<usize> = (0..self.pieces.len())
            .filter(|&i| self.pieces[i].scene_atlas && self.pieces[i].baked)
            .collect();
        if pieces.is_empty() {
            return None;
        }
        let uv1: Vec<Vec<[f32; 2]>> = pieces
            .iter()
            .map(|&i| self.pieces[i].uv1.iter().map(|u| [u.x, u.y]).collect())
            .collect();
        let prepared: Vec<viewport_lib_lightbake::PreparedObject> = pieces
            .iter()
            .zip(&uv1)
            .map(|(&i, uv1)| {
                let p = &self.pieces[i];
                viewport_lib_lightbake::PreparedObject {
                    positions: &p.pos,
                    normals: &p.nrm,
                    uv1,
                    indices: &p.idx,
                    width: p.atlas_w,
                    height: p.atlas_h,
                    model: p.xf.to_cols_array_2d(),
                }
            })
            .collect();
        // 1024 fits each hero's atlas (the torus packs to ~980), so no rect is
        // clamped; objects still spill to a second page, which the array handles.
        let job = viewport_lib_lightbake::SceneBakeJob::new(
            &prepared,
            &viewport_lib_lightbake::SceneBakeOptions {
                page_size: 1024,
                padding: 8,
                denoise: self.denoise,
                ..Default::default()
            },
        );
        self.timings.scene_objects = pieces.len() as u32;
        Some(SceneStage {
            job,
            passes: LivePasses {
                device: ctx.device.clone(),
                queue: ctx.queue.clone(),
                tracer: tools.tracer,
                gbuffer_pass: tools.gbuffer,
                settings,
                gbuffer: None,
                solve: None,
            },
            pieces,
        })
    }

    /// Upload one piece's atlases from the cleanup thread.
    fn upload_piece(&mut self, ctx: &mut ShowcaseCtx, live: &mut LiveBake, e: PieceEncoded) {
        let t = Instant::now();
        let (device, queue) = (ctx.device, ctx.queue);
        self.timings.directionality.0 += e.directionality.0;
        self.timings.directionality.1 += e.directionality.1;
        self.timings.cleanup_ms += e.stages.iter().sum::<f32>();
        self.timings.denoise_ms += e.stages[0];
        self.timings.stitch_ms += e.stages[1];
        self.timings.dilate_ms += e.stages[2];
        self.timings.encode_ms += e.stages[3];
        let res = ctx.session.resources_mut();
        // Single textures go through the upload jobs and are picked up on a
        // later frame (the synchronous upload waits for the job, which competes
        // with the cleanup threads for the CPU). The multi-page hero's texture
        // array is written directly.
        if let Some(dir) = e.direction {
            let job = res
                .begin_upload_texture(device, queue, vpl::TextureData::hdr(e.width, e.height, dir))
                .unwrap();
            live.texture_jobs.push((e.piece, true, job));
        }
        if e.pages > 1 {
            let tex = res
                .upload_texture_hdr_layers(device, queue, e.width, e.height, e.pages, &e.layers)
                .unwrap();
            self.pieces[e.piece].tex = Some(tex);
        } else {
            let job = res
                .begin_upload_texture(
                    device,
                    queue,
                    vpl::TextureData::hdr(e.width, e.height, e.layers),
                )
                .unwrap();
            live.texture_jobs.push((e.piece, false, job));
        }
        let (sum, n) = self.timings.directionality;
        self.directionality = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
        self.timings.upload_ms += ms_since(t);
        self.applied = None;
    }

    /// Upload the shared scene atlas and point its heroes at their placements.
    fn upload_scene_atlas(
        &mut self,
        ctx: &mut ShowcaseCtx,
        pieces: &[usize],
        bake: viewport_lib_lightbake::PreparedSceneBake,
    ) {
        let t = Instant::now();
        let tex = ctx
            .session
            .resources_mut()
            .upload_texture_hdr_layers(
                ctx.device,
                ctx.queue,
                bake.page_size,
                bake.page_size,
                bake.layers,
                &bake.radiance,
            )
            .unwrap();
        self.scene_atlas_tex = Some(tex);
        self.scene_objects = pieces.len() as u32;
        self.scene_pages = bake.layers;
        self.scene_page_size = bake.page_size;
        for (k, &i) in pieces.iter().enumerate() {
            let pl = bake.placements[k];
            let p = &mut self.pieces[i];
            p.tex = Some(tex);
            p.dir_tex = None;
            p.scene_scale_bias = pl.scale_bias;
            p.scene_layer = pl.layer;
        }
        self.timings.upload_ms += ms_since(t);
        self.applied = None;
    }

    fn finish_bake(&mut self, live: &LiveBake) {
        if self.baked_target.is_none() {
            // The scene was lit for realtime while there were no lightmaps;
            // rebuild it for the baked mode now that there are.
            self.shown = None;
        }
        self.baked_target = Some(live.target);
        self.built = true;
        self.bake_ms = ms_since(live.started) as u32;
        if vpl::resources::build_log::enabled() {
            let builds = vpl::resources::build_log::drain();
            self.timings.builds = builds.len() as u32;
            self.timings.build_ms = builds.iter().map(|(_, ms)| ms).sum();
        }
        let mut frames = live.frame_ms.clone();
        frames.sort_by(f32::total_cmp);
        self.timings.median_frame_ms = frames.get(frames.len() / 2).copied().unwrap_or(0.0);
        let t = self.timings;
        eprintln!(
            "lightmap bake: {} ms over {} frames | on this thread {:.0} ms, worst frame {:.1} ms | \
             frame time median {:.1} ms, worst {:.1} ms | unwrap {:.0} ms ({}x, at setup) | \
             tracer setup {:.0} ms ({}x, {}) | per-piece solves {:.0} ms | per-piece cleanup {:.0} ms \
             on a worker (denoise {:.0}, stitch {:.0}, dilate {:.0}, encode {:.0}) | scene atlas {:.0} ms \
             ({} objects) | upload {:.0} ms | builds {} ({:.0} ms)",
            self.bake_ms,
            t.frames,
            t.work_ms,
            t.worst_work_ms,
            t.median_frame_ms,
            t.worst_frame_ms,
            t.unwrap_ms,
            t.unwraps,
            t.tracer_ms,
            t.tracers,
            if t.hardware { "hardware" } else { "software" },
            t.trace_ms,
            t.cleanup_ms,
            t.denoise_ms,
            t.stitch_ms,
            t.dilate_ms,
            t.encode_ms,
            t.scene_ms,
            t.scene_objects,
            t.upload_ms,
            t.builds,
            t.build_ms,
        );
    }

    /// How far the running bake has got, as a fraction and a line of text.
    fn bake_progress(&self) -> Option<(f32, String)> {
        let live = self.live.as_ref()?;
        let solves_done =
            live.piece_total - live.pending.len() - usize::from(live.solving.is_some());
        let (scene_done, scene_total) = match &live.scene {
            Some(stage) => (stage.job.objects_baked(), stage.job.total()),
            None if live.scene_done => (1, 1),
            None => (0, 1),
        };
        let cleaned = usize::from(live.encoded);
        let fraction = (solves_done + scene_done + cleaned) as f32
            / (live.piece_total + scene_total + 1) as f32;
        Some((
            fraction,
            format!(
                "Baking: piece solves {solves_done}/{}, scene objects {scene_done}/{scene_total}",
                live.piece_total
            ),
        ))
    }

    /// Build (or rebuild) the scene nodes for the current mode.
    fn rebuild(&mut self, session: &mut vpl::ViewportInstance) {
        if !self.nodes.is_empty() {
            let ids = std::mem::take(&mut self.nodes);
            session.scene_mut().remove_many(&ids);
        }
        self.atlas_node = None;

        let baked_mode = self.baked_mode();
        // Until the first bake lands there are no lightmaps to lean on, so the
        // baked modes are lit the way the realtime one is.
        let lighting = if self.baked_target.is_none() {
            REALTIME_MODE
        } else {
            self.mode
        };

        // Lighting: pure-baked modes lean on the lightmaps (runtime lights off, dim
        // ambient for the walls); Mixed keeps the realtime key on so the dynamic
        // occluder casts a realtime shadow onto the baked floor; realtime mode
        // lights everything with one key light plus hemisphere ambient.
        {
            let l = &mut session.effects_mut().lighting;
            if lighting == BAKED_MODE || lighting == EMISSIVE_MODE {
                l.lights = Vec::new();
                l.hemisphere_intensity = 0.28;
                l.sky_colour = [0.5, 0.54, 0.62].into();
                l.ground_colour = [0.16, 0.16, 0.18].into();
            } else if lighting == MIXED_MODE {
                // The lightmap (consumed subtractively) carries the static lighting;
                // the realtime directional stays on. On lightmapped surfaces its
                // direct is baked (suppressed), but its realtime shadow darkens the
                // baked term, so the dynamic occluder casts onto the baked floor.
                let mut key = LightSource::default();
                key.kind = LightKind::Directional {
                    direction: LIGHT_DIR.to_array(),
                };
                key.colour = [1.0, 0.98, 0.95].into();
                key.intensity = 1.1;
                key.cast_shadows = true;
                l.lights = vec![key];
                l.hemisphere_intensity = 0.15;
                l.sky_colour = [0.5, 0.54, 0.62].into();
                l.ground_colour = [0.16, 0.16, 0.18].into();
                l.shadows.enabled = true;
                l.shadows.extent_override = Some(13.0);
                // Soft (PCSS) shadow with a wide penumbra, so the dynamic object's
                // realtime shadow reads as a soft contact shadow that blends with the
                // baked GI rather than a hard-edged blot on the curved baked heroes.
                l.shadows.filter = ShadowFilter::Pcss;
                l.shadows.pcss_light_radius = 0.05;
            } else {
                let mut key = LightSource::default();
                key.kind = LightKind::Directional {
                    direction: LIGHT_DIR.to_array(),
                };
                key.colour = [1.0, 0.98, 0.95].into();
                key.intensity = 1.1;
                // The light always keeps its shadow cascades; whether a shadow
                // actually appears is gated per-object below, so the toggle is
                // honoured reliably by the shadow pass.
                key.cast_shadows = true;
                l.lights = vec![key];
                // Low ambient so the realtime shadow (when the toggle is on) is
                // not washed out by fill light.
                l.hemisphere_intensity = 0.18;
                l.sky_colour = [0.6, 0.64, 0.72].into();
                l.ground_colour = [0.2, 0.2, 0.22].into();
                // Shadow rendering persists on the shared session across
                // showcases, so set it explicitly rather than assuming a prior
                // showcase left it on. Fit the shadow frustum to this room (auto
                // is 20, looser than the scene needs).
                l.shadows.enabled = true;
                l.shadows.extent_override = Some(13.0);
            }
        }

        for p in &self.pieces {
            // Every piece keeps its true albedo: the baked lightmap now stores
            // material-independent incident radiance (E/pi), and Replace mode
            // multiplies it by the material's base_colour (albedo).
            let mut mat = Material::pbr(p.albedo, 0.0, 0.9);
            mat.backface_policy = BackfacePolicy::Identical;
            // Normal-mapped pieces carry their map in both modes; combined with the
            // directional lightmap (baked mode) the bumps pick up the baked light.
            if let Some(nt) = p.normal_tex {
                mat.normal_map_id = Some(nt);
                mat.normal_strength = 1.0;
            }
            let id = session.scene_mut().add(Some(p.mesh), p.xf, mat);
            // Per-object cast-shadows: the shadow pass skips items with this off, so
            // the toggle reliably shows/hides the realtime shadow. Baked/Emissive
            // have no realtime light (inert). In Mixed the static pieces' shadows are
            // baked, so they must not also cast a realtime shadow (that would double
            // up); only the dynamic occluder casts.
            let mut ap = ItemSettings::default();
            ap.cast_shadows = self.realtime_shadows && self.mode != MIXED_MODE;
            session.scene_mut().set_appearance(id, ap);
            self.nodes.push(id);
        }

        // Mixed GI: a floating dynamic sphere, not lightmapped, lit fully in realtime
        // by the key light and casting a realtime shadow onto the baked floor. This
        // is the dynamic object a subtractive setup exists to support: baked static
        // GI plus a moving object that shadows it in realtime.
        if self.mode == MIXED_MODE {
            if let Some(sphere) = self.dynamic_occluder {
                let mut mat = Material::pbr([0.85, 0.85, 0.88], 0.0, 0.6);
                mat.backface_policy = BackfacePolicy::Identical;
                let id = session
                    .scene_mut()
                    .add(Some(sphere), dynamic_occluder_xf(), mat);
                let mut ap = ItemSettings::default();
                ap.cast_shadows = true;
                session.scene_mut().set_appearance(id, ap);
                self.nodes.push(id);
            }
        }

        // Emissive GI: show the glowing ceiling panel that lit the bake. It carries
        // the same radiance as the emitter in the trace scene (emissive material,
        // no lightmap), so the source of the soft lighting is visible. Not a shadow
        // caster or receiver: it is the light, not lit geometry.
        if self.mode == EMISSIVE_MODE {
            if let Some(panel) = self.emissive_panel {
                let mut mat = Material::pbr([0.0, 0.0, 0.0], 0.0, 1.0);
                mat.emissive = PANEL_RADIANCE.into();
                mat.backface_policy = BackfacePolicy::Identical;
                let id = session.scene_mut().add(Some(panel), panel_xf(), mat);
                let mut ap = ItemSettings::default();
                ap.cast_shadows = false;
                ap.receive_shadows = false;
                session.scene_mut().set_appearance(id, ap);
                self.nodes.push(id);
            }
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

    /// Every baked atlas the poster can show, as `(texture, layer, label)`: each
    /// page of the shared scene atlas, then each non-scene baked piece's own
    /// texture (the cuboid's directional atlas, and every page of the multi-page
    /// torus). The panel's page selector indexes this list.
    fn atlas_sources(&self) -> Vec<(TextureId, u32, String)> {
        let mut v = Vec::new();
        if let Some(tex) = self.scene_atlas_tex {
            for layer in 0..self.scene_pages.max(1) {
                v.push((tex, layer, format!("Scene atlas p{layer}")));
            }
        }
        for p in &self.pieces {
            if !p.baked || p.scene_atlas {
                continue;
            }
            let Some(tex) = p.tex else { continue };
            let pages = p.atlas_count.max(1);
            if pages > 1 {
                for layer in 0..pages {
                    v.push((tex, layer, format!("Multi-page p{layer}")));
                }
            } else {
                let label = if p.normal_tex.is_some() {
                    "Cuboid (directional)".to_string()
                } else {
                    "Object".to_string()
                };
                v.push((tex, 0, label));
            }
        }
        v
    }

    /// Attach or clear the baked lightmaps to match the current mode.
    fn apply_lightmaps(&mut self, ctx: &mut ShowcaseCtx) {
        let state = (self.mode, self.show_atlas, self.atlas_view);
        if self.applied == Some(state) {
            return;
        }
        let baked_mode = self.baked_mode();
        let device = ctx.device;

        // The atlas poster shows whichever baked atlas the page selector picks: a
        // scene-atlas page, the cuboid's directional atlas, or a page of the
        // multi-page torus. Each source is a (texture, layer) pair sampled full-quad.
        let sources = self.atlas_sources();
        let poster = sources
            .get(self.atlas_view.min(sources.len().saturating_sub(1)))
            .map(|&(tex, layer, _)| (tex, layer));

        // Mixed consumes the same baked atlas subtractively: the realtime key's
        // direct is suppressed on these static receivers (baked in) and its shadow
        // darkens the baked term. Baked/Emissive replace the indirect diffuse.
        let lm_mode = if self.mode == MIXED_MODE {
            LightmapMode::Subtractive
        } else {
            LightmapMode::Replace
        };

        let res = ctx.session.resources_mut();
        for p in &self.pieces {
            if !p.baked {
                continue;
            }
            match (baked_mode, p.tex) {
                (true, Some(tex)) => {
                    if p.scene_atlas {
                        // Scene atlas: many objects share `tex`; this object samples
                        // its packed layer + sub-rect.
                        let _ = res.set_scene_lightmap(
                            device,
                            p.mesh,
                            &p.uv1,
                            tex,
                            p.scene_layer,
                            p.scene_scale_bias,
                            lm_mode,
                        );
                    } else {
                        // Directional when a dominant-direction atlas was baked (the
                        // normal-mapped pieces), flat otherwise.
                        let data = match p.dir_tex {
                            Some(direction) => LightmapData::DominantDirection {
                                radiance: tex,
                                direction,
                            },
                            None => LightmapData::NonDirectional { radiance: tex },
                        };
                        if p.atlas_count > 1 {
                            // Multi-page: the lightmap is a texture array; each vertex
                            // carries the atlas page it was packed onto.
                            let _ = res.set_lightmap_paged(
                                device, p.mesh, &p.uv1, &p.pages, data, lm_mode,
                            );
                        } else {
                            let _ = res.set_lightmap(device, p.mesh, &p.uv1, data, lm_mode);
                        }
                    }
                }
                _ => {
                    let _ = res.clear_lightmap(p.mesh);
                }
            }
        }
        if let (Some(atlas), Some((tex, layer))) = (self.atlas_mesh, poster) {
            if baked_mode && self.show_atlas {
                // Sample the chosen atlas layer full-quad (identity scale/bias).
                let _ = res.set_scene_lightmap(
                    device,
                    atlas,
                    &self.atlas_uv,
                    tex,
                    layer,
                    [1.0, 1.0, 0.0, 0.0],
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
        self.baked_target = None;
        self.live = None;
        self.built = false;
        self.timings = BakeTimings::default();
        self.applied = None;
        self.need_reencode = false;

        let mut pieces = Vec::new();

        // Floor. The floor, torus, and sphere are the non-directional single-page
        // heroes; instead of each getting its own lightmap texture they are packed
        // into one shared scene atlas (see `encode_all`), so the render exercises
        // the scene-atlas load path (`set_scene_lightmap`) rather than one texture
        // per mesh. The directional cuboid and the multi-page torus keep their own
        // atlases (they use paths a shared atlas does not cover here).
        let floor = primitives::plane(20.0, 14.0);
        let mut floor_piece = make_piece(ctx, &floor, Mat4::IDENTITY, FLOOR_ALBEDO, true);
        floor_piece.scene_atlas = true;
        pieces.push(floor_piece);

        // A procedural bump normal map for the directional-lightmap demo.
        let bump = make_bump_normal_map(ctx, 512, 4.0);

        // Three hero objects of different topology, each UV-unwrapped with xatlas
        // so its lightmap UVs are unique. The torus is the seam-free hero (and
        // drives the atlas panel); the sphere is a smooth radiance hero; the
        // cuboid carries the bump normal map, so its baked lightmap is directional
        // and the bumps catch the raked light. The cuboid gets the normal map (not
        // the sphere) because its per-face charts meet at real geometric edges, so
        // the directional atlas has no smooth-surface seams to show.
        //
        // Icosphere, not a UV sphere: a UV sphere's pole collapses many triangles
        // to one point with degenerate UVs, leaving an uncovered (black) texel
        // patch at the pole. The icosphere has uniform triangles and no pole.
        //
        // Multi-page hero: a finely tessellated torus with enough chart area that
        // its unwrap spills across several atlas pages. Its lightmap loads as a
        // texture array and is sampled with a per-vertex page index : the case the
        // single-atlas heroes never reach.
        let torus = primitives::torus(1.9, 0.7, 64, 32);
        let sphere = primitives::icosphere(1.5, 4);
        let box_mesh = primitives::cuboid(2.4, 2.4, 2.4);
        let knot = primitives::torus(1.3, 0.5, 96, 48);

        // The four unwraps are independent xatlas runs, so they run together and
        // the batch takes about as long as the slowest (the fine torus).
        let mut unwrapped = unwrap_batch(
            &[
                (&torus, 0.0),
                (&sphere, 0.0),
                (&box_mesh, 0.0),
                (&knot, MULTIPAGE_START_DENSITY),
            ],
            &mut self.timings,
        )
        .into_iter();
        let mut next = || unwrapped.next().expect("one unwrap per hero");

        let (mut torus_piece, torus_charts) = build_piece_from_unwrap(
            ctx,
            &torus,
            Mat4::from_translation(Vec3::new(0.0, 1.0, 1.1)) * Mat4::from_rotation_x(0.35),
            TORUS_ALBEDO,
            None,
            next(),
        );
        self.torus_charts = torus_charts;
        self.torus_atlas = (torus_piece.atlas_w, torus_piece.atlas_h);
        torus_piece.scene_atlas = true;
        pieces.push(torus_piece);

        let (mut sphere_piece, _) = build_piece_from_unwrap(
            ctx,
            &sphere,
            Mat4::from_translation(Vec3::new(-4.6, -1.5, 1.5)),
            SPHERE_ALBEDO,
            None,
            next(),
        );
        sphere_piece.scene_atlas = true;
        pieces.push(sphere_piece);

        let (box_piece, _) = build_piece_from_unwrap(
            ctx,
            &box_mesh,
            Mat4::from_translation(Vec3::new(4.8, -1.2, 1.2)) * Mat4::from_rotation_z(0.5),
            BOX_ALBEDO,
            Some(bump),
            next(),
        );
        pieces.push(box_piece);

        let knot_unwrap = spill_to_pages(&knot, next(), &mut self.timings);
        let (knot_piece, _) = build_piece_from_unwrap(
            ctx,
            &knot,
            Mat4::from_translation(Vec3::new(0.0, -4.2, 1.35)) * Mat4::from_rotation_x(1.1),
            KNOT_ALBEDO,
            None,
            knot_unwrap,
        );
        self.knot_pages = knot_piece.atlas_count;
        self.knot_atlas = (knot_piece.atlas_w, knot_piece.atlas_h);
        pieces.push(knot_piece);

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

        // The Emissive-mode ceiling light mesh (shown as a glowing quad in that
        // mode).
        self.emissive_panel = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &panel_mesh())
                .unwrap(),
        );

        // The Mixed-mode dynamic occluder (a floating sphere that casts a realtime
        // shadow onto the baked floor).
        self.dynamic_occluder = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &primitives::icosphere(1.1, 3))
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

        if self.baked_mode() {
            // Re-trace on first entry, a rebake request, a sample-count change, or a
            // switch to a different bake scene variant (Emissive integrates a
            // different light than Baked/Mixed, so its cached bake is stale);
            // re-encode only (no tracing) when the denoiser is toggled. Baked and
            // Mixed share variant 0, so switching between them reuses the bake and
            // only re-consumes it (Replace vs Subtractive) in rebuild.
            let target = (self.samples, self.bake_scene_variant());
            let current = self.live.as_ref().map(|l| l.target).or(self.baked_target);
            if self.request_rebake || current != Some(target) {
                self.request_rebake = false;
                self.need_reencode = false;
                self.start_bake(ctx, true);
            } else if self.need_reencode {
                self.need_reencode = false;
                // A bake still running restarts with the new setting; a finished
                // one cleans up its cached raw atlases again.
                let trace = self.live.is_some();
                self.start_bake(ctx, trace);
            }
        }
        if self.live.is_some() {
            self.step_bake(ctx);
        } else {
            let ms = ctx.dt * 1000.0;
            self.idle_frame_ms = if self.idle_frame_ms == 0.0 {
                ms
            } else {
                self.idle_frame_ms * 0.95 + ms * 0.05
            };
        }

        self.apply_lightmaps(ctx);
        ctx.drive_camera();
    }

    fn description(&self) -> &str {
        match self.mode {
            BAKED_MODE => {
                "Baked GI: torus, sphere, cuboid, and a finely tessellated multi-page torus on a \
                 floor, all path-traced offline : unwrap, texel G-buffer, GI solve, denoise, \
                 seam-stitch, encode. HDR radiance, soft contact shadows, red/green colour bleed, \
                 and inter-object occlusion are baked in. The cuboid has a directional lightmap so \
                 its bump normal map catches the baked light direction; the front torus spilled its \
                 unwrap across several atlas pages and loads as a texture array; and the floor, \
                 large torus, and sphere are packed into one shared scene atlas (per-object layer + \
                 UV offset), so the scene bakes into a handful of atlases, not one per mesh (see \
                 Bake stats)."
            }
            EMISSIVE_MODE => {
                "Emissive GI: the directional key is replaced by a glowing ceiling panel, so the \
                 whole room is lit by an area light. The bake finds it with area-light next-event \
                 estimation, so it stays low-noise even at few samples : soft, directionless \
                 shading and soft contact shadows a single directional light cannot produce. Same \
                 unwrap, atlas, and encode path as Baked GI."
            }
            MIXED_MODE => {
                "Mixed: the same baked lighting as Baked GI, but the realtime key light stays on and \
                 the lightmap is consumed subtractively. On the baked static geometry the key's \
                 direct is already baked (so it is suppressed to avoid double counting), while its \
                 realtime shadow still darkens the baked term : the floating sphere is a dynamic, \
                 non-lightmapped object, lit fully in realtime and casting a real shadow across the \
                 baked floor. Baked GI and dynamic objects coexist (Unity Subtractive parity)."
            }
            _ => {
                "Realtime only: the same room lit by one realtime light and flat ambient. \
                 No bounce, no colour bleed, no baked occlusion : switch to Baked GI or Emissive \
                 GI to see what the offline solve adds."
            }
        }
    }

    fn has_controls(&self) -> bool {
        true
    }

    fn top_overlay(&mut self, ui: &mut egui::Ui) {
        if let Some(i) = crate::ui::segmented(
            ui,
            self.mode,
            &["Baked GI", "Emissive GI", "Mixed", "Realtime only"],
        ) {
            self.mode = i;
        }
        if let Some((fraction, text)) = self.bake_progress() {
            ui.add(
                egui::ProgressBar::new(fraction)
                    .desired_width(320.0)
                    .text(text),
            );
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

        ui.add_enabled_ui(self.baked_mode(), |ui| {
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
            if self.show_atlas {
                // Page selector: the poster shows one baked atlas layer at a time
                // (scene-atlas pages, the cuboid's atlas, the multi-page torus's
                // pages). Labels are collected first so the combo can mutate state.
                let labels: Vec<String> = self
                    .atlas_sources()
                    .into_iter()
                    .map(|(_, _, l)| l)
                    .collect();
                if !labels.is_empty() {
                    let cur = self.atlas_view.min(labels.len() - 1);
                    egui::ComboBox::from_label("Atlas page")
                        .selected_text(labels[cur].clone())
                        .show_ui(ui, |ui| {
                            for (i, label) in labels.iter().enumerate() {
                                if ui.selectable_label(cur == i, label).clicked() {
                                    self.atlas_view = i;
                                    self.applied = None; // re-bind the poster
                                }
                            }
                        });
                }
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
        ui.label(format!(
            "Multi-page hero: {} pages ({} x {} each)",
            self.knot_pages, self.knot_atlas.0, self.knot_atlas.1
        ));
        ui.label(format!(
            "Scene atlas: {} objects in {} page(s) ({}^2)",
            self.scene_objects, self.scene_pages, self.scene_page_size
        ));
        ui.label(format!(
            "Samples: {}",
            self.baked_target.map_or(0, |(samples, _)| samples)
        ));
        ui.label(format!("Bake time: {} ms", self.bake_ms));
        let t = &self.timings;
        ui.label(format!(
            "  over {} frames; worst frame {:.1} ms (idle {:.1})",
            t.frames, t.worst_frame_ms, self.idle_frame_ms
        ));
        ui.label(format!(
            "  bake work on this thread: {:.0} ms, worst frame {:.1} ms",
            t.work_ms, t.worst_work_ms
        ));
        ui.label(format!(
            "  unwrap (setup): {:.0} ms, {} calls",
            t.unwrap_ms, t.unwraps
        ));
        ui.label(format!(
            "  tracer setup: {:.0} ms, {} tracers ({})",
            t.tracer_ms,
            t.tracers,
            if t.hardware { "hardware" } else { "software" }
        ));
        ui.label(format!("  per-piece solves: {:.0} ms", t.trace_ms));
        ui.label(format!(
            "  per-piece cleanup (worker): {:.0} ms",
            t.cleanup_ms
        ));
        ui.label(format!(
            "    denoise {:.0}, stitch {:.0}, dilate {:.0}, encode {:.0}",
            t.denoise_ms, t.stitch_ms, t.dilate_ms, t.encode_ms
        ));
        ui.label(format!(
            "  scene atlas: {:.0} ms, {} objects",
            t.scene_ms, t.scene_objects
        ));
        ui.label(format!("  upload: {:.0} ms", t.upload_ms));
        if vpl::resources::build_log::enabled() {
            ui.label(format!(
                "  builds: {} pipelines/modules, {:.0} ms",
                t.builds, t.build_ms
            ));
        } else {
            ui.label("  builds: set VPL_BUILD_LOG to count");
        }
        ui.label(format!("Mean directionality: {:.2}", self.directionality));
        ui.add_space(8.0);
        ui.label(
            "Denoise off shows the raw Monte-Carlo noise; the atlas panel shows the \
             torus' baked lightmap in UV space.",
        );
    }
}

/// Where a bake's time went, shown in the side panel and printed when it
/// lands. The bake runs across frames, so the stages are wall-clock spans, and
/// the per-frame figures are what it cost the thread that draws.
#[derive(Default, Clone, Copy)]
struct BakeTimings {
    /// xatlas, at setup.
    unwrap_ms: f32,
    unwraps: u32,
    /// `Tracer::new`: kernel pipelines, scene upload and BVH.
    tracer_ms: f32,
    tracers: u32,
    /// Whether the solves traced in hardware.
    hardware: bool,
    /// From the start of the bake until the per-piece solves are in.
    trace_ms: f32,
    /// Per-piece cleanup on the worker thread, and its stages.
    cleanup_ms: f32,
    denoise_ms: f32,
    stitch_ms: f32,
    dilate_ms: f32,
    encode_ms: f32,
    /// The scene atlas job, from its start to its atlas.
    scene_ms: f32,
    scene_objects: u32,
    upload_ms: f32,
    /// Frames the bake ran over; its own work on this thread, in all and in the
    /// worst frame; the worst frame time while it ran.
    frames: u32,
    work_ms: f32,
    worst_work_ms: f32,
    worst_frame_ms: f32,
    median_frame_ms: f32,
    /// Directionality sum and count over the per-piece atlases.
    directionality: (f64, u64),
    /// Pipelines and shader modules built during the bake. Only counted when
    /// the build log is on (`VPL_BUILD_LOG`).
    builds: u32,
    build_ms: f32,
}

/// A bake running across frames.
struct LiveBake {
    /// The samples and scene variant it bakes for.
    target: (u32, usize),
    started: Instant,
    /// Per-piece solves not yet started, as (piece, atlas page), and how many
    /// there were.
    pending: VecDeque<(usize, u32)>,
    piece_total: usize,
    /// The tracer and G-buffer pipeline, on their way from the worker that
    /// builds them; used by the per-piece solves, then handed to the scene job.
    tools_rx: Option<mpsc::Receiver<BakeTools>>,
    tools: Option<BakeTools>,
    solving: Option<(usize, u32, PieceSolve)>,
    /// Cleaned pieces waiting to upload, one a frame, and texture uploads in
    /// flight as (piece, whether it is the direction atlas, job).
    uploads: VecDeque<PieceEncoded>,
    texture_jobs: Vec<(usize, bool, vpl::resources::JobId)>,
    /// Frame times while the bake ran.
    frame_ms: Vec<f32>,
    /// The per-piece cleanup, running on a worker thread.
    encode: Option<mpsc::Receiver<Vec<PieceEncoded>>>,
    encode_started: bool,
    /// The per-piece atlases are cleaned and uploaded.
    encoded: bool,
    scene: Option<SceneStage>,
    scene_started: Instant,
    scene_done: bool,
}

/// What the GPU passes are built from, made on a worker thread at the start of
/// a bake.
struct BakeTools {
    tracer: Tracer,
    gbuffer: std::sync::Arc<vpl::bake::TexelGBufferPass>,
    ms: f32,
}

/// One per-piece solve in flight: its G-buffer on the way back, then the GI
/// solve over it.
enum PieceSolve {
    Gbuffer(vpl::bake::TexelGBufferJob),
    Trace(vpl::bake::TexelGBuffer, vpl::raytrace::DirectionalBakeJob),
}

/// The scene-atlas job and the pieces it bakes, in its object order.
struct SceneStage {
    job: viewport_lib_lightbake::SceneBakeJob,
    passes: LivePasses,
    pieces: Vec<usize>,
}

/// The renderer's texel G-buffer and GI solve as polled passes for the scene
/// bake job: each starts a job and polls it, submitting a few dispatches of the
/// solve per poll, against a fixed occluder scene.
struct LivePasses {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tracer: Tracer,
    gbuffer_pass: std::sync::Arc<vpl::bake::TexelGBufferPass>,
    settings: RtSettings,
    gbuffer: Option<vpl::bake::TexelGBufferJob>,
    solve: Option<vpl::raytrace::DirectionalBakeJob>,
}

impl viewport_lib_lightbake::SceneBakeJobPasses for LivePasses {
    fn start_texel_gbuffer(
        &mut self,
        geom: &viewport_lib_lightbake::BakeGeometry<'_>,
        width: u32,
        height: u32,
    ) {
        self.gbuffer = Some(self.gbuffer_pass.begin(
            &self.device,
            &self.queue,
            &TexelGeometry {
                positions: geom.positions,
                normals: geom.normals,
                uv1: geom.uv1,
                indices: geom.indices,
                model: Mat4::from_cols_array_2d(&geom.model),
            },
            width,
            height,
        ));
    }

    fn poll_texel_gbuffer(&mut self) -> Option<viewport_lib_lightbake::TexelGbuffer> {
        let g = self.gbuffer.as_mut()?.poll(&self.device)?;
        self.gbuffer = None;
        Some(viewport_lib_lightbake::TexelGbuffer {
            width: g.width,
            height: g.height,
            world_pos: g.world_pos,
            world_normal: g.world_normal,
        })
    }

    fn start_solve_gi(&mut self, gbuffer: &viewport_lib_lightbake::TexelGbuffer) {
        let surfaces = TexelSurfaces {
            width: gbuffer.width,
            height: gbuffer.height,
            world_pos: &gbuffer.world_pos,
            world_normal: &gbuffer.world_normal,
        };
        let hardware = self.tracer.backend() == vpl::raytrace::RtBackend::Hardware;
        self.solve = Some(
            self.tracer
                .begin_directional(&self.device, &surfaces, &self.settings)
                .samples_per_dispatch(samples_per_dispatch(
                    gbuffer.width * gbuffer.height,
                    hardware,
                )),
        );
    }

    fn poll_solve_gi(&mut self) -> Option<viewport_lib_lightbake::GiBake> {
        let job = self.solve.as_mut()?;
        job.step(&self.device, &self.queue, 1);
        let bake = job.poll(&self.device)?;
        self.solve = None;
        Some(viewport_lib_lightbake::GiBake {
            irradiance: bake.irradiance,
        })
    }
}

/// What the cleanup thread needs from one piece, copied so it can run there.
struct PieceInput {
    piece: usize,
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    uv1: Vec<Vec2>,
    idx: Vec<u32>,
    pages: Vec<u32>,
    width: u32,
    height: u32,
    page_count: u32,
    raw_irradiance: Vec<Vec<[f32; 4]>>,
    raw_direction: Vec<Vec<[f32; 4]>>,
    gbuf_nrm: Vec<Vec<[f32; 4]>>,
    gbuf_pos: Vec<Vec<[f32; 4]>>,
    /// Normal-mapped pieces get the directional atlas too.
    directional: bool,
}

impl PieceInput {
    fn new(piece: usize, p: &Piece) -> Self {
        Self {
            piece,
            pos: p.pos.clone(),
            nrm: p.nrm.clone(),
            uv1: p.uv1.clone(),
            idx: p.idx.clone(),
            pages: p.pages.clone(),
            width: p.atlas_w,
            height: p.atlas_h,
            page_count: p.atlas_count.max(1),
            raw_irradiance: p.raw_irradiance.clone(),
            raw_direction: p.raw_direction.clone(),
            gbuf_nrm: p.gbuf_nrm.clone(),
            gbuf_pos: p.gbuf_pos.clone(),
            directional: p.normal_tex.is_some(),
        }
    }
}

/// One piece's atlases, ready to upload.
struct PieceEncoded {
    piece: usize,
    width: u32,
    height: u32,
    pages: u32,
    /// Linear diffuse radiance (E / pi), layer-major across pages.
    layers: Vec<f32>,
    /// Dominant direction xyz + directionality w, for a directional piece.
    direction: Option<Vec<f32>>,
    /// Directionality sum and count over covered texels.
    directionality: (f64, u64),
    /// Denoise, stitch, dilate and encode, in ms.
    stages: [f32; 4],
}

/// Denoise (or not), stitch, dilate and encode one piece's raw bake. Runs on
/// the cleanup thread.
fn encode_piece(input: &PieceInput, denoise_on: bool) -> PieceEncoded {
    let (aw, ah) = (input.width, input.height);
    let atlas_count = input.page_count;
    let inv_pi = 1.0 / std::f32::consts::PI;
    let page_texels = (aw * ah) as usize;
    let p = atlas_count as usize;
    let mut stages = [0.0f32; 4];

    // Denoise is optional and local, so it runs per page on that page's own
    // atlas. Dilation is not optional; it comes after the stitch below.
    let t = Instant::now();
    let mut denoised_pages: Vec<Vec<[f32; 4]>> = Vec::with_capacity(p);
    for page in 0..p {
        let raw = &input.raw_irradiance[page];
        if raw.is_empty() {
            denoised_pages.push(vec![[0.0f32; 4]; page_texels]);
            continue;
        }
        denoised_pages.push(if denoise_on {
            denoise(
                raw,
                &input.gbuf_pos[page],
                &input.gbuf_nrm[page],
                aw,
                ah,
                &DenoiseParams::default(),
            )
        } else {
            raw.clone()
        });
    }
    stages[0] = ms_since(t);

    // Stitch every chart seam at once, including cuts whose two charts landed
    // on different atlas pages. Stack the pages into one tall atlas (page k ->
    // vertical band k) and shift each vertex's UV into its band; stitch welds
    // the two sides of a cut by 3D position, so a cross-page cut reconciles
    // exactly like a within-page one. Single-page pieces stack to themselves.
    let t = Instant::now();
    let stacked: Vec<[f32; 4]> = denoised_pages.concat();
    let inv_p = 1.0 / p as f32;
    let uv_stacked: Vec<[f32; 2]> = input
        .uv1
        .iter()
        .enumerate()
        .map(|(v, u)| {
            let page = input
                .pages
                .get(v)
                .copied()
                .unwrap_or(0)
                .min(atlas_count - 1) as f32;
            [u.x, (u.y + page) * inv_p]
        })
        .collect();
    let stitched = stitch(
        &stacked,
        aw,
        ah * atlas_count,
        &StitchGeometry {
            positions: &input.pos,
            uv1: &uv_stacked,
            indices: &input.idx,
            normals: Some(&input.nrm),
        },
        &StitchParams::default(),
    );
    stages[1] = ms_since(t);

    // Per page: dilate its band into the gutter, encode, and write its layer.
    let mut layers = vec![0.0f32; page_texels * 4 * p];
    let mut direction = None;
    let (mut dir_sum, mut dir_n) = (0.0f64, 0u64);
    for page in 0..p {
        if input.raw_irradiance[page].is_empty() {
            continue; // leaves this layer zeroed (no charts on this page)
        }
        let band = &stitched[page * page_texels..(page + 1) * page_texels];
        let t = Instant::now();
        let cleaned = dilate(band, aw, ah, 6);
        stages[2] += ms_since(t);
        // Encode into the neutral directional lightmap (exercises the encoder
        // and yields the directionality stat); the display samples the
        // radiance channel.
        let t = Instant::now();
        let lm = encode(
            aw,
            ah,
            &cleaned,
            Some(&input.raw_direction[page]),
            &input.gbuf_nrm[page],
            Encoding::DominantDirection,
        );
        stages[3] += ms_since(t);
        if let Some(dir) = lm.direction() {
            for d in dir {
                if d[3] > 0.0 {
                    dir_sum += d[3] as f64;
                    dir_n += 1;
                }
            }
        }
        // Linear diffuse radiance (incident irradiance / pi). The material keeps
        // the true albedo, so Replace mode (base_colour * lm.rgb) gives
        // albedo * E/pi; the renderer's HDR pipeline tonemaps once for display.
        let base = page * page_texels * 4;
        for (t, px) in lm.radiance().iter().enumerate() {
            if px[3] <= 0.5 {
                continue;
            }
            layers[base + t * 4] = px[0] * inv_pi;
            layers[base + t * 4 + 1] = px[1] * inv_pi;
            layers[base + t * 4 + 2] = px[2] * inv_pi;
            layers[base + t * 4 + 3] = 1.0;
        }

        // Normal-mapped pieces get the directional atlas too, so the bumps
        // respond to where the baked light came from. These are single-page.
        if atlas_count == 1
            && input.directional
            && let Some(dir) = lm.direction()
        {
            // Dilate the direction atlas into the gutter using the radiance
            // coverage as the mask (its own w is directionality, not coverage),
            // so bilinear at a chart edge reads a real direction.
            let covered: Vec<bool> = cleaned.iter().map(|c| c[3] > 0.5).collect();
            let dir = dilate_masked(dir, &covered, aw as usize, ah as usize, 6);
            direction = Some(dir.iter().flatten().copied().collect());
        }
    }
    PieceEncoded {
        piece: input.piece,
        width: aw,
        height: ah,
        pages: atlas_count,
        layers,
        direction,
        directionality: (dir_sum, dir_n),
        stages,
    }
}

fn texel_surfaces(g: &vpl::bake::TexelGBuffer) -> TexelSurfaces<'_> {
    TexelSurfaces {
        width: g.width,
        height: g.height,
        world_pos: &g.world_pos,
        world_normal: &g.world_normal,
    }
}

fn ms_since(t: std::time::Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
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
        pages: Vec::new(),
        atlas_count: 1,
        normal_tex: None,
        raw_irradiance: Vec::new(),
        raw_direction: Vec::new(),
        gbuf_pos: Vec::new(),
        gbuf_nrm: Vec::new(),
        scene_atlas: false,
        scene_scale_bias: [1.0, 1.0, 0.0, 0.0],
        scene_layer: 0,
        tex: None,
        dir_tex: None,
    }
}

/// Unwrap options for a hero: a fixed `ATLAS` page size and `texels_per_unit`
/// density (0 lets xatlas estimate a density that fits one page). A fixed page
/// size plus a high density makes the charts overflow one page and spill onto
/// more, which is how the multi-page hero is produced.
fn hero_unwrap_options(texels_per_unit: f32) -> viewport_lib_lightbake::UnwrapOptions {
    viewport_lib_lightbake::UnwrapOptions {
        resolution: ATLAS,
        texels_per_unit,
        padding: 6,
        ..Default::default()
    }
}

/// Unwrap several primitives at once, each at its own density, results in
/// order.
fn unwrap_batch(
    meshes: &[(&MeshData, f32)],
    timings: &mut BakeTimings,
) -> Vec<viewport_lib_lightbake::UnwrapResult> {
    let jobs: Vec<_> = meshes
        .iter()
        .map(|(mesh, tpu)| {
            (
                viewport_lib_lightbake::UnwrapInput {
                    positions: &mesh.positions,
                    normals: Some(&mesh.normals),
                    indices: &mesh.indices,
                },
                hero_unwrap_options(*tpu),
            )
        })
        .collect();
    let t = std::time::Instant::now();
    let results = viewport_lib_lightbake::unwrap_many(&jobs)
        .into_iter()
        .map(|r| r.expect("unwrap piece"))
        .collect();
    timings.unwrap_ms += ms_since(t);
    timings.unwraps += jobs.len() as u32;
    results
}

/// Starting density for the multi-page hero. The pages are the same full size
/// as every other hero (`ATLAS`) and the density starts high, so each page is as
/// sharp as the single-page bakes; only the page count is contrived.
const MULTIPAGE_START_DENSITY: f32 = 48.0;

/// Make sure the multi-page hero really spans two or more pages: starting from
/// its unwrap at [`MULTIPAGE_START_DENSITY`], raise the density until the charts
/// no longer fit one page and xatlas spills onto more.
fn spill_to_pages(
    mesh: &MeshData,
    first: viewport_lib_lightbake::UnwrapResult,
    timings: &mut BakeTimings,
) -> viewport_lib_lightbake::UnwrapResult {
    let mut tpu = MULTIPAGE_START_DENSITY;
    let mut u = first;
    while u.atlas_count < 2 && tpu < 320.0 {
        tpu *= 1.3;
        u = unwrap_batch(&[(mesh, tpu)], timings).remove(0);
    }
    u
}

/// Build a baked [`Piece`] from an unwrap result. `normal_tex` opts the piece
/// into normal mapping + a directional lightmap. Carries the per-vertex atlas
/// page so a multi-page unwrap (`atlas_count > 1`) loads as a texture array.
fn build_piece_from_unwrap(
    ctx: &mut SetupCtx,
    mesh: &MeshData,
    xf: Mat4,
    albedo: [f32; 3],
    normal_tex: Option<TextureId>,
    unwrapped: viewport_lib_lightbake::UnwrapResult,
) -> (Piece, u32) {
    let charts = unwrapped.chart_count;
    // Carry the art UV0 onto the re-indexed mesh (gathered by xref) so normal
    // mapping still has texture coordinates after the unwrap split the vertices.
    let uv0: Option<Vec<[f32; 2]>> = mesh
        .uvs
        .as_ref()
        .map(|uvs| unwrapped.xref.iter().map(|&x| uvs[x as usize]).collect());
    let id = build_mesh_uv0(
        ctx,
        &unwrapped.positions,
        &unwrapped.normals,
        uv0.as_deref(),
        &unwrapped.indices,
    );
    // An unassigned vertex (u32::MAX) sits on no page; clamp it to page 0 so it
    // never indexes past the array (it is degenerate and not visibly shaded).
    let pages: Vec<u32> = unwrapped
        .atlas_index
        .iter()
        .map(|&a| if a == u32::MAX { 0 } else { a })
        .collect();
    let piece = Piece {
        mesh: id,
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
        xf,
        albedo,
        baked: true,
        pages,
        atlas_count: unwrapped.atlas_count.max(1),
        normal_tex,
        raw_irradiance: Vec::new(),
        raw_direction: Vec::new(),
        gbuf_pos: Vec::new(),
        gbuf_nrm: Vec::new(),
        scene_atlas: false,
        scene_scale_bias: [1.0, 1.0, 0.0, 0.0],
        scene_layer: 0,
        tex: None,
        dir_tex: None,
    };
    (piece, charts)
}

/// Indices of the triangles whose vertices sit on atlas `page`. All three
/// vertices of a triangle share a page (charts do not split across pages), so
/// testing the first vertex is enough. An empty `pages` slice (non-unwrapped
/// piece) means one page holding every triangle.
fn page_indices(idx: &[u32], pages: &[u32], page: u32) -> Vec<u32> {
    if pages.is_empty() {
        return idx.to_vec();
    }
    idx.chunks_exact(3)
        .filter(|t| pages[t[0] as usize] == page)
        .flat_map(|t| t.iter().copied())
        .collect()
}

/// Build a mesh from raw arrays (used for the unwrapped, re-indexed torus).
fn build_mesh_uv0(
    ctx: &mut SetupCtx,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uv0: Option<&[[f32; 2]]>,
    indices: &[u32],
) -> MeshId {
    let mut m = MeshData::default();
    m.positions = positions.to_vec();
    m.normals = normals.to_vec();
    m.indices = indices.to_vec();
    // Art UV0 (for normal mapping); the lightmap UV1 rides `set_lightmap`, not
    // the mesh. Tangents are auto-computed from UV0 when present.
    m.uvs = uv0.map(|u| u.to_vec());
    ctx.session
        .resources_mut()
        .upload_mesh_data(ctx.device, &m)
        .unwrap()
}

/// A procedural tangent-space normal map: a grid of rounded bumps. Demonstrates
/// the directional lightmap, since each bump's slopes face different directions
/// and pick up the baked dominant light accordingly.
fn make_bump_normal_map(ctx: &mut SetupCtx, size: u32, bumps: f32) -> TextureId {
    let n = (size * size) as usize;
    let mut rgba = vec![0u8; n * 4];
    let tau = std::f32::consts::TAU;
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            // Height field of rounded bumps; slope gives the tangent-space normal.
            let amp = 0.3;
            let dhdu = amp * bumps * tau * (u * bumps * tau).cos() * (v * bumps * tau).sin();
            let dhdv = amp * bumps * tau * (u * bumps * tau).sin() * (v * bumps * tau).cos();
            let nrm = glam::Vec3::new(-dhdu, -dhdv, 1.0).normalize();
            let i = ((y * size + x) * 4) as usize;
            rgba[i] = ((nrm.x * 0.5 + 0.5) * 255.0) as u8;
            rgba[i + 1] = ((nrm.y * 0.5 + 0.5) * 255.0) as u8;
            rgba[i + 2] = ((nrm.z * 0.5 + 0.5) * 255.0) as u8;
            rgba[i + 3] = 255;
        }
    }
    ctx.session
        .resources_mut()
        .upload_texture(
            ctx.device,
            ctx.queue,
            vpl::TextureData::normal_map(size, size, rgba.to_vec()),
        )
        .unwrap()
}

/// Grow `src` into texels that are uncovered but neighbour a covered one,
/// averaging covered neighbours. `covered` is an external coverage mask (the
/// radiance coverage), used because the direction atlas' own `w` is
/// directionality, not coverage.
fn dilate_masked(
    src: &[[f32; 4]],
    covered: &[bool],
    w: usize,
    h: usize,
    iters: u32,
) -> Vec<[f32; 4]> {
    let mut cur = src.to_vec();
    let mut cov = covered.to_vec();
    for _ in 0..iters {
        let mut nxt = cur.clone();
        let mut ncov = cov.clone();
        for y in 0..h {
            for x in 0..w {
                let ci = y * w + x;
                if cov[ci] {
                    continue;
                }
                let mut acc = [0.0f32; 4];
                let mut cnt = 0.0f32;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (sx, sy) = (x as i32 + dx, y as i32 + dy);
                    if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                        continue;
                    }
                    let si = sy as usize * w + sx as usize;
                    if cov[si] {
                        for c in 0..4 {
                            acc[c] += cur[si][c];
                        }
                        cnt += 1.0;
                    }
                }
                if cnt > 0.0 {
                    for c in 0..4 {
                        nxt[ci][c] = acc[c] / cnt;
                    }
                    ncov[ci] = true;
                }
            }
        }
        cur = nxt;
        cov = ncov;
    }
    cur
}

/// The Emissive-mode ceiling light: a horizontal quad. One mesh definition, used
/// both as the emissive occluder in the trace scene and as the glowing node in the
/// rendered scene, so the two stay in sync.
fn panel_mesh() -> MeshData {
    primitives::plane(6.0, 4.0)
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
