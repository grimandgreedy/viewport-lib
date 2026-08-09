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
//! The top chip switches three modes. **Baked GI** lights the bake with a
//! directional key. **Emissive GI** swaps that for a glowing ceiling panel, so the
//! room is lit entirely by an area light the bake finds with area-light next-event
//! estimation (low-noise soft shading and soft contact shadows a directional light
//! cannot give). **Realtime only** is the flat comparison: the same room under one
//! realtime light with flat ambient, no bounce or baked occlusion. The side panel
//! re-bakes at different sample counts, toggles the denoiser (watch the noise
//! return), shows the baked atlas on a floating panel, and reports how many pages
//! the multi-page hero spilled into.

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
const SPHERE_ALBEDO: [f32; 3] = [0.78, 0.80, 0.85];
const BOX_ALBEDO: [f32; 3] = [0.80, 0.72, 0.55];
const KNOT_ALBEDO: [f32; 3] = [0.72, 0.58, 0.82];

/// Modes. Baked GI and Emissive GI both path-trace a lightmap; they differ only in
/// the light source the bake integrates against. Realtime is the flat comparison.
const BAKED_MODE: usize = 0;
const EMISSIVE_MODE: usize = 1;
const REALTIME_MODE: usize = 2;

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
    baked_at: Option<u32>,
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
    directionality: f32,
    request_rebake: bool,
    /// The glowing ceiling panel shown (and emitting) in Emissive GI mode.
    emissive_panel: Option<MeshId>,
    /// Which mode's scene the cached bake was traced against (Baked vs Emissive
    /// integrate different lights), so a switch between them forces a re-trace.
    baked_kind: Option<usize>,
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
            directionality: 0.0,
            request_rebake: false,
            emissive_panel: None,
            baked_kind: None,
        }
    }

    /// True in the two baked modes (Baked GI, Emissive GI), false for Realtime.
    fn baked_mode(&self) -> bool {
        self.mode != REALTIME_MODE
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
                    base_colour: p.albedo,
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
                    base_colour: [0.0, 0.0, 0.0],
                    emissive: PANEL_RADIANCE,
                    ..RtMaterial::default()
                },
            );
        } else {
            // Baked GI: a directional key. Tuned for the HDR display path: the baked
            // radiance feeds the renderer's tonemapper once (linear upload).
            scene.add_light(RtLight::Directional {
                direction: LIGHT_DIR.normalize().to_array(),
                colour: [2.1, 2.05, 1.9],
            });
        }
        scene
    }

    /// Trace the shared room and, for each baked piece, path-trace its lightmap
    /// and cache the raw irradiance + direction atlases (no denoise/encode yet).
    fn trace_all(&mut self, ctx: &mut ShowcaseCtx) {
        let device = ctx.device;
        let queue = ctx.queue;

        // The scene every bake traces against: all pieces in world space (every
        // piece is an occluder, including the scene-atlas heroes), plus a key light
        // and a soft sky.
        let scene = self.build_rt_scene();

        let settings = RtSettings {
            samples: self.samples,
            max_bounces: 4,
            denoise: false,
            seed: 0,
        };
        for i in 0..self.pieces.len() {
            // Scene-atlas heroes are baked in one `bake_scene_prepared` call in
            // encode_all (which owns their gbuffer + GI), not here.
            if !self.pieces[i].baked || self.pieces[i].scene_atlas {
                continue;
            }
            let (pos, nrm, uv1, idx, pages, xf, aw, ah, atlas_count) = {
                let p = &self.pieces[i];
                (
                    p.pos.clone(),
                    p.nrm.clone(),
                    p.uv1.iter().map(|u| [u.x, u.y]).collect::<Vec<_>>(),
                    p.idx.clone(),
                    p.pages.clone(),
                    p.xf,
                    p.atlas_w,
                    p.atlas_h,
                    p.atlas_count.max(1),
                )
            };
            // Bake each atlas page on its own: a page holds a disjoint set of
            // charts (all three vertices of a triangle share a page), so its texel
            // G-buffer must rasterise only that page's triangles. Single-page
            // pieces run this loop once over the whole mesh.
            let mut irr_pages = Vec::with_capacity(atlas_count as usize);
            let mut dir_pages = Vec::with_capacity(atlas_count as usize);
            let mut gp_pages = Vec::with_capacity(atlas_count as usize);
            let mut gn_pages = Vec::with_capacity(atlas_count as usize);
            for page in 0..atlas_count {
                let idx_k = page_indices(&idx, &pages, page);
                if idx_k.is_empty() {
                    irr_pages.push(Vec::new());
                    dir_pages.push(Vec::new());
                    gp_pages.push(Vec::new());
                    gn_pages.push(Vec::new());
                    continue;
                }
                let gbuf = rasterize_texel_gbuffer(
                    device,
                    queue,
                    &TexelGeometry {
                        positions: &pos,
                        normals: &nrm,
                        uv1: &uv1,
                        indices: &idx_k,
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
                irr_pages.push(to_rgba4(&bake.irradiance));
                dir_pages.push(to_rgba4(&bake.direction));
                gp_pages.push(gbuf.world_pos);
                gn_pages.push(gbuf.world_normal);
            }
            let p = &mut self.pieces[i];
            p.raw_irradiance = irr_pages;
            p.raw_direction = dir_pages;
            p.gbuf_pos = gp_pages;
            p.gbuf_nrm = gn_pages;
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
            // Scene-atlas heroes are skipped here; they are baked together below
            // via bake_scene_prepared. Non-scene pieces (whose raw was cached by
            // trace_all) run the per-page cleanup + upload path.
            if !self.pieces[i].baked || self.pieces[i].raw_irradiance.is_empty() {
                continue;
            }
            let (aw, ah) = (self.pieces[i].atlas_w, self.pieces[i].atlas_h);
            let atlas_count = self.pieces[i].atlas_count.max(1);
            let idx = self.pieces[i].idx.clone();
            let pages = self.pieces[i].pages.clone();
            let inv_pi = 1.0 / std::f32::consts::PI;
            let page_texels = (aw * ah) as usize;
            let p = atlas_count as usize;

            // Denoise is optional (the toggle) and local, so it runs per page on
            // that page's own atlas. Dilation is not optional; it comes after the
            // stitch below.
            let mut denoised_pages: Vec<Vec<[f32; 4]>> = Vec::with_capacity(p);
            for page in 0..p {
                let raw = &self.pieces[i].raw_irradiance[page];
                if raw.is_empty() {
                    denoised_pages.push(vec![[0.0f32; 4]; page_texels]);
                    continue;
                }
                let d = if self.denoise {
                    denoise(
                        raw,
                        &self.pieces[i].gbuf_pos[page],
                        &self.pieces[i].gbuf_nrm[page],
                        aw,
                        ah,
                        &DenoiseParams::default(),
                    )
                } else {
                    raw.clone()
                };
                denoised_pages.push(d);
            }

            // Stitch every chart seam at once, including cuts whose two charts
            // landed on different atlas pages. Stack the pages into one tall atlas
            // (page k -> vertical band k) and shift each vertex's UV into its band;
            // stitch welds the two sides of a cut by 3D position, so a cross-page
            // cut reconciles exactly like a within-page one. Stitching each page in
            // isolation would leave those cross-page seams visible. Single-page
            // pieces stack to themselves (band 0), so this is a no-op for them.
            let stacked: Vec<[f32; 4]> = denoised_pages.concat();
            let inv_p = 1.0 / p as f32;
            let uv_stacked: Vec<[f32; 2]> = self.pieces[i]
                .uv1
                .iter()
                .enumerate()
                .map(|(v, u)| {
                    let page = pages.get(v).copied().unwrap_or(0).min(atlas_count - 1) as f32;
                    [u.x, (u.y + page) * inv_p]
                })
                .collect();
            let stitched = stitch(
                &stacked,
                aw,
                ah * atlas_count,
                &StitchGeometry {
                    positions: &self.pieces[i].pos,
                    uv1: &uv_stacked,
                    indices: &idx,
                },
                &StitchParams::default(),
            );

            // Per page: dilate its band into the gutter, encode, and write its
            // layer. Radiance is concatenated layer-major (page 0, then page 1, ...)
            // so a multi-page piece uploads as one texture array; single-page
            // pieces produce one layer.
            let mut layers = vec![0.0f32; page_texels * 4 * p];
            // The directional atlas is only uploaded for the (single-page) normal-
            // mapped pieces; the multi-page hero is non-directional.
            let mut dir_tex: Option<TextureId> = None;
            for page in 0..p {
                if self.pieces[i].raw_irradiance[page].is_empty() {
                    continue; // leaves this layer zeroed (no charts on this page)
                }
                let band = &stitched[page * page_texels..(page + 1) * page_texels];
                let cleaned = dilate(band, aw, ah, 6);
                // Encode into the neutral directional lightmap (exercises the
                // encoder and yields the directionality stat); the display samples
                // the radiance channel.
                let lm = encode(
                    aw,
                    ah,
                    &cleaned,
                    Some(&self.pieces[i].raw_direction[page]),
                    &self.pieces[i].gbuf_nrm[page],
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
                // Write this page's linear diffuse radiance (incident irradiance /
                // pi) into its layer. The material keeps the true albedo, so
                // Replace mode (base_colour * lm.rgb) gives albedo * E/pi. No albedo
                // baked in, no exposure here: the renderer's HDR pipeline tonemaps
                // once for display.
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

                // Normal-mapped pieces get the directional atlas too (dominant
                // direction xyz + directionality w, uploaded linear/raw), so the
                // bumps respond to where the baked light came from. These are all
                // single-page, so only page 0 runs.
                if atlas_count == 1 && self.pieces[i].normal_tex.is_some() {
                    if let Some(dir) = lm.direction() {
                        // Dilate the direction atlas into the gutter using the
                        // radiance coverage as the mask (its own w is directionality,
                        // not coverage), so bilinear at a chart edge reads a real
                        // direction rather than the w=0 gutter (which fades to flat).
                        let covered: Vec<bool> = cleaned.iter().map(|c| c[3] > 0.5).collect();
                        let dir = dilate_masked(dir, &covered, aw as usize, ah as usize, 6);
                        let mut dirbuf = vec![0.0f32; dir.len() * 4];
                        for (t, d) in dir.iter().enumerate() {
                            dirbuf[t * 4] = d[0];
                            dirbuf[t * 4 + 1] = d[1];
                            dirbuf[t * 4 + 2] = d[2];
                            dirbuf[t * 4 + 3] = d[3];
                        }
                        dir_tex = Some(
                            ctx.session
                                .resources_mut()
                                .upload_texture_hdr(device, queue, aw, ah, &dirbuf)
                                .unwrap(),
                        );
                    }
                }
            }

            // Single texture for single-page pieces; an N-layer texture array for
            // the multi-page hero, which set_lightmap_paged then samples per vertex.
            let res = ctx.session.resources_mut();
            let tex = if atlas_count > 1 {
                res.upload_texture_hdr_layers(device, queue, aw, ah, atlas_count, &layers)
                    .unwrap()
            } else {
                res.upload_texture_hdr(device, queue, aw, ah, &layers)
                    .unwrap()
            };
            self.pieces[i].tex = Some(tex);
            self.pieces[i].dir_tex = dir_tex;
        }

        // Scene-atlas heroes: bake them all in one `bake_scene_prepared` call. Its
        // injected passes run the renderer's texel G-buffer + GI solve (against the
        // same occluder scene), then the orchestrator denoises, stitches, encodes,
        // packs into one shared atlas, and returns a placement per object. This
        // runs the whole one-call scene bake under the live renderer, exercising the
        // real GPU-passes path rather than the headless mock.
        let scene_pieces: Vec<usize> = (0..self.pieces.len())
            .filter(|&i| self.pieces[i].scene_atlas && self.pieces[i].baked)
            .collect();
        if !scene_pieces.is_empty() {
            // Owned, already-unwrapped geometry for each hero; PreparedObject
            // borrows it.
            let owned: Vec<(
                Vec<[f32; 3]>,
                Vec<[f32; 3]>,
                Vec<[f32; 2]>,
                Vec<u32>,
                u32,
                u32,
                [[f32; 4]; 4],
            )> = scene_pieces
                .iter()
                .map(|&i| {
                    let p = &self.pieces[i];
                    (
                        p.pos.clone(),
                        p.nrm.clone(),
                        p.uv1.iter().map(|u| [u.x, u.y]).collect(),
                        p.idx.clone(),
                        p.atlas_w,
                        p.atlas_h,
                        p.xf.to_cols_array_2d(),
                    )
                })
                .collect();
            let prepared: Vec<viewport_lib_bake::PreparedObject> = owned
                .iter()
                .map(
                    |(pos, nrm, uv1, idx, w, h, model)| viewport_lib_bake::PreparedObject {
                        positions: pos,
                        normals: nrm,
                        uv1,
                        indices: idx,
                        width: *w,
                        height: *h,
                        model: *model,
                    },
                )
                .collect();
            let rt = self.build_rt_scene();
            let mut passes = ScenePasses {
                device,
                queue,
                scene: &rt,
                settings: RtSettings {
                    samples: self.samples,
                    max_bounces: 4,
                    denoise: false,
                    seed: 0,
                },
            };
            // 1024 fits each hero's atlas (the torus packs to ~980), so no rect is
            // clamped; objects still spill to a second page, which the array handles.
            let opts = viewport_lib_bake::SceneBakeOptions {
                page_size: 1024,
                padding: 8,
                denoise: self.denoise,
                ..Default::default()
            };
            let bake = viewport_lib_bake::bake_scene_prepared(&prepared, &mut passes, &opts);
            let tex = ctx
                .session
                .resources_mut()
                .upload_texture_hdr_layers(
                    device,
                    queue,
                    bake.page_size,
                    bake.page_size,
                    bake.layers,
                    &bake.radiance,
                )
                .unwrap();
            self.scene_atlas_tex = Some(tex);
            self.scene_objects = scene_pieces.len() as u32;
            self.scene_pages = bake.layers;
            self.scene_page_size = bake.page_size;
            for (k, &i) in scene_pieces.iter().enumerate() {
                let pl = bake.placements[k];
                self.pieces[i].tex = Some(tex);
                self.pieces[i].dir_tex = None;
                self.pieces[i].scene_scale_bias = pl.scale_bias;
                self.pieces[i].scene_layer = pl.layer;
            }
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
    fn rebuild(&mut self, session: &mut viewport_lib::ViewportInstance) {
        if !self.nodes.is_empty() {
            let ids = std::mem::take(&mut self.nodes);
            session.scene_mut().remove_many(&ids);
        }
        self.atlas_node = None;

        let baked_mode = self.baked_mode();

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
            // Normal-mapped pieces carry their map in both modes; combined with the
            // directional lightmap (baked mode) the bumps pick up the baked light.
            if let Some(nt) = p.normal_tex {
                mat.normal_map_id = Some(nt);
                mat.normal_strength = 1.0;
            }
            let id = session.scene_mut().add(Some(p.mesh), p.xf, mat);
            // Per-object cast-shadows: the shadow pass skips items with this off,
            // so the toggle reliably shows/hides the realtime shadow. Baked mode
            // has no realtime light, so this is inert there.
            let mut ap = ItemSettings::default();
            ap.cast_shadows = self.realtime_shadows;
            session.scene_mut().set_appearance(id, ap);
            self.nodes.push(id);
        }

        // Emissive GI: show the glowing ceiling panel that lit the bake. It carries
        // the same radiance as the emitter in the trace scene (emissive material,
        // no lightmap), so the source of the soft lighting is visible. Not a shadow
        // caster or receiver: it is the light, not lit geometry.
        if self.mode == EMISSIVE_MODE {
            if let Some(panel) = self.emissive_panel {
                let mut mat = Material::pbr([0.0, 0.0, 0.0], 0.0, 1.0);
                mat.emissive = PANEL_RADIANCE;
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
                            LightmapMode::Replace,
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
                                device,
                                p.mesh,
                                &p.uv1,
                                &p.pages,
                                data,
                                LightmapMode::Replace,
                            );
                        } else {
                            let _ = res.set_lightmap(
                                device,
                                p.mesh,
                                &p.uv1,
                                data,
                                LightmapMode::Replace,
                            );
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
        self.baked_at = None;
        self.built = false;
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
        let torus = primitives::torus(1.9, 0.7, 64, 32);
        let (mut torus_piece, torus_charts) = unwrap_piece(
            ctx,
            &torus,
            Mat4::from_translation(Vec3::new(0.0, 1.0, 1.1)) * Mat4::from_rotation_x(0.35),
            TORUS_ALBEDO,
            None,
        );
        self.torus_charts = torus_charts;
        self.torus_atlas = (torus_piece.atlas_w, torus_piece.atlas_h);
        torus_piece.scene_atlas = true;
        pieces.push(torus_piece);

        // Icosphere, not a UV sphere: a UV sphere's pole collapses many triangles
        // to one point with degenerate UVs, leaving an uncovered (black) texel
        // patch at the pole. The icosphere has uniform triangles and no pole.
        let sphere = primitives::icosphere(1.5, 4);
        let (mut sphere_piece, _) = unwrap_piece(
            ctx,
            &sphere,
            Mat4::from_translation(Vec3::new(-4.6, -1.5, 1.5)),
            SPHERE_ALBEDO,
            None,
        );
        sphere_piece.scene_atlas = true;
        pieces.push(sphere_piece);

        let box_mesh = primitives::cuboid(2.4, 2.4, 2.4);
        let (box_piece, _) = unwrap_piece(
            ctx,
            &box_mesh,
            Mat4::from_translation(Vec3::new(4.8, -1.2, 1.2)) * Mat4::from_rotation_z(0.5),
            BOX_ALBEDO,
            Some(bump),
        );
        pieces.push(box_piece);

        // Multi-page hero: a finely tessellated torus with enough chart area that
        // its unwrap spills across several atlas pages. Its lightmap loads as a
        // texture array and is sampled with a per-vertex page index : the case the
        // single-atlas heroes never reach.
        let knot = primitives::torus(1.3, 0.5, 96, 48);
        let (knot_piece, _) = unwrap_piece_multipage(
            ctx,
            &knot,
            Mat4::from_translation(Vec3::new(0.0, -4.2, 1.35)) * Mat4::from_rotation_x(1.1),
            KNOT_ALBEDO,
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
            // switch between Baked and Emissive (they integrate different lights, so
            // the cached bake is stale); re-encode only (cheap, no tracing) when the
            // denoiser is toggled.
            let need_trace = self.request_rebake
                || self.baked_at.is_none()
                || self.baked_at != Some(self.samples)
                || self.baked_kind != Some(self.mode);
            if need_trace {
                self.request_rebake = false;
                let t0 = std::time::Instant::now();
                self.trace_all(ctx);
                self.encode_all(ctx);
                self.bake_ms = t0.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;
                self.need_reencode = false;
                self.built = true;
                self.baked_kind = Some(self.mode);
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
        if let Some(i) =
            crate::ui::segmented(ui, self.mode, &["Baked GI", "Emissive GI", "Realtime only"])
        {
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

/// Injected GPU passes for `bake_scene_prepared`: the renderer's texel G-buffer
/// rasteriser and GI solve, run against a fixed occluder scene. The orchestrator
/// in viewport-lib-bake owns the CPU steps (denoise, stitch, encode, pack) and
/// calls these two for the GPU work, so the bake crate stays GPU-free.
struct ScenePasses<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    scene: &'a RtScene,
    settings: RtSettings,
}

impl viewport_lib_bake::SceneBakePasses for ScenePasses<'_> {
    fn texel_gbuffer(
        &mut self,
        geom: &viewport_lib_bake::BakeGeometry<'_>,
        width: u32,
        height: u32,
    ) -> viewport_lib_bake::TexelGbuffer {
        let g = rasterize_texel_gbuffer(
            self.device,
            self.queue,
            &TexelGeometry {
                positions: geom.positions,
                normals: geom.normals,
                uv1: geom.uv1,
                indices: geom.indices,
                model: Mat4::from_cols_array_2d(&geom.model),
            },
            width,
            height,
        );
        viewport_lib_bake::TexelGbuffer {
            width: g.width,
            height: g.height,
            world_pos: g.world_pos,
            world_normal: g.world_normal,
        }
    }

    fn solve_gi(&mut self, gbuffer: &viewport_lib_bake::TexelGbuffer) -> viewport_lib_bake::GiBake {
        let bake = bake_lightmap_directional(
            self.device,
            self.queue,
            self.scene,
            &TexelSurfaces {
                width: gbuffer.width,
                height: gbuffer.height,
                world_pos: &gbuffer.world_pos,
                world_normal: &gbuffer.world_normal,
            },
            &self.settings,
        );
        viewport_lib_bake::GiBake {
            irradiance: bake.irradiance,
        }
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

/// Unwrap a primitive with xatlas. `resolution` fixes the atlas (page) size in
/// texels; `texels_per_unit` sets the lightmap density (0 lets xatlas estimate a
/// density that fits one page). A fixed page size plus a high density makes the
/// charts overflow one page and spill onto more, which is how the multi-page
/// hero is produced.
fn do_unwrap(
    mesh: &MeshData,
    resolution: u32,
    texels_per_unit: f32,
) -> viewport_lib_bake::UnwrapResult {
    viewport_lib_bake::unwrap(
        &viewport_lib_bake::UnwrapInput {
            positions: &mesh.positions,
            normals: Some(&mesh.normals),
            indices: &mesh.indices,
        },
        &viewport_lib_bake::UnwrapOptions {
            resolution,
            texels_per_unit,
            padding: 6,
            ..Default::default()
        },
    )
    .expect("unwrap piece")
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
    unwrapped: viewport_lib_bake::UnwrapResult,
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

/// Unwrap a primitive with xatlas and build a baked [`Piece`] from the result.
/// `normal_tex` opts the piece into normal mapping + a directional lightmap.
fn unwrap_piece(
    ctx: &mut SetupCtx,
    mesh: &MeshData,
    xf: Mat4,
    albedo: [f32; 3],
    normal_tex: Option<TextureId>,
) -> (Piece, u32) {
    let unwrapped = do_unwrap(mesh, ATLAS, 0.0);
    build_piece_from_unwrap(ctx, mesh, xf, albedo, normal_tex, unwrapped)
}

/// Unwrap a primitive deliberately into two or more atlas pages, so the piece
/// exercises the multi-page load path. The trick is only to make a small object
/// spill: the pages are the same full size as every other hero (`ATLAS`) and the
/// density starts high, so each page is as sharp as the single-page bakes. Only
/// the page count is contrived here, never the per-page quality. The density is
/// raised until the charts no longer fit one page and xatlas spills onto more.
fn unwrap_piece_multipage(
    ctx: &mut SetupCtx,
    mesh: &MeshData,
    xf: Mat4,
    albedo: [f32; 3],
) -> (Piece, u32) {
    let mut tpu = 48.0f32;
    let unwrapped = loop {
        let u = do_unwrap(mesh, ATLAS, tpu);
        if u.atlas_count >= 2 || tpu >= 320.0 {
            break u;
        }
        tpu *= 1.3;
    };
    build_piece_from_unwrap(ctx, mesh, xf, albedo, None, unwrapped)
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
        .upload_normal_map(ctx.device, ctx.queue, size, size, &rgba)
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
