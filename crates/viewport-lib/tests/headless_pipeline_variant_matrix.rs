//! Phase 0 completeness matrix for the pipeline-variant-specialization plan
//! (`docs/adrs/0002-pipeline-variant-management.md`).
//!
//! Every reachable (pass, variant-key) combination must resolve to a pipeline
//! that renders the right thing. This file exercises the axes that matter per
//! pass -- facedness (one-sided vs two-sided), alpha-cutout, and LDR vs HDR --
//! across the opaque, OIT, and shadow passes, for both the per-object and the
//! instanced draw routes.
//!
//! Every cell here passes; this is a regression backstop for the pipeline-key
//! refactor. `shadow_alpha_mask_matrix` covers both draw routes for a masked
//! caster (a material that discards to nothing in the colour pass must not
//! still cast a full opaque shadow), including the per-object route, whose
//! alpha-cutout pipeline was the first of two filed gaps this plan closes.
//!
//! The other filed gap (`material-plugin-opaque-pipelines-no-early-z-nodiscard-variant`)
//! is closed too but is not covered here: it was a missing fast-path pipeline
//! twin, not a rendering difference, so a plugin material draws the same
//! pixels whether or not the discard-free twin exists. There is no pixel-level
//! signal that distinguishes the two paths; see
//! `mesh_sidecar::shade::tests::material_plugin_pipelines_resolve_every_key_once_built`
//! for the completeness check that covers it instead.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;
use viewport_lib::{PipelineMode, TextureId};

// ---------------------------------------------------------------------------
// Shared geometry and scene helpers
// ---------------------------------------------------------------------------

/// A flat quad in the z = 0 plane. `front_faces_positive_z` controls the
/// winding: `true` puts the geometric front face at +Z (visible to a camera or
/// light above), `false` winds it the other way (front face at -Z, so a viewer
/// or light above sees only the back face under `BackfacePolicy::Cull`).
fn quad_mesh(half: f32, front_faces_positive_z: bool) -> MeshData {
    let positions = vec![
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ];
    let normal_z = if front_faces_positive_z { 1.0 } else { -1.0 };
    let normals = vec![[0.0, 0.0, normal_z]; 4];
    let indices: Vec<u32> = if front_faces_positive_z {
        vec![0, 1, 2, 2, 3, 0]
    } else {
        vec![0, 2, 1, 0, 3, 2]
    };
    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}

/// Count pixels with a visible red channel (unlit red on a black background).
fn coverage(px: &[u8]) -> usize {
    px.chunks_exact(4).filter(|p| p[0] > 40).count()
}

fn top_down_camera(distance: f32) -> Camera {
    let mut c = Camera::default();
    c.center = glam::Vec3::ZERO;
    c.distance = distance;
    c.orientation = glam::Quat::from_rotation_x(0.15);
    c
}

/// `generation` must be distinct for each distinct scene content within a
/// renderer's lifetime: the instanced-batch cache keys on `scene.generation`
/// rather than diffing content, so two calls with the same generation replay
/// stale batches even when the items differ (see `headless_instanced_free_cache.rs`).
fn base_frame(target: PipelineMode, size: u32, generation: u64) -> FrameData {
    let mut frame = FrameData::default();
    frame.scene.generation = generation;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0].into());
    frame.effects.display.mode = target;
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&top_down_camera(6.0));
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame
}

/// A distant, harmless item that pushes the scene's visible-item count above
/// the instancing threshold (1) without appearing on screen, so the item(s)
/// under test route through the instanced draw path instead of per-object.
fn route_filler(mesh_id: MeshId) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model =
        glam::Mat4::from_translation(glam::Vec3::new(-500.0, -500.0, -500.0)).to_cols_array_2d();
    item.material = Material::from_colour([0.1, 0.9, 0.1]);
    item
}

// ---------------------------------------------------------------------------
// Opaque pass: {one-sided, two-sided} x {LDR, HDR} x {per-object, instanced}
// ---------------------------------------------------------------------------

#[test]
fn opaque_two_sided_matrix() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Front face points away from the top-down camera: `Cull` sees nothing,
    // `Identical` must draw the back face.
    let mesh = quad_mesh(1.5, false);
    for &target in &[PipelineMode::Direct, PipelineMode::Hdr] {
        for &instanced in &[false, true] {
            let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
            let mesh_id = renderer
                .resources_mut()
                .upload_mesh_data(&device, &mesh)
                .unwrap();
            let gen_ctr = std::cell::Cell::new(0u64);
            let render = |renderer: &mut ViewportRenderer, two_sided: bool| {
                gen_ctr.set(gen_ctr.get() + 1);
                let mut frame = base_frame(target, 128, gen_ctr.get());
                let mut item = SceneRenderItem::default();
                item.mesh_id = mesh_id;
                item.material.base_colour = [1.0, 0.0, 0.0].into();
                item.settings.unlit = true;
                if two_sided {
                    item.material.backface_policy = BackfacePolicy::Identical;
                }
                let mut items = vec![item];
                if instanced {
                    items.push(route_filler(mesh_id));
                }
                frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
                coverage(&renderer.render_offscreen(&device, &queue, &frame, 128, 128))
            };

            let one_sided = render(&mut renderer, false);
            assert_eq!(
                one_sided, 0,
                "target={target:?} instanced={instanced}: one-sided back-facing quad \
                 should cull to nothing (coverage {one_sided})"
            );
            let two_sided = render(&mut renderer, true);
            assert!(
                two_sided > 1000,
                "target={target:?} instanced={instanced}: two-sided quad did not draw \
                 its back face (coverage {two_sided})"
            );
        }
    }
}

#[test]
fn opaque_alpha_mask_matrix() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Front-facing quad: visible under either backface policy, so the mask
    // axis is tested independently of facedness.
    let mesh = quad_mesh(1.5, true);
    for &target in &[PipelineMode::Direct, PipelineMode::Hdr] {
        for &instanced in &[false, true] {
            for &two_sided in &[false, true] {
                let mut renderer =
                    ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
                let mesh_id = renderer
                    .resources_mut()
                    .upload_mesh_data(&device, &mesh)
                    .unwrap();
                // Alpha below the 0.5 cutoff must discard every fragment;
                // alpha above it must pass through untouched. Mask discard
                // reads the *texture* alpha channel, not the item's base
                // colour or opacity, so a real (1x1) texture is required.
                let tex_below = renderer
                    .resources_mut()
                    .upload_texture(&device, &queue, 1, 1, &[255, 255, 255, 20])
                    .unwrap();
                let tex_above = renderer
                    .resources_mut()
                    .upload_texture(&device, &queue, 1, 1, &[255, 255, 255, 220])
                    .unwrap();

                let gen_ctr = std::cell::Cell::new(0u64);
                let render = |renderer: &mut ViewportRenderer, tex: TextureId| {
                    gen_ctr.set(gen_ctr.get() + 1);
                    let mut frame = base_frame(target, 128, gen_ctr.get());
                    let mut item = SceneRenderItem::default();
                    item.mesh_id = mesh_id;
                    item.material.base_colour = [1.0, 0.0, 0.0].into();
                    item.material.texture_id = Some(tex);
                    item.material.alpha_mode = AlphaMode::Mask(0.5);
                    item.settings.unlit = true;
                    if two_sided {
                        item.material.backface_policy = BackfacePolicy::Identical;
                    }
                    let mut items = vec![item];
                    if instanced {
                        items.push(route_filler(mesh_id));
                    }
                    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
                    coverage(&renderer.render_offscreen(&device, &queue, &frame, 128, 128))
                };

                let passes = render(&mut renderer, tex_above);
                assert!(
                    passes > 1000,
                    "target={target:?} instanced={instanced} two_sided={two_sided}: \
                     alpha above the cutoff should render (coverage {passes})"
                );
                let discarded = render(&mut renderer, tex_below);
                assert_eq!(
                    discarded, 0,
                    "target={target:?} instanced={instanced} two_sided={two_sided}: \
                     alpha below the cutoff should discard fully (coverage {discarded})"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OIT (transparent, HDR-only): {one-sided, two-sided} x {per-object, instanced}
// ---------------------------------------------------------------------------

#[test]
fn oit_two_sided_matrix() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mesh = quad_mesh(1.5, false);
    for &instanced in &[false, true] {
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &mesh)
            .unwrap();
        let gen_ctr = std::cell::Cell::new(0u64);
        let render = |renderer: &mut ViewportRenderer, two_sided: bool| {
            gen_ctr.set(gen_ctr.get() + 1);
            let mut frame = base_frame(PipelineMode::Hdr, 128, gen_ctr.get());
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.material.base_colour = [1.0, 0.0, 0.0].into();
            item.settings.unlit = true;
            item.settings.opacity = 0.75;
            if two_sided {
                item.material.backface_policy = BackfacePolicy::Identical;
            }
            let mut items = vec![item];
            if instanced {
                items.push(route_filler(mesh_id));
            }
            frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
            coverage(&renderer.render_offscreen(&device, &queue, &frame, 128, 128))
        };

        let one_sided = render(&mut renderer, false);
        assert_eq!(
            one_sided, 0,
            "instanced={instanced}: one-sided transparent quad should cull to nothing \
             (coverage {one_sided})"
        );
        let two_sided = render(&mut renderer, true);
        assert!(
            two_sided > 1000,
            "instanced={instanced}: two-sided transparent quad lost its back faces on \
             the OIT path (coverage {two_sided})"
        );
    }
}

/// Centre-pixel red channel of a rendered frame.
fn centre_red(px: &[u8], size: usize) -> u8 {
    let idx = ((size / 2) * size + size / 2) * 4;
    px[idx]
}

/// Premultiplied-alpha blend axis on the OIT pass.
///
/// Unlike the other axes here, premultiplied blend is not a separate pipeline:
/// weighted-blended OIT fixes its accum/reveal blend equations, so the straight
/// vs premultiplied distinction is a per-object uniform branch in
/// `mesh_oit.wgsl` (skip the `* alpha` on RGB), read from `alpha_mode == 3`.
/// The observable is a colour difference, not coverage: feed the *same*
/// premultiplied texture (RGB already multiplied down by alpha) through both
/// modes. Straight `Blend` multiplies by alpha a second time and darkens it
/// (the dark-edge halo the mode exists to fix); `BlendPremultiplied` composites
/// it as authored and stays brighter.
#[test]
fn oit_premultiplied_blend_matrix() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // One-sided quad, front face toward the top-down camera, so it renders
    // opaque-coverage-wise and the whole centre is the surface under test.
    let mesh = quad_mesh(1.5, true);
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();
    // A premultiplied grey: RGB already scaled by the 0.5 alpha it carries.
    let tex = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 1, 1, &[128, 128, 128, 128])
        .unwrap();

    let gen_ctr = std::cell::Cell::new(0u64);
    let render = |renderer: &mut ViewportRenderer, mode: AlphaMode| {
        gen_ctr.set(gen_ctr.get() + 1);
        let mut frame = base_frame(PipelineMode::Hdr, 128, gen_ctr.get());
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.material.base_colour = [1.0, 1.0, 1.0].into();
        item.material.texture_id = Some(tex);
        item.material.alpha_mode = mode;
        item.settings.unlit = true;
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        let px = renderer.render_offscreen(&device, &queue, &frame, 128, 128);
        centre_red(&px, 128)
    };

    let straight = render(&mut renderer, AlphaMode::Blend);
    let premult = render(&mut renderer, AlphaMode::BlendPremultiplied);

    // Both must actually render the surface.
    assert!(
        straight > 10,
        "straight-blend premultiplied texture rendered nothing (red {straight})"
    );
    assert!(
        premult > 10,
        "premultiplied-blend texture rendered nothing (red {premult})"
    );
    // Premultiplied avoids the second alpha multiply, so it composites brighter
    // than straight blend for the same premultiplied input.
    assert!(
        premult as i32 - straight as i32 > 20,
        "premultiplied blend ({premult}) should be clearly brighter than straight \
         blend ({straight}) for the same premultiplied texture; the axis is not \
         reaching the OIT shader"
    );
}

// ---------------------------------------------------------------------------
// Shadow pass: {one-sided, two-sided} x {plain, alpha-mask} x {per-object, instanced}
// ---------------------------------------------------------------------------

/// A floor quad, plus an optional small caster quad hovering above it, as one
/// mesh. Combining them into a single item lets the per-object route be forced
/// just by keeping the scene's visible-item count at 1 (below the instancing
/// threshold), with no other per-item exclusion needed. `caster_front_positive_z`
/// sets the caster's winding (see `shadow_two_sided_matrix` for why it matters).
fn floor_with_caster(include_caster: bool, caster_front_positive_z: bool) -> MeshData {
    let mut mesh = quad_mesh(1.5, true);
    if include_caster {
        let mut caster = quad_mesh(0.4, caster_front_positive_z);
        for p in caster.positions.iter_mut() {
            p[2] += 1.0;
        }
        let base = mesh.positions.len() as u32;
        mesh.positions.extend(caster.positions);
        mesh.normals.extend(caster.normals);
        mesh.indices.extend(caster.indices.iter().map(|i| i + base));
    }
    mesh
}

#[test]
fn shadow_two_sided_matrix() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for &instanced in &[false, true] {
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let gen_ctr = std::cell::Cell::new(0u64);
        let render = |renderer: &mut ViewportRenderer, mesh_id: MeshId, two_sided: bool| {
            gen_ctr.set(gen_ctr.get() + 1);
            let mut frame = base_frame(PipelineMode::Hdr, 128, gen_ctr.get());
            frame.effects.lighting.shadows.cascade_count = 1;
            frame.effects.lighting.shadows.atlas_resolution = 512;
            frame.effects.lighting.shadows.extent_override = Some(3.0);
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.material.base_colour = [0.7, 0.7, 0.7].into();
            if two_sided {
                item.material.backface_policy = BackfacePolicy::Identical;
            }
            let mut items = vec![item];
            if instanced {
                items.push(route_filler(mesh_id));
            }
            frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
            renderer.render_offscreen(&device, &queue, &frame, 128, 128)
        };

        // The default shadow pipeline culls *front* faces (the opposite
        // convention from the colour pass): for a closed solid the surface's
        // own front face is never compared against itself in the shadow map,
        // but for an open, single-sided surface it means the face pointing
        // *toward* the light is the one that gets removed. Wind the caster so
        // its front face points toward the light (matching the top-down
        // camera too, so it is equally visible in the colour pass in both
        // cases below and colour-pass visibility does not confound the
        // shadow comparison): `Cull` (the plain `pipeline`, cull-front) drops
        // it from the shadow pass entirely; `Identical` (`pipeline_two_sided`,
        // cull-none) keeps it.
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &floor_with_caster(true, true))
            .unwrap();

        let one_sided = render(&mut renderer, mesh_id, false);
        let two_sided = render(&mut renderer, mesh_id, true);
        let diffs = two_sided
            .iter()
            .zip(&one_sided)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diffs > 50,
            "instanced={instanced}: a two-sided caster should cast a shadow that a \
             one-sided (cull-front) caster of the same visible geometry does not \
             (got {diffs} differing bytes between them)"
        );
    }
}

/// A masked caster (invisible in the colour pass) must not still cast a full
/// opaque shadow, on both the per-object and instanced shadow routes.
/// `route_instanced=false` forces the caster onto the per-object path via a
/// styled backface policy (`Tint`, which `backface_needs_per_object` requires
/// per-item state for); two-sidedness itself is not under test here.
#[test]
fn shadow_alpha_mask_matrix() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let floor_mesh = quad_mesh(1.5, true);
    let caster_mesh = {
        let mut b = box_mesh();
        for p in b.positions.iter_mut() {
            for c in p.iter_mut() {
                *c *= 0.35;
            }
        }
        b
    };

    for &route_instanced in &[false, true] {
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let floor_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &floor_mesh)
            .unwrap();
        let caster_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &caster_mesh)
            .unwrap();
        // Mask discard reads texture alpha, not base colour or opacity.
        let tex_below = renderer
            .resources_mut()
            .upload_texture(&device, &queue, 1, 1, &[255, 255, 255, 20])
            .unwrap();
        let tex_above = renderer
            .resources_mut()
            .upload_texture(&device, &queue, 1, 1, &[255, 255, 255, 220])
            .unwrap();

        let gen_ctr = std::cell::Cell::new(0u64);
        let render = |renderer: &mut ViewportRenderer, caster_tex: Option<TextureId>| {
            gen_ctr.set(gen_ctr.get() + 1);
            let mut frame = base_frame(PipelineMode::Hdr, 128, gen_ctr.get());
            frame.effects.lighting.shadows.cascade_count = 1;
            frame.effects.lighting.shadows.atlas_resolution = 512;
            frame.effects.lighting.shadows.extent_override = Some(3.0);
            let mut floor = SceneRenderItem::default();
            floor.mesh_id = floor_id;
            floor.material.base_colour = [0.7, 0.7, 0.7].into();
            let mut items = vec![floor];
            if let Some(tex) = caster_tex {
                let mut caster = SceneRenderItem::default();
                caster.mesh_id = caster_id;
                caster.model =
                    glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.6)).to_cols_array_2d();
                caster.material.base_colour = [1.0, 0.0, 0.0].into();
                caster.material.texture_id = Some(tex);
                caster.material.alpha_mode = AlphaMode::Mask(0.5);
                if !route_instanced {
                    // `Tint` needs per-item back-face state, which forces the
                    // per-object path regardless of the scene's global
                    // instancing mode; two-sidedness itself is not under test
                    // here (a closed box occludes the light the same way
                    // either way).
                    caster.material.backface_policy = BackfacePolicy::Tint(0.3);
                }
                items.push(caster);
            }
            frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
            renderer.render_offscreen(&device, &queue, &frame, 128, 128)
        };

        let baseline = render(&mut renderer, None);

        let below_cutoff = render(&mut renderer, Some(tex_below));
        let bug_diffs = below_cutoff
            .iter()
            .zip(&baseline)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            bug_diffs, 0,
            "route_instanced={route_instanced}: a caster discarded below the alpha \
             cutoff must cast no shadow (byte-identical to no caster at all; \
             got {bug_diffs} differing bytes)"
        );

        let above_cutoff = render(&mut renderer, Some(tex_above));
        let diffs = above_cutoff
            .iter()
            .zip(&baseline)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diffs > 50,
            "route_instanced={route_instanced}: a visible caster above the cutoff \
             should render and cast a shadow (got {diffs} differing bytes)"
        );
    }
}
