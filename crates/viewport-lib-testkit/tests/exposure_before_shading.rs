//! Exposure applied where radiance is read.
//!
//! The camera's exposure multiplies every radiance source in the shaders, so a
//! lit surface carries exposed radiance before the composite. These tests pin
//! what follows from that: the auto-exposure meter settles on the EV it
//! measures in absolute radiance, the `Direct` path takes a fixed exposure and
//! reports an automatic one, viewports over one scene expose independently,
//! a capture holds absolute radiance whatever the frame's exposure, and an
//! emitter's exposure weight sets how far the exposure reaches it.
//!
//! The float tests read back a half-float target, so a ratio is measured to
//! half-float precision rather than through 8-bit quantisation.

use viewport_lib::Camera;
use viewport_lib::wgpu;
use viewport_lib::{
    ExposureSettings, FrameData, LightingSettings, NoticeDetail, PipelineMode, ToneMapping,
};
use viewport_lib_testkit::{BuiltScene, Harness, frame_for, scene_by_name};

const W: u32 = 400;
const H: u32 = 300;

/// The adapted EV each automatic chart settles on, metered in absolute
/// radiance.
const SETTLED_EV: [(&str, f32); 2] = [
    ("exposure_auto", 15.152_625),
    ("exposure_auto_partial", 0.196_618_26),
];

/// The linear multiplier of a manual EV100, as `ExposureSettings` resolves it.
fn multiplier(ev: f32) -> f32 {
    1.0 / (1.2 * 2f32.powf(ev))
}

/// The exposure chart, built once for frames over it.
fn build_chart(h: &mut Harness) -> (BuiltScene, Camera) {
    let scene = scene_by_name("exposure_manual_ev15").expect("exposure chart in the catalogue");
    let built = h.build_scene(&scene);
    (built, scene.cameras[0].camera.clone())
}

/// A frame of the chart under unit lighting, so a Passthrough frame stays
/// inside the `Direct` path's [0, 1] clamp over most of the frame.
fn unit_chart(chart: &(BuiltScene, Camera), size: [u32; 2]) -> FrameData {
    let mut frame = frame_for(&chart.0, &chart.1, [size[0] as f32, size[1] as f32]);
    let mut lighting = LightingSettings::default();
    for light in &mut lighting.lights {
        light.cast_shadows = false;
    }
    frame.effects.lighting = lighting;
    frame
}

/// Per-channel ratios `dim / bright` over the channels where `bright` lies
/// inside `(lo, hi)`.
fn ratios(bright: &[[f32; 4]], dim: &[[f32; 4]], lo: f32, hi: f32) -> Vec<f32> {
    let mut out = Vec::new();
    for (b, d) in bright.iter().zip(dim) {
        for c in 0..3 {
            if b[c] > lo && b[c] < hi {
                out.push(d[c] / b[c]);
            }
        }
    }
    out
}

/// Assert every ratio is within `tol` (relative) of `expected`.
fn assert_ratios(label: &str, ratios: &[f32], expected: f32, tol: f32) {
    assert!(
        ratios.len() > 1000,
        "{label}: only {} measurable channels",
        ratios.len()
    );
    let worst = ratios
        .iter()
        .map(|r| (r / expected - 1.0).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= tol,
        "{label}: a channel ratio is {worst} away from {expected} (relative; allowed {tol})"
    );
}

/// `exposure_auto` meters the same EV with the exposure applied at the
/// radiance reads as it did when the composite applied it.
#[test]
fn auto_exposure_settles_on_the_metered_ev() {
    for (name, expected) in SETTLED_EV {
        let Some(mut harness) = Harness::new() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let vp0 = harness.renderer.create_viewport(&harness.device);
        let scene = scene_by_name(name).expect("scene in the catalogue");
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        // The first frame shades with the seed exposure and meters; the rest
        // shade with what it resolved and must hold it.
        for i in 0..4 {
            let _ = harness.render(&frame, W, H);
            let st = harness
                .renderer
                .exposure_state(&harness.device, &harness.queue, vp0)
                .expect("exposure state after an HDR frame");
            assert!(
                (st.current_ev - expected).abs() <= 0.01,
                "{name} frame {i}: EV {} differs from {expected}",
                st.current_ev
            );
        }
    }
}

/// The `Direct` path applies a fixed exposure at the radiance reads: the chart
/// under `Manual { ev: 15 }` is the Passthrough chart scaled by the
/// multiplier, pixel for pixel. `Automatic` has no meter there, so it draws as
/// Passthrough and raises a notice.
#[test]
fn direct_path_takes_the_fixed_exposure() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let chart = build_chart(&mut h);
    let mut frame = unit_chart(&chart, [W, H]);
    frame.effects.display.mode = PipelineMode::Direct;

    frame.effects.display.exposure = ExposureSettings::passthrough();
    let passthrough = h.render_float(&frame, W, H);
    frame.effects.display.exposure = ExposureSettings::manual(15.0);
    let ev15 = h.render_float(&frame, W, H);

    // Lit radiance is clamped to 1.0 on this path, so measure below it.
    let r = ratios(&passthrough, &ev15, 0.02, 0.95);
    assert_ratios("Direct, ev 15 / passthrough", &r, multiplier(15.0), 0.01);

    let _ = h.renderer.take_notices();
    frame.effects.display.exposure = ExposureSettings::automatic();
    let auto = h.render_float(&frame, W, H);
    assert!(
        auto == passthrough,
        "Automatic on the Direct path did not draw as Passthrough"
    );
    let notices = h.renderer.take_notices();
    assert!(
        notices
            .iter()
            .any(|n| n.detail == NoticeDetail::AutoExposureOnLdr),
        "no AutoExposureOnLdr notice: {notices:?}"
    );
}

/// Read a half-float offscreen target back to linear RGBA.
fn readback_float(h: &Harness, texture: &wgpu::Texture, w: u32, hgt: u32) -> Vec<[f32; 4]> {
    let bpp = 8u32;
    let unpadded = w * bpp;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = (unpadded + align - 1) & !(align - 1);
    let staging = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exposure_mv_staging"),
        size: (padded * hgt) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = h
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("exposure_mv_copy"),
        });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(hgt),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: hgt,
            depth_or_array_layers: 1,
        },
    );
    h.queue.submit(std::iter::once(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    h.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    let _ = rx.recv();
    let mut out = Vec::with_capacity((w * hgt) as usize);
    {
        let mapped = viewport_lib::gpu::mapped_range(staging.slice(..));
        for row in 0..hgt as usize {
            let start = row * padded as usize;
            for texel in mapped[start..start + unpadded as usize].chunks_exact(8) {
                let mut px = [0f32; 4];
                for (c, v) in px.iter_mut().enumerate() {
                    *v = half_to_f32(u16::from_le_bytes([texel[c * 2], texel[c * 2 + 1]]));
                }
                out.push(px);
            }
        }
    }
    staging.unmap();
    out
}

/// Decode an IEEE half into an `f32`.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => fraction * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Two viewports over one scene, at EV 0 and EV 15, each shade with their own
/// exposure. Reinhard is inverted per pixel to recover the exposed radiance,
/// and the two viewports differ by the ratio of their multipliers.
#[test]
fn viewports_over_one_scene_expose_independently() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    const S: u32 = 200;
    let chart = build_chart(&mut h);
    let vp0 = h.renderer.create_viewport(&h.device);
    let vp1 = h.renderer.create_viewport(&h.device);
    let mut frame0 = unit_chart(&chart, [S, S]);
    let mut frame1 = unit_chart(&chart, [S, S]);
    for (frame, vp, ev) in [(&mut frame0, vp0, 0.0), (&mut frame1, vp1, 15.0)] {
        frame.camera.viewport_index = vp.index();
        frame.effects.display.exposure = ExposureSettings::manual(ev);
        frame.effects.display.operator = ToneMapping::Reinhard;
    }

    let target = |h: &Harness| {
        h.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("exposure_mv_target"),
            size: wgpu::Extent3d {
                width: S,
                height: S,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Harness::FLOAT_TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let tex0 = target(&h);
    let tex1 = target(&h);
    let view0 = tex0.create_view(&wgpu::TextureViewDescriptor::default());
    let view1 = tex1.create_view(&wgpu::TextureViewDescriptor::default());

    let (scene_fx, _) = frame0.effects.split();
    let token = h
        .renderer
        .owned()
        .prepare_scene(&h.device, &h.queue, &frame0, &scene_fx);
    h.renderer
        .owned()
        .prepare_viewport(&h.device, &h.queue, &token, vp0, &frame0);
    h.renderer
        .owned()
        .prepare_viewport(&h.device, &h.queue, &token, vp1, &frame1);
    let cmd0 = h
        .renderer
        .owned()
        .render_viewport(&h.device, &h.queue, &view0, vp0, &frame0);
    let cmd1 = h
        .renderer
        .owned()
        .render_viewport(&h.device, &h.queue, &view1, vp1, &frame1);
    h.queue.submit([cmd0, cmd1]);

    let unreinhard = |img: Vec<[f32; 4]>| -> Vec<[f32; 4]> {
        img.into_iter()
            .map(|p| {
                let inv = |y: f32| y / (1.0 - y);
                [inv(p[0]), inv(p[1]), inv(p[2]), p[3]]
            })
            .collect()
    };
    let lin0 = unreinhard(readback_float(&h, &tex0, S, S));
    let lin1 = unreinhard(readback_float(&h, &tex1, S, S));

    // The EV 15 view sits in half-float subnormals, whose spacing is under a
    // percent of the value over this range.
    let r = ratios(&lin0, &lin1, 0.25, 3.0);
    assert_ratios(
        "viewport ev 15 / viewport ev 0",
        &r,
        multiplier(15.0) / multiplier(0.0),
        0.02,
    );
}

/// A capture holds absolute radiance: the same panorama under any exposure.
#[test]
fn captures_ignore_the_frame_exposure() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene =
        scene_by_name("light_transport_emitter_sunny16").expect("emitter chart in the catalogue");
    let built = h.build_scene(&scene);
    let mut frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let _ = h.render(&frame, W, H);

    let mut capture = |frame: &mut FrameData, exposure: ExposureSettings| {
        frame.effects.display.exposure = exposure;
        h.renderer
            .capture_equirect(&h.device, &h.queue, frame, [0.0, 0.0, 2.0], 32, 32)
    };
    let passthrough = capture(&mut frame, ExposureSettings::passthrough());
    let ev15 = capture(&mut frame, ExposureSettings::manual(15.0));
    assert_eq!(
        frame.effects.display.exposure,
        ExposureSettings::manual(15.0),
        "the capture did not restore the frame's exposure"
    );

    let brightest = passthrough.rgba.iter().copied().fold(0.0f32, f32::max);
    assert!(
        brightest > 100.0,
        "the passthrough panorama is not daylight radiance (max {brightest})"
    );
    assert!(
        passthrough.rgba == ev15.rgba,
        "the panorama changed with the frame's exposure"
    );
}

/// An emitter at exposure weight 1 scales with the exposure multiplier, one at
/// weight 0 draws the same under any exposure, and one at 0.5 takes the
/// midpoint scale. `light_transport_emitter_weights` holds 2 000 nit black
/// cards at weights 1, 0.5 and 0; with no lights the cards carry only their
/// emissive.
#[test]
fn emissive_exposure_weight_sets_the_exposure_scale() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    const NITS: f32 = 2_000.0;
    let scene =
        scene_by_name("light_transport_emitter_weights").expect("emitter weights in the catalogue");
    let built = h.build_scene(&scene);
    let mut frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let mut lighting = LightingSettings::default();
    lighting.lights.clear();
    lighting.hemisphere_intensity = 0.0;
    frame.effects.lighting = lighting;
    let m = multiplier(15.0);

    let mut render = |frame: &mut FrameData, mode: PipelineMode, exposure: ExposureSettings| {
        frame.effects.display.mode = mode;
        frame.effects.display.exposure = exposure;
        h.render_float(frame, W, H)
    };
    let direct_pass = render(
        &mut frame,
        PipelineMode::Direct,
        ExposureSettings::passthrough(),
    );
    let direct_ev15 = render(
        &mut frame,
        PipelineMode::Direct,
        ExposureSettings::manual(15.0),
    );
    let hdr_pass = render(
        &mut frame,
        PipelineMode::Hdr,
        ExposureSettings::passthrough(),
    );
    let hdr_ev15 = render(
        &mut frame,
        PipelineMode::Hdr,
        ExposureSettings::manual(15.0),
    );

    // Card interiors: exactly the emitter's radiance under Passthrough.
    let (mut unit, mut half, mut zero) = (0usize, 0usize, 0usize);
    for (i, (pass, ev15)) in direct_pass.iter().zip(&direct_ev15).enumerate() {
        if !(0..3).all(|c| pass[c] == NITS) {
            continue;
        }
        let v = ev15[0];
        if v == NITS {
            zero += 1;
            // Tone mapped, the weight-0 card is still the same in both.
            assert_eq!(
                hdr_pass[i], hdr_ev15[i],
                "weight-0 pixel {i} changed with the exposure on the HDR path"
            );
        } else if (v / (NITS * m) - 1.0).abs() <= 0.002 {
            unit += 1;
        } else if (v / (NITS * (0.5 * m + 0.5)) - 1.0).abs() <= 0.002 {
            half += 1;
        } else {
            panic!("card pixel {i} reads {v} at EV 15, matching no weight");
        }
    }
    for (label, n) in [("weight 1", unit), ("weight 0.5", half), ("weight 0", zero)] {
        assert!(n > 1000, "{label}: only {n} card pixels");
    }
}
