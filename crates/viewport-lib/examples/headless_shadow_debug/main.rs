//! Headless shadow debugging: renders the lighting-shadows test scene to PNG
//! files so shadow bugs can be inspected without a window.
//!
//! Usage:
//!   cargo run --release --example headless-shadow-debug -- <dir_x> <dir_y> <dir_z> [out_prefix]
//!
//! Writes <out_prefix>_normal.ppm, _shadowfactor.ppm, _ndotl.ppm,
//! _cascade.ppm, and _atlas.ppm to the current directory.

use viewport_lib::{
    AtlasViewerCorner, BackfacePolicy, Camera, CameraFrame, DebugOutputMode, DebugQuantity,
    DebugVis, FrameData, LightKind, LightSource, LightingSettings, Material, SceneFrame,
    SceneRenderItem, ShadowFilter, ViewportRenderer, primitives,
};

const W: u32 = 1280;
const H: u32 = 720;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = if args.len() >= 4 {
        [
            args[1].parse::<f32>().unwrap(),
            args[2].parse::<f32>().unwrap(),
            args[3].parse::<f32>().unwrap(),
        ]
    } else {
        [0.4, 0.3, -0.85]
    };
    let prefix = if args.len() >= 5 {
        args[4].clone()
    } else {
        "shadow_debug".to_string()
    };

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("headless-shadow-debug"),
        required_limits: viewport_lib::ViewportRenderer::recommended_device_limits(&adapter),
        ..Default::default()
    }))
    .expect("device");

    let format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let mut renderer = ViewportRenderer::new(&device, format);

    let (m_ground, m_sphere, m_cube, m_torus);
    {
        let res = renderer.resources_mut();
        m_ground = res
            .upload_mesh_data(&device, &primitives::cuboid(24.0, 24.0, 0.5))
            .expect("ground");
        m_sphere = res
            .upload_mesh_data(&device, &primitives::sphere(0.6, 32, 16))
            .expect("sphere");
        m_cube = res
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .expect("cube");
        m_torus = res
            .upload_mesh_data(&device, &primitives::torus(0.5, 0.18, 40, 20))
            .expect("torus");
    }

    // Same scene as the eframe_lighting_shadows Basic tab.
    let build_items = || {
        let mut items = Vec::new();
        let mut ground = SceneRenderItem::default();
        ground.mesh_id = m_ground;
        ground.model =
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
        ground.material = Material::from_colour([0.88, 0.84, 0.76]);
        ground.material.roughness = 0.85;
        ground.material.backface_policy = BackfacePolicy::Cull;
        items.push(ground);

        let mut sphere = SceneRenderItem::default();
        sphere.mesh_id = m_sphere;
        sphere.model =
            glam::Mat4::from_translation(glam::Vec3::new(-4.0, 0.0, 0.6)).to_cols_array_2d();
        sphere.material = Material::from_colour([0.78, 0.90, 0.80]);
        items.push(sphere);

        let mut cube = SceneRenderItem::default();
        cube.mesh_id = m_cube;
        cube.model =
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.5)).to_cols_array_2d();
        cube.material = Material::from_colour([0.78, 0.83, 0.95]);
        items.push(cube);

        let mut torus = SceneRenderItem::default();
        torus.mesh_id = m_torus;
        torus.model =
            glam::Mat4::from_translation(glam::Vec3::new(4.0, 0.0, 0.18)).to_cols_array_2d();
        torus.material = Material::from_colour([0.95, 0.82, 0.74]);
        items.push(torus);
        items
    };

    // Camera: orbit distance 18 looking down at the scene by default; optional
    // args 5..=10 override center xyz, distance, and rz/rx orientation angles.
    let f = |i: usize, d: f32| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let mut camera = Camera {
        distance: f(8, 18.0),
        ..Camera::default()
    };
    camera.center = glam::Vec3::new(f(5, 0.0), f(6, 0.0), f(7, 0.0));
    camera.orientation =
        glam::Quat::from_rotation_z(f(9, 0.6)) * glam::Quat::from_rotation_x(f(10, 1.0));
    camera.set_aspect_ratio(W as f32, H as f32);

    let build_lighting = |dbg: Option<(DebugQuantity, DebugQuantity, DebugQuantity, f32)>,
                          shadows_on: bool,
                          hemi: f32| {
        let mut l = LightingSettings::default();
        l.lights = vec![{
            let mut src = LightSource::default();
            src.kind = LightKind::Directional { direction: dir };
            src.colour = [1.0, 0.97, 0.90];
            src.intensity = 0.8;
            src
        }];
        l.shadows.enabled = shadows_on;
        l.shadows.bias = 0.0;
        l.shadows.cascade_count = 4;
        l.shadows.filter = ShadowFilter::Pcf;
        l.shadows.atlas_resolution = args.get(11).and_then(|s| s.parse().ok()).unwrap_or(4096);
        l.hemisphere_intensity = hemi;
        l.sky_colour = [0.8, 0.9, 1.0];
        l.ground_colour = [0.5, 0.55, 0.6];
        if let Some((r, g, b, scale)) = dbg {
            let mut dv = DebugVis::default();
            dv.active = true;
            dv.mode = DebugOutputMode::Replace;
            dv.channel_r = r;
            dv.channel_g = g;
            dv.channel_b = b;
            dv.scale = scale;
            l.debug_vis = dv;
        }
        l
    };

    // Offscreen targets.
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = (W * 4 + 255) / 256 * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let outputs: Vec<(
        &str,
        Option<(DebugQuantity, DebugQuantity, DebugQuantity, f32)>,
        bool,
    )> = vec![
        ("normal", None, false),
        ("noshadow", None, false),
        ("nohemi", None, false),
        ("nohemi_noshadow", None, false),
        (
            "shadowfactor",
            Some((
                DebugQuantity::ShadowFactor,
                DebugQuantity::Zero,
                DebugQuantity::Zero,
                1.0,
            )),
            false,
        ),
        (
            "ndotl",
            Some((
                DebugQuantity::NdotL,
                DebugQuantity::Zero,
                DebugQuantity::Zero,
                1.0,
            )),
            false,
        ),
        (
            "cascade",
            Some((
                DebugQuantity::CascadeIndex,
                DebugQuantity::CascadeIndex,
                DebugQuantity::CascadeIndex,
                1.0,
            )),
            false,
        ),
        (
            "depthcmp",
            Some((
                DebugQuantity::BiasedDepth,
                DebugQuantity::SurfaceDepth,
                DebugQuantity::NdotL,
                1.0,
            )),
            false,
        ),
        ("atlas", None, true),
    ];

    for (name, dbg, show_atlas) in outputs {
        let mut fd = FrameData::new(
            CameraFrame::from_camera(&camera, [W as f32, H as f32]),
            SceneFrame::from_surface_items(build_items()),
        );
        let shadows_on = !name.contains("noshadow");
        let hemi = if name.contains("nohemi") { 0.0 } else { 0.2 };
        fd.effects.lighting = build_lighting(dbg, shadows_on, hemi);
        fd.effects.debug.show_shadow_atlas = show_atlas;
        fd.effects.debug.atlas_viewer_corner = AtlasViewerCorner::BottomRight;
        fd.effects.debug.atlas_viewer_scale = 0.6;

        let cmds = renderer.pass().prepare(&device, &queue, &fd);
        queue.submit(cmds);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("paint"),
        });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.12,
                            g: 0.12,
                            b: 0.13,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            renderer.pass_view().paint(&mut rp, &fd);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(H),
                },
            },
            wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        rx.recv().expect("map").expect("map ok");
        let data = slice.get_mapped_range();

        let path = format!("{}_{}.ppm", prefix, name);
        let mut out = Vec::with_capacity((W * H * 3) as usize + 32);
        out.extend_from_slice(format!("P6\n{} {}\n255\n", W, H).as_bytes());
        for y in 0..H {
            let row = &data[(y * bytes_per_row) as usize..];
            for x in 0..W {
                let px = &row[(x * 4) as usize..(x * 4 + 4) as usize];
                // BGRA -> RGB
                out.push(px[2]);
                out.push(px[1]);
                out.push(px[0]);
            }
        }
        std::fs::write(&path, &out).expect("write ppm");
        drop(data);
        readback.unmap();
        println!("wrote {}", path);
    }
}
