//! Precision test bed for scenes far from the world origin (eframe / egui).
//!
//! One small scene (a building shell with a point light inside, a clip plane
//! through it, a decal on the back wall and a label at the door, seen by an
//! orbit camera a few metres away) placed 0, 10 km, 1 000 km or 6 400 km
//! from the world origin. Everything that carries a world position here is
//! f64: the nodes' `Transform`s, `LightKind::Point`'s position,
//! `ClipShape::Plane`'s distance, the label's `OverlayOrigin::World` anchor,
//! the camera's `Pose` and `PickHit::world_pos`.
//!
//! - **Distance** moves the whole scene, light, clip plane, decal, label and
//!   camera target along +X.
//! - **World origin** picks what the renderer measures GPU positions from:
//!   `WorldOrigin::Zero` (world coordinates, which run out of f32 precision
//!   far out), `WorldOrigin::Fixed` at the scene centre, or
//!   `WorldOrigin::snapped()`, which follows the camera on a grid. It is set
//!   per frame through `SceneFrame::world_origin`. The readout shows the
//!   origin the renderer drew relative to, and counts the frames it moved on.
//! - **Difference** adds a post-effect stage (`PostEffectStage`) that shows
//!   the absolute difference between each frame and the one before, times the
//!   gain, while the camera drifts slowly round the building: under `Zero` at
//!   1 000 km and beyond every surface shimmers as positions snap between f32
//!   steps; under `snapped()` only true motion at the edges shows.
//! - A click picks the surface under the cursor and prints the hit's world
//!   position with its distance from the nearest authored surface.
//!
//! The example is meant to grow: further precision checks (other item types,
//! shadows, picking modes) get added here as controls on the same scene.
//!
//! `VPL_CAPTURE=<dir>` renders a short tour (the opening view, `Zero`, 6 400
//! km under `Zero`, then `snapped()`), writes one BMP per step to
//! `<dir>/precision_<step>.bmp` and closes the window.
//!
//! Left/middle drag orbits, right drag pans, scroll zooms.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::eframe::egui;
use glam::{DVec3, Mat3, Vec3};
use viewport_lib as vpl;
use viewport_lib::Colour;
pub use viewport_lib_examples_eframe::eframe;
use viewport_lib_plugins::item_types::decal::{DecalBlendMode, DecalItem};
use vpl::input::adapters::{from_egui, modifiers_from_egui};
use vpl::wgpu;
use vpl::{
    BindingPreset, Camera, CameraFrame, ClipObject, ClipShape, FrameData, InputOwnership,
    KeyboardOwnership, LabelItem, LightKind, LightSource, LightingSettings, Material, MaterialId,
    OffscreenViewportTarget, OrbitCameraController, PickBackend, PickMask, PointerOwnership,
    PostEffectContext, PostEffectResizeContext, PostEffectStage, Selection, TextureId, Transform,
    ViewportContext, ViewportInput, ViewportRenderer, WorldOrigin, scene::Scene,
};

/// The distances the selector steps through, in metres.
const DISTANCES: [(f64, &str); 4] = [
    (0.0, "0"),
    (10_000.0, "10 km"),
    (1_000_000.0, "1 000 km"),
    (6_400_000.0, "6 400 km"),
];

/// The building's boxes: (name, local centre, size), metres.
const BOXES: [(&str, [f64; 3], [f32; 3]); 8] = [
    ("Floor", [0.0, 0.0, -0.1], [4.0, 4.0, 0.2]),
    ("Back wall", [0.0, 1.9, 1.2], [4.0, 0.2, 2.4]),
    ("Left wall", [-1.9, 0.0, 1.2], [0.2, 4.0, 2.4]),
    ("Right wall", [1.9, 0.0, 1.2], [0.2, 4.0, 2.4]),
    ("Front wall left", [-1.25, -1.9, 1.2], [1.5, 0.2, 2.4]),
    ("Front wall right", [1.25, -1.9, 1.2], [1.5, 0.2, 2.4]),
    ("Lintel", [0.0, -1.9, 2.1], [1.0, 0.2, 0.6]),
    ("Plinth", [0.6, 0.6, 0.25], [0.5, 0.5, 0.5]),
];
/// The point light inside the building, local.
const LIGHT_AT: [f64; 3] = [-0.6, 0.4, 1.9];
/// Where the clip plane cuts, along +X from the centre.
const CLIP_X: f64 = 0.4;

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    eframe::run_native(
        "viewport-lib : Precision far from the origin",
        viewport_lib_examples_eframe::native_options([1400.0, 900.0]),
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            // The target is the sRGB variant of egui's format so the tone map's
            // linear output is encoded on write; egui samples its non-sRGB view.
            let format = OffscreenViewportTarget::render_format(rs.target_format);
            let mut renderer = ViewportRenderer::new(&rs.device, format);
            viewport_lib_plugins::item_types::install(&mut renderer, &rs.device);
            Ok(Box::new(App::new(
                rs.device.clone(),
                rs.queue.clone(),
                renderer,
            )))
        }),
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum OriginSel {
    Zero,
    Fixed,
    Snapped,
}

/// The settings the difference stage reads, shared with the app.
struct DiffSettings {
    enabled: AtomicBool,
    gain_bits: AtomicU32,
}

/// The app-owned offscreen render target and its egui texture registration.
struct Target {
    inner: OffscreenViewportTarget,
    id: egui::TextureId,
}

struct App {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: ViewportRenderer,
    target: Option<Target>,
    camera: Camera,
    input: ViewportInput,
    orbit: OrbitCameraController,
    scene: Scene,
    selection: Selection,
    nodes: Vec<(vpl::NodeId, [f64; 3], Mat3)>,
    distance: usize,
    origin: OriginSel,
    clip: bool,
    drift: bool,
    difference: bool,
    gain: f32,
    diff: Arc<DiffSettings>,
    decal_texture: TextureId,
    /// The origin the renderer last drew relative to, and how often it moved.
    seen_origin: Option<DVec3>,
    origin_moves: u32,
    last_pick: Option<(DVec3, f64)>,
    capture: Option<Capture>,
}

fn upload_material(renderer: &mut ViewportRenderer, material: Material) -> MaterialId {
    renderer
        .resources_mut()
        .upload_material(material)
        .expect("material upload")
}

impl App {
    fn new(device: wgpu::Device, queue: wgpu::Queue, mut renderer: ViewportRenderer) -> Self {
        let mut scene = Scene::new();
        let mut nodes = Vec::new();
        let cube = renderer
            .resources_mut()
            .upload_mesh_data(&device, &vpl::primitives::cube(1.0))
            .expect("cube upload");
        let wall = upload_material(
            &mut renderer,
            Material::pbr(Colour::linear_rgb(0.75, 0.72, 0.68), 0.0, 0.8),
        );
        let floor = upload_material(
            &mut renderer,
            Material::pbr(Colour::linear_rgb(0.45, 0.4, 0.36), 0.0, 0.7),
        );
        for (name, centre, size) in BOXES {
            let mat = if name == "Floor" { floor } else { wall };
            let rs = Mat3::from_diagonal(Vec3::from(size));
            let id = scene.add_named(
                name,
                Some(cube),
                Transform::from_parts(DVec3::from(centre), rs),
                mat,
            );
            nodes.push((id, centre, rs));
        }
        // The ground, left unclipped so the cut only opens the building.
        let ground = renderer
            .resources_mut()
            .upload_mesh_data(&device, &vpl::primitives::cuboid(40.0, 40.0, 0.1))
            .expect("ground upload");
        let ground_mat = upload_material(
            &mut renderer,
            Material::pbr(Colour::linear_rgb(0.3, 0.38, 0.28), 0.0, 0.95),
        );
        let local = [0.0, 0.0, -0.26];
        let id = scene.add_named(
            "Ground",
            Some(ground),
            Transform::from_translation(DVec3::from(local)),
            ground_mat,
        );
        let mut settings = vpl::ItemSettings::default();
        settings.ignore_clip = true;
        scene.set_appearance(id, settings);
        nodes.push((id, local, Mat3::IDENTITY));

        let decal_texture = renderer
            .resources_mut()
            .upload_texture(&device, &queue, checker())
            .expect("decal texture upload");

        let gain = 8.0f32;
        let diff = Arc::new(DiffSettings {
            enabled: AtomicBool::new(false),
            gain_bits: AtomicU32::new(gain.to_bits()),
        });
        let stage = DiffStage {
            settings: diff.clone(),
            bgl: None,
            pipeline: None,
            uniform: None,
            per_viewport: HashMap::new(),
        };
        renderer
            .add_post_effect_stage(
                Box::new(stage),
                vpl::plugin_api::post_effect::stage_order::EXTERNAL_DEFAULT + 50,
            )
            .expect("difference stage");

        let mut app = Self {
            device,
            queue,
            renderer,
            target: None,
            camera: Camera::orbit(DVec3::ZERO, 7.0, glam::Quat::IDENTITY),
            input: ViewportInput::from_preset(BindingPreset::Default),
            orbit: OrbitCameraController::new_stateless(),
            scene,
            selection: Selection::new(),
            nodes,
            distance: 2,
            origin: OriginSel::Snapped,
            clip: true,
            drift: true,
            difference: false,
            gain,
            diff,
            decal_texture,
            seen_origin: None,
            origin_moves: 0,
            last_pick: None,
            capture: Capture::from_env(),
        };
        app.place();
        app.camera = Camera::orbit(
            app.base() + DVec3::new(0.0, 0.0, 1.0),
            7.0,
            glam::Quat::from_rotation_z(0.5) * glam::Quat::from_rotation_x(1.0),
        );
        app
    }

    /// The scene centre: the selected distance along +X.
    fn base(&self) -> DVec3 {
        DVec3::new(DISTANCES[self.distance].0, 0.0, 0.0)
    }

    fn world_origin(&self) -> WorldOrigin {
        match self.origin {
            OriginSel::Zero => WorldOrigin::Zero,
            OriginSel::Fixed => WorldOrigin::Fixed(self.base()),
            OriginSel::Snapped => WorldOrigin::snapped(),
        }
    }

    /// Distance from `p` to the nearest authored box surface, in f64.
    fn surface_distance(&self, p: DVec3) -> f64 {
        let base = self.base();
        BOXES
            .iter()
            .map(|(_, c, size)| {
                let half = Vec3::from(*size).as_dvec3() * 0.5;
                let q = (p - base - DVec3::from(*c)).abs() - half;
                let outside = q.max(DVec3::ZERO).length();
                let inside = q.max_element().min(0.0);
                (outside + inside).abs()
            })
            .fold(f64::INFINITY, f64::min)
    }

    fn lighting(&self) -> LightingSettings {
        let mut sun = LightSource::default();
        sun.kind = LightKind::Directional {
            direction: [0.4, -0.6, 1.0],
        };
        sun.intensity = 1.6;
        let mut lamp = LightSource::default();
        lamp.kind = LightKind::Point {
            position: self.base() + DVec3::from(LIGHT_AT),
            range: 8.0,
            radius: 0.05,
        };
        lamp.colour = Colour::linear_rgb(1.0, 0.75, 0.45);
        lamp.intensity = 6.0;
        lamp.cast_shadows = true;
        let mut l = LightingSettings::default();
        l.lights = vec![sun, lamp];
        l.shadows.enabled = true;
        l.hemisphere_intensity = 0.35;
        l
    }

    /// Move every node to the current distance.
    fn place(&mut self) {
        let base = self.base();
        for (id, local, rs) in &self.nodes {
            self.scene
                .set_local_transform(*id, Transform::from_parts(base + DVec3::from(*local), *rs));
        }
    }

    /// Move the scene to distance index `i` and recentre the camera on it.
    fn set_distance(&mut self, i: usize) {
        self.distance = i;
        self.place();
        self.last_pick = None;
        let centre = self.base() + DVec3::new(0.0, 0.0, 1.0);
        self.camera.set_orbit_centre(centre);
    }

    /// The frame for this tick: the scene plus everything placed per frame.
    fn frame_data(&mut self, size: [f32; 2], pixels_per_point: f32) -> FrameData {
        let mut fd = FrameData::from_scene(
            CameraFrame::from_camera(&self.camera, size).with_pixels_per_point(pixels_per_point),
            &mut self.scene,
            &self.selection,
        );
        let base = self.base();
        fd.effects.lighting = self.lighting();
        fd.viewport.show_axes_indicator = true;
        fd.scene.world_origin = Some(self.world_origin());
        if self.clip {
            let mut clip = ClipObject::default();
            clip.shape = ClipShape::Plane {
                normal: [-1.0, 0.0, 0.0],
                distance: base.x + CLIP_X,
                cap_colour: None,
            };
            fd.effects.clip.objects = vec![clip];
        }
        // A decal on the inside of the back wall. Its projector is an f32
        // matrix in world space.
        let at = base + DVec3::new(-0.6, 1.8, 1.3);
        let mut decal = DecalItem::default();
        decal.transform = (glam::Mat4::from_translation(at.as_vec3())
            * glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * glam::Mat4::from_scale(Vec3::new(1.0, 1.0, 0.4)))
        .to_cols_array_2d();
        decal.texture_id = self.decal_texture;
        decal.blend_mode = DecalBlendMode::Replace;
        decal.alpha = 1.0;
        fd.scene.items_mut::<DecalItem>().push(decal);
        fd.overlays.labels.push(
            LabelItem::new("Door")
                .with_world_anchor(base + DVec3::new(0.0, -2.0, 2.6))
                .with_font_size(14.0),
        );
        self.diff.enabled.store(self.difference, Ordering::Relaxed);
        self.diff
            .gain_bits
            .store(self.gain.to_bits(), Ordering::Relaxed);
        fd
    }

    /// Set up step `step` of the capture tour. Returns false past the last.
    fn tour(&mut self, step: u32) -> bool {
        match step {
            0 => {}
            1 => self.origin = OriginSel::Zero,
            2 => self.set_distance(3),
            3 => self.origin = OriginSel::Snapped,
            _ => return false,
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Difference stage
// ---------------------------------------------------------------------------

const DIFF_WGSL: &str = r#"
@group(0) @binding(0) var current: texture_2d<f32>;
@group(0) @binding(1) var previous: texture_2d<f32>;
@group(0) @binding(2) var<uniform> gain: vec4<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(3.0, 1.0), vec2(-1.0, 1.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let a = textureLoad(current, p, 0).rgb;
    let b = textureLoad(previous, p, 0).rgb;
    return vec4<f32>(min(abs(a - b) * gain.x, vec3<f32>(1.0)), 1.0);
}
"#;

struct DiffViewport {
    input: wgpu::Texture,
    input_view: wgpu::TextureView,
    previous: wgpu::Texture,
    size: [u32; 2],
    bind_group: wgpu::BindGroup,
}

/// A display-space stage drawing `|frame - previous frame| * gain`, then
/// keeping this frame for the next one.
struct DiffStage {
    settings: Arc<DiffSettings>,
    bgl: Option<wgpu::BindGroupLayout>,
    pipeline: Option<wgpu::RenderPipeline>,
    uniform: Option<wgpu::Buffer>,
    per_viewport: HashMap<usize, DiffViewport>,
}

impl PostEffectStage for DiffStage {
    fn type_name(&self) -> &'static str {
        "frame_difference"
    }

    fn enabled(&self) -> bool {
        self.settings.enabled.load(Ordering::Relaxed)
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        self.bgl = Some(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("frame_difference"),
                entries: &[
                    texture(0),
                    texture(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }),
        );
        self.uniform = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_difference_gain"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.pipeline = None;
        self.per_viewport.clear();
    }

    fn on_viewport_resized(&mut self, device: &wgpu::Device, ctx: &PostEffectResizeContext<'_>) {
        let (Some(bgl), Some(uniform)) = (&self.bgl, &self.uniform) else {
            return;
        };
        if self.pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("frame_difference"),
                source: wgpu::ShaderSource::Wgsl(DIFF_WGSL.into()),
            });
            self.pipeline = Some(vpl::plugin_api::post_effect::build_post_effect_pipeline(
                device,
                "frame_difference",
                &shader,
                bgl,
                ctx.target_format,
                None,
            ));
        }
        let size = [ctx.scene_size[0].max(1), ctx.scene_size[1].max(1)];
        let make = |label, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ctx.target_format,
                usage,
                view_formats: &[],
            })
        };
        let input = make(
            "frame_difference_input",
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let previous = make(
            "frame_difference_previous",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let input_view = input.create_view(&wgpu::TextureViewDescriptor::default());
        let previous_view = previous.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame_difference"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&previous_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        self.per_viewport.insert(
            ctx.viewport_index,
            DiffViewport {
                input,
                input_view,
                previous,
                size,
                bind_group,
            },
        );
    }

    fn prepare(&mut self, queue: &wgpu::Queue, _ctx: &PostEffectContext<'_>) {
        if let Some(uniform) = &self.uniform {
            let gain = f32::from_bits(self.settings.gain_bits.load(Ordering::Relaxed));
            queue.write_buffer(uniform, 0, bytemuck::cast_slice(&[gain, 0.0, 0.0, 0.0]));
        }
    }

    fn input_view(&self, viewport_index: usize) -> &wgpu::TextureView {
        &self.per_viewport[&viewport_index].input_view
    }

    fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        ctx: &PostEffectContext<'_>,
    ) {
        let (Some(pipeline), Some(vp)) =
            (&self.pipeline, self.per_viewport.get(&ctx.viewport_index))
        else {
            return;
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame_difference"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &vp.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // Keep this frame for the next one.
        let copy = |texture| wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };
        encoder.copy_texture_to_texture(
            copy(&vp.input),
            copy(&vp.previous),
            wgpu::Extent3d {
                width: vp.size[0],
                height: vp.size[1],
                depth_or_array_layers: 1,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Decal texture
// ---------------------------------------------------------------------------

/// A small checker for the decal.
fn checker() -> vpl::TextureData {
    let n = 8u32;
    let px = (0..n * n)
        .flat_map(|i| {
            let c = if (i % n + i / n) % 2 == 0 {
                [220, 70, 40]
            } else {
                [240, 220, 200]
            };
            [c[0], c[1], c[2], 255]
        })
        .collect();
    vpl::TextureData::srgb(n, n, px)
}

// ---------------------------------------------------------------------------
// Capture tour
// ---------------------------------------------------------------------------

/// The `VPL_CAPTURE` tour: its output directory and how far it has got.
struct Capture {
    dir: std::path::PathBuf,
    step: u32,
    frames: u32,
}

/// Frames drawn per tour step before the capture, so pipeline compiles and
/// shadows settle.
const CAPTURE_SETTLE_FRAMES: u32 = 45;

impl Capture {
    fn from_env() -> Option<Self> {
        let dir = std::env::var_os("VPL_CAPTURE")?;
        Some(Self {
            dir: dir.into(),
            step: 0,
            frames: 0,
        })
    }
}

/// Write RGBA8 pixels as a 32-bit BMP (bottom-up rows, BGRA).
fn write_bmp(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let size = 54 + w * h * 4;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(b"BM");
    for v in [size, 0, 54, 40] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    for v in [0u32, w * h * 4, 2835, 2835, 0, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for y in (0..h).rev() {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            out.extend_from_slice(&[rgba[i + 2], rgba[i + 1], rgba[i], 255]);
        }
    }
    std::fs::write(path, out)
}

impl App {
    /// Advance the capture tour by one frame. Returns `None` with no tour,
    /// otherwise whether the tour has finished.
    fn capture_step(&mut self, fd: &FrameData, size_px: [u32; 2]) -> Option<bool> {
        let cap = self.capture.as_mut()?;
        cap.frames += 1;
        if cap.frames < CAPTURE_SETTLE_FRAMES {
            return Some(false);
        }
        let [w, h] = size_px;
        let pixels = self
            .renderer
            .render_offscreen(&self.device, &self.queue, fd, w, h);
        let path = cap.dir.join(format!("precision_{}.bmp", cap.step));
        if let Err(e) = write_bmp(&path, w, h, &pixels) {
            eprintln!("capture: {}: {e}", path.display());
        }
        cap.step += 1;
        cap.frames = 0;
        let step = cap.step;
        Some(!self.tour(step))
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let rs = frame
            .wgpu_render_state()
            .expect("wgpu backend required")
            .clone();

        egui::SidePanel::left("precision_panel")
            .min_width(300.0)
            .max_width(380.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

                let ppp = ui.ctx().pixels_per_point();
                let size_px = [
                    (rect.width() * ppp).round().max(1.0) as u32,
                    (rect.height() * ppp).round().max(1.0) as u32,
                ];
                if self
                    .target
                    .as_ref()
                    .map_or(true, |t| t.inner.size() != size_px)
                {
                    if let Some(old) = self.target.take() {
                        rs.renderer.write().free_texture(&old.id);
                    }
                    let inner = OffscreenViewportTarget::new(&rs.device, rs.target_format, size_px);
                    let id = rs.renderer.write().register_native_texture(
                        &rs.device,
                        inner.sample_view(),
                        wgpu::FilterMode::Linear,
                    );
                    self.target = Some(Target { inner, id });
                }

                // Camera input. A capture tour keeps the opening camera.
                let size = [rect.width(), rect.height()];
                self.input.begin_frame(ViewportContext::new(
                    InputOwnership {
                        pointer: if response.hovered() {
                            PointerOwnership::Owned
                        } else {
                            PointerOwnership::Elsewhere
                        },
                        keyboard: if response.hovered() && !ui.ctx().wants_keyboard_input() {
                            KeyboardOwnership::Owned
                        } else {
                            KeyboardOwnership::Elsewhere
                        },
                    },
                    size,
                ));
                let origin = glam::Vec2::new(rect.left(), rect.top());
                ui.input(|i| {
                    self.input.push_event(vpl::ViewportEvent::ModifiersChanged(
                        modifiers_from_egui(&i.modifiers),
                    ));
                    for event in &i.events {
                        if let Some(ev) = from_egui(event, origin) {
                            self.input.push_event(ev);
                        }
                    }
                });
                let dt = ctx.input(|i| i.stable_dt).min(0.1);
                if self.capture.is_none() {
                    if self.drift {
                        let o = self.camera.orientation();
                        self.camera
                            .set_orbit_orientation(glam::Quat::from_rotation_z(dt * 0.05) * o);
                    }
                    let action = self.input.resolve();
                    self.orbit.set_viewport_size(size);
                    self.orbit.apply(&mut self.camera, &action, dt);
                }
                self.camera.set_aspect_ratio(rect.width(), rect.height());

                let fd = self.frame_data(size, ppp);
                let target = self.target.as_ref().unwrap();
                let cmd = self.renderer.owned().render(
                    &self.device,
                    &self.queue,
                    target.inner.render_view(),
                    &fd,
                );
                self.queue.submit(std::iter::once(cmd));
                let tex_id = target.id;

                // What the renderer drew relative to this frame.
                let drawn = self.renderer.world_origin();
                if self.seen_origin.is_some_and(|o| o != drawn) {
                    self.origin_moves += 1;
                }
                self.seen_origin = Some(drawn);

                // Resolve a click against the frame just drawn.
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let local = glam::Vec2::new(pos.x - rect.left(), pos.y - rect.top());
                        let hit = self.renderer.pick_object(
                            PickBackend::Gpu,
                            local,
                            &fd,
                            &self.device,
                            &self.queue,
                            PickMask::OBJECT,
                        );
                        self.last_pick =
                            hit.map(|h| (h.world_pos, self.surface_distance(h.world_pos)));
                    }
                }

                if let Some(done) = self.capture_step(&fd, size_px) {
                    if done {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    ctx.request_repaint();
                }

                ui.painter().image(
                    tex_id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            });

        if self.drift || self.difference {
            ctx.request_repaint();
        }
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

impl App {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Precision far from the origin");
        ui.separator();
        ui.label("Distance from the world origin:");
        let mut distance = self.distance;
        ui.horizontal(|ui| {
            for (i, (_, label)) in DISTANCES.iter().enumerate() {
                ui.selectable_value(&mut distance, i, *label);
            }
        });
        if distance != self.distance {
            self.set_distance(distance);
        }
        ui.separator();
        ui.label("World origin:");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.origin, OriginSel::Zero, "Zero");
            ui.selectable_value(&mut self.origin, OriginSel::Fixed, "Fixed at the scene");
            ui.selectable_value(&mut self.origin, OriginSel::Snapped, "Snapped");
        });
        ui.checkbox(&mut self.clip, "Clip plane through the building");
        ui.checkbox(&mut self.drift, "Camera drifts slowly");
        ui.checkbox(&mut self.difference, "Difference from the previous frame");
        ui.add_enabled(
            self.difference,
            egui::Slider::new(&mut self.gain, 1.0..=64.0)
                .logarithmic(true)
                .text("Gain"),
        );

        ui.separator();
        let fmt = |v: DVec3| format!("({:.3}, {:.3}, {:.3})", v.x, v.y, v.z);
        egui::Grid::new("precision_readout")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Scene centre");
                ui.label(fmt(self.base()));
                ui.end_row();
                ui.label("Camera eye (Pose)");
                ui.label(fmt(self.camera.pose.position));
                ui.end_row();
                ui.label("Drawn relative to");
                ui.label(self.seen_origin.map_or("-".to_string(), fmt));
                ui.end_row();
                ui.label("Origin moves");
                ui.label(self.origin_moves.to_string());
                ui.end_row();
            });
        ui.weak(
            "Under Snapped the origin jumps to the grid point nearest the eye when the eye \
             strays more than about half a cell: each jump counts above.",
        );
        ui.separator();
        ui.label("Click a surface to pick it.");
        if let Some((pos, dist)) = self.last_pick {
            ui.label(format!("world_pos {}", fmt(pos)));
            ui.label(format!("{:.4} m from the nearest authored surface", dist));
        }
    }
}
