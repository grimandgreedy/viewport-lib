//! Showcase 61: Large Worlds.
//!
//! One small scene (a building shell with a point light inside, a clip plane
//! through it, a decal on the back wall and a label at the door, seen by an
//! orbit camera a few metres away) placed 0, 10 km, 1 000 km or 6 400 km
//! from the world origin. Everything that carries a world position here is
//! f64: the nodes' `Transform`s, `LightKind::Point`'s position,
//! `ClipShape::Plane`'s distance, the label's `OverlayOrigin::World` anchor,
//! the camera's `Pose` and `PickHit::world_pos`.
//!
//! The **World origin** chip picks what the renderer measures GPU positions
//! from: `WorldOrigin::Zero` (world coordinates, which run out of f32
//! precision far out), `WorldOrigin::Fixed` at the scene centre, or
//! `WorldOrigin::snapped()`, which follows the camera on a grid. It is set
//! per frame through `SceneFrame::world_origin`. The readout shows the origin
//! the renderer drew relative to, and counts the frames it moved on.
//!
//! **Difference** adds a post-effect stage (`PostEffectStage`) that shows the
//! absolute difference between each frame and the one before, amplified,
//! while the camera drifts slowly round the building: under `Zero` at
//! 1 000 km and beyond every surface shimmers as positions snap between f32
//! steps; under `snapped()` only true motion at the edges shows.
//!
//! A click picks the surface under the cursor and prints the hit's
//! world position with its distance from the nearest authored surface.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::App;
use crate::eframe::egui;
use crate::shared::upload_material;
use glam::{DVec3, Mat3, Vec3};
use viewport_lib as vpl;
use viewport_lib::Colour;
use viewport_lib_plugins::item_types::decal::{DecalBlendMode, DecalItem};
use vpl::wgpu;
use vpl::{
    ClipObject, ClipShape, LabelItem, LightKind, LightSource, LightingSettings, Material,
    PostEffectContext, PostEffectResizeContext, PostEffectStage, TextureId, Transform,
    ViewportRenderer, WorldOrigin, scene::Scene,
};

/// The distances the slider steps through, in metres.
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum OriginSel {
    Zero,
    Fixed,
    Snapped,
}

/// The settings the difference stage reads, shared with the showcase.
struct DiffSettings {
    enabled: AtomicBool,
    gain_bits: AtomicU32,
}

pub(crate) struct LargeWorldsState {
    built: bool,
    scene: Scene,
    nodes: Vec<(vpl::NodeId, [f64; 3], Mat3)>,
    pub distance: usize,
    pub origin: OriginSel,
    pub clip: bool,
    pub drift: bool,
    pub difference: bool,
    pub gain: f32,
    diff: Option<Arc<DiffSettings>>,
    decal_texture: Option<TextureId>,
    /// The origin the renderer last drew relative to, and how often it moved.
    seen_origin: Option<DVec3>,
    origin_moves: u32,
    last_pick: Option<(DVec3, f64)>,
}

impl Default for LargeWorldsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            nodes: Vec::new(),
            distance: 2,
            origin: OriginSel::Snapped,
            clip: true,
            drift: true,
            difference: false,
            gain: 8.0,
            diff: None,
            decal_texture: None,
            seen_origin: None,
            origin_moves: 0,
            last_pick: None,
        }
    }
}

impl LargeWorldsState {
    /// The scene centre: the authored distance along +X.
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
        "showcase_frame_difference"
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
// Build
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

fn build(app: &mut App, renderer: &mut ViewportRenderer) {
    let st = &mut app.worlds_state;
    st.scene = Scene::new();
    st.nodes.clear();
    let cube = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &vpl::primitives::cube(1.0))
        .expect("cube upload");
    let wall = upload_material(
        renderer,
        Material::pbr(Colour::linear_rgb(0.75, 0.72, 0.68), 0.0, 0.8),
    );
    let floor = upload_material(
        renderer,
        Material::pbr(Colour::linear_rgb(0.45, 0.4, 0.36), 0.0, 0.7),
    );
    for (name, centre, size) in BOXES {
        let mat = if name == "Floor" { floor } else { wall };
        let rs = Mat3::from_diagonal(Vec3::from(size));
        let id = st.scene.add_named(
            name,
            Some(cube),
            Transform::from_parts(DVec3::from(centre), rs),
            mat,
        );
        st.nodes.push((id, centre, rs));
    }
    // The ground, left unclipped so the cut only opens the building.
    let ground = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &vpl::primitives::cuboid(40.0, 40.0, 0.1))
        .expect("ground upload");
    let ground_mat = upload_material(
        renderer,
        Material::pbr(Colour::linear_rgb(0.3, 0.38, 0.28), 0.0, 0.95),
    );
    let local = [0.0, 0.0, -0.26];
    let id = st.scene.add_named(
        "Ground",
        Some(ground),
        Transform::from_translation(DVec3::from(local)),
        ground_mat,
    );
    let mut settings = vpl::ItemSettings::default();
    settings.ignore_clip = true;
    st.scene.set_appearance(id, settings);
    st.nodes.push((id, local, Mat3::IDENTITY));
    st.place();

    if st.decal_texture.is_none() {
        st.decal_texture = Some(
            renderer
                .resources_mut()
                .upload_texture(&app.device, &app.queue, checker())
                .expect("decal texture upload"),
        );
    }
    if st.diff.is_none() {
        let settings = Arc::new(DiffSettings {
            enabled: AtomicBool::new(false),
            gain_bits: AtomicU32::new(st.gain.to_bits()),
        });
        let stage = DiffStage {
            settings: settings.clone(),
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
        st.diff = Some(settings);
    }
    st.built = true;
    let base = st.base();
    app.camera = vpl::Camera::orbit(
        base + DVec3::new(0.0, 0.0, 1.0),
        7.0,
        glam::Quat::from_rotation_z(0.5) * glam::Quat::from_rotation_x(1.0),
    );
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

fn controls(app: &mut App, ui: &mut egui::Ui, frame: &crate::eframe::Frame) {
    let mut moved = false;
    {
        let st = &mut app.worlds_state;
        ui.label("Distance from the world origin:");
        ui.horizontal(|ui| {
            for (i, (_, label)) in DISTANCES.iter().enumerate() {
                moved |= ui.selectable_value(&mut st.distance, i, *label).changed();
            }
        });
        ui.separator();
        ui.label("World origin:");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut st.origin, OriginSel::Zero, "Zero");
            ui.selectable_value(&mut st.origin, OriginSel::Fixed, "Fixed at the scene");
            ui.selectable_value(&mut st.origin, OriginSel::Snapped, "Snapped");
        });
        ui.checkbox(&mut st.clip, "Clip plane through the building");
        ui.checkbox(&mut st.drift, "Camera drifts slowly");
        ui.checkbox(&mut st.difference, "Difference from the previous frame");
        ui.add_enabled(
            st.difference,
            egui::Slider::new(&mut st.gain, 1.0..=64.0)
                .logarithmic(true)
                .text("Gain"),
        );
    }
    if moved {
        let st = &mut app.worlds_state;
        st.place();
        st.last_pick = None;
        let centre = st.base() + DVec3::new(0.0, 0.0, 1.0);
        app.camera.set_orbit_center(centre);
    }

    // What the renderer drew relative to last frame.
    if let Some(rs) = frame.wgpu_render_state() {
        let guard = rs.renderer.read();
        if let Some(renderer) = guard.callback_resources.get::<ViewportRenderer>() {
            let origin = renderer.world_origin();
            let st = &mut app.worlds_state;
            if st.seen_origin.is_some_and(|o| o != origin) {
                st.origin_moves += 1;
            }
            st.seen_origin = Some(origin);
        }
    }
    let st = &app.worlds_state;
    ui.separator();
    let fmt = |v: DVec3| format!("({:.3}, {:.3}, {:.3})", v.x, v.y, v.z);
    egui::Grid::new("worlds_readout")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("Scene centre");
            ui.label(fmt(st.base()));
            ui.end_row();
            ui.label("Camera eye (Pose)");
            ui.label(fmt(app.camera.pose.position));
            ui.end_row();
            ui.label("Drawn relative to");
            ui.label(st.seen_origin.map_or("-".to_string(), fmt));
            ui.end_row();
            ui.label("Origin moves");
            ui.label(st.origin_moves.to_string());
            ui.end_row();
        });
    ui.weak(
        "Under Snapped the origin jumps to the grid point nearest the eye when the eye \
         strays more than about half a cell: each jump counts above.",
    );
    ui.separator();
    ui.label("Click a surface to pick it.");
    if let Some((pos, dist)) = st.last_pick {
        ui.label(format!("world_pos {}", fmt(pos)));
        ui.label(format!("{:.4} m from the nearest authored surface", dist));
    }
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScLargeWorlds;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScLargeWorlds = ScLargeWorlds;

impl crate::Showcase for ScLargeWorlds {
    fn needs_build(&self, app: &crate::App) -> bool {
        !app.worlds_state.built
    }

    fn build(&self, app: &mut crate::App, renderer: &mut ViewportRenderer) {
        build(app, renderer)
    }

    fn scene(
        &self,
        app: &mut crate::App,
        _frame: &crate::eframe::Frame,
        _out: &mut crate::SceneOverrides,
    ) -> crate::SceneContents {
        let st = &mut app.worlds_state;
        let items = st.scene.collect_render_items(&vpl::Selection::new());
        crate::SceneContents {
            items,
            bg_colour: None,
            lighting: st.lighting(),
            scene_gen: st.scene.version(),
            sel_gen: 0,
        }
    }

    fn advance(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        if app.worlds_state.drift && app.capture.is_none() {
            let dt = cx.egui.input(|i| i.stable_dt).min(0.1);
            let o = app.camera.orientation();
            app.camera
                .set_orbit_orientation(glam::Quat::from_rotation_z(dt * 0.05) * o);
        }
    }

    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, _ctx: &crate::FrameCtx) {
        let st = &app.worlds_state;
        let base = st.base();
        fd.scene.world_origin = Some(st.world_origin());
        if st.clip {
            let mut clip = ClipObject::default();
            clip.shape = ClipShape::Plane {
                normal: [-1.0, 0.0, 0.0],
                distance: base.x + CLIP_X,
                cap_colour: None,
            };
            fd.effects.clip.objects = vec![clip];
        }
        if let Some(texture) = st.decal_texture {
            // A decal on the inside of the back wall. Its projector is an f32
            // matrix in world space.
            let at = base + DVec3::new(-0.6, 1.8, 1.3);
            let mut decal = DecalItem::default();
            decal.transform = (glam::Mat4::from_translation(at.as_vec3())
                * glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
                * glam::Mat4::from_scale(Vec3::new(1.0, 1.0, 0.4)))
            .to_cols_array_2d();
            decal.texture_id = texture;
            decal.blend_mode = DecalBlendMode::Replace;
            decal.alpha = 1.0;
            fd.scene.items_mut::<DecalItem>().push(decal);
        }
        fd.overlays.labels.push(
            LabelItem::new("Door")
                .with_world_anchor(base + DVec3::new(0.0, -2.0, 2.6))
                .with_font_size(14.0),
        );
        if let Some(diff) = &st.diff {
            diff.enabled.store(st.difference, Ordering::Relaxed);
            diff.gain_bits.store(st.gain.to_bits(), Ordering::Relaxed);
        }
    }

    fn tick(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        if app.worlds_state.drift || app.worlds_state.difference {
            cx.egui.request_repaint();
        }
    }

    fn on_click(&self, app: &mut crate::App, cx: &crate::ClickCtx) {
        app.pending_pick = Some(cx.pos);
    }

    fn on_pick(&self, app: &mut crate::App, hit: Option<vpl::PickHit>) {
        let st = &mut app.worlds_state;
        st.last_pick = hit.map(|h| (h.world_pos, st.surface_distance(h.world_pos)));
    }

    fn leave(&self, app: &mut crate::App) {
        if let Some(diff) = &app.worlds_state.diff {
            diff.enabled.store(false, Ordering::Relaxed);
        }
    }

    fn controls(&self, app: &mut crate::App, ui: &mut egui::Ui, frame: &crate::eframe::Frame) {
        controls(app, ui, frame)
    }

    fn tour(&self, app: &mut crate::App, step: u32) -> bool {
        let st = &mut app.worlds_state;
        match step {
            0 => {}
            1 => st.origin = OriginSel::Zero,
            2 => {
                st.distance = 3;
                st.place();
                let centre = st.base() + DVec3::new(0.0, 0.0, 1.0);
                app.camera.set_orbit_center(centre);
            }
            3 => st.origin = OriginSel::Snapped,
            _ => return false,
        }
        true
    }
}
