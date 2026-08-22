//! Render-path zoo: one mesh object per distinct path through the renderer.
//!
//! Each object is a different shape configured to take one reachable scene/shadow
//! path: instanced vs per-object, opaque vs transparent, one-sided vs two-sided,
//! plus the shading modifiers (matcap, scalar attribute, param-vis), a bound
//! position-override buffer, and the two heavy paths (compute-filter and GPU
//! skinning). The left panel drives lighting and shadows and toggles the HDR
//! (post-process + OIT) vs LDR inline path, GPU culling, and wireframe. The
//! light-kind selector switches between directional cascade shadows and
//! point/spot cube-face shadows.
//!
//! Click an object to pick it (unified GPU/CPU picker) and outline it; the legend
//! shows which path the picked object exercises. This makes a whole class of
//! path-specific bug visible at a glance: if any object renders or shadows wrong,
//! its path is broken.
//!
//! Controls:
//!   Left click   : pick one object (Shift to add)
//!   Left drag    : rubber-band box select (Shift to add)
//!   Middle drag  : orbit
//!   Right drag   : pan
//!   Scroll       : zoom

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    AttributeData, AttributeKind, AttributeRef, BackfacePattern, BackfacePolicy, BuiltinColourmap,
    BuiltinMatcap, ButtonState, Camera, CameraFrame, ColourmapId, ComputeFilterItem,
    ComputeFilterKind, FrameData, LightKind, LightSource, LightingSettings, MatcapId, Material,
    MeshId, OrbitCameraController, ParamVis, ParamVisMode, PatternConfig, PickId, PickMask,
    SceneFrame, SceneRenderItem, ScrollUnits, ShadingModel, ShadowFilter, ViewportContext,
    ViewportEvent, ViewportRenderer,
    material::AlphaMode,
    plugins::skinning::{SkinWeights, SkinningPlugin},
    primitives,
};

/// Legend: (pick id, short name, path it exercises). Order matches the grid.
const PATHS: &[(u64, &str, &str)] = &[
    (1, "Cube", "Instanced opaque, one-sided (Cull)"),
    (2, "Hemisphere", "Instanced opaque, two-sided (Identical)"),
    (3, "Sphere", "Instanced transparent, one-sided (OIT/blend)"),
    (
        4,
        "Torus",
        "Per-object transparent, two-sided (OIT back-cull)",
    ),
    (5, "Cone", "Per-object two-sided: DifferentColour backface"),
    (6, "Cylinder", "Per-object two-sided: Tint backface"),
    (7, "Ring", "Per-object two-sided: Pattern backface"),
    (8, "Icosphere", "Per-object: matcap shading"),
    (9, "Sphere", "Per-object: scalar attribute + colourmap"),
    (10, "Ellipsoid", "Per-object: UV param-vis (checker)"),
    (11, "Sphere", "Per-object: compute-filter (clip plane)"),
    (
        12,
        "Capsule",
        "Per-object: GPU skinning (per-instance deform)",
    ),
    (
        13,
        "Arrow",
        "Per-object transparent, one-sided (scalar + opacity)",
    ),
    (
        14,
        "Sphere",
        "Per-object: position-override buffer (radial bump)",
    ),
];

/// Three drag-value fields for a vec3, laid out horizontally.
fn ui_vec3(ui: &mut egui::Ui, v: &mut [f32; 3], speed: f64) {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut v[0]).speed(speed).prefix("x: "));
        ui.add(egui::DragValue::new(&mut v[1]).speed(speed).prefix("y: "));
        ui.add(egui::DragValue::new(&mut v[2]).speed(speed).prefix("z: "));
    });
}

/// Grid position for a pick id (1-based), laid out 4 columns on the ground.
fn grid_pos(pick_id: u64) -> glam::Vec3 {
    let i = (pick_id - 1) as i32;
    let col = i % 4;
    let row = i / 4;
    let x = (col - 1) as f32 * 3.0 - 1.5;
    let y = (1 - row) as f32 * 3.0;
    glam::Vec3::new(x, y, 0.95)
}

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Render Paths",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 900.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            // Request the features the renderer can use when the adapter has them
            // (eframe does not request any by default, which leaves
            // is_gpu_culling_supported() == false and gpu_frame_ms == None).
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                    eframe::egui_wgpu::WgpuSetupCreateNew {
                        device_descriptor: std::sync::Arc::new(|adapter| {
                            use eframe::wgpu;
                            let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                                wgpu::Limits::downlevel_webgl2_defaults()
                            } else {
                                viewport_lib::ViewportRenderer::recommended_device_limits(adapter)
                            };
                            wgpu::DeviceDescriptor {
                                label: Some("viewport-lib render-paths device"),
                                required_features:
                                    viewport_lib::ViewportRenderer::recommended_device_features(
                                        adapter,
                                    ),
                                required_limits: wgpu::Limits {
                                    max_texture_dimension_2d: 8192,
                                    ..base_limits
                                },
                                ..Default::default()
                            }
                        }),
                        ..Default::default()
                    },
                ),
                ..Default::default()
            },
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let device = &rs.device;
            let queue = &rs.queue;
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);
            // pick()/pick_rect() read a CPU-side cache that prepare() only builds
            // when this is on. Off by default, so enable it for picking to work.
            renderer.set_cpu_pick_cache(true);

            let meshes;
            let skinning;
            let skin_pivot_z;
            {
                let res = renderer.resources_mut();
                res.ensure_matcaps_initialized(device, queue);
                res.ensure_colourmaps_initialized(device, queue);

                // One mesh per object so the non-instanced objects each get their own
                // GPU state, and the instanced ones each form their own batch.
                let ground = res
                    .upload_mesh_data(device, &primitives::cuboid(26.0, 22.0, 0.5))
                    .expect("ground");
                let cube = res
                    .upload_mesh_data(device, &primitives::cube(1.3))
                    .expect("cube");
                let hemisphere = res
                    .upload_mesh_data(device, &primitives::hemisphere(0.9, 32, 16))
                    .expect("hemisphere");
                let sphere_t = res
                    .upload_mesh_data(device, &primitives::sphere(0.8, 32, 16))
                    .expect("sphere_t");
                let torus = res
                    .upload_mesh_data(device, &primitives::torus(0.6, 0.22, 32, 18))
                    .expect("torus");
                let cone = res
                    .upload_mesh_data(device, &primitives::cone(0.8, 1.6, 32))
                    .expect("cone");
                let cylinder = res
                    .upload_mesh_data(device, &primitives::cylinder(0.6, 1.6, 32))
                    .expect("cylinder");
                let ring = res
                    .upload_mesh_data(device, &primitives::ring(0.4, 0.9, 40))
                    .expect("ring");
                let icosphere = res
                    .upload_mesh_data(device, &primitives::icosphere(0.85, 2))
                    .expect("icosphere");

                // Scalar-attribute sphere: attach a per-vertex scalar so the item can
                // activate `active_attribute` (which forces the per-object path).
                let mut scalar_mesh = primitives::sphere(0.85, 40, 20);
                let scalars: Vec<f32> = scalar_mesh
                    .positions
                    .iter()
                    .map(|p| p[2]) // colour by height
                    .collect();
                scalar_mesh
                    .attributes
                    .insert("height".to_string(), AttributeData::Vertex(scalars));
                let scalar = res.upload_mesh_data(device, &scalar_mesh).expect("scalar");

                let ellipsoid = res
                    .upload_mesh_data(device, &primitives::ellipsoid(0.9, 0.6, 1.0, 32, 16))
                    .expect("ellipsoid");
                let filter = res
                    .upload_mesh_data(device, &primitives::sphere(0.9, 36, 18))
                    .expect("filter");

                // Skinned capsule: build weights from vertex height, attach to the
                // skinning deformer. The joint palette is uploaded per frame in the callback.
                let capsule = primitives::capsule(0.4, 1.8, 24, 12);
                let (zmin, zmax) = capsule
                    .positions
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                        (lo.min(p[2]), hi.max(p[2]))
                    });
                skin_pivot_z = 0.5 * (zmin + zmax);
                let mut joint_indices = Vec::with_capacity(capsule.positions.len());
                let mut joint_weights = Vec::with_capacity(capsule.positions.len());
                for p in &capsule.positions {
                    let t = ((p[2] - zmin) / (zmax - zmin)).clamp(0.0, 1.0); // 0 at base, 1 at top
                    joint_indices.push([0u8, 1, 0, 0]);
                    joint_weights.push([1.0 - t, t, 0.0, 0.0]);
                }
                let skinned = res.upload_mesh_data(device, &capsule).expect("capsule");
                let plugin = SkinningPlugin::install(res, device).expect("skinning install");
                plugin.attach_weights(
                    res,
                    device,
                    skinned,
                    &SkinWeights {
                        joint_indices,
                        joint_weights,
                    },
                );
                skinning = plugin;

                // Transparent scalar arrow: active_attribute forces per-object, and
                // opacity + Cull make it the per-object transparent one-sided path.
                let mut arrow_mesh = primitives::arrow(0.18, 0.4, 0.35, 24);
                let arrow_scalars: Vec<f32> = arrow_mesh.positions.iter().map(|p| p[2]).collect();
                arrow_mesh
                    .attributes
                    .insert("height".to_string(), AttributeData::Vertex(arrow_scalars));
                let scalar_t = res.upload_mesh_data(device, &arrow_mesh).expect("scalar_t");

                // Position-override sphere: bind a per-vertex position buffer (a radial
                // bump of the rest sphere). A bound override buffer forces the per-object
                // path via the `is_instanceable` override-buffer exclusion.
                let sphere_o = primitives::sphere(0.85, 40, 20);
                let mut flat: Vec<f32> = Vec::with_capacity(sphere_o.positions.len() * 3);
                for p in &sphere_o.positions {
                    let bump = 1.0 + 0.15 * (6.0 * p[0].atan2(p[1])).sin();
                    flat.extend_from_slice(&[p[0] * bump, p[1] * bump, p[2] * bump]);
                }
                let override_pos = res.upload_mesh_data(device, &sphere_o).expect("override");
                let bytes: &[u8] = bytemuck::cast_slice(&flat);
                let override_buf = device.create_buffer(&eframe::wgpu::BufferDescriptor {
                    label: Some("position_override"),
                    size: bytes.len() as u64,
                    usage: eframe::wgpu::BufferUsages::STORAGE
                        | eframe::wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&override_buf, 0, bytes);
                res.set_position_override_buffer(override_pos, override_buf)
                    .expect("override buffer");

                meshes = Meshes {
                    ground,
                    cube,
                    hemisphere,
                    sphere_t,
                    torus,
                    cone,
                    cylinder,
                    ring,
                    icosphere,
                    scalar,
                    ellipsoid,
                    filter,
                    skinned,
                    scalar_t,
                    override_pos,
                };
            }

            let matcap = renderer
                .resources()
                .builtin_matcap_id(BuiltinMatcap::Ceramic);
            let colourmap = renderer
                .resources()
                .builtin_colourmap_id(BuiltinColourmap::Viridis);

            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App::new(
                meshes,
                matcap,
                colourmap,
                skinning,
                skin_pivot_z,
            )))
        }),
    )
}

struct Meshes {
    ground: MeshId,
    cube: MeshId,
    hemisphere: MeshId,
    sphere_t: MeshId,
    torus: MeshId,
    cone: MeshId,
    cylinder: MeshId,
    ring: MeshId,
    icosphere: MeshId,
    scalar: MeshId,
    ellipsoid: MeshId,
    filter: MeshId,
    skinned: MeshId,
    scalar_t: MeshId,
    override_pos: MeshId,
}

struct App {
    camera: Camera,
    controller: OrbitCameraController,

    meshes: Meshes,
    matcap: MatcapId,
    colourmap: ColourmapId,
    skinning: SkinningPlugin,
    skin_pivot_z: f32,

    // Selected pick ids (click adds one, left-drag box adds many), animation time.
    selection: std::collections::HashSet<u64>,
    time: f32,
    // Left-drag rubber-band state, in viewport-local pixels.
    drag_start: Option<glam::Vec2>,
    last_cursor: glam::Vec2,

    // Render-path toggles.
    hdr: bool,
    // Flip every lit (non-matcap) material to PBR when on, Phong when off.
    pbr: bool,
    gpu_culling: bool,
    gpu_culling_supported: bool,
    wireframe: bool,

    // Lighting / shadow controls.
    light_kind: u8, // 0 = Directional, 1 = Point, 2 = Spot
    dir_direction: [f32; 3],
    point_position: [f32; 3],
    point_range: f32,
    spot_position: [f32; 3],
    spot_direction: [f32; 3],
    spot_range: f32,
    spot_inner_deg: f32,
    spot_outer_deg: f32,
    light_colour: [f32; 3],
    light_intensity: f32,
    shadows_enabled: bool,
    shadow_cascade_count: u32,
    shadow_filter: ShadowFilter,
    shadow_atlas_resolution: u32,
    hemisphere_intensity: f32,

    // Shared with the paint callback (one frame behind).
    instancing_status: std::sync::Arc<std::sync::Mutex<(bool, usize)>>,
}

impl App {
    fn new(
        meshes: Meshes,
        matcap: MatcapId,
        colourmap: ColourmapId,
        skinning: SkinningPlugin,
        skin_pivot_z: f32,
    ) -> Self {
        Self {
            camera: Camera {
                distance: 22.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            meshes,
            matcap,
            colourmap,
            skinning,
            skin_pivot_z,
            selection: std::collections::HashSet::new(),
            time: 0.0,
            drag_start: None,
            last_cursor: glam::Vec2::ZERO,
            hdr: true,
            pbr: false,
            gpu_culling: false,
            gpu_culling_supported: false,
            wireframe: false,
            light_kind: 0,
            dir_direction: [0.3, 0.5, 0.8],
            point_position: [0.0, 0.0, 6.0],
            point_range: 30.0,
            spot_position: [0.0, -6.0, 8.0],
            spot_direction: [0.0, 0.5, -0.8],
            spot_range: 40.0,
            spot_inner_deg: 20.0,
            spot_outer_deg: 32.0,
            light_colour: [1.0, 0.97, 0.90],
            light_intensity: 0.85,
            shadows_enabled: true,
            shadow_cascade_count: 4,
            shadow_filter: ShadowFilter::Pcf,
            shadow_atlas_resolution: 4096,
            hemisphere_intensity: 0.25,
            instancing_status: std::sync::Arc::new(std::sync::Mutex::new((false, 0))),
        }
    }

    /// Two-joint palette for the skinned capsule: root fixed, upper joint bends
    /// about the mid-height pivot. `palette[i] = pose_i * inverse_bind_i`.
    fn skin_palette(&self) -> [glam::Mat4; 2] {
        let angle = (self.time * 1.5).sin() * 0.6;
        let pz = self.skin_pivot_z;
        let bend = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, pz))
            * glam::Mat4::from_rotation_x(angle)
            * glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -pz));
        [glam::Mat4::IDENTITY, bend]
    }

    fn build_lighting(&self) -> LightingSettings {
        let mut s = LightingSettings::default();
        let mut light = LightSource::default();
        light.kind = match self.light_kind {
            1 => LightKind::Point {
                position: self.point_position,
                range: self.point_range,
                radius: 0.1,
            },
            2 => LightKind::Spot {
                position: self.spot_position,
                direction: self.spot_direction,
                range: self.spot_range,
                inner_angle: self.spot_inner_deg.to_radians(),
                outer_angle: self.spot_outer_deg.to_radians(),
                radius: 0.1,
            },
            _ => LightKind::Directional {
                direction: self.dir_direction,
            },
        };
        light.colour = self.light_colour;
        // Normalised 0..1 slider scaled into the selected kind's unit: directional
        // is illuminance (direct); point/spot are candela with 1/d^2 falloff, so
        // they need a much larger number to read at the light's throw distance.
        light.intensity = match self.light_kind {
            1 => self.light_intensity * 40.0,
            2 => self.light_intensity * 120.0,
            _ => self.light_intensity,
        };
        s.lights = vec![light];
        s.shadows.enabled = self.shadows_enabled;
        s.shadows.cascade_count = self.shadow_cascade_count;
        s.shadows.filter = self.shadow_filter;
        s.shadows.atlas_resolution = self.shadow_atlas_resolution;
        s.hemisphere_intensity = self.hemisphere_intensity;
        s
    }

    /// One render item per object, plus the ground. Returns the scene items and the
    /// compute-filter specs (the clipped sphere needs an entry on the effects frame).
    fn build_scene(&self) -> (Vec<SceneRenderItem>, Vec<ComputeFilterItem>) {
        let mut items = Vec::new();

        let mut place = |mesh: MeshId, pick_id: u64, configure: &dyn Fn(&mut SceneRenderItem)| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh;
            item.model = glam::Mat4::from_translation(grid_pos(pick_id)).to_cols_array_2d();
            item.material = Material::from_colour([0.80, 0.82, 0.88]);
            item.settings.pick_id = PickId(pick_id);
            item.settings.selected = self.selection.contains(&pick_id);
            configure(&mut item);
            items.push(item);
        };

        // Vivid base colours so every object stands out against the white platform.
        // Matcap (8) and scalar (9) ignore base_colour (their own shading wins).

        // 1: instanced opaque, one-sided.
        place(self.meshes.cube, 1, &|it| {
            it.material.base_colour = [0.85, 0.10, 0.10]; // red
            it.material.backface_policy = BackfacePolicy::Cull;
        });
        // 2: instanced opaque, two-sided (the path the two-sided shadow fix covers).
        place(self.meshes.hemisphere, 2, &|it| {
            it.material.base_colour = [0.95, 0.45, 0.05]; // orange
            it.material.backface_policy = BackfacePolicy::Identical;
        });
        // 3: instanced transparent, one-sided.
        place(self.meshes.sphere_t, 3, &|it| {
            it.material.base_colour = [0.93, 0.83, 0.10]; // yellow
            it.material.backface_policy = BackfacePolicy::Cull;
            it.material.alpha_mode = AlphaMode::Blend;
            it.settings.opacity = 0.5;
        });
        // 4: per-object transparent, two-sided (OIT is back-culled, so this stays per-object).
        place(self.meshes.torus, 4, &|it| {
            it.material.base_colour = [0.15, 0.80, 0.25]; // green
            it.material.backface_policy = BackfacePolicy::Identical;
            it.material.alpha_mode = AlphaMode::Blend;
            it.settings.opacity = 0.5;
        });
        // 5: per-object two-sided, DifferentColour backface.
        place(self.meshes.cone, 5, &|it| {
            it.material.base_colour = [0.05, 0.70, 0.65]; // teal (front)
            it.material.backface_policy = BackfacePolicy::DifferentColour([0.95, 0.35, 0.10]);
        });
        // 6: per-object two-sided, Tint backface.
        place(self.meshes.cylinder, 6, &|it| {
            it.material.base_colour = [0.15, 0.35, 0.95]; // blue
            it.material.backface_policy = BackfacePolicy::Tint(0.4);
        });
        // 7: per-object two-sided, Pattern backface.
        place(self.meshes.ring, 7, &|it| {
            it.material.base_colour = [0.60, 0.20, 0.90]; // violet (front)
            it.material.backface_policy = BackfacePolicy::Pattern(PatternConfig {
                pattern: BackfacePattern::Hatching,
                colour: [0.1, 0.4, 0.8],
                ..Default::default()
            });
        });
        // 8: per-object matcap (base_colour ignored).
        place(self.meshes.icosphere, 8, &|it| {
            it.material.shading_model = ShadingModel::Matcap(self.matcap);
        });
        // 9: per-object scalar attribute + colourmap (base_colour ignored).
        place(self.meshes.scalar, 9, &|it| {
            it.active_attribute = Some(AttributeRef {
                name: "height".to_string(),
                kind: AttributeKind::Vertex,
            });
            it.colourmap_id = Some(self.colourmap);
        });
        // 10: per-object UV param-vis.
        place(self.meshes.ellipsoid, 10, &|it| {
            it.material.base_colour = [0.95, 0.15, 0.65]; // magenta
            it.material.param_vis = Some(ParamVis {
                mode: ParamVisMode::Checker,
                scale: 8.0,
            });
        });
        // 11: per-object compute-filter (handled below; the pending filter result forces per-object).
        place(self.meshes.filter, 11, &|it| {
            it.material.base_colour = [0.10, 0.75, 0.90]; // cyan
            it.material.backface_policy = BackfacePolicy::Identical; // show the cut interior
        });
        // 12: per-object GPU skinning. deform_instance binds the per-instance palette.
        place(self.meshes.skinned, 12, &|it| {
            it.deform_instance = Some(0);
            it.material = Material::from_colour([0.90, 0.55, 0.05]); // amber
            it.settings.pick_id = PickId(12);
            it.settings.selected = self.selection.contains(&12);
        });
        // 13: per-object transparent one-sided (scalar attribute forces per-object).
        place(self.meshes.scalar_t, 13, &|it| {
            it.active_attribute = Some(AttributeRef {
                name: "height".to_string(),
                kind: AttributeKind::Vertex,
            });
            it.colourmap_id = Some(self.colourmap);
            it.material.backface_policy = BackfacePolicy::Cull;
            it.material.alpha_mode = AlphaMode::Blend;
            it.settings.opacity = 0.5;
        });
        // 14: per-object via a bound position-override buffer.
        place(self.meshes.override_pos, 14, &|it| {
            it.material.base_colour = [0.55, 0.85, 0.20]; // lime
            it.material.backface_policy = BackfacePolicy::Cull;
        });

        // Ground platform: instanced opaque, receives every object's shadow.
        let mut ground = SceneRenderItem::default();
        ground.mesh_id = self.meshes.ground;
        ground.model =
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
        ground.material = Material::from_colour([0.86, 0.85, 0.82]);
        ground.material.roughness = 0.9;
        ground.material.backface_policy = BackfacePolicy::Cull;
        items.push(ground);

        // Flip every lit (non-matcap) material between PBR and Phong. Matcap
        // keeps its own model; scalar/param-vis items still shade through the
        // selected lit model underneath. Best viewed with the HDR path on.
        for it in &mut items {
            if !matches!(it.material.shading_model, ShadingModel::Matcap(_)) {
                it.material.shading_model = if self.pbr {
                    ShadingModel::Pbr
                } else {
                    ShadingModel::Phong
                };
            }
        }

        // Clip the filtered sphere to its top half (local mesh space, before the model transform).
        let mut clip = ComputeFilterItem::default();
        clip.mesh_id = self.meshes.filter;
        clip.kind = ComputeFilterKind::Clip {
            plane_normal: [0.0, 0.0, 1.0],
            plane_dist: 0.0,
        };
        let filters = vec![clip];

        (items, filters)
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Animate the skinned object continuously.
        self.time = ctx.input(|i| i.time) as f32;
        ctx.request_repaint();

        egui::SidePanel::left("paths_panel")
            .min_width(280.0)
            .max_width(340.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.ui_panel(ui);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.controller.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            });

            ui.input(|i| {
                self.controller.push_event(ViewportEvent::ModifiersChanged(
                    viewport_lib::Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    },
                ));
                if let Some(p) = i.pointer.interact_pos() {
                    let local = glam::Vec2::new(p.x - rect.left(), p.y - rect.top());
                    self.last_cursor = local;
                    self.controller
                        .push_event(ViewportEvent::PointerMoved { position: local });
                }
                for event in &i.events {
                    match event {
                        egui::Event::PointerButton {
                            button, pressed, ..
                        } => {
                            // Left is reserved for selection (click = pick, drag = box).
                            // Orbit is on middle, pan on right, so left never reaches
                            // the camera controller.
                            if *button == egui::PointerButton::Primary {
                                self.drag_start = if *pressed {
                                    Some(self.last_cursor)
                                } else {
                                    self.drag_start
                                };
                                continue;
                            }
                            let vp_button = match button {
                                egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
                                egui::PointerButton::Middle => viewport_lib::MouseButton::Middle,
                                _ => continue,
                            };
                            self.controller.push_event(ViewportEvent::MouseButton {
                                button: vp_button,
                                state: if *pressed {
                                    ButtonState::Pressed
                                } else {
                                    ButtonState::Released
                                },
                            });
                        }
                        egui::Event::MouseWheel { unit, delta, .. } => {
                            let units = match unit {
                                egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                                egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                                egui::MouseWheelUnit::Page => ScrollUnits::Pages,
                            };
                            self.controller.push_event(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units,
                            });
                        }
                        _ => {}
                    }
                }
            });

            let w = rect.width();
            let h = rect.height();
            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(w, h);

            // Resolve selection against the renderer held in the egui render state
            // (the picker uses the previous frame's data). A left click is a single
            // `pick`; a finished left-drag is a rubber-band `pick_rect`. Shift adds
            // to the current selection instead of replacing it.
            let shift = ctx.input(|i| i.modifiers.shift);
            let vp_size = glam::Vec2::new(w, h);
            let view_proj = self.camera.view_proj_matrix();
            let click = response.clicked();
            let drag_done = response.drag_stopped();
            if click || drag_done {
                if let Some(rs) = frame.wgpu_render_state() {
                    let guard = rs.renderer.read();
                    if let Some(renderer) = guard.callback_resources.get::<ViewportRenderer>() {
                        if click {
                            let local = response
                                .interact_pointer_pos()
                                .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()))
                                .unwrap_or(self.last_cursor);
                            if !shift {
                                self.selection.clear();
                            }
                            if let Some(hit) =
                                renderer.pick(local, vp_size, view_proj, PickMask::OBJECT)
                            {
                                self.selection.insert(hit.id);
                            }
                        } else if let Some(start) = self.drag_start {
                            let end = self.last_cursor;
                            if (end - start).length() > 4.0 {
                                if !shift {
                                    self.selection.clear();
                                }
                                // pick_rect tests rect_min <= p <= rect_max as-is, so
                                // normalize the corners (a drag can go any direction).
                                let r_min = start.min(end);
                                let r_max = start.max(end);
                                let result = renderer.pick_rect(
                                    r_min,
                                    r_max,
                                    vp_size,
                                    view_proj,
                                    PickMask::OBJECT,
                                );
                                for id in result.objects {
                                    self.selection.insert(id);
                                }
                            }
                        }
                    }
                }
            }
            if !ctx.input(|i| i.pointer.primary_down()) {
                self.drag_start = None;
            }

            // Read GPU-culling support once (the renderer knows after first prepare).
            if let Some(rs) = frame.wgpu_render_state() {
                let guard = rs.renderer.read();
                if let Some(renderer) = guard.callback_resources.get::<ViewportRenderer>() {
                    self.gpu_culling_supported = renderer.is_gpu_culling_supported();
                }
            }

            let (items, filters) = self.build_scene();
            let ppp = ui.ctx().pixels_per_point();
            let mut fd = FrameData::new(
                CameraFrame::from_camera(&self.camera, [w, h]).with_pixels_per_point(ppp),
                SceneFrame::from_surface_items(items),
            );
            fd.effects.lighting = self.build_lighting();
            fd.effects.compute_filter_items = filters;
            // HDR path (post-process + OIT transparency) vs LDR inline path.
            fd.effects.display.mode = if self.hdr {
                viewport_lib::PipelineMode::Hdr
            } else {
                viewport_lib::PipelineMode::Direct
            };
            fd.viewport.wireframe_mode = self.wireframe;
            fd.viewport.show_axes_indicator = true;
            fd.interaction.outline_selected = !self.selection.is_empty();
            fd.interaction.outline_colour = [1.0, 0.85, 0.0, 1.0];
            fd.interaction.outline_width_px = 3.0;

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback {
                        frame: fd,
                        gpu_culling: self.gpu_culling,
                        skinning: Some(self.skinning.clone()),
                        skinned_mesh: self.meshes.skinned,
                        skin_palette: self.skin_palette(),
                        instancing_status: self.instancing_status.clone(),
                    },
                ));

            let (is_instanced, batch_count) = self
                .instancing_status
                .lock()
                .map(|g| *g)
                .unwrap_or((false, 0));
            let status = format!(
                "Instanced batches: {}   Path: {}   GPU culling: {}   Selected: {}",
                batch_count,
                if is_instanced {
                    "instanced+per-object"
                } else {
                    "per-object only"
                },
                if self.gpu_culling && self.gpu_culling_supported {
                    "on"
                } else {
                    "off"
                },
                self.selection.len(),
            );
            ui.painter().text(
                egui::pos2(rect.left() + 8.0, rect.bottom() - 20.0),
                egui::Align2::LEFT_BOTTOM,
                &status,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgba_premultiplied(20, 20, 20, 220),
            );

            // Rubber-band overlay while left-dragging a selection box.
            if let Some(start) = self.drag_start {
                if response.dragged() && (self.last_cursor - start).length() > 4.0 {
                    let a = egui::pos2(rect.left() + start.x, rect.top() + start.y);
                    let b = egui::pos2(
                        rect.left() + self.last_cursor.x,
                        rect.top() + self.last_cursor.y,
                    );
                    ui.painter().rect(
                        egui::Rect::from_two_pos(a, b),
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(255, 215, 0, 20),
                        egui::Stroke::new(
                            1.5,
                            egui::Color32::from_rgba_unmultiplied(255, 215, 0, 200),
                        ),
                        egui::StrokeKind::Outside,
                    );
                }
            }

            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}

impl App {
    fn ui_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Render Paths");
        ui.label("One object per reachable path. Click an object to pick + outline it.");
        ui.separator();

        egui::CollapsingHeader::new("Path toggles")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut self.hdr, "HDR path (post-process + OIT)");
                if !self.hdr {
                    ui.label("  LDR inline path (straight-alpha transparency)");
                }
                ui.checkbox(&mut self.pbr, "PBR shading (off = Phong)");
                ui.add_enabled(
                    self.gpu_culling_supported,
                    egui::Checkbox::new(&mut self.gpu_culling, "GPU-driven culling (indirect)"),
                );
                if !self.gpu_culling_supported {
                    ui.label("  (not supported on this device)");
                }
                ui.checkbox(&mut self.wireframe, "Wireframe mode");
            });

        egui::CollapsingHeader::new("Light")
            .default_open(true)
            .show(ui, |ui| {
                // Light kind drives which shadow path runs: directional cascades,
                // or point/spot cube-face shadows.
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.light_kind, 0, "Directional");
                    ui.radio_value(&mut self.light_kind, 1, "Point");
                    ui.radio_value(&mut self.light_kind, 2, "Spot");
                });
                match self.light_kind {
                    1 => {
                        ui.label("Position:");
                        ui_vec3(ui, &mut self.point_position, 0.1);
                        ui.add(egui::Slider::new(&mut self.point_range, 2.0..=80.0).text("Range"));
                    }
                    2 => {
                        ui.label("Position:");
                        ui_vec3(ui, &mut self.spot_position, 0.1);
                        ui.label("Direction:");
                        ui_vec3(ui, &mut self.spot_direction, 0.01);
                        ui.add(egui::Slider::new(&mut self.spot_range, 2.0..=80.0).text("Range"));
                        ui.add(
                            egui::Slider::new(&mut self.spot_inner_deg, 1.0..=88.0)
                                .text("Inner deg"),
                        );
                        self.spot_outer_deg = self.spot_outer_deg.max(self.spot_inner_deg);
                        ui.add(
                            egui::Slider::new(&mut self.spot_outer_deg, self.spot_inner_deg..=89.0)
                                .text("Outer deg"),
                        );
                    }
                    _ => {
                        ui.label("Direction (toward light):");
                        ui_vec3(ui, &mut self.dir_direction, 0.01);
                    }
                }
                ui.horizontal(|ui| {
                    ui.label("Colour:");
                    ui.color_edit_button_rgb(&mut self.light_colour);
                });
                ui.add(egui::Slider::new(&mut self.light_intensity, 0.0..=1.0).text("Intensity"));
                ui.add(
                    egui::Slider::new(&mut self.hemisphere_intensity, 0.0..=1.0).text("Ambient"),
                );
            });

        egui::CollapsingHeader::new("Shadows")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut self.shadows_enabled, "Enabled");
                ui.label("Cascades:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.shadow_cascade_count, 1, "1");
                    ui.radio_value(&mut self.shadow_cascade_count, 2, "2");
                    ui.radio_value(&mut self.shadow_cascade_count, 4, "4");
                });
                ui.label("Filter:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::Pcf, "PCF");
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::Pcss, "PCSS");
                });
                ui.label("Atlas resolution:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.shadow_atlas_resolution, 1024, "1K");
                    ui.radio_value(&mut self.shadow_atlas_resolution, 2048, "2K");
                    ui.radio_value(&mut self.shadow_atlas_resolution, 4096, "4K");
                });
            });

        egui::CollapsingHeader::new("Legend")
            .default_open(true)
            .show(ui, |ui| {
                ui.label("Left-click: pick. Left-drag: box select (Shift adds).");
                ui.label("Middle-drag: orbit. Right-drag: pan. Scroll: zoom.");
                ui.separator();
                for (id, name, path) in PATHS {
                    let picked = self.selection.contains(id);
                    let text = format!("{}. {} : {}", id, name, path);
                    if ui.selectable_label(picked, text).clicked() {
                        if picked {
                            self.selection.remove(id);
                        } else {
                            self.selection.insert(*id);
                        }
                    }
                }
                if !self.selection.is_empty() {
                    ui.separator();
                    if ui.button("Clear selection").clicked() {
                        self.selection.clear();
                    }
                }
            });
    }
}
