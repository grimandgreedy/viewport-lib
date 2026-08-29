//! Lighting and shadows test example using eframe / egui.
//!
//! A bench for checking the lighting subsystem before a release. The left panel
//! drives every lighting knob: the light source (authored in nominal units or
//! real photometric lux/candela/lumens), hemisphere ambient, shadows, the HDR
//! display transform (pipeline mode, exposure model, tone-map operator), and an
//! image-based environment. Switch between scene tabs at the top to test
//! different material configurations.
//!
//! The viewport renders through the HDR pipeline, so exposure and tone mapping
//! apply here; the "Display & Exposure" section drives them and reads back the
//! live EV (metered on the GPU under auto-exposure).
//!
//! Structure mirrors `eframe_exposure`: a `ViewportInstance` renders into an
//! app-owned offscreen texture that egui displays as an image, and events come
//! through the `from_egui` adapter driving an `OrbitCameraController`. The
//! offscreen texture is the sRGB variant of egui's format so the tone map's
//! linear output gets the hardware linear->sRGB encode on write. The scene items
//! are rebuilt per frame (per-tab geometry, live material toggles) and injected
//! into the instance's assembled frame rather than held in its scene graph.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom

use eframe::{egui, wgpu};
use viewport_lib::input::adapters::from_egui;
use viewport_lib::{
    AtlasViewerCorner, AutoExposure, BackfacePolicy, BuiltinMatcap, Candela, DebugOutputMode,
    DebugQuantity, DebugVis, DisplaySettings, EnvironmentSettings, ExposureMode, ExposureReadback,
    ExposureSettings, LightKind, LightSource, LightingSettings, Lumen, Lux, MatcapId, Material,
    MeshId, Modifiers, OrbitCameraController, PipelineMode, SceneFrame, SceneRenderItem,
    ShadowDebugStats, ShadowFilter, ToneMapping, ViewportContext, ViewportEvent, ViewportId,
    ViewportInstance, primitives,
};

// Percy photo: pre-converted raw RGBA (2203 x 2009).
const PERCY_WIDTH: u32 = 2203;
const PERCY_HEIGHT: u32 = 2009;
const PERCY_RGBA: &[u8] = include_bytes!("../eframe_showcase/percy.rgba");

// Equirectangular environment resolution. Small is fine: it feeds the IBL
// prefilter (diffuse irradiance + a few specular mips), not a sharp backdrop.
const ENV_W: u32 = 64;
const ENV_H: u32 = 32;

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Lighting & Shadows",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 900.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            // eframe creates its device with default limits (8 storage buffers
            // per stage), but the auto-exposure compute path needs the higher
            // limits the renderer recommends. Request them, or bring-up panics on
            // backends that enforce the limit (Vulkan, DX12).
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                    eframe::egui_wgpu::WgpuSetupCreateNew {
                        device_descriptor: std::sync::Arc::new(|adapter| {
                            let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                                wgpu::Limits::downlevel_webgl2_defaults()
                            } else {
                                viewport_lib::ViewportRenderer::recommended_device_limits(adapter)
                            };
                            wgpu::DeviceDescriptor {
                                label: Some("viewport-lib lighting device"),
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
            // The session draws into an app-owned offscreen texture built as the
            // sRGB variant of egui's format (so the tone map's linear output is
            // sRGB-encoded on write); build the session for that same format.
            let format = rs.target_format.add_srgb_suffix();

            let mut session = ViewportInstance::new(device, format);

            // Mint an explicit viewport id and tag every frame with it (see
            // App::vp_id). Naming the slot lets `exposure_state` read back the EV
            // for the same slot the frame renders into.
            let vp_id = session.renderer_mut().create_viewport(device);

            let (
                m_ground,
                m_sphere,
                m_cube,
                m_torus,
                m_ground2,
                m_clay,
                m_ceramic,
                m_metal,
                m_rough,
                m_cube2,
                m_percy,
                tex_percy,
            );
            {
                let res = session.resources_mut();

                res.ensure_matcaps_initialized(device, queue);

                // Tab 1: Basic geometry
                m_ground = res
                    .upload_mesh_data(device, &primitives::cuboid(24.0, 24.0, 0.5))
                    .expect("ground");
                m_sphere = res
                    .upload_mesh_data(device, &primitives::sphere(0.6, 32, 16))
                    .expect("sphere");
                m_cube = res
                    .upload_mesh_data(device, &primitives::cube(1.0))
                    .expect("cube");
                m_torus = res
                    .upload_mesh_data(device, &primitives::torus(0.5, 0.18, 40, 20))
                    .expect("torus");

                // Tab 2: Material variety.
                // Each sphere needs its own mesh upload so the non-instanced path gives each
                // object independent GPU state (object_uniform_buf / bind_group).
                m_ground2 = res
                    .upload_mesh_data(device, &primitives::cuboid(24.0, 24.0, 0.5))
                    .expect("ground2");
                m_clay = res
                    .upload_mesh_data(device, &primitives::sphere(0.7, 40, 20))
                    .expect("clay sphere");
                m_ceramic = res
                    .upload_mesh_data(device, &primitives::sphere(0.7, 40, 20))
                    .expect("ceramic sphere");
                m_metal = res
                    .upload_mesh_data(device, &primitives::sphere(0.7, 40, 20))
                    .expect("metal sphere");
                m_rough = res
                    .upload_mesh_data(device, &primitives::sphere(0.7, 40, 20))
                    .expect("rough sphere");
                m_cube2 = res
                    .upload_mesh_data(device, &primitives::cube(1.2))
                    .expect("cube2");

                let percy_aspect = PERCY_HEIGHT as f32 / PERCY_WIDTH as f32;
                m_percy = res
                    .upload_mesh_data(device, &primitives::plane(4.5, 4.5 * percy_aspect))
                    .expect("percy plane");
                tex_percy = res
                    .upload_texture(device, queue, PERCY_WIDTH, PERCY_HEIGHT, PERCY_RGBA)
                    .expect("percy texture");
            }

            // Procedural sky/ground gradient for the IBL environment. Stored as
            // relative radiance; EnvironmentSettings::intensity scales it to nits
            // at render time, so the same map reads from near-black up to a bright
            // daytime sky depending on the exposure regime under test.
            let env = equirect_gradient([0.55, 0.70, 1.0], [0.30, 0.32, 0.34], ENV_W, ENV_H);
            session
                .renderer_mut()
                .upload_environment_map(device, queue, &env, ENV_W, ENV_H)
                .expect("environment map");

            let matcap_clay = session
                .renderer_mut()
                .resources()
                .builtin_matcap_id(BuiltinMatcap::Clay);
            let matcap_ceramic = session
                .renderer_mut()
                .resources()
                .builtin_matcap_id(BuiltinMatcap::Ceramic);

            // Dark neutral background so the scene sits on the same plate as before.
            session.viewport_frame_mut().background_colour =
                Some([65.0 / 255.0, 65.0 / 255.0, 65.0 / 255.0, 1.0]);
            session.camera_mut().distance = 18.0;

            Ok(Box::new(App::new(
                session,
                vp_id,
                m_ground,
                m_sphere,
                m_cube,
                m_torus,
                m_ground2,
                m_clay,
                m_ceramic,
                m_metal,
                m_rough,
                m_cube2,
                m_percy,
                tex_percy,
                matcap_clay,
                matcap_ceramic,
            )))
        }),
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Basic,
    Materials,
}

struct App {
    // Renderer + scene + camera + input, driven once per frame.
    session: ViewportInstance,
    orbit: OrbitCameraController,
    // App-owned offscreen render target that egui displays as an image.
    target: Option<Target>,
    // Viewport slot this frame renders into (tagged on the assembled frame). Named
    // so exposure_state can read back the EV for the same slot.
    vp_id: ViewportId,

    // Scene selection
    tab: Tab,

    // Mesh / texture / matcap IDs
    m_ground: MeshId,
    m_sphere: MeshId,
    m_cube: MeshId,
    m_torus: MeshId,
    m_ground2: MeshId,
    m_clay: MeshId,
    m_ceramic: MeshId,
    m_metal: MeshId,
    m_rough: MeshId,
    m_cube2: MeshId,
    m_percy: MeshId,
    tex_percy: viewport_lib::TextureId,
    matcap_clay: MatcapId,
    matcap_ceramic: MatcapId,

    // Light source parameters.
    // Stored per-kind so switching kinds does not lose previously entered values.
    light_kind: u8, // 0 = Directional, 1 = Point, 2 = Spot
    light_colour: [f32; 3],
    // Author intensity either in nominal units (the faithful "colour is data"
    // scale, read directly at EV 0) or in real photometric units. The unit and
    // slider range change with this toggle; the two regimes are what the exposure
    // controls exist to reconcile.
    photometric: bool,
    // Nominal-mode magnitudes: directional is a plain multiplier (~pi reads as a
    // lit surface at EV 0); point/spot are candela.
    dir_nominal: f32,
    point_candela: f32,
    spot_candela: f32,
    // Photometric-mode magnitudes: directional in lux, point/spot bulbs in lumens.
    dir_lux: f32,
    point_lumens: f32,
    spot_lumens: f32,
    // Per-light geometry / behaviour shared across kinds.
    light_radius: f32, // point/spot source radius: near-clamp + penumbra size
    light_importance: f32,
    light_cast_shadows: bool,
    dir_direction: [f32; 3],
    point_position: [f32; 3],
    point_range: f32,
    spot_position: [f32; 3],
    spot_direction: [f32; 3],
    spot_range: f32,
    spot_inner_deg: f32,
    spot_outer_deg: f32,

    // Display transform (HDR -> pixels): pipeline mode, exposure, tone map.
    hdr_pipeline: bool, // true = Hdr (exposure + tonemap), false = Direct passthrough
    exposure_mode: u8,  // 0 = Manual, 1 = Physical camera, 2 = Automatic
    manual_ev: f32,
    aperture: f32,
    shutter_denom: f32,
    iso: f32,
    auto: AutoExposure,
    exposure_compensation: f32,
    tone_mapping: ToneMapping,
    // Live EV read back from the GPU (Some once the first HDR frame has run).
    last_exposure: Option<ExposureReadback>,

    // Image-based environment (IBL + skybox).
    env_enabled: bool,
    env_intensity: f32, // absolute nits scale applied to the stored gradient
    env_rotation: f32,
    env_show_skybox: bool,

    // Shadow settings
    shadows_enabled: bool,
    shadow_bias: f32,
    shadow_cascade_count: u32,
    shadow_filter: ShadowFilter,
    pcss_light_radius: f32,
    shadow_atlas_resolution: u32,
    shadow_extent_enabled: bool,
    shadow_extent_value: f32,

    // Hemisphere ambient
    hemisphere_intensity: f32,
    sky_colour: [f32; 3],
    ground_colour: [f32; 3],

    // Debug visualization
    debug_vis_active: bool,
    debug_vis_mode_replace: bool,
    debug_vis_splitscreen: bool,
    debug_vis_split_x: f32,
    debug_vis_r: DebugQuantity,
    debug_vis_g: DebugQuantity,
    debug_vis_b: DebugQuantity,
    debug_vis_scale: f32,

    // Shadow atlas viewer
    show_shadow_atlas: bool,
    atlas_viewer_corner: AtlasViewerCorner,
    atlas_viewer_scale: f32,

    // Pixel inspector (D7)
    pixel_inspector_active: bool,
    // Pixel to read back after the next render (set on click).
    pixel_read_req: Option<(u32, u32)>,
    last_picked_pos: Option<(u32, u32)>,
    last_picked_values: Option<[f32; 4]>,

    // Scene toggles
    show_platform: bool,
    // When true every lit (non-matcap) material renders as PBR; when false as
    // Phong. Flips the whole scene between the two shading models for A/B
    // comparison, the same flip the headless PBR/Phong repros use.
    pbr_enabled: bool,

    // Diagnostics latched after each render.
    instancing_status: (bool, usize),
    shadow_stats: Option<ShadowDebugStats>,
}

/// App-owned offscreen render target displayed by egui as an image.
struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    id: egui::TextureId,
    size: [u32; 2],
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        session: ViewportInstance,
        vp_id: ViewportId,
        m_ground: MeshId,
        m_sphere: MeshId,
        m_cube: MeshId,
        m_torus: MeshId,
        m_ground2: MeshId,
        m_clay: MeshId,
        m_ceramic: MeshId,
        m_metal: MeshId,
        m_rough: MeshId,
        m_cube2: MeshId,
        m_percy: MeshId,
        tex_percy: viewport_lib::TextureId,
        matcap_clay: MatcapId,
        matcap_ceramic: MatcapId,
    ) -> Self {
        Self {
            session,
            orbit: OrbitCameraController::viewport_primitives(),
            target: None,
            vp_id,
            tab: Tab::Basic,
            m_ground,
            m_sphere,
            m_cube,
            m_torus,
            m_ground2,
            m_clay,
            m_ceramic,
            m_metal,
            m_rough,
            m_cube2,
            m_percy,
            tex_percy,
            matcap_clay,
            matcap_ceramic,
            // Match showcase_27 defaults: a known-problematic setup useful for reproducing bugs.
            light_kind: 0,
            light_colour: [1.0, 0.97, 0.90],
            // Nominal by default: the faithful "colour is data" regime at EV 0.
            photometric: false,
            dir_nominal: std::f32::consts::PI,
            point_candela: 40.0,
            spot_candela: 120.0,
            dir_lux: Lux::OVERCAST.0,
            point_lumens: Lumen::INCANDESCENT_100W.0,
            spot_lumens: Lumen::INCANDESCENT_100W.0,
            light_radius: 0.1,
            light_importance: 1.0,
            light_cast_shadows: true,
            dir_direction: [0.3, 0.5, 0.8],
            point_position: [0.0, 0.0, 5.0],
            point_range: 20.0,
            spot_position: [0.0, -8.0, 6.0],
            spot_direction: [0.0, 0.5, -0.6],
            spot_range: 20.0,
            spot_inner_deg: 15.0,
            spot_outer_deg: 25.0,
            // Display / exposure: neutral by default so the faithful lighting
            // reads as authored (HDR pipeline, manual EV 0, KhronosNeutral).
            hdr_pipeline: true,
            exposure_mode: 0,
            manual_ev: 0.0,
            aperture: 16.0,
            shutter_denom: 125.0,
            iso: 100.0,
            auto: AutoExposure::default(),
            exposure_compensation: 0.0,
            tone_mapping: ToneMapping::KhronosNeutral,
            last_exposure: None,
            env_enabled: false,
            env_intensity: 4_000.0,
            env_rotation: 0.0,
            env_show_skybox: true,
            shadows_enabled: true,
            shadow_bias: 0.0,
            shadow_cascade_count: 4,
            shadow_filter: ShadowFilter::Pcf,
            pcss_light_radius: 0.02,
            shadow_atlas_resolution: 4096,
            shadow_extent_enabled: false,
            shadow_extent_value: 20.0,
            hemisphere_intensity: 0.2,
            sky_colour: [0.8, 0.9, 1.0],
            ground_colour: [0.5, 0.55, 0.6],
            debug_vis_active: false,
            debug_vis_mode_replace: true,
            debug_vis_splitscreen: false,
            debug_vis_split_x: 0.5,
            debug_vis_r: DebugQuantity::ShadowFactor,
            debug_vis_g: DebugQuantity::Zero,
            debug_vis_b: DebugQuantity::Zero,
            debug_vis_scale: 1.0,
            show_shadow_atlas: false,
            atlas_viewer_corner: AtlasViewerCorner::BottomRight,
            atlas_viewer_scale: 0.3,
            pixel_inspector_active: false,
            pixel_read_req: None,
            last_picked_pos: None,
            last_picked_values: None,
            show_platform: true,
            pbr_enabled: true,
            instancing_status: (false, 0),
            shadow_stats: None,
        }
    }

    /// Build the single light from the panel state, using the typed photometric
    /// constructors so the intensity carries the right unit (lux for directional,
    /// candela for point/spot). Nominal mode sets a plain magnitude instead.
    fn build_light_source(&self) -> LightSource {
        let inner = self.spot_inner_deg.to_radians();
        let outer = self.spot_outer_deg.to_radians();
        let mut light = match (self.light_kind, self.photometric) {
            (0, false) => {
                let mut l = LightSource::default();
                l.kind = LightKind::Directional {
                    direction: self.dir_direction,
                };
                l.intensity = self.dir_nominal;
                l
            }
            (0, true) => LightSource::directional_lux(self.dir_direction, Lux(self.dir_lux)),
            (1, false) => LightSource::point_candela(
                self.point_position,
                Candela(self.point_candela),
                self.point_range,
                self.light_radius,
            ),
            (1, true) => LightSource::point_lumens(
                self.point_position,
                Lumen(self.point_lumens),
                self.point_range,
                self.light_radius,
            ),
            (_, false) => LightSource::spot_candela(
                self.spot_position,
                self.spot_direction,
                Candela(self.spot_candela),
                self.spot_range,
                inner,
                outer,
                self.light_radius,
            ),
            (_, true) => LightSource::spot_lumens(
                self.spot_position,
                self.spot_direction,
                Lumen(self.spot_lumens),
                self.spot_range,
                inner,
                outer,
                self.light_radius,
            ),
        };
        light.colour = self.light_colour;
        light.importance = self.light_importance;
        light.cast_shadows = self.light_cast_shadows;
        light
    }

    fn build_lighting(&self) -> LightingSettings {
        {
            let mut _t = LightingSettings::default();
            _t.lights = vec![self.build_light_source()];
            _t.shadows.enabled = self.shadows_enabled;
            _t.shadows.bias = self.shadow_bias;
            _t.shadows.cascade_count = self.shadow_cascade_count;
            _t.shadows.filter = self.shadow_filter;
            _t.shadows.pcss_light_radius = self.pcss_light_radius;
            _t.shadows.atlas_resolution = self.shadow_atlas_resolution;
            _t.shadows.extent_override = if self.shadow_extent_enabled {
                Some(self.shadow_extent_value)
            } else {
                None
            };
            _t.hemisphere_intensity = self.hemisphere_intensity;
            _t.sky_colour = self.sky_colour;
            _t.ground_colour = self.ground_colour;
            _t
        }
    }

    /// Channel-visualization config from the panel. Attached to
    /// `EffectsFrame::debug.debug_vis`.
    fn build_debug_vis(&self) -> DebugVis {
        let mut dv = DebugVis::default();
        dv.active = self.debug_vis_active;
        dv.mode = if self.debug_vis_splitscreen {
            DebugOutputMode::SplitScreen
        } else if self.debug_vis_mode_replace {
            DebugOutputMode::Replace
        } else {
            DebugOutputMode::TintOverlay
        };
        dv.split_x = self.debug_vis_split_x;
        dv.channel_r = self.debug_vis_r;
        dv.channel_g = self.debug_vis_g;
        dv.channel_b = self.debug_vis_b;
        dv.scale = self.debug_vis_scale;
        dv
    }

    /// Exposure settings from the panel. `dt` feeds auto-exposure smoothing.
    fn build_exposure(&self, dt: f32) -> ExposureSettings {
        let mode = match self.exposure_mode {
            0 => ExposureMode::Manual { ev: self.manual_ev },
            1 => ExposureMode::PhysicalCamera {
                aperture: self.aperture,
                shutter: 1.0 / self.shutter_denom.max(1.0),
                iso: self.iso.max(1.0),
            },
            _ => {
                let mut auto = self.auto;
                auto.dt = dt;
                ExposureMode::Automatic(auto)
            }
        };
        ExposureSettings::from_mode(mode).with_compensation(self.exposure_compensation)
    }

    /// Display transform from the panel: pipeline mode, exposure, tone map.
    fn build_display(&self, dt: f32) -> DisplaySettings {
        let mut d = DisplaySettings::default();
        d.mode = if self.hdr_pipeline {
            PipelineMode::Hdr
        } else {
            PipelineMode::Direct
        };
        d.exposure = self.build_exposure(dt);
        d.operator = self.tone_mapping;
        d
    }

    /// Environment settings, or `None` when the IBL environment is disabled.
    fn build_environment(&self) -> Option<EnvironmentSettings> {
        self.env_enabled.then(|| EnvironmentSettings {
            intensity: self.env_intensity,
            rotation: self.env_rotation,
            show_skybox: self.env_show_skybox,
        })
    }

    /// The EV the panel controls imply, for the readout. `Manual`/`Physical` are
    /// known here; `Automatic` is metered on the GPU, so its value comes from the
    /// exposure readback (`last_exposure`) instead.
    fn panel_ev(&self) -> Option<f32> {
        match self.exposure_mode {
            0 => Some(self.manual_ev),
            1 => {
                let n = self.aperture;
                let t = 1.0 / self.shutter_denom.max(1.0);
                let iso = self.iso.max(1.0);
                Some((n * n / t).log2() + (100.0 / iso).log2())
            }
            _ => None,
        }
    }

    /// True while the exposure model needs a live GPU EV readback (Automatic
    /// only). Gates the blocking `exposure_state` poll to when it is useful.
    fn wants_exposure_readback(&self) -> bool {
        self.hdr_pipeline && self.exposure_mode == 2
    }

    /// Force a lit material to the shading model selected by the PBR toggle.
    /// Matcap materials keep their matcap; PBR and Phong swap.
    fn apply_pbr_toggle(&self, m: &mut Material) {
        match m.shading_model {
            viewport_lib::ShadingModel::Matcap(_) => {}
            _ => {
                m.shading_model = if self.pbr_enabled {
                    viewport_lib::ShadingModel::Pbr
                } else {
                    viewport_lib::ShadingModel::Phong
                };
            }
        }
    }

    fn build_basic_items(&self) -> Vec<SceneRenderItem> {
        let mut items = Vec::new();

        // Ground platform: light warm sand. Cuboid with top surface at z=0.
        if self.show_platform {
            let mut ground = SceneRenderItem::default();
            ground.mesh_id = self.m_ground;
            ground.model =
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
            ground.material = Material::from_colour([0.88, 0.84, 0.76]);
            ground.material.roughness = 0.85;
            ground.material.backface_policy = BackfacePolicy::Cull;
            items.push(ground);
        }

        // Sphere: light sage green. Acne is easy to spot on light curved surfaces.
        let mut sphere = SceneRenderItem::default();
        sphere.mesh_id = self.m_sphere;
        sphere.model =
            glam::Mat4::from_translation(glam::Vec3::new(-4.0, 0.0, 0.6)).to_cols_array_2d();
        sphere.material = Material::from_colour([0.78, 0.90, 0.80]);
        items.push(sphere);

        // Cube: light periwinkle. Flat faces isolate bias and gap artifacts.
        let mut cube = SceneRenderItem::default();
        cube.mesh_id = self.m_cube;
        cube.model =
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.5)).to_cols_array_2d();
        cube.material = Material::from_colour([0.78, 0.83, 0.95]);
        items.push(cube);

        // Torus: light peach. Mixed geometry useful for bright-spot test.
        // Z offset 0.18 = torus minor radius, places the bottom of the tube on the ground.
        let mut torus = SceneRenderItem::default();
        torus.mesh_id = self.m_torus;
        torus.model =
            glam::Mat4::from_translation(glam::Vec3::new(4.0, 0.0, 0.18)).to_cols_array_2d();
        torus.material = Material::from_colour([0.95, 0.82, 0.74]);
        items.push(torus);

        for it in &mut items {
            self.apply_pbr_toggle(&mut it.material);
        }
        items
    }

    fn build_materials_items(&self) -> Vec<SceneRenderItem> {
        let mut items = Vec::new();

        // Ground platform: neutral light grey, top surface at z=0.
        if self.show_platform {
            let mut ground = SceneRenderItem::default();
            ground.mesh_id = self.m_ground2;
            ground.model =
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
            ground.material = Material::from_colour([0.85, 0.85, 0.85]);
            ground.material.roughness = 0.85;
            items.push(ground);
        }

        // Clay matcap sphere (blendable): off-white base exposes matcap tint while keeping
        // surfaces light enough to show shadow detail.
        let mut clay = SceneRenderItem::default();
        clay.mesh_id = self.m_clay;
        clay.model =
            glam::Mat4::from_translation(glam::Vec3::new(-4.5, 2.5, 0.7)).to_cols_array_2d();
        clay.material = Material::from_colour([0.92, 0.90, 0.88]);
        clay.material.shading_model = viewport_lib::ShadingModel::Matcap(self.matcap_clay);
        items.push(clay);

        // Ceramic matcap sphere (static): white base, high-contrast sheen.
        // Good for checking shadow edge quality.
        let mut ceramic = SceneRenderItem::default();
        ceramic.mesh_id = self.m_ceramic;
        ceramic.model =
            glam::Mat4::from_translation(glam::Vec3::new(-1.5, 2.5, 0.7)).to_cols_array_2d();
        ceramic.material = Material::from_colour([1.0, 1.0, 1.0]);
        ceramic.material.shading_model = viewport_lib::ShadingModel::Matcap(self.matcap_ceramic);
        items.push(ceramic);

        // PBR metallic sphere: near-white, very smooth. Specular response reveals shadow
        // terminator and banding at glancing angles.
        let mut metal = SceneRenderItem::default();
        metal.mesh_id = self.m_metal;
        metal.model =
            glam::Mat4::from_translation(glam::Vec3::new(1.5, 2.5, 0.7)).to_cols_array_2d();
        metal.material = Material::pbr([0.95, 0.95, 0.93], 1.0, 0.08);
        items.push(metal);

        // PBR rough sphere: near-white diffuse. High roughness maximises shadow acne visibility.
        let mut rough = SceneRenderItem::default();
        rough.mesh_id = self.m_rough;
        rough.model =
            glam::Mat4::from_translation(glam::Vec3::new(4.5, 2.5, 0.7)).to_cols_array_2d();
        rough.material = Material::pbr([0.95, 0.95, 0.95], 0.0, 0.95);
        items.push(rough);

        // Plain diffuse cube: light cream. Flat surfaces at cardinal angles show cascade bands
        // as clean rectangular stripes rather than complex shadow shapes.
        let mut cube = SceneRenderItem::default();
        cube.mesh_id = self.m_cube2;
        cube.model =
            glam::Mat4::from_translation(glam::Vec3::new(-3.5, -2.5, 0.6)).to_cols_array_2d();
        cube.material = Material::from_colour([0.96, 0.94, 0.90]);
        items.push(cube);

        // Percy photo plane: lying flat on the ground, slightly raised to avoid Z-fighting.
        // Real-world image content makes acne and bias gaps easy to judge perceptually.
        let mut percy = SceneRenderItem::default();
        percy.mesh_id = self.m_percy;
        percy.model =
            glam::Mat4::from_translation(glam::Vec3::new(2.5, -2.5, 0.005)).to_cols_array_2d();
        percy.material = Material::default();
        percy.material.texture_id = Some(self.tex_percy);
        percy.material.backface_policy = BackfacePolicy::Cull;
        items.push(percy);

        for it in &mut items {
            self.apply_pbr_toggle(&mut it.material);
        }
        items
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

        egui::SidePanel::left("lighting_panel")
            .min_width(250.0)
            .max_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.ui_lighting_panel(ui);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Tab bar
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Basic, "Basic");
                ui.selectable_value(&mut self.tab, Tab::Materials, "Materials");
            });
            ui.separator();

            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            // Offscreen target in physical pixels; (re)created when the viewport
            // resizes. sRGB variant so the tone map's linear output is encoded on
            // write and egui displays it correctly.
            let ppp = ui.ctx().pixels_per_point();
            let size = [
                (rect.width() * ppp).round().max(1.0) as u32,
                (rect.height() * ppp).round().max(1.0) as u32,
            ];
            if self.target.as_ref().map_or(true, |t| t.size != size) {
                let texture = rs.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("lighting_offscreen"),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: rs.target_format.add_srgb_suffix(),
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let id = rs.renderer.write().register_native_texture(
                    &rs.device,
                    &view,
                    wgpu::FilterMode::Linear,
                );
                self.target = Some(Target {
                    _texture: texture,
                    view,
                    id,
                    size,
                });
            }
            let target = self.target.as_ref().unwrap();

            // Feed input to the instance's own input pipeline via the egui adapter.
            self.session.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            });
            self.session.set_pixels_per_point(ppp);
            let origin = glam::Vec2::new(rect.left(), rect.top());
            ui.input(|i| {
                self.session
                    .handle_event(ViewportEvent::ModifiersChanged(Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    }));
                for event in &i.events {
                    if let Some(ev) = from_egui(event, origin) {
                        self.session.handle_event(ev);
                    }
                }
            });

            // Frame time for auto-exposure smoothing. Exposure lives outside the
            // scene, so a control change would not dirty anything: request a
            // repaint every frame while auto-exposure is active so it keeps
            // metering and adapting.
            let dt = ctx.input(|i| i.stable_dt).min(0.1);
            if self.wants_exposure_readback() {
                ctx.request_repaint();
            }

            // Persistent effect state on the instance (built from immutable self
            // borrows first, then stamped in).
            let lighting = self.build_lighting();
            let display = self.build_display(dt);
            let environment = self.build_environment();
            let debug_vis = self.build_debug_vis();
            {
                let eff = self.session.effects_mut();
                eff.lighting = lighting;
                eff.display = display;
                eff.environment = environment;
                eff.debug.show_shadow_atlas = self.show_shadow_atlas;
                eff.debug.atlas_viewer_corner = self.atlas_viewer_corner;
                eff.debug.atlas_viewer_scale = self.atlas_viewer_scale;
                eff.debug.debug_vis = debug_vis;
            }

            // Build this tab's items, drive the camera, and inject the items into
            // the assembled frame (the instance's own scene graph stays empty), and
            // tag the viewport slot so exposure_state reads the right one.
            let items = match self.tab {
                Tab::Basic => self.build_basic_items(),
                Tab::Materials => self.build_materials_items(),
            };
            let vp_index = self.vp_id.index();
            self.session.update_orbit_with(&mut self.orbit, |fd| {
                fd.scene = SceneFrame::from_surface_items(items);
                fd.camera.viewport_index = vp_index;
            });

            // Pixel inspector: queue the clicked pixel (physical pixels) before the
            // render so the readback below reflects this frame.
            if self.pixel_inspector_active && self.debug_vis_active && response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let px = ((pos.x - rect.left()) * ppp) as u32;
                    let py = ((pos.y - rect.top()) * ppp) as u32;
                    self.last_picked_pos = Some((px, py));
                    self.pixel_read_req = Some((px, py));
                }
            }

            // Render into the offscreen target.
            let cmd = self.session.render(&rs.device, &rs.queue, &target.view);
            rs.queue.submit(std::iter::once(cmd));

            // Latch diagnostics from this frame's render.
            let (inst, batches, stats) = {
                let r = self.session.renderer_mut();
                (
                    r.is_using_instanced_path(),
                    r.instanced_batch_count(),
                    r.shadow_debug_stats(),
                )
            };
            self.instancing_status = (inst, batches);
            self.shadow_stats = Some(stats);

            // Metered EV readback (auto-exposure only; blocking device poll).
            if self.wants_exposure_readback() {
                if let Some(rb) = self
                    .session
                    .renderer_mut()
                    .exposure_state(&rs.device, &rs.queue, self.vp_id)
                {
                    self.last_exposure = Some(rb);
                }
            }

            // Pixel inspector readback for this frame's queued pixel.
            if let Some((px, py)) = self.pixel_read_req.take() {
                self.last_picked_values = self
                    .session
                    .renderer_mut()
                    .read_debug_pixel(&rs.device, &rs.queue, px, py);
            }

            // Display the rendered texture.
            ui.painter().image(
                target.id,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Status bar: shader path and cascade count.
            let (is_instanced, batch_count) = self.instancing_status;
            let path_label = if is_instanced {
                format!("Shader path: Instanced ({} batches)", batch_count)
            } else {
                "Shader path: Per-object".to_string()
            };
            let status_text = format!("{}   Cascades: {}", path_label, self.shadow_cascade_count);
            ui.painter().text(
                egui::pos2(rect.left() + 8.0, rect.bottom() - 20.0),
                egui::Align2::LEFT_BOTTOM,
                &status_text,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgba_premultiplied(230, 230, 230, 220),
            );

            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Lighting panel UI
// ---------------------------------------------------------------------------

impl App {
    fn ui_lighting_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Lighting");
        ui.separator();

        // Posture presets: set light magnitudes and the matching exposure together
        // (the pairing `EffectsFrame::with_posture` guarantees), seeded into the
        // panel state so the individual controls stay authoritative afterwards.
        ui.horizontal(|ui| {
            ui.label("Posture:");
            if ui.button("Faithful").clicked() {
                self.apply_posture_faithful();
            }
            if ui.button("Physical daylight").clicked() {
                self.apply_posture_daylight();
            }
        });
        ui.add_space(4.0);

        // Light source
        egui::CollapsingHeader::new("Light Source")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.light_kind, 0, "Directional");
                    ui.radio_value(&mut self.light_kind, 1, "Point");
                    ui.radio_value(&mut self.light_kind, 2, "Spot");
                });
                ui.add_space(4.0);

                match self.light_kind {
                    0 => {
                        ui.label("Direction (toward light):");
                        ui_vec3(ui, &mut self.dir_direction, 0.01);
                    }
                    1 => {
                        ui.label("Position:");
                        ui_vec3(ui, &mut self.point_position, 0.1);
                        ui.add(egui::Slider::new(&mut self.point_range, 1.0..=100.0).text("Range"));
                        ui.add(
                            egui::Slider::new(&mut self.light_radius, 0.0..=2.0)
                                .text("Source radius"),
                        );
                    }
                    _ => {
                        ui.label("Position:");
                        ui_vec3(ui, &mut self.spot_position, 0.1);
                        ui.label("Direction:");
                        ui_vec3(ui, &mut self.spot_direction, 0.01);
                        ui.add(egui::Slider::new(&mut self.spot_range, 1.0..=100.0).text("Range"));
                        ui.add(
                            egui::Slider::new(&mut self.light_radius, 0.0..=2.0)
                                .text("Source radius"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.spot_inner_deg, 1.0..=89.0)
                                .text("Inner angle"),
                        );
                        // Keep outer >= inner
                        self.spot_outer_deg = self.spot_outer_deg.max(self.spot_inner_deg);
                        ui.add(
                            egui::Slider::new(&mut self.spot_outer_deg, self.spot_inner_deg..=89.0)
                                .text("Outer angle"),
                        );
                    }
                }

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Colour:");
                    ui.color_edit_button_rgb(&mut self.light_colour);
                });

                ui.add_space(4.0);
                ui.checkbox(&mut self.photometric, "Photometric units");
                // Intensity in the unit that matches the kind and mode. Nominal is
                // the faithful scale (reads at EV 0); photometric is real lux for
                // directional and lumens for point/spot bulbs.
                match (self.light_kind, self.photometric) {
                    (0, false) => {
                        ui.add(
                            egui::Slider::new(&mut self.dir_nominal, 0.0..=8.0).text("Intensity"),
                        );
                    }
                    (0, true) => {
                        ui.add(
                            egui::Slider::new(&mut self.dir_lux, 1.0..=120_000.0)
                                .logarithmic(true)
                                .text("Illuminance (lux)"),
                        );
                    }
                    (1, false) | (2, false) => {
                        let v = if self.light_kind == 1 {
                            &mut self.point_candela
                        } else {
                            &mut self.spot_candela
                        };
                        ui.add(egui::Slider::new(v, 0.0..=2_000.0).text("Intensity (cd)"));
                    }
                    (1, true) | (2, true) => {
                        let v = if self.light_kind == 1 {
                            &mut self.point_lumens
                        } else {
                            &mut self.spot_lumens
                        };
                        ui.add(
                            egui::Slider::new(v, 1.0..=10_000.0)
                                .logarithmic(true)
                                .text("Flux (lumens)"),
                        );
                    }
                    _ => {}
                }

                ui.add_space(4.0);
                ui.add(egui::Slider::new(&mut self.light_importance, 0.0..=4.0).text("Importance"));
                ui.checkbox(&mut self.light_cast_shadows, "Cast shadows");
            });

        ui.add_space(4.0);

        // Hemisphere ambient
        egui::CollapsingHeader::new("Hemisphere Ambient")
            .default_open(true)
            .show(ui, |ui| {
                // Same linear scale as the key light. Small in the nominal regime;
                // thousands under photometric daylight (double-click to type).
                ui.add(
                    egui::Slider::new(&mut self.hemisphere_intensity, 0.0..=20_000.0)
                        .text("Intensity"),
                );
                ui.horizontal(|ui| {
                    ui.label("Sky:   ");
                    ui.color_edit_button_rgb(&mut self.sky_colour);
                });
                ui.horizontal(|ui| {
                    ui.label("Ground:");
                    ui.color_edit_button_rgb(&mut self.ground_colour);
                });
            });

        ui.add_space(4.0);

        self.ui_display_exposure(ui);

        ui.add_space(4.0);

        self.ui_environment(ui);

        ui.add_space(4.0);

        // Scene toggles
        egui::CollapsingHeader::new("Scene")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.show_platform, "Show platform");
                ui.checkbox(&mut self.pbr_enabled, "PBR shading (off = Phong)");
            });

        ui.add_space(4.0);

        // Shadows
        egui::CollapsingHeader::new("Shadows")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut self.shadows_enabled, "Enabled");
                ui.add_space(2.0);

                ui.add(
                    egui::Slider::new(&mut self.shadow_bias, 0.0..=0.005)
                        .text("Bias")
                        .custom_formatter(|v, _| format!("{:.5}", v)),
                );

                ui.add_space(4.0);
                ui.label("Cascade count:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.shadow_cascade_count, 1, "1");
                    ui.radio_value(&mut self.shadow_cascade_count, 2, "2");
                    ui.radio_value(&mut self.shadow_cascade_count, 4, "4");
                });

                ui.add_space(4.0);
                ui.label("Filter (cheapest to most expensive):");
                ui.horizontal_wrapped(|ui| {
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::Hard, "Hard");
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::Pcf, "PCF");
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::PcfHigh, "PCF high");
                });
                ui.horizontal_wrapped(|ui| {
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::PcssFast, "PCSS fast");
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::Pcss, "PCSS");
                });
                if matches!(
                    self.shadow_filter,
                    ShadowFilter::Pcss | ShadowFilter::PcssFast
                ) {
                    ui.add(
                        egui::Slider::new(&mut self.pcss_light_radius, 0.001..=0.2)
                            .text("PCSS radius"),
                    );
                }

                ui.add_space(4.0);
                ui.label("Atlas resolution:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.shadow_atlas_resolution, 1024, "1K");
                    ui.radio_value(&mut self.shadow_atlas_resolution, 2048, "2K");
                    ui.radio_value(&mut self.shadow_atlas_resolution, 4096, "4K");
                });

                ui.add_space(4.0);
                ui.checkbox(&mut self.shadow_extent_enabled, "Override extent");
                if self.shadow_extent_enabled {
                    ui.add(
                        egui::Slider::new(&mut self.shadow_extent_value, 2.0..=100.0)
                            .text("Extent (m)"),
                    );
                }
            });

        ui.add_space(4.0);

        egui::CollapsingHeader::new("Shadow Atlas Viewer")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.show_shadow_atlas, "Show atlas");
                    if self.show_shadow_atlas {
                        egui::ComboBox::from_id_salt("atlas_corner")
                            .selected_text(atlas_corner_label(self.atlas_viewer_corner))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.atlas_viewer_corner,
                                    AtlasViewerCorner::TopLeft,
                                    "Top-left",
                                );
                                ui.selectable_value(
                                    &mut self.atlas_viewer_corner,
                                    AtlasViewerCorner::TopRight,
                                    "Top-right",
                                );
                                ui.selectable_value(
                                    &mut self.atlas_viewer_corner,
                                    AtlasViewerCorner::BottomLeft,
                                    "Bottom-left",
                                );
                                ui.selectable_value(
                                    &mut self.atlas_viewer_corner,
                                    AtlasViewerCorner::BottomRight,
                                    "Bottom-right",
                                );
                            });
                    }
                });
                if self.show_shadow_atlas {
                    ui.add(egui::Slider::new(&mut self.atlas_viewer_scale, 0.1..=0.6).text("Size"));
                }
            });

        ui.add_space(4.0);

        egui::CollapsingHeader::new("Debug Visualization")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.debug_vis_active, "Active");
                if self.debug_vis_active {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        if ui.small_button("Atlas UV").clicked() {
                            self.debug_vis_r = viewport_lib::DebugQuantity::AtlasUvX;
                            self.debug_vis_g = viewport_lib::DebugQuantity::AtlasUvY;
                            self.debug_vis_b = viewport_lib::DebugQuantity::Zero;
                        }
                        if ui.small_button("Depth compare").clicked() {
                            self.debug_vis_r = viewport_lib::DebugQuantity::BiasedDepth;
                            self.debug_vis_g = viewport_lib::DebugQuantity::SurfaceDepth;
                            self.debug_vis_b = viewport_lib::DebugQuantity::NdotL;
                        }
                        if ui.small_button("Direct light").clicked() {
                            self.debug_vis_r = viewport_lib::DebugQuantity::DirectLightLuminance;
                            self.debug_vis_g = viewport_lib::DebugQuantity::AmbientLuminance;
                            self.debug_vis_b = viewport_lib::DebugQuantity::Zero;
                            self.debug_vis_scale = 1.0;
                        }
                        if ui.small_button("IBL split").clicked() {
                            self.debug_vis_r = viewport_lib::DebugQuantity::IblDiffuseLuminance;
                            self.debug_vis_g = viewport_lib::DebugQuantity::IblSpecularLuminance;
                            self.debug_vis_b = viewport_lib::DebugQuantity::Zero;
                            self.debug_vis_scale = 1.0;
                        }
                    });
                    ui.add_space(4.0);
                    ui.label("Mode:");
                    ui.horizontal(|ui| {
                        if ui
                            .radio(
                                !self.debug_vis_splitscreen && self.debug_vis_mode_replace,
                                "Replace",
                            )
                            .clicked()
                        {
                            self.debug_vis_splitscreen = false;
                            self.debug_vis_mode_replace = true;
                        }
                        if ui
                            .radio(
                                !self.debug_vis_splitscreen && !self.debug_vis_mode_replace,
                                "Tint overlay",
                            )
                            .clicked()
                        {
                            self.debug_vis_splitscreen = false;
                            self.debug_vis_mode_replace = false;
                        }
                        if ui
                            .radio(self.debug_vis_splitscreen, "Split screen")
                            .clicked()
                        {
                            self.debug_vis_splitscreen = true;
                        }
                    });
                    if self.debug_vis_splitscreen {
                        ui.add(
                            egui::Slider::new(&mut self.debug_vis_split_x, 0.0..=1.0).text("Split"),
                        );
                    }
                    ui.add_space(4.0);
                    ui.label("R channel:");
                    egui::ComboBox::from_id_salt("dbg_r")
                        .selected_text(debug_quantity_label(self.debug_vis_r))
                        .show_ui(ui, |ui| {
                            for &q in DebugQuantity::all_variants() {
                                ui.selectable_value(
                                    &mut self.debug_vis_r,
                                    q,
                                    debug_quantity_label(q),
                                );
                            }
                        });
                    ui.label("G channel:");
                    egui::ComboBox::from_id_salt("dbg_g")
                        .selected_text(debug_quantity_label(self.debug_vis_g))
                        .show_ui(ui, |ui| {
                            for &q in DebugQuantity::all_variants() {
                                ui.selectable_value(
                                    &mut self.debug_vis_g,
                                    q,
                                    debug_quantity_label(q),
                                );
                            }
                        });
                    ui.label("B channel:");
                    egui::ComboBox::from_id_salt("dbg_b")
                        .selected_text(debug_quantity_label(self.debug_vis_b))
                        .show_ui(ui, |ui| {
                            for &q in DebugQuantity::all_variants() {
                                ui.selectable_value(
                                    &mut self.debug_vis_b,
                                    q,
                                    debug_quantity_label(q),
                                );
                            }
                        });
                    ui.add_space(4.0);
                    ui.add(
                        egui::Slider::new(&mut self.debug_vis_scale, 0.1..=100.0)
                            .text("Scale")
                            .logarithmic(true),
                    );
                }
            });

        ui.add_space(4.0);

        egui::CollapsingHeader::new("Pixel Inspector")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.pixel_inspector_active, "Active");
                if self.pixel_inspector_active {
                    if !self.debug_vis_active {
                        ui.label("Enable Debug Visualization first.");
                    } else {
                        ui.label("Click a pixel in the viewport to read its debug values.");
                    }
                }

                if let Some((px, py)) = self.last_picked_pos {
                    ui.add_space(4.0);
                    ui.label(format!("Last picked: ({}, {})", px, py));
                    if let Some(vals) = self.last_picked_values {
                        ui.horizontal(|ui| {
                            ui.label("R");
                            ui.label(format!(
                                "{} ({:.4})",
                                debug_quantity_label(self.debug_vis_r),
                                vals[0]
                            ));
                        });
                        ui.horizontal(|ui| {
                            ui.label("G");
                            ui.label(format!(
                                "{} ({:.4})",
                                debug_quantity_label(self.debug_vis_g),
                                vals[1]
                            ));
                        });
                        ui.horizontal(|ui| {
                            ui.label("B");
                            ui.label(format!(
                                "{} ({:.4})",
                                debug_quantity_label(self.debug_vis_b),
                                vals[2]
                            ));
                        });
                    } else {
                        ui.label("(reading...)");
                    }
                }
            });

        ui.add_space(4.0);

        // D8: Frame stats footer.
        egui::CollapsingHeader::new("Frame Stats")
            .default_open(false)
            .show(ui, |ui| {
                let stats = self.shadow_stats;
                if let Some(s) = stats {
                    let path = if s.using_instanced_path {
                        format!("Instanced ({} batches)", s.instanced_batch_count)
                    } else {
                        "Per-object".to_string()
                    };
                    ui.label(format!("Path:     {}", path));
                    let splits: Vec<String> = s.cascade_splits[..s.cascade_count as usize]
                        .iter()
                        .map(|v| format!("{:.1}", v))
                        .collect();
                    let splits_str = splits.join(" / ");
                    ui.label(format!(
                        "Cascades: {}  splits: {}",
                        s.cascade_count, splits_str
                    ));
                    let extent_label = if self.shadow_extent_enabled {
                        format!("{:.1} m (override)", s.shadow_extent_world)
                    } else {
                        format!("{:.1} m (auto)", s.shadow_extent_world)
                    };
                    ui.label(format!(
                        "Atlas:    {} px   Extent: {}",
                        s.shadow_atlas_resolution, extent_label
                    ));
                    let contact = if s.contact_shadow_active {
                        "enabled"
                    } else {
                        "disabled"
                    };
                    ui.label(format!("Contact:  {}", contact));
                } else {
                    ui.label("(no data yet)");
                }
            });

        ui.add_space(4.0);

        // D9: Diagnostic presets.
        egui::CollapsingHeader::new("Diagnostic Presets")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("One-click setups for each known bug class.");
                ui.add_space(4.0);

                if ui.button("Peter-panning / shadow gap").clicked() {
                    self.apply_preset_peter_panning();
                }
                if ui.button("Shadow acne").clicked() {
                    self.apply_preset_acne();
                }
                if ui.button("Cascade band / seam").clicked() {
                    self.apply_preset_cascade_band();
                }
                if ui.button("Ghost shapes (orbit)").clicked() {
                    self.apply_preset_ghost_shapes();
                }
                if ui.button("Contact shadow bright spot").clicked() {
                    self.apply_preset_contact_bright_spot();
                }
            });
    }

    fn ui_display_exposure(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Display & Exposure")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("Pipeline:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.hdr_pipeline, true, "HDR");
                    ui.radio_value(&mut self.hdr_pipeline, false, "Direct");
                });
                if !self.hdr_pipeline {
                    ui.label("Direct passthrough: exposure, tone map, and post");
                    ui.label("effects are skipped.");
                    return;
                }

                ui.add_space(4.0);
                ui.label("Exposure mode:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.exposure_mode, 0, "Manual");
                    ui.radio_value(&mut self.exposure_mode, 1, "Physical");
                    ui.radio_value(&mut self.exposure_mode, 2, "Auto");
                });

                match self.exposure_mode {
                    0 => {
                        ui.add(egui::Slider::new(&mut self.manual_ev, -6.0..=16.0).text("EV100"));
                    }
                    1 => {
                        ui.add(
                            egui::Slider::new(&mut self.aperture, 1.0..=22.0).text("Aperture f/"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.shutter_denom, 4.0..=4000.0)
                                .logarithmic(true)
                                .text("Shutter 1/s"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.iso, 50.0..=6400.0)
                                .logarithmic(true)
                                .text("ISO"),
                        );
                    }
                    _ => {
                        ui.add(
                            egui::Slider::new(&mut self.auto.adaptation, 0.0..=1.0)
                                .text("Adaptation"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.auto.min_ev, -10.0..=6.0).text("EV min"),
                        );
                        ui.add(egui::Slider::new(&mut self.auto.max_ev, 4.0..=20.0).text("EV max"));
                        ui.add(
                            egui::Slider::new(&mut self.auto.speed_up, 0.1..=10.0).text("Speed up"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.auto.speed_down, 0.1..=10.0)
                                .text("Speed down"),
                        );
                    }
                }

                ui.add_space(4.0);
                ui.add(
                    egui::Slider::new(&mut self.exposure_compensation, -3.0..=3.0)
                        .text("Compensation"),
                );

                ui.add_space(4.0);
                ui.label("Tone map:");
                ui.horizontal_wrapped(|ui| {
                    ui.radio_value(
                        &mut self.tone_mapping,
                        ToneMapping::KhronosNeutral,
                        "Khronos",
                    );
                    ui.radio_value(&mut self.tone_mapping, ToneMapping::Aces, "ACES");
                    ui.radio_value(&mut self.tone_mapping, ToneMapping::Reinhard, "Reinhard");
                });

                ui.add_space(4.0);
                // EV readout: known in-app for Manual/Physical; metered on the GPU
                // for Automatic, read back via exposure_state.
                match self.panel_ev() {
                    Some(ev) => {
                        ui.label(format!("EV100: {:.2}", ev - self.exposure_compensation));
                    }
                    None => match self.last_exposure {
                        Some(rb) => {
                            let settling = if rb.adapting { " (adapting)" } else { "" };
                            ui.label(format!(
                                "EV100: {:.2} -> {:.2}{}",
                                rb.current_ev, rb.target_ev, settling
                            ));
                        }
                        None => {
                            ui.label("EV100: metering...");
                        }
                    },
                }
            });
    }

    fn ui_environment(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Environment (IBL)")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.env_enabled, "Enabled");
                if self.env_enabled {
                    // Absolute nits: near-black at ~1 under photometric exposure,
                    // thousands for a daytime sky. Double-click to type a value.
                    ui.add(
                        egui::Slider::new(&mut self.env_intensity, 0.0..=20_000.0)
                            .text("Intensity (nits)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.env_rotation, 0.0..=std::f32::consts::TAU)
                            .text("Rotation"),
                    );
                    ui.checkbox(&mut self.env_show_skybox, "Show skybox");
                }
            });
    }

    /// Faithful posture: nominal magnitudes at neutral EV 0 ("colour is data").
    /// Mirrors `LightingPosture::Faithful` into the panel state.
    fn apply_posture_faithful(&mut self) {
        self.photometric = false;
        self.dir_nominal = std::f32::consts::PI;
        self.hemisphere_intensity = 1.5;
        self.hdr_pipeline = true;
        self.exposure_mode = 0;
        self.manual_ev = 0.0;
        self.exposure_compensation = 0.0;
        self.tone_mapping = ToneMapping::KhronosNeutral;
    }

    /// Physical-daylight posture: real daylight lux mapped down by auto-exposure.
    /// Mirrors `LightingPosture::PhysicalDaylight` into the panel state.
    fn apply_posture_daylight(&mut self) {
        self.photometric = true;
        self.dir_lux = Lux::FULL_DAYLIGHT.0;
        // Clear-sky fill proportional to the daylight key (see LightingSettings::daylight).
        self.hemisphere_intensity = 8_000.0;
        self.hdr_pipeline = true;
        self.exposure_mode = 2;
        self.auto = AutoExposure::default();
        self.exposure_compensation = 0.0;
        self.tone_mapping = ToneMapping::KhronosNeutral;
    }
}

// ---------------------------------------------------------------------------
// Diagnostic preset helpers
// ---------------------------------------------------------------------------

impl App {
    fn apply_preset_peter_panning(&mut self) {
        // Side-on view of the sphere at (-4, 0, 0.6), low camera angle.
        // SplitScreen: left = normal, right = ShadowFactor.
        self.tab = Tab::Basic;
        self.session.camera_mut().center = glam::Vec3::new(-4.0, 0.0, 0.6);
        self.session.camera_mut().distance = 7.0;
        // Looking from the Y+ side, slightly above horizontal.
        self.session.camera_mut().orientation =
            glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)
                * glam::Quat::from_rotation_x(1.42);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::ShadowFactor;
        self.debug_vis_g = DebugQuantity::Zero;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 1.0;
        self.debug_vis_split_x = 0.5;
    }

    fn apply_preset_acne(&mut self) {
        // Steep top-down view of the rough sphere in the Materials tab.
        // SplitScreen: left = normal, right = ShadowFactor (acne appears as bright speckles).
        self.tab = Tab::Materials;
        self.session.camera_mut().center = glam::Vec3::new(4.5, 2.5, 0.7);
        self.session.camera_mut().distance = 4.0;
        // Looking down at steep angle.
        self.session.camera_mut().orientation =
            glam::Quat::from_rotation_z(0.3) * glam::Quat::from_rotation_x(0.5);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::ShadowFactor;
        self.debug_vis_g = DebugQuantity::Zero;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 1.0;
        self.debug_vis_split_x = 0.5;
    }

    fn apply_preset_cascade_band(&mut self) {
        // Pulled back to see the full scene from a moderate angle.
        // SplitScreen: left = normal, right = CascadeIndex (bands as distinct colours).
        self.tab = Tab::Basic;
        self.session.camera_mut().center = glam::Vec3::new(0.0, 0.0, 0.5);
        self.session.camera_mut().distance = 22.0;
        self.session.camera_mut().orientation =
            glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.0);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::CascadeIndex;
        self.debug_vis_g = DebugQuantity::Zero;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 0.33;
        self.debug_vis_split_x = 0.5;
    }

    fn apply_preset_ghost_shapes(&mut self) {
        // Mid-distance orbit view. Orbit while watching AtlasUV on the right half.
        // Unstable UVs during orbit point to cascade matrix instability.
        self.tab = Tab::Basic;
        self.session.camera_mut().center = glam::Vec3::new(0.0, 0.0, 0.5);
        self.session.camera_mut().distance = 12.0;
        self.session.camera_mut().orientation =
            glam::Quat::from_rotation_z(1.0) * glam::Quat::from_rotation_x(1.15);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::AtlasUvX;
        self.debug_vis_g = DebugQuantity::AtlasUvY;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 1.0;
        self.debug_vis_split_x = 0.5;
    }

    fn apply_preset_contact_bright_spot(&mut self) {
        // Looking at the torus from above at a low angle so the inner ring is visible.
        // SplitScreen: left = normal, right = ContactShadowFactor.
        self.tab = Tab::Basic;
        self.session.camera_mut().center = glam::Vec3::new(4.0, 0.0, 0.18);
        self.session.camera_mut().distance = 5.0;
        self.session.camera_mut().orientation =
            glam::Quat::from_rotation_z(-0.4) * glam::Quat::from_rotation_x(1.3);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::ContactShadowFactor;
        self.debug_vis_g = DebugQuantity::Zero;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 1.0;
        self.debug_vis_split_x = 0.5;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Three drag-value fields for a vec3, laid out horizontally.
fn ui_vec3(ui: &mut egui::Ui, v: &mut [f32; 3], speed: f64) {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut v[0]).speed(speed).prefix("x: "));
        ui.add(egui::DragValue::new(&mut v[1]).speed(speed).prefix("y: "));
        ui.add(egui::DragValue::new(&mut v[2]).speed(speed).prefix("z: "));
    });
}

/// Equirectangular sky/ground gradient (RGBA f32), sky at the top row. Values are
/// relative radiance; `EnvironmentSettings::intensity` scales them to nits.
fn equirect_gradient(sky: [f32; 3], ground: [f32; 3], w: u32, h: u32) -> Vec<f32> {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let v = y as f32 / (h - 1).max(1) as f32;
        let t = v * v * (3.0 - 2.0 * v); // soft horizon
        for _x in 0..w {
            px.push(sky[0] + (ground[0] - sky[0]) * t);
            px.push(sky[1] + (ground[1] - sky[1]) * t);
            px.push(sky[2] + (ground[2] - sky[2]) * t);
            px.push(1.0);
        }
    }
    px
}

/// Display label for an AtlasViewerCorner variant.
#[allow(unreachable_patterns)]
fn atlas_corner_label(c: AtlasViewerCorner) -> &'static str {
    match c {
        AtlasViewerCorner::TopLeft => "Top-left",
        AtlasViewerCorner::TopRight => "Top-right",
        AtlasViewerCorner::BottomLeft => "Bottom-left",
        AtlasViewerCorner::BottomRight => "Bottom-right",
        _ => "?",
    }
}

/// Display label for a DebugQuantity variant.
fn debug_quantity_label(q: DebugQuantity) -> &'static str {
    match q {
        DebugQuantity::Zero => "Zero (black)",
        DebugQuantity::One => "One (white)",
        DebugQuantity::CascadeIndex => "Cascade index",
        DebugQuantity::ShadowFactor => "Shadow factor",
        DebugQuantity::ContactShadowFactor => "Contact shadow",
        DebugQuantity::NdotL => "N dot L",
        DebugQuantity::NormalBiasMagnitude => "Normal bias magnitude",
        DebugQuantity::AtlasUvX => "Atlas UV X",
        DebugQuantity::AtlasUvY => "Atlas UV Y",
        DebugQuantity::TileUvX => "Tile UV X",
        DebugQuantity::TileUvY => "Tile UV Y",
        DebugQuantity::BiasedDepth => "Biased depth",
        DebugQuantity::SurfaceDepth => "Surface depth",
        DebugQuantity::WorldNormalX => "World normal X",
        DebugQuantity::WorldNormalY => "World normal Y",
        DebugQuantity::WorldNormalZ => "World normal Z",
        DebugQuantity::Roughness => "Roughness",
        DebugQuantity::Metallic => "Metallic",
        DebugQuantity::AoFactor => "AO factor",
        DebugQuantity::DirectLightLuminance => "Direct light lum.",
        DebugQuantity::AmbientLuminance => "Ambient lum.",
        DebugQuantity::IblDiffuseLuminance => "IBL diffuse lum.",
        DebugQuantity::IblSpecularLuminance => "IBL specular lum.",
        DebugQuantity::EmissiveLuminance => "Emissive lum.",
        _ => "Unknown",
    }
}
