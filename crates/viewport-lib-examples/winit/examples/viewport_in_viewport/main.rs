//! Three viewports composited inside a fourth, with their backgrounds fading
//! between opaque and absent.
//!
//! The parent viewport draws its own scene. Three children each draw a
//! different primitive over their own background colour, render into offscreen
//! targets, and are composited into the parent's frame as they drift around it.
//! Each child's background opacity sweeps between 0 and 1 and pauses briefly at
//! each end, so all three regimes are on screen at once:
//!
//! - At alpha 1 a child is an opaque inset panel in its own colour.
//! - At alpha 0 only its geometry is left and the parent's scene shows through.
//! - In between it is a translucent plate the child's own content sits on.
//!
//! `ViewportFrame::background_colour` is premultiplied, so a child rendered at
//! alpha 0 comes back carrying only what it drew, with alpha as coverage.
//! Compositing that needs a blend that takes premultiplied input, which is what
//! `create_blit_composite` + `blit_composite_rect` are: a plain `blit` has no
//! blend state and would replace the parent instead of drawing over it.
//!
//! The children are separate `ViewportInstance`s rather than viewports on one
//! renderer. Four independent scenes through one `ViewportRenderer` would have
//! to keep their `SceneFrame::generation` values apart to avoid the instanced
//! batch cache reusing one child's batches for another, and each render path
//! would pump the shared upload runner. Neither is what this example is about.

use std::cell::RefCell;
use std::rc::Rc;

use viewport_lib as vpl;
use vpl::{
    AppConfigV2, BlitTexture, Colour, Material, OffscreenViewportTarget, ViewportAppV2,
    ViewportContext, ViewportInstance, WindowConfig, primitives,
};

/// Seconds for one full fade, and seconds held at each end.
const FADE_SECS: f32 = 2.5;
const HOLD_SECS: f32 = 1.2;

/// Background opacity at time `t`, offset per child by `phase`.
///
/// A triangle wave with a flat top and bottom: rising, held at 1, falling, held
/// at 0. The holds are what make the endpoints readable, which is the whole
/// point of sweeping rather than toggling: at alpha 1 the child should be
/// indistinguishable from an ordinary opaque viewport, and that is only
/// checkable if it sits there long enough to look at.
fn background_alpha(t: f32, phase: f32) -> f32 {
    let period = 2.0 * (FADE_SECS + HOLD_SECS);
    let u = (t + phase * period / 3.0).rem_euclid(period);
    if u < FADE_SECS {
        u / FADE_SECS
    } else if u < FADE_SECS + HOLD_SECS {
        1.0
    } else if u < 2.0 * FADE_SECS + HOLD_SECS {
        1.0 - (u - FADE_SECS - HOLD_SECS) / FADE_SECS
    } else {
        0.0
    }
}

/// One embedded viewport: its own scene, the offscreen target it renders into,
/// the composite handle, and the colour and phase that tell it apart.
struct Child {
    session: ViewportInstance,
    target: OffscreenViewportTarget,
    blit: Option<BlitTexture>,
    /// Background colour at full opacity. The alpha is replaced each frame.
    tint: [f32; 3],
    /// Offset into the fade cycle, so the three are never in step.
    phase: f32,
    /// Phase of the drift ellipse, so the three do not overlap constantly.
    orbit_phase: f32,
    /// Top-left of this child in the parent, physical pixels. Computed in the
    /// per-frame callback, which has the clock, and read in the paint hook,
    /// which does not.
    origin: (u32, u32),
}

impl Child {
    fn new(
        device: &vpl::wgpu::Device,
        surface_format: vpl::wgpu::TextureFormat,
        size: [u32; 2],
        mesh: &vpl::resources::MeshData,
        colour: [f32; 3],
        tint: [f32; 3],
        phase: f32,
        orbit_phase: f32,
    ) -> Self {
        // The offscreen instance targets the sRGB render format, so compositing
        // into the (sRGB) window surface encodes exactly once.
        let mut session = ViewportInstance::new(
            device,
            OffscreenViewportTarget::render_format(surface_format),
        );
        let id = session
            .resources_mut()
            .upload_mesh_data(device, mesh)
            .expect("upload child mesh");
        session.scene_mut().add(
            Some(id),
            glam::Mat4::IDENTITY,
            Material::from_colour(Colour::from_linear_rgb_array(colour)),
        );
        session.camera_mut().distance = 3.4;
        Self {
            session,
            target: OffscreenViewportTarget::new(device, surface_format, size),
            blit: None,
            tint,
            phase,
            orbit_phase,
            origin: (0, 0),
        }
    }

    /// Where this child sits in the parent this frame, in physical pixels. It
    /// drifts on a slow ellipse so the composite is exercised over a moving
    /// backdrop rather than a static one.
    fn place(&mut self, t: f32, parent: [u32; 2], size: [u32; 2]) {
        let a = t * 0.25 + self.orbit_phase * std::f32::consts::TAU;
        let free_w = parent[0].saturating_sub(size[0]) as f32;
        let free_h = parent[1].saturating_sub(size[1]) as f32;
        // 0.5 +- 0.4 keeps the whole child inside the parent at both extremes.
        let x = (0.5 + 0.4 * a.cos()) * free_w;
        let y = (0.5 + 0.4 * (a * 1.3).sin()) * free_h;
        self.origin = (x as u32, y as u32);
    }

    fn resize(&mut self, device: &vpl::wgpu::Device, size: [u32; 2]) {
        if self.target.resize(device, size) {
            // The texture was recreated, so the old handle points at a dead
            // view: drop it and build a new one against the new one.
            self.blit = None;
        }
    }

    /// Render this child into its offscreen target at this frame's background
    /// opacity, spinning its primitive so the two kinds of transparency (the
    /// background's, and the geometry's own edges) are both in motion.
    fn render(
        &mut self,
        device: &vpl::wgpu::Device,
        queue: &vpl::wgpu::Queue,
        t: f32,
        logical: [f32; 2],
        ppp: f32,
    ) {
        let alpha = background_alpha(t, self.phase);
        self.session.set_pixels_per_point(ppp);
        self.session.viewport_frame_mut().background_colour = Some(Colour::srgb(
            self.tint[0],
            self.tint[1],
            self.tint[2],
            alpha,
        ));
        // Identity looks down -Z from +Z in this Z-up world, so tilt off the
        // pole and then spin about Z.
        self.session.camera_mut().orientation =
            glam::Quat::from_rotation_z(t * 0.6) * glam::Quat::from_rotation_x(1.1);
        // `frame` is the assemble entry for a consumer moving the camera itself
        // rather than through an orbit controller. It has to run before `render`,
        // which draws the retained frame as it stands: without it the instance
        // keeps its default viewport size and the pass is rejected for a depth
        // attachment that does not match the offscreen colour target.
        self.session.frame(ViewportContext {
            hovered: false,
            focused: false,
            viewport_size: logical,
        });
        let cmd = self
            .session
            .render(device, queue, self.target.render_view());
        queue.submit(std::iter::once(cmd));
    }
}

#[derive(Default)]
struct Scene {
    children: Vec<Child>,
}

fn main() {
    let state: Rc<RefCell<Scene>> = Rc::new(RefCell::new(Scene::default()));
    let paint_state = state.clone();

    ViewportAppV2::new(AppConfigV2::default())
        .window_with_paint(
            WindowConfig::default()
                .with_title("viewport-lib : viewport in viewport")
                .with_window_size(1280, 800),
            // The parent's own scene: a ring of bars the children drift over, so
            // there is something recognisable to see through them.
            |vp, device| {
                let bar = vp
                    .resources_mut()
                    .upload_mesh_data(device, &primitives::cube(1.0))
                    .expect("upload parent mesh");
                for i in 0..12 {
                    let a = i as f32 / 12.0 * std::f32::consts::TAU;
                    let m = glam::Mat4::from_translation(glam::Vec3::new(
                        a.cos() * 3.2,
                        a.sin() * 3.2,
                        0.0,
                    )) * glam::Mat4::from_rotation_z(a)
                        * glam::Mat4::from_scale(glam::Vec3::new(0.5, 0.5, 2.2));
                    vp.scene_mut().add(
                        Some(bar),
                        m,
                        Material::from_colour(Colour::linear_rgb(0.30, 0.32, 0.38)),
                    );
                }
                vp.camera_mut().distance = 11.0;
                vp.viewport_frame_mut().background_colour =
                    Some(Colour::srgb_rgb(0.07, 0.08, 0.11));
            },
            move |ctx| {
                let device = ctx.device().clone();
                let queue = ctx.queue().clone();
                let [pw, ph] = ctx.surface_size();
                let [wl, _hl] = ctx.viewport_size();
                let fmt = ctx.surface_format();
                let ppp = pw as f32 / wl.max(1.0);
                let t = ctx.time();

                // A child is a third of the window, in physical pixels. The
                // logical size is derived from that rather than computed
                // alongside it: the instance sizes its depth attachment as
                // `round(logical * ppp)`, so deriving one from the other is what
                // keeps it equal to the offscreen colour target. Computing both
                // from the window independently leaves them one pixel apart at
                // some window sizes, and the render pass is rejected.
                let size = [(pw / 3).max(1), (ph / 3).max(1)];
                let logical = [size[0] as f32 / ppp, size[1] as f32 / ppp];

                let mut scene = state.borrow_mut();
                if scene.children.is_empty() {
                    // Three different primitives, three different background
                    // colours, three different phases.
                    scene.children.push(Child::new(
                        &device,
                        fmt,
                        size,
                        &primitives::torus(0.8, 0.3, 32, 16),
                        [0.95, 0.55, 0.15],
                        [0.55, 0.12, 0.10],
                        0.0,
                        0.0,
                    ));
                    scene.children.push(Child::new(
                        &device,
                        fmt,
                        size,
                        &primitives::sphere(1.0, 32, 16),
                        [0.35, 0.80, 0.95],
                        [0.08, 0.30, 0.45],
                        1.0,
                        0.33,
                    ));
                    scene.children.push(Child::new(
                        &device,
                        fmt,
                        size,
                        &primitives::cone(0.9, 1.6, 28),
                        [0.60, 0.95, 0.45],
                        [0.14, 0.38, 0.14],
                        2.0,
                        0.66,
                    ));
                }

                for child in scene.children.iter_mut() {
                    child.resize(&device, size);
                    child.place(t, [pw, ph], size);
                    child.render(&device, &queue, t, logical, ppp);
                    if child.blit.is_none() {
                        // `create_blit_composite`, not `create_blit`: this one
                        // builds the premultiplied-blend pipelines, and a plain
                        // blit would replace the parent rather than draw over it.
                        let blit = ctx
                            .renderer_mut()
                            .create_blit_composite(&device, child.target.render_view());
                        child.blit = Some(blit);
                    }
                }
            },
            move |pctx| {
                let scene = paint_state.borrow();
                let [pw, ph] = pctx.surface_size();
                let size = [(pw / 3).max(1), (ph / 3).max(1)];
                // The window's own render is already in the pass, so each child
                // composites over it in turn. Back to front is just submission
                // order here: the children overlap each other as they drift.
                for child in scene.children.iter() {
                    if let Some(blit) = child.blit.as_ref() {
                        let (x, y) = child.origin;
                        pctx.blit_composite_rect(blit, x, y, size[0], size[1]);
                    }
                }
            },
        )
        .run();
}
