//! Showcase 21: Image Textures.
//!
//! Demonstrates UV-mapped image textures uploaded via `upload_texture`.
//!
//! Layout (Z-up, 2x2 grid in X-Z plane at Y=0):
//!   Back-left   (-3, 0, +3):  Plane (two-sided) : Percy photo
//!   Back-right  (+3, 0, +3):  UV sphere : procedural checkerboard
//!   Front-left  (-3, 0, -3):  Cube : procedural color-gradient per face
//!   Front-right (+3, 0, -3):  Torus : procedural stripe pattern
//!
//! The Percy photo is stored as pre-converted raw RGBA bytes alongside this
//! file, so no image-parsing dependency or build script is required.

use crate::App;
use eframe::egui;
use viewport_lib::{Material, ViewportRenderer, scene::Scene};

// Percy photo : pre-converted to raw RGBA (2203 × 2009).
const PERCY_WIDTH: u32 = 2203;
const PERCY_HEIGHT: u32 = 2009;
const PERCY_RGBA: &[u8] = include_bytes!("percy.rgba");

impl App {
    pub(crate) fn build_texture_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.texture_scene = Scene::new();

        let res = renderer.resources_mut();

        // --- Percy photo on a plane ---
        let percy_tex = res
            .upload_texture(
                &self.device,
                &self.queue,
                PERCY_WIDTH,
                PERCY_HEIGHT,
                PERCY_RGBA,
            )
            .expect("percy texture upload");

        let plane = viewport_lib::geometry::primitives::plane(4.5, 4.5 * PERCY_HEIGHT as f32 / PERCY_WIDTH as f32);
        let plane_id = res
            .upload_mesh_data(&self.device, &plane)
            .expect("plane mesh upload");

        self.texture_plane_node = self.texture_scene.add_named(
            "Percy Plane",
            Some(viewport_lib::MeshId::from_index(plane_id)),
            glam::Mat4::from_translation(glam::Vec3::new(-3.0, 0.0, 3.0)),
            {
                let mut m = Material::default();
                m.texture_id = Some(percy_tex);
                m.ambient = 0.9;
                m.diffuse = 0.4;
                m
            },
        );
        // (two-sided set in build_frame_data via texture_plane_node)

        // --- Checkerboard on a sphere ---
        let checker = make_checkerboard(256, 8, [220, 220, 220, 255], [40, 40, 40, 255]);
        let checker_tex = res
            .upload_texture(&self.device, &self.queue, 256, 256, &checker)
            .expect("checker texture upload");

        let sphere = viewport_lib::geometry::primitives::sphere(1.8, 48, 24);
        let sphere_id = res
            .upload_mesh_data(&self.device, &sphere)
            .expect("sphere mesh upload");

        self.texture_scene.add_named(
            "Checker Sphere",
            Some(viewport_lib::MeshId::from_index(sphere_id)),
            glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 3.0)),
            {
                let mut m = Material::default();
                m.texture_id = Some(checker_tex);
                m.ambient = 0.3;
                m.diffuse = 0.8;
                m.specular = 0.4;
                m.shininess = 32.0;
                m
            },
        );

        // --- Color-gradient on a cube ---
        let gradient = make_gradient(256);
        let gradient_tex = res
            .upload_texture(&self.device, &self.queue, 256, 256, &gradient)
            .expect("gradient texture upload");

        let cube = viewport_lib::geometry::primitives::cube(3.0);
        let cube_id = res
            .upload_mesh_data(&self.device, &cube)
            .expect("cube mesh upload");

        self.texture_scene.add_named(
            "Gradient Cube",
            Some(viewport_lib::MeshId::from_index(cube_id)),
            glam::Mat4::from_translation(glam::Vec3::new(-3.0, 0.0, -3.0)),
            {
                let mut m = Material::default();
                m.texture_id = Some(gradient_tex);
                m.ambient = 0.3;
                m.diffuse = 0.8;
                m
            },
        );

        // --- Stripe pattern on a torus ---
        let stripes = make_stripes(256, 16, [180, 100, 30, 255], [230, 200, 140, 255]);
        let stripes_tex = res
            .upload_texture(&self.device, &self.queue, 256, 256, &stripes)
            .expect("stripes texture upload");

        let torus = viewport_lib::geometry::primitives::torus(1.5, 0.5, 48, 24);
        let torus_id = res
            .upload_mesh_data(&self.device, &torus)
            .expect("torus mesh upload");

        self.texture_scene.add_named(
            "Stripe Torus",
            Some(viewport_lib::MeshId::from_index(torus_id)),
            glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, -3.0)),
            {
                let mut m = Material::default();
                m.texture_id = Some(stripes_tex);
                m.ambient = 0.3;
                m.diffuse = 0.8;
                m.specular = 0.6;
                m.shininess = 64.0;
                m
            },
        );

        self.texture_built = true;
    }

    pub(crate) fn controls_textures(&mut self, ui: &mut egui::Ui) {
        ui.label("UV-mapped image textures on four primitives:");
        ui.label("  Back-left:  plane : Percy photo");
        ui.label("  Left:    sphere : procedural checkerboard");
        ui.label("  Right:   cube : procedural color gradient");
        ui.label("  Front:   torus : procedural stripes");
        ui.separator();
        ui.label("Textures are uploaded as raw RGBA via upload_texture().");
        ui.label("The photo is stored as pre-converted raw RGBA bytes.");
    }
}

// ---------------------------------------------------------------------------
// Procedural texture generators
// ---------------------------------------------------------------------------

/// Checkerboard: `cells` × `cells` grid alternating between `a` and `b`.
fn make_checkerboard(size: usize, cells: usize, a: [u8; 4], b: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * size * 4);
    for row in 0..size {
        for col in 0..size {
            let cx = col * cells / size;
            let cy = row * cells / size;
            let color = if (cx + cy) % 2 == 0 { a } else { b };
            out.extend_from_slice(&color);
        }
    }
    out
}

/// Smooth HSV hue gradient scrolling across both axes.
fn make_gradient(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * size * 4);
    for row in 0..size {
        for col in 0..size {
            let h = (col as f32 / size as f32 + row as f32 / size as f32 * 0.3).fract();
            let (r, g, b) = hsv_to_rgb(h, 0.75, 0.92);
            out.push((r * 255.0) as u8);
            out.push((g * 255.0) as u8);
            out.push((b * 255.0) as u8);
            out.push(255);
        }
    }
    out
}

/// Horizontal stripes alternating between `a` and `b`.
fn make_stripes(size: usize, count: usize, a: [u8; 4], b: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * size * 4);
    for row in 0..size {
        let band = row * count / size;
        let color = if band % 2 == 0 { a } else { b };
        for _ in 0..size {
            out.extend_from_slice(&color);
        }
    }
    out
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = h * 6.0;
    let i = h6.floor() as u32 % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}
