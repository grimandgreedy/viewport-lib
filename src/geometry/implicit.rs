//! CPU sphere-marching of implicit surfaces (signed-distance functions).
//!
//! [`march_implicit_surface`] and [`march_implicit_surface_color`] accept a
//! user-supplied SDF closure, fire rays from the camera for each pixel, and
//! produce a [`crate::renderer::types::ScreenImageItem`] with per-pixel NDC
//! depth suitable for depth-compositing against scene geometry via Phase 12.
//!
//! # Usage
//!
//! ```rust,ignore
//! let opts = ImplicitRenderOptions {
//!     width: 320,
//!     height: 240,
//!     ..Default::default()
//! };
//! // Sphere of radius 1.5:
//! let img = march_implicit_surface(&camera, &opts, |p| p.length() - 1.5);
//! fd.scene.screen_images.push(img);
//! ```
//!
//! For colored surfaces supply a closure returning `(sdf_value, [r, g, b, a])`:
//!
//! ```rust,ignore
//! let img = march_implicit_surface_color(&camera, &opts, |p| {
//!     let d = p.length() - 1.5;
//!     let color = [200u8, 100, 50, 255];
//!     (d, color)
//! });
//! ```
//!
//! The returned item has `depth: Some(depths)` and `anchor: TopLeft` with
//! `scale: 1.0`. Adjust `scale` on the returned item if you rendered at a
//! reduced resolution (e.g. `scale = 2.0` for half-resolution rendering that
//! still covers the full viewport).

use crate::camera::camera::{Camera, Projection};
use crate::renderer::{ImageAnchor, ScreenImageItem};
use glam::Vec3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for sphere-marching an implicit surface.
///
/// Resolution, step quality, and appearance can all be tuned here. Reducing
/// `width`/`height` is the most effective way to improve performance — halving
/// both dimensions cuts render time to ~1/4 while still producing a readable
/// result.
#[derive(Clone, Debug)]
pub struct ImplicitRenderOptions {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Maximum number of sphere-march steps per ray.
    ///
    /// Increase for thin or complex surfaces; decrease for performance. Default: 128.
    pub max_steps: u32,
    /// Fraction of the SDF value to advance per step.
    ///
    /// Must be in `(0.0, 1.0]`. Use `< 1.0` for SDFs that are not exact (e.g.
    /// smooth-min blends). Default: `0.9`.
    pub step_scale: f32,
    /// Distance threshold for declaring a surface hit: `|sdf(pos)| < hit_threshold`.
    ///
    /// Smaller values give sharper edges but may need more steps or a smaller
    /// `step_scale`. Default: `5e-4`.
    pub hit_threshold: f32,
    /// Maximum ray travel distance; rays that exceed this without a hit are
    /// treated as background. Default: `1000.0`.
    pub max_distance: f32,
    /// RGBA8 surface color used by [`march_implicit_surface`].
    ///
    /// Ignored by [`march_implicit_surface_color`] (color comes from the closure).
    /// Default: light grey `[200, 200, 200, 255]`.
    pub surface_color: [u8; 4],
    /// RGBA8 background color for pixels that miss the surface. Default: fully
    /// transparent black `[0, 0, 0, 0]`.
    pub background: [u8; 4],
}

impl Default for ImplicitRenderOptions {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            max_steps: 128,
            step_scale: 0.9,
            hit_threshold: 5e-4,
            max_distance: 1000.0,
            surface_color: [200, 200, 200, 255],
            background: [0, 0, 0, 0],
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Sphere-march a signed-distance function and produce a depth-composited
/// [`ScreenImageItem`].
///
/// Each pixel fires a ray from the camera and sphere-marches by calling
/// `sdf`. Hit points are shaded with simple diffuse + ambient lighting derived
/// from the SDF gradient (6 extra SDF evaluations per hit point via central
/// differences).
///
/// The returned item has `depth: Some(depths)`. Background pixels carry depth
/// `1.0` (far plane) so scene geometry is never occluded by them.
///
/// See the module-level documentation for a usage example.
pub fn march_implicit_surface<F>(
    camera: &Camera,
    options: &ImplicitRenderOptions,
    sdf: F,
) -> ScreenImageItem
where
    F: Fn(Vec3) -> f32,
{
    let color = options.surface_color;
    march_impl(camera, options, move |p| (sdf(p), color))
}

/// Sphere-march a colored signed-distance function and produce a depth-composited
/// [`ScreenImageItem`].
///
/// The closure `sdf_color` returns `(sdf_value, rgba8_color)`. The SDF value
/// drives the ray-march; the color is modulated by the same diffuse + ambient
/// shading as [`march_implicit_surface`]. The color closure is also called
/// (6 times per hit point) for normal estimation — only the SDF value is
/// used in those calls.
///
/// The returned item has `depth: Some(depths)`. Background pixels carry depth
/// `1.0` (far plane) so scene geometry is never occluded by them.
///
/// See the module-level documentation for a usage example.
pub fn march_implicit_surface_color<F>(
    camera: &Camera,
    options: &ImplicitRenderOptions,
    sdf_color: F,
) -> ScreenImageItem
where
    F: Fn(Vec3) -> (f32, [u8; 4]),
{
    march_impl(camera, options, sdf_color)
}

// ---------------------------------------------------------------------------
// Core implementation
// ---------------------------------------------------------------------------

fn march_impl<F>(camera: &Camera, options: &ImplicitRenderOptions, sdf_color: F) -> ScreenImageItem
where
    F: Fn(Vec3) -> (f32, [u8; 4]),
{
    let w = options.width.max(1);
    let h = options.height.max(1);

    let eye = camera.eye_position();
    // Look direction: eye -> center.  Fall back to orientation when eye==center.
    let forward = {
        let diff = camera.center - eye;
        if diff.length_squared() > 1e-10 {
            diff.normalize()
        } else {
            -(camera.orientation * Vec3::Z)
        }
    };
    let right = camera.orientation * Vec3::X;
    let up = camera.orientation * Vec3::Y;

    // Perspective: half-extents of the image plane at unit distance.
    let half_h_persp = (camera.fov_y / 2.0).tan();
    let half_w_persp = half_h_persp * camera.aspect;

    // Orthographic: half-extents in world units.
    let orth_half_h = camera.distance * half_h_persp;
    let orth_half_w = camera.distance * half_w_persp;

    let is_ortho = matches!(camera.projection, Projection::Orthographic);

    let znear = camera.znear;
    // Effective far matches Camera::proj_matrix so NDC depths are consistent.
    let effective_zfar = camera.zfar.max(camera.distance * 3.0);

    // Finite-difference step for normal estimation.
    let eps = (options.hit_threshold * 100.0).max(1e-5_f32);

    // Simple diffuse light in world space.
    const LIGHT: Vec3 = Vec3::new(0.577_350_26, 0.577_350_26, 0.577_350_26);
    const AMBIENT: f32 = 0.25_f32;

    let count = (w * h) as usize;
    let mut pixels = vec![[0u8; 4]; count];
    let mut depths = vec![1.0_f32; count];

    for py in 0..h {
        for px in 0..w {
            // NDC: x in [-1, 1] left->right, y in [-1, 1] bottom->top.
            let ndc_x = (px as f32 + 0.5) / w as f32 * 2.0 - 1.0;
            let ndc_y = 1.0 - (py as f32 + 0.5) / h as f32 * 2.0;

            let (ray_o, ray_d): (Vec3, Vec3) = if is_ortho {
                let o = eye + right * (ndc_x * orth_half_w) + up * (ndc_y * orth_half_h);
                (o, forward)
            } else {
                let d = (forward
                    + right * (ndc_x * half_w_persp)
                    + up * (ndc_y * half_h_persp))
                    .normalize();
                (eye, d)
            };

            // Sphere-march.
            let mut t = znear;
            let mut hit = false;
            let mut hit_pos = Vec3::ZERO;
            let mut hit_color = options.surface_color;

            for _ in 0..options.max_steps {
                let pos = ray_o + ray_d * t;
                let (d, color) = sdf_color(pos);
                if d.abs() < options.hit_threshold {
                    hit = true;
                    hit_pos = pos;
                    hit_color = color;
                    break;
                }
                t += d * options.step_scale;
                if t > options.max_distance {
                    break;
                }
            }

            let idx = (py * w + px) as usize;
            if hit {
                // Normal from central differences.
                let nx = sdf_color(hit_pos + Vec3::X * eps).0
                    - sdf_color(hit_pos - Vec3::X * eps).0;
                let ny = sdf_color(hit_pos + Vec3::Y * eps).0
                    - sdf_color(hit_pos - Vec3::Y * eps).0;
                let nz = sdf_color(hit_pos + Vec3::Z * eps).0
                    - sdf_color(hit_pos - Vec3::Z * eps).0;
                let normal = Vec3::new(nx, ny, nz).normalize_or_zero();

                // Diffuse + ambient shading.
                let diffuse = normal.dot(LIGHT).max(0.0);
                let shade = (AMBIENT + (1.0 - AMBIENT) * diffuse).min(1.0);

                pixels[idx] = [
                    (hit_color[0] as f32 * shade) as u8,
                    (hit_color[1] as f32 * shade) as u8,
                    (hit_color[2] as f32 * shade) as u8,
                    hit_color[3],
                ];

                // NDC depth (wgpu: 0 = near plane, 1 = far plane).
                // Formula from Phase 12 showcase: zfar*(d-znear)/(d*(zfar-znear))
                // where d is the positive view-space depth.
                let view_depth = (hit_pos - eye).dot(forward);
                depths[idx] = if view_depth > znear {
                    (effective_zfar * (view_depth - znear)
                        / (view_depth * (effective_zfar - znear)))
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                };
            } else {
                pixels[idx] = options.background;
                // Far-plane depth: never occludes scene geometry.
                depths[idx] = 1.0;
            }
        }
    }

    ScreenImageItem {
        pixels,
        width: w,
        height: h,
        anchor: ImageAnchor::TopLeft,
        scale: 1.0,
        alpha: 1.0,
        depth: Some(depths),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Camera;

    fn default_cam() -> Camera {
        Camera {
            center: glam::Vec3::ZERO,
            distance: 6.0,
            orientation: glam::Quat::IDENTITY,
            fov_y: std::f32::consts::FRAC_PI_4,
            aspect: 1.0,
            znear: 0.1,
            zfar: 100.0,
            ..Camera::default()
        }
    }

    #[test]
    fn march_sphere_hits_center() {
        let cam = default_cam();
        let opts = ImplicitRenderOptions {
            width: 64,
            height: 64,
            max_steps: 256,
            hit_threshold: 1e-4,
            max_distance: 200.0,
            surface_color: [255, 0, 0, 255],
            ..Default::default()
        };
        // Unit sphere at origin — camera is at z=6, looking at origin.
        let img = march_implicit_surface(&cam, &opts, |p| p.length() - 1.0);

        assert_eq!(img.pixels.len(), 64 * 64);
        assert_eq!(img.depth.as_ref().map(|d| d.len()), Some(64 * 64));

        // Centre pixel should have hit the sphere (alpha > 0).
        let cx = 32usize;
        let cy = 32usize;
        let center_px = img.pixels[cy * 64 + cx];
        assert!(
            center_px[3] == 255,
            "centre pixel should have alpha=255 (sphere hit), got {:?}",
            center_px
        );
    }

    #[test]
    fn march_sphere_depth_in_range() {
        let cam = default_cam();
        let opts = ImplicitRenderOptions {
            width: 32,
            height: 32,
            max_steps: 256,
            hit_threshold: 1e-4,
            max_distance: 200.0,
            ..Default::default()
        };
        let img = march_implicit_surface(&cam, &opts, |p| p.length() - 1.0);

        let depths = img.depth.as_ref().unwrap();
        let cx = 16usize;
        let cy = 16usize;
        let d = depths[cy * 32 + cx];
        assert!(d > 0.0 && d < 1.0, "centre depth should be in (0,1), got {d}");
    }

    #[test]
    fn march_miss_returns_background() {
        let cam = default_cam();
        let opts = ImplicitRenderOptions {
            width: 8,
            height: 8,
            max_steps: 64,
            max_distance: 0.01, // effectively no march — all rays miss
            background: [0, 0, 0, 0],
            ..Default::default()
        };
        let img = march_implicit_surface(&cam, &opts, |p| p.length() - 1.0);

        for (i, px) in img.pixels.iter().enumerate() {
            assert_eq!(
                *px,
                [0, 0, 0, 0],
                "pixel {i} should be background colour"
            );
        }
        let depths = img.depth.as_ref().unwrap();
        for d in depths.iter() {
            assert!(
                (d - 1.0).abs() < 1e-6,
                "missed pixels should have far-plane depth (1.0), got {d}"
            );
        }
    }

    #[test]
    fn march_color_closure_applies_color() {
        let cam = default_cam();
        let opts = ImplicitRenderOptions {
            width: 32,
            height: 32,
            max_steps: 256,
            hit_threshold: 1e-4,
            max_distance: 200.0,
            ..Default::default()
        };
        let target_alpha = 200u8;
        let img = march_implicit_surface_color(&cam, &opts, |p| {
            (p.length() - 1.0, [0, 255, 0, target_alpha])
        });

        // Centre pixel: alpha must match what the closure returns.
        let cx = 16usize;
        let cy = 16usize;
        let px = img.pixels[cy * 32 + cx];
        assert_eq!(px[3], target_alpha, "alpha should pass through unchanged");
        // Green channel should be non-zero (diffuse shading dims it, but it started at 255).
        assert!(px[1] > 0, "green channel should survive shading");
        // Red and blue should be zero (closure returns 0 for both).
        assert_eq!(px[0], 0, "red should be 0");
        assert_eq!(px[2], 0, "blue should be 0");
    }

    #[test]
    fn output_dimensions_match_options() {
        let cam = default_cam();
        let opts = ImplicitRenderOptions {
            width: 17,
            height: 11,
            ..Default::default()
        };
        let img = march_implicit_surface(&cam, &opts, |p| p.length() - 1.0);
        assert_eq!(img.width, 17);
        assert_eq!(img.height, 11);
        assert_eq!(img.pixels.len(), 17 * 11);
        assert_eq!(img.depth.as_ref().unwrap().len(), 17 * 11);
    }

    #[test]
    fn orthographic_camera_hits_sphere() {
        let mut cam = default_cam();
        cam.projection = Projection::Orthographic;
        let opts = ImplicitRenderOptions {
            width: 32,
            height: 32,
            max_steps: 256,
            hit_threshold: 1e-4,
            max_distance: 200.0,
            surface_color: [255, 255, 255, 255],
            ..Default::default()
        };
        let img = march_implicit_surface(&cam, &opts, |p| p.length() - 1.0);
        let cx = 16usize;
        let cy = 16usize;
        assert_eq!(
            img.pixels[cy * 32 + cx][3],
            255,
            "orthographic centre pixel should hit the sphere"
        );
    }
}
