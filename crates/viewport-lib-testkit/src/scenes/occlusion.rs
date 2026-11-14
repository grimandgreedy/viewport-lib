//! Screen-space occlusion scenes: a unit box floating just above a grey
//! plane under a high sun and the hemisphere fill, with ambient occlusion on
//! in one and contact shadows in the other. The sun leans toward the camera's
//! side, so the plane it shades beside the box is in view.
//!
//! Ambient occlusion darkens the fill-lit plane in the box's shadow; contact
//! shadows, with the sun's shadow map off, darken the sunlit plane under the
//! box. The sunlit plane away from the box is untouched by either.

use glam::{Mat4, Vec3};
use viewport_lib::{
    Colour, LightKind, LightSource, LightingSettings, Material, PostProcessSettings, primitives,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, item_model, orbit_camera, upload};

/// The plane, and the box with its underside 0.15 above it.
fn set(ctx: &mut BuildCtx<'_>) -> Vec<viewport_lib::MeshItem> {
    let plane = upload(ctx, &primitives::plane(12.0, 12.0));
    let cube = upload(ctx, &primitives::cube(1.0));
    let grey = Material::pbr(Colour::linear_rgb(0.6, 0.6, 0.6), 0.0, 0.8);
    vec![
        item_model(ctx, plane, Mat4::IDENTITY, grey),
        item_model(
            ctx,
            cube,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.65)),
            grey,
        ),
    ]
}

/// A high sun over the default hemisphere fill.
fn lighting(sun_shadows: bool) -> LightingSettings {
    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [-0.5, 0.6, 1.0],
    };
    sun.cast_shadows = sun_shadows;
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![sun];
    lighting
}

fn camera() -> Vec<NamedCamera> {
    let mut camera = orbit_camera(Vec3::ZERO, 6.0, 0.4, 0.45);
    camera.aspect = 4.0 / 3.0;
    vec![NamedCamera {
        name: "high",
        camera,
    }]
}

fn build_ssao(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut post = PostProcessSettings::default();
    post.ssao = true;
    BuiltScene {
        items: set(ctx),
        lighting: lighting(true),
        post_process: Some(post),
        ..Default::default()
    }
}

fn build_contact(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut post = PostProcessSettings::default();
    post.contact_shadows.enabled = true;
    BuiltScene {
        items: set(ctx),
        lighting: lighting(false),
        post_process: Some(post),
        ..Default::default()
    }
}

/// The ambient occlusion and contact shadow scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "occlusion_ssao",
            cameras: camera(),
            build: build_ssao,
        },
        NamedScene {
            name: "occlusion_contact",
            cameras: camera(),
            build: build_contact,
        },
    ]
}
