//! A scene drawn at a reference resolution and scaled into the viewport.
//!
//! The scene is 100 x 60 reference pixels under an orthographic `Camera2d`,
//! one world unit to a reference pixel: a plate, three squares and a tilted
//! bar whose stepped edges show the nearest upscale as whole blocks. At the
//! golden's 400 x 300 it draws at four times, centred, with purple bars above
//! and below. Two screen overlays draw over the upscaled result at the
//! viewport's own resolution: a translucent disc hung from the middle square's
//! world position and a rounded bar in the top-right corner, both with smooth
//! edges where the scene's are blocky.

use glam::{Mat4, Quat, Vec2, Vec3};
use viewport_lib::{
    Alignment, AnchorX, AnchorY, Camera2d, Colour, Material, OverlayAnchoring, OverlayFill,
    OverlayShape, OverlayShapeItem, Plane2d, Presentation, ReferenceFit, SceneRenderItem,
    primitives,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, material, rigs};

/// The reference size the scene draws at.
pub const REFERENCE_SIZE: [u32; 2] = [100, 60];

/// The colour of the bars around the scaled scene.
pub const BAR_COLOUR: Colour = Colour::linear(0.12, 0.03, 0.16, 1.0);

/// The presentation scenes appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "presentation_reference",
        cameras: vec![NamedCamera {
            name: "top",
            camera: camera(),
        }],
        build: build_presentation_reference,
    }]
}

/// The orthographic camera over the reference target: one world unit to a
/// reference pixel.
pub fn camera() -> viewport_lib::Camera {
    Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(
            REFERENCE_SIZE[0] as f32,
            REFERENCE_SIZE[1] as f32,
        ))
        .camera()
}

/// An unlit quad `size` across, centred on `centre` at height `z` and turned
/// `angle` radians about +Z.
fn quad(
    ctx: &mut BuildCtx<'_>,
    centre: Vec2,
    size: Vec2,
    angle: f32,
    z: f32,
    colour: Colour,
) -> SceneRenderItem {
    let mesh = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::plane(size.x, size.y))
        .expect("quad upload");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = Mat4::from_rotation_translation(
        Quat::from_rotation_z(angle),
        Vec3::new(centre.x, centre.y, z),
    )
    .into();
    item.material = material(ctx, Material::from_colour(colour));
    item.settings.unlit = true;
    item
}

fn build_presentation_reference(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let items = vec![
        quad(
            ctx,
            Vec2::ZERO,
            Vec2::new(100.0, 60.0),
            0.0,
            -1.0,
            Colour::srgb_rgb(0.12, 0.3, 0.32),
        ),
        quad(
            ctx,
            Vec2::new(-30.0, 12.0),
            Vec2::splat(12.0),
            0.0,
            0.0,
            Colour::srgb_rgb(0.9, 0.25, 0.2),
        ),
        quad(
            ctx,
            Vec2::new(0.0, 12.0),
            Vec2::splat(12.0),
            0.0,
            0.0,
            Colour::srgb_rgb(0.95, 0.8, 0.2),
        ),
        quad(
            ctx,
            Vec2::new(30.0, 12.0),
            Vec2::splat(12.0),
            0.0,
            0.0,
            Colour::srgb_rgb(0.25, 0.45, 0.95),
        ),
        quad(
            ctx,
            Vec2::new(0.0, -14.0),
            Vec2::new(70.0, 5.0),
            0.25,
            0.0,
            Colour::srgb_rgb(0.4, 0.9, 0.35),
        ),
    ];

    let mut overlays = viewport_lib::OverlayFrame::default();
    let mut disc = OverlayShapeItem::new(OverlayShape::Circle, [0.0, 0.0], [44.0, 44.0])
        .with_fill(OverlayFill::Solid(Colour::linear(1.0, 1.0, 1.0, 0.45)));
    disc.anchoring = OverlayAnchoring::world([0.0f32, 12.0, 0.0]).with_align(Alignment::CENTRE);
    overlays.shapes.push(disc);
    let mut bar = OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 7.0 },
        [-10.0, 10.0],
        [90.0, 14.0],
    )
    .with_fill(OverlayFill::Solid(Colour::srgb(0.95, 0.95, 0.98, 0.9)));
    bar.anchoring = OverlayAnchoring::viewport(AnchorX::Right, AnchorY::Top)
        .with_align(Alignment::new(AnchorX::Right, AnchorY::Top));
    overlays.shapes.push(bar);

    BuiltScene {
        items,
        overlays,
        lighting: rigs::from_above(),
        presentation: Some(Presentation::reference(
            REFERENCE_SIZE,
            ReferenceFit::IntegerBars { colour: BAR_COLOUR },
        )),
        ..Default::default()
    }
}
