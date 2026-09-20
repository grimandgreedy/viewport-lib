//! Retained overlay geometry: a handle to a group of overlay items compiled
//! once into GPU buffers, plus the per-frame submission that draws it.

use crate::overlay::OverlayClip;

crate::slot_handle! {
    /// Handle to a compiled overlay-geometry group.
    ///
    /// A consumer builds a group of overlay items (polylines and vector shapes
    /// today; more families later) once with `ViewportRenderer::compile_overlay_geometry`,
    /// keeps the returned id, and each frame submits a [`RetainedOverlay`] that
    /// references it instead of re-tessellating the items. The renderer tessellates
    /// and uploads the group once and re-draws it from the cached buffer, applying
    /// the per-frame translate, opacity, and clip. Release it with
    /// `free_overlay_geometry`.
    ///
    /// Carries the slot index plus the generation the slot had when the handle was
    /// issued; a freed handle resolves to nothing rather than aliasing a group
    /// compiled later into the reused slot.
    pub struct OverlayGeometryId;
}

/// A retained overlay group submitted for one frame.
///
/// References a group compiled with `compile_overlay_geometry` and carries the
/// cheap per-frame parameters that do not require re-tessellation: a `translate`
/// (so a scroll container just updates the offset), an `opacity` multiplier, a
/// `z_order` for cross-family draw order, and an optional outer clip. The
/// group's own geometry is fixed in the handle.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RetainedOverlay {
    /// The compiled group to draw.
    pub id: OverlayGeometryId,
    /// Translate, rotate, and scale the whole group, in logical pixels and
    /// radians. A scroll container updates `translate` per frame instead of
    /// re-compiling; a pop or a flip rides `scale` and `rotation`.
    ///
    /// This is the group half of the composition contract on
    /// [`OverlayTransform`]: an item inside the group carries its own
    /// transform, and the two compose as `group_T . item_T`.
    ///
    /// Retained glyphs are bitmaps baked at their original size, so a large
    /// sustained `scale` looks soft: re-compile at the target size when crisp
    /// large text is needed.
    ///
    /// [`OverlayTransform`]: crate::overlay::OverlayTransform
    pub transform: crate::overlay::OverlayTransform,
    /// Opacity multiplier in `[0, 1]` applied to the group's alpha. `1.0` leaves
    /// the compiled colours unchanged.
    pub opacity: f32,
    /// Cross-family draw order, low to high, matching the `z_order` on the
    /// immediate overlay items. Default `0`.
    pub z_order: i32,
    /// What this item is clipped to: an axis-aligned box, a mask shape, or
    /// both. The default clips nothing.
    pub clip: OverlayClip,
    /// Per-frame colour multiplier applied to the whole group, identity
    /// `[1, 1, 1, 1]`. A colour flash, fade, or tint rides this instead of
    /// re-compiling the group. On SDF shapes the tint reaches the fill, border, and
    /// gradient colours but not the drop shadow (whose colour is baked).
    pub tint: [f32; 4],
    /// Origin the group hangs from, resolved every frame: a viewport corner
    /// that follows a resize, or a world point projected through the camera.
    /// `transform.translate` is then a nudge from that origin, and a world
    /// anchor behind the camera or off screen culls the group for the frame.
    ///
    /// `None` (the default) leaves the group where its `translate` puts it. A
    /// group compiled from a single `LabelItem` carries the label's own anchor;
    /// setting this overrides it.
    pub anchor: Option<crate::overlay::OverlayAnchor>,
    /// How the group's extent box sits horizontally on the resolved anchor.
    /// `Left` (the default) puts its left edge there. Ignored when `anchor` is
    /// `None`.
    pub align_x: crate::overlay::AnchorX,
    /// How the group's extent box sits vertically on the resolved anchor.
    /// `Top` (the default) puts its top edge there. Ignored when `anchor` is
    /// `None`.
    pub align_y: crate::overlay::AnchorY,
    /// Animation tracks resolved each frame against `OverlayFrame::time`.
    ///
    /// Every channel a track can drive rides the per-draw instance, so an
    /// animated group re-draws from its compiled buffers with no
    /// re-tessellation. Boxed and `None` for a static group.
    pub animations: Option<Box<crate::overlay::OverlayAnimations>>,
}

impl RetainedOverlay {
    /// A submission of `id` at the origin, fully opaque, `z_order` 0,
    /// unclipped, untinted, with an identity transform.
    pub fn new(id: OverlayGeometryId) -> Self {
        Self {
            id,
            transform: crate::overlay::OverlayTransform::IDENTITY,
            opacity: 1.0,
            z_order: 0,
            clip: OverlayClip::default(),
            tint: [1.0, 1.0, 1.0, 1.0],
            anchor: None,
            align_x: crate::overlay::AnchorX::Left,
            align_y: crate::overlay::AnchorY::Top,
            animations: None,
        }
    }

    /// Set the group transform: translate, rotate, scale, and pivot at once.
    pub fn with_transform(mut self, transform: crate::overlay::OverlayTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Set the per-frame translate in logical pixels.
    pub fn with_translate(mut self, translate: [f32; 2]) -> Self {
        self.transform.translate = translate;
        self
    }

    /// Set the per-frame rotation in radians about the group pivot.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.transform.rotation = radians;
        self
    }

    /// Set the centre of rotation and scaling, in logical pixels from the
    /// group's local origin.
    pub fn with_rotation_pivot(mut self, pivot: [f32; 2]) -> Self {
        self.transform.pivot = pivot;
        self
    }

    /// Set the opacity multiplier.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set the cross-family draw order.
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Set the outer clip bounding box in logical pixels.
    pub fn with_clip_rect(mut self, clip_rect: [f32; 4]) -> Self {
        self.clip.rect = Some(clip_rect);
        self
    }

    /// Clip the group to the mask shape whose `provides_mask` matches `clip_id`
    /// (registered by an overlay shape submitted in the same frame), for shaped
    /// clipping such as a rounded-rect scroll viewport.
    pub fn with_clip_mask(mut self, clip_id: u32) -> Self {
        self.clip.mask = Some(clip_id);
        self
    }

    /// Set the per-frame colour multiplier (identity `[1, 1, 1, 1]`).
    pub fn with_tint(mut self, tint: [f32; 4]) -> Self {
        self.tint = tint;
        self
    }

    /// Anchor the group to a viewport corner or a world point, resolved every
    /// frame. `translate` becomes a nudge from the resolved origin.
    pub fn with_anchor(mut self, anchor: crate::overlay::OverlayAnchor) -> Self {
        self.anchor = Some(anchor);
        self
    }

    /// Set how the group's extent box sits on the resolved anchor.
    pub fn with_align(mut self, x: crate::overlay::AnchorX, y: crate::overlay::AnchorY) -> Self {
        self.align_x = x;
        self.align_y = y;
        self
    }

    /// Set the animation tracks. Every channel they drive rides the instance,
    /// so an animated group never re-compiles.
    pub fn with_animations(mut self, animations: crate::overlay::OverlayAnimations) -> Self {
        self.animations = Some(Box::new(animations));
        self
    }

    /// Set the per-frame uniform scale about the group pivot (identity `1.0`).
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.transform.scale = scale;
        self
    }
}
