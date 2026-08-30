// ---------------------------------------------------------------------------
// Section view / clip plane / clip volume
// ---------------------------------------------------------------------------

/// A world-space half-space clipping plane for section views.
///
/// The shape of a [`ClipObject`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ClipShape {
    /// Half-space plane : fragments where `dot(p, normal) + distance >= 0` are kept.
    Plane {
        /// Unit normal pointing into the preserved half-space.
        normal: [f32; 3],
        /// Signed distance from origin along `normal`.
        distance: f32,
        /// Cap fill colour override. `None` = use the clipped mesh's base_colour.
        cap_colour: Option<[f32; 4]>,
    },
    /// Oriented box : fragments inside the box are kept.
    Box {
        /// World-space center of the box.
        center: [f32; 3],
        /// Half-extents along each local axis.
        half_extents: [f32; 3],
        /// 3x3 rotation matrix columns.
        orientation: [[f32; 3]; 3],
    },
    /// Sphere : fragments inside the sphere are kept.
    Sphere {
        /// World-space center of the sphere.
        center: [f32; 3],
        /// Radius of the sphere.
        radius: f32,
    },
    /// Cylinder : fragments inside the cylinder are kept.
    Cylinder {
        /// World-space center (midpoint of the axis segment).
        center: [f32; 3],
        /// Unit axis direction.
        axis: [f32; 3],
        /// Radius.
        radius: f32,
        /// Half-length along the axis (total length = 2 * half_length).
        half_length: f32,
    },
}

/// A clip object : defines a clipping region and optional visual boundary rendering.
///
/// Push into `EffectsFrame::clip_objects` each frame. Up to 6 `Plane` variants and
/// up to 4 `Box`, `Sphere`, or `Cylinder` variants are supported simultaneously; all
/// active clip objects apply cumulatively (AND semantics). Entries beyond the limit
/// are silently ignored.
///
/// The renderer does not draw a boundary for a clip object; it only performs the
/// clip operation (and the section cap fill). To show where a clip sits, build the
/// visual with [`clip_plane::visual`](crate::clip_plane::visual) - an outline
/// [`PolylineItem`](crate::PolylineItem) for any shape, plus a translucent fill mesh
/// for the plane - and submit it as an ordinary scene primitive with
/// `ItemSettings::ignore_clip = true` so it stays visible through active clips.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClipObject {
    /// The clipping shape (plane, box, sphere, or cylinder).
    pub shape: ClipShape,
    /// Whether this object clips rendered geometry via the GPU clip-plane uniform.
    ///
    /// Set to `false` to produce only a visual indicator without affecting geometry.
    /// Default: `true`.
    pub clip_geometry: bool,
    /// Whether this object is active. Disabled objects are ignored entirely.
    pub enabled: bool,
}

impl Default for ClipObject {
    fn default() -> Self {
        Self {
            shape: ClipShape::Plane {
                normal: [0.0, 0.0, 1.0],
                distance: 0.0,
                cap_colour: None,
            },
            clip_geometry: true,
            enabled: true,
        }
    }
}

impl ClipObject {
    /// Create a half-space plane clip object.
    pub fn plane(normal: [f32; 3], distance: f32) -> Self {
        Self {
            shape: ClipShape::Plane {
                normal,
                distance,
                cap_colour: None,
            },
            ..Default::default()
        }
    }
    /// Create an oriented box clip object.
    pub fn box_shape(center: [f32; 3], half_extents: [f32; 3], orientation: [[f32; 3]; 3]) -> Self {
        Self {
            shape: ClipShape::Box {
                center,
                half_extents,
                orientation,
            },
            ..Default::default()
        }
    }
    /// Create a sphere clip object.
    pub fn sphere(center: [f32; 3], radius: f32) -> Self {
        Self {
            shape: ClipShape::Sphere { center, radius },
            ..Default::default()
        }
    }

    /// Create a cylinder clip object.
    ///
    /// `axis` must be a unit vector.
    pub fn cylinder(center: [f32; 3], axis: [f32; 3], radius: f32, half_length: f32) -> Self {
        Self {
            shape: ClipShape::Cylinder {
                center,
                axis,
                radius,
                half_length,
            },
            ..Default::default()
        }
    }
}
