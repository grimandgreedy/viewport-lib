// ---------------------------------------------------------------------------
// Section view / clip plane / clip volume
// ---------------------------------------------------------------------------

/// A world-space half-space clipping plane for section views.
///
/// The shape of a [`ClipObject`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClipShape {
    /// Half-space plane : fragments where `dot(p, normal) + distance >= 0` are kept.
    Plane {
        /// Unit normal pointing into the preserved half-space.
        normal: [f32; 3],
        /// Signed distance from origin along `normal`.
        distance: f32,
        /// Cap fill color override. `None` = use the clipped mesh's base_color.
        cap_color: Option<[f32; 4]>,
        /// World-space point on the plane used to position the overlay quad.
        ///
        /// When `None`, the renderer falls back to `normal * (-distance)`, which is
        /// the closest point on the plane to the world origin.  Set this to the
        /// user-facing origin (e.g. a drag handle position) so the overlay stays
        /// centred under the gizmo when the plane is translated laterally.
        display_center: Option<[f32; 3]>,
    },
    /// Oriented box : fragments inside the box are kept.
    Box {
        /// World-space center of the box.
        center: [f32; 3],
        /// Half-extents along each local axis.
        half_extents: [f32; 3],
        /// 3×3 rotation matrix columns.
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
/// Set `color` to `Some(rgba)` to have the renderer draw the clip boundary automatically.
/// For planes this produces a semi-transparent fill quad + border; for box/sphere, a
/// wireframe outline. Leave `color` as `None` for silent clipping with no visual.
///
/// The `hovered` and `active` flags are written by `ClipPlaneController` and read by
/// the renderer to vary the plane overlay appearance (brighter when hovered, tinted when active).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipObject {
    /// The clipping shape (plane, box, or sphere).
    pub shape: ClipShape,
    /// RGBA fill color for the plane quad. `None` = no fill drawn.
    ///
    /// When both `color` and `edge_color` are `None`, no visual is drawn at all.
    pub color: Option<[f32; 4]>,
    /// RGBA color for the plane border edge. `None` = derive from `color` (original behaviour).
    ///
    /// Set independently to show a visible edge while keeping the fill transparent.
    pub edge_color: Option<[f32; 4]>,
    /// Whether this object clips rendered geometry via the GPU clip-plane uniform.
    ///
    /// Set to `false` to produce only a visual indicator without affecting geometry.
    /// Default: `true`.
    pub clip_geometry: bool,
    /// Whether this object is active. Disabled objects are ignored entirely.
    pub enabled: bool,
    /// Visual and hit-test half-extent for `Plane` shapes (world units). Default `4.5`.
    pub extent: f32,
    /// Hover state : set by `ClipPlaneController`, read by renderer.
    pub hovered: bool,
    /// Active drag state : set by `ClipPlaneController`, read by renderer.
    pub active: bool,
}

impl Default for ClipObject {
    fn default() -> Self {
        Self {
            shape: ClipShape::Plane {
                normal: [0.0, 0.0, 1.0],
                distance: 0.0,
                cap_color: None,
                display_center: None,
            },
            color: None,
            edge_color: None,
            clip_geometry: true,
            enabled: true,
            extent: 4.5,
            hovered: false,
            active: false,
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
                cap_color: None,
                display_center: None,
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
