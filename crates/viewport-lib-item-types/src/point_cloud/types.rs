use viewport_lib::{Colour, ColourSource, ItemSettings, SizeSource};

const IDENTITY_MAT4: [[f32; 4]; 4] = glam::Mat4::IDENTITY.to_cols_array_2d();

viewport_lib::resources::handle::slot_handle! {
    /// Handle to a point cloud uploaded once through
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload).
    ///
    /// Name it from a [`PointCloudRefItem`] to draw the stored cloud without
    /// resubmitting its points. Carries the slot index plus the generation the
    /// slot had when the handle was issued, so a handle whose cloud was
    /// dropped resolves to nothing rather than aliasing its successor.
    pub struct PointCloudId;
}

/// Render mode for point cloud items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointRenderMode {
    /// Flat disc: billboard quad clipped to a circle. Fast, no shading.
    #[default]
    ScreenSpaceCircle,
    /// Shaded sphere: billboard quad with hemisphere normal shading (ambient + diffuse + specular).
    /// Points look like small lit spheres without actual geometry cost.
    Sphere,
}

/// A point cloud item to render in the viewport.
///
/// Sizes are in **pixels**: a point is a screen-space billboard, so its radius
/// is a screen measurement rather than a world one.
///
/// A point cloud has **no natural scalar**, so
/// [`ColourSource::Natural`](viewport_lib::ColourSource::Natural) colours every
/// point opaque white and [`SizeSource::Natural`](viewport_lib::SizeSource)
/// gives every point the bottom of its output range. Name what you want
/// instead.
///
/// ```no_run
/// # use viewport_lib_item_types::PointCloudItem;
/// # use viewport_lib::{ColourSource, SizeSource};
/// # let (positions, temperatures) = (Vec::new(), Vec::new());
/// let mut cloud = PointCloudItem::default();
/// cloud.positions = positions;
/// cloud.colour = ColourSource::Scalar {
///     values: temperatures,
///     range: None,
///     colourmap: None,
/// };
/// cloud.size = SizeSource::Uniform(6.0);
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct PointCloudItem {
    /// World-space positions (one vec3 per point).
    pub positions: Vec<[f32; 3]>,

    /// How each point is coloured. Default: opaque white for every point.
    ///
    /// Points past the end of a short
    /// [`PerSample`](viewport_lib::ColourSource::PerSample) list are drawn
    /// opaque white.
    pub colour: ColourSource,
    /// Each point's radius, in pixels. Default: four pixels for every point.
    pub size: SizeSource,

    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Render mode. Default: ScreenSpaceCircle.
    pub render_mode: PointRenderMode,
    /// Optional per-point opacity values in `[0, 1]`. If non-empty, scales each point's alpha.
    pub transparencies: Vec<f32>,
    /// When true, each point is rendered as a soft Gaussian splat instead of a flat circle.
    /// The alpha falls off as `exp(-3 * d^2)` where `d` is the normalised distance from the
    /// point centre. Default: false.
    pub gaussian: bool,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for PointCloudItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            colour: ColourSource::Solid(Colour::WHITE),
            size: SizeSource::Uniform(4.0),
            model: IDENTITY_MAT4,
            render_mode: PointRenderMode::ScreenSpaceCircle,
            transparencies: Vec::new(),
            gaussian: false,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded point cloud.
///
/// **Picking is GPU only.** The points live on the GPU and the plugin keeps no
/// CPU copy of them, so a reference answers `PickBackend::Gpu` and is invisible
/// to the CPU ray and rect pickers. An inline [`PointCloudItem`] answers both.
/// Submit the inline form if you need CPU picking, or keep your own positions
/// and test against them.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PointCloudRefItem {
    /// Handle to GPU buffers produced by
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload)
    /// or `Uploads::begin_upload`.
    pub source: PointCloudId,
    /// Per-frame model matrix. Composes on top of the model baked into the
    /// upload, so identity here renders the points at their original
    /// transform.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl PointCloudRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: PointCloudId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

impl PointCloudItem {
    /// Collect the point primitives a [`DebugDraw`](viewport_lib::runtime::DebugDraw)
    /// has accumulated into one cloud.
    ///
    /// `None` when there are no point primitives, or when the buffer is
    /// disabled. Dev-layer points are skipped unless `dev_enabled` is set, the
    /// same filtering the polyline and label conversions apply.
    pub fn from_debug_draw(dd: &viewport_lib::runtime::DebugDraw) -> Option<Self> {
        use viewport_lib::runtime::{DebugLayer, DebugPrim};

        if !dd.enabled {
            return None;
        }
        let mut positions = Vec::new();
        let mut colours = Vec::new();
        let mut radii = Vec::new();
        for prim in dd.prims() {
            if prim.layer() == DebugLayer::Dev && !dd.dev_enabled {
                continue;
            }
            if let DebugPrim::Point {
                position,
                radius,
                colour,
                ..
            } = prim
            {
                positions.push((*position).into());
                colours.push(*colour);
                radii.push(*radius);
            }
        }
        if positions.is_empty() {
            return None;
        }
        Some(Self {
            positions,
            colour: ColourSource::PerSample(colours),
            size: SizeSource::PerSample(radii),
            ..Self::default()
        })
    }
}
