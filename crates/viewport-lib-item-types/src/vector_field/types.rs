//! The vector field item type: one instanced mesh per sample, oriented along
//! the sample's vector, plus the handle and reference form for a field
//! uploaded once.

use viewport_lib::ItemSettings;
use viewport_lib::{ColourSource, MeshId, SizeSource};

viewport_lib::resources::handle::slot_handle! {
    /// Handle to a vector field uploaded once through
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload).
    ///
    /// Name it from a [`VectorFieldRefItem`] to draw the stored field without
    /// rebuilding its instance buffer.
    pub struct VectorFieldId;
}

const IDENTITY_MAT4: [[f32; 4]; 4] = glam::Mat4::IDENTITY.to_cols_array_2d();

/// Per-frame reference to a pre-uploaded vector field.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VectorFieldRefItem {
    /// Handle to GPU buffers produced by
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload)
    /// or `Uploads::begin_upload`.
    pub source: VectorFieldId,
    /// Per-frame model matrix. Composes on top of the per-instance transforms
    /// baked at upload time.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl VectorFieldRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: VectorFieldId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// A sampled vector field: one world-space vector at each of a set of positions,
/// drawn as an instanced mesh per sample.
///
/// Each instance is [`shape`](Self::shape) placed at `positions[i]`, rotated so
/// the mesh's local `+Z` points along `vectors[i]`, and scaled by
/// [`size`](Self::size) times [`scale`](Self::scale).
///
/// The field's **natural scalar is the vector magnitude**, so the default
/// [`ColourSource::Natural`] colours by magnitude and the default
/// [`SizeSource::Natural`] sizes by it, neither costing an upload.
///
/// ```no_run
/// # use viewport_lib_item_types::VectorFieldItem;
/// # let arrow_mesh: viewport_lib::MeshId = unimplemented!();
/// # let (positions, vectors) = (Vec::new(), Vec::new());
/// let mut field = VectorFieldItem::new(arrow_mesh);
/// field.positions = positions;
/// field.vectors = vectors;
/// field.scale = 0.3;
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct VectorFieldItem {
    /// Sample positions, one per instance.
    pub positions: Vec<[f32; 3]>,
    /// The world-space vector at each position. A sample with no vector, or a
    /// zero one, is drawn unrotated.
    pub vectors: Vec<[f32; 3]>,

    /// The mesh drawn at every sample, oriented along its local `+Z` and sized
    /// for a unit vector: [`primitives::arrow`](viewport_lib::primitives::arrow)
    /// is the usual one, and any uploaded mesh works.
    ///
    /// A field whose shape is unset or stale draws nothing.
    pub shape: MeshId,
    /// Multiplier on whatever [`size`](Self::size) resolves to, for scaling a
    /// whole field without touching its encoding. Default: 1.0.
    pub scale: f32,

    /// How each sample is coloured. Default: by magnitude, auto-ranged, through
    /// the item's colourmap.
    ///
    /// Samples past the end of a short
    /// [`PerSample`](ColourSource::PerSample) list are drawn opaque white.
    pub colour: ColourSource,
    /// How each sample is sized, before [`scale`](Self::scale).
    ///
    /// The default maps magnitude over `0..1` onto `0.05..1.0`, so the field's
    /// weakest samples stay visible rather than collapsing to nothing.
    pub size: SizeSource,

    /// World-space model matrix, composed on top of the per-instance transform.
    /// Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity,
    /// selection state).
    pub settings: ItemSettings,
}

impl VectorFieldItem {
    /// An empty field drawn with `shape`, at the default encoding.
    pub fn new(shape: MeshId) -> Self {
        Self {
            shape,
            ..Self::default()
        }
    }

    /// Number of samples that will be drawn.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// True when the field has no samples.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

impl Default for VectorFieldItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            vectors: Vec::new(),
            shape: MeshId::INVALID,
            scale: 1.0,
            colour: ColourSource::default(),
            size: SizeSource::Natural {
                domain: Some((0.0, 1.0)),
                output: (0.05, 1.0),
            },
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}
