//! The tensor field item type: one instanced mesh per sample, shaped by the
//! sample's tensor, plus the handle and reference form for a field uploaded
//! once.

use super::eigen::{SymmetricEigen, symmetric_eigen_3x3};
use viewport_lib::ItemSettings;
use viewport_lib::{ColourSource, MeshId, SizeSource};

viewport_lib::resources::handle::slot_handle! {
    /// Handle to a tensor field uploaded once through
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload).
    ///
    /// Name it from a [`TensorFieldRefItem`] to draw the stored field without
    /// rebuilding its instance buffer.
    pub struct TensorFieldId;
}

const IDENTITY_MAT4: [[f32; 4]; 4] = glam::Mat4::IDENTITY.to_cols_array_2d();

/// Per-frame reference to a pre-uploaded tensor field.
///
/// **Picking is GPU only.** The samples live on the GPU and the plugin keeps no
/// CPU copy of them, so a reference answers `PickBackend::Gpu` and is invisible
/// to the CPU ray and rect pickers. An inline [`TensorFieldItem`] answers both.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TensorFieldRefItem {
    /// Handle to GPU buffers produced by
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload)
    /// or `Uploads::begin_upload`.
    pub source: TensorFieldId,
    /// Per-frame model matrix. Composes on top of the per-instance
    /// transforms baked at upload time.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl TensorFieldRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: TensorFieldId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// The tensors themselves, in whichever form the caller has them.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TensorSource {
    /// The six independent components of a symmetric tensor per sample,
    /// `[xx, yy, zz, xy, xz, yz]`: what a solver writes, what the common file
    /// formats store, and what a caller normally has. The item decomposes them.
    Components(Vec<[f32; 6]>),

    /// A decomposition the caller did themselves, for a particular convention
    /// around repeated eigenvalues or eigenvector signs.
    ///
    /// `vectors[i]` holds the three eigenvectors as column vectors, matching
    /// `values[i]` in order. They should be orthonormal; nothing checks.
    Eigen {
        /// Three eigenvalues per sample.
        values: Vec<[f32; 3]>,
        /// Three unit eigenvectors per sample, in the same order.
        vectors: Vec<[[f32; 3]; 3]>,
    },
}

impl Default for TensorSource {
    fn default() -> Self {
        Self::Components(Vec::new())
    }
}

impl TensorSource {
    /// Number of tensors the source carries.
    pub fn len(&self) -> usize {
        match self {
            Self::Components(c) => c.len(),
            Self::Eigen { values, vectors } => values.len().min(vectors.len()),
        }
    }

    /// True when the source carries no tensors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The decomposition of tensor `i`, solving for it when the source holds
    /// raw components.
    ///
    /// Out of range gives the identity: three unit eigenvalues on the standard
    /// basis, so a short source draws unit spheres rather than nothing.
    pub fn eigen_at(&self, i: usize) -> SymmetricEigen {
        match self {
            Self::Components(c) => c
                .get(i)
                .map(|t| symmetric_eigen_3x3(*t))
                .unwrap_or(IDENTITY_EIGEN),
            Self::Eigen { values, vectors } => match (values.get(i), vectors.get(i)) {
                (Some(v), Some(e)) => SymmetricEigen {
                    values: *v,
                    vectors: *e,
                },
                _ => IDENTITY_EIGEN,
            },
        }
    }
}

const IDENTITY_EIGEN: SymmetricEigen = SymmetricEigen {
    values: [1.0; 3],
    vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
};

/// A sampled tensor field, drawn as one anisotropically scaled mesh per sample.
///
/// Each instance is [`shape`](Self::shape) at `positions[i]`, with its local
/// axes rotated onto the tensor's eigenvectors and scaled by the absolute
/// eigenvalues times [`scale`](Self::scale). With a unit sphere for a shape that
/// is the usual stress ellipsoid.
///
/// The field's **natural scalar is the dominant (first, signed) eigenvalue**, so
/// the default [`ColourSource::Natural`] separates tension from compression once
/// the range spans zero.
///
/// ```no_run
/// # use viewport_lib_plugins::item_types::tensor_field::{TensorFieldItem, TensorSource};
/// # let sphere: viewport_lib::MeshId = unimplemented!();
/// # let (positions, components) = (Vec::new(), Vec::new());
/// let mut field = TensorFieldItem::new(sphere);
/// field.positions = positions;
/// // [xx, yy, zz, xy, xz, yz] per sample, straight out of the solver.
/// field.tensors = TensorSource::Components(components);
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct TensorFieldItem {
    /// Sample positions, one per instance.
    pub positions: Vec<[f32; 3]>,
    /// The tensor at each position.
    pub tensors: TensorSource,

    /// The mesh drawn at every sample, scaled along its local axes by the
    /// eigenvalues: [`primitives::icosphere`](viewport_lib::primitives::icosphere)
    /// is the usual one.
    ///
    /// A field whose shape is unset or stale draws nothing.
    pub shape: MeshId,
    /// Multiplier on the eigenvalue-derived extent, for scaling a whole field
    /// without touching its data. Default: 1.0.
    pub scale: f32,

    /// How each sample is coloured. Default: by the dominant eigenvalue,
    /// auto-ranged, through the item's colourmap. A diverging colourmap reads
    /// best when the range spans zero.
    ///
    /// Samples past the end of a short
    /// [`PerSample`](ColourSource::PerSample) list are drawn opaque white.
    pub colour: ColourSource,
    /// A further per-sample multiplier on the extent, on top of the
    /// eigenvalues. Default: one, which leaves the tensor's own magnitudes to
    /// set the size.
    ///
    /// [`SizeSource::Natural`] reads the dominant eigenvalue here, which the
    /// glyph is already scaled by, so it compounds rather than replaces it.
    pub size: SizeSource,

    /// World-space model matrix, composed on top of the per-instance transform.
    /// Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity,
    /// selection state).
    pub settings: ItemSettings,
}

impl TensorFieldItem {
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

impl Default for TensorFieldItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            tensors: TensorSource::default(),
            shape: MeshId::INVALID,
            scale: 1.0,
            colour: ColourSource::default(),
            size: SizeSource::Uniform(1.0),
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}
