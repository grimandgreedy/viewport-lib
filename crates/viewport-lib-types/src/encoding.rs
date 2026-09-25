//! How an item maps its per-sample data to colour and size.
//!
//! An item that draws many samples (a point cloud, a vector or tensor field)
//! decides each sample's colour and size the same handful of ways: one value
//! for all of them, one value per sample, or a scalar mapped through a
//! colourmap or a size range. [`ColourSource`] and [`SizeSource`] say which,
//! so an item carries one field per role rather than a set of loosely related
//! ones whose precedence is only documented.
//!
//! ```rust
//! # use viewport_lib_types::encoding::{ColourSource, SizeSource};
//! // Colour by a scalar the caller supplies, over an explicit domain.
//! let colour = ColourSource::Scalar {
//!     values: vec![0.0, 0.5, 1.0],
//!     range: Some((0.0, 1.0)),
//!     colourmap: None,
//! };
//!
//! // Size from the item's own natural scalar, auto-ranged.
//! let size = SizeSource::Natural { domain: None, output: (0.2, 1.0) };
//! ```

use crate::colour::Colour;
use crate::colourmap::ColourmapId;

/// How an item colours its samples.
///
/// The variants are mutually exclusive by construction, which is the point:
/// an item carrying separate per-sample colour and per-sample scalar fields
/// has to define which wins, and a caller cannot see that from the type.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ColourSource {
    /// One colour for every sample.
    Solid(Colour),

    /// One colour per sample, in submission order.
    ///
    /// Shorter than the sample count leaves the remainder at the item's own
    /// fallback; longer is ignored. Items document their fallback.
    PerSample(Vec<Colour>),

    /// A scalar per sample, mapped through a colourmap.
    Scalar {
        /// One value per sample, in submission order.
        values: Vec<f32>,
        /// The scalar domain mapped onto the colourmap. `None` takes it from
        /// `values`.
        range: Option<(f32, f32)>,
        /// `None` is the item's default colourmap.
        colourmap: Option<ColourmapId>,
    },

    /// The scalar the item already holds, mapped through a colourmap.
    ///
    /// Each item documents what its natural scalar is: the vector magnitude
    /// for a vector field, for instance. Nothing is uploaded for this, which
    /// is why it is worth having separately from [`Scalar`](Self::Scalar):
    /// the value is derived from data the item has sent anyway.
    ///
    /// An item with no natural scalar treats this as
    /// [`Solid`](Self::Solid) with its default colour, and says so.
    Natural {
        /// The scalar domain mapped onto the colourmap. `None` takes it from
        /// the data.
        range: Option<(f32, f32)>,
        /// `None` is the item's default colourmap.
        colourmap: Option<ColourmapId>,
    },
}

impl Default for ColourSource {
    /// Colour by the item's natural scalar, auto-ranged, in its default
    /// colourmap: what a scalar-carrying item does when told nothing.
    fn default() -> Self {
        Self::Natural {
            range: None,
            colourmap: None,
        }
    }
}

impl ColourSource {
    /// Per-sample values this source carries, for an item validating lengths
    /// against its sample count. `None` when the source is not per-sample.
    pub fn per_sample_len(&self) -> Option<usize> {
        match self {
            Self::PerSample(colours) => Some(colours.len()),
            Self::Scalar { values, .. } => Some(values.len()),
            Self::Solid(_) | Self::Natural { .. } => None,
        }
    }
}

/// How an item sizes its samples.
///
/// The scalar variants map a domain onto an output range rather than scaling
/// by the raw value: a field whose magnitudes are in the thousands should not
/// produce samples a thousand units across, and the caller should not have to
/// normalise first.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SizeSource {
    /// One size for every sample.
    Uniform(f32),

    /// A scalar per sample, mapped from `domain` onto `output`.
    Scalar {
        /// One value per sample, in submission order.
        values: Vec<f32>,
        /// The input domain. `None` takes it from `values`.
        domain: Option<(f32, f32)>,
        /// The size range the domain maps onto. A sample at the bottom of the
        /// domain takes `output.0`, one at the top takes `output.1`.
        output: (f32, f32),
    },

    /// The scalar the item already holds, mapped the same way. See
    /// [`ColourSource::Natural`] for what "natural" means per item.
    Natural {
        /// The input domain. `None` takes it from the data.
        domain: Option<(f32, f32)>,
        /// The size range the domain maps onto.
        output: (f32, f32),
    },
}

impl Default for SizeSource {
    /// One unit per sample.
    fn default() -> Self {
        Self::Uniform(1.0)
    }
}

impl SizeSource {
    /// Per-sample values this source carries. `None` when the source is not
    /// per-sample.
    pub fn per_sample_len(&self) -> Option<usize> {
        match self {
            Self::Scalar { values, .. } => Some(values.len()),
            Self::Uniform(_) | Self::Natural { .. } => None,
        }
    }

    /// The size range samples land in, for an item sizing its bounds before
    /// it has resolved any scalar.
    pub fn output_range(&self) -> (f32, f32) {
        match self {
            Self::Uniform(size) => (*size, *size),
            Self::Scalar { output, .. } | Self::Natural { output, .. } => *output,
        }
    }
}

/// Map `value` from `domain` onto `output`, clamped to `output` at both ends.
///
/// The shared resolution step behind both scalar sources, so an item type does
/// not reimplement the degenerate cases. An empty domain (`lo == hi`, which is
/// what a constant field auto-ranges to) maps everything to `output.0`, rather
/// than dividing by zero or picking an arbitrary end.
pub fn map_range(value: f32, domain: (f32, f32), output: (f32, f32)) -> f32 {
    let (lo, hi) = domain;
    let span = hi - lo;
    if !span.is_finite() || span.abs() < f32::EPSILON {
        return output.0;
    }
    let t = ((value - lo) / span).clamp(0.0, 1.0);
    output.0 + t * (output.1 - output.0)
}

/// The `(min, max)` of `values`, for a source whose range or domain is `None`.
///
/// `None` when `values` is empty or holds nothing finite, which leaves the
/// item to fall back to whatever it considers sensible rather than producing
/// an infinite domain.
pub fn auto_range(values: &[f32]) -> Option<(f32, f32)> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo <= hi).then_some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_constant_domain_maps_to_the_bottom_of_the_output() {
        // What auto_range gives for a field whose samples are all equal. The
        // old glyph path divided by this span and relied on a branch to avoid
        // it; here there is one answer and it is not a division.
        assert_eq!(map_range(5.0, (5.0, 5.0), (0.2, 1.0)), 0.2);
    }

    #[test]
    fn values_outside_the_domain_clamp_rather_than_extrapolate() {
        assert_eq!(map_range(-10.0, (0.0, 1.0), (0.0, 4.0)), 0.0);
        assert_eq!(map_range(10.0, (0.0, 1.0), (0.0, 4.0)), 4.0);
    }

    #[test]
    fn an_inverted_output_range_is_honoured() {
        // Large values map small: legitimate for a size source, so it is not
        // normalised away.
        assert_eq!(map_range(1.0, (0.0, 1.0), (4.0, 1.0)), 1.0);
        assert_eq!(map_range(0.0, (0.0, 1.0), (4.0, 1.0)), 4.0);
    }

    #[test]
    fn auto_range_skips_non_finite_values() {
        let r = auto_range(&[1.0, f32::NAN, 3.0, f32::INFINITY]);
        assert_eq!(r, Some((1.0, 3.0)));
    }

    #[test]
    fn auto_range_is_none_when_nothing_is_finite() {
        assert_eq!(auto_range(&[]), None);
        assert_eq!(auto_range(&[f32::NAN, f32::NEG_INFINITY]), None);
    }

    #[test]
    fn per_sample_len_reports_only_the_per_sample_variants() {
        assert_eq!(ColourSource::Solid(Colour::WHITE).per_sample_len(), None);
        assert_eq!(ColourSource::default().per_sample_len(), None);
        assert_eq!(
            ColourSource::PerSample(vec![Colour::WHITE; 3]).per_sample_len(),
            Some(3)
        );
        assert_eq!(
            ColourSource::Scalar {
                values: vec![0.0; 7],
                range: None,
                colourmap: None,
            }
            .per_sample_len(),
            Some(7)
        );
    }

    #[test]
    fn a_uniform_size_reports_its_own_value_as_the_range() {
        assert_eq!(SizeSource::Uniform(2.0).output_range(), (2.0, 2.0));
        assert_eq!(
            SizeSource::Natural {
                domain: None,
                output: (0.1, 3.0)
            }
            .output_range(),
            (0.1, 3.0)
        );
    }
}
