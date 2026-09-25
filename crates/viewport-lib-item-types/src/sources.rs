//! Resolving [`ColourSource`] and [`SizeSource`] against a set of samples.
//!
//! Both field item types do the same thing with them: turn a source plus the
//! item's own natural scalar into one value per sample, ready to write into an
//! instance record. Only the colourmap lookup is left to the shader, because it
//! needs the LUT texture.

use viewport_lib::Colour;
use viewport_lib::encoding::{ColourSource, SizeSource, auto_range, map_range};

/// Colour for a sample past the end of a short `ColourSource::PerSample` list.
pub(crate) const MISSING_COLOUR: Colour = Colour::WHITE;

/// An explicit domain, or the data's own, or the unit interval when the data
/// offers nothing finite.
pub(crate) fn resolved_domain(domain: Option<(f32, f32)>, values: &[f32]) -> (f32, f32) {
    domain.or_else(|| auto_range(values)).unwrap_or((0.0, 1.0))
}

/// Every sample's resolved size, before any global scale the item applies.
///
/// `natural` is the item's own scalar, one per sample, which the
/// [`SizeSource::Natural`] case reads.
pub(crate) fn sample_sizes(size: &SizeSource, count: usize, natural: &[f32]) -> Vec<f32> {
    match size {
        SizeSource::Uniform(s) => vec![*s; count],
        SizeSource::Scalar {
            values,
            domain,
            output,
        } => {
            let domain = resolved_domain(*domain, values);
            (0..count)
                .map(|i| map_range(values.get(i).copied().unwrap_or(0.0), domain, *output))
                .collect()
        }
        SizeSource::Natural { domain, output } => {
            let domain = resolved_domain(*domain, natural);
            (0..count)
                .map(|i| map_range(natural.get(i).copied().unwrap_or(0.0), domain, *output))
                .collect()
        }
        // A variant added later sizes uniformly until this knows better.
        _ => vec![1.0; count],
    }
}

/// How the shader gets each sample's colour: straight from the instance record,
/// or through the LUT over a resolved range.
pub(crate) struct ColourPlan {
    /// Per-instance RGBA, used when `lut_range` is `None`.
    pub(crate) colours: Vec<[f32; 4]>,
    /// Per-instance scalar, used when `lut_range` is `Some`.
    pub(crate) scalars: Vec<f32>,
    /// The scalar domain the colourmap spans, when the LUT is in play.
    pub(crate) lut_range: Option<(f32, f32)>,
}

/// Resolve a colour source over `count` samples, with `natural` supplying the
/// [`ColourSource::Natural`] case.
pub(crate) fn colour_plan(colour: &ColourSource, count: usize, natural: &[f32]) -> ColourPlan {
    match colour {
        ColourSource::Solid(c) => ColourPlan {
            colours: vec![c.to_linear_rgba(); count],
            scalars: vec![0.0; count],
            lut_range: None,
        },
        ColourSource::PerSample(list) => ColourPlan {
            colours: (0..count)
                .map(|i| {
                    list.get(i)
                        .copied()
                        .unwrap_or(MISSING_COLOUR)
                        .to_linear_rgba()
                })
                .collect(),
            scalars: vec![0.0; count],
            lut_range: None,
        },
        ColourSource::Scalar { values, range, .. } => ColourPlan {
            colours: vec![[1.0; 4]; count],
            scalars: (0..count)
                .map(|i| values.get(i).copied().unwrap_or(0.0))
                .collect(),
            lut_range: Some(resolved_domain(*range, values)),
        },
        ColourSource::Natural { range, .. } => ColourPlan {
            colours: vec![[1.0; 4]; count],
            scalars: (0..count)
                .map(|i| natural.get(i).copied().unwrap_or(0.0))
                .collect(),
            lut_range: Some(resolved_domain(*range, natural)),
        },
        // A variant added later falls back to flat white.
        _ => ColourPlan {
            colours: vec![MISSING_COLOUR.to_linear_rgba(); count],
            scalars: vec![0.0; count],
            lut_range: None,
        },
    }
}

/// The colourmap a source names, when it names one.
pub(crate) fn requested_colourmap(
    colour: &ColourSource,
) -> Option<viewport_lib::resources::ColourmapId> {
    match colour {
        ColourSource::Scalar { colourmap, .. } | ColourSource::Natural { colourmap, .. } => {
            *colourmap
        }
        _ => None,
    }
}
