//! Golden-image comparison: exact by default, with per-scene tolerance overrides.
//!
//! The comparison is exact (a single differing channel fails) unless a scene has
//! a measured tolerance in the manifest. Tolerance exists only to absorb
//! non-deterministic rounding on the same intended image, never to accept an
//! intended change: a real visual change is a re-bless (a new reference), not a
//! wider tolerance. Because a widened tolerance permanently blinds the test to a
//! band of regressions, exceptions are per-scene and carry the reason they were
//! measured (see the manifest file).
//!
//! A reference image lives at `<dir>/<name>.png`. On a missing reference (or with
//! `BLESS` set) the actual frame is written as the new reference. On a mismatch
//! the actual frame and a red diff image are written next to the reference for
//! inspection.

use std::collections::HashMap;
use std::path::Path;

pub use image::{Rgba, RgbaImage};

/// How much a scene's render may differ from its reference and still pass.
///
/// The default is exact: any pixel with any channel off by 1 fails. A non-zero
/// tolerance should only ever be set from a measured non-determinism floor (the
/// double-render probe), never to accept an intended visual change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tolerance {
    /// Largest per-channel absolute difference (0..=255) a pixel may have and
    /// still count as unchanged. `0` is exact.
    pub max_channel_delta: u8,
    /// How many pixels may exceed `max_channel_delta` before the image fails.
    pub max_pixels_over: u64,
}

impl Tolerance {
    /// Bit-exact: no channel may differ, no pixel may be over.
    pub const EXACT: Tolerance = Tolerance {
        max_channel_delta: 0,
        max_pixels_over: 0,
    };
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::EXACT
    }
}

/// The measured difference between a reference and an actual frame, against a
/// given [`Tolerance`].
#[derive(Clone, Copy, Debug)]
pub struct Comparison {
    /// Worst per-channel absolute difference seen anywhere in the image.
    pub max_channel_delta: u8,
    /// Number of pixels with any channel over the tolerance's `max_channel_delta`.
    pub pixels_over: u64,
    /// Total pixels compared.
    pub total_pixels: u64,
}

impl Comparison {
    /// True when the image is within `tol`.
    pub fn passes(&self, tol: Tolerance) -> bool {
        self.pixels_over <= tol.max_pixels_over
    }
}

/// Compare `reference` against `actual`, counting pixels over `tol.max_channel_delta`.
///
/// Panics if the two images differ in size (a size change is a definite failure
/// the caller should have caught).
pub fn compare(reference: &RgbaImage, actual: &RgbaImage, tol: Tolerance) -> Comparison {
    assert_eq!(
        reference.dimensions(),
        actual.dimensions(),
        "reference/actual size mismatch"
    );
    let mut worst = 0u8;
    let mut over = 0u64;
    let total = u64::from(reference.width()) * u64::from(reference.height());
    for (r, a) in reference.pixels().zip(actual.pixels()) {
        let mut pixel_over = false;
        for c in 0..4 {
            let d = (i16::from(r.0[c]) - i16::from(a.0[c])).unsigned_abs() as u8;
            worst = worst.max(d);
            if d > tol.max_channel_delta {
                pixel_over = true;
            }
        }
        if pixel_over {
            over += 1;
        }
    }
    Comparison {
        max_channel_delta: worst,
        pixels_over: over,
        total_pixels: total,
    }
}

/// A red-over-grey diff: red where a pixel is over `tol.max_channel_delta`, a
/// dimmed greyscale of the reference elsewhere.
pub fn diff_image(reference: &RgbaImage, actual: &RgbaImage, tol: Tolerance) -> RgbaImage {
    let mut out = RgbaImage::new(reference.width(), reference.height());
    for (x, y, p) in out.enumerate_pixels_mut() {
        let r = reference.get_pixel(x, y);
        let a = actual.get_pixel(x, y);
        let over = (0..4).any(|c| {
            (i16::from(r.0[c]) - i16::from(a.0[c])).unsigned_abs() as u8 > tol.max_channel_delta
        });
        *p = if over {
            Rgba([255, 0, 0, 255])
        } else {
            let g = (u32::from(r.0[0]) + u32::from(r.0[1]) + u32::from(r.0[2])) / 6;
            Rgba([g as u8, g as u8, g as u8, 255])
        };
    }
    out
}

/// The result of checking one scene's frame against its reference.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// The reference was written (missing reference, or `bless`).
    Blessed,
    /// The actual frame matched the reference within tolerance.
    Match,
    /// The actual frame differed beyond tolerance. Actual and diff images were
    /// written next to the reference.
    Mismatch {
        /// Observed difference.
        comparison: Comparison,
        /// Tolerance it was checked against.
        tolerance: Tolerance,
    },
}

/// Check `actual` against the reference at `<dir>/<name>.png`.
///
/// - `bless` (or a missing reference): writes `actual` as the reference, returns
///   [`Outcome::Blessed`].
/// - otherwise compares within `tol`: [`Outcome::Match`], or [`Outcome::Mismatch`]
///   after writing `<name>.actual.png` and `<name>.diff.png` next to the reference.
///
/// A dimension change is reported as a mismatch with a saturated comparison.
pub fn check(dir: &Path, name: &str, actual: &RgbaImage, tol: Tolerance, bless: bool) -> Outcome {
    let ref_path = dir.join(format!("{name}.png"));
    if bless || !ref_path.exists() {
        actual.save(&ref_path).expect("write reference");
        return Outcome::Blessed;
    }
    let reference = image::open(&ref_path).expect("open reference").to_rgba8();
    if reference.dimensions() != actual.dimensions() {
        actual
            .save(dir.join(format!("{name}.actual.png")))
            .expect("write actual");
        return Outcome::Mismatch {
            comparison: Comparison {
                max_channel_delta: 255,
                pixels_over: u64::from(actual.width()) * u64::from(actual.height()),
                total_pixels: u64::from(actual.width()) * u64::from(actual.height()),
            },
            tolerance: tol,
        };
    }
    let comparison = compare(&reference, actual, tol);
    if comparison.passes(tol) {
        Outcome::Match
    } else {
        actual
            .save(dir.join(format!("{name}.actual.png")))
            .expect("write actual");
        diff_image(&reference, actual, tol)
            .save(dir.join(format!("{name}.diff.png")))
            .expect("write diff");
        Outcome::Mismatch {
            comparison,
            tolerance: tol,
        }
    }
}

/// Load per-scene tolerance overrides from a manifest file.
///
/// Format: one `<name> <max_channel_delta> <max_pixels_over>` per line; blank
/// lines and `#` comments ignored. A missing file yields an empty map (so every
/// scene is exact). Scenes absent from the map use [`Tolerance::EXACT`].
pub fn load_tolerances(path: &Path) -> HashMap<String, Tolerance> {
    let mut map = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return map;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        if let (Some(name), Some(d), Some(p)) = (it.next(), it.next(), it.next()) {
            if let (Ok(max_channel_delta), Ok(max_pixels_over)) = (d.parse(), p.parse()) {
                map.insert(
                    name.to_string(),
                    Tolerance {
                        max_channel_delta,
                        max_pixels_over,
                    },
                );
            }
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, px: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba(px))
    }

    #[test]
    fn identical_images_pass_exact() {
        let a = solid(4, 4, [10, 20, 30, 255]);
        let b = a.clone();
        let c = compare(&a, &b, Tolerance::EXACT);
        assert_eq!(c.max_channel_delta, 0);
        assert_eq!(c.pixels_over, 0);
        assert!(c.passes(Tolerance::EXACT));
    }

    #[test]
    fn one_off_pixel_fails_exact_but_passes_within_tolerance() {
        let a = solid(4, 4, [10, 20, 30, 255]);
        let mut b = a.clone();
        b.put_pixel(1, 1, Rgba([12, 20, 30, 255])); // channel 0 off by 2
        let c = compare(&a, &b, Tolerance::EXACT);
        assert_eq!(c.max_channel_delta, 2);
        assert_eq!(c.pixels_over, 1);
        assert!(!c.passes(Tolerance::EXACT));

        // A tolerance that allows delta 2 and one over-pixel passes.
        let tol = Tolerance {
            max_channel_delta: 2,
            max_pixels_over: 1,
        };
        let c2 = compare(&a, &b, tol);
        assert_eq!(c2.pixels_over, 0); // delta 2 is not "over" at threshold 2
        assert!(c2.passes(tol));
    }
}
