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
//! A reference image lives at `<dir>/<name>.png`. What [`check`] does with it
//! depends on the [`GoldenMode`], read from the environment by
//! [`GoldenMode::from_env`]:
//!
//! - `Compare` (the default): compares against the reference. A missing
//!   reference is a failure ([`Outcome::Missing`]), not a fresh bless, so a
//!   backend without a blessed set cannot pass by photographing itself.
//! - `Bless` (`BLESS=1`): writes the frame as the new reference.
//! - `RenderOnly` (`GOLDEN_RENDER_ONLY=1`): writes the frame under
//!   `<target>/golden-render/<dir name>/` and never touches the reference
//!   directory. For backends that have no blessed set yet: it checks that every
//!   scene renders without claiming the pixels are right.
//!
//! On a mismatch the actual frame and a red diff image are written next to the
//! reference for inspection.
//!
//! A reference directory blessed on a software rasteriser can carry a driver
//! stamp ([`DRIVER_STAMP_FILE`]); [`check_driver_stamp`] fails the run when the
//! live driver differs, so a Mesa upgrade reads as "re-bless", not as pixel
//! noise.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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

/// The largest mean per-channel difference over any `block` x `block` tile
/// (0..=255 scale; edge tiles are averaged over the pixels they hold).
///
/// A difference scattered over isolated pixels keeps every tile's mean low
/// however large each pixel's difference; a change to a whole surface moves
/// every tile it covers.
pub fn max_block_mean_delta(reference: &RgbaImage, actual: &RgbaImage, block: u32) -> f32 {
    assert_eq!(
        reference.dimensions(),
        actual.dimensions(),
        "reference/actual size mismatch"
    );
    let (w, h) = reference.dimensions();
    let mut worst = 0f32;
    for by in (0..h).step_by(block as usize) {
        for bx in (0..w).step_by(block as usize) {
            let mut sum = 0u64;
            let mut n = 0u64;
            for y in by..(by + block).min(h) {
                for x in bx..(bx + block).min(w) {
                    let (r, a) = (reference.get_pixel(x, y), actual.get_pixel(x, y));
                    for c in 0..4 {
                        sum += (i16::from(r.0[c]) - i16::from(a.0[c])).unsigned_abs() as u64;
                    }
                    n += 4;
                }
            }
            worst = worst.max(sum as f32 / n as f32);
        }
    }
    worst
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

/// What [`check`] does with a frame and its reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GoldenMode {
    /// Compare against the reference; a missing reference fails.
    Compare,
    /// Write the frame as the reference.
    Bless,
    /// Write the frame under the render-only directory, leaving references alone.
    RenderOnly,
}

impl GoldenMode {
    /// The mode named by the environment: `BLESS` selects [`Bless`](Self::Bless),
    /// `GOLDEN_RENDER_ONLY` selects [`RenderOnly`](Self::RenderOnly), neither
    /// gives [`Compare`](Self::Compare).
    ///
    /// # Panics
    ///
    /// When both are set: a run cannot bless and leave the references alone.
    pub fn from_env() -> Self {
        Self::from_flags(
            std::env::var_os("BLESS").is_some(),
            std::env::var_os("GOLDEN_RENDER_ONLY").is_some(),
        )
    }

    /// The mode for the given `BLESS` / `GOLDEN_RENDER_ONLY` flags.
    ///
    /// # Panics
    ///
    /// When both flags are set.
    pub fn from_flags(bless: bool, render_only: bool) -> Self {
        match (bless, render_only) {
            (true, true) => panic!("BLESS and GOLDEN_RENDER_ONLY are both set; pick one"),
            (true, false) => Self::Bless,
            (false, true) => Self::RenderOnly,
            (false, false) => Self::Compare,
        }
    }
}

/// The result of checking one scene's frame against its reference.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// The reference was written ([`GoldenMode::Bless`]).
    Blessed,
    /// The frame was written to `path` outside the reference directory
    /// ([`GoldenMode::RenderOnly`]).
    Rendered {
        /// Where the frame was written.
        path: PathBuf,
    },
    /// The actual frame matched the reference within tolerance.
    Match,
    /// There is no reference to compare against ([`GoldenMode::Compare`]).
    /// Nothing was written. [`missing_message`] formats the failure.
    Missing {
        /// The reference path that does not exist.
        reference: PathBuf,
    },
    /// The actual frame differed beyond tolerance. Actual and diff images were
    /// written next to the reference.
    Mismatch {
        /// Observed difference.
        comparison: Comparison,
        /// Tolerance it was checked against.
        tolerance: Tolerance,
    },
}

/// The failure text for a scene with no reference in `dir`.
pub fn missing_message(name: &str, dir: &Path) -> String {
    format!(
        "no reference for {name} in {}: bless on the reference machine with BLESS=1",
        dir.display()
    )
}

/// Check `actual` against the reference at `<dir>/<name>.png`.
///
/// - [`GoldenMode::Bless`]: writes `actual` as the reference (creating `dir`),
///   returns [`Outcome::Blessed`].
/// - [`GoldenMode::RenderOnly`]: writes `actual` to
///   `<render_only_dir(dir)>/<name>.png`, returns [`Outcome::Rendered`]. `dir`
///   is not read or created.
/// - [`GoldenMode::Compare`]: [`Outcome::Missing`] when the reference does not
///   exist; otherwise compares within `tol` and returns [`Outcome::Match`], or
///   [`Outcome::Mismatch`] after writing `<name>.actual.png` and
///   `<name>.diff.png` next to the reference.
///
/// A dimension change is reported as a mismatch with a saturated comparison.
pub fn check(
    dir: &Path,
    name: &str,
    actual: &RgbaImage,
    tol: Tolerance,
    mode: GoldenMode,
) -> Outcome {
    let ref_path = dir.join(format!("{name}.png"));
    match mode {
        GoldenMode::Bless => {
            std::fs::create_dir_all(dir).expect("create reference dir");
            actual.save(&ref_path).expect("write reference");
            return Outcome::Blessed;
        }
        GoldenMode::RenderOnly => {
            let out = render_only_dir(dir);
            std::fs::create_dir_all(&out).expect("create render-only dir");
            let path = out.join(format!("{name}.png"));
            actual.save(&path).expect("write render-only frame");
            return Outcome::Rendered { path };
        }
        GoldenMode::Compare => {}
    }
    if !ref_path.exists() {
        return Outcome::Missing {
            reference: ref_path,
        };
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

/// Where [`GoldenMode::RenderOnly`] writes the frames for the reference
/// directory `reference_dir`: `<target>/golden-render/<dir name>/`.
///
/// `<target>` is `CARGO_TARGET_DIR` when set, else the cargo target directory
/// the running test binary was built into, else the system temp directory.
pub fn render_only_dir(reference_dir: &Path) -> PathBuf {
    let leaf = reference_dir
        .file_name()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("unknown"));
    target_dir().join("golden-render").join(leaf)
}

fn target_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR") {
        return PathBuf::from(dir);
    }
    // Test binaries live at `<target>/<profile>/deps/`, examples at
    // `<target>/<profile>/examples/`.
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            exe.ancestors()
                .find(|a| {
                    a.file_name()
                        .is_some_and(|n| n == "deps" || n == "examples")
                })
                .and_then(|d| d.parent()?.parent())
                .map(Path::to_path_buf)
        })
        .unwrap_or_else(std::env::temp_dir)
}

/// The file in a reference directory that records the driver the set was
/// blessed on (the adapter's `driver_info`, as `vulkaninfo` prints
/// `driverInfo`).
pub const DRIVER_STAMP_FILE: &str = "MESA_VERSION";

/// Check the driver stamp in `dir` against the live adapter's `driver_info`.
///
/// `Ok` when `dir` has no [`DRIVER_STAMP_FILE`] or the stamp matches (leading
/// and trailing whitespace ignored). `Err` names both versions: a software
/// rasteriser's output moves with its version, so a changed driver means the
/// set needs a re-bless rather than a pixel investigation.
pub fn check_driver_stamp(dir: &Path, live_driver_info: &str) -> Result<(), String> {
    let path = dir.join(DRIVER_STAMP_FILE);
    let Ok(stamp) = std::fs::read_to_string(&path) else {
        return Ok(());
    };
    let (stamp, live) = (stamp.trim(), live_driver_info.trim());
    if stamp == live {
        Ok(())
    } else {
        Err(format!(
            "re-bless: Mesa changed: {} was blessed on '{stamp}', this adapter reports '{live}'",
            dir.display()
        ))
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

    /// A fresh, empty directory under the system temp dir.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vpl-golden-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn mode_from_flags() {
        assert_eq!(GoldenMode::from_flags(false, false), GoldenMode::Compare);
        assert_eq!(GoldenMode::from_flags(true, false), GoldenMode::Bless);
        assert_eq!(GoldenMode::from_flags(false, true), GoldenMode::RenderOnly);
    }

    #[test]
    #[should_panic(expected = "both set")]
    fn mode_rejects_bless_with_render_only() {
        let _ = GoldenMode::from_flags(true, true);
    }

    #[test]
    fn compare_without_reference_is_missing_and_writes_nothing() {
        let dir = temp_dir("missing");
        let img = solid(4, 4, [1, 2, 3, 255]);
        let outcome = check(&dir, "absent", &img, Tolerance::EXACT, GoldenMode::Compare);
        match outcome {
            Outcome::Missing { reference } => assert_eq!(reference, dir.join("absent.png")),
            other => panic!("expected Missing, got {other:?}"),
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        let msg = missing_message("absent", &dir);
        assert!(msg.starts_with("no reference for absent in "));
        assert!(msg.ends_with(": bless on the reference machine with BLESS=1"));

        // A reference directory that does not exist at all is missing too, and
        // is not created.
        let gone = dir.join("no-such-backend");
        assert!(matches!(
            check(&gone, "absent", &img, Tolerance::EXACT, GoldenMode::Compare),
            Outcome::Missing { .. }
        ));
        assert!(!gone.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn bless_then_compare_matches() {
        let dir = temp_dir("bless").join("backend");
        let img = solid(4, 4, [9, 8, 7, 255]);
        assert!(matches!(
            check(&dir, "scene", &img, Tolerance::EXACT, GoldenMode::Bless),
            Outcome::Blessed
        ));
        assert!(dir.join("scene.png").exists());
        assert!(matches!(
            check(&dir, "scene", &img, Tolerance::EXACT, GoldenMode::Compare),
            Outcome::Match
        ));
        std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
    }

    #[test]
    fn render_only_leaves_the_reference_dir_alone() {
        let root = temp_dir("render-only");
        let dir = root.join("vpl-golden-render-only-test");
        let img = solid(4, 4, [4, 5, 6, 255]);
        let outcome = check(
            &dir,
            "scene",
            &img,
            Tolerance::EXACT,
            GoldenMode::RenderOnly,
        );
        let Outcome::Rendered { path } = outcome else {
            panic!("expected Rendered, got {outcome:?}");
        };
        assert!(!dir.exists(), "render-only created the reference dir");
        assert!(path.exists());
        assert_eq!(path, render_only_dir(&dir).join("scene.png"));
        let snapshots = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
        assert!(!path.starts_with(&snapshots));
        assert!(
            path.parent()
                .unwrap()
                .ends_with("golden-render/vpl-golden-render-only-test")
        );
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn driver_stamp_absent_matching_and_changed() {
        let dir = temp_dir("stamp");
        assert!(check_driver_stamp(&dir, "Mesa 24.0.5").is_ok());
        std::fs::write(dir.join(DRIVER_STAMP_FILE), "Mesa 24.0.5\n").unwrap();
        assert!(check_driver_stamp(&dir, "Mesa 24.0.5").is_ok());
        let err = check_driver_stamp(&dir, "Mesa 25.0.7").unwrap_err();
        assert!(err.starts_with("re-bless: Mesa changed"), "{err}");
        assert!(err.contains("Mesa 24.0.5") && err.contains("Mesa 25.0.7"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
