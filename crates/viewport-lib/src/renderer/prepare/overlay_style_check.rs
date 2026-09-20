//! Debug-build reporting for overlay style fields the drawing family ignores.
//!
//! `OverlayStyle` is shared across the overlay item types, so a field can be
//! set on an item whose coverage backend cannot draw it. That is a documented
//! situation rather than a bug, but silently dropping the effect is how a
//! consumer ends up debugging their own correct code. This logs the mismatch
//! once per (family, field) pair for the life of the process, so a per-frame
//! loop over a thousand items produces one line, and compiles to nothing in
//! release.
//!
//! `OverlayStyleSupport::inert_fields` is the same check as a plain function,
//! for a consumer who wants it in a test.

#[cfg(debug_assertions)]
use std::sync::{Mutex, OnceLock};

#[cfg(debug_assertions)]
fn seen() -> &'static Mutex<std::collections::HashSet<(&'static str, &'static str)>> {
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(&'static str, &'static str)>>> =
        OnceLock::new();
    SEEN.get_or_init(Default::default)
}

/// The (family, field) pairs from this call that had not been reported before.
/// Split out so the dedupe can be tested without capturing log output.
#[cfg(debug_assertions)]
fn take_unreported(
    family: &'static str,
    support: crate::renderer::types::OverlayStyleSupport,
    style: &crate::renderer::types::OverlayStyle,
) -> Vec<&'static str> {
    let inert = support.inert_fields(style);
    if inert.is_empty() {
        return Vec::new();
    }
    let Ok(mut seen) = seen().lock() else {
        return Vec::new();
    };
    inert
        .into_iter()
        .filter(|field| seen.insert((family, *field)))
        .collect()
}

/// Log any style field `family` does not draw that `style` sets, once each.
#[cfg(debug_assertions)]
pub(super) fn warn_inert_style(
    family: &'static str,
    support: crate::renderer::types::OverlayStyleSupport,
    style: &crate::renderer::types::OverlayStyle,
) {
    for field in take_unreported(family, support, style) {
        tracing::warn!(
            "overlay: `{family}` does not draw `style.{field}`, so the value set on it has \
             no effect. Query OverlayStyleSupport for what this family draws."
        );
    }
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;
    use crate::renderer::types::{OverlayStyle, OverlayStyleSupport, ShadowLayer};

    /// A per-frame loop over many offending items logs one line per field, not
    /// one per item: an overlay frame can carry thousands.
    #[test]
    fn each_inert_field_is_reported_once() {
        let style = OverlayStyle::default()
            .with_inner_shadows(vec![ShadowLayer::default()])
            .with_texture(crate::renderer::types::OverlayTextureId::INVALID);
        let support = OverlayStyleSupport::for_glyphs();

        // A name of this test's own, so the process-wide set is not shared with
        // whatever the renderer reported in another test.
        let first = take_unreported("TestFamily", support, &style);
        assert_eq!(first, ["texture"]);
        for _ in 0..1000 {
            assert!(take_unreported("TestFamily", support, &style).is_empty());
        }
    }
}

/// Release builds do no checking.
#[cfg(not(debug_assertions))]
pub(super) fn warn_inert_style(
    _family: &'static str,
    _support: crate::renderer::types::OverlayStyleSupport,
    _style: &crate::renderer::types::OverlayStyle,
) {
}
