//! Font handle for overlay text items.

/// Handle to an uploaded font, used by overlay label and glyph-run items.
///
/// Obtain one from `DeviceResources::upload_font`. Pass `None` (or omit the
/// field) on overlay items to use the built-in default font; pass
/// `Some(handle)` to use a user-supplied TTF font.
///
/// This is a plain index into the renderer's font store: the store itself (and
/// the font rasterization behind it) lives in `viewport-lib`, but the handle a
/// consumer names on an overlay item is pure data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontHandle(pub usize);
