//! What an overlay item is clipped to.

/// The clipping applied to a single overlay item: an axis-aligned box, a mask
/// shape, or both.
///
/// Both are optional and compose by intersection, so an item naming both draws
/// only where they overlap. The default clips nothing.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct OverlayClip {
    /// Axis-aligned box in logical pixels `[x0, y0, x1, y1]`, in framebuffer
    /// space. Fragments outside it are discarded.
    ///
    /// Framebuffer space is the definition, not an approximation: the box stays
    /// axis-aligned on screen and does **not** turn with the item's own
    /// rotation or with the rotation of a retained group containing it, the
    /// same way a scissor rect behaves everywhere else. For a clip that follows
    /// rotated content, use `mask`, which is evaluated per fragment against a
    /// shape that can itself rotate.
    pub rect: Option<[f32; 4]>,
    /// The id of the mask shape this item is clipped to: the shape whose
    /// `provides_mask` matches. The mask's distance field is used rather than
    /// its bounding box, so a rounded or circular mask clips to its real
    /// boundary, and masks may nest. A missing mask draws the item unclipped.
    pub mask: Option<u32>,
}

impl OverlayClip {
    /// Clip to an axis-aligned box in logical pixels `[x0, y0, x1, y1]`.
    pub fn rect(rect: [f32; 4]) -> Self {
        Self {
            rect: Some(rect),
            mask: None,
        }
    }

    /// Clip to the mask shape with this id.
    pub fn mask(mask_id: u32) -> Self {
        Self {
            rect: None,
            mask: Some(mask_id),
        }
    }

    /// Add an axis-aligned box to this clip.
    pub fn with_rect(mut self, rect: [f32; 4]) -> Self {
        self.rect = Some(rect);
        self
    }

    /// Add a mask shape to this clip.
    pub fn with_mask(mut self, mask_id: u32) -> Self {
        self.mask = Some(mask_id);
        self
    }

    /// Whether this clips anything at all.
    pub fn is_set(&self) -> bool {
        self.rect.is_some() || self.mask.is_some()
    }
}
