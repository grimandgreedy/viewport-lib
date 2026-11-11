//! Where an item sits in the painter sequence, and the helper that composes
//! one from a layer, a sort value and a member index.

/// An item's place in the painter sequence: drawn after everything on a lower
/// `layer`, and after everything on its own layer with a lower `key`.
///
/// The renderer only sorts placements: it never derives one from an item's
/// position or groups items itself. How the numbers are chosen is up to the
/// application. [`painter_key`] composes one from a layer, a continuous sort
/// value (y in a top-down game, a depth in a side view) and a small member
/// index, which is enough to keep the parts of one entity together and in a
/// fixed order: give every part the entity's sort value and its own member
/// index.
///
/// ```
/// use viewport_lib_types::render_item::painter_key;
///
/// // A character on layer 3 at y = 12.5, drawn from a body sprite, a
/// // weapon sprite and a flat shadow mesh. A top-down game sorts by -y so
/// // what is lower on the screen draws on top.
/// let sort = -12.5;
/// let shadow = painter_key(3, sort, 0);
/// let body = painter_key(3, sort, 1);
/// let weapon = painter_key(3, sort, 2);
/// assert!(shadow.key < body.key && body.key < weapon.key);
///
/// // Another character a little further down the screen draws over all
/// // three parts of the first.
/// let other = painter_key(3, -12.0, 0);
/// assert!(other.key > weapon.key);
/// ```
///
/// Placements that compare equal keep the renderer's tiebreak: the order the
/// item types were registered, then the order each reported its draws. A
/// consumer that needs a defined order between two item types at one sort
/// value puts it in the member index.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PainterPlacement {
    /// Painter layer: lower layers draw first.
    pub layer: u32,
    /// Order within the layer: lower keys draw first. Compared with
    /// `f32::total_cmp`.
    pub key: f32,
    /// Not lit by 2D lighting: the content does not sample the light layer
    /// of the painter range `layer` lies in, so it keeps its colour (an
    /// emissive part, a glow). It draws over the range's lit content,
    /// whatever their layers and keys, in its own layer and key order among
    /// the range's other unlit content. Outside every light range it draws in
    /// its place. Default `false`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unlit: bool,
}

impl PainterPlacement {
    /// A placement on `layer` at `key`.
    pub const fn new(layer: u32, key: f32) -> Self {
        Self {
            layer,
            key,
            unlit: false,
        }
    }

    /// This placement with [`unlit`](Self::unlit) set to `unlit`.
    pub const fn with_unlit(mut self, unlit: bool) -> Self {
        self.unlit = unlit;
        self
    }
}

/// How many painter mask slots a frame has: a mask item draws into one of
/// slots `0` to `PAINTER_MASK_SLOTS - 1`.
pub const PAINTER_MASK_SLOTS: u8 = 4;

/// Whether painter content is cut by a painter mask, and which side of it
/// shows.
///
/// A mask item draws its coverage into one slot and names the painter layers
/// it applies to. Content on a layer in that range with
/// [`VisibleInside`](Self::VisibleInside) shows only where the mask covers it,
/// and with [`VisibleOutside`](Self::VisibleOutside) only where it does not.
/// Content on a layer outside the range, or testing a slot no mask drew into
/// this frame, is not cut. Slots above `PAINTER_MASK_SLOTS - 1` wrap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MaskTest {
    /// Not cut by any mask. The default.
    #[default]
    None,
    /// Shown only inside the mask in this slot.
    VisibleInside(u8),
    /// Shown only outside the mask in this slot.
    VisibleOutside(u8),
}

impl MaskTest {
    /// The test as the shaders read it: the slot in bits 0-1 and the side in
    /// bits 2-3 (`0` none, `1` inside, `2` outside).
    pub const fn word(self) -> u32 {
        match self {
            Self::None => 0,
            Self::VisibleInside(slot) => (slot as u32 & 3) | (1 << 2),
            Self::VisibleOutside(slot) => (slot as u32 & 3) | (2 << 2),
        }
    }
}

/// How many member indices [`painter_key`] keeps distinct: `0` to
/// `PAINTER_MEMBERS - 1`.
pub const PAINTER_MEMBERS: u32 = 1 << PAINTER_MEMBER_BITS;

/// Bits of the key's significand given to the member index.
const PAINTER_MEMBER_BITS: u32 = 6;

/// Compose a painter placement from a `layer`, a continuous `sort` value and a
/// `member` index.
///
/// The returned keys order by `sort` first and by `member` second: for two
/// sort values far enough apart to stay distinct (below), every member of the
/// lower one draws before every member of the higher one, and within one sort
/// value the members draw in index order.
///
/// The member index takes the low six bits of the key's significand, so the
/// sort value keeps 18 significant bits: values are told apart when they
/// differ by more than about `|sort| / 2^18` (0.004 at 1000, 0.04 at 10 000).
/// Sort values closer than that land in one bucket, where their members tie
/// and the renderer's tiebreak applies. `member` is clamped to one less than
/// [`PAINTER_MEMBERS`]. A NaN sort value is treated as zero and an infinite
/// one as the largest finite value of its sign.
pub fn painter_key(layer: u32, sort: f32, member: u32) -> PainterPlacement {
    let sort = if sort.is_nan() {
        0.0
    } else {
        // `+ 0.0` folds -0 into +0, so the two zeros share a bucket.
        sort.clamp(-f32::MAX, f32::MAX) + 0.0
    };
    let bits = sort.to_bits();
    // Map the float onto an unsigned integer that increases with it: flip
    // every bit of a negative value, only the sign bit of a positive one.
    let ordered = if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    };
    let mask = PAINTER_MEMBERS - 1;
    let composed = (ordered & !mask) | member.min(mask);
    let bits = if composed & 0x8000_0000 != 0 {
        composed & 0x7fff_ffff
    } else {
        !composed
    };
    PainterPlacement::new(layer, f32::from_bits(bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(a: PainterPlacement, b: PainterPlacement) -> std::cmp::Ordering {
        a.layer.cmp(&b.layer).then(a.key.total_cmp(&b.key))
    }

    #[test]
    fn keys_order_by_layer_then_sort_then_member() {
        let mut placements = Vec::new();
        for layer in [0u32, 1, 7] {
            for y in [-2000.0f32, -1.25, 0.0, 0.5, 3.0, 640.0, 9000.0] {
                for member in 0..40 {
                    placements.push(((layer, y, member), painter_key(layer, y, member)));
                }
            }
        }
        let mut sorted = placements.clone();
        sorted.sort_by(|a, b| order(a.1, b.1));
        let expected: Vec<_> = {
            let mut e = placements.clone();
            e.sort_by(|a, b| {
                a.0.0
                    .cmp(&b.0.0)
                    .then(a.0.1.total_cmp(&b.0.1))
                    .then(a.0.2.cmp(&b.0.2))
            });
            e
        };
        assert_eq!(
            sorted.iter().map(|p| p.0).collect::<Vec<_>>(),
            expected.iter().map(|p| p.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn members_stay_distinct_across_a_y_range() {
        // Whole and half pixels over a 4096-pixel map, with a few dozen
        // members each: every key distinct and in order.
        let mut last: Option<f32> = None;
        let mut y = -2048.0f32;
        while y <= 2048.0 {
            for member in 0..48 {
                let key = painter_key(0, y, member).key;
                assert!(key.is_finite(), "y {y} member {member}");
                if let Some(prev) = last {
                    assert!(key > prev, "y {y} member {member}: {key} after {prev}");
                }
                last = Some(key);
            }
            y += 0.5;
        }
    }

    #[test]
    fn keys_stay_close_to_the_sort_value() {
        for y in [-5000.0f32, -3.0, 0.0, 1.0, 250.0, 70000.0] {
            for member in [0, 63] {
                let key = painter_key(0, y, member).key;
                assert!(
                    (key - y).abs() <= y.abs() * 1e-5 + 1e-30,
                    "{y} {member} {key}"
                );
            }
        }
    }

    #[test]
    fn edge_values_give_finite_keys() {
        for y in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            -f32::MAX,
            -0.0,
            f32::MIN_POSITIVE,
        ] {
            for member in [0, 63, 1000] {
                let key = painter_key(2, y, member).key;
                assert!(!key.is_nan(), "{y} {member}");
                assert!(key.is_finite(), "{y} {member}");
            }
        }
        assert_eq!(painter_key(0, -0.0, 5), painter_key(0, 0.0, 5));
        assert_eq!(painter_key(0, 1.0, 1000), painter_key(0, 1.0, 63));
    }
}
