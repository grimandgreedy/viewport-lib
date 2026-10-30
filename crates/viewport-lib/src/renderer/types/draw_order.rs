//! Sort keys for draw order within a pass.
//!
//! A blended draw's key is one `u64`, drawn in ascending order:
//!
//! | Bits    | Field                                                    |
//! | ------- | -------------------------------------------------------- |
//! | 63..61  | channel, reserved (0)                                    |
//! | 60..53  | `draw_priority + 128`                                    |
//! | 52..21  | `!sortable_bits(depth)` (back to front), or 0 with a blend order |
//! | 20..5   | `blend_order`                                            |
//! | 4..0    | reserved (0)                                             |

/// Maps an `f32` to a `u32` whose unsigned order matches `f32::total_cmp`:
/// a negative value has every bit flipped, a positive one its sign bit set.
pub(crate) fn sortable_bits(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// The sort key of a blended draw. `depth` grows away from the eye (view
/// depth or distance); with equal priority and no blend order, the farther
/// draw sorts first.
pub(crate) fn blend_key(priority: i8, blend_order: Option<u16>, depth: f32) -> u64 {
    let priority = (priority as i16 + 128) as u64;
    let (depth, order) = match blend_order {
        Some(n) => (0u64, n as u64),
        None => (!sortable_bits(depth) as u64, 0u64),
    };
    (priority << 53) | (depth << 21) | (order << 5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sortable_bits_orders_signs() {
        let values = [
            f32::NEG_INFINITY,
            -1.0e9,
            -2.5,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            3.5e7,
            f32::INFINITY,
        ];
        for pair in values.windows(2) {
            assert!(
                sortable_bits(pair[0]) < sortable_bits(pair[1]),
                "{} should sort before {}",
                pair[0],
                pair[1]
            );
        }
        assert_eq!(sortable_bits(0.0), 0x8000_0000);
        assert_eq!(sortable_bits(-0.0), 0x7FFF_FFFF);
    }

    #[test]
    fn equal_priority_sorts_back_to_front() {
        let far = blend_key(0, None, 20.0);
        let mid = blend_key(0, None, 5.0);
        let near = blend_key(0, None, 0.5);
        let behind = blend_key(0, None, -3.0);
        assert!(far < mid && mid < near && near < behind);
    }

    #[test]
    fn priority_dominates_depth() {
        // A near item at priority -1 draws before a far one at 0, and a far
        // item at priority 1 after a near one at 0.
        assert!(blend_key(-1, None, 0.1) < blend_key(0, None, 1000.0));
        assert!(blend_key(0, None, 0.1) < blend_key(1, None, 1000.0));
        assert!(
            blend_key(i8::MIN, None, f32::NEG_INFINITY) < blend_key(i8::MAX, None, f32::INFINITY)
        );
        assert!(blend_key(i8::MAX, Some(u16::MAX), 0.0) < 1 << 61);
    }

    #[test]
    fn blend_order_replaces_depth() {
        // A fixed order ignores depth: lower order first, whatever the distance.
        assert!(blend_key(0, Some(1), 0.1) < blend_key(0, Some(2), 1000.0));
        assert_eq!(blend_key(0, Some(7), 0.1), blend_key(0, Some(7), 1000.0));
        // It draws before the depth-sorted draws of its priority and never
        // crosses a priority.
        assert!(blend_key(0, Some(u16::MAX), 0.0) < blend_key(0, None, 1.0e30));
        assert!(blend_key(0, None, 0.1) < blend_key(1, Some(0), 0.0));
        // The reserved low bits stay clear.
        assert_eq!(blend_key(0, Some(u16::MAX), 0.0) & 0x1F, 0);
    }
}
