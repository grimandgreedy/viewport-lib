//! A cheap hasher for the maps the renderer rebuilds or probes every frame.
//!
//! The standard `HashMap` hashes with SipHash, which costs hundreds of
//! nanoseconds on a key of a few hundred bytes. These maps are keyed by data
//! the renderer derives itself (material blocks, packed integers), so they
//! need speed, not resistance to chosen keys.

use std::hash::{BuildHasherDefault, Hasher};

/// A `HashMap` hashed with [`FastHasher`].
pub(crate) type FastMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FastHasher>>;

/// Multiply-rotate over eight bytes at a time, with a final mix so keys that
/// differ only in their high bits still spread across buckets.
#[derive(Default, Clone, Copy)]
pub(crate) struct FastHasher(u64);

const K: u64 = 0x517c_c1b7_2722_0a95;

impl FastHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        let h = self.0;
        (h ^ (h >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9) ^ (h >> 32)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut tail = [0u8; 8];
            tail[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(tail));
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_differing_only_in_high_fields_spread() {
        // Packed (x, y, z) cells where only y and z vary would all land in
        // one bucket chain without the final mix.
        let mut low_bits = std::collections::HashSet::new();
        for y in 0..64u64 {
            let mut h = FastHasher::default();
            h.write_u64(7 | (y << 21));
            low_bits.insert(h.finish() & 0xff);
        }
        assert!(
            low_bits.len() > 32,
            "only {} distinct low bytes",
            low_bits.len()
        );
    }

    #[test]
    fn byte_keys_hash_by_content() {
        let a = [3u8; 304];
        let mut b = [3u8; 304];
        let hash = |k: &[u8; 304]| {
            let mut h = FastHasher::default();
            std::hash::Hash::hash(k, &mut h);
            h.finish()
        };
        assert_eq!(hash(&a), hash(&b));
        b[300] = 4;
        assert_ne!(hash(&a), hash(&b));
    }
}
