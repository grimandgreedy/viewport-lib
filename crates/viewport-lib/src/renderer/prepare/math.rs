//! Small numeric helpers used by the prepare passes.

/// Hash a byte slice for per-batch dirty detection.
///
/// Used by the partial-upload path to avoid reading back the cached instance
/// buffer: a hash mismatch means the batch changed; a match means it is clean.
pub(super) fn hash_instance_bytes(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}
