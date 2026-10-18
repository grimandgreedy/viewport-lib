// Reader for a mesh's sidecar record, in the chunk bound at group 1 binding 6
// (`sidecar_pool`, declared by the including shader). The layout matches the
// renderer's `sidecar_pool` module:
//
//   word 0     first vertex of the mesh in its slab chunk, for the uv1 stream
//   words 1, 2 position override window: base element, element count
//   words 3, 4 normal override window
//   word 5     extension attribute stream index
//   word 6     lightmap UV1 stream index
//   word 7     stream count
//   word 8..   the stream table, (offset, len) per stream: words from the
//              record base, elements; then the streams
//
// A mesh without a record reads the empty header bound in its place (record
// base 0, empty windows, no streams). Every read is clamped to the bound
// buffer, so a malformed index reads garbage rather than out of bounds.

const SC_UV1_BASE: u32 = 0u;
const SC_POS_BASE: u32 = 1u;
const SC_POS_LEN: u32 = 2u;
const SC_NRM_BASE: u32 = 3u;
const SC_NRM_LEN: u32 = 4u;
const SC_EXTENSION: u32 = 5u;
const SC_LIGHTMAP_UV: u32 = 6u;
const SC_TABLE: u32 = 8u;
const SC_NO_STREAM: u32 = 0xffffffffu;
// Bits per stream index in the instance selection word; all ones is none.
const SC_SELECT_BITS: u32 = 10u;
const SC_SELECT_NONE: u32 = 0x3ffu;

fn sc_word(i: u32) -> u32 {
    return sidecar_pool[min(i, arrayLength(&sidecar_pool) - 1u)];
}

fn sc_f32(i: u32) -> f32 {
    return bitcast<f32>(sc_word(i));
}

fn sc_vec3(i: u32) -> vec3<f32> {
    return vec3<f32>(sc_f32(i), sc_f32(i + 1u), sc_f32(i + 2u));
}

fn sc_vec4(i: u32) -> vec4<f32> {
    return vec4<f32>(sc_f32(i), sc_f32(i + 1u), sc_f32(i + 2u), sc_f32(i + 3u));
}

// Stream `index` of the record at `rec`: x = first word in the chunk,
// y = element count. Count 0 when there is no such stream.
fn sc_stream(rec: u32, index: u32) -> vec2<u32> {
    if index == SC_NO_STREAM {
        return vec2<u32>(0u, 0u);
    }
    let entry = rec + SC_TABLE + 2u * index;
    return vec2<u32>(rec + sc_word(entry), sc_word(entry + 1u));
}

// The stream index in field `field` (0 scalar, 1 face colour, 2 warp) of an
// instance selection word.
fn sc_selected(word: u32, field: u32) -> u32 {
    let i = (word >> (field * SC_SELECT_BITS)) & SC_SELECT_NONE;
    return select(i, SC_NO_STREAM, i == SC_SELECT_NONE);
}
