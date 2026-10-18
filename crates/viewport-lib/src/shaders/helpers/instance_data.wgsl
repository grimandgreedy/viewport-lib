// One record per mesh draw instance, in the instance storage buffer every
// mesh draw binds at group 1 binding 0 and selects with
// @builtin(instance_index). Layout matches the renderer's `InstanceData`
// (160 bytes); keep the offsets in step with its doc table.
//
// Material scalars live in `material_gpu_buf`, read via `material_id`, and the
// override windows in the mesh's sidecar record. The sidecar and stream words
// are read by the draws that bind a mesh sub-slice and its record chunk; the
// chunk-bound instanced draws ignore them.
struct InstanceData {
    model: mat4x4<f32>,                   // offset 0
    colour: vec4<f32>,                    // offset 64 : base colour, opacity in .a
    flags: u32,                           // offset 80 : INST_* bits below
    material_id: u32,                     // offset 84 : index into material_gpu_buf
    alpha_cutoff: f32,                    // offset 88 : AlphaMode::Mask cutoff
    light_probe_index: u32,               // offset 92 : base block in light_probe_sh
    custom_data_id: u32,                  // offset 96 : index into instance_custom_data_buf
    backface_pattern_scale: f32,          // offset 100 : styled back-face Pattern world scale
    object_mask: u32,                     // offset 104 : layer mask, AND-tested per light
    sidecar_record: u32,                  // offset 108 : record base word in the bound sidecar chunk
    deform_flags: u32,                    // offset 112 : bit i set when deformer slot i is active
    colour_stream: u32,                   // offset 116 : first chunk word of the scalar or face-colour stream, 0 none
    warp_scale: f32,                      // offset 120 : vertex warp scale (INST_HAS_WARP)
    lightmap_layer: u32,                  // offset 124 : base lightmap atlas page
    lightmap_scale_bias: vec4<f32>,       // offset 128 : lm_uv = uv1 * .xy + .zw
    warp_stream: u32,                     // offset 144 : first chunk word of the warp stream, 0 none
    lightmap_stream: u32,                 // offset 148 : first chunk word of the lightmap UV1 stream, 0 none
    uv1_base: u32,                        // offset 152 : first vertex of the mesh in its slab chunk
    _pad: u32,                            // offset 156
};

const INST_SELECTED: u32 = 1u;
const INST_WIREFRAME: u32 = 2u;
const INST_HAS_TEXTURE: u32 = 4u;
const INST_HAS_NORMAL_MAP: u32 = 8u;
const INST_HAS_AO_MAP: u32 = 16u;
const INST_UNLIT: u32 = 32u;
// Sample the shadow atlas; without it the fragment is unshadowed.
const INST_RECEIVE_SHADOWS: u32 = 64u;
// The mesh is a closed surface: its back faces shade as inside the solid's
// own shadow instead of sampling the shadow map.
const INST_MESH_CLOSED: u32 = 128u;
// A two-sided receiver (styled back-face policy on an open mesh), which takes
// the cull-none path's receiver bias.
const INST_TWO_SIDED_RECEIVER: u32 = 256u;
// AlphaMode::Mask: discard below alpha_cutoff.
const INST_ALPHA_MASK: u32 = 512u;
const INST_HAS_LIGHT_PROBE: u32 = 1024u;
// Exempt from the global clip planes and volumes.
const INST_IGNORE_CLIP: u32 = 2048u;
const INST_HAS_POSITION_OVERRIDE: u32 = 4096u;
const INST_HAS_NORMAL_OVERRIDE: u32 = 8192u;
// The item has an active scalar attribute (the per-vertex scalar buffer).
const INST_HAS_ATTRIBUTE: u32 = 16384u;
// The attribute is a per-face colour buffer rather than a scalar.
const INST_USE_FACE_COLOUR: u32 = 32768u;
// The item has a GPU vertex warp (the per-vertex warp buffer).
const INST_HAS_WARP: u32 = 65536u;
// The mesh has a sidecar record at `sidecar_record` (base 0 is a valid record).
const INST_HAS_SIDECAR: u32 = 131072u;
