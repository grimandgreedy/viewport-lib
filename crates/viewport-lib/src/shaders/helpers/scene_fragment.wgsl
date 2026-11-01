// @viewport-wgsl-version: 1
// The HDR scene pass's fragment output: the scene colour at location 0 and
// the light split at location 1. Its red channel is the indirect share of the
// pixel's light (ambient, IBL, probes, lightmaps), its green channel the
// direct share (the light loop); what is left is emission. The composite
// scales ambient occlusion by the indirect share and contact shadows by the
// direct share, so neither darkens emission. An unlit colour has both at 0.
//
// The shares ride in x and y with the colour's alpha in w, so a pipeline that
// gives the target the colour target's blend mixes the two alike.

// The fragment's shares, for a body to set with `scene_light_split` before
// the scene-pass entry returns `scene_fragment(colour)`. Zero, the unlit
// value, unless the body sets them.
var<private> scene_indirect_fraction: f32;
var<private> scene_direct_fraction: f32;

struct SceneFragment {
    @location(0) colour: vec4<f32>,
    @location(1) indirect_fraction: vec4<f32>,
};

// Set the indirect and direct shares of `indirect + direct + emitted` by
// luminance; 0 for black.
fn scene_light_split(indirect: vec3<f32>, direct: vec3<f32>, emitted: vec3<f32>) {
    let w = vec3<f32>(0.2126, 0.7152, 0.0722);
    let i = max(dot(indirect, w), 0.0);
    let d = max(dot(direct, w), 0.0);
    let e = max(dot(emitted, w), 0.0);
    let total = max(i + d + e, 1e-6);
    scene_indirect_fraction = i / total;
    scene_direct_fraction = d / total;
}

// The colour is clamped to 65504, the largest finite half float, so an
// unbounded radiance cannot write an infinity into the Rgba16Float target.
fn scene_fragment(colour: vec4<f32>) -> SceneFragment {
    return SceneFragment(
        vec4<f32>(min(colour.rgb, vec3<f32>(65504.0)), colour.a),
        vec4<f32>(scene_indirect_fraction, scene_direct_fraction, 0.0, colour.a),
    );
}
