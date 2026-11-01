// @viewport-wgsl-version: 1
// The HDR scene pass's fragment output: the scene colour at location 0 and
// the indirect fraction at location 1, the share of the pixel's light that
// is indirect (ambient, IBL, probes, lightmaps). The composite scales ambient
// occlusion and contact shadows by the fraction, so they darken the indirect
// light only. Direct light and emission count as direct; an unlit colour has
// fraction 0.
//
// The fraction rides in x with the colour's alpha in w, so a pipeline that
// gives the fraction target the colour target's blend mixes the two alike.

// The fragment's indirect fraction, for a body to set before the scene-pass
// entry returns `scene_fragment(colour, scene_indirect_fraction)`. Zero, the
// unlit value, unless the body assigns it.
var<private> scene_indirect_fraction: f32;

struct SceneFragment {
    @location(0) colour: vec4<f32>,
    @location(1) indirect_fraction: vec4<f32>,
};

// The indirect share of `indirect + direct` by luminance; 0 for black.
fn indirect_fraction(indirect: vec3<f32>, direct: vec3<f32>) -> f32 {
    let w = vec3<f32>(0.2126, 0.7152, 0.0722);
    let i = max(dot(indirect, w), 0.0);
    let d = max(dot(direct, w), 0.0);
    return i / max(i + d, 1e-6);
}

fn scene_fragment(colour: vec4<f32>, fraction: f32) -> SceneFragment {
    return SceneFragment(colour, vec4<f32>(fraction, 0.0, 0.0, colour.a));
}
