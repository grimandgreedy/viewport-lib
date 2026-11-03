// Tone-map operators that do more than a per-channel curve: AgX, the ACES
// RRT and ODT approximation, and the generic curve. AgX and ACES take and
// return linear Rec.2020. The matrices are columns (WGSL order) and mirror the
// row-major f64 constants in viewport-lib-types' colour_space and
// tone_operators modules; a test checks the two agree.

const SRGB_TO_REC2020 = mat3x3<f32>(
    vec3<f32>(0.627403896, 0.0690972894, 0.0163914389),
    vec3<f32>(0.329283038, 0.919540395, 0.0880133079),
    vec3<f32>(0.0433130657, 0.0113623156, 0.895595253),
);
const REC2020_TO_SRGB = mat3x3<f32>(
    vec3<f32>(1.660491, -0.124550475, -0.0181507634),
    vec3<f32>(-0.587641139, 1.1328999, -0.100578898),
    vec3<f32>(-0.0728498633, -0.0083494226, 1.11872966),
);
const AGX_INSET = mat3x3<f32>(
    vec3<f32>(0.856627153, 0.137318973, 0.111898213),
    vec3<f32>(0.0951212405, 0.761241991, 0.0767994186),
    vec3<f32>(0.0482516061, 0.101439036, 0.811302368),
);
const AGX_OUTSET = mat3x3<f32>(
    vec3<f32>(1.12710058, -0.141329763, -0.141329763),
    vec3<f32>(-0.110606643, 1.1578237, -0.110606643),
    vec3<f32>(-0.0164939387, -0.0164939387, 1.25193641),
);
const REC2020_TO_AP0 = mat3x3<f32>(
    vec3<f32>(0.668685577, 0.0449002066, 0.0),
    vec3<f32>(0.151817682, 0.862145499, 0.0278271136),
    vec3<f32>(0.177189679, 0.101922445, 1.05170358),
);
const AP0_TO_AP1 = mat3x3<f32>(
    vec3<f32>(1.45143932, -0.0765537734, 0.00831614843),
    vec3<f32>(-0.236510747, 1.1762297, -0.00603244979),
    vec3<f32>(-0.214928569, -0.0996759264, 0.997716301),
);
const AP1_TO_XYZ = mat3x3<f32>(
    vec3<f32>(0.662454181, 0.272228717, -0.00557464949),
    vec3<f32>(0.134004206, 0.674081766, 0.00406073353),
    vec3<f32>(0.156187687, 0.0536895174, 1.0103391),
);
const XYZ_TO_AP1 = mat3x3<f32>(
    vec3<f32>(1.64102338, -0.663662859, 0.0117218943),
    vec3<f32>(-0.324803294, 1.61533159, -0.008284442),
    vec3<f32>(-0.236424695, 0.0167563477, 0.988394859),
);
const AP1_TO_REC2020 = mat3x3<f32>(
    vec3<f32>(1.04179138, -0.00168312772, -0.00520968653),
    vec3<f32>(-0.0107415627, 1.00036605, -0.0226414457),
    vec3<f32>(-0.00696187507, -0.00140821099, 0.952302415),
);
const AP1_LUMINANCE = vec3<f32>(0.272228717, 0.674081766, 0.0536895174);

const REC709_LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

// Log2 encoding range of AgX, in stops about 1.
const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

const AGX_LOOK_PUNCHY: u32 = 1u;
const AGX_LOOK_GOLDEN: u32 = 2u;

const HALF_MAX: f32 = 65504.0;

// AgX's 7th-order fit of its default-contrast sigmoid, on [0, 1].
fn agx_sigmoid(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    return -17.86 * x6 * x
        + 78.01 * x6
        - 126.7 * x4 * x
        + 92.06 * x4
        - 28.72 * x2 * x
        + 4.361 * x2
        - 0.1718 * x
        + 0.002857;
}

// ASC CDL slope and power, then saturation about Rec.709 luma.
fn agx_look(v: vec3<f32>, look: u32) -> vec3<f32> {
    if look != AGX_LOOK_PUNCHY && look != AGX_LOOK_GOLDEN {
        return v;
    }
    let luma = dot(v, REC709_LUMA);
    var slope = vec3<f32>(1.0);
    var power = vec3<f32>(1.35);
    var sat = 1.4;
    if look == AGX_LOOK_GOLDEN {
        slope = vec3<f32>(1.0, 0.9, 0.5);
        power = vec3<f32>(0.8);
        sat = 1.3;
    }
    let graded = pow(v * slope, power);
    return luma + sat * (graded - luma);
}

fn agx(colour: vec3<f32>, look: u32) -> vec3<f32> {
    var v = AGX_INSET * max(colour, vec3<f32>(0.0));
    v = (log2(max(v, vec3<f32>(1e-10))) - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);
    v = agx_sigmoid(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)));
    v = agx_look(v, look);
    v = AGX_OUTSET * v;
    return pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn aces_rgb_to_saturation(rgb: vec3<f32>) -> f32 {
    let tiny = 1e-5;
    let mi = min(rgb.r, min(rgb.g, rgb.b));
    let ma = max(rgb.r, max(rgb.g, rgb.b));
    return (max(ma, tiny) - max(mi, tiny)) / max(ma, 1e-2);
}

fn aces_rgb_to_yc(rgb: vec3<f32>) -> f32 {
    let yc_radius_weight = 1.75;
    let r = rgb.r;
    let g = rgb.g;
    let b = rgb.b;
    let chroma = sqrt(max(b * (b - g) + g * (g - r) + r * (r - b), 0.0));
    return (b + g + r + yc_radius_weight * chroma) / 3.0;
}

fn aces_sigmoid_shaper(x: f32) -> f32 {
    let t = max(1.0 - abs(x / 2.0), 0.0);
    let y = 1.0 + sign(x) * (1.0 - t * t);
    return y / 2.0;
}

fn aces_glow_fwd(yc_in: f32, glow_gain_in: f32, glow_mid: f32) -> f32 {
    if yc_in <= 2.0 / 3.0 * glow_mid {
        return glow_gain_in;
    }
    if yc_in >= 2.0 * glow_mid {
        return 0.0;
    }
    return glow_gain_in * (glow_mid / yc_in - 0.5);
}

fn aces_rgb_to_hue(rgb: vec3<f32>) -> f32 {
    var hue = 0.0;
    if !(rgb.r == rgb.g && rgb.g == rgb.b) {
        hue = degrees(atan2(sqrt(3.0) * (rgb.g - rgb.b), 2.0 * rgb.r - rgb.g - rgb.b));
    }
    return select(hue, hue + 360.0, hue < 0.0);
}

fn aces_centre_hue(hue: f32, centre: f32) -> f32 {
    var h = hue - centre;
    if h < -180.0 {
        h = h + 360.0;
    } else if h > 180.0 {
        h = h - 360.0;
    }
    return h;
}

fn aces_dark_to_dim_surround(linear_cv: vec3<f32>) -> vec3<f32> {
    let dim_surround_gamma = 0.9811;
    let xyz = AP1_TO_XYZ * linear_cv;
    let sum = max(xyz.x + xyz.y + xyz.z, 1e-5);
    let x = xyz.x / sum;
    let y = xyz.y / sum;
    let big_y = pow(clamp(xyz.y, 0.0, HALF_MAX), dim_surround_gamma);
    let m = big_y / max(y, 1e-5);
    return XYZ_TO_AP1 * vec3<f32>(x * m, big_y, (1.0 - x - y) * m);
}

// The ACES RRT and ODT approximation (100 nit dim-surround display).
fn aces_rrt(colour: vec3<f32>) -> vec3<f32> {
    let glow_gain = 0.05;
    let glow_mid = 0.08;
    let red_scale = 0.82;
    let red_pivot = 0.03;
    let red_hue = 0.0;
    let red_width = 135.0;
    let rrt_sat = 0.96;
    let odt_sat = 0.93;
    let brightness = 1.0;

    var ap0 = REC2020_TO_AP0 * colour;

    let saturation = aces_rgb_to_saturation(ap0);
    let yc_in = aces_rgb_to_yc(ap0);
    let s = aces_sigmoid_shaper((saturation - 0.4) / 0.2);
    let added_glow = 1.0 + aces_glow_fwd(yc_in, glow_gain * s, glow_mid);
    ap0 = ap0 * added_glow;

    let hue = aces_rgb_to_hue(ap0);
    let centred = aces_centre_hue(hue, red_hue);
    var hue_weight = smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centred / red_width));
    hue_weight = hue_weight * hue_weight;
    ap0.r = ap0.r + hue_weight * saturation * (red_pivot - ap0.r) * (1.0 - red_scale);

    var ap1 = clamp(AP0_TO_AP1 * ap0, vec3<f32>(0.0), vec3<f32>(HALF_MAX));
    ap1 = mix(vec3<f32>(dot(ap1, AP1_LUMINANCE)), ap1, rrt_sat);
    ap1 = ap1 * brightness;

    let a = 2.785085;
    let b = 0.107772;
    let c = 2.936045;
    let d = 0.887122;
    let e = 0.806889;
    let post = (ap1 * (a * ap1 + b)) / (ap1 * (c * ap1 + d) + e);

    var linear_cv = aces_dark_to_dim_surround(post);
    linear_cv = mix(vec3<f32>(dot(linear_cv, AP1_LUMINANCE)), linear_cv, odt_sat);
    return AP1_TO_REC2020 * linear_cv;
}

// The generic curve with precomputed scales: x^c * out / (x^c + in).
fn generic_curve(colour: vec3<f32>, contrast: f32, input_scale: f32, output_scale: f32) -> vec3<f32> {
    let p = pow(max(colour, vec3<f32>(0.0)), vec3<f32>(contrast));
    return output_scale * p / (p + input_scale);
}
