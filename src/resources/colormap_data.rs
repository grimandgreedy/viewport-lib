/// Built-in colormap LUT data (256 RGBA samples each).
///
/// Each function returns a `[[u8; 4]; 256]` array suitable for uploading to a
/// 256×1 GPU texture.  All values are in linear (non-sRGB) space to match the
/// `Rgba8Unorm` texture format used by the LUT sampler.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline]
fn to_u8(x: f32) -> u8 {
    (clamp01(x) * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------------------
// Greyscale: linear ramp black -> white
// ---------------------------------------------------------------------------

/// Linear greyscale ramp from black (index 0) to white (index 255).
pub fn greyscale_rgba() -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];
    for i in 0..256 {
        let v = i as u8;
        lut[i] = [v, v, v, 255];
    }
    lut
}

// ---------------------------------------------------------------------------
// Viridis: Smith 2015 polynomial approximation
// ---------------------------------------------------------------------------

/// Viridis colormap : perceptually uniform, colorblind-friendly.
///
/// Uses the Smith 2015 polynomial approximation.
pub fn viridis_rgba() -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];
    for i in 0..256 {
        let t = i as f32 / 255.0;
        // Degree-6 polynomial approximation (Zucker/Smith coefficients).
        // Range: dark purple (t=0) -> blue -> teal -> green -> yellow (t=1).
        let r = 0.277_727_33
            + t * (0.105_093_04
                + t * (-0.330_861_83
                    + t * (-4.634_230_5
                        + t * (6.228_269_9
                            + t * (4.776_385_0 - t * 5.435_455_9)))));
        let g = 0.005_407_345
            + t * (1.404_613_5
                + t * (0.214_847_56
                    + t * (-5.799_101_0
                        + t * (14.179_933
                            + t * (-13.745_145 + t * 4.645_852_6)))));
        let b = 0.334_099_81
            + t * (1.384_590_2
                + t * (0.095_095_16
                    + t * (-19.332_441
                        + t * (56.690_552
                            + t * (-65.353_033 + t * 26.312_435)))));
        lut[i] = [to_u8(r), to_u8(g), to_u8(b), 255];
    }
    lut
}

// ---------------------------------------------------------------------------
// Plasma: polynomial approximation
// ---------------------------------------------------------------------------

/// Plasma colormap : perceptually uniform, colorblind-friendly.
///
/// Uses a polynomial approximation of the Matplotlib plasma colormap.
pub fn plasma_rgba() -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];
    for i in 0..256 {
        let t = i as f32 / 255.0;
        // Degree-6 polynomial approximation (Zucker/Kall coefficients).
        // Range: dark blue (t=0) -> pink (t=0.5) -> orange -> yellow (t=1).
        let r = 0.058_732_344
            + t * (2.176_514_6
                + t * (-2.689_460_5
                    + t * (6.130_348_3
                        + t * (-11.107_436
                            + t * (10.023_066 - t * 3.658_713_8)))));
        let g = 0.023_336_709
            + t * (0.238_383_42
                + t * (-7.455_851
                    + t * (42.346_188
                        + t * (-82.666_31
                            + t * (71.413_62 - t * 22.931_534)))));
        let b = 0.543_340_18
            + t * (0.753_960_46
                + t * (3.110_800
                    + t * (-28.518_854
                        + t * (60.139_847
                            + t * (-54.072_186 + t * 18.191_908)))));
        lut[i] = [to_u8(r), to_u8(g), to_u8(b), 255];
    }
    lut
}

// ---------------------------------------------------------------------------
// Coolwarm: diverging blue -> white -> red (cubic Hermite)
// ---------------------------------------------------------------------------

/// Diverging coolwarm colormap: blue at t=0, white at t=0.5, red at t=1.
pub fn coolwarm_rgba() -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];

    // Control points in linear [0,1] space.
    let cold = [59.0 / 255.0_f32, 76.0 / 255.0, 192.0 / 255.0]; // blue
    let mid = [220.0 / 255.0_f32, 220.0 / 255.0, 220.0 / 255.0]; // near-white
    let warm = [180.0 / 255.0_f32, 4.0 / 255.0, 38.0 / 255.0]; // red

    for i in 0..256 {
        let t = i as f32 / 255.0;
        // Piecewise smooth cubic Hermite blend across two halves.
        let (r, g, b) = if t <= 0.5 {
            let s = t * 2.0; // [0,1] in the cold half
            let h = s * s * (3.0 - 2.0 * s); // smoothstep
            (
                cold[0] + h * (mid[0] - cold[0]),
                cold[1] + h * (mid[1] - cold[1]),
                cold[2] + h * (mid[2] - cold[2]),
            )
        } else {
            let s = (t - 0.5) * 2.0; // [0,1] in the warm half
            let h = s * s * (3.0 - 2.0 * s);
            (
                mid[0] + h * (warm[0] - mid[0]),
                mid[1] + h * (warm[1] - mid[1]),
                mid[2] + h * (warm[2] - mid[2]),
            )
        };
        lut[i] = [to_u8(r), to_u8(g), to_u8(b), 255];
    }
    lut
}

// ---------------------------------------------------------------------------
// Rainbow: HSV hue sweep 240° (blue) -> 0° (red) at full saturation/value
// ---------------------------------------------------------------------------

/// Rainbow colormap: HSV hue sweep from 240° (blue) at t=0 to 0° (red) at t=1.
pub fn rainbow_rgba() -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];
    for i in 0..256 {
        let t = i as f32 / 255.0;
        // Hue sweeps from 240° down to 0° as t goes from 0 to 1.
        let hue = 240.0 * (1.0 - t); // degrees
        let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
        lut[i] = [to_u8(r), to_u8(g), to_u8(b), 255];
    }
    lut
}

/// Convert HSV (hue in degrees [0,360), saturation [0,1], value [0,1]) to RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (v, v, v);
    }
    let h = h % 360.0;
    let sector = (h / 60.0).floor() as i32;
    let frac = h / 60.0 - sector as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * frac);
    let t = v * (1.0 - s * (1.0 - frac));
    match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// ---------------------------------------------------------------------------
// Custom colormap LUT generation
// ---------------------------------------------------------------------------

/// Linearly interpolate between colour stops to produce a 256-sample LUT.
///
/// `stops` is a slice of `(position, rgba)` pairs where `position` is in `[0, 1]`.
/// Stops are sorted by position before interpolation.
/// The result is clamped: the first stop colour fills `t < stops[0].0` and the last
/// stop colour fills `t > stops.last().0`.
///
/// Returns a `[[u8; 4]; 256]` suitable for uploading to a 256×1 GPU texture.
pub fn lerp_colormap_lut(stops: &[(f32, [u8; 4])]) -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];
    if stops.is_empty() {
        return lut;
    }
    // Sort by position.
    let mut sorted: Vec<(f32, [u8; 4])> = stops.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..256 {
        let t = i as f32 / 255.0;

        // Find the surrounding stops.
        let first = sorted[0];
        let last = *sorted.last().unwrap();

        if t <= first.0 {
            lut[i] = first.1;
            continue;
        }
        if t >= last.0 {
            lut[i] = last.1;
            continue;
        }

        // Binary search for the lower stop.
        let pos = sorted.partition_point(|s| s.0 <= t);
        let lo = sorted[pos - 1];
        let hi = sorted[pos];

        let range = hi.0 - lo.0;
        let frac = if range > 1.0e-7 {
            (t - lo.0) / range
        } else {
            0.0
        };
        lut[i] = [
            (lo.1[0] as f32 + (hi.1[0] as f32 - lo.1[0] as f32) * frac + 0.5) as u8,
            (lo.1[1] as f32 + (hi.1[1] as f32 - lo.1[1] as f32) * frac + 0.5) as u8,
            (lo.1[2] as f32 + (hi.1[2] as f32 - lo.1[2] as f32) * frac + 0.5) as u8,
            (lo.1[3] as f32 + (hi.1[3] as f32 - lo.1[3] as f32) * frac + 0.5) as u8,
        ];
    }
    lut
}

/// Parse a ParaView XML colormap file.
///
/// Supports the standard format:
/// ```xml
/// <ColorMaps>
///   <ColorMap name="MyMap" space="RGB">
///     <Point x="0.0" r="0.0" g="0.0" b="0.0" o="1.0"/>
///     <Point x="1.0" r="1.0" g="1.0" b="1.0" o="1.0"/>
///   </ColorMap>
/// </ColorMaps>
/// ```
///
/// Returns a list of `(name, stops)` pairs, where each stop is `(position_0_1, [r, g, b, a])`.
/// Uses simple string parsing without any XML library dependency.
pub fn parse_paraview_xml_colormap(xml: &str) -> Vec<(String, Vec<(f32, [u8; 4])>)> {
    let mut result = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_stops: Vec<(f32, [u8; 4])> = Vec::new();

    for line in xml.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("<ColorMap") {
            // Extract name attribute.
            current_name = attr_value(trimmed, "name").map(|s| s.to_string());
            current_stops.clear();
        } else if trimmed.starts_with("</ColorMap>") {
            if let Some(name) = current_name.take() {
                if !current_stops.is_empty() {
                    result.push((name, current_stops.clone()));
                }
            }
            current_stops.clear();
        } else if trimmed.starts_with("<Point") {
            // Parse x, r, g, b, o attributes.
            if let (Some(x), Some(r), Some(g), Some(b)) = (
                attr_f32(trimmed, "x"),
                attr_f32(trimmed, "r"),
                attr_f32(trimmed, "g"),
                attr_f32(trimmed, "b"),
            ) {
                let a = attr_f32(trimmed, "o").unwrap_or(1.0);
                let stop = (
                    x,
                    [
                        (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                        (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                        (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                        (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    ],
                );
                current_stops.push(stop);
            }
        }
    }

    result
}

/// Serialize a colormap to the ParaView XML colormap format.
pub fn export_paraview_xml_colormap(name: &str, stops: &[(f32, [u8; 4])]) -> String {
    let mut out = String::new();
    out.push_str("<ColorMaps>\n");
    out.push_str(&format!("  <ColorMap name=\"{}\" space=\"RGB\">\n", name));
    for &(pos, rgba) in stops {
        let r = rgba[0] as f32 / 255.0;
        let g = rgba[1] as f32 / 255.0;
        let b = rgba[2] as f32 / 255.0;
        let o = rgba[3] as f32 / 255.0;
        out.push_str(&format!(
            "    <Point x=\"{:.6}\" r=\"{:.6}\" g=\"{:.6}\" b=\"{:.6}\" o=\"{:.6}\"/>\n",
            pos, r, g, b, o
        ));
    }
    out.push_str("  </ColorMap>\n");
    out.push_str("</ColorMaps>\n");
    out
}

// ---------------------------------------------------------------------------
// XML parsing helpers
// ---------------------------------------------------------------------------

/// Extract the string value of a named XML attribute from a tag line.
/// e.g. `attr_value(r#"<Point x="0.5"/>"#, "x")` returns `Some("0.5")`.
fn attr_value<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let search = format!("{}=\"", name);
    let start = tag.find(search.as_str())? + search.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

fn attr_f32(tag: &str, name: &str) -> Option<f32> {
    attr_value(tag, name)?.parse().ok()
}
