//! Fill records for the overlay fill module (`helpers/overlay_fill.wgsl`).
//
// Every overlay pipeline evaluates an item's fill per fragment from one record
// in a data texture. This module writes the records; the shader's header
// comment documents the layout, and the two have to agree texel for texel.

use crate::renderer::types::{GradientStop, OverlayFill};

/// Texels in a record's fixed header, before its stops.
const HEADER_TEXELS: usize = 8;

/// The kind codes the shader switches on.
mod kind {
    pub const LINEAR: f32 = 1.0;
    pub const RADIAL: f32 = 2.0;
    pub const CONICAL: f32 = 3.0;
    pub const SWEEP: f32 = 4.0;
    pub const TEXTURE: f32 = 5.0;
    pub const SOLID: f32 = 6.0;
}

/// The record index the shaders read as "no fill": the caller's own colour
/// passes through.
pub(crate) const NO_FILL: f32 = -1.0;

/// One frame's (or one compiled group's) fill records, appended as items are
/// emitted and uploaded once as a data texture.
#[derive(Clone, Default)]
pub(crate) struct FillTable {
    texels: Vec<[f32; 4]>,
    /// Whether a texture fill is recorded as one. A compiled group has no
    /// image bound when it draws, so there a texture fill records its tint.
    no_textures: bool,
}

impl FillTable {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// A table whose texture fills record their tint as a solid colour, for a
    /// compiled group, which draws with no image bound.
    pub(crate) fn without_textures() -> Self {
        Self {
            texels: Vec::new(),
            no_textures: true,
        }
    }

    /// Whether a texture fill draws its image from this table.
    pub(crate) fn draws_textures(&self) -> bool {
        !self.no_textures
    }

    /// Number of records written.
    #[cfg(test)]
    pub(crate) fn texel_count(&self) -> usize {
        self.texels.len()
    }

    /// The texels to upload: never empty, so the binding always has a texel to
    /// point at.
    pub(crate) fn texels(&self) -> &[[f32; 4]] {
        if self.texels.is_empty() {
            &[[0.0; 4]]
        } else {
            &self.texels
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.texels.is_empty()
    }

    /// Append the record for `fill` over a box of half-extents `half` and
    /// return its index for the vertex or instance that reads it.
    ///
    /// `opacity` multiplies every colour's alpha and then `tint` every
    /// component, in that order: the shape families fold an item's own
    /// opacity and tint into the record, the glyph families carry them on the
    /// vertex and pass the identity. `image_size` is the bound texture's pixel
    /// size, which a nine-slice fill needs.
    pub(crate) fn push(
        &mut self,
        fill: &OverlayFill,
        half: [f32; 2],
        opacity: f32,
        tint: [f32; 4],
        image_size: [u32; 2],
    ) -> f32 {
        let scale = |mut c: [f32; 4]| {
            c[3] *= opacity;
            for (c, t) in c.iter_mut().zip(tint) {
                *c *= t;
            }
            c
        };
        let base = self.texels.len();
        let mut header = [[0.0f32; 4]; HEADER_TEXELS];
        header[0] = [0.0, 0.0, -1.0, 0.0];
        header[1] = [half[0], half[1], 0.0, 0.0];
        // The texture transform defaults to the identity, so a record that is
        // not a texture still samples a defined coordinate.
        header[4] = [0.0, 0.0, 1.0, 1.0];
        let mut stops: &[GradientStop] = &[];
        match fill {
            OverlayFill::Solid(c) => {
                header[0][0] = kind::SOLID;
                header[3] = scale(c.to_linear_rgba());
            }
            OverlayFill::LinearGradient { stops: s, angle } => {
                header[0][0] = kind::LINEAR;
                header[1][2] = *angle;
                stops = s;
            }
            OverlayFill::RadialGradient { stops: s } => {
                header[0][0] = kind::RADIAL;
                stops = s;
            }
            OverlayFill::ConicalGradient {
                stops: s,
                offset_angle,
            } => {
                header[0][0] = kind::CONICAL;
                header[1][2] = *offset_angle;
                stops = s;
            }
            OverlayFill::SweepGradient {
                stops: s,
                centre,
                start_angle,
                end_angle,
            } => {
                header[0][0] = kind::SWEEP;
                let span = end_angle - start_angle;
                header[1][2] = *start_angle;
                header[1][3] = if span > 0.0 {
                    span
                } else {
                    std::f32::consts::TAU
                };
                header[2][0] = (centre[0] * 2.0 - 1.0) * half[0];
                header[2][1] = (centre[1] * 2.0 - 1.0) * half[1];
                stops = s;
            }
            OverlayFill::Texture {
                transform,
                nine_slice,
                tint: tex_tint,
                ..
            } => {
                header[3] = scale(tex_tint.to_linear_rgba());
                if self.no_textures {
                    header[0][0] = kind::SOLID;
                } else {
                    header[0][0] = kind::TEXTURE;
                    header[4] = [
                        transform.offset[0],
                        transform.offset[1],
                        transform.scale[0],
                        transform.scale[1],
                    ];
                    header[5] = [
                        transform.rotation,
                        super::overlay_geometry::tile_mode_to_f(transform.tile_mode),
                        if transform.flip_x { 1.0 } else { 0.0 },
                        if transform.flip_y { 1.0 } else { 0.0 },
                    ];
                    if let Some(ns) = nine_slice {
                        // Pixel insets as texture-UV and box fractions, for the
                        // shader's piecewise remap.
                        let tw = (image_size[0] as f32).max(1.0);
                        let th = (image_size[1] as f32).max(1.0);
                        let bw = (half[0] * 2.0).max(1.0);
                        let bh = (half[1] * 2.0).max(1.0);
                        let i = ns.insets_px;
                        header[0][3] = 1.0;
                        header[2][2] = super::overlay_geometry::tile_mode_to_f(ns.centre_mode);
                        header[2][3] = super::overlay_geometry::tile_mode_to_f(ns.edge_mode);
                        header[6] = [
                            (i[0] / th).clamp(0.0, 0.5),
                            (i[1] / tw).clamp(0.0, 0.5),
                            (i[2] / th).clamp(0.0, 0.5),
                            (i[3] / tw).clamp(0.0, 0.5),
                        ];
                        header[7] = [
                            (i[0] / bh).clamp(0.0, 0.5),
                            (i[1] / bw).clamp(0.0, 0.5),
                            (i[2] / bh).clamp(0.0, 0.5),
                            (i[3] / bw).clamp(0.0, 0.5),
                        ];
                    }
                }
            }
            // `OverlayFill` is non_exhaustive; a fill this renderer does not
            // know draws nothing.
            _ => {
                header[0][0] = kind::SOLID;
            }
        }
        // Stops sorted by position, positions clamped to [0, 1]. An empty list
        // is one transparent stop, so the shader always has a stop to read.
        let mut sorted: Vec<GradientStop> = stops.to_vec();
        sorted.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if sorted.is_empty() && header[0][0] != kind::SOLID && header[0][0] != kind::TEXTURE {
            sorted.push(GradientStop::new(0.0, crate::Colour::TRANSPARENT));
        }
        header[0][1] = sorted.len() as f32;
        self.texels.extend_from_slice(&header);
        for s in &sorted {
            self.texels.push(scale(s.colour.to_linear_rgba()));
        }
        for chunk in sorted.chunks(4) {
            let mut p = [1.0f32; 4];
            for (slot, s) in p.iter_mut().zip(chunk) {
                *slot = s.position.clamp(0.0, 1.0);
            }
            self.texels.push(p);
        }
        base as f32
    }
}

/// The colour a gradient-free consumer of `fill` uses: the solid colour, the
/// first stop of a gradient, or a texture fill's tint.
pub(crate) fn leading_colour(fill: &OverlayFill) -> [f32; 4] {
    match fill {
        OverlayFill::Solid(c) => c.to_linear_rgba(),
        OverlayFill::Texture { tint, .. } => tint.to_linear_rgba(),
        other => {
            let mut stops: Vec<GradientStop> = other.stops().unwrap_or(&[]).to_vec();
            stops.sort_by(|a, b| {
                a.position
                    .partial_cmp(&b.position)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            stops
                .first()
                .map_or([0.0; 4], |s| s.colour.to_linear_rgba())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Colour;

    #[test]
    fn a_record_holds_every_stop() {
        let stops: Vec<GradientStop> = (0..8)
            .map(|i| {
                GradientStop::new(
                    i as f32 / 7.0,
                    Colour::linear(i as f32 / 7.0, 0.0, 0.0, 1.0),
                )
            })
            .collect();
        let mut table = FillTable::new();
        let first = table.push(
            &OverlayFill::LinearGradient { stops, angle: 0.0 },
            [10.0, 10.0],
            1.0,
            [1.0; 4],
            [1, 1],
        );
        // The header, eight colours, and two texels of positions.
        assert_eq!(first, 0.0);
        assert_eq!(table.texel_count(), HEADER_TEXELS + 8 + 2);
        assert_eq!(table.texels()[0][1], 8.0);
        assert_eq!(table.texels()[HEADER_TEXELS + 8 + 1][3], 1.0);
        // The next record starts where this one ends.
        let second = table.push(
            &OverlayFill::radial_gradient(Colour::WHITE, Colour::BLACK),
            [4.0, 4.0],
            1.0,
            [1.0; 4],
            [1, 1],
        );
        assert_eq!(second as usize, HEADER_TEXELS + 8 + 2);
    }

    #[test]
    fn opacity_and_tint_fold_into_every_colour() {
        let mut table = FillTable::new();
        table.push(
            &OverlayFill::linear_gradient(
                Colour::linear(1.0, 1.0, 1.0, 1.0),
                Colour::linear(0.5, 0.5, 0.5, 1.0),
                0.0,
            ),
            [1.0, 1.0],
            0.5,
            [1.0, 0.5, 1.0, 0.5],
            [1, 1],
        );
        assert_eq!(table.texels()[HEADER_TEXELS], [1.0, 0.5, 1.0, 0.25]);
        assert_eq!(table.texels()[HEADER_TEXELS + 1], [0.5, 0.25, 0.5, 0.25]);
    }

    #[test]
    fn a_compiled_group_records_a_texture_as_its_tint() {
        let fill = OverlayFill::texture(crate::renderer::types::OverlayTextureId::INVALID)
            .with_tint(Colour::linear(0.2, 0.4, 0.6, 1.0));
        let mut table = FillTable::without_textures();
        table.push(&fill, [1.0, 1.0], 1.0, [1.0; 4], [1, 1]);
        assert_eq!(table.texels()[0][0], kind::SOLID);
        assert_eq!(table.texels()[3], [0.2, 0.4, 0.6, 1.0]);
    }
}
