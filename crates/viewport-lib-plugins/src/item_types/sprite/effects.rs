//! Sprite effects on the GPU: how the effect shaders are composed from the
//! plain ones, the group-3 layout and the per-batch effect block, and the
//! registry of custom sprite materials.
//!
//! An effect pipeline is the plain sprite or lit sprite shader with
//! `sprite_effects.wgsl` spliced after it: the body's vertex entry becomes
//! `sprite_vertex`, a new `vs_main` grows the quad for an outline and fills
//! the effect varyings, and the body's `<sprite-fx:...>` markers become calls
//! into the effect functions. The plain modules are never touched, so a frame
//! with no effects compiles and draws exactly what it did before.
//!
//! A custom material is a [`MaterialPlugin`] whose body defines
//! `fn shade_sprite(f: SpriteFragment) -> vec4<f32>`, the particle
//! fragment-hook contract with a sprite's fields added. Its identifiers are
//! prefixed with its name, its parameters fill lanes 1.. of `hook_params` and
//! a `params` struct, and its textures bind as `htex0..htex3` with
//! `sprite_samp`; one pipeline family is built per material, all over one
//! group-3 layout.

use viewport_lib::error::{ViewportError, ViewportResult};
use viewport_lib::plugin_api::wgsl_compose;
use viewport_lib::resources::DeviceResources;
use viewport_lib::{MaterialPlugin, ParamDesc};

use super::types::{PaletteIndex, SpriteEffects};

/// Texture slots a sprite material reads (`htex0..htex3`), as the particle
/// fragment hook has. Every effect pipeline declares all four; unused slots
/// bind 1x1 white.
pub(super) const MATERIAL_TEXTURES: usize = 4;

/// Parameter lanes in the effect block. Lane 0 is reserved (its `.w` carries
/// the time), so a material declares at most one fewer parameters.
const LANES: usize = 8;

/// The effect stage spliced after a sprite body.
const SPRITE_FX: &str = crate::item_types::shader::wgsl_source!("sprite_effects");

/// The effect varyings, after the body's own `last` locations.
fn varyings(first: u32) -> String {
    format!(
        "@location({}) @interpolate(flat) fx_word: u32,\n    \
         @location({}) fx_local_uv: vec2<f32>,\n    \
         @location({}) @interpolate(flat) fx_outline_uv: vec2<f32>,\n    \
         @location({}) @interpolate(flat) fx_uv_rect: vec4<f32>,",
        first,
        first + 1,
        first + 2,
        first + 3
    )
}

/// A custom material's part of an effect shader.
pub(super) struct MaterialSource<'a> {
    pub name: &'a str,
    pub prefixed_body: &'a str,
    pub params: &'a [ParamDesc],
}

fn replace_once(source: &mut String, from: &str, to: &str) -> Result<(), String> {
    match source.matches(from).count() {
        1 => {
            *source = source.replacen(from, to, 1);
            Ok(())
        }
        n => Err(format!(
            "expected one '{from}' in the sprite shader, found {n}"
        )),
    }
}

/// The effect variant of `base`, an assembled sprite (`lit == false`) or lit
/// sprite module, with `material`'s stage when there is one.
pub(super) fn compose(
    base: &str,
    lit: bool,
    material: Option<&MaterialSource<'_>>,
) -> Result<String, String> {
    let mut out = base.to_string();
    replace_once(
        &mut out,
        "@vertex\nfn vs_main(in: VertexIn) -> VertexOut {",
        "fn sprite_vertex(in: VertexIn, fx_corner: vec2<f32>) -> VertexOut {",
    )?;
    replace_once(&mut out, "= quad_corner(in.vertex_index);", "= fx_corner;")?;
    replace_once(
        &mut out,
        "// <sprite-fx:vertex-in>",
        "@location(1) fx_word: u32,",
    )?;
    replace_once(
        &mut out,
        "// <sprite-fx:varyings>",
        &varyings(if lit { 8 } else { 4 }),
    )?;
    replace_once(
        &mut out,
        "// <sprite-fx:base>",
        "colour = sprite_fx_base(colour, in);",
    )?;
    replace_once(
        &mut out,
        "// <sprite-fx:shaded>",
        "colour = sprite_fx_shaded(colour, in);",
    )?;
    let mut fx = SPRITE_FX.to_string();
    if let Some(m) = material {
        let call = if m.params.is_empty() {
            format!(
                "c = {}__shade_sprite(sprite_fx_fragment(in, texel));",
                m.name
            )
        } else {
            format!(
                "params = sprite_material_params();\n    c = {}__shade_sprite(sprite_fx_fragment(in, texel));",
                m.name
            )
        };
        replace_once(&mut fx, "// <sprite-fx:material>", &call)?;
    }
    out.push('\n');
    out.push_str(&fx);
    if let Some(m) = material {
        out.push('\n');
        out.push_str(&material_declarations(m));
        out.push('\n');
        out.push_str(m.prefixed_body);
        out.push('\n');
    }
    Ok(out)
}

/// A material's `params` struct, filled from lanes 1.. of the effect block.
fn material_declarations(m: &MaterialSource<'_>) -> String {
    let mut out = String::new();
    if !m.params.is_empty() {
        out.push_str("struct SpriteMaterialParams {\n");
        for p in m.params {
            out.push_str(&format!("    {}: {},\n", p.name, p.kind.wgsl()));
        }
        out.push_str("};\nvar<private> params: SpriteMaterialParams;\n");
        out.push_str("fn sprite_material_params() -> SpriteMaterialParams {\n");
        out.push_str("    var p: SpriteMaterialParams;\n");
        for (i, p) in m.params.iter().enumerate() {
            let swizzle = match p.kind {
                viewport_lib::ParamKind::F32 => ".x",
                viewport_lib::ParamKind::Vec2 => ".xy",
                viewport_lib::ParamKind::Vec3 => ".xyz",
                viewport_lib::ParamKind::Vec4 => "",
            };
            out.push_str(&format!(
                "    p.{} = sprite_fx.lanes[{}]{swizzle};\n",
                p.name,
                i + 1
            ));
        }
        out.push_str("    return p;\n}\n");
    }
    out
}

/// The per-batch effect block, matching `SpriteFx` in `sprite_effects.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SpriteFxUniform {
    flash_colour: [f32; 4],
    edge_colour: [f32; 4],
    outline_colour: [f32; 4],
    edge_width: f32,
    noise_scale: f32,
    outline_px: f32,
    flags: u32,
    lanes: [[f32; 4]; LANES],
}

const FX_FLAG_NOISE: u32 = 1;
const FX_FLAG_PALETTE: u32 = 2;
const FX_FLAG_PALETTE_LUMINANCE: u32 = 4;

/// Bytes per sprite of the effect stream: one packed word.
pub(super) const FX_STRIDE: u32 = 4;

/// The group-3 layout of every effect pipeline: the lit normal map, the
/// effect block, the noise and the palette, then a material's sampler and its
/// four texture slots.
pub(super) fn fx_layout(device: &viewport_lib::gpu::Device) -> viewport_lib::gpu::BindGroupLayout {
    use viewport_lib::gpu::{
        BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, ShaderStages,
        TextureSampleType, TextureViewDimension,
    };
    let texture = |binding| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampler = |binding| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    };
    let mut entries = vec![
        texture(0),
        sampler(1),
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        texture(3),
        texture(4),
        sampler(5),
    ];
    entries.extend((0..MATERIAL_TEXTURES as u32).map(|i| texture(6 + i)));
    device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
        label: Some("sprite_fx_bgl"),
        entries: &entries,
    })
}

/// A sprite material as the registry holds it.
pub(crate) struct SpriteMaterialEntry {
    pub(super) name: String,
    pub(super) prefixed_body: String,
    pub(super) params: Vec<ParamDesc>,
    pub(super) texture_count: u32,
    pub(super) sampler: viewport_lib::gpu::Sampler,
}

impl SpriteMaterialEntry {
    pub(super) fn source(&self) -> MaterialSource<'_> {
        MaterialSource {
            name: &self.name,
            prefixed_body: &self.prefixed_body,
            params: &self.params,
        }
    }
}

fn invalid(reason: String) -> ViewportError {
    ViewportError::ShadeShaderInvalid { reason }
}

/// The two plain modules every effect family composes from: the sprite and
/// the lit sprite shader as their pipelines build them.
pub(super) fn base_sources() -> (String, String) {
    use crate::item_types::shader::{lit_shader, scene_shader, wgsl_source};
    use viewport_lib::plugin_api::shared_wgsl;
    (
        scene_shader(&[super::SPRITE_COMMON], wgsl_source!("sprite")),
        lit_shader(
            &[
                shared_wgsl::SHARED_CLIP_VOLUME_WGSL,
                shared_wgsl::SHARED_CSM_WGSL,
                shared_wgsl::SHARED_SHADOW_ATLAS_WGSL,
                super::SPRITE_COMMON,
            ],
            wgsl_source!("sprite_lit"),
        ),
    )
}

/// Check `plugin` against the sprite material contract and compose and
/// validate both of its effect modules. Stores nothing.
pub(super) fn validate_material(
    device: &viewport_lib::gpu::Device,
    resources: &DeviceResources,
    existing: &[SpriteMaterialEntry],
    plugin: &dyn MaterialPlugin,
) -> ViewportResult<SpriteMaterialEntry> {
    let name = plugin.name();
    if !wgsl_compose::is_identifier(name) {
        return Err(invalid(format!(
            "sprite material name '{name}' is not a WGSL identifier"
        )));
    }
    if existing.iter().any(|e| e.name == name) {
        return Err(ViewportError::ShadeNameTaken {
            name: name.to_string(),
        });
    }
    let texture_count = plugin.texture_count();
    if texture_count as usize > MATERIAL_TEXTURES {
        return Err(invalid(format!(
            "sprite material '{name}' declares {texture_count} textures, at most \
             {MATERIAL_TEXTURES} fit"
        )));
    }
    let params = plugin.param_layout();
    if params.len() > LANES - 1 {
        return Err(invalid(format!(
            "sprite material '{name}' declares {} parameters, at most {} fit",
            params.len(),
            LANES - 1
        )));
    }
    for (i, p) in params.iter().enumerate() {
        if !wgsl_compose::is_identifier(&p.name) || params[..i].iter().any(|q| q.name == p.name) {
            return Err(invalid(format!(
                "sprite material '{name}': parameter '{}' is not a fresh WGSL identifier",
                p.name
            )));
        }
        if !p.default.fits(p.kind) {
            return Err(invalid(format!(
                "sprite material '{name}': the default of '{}' does not fit its kind",
                p.name
            )));
        }
    }
    let prefixed_body = wgsl_compose::prefix_identifiers(name, &plugin.wgsl_body());
    if !prefixed_body.contains(&format!("fn {name}__shade_sprite(")) {
        return Err(invalid(format!(
            "sprite material '{name}' defines no `shade_sprite` function"
        )));
    }
    if wgsl_compose::contains_identifier(&prefixed_body, &format!("{name}__params")) {
        return Err(invalid(format!(
            "sprite material '{name}' declares the reserved identifier 'params'"
        )));
    }
    let entry = SpriteMaterialEntry {
        name: name.to_string(),
        prefixed_body,
        params,
        texture_count,
        sampler: resources.sampler_for(device, plugin.sampler()),
    };
    let (unlit, lit) = base_sources();
    for (label, base, is_lit) in [("sprite", &unlit, false), ("sprite_lit", &lit, true)] {
        let composed = compose(base, is_lit, Some(&entry.source()))
            .map_err(|e| invalid(format!("sprite material '{name}': {e}")))?;
        wgsl_compose::validate_module(
            device,
            &format!("sprite_material_{name}_{label}"),
            &composed,
        )?;
    }
    Ok(entry)
}

/// The texture views and block one effect batch binds at group 3, resolved
/// before the build so the build can run on a worker.
#[derive(Clone)]
pub(super) struct FxBindings {
    /// The pipeline family: 0 for the built-in effects, `1 + i` for material `i`.
    pub family: u32,
    pub bgl: viewport_lib::gpu::BindGroupLayout,
    pub uniform: SpriteFxUniform,
    pub noise_view: viewport_lib::gpu::TextureView,
    pub palette_view: viewport_lib::gpu::TextureView,
    /// The material's sampler, or `None` to bind the batch's own.
    pub material_sampler: Option<viewport_lib::gpu::Sampler>,
    /// The four material texture slots.
    pub material_views: Vec<viewport_lib::gpu::TextureView>,
}

/// Resolve `fx` into the bindings its batch's group 3 needs.
pub(super) fn resolve(
    resources: &DeviceResources,
    fx_bgl: &viewport_lib::gpu::BindGroupLayout,
    materials: &[SpriteMaterialEntry],
    fx: &SpriteEffects,
    report: bool,
) -> FxBindings {
    use viewport_lib::resources::TextureRequirement;
    if report {
        resources.check_texture_slot(
            fx.dissolve_noise,
            TextureRequirement::linear("SpriteEffects::dissolve_noise"),
        );
        resources.check_texture_slot(
            fx.palette.map(|p| p.texture),
            TextureRequirement::colour("SpriteEffects::palette"),
        );
    }
    let white = || {
        resources
            .fallback_texture_view(viewport_lib::TextureSlot::Albedo)
            .clone()
    };
    let view_of = |id: Option<viewport_lib::resources::TextureId>| {
        id.and_then(|id| resources.texture_view(id)).cloned()
    };
    let mut flags = 0;
    let noise_view = match view_of(fx.dissolve_noise) {
        Some(v) => {
            flags |= FX_FLAG_NOISE;
            v
        }
        None => white(),
    };
    let palette_view = match view_of(fx.palette.map(|p| p.texture)) {
        Some(v) => {
            flags |= FX_FLAG_PALETTE;
            if fx
                .palette
                .is_some_and(|p| p.index == PaletteIndex::Luminance)
            {
                flags |= FX_FLAG_PALETTE_LUMINANCE;
            }
            v
        }
        None => white(),
    };
    let material = fx
        .material
        .as_ref()
        .and_then(|m| materials.get(m.id.0 as usize).map(|e| (m, e)));
    let mut lanes = [[0.0f32; 4]; LANES];
    let (family, material_sampler, material_views) = match material {
        Some((m, entry)) => {
            for (i, p) in entry.params.iter().enumerate() {
                let value = m
                    .params
                    .iter()
                    .find(|(n, v)| *n == p.name && v.fits(p.kind))
                    .map_or(p.default, |(_, v)| *v);
                lanes[i + 1] = value.to_vec4();
            }
            let views = (0..MATERIAL_TEXTURES)
                .map(|i| {
                    let id = (i < entry.texture_count as usize)
                        .then(|| m.textures.get(i).copied().flatten())
                        .flatten();
                    view_of(id).unwrap_or_else(white)
                })
                .collect();
            (1 + m.id.0, Some(entry.sampler.clone()), views)
        }
        None => (0, None, (0..MATERIAL_TEXTURES).map(|_| white()).collect()),
    };
    let outline = fx.outline.filter(|o| o.width_px > 0.0);
    FxBindings {
        family,
        bgl: fx_bgl.clone(),
        uniform: SpriteFxUniform {
            flash_colour: fx.flash_colour.to_linear_rgba(),
            edge_colour: fx.dissolve_edge_colour.to_linear_rgba(),
            outline_colour: outline.map_or([0.0; 4], |o| o.colour.to_linear_rgba()),
            edge_width: fx.dissolve_edge_width.max(0.0),
            noise_scale: fx.noise_scale,
            outline_px: outline.map_or(0.0, |o| o.width_px),
            flags,
            lanes,
        },
        noise_view,
        palette_view,
        material_sampler,
        material_views,
    }
}

/// Every texture id `fx` binds, for the residency check of a stored batch.
pub(super) fn texture_ids(
    fx: &SpriteEffects,
) -> impl Iterator<Item = viewport_lib::resources::TextureId> + '_ {
    fx.dissolve_noise
        .into_iter()
        .chain(fx.palette.map(|p| p.texture))
        .chain(
            fx.material
                .iter()
                .flat_map(|m| m.textures.iter().copied().flatten()),
        )
}

/// Build one effect batch's group-3 bind group.
pub(super) fn bind_group(
    device: &viewport_lib::gpu::Device,
    fx: &FxBindings,
    normal_view: &viewport_lib::gpu::TextureView,
    sampler: &viewport_lib::gpu::Sampler,
    uniform_buf: &viewport_lib::gpu::Buffer,
) -> viewport_lib::gpu::BindGroup {
    use viewport_lib::gpu::{BindGroupEntry, BindingResource};
    let mut entries = vec![
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(normal_view),
        },
        BindGroupEntry {
            binding: 1,
            resource: BindingResource::Sampler(sampler),
        },
        BindGroupEntry {
            binding: 2,
            resource: uniform_buf.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 3,
            resource: BindingResource::TextureView(&fx.noise_view),
        },
        BindGroupEntry {
            binding: 4,
            resource: BindingResource::TextureView(&fx.palette_view),
        },
        BindGroupEntry {
            binding: 5,
            resource: BindingResource::Sampler(fx.material_sampler.as_ref().unwrap_or(sampler)),
        },
    ];
    for (i, view) in fx.material_views.iter().enumerate() {
        entries.push(BindGroupEntry {
            binding: 6 + i as u32,
            resource: BindingResource::TextureView(view),
        });
    }
    device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
        label: Some("sprite_fx_bg"),
        layout: &fx.bgl,
        entries: &entries,
    })
}

/// The packed effect words of an item's sprites, in drawn order.
pub(super) fn instance_words(
    fx: &SpriteEffects,
    count: usize,
    source: impl Fn(usize) -> usize,
) -> Vec<u32> {
    (0..count).map(|d| fx.instance_word(source(d))).collect()
}

/// Encode a run of caller-supplied effect values for a ranged write.
pub(crate) fn encode_effects(values: &[super::channels::SpriteEffect]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| {
            super::types::pack_effect_word(v.flash, v.dissolve, v.palette_row).to_le_bytes()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_bodies_compose() {
        let (unlit, lit) = base_sources();
        for (base, is_lit) in [(&unlit, false), (&lit, true)] {
            let out = compose(base, is_lit, None).expect("composes");
            assert!(out.contains("fn sprite_vertex(in: VertexIn, fx_corner: vec2<f32>)"));
            assert_eq!(out.matches("@vertex").count(), 1);
            assert!(out.contains("colour = sprite_fx_base(colour, in);"));
            assert!(out.contains("colour = sprite_fx_shaded(colour, in);"));
            // Only the material slot stays, as a comment, when there is no material.
            assert!(
                !out.replace("// <sprite-fx:material>", "")
                    .contains("<sprite-fx:"),
                "every body marker replaced"
            );
        }
    }

    #[test]
    fn material_params_unpack_from_the_block() {
        let params = [
            ParamDesc::f32("rim", 1.0),
            ParamDesc::vec3("tint", [1.0; 3]),
        ];
        let src = MaterialSource {
            name: "glow",
            prefixed_body: "fn glow__shade_sprite(f: SpriteFragment) -> vec4<f32> { return f.colour; }",
            params: &params,
        };
        let decl = material_declarations(&src);
        assert!(decl.contains("p.rim = sprite_fx.lanes[1].x;"));
        assert!(decl.contains("p.tint = sprite_fx.lanes[2].xyz;"));
        let (unlit, _) = base_sources();
        let out = compose(&unlit, false, Some(&src)).unwrap();
        assert!(out.contains("params = sprite_material_params();"));
        assert!(out.contains("c = glow__shade_sprite(sprite_fx_fragment(in, texel));"));
    }

    #[test]
    fn block_size_matches_the_shader() {
        assert_eq!(std::mem::size_of::<SpriteFxUniform>(), 3 * 16 + 16 + 8 * 16);
    }
}
